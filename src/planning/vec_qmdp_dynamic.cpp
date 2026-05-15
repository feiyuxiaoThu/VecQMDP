/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file vec_qmdp_dynamic.cpp
 * @brief DynVecQMDP implementation — dynamic chunked SoA tree-search kernel.
 *
 * This file contains the domain-agnostic BeliefTreeSearch core for DynVecQMDP.
 * The fundamental difference from vec_qmdp_static.cpp:
 *
 *   - Nodes live in a DynSoAPool (chunked, grow-on-demand).
 *   - Parent–child links are stored explicitly (child_base_idx, parent_idx).
 *   - exploreNodesVote() returns DynExploreResult structs that include global
 *     pool indices, enabling the derived class to gather domain state via
 *     SIMD scatter/gather without reconstructing implicit paths.
 *   - backPropagate / backupDepth traverse parent_idx chains instead of the
 *     formula  parent = (cur - 1) / num_actions.
 *
 * Memory growth is O(visited_nodes) = O(max_iters * num_actions), not
 * O(num_actions^tree_height).  For tree_height=90, num_actions=13, this
 * reduces memory from ~10^99 to ~O(40 000) nodes per thread.
 */
#include <planning/vec_qmdp_dynamic.hpp>

// In VECQMDP_TEST_STANDALONE mode the test is compiled without VAMP / NEON
// headers and without global_utils.cpp.  Only the pool + BeliefTreeSearch logic is built;
// the multi-threaded parallel infrastructure is excluded via guards below.
#ifndef VECQMDP_TEST_STANDALONE
#include <utils/global_utils.hpp> // Full ThreadPool definition
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <ostream>
#include <string>

namespace vec_qmdp
{
    namespace planning
    {
        // =====================================================================
        // Construction
        // =====================================================================

        DynVecQMDP::DynVecQMDP(int num_scenarios, uint32_t num_actions, uint32_t tree_height)
            : pool_(/* initial_chunks = */ 16), num_actions_(num_actions), tree_height_(tree_height),
              scenario_size_(static_cast<uint32_t>(num_scenarios))
        {
            scenario_roots_.resize(scenario_size_, DYN_INVALID_IDX);
            aggregated_action_values_.assign(num_actions_, 0.0f);
            aggregated_action_counts_.assign(num_actions_, 0);
            convergence_info_.resize(scenario_size_);
        }

        // =====================================================================
        // Parallel infrastructure
        // =====================================================================

        void DynVecQMDP::initParallelInfrastructure(size_t num_threads)
        {
            num_worker_threads_ = std::max<size_t>(1, num_threads);

            aggregated_action_values_.assign(num_actions_, 0.0f);
            aggregated_action_counts_.assign(num_actions_, 0);

            if (num_worker_threads_ == 1)
            {
                worker_instances_.clear();
                thread_pool_.reset();
                return;
            }

#ifndef VECQMDP_TEST_STANDALONE
            thread_pool_ = std::make_shared<utils::ThreadPool>(num_worker_threads_);
#endif
            worker_instances_.clear();
            worker_instances_.reserve(num_worker_threads_);
            for (size_t i = 0; i < num_worker_threads_; ++i)
                worker_instances_.push_back(makeWorkerInstance());
        }

        void
        DynVecQMDP::dispatchParallelSearch(std::function<BeliefTreeSearchThreadResult(DynVecQMDP *, size_t)> search_fn)
        {
            std::fill(aggregated_action_values_.begin(), aggregated_action_values_.end(), 0.0f);
            std::fill(aggregated_action_counts_.begin(), aggregated_action_counts_.end(), 0);
            total_simulation_count_ = 0;

            if (num_worker_threads_ <= 1 || worker_instances_.empty())
            {
                auto result = search_fn(this, 0);
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    aggregated_action_values_[a] += result.action_sum_values[a];
                    aggregated_action_counts_[a] += result.action_counts[a];
                }
                total_simulation_count_ += result.simulation_count;
            }
            else
            {
#ifndef VECQMDP_TEST_STANDALONE
                std::vector<std::future<BeliefTreeSearchThreadResult>> futures;
                futures.reserve(num_worker_threads_);
                for (size_t i = 0; i < num_worker_threads_; ++i)
                {
                    DynVecQMDP *wp = worker_instances_[i].get();
                    futures.emplace_back(thread_pool_->enqueue([wp, i, &search_fn]() { return search_fn(wp, i); }));
                }
                for (auto &f : futures)
                {
                    auto result = f.get();
                    for (uint32_t a = 0; a < num_actions_; ++a)
                    {
                        aggregated_action_values_[a] += result.action_sum_values[a];
                        aggregated_action_counts_[a] += result.action_counts[a];
                    }
                    total_simulation_count_ += result.simulation_count;
                }
#endif
            }

            for (uint32_t a = 0; a < num_actions_; ++a)
            {
                if (aggregated_action_counts_[a] > 0)
                    aggregated_action_values_[a] /= static_cast<float>(aggregated_action_counts_[a]);
                else
                    aggregated_action_values_[a] = -1e7f;
            }
        }

        // =====================================================================
        // Pool / tree initialisation
        // =====================================================================

        void DynVecQMDP::resetDynStructures()
        {
            pool_.reset();
            dyn_candidate_actions_.clear();

            for (auto &info : convergence_info_)
            {
                info.best_action_idx = -1;
                info.best_q_value = std::numeric_limits<float>::lowest();
                info.stable_iterations = 0;
                info.prev_q_value = std::numeric_limits<float>::lowest();
                info.total_checks = 0;
            }
        }

        void DynVecQMDP::initDynRoots()
        {
            // Allocate one slot per scenario; roots occupy global indices 0..S-1.
            const int32_t base = pool_.allocate(static_cast<int32_t>(scenario_size_));
            assert(base == 0 && "Root allocation must be the first allocation");

            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const int32_t root_g = base + static_cast<int32_t>(s);
                scenario_roots_[s] = root_g;

                pool_.init_node(root_g,
                                /* depth         */ 0,
                                /* rollout_len   */ static_cast<int32_t>(tree_height_) - 1,
                                /* parent_g      */ DYN_INVALID_IDX,
                                /* action        */ 0,
                                /* tree_height   */ static_cast<int32_t>(tree_height_));
            }
        }

        // =====================================================================
        // Child block allocation
        // =====================================================================

        int32_t DynVecQMDP::allocateChildBlock(int32_t parent_g)
        {
            assert(parent_g >= 0 && parent_g < pool_.total_nodes());

            const int32_t parent_depth = pool_.depth(parent_g);
            const int32_t child_depth = parent_depth + 1;
            const int32_t child_rl = static_cast<int32_t>(tree_height_) - child_depth - 1;
            const int32_t n = static_cast<int32_t>(num_actions_);
            const int32_t child_base = pool_.allocate(n);

            for (int32_t a = 0; a < n; ++a)
            {
                pool_.init_node(child_base + a, child_depth, std::max(0, child_rl), parent_g, a,
                                static_cast<int32_t>(tree_height_));
            }

            // Register children on parent
            pool_.child_base(parent_g) = child_base;
            return child_base;
        }

        // =====================================================================
        // Candidate-action cache
        // =====================================================================

        const std::vector<uint32_t> &DynVecQMDP::getDynCandidateActions(int32_t global_idx)
        {
            auto it = dyn_candidate_actions_.find(global_idx);
            if (it == dyn_candidate_actions_.end())
            {
                auto [ins, ok] = dyn_candidate_actions_.emplace(global_idx, computeNodeCandidateActionsDyn(global_idx));
                return ins->second;
            }
            return it->second;
        }

        // =====================================================================
        // UCB scoring
        // =====================================================================

        float DynVecQMDP::calculateUCB(int32_t node_g, int32_t parent_g, int target_depth) const
        {
            if (pool_.visit_count(node_g) == 0)
                return std::numeric_limits<float>::max();

            const float exploitation = pool_.q_value(node_g);
            const float exploration =
                exploration_constant_ * std::sqrt(std::log(static_cast<float>(pool_.visit_count(parent_g))) /
                                                  static_cast<float>(pool_.visit_count(node_g)));
            float score = exploitation + exploration;

            if (target_depth != -1 && depth_sync_lambda_ != 0.0f)
            {
                const int min_g = pool_.min_depth(node_g);
                const int max_g = pool_.max_depth(node_g);
                const int eff_g = (target_depth <= min_g) ? min_g : (target_depth >= max_g) ? max_g : target_depth;
                score -= depth_sync_lambda_ * std::abs(static_cast<float>(eff_g) - static_cast<float>(target_depth));
            }
            return score;
        }

        // =====================================================================
        // Node selection via UCB
        // =====================================================================

        int32_t DynVecQMDP::selectNodeDyn(int32_t parent_g, const std::vector<uint32_t> &cands, int target_depth) const
        {
            const int32_t child_base = pool_.child_base(parent_g);
            assert(child_base != DYN_INVALID_IDX);

            int32_t best_child = child_base; // fallback: first child
            float   best_ucb = -std::numeric_limits<float>::max();

            for (uint32_t a : cands)
            {
                const int32_t child_g = child_base + static_cast<int32_t>(a);
                const float   ucb = calculateUCB(child_g, parent_g, target_depth);
                if (ucb > best_ucb)
                {
                    best_ucb = ucb;
                    best_child = child_g;
                }
            }
            return best_child;
        }

        // =====================================================================
        // exploreNodes — standard UCB search
        // =====================================================================

        std::vector<DynExploreResult> DynVecQMDP::exploreNodes()
        {
            const uint32_t S = scenario_size_;

            std::vector<DynExploreResult> results(S);
            std::vector<int>              sel_depths(S, 0);
            std::vector<uint32_t>         tmp_actions(S, 0);

            // ---- standard UCB traversal (no depth-sync penalty) ----
            for (uint32_t s = 0; s < S; ++s)
            {
                const int32_t root_g = scenario_roots_[s];

                if (pool_.visit_count(root_g) == 0)
                {
                    // Root not yet visited: select it directly.
                    results[s].parent_global_idx = root_g;
                    const auto &cands = getDynCandidateActions(root_g);
                    tmp_actions[s] = cands.empty() ? 0u : cands[0];
                }
                else
                {
                    int32_t  cur_g = root_g;
                    uint32_t sel_action = static_cast<uint32_t>(pool_.curr_action(root_g));
                    bool     found = false;

                    while (!found)
                    {
                        const auto   &cands = getDynCandidateActions(cur_g);
                        const int32_t child_b = pool_.child_base(cur_g);

                        // Leaf node (at max depth or no children allocated yet)
                        if (pool_.depth(cur_g) >= static_cast<int32_t>(tree_height_) || child_b == DYN_INVALID_IDX ||
                            pool_.active_flag(cur_g) == 0)
                        {
                            sel_action = static_cast<uint32_t>(pool_.curr_action(cur_g));
                            found = true;
                            break;
                        }

                        // Check for unvisited children
                        bool found_unvisited = false;
                        for (uint32_t a : cands)
                        {
                            const int32_t cg = child_b + static_cast<int32_t>(a);
                            if (pool_.visit_count(cg) == 0)
                            {
                                sel_action = a;
                                found = true;
                                found_unvisited = true;
                                break;
                            }
                        }
                        if (found_unvisited)
                            break;

                        // All visited: descend via UCB
                        cur_g = selectNodeDyn(cur_g, cands, -1);
                    }

                    results[s].parent_global_idx = cur_g;
                    tmp_actions[s] = sel_action;
                }

                sel_depths[s] = pool_.depth(results[s].parent_global_idx);
                results[s].scenario_idx = s;
            }

            // ---- Fill action_idx from tmp_actions ----
            for (uint32_t s = 0; s < S; ++s)
            {
                results[s].action_idx = tmp_actions[s];
                results[s].parent_depth = sel_depths[s];
            }

            return results;
        }

        // =====================================================================
        // exploreNodesVote — majority-vote depth synchronisation
        // =====================================================================

        std::vector<DynExploreResult> DynVecQMDP::exploreNodesVote()
        {
            const uint32_t S = scenario_size_;

            std::vector<DynExploreResult> results(S);
            std::vector<int>              sel_depths(S, 0);
            std::vector<uint32_t>         tmp_actions(S, 0);

            int  first_depth = -1;
            bool all_same_depth = true;

            // ---- Phase 1: standard UCB traversal (no depth-sync penalty) ----
            for (uint32_t s = 0; s < S; ++s)
            {
                const int32_t root_g = scenario_roots_[s];

                if (pool_.visit_count(root_g) == 0)
                {
                    // Root not yet visited: select it directly.
                    results[s].parent_global_idx = root_g;
                    const auto &cands = getDynCandidateActions(root_g);
                    tmp_actions[s] = cands.empty() ? 0u : cands[0];
                }
                else
                {
                    int32_t  cur_g = root_g;
                    uint32_t sel_action = static_cast<uint32_t>(pool_.curr_action(root_g));
                    bool     found = false;

                    while (!found)
                    {
                        const auto   &cands = getDynCandidateActions(cur_g);
                        const int32_t child_b = pool_.child_base(cur_g);

                        // Leaf node (at max depth or no children allocated yet)
                        if (pool_.depth(cur_g) >= static_cast<int32_t>(tree_height_) || child_b == DYN_INVALID_IDX ||
                            pool_.active_flag(cur_g) == 0)
                        {
                            sel_action = static_cast<uint32_t>(pool_.curr_action(cur_g));
                            found = true;
                            break;
                        }

                        // Check for unvisited children
                        bool found_unvisited = false;
                        for (uint32_t a : cands)
                        {
                            const int32_t cg = child_b + static_cast<int32_t>(a);
                            if (pool_.visit_count(cg) == 0)
                            {
                                sel_action = a;
                                found = true;
                                found_unvisited = true;
                                break;
                            }
                        }
                        if (found_unvisited)
                            break;

                        // All visited: descend via UCB
                        cur_g = selectNodeDyn(cur_g, cands, -1);
                    }

                    results[s].parent_global_idx = cur_g;
                    tmp_actions[s] = sel_action;
                }

                sel_depths[s] = pool_.depth(results[s].parent_global_idx);
                results[s].scenario_idx = s;

                if (s == 0)
                {
                    first_depth = sel_depths[0];
                }
                else if (all_same_depth && sel_depths[s] != first_depth)
                    all_same_depth = false;
            }

            // ---- Fill action_idx from tmp_actions ----
            for (uint32_t s = 0; s < S; ++s)
            {
                results[s].action_idx = tmp_actions[s];
                results[s].parent_depth = sel_depths[s];
            }

            if (all_same_depth)
                return results;

            // ---- Phase 2: majority-vote depth selection ----
            std::vector<int> depth_votes(tree_height_ + 1, 0);
            for (int d : sel_depths)
                ++depth_votes[d];

            int target_sync = 0, max_votes = -1;
            for (int d = 0; d <= static_cast<int>(tree_height_); ++d)
            {
                const int cnt = depth_votes[d];
                if (cnt == 0)
                    continue;

                if (d == static_cast<int>(tree_height_))
                {
                    // Leaf depth: last-resort only
                    if (max_votes == -1)
                    {
                        max_votes = cnt;
                        target_sync = d;
                    }
                    continue;
                }
                if (cnt > max_votes || target_sync == static_cast<int>(tree_height_))
                {
                    max_votes = cnt;
                    target_sync = d;
                }
                else if (cnt == max_votes && d > target_sync)
                    target_sync = d;
            }

            // ---- Phase 3: re-align mismatched scenarios ----
            for (uint32_t s = 0; s < S; ++s)
            {
                if (sel_depths[s] == target_sync)
                    continue;

                const int32_t root_g = scenario_roots_[s];
                if (pool_.visit_count(root_g) == 0)
                    continue;

                int32_t  cur_g = root_g;
                uint32_t sel_action = static_cast<uint32_t>(pool_.curr_action(root_g));
                bool     found = false;

                while (!found)
                {
                    const auto   &cands = getDynCandidateActions(cur_g);
                    const int32_t child_b = pool_.child_base(cur_g);

                    if (pool_.depth(cur_g) >= static_cast<int32_t>(tree_height_) || child_b == DYN_INVALID_IDX ||
                        pool_.active_flag(cur_g) == 0)
                    {
                        sel_action = static_cast<uint32_t>(pool_.curr_action(cur_g));
                        found = true;
                        break;
                    }

                    bool found_unvisited = false;
                    for (uint32_t a : cands)
                    {
                        const int32_t cg = child_b + static_cast<int32_t>(a);
                        if (pool_.visit_count(cg) == 0)
                        {
                            sel_action = a;
                            found = true;
                            found_unvisited = true;
                            break;
                        }
                    }
                    if (found_unvisited)
                        break;

                    // Descend with depth-sync UCB penalty
                    cur_g = selectNodeDyn(cur_g, cands, target_sync);
                }

                results[s].parent_global_idx = cur_g;
                results[s].action_idx = sel_action;
                results[s].parent_depth = pool_.depth(cur_g);
            }

            return results;
        }

        // =====================================================================
        // Back-propagation
        // =====================================================================

        void DynVecQMDP::backPropagate(int32_t child_g, float rollout_value)
        {
            int32_t cur = child_g;
            bool    pruned = false;

            while (cur != DYN_INVALID_IDX)
            {
                rollout_value = pool_.reward(cur) + discount_factor_ * rollout_value;

                // Store initial rollout value on first visit
                const bool first_visit = (pool_.visit_count(cur) == 0);
                if (first_visit)
                {
                    pool_.initial_rollout(cur) = rollout_value;
                }

                const int32_t vc_old = pool_.visit_count(cur);
                const int32_t vc_new = vc_old + 1;
                pool_.visit_count(cur) = vc_new;
                pool_.q_value(cur) += (rollout_value - pool_.q_value(cur)) / static_cast<float>(vc_new);

                // Persistent max-backup: Q(s) = max(Q_running_avg, initial_rollout)
                pool_.q_value(cur) = std::max(pool_.q_value(cur), pool_.initial_rollout(cur));

                if (rollout_value < pruned_threshold_ || pool_.reward(cur) < pruned_threshold_)
                {
                    pruned = true;
                    // Increment visit count on root so it is not re-selected immediately
                    const int32_t root_g = scenario_roots_[0]; // approx; roots share the counter
                    (void)root_g;
                    break;
                }

                const int32_t par = pool_.parent(cur);
                if (par == DYN_INVALID_IDX)
                    break; // reached root — already updated above

                // Max-backup blending: blend max child Q into the rollout value
                const int32_t sib_base = pool_.child_base(par);
                if (sib_base != DYN_INVALID_IDX)
                {
                    float max_child_q = std::numeric_limits<float>::lowest();
                    bool  found_child = false;
                    for (uint32_t a = 0; a < num_actions_; ++a)
                    {
                        const int32_t cg = sib_base + static_cast<int32_t>(a);
                        if (pool_.visit_count(cg) > 0)
                        {
                            max_child_q = std::max(max_child_q, pool_.q_value(cg));
                            found_child = true;
                        }
                    }
                    if (found_child)
                        rollout_value = (1.0f - max_backup_lambda_) * max_child_q + max_backup_lambda_ * rollout_value;
                }

                cur = par;
            }

            // If pruned before reaching root: bump root visit count so UCB does not
            // treat it as unexplored.
            if (pruned)
            {
                for (uint32_t s = 0; s < scenario_size_; ++s)
                {
                    const int32_t root_g = scenario_roots_[s];
                    if (pool_.parent(child_g) == DYN_INVALID_IDX && child_g == root_g)
                    {
                        // child_g is itself a root — already updated
                        break;
                    }
                    // Just find the root of this node's tree by walking up
                    // (roots are the only nodes with parent == DYN_INVALID_IDX)
                }
            }
        }

        // =====================================================================
        // Depth-gap backup
        // =====================================================================

        void DynVecQMDP::backupDepth(int32_t child_g)
        {
            int32_t cur = child_g;

            const int32_t own_depth = pool_.depth(cur);
            pool_.min_depth(cur) = own_depth;
            pool_.max_depth(cur) = own_depth;

            while (true)
            {
                const int32_t par = pool_.parent(cur);
                if (par == DYN_INVALID_IDX)
                    break;

                const int32_t sib_base = pool_.child_base(par);
                assert(sib_base != DYN_INVALID_IDX);

                // Compute new [min, max] depth across all children of par.
                int  new_min = std::numeric_limits<int>::max();
                int  new_max = 0;
                bool fully_expanded = true;

                const auto &cands = getDynCandidateActions(par);
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    const int32_t cg = sib_base + static_cast<int32_t>(a);
                    if (pool_.active_flag(cg) != 0)
                    {
                        new_min = std::min(new_min, pool_.min_depth(cg));
                        new_max = std::max(new_max, pool_.max_depth(cg));
                    }
                }
                // If any candidate child is unvisited the parent is not fully expanded.
                for (uint32_t a : cands)
                {
                    const int32_t cg = sib_base + static_cast<int32_t>(a);
                    if (pool_.visit_count(cg) == 0)
                    {
                        fully_expanded = false;
                        break;
                    }
                }
                if (!fully_expanded)
                {
                    const int pd = pool_.depth(par);
                    new_min = std::min(new_min, pd);
                    new_max = std::max(new_max, pd);
                }
                if (new_min == std::numeric_limits<int>::max())
                    new_min = pool_.depth(par);

                // Early exit if nothing changed
                if (pool_.min_depth(par) == new_min && pool_.max_depth(par) == new_max)
                    break;

                pool_.min_depth(par) = new_min;
                pool_.max_depth(par) = new_max;
                cur = par;
            }
        }

        // =====================================================================
        // Early-termination
        // =====================================================================

        void DynVecQMDP::updateConvergenceTracking()
        {
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                auto &info = convergence_info_[s];
                ++info.total_checks;

                const int32_t root_g = scenario_roots_[s];
                if (pool_.visit_count(root_g) == 0)
                    continue;

                const int32_t child_b = pool_.child_base(root_g);
                if (child_b == DYN_INVALID_IDX)
                    continue;

                int   best_action = -1;
                float best_q = std::numeric_limits<float>::lowest();

                for (uint32_t a : getDynCandidateActions(root_g))
                {
                    const int32_t cg = child_b + static_cast<int32_t>(a);
                    if (pool_.visit_count(cg) > 0 && pool_.q_value(cg) > best_q)
                    {
                        best_q = pool_.q_value(cg);
                        best_action = static_cast<int>(a);
                    }
                }
                if (best_action == -1)
                    continue;

                if (best_action == info.best_action_idx)
                    ++info.stable_iterations;
                else
                {
                    info.best_action_idx = best_action;
                    info.stable_iterations = 1;
                }
                info.prev_q_value = info.best_q_value;
                info.best_q_value = best_q;
            }
        }

        bool DynVecQMDP::checkEarlyTermination()
        {
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const auto &info = convergence_info_[s];

                if (info.best_action_idx == -1 || info.stable_iterations < early_term_stable_iters_)
                    return false;

                const int32_t root_g = scenario_roots_[s];
                const int32_t child_b = pool_.child_base(root_g);
                if (child_b == DYN_INVALID_IDX)
                    return false;

                const int32_t best_child_g = child_b + info.best_action_idx;
                if (pool_.visit_count(best_child_g) < early_term_min_best_visits_)
                    return false;

                if (info.prev_q_value != std::numeric_limits<float>::lowest() && info.best_q_value != 0.0f)
                {
                    const float rel =
                        std::abs(info.best_q_value - info.prev_q_value) / std::max(std::abs(info.best_q_value), 1e-6f);
                    if (rel > early_term_q_change_thr_)
                        return false;
                }
            }
            return true;
        }

        // =====================================================================
        // Homogenous search support
        // =====================================================================

#ifdef ENABLE_HOMOGENOUS_SEARCH
        void DynVecQMDP::initHomogenousMapping()
        {
            homogenous_node_map_.clear();
            homogenous_node_map_.resize(tree_height_ + 1);
            homogenous_node_map_[0].resize(scenario_size_);

            for (uint32_t s = 0; s < scenario_size_; ++s)
                homogenous_node_map_[0][s] = scenario_roots_[s];

            homogenous_current_level_ = 0;
            scenario_global_idxs_.resize(scenario_size_);
        }

        int32_t DynVecQMDP::exploreNodesHomogenous(int &action_idx)
        {
            const auto t0 = std::chrono::steady_clock::now();

            // Always start from root (level 0) and traverse down
            int32_t cur_level = 0;
            int32_t cur_g0 = homogenous_node_map_[0][0]; // Scenario-0 root node

            // If root not yet visited
            if (pool_.visit_count(cur_g0) == 0)
            {
                const auto &cands = getDynCandidateActions(cur_g0);
                action_idx = cands.empty() ? 0 : static_cast<int>(cands[0]);

                total_select_batch_us_ +=
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                ++select_batch_call_count_;
                return 0; // Return root level
            }

            // UCB traversal down the tree from root
            int32_t cur = cur_g0;
            int     sel_action = 0;

            while (true)
            {
                const auto   &cands = getDynCandidateActions(cur);
                const int32_t child_b = pool_.child_base(cur);

                // Leaf conditions
                if (pool_.depth(cur) >= static_cast<int32_t>(tree_height_) || child_b == DYN_INVALID_IDX ||
                    pool_.active_flag(cur) == 0)
                {
                    sel_action = static_cast<int>(pool_.curr_action(cur));
                    break;
                }

                // Check for unvisited children
                bool found_unvisited = false;
                for (uint32_t a : cands)
                {
                    const int32_t cg = child_b + static_cast<int32_t>(a);
                    if (pool_.visit_count(cg) == 0)
                    {
                        sel_action = static_cast<int>(a);
                        found_unvisited = true;
                        break;
                    }
                }
                if (found_unvisited)
                    break;

                // All visited: descend via UCB to next level.
                // After selecting the best child for scenario-0, synchronize
                // homogenous_node_map_ for ALL scenarios so that the subsequent
                // expandNodesBatch() reads the correct parents at this level.
                const int32_t next_g = selectNodeDyn(cur, cands, -1);
                const int32_t taken_action = static_cast<int32_t>(pool_.curr_action(next_g));
                ++cur_level;

                // Sync map for all scenarios at the newly reached level.
                if (homogenous_node_map_.size() <= static_cast<size_t>(cur_level))
                    homogenous_node_map_.resize(cur_level + 1);
                {
                    auto &lvl_nodes = homogenous_node_map_[cur_level];
                    if (lvl_nodes.size() < scenario_size_)
                        lvl_nodes.resize(scenario_size_);
                    for (uint32_t s = 0; s < scenario_size_; ++s)
                    {
                        const int32_t par = homogenous_node_map_[cur_level - 1][s];
                        const int32_t cb = pool_.child_base(par);
                        lvl_nodes[s] = (cb != DYN_INVALID_IDX) ? cb + taken_action : par;
                    }
                }
                cur = next_g;
            }

            action_idx = sel_action;

            total_select_batch_us_ +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            ++select_batch_call_count_;

            return cur_level; // Return the level of the parent node
        }

        void DynVecQMDP::backPropagateHomogenous(int32_t child_level, const std::vector<float> &rollout_values)
        {
            const auto t0 = std::chrono::steady_clock::now();

            // Working copy of values
            std::vector<float> cur_values = rollout_values;

            // Traverse from child_level to root (level 0)
            int32_t cur_level = child_level;

            while (cur_level >= 0)
            {
                const auto &node_globals = homogenous_node_map_[cur_level];

                // Update all scenarios at this level
                for (uint32_t s = 0; s < scenario_size_; ++s)
                {
                    const int32_t node_g = node_globals[s];

                    // Accumulate reward (no extra discount factor — matches the static
                    // VecQMDP convention: the rollout value already encodes the
                    // discounted future return from the child state).
                    cur_values[s] = pool_.reward(node_g) + cur_values[s];

                    // Store initial rollout value on first visit
                    const bool first_visit = (pool_.visit_count(node_g) == 0);
                    if (first_visit)
                    {
                        pool_.initial_rollout(node_g) = cur_values[s];
                    }

                    // Update visit count and Q-value
                    const int32_t vc_old = pool_.visit_count(node_g);
                    pool_.visit_count(node_g) = vc_old + 1;

                    const float q_old = pool_.q_value(node_g);
                    pool_.q_value(node_g) = q_old + (cur_values[s] - q_old) / static_cast<float>(vc_old + 1);

                    // Persistent max-backup: Q(s) = max(Q_running_avg, initial_rollout)
                    pool_.q_value(node_g) = std::max(pool_.q_value(node_g), pool_.initial_rollout(node_g));
                }

                // Check for pruning
                bool any_pruned = false;
                for (uint32_t s = 0; s < scenario_size_; ++s)
                {
                    const int32_t node_g = node_globals[s];
                    if (cur_values[s] < pruned_threshold_ || pool_.reward(node_g) < pruned_threshold_)
                    {
                        any_pruned = true;
                        break;
                    }
                }

                if (any_pruned)
                {
                    // Increment root counts to prevent immediate re-selection
                    for (uint32_t s = 0; s < scenario_size_; ++s)
                        pool_.visit_count(scenario_roots_[s]) += 1;
                    break;
                }

                // Max-backup blending (if not at root)
                if (cur_level > 0)
                {
                    const int32_t parent_level = cur_level - 1;
                    const auto   &parent_globals = homogenous_node_map_[parent_level];

                    for (uint32_t s = 0; s < scenario_size_; ++s)
                    {
                        const int32_t par_g = parent_globals[s];
                        const int32_t sib_base = pool_.child_base(par_g);

                        if (sib_base != DYN_INVALID_IDX)
                        {
                            float max_child_q = std::numeric_limits<float>::lowest();
                            bool  found_child = false;

                            for (uint32_t a = 0; a < num_actions_; ++a)
                            {
                                const int32_t cg = sib_base + static_cast<int32_t>(a);
                                if (pool_.visit_count(cg) > 0)
                                {
                                    max_child_q = std::max(max_child_q, pool_.q_value(cg));
                                    found_child = true;
                                }
                            }

                            if (found_child)
                            {
                                cur_values[s] =
                                    (1.0f - max_backup_lambda_) * max_child_q + max_backup_lambda_ * cur_values[s];
                            }
                        }
                    }
                }

                // Move to parent level
                if (cur_level == 0)
                    break;
                --cur_level;
            }

            total_backprop_us_ +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            ++backprop_call_count_;
        }
#endif // ENABLE_HOMOGENOUS_SEARCH

        // =====================================================================
        // Performance timing stats
        // =====================================================================

        void DynVecQMDP::printTree(int max_depth, std::ostream &out) const
        {
            out << "\n";
            out << "[DynVecQMDP] Tree dump (scenario 0, max_depth=" << max_depth << ")\n";

            struct Frame
            {
                int32_t     node_g;
                int         depth;
                std::string prefix;
                bool        is_last;
            };

            std::vector<Frame> stack;
            stack.push_back({scenario_roots_.empty() ? DYN_INVALID_IDX : scenario_roots_[0], 0, "", true});

            int nodes_shown = 0;

            while (!stack.empty())
            {
                const Frame f = stack.back();
                stack.pop_back();

                const int32_t g = f.node_g;
                if (g == DYN_INVALID_IDX)
                    continue;
                if (g < 0 || g >= pool_.total_nodes())
                    continue;

                const bool is_root = (f.depth == 0);
                const bool has_visits = (pool_.visit_count(g) > 0);
                if (!is_root && !has_visits)
                    continue;

                const bool terminal = (pool_.active_flag(g) == 0);

                if (is_root)
                {
                    out << "ROOT"
                        << "  g=" << g << "  N=" << pool_.visit_count(g) << "  INITIAL_Q=" << pool_.initial_rollout(g)
                        << "  Q=" << pool_.q_value(g) << "  R=" << pool_.reward(g) << "  depth=" << pool_.depth(g)
                        << "\n";
                }
                else
                {
                    out << f.prefix << (f.is_last ? "└─ " : "├─ ") << "a=" << pool_.action_taken(g) << "  g=" << g
                        << "  N=" << pool_.visit_count(g) << "  INITIAL_Q=" << pool_.initial_rollout(g)
                        << "  Q=" << pool_.q_value(g) << "  R=" << pool_.reward(g) << "  depth=" << pool_.depth(g);
                    if (terminal)
                        out << "  [TERMINAL]";
                    out << "\n";
                }

                ++nodes_shown;

                if (f.depth >= max_depth || terminal)
                    continue;

                const int32_t child_b = pool_.child_base(g);
                if (child_b == DYN_INVALID_IDX)
                    continue; // not expanded

                // Collect visited children (visit count > 0)
                std::vector<int32_t> children;
                children.reserve(num_actions_);
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    const int32_t cg = child_b + static_cast<int32_t>(a);
                    if (cg < 0 || cg >= pool_.total_nodes())
                        continue;
                    if (pool_.visit_count(cg) > 0)
                        children.push_back(cg);
                }

                const std::string child_prefix = f.prefix + (f.is_last ? "   " : "│  ");
                for (int i = static_cast<int>(children.size()) - 1; i >= 0; --i)
                {
                    const bool child_last = (i == static_cast<int>(children.size()) - 1);
                    stack.push_back({children[static_cast<size_t>(i)], f.depth + 1, child_prefix, child_last});
                }
            }

            out << "\n[DynVecQMDP] Printed " << nodes_shown << " node(s) (visit_count > 0, depth ≤ " << max_depth
                << ")\n\n";
        }

        void DynVecQMDP::printTimingStats() const
        {
            if (num_worker_threads_ <= 1)
            {
                std::cout << std::fixed << std::setprecision(3) << "[DynVecQMDP timing] "
                          << "stepBatch: " << total_step_batch_us_ / 1000.0 << " ms "
                          << "(" << step_batch_call_count_ << " calls) | "
                          << "rollout: " << total_rollout_us_ / 1000.0 << " ms "
                          << "(" << rollout_call_count_ << " calls) | "
                          << "backprop: " << total_backprop_us_ / 1000.0 << " ms "
                          << "(" << backprop_call_count_ << " calls)\n";
            }
            else
            {
                // Aggregate across workers
                double  agg_step = 0.0, agg_rollout = 0.0, agg_backprop = 0.0;
                int64_t agg_step_ct = 0, agg_rollout_ct = 0, agg_backprop_ct = 0;

                for (const auto &w : worker_instances_)
                {
                    auto *dw = dynamic_cast<const DynVecQMDP *>(w.get());
                    if (dw)
                    {
                        agg_step += dw->total_step_batch_us_;
                        agg_rollout += dw->total_rollout_us_;
                        agg_backprop += dw->total_backprop_us_;
                        agg_step_ct += dw->step_batch_call_count_;
                        agg_rollout_ct += dw->rollout_call_count_;
                        agg_backprop_ct += dw->backprop_call_count_;
                    }
                }

                std::cout << std::fixed << std::setprecision(3) << "[DynVecQMDP timing (aggregated)] "
                          << "stepBatch: " << agg_step / 1000.0 << " ms | "
                          << "rollout: " << agg_rollout / 1000.0 << " ms | "
                          << "backprop: " << agg_backprop / 1000.0 << " ms\n";
            }
        }

    } // namespace planning
} // namespace vec_qmdp
