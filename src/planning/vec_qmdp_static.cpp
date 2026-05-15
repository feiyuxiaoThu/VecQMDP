/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file vec_qmdp_static.cpp
 * @brief Generic BeliefTreeSearch tree-search kernel — VecQMDP implementation.
 *
 * All structural values (tree shape, thresholds, …) come from protected
 * member variables set by the derived class.  No utils/ constants are
 * referenced here.
 */
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <planning/vec_qmdp_static.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        using AlignedVectorInt = utils::AlignedVectorInt;
        // -----------------------------------------------------------------------
        // Construction / Initialisation
        // -----------------------------------------------------------------------

        VecQMDP::VecQMDP(int num_scenarios, uint32_t num_actions, uint32_t tree_height)
            : num_actions_(num_actions), tree_height_(tree_height)
        {
            // Compute per-layer node counts: layer k has num_actions_^k nodes
            tree_node_sizes_per_depth_.resize(tree_height_ + 1);
            uint32_t power = 1;
            uint32_t total = 0;
            for (uint32_t k = 0; k <= tree_height_; ++k)
            {
                tree_node_sizes_per_depth_[k] = power;
                total += power;
                power *= num_actions_;
            }
            tree_node_size_ = total;

            assert(num_scenarios % IVectorT_qmdp::num_scalars == 0);
            scenario_size_ = static_cast<uint32_t>(num_scenarios);
            node_size_ = tree_node_size_ * scenario_size_;

            initializeTreeStructures();
        }

        // -----------------------------------------------------------------------
        // Parallel infrastructure
        // -----------------------------------------------------------------------
        void VecQMDP::initParallelInfrastructure(size_t num_threads)
        {
            num_worker_threads_ = std::max<size_t>(1, num_threads);

            aggregated_action_values_.assign(num_actions_, 0.0f);
            aggregated_action_counts_.assign(num_actions_, 0);

            if (num_worker_threads_ == 1)
            {
                // Single-threaded path: no thread pool needed; the instance itself
                // acts as the sole worker.
                worker_instances_.clear();
                thread_pool_.reset();
                return;
            }

            thread_pool_ = std::make_shared<utils::ThreadPool>(num_worker_threads_);

            worker_instances_.clear();
            worker_instances_.reserve(num_worker_threads_);
            for (size_t i = 0; i < num_worker_threads_; ++i)
                worker_instances_.push_back(makeWorkerInstance());
        }

        void VecQMDP::dispatchParallelSearch(std::function<BeliefTreeSearchThreadResult(VecQMDP *, size_t)> search_fn)
        {
            // Reset aggregation buffers.
            std::fill(aggregated_action_values_.begin(), aggregated_action_values_.end(), 0.0f);
            std::fill(aggregated_action_counts_.begin(), aggregated_action_counts_.end(), 0);
            aggregated_simulation_count_ = 0;

            if (num_worker_threads_ <= 1 || worker_instances_.empty())
            {
                // Fallback: run on the current instance itself (thread 0).
                auto result = search_fn(this, 0);
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    aggregated_action_values_[a] += result.action_sum_values[a];
                    aggregated_action_counts_[a] += result.action_counts[a];
                }
                aggregated_simulation_count_ += result.simulation_count;
            }
            else
            {
                // Submit one task per worker to the thread pool.
                std::vector<std::future<BeliefTreeSearchThreadResult>> futures;
                futures.reserve(num_worker_threads_);
                for (size_t i = 0; i < num_worker_threads_; ++i)
                {
                    VecQMDP *worker_ptr = worker_instances_[i].get();
                    futures.emplace_back(
                        thread_pool_->enqueue([worker_ptr, i, &search_fn]() { return search_fn(worker_ptr, i); }));
                }

                // Collect and aggregate.
                for (auto &f : futures)
                {
                    auto result = f.get();
                    for (uint32_t a = 0; a < num_actions_; ++a)
                    {
                        aggregated_action_values_[a] += result.action_sum_values[a];
                        aggregated_action_counts_[a] += result.action_counts[a];
                    }
                    aggregated_simulation_count_ += result.simulation_count;
                }
            }

            // Convert sums to per-action averages.
            for (uint32_t a = 0; a < num_actions_; ++a)
            {
                if (aggregated_action_counts_[a] > 0)
                    aggregated_action_values_[a] /= static_cast<float>(aggregated_action_counts_[a]);
                else
                    aggregated_action_values_[a] = utils::ACTION_VALUE_INITIAL_MIN;
            }
        }

        // -----------------------------------------------------------------------
        // Construction / Initialisation
        // -----------------------------------------------------------------------

        void VecQMDP::initializeTreeStructures()
        {
            node_active_flags_.resize(node_size_, 0xFFFFFFFF);
            node_curr_action_idxs_.resize(node_size_, 0);

            q_node_values_.resize(node_size_, 0.0f);
            node_rewards_.resize(node_size_, 0.0f);
            node_count_.resize(node_size_, 0);
            node_depth_.resize(node_size_, 0);
            node_rollout_len_.resize(node_size_, 0);
            node_min_depth_.resize(node_size_, static_cast<int>(tree_height_) + 1);
            node_max_depth_.resize(node_size_, -1);
            node_initial_rollout_.resize(node_size_, std::numeric_limits<float>::lowest());
            node_candidate_actions_.resize(node_size_);
            node_actions_computed_.resize(node_size_, false);
            tree_convergence_info_.resize(scenario_size_);

            // Precompute per-scenario memory offsets (invariant after construction)
            scen_offsets_.resize(scenario_size_);
            for (uint32_t s = 0; s < scenario_size_; ++s)
                scen_offsets_[s] = static_cast<int>(s * tree_node_size_);

            // Pre-fill depth and rollout-length for every node in every tree
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const uint32_t root = s * tree_node_size_;
                uint32_t       start_node = root;
                uint32_t       end_node = root;

                for (int layer = 0; layer <= static_cast<int>(tree_height_); ++layer)
                {
                    start_node += (layer > 0) ? tree_node_sizes_per_depth_[layer - 1] : 0;
                    end_node += tree_node_sizes_per_depth_[layer];

                    std::fill(node_depth_.begin() + start_node, node_depth_.begin() + end_node, layer);
                    std::fill(node_rollout_len_.begin() + start_node, node_rollout_len_.begin() + end_node,
                              static_cast<int>(tree_height_) - layer - 1);
                }
            }
        }

        void VecQMDP::initUCBTables()
        {
            ucb_coeff_table_.resize(UCB_TABLE_SIZE, 0.0f);
            inv_sqrt_table_.resize(UCB_TABLE_SIZE, 0.0f);
            // n = 0: undefined (log(0) = -inf); keep 0 so these nodes win UCB max check
            // naturally (they appear as "unvisited" in the count==0 check instead).
            ucb_coeff_table_[0] = 0.0f;
            inv_sqrt_table_[0] = 0.0f;
            for (uint32_t n = 1; n < UCB_TABLE_SIZE; ++n)
            {
                const float fn = static_cast<float>(n);
                ucb_coeff_table_[n] = exploration_constant_ * std::sqrt(std::log(fn));
                inv_sqrt_table_[n] = 1.0f / std::sqrt(fn);
            }
        }

        void VecQMDP::resetTreeStructures()
        {
            std::fill(node_count_.begin(), node_count_.end(), 0);
            std::fill(q_node_values_.begin(), q_node_values_.end(), utils::ACTION_VALUE_INITIAL_MIN);
            std::fill(node_rewards_.begin(), node_rewards_.end(), 0.0f);
            std::fill(node_min_depth_.begin(), node_min_depth_.end(), static_cast<int>(tree_height_) + 1);
            std::fill(node_max_depth_.begin(), node_max_depth_.end(), -1);
            std::fill(node_initial_rollout_.begin(), node_initial_rollout_.end(), std::numeric_limits<float>::lowest());

            for (auto &v : node_candidate_actions_)
                v.clear();
            std::fill(node_actions_computed_.begin(), node_actions_computed_.end(), false);

            for (auto &info : tree_convergence_info_)
            {
                info.best_action_idx = -1;
                info.best_q_value = std::numeric_limits<float>::lowest();
                info.stable_iterations = 0;
                info.prev_q_value = std::numeric_limits<float>::lowest();
                info.total_checks = 0;
            }
            depth_difference_count.clear();
        }

        // -----------------------------------------------------------------------
        // Node Selection (UCB)
        // -----------------------------------------------------------------------

        float VecQMDP::calculateUCB(uint32_t node_idx, uint32_t parent_index, int target_depth)
        {
            if (node_count_[node_idx] == 0)
                return std::numeric_limits<float>::max();

            const float exploitation = q_node_values_[node_idx];
            const float exploration =
                exploration_constant_ * std::sqrt(std::log(static_cast<float>(node_count_[parent_index])) /
                                                  static_cast<float>(node_count_[node_idx]));
            float score = exploitation + exploration;

            if (target_depth != -1)
            {
                const int min_gap = node_min_depth_[node_idx];
                const int max_gap = node_max_depth_[node_idx];
                const int eff_gap = (target_depth <= min_gap)   ? min_gap
                                    : (target_depth >= max_gap) ? max_gap
                                                                : target_depth;
                score -= depth_sync_lambda_ * std::abs(static_cast<float>(eff_gap) - static_cast<float>(target_depth));
            }
            return score;
        }

        uint32_t VecQMDP::selectNode(uint32_t parent_relative_index, const std::vector<uint32_t> &candidate_actions,
                                     uint32_t scenario_idx, int target_depth)
        {
            const uint32_t global_parent = getNodeIdx(parent_relative_index, scenario_idx);
            const uint32_t child_rel_start = num_actions_ * parent_relative_index + 1;
            uint32_t       best_child = 0;
            float          best_ucb = -std::numeric_limits<float>::max();

            for (uint32_t a : candidate_actions)
            {
                const uint32_t child_rel = child_rel_start + a;
                if (child_rel >= tree_node_size_)
                    return child_rel;

                const uint32_t child_global = getNodeIdx(child_rel, scenario_idx);
                const float    ucb = calculateUCB(child_global, global_parent, target_depth);
                if (ucb > best_ucb)
                {
                    best_ucb = ucb;
                    best_child = child_rel;
                }
            }
            return best_child;
        }

        // -----------------------------------------------------------------------
        // Exploration — Majority-vote depth synchronisation
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // exploreNodes — SIMD-parallel UCB tree traversal
        //
        // Key optimisations vs the old scalar version:
        //  1. UCB coefficients come from precomputed tables (no log/sqrt per call).
        //  2. node_count_ and q_node_values_ are gathered for ALL scenarios in one
        //     AVX2 _mm256_i32gather instruction, so 8 cache-miss requests fire in
        //     parallel instead of sequentially.
        //  3. UCB arithmetic (multiply + add) runs as SIMD float ops across all 8
        //     scenarios simultaneously.
        //  4. scen_offsets_ is precomputed once at construction time.
        //
        // Correctness invariant: at the start of every while-loop iteration all
        // *active* (not-yet-done) scenarios are at the same tree depth, so the
        // candidate-action list obtained from the representative scenario is valid
        // for every active scenario.
        // -----------------------------------------------------------------------
        AlignedVectorInt VecQMDP::exploreNodes(std::vector<int> &target_action_idxs)
        {
            // const auto t0 = std::chrono::steady_clock::now();

            constexpr uint32_t BATCH = static_cast<uint32_t>(IVectorT_qmdp::num_scalars); // 8

            AlignedVectorInt result_nodes(scenario_size_, 0);
            target_action_idxs.assign(scenario_size_, 0);

            // Process scenarios in SIMD-width batches so that scenario_size_ is not
            // limited to IVectorT_qmdp::num_scalars (= 8).
            for (uint32_t batch_start = 0; batch_start < scenario_size_; batch_start += BATCH)
            {
                const uint32_t batch_end = std::min(batch_start + BATCH, scenario_size_);
                const uint32_t valid_num = batch_end - batch_start;

                // Build per-batch scenario-offset array.  Lanes beyond valid_num are
                // padded with the last valid offset so SIMD gathers stay in-bounds;
                // those lanes are marked done immediately below.
                alignas(32) int32_t batch_off[BATCH];
                for (uint32_t i = 0; i < BATCH; ++i)
                {
                    const uint32_t s = (i < valid_num) ? (batch_start + i) : (batch_end - 1);
                    batch_off[i] = static_cast<int32_t>(scen_offsets_[s]);
                }
                const IVectorT_qmdp scen_off_v(batch_off);

                // Per-scenario mutable state for this batch.
                alignas(32) int32_t father_rel_arr[BATCH] = {};
                AlignedVectorInt    explore_nodes(BATCH, 0);
                AlignedVectorInt    tmp_actions(BATCH, 0);
                alignas(32) int32_t done_arr[BATCH] = {};
                int                 num_done = 0;

                // ---- Initialise: check whether each scenario root has been visited ----
                {
                    const IVectorT_qmdp root_count_v = IVectorT_qmdp::gather(node_count_.data(), scen_off_v);

                    alignas(32) int32_t rc[BATCH];
                    root_count_v.to_array(rc);

                    for (uint32_t i = 0; i < BATCH; ++i)
                    {
                        if (i >= valid_num)
                        {
                            // Padding lane — mark done immediately.
                            done_arr[i] = -1;
                            ++num_done;
                            continue;
                        }
                        explore_nodes[i] = batch_off[i];
                        tmp_actions[i] = node_curr_action_idxs_[batch_off[i]];
                        if (rc[i] == 0)
                        {
                            done_arr[i] = -1;
                            ++num_done;
                        }
                    }
                }

                // ---- Main traversal loop (one depth level per iteration) ----
                while (num_done < static_cast<int>(BATCH))
                {
                    const IVectorT_qmdp father_rel_v(father_rel_arr);
                    const IVectorT_qmdp father_glob_v = father_rel_v + scen_off_v;
                    const IVectorT_qmdp cs_v = father_rel_v * static_cast<int32_t>(num_actions_) + 1;

                    const IVectorT_qmdp parent_count_v = IVectorT_qmdp::gather(node_count_.data(), father_glob_v);
                    const FVectorT_qmdp ucb_coeff_v = FVectorT_qmdp::gather(ucb_coeff_table_.data(), parent_count_v);

                    alignas(32) float   best_ucb_arr[BATCH];
                    alignas(32) int32_t best_crel_arr[BATCH];
                    for (uint32_t i = 0; i < BATCH; ++i)
                    {
                        best_ucb_arr[i] = -std::numeric_limits<float>::max();
                        best_crel_arr[i] = father_rel_arr[i];
                    }

                    alignas(32) int32_t fg_arr[BATCH];
                    father_glob_v.to_array(fg_arr);

                    // Candidate actions from the first undone scenario (all active
                    // scenarios are at the same depth, so the list is identical).
                    uint32_t rep = 0;
                    while (done_arr[rep])
                        ++rep;
                    const uint32_t rep_glob =
                        static_cast<uint32_t>(father_rel_arr[rep]) + static_cast<uint32_t>(batch_off[rep]);
                    const auto &cands = getNodeCandidateActions(rep_glob);

                    for (uint32_t a : cands)
                    {
                        if (num_done >= static_cast<int>(BATCH))
                            break;

                        const IVectorT_qmdp child_rel_v = cs_v + static_cast<int32_t>(a);
                        const IVectorT_qmdp oob_v = (child_rel_v >= static_cast<int32_t>(tree_node_size_));

                        const IVectorT_qmdp child_glob_v = child_rel_v + scen_off_v;
                        const IVectorT_qmdp safe_cg_v =
                            IVectorT_qmdp::select(oob_v, IVectorT_qmdp::fill(0), child_glob_v);

                        IVectorT_qmdp child_count_v = IVectorT_qmdp::gather(node_count_.data(), safe_cg_v);
                        child_count_v = IVectorT_qmdp::select(oob_v, IVectorT_qmdp::fill(-1), child_count_v);

                        const FVectorT_qmdp child_q_v = FVectorT_qmdp::gather(q_node_values_.data(), safe_cg_v);

                        const IVectorT_qmdp cc_safe_v = child_count_v.max(IVectorT_qmdp::fill(1));
                        const FVectorT_qmdp inv_sqrt_v = FVectorT_qmdp::gather(inv_sqrt_table_.data(), cc_safe_v);
                        const FVectorT_qmdp ucb_v = child_q_v + ucb_coeff_v * inv_sqrt_v;

                        alignas(32) int32_t oob_arr[BATCH], cc_arr[BATCH], cr_arr[BATCH];
                        alignas(32) float   ucb_s[BATCH];
                        oob_v.to_array(oob_arr);
                        child_count_v.to_array(cc_arr);
                        child_rel_v.to_array(cr_arr);
                        ucb_v.to_array(ucb_s);

                        for (uint32_t i = 0; i < BATCH; ++i)
                        {
                            if (done_arr[i])
                                continue;

                            if (oob_arr[i])
                            {
                                tmp_actions[i] = node_curr_action_idxs_[fg_arr[i]];
                                explore_nodes[i] = fg_arr[i];
                                done_arr[i] = -1;
                                ++num_done;
                                continue;
                            }

                            if (cc_arr[i] == 0)
                            {
                                tmp_actions[i] = static_cast<int>(a);
                                explore_nodes[i] = fg_arr[i];
                                done_arr[i] = -1;
                                ++num_done;
                                continue;
                            }

                            if (cc_arr[i] > 0 && ucb_s[i] > best_ucb_arr[i])
                            {
                                best_ucb_arr[i] = ucb_s[i];
                                best_crel_arr[i] = cr_arr[i];
                            }
                        }
                    }

                    for (uint32_t i = 0; i < BATCH; ++i)
                        if (!done_arr[i])
                            father_rel_arr[i] = best_crel_arr[i];
                }

                // Copy batch results into the output vectors.
                for (uint32_t i = 0; i < valid_num; ++i)
                {
                    result_nodes[batch_start + i] = explore_nodes[i];
                    target_action_idxs[batch_start + i] = tmp_actions[i];
                }
            }

            // total_select_batch_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++select_batch_call_count_;
            return result_nodes;
        }

#ifdef ENABLE_HOMOGENOUS_SEARCH
        // -----------------------------------------------------------------------
        // exploreNodesHomogenous — single-scenario UCB traversal
        //
        // Explores only scenario-0's subtree.  Because all scenarios share the
        // same tree path under ENABLE_HOMOGENOUS_SEARCH, the relative node index
        // and action found for scenario 0 are valid for every other scenario.
        // The caller can derive scenario-s's global node index as:
        //   global_s = returned_relative_idx + s * tree_node_size_
        // -----------------------------------------------------------------------
        uint32_t VecQMDP::exploreNodesHomogenous(int &action_idx)
        {
#ifdef PRINT_TIME
            const auto t0 = std::chrono::steady_clock::now();
#endif

            // Scenario-0: global index == relative index (offset = 0).
            uint32_t father_rel = 0;
            int      sel_action = node_curr_action_idxs_[0];

            if (node_count_[0] == 0)
            {
                // Root unvisited — expand from root.
                action_idx = sel_action;
#ifdef PRINT_TIME
                total_select_batch_us_ +=
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                ++select_batch_call_count_;
#endif
                return 0u;
            }

            while (true)
            {
                // For scenario 0, global == relative.
                const uint32_t father_glob = father_rel;
                const auto    &cands = getNodeCandidateActions(father_glob);
                const uint32_t cs = num_actions_ * father_rel + 1;

                const float ucb_coeff = ucb_coeff_table_[node_count_[father_glob]];

                int   best_crel = static_cast<int>(father_rel);
                float best_ucb = -std::numeric_limits<float>::max();
                bool  found = false;

                for (uint32_t a : cands)
                {
                    const uint32_t cr = cs + a;

                    if (cr >= tree_node_size_)
                    {
                        // OOB — tree boundary reached; stay at current parent.
                        sel_action = node_curr_action_idxs_[father_glob];
                        found = true;
                        break;
                    }

                    if (node_count_[cr] == 0)
                    {
                        // Unvisited child — expand here.
                        sel_action = static_cast<int>(a);
                        found = true;
                        break;
                    }

                    // Visited child — compute UCB and track best.
                    const float ucb = q_node_values_[cr] + ucb_coeff * inv_sqrt_table_[node_count_[cr]];
                    if (ucb > best_ucb)
                    {
                        best_ucb = ucb;
                        best_crel = static_cast<int>(cr);
                    }
                }

                if (found)
                    break;

                // All children visited — descend to the best UCB child.
                father_rel = static_cast<uint32_t>(best_crel);
            }

            action_idx = sel_action;
#ifdef PRINT_TIME
            total_select_batch_us_ +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            ++select_batch_call_count_;
#endif
            return father_rel; // scenario-0 global == relative
        }

        // -----------------------------------------------------------------------
        // backPropagateHomogenous — SIMD-vectorised batch back-propagation
        //
        // All scenarios share the same relative tree path (ENABLE_HOMOGENOUS_SEARCH
        // guarantee), so we process them in SIMD batches of width BATCH =
        // FVectorT_qmdp::num_scalars rather than one-by-one.
        //
        // Per-level work (fully vectorised):
        //   1. gather node_rewards for all BATCH scenarios simultaneously.
        //   2. Accumulate into cv_v (running per-scenario return).
        //   3. Masked count++ and Q running-mean update (mask = ~pruned_v).
        //   4. Update cumulative pruned bitmask: prune if cv < thr OR reward < thr.
        //   5. Max-backup blend over all children of parent: gather child counts
        //      and Q-values, build max/found masks, select-blend cv_v.
        //   6. scatter updated counts and Q-values back; store cv_v / pruned_v.
        //
        // Root update: all scenarios increment count; active only update Q.
        //
        // Pruning semantics match backPropagate():
        //   - The node that triggers pruning IS counted and has Q updated.
        //   - All strictly-ancestor nodes are skipped for that scenario.
        //   - Root count is incremented even for pruned scenarios.
        //
        // A scalar fallback handles the remainder when scenario_size_ % BATCH != 0.
        // -----------------------------------------------------------------------
        void VecQMDP::backPropagateHomogenous(uint32_t node_idx, AlignedVectorFloat &rollout_values)
        {
#ifdef PRINT_TIME
            const auto t0 = std::chrono::steady_clock::now();
#endif

            constexpr uint32_t BATCH = static_cast<uint32_t>(FVectorT_qmdp::num_scalars);
            const uint32_t     full_batches = scenario_size_ / BATCH;
            const uint32_t     simd_size = full_batches * BATCH;

            // Per-scenario pruned bitmask stored as float lanes:
            //   0x00000000 (0.0f)      = active
            //   0xFFFFFFFF (SIMD mask) = pruned
            // Scalar remainder path uses 1.0f as the pruned sentinel (also != 0.0f).
            AlignedVectorInt pruned_flags(scenario_size_, 0);

            // ----------------------------------------------------------------
            // Main traversal: node_idx → root (exclusive).
            // ----------------------------------------------------------------
            uint32_t cur = node_idx;
            while (cur > 0)
            {
                const uint32_t parent = (cur - 1) / num_actions_;
                const uint32_t cs = parent * num_actions_ + 1; // first child rel-idx

                // ---- SIMD path: full-width batches ----
                for (uint32_t b = 0; b < full_batches; ++b)
                {
                    const int32_t bs = b * BATCH;

                    // scen_offsets_[s] = s * tree_node_size_; load BATCH of them.
                    const IVectorT_qmdp scen_off_v = IVectorT_qmdp::load_contiguous(scen_offsets_.data(), bs);

                    // Global node indices: g[i] = cur + scen_offsets_[bs+i].
                    const IVectorT_qmdp g_v = cur + scen_off_v;

                    // Load working state for this batch.
                    FVectorT_qmdp cv_v = FVectorT_qmdp::load_contiguous(rollout_values.data(), bs);
                    FVectorT_qmdp pruned_v =
                        IVectorT_qmdp::load_contiguous(pruned_flags.data(), bs).template as<FVectorT_qmdp>();

                    // Active mask entering this level:
                    //   0xFFFFFFFF where NOT yet pruned, 0x0 where already pruned.
                    const FVectorT_qmdp active_f = ~pruned_v;

                    // 1. Accumulate node reward (all lanes; pruned values unused).
                    const FVectorT_qmdp rewards_v = FVectorT_qmdp::gather(node_rewards_.data(), g_v);
                    cv_v = cv_v + rewards_v;

                    // 2. Count++ and Q running-mean — active scenarios only.
                    //    Pruned lanes write back the unchanged count/Q via select.
                    const IVectorT_qmdp counts_old_v = IVectorT_qmdp::gather(node_count_.data(), g_v);

                    // Check for first visit: store initial rollout value
                    const FVectorT_qmdp first_visit_f = (counts_old_v == 0).template as<FVectorT_qmdp>();
                    FVectorT_qmdp       init_rollout_v = FVectorT_qmdp::gather(node_initial_rollout_.data(), g_v);
                    // Update initial rollout only where first_visit AND active
                    init_rollout_v = FVectorT_qmdp::select(first_visit_f & active_f, cv_v, init_rollout_v);
                    init_rollout_v.scatter(node_initial_rollout_.data(), g_v);

                    const IVectorT_qmdp counts_new_v =
                        IVectorT_qmdp::select(active_f.template as<IVectorT_qmdp>(), counts_old_v + 1, counts_old_v);
                    counts_new_v.scatter(node_count_.data(), g_v);

                    FVectorT_qmdp q_v = FVectorT_qmdp::gather(q_node_values_.data(), g_v);
                    // Clamp denominator to ≥1 so pruned lanes (count may be 0)
                    // don't produce inf/nan — the result is discarded by select anyway.
                    const FVectorT_qmdp cnt_f = counts_new_v.max(1).template convert<FVectorT_qmdp>();
                    // On first visit, assign cv_v directly to avoid catastrophic cancellation:
                    // q_init = -10^7, so (cv - q_init) ~ 10^7 where float32 ULP=1, losing decimal precision.
                    FVectorT_qmdp new_q_v =
                        FVectorT_qmdp::select(first_visit_f & active_f, cv_v, q_v + (cv_v - q_v) / cnt_f);

                    // Persistent max-backup: Q(s) = max(Q_running_avg, initial_rollout)
                    new_q_v = new_q_v.max(init_rollout_v);

                    FVectorT_qmdp::select(active_f, new_q_v, q_v).scatter(q_node_values_.data(), g_v);

                    // 3. Update cumulative pruned mask.
                    //    Once pruned, always pruned (bitwise OR accumulation).
                    const FVectorT_qmdp thr_v = FVectorT_qmdp::fill(pruned_threshold_);
                    pruned_v = pruned_v | (cv_v < thr_v) | (rewards_v < thr_v);

                    // 4. Max-backup blend for scenarios still active after pruning.
                    const FVectorT_qmdp active2_f = ~pruned_v;
                    FVectorT_qmdp       max_q_v = FVectorT_qmdp::fill(-std::numeric_limits<float>::max());
                    FVectorT_qmdp       found_v = FVectorT_qmdp::fill(0.0f);

                    for (uint32_t a = 0; a < num_actions_; ++a)
                    {
                        const IVectorT_qmdp gci_v = cs + a + scen_off_v;
                        const IVectorT_qmdp child_cnt = IVectorT_qmdp::gather(node_count_.data(), gci_v);
                        // has_v: 0xFFFFFFFF where child has been visited.
                        const FVectorT_qmdp has_v = (child_cnt > 0).template as<FVectorT_qmdp>();
                        const FVectorT_qmdp child_q_v = FVectorT_qmdp::gather(q_node_values_.data(), gci_v);
                        // Keep max only over visited children.
                        max_q_v = max_q_v.max(child_q_v);
                        found_v = found_v | has_v;
                    }

                    // Blend: cv = (1−λ)·max_q + λ·cv, only where active AND found.
                    cv_v = FVectorT_qmdp::select(
                        active2_f & found_v, (1.0f - max_backup_lambda_) * max_q_v + max_backup_lambda_ * cv_v, cv_v);

                    // 5. Write back working state.
                    cv_v.to_array(rollout_values.data() + bs);
                    pruned_v.template as<IVectorT_qmdp>().to_array(pruned_flags.data() + bs);
                }

                cur = parent;
            }

            // ----------------------------------------------------------------
            // Root update (cur == 0):
            //   - All scenarios  : increment root visit count, update root Q-value.
            // ----------------------------------------------------------------

            // SIMD path.
            for (uint32_t b = 0; b < full_batches; ++b)
            {
                const uint32_t bs = b * BATCH;

                const IVectorT_qmdp g_v = IVectorT_qmdp::load_contiguous(scen_offsets_.data(), bs);
                const FVectorT_qmdp cv_v = FVectorT_qmdp::load_contiguous(rollout_values.data(), bs);

                // Check for first visit and store initial rollout
                const IVectorT_qmdp counts_old_v = IVectorT_qmdp::gather(node_count_.data(), g_v);
                const FVectorT_qmdp first_visit_f = (counts_old_v == 0).template as<FVectorT_qmdp>();
                FVectorT_qmdp       init_rollout_v = FVectorT_qmdp::gather(node_initial_rollout_.data(), g_v);
                init_rollout_v = FVectorT_qmdp::select(first_visit_f, cv_v, init_rollout_v);
                init_rollout_v.scatter(node_initial_rollout_.data(), g_v);

                // increment root count.
                const IVectorT_qmdp counts_v = counts_old_v + 1;
                counts_v.scatter(node_count_.data(), g_v);

                // update root Q.
                FVectorT_qmdp q_v = FVectorT_qmdp::gather(q_node_values_.data(), g_v);
                // On first visit, assign cv_v directly to avoid catastrophic cancellation.
                FVectorT_qmdp new_q_v = FVectorT_qmdp::select(
                    first_visit_f, cv_v, q_v + (cv_v - q_v) / counts_v.template convert<FVectorT_qmdp>());

                // Persistent max-backup: Q(s) = max(Q_running_avg, initial_rollout)
                new_q_v = new_q_v.max(init_rollout_v);

                new_q_v.scatter(q_node_values_.data(), g_v);
            }

            // Scalar fallback.
            for (uint32_t s = simd_size; s < scenario_size_; ++s)
            {
                const uint32_t g = scen_offsets_[s];

                // Store initial rollout on first visit
                if (node_count_[g] == 0)
                {
                    node_initial_rollout_[g] = rollout_values[s];
                }

                ++node_count_[g];
                q_node_values_[g] += (rollout_values[s] - q_node_values_[g]) / static_cast<float>(node_count_[g]);

                // Persistent max-backup
                q_node_values_[g] = std::max(q_node_values_[g], node_initial_rollout_[g]);
            }

#ifdef PRINT_TIME
            total_backprop_us_ +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            ++backprop_call_count_;
#endif
        }
#endif // ENABLE_HOMOGENOUS_SEARCH

        // -----------------------------------------------------------------------
        // Exploration — Majority-vote depth synchronisation
        // -----------------------------------------------------------------------

        AlignedVectorInt VecQMDP::exploreNodesVote(IVectorT_qmdp &target_action_idxs)
        {
            AlignedVectorInt explore_nodes(scenario_size_, 0);
            std::vector<int> sel_depths(scenario_size_, 0);
            AlignedVectorInt tmp_actions(scenario_size_, 0);

            int  first_depth = -1;
            bool all_same_depth = true;

            // ---- Phase 1: Standard UCB probe (no depth-sync penalty) ----
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                uint32_t father_rel = 0;
                uint32_t father_glob = getNodeIdx(0, s);
                uint32_t sel_action = static_cast<uint32_t>(node_curr_action_idxs_[father_glob]);

                if (node_count_[father_glob] == 0)
                {
                    explore_nodes[s] = father_glob;
                }
                else
                {
                    bool found = false;
                    while (!found)
                    {
                        const auto    &cands = getNodeCandidateActions(father_glob);
                        const uint32_t cs = num_actions_ * father_rel + 1;

                        for (uint32_t a : cands)
                        {
                            const uint32_t cr = cs + a;
                            if (cr >= tree_node_size_)
                            {
                                sel_action = static_cast<uint32_t>(node_curr_action_idxs_[father_glob]);
                                found = true;
                                break;
                            }
                            if (node_count_[getNodeIdx(cr, s)] == 0)
                            {
                                sel_action = a;
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                        {
                            father_rel = selectNode(father_rel, cands, s, -1);
                            father_glob = getNodeIdx(father_rel, s);
                        }
                    }
                    explore_nodes[s] = father_glob;
                }

                sel_depths[s] = static_cast<int>(node_depth_[explore_nodes[s]]);
                tmp_actions[s] = static_cast<int>(sel_action);

                if (s == 0)
                {
                    first_depth = sel_depths[0];
                }
                else if (all_same_depth && sel_depths[s] != first_depth)
                {
                    all_same_depth = false;
                }
            }

            if (all_same_depth)
            {
                target_action_idxs = IVectorT_qmdp(tmp_actions.data());
                return explore_nodes;
            }

            // ---- Phase 2: Majority-vote to choose a sync depth ----
            std::vector<int> depth_votes(tree_height_ + 1, 0);
            for (int d : sel_depths)
                depth_votes[d]++;

            int target_sync = 0, max_votes = -1;
            for (int d = 0; d <= static_cast<int>(tree_height_); ++d)
            {
                const int cnt = depth_votes[d];
                if (cnt == 0)
                    continue;

                if (d == static_cast<int>(tree_height_))
                {
                    // leaf depth: last-resort only
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
                {
                    target_sync = d;
                }
            }

            // ---- Phase 3: Re-align mismatched scenarios ----
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                if (sel_depths[s] != target_sync)
                {
                    uint32_t father_rel = 0;
                    uint32_t father_glob = getNodeIdx(0, s);
                    uint32_t sel_action = static_cast<uint32_t>(node_curr_action_idxs_[father_glob]);

                    if (node_count_[father_glob] > 0)
                    {
                        bool found = false;
                        while (!found)
                        {
                            const auto    &cands = getNodeCandidateActions(father_glob);
                            const uint32_t cs = num_actions_ * father_rel + 1;

                            for (uint32_t a : cands)
                            {
                                const uint32_t cr = cs + a;
                                if (cr >= tree_node_size_)
                                {
                                    sel_action = static_cast<uint32_t>(node_curr_action_idxs_[father_glob]);
                                    found = true;
                                    break;
                                }
                                if (node_count_[getNodeIdx(cr, s)] == 0)
                                {
                                    sel_action = a;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                father_rel = selectNode(father_rel, cands, s, target_sync);
                                father_glob = getNodeIdx(father_rel, s);
                            }
                        }
                    }
                    explore_nodes[s] = father_glob;
                    tmp_actions[s] = static_cast<int>(sel_action);
                }
            }

            target_action_idxs = IVectorT_qmdp(tmp_actions.data());
            return explore_nodes;
        }

        // -----------------------------------------------------------------------
        // Back-propagation
        // -----------------------------------------------------------------------

        void VecQMDP::backPropagate(uint32_t node_idx, uint32_t scenario_idx, float rollout_value)
        {
            // const auto t0 = std::chrono::steady_clock::now();
            uint32_t cur = node_idx;
            uint32_t global_cur = getNodeIdx(cur, scenario_idx);
            bool     pruned = false;

            while (cur > 0)
            {
                rollout_value += node_rewards_[global_cur];

                // Store initial rollout value on first visit
                const bool first_visit = (node_count_[global_cur] == 0);
                if (first_visit)
                {
                    node_initial_rollout_[global_cur] = rollout_value;
                }

                ++node_count_[global_cur];
                // On first visit, assign directly to avoid catastrophic cancellation:
                // q_init = -10^7, so (rollout_value - q_init) ~ 10^7 where float32 ULP=1.
                if (node_count_[global_cur] == 1)
                    q_node_values_[global_cur] = rollout_value;
                else
                    q_node_values_[global_cur] +=
                        (rollout_value - q_node_values_[global_cur]) / static_cast<float>(node_count_[global_cur]);

                // Persistent max-backup: Q(s) = max(initial_rollout, current_Q)
                q_node_values_[global_cur] = std::max(q_node_values_[global_cur], node_initial_rollout_[global_cur]);

                if (rollout_value < pruned_threshold_ || node_rewards_[global_cur] < pruned_threshold_)
                {
                    pruned = true;
                    break;
                }

                const uint32_t parent = (cur - 1) / num_actions_;
                const uint32_t global_par = getNodeIdx(parent, scenario_idx);

                float max_child_q = std::numeric_limits<float>::lowest();
                bool  found_child = false;
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    const uint32_t ci = parent * num_actions_ + 1 + a;
                    const uint32_t gci = getNodeIdx(ci, scenario_idx);
                    if (node_count_[gci] > 0)
                    {
                        max_child_q = std::max(max_child_q, q_node_values_[gci]);
                        found_child = true;
                    }
                }
                if (found_child)
                    rollout_value = (1.0f - max_backup_lambda_) * max_child_q + max_backup_lambda_ * rollout_value;

                cur = parent;
                global_cur = global_par;
            }

            if (!pruned)
            {
                // Root update: store initial rollout on first visit
                const bool first_visit = (node_count_[global_cur] == 0);
                if (first_visit)
                {
                    node_initial_rollout_[global_cur] = rollout_value;
                }

                ++node_count_[global_cur];
                // On first visit, assign directly to avoid catastrophic cancellation.
                if (node_count_[global_cur] == 1)
                    q_node_values_[global_cur] = rollout_value;
                else
                    q_node_values_[global_cur] +=
                        (rollout_value - q_node_values_[global_cur]) / static_cast<float>(node_count_[global_cur]);

                // Persistent max-backup at root
                q_node_values_[global_cur] = std::max(q_node_values_[global_cur], node_initial_rollout_[global_cur]);
            }
            else
            {
                ++node_count_[getNodeIdx(0, scenario_idx)];
            }
            // total_backprop_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++backprop_call_count_;
        }

        // -----------------------------------------------------------------------
        // Depth-gap backup (for depth-synchronisation)
        // -----------------------------------------------------------------------

        void VecQMDP::backupDepth(uint32_t node_idx, uint32_t scenario_idx)
        {
            uint32_t cur = node_idx;
            uint32_t global_cur = getNodeIdx(cur, scenario_idx);

            const int own_depth = static_cast<int>(node_depth_[global_cur]);
            node_min_depth_[global_cur] = own_depth;
            node_max_depth_[global_cur] = own_depth;

            while (cur > 0)
            {
                const uint32_t parent = (cur - 1) / num_actions_;
                const uint32_t global_parent = getNodeIdx(parent, scenario_idx);
                const uint32_t child_rel_start = num_actions_ * parent + 1;
                const uint32_t child_glob_start = getNodeIdx(child_rel_start, scenario_idx);

                // SIMD pass over the first (num_actions_ - 1) children
                IVectorT_qmdp active_batch =
                    IVectorT_qmdp::load_contiguous_unaligned(node_active_flags_.data(), child_glob_start);
                IVectorT_qmdp min_batch =
                    IVectorT_qmdp::load_contiguous_unaligned(node_min_depth_.data(), child_glob_start);
                IVectorT_qmdp max_batch =
                    IVectorT_qmdp::load_contiguous_unaligned(node_max_depth_.data(), child_glob_start);

                IVectorT_qmdp min_proc =
                    IVectorT_qmdp::select(active_batch, min_batch, IVectorT_qmdp(std::numeric_limits<int>::max()));
                IVectorT_qmdp max_proc = IVectorT_qmdp::select(active_batch, max_batch, IVectorT_qmdp(0));

                // Last child handled serially (index = num_actions_ - 1)
                const uint32_t last = num_actions_ - 1;
                const bool     last_active = (node_active_flags_[child_glob_start + last] != 0);
                const int      last_min =
                    last_active ? node_min_depth_[child_glob_start + last] : std::numeric_limits<int>::max();
                const int last_max = last_active ? node_max_depth_[child_glob_start + last] : 0;

                int new_min = std::min(min_proc.hmin(), last_min);
                int new_max = std::max(max_proc.hmax(), last_max);

                const auto &cands = getNodeCandidateActions(global_parent);
                bool        fully_expanded = true;
                for (uint32_t a : cands)
                {
                    if (node_count_[child_glob_start + a] == 0)
                    {
                        fully_expanded = false;
                        break;
                    }
                }
                if (!fully_expanded)
                {
                    const int pd = static_cast<int>(node_depth_[global_parent]);
                    new_min = std::min(new_min, pd);
                    new_max = std::max(new_max, pd);
                }

                if (node_min_depth_[global_parent] == new_min && node_max_depth_[global_parent] == new_max)
                    break;

                node_min_depth_[global_parent] = new_min;
                node_max_depth_[global_parent] = new_max;
                cur = parent;
            }
        }

        // -----------------------------------------------------------------------
        // Early-termination
        // -----------------------------------------------------------------------

        void VecQMDP::updateConvergenceTracking()
        {
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                auto &info = tree_convergence_info_[s];
                ++info.total_checks;

                const uint32_t root = getNodeIdx(0, s);
                if (node_count_[root] == 0)
                    continue;

                int   best_action = -1;
                float best_q = std::numeric_limits<float>::lowest();

                for (uint32_t a : getNodeCandidateActions(root))
                {
                    const uint32_t gci = getNodeIdx(1 + a, s);
                    if (node_count_[gci] > 0 && q_node_values_[gci] > best_q)
                    {
                        best_q = q_node_values_[gci];
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

        bool VecQMDP::checkEarlyTermination()
        {
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const auto &info = tree_convergence_info_[s];

                if (info.best_action_idx == -1 || info.stable_iterations < early_term_stable_iters_)
                    return false;

                const uint32_t best_child = getNodeIdx(1 + info.best_action_idx, s);
                if (node_count_[best_child] < early_term_min_best_visits_)
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

        void VecQMDP::printTree(int max_depth, std::ostream &out) const
        {
            out << "\n";
            out << "[VecQMDP] Tree dump (scenario 0, max_depth=" << max_depth << ")\n";

            struct Frame
            {
                uint32_t    node;
                int         depth;
                std::string prefix;
                bool        is_last;
            };

            std::vector<Frame> stack;
            stack.push_back({0, 0, "", true});

            int nodes_shown = 0;

            while (!stack.empty())
            {
                const Frame f = stack.back();
                stack.pop_back();

                const uint32_t g = f.node;
                if (g >= node_size_)
                    continue;

                const bool is_root = (g == 0);
                const bool has_visits = (node_count_[g] > 0);
                if (!is_root && !has_visits)
                    continue;

                const bool terminal = (node_active_flags_[g] == 0);

                if (f.depth == 0)
                {
                    out << "ROOT"
                        << "  N=" << node_count_[g] << "  Q=" << q_node_values_[g] << "  R=" << node_rewards_[g]
                        << "  depth=" << node_depth_[g] << "\n";
                }
                else
                {
                    out << f.prefix << (f.is_last ? "└─ " : "├─ ") << "a=" << node_curr_action_idxs_[g]
                        << "  N=" << node_count_[g] << "  Q=" << q_node_values_[g] << "  R=" << node_rewards_[g]
                        << "  depth=" << node_depth_[g];
                    if (terminal)
                        out << "  [TERMINAL]";
                    out << "\n";
                }
                ++nodes_shown;

                if (f.depth >= max_depth || terminal)
                    continue;

                // Collect visited children (visit count > 0)
                std::vector<uint32_t> children;
                children.reserve(num_actions_);
                const uint32_t base_rel = g; // scenario 0: global == relative
                for (uint32_t a = 0; a < num_actions_; ++a)
                {
                    const uint32_t child_rel = base_rel * num_actions_ + a + 1;
                    if (child_rel >= tree_node_size_)
                        break;
                    const uint32_t child_g = child_rel;
                    if (child_g < node_size_ && node_count_[child_g] > 0)
                        children.push_back(child_g);
                }

                const std::string child_prefix = f.prefix + (f.is_last ? "   " : "│  ");
                for (int i = static_cast<int>(children.size()) - 1; i >= 0; --i)
                {
                    const bool child_last = (i == static_cast<int>(children.size()) - 1);
                    stack.push_back({children[static_cast<size_t>(i)], f.depth + 1, child_prefix, child_last});
                }
            }

            out << "\n[VecQMDP] Printed " << nodes_shown << " node(s) (visit_count > 0, depth ≤ " << max_depth
                << ")\n\n";
        }

    } // namespace planning
} // namespace vec_qmdp