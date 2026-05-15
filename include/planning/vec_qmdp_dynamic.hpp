/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file vec_qmdp_dynamic.hpp
 * @brief VecQMDP variant with dynamic chunked SoA memory pool.
 *
 * DynVecQMDP implements the same BeliefTreeSearch kernel as VecQMDP (majority-vote
 * depth synchronisation, UCB selection, back-propagation, early termination)
 * but replaces the pre-allocated implicit balanced tree with a dynamically
 * grown pool of SoA chunks (see dynamic_soa_pool.hpp).
 *
 * Key differences from VecQMDP
 * -----------------------------
 *  - Tree topology is stored explicitly via child_base_idx / parent_idx fields.
 *    No exponential pre-allocation is required.
 *  - exploreNodesVote() returns a vector of DynExploreResult structs that carry
 *    the global node index; the derived class uses these to gather domain state
 *    without reconstructing positions from implicit paths.
 *  - allocateChildBlock(parent_g) allocates num_actions_ consecutive slots,
 *    initialises their BeliefTreeSearch fields, and writes child_base_idx on the parent.
 *    The derived class is responsible for also initialising domain-specific
 *    state (e.g. robot_pos, rock_bits) at the returned global indices.
 *  - backPropagate / backupDepth traverse via parent_idx rather than the
 *    parent = (cur - 1) / num_actions formula.
 *
 * Config toggle
 * -------------
 * Define VECQMDP_USE_DYNAMIC_POOL before including this header (or pass
 * -DVECQMDP_USE_DYNAMIC_POOL via CMake) to select the dynamic backend.
 * When the macro is absent the original VecQMDP is used unchanged.
 *
 * To port a domain:
 *   1.  Derive from DynVecQMDP instead of VecQMDP.
 *   2.  Implement computeNodeCandidateActionsDyn(int32_t global_idx).
 *   3.  Call allocateChildBlock(parent_g) inside expandNodesBatch, then
 *       scatter domain state to the returned child global indices.
 *   4.  Pass child global indices to backPropagate / backupDepth.
 */
#pragma once

// ============================================================================
// Feature toggle: Enable homogenous search optimization
// ============================================================================
// #define ENABLE_HOMOGENOUS_SEARCH

#include <planning/dynamic_soa_pool.hpp> // DynSoAPool, DynSoAChunk

// Minimal standard headers — deliberately avoid pulling in vamp/vector.hh
// which has a platform-specific NEON bug on Apple Silicon.
#include <cmath>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare ThreadPool so callers are not forced to include the full
// global_utils.hpp (which drags in vamp/vector.hh) when they only need the
// dynamic pool interface.
namespace vec_qmdp
{
    namespace utils
    {
        class ThreadPool;
    }
} // namespace vec_qmdp

namespace vec_qmdp
{
    namespace planning
    {
        // =====================================================================
        // Result type returned by exploreNodesVote (dynamic variant)
        // =====================================================================

        /**
         * @brief Per-scenario output of DynVecQMDP::exploreNodesVote.
         *
         * The derived class uses global_idx to gather domain state for the
         * expansion step and later passes child_global_idx to backPropagate.
         */
        struct DynExploreResult
        {
            int32_t  parent_global_idx; ///< global pool index of the node to expand
            int32_t  parent_depth;      ///< depth of that node (0 = root)
            uint32_t action_idx;        ///< action to take from that parent
            uint32_t scenario_idx;      ///< owning scenario (0 … scenario_size_-1)
        };

        // =====================================================================
        // DynVecQMDP — dynamic-pool BeliefTreeSearch base class
        // =====================================================================

        class DynVecQMDP
        {
          public:
            // ---- Parallel-search result (same shape as VecQMDP) ----
            struct BeliefTreeSearchThreadResult
            {
                std::vector<float> action_sum_values;
                std::vector<int>   action_counts;
                double             tree_search_time_ms{0.0};
                int                simulation_count{0};
            };

            /**
             * @param num_scenarios  Number of parallel scenario trees.
             * @param num_actions    Branching factor (actions per node).
             * @param tree_height    Maximum search depth (layers 0 … tree_height).
             */
            DynVecQMDP(int num_scenarios, uint32_t num_actions, uint32_t tree_height);
            virtual ~DynVecQMDP() = default;

            // ----------------------------------------------------------------
            // Tunable hyperparameter setters (identical API to VecQMDP)
            // ----------------------------------------------------------------

            void setDiscountFactor(double v) { discount_factor_ = v; }
            void setExplorationConstant(float v) { exploration_constant_ = v; }
            void setMaxBackupLambda(float v) { max_backup_lambda_ = v; }
            void setDepthSyncLambda(float v) { depth_sync_lambda_ = v; }
            void setPrunedThreshold(float v) { pruned_threshold_ = v; }
            void setMaxPlanningTimeMs(double v) { max_planning_time_ms_ = v; }
            void setEarlyTermStableIterations(int v) { early_term_stable_iters_ = v; }
            void setEarlyTermMinBestVisits(int v) { early_term_min_best_visits_ = v; }
            void setEarlyTermQChangeThreshold(float v) { early_term_q_change_thr_ = v; }
            void setEarlyTermMinExpandCalls(int v) { early_term_min_expand_calls_ = v; }
            void setEarlyTermCheckInterval(int v) { early_term_check_interval_ = v; }

            // ----------------------------------------------------------------
            // Core tree-search interface
            // ----------------------------------------------------------------

            std::vector<DynExploreResult> exploreNodes();

            /**
             * @brief Explore one node per scenario using majority-vote depth sync.
             * @return One DynExploreResult per scenario (size = scenario_size_).
             */
            std::vector<DynExploreResult> exploreNodesVote();

            /**
             * @brief Allocate num_actions_ consecutive child slots for @p parent_g,
             *        initialise their BeliefTreeSearch fields, and set parent's child_base_idx.
             * @param parent_g  Global index of the parent node.
             * @return Global index of the first child slot (child for action a is at
             *         return_value + a).
             *
             * After this call the derived class must write domain state at every
             * returned child index before the next exploreNodesVote call.
             */
            int32_t allocateChildBlock(int32_t parent_g);

            /**
             * @brief Back-propagate a cumulative return up to the root.
             * @param child_g       Global index of the newly expanded child.
             * @param rollout_value Discounted return from the rollout.
             */
            void backPropagate(int32_t child_g, float rollout_value);

            /**
             * @brief Update depth-gap [min_depth, max_depth] on the path to root.
             * @param child_g  Global index of the newly expanded child.
             */
            void backupDepth(int32_t child_g);

#ifdef ENABLE_HOMOGENOUS_SEARCH
            /**
             * @brief Homogenous-search node selection (scenario-0 only).
             *
             * Explores only scenario-0's tree and returns the level where selection
             * stopped (due to unvisited child or tree boundary).
             *
             * @param[out] action_idx  Action selected at this level.
             * @return Current level in homogenous tree.
             */
            int32_t exploreNodesHomogenous(int &action_idx);

            /**
             * @brief Homogenous back-propagation from child_level to root.
             *
             * All scenarios at child_level are processed in SIMD batches,
             * traversing up the tree level-by-level.
             *
             * @param child_level    Level of newly expanded children.
             * @param rollout_values Rollout return per scenario [scenario_size_].
             */
            void backPropagateHomogenous(int32_t child_level, const std::vector<float> &rollout_values);
#endif // ENABLE_HOMOGENOUS_SEARCH

            // ----------------------------------------------------------------
            // Early-termination (same semantics as VecQMDP)
            // ----------------------------------------------------------------

            void updateConvergenceTracking();
            bool checkEarlyTermination();

            // ----------------------------------------------------------------
            // Read-only accessors
            // ----------------------------------------------------------------

            inline int32_t getRootGlobalIdx(uint32_t scenario_idx) const { return scenario_roots_[scenario_idx]; }

            inline float getRootActionQValue(uint32_t scenario_idx, uint32_t action_idx) const
            {
                const int32_t root_g = scenario_roots_[scenario_idx];
                const int32_t child_b = pool_.child_base(root_g);
                if (child_b == DYN_INVALID_IDX)
                    return -1e9f;
                return pool_.q_value(child_b + static_cast<int32_t>(action_idx));
            }

            inline int32_t getRootVisitCount(uint32_t scenario_idx, uint32_t action_idx) const
            {
                const int32_t root_g = scenario_roots_[scenario_idx];
                const int32_t child_b = pool_.child_base(root_g);
                if (child_b == DYN_INVALID_IDX)
                    return 0;
                return pool_.visit_count(child_b + static_cast<int32_t>(action_idx));
            }

            inline int32_t getTotalNodesAllocated() const noexcept { return pool_.total_nodes(); }

            inline size_t getPoolMemoryBytes() const noexcept { return pool_.memory_bytes(); }

            // ---- Parallel-search accessors ----
            inline const std::vector<float> &getAggregatedActionValues() const { return aggregated_action_values_; }
            inline const std::vector<std::shared_ptr<DynVecQMDP>> &getWorkerInstances() const
            {
                return worker_instances_;
            }
            inline size_t getNumWorkerThreads() const { return num_worker_threads_; }
            inline int    getTotalSimulationCount() const { return total_simulation_count_; }

            /**
             * Print the BeliefTreeSearch search tree for scenario-0 (dynamic pool).
             * Only nodes with visit count > 0 are shown.
             * @param max_depth Maximum tree depth to print (root has depth 0).
             */
            void printTree(int max_depth = 10, std::ostream &out = std::cout) const;

          protected:
            // ================================================================
            // Pool and topology
            // ================================================================

            DynSoAPool           pool_;           ///< thread-private node pool
            std::vector<int32_t> scenario_roots_; ///< root global index per scenario

            // ================================================================
            // BeliefTreeSearch structural parameters
            // ================================================================

            uint32_t num_actions_;
            uint32_t tree_height_;
            uint32_t scenario_size_;

            // ================================================================
            // BeliefTreeSearch hyperparameters
            // ================================================================

            double discount_factor_{0.95};
            float  exploration_constant_{1.0f};
            float  max_backup_lambda_{0.001f};
            float  depth_sync_lambda_{0.0f};
            float  pruned_threshold_{-500.0f};
            double max_planning_time_ms_{100.0};

            // ================================================================
            // Early-termination parameters
            // ================================================================

            int   early_term_stable_iters_{5};
            int   early_term_min_best_visits_{10};
            float early_term_q_change_thr_{0.01f};
            int   early_term_min_expand_calls_{50};
            int   early_term_check_interval_{10};

            // ================================================================
            // Parallel infrastructure
            // ================================================================

            size_t                                   num_worker_threads_{1};
            std::shared_ptr<utils::ThreadPool>       thread_pool_;
            std::vector<std::shared_ptr<DynVecQMDP>> worker_instances_;
            std::vector<float>                       aggregated_action_values_;
            std::vector<int>                         aggregated_action_counts_;
            int                                      total_simulation_count_{0};

            void initParallelInfrastructure(size_t num_threads);
            void dispatchParallelSearch(std::function<BeliefTreeSearchThreadResult(DynVecQMDP *, size_t)> search_fn);

            // ================================================================
            // Performance timing stats
            // ================================================================

            double  total_step_batch_us_{0.0};
            int64_t step_batch_call_count_{0};
            double  total_rollout_us_{0.0};
            int64_t rollout_call_count_{0};
            double  total_backprop_us_{0.0};
            int64_t backprop_call_count_{0};
            double  total_select_batch_us_{0.0};
            int64_t select_batch_call_count_{0};

            virtual void printTimingStats() const;

#ifdef ENABLE_HOMOGENOUS_SEARCH
            // ================================================================
            // Homogenous search support
            // ================================================================

            /**
             * @brief Scenario-to-global-index mapping for homogenous search.
             *
             * homogenous_node_map_[level][s] = global pool index for scenario s at level.
             * Level 0 = root, increments with each expansion.
             */
            std::vector<std::vector<int32_t>> homogenous_node_map_;

            /**
             * @brief Current depth in homogenous tree (number of expansions from root).
             * Reset to 0 at start of each search round.
             */
            int32_t homogenous_current_level_{0};

            /**
             * @brief SIMD index buffer: scenario_global_idxs_[s] = current node for scenario s.
             * Refilled at each expansion for efficient SIMD gather/scatter.
             */
            std::vector<int32_t> scenario_global_idxs_;

            /**
             * @brief Initialize homogenous mapping with root nodes.
             * Call after initDynRoots().
             */
            void initHomogenousMapping();
#endif // ENABLE_HOMOGENOUS_SEARCH

            // ================================================================
            // Protected helpers called by derived class
            // ================================================================

            /**
             * @brief Allocate and initialise scenario_size_ root nodes.
             *
             * Must be called once per search round (after pool_.reset()) before
             * the first exploreNodesVote.  The derived class should also initialise
             * domain state at each returned root index.
             */
            void initDynRoots();

            /**
             * @brief Reset pool and convergence state for a new search round.
             *
             * Call at the start of each runSearch / runSearchInternal invocation.
             * Domain-specific arrays in the derived class must be reset separately.
             */
            void resetDynStructures();

            /// Cached-wrapper: calls computeNodeCandidateActionsDyn on first access.
            const std::vector<uint32_t> &getDynCandidateActions(int32_t global_idx);

          private:
            // ================================================================
            // Domain hooks (pure virtual)
            // ================================================================

            /**
             * @brief Return the priority-ordered list of valid actions at @p global_idx.
             *
             * This mirrors VecQMDP::computeNodeCandidateActions but accepts a global
             * pool index rather than a relative tree index.  Called once per node;
             * result is cached by getDynCandidateActions().
             */
            virtual std::vector<uint32_t> computeNodeCandidateActionsDyn(int32_t global_idx) = 0;

            /// Factory: create an independent single-threaded copy for the thread pool.
            virtual std::shared_ptr<DynVecQMDP> makeWorkerInstance() const = 0;

            // ================================================================
            // Internal UCB / selection helpers
            // ================================================================

            float calculateUCB(int32_t node_g, int32_t parent_g, int target_depth = -1) const;

            int32_t selectNodeDyn(int32_t parent_g, const std::vector<uint32_t> &candidate_actions,
                                  int target_depth = -1) const;

            // ================================================================
            // Candidate-action cache
            // ================================================================

            std::unordered_map<int32_t, std::vector<uint32_t>> dyn_candidate_actions_;

            // ================================================================
            // Convergence tracking
            // ================================================================

            struct DynConvergenceInfo
            {
                int   best_action_idx = -1;
                float best_q_value = std::numeric_limits<float>::lowest();
                int   stable_iterations = 0;
                float prev_q_value = std::numeric_limits<float>::lowest();
                int   total_checks = 0;
            };
            std::vector<DynConvergenceInfo> convergence_info_;
        };

    } // namespace planning
} // namespace vec_qmdp
