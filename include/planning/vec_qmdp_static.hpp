/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file vec_qmdp_static.hpp
 * @brief Abstract base class for SIMD-vectorised BeliefTreeSearch tree search (VecQMDP).
 *
 * VecQMDP provides a domain-agnostic, high-performance BeliefTreeSearch kernel that
 * operates on a pre-allocated, implicit balanced tree stored in flat
 * SIMD-aligned arrays.  It is designed to be subclassed for any specific
 * planning domain.
 *
 * Core capabilities provided by this base class:
 *   - Per-scenario search trees with vectorised (SIMD) back-propagation.
 *   - Three exploration strategies: batch UCB (exploreNodes), majority-vote
 *     depth synchronisation (exploreNodesVote), and homogenous single-path
 *     mode (exploreNodesHomogenous).
 *   - UCB selection with precomputed coefficient lookup tables.
 *   - Max-backup blending via max_backup_lambda_.
 *   - Automatic early-termination on Q-value convergence.
 *   - Built-in multi-threaded parallel search via a shared thread pool.
 *
 * ============================================================================
 * How to implement a new domain
 * ============================================================================
 *
 * 1. Derive from VecQMDP and implement two pure-virtual hooks:
 *
 *    a) computeNodeCandidateActions(uint32_t node_idx)
 *       Return the priority-ordered list of valid action indices from the
 *       given tree node.  Called once per node; the result is cached.
 *
 *    b) makeWorkerInstance()
 *       Create a new, independent, single-threaded instance of the same
 *       derived class.  Used by initParallelInfrastructure() for multi-
 *       threaded search.
 *
 * 2. Implement a domain-specific expandNodesBatch() method that:
 *    - Receives the selected node indices and action indices from
 *      exploreNodes() / exploreNodesVote() / exploreNodesHomogenous().
 *    - Simulates one step forward (state transition + reward).
 *    - Writes child node state into the pre-allocated tree arrays.
 *    - Calls backPropagate() and backupDepth() from the base class.
 *
 * 3. Implement a domain-specific search entry point (e.g. runSearch() or
 *    beliefTreeSearch()) that orchestrates the BeliefTreeSearch loop:
 *      while (budget remaining) {
 *          nodes = exploreNodes*();          // base class
 *          expandNodesBatch(nodes, actions); // your domain logic
 *      }
 *
 * 4. Optionally override printTree() for domain-specific tree visualisation.
 *
 * See the examples/ directory for complete domain implementations:
 *   - RockSampleStaticVecQMDP : discrete grid-world POMDP (Rock Sample).
 *   - VecQMDP_AD              : continuous autonomous-driving planner.
 */
#pragma once

// #define ENABLE_HOMOGENOUS_SEARCH
// #define PRINT_TIME

#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utils/aligned_allocator.hpp>
#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        /**
         * @class VecQMDP
         * @brief Abstract base class for SIMD-vectorised BeliefTreeSearch tree search.
         *
         * Derived classes must implement the pure-virtual domain hooks
         * (see the "Domain hooks" section in protected) and provide their
         * own expandNodesBatch() and search-loop methods.
         *
         * The base class owns all tree bookkeeping arrays, UCB tables,
         * parallel infrastructure, and provides exploration / back-propagation
         * / early-termination algorithms that are fully domain-agnostic.
         */
        class VecQMDP
        {
          public:
            // =================== Type aliases ================================
            using AlignedVectorFloat = utils::AlignedVectorFloat;
            using AlignedVectorInt = utils::AlignedVectorInt;
            using AlignedVectorBool = utils::AlignedVectorBool;
            using FVectorT_qmdp = utils::FVectorT_qmdp;
            using IVectorT_qmdp = utils::IVectorT_qmdp;

            int vector_with = FVectorT_qmdp::num_scalars_per_row;

            // =================== Timing stats =======================================
            double  total_step_batch_us_{0.0};
            int64_t step_batch_call_count_{0};
            double  total_rollout_us_{0.0};
            int64_t rollout_call_count_{0};
            double  total_backprop_us_{0.0};
            int64_t backprop_call_count_{0};
            double  total_select_batch_us_{0.0};
            int64_t select_batch_call_count_{0};

            // =================== UCB table size =====================================
            // Tables are sized to cover node visit counts up to this limit.
            // Call initUCBTables() after setting exploration_constant_ and before search.
            static constexpr uint32_t UCB_TABLE_SIZE = 65536u;

            // =================== Parallel search result ===================

            /// Aggregated result returned by each worker thread after one BeliefTreeSearch run.
            struct BeliefTreeSearchThreadResult
            {
                std::vector<float> action_sum_values; ///< summed Q-values per action
                std::vector<int>   action_counts;     ///< valid scenario count per action
                double             tree_search_time_ms{0.0};
                int                simulation_count{0};
            };

            /**
             * @param num_scenarios  Number of parallel scenario trees.
             * @param num_actions    Branching factor (actions per node).
             * @param tree_height    Maximum search depth (layers 0 … tree_height).
             */
            VecQMDP(int num_scenarios, uint32_t num_actions, uint32_t tree_height);
            virtual ~VecQMDP() = default;

            // =================== Tunable hyperparameter setters ===================

            void setDiscountFactor(double v) { discount_factor_ = v; }
            void setExplorationConstant(float v) { exploration_constant_ = v; }
            void setMaxBackupLambda(float v) { max_backup_lambda_ = v; }
            void setDepthSyncLambda(float v) { depth_sync_lambda_ = v; }
            void setPrunedThreshold(float v) { pruned_threshold_ = v; }
            void setMaxPlanningTimeMs(double v) { max_planning_time_ms_ = v; }

            /// Early-termination: minimum stable iterations before checking.
            void setEarlyTermStableIterations(int v) { early_term_stable_iters_ = v; }
            /// Early-termination: minimum visits to the best child before checking.
            void setEarlyTermMinBestVisits(int v) { early_term_min_best_visits_ = v; }
            /// Early-termination: relative Q-change tolerance.
            void setEarlyTermQChangeThreshold(float v) { early_term_q_change_thr_ = v; }
            /// Early-termination: minimum expand calls before checking is enabled.
            void setEarlyTermMinExpandCalls(int v) { early_term_min_expand_calls_ = v; }
            /// Early-termination: check interval (every N iterations).
            void setEarlyTermCheckInterval(int v) { early_term_check_interval_ = v; }

            /// Build UCB lookup tables: ucb_coeff[n] = C*sqrt(log(n)),
            /// inv_sqrt[n] = 1/sqrt(n).  Must be called after setExplorationConstant()
            /// and before the first exploreNodes() call.
            void initUCBTables();

            // =================== Core tree-search interface ===================

            /// Select the best child of @p parent_index via UCB.
            /// Pass target_depth = -1 to disable the depth-sync penalty.
            uint32_t selectNode(uint32_t parent_index, const std::vector<uint32_t> &candidate_actions,
                                uint32_t scenario_index, int target_depth = -1);

            /// SIMD-parallel UCB tree traversal for all scenarios.
            /// @p target_action_idxs is resized to scenario_size_ and filled with the
            /// flat action index selected for each scenario.
            /// Supports arbitrary scenario_size_ by processing scenarios in SIMD-width
            /// batches internally (not limited to IVectorT_qmdp::num_scalars = 8).
            AlignedVectorInt exploreNodes(std::vector<int> &target_action_idxs);

            /// Explore one node per scenario (majority-vote depth-sync strategy).
            /// Fills @p target_action_idxs with the flat action index selected for
            /// each scenario.  Any domain-specific decoding of those indices
            /// (e.g. path/offset decomposition) must be done by the derived class.
            AlignedVectorInt exploreNodesVote(IVectorT_qmdp &target_action_idxs);

            /// Back-propagate a cumulative return to the root of @p scenario_index.
            void backPropagate(uint32_t node_idx, uint32_t scenario_index, float rollout_value);

            /// Update depth-gap [min, max] on the path from @p node_idx to the root.
            void backupDepth(uint32_t node_idx, uint32_t scenario_index);

#ifdef ENABLE_HOMOGENOUS_SEARCH
            /// Homogenous-search variant: explores only scenario-0's subtree and
            /// returns the single relative node index and action that represent ALL
            /// scenarios (they share the same tree path by design).
            ///
            /// @param[out] action_idx  Flat action index selected for this step.
            /// @return Scenario-0 global node index (= relative index, since
            ///         scenario-0 offset is 0) of the selected parent node.
            uint32_t exploreNodesHomogenous(int &action_idx);

            /// Homogenous back-propagation: propagates rollout_values[s] for every
            /// scenario s along the common relative tree path from @p node_idx to the
            /// root in a single traversal.  More efficient than calling
            /// backPropagate() scenario_size_ times when all scenarios share the
            /// same path (which is guaranteed under ENABLE_HOMOGENOUS_SEARCH).
            void backPropagateHomogenous(uint32_t node_idx, AlignedVectorFloat &rollout_values);
#endif // ENABLE_HOMOGENOUS_SEARCH

            // =================== Early-termination ===================
            void updateConvergenceTracking();
            bool checkEarlyTermination();

            // =================== Read-only accessors ===================
            inline int      getNodeCount(uint32_t node_idx) const { return node_count_[node_idx]; }
            inline float    getQNodeValue(uint32_t node_idx) const { return q_node_values_[node_idx]; }
            inline uint32_t getNodeIdx(uint32_t base_index, uint32_t scenario_index) const
            {
                return base_index + scenario_index * tree_node_size_;
            }
            inline uint32_t getScenarioIdx(uint32_t node_idx) const { return node_idx / tree_node_size_; }
            inline uint32_t getNodeDepth(uint32_t node_idx) const { return node_depth_[node_idx]; }

            /**
             * Print the BeliefTreeSearch search tree for scenario-0.
             * Only nodes with visit count > 0 are shown.
             * Override in derived classes for domain-specific visualisation.
             * @param max_depth Maximum tree depth to print (root has depth 0).
             */
            virtual void printTree(int max_depth = 10, std::ostream &out = std::cout) const;

            // =================== Parallel-search accessors ===================

            /// Action Q-values averaged across all worker threads after a parallel search.
            inline const std::vector<float> &getAggregatedActionValues() const { return aggregated_action_values_; }

            /// Direct access to the per-thread worker instances (e.g. for second-level
            /// aggregation in derived classes).
            inline const std::vector<std::shared_ptr<VecQMDP>> &getWorkerInstances() const { return worker_instances_; }

            inline size_t getNumWorkerThreads() const { return num_worker_threads_; }

          protected:
            // =================== Parallel infrastructure ===================

            size_t                                num_worker_threads_{1};
            std::shared_ptr<utils::ThreadPool>    thread_pool_;
            std::vector<std::shared_ptr<VecQMDP>> worker_instances_;

            /// Per-action aggregated averages populated by dispatchParallelSearch().
            std::vector<float> aggregated_action_values_;
            std::vector<int>   aggregated_action_counts_;
            int                aggregated_simulation_count_{0};

            /// Allocate @p num_threads worker instances (via makeWorkerInstance) and
            /// start the internal thread pool.  Call once from the derived constructor
            /// after all domain-specific state has been initialised.
            void initParallelInfrastructure(size_t num_threads);

            /// Dispatch @p search_fn to every worker instance in parallel, then
            /// aggregate action Q-values into aggregated_action_values_.
            void
            dispatchParallelSearch(std::function<BeliefTreeSearchThreadResult(VecQMDP *, size_t thread_idx)> search_fn);

            // =================== Tree-structure parameters ===================
            // Set once in the constructor from the arguments supplied by the
            // derived class.  Do NOT reference utils:: constants here.

            uint32_t              num_actions_;    ///< branching factor
            uint32_t              tree_height_;    ///< maximum depth (0-indexed, inclusive)
            uint32_t              tree_node_size_; ///< total nodes per scenario tree
            std::vector<uint32_t> tree_node_sizes_per_depth_;
            ///< [layer] = num_actions_^layer

            // =================== BeliefTreeSearch hyperparameters ===================
            double discount_factor_;
            float  exploration_constant_;
            float  max_backup_lambda_;
            float  depth_sync_lambda_;
            float  pruned_threshold_;
            double max_planning_time_ms_;

            // =================== Early-termination parameters ===================
            int   early_term_stable_iters_{5};
            int   early_term_min_best_visits_{10};
            float early_term_q_change_thr_{0.01f};
            int   early_term_min_expand_calls_{50};
            int   early_term_check_interval_{10};

            // =================== Dimension info ===================
            uint32_t node_size_;     ///< tree_node_size_ * scenario_size_
            uint32_t scenario_size_; ///< number of parallel scenarios

            // =================== Per-node task-agnostic state ===================

            /// 0xFFFFFFFF = active, 0 = terminated.  Written by the derived class.
            AlignedVectorInt node_active_flags_;

            /// Action index taken to reach each node.  Written by the derived class.
            AlignedVectorInt node_curr_action_idxs_;

            // =================== Tree bookkeeping ===================
            AlignedVectorFloat q_node_values_;
            AlignedVectorFloat node_rewards_;
            AlignedVectorInt   node_count_;
            AlignedVectorInt   node_depth_;
            AlignedVectorInt   node_rollout_len_;
            AlignedVectorInt   node_min_depth_;
            AlignedVectorInt   node_max_depth_;
            AlignedVectorFloat node_initial_rollout_; ///< Initial rollout value (persistent lower bound)

            std::vector<std::vector<uint32_t>> node_candidate_actions_;
            AlignedVectorBool                  node_actions_computed_;

            // =================== UCB lookup tables (SIMD-friendly) ==================
            // ucb_coeff_table_[n] = exploration_constant_ * sqrt(log(n)), n = 0..UCB_TABLE_SIZE-1
            // inv_sqrt_table_[n]  = 1.0f / sqrt(n),                        n = 1..UCB_TABLE_SIZE-1
            // Both are indexed by integer visit counts and used in exploreNodes().
            AlignedVectorFloat ucb_coeff_table_;
            AlignedVectorFloat inv_sqrt_table_;

            // Precomputed per-scenario memory offsets: scen_offsets_[s] = s * tree_node_size_
            // Set once in initializeTreeStructures(), reused every exploreNodes() call.
            AlignedVectorInt scen_offsets_;

            std::unordered_map<int, int> depth_difference_count;

            // =================== Protected helpers ===================

            void initializeTreeStructures();
            void resetTreeStructures();

            /// Cached wrapper: calls computeNodeCandidateActions on first access.
            inline const std::vector<uint32_t> &getNodeCandidateActions(uint32_t node_idx)
            {
                if (!node_actions_computed_[node_idx])
                {
                    node_candidate_actions_[node_idx] = computeNodeCandidateActions(node_idx);
                    node_actions_computed_[node_idx] = true;
                }
                return node_candidate_actions_[node_idx];
            }

            // =================== Domain hooks (pure virtual) =====================
            //
            // Every derived class MUST implement the following two methods.
            // They are the minimum interface required to plug a new domain into
            // the VecQMDP tree-search engine.
            //
            // See the class-level documentation and examples/ for guidance.
            // =================================================================

            /**
             * Compute the priority-ordered list of valid action indices at
             * @p node_idx.  Called once per node; the result is cached
             * internally by getNodeCandidateActions().
             *
             * The ordering determines UCB tie-breaking: actions listed first
             * are explored preferentially when they are equally promising.
             *
             * @param node_idx  Global node index (encodes both the tree-relative
             *                  position and the scenario index).
             * @return A vector of action indices, ordered by domain-specific
             *         priority (most preferred first).
             */
            virtual std::vector<uint32_t> computeNodeCandidateActions(uint32_t node_idx) = 0;

            /**
             * Factory method: create a new, independent instance of the same
             * derived type configured for single-threaded use (num_threads = 1).
             *
             * Called by initParallelInfrastructure() to populate
             * worker_instances_.  The returned instance must carry the same
             * domain parameters (grid size, action space, etc.) but does NOT
             * need to share mutable state with the parent.
             *
             * @return A shared pointer to a freshly constructed single-threaded
             *         worker of the concrete derived type.
             */
            virtual std::shared_ptr<VecQMDP> makeWorkerInstance() const = 0;

          private:
            // =================== Internal algorithms ===================

            float calculateUCB(uint32_t node_idx, uint32_t parent_index, int target_depth = -1);

            inline IVectorT_qmdp getNodeidxs(const IVectorT_qmdp &base_idxs, const IVectorT_qmdp &scenario_idxs) const
            {
                return base_idxs + scenario_idxs * static_cast<int32_t>(tree_node_size_);
            }

            // =================== Convergence tracking ===================
            struct TreeConvergenceInfo
            {
                int   best_action_idx = -1;
                float best_q_value = std::numeric_limits<float>::lowest();
                int   stable_iterations = 0;
                float prev_q_value = std::numeric_limits<float>::lowest();
                int   total_checks = 0;
            };
            std::vector<TreeConvergenceInfo> tree_convergence_info_;
        };

    } // namespace planning
} // namespace vec_qmdp
