/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file vec_qmdp_ad.hpp
 * @brief Autonomous-driving instantiation of VecQMDP (VecQMDP_AD).
 *
 * VecQMDP_AD extends the domain-agnostic VecQMDP base with all data
 * structures and methods specific to on-road motion planning:
 *   - Ego-vehicle kinematic state buffers (flat, SIMD-aligned).
 *   - Exogenous-agent trajectory buffers sampled from NetBelief.
 *   - Frenet-frame coordinates and STRtree collision structures.
 *   - AD-specific action encoding (path × lateral-offset grid).
 *   - StepBatch simulation via ContextQMDP.
 *
 * Public interface used by QMDPTrajectoryPlanner:
 *   sampleScenarios()   — populate state buffers from a NetBelief sample.
 *   expandNodesBatch()  — simulate one step + rollout for a batch of nodes.
 *   beliefTreeSearch()        — run the full beliefTreeSearch loop and return elapsed time (µs).
 */
#pragma once

#include <chrono>
#include <core/net_belief.hpp>
#include <core/state.hpp>
#include <memory>
#include <planning/context_qmdp.hpp>
#include <planning/vec_qmdp_static.hpp>
#include <random>
#include <utils/params.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        // -----------------------------------------------------------------------
        // Action-transition table for the AD action space
        // (NUM_REF_PATH paths × NUM_LATERAL_OFFSETS offsets = 9 actions).
        // Built once at program start; reused across all VecQMDP_AD instances.
        // -----------------------------------------------------------------------
        struct ActionTransition
        {
            uint32_t target_action_idx;
            uint32_t target_path_idx;

            ActionTransition(uint32_t a, uint32_t p) : target_action_idx(a), target_path_idx(p) {}
        };

        /// Priority-ordered candidate transition table.
        /// STATIC_TRANSITION_TABLE[current_action_id] = ordered list of reachable actions.
        static const std::vector<std::vector<ActionTransition>> STATIC_TRANSITION_TABLE = []()
        {
            std::vector<std::vector<ActionTransition>> table(utils::NUM_ACTIONS);

            for (int curr = 0; curr < static_cast<int>(utils::NUM_ACTIONS); ++curr)
            {
                int curr_path = curr / static_cast<int>(utils::NUM_LATERAL_OFFSETS);

                // Center-out expansion: delta = 0 (same action), then ±1, ±2, ...
                for (int delta = 0; delta < static_cast<int>(utils::NUM_ACTIONS); ++delta)
                {
                    auto try_add = [&](int target)
                    {
                        if (target < 0 || target >= static_cast<int>(utils::NUM_ACTIONS))
                            return;
                        int target_path = target / static_cast<int>(utils::NUM_LATERAL_OFFSETS);
                        // Disallow jumping across non-adjacent paths
                        if (std::abs(curr_path - target_path) <= 1)
                            table[curr].emplace_back(static_cast<uint32_t>(target), static_cast<uint32_t>(target_path));
                    };

                    if (delta == 0)
                    {
                        try_add(curr);
                    }
                    else
                    {
                        try_add(curr - delta);
                        try_add(curr + delta);
                    }
                }
            }
            return table;
        }();

        // -----------------------------------------------------------------------

        class VecQMDP_AD : public VecQMDP
        {
          public:
            using STRtree = collision::STRtree;

            /**
             * Constructs the AD planner with the action-space and tree-height
             * values defined in utils/params.hpp (NUM_ACTIONS, TREE_HEIGHT).
             *
             * @param num_scenarios_per_worker  Scenarios processed by each worker thread.
             * @param num_threads               Number of parallel worker threads (>= 1).
             *                                  When > 1, initParallelInfrastructure is
             *                                  called automatically; each worker receives
             *                                  its own VecQMDP_AD(num_scenarios, 1).
             */
            explicit VecQMDP_AD(int num_scenarios_per_worker, size_t num_threads = 1);
            ~VecQMDP_AD() override = default;

            // ------------------- AD-specific public interface -------------------

            /// Sample exogenous trajectories from @p belief and seed all root nodes.
            void sampleScenarios(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                 const std::shared_ptr<core::NetBelief> &belief, const size_t &curr_path_idx);

            /// Simulate one tree step + rollout for each node in @p node_idxs.
            void expandNodesBatch(const AlignedVectorInt &node_idxs, IVectorT_qmdp &target_action_idxs,
                                  IVectorT_qmdp &target_path_idxs, FVectorT_qmdp &target_offsets,
                                  const int &expand_nodes_calls);

            /// Run the full beliefTreeSearch search loop on this single instance.
            /// @return Elapsed wall-clock time in microseconds.
            double beliefTreeSearch(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                    const std::shared_ptr<core::NetBelief>   &belief,
                                    const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                    const size_t                             &curr_path_idx);

            /// Run beliefTreeSearch across all worker threads and aggregate action Q-values into
            /// aggregated_action_values_ (accessible via getAggregatedActionValues()).
            void parallelBeliefTreeSearch(const std::shared_ptr<core::EgoState>    &ego_state,
                                          const std::shared_ptr<core::NetBelief>   &belief,
                                          const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                          const size_t                             &curr_path_idx);

            /// Compute the best second-level action given the first best action.
            /// Iterates over all worker instances' second-layer Q-values.
            void getSecondBestAction(int first_best_action_idx, int &second_best_action_idx,
                                     int &second_best_target_path_idx, float &second_best_target_offset,
                                     float &second_best_value) const;

          private:
            // ------------------- VecQMDP hook implementation -------------------

            /// Compute the ordered set of valid action indices reachable from @p node_idx.
            std::vector<uint32_t> computeNodeCandidateActions(uint32_t node_idx) override;

            /// Factory: create a single-threaded VecQMDP_AD worker with the same
            /// scenario count.  Called by initParallelInfrastructure.
            std::shared_ptr<VecQMDP> makeWorkerInstance() const override;

            // ------------------- AD-specific action decoding -------------------

            /// Map a flat action index to (reference-path index, lateral offset).
            /// AD-specific: not a base-class virtual — called only within VecQMDP_AD.
            void decomposeAction(uint32_t action_idx, int &path_idx_out, float &offset_out) const;

            /// Vectorised wrapper: decompose every action in @p action_idxs into
            /// the corresponding path-index and lateral-offset SIMD vectors.
            void decomposeActionBatch(const IVectorT_qmdp &action_idxs, IVectorT_qmdp &path_idxs_out,
                                      FVectorT_qmdp &offsets_out) const;

            // ------------------- Initialisation / reset -------------------

            /// Allocate AD-specific flat state buffers (called from constructor).
            void initializeDataStructures();

            /// Reset mutable AD state before each planning call.
            void resetDataStructures();

            // ------------------- Ego helpers -------------------

            void updateEgoInitialState(const std::shared_ptr<core::EgoState> &ego_state);

            /// Encode (path_idx, offset_idx) → flat action index.
            inline uint32_t encodeAction(uint32_t path_idx, uint32_t offset_idx) const
            {
                return path_idx * utils::NUM_LATERAL_OFFSETS + offset_idx;
            }

            /// Decode flat action index → (path_idx, offset_idx).
            inline std::pair<uint32_t, uint32_t> decodeAction(uint32_t action_idx) const
            {
                return {utils::ACTION_TO_PATH[action_idx], utils::ACTION_TO_OFFSET[action_idx]};
            }

            /// Find the offset table index closest to @p offset_f.
            inline uint32_t convertOffsetToIndexRobust(float offset_f) const
            {
                constexpr float EPS = 1e-6f;
                for (uint32_t i = 0; i < utils::NUM_LATERAL_OFFSETS; ++i)
                    if (std::abs(offset_f - utils::PATH_OFFSETS_FLOAT[i]) < EPS)
                        return i;
                return 1; // fallback: zero-offset index
            }

            /// True if @p target_path_idx is reachable (adjacent) from @p current_path_idx.
            inline bool isPathReachable(uint32_t current_path_idx, uint32_t target_path_idx) const
            {
                return std::abs(static_cast<int>(current_path_idx) - static_cast<int>(target_path_idx)) <= 1;
            }

            // ------------------- Exo index helpers -------------------

            /// Flat index layout: [scenario][time][vehicle]
            inline uint32_t getExoIdxTimeVehicle(uint32_t s, uint32_t t, uint32_t v) const
            {
                return s * exo_total_size_ + t * vehicle_size_ + v;
            }

            /// Flat index layout: [scenario][vehicle][time]
            inline uint32_t getExoIdxVehicleTime(uint32_t s, uint32_t v, uint32_t t) const
            {
                return s * exo_total_size_ + v * global_time_size_ + t;
            }

            /// Flat index for the exogenous data belonging to @p node_idx at (v, t).
            inline uint32_t getNodeExoIdx(uint32_t node_idx, uint32_t t = 0, uint32_t v = 0) const
            {
                return getExoIdxVehicleTime(getScenarioIdx(node_idx), v, t);
            }

          protected:
            // ------------------- Shared context (step simulator) -------------------
            std::shared_ptr<ContextQMDP> context_qmdp_;

            // ------------------- AD-specific dimension info -------------------
            uint32_t global_time_size_;  ///< DUMMY_TIME_STEPS
            uint32_t step_time_size_;    ///< STEP_TIME_LEN
            uint32_t vehicle_size_;      ///< MAX_SIM_VEHICLES
            uint32_t path_num_;          ///< NUM_REF_PATH
            uint32_t exo_total_size_;    ///< DUMMY_TIME_STEPS * MAX_SIM_VEHICLES
            uint32_t vehicle_valid_num_; ///< Number of currently observed exo agents

            // ------------------- Ego initial state -------------------
            float initial_ego_x_;
            float initial_ego_y_;
            float initial_ego_v_;
            float initial_ego_a_;
            float initial_ego_theta_;

            // ------------------- Per-node ego state buffers -------------------
            AlignedVectorFloat                 ego_xs_flat_;
            AlignedVectorFloat                 ego_ys_flat_;
            AlignedVectorFloat                 ego_vs_flat_;
            AlignedVectorFloat                 ego_as_flat_;
            AlignedVectorFloat                 ego_thetas_flat_;
            AlignedVectorInt                   ego_curr_ref_path_idxs_;
            AlignedVectorFloat                 ego_curr_lateral_offsets_;
            std::vector<std::shared_ptr<Path>> ego_ref_paths_;

            // ------------------- Exogenous agent state buffers -------------------
            /// Layout: [scenario][vehicle][time] — SIMD-friendly sequential access
            AlignedVectorFloat exo_xs_flat_;
            AlignedVectorFloat exo_ys_flat_;
            AlignedVectorFloat exo_vs_flat_;
            AlignedVectorFloat exo_thetas_flat_;
            AlignedVectorFloat exo_cos_thetas_flat_;
            AlignedVectorFloat exo_sin_thetas_flat_;
            AlignedVectorFloat exo_bb_extent_xs_; ///< half-widths  [vehicle]
            AlignedVectorFloat exo_bb_extent_ys_; ///< half-lengths [vehicle]

            /// Frenet s-coordinate per reference path: [ref_path][scenario*vehicle*time]
            std::vector<AlignedVectorFloat> exo_ss_flat_;
            /// Frenet l-coordinate per reference path
            std::vector<AlignedVectorFloat> exo_ls_flat_;
            /// Projected l-radius for collision broadphase per reference path
            std::vector<AlignedVectorFloat> exo_ls_projected_radius_flat_;

            /// STRtrees indexed [ref_path][scenario * time]
            std::vector<std::vector<std::shared_ptr<STRtree>>> exo_strtrees_;

            // ------------------- Per-node exo inactive flags -------------------
            /// node_exo_inactive_flags_[node_idx][vehicle] == 0xFFFFFFFF when inactive
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> node_exo_inactive_flags_;
        };

    } // namespace planning
} // namespace vec_qmdp
