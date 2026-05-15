/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file rock_sample_static_vecqmdp.hpp
 * @brief Static (pre-allocated) VecQMDP solver for the Rock Sample POMDP.
 *
 * Derives from vec_qmdp::planning::VecQMDP and uses the fully pre-allocated
 * balanced tree rather than the dynamic pool used by DynVecQMDP.
 *
 * Key differences from RockSampleVecQMDP (DynVecQMDP variant)
 * ------------------------------------------------------------
 *  - The entire tree is allocated at construction time:
 *      node_size = tree_node_size × scenario_size
 *    where tree_node_size = sum_{k=0}^{tree_height} num_actions^k.
 *    For tree_height = 4 and num_actions = 13: tree_node_size ≈ 31 K → ~17 MB ✓
 *    For tree_height = 6 and num_actions = 13: tree_node_size ≈ 5.2 M → ~2.5 GB ✗
 *    => Recommended: tree_height ≤ 4 for the 13-action rock sample.
 *
 *  - scenario_size is FIXED to IVectorT::num_scalars (= SIMD width, typically 8).
 *    The num_scenarios constructor argument is accepted but IGNORED; the actual
 *    value is always IVectorT::num_scalars.  To increase effective coverage,
 *    increase num_threads — each worker independently uses 8 scenarios.
 *
 *  - Domain state (robot pos, rock bits) is stored in flat arrays indexed by
 *    getNodeIdx(tree_relative_pos, scenario_idx).
 *
 *  - Backpropagation uses the implicit heap formula:
 *      parent_rel = (child_rel - 1) / num_actions
 *    instead of the explicit parent pointer used by DynVecQMDP.
 *
 * Solver selection
 * ----------------
 * Define USE_STATIC_SOLVER in main.cpp (or via -DUSE_STATIC_SOLVER) to
 * compile against this class instead of RockSampleVecQMDP.
 *
 * Rollout policy
 * --------------
 * Define ROLLOUT_ENT_STATIC to enable the ENT (Explore Nearest in Thresholded
 * state) lower bound.  When undefined, the always-east lower bound is used.
 */
#pragma once

extern int STEP;
extern bool ENABLE_DEBUG;

// ============================================================================
// Rollout policy selection
// ============================================================================
#define ROLLOUT_ENT_STATIC  ///< ENT rollout; comment out for always-east

#include <planning/vec_qmdp_static.hpp>
#include <utils/global_utils.hpp>
#include <utils/params.hpp>

#include <despot/util/coord.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rock_sample_vecqmdp {

// ============================================================================
// RockSampleStaticVecQMDP  (pre-allocated tree edition)
// ============================================================================

class RockSampleStaticVecQMDP : public vec_qmdp::planning::VecQMDP
{
public:
    using IVectorT = vec_qmdp::utils::IVectorT_qmdp;
    using FVectorT = vec_qmdp::utils::FVectorT_qmdp;
    using BeliefTreeSearchThreadResult = vec_qmdp::planning::VecQMDP::BeliefTreeSearchThreadResult;

    /// Action indices — identical to RockSampleVecQMDP.
    enum Action : int {
        A_NORTH  = 0,
        A_EAST   = 1,
        A_SOUTH  = 2,
        A_WEST   = 3,
        A_SAMPLE = 4
        // A_SENSE_k = 5 + k  (k = 0 .. num_rocks-1)
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @param size                     Grid side-length.
     * @param num_rocks                Number of rocks.
     * @param rock_pos                 Rock positions indexed [0, num_rocks).
     * @param start_x / start_y        Robot start position.
     * @param half_efficiency_distance Half-efficiency distance for sensing.
     * @param tree_height              BeliefTreeSearch lookahead depth.
     *                                 MUST be ≤ 4 for the 13-action rock sample!
     * @param num_scenarios            Accepted but ignored; always uses
     *                                 IVectorT::num_scalars (= 8).
     * @param num_threads              Parallel workers.
     */
    RockSampleStaticVecQMDP(int                               size,
                            int                               num_rocks,
                            const std::vector<despot::Coord>& rock_pos,
                            int                               start_x,
                            int                               start_y,
                            double                            half_efficiency_distance,
                            uint32_t                          tree_height  = 4,
                            int                               num_scenarios = 8,
                            int                               num_threads  = 1);

    // -----------------------------------------------------------------------
    // Belief initialisation
    // -----------------------------------------------------------------------

    /**
     * Seed root nodes from robot position and rock probabilities.
     * Scenarios are sampled from the independent rock probabilities.
     * @param robot_pos    Robot position (flat index: y*size+x).
     * @param rock_probs   P(rock_i is good) for each rock.
     */
    void sampleScenarios(int robot_pos, const std::vector<float>& rock_probs);

    // -----------------------------------------------------------------------
    // Node expansion
    // -----------------------------------------------------------------------

    /**
     * Expand one node per scenario using the pre-allocated static tree.
     * Reads parent domain state via SIMD gather, runs stepBatch, writes child
     * domain state.  Terminal parents are skipped.
     *
     * @param parent_nodes  AlignedVectorInt[s] = global tree index of the
     *                      parent to expand for scenario s.
     * @param action_idxs   std::vector<int>[s] = action index for scenario s.
     */
    void expandNodesBatch(const AlignedVectorInt& parent_nodes,
                          const std::vector<int>& action_idxs);

#ifdef ENABLE_HOMOGENOUS_SEARCH
    /**
     * Homogenous expand: all scenarios share the same relative parent node and
     * the same action.  The parent for scenario s is:
     *   global_parent_s = parent_rel + s * tree_node_size_
     *
     * @param parent_rel  Relative tree index of the parent (scenario-0 == global).
     * @param action_idx  Single action applied to every scenario.
     */
    void expandNodesBatch(uint32_t parent_rel, int action_idx);
#endif // ENABLE_HOMOGENOUS_SEARCH

    // -----------------------------------------------------------------------
    // BeliefTreeSearch
    // -----------------------------------------------------------------------

    /// Core BeliefTreeSearch entry point for a single worker.
    /// Resets tree structures, seeds scenarios, and runs the search loop.
    void beliefTreeSearch(int robot_pos,
                          const std::vector<float>& rock_probs,
                          int max_iters);

    // -----------------------------------------------------------------------
    // Parallel search entry point
    // -----------------------------------------------------------------------

    /**
     * Run the full parallel BeliefTreeSearch search and return the best action index.
     * @param robot_pos    Current robot position (flat index).
     * @param rock_probs   Current belief: P(rock_i is good) for each rock.
     * @param max_iters    Maximum BeliefTreeSearch iterations per worker.
     */
    int parallelBeliefTreeSearch(int robot_pos, const std::vector<float>& rock_probs,
                              int max_iters = 5000);

    /// Return the best action index from aggregated Q-values after parallelBeliefTreeSearch().
    int getBestAction() const;

    /// Total simulation count across all workers.
    inline int getTotalSimulationCount() const { return total_simulation_count_static_; }

    /// Print detailed timing statistics collected during the last parallelBeliefTreeSearch() call.
    void printTimingStats() const;

    // -----------------------------------------------------------------------
    // QMDP augmentation
    // -----------------------------------------------------------------------

    void setSensePriorBonus(float v) { sense_prior_bonus_ = v; }
    void setEntropyLambda(double v)  { entropy_lambda_ = v; }

    // -----------------------------------------------------------------------
    // State encoding helpers
    // -----------------------------------------------------------------------

    int encodeState(int pos_idx, int rock_bits) const
    {
        return (pos_idx << num_rocks_) | rock_bits;
    }
    int decodePos(int state_id)   const { return state_id >> num_rocks_; }
    int decodeRocks(int state_id) const
    {
        return state_id & ((1 << num_rocks_) - 1);
    }

    void printTree(int max_depth = 10, std::ostream& out = std::cout) const override;

protected:
    // -----------------------------------------------------------------------
    // VecQMDP pure-virtual hooks
    // -----------------------------------------------------------------------

    std::vector<uint32_t>
        computeNodeCandidateActions(uint32_t node_idx) override;

    std::shared_ptr<vec_qmdp::planning::VecQMDP>
        makeWorkerInstance() const override;

private:
    // =================== Domain parameters ==================================
    int    size_;
    int    num_rocks_;
    int    start_x_, start_y_;
    double half_efficiency_distance_;

    std::vector<despot::Coord> rock_pos_;
    std::vector<int>           grid_;

    // =================== QMDP augmentation ==================================
    float  sense_prior_bonus_{0.0f};
    double entropy_lambda_{2.0};

    std::vector<float> current_rock_probs_;

    // =================== Static tree domain state ===========================
    //
    // Indexed by getNodeIdx(relative_tree_pos, scenario_idx).
    // Size = tree_node_size_ * scenario_size_  (same shape as the VecQMDP
    // base-class arrays).
    //
    std::vector<int> robot_pos_static_;  ///< pos_index = y*size+x
    std::vector<int> rock_bits_static_;  ///< rock-status bitmask

    // =================== Precomputed tables =================================
    std::vector<float> discount_pow_table_;  ///< 10 * gamma^k
    std::vector<int>   action_dx_;
    std::vector<int>   action_dy_;
    std::vector<int>   rock_pos_x_;
    std::vector<int>   rock_pos_y_;

    // =================== Per-scenario random engines ========================
    std::vector<std::mt19937> rngs_;

    // =================== Simulation counters ================================
    int total_simulation_count_static_{0};
    int simulation_count_static_{0};

    // Best rock index to sense (set by applyEntropyBonuses, read by getBestAction)
    mutable int best_sense_rock_{0};
    
    // =================== Private helpers ====================================

    /**
     * SIMD batch transition — identical physics to RockSampleVecQMDP::stepBatch.
     */
    void stepBatch(const IVectorT& pos_vec,
                   const IVectorT& rocks_vec,
                   const IVectorT& action_vec,
                   IVectorT&       out_pos,
                   IVectorT&       out_rocks,
                   IVectorT&       out_active,
                   FVectorT&       out_reward);

#ifdef ENABLE_HOMOGENOUS_SEARCH
    /**
     * Homogenous stepBatch: all SIMD lanes execute the same scalar action.
     * Branches on action category (sense / sample / move) to skip irrelevant
     * vector operations entirely, rather than blending per-lane masks.
     */
    void stepBatch(const IVectorT& pos_vec,
                   const IVectorT& rocks_vec,
                   int             action_idx,
                   IVectorT&       out_pos,
                   IVectorT&       out_rocks,
                   IVectorT&       out_active,
                   FVectorT&       out_reward);
#endif // ENABLE_HOMOGENOUS_SEARCH

    FVectorT rolloutEast(const IVectorT& new_pos_vec,
                         const IVectorT& active_vec) const;

    /**
     * ENT rollout — optimised general version.
     * Pre-filters candidate rocks (prob >= 0.1) once per call to eliminate
     * per-rock branching in the hot inner loop.
     */
    FVectorT rolloutENT(const IVectorT& new_pos_vec,
                        const IVectorT& new_rocks_vec,
                        const IVectorT& active_vec,
                        bool print = false) const;

#ifdef ENABLE_HOMOGENOUS_SEARCH
    /**
     * ENT rollout specialised for the homogenous-search case, where all 8
     * SIMD lanes share the same robot position at the start of each call.
     *
     * Per outer iteration:
     *  - Uniform positions → scalar distances + insertion-sort + SIMD
     *    is_good-only scan with early exit (test_zero / all()).
     *  - Non-uniform positions → falls back to the standard pre-filtered
     *    SIMD inner loop (same as rolloutENT).
     */
    FVectorT rolloutENT_Homogenous(const IVectorT& new_pos_vec,
                                   const IVectorT& new_rocks_vec,
                                   const IVectorT& active_vec) const;
#endif // ENABLE_HOMOGENOUS_SEARCH


    float computeEntropyBonus(int rock_idx) const;

    void applyEntropyBonuses();
};

} // namespace rock_sample_vecqmdp
