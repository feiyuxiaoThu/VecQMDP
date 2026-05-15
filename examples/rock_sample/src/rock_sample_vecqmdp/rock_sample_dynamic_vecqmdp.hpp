/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file rock_sample_dynamic_vecqmdp.hpp
 * @brief DynVecQMDP-based BeliefTreeSearch solver for the Rock Sample POMDP.
 *
 * Derives from vec_qmdp::planning::DynVecQMDP and stores per-node domain state
 * in two plain std::vector<int> arrays that grow alongside the dynamic SoA pool:
 *
 *   robot_pos_flat_[global_pool_idx]  --  robot grid position index (y*size+x)
 *   rock_bits_flat_[global_pool_idx]  --  rock-status bitmask (bit i = good)
 *
 * Unlike the static VecQMDP variant, no tree memory is pre-allocated at
 * construction: nodes are created on demand by allocateChildBlock() inside
 * expandNodesBatch(). For tree_height = 90 and num_actions = 13 this reduces
 * peak memory from O(num_actions^tree_height) to O(max_iters * num_actions).
 *
 * State encoding follows the DESPOT BaseRockSample convention:
 *   state_id = (pos_index << num_rocks) | rock_bits
 *
 * Threading:
 *   When num_threads > 1 the DynVecQMDP parallel infrastructure is used.
 *   Each worker independently runs BeliefTreeSearch on its own dynamic pool and the
 *   Q-values are averaged at the end.
 */
#pragma once

// ============================================================================
// Rollout policy selection
// Define ROLLOUT_ENT to use the ENT (Explore Nearest in Thresholded state)
// rollout, which greedily visits the nearest good rock in each scenario before
// heading east.  Comment out to use the simpler always-east lower bound.
// ============================================================================
#define ROLLOUT_ENT   ///< ENT rollout; default is always-east lower bound

#include <planning/vec_qmdp_dynamic.hpp>
#include <utils/global_utils.hpp>
#include <utils/params.hpp>

#include <despot/util/coord.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

extern int STEP;
extern bool ENABLE_DEBUG;

namespace rock_sample_vecqmdp {

// ============================================================================
// RockSampleVecQMDP  (dynamic-pool edition)
// ============================================================================

class RockSampleVecQMDP : public vec_qmdp::planning::DynVecQMDP
{
public:
    using IVectorT = vec_qmdp::utils::IVectorT_qmdp;
    using FVectorT = vec_qmdp::utils::FVectorT_qmdp;

    // Convenience aliases re-used from DynVecQMDP
    using BeliefTreeSearchThreadResult = vec_qmdp::planning::DynVecQMDP::BeliefTreeSearchThreadResult;
    using DynExploreResult = vec_qmdp::planning::DynExploreResult;

    /// Action indices — match DESPOT Compass + E_SAMPLE convention exactly.
    enum Action : int {
        A_NORTH  = 0,   ///< Move north  (+y)
        A_EAST   = 1,   ///< Move east   (+x)  — reaches goal when x == size-1
        A_SOUTH  = 2,   ///< Move south  (-y)
        A_WEST   = 3,   ///< Move west   (-x)
        A_SAMPLE = 4    ///< Sample rock at current cell
        // A_SENSE_k = 5 + k  (k = 0 .. num_rocks-1)
    };

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /**
     * @param size                     Grid side-length (square grid).
     * @param num_rocks                Number of rocks.
     * @param rock_pos                 Rock positions indexed [0, num_rocks).
     * @param start_x                  Robot start column.
     * @param start_y                  Robot start row.
     * @param half_efficiency_distance Half-efficiency distance for sensing.
     * @param tree_height              BeliefTreeSearch lookahead depth.
     * @param num_scenarios            Number of parallel BeliefTreeSearch scenario trees.
     * @param num_threads              Parallel workers (1 = single-threaded).
     */
    RockSampleVecQMDP(int                               size,
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
     * Seed root nodes from a flat particle set (state_id encoded).
     * Exactly scenario_size_ root nodes are populated; particles are cycled
     * if the set is smaller.
     *
     * Must be called after resetDynStructures() + initDynRoots() have been
     * invoked (i.e., from within parallelBeliefTreeSearch / beliefTreeSearch).  Roots are
     * always allocated at global indices 0 … scenario_size_-1.
     */
    void sampleScenarios(const std::vector<int>& state_ids);

    /**
     * @brief Initialize belief from probability distribution.
     * Sample rock states from independent Bernoulli distributions.
     *
     * @param robot_pos  Robot starting position (pos_idx = y * size + x).
     * @param rock_probs Probability that each rock is good [num_rocks_].
     */
    void sampleScenarios(int robot_pos, const std::vector<float>& rock_probs);

    // -----------------------------------------------------------------
    // Node expansion
    // -----------------------------------------------------------------

    /**
     * Expand all scenario_size_ nodes from explore_results, processing them
     * in SIMD batches of IVectorT::num_scalars.  Allocates child blocks as
     * needed and writes domain + pool state for every expanded child.
     */
    void expandNodesBatch(const std::vector<DynExploreResult>& explore_results);

#ifdef ENABLE_HOMOGENOUS_SEARCH
    /**
     * @brief Expand all scenarios at parent_level with action_idx (homogenous).
     *
     * All scenarios share the same tree path, so we expand child_base + action_idx
     * for every scenario simultaneously.
     *
     * @param parent_level  Current level in homogenous tree.
     * @param action_idx    Action to take from all parents.
     */
    void expandNodesBatch(int32_t parent_level, int action_idx);
#endif // ENABLE_HOMOGENOUS_SEARCH

    // -----------------------------------------------------------------
    // Single-threaded search
    // -----------------------------------------------------------------

    /// Core single-threaded BeliefTreeSearch loop (state_ids overload).
    /// Handles reset, sampling, and the full search loop internally.
    void beliefTreeSearch(const std::vector<int>& state_ids, int max_iters);

    /// Core single-threaded BeliefTreeSearch loop (robot_pos / rock_probs overload).
    /// Handles reset, sampling, and the full search loop internally.
    void beliefTreeSearch(int robot_pos, const std::vector<float>& rock_probs, int max_iters);

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /**
     * Run the full parallel BeliefTreeSearch search and return the best action index.
     * @param state_ids  Current belief particles (state_id encoded).
     * @param max_iters  Maximum BeliefTreeSearch iterations per worker.
     */
    int parallelBeliefTreeSearch(const std::vector<int>& state_ids, int max_iters = 5000);

    /**
     * Run the full parallel BeliefTreeSearch search and return the best action index.
     * @param robot_pos   Robot starting position (pos_idx = y * size + x).
     * @param rock_probs  Probability that each rock is good [num_rocks_].
     * @param max_iters   Maximum BeliefTreeSearch iterations per worker.
     */
    int parallelBeliefTreeSearch(int robot_pos, const std::vector<float>& rock_probs, int max_iters = 5000);

    /// Return the best action index from aggregated Q-values after parallelBeliefTreeSearch().
    int getBestAction() const;
    // -----------------------------------------------------------------------
    // Post-search visualization  (available only when macros are defined)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // QMDP augmentation parameters
    // -----------------------------------------------------------------------

    /// Fixed reward bonus added to sense-action nodes during tree search.
    void setSensePriorBonus(float v)  { sense_prior_bonus_ = v; }

    /// Weight lambda for the information-gain bonus on sense actions.
    void setEntropyLambda(double v)   { entropy_lambda_ = v; }

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

protected:
    // -----------------------------------------------------------------------
    // DynVecQMDP pure-virtual hooks
    // -----------------------------------------------------------------------

    std::vector<uint32_t>
        computeNodeCandidateActionsDyn(int32_t global_idx) override;

    std::shared_ptr<vec_qmdp::planning::DynVecQMDP>
        makeWorkerInstance() const override;

private:
    // =================== Domain parameters ==================================
    int    size_;
    int    num_rocks_;
    int    start_x_, start_y_;
    double half_efficiency_distance_;

    std::vector<despot::Coord> rock_pos_;
    std::vector<int>           grid_;          ///< grid_[y*size+x] = rock index, -1 if empty

    // =================== QMDP augmentation ==================================
    // sense_prior_bonus_ is kept as a tunable parameter but is no longer
    // injected into the BeliefTreeSearch tree rewards (see expandNodesBatch).  It may
    // be repurposed for UCB initialisation in the future.
    float  sense_prior_bonus_{0.0f};

    // Scale factor for the Shannon information-gain bonus applied post-search.
    // Game rewards are in units of ±10; with max IG = 1 bit this lambda of
    // 5.0 makes a fully-uncertain rock worth ~5 sense-reward, which is
    // large enough to beat nearby movement but correctly collapses to 0 when
    // belief is already converged.
    double entropy_lambda_{2.0};

    std::vector<int> current_belief_particles_;

    /// Per-scenario rollout value (always-east lower bound); indexed by
    /// scenario index within the current batch (0 … SIMD_BATCH - 1).
    std::vector<float> rollout_values_;

    // =================== Dynamic DOD state buffers ==========================
    //
    // Indexed by global pool index (same as the DynSoAPool index).
    // Grown in lockstep with pool_ via ensureDomainCapacity().
    //
    std::vector<int> robot_pos_flat_;  ///< pos_index = y*size+x
    std::vector<int> rock_bits_flat_;  ///< rock-status bitmask (bit i = rock i is good)

    // =================== Precomputed rollout value table ====================
    //
    // discount_pow_table_[k] = 10.0f * gamma^k  for k = 0 .. size_
    //
    std::vector<float> discount_pow_table_;

    // =================== Precomputed action-delta lookup tables ==============
    std::vector<int> action_dx_;  ///< x-displacement per action
    std::vector<int> action_dy_;  ///< y-displacement per action

    // Separate x/y arrays for rock positions — required for SIMD gather.
    std::vector<int> rock_pos_x_;
    std::vector<int> rock_pos_y_;

    // =================== Per-scenario random engines ========================
    std::vector<std::mt19937> rngs_;

    // =================== Current rock good probabilities ====================
    std::vector<float> current_rock_probs_;

    // =================== exp data ===========================================
    int simulation_count_{0};

    // Best rock index to sense (set by applyEntropyBonuses, read by getBestAction)
    mutable int best_sense_rock_{0};

    // =================== Timing stats =======================================
    // Accumulated inside expandNodesBatch(); reset at the start of parallelBeliefTreeSearch()
    // and (for workers) inside the parallel-dispatch lambda.
    double  total_step_batch_us_{0.0};   ///< total µs spent in stepBatch calls
    int64_t step_batch_call_count_{0};   ///< number of stepBatch batch calls
    double  total_rollout_us_{0.0};      ///< total µs spent in rollout calls
    int64_t rollout_call_count_{0};      ///< number of rollout batch calls

    // =================== Private helpers ====================================

    /**
     * Ensure robot_pos_flat_ and rock_bits_flat_ can hold at least min_size
     * entries (indexed 0 … min_size - 1).  Called whenever new pool nodes are
     * allocated.
     */
    void ensureDomainCapacity(int32_t min_size);

    /**
     * Wrapper around DynVecQMDP::allocateChildBlock that also grows the domain
     * buffers to cover the newly allocated child slots.
     */
    int32_t allocateDomainChildBlock(int32_t parent_g);

    /**
     * Scalar reference step (not on the hot path).
     * @return Immediate reward; sets out_pos, out_rocks, out_terminal.
     */
    float applyAction(int           pos,
                      int           rock_bits,
                      int           action,
                      std::mt19937& rng,
                      int&          out_pos,
                      int&          out_rocks,
                      bool&         out_terminal) const;

    /**
     * SIMD batch transition for SIMD_BATCH scenarios simultaneously.
     */
    void stepBatch(const IVectorT& pos_vec,
                   const IVectorT& rocks_vec,
                   const IVectorT& action_vec,
                   IVectorT&       out_pos,
                   IVectorT&       out_rocks,
                   IVectorT&       out_active,
                   FVectorT&       out_reward);

    /**
     * Always-east rollout lower bound (SIMD batch).
     * Returns rollout value = 10 * gamma^(size-1-x) for active lanes, 0 for
     * terminated lanes.
     */
    FVectorT rolloutEast(const IVectorT& new_pos_vec,
                         const IVectorT& active_vec) const;

    /**
     * ENT (Explore Nearest in Thresholded state) rollout (SIMD batch).
     * For each lane, greedily visits the nearest good rock (bit set in
     * new_rocks_vec), accumulates discounted +10 rewards, then adds the
     * always-east exit bonus from the final position.
     * Returns 0 for terminated lanes.
     */
    FVectorT rolloutENT(const IVectorT& new_pos_vec,
                        const IVectorT& new_rocks_vec,
                        const IVectorT& active_vec,
                        bool print = false) const;

    /// Compute the Shannon entropy information-gain bonus for sensing rock_idx.
    float computeEntropyBonus(int rock_idx,
                              const std::vector<int>& state_ids) const;

    /// Compute the Shannon entropy information-gain bonus using probabilistic belief.
    float computeEntropyBonus(int rock_idx, int robot_pos) const;

    /// Add entropy information-gain bonuses to aggregated_action_values_ for
    /// all sense actions after parallel search completes.
    void applyEntropyBonuses(const std::vector<int>& state_ids);

    /// Add entropy information-gain bonuses using probabilistic belief.
    void applyEntropyBonuses(int robot_pos);
};

} // namespace rock_sample_vecqmdp
