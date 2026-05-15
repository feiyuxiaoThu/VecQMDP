/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file rock_sample_dynamic_vecqmdp.cpp
 * @brief DynVecQMDP BeliefTreeSearch solver implementation for the Rock Sample POMDP.
 *
 * Key design choices vs. the static VecQMDP version:
 *
 *  Dynamic pool
 *    Nodes live in DynSoAPool and are allocated on demand.  No exponential
 *    pre-allocation is required, so tree_height = 90 is tractable.
 *
 *  DOD state layout
 *    robot_pos_flat_[global_pool_idx] = y * size + x  (int)
 *    rock_bits_flat_[global_pool_idx] = bitmask (bit i = rock i is good)
 *    These plain vectors grow in lockstep with the pool via
 *    ensureDomainCapacity().
 *
 *  SIMD batching
 *    expandNodesBatch() iterates over all scenario_size_ scenarios in
 *    batches of IVectorT::num_scalars (= 8 for AVX2).  Per-batch SIMD
 *    gathers gather parent state; per-batch stepBatch() computes all
 *    transitions; a scalar scatter loop writes child state back.
 *
 *  Parallelism
 *    When num_threads > 1, DynVecQMDP::initParallelInfrastructure() spawns
 *    N worker instances.  dispatchParallelSearch() runs beliefTreeSearch()
 *    on each worker in its own thread and averages the per-action Q-values.
 */
#include "rock_sample_dynamic_vecqmdp.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <bitset>

namespace rock_sample_vecqmdp {

// ============================================================================
// Construction
// ============================================================================

RockSampleVecQMDP::RockSampleVecQMDP(
    int                               size,
    int                               num_rocks,
    const std::vector<despot::Coord>& rock_pos,
    int                               start_x,
    int                               start_y,
    double                            half_efficiency_distance,
    uint32_t                          tree_height,
    int                               num_scenarios,
    int                               num_threads)
    : DynVecQMDP(num_scenarios,
                 6u,
                 tree_height)
    , size_(size)
    , num_rocks_(num_rocks)
    , start_x_(start_x)
    , start_y_(start_y)
    , half_efficiency_distance_(half_efficiency_distance)
    , rock_pos_(rock_pos)
{
    if (num_rocks_ > 30)
        throw std::invalid_argument("num_rocks exceeds int bitmask capacity (30)");

    // Build flat grid: grid_[y*size+x] = rock index, -1 if empty
    grid_.assign(size_ * size_, -1);
    for (int i = 0; i < num_rocks_; ++i)
        grid_[rock_pos_[i].y * size_ + rock_pos_[i].x] = i;

    // Pre-allocate domain buffers for the root nodes (scenario_size_ entries).
    // Further growth happens in allocateDomainChildBlock().
    ensureDomainCapacity(static_cast<int32_t>(scenario_size_));

    // Action-delta lookup tables (indexed by action index, size=6: 4 moves + sample + sense)
    action_dx_.assign(6u, 0);
    action_dy_.assign(6u, 0);
    action_dy_[A_NORTH] = +1;
    action_dx_[A_EAST]  = +1;
    action_dy_[A_SOUTH] = -1;
    action_dx_[A_WEST]  = -1;

    // Separate x/y arrays for rock positions (needed for SIMD gather)
    rock_pos_x_.resize(static_cast<size_t>(num_rocks_));
    rock_pos_y_.resize(static_cast<size_t>(num_rocks_));
    for (int i = 0; i < num_rocks_; ++i)
    {
        rock_pos_x_[i] = rock_pos_[i].x;
        rock_pos_y_[i] = rock_pos_[i].y;
    }

    // Per-scenario RNGs
    rngs_.reserve(scenario_size_);
    for (uint32_t s = 0; s < scenario_size_; ++s)
        rngs_.emplace_back(static_cast<uint32_t>(
            vec_qmdp::utils::RANDOM_SEED) + s * 1009u + 999983u);

    // Per-SIMD-batch rollout value storage (8 lanes = one IVectorT batch)
    rollout_values_.assign(scenario_size_, 0.0f);

    // Current rock good probabilities
    current_rock_probs_.assign(num_rocks_, 0.0f);

    // Hyperparameter defaults
    setDepthSyncLambda(0.0f);
    setDiscountFactor(0.95);

    // Precompute 10 * gamma^k for k = 0 .. 2*size_.
    // Base rollout (always-east) needs indices 0..size_.
    // Good-rock bonus needs indices up to 2*(size_-1)+1 ≤ 2*size_-1.
    discount_pow_table_.resize(static_cast<size_t>(2 * size_) + 1);
    for (int k = 0; k <= 2 * size_; ++k)
        discount_pow_table_[k] = 10.0f * static_cast<float>(
            std::pow(discount_factor_, static_cast<double>(k)));
            
    setEntropyLambda(10.0);
    setExplorationConstant(2.0f);
    setMaxBackupLambda(0.000f);
    setPrunedThreshold(-50.0f);
    setMaxPlanningTimeMs(500.0);
    setEarlyTermMinExpandCalls(500);
    setEarlyTermCheckInterval(15);
    setEarlyTermStableIterations(50);
    setEarlyTermMinBestVisits(20);
    setEarlyTermQChangeThreshold(0.1f);

    // Spawn worker pool after all domain state is initialised
    if (num_threads > 1)
        initParallelInfrastructure(static_cast<size_t>(num_threads));
}

// ============================================================================
// Belief initialisation
// ============================================================================

void RockSampleVecQMDP::sampleScenarios(const std::vector<int>& state_ids)
{
    const size_t n = state_ids.size();
    if (n == 0) return;

    // After pool_.reset() + initDynRoots(), roots occupy global indices 0..S-1.
    for (uint32_t s = 0; s < scenario_size_; ++s)
    {
        const int sid = state_ids[s % n];
        // scenario_roots_[s] == s after initDynRoots() (first alloc starts at 0)
        robot_pos_flat_[s] = decodePos(sid);
        rock_bits_flat_[s] = decodeRocks(sid);
    }
}

void RockSampleVecQMDP::sampleScenarios(int robot_pos, const std::vector<float>& rock_probs)
{
    for (uint32_t s = 0; s < scenario_size_; ++s)
    {
        const int32_t root_g = scenario_roots_[s];
        robot_pos_flat_[root_g] = robot_pos;

        // Sample rock states from Bernoulli distributions
        int rock_bits = 0;
        for (int r = 0; r < num_rocks_; ++r)
        {
            std::uniform_real_distribution<float> ud(0.0f, 1.0f);
            if (ud(rngs_[s]) < rock_probs[r])
            {
                rock_bits |= (1 << r);
            }
        }
        rock_bits_flat_[root_g] = rock_bits;

        if (s == 0) // Debug output for the first scenario
            std::cout << "[DEBUG] Scenario " << s << ": robot_pos=" << robot_pos
                    << ", rock_bits=" << std::bitset<30>(rock_bits) << "\n";
    }
}

// ============================================================================
// expandNodesBatch — SIMD-batched node expansion (dynamic pool edition)
// ============================================================================
//
// Processes all scenario_size_ scenarios in SIMD batches of
// IVectorT::num_scalars (= 8 for AVX2).  For each batch:
//
//   1. Allocate child blocks for parents that don't have children yet.
//   2. IVectorT::gather  — load all 8 parent states in two SIMD instructions.
//   3. stepBatch         — compute all 8 transitions in parallel.
//   4. Rollout + sense bonus computation (SIMD).
//   5. Scalar scatter-write — results to (different) child global indices.
//
void RockSampleVecQMDP::expandNodesBatch(
    const std::vector<DynExploreResult>& explore_results)
{
    constexpr uint32_t BATCH = static_cast<uint32_t>(IVectorT::num_scalars); // 8

    for (uint32_t batch_start = 0; batch_start < scenario_size_; batch_start += BATCH)
    {
        const uint32_t batch_end  = std::min(batch_start + BATCH, scenario_size_);
        const uint32_t batch_size = batch_end - batch_start;

        // ---- Collect per-lane indices, allocate children if needed ----
        int32_t parent_g_arr[8] = {};
        int32_t action_arr  [8] = {};
        int32_t child_g_arr [8] = {};
        bool    valid       [8] = {};

        for (uint32_t i = 0; i < BATCH; ++i)
        {
            if (i >= batch_size)
            {
                // Padding lane: point at root[0] as harmless dummy parent.
                parent_g_arr[i] = scenario_roots_[0];
                action_arr  [i] = 0;
                child_g_arr [i] = vec_qmdp::planning::DYN_INVALID_IDX;
                valid       [i] = false;
                continue;
            }

            const uint32_t s         = batch_start + i;
            const int32_t  parent_g  = explore_results[s].parent_global_idx;
            const uint32_t action    = explore_results[s].action_idx;

            parent_g_arr[i] = parent_g;
            action_arr  [i] = static_cast<int32_t>(action);

            // Allocate child block on first visit to this parent
            if (pool_.child_base(parent_g) == vec_qmdp::planning::DYN_INVALID_IDX)
                allocateDomainChildBlock(parent_g);

            const int32_t child_base = pool_.child_base(parent_g);
            child_g_arr[i] = child_base + static_cast<int32_t>(action);

            // Only expand if still within depth limit
            valid[i] = (pool_.depth(parent_g) < static_cast<int32_t>(tree_height_))
                        && (pool_.active_flag(parent_g) != 0);
        }

        // ---- SIMD gather: load parent states for all 8 lanes ----
        const IVectorT parent_g_vec(parent_g_arr);
        const IVectorT pos_vec   = IVectorT::gather(
            robot_pos_flat_.data(), parent_g_vec);
        const IVectorT rocks_vec = IVectorT::gather(
            rock_bits_flat_.data(), parent_g_vec);
        const IVectorT action_vec(action_arr);

        // ---- SIMD transition ----
        IVectorT new_pos_vec, new_rocks_vec, active_vec;
        FVectorT reward_vec;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            stepBatch(pos_vec, rocks_vec, action_vec,
                      new_pos_vec, new_rocks_vec, active_vec, reward_vec);
            // total_step_batch_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++step_batch_call_count_;
        }

        simulation_count_ += BATCH;

        // ---- Rollout lower bound (SIMD) ----
        // Switch between always-east and ENT via the ROLLOUT_ENT macro.
        #ifdef ROLLOUT_ENT
        FVectorT rollout_f;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            rollout_f = rolloutENT(new_pos_vec, new_rocks_vec, active_vec);
            // total_rollout_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++rollout_call_count_;
        }
        #else
        FVectorT rollout_f;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            rollout_f = rolloutEast(new_pos_vec, active_vec);
            // total_rollout_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++rollout_call_count_;
        }
        #endif

        const IVectorT is_sense = (action_vec > static_cast<int32_t>(A_SAMPLE));
        rollout_f = FVectorT::select(is_sense.as<FVectorT>(), 
                                    rollout_f * FVectorT::fill(discount_factor_), 
                                    rollout_f);

        rollout_f.to_array_unaligned(
            rollout_values_.data() + static_cast<ptrdiff_t>(batch_start));

        // Sense actions have zero immediate reward in the tree; their value
        // comes solely from the post-search entropy-based bonus applied in
        // applyEntropyBonuses().  Adding a fixed bonus here would bias UCB
        // toward sense actions regardless of belief state, causing the agent
        // to keep sensing even when the belief is fully converged.
        const FVectorT& final_reward_vec = reward_vec;

        // ---- Extract SIMD results to plain arrays ----
        const auto new_pos_arr   = new_pos_vec.to_array();
        const auto new_rocks_arr = new_rocks_vec.to_array();
        const auto active_arr    = active_vec.to_array();
        const auto final_rew_arr = final_reward_vec.to_array();

        // ---- Scalar scatter-write to child global indices ----
        for (uint32_t i = 0; i < BATCH; ++i)
        {
            if (!valid[i]) continue;

            const int32_t cg = child_g_arr[i];
            if (cg < 0) continue;

            const auto sz_cg = static_cast<size_t>(cg);
            robot_pos_flat_[sz_cg] = new_pos_arr  [i];
            rock_bits_flat_[sz_cg] = new_rocks_arr[i];

            pool_.reward     (cg) = final_rew_arr[i];
            pool_.active_flag(cg) = active_arr   [i];
            pool_.curr_action(cg) = action_arr   [i];
        }
    }
}

#ifdef ENABLE_HOMOGENOUS_SEARCH
// ============================================================================
// expandNodesBatch (homogenous) — All scenarios at same level with same action
// ============================================================================

void RockSampleVecQMDP::expandNodesBatch(int32_t parent_level, int action_idx)
{
    constexpr uint32_t BATCH = static_cast<uint32_t>(IVectorT::num_scalars); // 8

    // Step 1: Allocate child blocks for all scenarios
    std::vector<int32_t> parent_globals(scenario_size_);
    std::vector<int32_t> child_globals(scenario_size_);

    for (uint32_t s = 0; s < scenario_size_; ++s)
    {
        const int32_t parent_g = homogenous_node_map_[parent_level][s];
        parent_globals[s] = parent_g;

        int32_t child_base = pool_.child_base(parent_g);
        if (child_base == vec_qmdp::planning::DYN_INVALID_IDX)
        {
            // Allocate child block + ensure domain array capacity
            child_base = allocateDomainChildBlock(parent_g);
        }
        child_globals[s] = child_base + action_idx;
    }

    // Step 2: Update homogenous_node_map_ for next level
    const int32_t child_level = parent_level + 1;
    if (homogenous_node_map_.size() <= static_cast<size_t>(child_level))
        homogenous_node_map_.resize(child_level + 1);
    homogenous_node_map_[child_level] = child_globals;

    // Step 3: SIMD batch processing
    std::vector<float> rollout_values(scenario_size_, 0.0f);

    for (uint32_t bs = 0; bs < scenario_size_; bs += BATCH)
    {
        const uint32_t valid_num = std::min(BATCH, scenario_size_ - bs);

        // Gather parent state (scalar loop - simpler than SIMD for cross-chunk)
        alignas(32) int32_t pos_arr[BATCH], rocks_arr[BATCH], active_arr[BATCH];
        for (uint32_t i = 0; i < valid_num; ++i)
        {
            const int32_t pg = parent_globals[bs + i];
            pos_arr[i]    = robot_pos_flat_[pg];
            rocks_arr[i]  = rock_bits_flat_[pg];
            active_arr[i] = pool_.active_flag(pg);
        }
        // Padding for unused lanes
        for (uint32_t i = valid_num; i < BATCH; ++i)
        {
            pos_arr[i] = pos_arr[valid_num - 1];
            rocks_arr[i] = rocks_arr[valid_num - 1];
            active_arr[i] = 0;  // Inactive padding
        }

        IVectorT pos_vec(pos_arr), rocks_vec(rocks_arr), active_vec(active_arr);
        IVectorT action_vec = IVectorT::fill(action_idx);

        // SIMD transition
        IVectorT new_pos_vec, new_rocks_vec;
        FVectorT reward_vec;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            stepBatch(pos_vec, rocks_vec, action_vec,
                      new_pos_vec, new_rocks_vec, active_vec, reward_vec);
            // total_step_batch_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++step_batch_call_count_;
        }

        simulation_count_ += BATCH;

        // Rollout
        FVectorT rollout_f;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            #ifdef ROLLOUT_ENT
                rollout_f = rolloutENT(new_pos_vec, new_rocks_vec, active_vec, bs == 0 && STEP == 4 && action_idx == 0 && parent_level == 0);
            #else
                rollout_f = rolloutEast(new_pos_vec, active_vec);
            #endif
            // total_rollout_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++rollout_call_count_;
        }

        // Apply sense-action discount if needed
        if (action_idx > static_cast<int>(A_SAMPLE))
            rollout_f = rollout_f * FVectorT::fill(static_cast<float>(discount_factor_));

        // Scatter child state (scalar loop)
        const auto new_pos_arr   = new_pos_vec.to_array();
        const auto new_rocks_arr = new_rocks_vec.to_array();
        const auto active_arr_out= active_vec.to_array();
        const auto reward_arr    = reward_vec.to_array();
        const auto rollout_arr   = rollout_f.to_array();

        for (uint32_t i = 0; i < valid_num; ++i)
        {
            const int32_t child_g = child_globals[bs + i];

            robot_pos_flat_[child_g] = new_pos_arr[i];
            rock_bits_flat_[child_g] = new_rocks_arr[i];
            pool_.reward(child_g)    = reward_arr[i];
            pool_.active_flag(child_g) = active_arr_out[i];
            pool_.curr_action(child_g) = action_idx;

            // Store rollout value (zero for terminal)
            const bool is_terminal = (active_arr[i] == 0);
            rollout_values[bs + i] = is_terminal ? 0.0f : rollout_arr[i];
        }
    }

    // Step 4: Unified backpropagation
    backPropagateHomogenous(child_level, rollout_values);

    // Step 5: Update homogenous mapping size if needed
    if (static_cast<size_t>(child_level + 1) > homogenous_node_map_.size())
    {
        homogenous_node_map_.resize(child_level + 1);
    }
}
#endif // ENABLE_HOMOGENOUS_SEARCH

// ============================================================================
// stepBatch — SIMD transition for one batch of IVectorT::num_scalars scenarios
// ============================================================================

void RockSampleVecQMDP::stepBatch(
    const IVectorT& pos_vec,
    const IVectorT& rocks_vec,
    const IVectorT& action_vec,
    IVectorT&       out_pos,
    IVectorT&       out_rocks,
    IVectorT&       out_active,
    FVectorT&       out_reward)
{
    const auto sz = static_cast<int32_t>(size_);

    // 1. Decode (x, y) from flat position index
    const IVectorT x_vec = pos_vec % sz;
    const IVectorT y_vec = pos_vec / sz;

    // 2. Movement: gather per-lane deltas, compute candidate new pos
    const IVectorT dx = IVectorT::gather(action_dx_.data(), action_vec);
    const IVectorT dy = IVectorT::gather(action_dy_.data(), action_vec);

    const IVectorT nx = x_vec + dx;
    const IVectorT ny = y_vec + dy;

    const IVectorT is_move    = action_vec < static_cast<int32_t>(A_SAMPLE);
    const IVectorT is_east    = (action_vec == static_cast<int32_t>(A_EAST));
    const IVectorT east_exit  = is_east & (nx >= sz);
    const IVectorT x_oob      = (nx < 0) | (nx >= sz);
    const IVectorT y_oob      = (ny < 0) | (ny >= sz);
    const IVectorT move_oob   = is_move & (x_oob | y_oob) & ~east_exit;
    const IVectorT valid_move = is_move & ~move_oob & ~east_exit;

    const IVectorT moved_pos  = ny * sz + nx;

    // 3. Sample action
    const IVectorT is_sample    = (action_vec == static_cast<int32_t>(A_SAMPLE));
    const IVectorT rock_at_vec  = IVectorT::gather(grid_.data(), pos_vec);
    const IVectorT rock_present = (rock_at_vec >= 0);

    // Variable-bit-shift — must be done with a scalar loop
    alignas(32) int32_t samp_set_arr[8], samp_clr_arr[8];
    {
        const auto ra = rock_at_vec.to_array();
        for (int s = 0; s < 8; ++s)
        {
            const int r     = ra[s];
            samp_set_arr[s] = (r >= 0) ? (1 << r) : 0;
            samp_clr_arr[s] = ~samp_set_arr[s];
        }
    }
    const IVectorT samp_set(samp_set_arr);
    const IVectorT samp_clr(samp_clr_arr);

    const IVectorT is_good_sample = (rocks_vec & samp_set) != 0;
    const IVectorT sampled_rocks  = rocks_vec & samp_clr;

    FVectorT samp_rew = FVectorT::select(
        is_good_sample.as<FVectorT>(),
        FVectorT::fill(+10.0f), FVectorT::fill(-10.0f));
    samp_rew = FVectorT::select(
        (~rock_present).as<FVectorT>(),
        FVectorT::fill(-100.0f), samp_rew);

    // 4. Sense action — no state change, no reward (QMDP semantics)
    const IVectorT is_sense = (action_vec > static_cast<int32_t>(A_SAMPLE));

    // 5. Combine outputs
    out_pos    = IVectorT::select(valid_move, moved_pos, pos_vec);
    out_rocks  = rocks_vec;
    out_rocks  = IVectorT::select(is_sample, sampled_rocks, out_rocks);
    out_active = IVectorT::fill(-1);
    out_active = IVectorT::select(east_exit, IVectorT::fill(0), out_active);

    out_reward = FVectorT::fill(0.0f);
    out_reward = FVectorT::select(is_sense.as<FVectorT>(), 
                                  FVectorT::fill(-0.1f), out_reward);
    out_reward = FVectorT::select(move_oob.as<FVectorT>(),
                                  FVectorT::fill(-100.0f), out_reward);
    out_reward = FVectorT::select(east_exit.as<FVectorT>(),
                                  FVectorT::fill(+10.0f),  out_reward);
    out_reward = FVectorT::select(is_sample.as<FVectorT>(), samp_rew, out_reward);
}

// ============================================================================
// Rollout policies  (SIMD batch, one lane = one scenario)
// ============================================================================

// ----------------------------------------------------------------------------
// rolloutEast — always-east lower bound
//
// Value = 10 * gamma^(size-1-x) for active lanes; 0 for terminated lanes.
// Matches the DESPOT RockSampleEastScenarioLowerBound per-scenario value.
// ----------------------------------------------------------------------------
auto RockSampleVecQMDP::rolloutEast(const IVectorT& new_pos_vec,
                                     const IVectorT& active_vec) const -> FVectorT
{
    const IVectorT new_x_vec =
        new_pos_vec % static_cast<int32_t>(size_);
    const IVectorT steps_vec =
        IVectorT::fill(static_cast<int32_t>(size_ - 1)) - new_x_vec;

    FVectorT rollout_f = FVectorT::gather(discount_pow_table_.data(), steps_vec);

    const IVectorT zero_mask = (active_vec == IVectorT::fill(0));
    return FVectorT::select(zero_mask.as<FVectorT>(), FVectorT::fill(0.0f), rollout_f);
}

// ============================================================================
// rolloutENT (optimized) — pre-filters candidate rocks into a small stack
// array to eliminate per-rock branching in the hot loop; uses unary negation
// for abs (one fewer SIMD op per axis vs. the fill(0)-subtract pattern);
// and shrinks both outer and inner loop bounds from num_rocks_ to n_cands.
// ============================================================================
auto RockSampleVecQMDP::rolloutENT(const IVectorT& new_pos_vec,
                                   const IVectorT& new_rocks_vec,
                                   const IVectorT& active_vec,
                                   bool print) const -> FVectorT
{
    // ---- Build candidate list (rocks with belief probability >= 0.1) ----
    // Stack-allocated to avoid heap traffic in the hot path.
    constexpr int MAX_CANDS = 30;
    int cand_rx [MAX_CANDS];   // rock x position
    int cand_ry [MAX_CANDS];   // rock y position
    int cand_bit[MAX_CANDS];   // (1 << rock_index) bitmask
    int n_cands = 0;
    for (int r = 0; r < num_rocks_; ++r) {
        if (current_rock_probs_[r] < 0.1f) continue;
        cand_rx [n_cands] = rock_pos_x_[r];
        cand_ry [n_cands] = rock_pos_y_[r];
        cand_bit[n_cands] = (1 << r);
        ++n_cands;
    }

    IVectorT cur_y_vec     = new_pos_vec / size_;
    IVectorT cur_x_vec     = new_pos_vec - (cur_y_vec * size_);
    IVectorT remaining_vec = new_rocks_vec;
    FVectorT cum_disc_vec  = FVectorT::fill(1.0f);
    FVectorT cum_val_vec   = FVectorT::fill(0.0f);

    const int32_t  NO_ROCK_DIST   = 2 * static_cast<int32_t>(size_);
    const IVectorT no_rock_dist_v  = IVectorT::fill(NO_ROCK_DIST);

    for (int iter = 0; iter < n_cands; ++iter) 
    {
        if (remaining_vec.test_zero()) break;

        IVectorT best_dist_vec = no_rock_dist_v;
        IVectorT best_x_vec    = cur_x_vec;
        IVectorT best_y_vec    = cur_y_vec;
        IVectorT best_bit_vec  = IVectorT::fill(0);

        for (int ci = 0; ci < n_cands; ++ci) 
        {
            const IVectorT rock_bit  = IVectorT::fill(cand_bit[ci]);
            const IVectorT is_good   = (remaining_vec & rock_bit) != 0;
            const IVectorT diff_x    = IVectorT::fill(cand_rx[ci]) - cur_x_vec;
            const IVectorT diff_y    = IVectorT::fill(cand_ry[ci]) - cur_y_vec;
            // abs via unary negation — one fewer SIMD op vs fill(0)-subtract
            const IVectorT dist      = (-diff_x).max(diff_x) + (-diff_y).max(diff_y);
            const IVectorT is_closer = is_good & (dist < best_dist_vec);

            best_dist_vec = IVectorT::select(is_closer, dist,                       best_dist_vec);
            best_x_vec    = IVectorT::select(is_closer, IVectorT::fill(cand_rx[ci]), best_x_vec);
            best_y_vec    = IVectorT::select(is_closer, IVectorT::fill(cand_ry[ci]), best_y_vec);
            best_bit_vec  = IVectorT::select(is_closer, rock_bit,                    best_bit_vec);
        }

        const IVectorT found_vec    = (best_dist_vec < no_rock_dist_v);
        const FVectorT step_disc_10 = FVectorT::gather(discount_pow_table_.data(), best_dist_vec);
        const FVectorT contrib      = cum_disc_vec * step_disc_10;

        cum_val_vec  = cum_val_vec +
            FVectorT::select(found_vec.as<FVectorT>(), contrib, 0.0f);

        const FVectorT new_cum_disc = cum_disc_vec * (step_disc_10 / 10.0f);
        cum_disc_vec = FVectorT::select(found_vec.as<FVectorT>(), new_cum_disc, cum_disc_vec);

        cur_x_vec     = IVectorT::select(found_vec, best_x_vec, cur_x_vec);
        cur_y_vec     = IVectorT::select(found_vec, best_y_vec, cur_y_vec);
        remaining_vec = IVectorT::select(found_vec,
                                          remaining_vec & (~best_bit_vec),
                                          remaining_vec);
    }

    IVectorT steps_to_exit =
        IVectorT::fill(static_cast<int32_t>(size_ - 1)) - cur_x_vec;
    cum_val_vec = cum_val_vec +
        cum_disc_vec * FVectorT::gather(discount_pow_table_.data(), steps_to_exit);

    IVectorT is_term = (active_vec == IVectorT::fill(0));
    FVectorT rollout_value = FVectorT::select(is_term.as<FVectorT>(), FVectorT::fill(0.0f), cum_val_vec);

    if (print)
    {
        std::cout << "x: " << cur_x_vec << " y: " << cur_y_vec << std::endl;
        std::cout << "rollout_value: " << rollout_value << std::endl;
    }

    return rollout_value;
}
// ============================================================================
// Core BeliefTreeSearch loop  (single-threaded; called by both serial path and workers)
// ============================================================================

void RockSampleVecQMDP::beliefTreeSearch(const std::vector<int>& state_ids,
                                       int max_iters)
{
    // ---- Setup: compute rock probs, reset structures, seed scenarios ----
    current_belief_particles_ = state_ids;

    std::fill(current_rock_probs_.begin(), current_rock_probs_.end(), 0.0f);
    for (int sid : state_ids)
    {
        int rocks = decodeRocks(sid);
        for (int i = 0; i < num_rocks_; ++i)
        {
            if ((rocks >> i) & 1) current_rock_probs_[i] += 1.0f;
        }
    }
    float inv_n = 1.0f / static_cast<float>(state_ids.size());
    for (int i = 0; i < num_rocks_; ++i)
        current_rock_probs_[i] *= inv_n;

    total_step_batch_us_ = 0.0;  step_batch_call_count_ = 0;
    total_rollout_us_    = 0.0;  rollout_call_count_    = 0;
    simulation_count_    = 0;

    resetDynStructures();
    ensureDomainCapacity(static_cast<int32_t>(scenario_size_));
    initDynRoots();
    sampleScenarios(state_ids);

    // ---- Core search loop ----
    const auto wall_start = std::chrono::steady_clock::now();

#ifdef ENABLE_HOMOGENOUS_SEARCH
    // Initialize homogenous mapping after roots are allocated
    initHomogenousMapping();
#endif

    int iter = 0;

    for (iter = 0; iter < max_iters; ++iter)
    {
        if (iter % 10 == 0)
        {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall_start).count();
            if (elapsed_ms >= max_planning_time_ms_) break;
        }

        #ifdef ENABLE_HOMOGENOUS_SEARCH
            // ---- Homogenous search path ----
            // 1. Selection — scenario-0 UCB traversal (all scenarios share path)
            int action_idx;
            const int32_t parent_level = exploreNodesHomogenous(action_idx);

            // 2. Expansion + Back-propagation — unified SIMD batch processing
            //    (backPropagateHomogenous is called inside expandNodesBatch)
            expandNodesBatch(parent_level, action_idx);
        #else
            // ---- Heterogeneous search path ----
            // 1. Selection — majority-vote depth-sync UCB traversal
            const std::vector<DynExploreResult> explore_results = exploreNodes();

            // 2. Expansion — SIMD-batched transition + child state initialisation
            expandNodesBatch(explore_results);

            // 3. Back-propagation — one call per scenario
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const int32_t parent_g  = explore_results[s].parent_global_idx;

                // If parent is terminal (active_flag == 0), backprop directly through it
                if (pool_.active_flag(parent_g) == 0) {
                    backPropagate(parent_g, rollout_values_[s]);
                    continue;
                }

                const int32_t child_b   = pool_.child_base(parent_g);
                if (child_b == vec_qmdp::planning::DYN_INVALID_IDX) continue;
                const int32_t child_g = child_b + static_cast<int32_t>(explore_results[s].action_idx);
                backPropagate(child_g, rollout_values_[s]);
            }
        #endif // ENABLE_HOMOGENOUS_SEARCH

        // // 4. Early-termination check every N iterations
        // if (iter > 0 && iter % early_term_check_interval_ == 0)
        // {
        //     updateConvergenceTracking();
        //     if (iter >= early_term_min_expand_calls_ &&
        //         checkEarlyTermination())
        //         break;
        // }
    }

    std::cout << "Search finished after " << iter << " iterations, "
              << simulation_count_ << " simulations.\n";
}

void RockSampleVecQMDP::beliefTreeSearch(int robot_pos,
                                       const std::vector<float>& rock_probs,
                                       int max_iters)
{
    // ---- Setup: reset structures, seed scenarios ----
    current_rock_probs_ = rock_probs;
    total_step_batch_us_ = 0.0;  step_batch_call_count_ = 0;
    total_rollout_us_    = 0.0;  rollout_call_count_    = 0;
    simulation_count_    = 0;

    resetDynStructures();
    ensureDomainCapacity(static_cast<int32_t>(scenario_size_));
    initDynRoots();
    sampleScenarios(robot_pos, rock_probs);

    // ---- Core search loop ----
    const auto wall_start = std::chrono::steady_clock::now();

    #ifdef ENABLE_HOMOGENOUS_SEARCH
        initHomogenousMapping();
    #endif

    int iter = 0;

    for (iter = 0; iter < max_iters; ++iter)
    {
        if (iter % 10 == 0)
        {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall_start).count();
            if (elapsed_ms >= max_planning_time_ms_) break;
        }

        #ifdef ENABLE_HOMOGENOUS_SEARCH
            int action_idx;
            const int32_t parent_level = exploreNodesHomogenous(action_idx);
            expandNodesBatch(parent_level, action_idx);
        #else
            const std::vector<DynExploreResult> explore_results = exploreNodes();
            expandNodesBatch(explore_results);
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                const int32_t parent_g = explore_results[s].parent_global_idx;
                if (pool_.active_flag(parent_g) == 0) {
                    backPropagate(parent_g, rollout_values_[s]);
                    continue;
                }
                const int32_t child_b = pool_.child_base(parent_g);
                if (child_b == vec_qmdp::planning::DYN_INVALID_IDX) continue;
                const int32_t child_g = child_b + static_cast<int32_t>(explore_results[s].action_idx);
                backPropagate(child_g, rollout_values_[s]);
            }
        #endif
    }

    std::cout << "Search finished after " << iter << " iterations, "
              << simulation_count_ << " simulations.\n";
}

// ============================================================================
// DynVecQMDP virtual hooks
// ============================================================================

std::vector<uint32_t>
RockSampleVecQMDP::computeNodeCandidateActionsDyn(int32_t global_idx)
{
    // QMDP semantics: sense actions are only useful at the first expansion
    // level (depth 0 parent → depth 1 children).
    const int depth = pool_.depth(global_idx);

    std::vector<uint32_t> actions;

    if (depth == 0)
    {
        // Single sense action at root — per-rock entropy bonuses are applied
        // post-search in applyEntropyBonuses(); best rock tracked in best_sense_rock_.
        actions.reserve(6u);
        actions.push_back(5u);
        for (uint32_t a = 0; a < 5u; ++a)
            actions.push_back(a);
    }
    else
    {
        // Deeper nodes: movement + sample only (no sense).
        actions.reserve(5u);
        for (uint32_t a = 0; a < 5u; ++a)
            actions.push_back(a);
    }

    return actions;
}

std::shared_ptr<vec_qmdp::planning::DynVecQMDP>
RockSampleVecQMDP::makeWorkerInstance() const
{
    return std::make_shared<RockSampleVecQMDP>(
        size_, num_rocks_, rock_pos_,
        start_x_, start_y_,
        half_efficiency_distance_,
        tree_height_,
        static_cast<int>(scenario_size_),
        1 /* single-threaded worker */);
}

// ============================================================================
// Search entry point
// ============================================================================

int RockSampleVecQMDP::parallelBeliefTreeSearch(const std::vector<int>& state_ids,
                                                int max_iters)
{
    // Compute rock probabilities from particle set (needed for entropy bonuses).
    current_belief_particles_ = state_ids;
    std::fill(current_rock_probs_.begin(), current_rock_probs_.end(), 0.0f);
    for (int sid : state_ids)
    {
        int rocks = decodeRocks(sid);
        for (int i = 0; i < num_rocks_; ++i)
        {
            if ((rocks >> i) & 1) current_rock_probs_[i] += 1.0f;
        }
    }
    const float inv_n = 1.0f / static_cast<float>(state_ids.size());
    for (int i = 0; i < num_rocks_; ++i)
        current_rock_probs_[i] *= inv_n;

    const std::vector<int> ids_copy        = state_ids;
    const auto             rock_probs_copy = current_rock_probs_;
    const int              iters           = max_iters;

    dispatchParallelSearch(
        [&ids_copy, &rock_probs_copy, iters](
            vec_qmdp::planning::DynVecQMDP* worker_base,
            size_t /*thread_idx*/) -> BeliefTreeSearchThreadResult
        {
            auto* w = static_cast<RockSampleVecQMDP*>(worker_base);
            w->current_rock_probs_ = rock_probs_copy;
            w->beliefTreeSearch(ids_copy, iters);

            // Collect first-level action Q-values from this worker's scenarios.
            BeliefTreeSearchThreadResult result;
            result.action_sum_values.assign(w->num_actions_, 0.0f);
            result.action_counts    .assign(w->num_actions_, 0);
            result.simulation_count = w->simulation_count_;

            for (uint32_t a = 0; a < w->num_actions_; ++a)
            {
                float sum_val   = 0.0f;
                int   valid_cnt = 0;
                for (uint32_t s = 0; s < w->scenario_size_; ++s)
                {
                    const int32_t vc = w->getRootVisitCount(s, a);
                    if (vc > 0)
                    {
                        sum_val += w->getRootActionQValue(s, a);
                        ++valid_cnt;
                    }
                }
                result.action_sum_values[a] = sum_val;
                result.action_counts[a]     = valid_cnt;
            }
            return result;
        });

    applyEntropyBonuses(state_ids);
    return getBestAction();
}

int RockSampleVecQMDP::parallelBeliefTreeSearch(int robot_pos,
                                                const std::vector<float>& rock_probs,
                                                int max_iters)
{
    current_rock_probs_ = rock_probs;

    const int  robot_pos_copy  = robot_pos;
    const auto rock_probs_copy = rock_probs;
    const int  iters           = max_iters;

    dispatchParallelSearch(
        [robot_pos_copy, &rock_probs_copy, iters](
            vec_qmdp::planning::DynVecQMDP* worker_base,
            size_t /*thread_idx*/) -> BeliefTreeSearchThreadResult
        {
            auto* w = static_cast<RockSampleVecQMDP*>(worker_base);
            w->current_rock_probs_ = rock_probs_copy;
            w->beliefTreeSearch(robot_pos_copy, rock_probs_copy, iters);

            // Collect first-level action Q-values from this worker's scenarios.
            BeliefTreeSearchThreadResult result;
            result.action_sum_values.assign(w->num_actions_, 0.0f);
            result.action_counts    .assign(w->num_actions_, 0);
            result.simulation_count = w->simulation_count_;

            for (uint32_t a = 0; a < w->num_actions_; ++a)
            {
                float sum_val   = 0.0f;
                int   valid_cnt = 0;
                for (uint32_t s = 0; s < w->scenario_size_; ++s)
                {
                    const int32_t vc = w->getRootVisitCount(s, a);
                    if (vc > 0)
                    {
                        sum_val += w->getRootActionQValue(s, a);
                        ++valid_cnt;
                    }
                }
                result.action_sum_values[a] = sum_val;
                result.action_counts[a]     = valid_cnt;
            }
            return result;
        });

    applyEntropyBonuses(robot_pos);
    return getBestAction();
}

int RockSampleVecQMDP::getBestAction() const
{
    const auto& vals = getAggregatedActionValues();
    int   best_a   = 0;
    float best_val = vals[0];
    for (uint32_t a = 1; a < num_actions_; ++a)
    {
        if (vals[a] > best_val)
        {
            best_val = vals[a];
            best_a   = static_cast<int>(a);
        }
    }
    if (best_a == 5) return 5 + best_sense_rock_;
    return best_a;
}

// ============================================================================
// Domain capacity management
// ============================================================================

void RockSampleVecQMDP::ensureDomainCapacity(int32_t min_size)
{
    if (min_size <= 0) return;
    const auto sz = static_cast<size_t>(min_size);
    if (robot_pos_flat_.size() < sz)
    {
        robot_pos_flat_.resize(sz, 0);
        rock_bits_flat_.resize(sz, 0);
    }
}

int32_t RockSampleVecQMDP::allocateDomainChildBlock(int32_t parent_g)
{
    const int32_t child_base = allocateChildBlock(parent_g);
    ensureDomainCapacity(child_base + static_cast<int32_t>(num_actions_));
    return child_base;
}

// ============================================================================
// Domain step function  (single-scenario, scalar; not on hot path)
// ============================================================================

float RockSampleVecQMDP::applyAction(int           pos,
                                      int           rock_bits,
                                      int           action,
                                      std::mt19937& rng,
                                      int&          out_pos,
                                      int&          out_rocks,
                                      bool&         out_terminal) const
{
    out_pos      = pos;
    out_rocks    = rock_bits;
    out_terminal = false;

    const int x = pos % size_;
    const int y = pos / size_;
    float reward = 0.0f;

    switch (action)
    {
    case A_NORTH:
        if (y + 1 < size_)  out_pos = (y + 1) * size_ + x;
        else                reward = -100.0f;
        break;

    case A_EAST:
        if (x + 1 < size_) {
            out_pos = y * size_ + (x + 1);
        } else {
            reward       = +10.0f;
            out_terminal = true;
        }
        break;

    case A_SOUTH:
        if (y - 1 >= 0)  out_pos = (y - 1) * size_ + x;
        else             reward = -100.0f;
        break;

    case A_WEST:
        if (x - 1 >= 0)  out_pos = y * size_ + (x - 1);
        else             reward = -100.0f;
        break;

    case A_SAMPLE:
    {
        const int rock_at = grid_[pos];
        if (rock_at >= 0) {
            const bool is_good = (rock_bits >> rock_at) & 1;
            reward    = is_good ? +10.0f : -10.0f;
            out_rocks = rock_bits & ~(1 << rock_at);
        } else {
            reward = -100.0f;
        }
        break;
    }

    default:
    {
        // Sense action — no state change, no reward
        (void)rng;
        break;
    }
    }

    return reward;
}

// ============================================================================
// Entropy information-gain bonus computation  (unchanged from static version)
// ============================================================================

float RockSampleVecQMDP::computeEntropyBonus(
    int rock_idx, const std::vector<int>& state_ids) const
{
    if (state_ids.empty()) return 0.0f;

    auto shannon = [](double p) -> double
    {
        if (p <= 0.0 || p >= 1.0) return 0.0;
        return -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
    };

    const int n = static_cast<int>(state_ids.size());

    constexpr int lane_w = static_cast<int>(IVectorT::num_scalars);

    const int32_t rock_mask_val = static_cast<int32_t>((1 << num_rocks_) - 1);
    const int32_t rock_bit_val  = static_cast<int32_t>(1 << rock_idx);
    const int32_t sz            = static_cast<int32_t>(size_);
    const auto*   sid_ptr       = reinterpret_cast<const int32_t*>(state_ids.data());

    IVectorT sum_good_v = IVectorT::fill(0);
    IVectorT sum_x_v    = IVectorT::fill(0);
    IVectorT sum_y_v    = IVectorT::fill(0);

    const int n_simd = (n / lane_w) * lane_w;

    for (int i = 0; i < n_simd; i += lane_w)
    {
        const IVectorT sid    = IVectorT::load_contiguous_unaligned(
                                    sid_ptr, static_cast<int32_t>(i));
        const IVectorT rock_v = sid & IVectorT::fill(rock_mask_val);
        const IVectorT pos_v  = sid >> static_cast<unsigned int>(num_rocks_);
        const IVectorT x_v    = pos_v % sz;
        const IVectorT y_v    = pos_v / sz;

        const IVectorT good_v = -((rock_v & IVectorT::fill(rock_bit_val))
                                  != IVectorT::fill(0));

        sum_good_v = sum_good_v + good_v;
        sum_x_v    = sum_x_v    + x_v;
        sum_y_v    = sum_y_v    + y_v;
    }

    int total_good = sum_good_v.hsum();
    int total_x    = sum_x_v.hsum();
    int total_y    = sum_y_v.hsum();

    for (int i = n_simd; i < n; ++i)
    {
        const int rock_bits_s = decodeRocks(state_ids[i]);
        if ((rock_bits_s >> rock_idx) & 1) ++total_good;
        const int pos_s = decodePos(state_ids[i]);
        total_x += pos_s % size_;
        total_y += pos_s / size_;
    }

    const double dn    = static_cast<double>(n);
    double p_good = static_cast<double>(total_good) / dn;
    double avg_x  = static_cast<double>(total_x)    / dn;
    double avg_y  = static_cast<double>(total_y)    / dn;

    if (p_good < 0.1 || p_good > 0.9) return 0.0f;

    const double h_current = shannon(p_good);
    if (h_current < 1e-9) return 0.0f;

    const double ddx  = avg_x - static_cast<double>(rock_pos_[rock_idx].x);
    const double ddy  = avg_y - static_cast<double>(rock_pos_[rock_idx].y);
    const double dist = std::sqrt(ddx * ddx + ddy * ddy);
    const double eta  = 0.5 * (1.0 + std::pow(2.0, -dist / half_efficiency_distance_));

    const double p_obs_good = eta * p_good + (1.0 - eta) * (1.0 - p_good);
    const double p_obs_bad  = 1.0 - p_obs_good;

    if (p_obs_good < 1e-9 || p_obs_bad < 1e-9) return 0.0f;

    const double p_g_given_og = (eta         * p_good) / p_obs_good;
    const double p_g_given_ob = ((1.0 - eta) * p_good) / p_obs_bad;

    const double h_after =
        p_obs_good * shannon(p_g_given_og) +
        p_obs_bad  * shannon(p_g_given_ob);

    const double ig = h_current - h_after;
    return static_cast<float>(entropy_lambda_ * ig);
}

float RockSampleVecQMDP::computeEntropyBonus(int rock_idx, int robot_pos) const
{
    auto shannon = [](double p) -> double
    {
        if (p <= 0.0 || p >= 1.0) return 0.0;
        return -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
    };

    const double p_good = static_cast<double>(current_rock_probs_[rock_idx]);

    if (p_good < 0.1 || p_good > 0.9) return 0.0f;

    const double h_current = shannon(p_good);
    if (h_current < 1e-9) return 0.0f;

    // Use the given robot position
    const double robot_x = static_cast<double>(robot_pos % size_);
    const double robot_y = static_cast<double>(robot_pos / size_);

    const double ddx  = robot_x - static_cast<double>(rock_pos_[rock_idx].x);
    const double ddy  = robot_y - static_cast<double>(rock_pos_[rock_idx].y);
    const double dist = std::sqrt(ddx * ddx + ddy * ddy);
    const double eta  = 0.5 * (1.0 + std::pow(2.0, -dist / half_efficiency_distance_));

    const double p_obs_good = eta * p_good + (1.0 - eta) * (1.0 - p_good);
    const double p_obs_bad  = 1.0 - p_obs_good;

    if (p_obs_good < 1e-9 || p_obs_bad < 1e-9) return 0.0f;

    const double p_g_given_og = (eta         * p_good) / p_obs_good;
    const double p_g_given_ob = ((1.0 - eta) * p_good) / p_obs_bad;

    const double h_after =
        p_obs_good * shannon(p_g_given_og) +
        p_obs_bad  * shannon(p_g_given_ob);

    const double ig = h_current - h_after;
    return static_cast<float>(entropy_lambda_ * ig);
}

void RockSampleVecQMDP::applyEntropyBonuses(const std::vector<int>& state_ids)
{
    if (aggregated_action_counts_[5] == 0) return;
    float best_bonus = -1e9f;
    int   best_rock  = 0;
    for (int r = 0; r < num_rocks_; ++r) {
        const float bonus = computeEntropyBonus(r, state_ids);
        if (bonus > best_bonus) { best_bonus = bonus; best_rock = r; }
    }
    aggregated_action_values_[5] += best_bonus;
    best_sense_rock_ = best_rock;
}

void RockSampleVecQMDP::applyEntropyBonuses(int robot_pos)
{
    if (aggregated_action_counts_[5] == 0) return;
    float best_bonus = -1e9f;
    int   best_rock  = 0;
    for (int r = 0; r < num_rocks_; ++r) {
        const float bonus = computeEntropyBonus(r, robot_pos);
        std::cout << "[DEBUG] Rock " << r << ": p_good=" << current_rock_probs_[r]
                  << ", bonus=" << bonus << "\n";
        if (bonus > best_bonus) { best_bonus = bonus; best_rock = r; }
    }
    // Debug output
    std::cout << "[DEBUG] Best entropy bonus: " << best_bonus
              << " for rock " << best_rock
              << " (prob=" << current_rock_probs_[best_rock] << ")\n";
    aggregated_action_values_[5] += best_bonus;
    best_sense_rock_ = best_rock;
}

} // namespace rock_sample_vecqmdp
