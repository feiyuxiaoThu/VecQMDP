/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file rock_sample_static_vecqmdp.cpp
 * @brief Static (pre-allocated tree) VecQMDP solver for Rock Sample.
 *
 * Design notes
 * ------------
 *  scenario_size_ = IVectorT::num_scalars (= 8 for AVX2)
 *
 *  Domain state layout (same shape as VecQMDP base arrays):
 *    robot_pos_static_[getNodeIdx(rel, s)] = pos
 *    rock_bits_static_[getNodeIdx(rel, s)] = rock_bits
 *  where rel ∈ [0, tree_node_size_) and s ∈ [0, scenario_size_).
 *
 *  Child formula (implicit heap):
 *    child_rel(parent_rel, action) = parent_rel * num_actions_ + action + 1
 *
 *  backPropagate / backupDepth in the static VecQMDP base class take the
 *  RELATIVE tree position (not the global index).
 *
 *  Terminal handling:
 *    When active_flag[parent_g] == 0, expandNodesBatch skips that lane. beliefTreeSearch then backprops
 *    through the terminal parent itself (which already carries node_rewards_
 *    = +10 from when it was first created).
 */
#include "rock_sample_static_vecqmdp.hpp"

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

int STEP = 0;
bool ENABLE_DEBUG = false;

namespace rock_sample_vecqmdp {

// ============================================================================
// Helper — compute tree_node_size and check memory before constructing base
// ============================================================================

namespace {

/// Return tree_node_size = sum_{k=0}^{h} n^k, or throw if memory > limit_mb.
uint64_t computeAndCheckTreeNodeSize(uint32_t tree_height, int num_actions,
                                     uint64_t limit_mb = 1024)
{
    uint64_t total = 0, power = 1;
    for (uint32_t k = 0; k <= tree_height; ++k) {
        total += power;
        if (total > 1000000000ULL) {  // > 1 billion nodes
            const uint64_t est = total * 8 * 12 * 4 / (1024 * 1024);
            throw std::runtime_error(
                "[RockSampleStaticVecQMDP] tree_height=" +
                std::to_string(tree_height) + " with " +
                std::to_string(num_actions) + " actions requires >" +
                std::to_string(est) + " MB per worker. "
                "Use tree_height ≤ 4 for the 13-action rock sample.");
        }
        power *= static_cast<uint64_t>(num_actions);
    }
    // Rough estimate: 12 int/float arrays * 4 bytes * (tree_size * 8 scenarios)
    const uint64_t est_mb = total * 8 * 12 * 4 / (1024 * 1024);
    if (est_mb > limit_mb) {
        std::cerr << "[RockSampleStaticVecQMDP] WARNING: tree_height=" << tree_height
                  << " with num_actions=" << num_actions
                  << " requires ~" << est_mb << " MB per worker "
                  << "(limit=" << limit_mb << " MB). "
                  << "Consider using tree_height ≤ 4.\n";
    }
    return total;
}

}  // anonymous namespace

// ============================================================================
// Construction
// ============================================================================

RockSampleStaticVecQMDP::RockSampleStaticVecQMDP(
    int                               size,
    int                               num_rocks,
    const std::vector<despot::Coord>& rock_pos,
    int                               start_x,
    int                               start_y,
    double                            half_efficiency_distance,
    uint32_t                          tree_height,
    int                               /*num_scenarios — fixed to IVectorT::num_scalars*/,
    int                               num_threads)
    : VecQMDP(
        // Check memory (throws on > ~1 GB per worker) BEFORE calling base ctor.
        [&]() -> int {
            computeAndCheckTreeNodeSize(tree_height, 6);
            return static_cast<int>(IVectorT::num_scalars);
        }(),
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

    // Flat grid: grid_[y*size+x] = rock index, -1 if empty
    grid_.assign(static_cast<size_t>(size_ * size_), -1);
    for (int i = 0; i < num_rocks_; ++i)
        grid_[rock_pos_[i].y * size_ + rock_pos_[i].x] = i;

    // Domain state arrays — same size as VecQMDP's node_count_ etc.
    robot_pos_static_.resize(node_size_, 0);
    rock_bits_static_.resize(node_size_, 0);

    // Action-delta lookup (size=6: 4 moves + sample + sense)
    action_dx_.assign(6u, 0);
    action_dy_.assign(6u, 0);
    action_dy_[A_NORTH] = +1;
    action_dx_[A_EAST]  = +1;
    action_dy_[A_SOUTH] = -1;
    action_dx_[A_WEST]  = -1;

    // Separate x/y rock-position arrays for SIMD gather
    rock_pos_x_.resize(static_cast<size_t>(num_rocks_));
    rock_pos_y_.resize(static_cast<size_t>(num_rocks_));
    for (int i = 0; i < num_rocks_; ++i) {
        rock_pos_x_[i] = rock_pos_[i].x;
        rock_pos_y_[i] = rock_pos_[i].y;
    }

    // Per-scenario RNGs
    rngs_.reserve(scenario_size_);
    for (uint32_t s = 0; s < scenario_size_; ++s)
        rngs_.emplace_back(static_cast<uint32_t>(
            vec_qmdp::utils::RANDOM_SEED) + s * 1009u + 999983u);

    // Rollout value storage (one entry per scenario)
    current_rock_probs_.assign(static_cast<size_t>(num_rocks_), 0.0f);

    // Hyperparameter defaults (match DynVecQMDP version)
    setDepthSyncLambda(0.0f);
    setDiscountFactor(0.95);

    // Precompute 10 * gamma^k for k = 0 .. 2*size_
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

    // Build UCB lookup tables now that exploration_constant_ is set.
    initUCBTables();

    initParallelInfrastructure(static_cast<size_t>(num_threads));
}

// ============================================================================
// Belief initialisation
// ============================================================================

void RockSampleStaticVecQMDP::sampleScenarios(int robot_pos,
                                         const std::vector<float>& rock_probs)
{
    // Sample rock states from independent Bernoulli distributions
    for (uint32_t s = 0; s < scenario_size_; ++s)
    {
        const uint32_t root_g = getNodeIdx(0u, s);
        robot_pos_static_[root_g] = robot_pos;

        // Sample each rock independently
        int rock_bits = 0;
        for (int r = 0; r < num_rocks_; ++r) {
            std::uniform_real_distribution<float> ud(0.0f, 1.0f);
            if (ud(rngs_[s]) < rock_probs[r]) {
                rock_bits |= (1 << r);
            }
        }
        rock_bits_static_[root_g] = rock_bits;
    
        if (s == 0) // Debug output for the first scenario
            std::cout << "[DEBUG] Scenario " << s << ": robot_pos=" << robot_pos
                    << ", rock_bits=" << std::bitset<30>(rock_bits) << "\n";
    }
}

// ============================================================================
// expandNodesBatch — SIMD expansion for the static pre-allocated tree
// ============================================================================

void RockSampleStaticVecQMDP::expandNodesBatch(
    const AlignedVectorInt& node_idxs,
    const std::vector<int>& action_idxs)
{
    constexpr uint32_t BATCH = static_cast<uint32_t>(IVectorT::num_scalars); // 8

    for (uint32_t batch_start = 0; batch_start < scenario_size_; batch_start += BATCH)
    {
        const uint32_t valid_num = std::min(BATCH, scenario_size_ - batch_start);

        IVectorT node_idxs_vec = IVectorT::load_contiguous(node_idxs.data(), batch_start);

        // Build per-batch action vector from the std::vector, padding unused lanes.
        alignas(32) int32_t action_buf[BATCH];
        for (uint32_t i = 0; i < BATCH; ++i)
        {
            const uint32_t s = (i < valid_num) ? (batch_start + i) : (batch_start + valid_num - 1);
            action_buf[i] = action_idxs[s];
        }
        const IVectorT action_vec(action_buf);

        // ---- SIMD gather: load parent domain state ----
        IVectorT pos_vec      = IVectorT::gather(robot_pos_static_.data(), node_idxs_vec);
        IVectorT rocks_vec    = IVectorT::gather(rock_bits_static_.data(), node_idxs_vec);
        IVectorT active_flags = IVectorT::gather(node_active_flags_.data(), node_idxs_vec);

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

        simulation_count_static_ += static_cast<int>(valid_num);

        // ---- Rollout lower bound ----
        FVectorT rollout_f;
        {
            // const auto t0 = std::chrono::steady_clock::now();
            #ifdef ROLLOUT_ENT_STATIC
                rollout_f = rolloutENT(new_pos_vec, new_rocks_vec, active_vec);
            #else
                rollout_f = rolloutEast(new_pos_vec, active_vec);
            #endif
            // total_rollout_us_ += std::chrono::duration<double, std::micro>(
            //     std::chrono::steady_clock::now() - t0).count();
            // ++rollout_call_count_;
        }

        // Sense actions: discount the rollout by one step.
        const IVectorT is_sense = (action_vec > static_cast<int32_t>(A_SAMPLE));
        rollout_f = FVectorT::select(is_sense.as<FVectorT>(),
                                     rollout_f * static_cast<float>(discount_factor_),
                                     rollout_f);

        // ---- Extract SIMD results ----
        IVectorT scenario_idxs     = node_idxs_vec / static_cast<int32_t>(tree_node_size_);
        IVectorT relative_idxs     = node_idxs_vec -
                                     (scenario_idxs * static_cast<int32_t>(tree_node_size_));
        IVectorT child_idxs        = relative_idxs * static_cast<int32_t>(num_actions_)
                                        + 1 + action_vec;
        IVectorT global_child_idxs = child_idxs +
                                     scenario_idxs * static_cast<int32_t>(tree_node_size_);

        // 3. Back-propagation
        for (uint32_t i = 0; i < valid_num; ++i)
        {
            const auto pair_idx = std::make_pair(0, i);
            const uint32_t scenario_idx = static_cast<uint32_t>(scenario_idxs[pair_idx]);

            if (active_flags[pair_idx] == 0u)
            {
                // Terminal parent: backprop through the parent itself.
                backPropagate(relative_idxs[pair_idx], scenario_idx, reward_vec[pair_idx]);
                continue;
            }
            const uint32_t child_rel = child_idxs[pair_idx];

            if (child_rel >= tree_node_size_)
                continue;  // at max depth, no write needed

            const uint32_t child_g = global_child_idxs[pair_idx];

            robot_pos_static_[child_g]      = new_pos_vec  [pair_idx];
            rock_bits_static_[child_g]      = new_rocks_vec[pair_idx];
            node_active_flags_[child_g]     = static_cast<int>(active_vec[pair_idx]);
            node_rewards_[child_g]          = reward_vec   [pair_idx];
            node_curr_action_idxs_[child_g] = action_vec   [pair_idx];

            backPropagate(child_rel, scenario_idx, rollout_f[pair_idx]);
        }
    }
}

#ifdef ENABLE_HOMOGENOUS_SEARCH
// ============================================================================
// expandNodesBatch (homogenous) — single relative parent + single action
//
// All scenarios share the same relative parent node (parent_rel) and the same
// action (action_idx).  The global parent for scenario s is:
//   parent_rel + s * tree_node_size_
//
// All scenarios (including terminal-parent lanes) are written via a single SIMD
// scatter, then backPropagateHomogenous is called once for all scenarios.
// Terminal-parent lanes receive rollout_value = 0; backPropagateHomogenous picks
// up their reward via node_rewards_[child_g] on the first traversal step, which
// is semantically equivalent to the old backPropagate(parent_rel, ..., reward).
// ============================================================================
void RockSampleStaticVecQMDP::expandNodesBatch(uint32_t parent_rel, int action_idx)
{
    constexpr uint32_t BATCH = static_cast<uint32_t>(IVectorT::num_scalars); // 8

    const uint32_t child_rel = parent_rel * num_actions_ + 1 +
                               static_cast<uint32_t>(action_idx);

    // Per-scenario rollout values for the single backPropagateHomogenous call.
    AlignedVectorFloat rollout_values(scenario_size_, 0.0f);

    const uint32_t full_batches = scenario_size_ / BATCH;
    const uint32_t simd_size    = full_batches * BATCH;

    // ---- Full SIMD batches ----
    for (uint32_t batch_start = 0; batch_start < simd_size; batch_start += BATCH)
    {
        IVectorT scen_off_v    = IVectorT::load_contiguous(scen_offsets_.data(), static_cast<int>(batch_start));
        IVectorT node_idxs_vec = scen_off_v + static_cast<int32_t>(parent_rel);

        // SIMD gather: load parent domain state
        IVectorT pos_vec    = IVectorT::gather(robot_pos_static_.data(), node_idxs_vec);
        IVectorT rocks_vec  = IVectorT::gather(rock_bits_static_.data(), node_idxs_vec);
        IVectorT active_vec = IVectorT::gather(node_active_flags_.data(), node_idxs_vec);

        // SIMD transition (scalar action — dispatches to per-category path)
        IVectorT new_pos_vec, new_rocks_vec;
        FVectorT reward_vec;
        {
            #ifdef PRINT_TIME
                const auto t0 = std::chrono::steady_clock::now();
            #endif
            stepBatch(pos_vec, rocks_vec, action_idx,
                      new_pos_vec, new_rocks_vec, active_vec, reward_vec);
            #ifdef PRINT_TIME
                total_step_batch_us_ += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();
                ++step_batch_call_count_;
            #endif
        }

        simulation_count_static_ += BATCH;

        // Rollout lower bound
        FVectorT rollout_f;
        {
            #ifdef PRINT_TIME
                const auto t0 = std::chrono::steady_clock::now();
            #endif

            #ifdef ROLLOUT_ENT_STATIC
                // #ifdef ENABLE_HOMOGENOUS_SEARCH
                //     // All lanes share the same robot position under homogenous search.
                //     rollout_f = rolloutENT_Homogenous(new_pos_vec, new_rocks_vec, active_vec);
                // #else
                    rollout_f = rolloutENT(new_pos_vec, new_rocks_vec, active_vec, batch_start == 0 && STEP == 4 && action_idx == 0 && parent_rel == 0);
                // #endif
            #else
                rollout_f = rolloutEast(new_pos_vec, active_vec);
            #endif

            #ifdef PRINT_TIME   
                total_rollout_us_ += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();
                ++rollout_call_count_;
            #endif
        }

        // Sense-action rollout discount (scalar branch — action is uniform).
        if (action_idx > static_cast<int>(A_SAMPLE))
            rollout_f = rollout_f * static_cast<float>(discount_factor_);

        // SIMD scatter: write child state for all scenarios.
        // Terminal-parent lanes also write their post-step state so that
        // backPropagateHomogenous can pick up reward_vec at the child level.
        if (child_rel < tree_node_size_)
        {
            const IVectorT child_g_vec = scen_off_v + static_cast<int32_t>(child_rel);
            new_pos_vec  .scatter(robot_pos_static_.data(),          child_g_vec);
            new_rocks_vec.scatter(rock_bits_static_.data(),          child_g_vec);
            active_vec   .scatter(node_active_flags_.data(),         child_g_vec);
            reward_vec   .scatter(node_rewards_.data(),              child_g_vec);
            IVectorT::fill(action_idx).scatter(node_curr_action_idxs_.data(), child_g_vec);
        }

        // Rollout values: rollout_f for active-parent lanes, 0 for terminal-parent.
        // Terminal-parent lanes start backprop from child with rollout=0; their
        // reward (stored in node_rewards_[child_g]) is added by
        // backPropagateHomogenous on the first traversal step.
        FVectorT::select(active_vec.template as<FVectorT>(), rollout_f, 0.0f)
                 .to_array(rollout_values.data() + batch_start);
    }

    // Single unified backpropagation for all scenarios (normal + terminal-parent).
    if (child_rel < tree_node_size_)
        backPropagateHomogenous(child_rel, rollout_values);
}
#endif // ENABLE_HOMOGENOUS_SEARCH

// ============================================================================
// stepBatch — identical physics to RockSampleVecQMDP::stepBatch
// ============================================================================

void RockSampleStaticVecQMDP::stepBatch(
    const IVectorT& pos_vec,
    const IVectorT& rocks_vec,
    const IVectorT& action_vec,
    IVectorT&       out_pos,
    IVectorT&       out_rocks,
    IVectorT&       out_active,
    FVectorT&       out_reward)
{
    // bool print = (STEP == 2) && ENABLE_DEBUG;

    // 1. Decode (x, y) from flat position index
    const IVectorT y_vec = pos_vec / size_;
    const IVectorT x_vec = pos_vec - (y_vec * size_);

    // 2. Movement: gather per-lane deltas, compute candidate new pos
    const IVectorT dx = IVectorT::gather(action_dx_.data(), action_vec);
    const IVectorT dy = IVectorT::gather(action_dy_.data(), action_vec);

    const IVectorT nx = x_vec + dx;
    const IVectorT ny = y_vec + dy;

    // std::cout << "new pos x: " << nx << " new pos y: " << ny << std::endl;

    const IVectorT is_move    = action_vec < static_cast<int32_t>(A_SAMPLE);
    const IVectorT is_east    = (action_vec == static_cast<int32_t>(A_EAST));
    const IVectorT east_exit  = is_east & (nx >= size_);
    const IVectorT x_oob      = (nx < 0) | (nx >= size_);
    const IVectorT y_oob      = (ny < 0) | (ny >= size_);
    const IVectorT move_oob   = is_move & (x_oob | y_oob) & ~east_exit;
    const IVectorT valid_move = is_move & ~move_oob & ~east_exit;

    const IVectorT moved_pos  = ny * size_ + nx;

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
    out_active = out_active & (~east_exit);

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
// stepBatch (homogenous) — scalar action_idx, all lanes execute the same action
//
// Because action_idx is uniform across lanes, we branch once on action category
// and compute only the SIMD work that is actually needed for that category:
//
//  sense  (action > A_SAMPLE) : no state change at all; trivial assignment.
//  sample (action == A_SAMPLE): only rock gather + bit-clear logic.
//  east   (action == A_EAST)  : movement + east-exit terminal check.
//  other  (N/S/W)             : movement + OOB penalty; no east-exit path.
// ============================================================================
#ifdef ENABLE_HOMOGENOUS_SEARCH
void RockSampleStaticVecQMDP::stepBatch(
    const IVectorT& pos_vec,
    const IVectorT& rocks_vec,
    int             action_idx,
    IVectorT&       out_pos,
    IVectorT&       out_rocks,
    IVectorT&       out_active,
    FVectorT&       out_reward)
{
    // ------------------------------------------------------------------
    // 1. Sense — no state change, uniform -0.1 reward
    // ------------------------------------------------------------------
    if (action_idx > static_cast<int>(A_SAMPLE))
    {
        out_pos    = pos_vec;
        out_rocks  = rocks_vec;
        out_reward = -0.1f;
        return;
    }

    // ------------------------------------------------------------------
    // 2. Sample — no movement; clear the rock bit at the current cell
    // ------------------------------------------------------------------
    if (action_idx == static_cast<int>(A_SAMPLE))
    {
        const IVectorT rock_at_vec  = IVectorT::gather(grid_.data(), pos_vec);
        const IVectorT rock_present = (rock_at_vec >= 0);

        // Variable-bit-shift (per-lane rock index) — scalar loop required
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
        FVectorT samp_rew = FVectorT::select(
            is_good_sample.as<FVectorT>(),
            FVectorT::fill(+10.0f), FVectorT::fill(-10.0f));
        samp_rew = FVectorT::select(
            (~rock_present).as<FVectorT>(),
            FVectorT::fill(-100.0f), samp_rew);

        out_pos    = pos_vec;
        out_rocks  = rocks_vec & samp_clr;
        out_reward = samp_rew;
        return;
    }

    // ------------------------------------------------------------------
    // 3. Movement (action_idx < A_SAMPLE)
    //    dx / dy are scalar constants — no gather needed.
    // ------------------------------------------------------------------
    const IVectorT y_vec = pos_vec / size_;
    const IVectorT x_vec = pos_vec - (y_vec * size_);

    const int32_t dx = static_cast<int32_t>(action_dx_[action_idx]);
    const int32_t dy = static_cast<int32_t>(action_dy_[action_idx]);

    const IVectorT nx = x_vec + dx;
    const IVectorT ny = y_vec + dy;

    out_rocks = rocks_vec;  // movement never changes rock state

    if (action_idx == static_cast<int>(A_EAST))
    {
        // EAST: only outcome is east-exit (terminal, +10) or valid move (0).
        // OOB on east boundary IS the exit condition, not a penalty.
        const IVectorT east_exit = (nx >= static_cast<int32_t>(size_));

        out_pos    = IVectorT::select(~east_exit, ny * size_ + nx, pos_vec);
        out_active = out_active & (~east_exit);
        out_reward = FVectorT::select(east_exit.as<FVectorT>(),
                                      FVectorT::fill(+10.0f), FVectorT::fill(0.0f));
    }
    else
    {
        // NORTH / SOUTH / WEST: cannot exit; OOB → -100 penalty.
        const IVectorT x_oob    = (nx < 0) | (nx >= static_cast<int32_t>(size_));
        const IVectorT y_oob    = (ny < 0) | (ny >= static_cast<int32_t>(size_));
        const IVectorT move_oob = x_oob | y_oob;

        out_pos    = IVectorT::select(~move_oob, ny * size_ + nx, pos_vec);
        out_reward = FVectorT::select(move_oob.as<FVectorT>(), -100.0f, 0.0f);
    }
}
#endif // ENABLE_HOMOGENOUS_SEARCH

// ============================================================================
// Rollout policies (identical to DynVecQMDP version)
// ============================================================================

auto RockSampleStaticVecQMDP::rolloutEast(
    const IVectorT& new_pos_vec,
    const IVectorT& active_vec) const -> FVectorT
{
    const IVectorT new_x   = new_pos_vec - (new_pos_vec / size_) * size_;
    const IVectorT steps   = IVectorT::fill(static_cast<int32_t>(size_ - 1)) - new_x;
    FVectorT val           = FVectorT::gather(discount_pow_table_.data(), steps);
    const IVectorT is_term = (active_vec == IVectorT::fill(0));
    return FVectorT::select(is_term.as<FVectorT>(), FVectorT::fill(0.0f), val);
}

// ============================================================================
// rolloutENT (optimized) — pre-filters candidate rocks into a small stack
// array to eliminate per-rock branching in the hot loop; uses unary negation
// for abs (one fewer SIMD op per axis vs. the fill(0)-subtract pattern);
// and shrinks both outer and inner loop bounds from num_rocks_ to n_cands.
// ============================================================================
auto RockSampleStaticVecQMDP::rolloutENT(
    const IVectorT& new_pos_vec,
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

    for (int iter = 0; iter < n_cands; ++iter) {
        if (remaining_vec.test_zero()) break;

        IVectorT best_dist_vec = no_rock_dist_v;
        IVectorT best_x_vec    = cur_x_vec;
        IVectorT best_y_vec    = cur_y_vec;
        IVectorT best_bit_vec  = IVectorT::fill(0);

        for (int ci = 0; ci < n_cands; ++ci) {
            const IVectorT rock_bit  = IVectorT::fill(cand_bit[ci]);
            const IVectorT is_good   = (remaining_vec & rock_bit) != 0;
            const IVectorT diff_x    = IVectorT::fill(cand_rx[ci]) - cur_x_vec;
            const IVectorT diff_y    = IVectorT::fill(cand_ry[ci]) - cur_y_vec;
            // abs via unary negation — one fewer SIMD op vs fill(0)-subtract
            const IVectorT dist      = (-diff_x).max(diff_x) + (-diff_y).max(diff_y);
            const IVectorT is_closer = is_good & (dist < best_dist_vec);

            best_dist_vec = IVectorT::select(is_closer, dist,                        best_dist_vec);
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

    return rollout_value;
}

#ifdef ENABLE_HOMOGENOUS_SEARCH
// ============================================================================
// rolloutENT_Homogenous — specialised ENT rollout for the homogenous search.
//
// Under ENABLE_HOMOGENOUS_SEARCH all scenarios follow the same action path,
// so all lanes start at the SAME robot position (only rock bits differ, since
// the belief has uniform agent position but uncertain rock states).
//
// Key optimisation — per outer iteration:
//   1. Check whether cur_x_vec / cur_y_vec are uniform across all 8 lanes
//      (hmin == hmax is a single horizontal instruction each).
//   2. If uniform (common case):
//        a. Compute all candidate Manhattan distances with scalar arithmetic
//           (one operation per rock vs. 7 SIMD ops in the standard path).
//        b. Sort candidates ascending by distance with insertion sort
//           (n_cands ≤ 30, typically 2-5; typically O(n) if previous order
//           is close to the new one).
//        c. Scan the sorted list with SIMD "is_good" checks only.
//           Pre-mark lanes that have no remaining candidate rock as already
//           "done", then break as soon as every searching lane has found its
//           nearest good rock — using a single test_zero per inner step.
//   3. If not uniform (positions diverged after heterogeneous first visits):
//        Fall back to the standard pre-filtered SIMD inner loop (same as
//        the optimised rolloutENT above).
// ============================================================================
auto RockSampleStaticVecQMDP::rolloutENT_Homogenous(
    const IVectorT& new_pos_vec,
    const IVectorT& new_rocks_vec,
    const IVectorT& active_vec) const -> FVectorT
{
    // ---- Build candidate list ----
    constexpr int MAX_CANDS = 30;
    int cand_rx  [MAX_CANDS];   // rock x
    int cand_ry  [MAX_CANDS];   // rock y
    int cand_bit [MAX_CANDS];   // (1 << rock_index)
    int cand_dist[MAX_CANDS];   // scalar Manhattan distance from current pos (recomputed per iter)
    int sort_ord [MAX_CANDS];   // sort_ord[i] = ci → i-th nearest candidate
    int n_cands = 0;
    for (int r = 0; r < num_rocks_; ++r) {
        if (current_rock_probs_[r] < 0.1f) continue;
        cand_rx  [n_cands] = rock_pos_x_[r];
        cand_ry  [n_cands] = rock_pos_y_[r];
        cand_bit [n_cands] = (1 << r);
        sort_ord [n_cands] = n_cands;   // identity permutation; sorted in-place each iteration
        ++n_cands;
    }

    IVectorT cur_y_vec     = new_pos_vec / size_;
    IVectorT cur_x_vec     = new_pos_vec - (cur_y_vec * size_);
    IVectorT remaining_vec = new_rocks_vec;
    FVectorT cum_disc_vec  = FVectorT::fill(1.0f);
    FVectorT cum_val_vec   = FVectorT::fill(0.0f);

    const int32_t  NO_ROCK_DIST  = 2 * static_cast<int32_t>(size_);
    const IVectorT no_rock_dist_v = IVectorT::fill(NO_ROCK_DIST);

    for (int iter = 0; iter < n_cands; ++iter) {
        if (remaining_vec.test_zero()) break;

        IVectorT best_dist_vec = no_rock_dist_v;
        IVectorT best_x_vec    = cur_x_vec;
        IVectorT best_y_vec    = cur_y_vec;
        IVectorT best_bit_vec  = IVectorT::fill(0);

        // ---- Uniformity check: are all 8 lanes at the same (x, y)? ----
        // hmin/hmax are single horizontal instructions; comparison is scalar.
        const int sx = cur_x_vec.hmin();
        const int sy = cur_y_vec.hmin();
        const bool uniform = (sx == cur_x_vec.hmax()) && (sy == cur_y_vec.hmax());

        if (uniform) {
            // ---- Optimised path: scalar distances + sorted scan + early exit ----

            // Compute scalar distances from common position to each candidate.
            for (int ci = 0; ci < n_cands; ++ci) {
                const int dx = cand_rx[ci] - sx;
                const int dy = cand_ry[ci] - sy;
                cand_dist[ci] = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
            }

            // Insertion sort sort_ord by cand_dist ascending.
            // n_cands ≤ 30; O(n) when close to previous order.
            for (int i = 1; i < n_cands; ++i) {
                const int key  = sort_ord[i];
                const int kdst = cand_dist[key];
                int j = i - 1;
                while (j >= 0 && cand_dist[sort_ord[j]] > kdst) {
                    sort_ord[j + 1] = sort_ord[j];
                    --j;
                }
                sort_ord[j + 1] = key;
            }

            // Pre-mark lanes that have NO remaining candidate rock as "done"
            // so they don't prevent the early-exit condition from triggering.
            IVectorT has_cand = IVectorT::fill(0);
            for (int ci = 0; ci < n_cands; ++ci)
                has_cand = has_cand | ((remaining_vec & IVectorT::fill(cand_bit[ci])) != 0);
            // lanes with no candidate → already done (0xFFFFFFFF); others → pending (0)
            IVectorT lanes_found = ~has_cand;

            // Scan sorted candidates with SIMD is_good checks only.
            // Because candidates are sorted nearest-first, the first good rock
            // a lane finds IS its nearest good rock — no distance comparison
            // needed after that lane is marked done.
            for (int si = 0; si < n_cands; ++si) {
                const int     ci       = sort_ord[si];
                const IVectorT rock_bit  = IVectorT::fill(cand_bit[ci]);
                const IVectorT dist_v    = IVectorT::fill(cand_dist[ci]);  // scalar broadcast
                const IVectorT is_good   = (remaining_vec & rock_bit) != 0;
                // Update only lanes that need search and have not yet found a rock.
                const IVectorT need_upd  = is_good & (~lanes_found);

                best_dist_vec = IVectorT::select(need_upd, dist_v,                      best_dist_vec);
                best_x_vec    = IVectorT::select(need_upd, IVectorT::fill(cand_rx[ci]), best_x_vec);
                best_y_vec    = IVectorT::select(need_upd, IVectorT::fill(cand_ry[ci]), best_y_vec);
                best_bit_vec  = IVectorT::select(need_upd, rock_bit,                     best_bit_vec);
                lanes_found   = lanes_found | need_upd;

                // Early exit: no pending-search lane still needs a rock.
                // (lanes_found == 0xFFFFFFFF for all 8 lanes)
                if (lanes_found.all()) break;
            }

        } else {
            // ---- Fallback: standard pre-filtered SIMD inner loop ----
            // Positions differ per lane; full per-lane distance computation needed.
            for (int ci = 0; ci < n_cands; ++ci) {
                const IVectorT rock_bit  = IVectorT::fill(cand_bit[ci]);
                const IVectorT is_good   = (remaining_vec & rock_bit) != 0;
                const IVectorT diff_x    = IVectorT::fill(cand_rx[ci]) - cur_x_vec;
                const IVectorT diff_y    = IVectorT::fill(cand_ry[ci]) - cur_y_vec;
                const IVectorT dist      = (-diff_x).max(diff_x) + (-diff_y).max(diff_y);
                const IVectorT is_closer = is_good & (dist < best_dist_vec);

                best_dist_vec = IVectorT::select(is_closer, dist,                       best_dist_vec);
                best_x_vec    = IVectorT::select(is_closer, IVectorT::fill(cand_rx[ci]), best_x_vec);
                best_y_vec    = IVectorT::select(is_closer, IVectorT::fill(cand_ry[ci]), best_y_vec);
                best_bit_vec  = IVectorT::select(is_closer, rock_bit,                    best_bit_vec);
            }
        }

        const IVectorT found_vec    = (best_dist_vec < no_rock_dist_v);
        const FVectorT step_disc_10 = FVectorT::gather(discount_pow_table_.data(), best_dist_vec);
        const FVectorT contrib      = cum_disc_vec * step_disc_10;

        cum_val_vec  = cum_val_vec +
            FVectorT::select(found_vec.as<FVectorT>(), contrib, FVectorT::fill(0.0f));

        const FVectorT new_cum_disc = cum_disc_vec * (step_disc_10 / FVectorT::fill(10.0f));
        cum_disc_vec = FVectorT::select(found_vec.as<FVectorT>(), new_cum_disc, cum_disc_vec);

        cur_x_vec     = IVectorT::select(found_vec, best_x_vec, cur_x_vec);
        cur_y_vec     = IVectorT::select(found_vec, best_y_vec, cur_y_vec);
        remaining_vec = IVectorT::select(found_vec,
                                          remaining_vec & (~best_bit_vec),
                                          remaining_vec);
    }

    const IVectorT steps_to_exit =
        IVectorT::fill(static_cast<int32_t>(size_ - 1)) - cur_x_vec;
    cum_val_vec = cum_val_vec +
        cum_disc_vec * FVectorT::gather(discount_pow_table_.data(), steps_to_exit);

    const IVectorT is_term = (active_vec == IVectorT::fill(0));
    return FVectorT::select(is_term.as<FVectorT>(), FVectorT::fill(0.0f), cum_val_vec);
}
#endif // ENABLE_HOMOGENOUS_SEARCH

// ============================================================================
// Core BeliefTreeSearch loop
// ============================================================================

void RockSampleStaticVecQMDP::beliefTreeSearch(int robot_pos,
                                             const std::vector<float>& rock_probs,
                                             int max_iters)
{
    // ---- Setup: reset structures, seed scenarios ----
    current_rock_probs_ = rock_probs;
    total_step_batch_us_   = 0.0;  step_batch_call_count_  = 0;
    total_rollout_us_      = 0.0;  rollout_call_count_     = 0;
    total_backprop_us_     = 0.0;  backprop_call_count_    = 0;
    total_select_batch_us_ = 0.0;  select_batch_call_count_ = 0;
    simulation_count_static_ = 0;

    resetTreeStructures();
    // resetTreeStructures() does NOT reset node_active_flags_; do it here.
    std::fill(node_active_flags_.begin(), node_active_flags_.end(), 0xFFFFFFFF);
    sampleScenarios(robot_pos, rock_probs);

    // ---- Core search loop ----
    const auto wall_start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < max_iters; ++iter) 
    {
        if (iter % 10 == 0)
        {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall_start).count();
            if (elapsed_ms >= max_planning_time_ms_) break;
        }

        #ifdef ENABLE_HOMOGENOUS_SEARCH
            // 1. Selection — explore only scenario-0's subtree; the relative index
            //    and action are shared by all scenarios.
            int action_idx;
            uint32_t parent_rel = exploreNodesHomogenous(action_idx);

            // 2. Expansion — single action applied uniformly to every scenario.
            expandNodesBatch(parent_rel, action_idx);
        #else
            // 1. Selection — independent UCB traversal for each scenario.
            std::vector<int> action_idxs;
            AlignedVectorInt explore_nodes = exploreNodes(action_idxs);

            // 2. Expansion — per-scenario actions.
            expandNodesBatch(explore_nodes, action_idxs);
        #endif // ENABLE_HOMOGENOUS_SEARCH
    }
}

// ============================================================================
// VecQMDP virtual hooks
// ============================================================================

std::vector<uint32_t>
RockSampleStaticVecQMDP::computeNodeCandidateActions(uint32_t node_idx)
{
    std::vector<uint32_t> actions;

    // Terminal nodes: return a single dummy action so exploreNodes can
    // select them.  expandNodesBatch / beliefTreeSearch detect active_flag==0
    // and handle them without actual expansion.
    if (node_active_flags_[node_idx] == 0u) {
        actions.push_back(static_cast<uint32_t>(A_EAST));
        return actions;
    }

    const int depth = static_cast<int>(node_depth_[node_idx]);

    if (depth == 0) {
        // Single sense action at root — per-rock entropy bonuses applied post-search.
        actions.reserve(6u);
        actions.push_back(5u);
        for (uint32_t a = 0; a < 5u; ++a)
            actions.push_back(a);
    } else {
        // Deeper: movement + sample only
        actions.reserve(5u);
        for (uint32_t a = 0; a < 5u; ++a)
            actions.push_back(a);
    }
    return actions;
}

std::shared_ptr<vec_qmdp::planning::VecQMDP>
RockSampleStaticVecQMDP::makeWorkerInstance() const
{
    return std::make_shared<RockSampleStaticVecQMDP>(
        size_, num_rocks_, rock_pos_,
        start_x_, start_y_,
        half_efficiency_distance_,
        tree_height_,
        static_cast<int>(scenario_size_),
        1 /* single-threaded worker */);
}

// ============================================================================
// Parallel search entry point
// ============================================================================

int RockSampleStaticVecQMDP::parallelBeliefTreeSearch(int robot_pos,
                                        const std::vector<float>& rock_probs,
                                        int max_iters)
{
    current_rock_probs_ = rock_probs;
    const int  iters           = max_iters;
    const auto rock_probs_copy = rock_probs;

    dispatchParallelSearch(
        [robot_pos, iters, &rock_probs_copy](
            vec_qmdp::planning::VecQMDP* worker, size_t /*thread_idx*/)
        -> BeliefTreeSearchThreadResult
        {
            auto* w = static_cast<RockSampleStaticVecQMDP*>(worker);
            w->beliefTreeSearch(robot_pos, rock_probs_copy, iters);

            // Collect first-level action Q-values from this worker's scenarios.
            BeliefTreeSearchThreadResult result;
            result.action_sum_values.assign(w->num_actions_, 0.0f);
            result.action_counts    .assign(w->num_actions_, 0);
            result.simulation_count = w->simulation_count_static_;

            for (uint32_t a = 0; a < w->num_actions_; ++a) {
                const uint32_t child_rel = 1u + a;  // child of root
                if (child_rel >= w->tree_node_size_) continue;
                float sum_val   = 0.0f;
                int   valid_cnt = 0;
                for (uint32_t s = 0; s < w->scenario_size_; ++s) {
                    const uint32_t cg = w->getNodeIdx(child_rel, s);
                    if (w->node_count_[cg] > 0) {
                        sum_val += w->q_node_values_[cg];
                        ++valid_cnt;
                    }
                }
                result.action_sum_values[a] = sum_val;
                result.action_counts[a]     = valid_cnt;
            }
            return result;
        });

    total_simulation_count_static_ = aggregated_simulation_count_;
    applyEntropyBonuses();
    return getBestAction();
}

int RockSampleStaticVecQMDP::getBestAction() const
{
    const auto& vals = getAggregatedActionValues();
    int   best_a   = 0;
    float best_val = vals[0];
    for (uint32_t a = 1; a < num_actions_; ++a) 
    {
        if (vals[a] > best_val) 
        { 
            best_val = vals[a]; best_a = static_cast<int>(a); 
        }
    }
    if (best_a == 5) 
        return 5 + best_sense_rock_;
    return best_a;
}

void RockSampleStaticVecQMDP::printTimingStats() const
{
    constexpr int BATCH_SZ = static_cast<int>(IVectorT::num_scalars);
    
    if (num_worker_threads_ <= 1) {
        // ---- Single-threaded timing summary ----
        const double avg_step_us =
            (step_batch_call_count_ > 0)
            ? total_step_batch_us_ / static_cast<double>(step_batch_call_count_)
            : 0.0;
        const double avg_rollout_us =
            (rollout_call_count_ > 0)
            ? total_rollout_us_ / static_cast<double>(rollout_call_count_)
            : 0.0;
        const double avg_backprop_us =
            (backprop_call_count_ > 0)
            ? total_backprop_us_ / static_cast<double>(backprop_call_count_)
            : 0.0;
        const double avg_select_batch_us =
            (select_batch_call_count_ > 0)
            ? total_select_batch_us_ / static_cast<double>(select_batch_call_count_)
            : 0.0;
        std::cout << std::fixed << std::setprecision(3)
            << "[StaticVecQMDP timing] \n" << "stepBatch: "
            << "total: " << total_step_batch_us_ / 1000.0 << " ms"
            << " cnt: " << step_batch_call_count_ << " batch calls"
            << " avg: " << avg_step_us / 1000 << " ms/batch"
            << " (" << avg_step_us / BATCH_SZ / 1000 << " ms/scenario)"
            << "\n" << "rollout: "
            << "total: " << total_rollout_us_ / 1000.0 << " ms"
            << " cnt: " << rollout_call_count_ << " batch calls"
            << " avg: " << avg_rollout_us / 1000 << " ms/batch"
            << " (" << avg_rollout_us / BATCH_SZ / 1000 << " ms/scenario)"
            << "\n" << "backprop: "
            << "total: " << total_backprop_us_ / 1000.0 << " ms"
            << " cnt: " << backprop_call_count_ << " batch calls"
            << " avg: " << avg_backprop_us / 1000 << " ms/scenario"
            << "\n" << "select: "
            << "total: " << total_select_batch_us_ / 1000.0 << " ms"
            << " cnt: " << select_batch_call_count_ << " batch calls"
            << " avg: " << avg_select_batch_us / 1000 << " ms/batch"
            << " (" << avg_select_batch_us / BATCH_SZ / 1000 << " ms/scenario)"
            << "\n" << "  [" << total_simulation_count_static_ << " total batch calls]\n";
    } else {
        // ---- Multi-threaded timing summary ----
        double  agg_step_us     = 0.0;
        double  agg_rollout_us  = 0.0;
        double  agg_backprop_us = 0.0;
        double  agg_select_batch_us = 0.0;
        int64_t agg_step_calls  = 0;
        int64_t agg_rlout_calls = 0;
        int64_t agg_backprop_calls = 0;
        int64_t agg_select_batch_calls = 0;

        for (const auto& wb : worker_instances_) {
            const auto* w = static_cast<const RockSampleStaticVecQMDP*>(wb.get());
            agg_step_us    += w->total_step_batch_us_;
            agg_rollout_us += w->total_rollout_us_;
            agg_backprop_us += w->total_backprop_us_;
            agg_select_batch_us += w->total_select_batch_us_;
            agg_step_calls += w->step_batch_call_count_;
            agg_rlout_calls += w->rollout_call_count_;
            agg_backprop_calls += w->backprop_call_count_;
            agg_select_batch_calls += w->select_batch_call_count_;
        }

        const double avg_step_us =
            (agg_step_calls > 0)
            ? agg_step_us / static_cast<double>(agg_step_calls)
            : 0.0;
        const double avg_rollout_us =
            (agg_rlout_calls > 0)
            ? agg_rollout_us / static_cast<double>(agg_rlout_calls)
            : 0.0;
        const double avg_backprop_us =
            (agg_backprop_calls > 0)
            ? agg_backprop_us / static_cast<double>(agg_backprop_calls)
            : 0.0;
        const double avg_select_batch_us =
            (agg_select_batch_calls > 0)
            ? agg_select_batch_us / static_cast<double>(agg_select_batch_calls)
            : 0.0;
        std::cout << std::fixed << std::setprecision(3)
            << "[StaticVecQMDP timing] stepBatch: "
            << "total: " << agg_step_us / 1000.0 << " ms"
            << " avg: " << avg_step_us / 1000 << " ms/batch"
            << " (" << avg_step_us / BATCH_SZ / 1000 << " ms/scenario)"
            << " | rollout: "
            << "total: " << agg_rollout_us / 1000 << " ms"
            << " avg: " << avg_rollout_us / 1000 << " ms/batch"
            << " (" << avg_rollout_us / BATCH_SZ / 1000 << " ms/scenario)"
            << " | backprop: "
            << "total: " << agg_backprop_us / 1000 << " ms"
            << " avg: " << avg_backprop_us / 1000 << " ms/scenario"
            << " | select: "
            << "total: " << agg_select_batch_us / 1000 << " ms"
            << " avg: " << avg_select_batch_us / 1000 << " ms/batch"
            << " (" << avg_select_batch_us / BATCH_SZ / 1000 << " ms/scenario)"
            << "  [" << total_simulation_count_static_ << " batch calls]\n";
    }
}

// ============================================================================
// Entropy information-gain bonus (using current rock probabilities)
// ============================================================================

float RockSampleStaticVecQMDP::computeEntropyBonus(int rock_idx) const
{
    auto shannon = [](double p) -> double {
        if (p <= 0.0 || p >= 1.0) return 0.0;
        return -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
    };

    const double p_good = static_cast<double>(current_rock_probs_[rock_idx]);

    if (p_good < 0.1 || p_good > 0.9) return 0.0f;

    const double h_current = shannon(p_good);
    if (h_current < 1e-9) return 0.0f;

    // Compute average robot position from root nodes
    double avg_x = 0.0;
    double avg_y = 0.0;
    for (uint32_t s = 0; s < scenario_size_; ++s) {
        const uint32_t root_g = getNodeIdx(0u, s);
        const int pos = robot_pos_static_[root_g];
        avg_x += static_cast<double>(pos % size_);
        avg_y += static_cast<double>(pos / size_);
    }
    avg_x /= static_cast<double>(scenario_size_);
    avg_y /= static_cast<double>(scenario_size_);

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

    return static_cast<float>(entropy_lambda_ * (h_current - h_after));
}

void RockSampleStaticVecQMDP::applyEntropyBonuses()
{
    if (aggregated_action_counts_[5] == 0) return;
    float best_bonus = -1e9f;
    int   best_rock  = 0;
    for (int r = 0; r < num_rocks_; ++r) {
        const float bonus = computeEntropyBonus(r);
        std::cout << "[DEBUG] Rock " << r << ": p_good=" << current_rock_probs_[r]
          << ", bonus=" << bonus << "\n";
        if (bonus > best_bonus) { best_bonus = bonus; best_rock = r; }
    }
    aggregated_action_values_[5] += best_bonus;
    best_sense_rock_ = best_rock;
}

void RockSampleStaticVecQMDP::printTree(int max_depth, std::ostream& out) const
{
    out << "\n";
    out << "[VecQMDP] Tree dump (scenario 0, max_depth=" << max_depth << ")\n";

    struct Frame {
        uint32_t    node;
        int         depth;
        std::string prefix;
        bool        is_last;
    };

    for (int i = 0; i < 10; ++i)
    {
        out << "node idx: " << i << " count:" << node_count_[i] << " depth:" << node_depth_[i] << "\n";
    }

    std::vector<Frame> stack;
    stack.push_back({0, 0, "", true});

    int nodes_shown = 0;

    while (!stack.empty()) {
        const Frame f = stack.back();
        stack.pop_back();

        const uint32_t g = f.node;
        if (g >= node_size_) continue;

        const bool is_root  = (g == 0);
        const bool has_visits = (node_count_[g] > 0);
        if (!is_root && !has_visits) continue;

        const bool terminal = (node_active_flags_[g] == 0);

        if (f.depth == 0) {
            out << "ROOT"
                << "  N=" << node_count_[g]
                << "  INITIAL_Q=" << node_initial_rollout_[g]
                << "  Q=" << q_node_values_[g]
                << "  R=" << node_rewards_[g]
                << "  depth=" << node_depth_[g]
                << "\n";
        } else {
            std::string action_name = "";
            if (node_curr_action_idxs_[g] == static_cast<int32_t>(A_EAST)) 
            {
                action_name = "EAST";
            }
            else if (node_curr_action_idxs_[g] == static_cast<int32_t>(A_SOUTH)) 
            {
                action_name = "A_SOUTH";
            } 
            else if (node_curr_action_idxs_[g] == static_cast<int32_t>(A_NORTH)) 
            {
                action_name = "A_NORTH";
            } 
            else if (node_curr_action_idxs_[g] == static_cast<int32_t>(A_WEST)) 
            {
                action_name = "A_WEST";
            } 
            else if (node_curr_action_idxs_[g] == static_cast<int32_t>(A_SAMPLE)) 
            {
                action_name = "SAMPLE";
            } 
            else
            {
                action_name = "SENSE_" + std::to_string(node_curr_action_idxs_[g] - 5);
            }
            out << f.prefix
                << (f.is_last ? "└─ " : "├─ ")
                << "a=" << action_name
                << "  N=" << node_count_[g]
                << "  INITIAL_Q=" << node_initial_rollout_[g]
                << "  Q=" << q_node_values_[g]
                << "  R=" << node_rewards_[g]
                << "  depth=" << node_depth_[g];
            if (terminal) out << "  [TERMINAL]";
            out << "\n";
        }
        ++nodes_shown;

        if (f.depth >= max_depth || terminal) continue;

        // Collect visited children (visit count > 0)
        std::vector<uint32_t> children;
        children.reserve(num_actions_);
        const uint32_t base_rel = g; // scenario 0: global == relative
        for (uint32_t a = 0; a < num_actions_; ++a) {
            const uint32_t child_rel = base_rel * num_actions_ + a + 1;
            if (child_rel >= tree_node_size_) break;
            const uint32_t child_g = child_rel;
            if (child_g < node_size_ && node_count_[child_g] > 0)
                children.push_back(child_g);
        }

        const std::string child_prefix = f.prefix + (f.is_last ? "   " : "│  ");
        for (int i = static_cast<int>(children.size()) - 1; i >= 0; --i) {
            const bool child_last = (i == static_cast<int>(children.size()) - 1);
            stack.push_back({children[static_cast<size_t>(i)], f.depth + 1,
                             child_prefix, child_last});
        }
    }

    out << "\n[VecQMDP] Printed " << nodes_shown
        << " node(s) (visit_count > 0, depth ≤ " << max_depth << ")\n\n";
}

} // namespace rock_sample_vecqmdp
