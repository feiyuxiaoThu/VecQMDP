/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file main.cpp
 * @brief Rock Sample POMDP solved with VecQMDP multi-threaded BeliefTreeSearch.
 *
 * Pipeline:
 *   1. Build the DESPOT RockSample model (transitions + observation probs).
 *   2. Initialise a uniform particle belief over all rock-status combinations
 *      at the start position.
 *   3. Sample one true world state.
 *   4. Repeat until terminal or max_steps:
 *        a. Run VecQMDP BeliefTreeSearch on the current belief  → best action.
 *        b. Execute the action in the true state.
 *        c. Update the particle belief with a standard particle filter.
 *   5. Report total discounted reward and timing statistics.
 *
 * Usage:
 *   ./Vec-QMDP_rock_sample_vecqmdp [size rocks [threads [iters [steps [rock_bits [tree_height [num_scenarios]]]]]]]
 *   Defaults: size=7, rocks=8, threads=8, iters=3000, steps=40, rock_bits=-1, tree_height=6, num_scenarios=512
 *   Static solver tip: pass tree_height=4 (e.g. argv[7]=4) to keep memory ≤ ~10 MB/worker.
 *
 * ============================================================================
 * Solver selection
 * ============================================================================
 * Uncomment USE_STATIC_SOLVER to use the static (pre-allocated tree) VecQMDP
 * solver instead of the default dynamic-pool DynVecQMDP solver.
 *
 * IMPORTANT memory constraint for the static solver:
 *   The entire tree is pre-allocated at construction.  For 13 actions the
 *   estimated memory per worker is:
 *     tree_height=3 →   ~0.5 MB   ✓
 *     tree_height=4 →   ~10  MB   ✓
 *     tree_height=5 →  ~200  MB   △ (feasible but large)
 *     tree_height=6 →   ~2.5 GB   ✗ (will likely OOM)
 *   => Use tree_height ≤ 4 with the static solver for rock sample.
 *   => The static solver fixes scenario_size to IVectorT::num_scalars (= 8)
 *      per worker regardless of the num_scenarios argument.
 */

// ============================================================================
// Solver selection macro — uncomment to switch to the static VecQMDP solver.
// ============================================================================
// #define USE_STATIC_SOLVER

#ifdef USE_STATIC_SOLVER
#  include "rock_sample_vecqmdp/rock_sample_static_vecqmdp.hpp"
#else
#  include "rock_sample_vecqmdp/rock_sample_dynamic_vecqmdp.hpp"
#endif

// DESPOT model headers (for model + observation probability)
#include "rock_sample_despot/rock_sample_despot.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

// ============================================================================
// Accessor shim — exposes protected BaseRockSample members
// ============================================================================

class RockSampleAccessor : public despot::RockSample
{
public:
    using despot::RockSample::RockSample;

    int                               getSize()       const { return size_;       }
    int                               getNumRocks()   const { return num_rocks_;  }
    const std::vector<despot::Coord>& getRockPos()    const { return rock_pos_;   }
    despot::Coord                     getStartPos()   const { return start_pos_;  }
    double                            getHalfEff()    const
    { return half_efficiency_distance_; }

    int coordToIndex(despot::Coord c) const { return CoordToIndex(c); }
    int numStates()                   const { return NumStates();      }
};

// ============================================================================
// Belief representation — independent rock probabilities
// ============================================================================

struct RockSampleBelief
{
    int                  robot_pos;     // Robot position (flat index)
    std::vector<float>   rock_probs;    // P(rock_i is good), one per rock

    RockSampleBelief(int pos, int num_rocks)
        : robot_pos(pos)
        , rock_probs(static_cast<size_t>(num_rocks), 0.5f) {}
};

// ============================================================================
// Helpers
// ============================================================================

namespace {

RockSampleBelief buildInitialBelief(const RockSampleAccessor& model)
{
    const int num_rocks = model.getNumRocks();
    const int pos       = model.coordToIndex(model.getStartPos());

    // Initialize with uniform belief: all rocks have 0.5 probability of being good
    return RockSampleBelief(pos, num_rocks);
}

void updateBelief(RockSampleBelief&         belief,
                  const RockSampleAccessor&  model,
                  int                        action,
                  despot::OBS_TYPE           obs,
                  std::mt19937&              rng)
{
    const int num_rocks = model.getNumRocks();
    std::uniform_real_distribution<double> ud(0.0, 1.0);

    // Decode robot position
    const int old_pos_idx = belief.robot_pos;
    const int size = model.getSize();
    const int old_x = old_pos_idx % size;
    const int old_y = old_pos_idx / size;

    // ---- Handle different action types ----

    // 1. Movement actions (NORTH=0, EAST=1, SOUTH=2, WEST=3)
    if (action < 4) {
        int new_x = old_x;
        int new_y = old_y;

        if (action == 0) new_y += 1;        // NORTH
        else if (action == 1) new_x += 1;   // EAST
        else if (action == 2) new_y -= 1;   // SOUTH
        else if (action == 3) new_x -= 1;   // WEST

        // Check bounds (excluding east exit)
        if (action == 1 && new_x >= size) {
            // East exit - terminal state, robot leaves
            belief.robot_pos = old_pos_idx; // Keep at boundary
        } else if (new_x >= 0 && new_x < size && new_y >= 0 && new_y < size) {
            // Valid move
            belief.robot_pos = new_y * size + new_x;
        }
        // else: out of bounds, stay in place
    }

    // 2. Sample action (action == 4)
    else if (action == 4) {
        // Check if there's a rock at current position
        const auto& rock_positions = model.getRockPos();
        for (int r = 0; r < num_rocks; ++r) {
            if (rock_positions[r].x == old_x && rock_positions[r].y == old_y) {
                // Rock sampled - set its probability to 0
                belief.rock_probs[r] = 0.0f;
                break;
            }
        }
    }

    // 3. Sense action (action >= 5)
    else if (action >= 5) {
        const int rock_idx = action - 5;
        if (rock_idx < 0 || rock_idx >= num_rocks) return;

        // Bayesian update: P(rock=good | obs) ∝ P(obs | rock=good) * P(rock=good)

        // Get rock position for distance calculation
        const auto& rock_pos = model.getRockPos()[rock_idx];
        const double dx = static_cast<double>(rock_pos.x - old_x);
        const double dy = static_cast<double>(rock_pos.y - old_y);
        const double dist = std::sqrt(dx * dx + dy * dy);

        // Observation model: eta is the accuracy at this distance
        const double half_eff = model.getHalfEff();
        const double eta = 0.5 * (1.0 + std::pow(2.0, -dist / half_eff));

        // Prior probability
        const double p_good = static_cast<double>(belief.rock_probs[rock_idx]);
        const double p_bad = 1.0 - p_good;

        // Observation likelihood
        // P(obs=GOOD | rock=good) = eta
        // P(obs=GOOD | rock=bad) = 1 - eta
        double likelihood_good, likelihood_bad;

        if (obs == despot::BaseRockSample::E_GOOD) {
            likelihood_good = eta;
            likelihood_bad = 1.0 - eta;
        } else { // E_BAD
            likelihood_good = 1.0 - eta;
            likelihood_bad = eta;
        }

        // Posterior (unnormalized)
        const double posterior_good = likelihood_good * p_good;
        const double posterior_bad = likelihood_bad * p_bad;
        const double normalizer = posterior_good + posterior_bad;

        // Update belief
        if (normalizer > 1e-12) {
            belief.rock_probs[rock_idx] = static_cast<float>(posterior_good / normalizer);
        }
    }
}

/// Human-readable action name.
std::string actionName(int action, int num_rocks)
{
    static const char* dirs[] = { "NORTH", "EAST", "SOUTH", "WEST" };
    if (action < 4)  return dirs[action];
    if (action == 4) return "SAMPLE";
    std::ostringstream ss;
    ss << "SENSE_" << (action - 5);
    return ss.str();
}

/// Print Q-values as a bar chart, highlighting the best action.
void printQValuesBar(const std::vector<float>& vals, int best_action,
                     int num_rocks, int step, double plan_ms, int simulation_count)
{
    const int na  = static_cast<int>(vals.size());
    const int BAR = 20;

    float max_q = -std::numeric_limits<float>::max();
    float min_q =  std::numeric_limits<float>::max();
    for (int a = 0; a < na; ++a) {
        if (vals[a] < -9000000.0f) continue;
        max_q = std::max(max_q, vals[a]);
        min_q = std::min(min_q, vals[a]);
    }
    const float range = (max_q > min_q) ? (max_q - min_q) : 1.0f;

    std::cout << "--- Q-values  step=" << step
              << "  best=" << actionName(best_action, num_rocks)
              << "  [" << std::fixed << std::setprecision(1)
              << plan_ms << " ms, " 
              << simulation_count / plan_ms << " #/ms] ---\n";

    for (int a = 0; a < na; ++a) {
        if (vals[a] < -9000000.0f) continue;
        const std::string name = actionName(a, num_rocks);
        const int filled = static_cast<int>(
            (vals[a] - min_q) / range * BAR + 0.5f);
        std::cout << "  " << std::setw(10) << std::left << name
                  << std::setw(8) << std::right << std::fixed
                  << std::setprecision(2) << vals[a] << "  |";
        for (int i = 0; i < BAR; ++i)
            std::cout << (i < filled ? '#' : '.');
        std::cout << "|" << (a == best_action ? " <-- BEST" : "") << "\n";
    }
    std::cout << "Best action: " << actionName(best_action, num_rocks) << std::endl;
    std::cout << "\n";
}

/// Print a compact header with solver configuration.
void printHeader(int size, int num_rocks, int num_threads,
                 int max_iters, int max_steps, int num_scenarios,
                 bool use_static_solver)
{
    std::cout
        << "========================================\n"
        << " Rock Sample VecQMDP Solver\n"
        << "========================================\n"
        << std::left
        << "  Problem   : RockSample(" << size << ", " << num_rocks << ")\n"
        << "  Solver    : "
        << (use_static_solver ? "Static VecQMDP (pre-allocated tree)"
                              : "Dynamic VecQMDP (on-demand pool)")
        << "\n"
        << "  Threads   : " << num_threads << "\n"
        << "  MaxIters  : " << max_iters << " per planning step\n"
        << "  MaxSteps  : " << max_steps << "\n";
    if (use_static_solver) {
        std::cout
            << "  Scenarios : " << vec_qmdp::utils::FloatVectorWidth
            << " per worker (fixed = SIMD width)"
            << "  total=" << num_threads * static_cast<int>(vec_qmdp::utils::FloatVectorWidth)
            << "\n";
    } else {
        std::cout
            << "  Scenarios : " << num_scenarios
            << " per worker  (SIMD width "
            << vec_qmdp::utils::FloatVectorWidth << ")\n";
    }
    std::cout << "========================================\n\n";
}

} // anonymous namespace

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------
    // Parse command-line arguments
    // ------------------------------------------------------------------
    int size        = 7;
    int num_rocks   = 8;
    int tree_height = 6;
    int num_scenarios = 512;
    int num_threads = 8;
    int max_iters   = 3000;
    int max_steps   = 40;
    int rock_bits   = -1; // -1 = random, >= 0 = fixed good/bad bitmask

    if (argc >= 3) { size        = std::atoi(argv[1]); num_rocks     = std::atoi(argv[2]); }
    if (argc >= 4) { num_threads = std::atoi(argv[3]); }
    if (argc >= 5) { max_iters   = std::atoi(argv[4]); }
    if (argc >= 6) { max_steps   = std::atoi(argv[5]); }
    if (argc >= 7) { rock_bits   = std::atoi(argv[6]); }
    if (argc >= 8) { tree_height = std::atoi(argv[7]); }
    if (argc >= 9) { num_scenarios= std::atoi(argv[8]); }

#ifdef USE_STATIC_SOLVER
    constexpr bool use_static_solver = true;
#else
    constexpr bool use_static_solver = false;
#endif

    printHeader(size, num_rocks, num_threads, max_iters, max_steps,
                num_scenarios, use_static_solver);

    // ------------------------------------------------------------------
    // Build DESPOT model (used for transitions and observation probabilities)
    // ------------------------------------------------------------------
    RockSampleAccessor model(size, num_rocks);

    // ------------------------------------------------------------------
    // Build VecQMDP solver
    // ------------------------------------------------------------------
    //
    // USE_STATIC_SOLVER selects RockSampleStaticVecQMDP (pre-allocated tree,
    // scenario_size fixed to IVectorT::num_scalars = 8 per worker).
    // Default (undefined) selects RockSampleVecQMDP (DynVecQMDP, dynamic pool,
    // arbitrary num_scenarios).
    //
#ifdef USE_STATIC_SOLVER
    using SolverType = rock_sample_vecqmdp::RockSampleStaticVecQMDP;
    // Static solver ignores num_scenarios; each worker uses IVectorT::num_scalars (=8).
    SolverType solver(
        model.getSize(),
        model.getNumRocks(),
        model.getRockPos(),
        model.getStartPos().x,
        model.getStartPos().y,
        model.getHalfEff(),
        tree_height,
        num_scenarios,   // accepted but ignored internally
        num_threads);
    std::cout << "Solver ready. Static tree: fully pre-allocated "
              << "(tree_height=" << tree_height
              << ", tree_node_size per scenario ≈ pre-computed).\n\n";
#else
    using SolverType = rock_sample_vecqmdp::RockSampleVecQMDP;
    SolverType solver(
        model.getSize(),
        model.getNumRocks(),
        model.getRockPos(),
        model.getStartPos().x,
        model.getStartPos().y,
        model.getHalfEff(),
        tree_height,
        num_scenarios,
        num_threads);
    std::cout << "Solver ready. Dynamic pool: nodes allocated on demand "
              << "(tree_height=" << tree_height << ", no pre-allocation).\n\n";
#endif

    // ------------------------------------------------------------------
    // Initial belief: uniform over all rock-status combinations
    // ------------------------------------------------------------------
    RockSampleBelief belief = buildInitialBelief(model);

    // Determine the true state: fixed bitmask or random sampling
    std::mt19937 env_rng(static_cast<uint32_t>(42));
    int true_state;
    if (rock_bits >= 0) {
        const int mask = (1 << num_rocks) - 1;
        const int pos  = belief.robot_pos;
        true_state = (pos << num_rocks) | (rock_bits & mask);
        std::cout << "[rock_sample_vecqmdp] Fixed rock bits: " << rock_bits
                  << " (0b";
        for (int w = 8; w > 0; --w)
            std::cout << ((rock_bits >> (w - 1)) & 1);
        std::cout << ")\n";
    } else {
        // Random: sample rock bits uniformly
        std::uniform_int_distribution<int> uni(0, (1 << num_rocks) - 1);
        const int random_rocks = uni(env_rng);
        true_state = (belief.robot_pos << num_rocks) | random_rocks;
    }

    // Print initial state
    {
        despot::RockSampleState ts(true_state);
        std::cout << "Initial state:\n";
        model.PrintState(ts);
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Planning loop
    // ------------------------------------------------------------------
    double total_reward = 0.0;
    double discount     = 1.0;
    bool   terminated   = false;

    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const int terminal_id = model.numStates() - 1;

    double total_plan_ms  = 0.0;
    int    steps_taken    = 0;

    const auto wall_start = std::chrono::steady_clock::now();

    for (int step = 0; step < max_steps && !terminated; ++step)
    {
        std::cout << "******************* step " << step << " *******************" << std::endl;

        STEP = step;

        // ---- Plan with VecQMDP BeliefTreeSearch ----
        const auto plan_start = std::chrono::steady_clock::now();
        const int best_action = solver.parallelBeliefTreeSearch(belief.robot_pos, belief.rock_probs, max_iters);
        const double plan_ms  = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - plan_start).count();
        total_plan_ms += plan_ms;
        ++steps_taken;

        // ---- Print timing statistics (outside of plan_ms measurement) ----
#ifdef USE_STATIC_SOLVER
        solver.printTimingStats();
#endif

        // ---- Q-value bar chart ----
        printQValuesBar(solver.getAggregatedActionValues(),
                        best_action, num_rocks, step, plan_ms,
                        solver.getTotalSimulationCount());

        // ---- Execute action in the true environment ----
        despot::RockSampleState ts(true_state);
        double           step_reward = 0.0;
        despot::OBS_TYPE obs         = despot::BaseRockSample::E_NONE;

        const bool terminal = model.Step(ts, ud(env_rng),
                                         best_action, step_reward, obs);

        total_reward += discount * step_reward;
        discount     *= 0.95;

        // ---- Visualize state + belief ----
        {
            const int state_for_viz = terminal ? true_state : ts.state_id;
            despot::RockSampleState viz_state(state_for_viz);
            // Convert belief.rock_probs (float) to vector<double> for visualization
            std::vector<double> rock_belief(belief.rock_probs.begin(),
                                           belief.rock_probs.end());
            model.PrintStateViz(viz_state, rock_belief);
        }

        std::cout << "Action: ";
        model.PrintAction(best_action);
        std::cout << "Reward: " << step_reward << "  Obs: ";
        model.PrintObs(ts, obs);

        if (terminal || ts.state_id == terminal_id)
        {
            std::cout << "\n*** TERMINAL — robot exited east boundary ***\n";
            terminated = true;
        }
        else
        {
            true_state = ts.state_id;

            // ---- Update belief via Bayesian update ----
            updateBelief(belief, model, best_action, obs, env_rng);
        }
    } // planning loop

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();

    std::cout << "\n========================================\n"
              << " Results\n"
              << "========================================\n"
              << std::fixed << std::setprecision(3)
              << "  Discounted reward  : " << total_reward << "\n"
              << "  Steps taken        : " << steps_taken
              << (terminated ? " (terminated)" : " (max steps)") << "\n"
              << "  Avg plan time/step : "
              << std::setprecision(1)
              << (steps_taken > 0 ? total_plan_ms / steps_taken : 0.0)
              << " ms\n"
              << "  Total wall time    : " << wall_ms << " ms\n"
              << "========================================\n";

    return 0;
}
