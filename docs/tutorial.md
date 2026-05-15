# VecQMDP Tutorial -- Implementing a Custom POMDP Domain

This tutorial walks through the process of implementing a new POMDP domain on
top of the VecQMDP belief tree search engine.  **Rock Sample** serves as the
running example throughout: a grid-world agent must sample rocks of uncertain
quality, sense their status, and exit the east boundary.

After reading this guide you will know how to:

1. Choose between the **Static** and **Dynamic** tree backends (and when to
   enable **Homogeneous Search**).
2. Convert a traditional Array-of-Structures (AoS) state representation into a
   SIMD-friendly Structure-of-Arrays (SoA) layout.
3. Redesign the action space so that transitions are fully data-parallel.
4. Use the VAMP SIMD vector interface (`IVectorT` / `FVectorT`) to write
   branchless, vectorised `stepBatch` and rollout functions.
5. Wire the domain into VecQMDP search loop and build it.

> **Prerequisites:** Familiarity with POMDP concepts (states, actions,
> observations, beliefs) and C++17.  No SIMD expertise is needed -- the library
> abstracts intrinsics behind a portable vector API.

---

## Table of Contents

- [1. Architectural Overview](#1-architectural-overview)
- [2. Choosing a Tree Backend](#2-choosing-a-tree-backend)
  - [2.1 Static VecQMDP](#21-static-vecqmdp)
  - [2.2 Dynamic DynVecQMDP](#22-dynamic-dynvecqmdp)
  - [2.3 Decision Flowchart](#23-decision-flowchart)
  - [2.4 Homogeneous Search Mode](#24-homogeneous-search-mode)
- [3. Data-Oriented Design: AoS to SoA](#3-data-oriented-design-aos-to-soa)
  - [3.1 The Traditional AoS State](#31-the-traditional-aos-state)
  - [3.2 The SoA Transformation](#32-the-soa-transformation)
  - [3.3 Precomputed Lookup Tables](#33-precomputed-lookup-tables)
  - [3.4 Conversion Guidelines](#34-conversion-guidelines)
- [4. Action Space Design for SIMD Transitions](#4-action-space-design-for-simd-transitions)
  - [4.1 The Problem with switch-case](#41-the-problem-with-switch-case)
  - [4.2 Table-Driven Action Translation](#42-table-driven-action-translation)
  - [4.3 Heterogeneous stepBatch (Full SIMD)](#43-heterogeneous-stepbatch-full-simd)
  - [4.4 Homogeneous stepBatch Specialisation](#44-homogeneous-stepbatch-specialisation)
- [5. SIMD Operations Reference](#5-simd-operations-reference)
  - [5.1 Vector Types](#51-vector-types)
  - [5.2 Construction and I/O](#52-construction-and-io)
  - [5.3 Arithmetic](#53-arithmetic)
  - [5.4 Comparison and Masking](#54-comparison-and-masking)
  - [5.5 Selection (Branchless Conditional)](#55-selection-branchless-conditional)
  - [5.6 Gather and Scatter](#56-gather-and-scatter)
  - [5.7 Horizontal Reductions](#57-horizontal-reductions)
  - [5.8 Type Casting](#58-type-casting)
- [6. Step-by-Step Implementation Guide](#6-step-by-step-implementation-guide)
  - [6.1 Derive from the Base Class](#61-derive-from-the-base-class)
  - [6.2 Constructor -- Tables and Hyperparameters](#62-constructor----tables-and-hyperparameters)
  - [6.3 sampleScenarios -- Initialise Root Belief](#63-samplescenarios----initialise-root-belief)
  - [6.4 stepBatch -- SIMD Transition Function](#64-stepbatch----simd-transition-function)
  - [6.5 Rollout Policy](#65-rollout-policy)
  - [6.6 expandNodesBatch -- Tying It All Together](#66-expandnodesbatch----tying-it-all-together)
  - [6.7 Wire Up main()](#67-wire-up-main)
- [7. Build Integration](#7-build-integration)
- [8. Further Reading](#9-further-reading)

---

## 1. Architectural Overview

VecQMDP is organised in three layers:

```
+------------------------------------------------------------------+
|                     Your Domain                                   |
|  (RockSampleStaticVecQMDP, RockSampleVecQMDP, VecQMDP_AD, ...)   |
+------------------------------------------------------------------+
|            VecQMDP / DynVecQMDP Base Class                        |
|  UCB selection, back-propagation, early termination,              |
|  parallel search infrastructure, precomputed UCB tables           |
+------------------------------------------------------------------+
|               VAMP SIMD Vector Interface                          |
|  FloatVector, IntVector, gather, scatter, select,                 |
|  arithmetic, comparisons, horizontal reductions                   |
+------------------------------------------------------------------+
```

The base class handles all generic tree-search machinery.  You only implement
four domain-specific hooks:

| Hook | Purpose |
|---|---|
| `sampleScenarios(...)` | Initialise root belief states for all scenarios |
| `expandNodesBatch(...)` | Forward-simulate one step + rollout for a batch of selected nodes |
| `computeNodeCandidateActions(node_idx)` | Return the priority-ordered list of valid actions at a node |
| `makeWorkerInstance()` | Clone factory for multi-threaded search |

All domain-specific physics live inside `expandNodesBatch`, which in turn calls
your `stepBatch` (SIMD-parallel transition) and a rollout function.

**Search loop overview** (base class drives, your code provides the physics):

```
while (budget_remaining) {
    (parent_node, action) = exploreNodes*();     // UCB traversal   (base)
    expandNodesBatch(parent_node, action);        // step + rollout  (YOU)
        |-- stepBatch(...)                        // SIMD transition (YOU)
        |-- rollout(...)                          // heuristic value (YOU)
        |-- scatter child state                   // SIMD write      (YOU)
    backPropagate*(child_node, rollout_values);   // value backup    (base)
}
best_action = argmax(root Q-values);
```

---

## 2. Choosing a Tree Backend

### 2.1 Static VecQMDP

**Header:** `include/planning/vec_qmdp_static.hpp`

**Base class:** `vec_qmdp::planning::VecQMDP`

The tree is an implicit A-ary heap stored in pre-allocated flat arrays:

```
tree_node_size = (A^(H+1) - 1) / (A - 1)

node_size = tree_node_size x scenario_size
```

Children of relative index `r` occupy indices `[A*r + 1, A*r + A]`.
Parent formula: `parent_rel = (child_rel - 1) / A` -- no pointer chasing.

**Advantages:**
- Zero allocation during search; all memory is contiguous and cache-friendly.
- Implicit parent/child navigation via integer arithmetic.

**Limitation:**
- Memory grows exponentially: O(A^H x S).

**Memory estimates (per worker, 8 scenarios):**

| Actions | Height | Tree Nodes | Approx. Memory |
|---------|--------|------------|-----------------|
| 9       | 10     | ~2.6 B     | ~10 GB -- infeasible |
| 13      | 3      | 2,380      | ~0.5 MB |
| 13      | 4      | 30,941     | ~10 MB |
| 13      | 5      | 402,234    | ~200 MB |

**Rule of thumb:** Use static when `A^H < ~100,000`.

### 2.2 Dynamic DynVecQMDP

**Header:** `include/planning/vec_qmdp_dynamic.hpp`

**Base class:** `vec_qmdp::planning::DynVecQMDP`

Nodes are allocated on demand via `DynSoAPool` (chunked pool, 32,768-slot
blocks).  Each node stores an explicit `parent_idx` field.

**Advantages:**
- Memory is O(visited_nodes), typically O(max_iters x A).
- Supports arbitrarily deep trees and large action spaces.

**Limitation:**
- Slightly higher per-expansion overhead (pool allocation, explicit topology).
- Random-access pool layout may cause more cache misses.

**Example:** Rock Sample with A=13, H=90 would require 13^90 (approx 10^100)
nodes with the static backend -- clearly infeasible.  DynVecQMDP keeps it at
~40,000 nodes per thread for a 3,000-iteration budget.

### 2.3 Decision Flowchart

```
Is A^H < ~100,000?
  +-- YES --> Use Static VecQMDP (vec_qmdp_static.hpp)
  |           Is expansion cost very cheap (< 100 ns per scenario)?
  |             +-- YES --> Enable Homogeneous Search
  |             +-- NO  --> Use standard exploreNodesVote
  +-- NO  --> Use Dynamic DynVecQMDP (vec_qmdp_dynamic.hpp)
              Is expansion cost very cheap?
                +-- YES --> Enable Homogeneous Search
                +-- NO  --> Use standard exploreNodesVote
```

**Rock Sample example:**
- A=13 (4 moves + sample + 8 sense), H=4 --> 30,941 nodes ~ 10 MB/worker --> Static is perfect.
- The transition function is trivially cheap (integer arithmetic) --> Homogeneous search yields further speedup.
- For deeper search (H=90), switch to the Dynamic backend.

### 2.4 Homogeneous Search Mode

Enabled by `#define ENABLE_HOMOGENOUS_SEARCH` in your domain header.

**Key idea:** Only scenario 0 performs UCB traversal.  The resulting tree path
is **shared identically across all S scenarios**.  Each scenario independently
evaluates its own state at the shared node.

**Implementation:** The base class provides:
- `exploreNodesHomogenous(action_idx)` -- scalar UCB descent on scenario 0 only;
  returns the relative node index.
- `backPropagateHomogenous(child_rel, rollout_values)` -- propagates all S
  scenarios returns along the shared path in SIMD batches.

**Trade-off:** One scalar UCB traversal replaces S independent SIMD traversals.
For Rock Sample, where `stepBatch` costs ~10 ns/scenario but UCB descent costs
~50 ns/node, this delivers a significant net speedup.

**When to use:**
- Expansion (`stepBatch` + rollout) is cheap relative to UCB selection overhead.
- State transitions are deterministic given the action.

**When NOT to use:**
- Scenarios have vastly different dynamics (e.g., autonomous driving with
  heterogeneous traffic participants).
- You need per-scenario search diversity to avoid premature convergence.

---

## 3. Data-Oriented Design: AoS to SoA

The single most important design decision for VecQMDP performance is the **state
layout**.  Traditional POMDP solvers store each scenario state as an object
(Array of Structures).  VecQMDP requires a **Structure of Arrays** (SoA) layout
so that SIMD instructions can process 8 scenarios in one instruction.

### 3.1 The Traditional AoS State

In the DESPOT baseline (`examples/rock_sample/src/base/base_rock_sample.h`),
state is a class:

```cpp
// Array-of-Structures (AoS) -- one object per scenario
class RockSampleState : public State {
public:
    int robot_position;    // encoded in state_id
    int rock_bits;         // encoded in state_id as a bitmask
};
```

The model stores domain parameters as struct members:

```cpp
class BaseRockSample {
protected:
    Grid<int>              grid_;       // 2D grid mapping (x,y) to rock index
    std::vector<Coord>     rock_pos_;   // rock positions (Coord structs)
    int                    size_;       // grid side length
    int                    num_rocks_;
    Coord                  start_pos_;  // robot start position (Coord struct)
    double                 half_efficiency_distance_;
};
```

The step function processes **one scenario at a time** with branching:

```cpp
// From examples/rock_sample/src/rock_sample_despot/rock_sample_despot.cpp
bool RockSample::Step(State& state, double rand_num,
                      ACT_TYPE action, double& reward, OBS_TYPE& obs) const
{
    RockSampleState& rs = static_cast<RockSampleState&>(state);
    reward = 0;
    obs = E_NONE;

    if (action < E_SAMPLE) {
        switch (action) {                            // <-- per-scenario branch
        case Compass::EAST:
            if (GetX(&rs) + 1 < size_) { IncX(&rs); break; }
            else { reward = +10; return true; }      // <-- terminal
        case Compass::NORTH:
            if (GetY(&rs) + 1 < size_) IncY(&rs);
            else reward = -100;
            break;
        case Compass::SOUTH: ...
        case Compass::WEST:  ...
        }
    }
    // ... sample and sense omitted for brevity ...
    return false;
}
```

**Problems for SIMD:**
- Each scenario takes a different branch through the `switch`.
- Pointer-based `State&` prevents contiguous memory access.
- SIMD lanes cannot diverge -- all 8 must execute the same instruction stream.

### 3.2 The SoA Transformation

In VecQMDP (`examples/rock_sample/src/rock_sample_vecqmdp/rock_sample_static_vecqmdp.hpp`),
state fields become **flat arrays indexed by global node index**:

```
global_index g(rel, scenario) = rel + scenario x tree_node_size
```

```cpp
// Structure-of-Arrays (SoA)
class RockSampleStaticVecQMDP : public vec_qmdp::planning::VecQMDP {
    // ---- Domain state arrays (one int per <node, scenario> pair) ----
    std::vector<int> robot_pos_static_;   // flat position: y * size + x
    std::vector<int> rock_bits_static_;   // bitmask: bit r = 1 iff rock r is good

    // ---- Precomputed lookup tables ----
    std::vector<int>   action_dx_;         // action --> delta-x  (size = num_actions)
    std::vector<int>   action_dy_;         // action --> delta-y
    std::vector<int>   rock_pos_x_;        // rock index --> x position
    std::vector<int>   rock_pos_y_;        // rock index --> y position
    std::vector<int>   grid_;              // flat grid: grid_[pos] = rock index or -1
    std::vector<float> discount_pow_table_; // k --> 10 * gamma^k (rollout speedup)
};
```

**Why this works for SIMD:**
- `robot_pos_static_[g]` for 8 consecutive scenarios lies in contiguous memory.
- A single `IVectorT::gather(robot_pos_static_.data(), index_vec)` loads all 8
  positions with one AVX2 instruction.
- No pointers, no virtual dispatch, no branch divergence.

### 3.3 Precomputed Lookup Tables

A core SoA pattern is **precomputing per-action or per-entity data** into flat
arrays that can be accessed via SIMD gather:

```cpp
// From rock_sample_static_vecqmdp.cpp -- constructor
RockSampleStaticVecQMDP::RockSampleStaticVecQMDP(...)
{
    // --- Action-delta lookup (4 moves + sample + sense) ---
    action_dx_.assign(6u, 0);
    action_dy_.assign(6u, 0);
    action_dy_[A_NORTH] = +1;   // North: dy = +1
    action_dx_[A_EAST]  = +1;   // East:  dx = +1
    action_dy_[A_SOUTH] = -1;   // South: dy = -1
    action_dx_[A_WEST]  = -1;   // West:  dx = -1
    // Sample (4) and Sense (5+k): dx = dy = 0

    // --- Separate x/y rock positions for SIMD gather ---
    rock_pos_x_.resize(num_rocks_);
    rock_pos_y_.resize(num_rocks_);
    for (int i = 0; i < num_rocks_; ++i) {
        rock_pos_x_[i] = rock_pos_[i].x;   // extract from Coord struct
        rock_pos_y_[i] = rock_pos_[i].y;
    }

    // --- Flat grid: position --> rock index ---
    grid_.assign(size_ * size_, -1);
    for (int i = 0; i < num_rocks_; ++i)
        grid_[rock_pos_[i].y * size_ + rock_pos_[i].x] = i;

    // --- Precomputed discount powers: 10 * gamma^k ---
    discount_pow_table_.resize(2 * size_ + 1);
    for (int k = 0; k <= 2 * size_; ++k)
        discount_pow_table_[k] = 10.0f * std::pow(discount_factor_, k);
}
```

**Design principle:** Any data that would appear inside a per-scenario loop
should be extracted into a table indexed by action, entity, or grid position --
enabling `gather` instead of branching.

### 3.4 Conversion Guidelines

| AoS Pattern | SoA Equivalent |
|---|---|
| `state.position` (Coord/struct) | Flat `int pos = y * width + x`; decode with `x = pos % w`, `y = pos / w` |
| `state.inventory[item]` | Bitmask `int inventory_bits` if items are boolean |
| `std::vector<Agent> agents` | Parallel arrays: `agent_x_[]`, `agent_y_[]`, `agent_heading_[]` |
| `if (state.fuel > 0)` branch | Mask: `IVectorT has_fuel = (fuel_vec > 0)` then `select` |
| `struct { double dist; int id; }` | Two separate arrays: `dist_[]`, `id_[]` |
| `rock_pos_[i].x`, `rock_pos_[i].y` | `rock_pos_x_[i]`, `rock_pos_y_[i]` |

**Rules of thumb:**
1. **Flatten structs** -- split compound types (Coord, Point) into separate scalar arrays.
2. **Encode enums and booleans as integers** -- use bitmasks where possible.
3. **Precompute anything action-indexed** into a lookup table.
4. **Use `AlignedVectorInt` / `AlignedVectorFloat`** for arrays accessed by SIMD gather/scatter (32-byte aligned).

---

## 4. Action Space Design for SIMD Transitions

### 4.1 The Problem with switch-case

The traditional DESPOT step function uses `switch (action)` with completely
different code paths per case.  SIMD registers cannot diverge: all 8 lanes must
execute the same instruction sequence.

Even in the heterogeneous `stepBatch` (where different scenarios may execute
different actions), we must compute results for **all** action categories
simultaneously and then `select` the correct output per lane.

### 4.2 Table-Driven Action Translation

Replace `switch-case` navigation with **precomputed delta tables** +
**SIMD gather**.

**Before (AoS, one scenario at a time):**
```cpp
switch (action) {
case Compass::EAST:  if (x+1 < size) x++; else { reward=10; return true; } break;
case Compass::NORTH: if (y+1 < size) y++; else reward=-100; break;
case Compass::SOUTH: if (y-1 >= 0)   y--; else reward=-100; break;
case Compass::WEST:  if (x-1 >= 0)   x--; else reward=-100; break;
}
```

**After (SoA, 8 scenarios in parallel):**
```cpp
// Precomputed tables (set once in constructor):
//   action_dx_ = { 0, +1,  0, -1, 0, 0 }   (N=0, E=1, S=2, W=3, Sample=4, Sense=5)
//   action_dy_ = {+1,  0, -1,  0, 0, 0 }

// In stepBatch -- 8 lanes simultaneously:
const IVectorT dx = IVectorT::gather(action_dx_.data(), action_vec);
const IVectorT dy = IVectorT::gather(action_dy_.data(), action_vec);
const IVectorT nx = x_vec + dx;
const IVectorT ny = y_vec + dy;
```

**The fundamental pattern:**
1. **Gather** per-action parameters from precomputed tables.
2. **Compute** results for all action categories simultaneously.
3. **Build masks** for each condition (boundary, terminal, valid).
4. **Select** the correct result per lane: `IVectorT::select(mask, if_true, if_false)`.

### 4.3 Heterogeneous stepBatch (Full SIMD)

When different scenarios may take different actions (standard mode), every action
category is computed branchlessly and results are blended via `select`:

```cpp
// From rock_sample_static_vecqmdp.cpp
void RockSampleStaticVecQMDP::stepBatch(
    const IVectorT& pos_vec,    // 8 positions
    const IVectorT& rocks_vec,  // 8 rock bitmasks
    const IVectorT& action_vec, // 8 action indices (may differ per lane)
    IVectorT& out_pos, IVectorT& out_rocks,
    IVectorT& out_active, FVectorT& out_reward)
{
    // 1. Decode 2D position from flat index
    const IVectorT y_vec = pos_vec / size_;
    const IVectorT x_vec = pos_vec - (y_vec * size_);

    // 2. Movement: gather per-lane deltas
    const IVectorT dx = IVectorT::gather(action_dx_.data(), action_vec);
    const IVectorT dy = IVectorT::gather(action_dy_.data(), action_vec);
    const IVectorT nx = x_vec + dx;
    const IVectorT ny = y_vec + dy;

    // 3. Branchless boundary checks via masks
    const IVectorT is_move    = action_vec < static_cast<int32_t>(A_SAMPLE);
    const IVectorT is_east    = (action_vec == static_cast<int32_t>(A_EAST));
    const IVectorT east_exit  = is_east & (nx >= size_);
    const IVectorT x_oob     = (nx < 0) | (nx >= size_);
    const IVectorT y_oob     = (ny < 0) | (ny >= size_);
    const IVectorT move_oob  = is_move & (x_oob | y_oob) & ~east_exit;
    const IVectorT valid_move = is_move & ~move_oob & ~east_exit;
    const IVectorT moved_pos = ny * size_ + nx;

    // 4. Sample action (branchless rock lookup)
    const IVectorT is_sample   = (action_vec == static_cast<int32_t>(A_SAMPLE));
    const IVectorT rock_at_vec = IVectorT::gather(grid_.data(), pos_vec);

    // 5. Sense action
    const IVectorT is_sense = (action_vec > static_cast<int32_t>(A_SAMPLE));

    // 6. Combine outputs via chained selects
    out_pos    = IVectorT::select(valid_move, moved_pos, pos_vec);
    out_rocks  = IVectorT::select(is_sample, sampled_rocks, rocks_vec);
    out_active = out_active & (~east_exit);

    out_reward = FVectorT::fill(0.0f);
    out_reward = FVectorT::select(is_sense.as<FVectorT>(),  -0.1f,   out_reward);
    out_reward = FVectorT::select(move_oob.as<FVectorT>(),  -100.0f, out_reward);
    out_reward = FVectorT::select(east_exit.as<FVectorT>(), +10.0f,  out_reward);
    out_reward = FVectorT::select(is_sample.as<FVectorT>(), samp_rew, out_reward);
    //  ^ Last select wins -- order matters for overlapping conditions.
}
```

**Note on variable-bit-shift:** AVX2 has no variable `1 << per_lane_index`
instruction.  Rock Sample requires per-scenario rock index lookups, so we fall
back to a scalar loop:

```cpp
// Scalar fallback for variable bit shift (no SIMD equivalent)
alignas(32) int32_t samp_set_arr[8], samp_clr_arr[8];
{
    const auto ra = rock_at_vec.to_array();  // extract 8 rock indices
    for (int s = 0; s < 8; ++s) {
        const int r     = ra[s];
        samp_set_arr[s] = (r >= 0) ? (1 << r) : 0;
        samp_clr_arr[s] = ~samp_set_arr[s];
    }
}
const IVectorT samp_set(samp_set_arr);   // reload into SIMD register
const IVectorT samp_clr(samp_clr_arr);

const IVectorT is_good_sample = (rocks_vec & samp_set) != 0;
const IVectorT sampled_rocks  = rocks_vec & samp_clr;
```

### 4.4 Homogeneous stepBatch Specialisation

When homogeneous search is enabled, all 8 scenarios execute the **same action**.
This allows a simpler implementation that branches once on the action category
(scalar) and only runs the SIMD work for that category:

```cpp
void RockSampleStaticVecQMDP::stepBatch(
    const IVectorT& pos_vec,
    const IVectorT& rocks_vec,
    int             action_idx,    // <-- scalar, same for all lanes
    IVectorT& out_pos, IVectorT& out_rocks,
    IVectorT& out_active, FVectorT& out_reward)
{
    // --- Sense: no state change, uniform reward ---
    if (action_idx > static_cast<int>(A_SAMPLE)) {
        out_pos = pos_vec; out_rocks = rocks_vec; out_reward = -0.1f;
        return;
    }

    // --- Sample: rock bit-clear logic only ---
    if (action_idx == static_cast<int>(A_SAMPLE)) {
        const IVectorT rock_at = IVectorT::gather(grid_.data(), pos_vec);
        // ... variable-bit-shift + select ...
        out_pos = pos_vec; out_rocks = rocks_vec & samp_clr; out_reward = samp_rew;
        return;
    }

    // --- Movement: dx/dy are scalar constants -- no gather needed! ---
    const IVectorT y_vec = pos_vec / size_;
    const IVectorT x_vec = pos_vec - (y_vec * size_);
    const int32_t dx = action_dx_[action_idx];
    const int32_t dy = action_dy_[action_idx];
    const IVectorT nx = x_vec + dx;    // scalar broadcast add
    const IVectorT ny = y_vec + dy;

    out_rocks = rocks_vec;  // movement never changes rocks

    if (action_idx == static_cast<int>(A_EAST)) {
        // East exit: terminal + reward
        const IVectorT east_exit = (nx >= static_cast<int32_t>(size_));
        out_pos    = IVectorT::select(~east_exit, ny * size_ + nx, pos_vec);
        out_active = out_active & (~east_exit);
        out_reward = FVectorT::select(east_exit.as<FVectorT>(), +10.0f, 0.0f);
    } else {
        // N/S/W: OOB penalty
        const IVectorT oob = (nx < 0) | (nx >= size_) | (ny < 0) | (ny >= size_);
        out_pos    = IVectorT::select(~oob, ny * size_ + nx, pos_vec);
        out_reward = FVectorT::select(oob.as<FVectorT>(), -100.0f, 0.0f);
    }
}
```

**Benefits:**
- Sense and sample skip all movement computation entirely.
- Movement uses scalar `dx`/`dy` -- no gather instruction at all.
- The scalar branch (`if/else`) is fine because the choice is the same for all 8 lanes.

---

## 5. SIMD Operations Reference

VecQMDP wraps AVX2 intrinsics via the VAMP vector interface
(`external/vamp/src/impl/vamp/vector/interface.hh`).  Two type aliases are
used throughout:

```cpp
using IVectorT = vec_qmdp::utils::IVectorT_qmdp;  // 8-wide int32 vector
using FVectorT = vec_qmdp::utils::FVectorT_qmdp;   // 8-wide float vector
```

### 5.1 Vector Types

| Type | Element Type | Width | Underlying |
|------|-------------|-------|------------|
| `IVectorT` | `int32_t` | 8 lanes | `__m256i` (AVX2) |
| `FVectorT` | `float`   | 8 lanes | `__m256`  (AVX2) |

### 5.2 Construction and I/O

```cpp
// Broadcast a scalar to all 8 lanes
IVectorT v = IVectorT::fill(42);           // [42, 42, 42, 42, 42, 42, 42, 42]
FVectorT f = FVectorT::fill(3.14f);

// Ascending sequence
IVectorT seq = IVectorT::iota(0);          // [0, 1, 2, 3, 4, 5, 6, 7]
IVectorT seq5 = IVectorT::iota(5);         // [5, 6, 7, 8, 9, 10, 11, 12]

// Construct from a C array (must be 32-byte aligned)
alignas(32) int32_t arr[8] = {1,2,3,4,5,6,7,8};
IVectorT v(arr);

// Extract to std::array
auto result = v.to_array();                // std::array<int32_t, 8>

// Extract to a raw pointer
v.to_array(output_ptr);

// Load from contiguous memory (faster than gather for sequential indices)
IVectorT loaded = IVectorT::load_contiguous(data_ptr, offset);
```

### 5.3 Arithmetic

All arithmetic operates element-wise across 8 lanes:

```cpp
IVectorT a = IVectorT::fill(10);
IVectorT b = IVectorT::fill(3);

IVectorT sum  = a + b;     // [13, 13, ...]
IVectorT diff = a - b;     // [7, 7, ...]
IVectorT prod = a * b;     // [30, 30, ...]
IVectorT quot = a / b;     // [3, 3, ...]  (integer division)
IVectorT neg  = -a;        // [-10, -10, ...]

FVectorT x = FVectorT::fill(2.0f);
FVectorT y = FVectorT::fill(0.5f);
FVectorT r = x * y + FVectorT::fill(1.0f);  // [2.0, 2.0, ...]
```

Scalar broadcast is implicit for binary operators:

```cpp
FVectorT scaled = f * 2.0f;        // broadcast 2.0f to all lanes
IVectorT offset = v + 5;           // broadcast 5 to all lanes
```

**Min / Max:**

```cpp
IVectorT lo = a.min(b);            // per-lane minimum
IVectorT hi = a.max(b);            // per-lane maximum

// abs via negation + max (used in Rock Sample for Manhattan distance):
const IVectorT abs_dx = (-diff_x).max(diff_x);
```

### 5.4 Comparison and Masking

Comparisons return mask vectors (all-1s for true, all-0s for false):

```cpp
IVectorT a = ..., b = ...;

IVectorT eq  = (a == b);    // per-lane equality
IVectorT neq = (a != b);    // per-lane inequality
IVectorT lt  = (a < b);     // per-lane less-than
IVectorT gt  = (a > b);     // per-lane greater-than
IVectorT le  = (a <= b);
IVectorT ge  = (a >= b);

// Scalar comparisons (broadcast)
IVectorT big = (a >= 10);   // which lanes are >= 10?
```

Bitwise operations on masks:

```cpp
IVectorT combined = mask_a & mask_b;   // AND
IVectorT either   = mask_a | mask_b;   // OR
IVectorT inverted = ~mask_a;           // NOT
IVectorT clear    = mask_a & ~mask_b;  // AND-NOT
```

### 5.5 Selection (Branchless Conditional)

`select` is the SIMD equivalent of `condition ? true_val : false_val`:

```cpp
// IVectorT::select(mask, when_true, when_false)
IVectorT result = IVectorT::select(mask, value_a, value_b);
// Per lane i: result[i] = mask[i] ? value_a[i] : value_b[i]

// FVectorT version -- mask must be float-width (use .as<FVectorT>())
FVectorT reward = FVectorT::select(
    is_terminal.as<FVectorT>(),       // int mask --> float mask
    FVectorT::fill(+10.0f),           // terminal reward
    FVectorT::fill(0.0f));            // non-terminal reward
```

**Chained selects** implement if/else-if chains:

```cpp
FVectorT reward = FVectorT::fill(0.0f);                              // default
reward = FVectorT::select(is_sense.as<FVectorT>(),  -0.1f,   reward); // sense
reward = FVectorT::select(is_oob.as<FVectorT>(),    -100.0f, reward); // OOB
reward = FVectorT::select(is_exit.as<FVectorT>(),   +10.0f,  reward); // exit
reward = FVectorT::select(is_sample.as<FVectorT>(), samp_r,  reward); // sample
// ^ Last select wins for lanes where multiple conditions match.
```

### 5.6 Gather and Scatter

Gather loads elements from a base array using per-lane indices.  Scatter writes
back.

```cpp
// Gather: result[i] = base[index_vec[i]]
const IVectorT values = IVectorT::gather(table.data(), index_vec);
const FVectorT floats = FVectorT::gather(float_table.data(), index_vec);

// Example: load per-action dx for 8 scenarios with different actions
const IVectorT dx = IVectorT::gather(action_dx_.data(), action_vec);

// Example: load domain state for 8 scenario nodes
IVectorT pos_vec = IVectorT::gather(robot_pos_static_.data(), node_idxs_vec);

// Scatter: base[index_vec[i]] = vector[i]
new_pos_vec.scatter(robot_pos_static_.data(), child_global_idxs);
reward_vec.scatter(node_rewards_.data(), child_global_idxs);
```

**Performance note:** Gather issues one cache-line request per unique address
among the 8 indices.  When indices are consecutive, use `load_contiguous`
instead (single load instruction):

```cpp
// Consecutive indices starting at batch_start -- faster than gather
IVectorT scen_off = IVectorT::load_contiguous(scen_offsets_.data(),
                                               static_cast<int>(batch_start));
```

### 5.7 Horizontal Reductions

```cpp
FVectorT v = ...;
float min_val = v.hmin();          // minimum across all 8 lanes
float max_val = v.hmax();          // maximum across all 8 lanes

// Test whether all lanes are zero (useful for early exit)
bool all_zero = v.test_zero();     // true if every lane == 0

// Predicate checks on mask vectors
bool any_true  = mask.any();       // at least one lane is true
bool all_true  = mask.all();       // every lane is true
bool none_true = mask.none();      // no lane is true
```

**Example -- early exit in rollout:**

```cpp
for (int iter = 0; iter < n_cands; ++iter) {
    if (remaining_vec.test_zero()) break;  // all rocks collected --> stop
    // ... find nearest good rock per lane ...
}
```

### 5.8 Type Casting

```cpp
// Reinterpret int mask as float mask (bit-preserving, no conversion)
FVectorT float_mask = int_mask.as<FVectorT>();

// Required when passing int comparisons to FVectorT::select:
FVectorT reward = FVectorT::select(
    (action_vec > 4).as<FVectorT>(),    // int comparison --> float mask
    FVectorT::fill(-0.1f),
    FVectorT::fill(0.0f));
```

---

## 6. Step-by-Step Implementation Guide

This section provides a concrete implementation pattern using Rock Sample as the
reference.  Replace domain-specific details with your own.

### 6.1 Derive from the Base Class

```cpp
// my_domain_static.hpp
#pragma once
#include <planning/vec_qmdp_static.hpp>
#include <utils/global_utils.hpp>

#define ENABLE_HOMOGENOUS_SEARCH   // optional -- enable if expansion is cheap

class MyDomainVecQMDP : public vec_qmdp::planning::VecQMDP
{
public:
    using IVectorT = vec_qmdp::utils::IVectorT_qmdp;
    using FVectorT = vec_qmdp::utils::FVectorT_qmdp;

    enum Action : int {
        A_UP = 0, A_RIGHT = 1, A_DOWN = 2, A_LEFT = 3,
        A_PICKUP = 4, NUM_ACTIONS = 5
    };

    MyDomainVecQMDP(/* domain params */, uint32_t tree_height, int num_threads);

    // ---- Required overrides ----
    void sampleScenarios(int start_pos, const std::vector<float>& probs);
    void expandNodesBatch(uint32_t parent_rel, int action_idx);
    std::vector<uint32_t> computeNodeCandidateActions(uint32_t node_idx) override;
    std::shared_ptr<VecQMDP> makeWorkerInstance() const override;

private:
    // ---- SoA domain state ----
    std::vector<int> position_;      // one int per (node, scenario)
    std::vector<int> item_bits_;     // one int per (node, scenario)

    // ---- Precomputed tables ----
    std::vector<int> action_dx_;     // action --> delta-x
    std::vector<int> action_dy_;     // action --> delta-y
    std::vector<int> grid_;          // flat grid: position --> cell type

    // ---- SIMD transition ----
    void stepBatch(const IVectorT& pos, const IVectorT& items, int action,
                   IVectorT& out_pos, IVectorT& out_items,
                   IVectorT& out_active, FVectorT& out_reward);

    // ---- Rollout ----
    FVectorT rollout(const IVectorT& pos, const IVectorT& active) const;
};
```

### 6.2 Constructor -- Tables and Hyperparameters

```cpp
MyDomainVecQMDP::MyDomainVecQMDP(/* params */, uint32_t tree_height,
                                  int num_threads)
    : VecQMDP(
          static_cast<int>(IVectorT::num_scalars),  // scenarios = SIMD width (8)
          NUM_ACTIONS,                                // branching factor
          tree_height)                                // max search depth
{
    // 1. Allocate domain state arrays (node_size_ set by base class)
    position_.resize(node_size_, 0);
    item_bits_.resize(node_size_, 0);

    // 2. Build action-delta table
    action_dx_ = { 0, +1,  0, -1, 0 };  // UP, RIGHT, DOWN, LEFT, PICKUP
    action_dy_ = {+1,  0, -1,  0, 0 };

    // 3. Build any other lookup tables your domain needs
    grid_.assign(width_ * height_, EMPTY);
    // ... populate grid ...

    // 4. Set hyperparameters
    setDiscountFactor(0.95);
    setExplorationConstant(2.0f);
    setMaxPlanningTimeMs(500.0);
    setPrunedThreshold(-50.0f);
    setEarlyTermMinExpandCalls(500);
    setEarlyTermCheckInterval(15);
    setEarlyTermStableIterations(50);
    setEarlyTermMinBestVisits(20);
    setEarlyTermQChangeThreshold(0.1f);

    // 5. Build UCB tables (MUST be called after setExplorationConstant)
    initUCBTables();

    // 6. Enable multi-threading
    initParallelInfrastructure(static_cast<size_t>(num_threads));
}
```

### 6.3 sampleScenarios -- Initialise Root Belief

Seed root nodes from your belief distribution:

```cpp
void MyDomainVecQMDP::sampleScenarios(int start_pos,
                                       const std::vector<float>& probs)
{
    for (uint32_t s = 0; s < scenario_size_; ++s)
    {
        const uint32_t root_g = getNodeIdx(0u, s);  // global index of root, scenario s
        position_[root_g] = start_pos;

        // Sample hidden state from belief
        int bits = 0;
        for (int i = 0; i < num_items_; ++i) {
            if (uniform_dist_(rng_[s]) < probs[i])
                bits |= (1 << i);
        }
        item_bits_[root_g] = bits;
    }
}
```

### 6.4 stepBatch -- SIMD Transition Function

Follow the **gather --> compute --> mask --> select** pattern:

```cpp
void MyDomainVecQMDP::stepBatch(
    const IVectorT& pos_vec, const IVectorT& items_vec, int action_idx,
    IVectorT& out_pos, IVectorT& out_items,
    IVectorT& out_active, FVectorT& out_reward)
{
    // 1. Decode 2D coordinates
    const IVectorT y = pos_vec / width_;
    const IVectorT x = pos_vec - (y * width_);

    // 2. Get deltas (scalar -- same for all lanes in homogeneous mode)
    const int32_t dx = action_dx_[action_idx];
    const int32_t dy = action_dy_[action_idx];
    const IVectorT nx = x + dx;
    const IVectorT ny = y + dy;

    // 3. Boundary check (branchless)
    const IVectorT oob = (nx < 0) | (nx >= width_) | (ny < 0) | (ny >= height_);

    // 4. Compute outputs
    out_pos    = IVectorT::select(~oob, ny * width_ + nx, pos_vec);
    out_items  = items_vec;
    out_reward = FVectorT::select(oob.as<FVectorT>(), -1.0f, 0.0f);
    // out_active is unchanged unless you have terminal conditions
}
```

### 6.5 Rollout Policy

Provide a fast heuristic lower-bound value estimate:

```cpp
FVectorT MyDomainVecQMDP::rollout(const IVectorT& pos_vec,
                                   const IVectorT& active_vec) const
{
    // Example: always-east rollout -- value = 10 * gamma^(steps_to_exit)
    const IVectorT x = pos_vec - (pos_vec / width_) * width_;
    const IVectorT steps = IVectorT::fill(static_cast<int32_t>(width_ - 1)) - x;
    FVectorT val = FVectorT::gather(discount_table_.data(), steps);

    // Zero out terminated scenarios
    const IVectorT is_term = (active_vec == 0);
    return FVectorT::select(is_term.as<FVectorT>(), 0.0f, val);
}
```

A smarter rollout (like Rock Sample's ENT -- Explore Nearest in Thresholded
state) can dramatically improve search quality, but keep it cheap: the rollout
runs once per expansion.

### 6.6 expandNodesBatch -- Tying It All Together

For homogeneous search with the static backend:

```cpp
void MyDomainVecQMDP::expandNodesBatch(uint32_t parent_rel, int action_idx)
{
    constexpr uint32_t BATCH = IVectorT::num_scalars;  // 8

    const uint32_t child_rel = parent_rel * num_actions_ + 1 + action_idx;

    AlignedVectorFloat rollout_values(scenario_size_, 0.0f);

    for (uint32_t b = 0; b < scenario_size_; b += BATCH)
    {
        // 1. Build scenario offset vector
        IVectorT scen_off = IVectorT::load_contiguous(scen_offsets_.data(),
                                                       static_cast<int>(b));

        // 2. SIMD gather: load parent domain state
        IVectorT parent_g  = scen_off + static_cast<int32_t>(parent_rel);
        IVectorT pos_vec   = IVectorT::gather(position_.data(),           parent_g);
        IVectorT items_vec = IVectorT::gather(item_bits_.data(),          parent_g);
        IVectorT active    = IVectorT::gather(node_active_flags_.data(),  parent_g);

        // 3. SIMD step transition
        IVectorT new_pos, new_items;
        FVectorT reward_vec;
        stepBatch(pos_vec, items_vec, action_idx,
                  new_pos, new_items, active, reward_vec);

        // 4. Rollout
        FVectorT rollout_f = rollout(new_pos, active);

        // 5. SIMD scatter: write child state
        if (child_rel < tree_node_size_) {
            IVectorT child_g = scen_off + static_cast<int32_t>(child_rel);
            new_pos.scatter(position_.data(),              child_g);
            new_items.scatter(item_bits_.data(),           child_g);
            active.scatter(node_active_flags_.data(),      child_g);
            reward_vec.scatter(node_rewards_.data(),       child_g);
            IVectorT::fill(action_idx).scatter(
                node_curr_action_idxs_.data(), child_g);
        }

        // 6. Store rollout values (zero for terminal-parent lanes)
        FVectorT::select(active.as<FVectorT>(), rollout_f, 0.0f)
                 .to_array(rollout_values.data() + b);
    }

    // 7. Back-propagate (base class handles this)
    if (child_rel < tree_node_size_)
        backPropagateHomogenous(child_rel, rollout_values);
}
```

### 6.7 Wire Up main()

```cpp
#include "my_domain_static.hpp"

int main(int argc, char* argv[])
{
    // 1. Create the solver
    MyDomainVecQMDP solver(
        /* domain params */,
        /*tree_height=*/4,
        /*num_threads=*/8);

    // 2. Initial belief
    int start_pos = /* ... */;
    std::vector<float> initial_probs(num_items, 0.5f);

    // 3. Planning loop
    for (int step = 0; step < max_steps; ++step) {
        // Plan: run parallel belief tree search
        int best_action = solver.parallelBeliefTreeSearch(
            start_pos, initial_probs, /*max_iters=*/3000);

        // Execute best_action in the true environment
        // ... get observation, update belief ...

        // Update start_pos and initial_probs for next step
    }

    return 0;
}
```

See `examples/rock_sample/src/rock_sample_vecqmdp/main.cpp` for a complete
working example with command-line argument parsing, belief updates, Q-value
visualisation, and timing statistics.

---

## 7. Build Integration

### 7.1 Add to the Build System

1. **Create a directory** for your domain under `examples/`:

```
examples/
+-- my_domain/
    +-- CMakeLists.txt
    +-- src/
        +-- my_domain_vecqmdp/
            +-- my_domain_static.hpp
            +-- my_domain_static.cpp
            +-- main.cpp
```

2. **Add a `CMakeLists.txt`** linking against the VecQMDP static library:

```cmake
add_executable(Vec-QMDP_my_domain
    src/my_domain_vecqmdp/my_domain_static.cpp
    src/my_domain_vecqmdp/main.cpp
)

target_include_directories(Vec-QMDP_my_domain PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/vamp/src/impl
    src/
)

target_link_libraries(Vec-QMDP_my_domain PRIVATE vec_qmdp)
```

3. **Register it** in the top-level or parent `CMakeLists.txt`:

```cmake
add_subdirectory(examples/my_domain)
```

### 7.2 Build Commands

```bash
# Standard optimised build
./scripts/build_vec_qmdp.sh --opt=O3

# Or build directly
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make Vec-QMDP_my_domain -j$(nproc)

# Run
./build/bin/Vec-QMDP_my_domain
```

---

## 9. Further Reading

- [Architecture Overview](architecture.md) -- full module map and data flow diagrams
- [VecQMDP Design Document](vec_qmdp.md) -- detailed algorithm descriptions for
  UCB selection, back-propagation, early termination, and depth synchronisation
- [State Representation](state.md) -- core state module documentation
- Rock Sample example source:
  - Static VecQMDP: `examples/rock_sample/src/rock_sample_vecqmdp/rock_sample_static_vecqmdp.hpp` / `.cpp`
  - Dynamic VecQMDP: `examples/rock_sample/src/rock_sample_vecqmdp/rock_sample_dynamic_vecqmdp.hpp` / `.cpp`
  - Main entry point: `examples/rock_sample/src/rock_sample_vecqmdp/main.cpp`
  - DESPOT baseline (AoS): `examples/rock_sample/src/rock_sample_despot/rock_sample_despot.cpp`
- SIMD interface: `external/vamp/src/impl/vamp/vector/interface.hh`
