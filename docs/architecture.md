# VecQMDP --- Project Architecture

## 1. Executive Summary

VecQMDP is a high-performance, SIMD-vectorised QMDP (Quick-MDP) motion planner for autonomous driving. It solves online POMDPs by combining:

- **Belief tree search** --- multi-threaded, SIMD-accelerated (AVX2/NEON) UCB-based search across multiple probabilistic scenarios.
- **Trajectory optimisation** --- vectorised proposal generation, LQR tracking, and cross-scenario reward evaluation.
- **Python integration** --- a Boost.Python shared library (`vec_qmdp_closed_planner.so`) consumed by the Python planner module.

The planner receives perception data and prediction beliefs, evaluates candidate driving actions under environmental uncertainty, and outputs a smooth, collision-free trajectory in real-time.

---

## 2. Repository Layout

```
VecQMDP/
├── include/                    # Public C++ headers
│   ├── collision/              # Spatial indexing (STRtree)
│   ├── core/                   # State representation and belief management
│   ├── planning/               # Planners, search, reward, tracker, optimiser
│   └── utils/                  # Shared infrastructure (SIMD, math, map, params)
├── src/                        # C++ implementations (.cpp mirrors include/)
│   ├── collision/
│   ├── core/
│   ├── planning/
│   └── utils/
├── python_planner/             # Python-side planner (imports compiled .so)
│   ├── vec_qmdp_planner_module.py   # Entry point
│   ├── controller/             # Python controllers
│   ├── qcpredictor/            # Neural-net prediction models
│   └── utils/                  # Python utilities
├── examples/rock_sample/       # Rock Sample POMDP example (static & dynamic backends)
├── external/                   # Vendored dependencies
│   ├── despot/                 # DESPOT POMDP solver (comparison baseline)
│   ├── geos/                   # GEOS geometry library (statically linked)
│   ├── highway/                # Google Highway (portable SIMD abstraction)
│   ├── vamp/                   # Vector-accelerated motion planning primitives
│   └── x86-simd-sort-main/    # SIMD sorting
├── scripts/
│   └── build_vec_qmdp.sh      # Canonical build script
├── docs/                       # Design documents
├── tests/                      # Unit tests
└── CMakeLists.txt              # Top-level build configuration
```

---

## 3. Build Architecture

```
CMakeLists.txt (top-level)
  ├── Dependency discovery: Eigen, Boost.Python, NumPy, GEOS
  ├── SIMD flags: -mavx2, -mfma, -march=native
  └── src/CMakeLists.txt
        ├── OBJECT lib: vec_qmdp_utils  (utils/*.cpp)
        ├── STATIC lib: vec_qmdp        (all src/*.cpp + vec_qmdp_utils)
        └── SHARED lib: vec_qmdp_closed_planner  (Python .so wrapper)
              ├── Links: vec_qmdp (static), Boost.Python, GEOS
              └── Copied to: python_planner/vec_qmdp_closed_planner.so
```

**Build commands:**
- Standard: `./scripts/build_vec_qmdp.sh --opt=O3`
- Clean build: `./scripts/build_vec_qmdp.sh --clean --opt=O3`
- With AddressSanitizer: `./scripts/build_vec_qmdp.sh --asan`

---

## 4. Module Dependency Graph

```
                     ┌──────────────────────────────────────┐
                     │   vec_qmdp_closed_planner (.so)      │
                     │   Boost.Python wrapper               │
                     └──────────────┬───────────────────────┘
                                    │ links
                     ┌──────────────▼───────────────────────┐
                     │          vec_qmdp (static lib)       │
                     └──┬─────┬─────┬──────┬────────────────┘
                        │     │     │      │
           ┌────────────▼─┐  ┌▼─────▼─┐  ┌─▼───────────────┐
           │  collision/  │  │ core/  │  │    planning/    │
           │  STRtree     │  │ State  │  │ QMDPTrajPlanner │
           └──────────────┘  │ NetBel │  │ ContextQMDP     │
                             └────────┘  │ VecQMDP_AD      │
                                         │ VecQMDP_Static  │
                                         │ VecQMDP_Dynamic │
                                         │ TrajOptimization│
                                         │ RewardFunctions │
                                         │ Tracker         │
                                         └───────┬─────────┘
                                                  │ depends on
                                         ┌───────▼──────────┐
                                         │    utils/        │
                                         │ params           │
                                         │ global_utils     │
                                         │ math_utils       │
                                         │ geometry_utils   │
                                         │ path_utils       │
                                         │ map_utils        │
                                         │ occupancy_map    │
                                         │ aligned_allocator│
                                         └──────────────────┘
```

---

## 5. Module Descriptions

### 5.1 Utils Layer (`include/utils/`, `src/utils/`)

The foundational infrastructure shared across all modules. Built as an OBJECT library (`vec_qmdp_utils`).

| File | Purpose |
|------|---------|
| `params.hpp` | Compile-time constants organised by category: vehicle geometry, SIMD lane widths, planning horizons, reward weights, timing budgets, early termination thresholds |
| `global_utils.hpp/cpp` | SIMD type aliases (`FVectorT_qmdp`, `IVectorT_qmdp`), `ThreadPool` with `std::future`-based interface, `PerfProfiler` with thread-local accumulation, `LOG()` macros with compile-time elimination, thread-local pre-sized workspaces (`StepBatchWorkspace`, `CrossScenarioEvaluationWorkspace`, etc.) |
| `aligned_allocator.hpp` | `AlignedAllocator<T, N>` --- STL-compatible allocator enforcing N-byte alignment (default 32B for AVX2). Uses `posix_memalign`. Powers `AlignedVectorFloat`/`AlignedVectorInt` throughout the codebase |
| `math_utils.hpp` | Scalar + SIMD helpers: `NormalizeAngle`, `SquaredDistance`, `IsAngleWithinThreshold<T>` (branchless SIMD cone test), `clipEdgeBatch<T>` (Liang-Barsky AABB clipping) |
| `geometry_utils.hpp/cpp` | `Point`, `Polygon` structs; coordinate transformations between Cartesian and Frenet frames |
| `path_utils.hpp/cpp` | `Path` class with SIMD-batched nearest-point search: two-phase anchor-segment coarse search + binary subdivision, incremental O(1) sequential updates, and sub-index interpolation. Stores per-point arrays (xs, ys, thetas, kappas) at 0.2 m spacing |
| `map_utils.hpp/cpp` | Boost.Python bridge to HD map; `ScopedGILAcquire`/`ScopedGILRelease` RAII wrappers; lane topology caching in `unordered_map` (successors, predecessors, neighbors, smooth paths, traffic lights); reference-path construction via `GetEgoRefPaths()` |
| `occupancy_map.hpp/cpp` | Wraps GEOS `PreparedPolygon` for fast drivable-area containment. Incremental polygon update via differential insert/delete. Batched point-in-polygon producing SIMD bitmasks (non-drivable, oncoming traffic, intersection, non-route) |

### 5.2 Core Layer (`include/core/`, `src/core/`)

| File | Purpose |
|------|---------|
| `state.hpp/cpp` | `EgoState` --- position, velocity, acceleration, heading, steering, junction proximity. `ObservedExoState` --- container for surrounding agents with Python/NumPy data import. `ExoStates` --- static SIMD-accelerated methods: `GetFrenetPointsBatch` (Cartesian---Frenet with velocity reversal/stopping logic), `buildSTRtreesFrenetBatch` (per-timestep spatial index construction with projected bounding boxes and safety margins) |
| `net_belief.hpp/cpp` | `NetBelief` --- manages sampled belief scenarios (joint probability over future exo-agent trajectories). Three-phase agent filtering: (1) Frenet-based spatial filtering by lateral distance, (2) future trajectory intersection via ML prediction mode with SIMD-batched checks, (3) collision exclusion and static-motion inference. Provides scenario-indexed access to filtered prediction data |

### 5.3 Planning Layer (`include/planning/`, `src/planning/`)

| File | Purpose |
|------|---------|
| `qmdp_trajectory_planner.hpp/cpp` | **Entry point.** `QMDPTrajectoryPlanner::planTrajectory()` orchestrates per-frame planning: receives `EgoState` + `Belief` + reference paths --- triggers parallel belief tree search via `VecQMDP_AD` --- selects best action with hysteresis preference bonus --- triggers parallel trajectory optimisation --- returns best trajectory. Manages `ThreadPool` and Python GIL for hybrid C++/Python execution |
| `vec_qmdp_static.hpp/cpp` | `VecQMDP<A,H,S>` --- abstract base class for SIMD-vectorised belief tree search with **pre-allocated** implicit balanced tree (O(A^H -- S) memory). Core algorithms: `exploreNodes` (SIMD-parallel UCB with precomputed coefficient tables and gather instructions), `exploreNodesVote` (majority-vote depth synchronisation with re-alignment penalty), `backPropagate` (running-mean Q-update with persistent max-backup and pruning), `backupDepth` (min/max depth-gap tracking), early termination on Q-value convergence |
| `vec_qmdp_dynamic.hpp/cpp` | `DynVecQMDP` --- alternative backend with **dynamic chunked SoA pool** (`DynSoAPool`). Nodes allocated on demand via `allocateChildBlock()`; O(visited_nodes) memory. Explicit `parent_idx`/`child_base_idx` topology. Same algorithmic building blocks as static variant. Best for large action spaces or deep trees (e.g., Rock Sample: A=13, H=90) |
| `dynamic_soa_pool.hpp` | `DynSoAPool` --- thread-private lock-free memory pool. Fixed-size chunks of 32768 slots (2^15) with bit-shift index decomposition. 64-byte-aligned SoA layout per chunk. Pre-warms 16 chunks (~24 MB). Logical reset without heap traffic |
| `vec_qmdp_ad.hpp/cpp` | `VecQMDP_AD` --- autonomous-driving domain deriving from `VecQMDP`. Implements `computeNodeCandidateActions` (constrained by lane adjacency via `STATIC_TRANSITION_TABLE`), `expandNodesBatch` (SIMD forward simulation via `ContextQMDP::StepBatch`), `makeWorkerInstance` (deep copy for multi-threaded search). Action encoding: 3 paths -- 3 lateral offsets = 9 actions |
| `context_qmdp.hpp/cpp` | `ContextQMDP` --- forward simulation engine. `StepBatch`: 2.0 s horizon at 0.2 s steps --- reference line identification --- behavioral decision (LF/LC) --- lead/follow vehicle search --- `LCAllowanceCheckBatch` feasibility --- IDM acceleration + Stanley steering --- reward accumulation. `generateProposalTrajectoryLF/LCBatch`: 0.1 s resolution proposal generation. `HasCollisionBatch`: two-stage broad-phase STRtree + narrow-phase SAT. `crossScenarioEvaluationBatch`: aggregated penalties (drivable area, speed, TTC) |
| `trajectory_optimization.hpp/cpp` | `TrajectoryOptimization` --- three-stage vectorised pipeline: (1) `importanceSampleScenarios` with weighted multi-modal diversity, (2) proposal generation with lateral offsets --- `Tracker::track()` LQR refinement, (3) `crossScenarioEvaluationBatch` reward scoring --- best trajectory selection. Safety governor: `checkAndGenerateEmergencyBrake` monitors TTC and triggers PD-controller hard braking |
| `reward_functions.hpp/cpp` | Inline SIMD-templated penalty functions: `movementPenalty` (speed deviation), `goalPenalty` (distance to Frenet goal), `actionPenalty` (oscillation suppression), `crashPenalty` (with `CollisionType` classification and no-at-fault logic), `lateralOffsetGuidance`, `directionCompliance`, `nonDrivableAreaPenalty`, `onNonRoute`, curvature-scaled penalties. All accept `<T, U>` template parameters for 8/16-wide SIMD evaluation |
| `tracker.hpp/cpp` | `Tracker` --- LQR controller for the kinematic bicycle model. Converts geometric proposals into dynamically feasible trajectories with velocity/acceleration profiles. Solves discrete-time algebraic Riccati equation for optimal gain matrix |

### 5.4 Collision Layer (`include/collision/`, `src/collision/`)

| File | Purpose |
|------|---------|
| `STRtree.hpp/cpp` | SIMD-accelerated R-tree (Sort-Tile-Recursive). SoA layout for cache-friendly AABB storage. Uses Google Highway `VQSort` for parallelised sorting and `hmin`/`hmax` for parent AABB computation. 8-wide SIMD intersection queries via `_mm256_movemask_ps` + `__builtin_ctz` for branchless traversal. Dual-subtree layout handles up to 96 leaf nodes (64+32) at depth 2. Query results sorted via unrolled insertion sort or `std::sort` |

### 5.5 Python Layer (`python_planner/`)

| File | Purpose |
|------|---------|
| `vec_qmdp_planner_module.py` | High-level Python planner entry point. Imports `vec_qmdp_closed_planner` (.so), manages aligned NumPy buffers via `aligned_array()` with `_alignment_buffers` GC protection, orchestrates neural-net predictions and C++ planner calls |
| `controller/` | Python-side controllers (fallback/override logic) |
| `qcpredictor/` | PyTorch prediction models for agent behaviour forecasting |
| `utils/` | Python utilities (data conversion, logging) |

---

## 6. Data Flow --- Single Planning Frame

```
  Python Module                     C++ Planner
  ─────────────                     ───────────
  1. Receive perception
     + HD map update
         │
  2. Run qcpredictor ──────► Belief scenarios (NumPy arrays)
         │
  3. Call planTrajectory() ──────────────────────────────►
                           4.QMDPTrajectoryPlanner::planTrajector()
                              │
                              ├─ 5. Build NetBelief
                              │     └─ filterDiscardedAgents()
                              │        (Frenet spatial + trajectory
                              │         intersection + collision exclusion)
                              │
                              ├─ 6. VecQMDP_AD::beliefTreeSearch()
                              │     [multi-threaded, per-worker clone]
                              │     ├─ exploreNodesVote()  (SIMD UCB + depth sync)
                              │     ├─ expandNodesBatch()
                              │     │   └─ ContextQMDP::StepBatch()
                              │     │      (IDM + Stanley + collision)
                              │     ├─ backPropagate()     (Q-update + pruning)
                              │     ├─ backupDepth()       (depth-gap tracking)
                              │     └─ early termination check
                              │
                              ├─ 7. getBestAction()
                              │     (aggregate Q-values + hysteresis bonus)
                              │
                              ├─ 8. TrajectoryOptimisation::optimize()
                              │     ├─ importanceSampleScenarios()
                              │     ├─ generateProposalTrajectory{LF,LC}Batch()
                              │     │   (lateral offsets, 0.1s resolution)
                              │     ├─ Tracker::track()    (LQR)
                              │     ├─ crossScenarioEvaluationBatch()
                              │     │   (STRtree + SAT + reward functions)
                              │     ├─ checkAndGenerateEmergencyBrake()
                              │     └─ Select best trajectory
                              │
  9. Receive trajectory ◄────────────── Return best trajectory
```

---

## 7. SIMD Architecture

VecQMDP processes **S scenarios simultaneously** using SIMD lanes:

- **AVX2 (x86):** 8 -- float32 per register --- scenario batches of 8
- **NEON (ARM):** 4 -- float32 per register --- scenario batches of 4
- **Highway abstraction:** `external/highway/` provides portable SIMD via `hwy::` namespace
- **VAMP library:** `external/vamp/` provides `FloatVector<Width, Rows>` / `IntVector<Width, Rows>` templates

### SIMD Type Hierarchy

```cpp
// Base width (typically 8 for AVX2)
constexpr int FloatVectorWidth = vamp::FloatVectorWidth;

// QMDP tree search (8 scenarios -- 1 row)
using FVectorT_qmdp = vamp::FloatVector<NUM_SCENARIOS_PER_THREAD, 1>;
using IVectorT_qmdp = vamp::IntVector<NUM_SCENARIOS_PER_THREAD, 1>;

// Trajectory optimisation (8 scenarios -- 9 lateral offsets)
using FVectorT_traj = vamp::FloatVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, LATERAL_OFFSETS_NUM>;
```

### Key SIMD-ised Operations

| Operation | Module | Technique |
|-----------|--------|-----------|
| UCB score computation | `vec_qmdp_static` | Precomputed `ucb_coeff_table_` + `inv_sqrt_table_`, `_mm256_i32gather`, fused multiply-add |
| Back-propagation | `vec_qmdp_static/dynamic` | Gather/scatter via `hwy::Gather`, masked count update |
| Reward evaluation | `reward_functions` | Template `<T, U>` over 8-wide float/int vectors |
| STRtree intersection | `STRtree` | 8-wide `_mm256_cmp_ps` + `_mm256_movemask_ps` + `__builtin_ctz` |
| Nearest-point search | `path_utils` | Anchor-segment SIMD distance, binary subdivision |
| Frenet conversion | `state` | Batch `GetFrenetPointsBatch` via `vamp::FloatVectorWidth` |
| Collision (SAT) | `context_qmdp` | SIMD corner computation + separating axis tests |

The `AlignedAllocator<T, 32>` ensures all SoA arrays are 32-byte aligned for AVX2.

---

## 8. VecQMDP Search --- Static vs Dynamic

### Static (`VecQMDP<A,H,S>`)
- Full balanced tree pre-allocated: $\sum_{k=0}^{H} A^k$ nodes per scenario
- Parent index computed implicitly: `parent = (child - 1) / A`
- Best for **small action spaces, shallow depth** (AD: A=9, H=10)
- Example: `VecQMDP_AD` with 9 actions, tree height 10 --- 7381 nodes/scenario

### Dynamic (`DynVecQMDP`)
- `DynSoAPool` allocates nodes in fixed-size chunks (32768 slots) on demand
- Explicit `parent_idx`/`child_base_idx` stored per node
- Best for **large action spaces or deep trees** (Rock Sample: A=13, H=90)
- Memory: O(max_iters -- A) vs O(A^H) --- e.g., ~30K nodes vs 10^100

### Shared Algorithms
Both backends share identical algorithmic building blocks:

| Algorithm | Description |
|-----------|-------------|
| `exploreNodes` | Standard SIMD-parallel UCB descent |
| `exploreNodesVote` | Three-phase majority-vote depth synchronisation |
| `exploreNodesHomogenous` | Single-path (scenario-0 only) for cheap domains |
| `backPropagate` | Running-mean Q-update with persistent max-backup and pruning |
| `backupDepth` | Min/max reachable depth tracking for depth-sync penalty |
| Early termination | Q-value convergence + stable best-action check |
| Parallel search | Worker cloning via `makeWorkerInstance()`, `ThreadPool` dispatch |

---

## 9. Key Design Patterns

### 9.1 Structure of Arrays (SoA)
All per-scenario data is stored column-major for SIMD. Example: `visit_counts[scenario][node]`, not `nodes[node].visit_count[scenario]`. This ensures contiguous memory access when processing all scenarios for a single node.

### 9.2 Aligned Allocation
- **C++:** `AlignedAllocator<T, 32>` for all SIMD containers
- **Python:** `aligned_array()` helper + `_alignment_buffers` module-global to prevent GC from dropping underlying buffers
- Prevents undefined behaviour from unaligned SIMD loads/stores

### 9.3 Static Core + Shared Wrapper
Native code is composed into a single static lib (`vec_qmdp`) and a Python-friendly shared lib (`vec_qmdp_closed_planner`). C++ symbols stay private; only Boost.Python bindings are exported.

### 9.4 Template-Heavy Reward Functions
`reward_functions.hpp` uses `template<typename T, typename U>` + `inline` to allow the compiler to auto-vectorise penalty loops without function-call overhead. `T` represents a SIMD float vector, `U` a SIMD int/mask vector.

### 9.5 Worker Cloning for Multi-Threading
`makeWorkerInstance()` creates deep copies of the search state for each thread. Each worker operates on its own tree with its own scenario set --- no shared mutable state, no locks during the search phase.

### 9.6 Thread-Local Workspaces
Pre-sized workspace structs (`StepBatchWorkspace`, `CrossScenarioEvaluationWorkspace`, etc.) are constructed once per thread and logically reset between planning cycles. This eliminates dynamic allocation during the hot search loop.

### 9.7 Python GIL Management
`ScopedGILAcquire`/`ScopedGILRelease` RAII wrappers ensure thread-safe Python interop. The main thread releases the GIL before spawning C++ workers; each worker acquires it only when calling Python map APIs.

---

## 10. External Dependencies

| Library | Location | Linking | Purpose |
|---------|----------|---------|---------|
| Eigen | System | Header-only | Linear algebra (LQR solver, matrix operations) |
| Boost.Python | System | Shared | Python --- C++ bindings |
| GEOS | `external/geos/` | Static | Geometry operations, `PreparedPolygon` containment |
| Highway | `external/highway/` | Header-only | Portable SIMD abstraction, `VQSort` sorting |
| x86-simd-sort | `external/x86-simd-sort-main/` | Header-only | AVX-512/AVX2 sorting |
| VAMP | `external/vamp/` | Header-only | `FloatVector`/`IntVector` SIMD type templates |
| NumPy | Python | Runtime | Array interface for Python planner |

---

## 11. Key Functions Reference

### 11.1 Planning Pipeline

| Function | Location | Description |
|----------|----------|-------------|
| `QMDPTrajectoryPlanner::planTrajectory()` | `planning/qmdp_trajectory_planner` | Main entry point: orchestrates search --- action selection --- trajectory optimisation |
| `VecQMDP_AD::beliefTreeSearch()` | `planning/vec_qmdp_ad` | Runs the full belief tree search loop; returns elapsed time (--s) |
| `VecQMDP_AD::sampleScenarios()` | `planning/vec_qmdp_ad` | Populates state buffers from a NetBelief sample |
| `VecQMDP_AD::expandNodesBatch()` | `planning/vec_qmdp_ad` | SIMD forward simulation + rollout for a batch of selected nodes |
| `TrajectoryOptimization::optimize()` | `planning/trajectory_optimization` | Full trajectory generation and refinement pipeline |

### 11.2 Tree Search Core

| Function | Location | Description |
|----------|----------|-------------|
| `exploreNodesVote()` | `vec_qmdp_static` / `vec_qmdp_dynamic` | Majority-vote depth-synchronised UCB traversal |
| `exploreNodes()` | `vec_qmdp_static` / `vec_qmdp_dynamic` | Standard SIMD UCB traversal |
| `backPropagate()` | `vec_qmdp_static` / `vec_qmdp_dynamic` | Per-scenario value propagation with max-backup |
| `backupDepth()` | `vec_qmdp_static` / `vec_qmdp_dynamic` | Min/max reachable depth update |
| `allocateChildBlock()` | `vec_qmdp_dynamic` | Dynamic node allocation in `DynSoAPool` |

### 11.3 Forward Simulation

| Function | Location | Description |
|----------|----------|-------------|
| `ContextQMDP::StepBatch()` | `planning/context_qmdp` | State transition: IDM + Stanley + collision for tree search |
| `ContextQMDP::generateProposalTrajectoryLFBatch()` | `planning/context_qmdp` | Lane-following proposal at 0.1s resolution |
| `ContextQMDP::generateProposalTrajectoryLCBatch()` | `planning/context_qmdp` | Lane-change proposal at 0.1s resolution |
| `ContextQMDP::HasCollisionBatch()` | `planning/context_qmdp` | Two-stage: STRtree broad-phase + SAT narrow-phase |
| `ContextQMDP::crossScenarioEvaluationBatch()` | `planning/context_qmdp` | Multi-scenario trajectory quality evaluation |
| `ContextQMDP::LCAllowanceCheckBatch()` | `planning/context_qmdp` | Lane-change feasibility (gaps, heading, IDM safety) |

### 11.4 Belief and State Management

| Function | Location | Description |
|----------|----------|-------------|
| `NetBelief::filterDiscardedAgents()` | `core/net_belief` | Three-phase agent filtering (spatial + temporal + collision) |
| `ExoStates::GetFrenetPointsBatch()` | `core/state` | SIMD Cartesian---Frenet conversion with velocity adjustment |
| `ExoStates::buildSTRtreesFrenetBatch()` | `core/state` | Per-timestep STRtree construction with projected bounding boxes |

### 11.5 Spatial Indexing

| Function | Location | Description |
|----------|----------|-------------|
| `STRtree::build()` | `collision/STRtree` | STR sort + vectorised parent AABB computation |
| `STRtree::queryBatch()` | `collision/STRtree` | 8-wide SIMD intersection query with branchless traversal |

### 11.6 Path and Map

| Function | Location | Description |
|----------|----------|-------------|
| `Path::NearestBatch()` | `utils/path_utils` | SIMD nearest-point search (anchor coarse + binary subdivision) |
| `Path::InterpolatedNearestBatch()` | `utils/path_utils` | Sub-index interpolation for smooth Frenet conversion |
| `MapUtils::GetEgoRefPaths()` | `utils/map_utils` | Build reference paths from HD map topology |
| `OccupancyMap::ContainsPointsInDrivableAreaBatch()` | `utils/occupancy_map` | Batched point-in-polygon with GEOS PreparedPolygon |

---

## 12. File Cross-Reference

| Header | Source | Module | Documentation |
|--------|--------|--------|---------------|
| `collision/STRtree.hpp` | `collision/STRtree.cpp` | STRtree | [STRtree.md](STRtree.md) |
| `core/net_belief.hpp` | `core/net_belief.cpp` | NetBelief | [net_belief.md](net_belief.md) |
| `core/state.hpp` | `core/state.cpp` | State | [state.md](state.md) |
| `planning/context_qmdp.hpp` | `planning/context_qmdp.cpp` | ContextQMDP | [context_qmdp.md](context_qmdp.md) |
| `planning/qmdp_trajectory_planner.hpp` | `planning/qmdp_trajectory_planner.cpp` | Planner | [qmdp_trajectory_planner.md](qmdp_trajectory_planner.md) |
| `planning/reward_functions.hpp` | `planning/reward_functions.cpp` | Rewards | [reward_function.md](reward_function.md) |
| `planning/tracker.hpp` | `planning/tracker.cpp` | Tracker | --- |
| `planning/trajectory_optimization.hpp` | `planning/trajectory_optimization.cpp` | TrajOpt | [trajectory_optimization.md](trajectory_optimization.md) |
| `planning/vec_qmdp_static.hpp` | `planning/vec_qmdp_static.cpp` | VecQMDP Static | [vec_qmdp.md](vec_qmdp.md) |
| `planning/vec_qmdp_dynamic.hpp` | `planning/vec_qmdp_dynamic.cpp` | VecQMDP Dynamic | [vec_qmdp.md](vec_qmdp.md) |
| `planning/dynamic_soa_pool.hpp` | --- (header-only) | DynSoAPool | [vec_qmdp.md](vec_qmdp.md) |
| `planning/vec_qmdp_ad.hpp` | `planning/vec_qmdp_ad.cpp` | VecQMDP_AD | [vec_qmdp.md](vec_qmdp.md) |
| `planning/vec_qmdp_closed_planner.hpp` | `planning/vec_qmdp_closed_planner.cpp` | PythonWrapper | --- |
| `utils/aligned_allocator.hpp` | --- (header-only) | AlignedAlloc | [utils.md](utils.md) |
| `utils/geometry_utils.hpp` | `utils/geometry_utils.cpp` | Geometry | [utils.md](utils.md) |
| `utils/global_utils.hpp` | `utils/global_utils.cpp` | Globals | [utils.md](utils.md) |
| `utils/map_utils.hpp` | `utils/map_utils.cpp` | Map | [utils.md](utils.md) |
| `utils/math_utils.hpp` | `utils/math_utils.cpp` | Math | [utils.md](utils.md) |
| `utils/occupancy_map.hpp` | `utils/occupancy_map.cpp` | OccMap | [utils.md](utils.md) |
| `utils/params.hpp` | `utils/params.cpp` | Params | [utils.md](utils.md) |
| `utils/path_utils.hpp` | `utils/path_utils.cpp` | PathUtils | [utils.md](utils.md) |

---

## 13. Configuration Parameters Summary

### 13.1 Planner Core

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_ACTIONS` | 9 | 3 paths -- 3 lateral offsets |
| `TREE_HEIGHT` | 4 | Planning tree depth |
| `TIME_STEP` | 0.2 s | Simulation time step |
| `STEP_TIME` | 2.0 s | Action duration per tree level |
| `NUM_SCENARIOS` | 64 | Total belief scenarios |
| `NUM_THREADS` | 8 | Worker thread count |
| `NUM_SCENARIOS_PER_THREAD` | 8 | Scenarios per worker |
| `MAX_SIM_VEHICLES` | 96 | Max simulated vehicles (aligned to 8) |
| `TREE_NODE_SIZE` | 7381 | Pre-computed total nodes |

### 13.2 Trajectory Optimisation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PROPOSAL_TRAJECTORY_SIZE` | 41 steps | 8.0 s at 0.2 s intervals |
| `LATERAL_OFFSETS_NUM` | 9 | Lateral offset candidates |
| `PATH_POINT_INTERVAL` | 0.2 m | Reference path discretisation |

### 13.3 Key Reward Weights

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CRASH_PENALTY` | -1000 | Collision penalty |
| `MOVEMENT_PENALTY` | 5.0 | Speed deviation weight |
| `ACTION_PENALTY` | -2.0 | Action switching penalty |
| `PRUNED_THRESHOLD` | -1000 | Return threshold for pruning |
