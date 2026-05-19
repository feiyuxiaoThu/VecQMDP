# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vec-QMDP is a high-performance, CPU-native parallel POMDP planner for autonomous driving. It uses SIMD-vectorized belief tree search (AVX2/NEON) with Data-Oriented Design to achieve real-time planning on standard CPUs. The C++ core exposes a Boost.Python shared library consumed by a Python planner module integrated with the nuPlan simulation framework. A self-contained RockSample benchmark demonstrates the search engine without external datasets.

## Build Commands

### Prerequisites
- conda environment `vec_qmdp` (Python 3.9) must be activated before building
- Eigen3 installed via conda (`conda install -c conda-forge eigen`)
- Boost >= 1.74 with Python/NumPy components
- GEOS static library must be built once: `cd external/geos && bash geos_with_simd.sh && cd build-avx && make install && cd ../../..`

### Build the shared library (nuPlan evaluation, main branch)
```bash
bash scripts/build_vec_qmdp.sh --opt=O3           # standard release
bash scripts/build_vec_qmdp.sh --clean --opt=O3    # clean build
bash scripts/build_vec_qmdp.sh --asan              # AddressSanitizer
bash scripts/build_vec_qmdp.sh --perf              # performance profiling
bash scripts/build_vec_qmdp.sh -h                  # all options
```
Output: `build/src/vec_qmdp_closed_planner.so` → copied to `python_planner/`.

### Build RockSample benchmark
```bash
bash examples/rock_sample/build_vec_qmdp.sh              # dynamic backend (default)
bash examples/rock_sample/build_vec_qmdp.sh --static-solver  # static backend
```
Before building RockSample, uncomment `#define ENABLE_HOMOGENOUS_SEARCH` in `include/planning/vec_qmdp_dynamic.hpp` or `vec_qmdp_static.hpp`. For nuPlan, this macro must remain commented out (heterogeneous search is required for belief diversity).

### Run nuPlan closed-loop evaluation
```bash
python scripts/simulation/sim_planner.py --split val14_split --challenge closed_loop_nonreactive_agents --worker sequential
python scripts/simulation/sim_planner.py --split test_hard --challenge closed_loop_nonreactive_agents --worker ray_distributed
```

### Run RockSample benchmark
```bash
bash examples/rock_sample/benchmark/run_vecqmdp.sh     # 20 simulations
bash examples/rock_sample/benchmark/run_despot.sh       # DESPOT baseline
python3 examples/rock_sample/benchmark/analyze.py       # compare results
```

### Install QCNet predictor
```bash
cd python_planner/qcpredictor && pip install -e . && cd ../..
```

## Architecture

### C++ Core (include/ + src/)

Four modules built as OBJECT libraries, composed into a static lib (`vec_qmdp`) and a Python shared lib (`vec_qmdp_closed_planner`):

- **utils/** — foundational infrastructure: `params.hpp` (compile-time constants), `global_utils` (SIMD type aliases, ThreadPool, perf profiler, LOG macros, thread-local workspaces), `aligned_allocator` (32-byte AVX2 alignment), `math_utils` (SIMD angle/distance helpers), `geometry_utils` (Point/Polygon, Cartesian↔Frenet), `path_utils` (SIMD nearest-point search), `map_utils` (HD map bridge, GIL management, lane topology cache), `occupancy_map` (GEOS PreparedPolygon batched point-in-polygon)
- **core/** — `state` (EgoState, ObservedExoState, SIMD Frenet conversion, batched STRtree construction), `net_belief` (three-phase agent filtering: spatial → trajectory intersection → collision exclusion)
- **planning/** — search engine and autonomous driving domain. Key classes:
  - `VecQMDP<A,H,S>` (static) — pre-allocated balanced tree, O(A^H) memory, best for small action spaces (AD: A=9, H=10)
  - `DynVecQMDP` (dynamic) — on-demand chunked SoA pool via `DynSoAPool`, O(visited_nodes) memory, best for large action spaces (RockSample: A=13, H=90)
  - Both share identical algorithms: `exploreNodes`/`exploreNodesVote` (SIMD UCB), `backPropagate` (running-mean Q-update), `backupDepth`, early termination
  - `VecQMDP_AD` — derives from `VecQMDP`, implements AD-specific action constraints and forward simulation
  - `ContextQMDP` — forward simulation engine (IDM + Stanley steering + collision checking via STRtree+SAT)
  - `TrajectoryOptimization` — proposal generation, LQR tracking via `Tracker`, cross-scenario evaluation, emergency brake
  - `QMDPTrajectoryPlanner` — top-level entry point orchestrating search → action selection → trajectory optimization
  - `reward_functions` — template `<T,U>` SIMD-templated penalty functions
- **collision/** — `STRtree` with SoA AABB layout, 8-wide SIMD intersection queries, Highway VQSort

### Python Layer (python_planner/)

- `vec_qmdp_planner_module.py` — entry point importing the compiled `.so`, managing aligned NumPy buffers, orchestrating QCNet predictions and C++ planner calls
- `qcpredictor/` — PyTorch-based agent behavior prediction models (QCNet)
- `utils/` — Python utilities for data conversion and logging

### Build Graph
```
CMakeLists.txt
  → src/CMakeLists.txt
    → OBJECT libs: vec_qmdp_{core,planning,utils,collision}
    → STATIC lib: vec_qmdp (all object libs)
    → SHARED lib: vec_qmdp_closed_planner (all object libs + Boost.Python + GEOS static)
    → copied to python_planner/vec_qmdp_closed_planner.so
```

## Key Design Patterns

- **Structure of Arrays (SoA):** All per-scenario data stored column-major for SIMD contiguity. `visit_counts[scenario][node]`, not `nodes[node].visit_count[scenario]`.
- **Aligned Allocation:** C++ uses `AlignedAllocator<T, 32>`, Python uses `aligned_array()` + `_alignment_buffers` to prevent GC from dropping buffers.
- **SIMD Type Hierarchy:** `FVectorT_qmdp = vamp::FloatVector<NUM_SCENARIOS_PER_THREAD, 1>` for tree search (8-wide); `FVectorT_traj = vamp::FloatVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, LATERAL_OFFSETS_NUM>` for trajectory optimization.
- **Template-heavy reward functions:** `template<typename T, typename U>` + `inline` for compiler auto-vectorization without function-call overhead.
- **Worker cloning:** `makeWorkerInstance()` deep-copies search state per thread; no shared mutable state, no locks during search.
- **Thread-local workspaces:** Pre-sized structs (`StepBatchWorkspace`, `CrossScenarioEvaluationWorkspace`) constructed once per thread, logically reset between frames.
- **GIL management:** `ScopedGILAcquire`/`ScopedGILRelease` RAII wrappers for C++/Python interop.

## Configuration

Planning parameters are compile-time constants in `include/utils/params.hpp`:
- `MAX_PLANNING_TIME` — time budget per frame (varies by split: 8.0s for val14/test_hard, 0.5s for test14_random)
- `NUM_THREADS` (default 8), `NUM_SCENARIOS_PER_THREAD` (default 8, must be multiple of 8)
- `NUM_ACTIONS` (9), `TREE_HEIGHT` (4), `TIME_STEP` (0.2s), `STEP_TIME` (2.0s)
- Key reward weights: `CRASH_PENALTY` (-1000), `MOVEMENT_PENALTY` (5.0), `ACTION_PENALTY` (-2.0)
- Changing any of these requires rebuilding the shared library.

## Environment Variables

Required for nuPlan evaluation (add to `~/.bashrc`):
```bash
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
export NUPLAN_HYDRA_CONFIG_PATH="$HOME/nuplan-devkit/nuplan/planning/script/config"
export NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="$HOME/VecQMDP/python_planner/qcpredictor:$HOME/nuplan-devkit:$HOME/VecQMDP/python_planner:$HOME/VecQMDP"
```

## Branches

- `main` — non-reactive evaluation (other agents follow log replay)
- `reactive` — reactive evaluation (other agents respond to ego); requires rebuilding after checkout

## External Dependencies

| Library | Location | Linking | Purpose |
|---------|----------|---------|---------|
| Eigen | System/conda | Header-only | LQR solver, matrix ops |
| Boost.Python | System | Shared | Python↔C++ bindings |
| GEOS | `external/geos/` | Static | Geometry, PreparedPolygon containment |
| Highway | `external/highway/` | Header-only | Portable SIMD abstraction, VQSort |
| x86-simd-sort | `external/x86-simd-sort-main/` | Header-only | AVX-512/AVX2 sorting |
| VAMP | `external/vamp/` | Header-only | FloatVector/IntVector SIMD templates |
| DESPOT | `external/despot/` | Static (examples only) | POMDP solver baseline for RockSample |

## Documentation

Detailed design docs in `docs/`: `architecture.md` (full module layout + dependency graph), `vec_qmdp.md` (search engine design), `tutorial.md` (implementing custom POMDP domains), `context_qmdp.md`, `net_belief.md`, `qmdp_trajectory_planner.md`, `trajectory_optimization.md`, `reward_function.md`, `state.md`, `STRtree.md`, `utils.md`.
