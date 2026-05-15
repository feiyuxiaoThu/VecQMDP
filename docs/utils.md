# Utils Module -- Design Document

## Overview

The `utils` module provides the foundational infrastructure shared across the entire VecQMDP planner. It is built as an OBJECT library (`vec_qmdp_utils`) and linked into both the static core (`vec_qmdp`) and the Python-facing shared library (`vec_qmdp_closed_planner`).

| File | Header | Purpose |
|---|---|---|
| `params` | `include/utils/params.hpp` | Compile-time constants and tuning parameters |
| `global_utils` | `include/utils/global_utils.hpp` | SIMD type aliases, thread pool, logging, performance profiling, thread-local workspaces |
| `aligned_allocator` | `include/utils/aligned_allocator.hpp` | STL-compatible aligned memory allocator |
| `math_utils` | `include/utils/math_utils.hpp` | Scalar and SIMD math helpers (distance, angle, clipping) |
| `geometry_utils` | `include/utils/geometry_utils.hpp` | Basic geometry primitives (`Point`, `Polygon`) |
| `path_utils` | `include/utils/path_utils.hpp` | Reference-path representation with SIMD-batched nearest-point search |
| `map_utils` | `include/utils/map_utils.hpp` | HD-map interface (Python/Boost.Python bridge), lane topology caching, reference-path construction |
| `occupancy_map` | `include/utils/occupancy_map.hpp` | Drivable-area polygon containment checks via GEOS `PreparedPolygon` |

---

## 1. Parameters (`params.hpp`)

`params.hpp` is a header-only file that defines all compile-time constants used by the planner. Constants are organised into the following categories:

### 1.1 Time and Discretization

| Constant | Value | Description |
|---|---|---|
| `TIME_STEP` | 0.2 s | Simulation time step |
| `STEP_TIME` | 2.0 s | Action duration per tree level |
| `ROLLOUT_TIME` | 2.0 s | Rollout horizon beyond the tree |
| `STEP_TIME_LEN` | 10 | Steps per action (`STEP_TIME / TIME_STEP`) |
| `LOOKAHEAD_TIME` | 4.8 s | Forward prediction horizon |
| `LOOKBACK_TIME` | 3.0 s | Backward reference-line extension |

### 1.2 Planner Core

| Constant | Value | Description |
|---|---|---|
| `TREE_HEIGHT` | 4 | Planning tree depth |
| `NUM_REF_PATH` | 3 | Number of reference paths (left, center, right) |
| `NUM_LATERAL_OFFSETS` | 3 | Lateral offsets per path (-1 m, 0 m, +1 m) |
| `NUM_ACTIONS` | 9 | Total actions = `NUM_REF_PATH * NUM_LATERAL_OFFSETS` |
| `TREE_NODE_SIZE` | 7381 | Pre-computed total nodes: 1 + 9 + 81 + 729 + 6561 |
| `MAX_OBS_VEHICLES` | 192 | Max observed vehicles |
| `MAX_SIM_VEHICLES` | 96 | Max simulated vehicles (aligned to multiples of 8) |

### 1.3 Parallelization

| Constant | Value | Description |
|---|---|---|
| `NUM_SCENARIOS` | 64 | Total belief scenarios |
| `NUM_THREADS` | 8 | Worker thread count |
| `NUM_SCENARIOS_PER_THREAD` | 8 | Scenarios per worker |

### 1.4 Early Termination

| Constant | Value | Description |
|---|---|---|
| `EARLY_TERM_CHECK_INTERVAL` | 5 | Check convergence every N iterations |
| `EARLY_TERM_MIN_EXPAND_CALLS` | 30 | Minimum expansions before checking |
| `EARLY_TERM_STABLE_ITERATIONS` | 5 | Required stable iterations |
| `EARLY_TERM_Q_CHANGE_THRESHOLD` | 0.05 | Relative Q-value change threshold |

### 1.5 Vehicle Geometry and Kinematics

Defines ego vehicle dimensions (`EGO_BB_EXTENT_X/Y`, `WHEEL_BASE`, `FRONT_TO_CENTER`, etc.), kinematic limits (`MAX_STEERING`, `MAX_VEL`, `MAX_ACC`, `MAX_DEC`), and IDM longitudinal model parameters (`DESIRED_TIME_HEADWAY`, `COMFORT_DEC`, `ACC_EXPONENT`).

### 1.6 Reward and Penalty

All penalty constants for the reward function: `CRASH_PENALTY` (-1000), `MOVEMENT_PENALTY` (5.0), `ACTION_PENALTY` (-2.0), `PRUNED_THRESHOLD` (-1000), lateral-offset penalties, non-drivable-area penalties, direction-violation penalties, TTC penalties, and steering penalties. Each constant is documented inline.

### 1.7 Other Categories

- **Trajectory optimisation:** `PROPOSAL_TRAJECTORY_SIZE` (41 steps = 8.0 s), `LATERAL_OFFSETS` (9 offsets), cross-evaluation parameters, emergency brake thresholds, PD controller gains.
- **STRtree spatial index:** `STRTREE_NODE_CAPACITY` (8), buffer sizes for the SIMD-accelerated R-tree.
- **Reference line:** `PATH_POINT_INTERVAL` (0.2 m), anchor segment lengths, curvature-to-speed lookup table.
- **Occupancy map:** radius offsets, safety margins.
- **Lane-changing:** lateral offset thresholds, heading-difference thresholds, time-headway parameters.
- **LQR tracking controller:** state indices, Q/R weight matrices, tracking horizon.
- **Agent filtering (NetBelief):** proximity distances, lateral thresholds for vehicle filtering.

---

## 2. Global Utilities (`global_utils.hpp` / `global_utils.cpp`)

### 2.1 SIMD Type Aliases

Centralises all SIMD vector type definitions used throughout the codebase. These wrap the VAMP library's `FloatVector` / `IntVector` templates:

```cpp
// Base width (typically 8 for AVX2)
constexpr int FloatVectorWidth = vamp::FloatVectorWidth;

// Aligned STL containers
using AlignedVectorFloat = std::vector<float, AlignedAllocator<float>>;
using AlignedVectorInt   = std::vector<int, AlignedAllocator<int>>;

// QMDP tree search vectors (width = NUM_SCENARIOS_PER_THREAD = 8, rows = 1)
using FVectorT_qmdp = vamp::FloatVector<NUM_SCENARIOS_PER_THREAD, 1>;
using IVectorT_qmdp = vamp::IntVector<NUM_SCENARIOS_PER_THREAD, 1>;

// Trajectory optimization vectors (width = 8, rows = LATERAL_OFFSETS_NUM = 9)
using FVectorT_traj = vamp::FloatVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, LATERAL_OFFSETS_NUM>;
using IVectorT_traj = vamp::IntVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, LATERAL_OFFSETS_NUM>;
```

The naming convention encodes the domain: `_qmdp` for tree search, `_traj` for trajectory optimisation, `_track` for LQR tracking, `_1` for single-row general-purpose vectors.

### 2.2 Thread-Local Workspaces

To eliminate dynamic allocation during the hot search loop, pre-sized workspace structs are defined for each major pipeline stage:

| Workspace | Purpose | Key Contents |
|---|---|---|
| `StepBatchWorkspace` | QMDP forward simulation | SIMD ego state vectors, path masks, collision flags, aligned buffers for lead/follow vehicle pairs |
| `GenerateProposalLFWorkspace` | Lane-following trajectory generation | Path nearest indices, acceleration/steering buffers, lead vehicle distance pairs |
| `GenerateProposalLCWorkspace` | Lane-changing trajectory generation | Dual-path tracking state (current + target), LC allowance flags, transition masks |
| `CrossScenarioEvaluationWorkspace` | Cross-scenario trajectory evaluation | Per-timestep SIMD arrays for ego corners, AABB queries, collision results, area masks |
| `OccupancyMapWorkspace` | Drivable-area point-in-polygon queries | Corner coordinates, area ID caches, pre-allocated SIMD segment buffers |

Each workspace is constructed once per thread with the appropriate batch size, and provides a `reset()` method for logical reinitialisation between planning cycles.

### 2.3 Thread Pool

A pImpl-pattern thread pool (`ThreadPool`) with a variadic `enqueue()` method returning `std::future`:

```cpp
ThreadPool pool(num_threads);
auto future = pool.enqueue([](int x) { return x * 2; }, 42);
int result = future.get();  // 84
```

Used by the parallel search infrastructure (`parallelBeliefTreeSearch`) and cross-scenario evaluation.

### 2.4 Performance Profiling

Two-tier profiling system with zero contention during measurement:

1. **Thread-local accumulation:** `thread_local std::map<std::string, PerformanceData> t_local_perf_data` -- each thread records timing data without any synchronisation.
2. **Global merge:** `commitPerformanceData()` acquires `g_perf_mutex` once and merges all thread-local data into `g_global_perf_data`.
3. **Hierarchical timers:** `FastScopedTimer` supports parent-child function nesting. A timer with `parent_function = ""` is top-level; otherwise it accumulates under `parent.sub_functions[name]`.
4. **Inline fast timers:** `fast_now()` / `fast_end()` provide lightweight chrono-based microsecond timing for fine-grained instrumentation.

### 2.5 Logging System

Compile-time-optimised logging with four severity levels:

| Level | Macro | Stream Macro | Optimised Away in Release? |
|---|---|---|---|
| DEBUG | `LOG_D(msg)` | `LOG_DS << ...` | Yes (`NDEBUG` or `__OPTIMIZE__`) |
| INFO | `LOG_I(msg)` | `LOG_IS << ...` | Yes |
| WARNING | `LOG_W(msg)` | `LOG_WS << ...` | No |
| ERROR | `LOG_E(msg)` | `LOG_ES << ...` | No |

The `Logger<L>` template class uses `constexpr if` to eliminate stream operations at compile time for DEBUG/INFO in optimised builds. `initializeLogger(file, level)` configures the output file and minimum level.

`PrintSafe(thread_id, msg)` provides a mutex-guarded thread-safe print with `[Tn]` prefix.

---

## 3. Aligned Allocator (`aligned_allocator.hpp`)

A header-only, STL-compatible allocator that guarantees SIMD-friendly memory alignment:

```cpp
template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator;
```

- Default alignment: **32 bytes** (suitable for AVX2 256-bit vectors).
- Uses `posix_memalign` for allocation and `free` for deallocation.
- Static assertions enforce that `Alignment` is a power of 2 and >= `alignof(T)`.
- Provides `rebind` and equality operators for full STL compatibility.

Used as the allocator for `AlignedVectorFloat`, `AlignedVectorInt`, and `AlignedVectorBool` throughout the codebase.

---

## 4. Math Utilities (`math_utils.hpp`)

Header-only scalar and SIMD math functions:

### 4.1 Distance and Angle

| Function | Signature | Description |
|---|---|---|
| `SquaredDistance` | `(x1, y1, x2, y2) -> float` | Squared Euclidean distance |
| `Distance` | `(x1, y1, x2, y2) -> float` | Euclidean distance |
| `NormalizeAngle` | `(angle) -> float` | Normalise to [-pi, pi) |
| `NormalizeAngleSIMD<T>` | `(angle_vec) -> T` | SIMD-batched angle normalisation |
| `InFront` / `InBehind` | `(x1, y1, theta, x2, y2) -> bool` | Relative position check using heading |

### 4.2 SIMD Geometry

| Function | Description |
|---|---|
| `IsAngleWithinThreshold<T>(vec_x, vec_y, ref_cos, ref_sin, threshold)` | Branchless SIMD angle-within-cone test using dot-product and cos-squared comparison |
| `clipEdgeBatch<T>(p, q, valid, t_min, t_max)` | SIMD Liang-Barsky line-clipping for batched AABB intersection |
| `clipEdgeSerial(p, q, valid, t_min, t_max)` | Scalar version of the above |
| `div_mod<T>(a, b)` | Returns `{a/b, a%b}` pair |

---

## 5. Geometry Utilities (`geometry_utils.hpp`)

Minimal geometry primitives:

```cpp
struct Point  { float x, y; };
struct Polygon { std::vector<Point> vertices; };
```

Used as lightweight value types for polygon containment and collision detection interfaces.

---

## 6. Path Utilities (`path_utils.hpp` / `path_utils.cpp`)

### 6.1 The `Path` Class

`Path` represents a discretised reference line (lane centerline or composite route segment). It stores per-point arrays at `PATH_POINT_INTERVAL` (0.2 m) spacing:

| Field | Type | Description |
|---|---|---|
| `xs_`, `ys_` | `AlignedVectorFloat` | Cartesian coordinates (SIMD-aligned) |
| `thetas_` | `AlignedVectorFloat` | Heading angles |
| `kappas_` | `AlignedVectorFloat` | Curvature values |
| `red_light_point_s_` | `float` | Traffic-light Frenet s-coordinate |

**Composite path metadata:**

| Field | Description |
|---|---|
| `comprised_ref_path_ids_` | Ordered list of lane/connector edge IDs composing this path |
| `comprised_ref_path_idxs_` | Start index of each edge within the path arrays |
| `max_curvature_vec_` / `min_desired_speed_vec_` | Per-segment max curvature and corresponding min desired speed |
| `max_curvature_idx_vec_` | End-point index for each curvature segment |
| `miss_goal_penalty_` | Penalty for deviating from the route goal (max across edges) |
| `goal_frenet_s_` | Frenet s-coordinate of the route goal on this path |

**Anchor segments** (`anchor_xs_`, `anchor_ys_`, `anchor_idxs_`, etc.) pre-compute coarse spatial partitions of the path for fast segment lookup during nearest-point search.

### 6.2 Nearest-Point Search

The `Path` class provides multiple nearest-point search strategies, all leveraging SIMD:

#### 6.2.1 Global Search (`NearestBatch` -- from scratch)

A two-phase SIMD search for batch queries with no prior estimate:

**Phase 1 -- Anchor-segment coarse search (`FindNearestSegmentBatch`):**
Iterates over pre-computed anchor segments, computes the SIMD-batched projection distance from each query point to each segment, and selects the closest segment per query point. This reduces the search range from the full path to a local window (`[begin_idx, end_idx]`).

**Phase 2 -- Binary subdivision:**
Within the local window, a three-point (begin, mid, end) distance comparison narrows the interval. When the midpoint is closest, a further quarter-point subdivision is performed. The loop terminates when the interval width drops below `FVectorT_1::num_scalars_per_row` (typically 8), at which point a final SIMD-width contiguous load evaluates all remaining candidates.

#### 6.2.2 Incremental Search (`NearestBatch` -- with previous index)

For sequential updates (e.g., simulation steps where the ego moves a small distance), the search starts from the previous nearest index and steps forward/backward along the path:

```
previous_idx -= TRACEBACK_STEPS * step_direction
for up to SAFETY_MAX_STEPS (25):
    next_idx = current + step_direction
    if distance improved: update best, reset patience
    else: increment patience counter
    if patience >= PATIENCE_LIMIT (3): stop
```

This yields O(1) amortised cost for smooth motion along the path.

#### 6.2.3 Interpolated Nearest (`InterpolatedNearestBatch`)

Given a discrete nearest index, computes sub-index interpolation by projecting the query point onto the segment between `nearest_idx` and `nearest_idx + 1`:

```
t = dot(query - P[i], P[i+1] - P[i]) / |P[i+1] - P[i]|^2
interpolated_x = P[i].x + t * (P[i+1].x - P[i].x)
interpolated_theta = theta[i] + t * NormalizeAngle(theta[i+1] - theta[i])
interpolated_path_index = i + t
```

All operations are SIMD-batched via `gather` and vectorised arithmetic.

### 6.3 Path Query Utilities

| Method | Description |
|---|---|
| `PointOnLeft(xs, ys, nearest_idxs)` | SIMD cross-product sign test: returns +1.0 if left of path, -1.0 if right |
| `GetFrenetS(idx)` | Converts discrete index to Frenet s-coordinate: `idx * PATH_POINT_INTERVAL` |
| `GetMaxCurvatureAndMinDesiredSpeedBatch(start, end)` | Scans curvature segments in [start, end] range, returns (max_curvature, min_speed) per SIMD lane |
| `GetNearestEdgeIdx(nearest_edge_idx, nearest_idx)` | Maps a path-point index to the comprising edge index |

---

## 7. Map Utilities (`map_utils.hpp` / `map_utils.cpp`)

### 7.1 Purpose

`MapUtils` bridges C++ planning code with the Python HD-map API (nuPlan-style). It wraps Boost.Python calls to query lane topology, build reference paths, and cache results for repeated lookups.

### 7.2 Python GIL Management

Two RAII wrappers ensure thread-safe Python interop in multi-threaded C++:

| Class | Action on Construction | Action on Destruction |
|---|---|---|
| `ScopedGILAcquire` | `PyGILState_Ensure()` | `PyGILState_Release()` |
| `ScopedGILRelease` | `PyEval_SaveThread()` | `PyEval_RestoreThread()` |

Pattern: the main thread (holding GIL) creates a `ScopedGILRelease` before spawning worker threads. Each worker creates a `ScopedGILAcquire` only when it needs to call Python.

### 7.3 Lane Topology Caching

All Python-side queries are cached in `std::unordered_map` lookups:

| Cache | Key | Value | Description |
|---|---|---|---|
| `LANE_ID_MAP_SUCCESSORS` | edge_id | `vector<string>` | Successor lane/connector IDs (sorted by miss-goal penalty, dead ends filtered) |
| `LANE_ID_MAP_ROUTE_SUCCESSORS` | edge_id | `vector<string>` | Route-only successors |
| `LANE_ID_MAP_PREDECESSORS` | edge_id | `vector<string>` | Predecessor IDs |
| `LANE_ID_MAP_LEFT_NEIGHBOR` | edge_id | `string` | Left-adjacent lane ID |
| `LANE_ID_MAP_RIGHT_NEIGHBOR` | edge_id | `string` | Right-adjacent lane ID |
| `LANE_ID_MAP_SMOOTH_PATH` | edge_id | `shared_ptr<Path>` | Smoothed centerline path for the edge |
| `LANE_ID_MAP_MISSED_GOAL_NUM` | edge_id | `float` | Miss-goal penalty (hops to route goal) |
| `LANE_ID_MAP_IS_TERMINATED` | edge_id | (set membership) | Whether the edge is a dead-end |
| `LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT` | connector_id | `bool` | Red-light status |
| `GLOBAL_ROUTE_EDGE_ID` | edge_id | `bool` | Whether the edge is on the global route |

The `cache_mutex_` protects successor lookups that may be accessed from multiple threads during parallel path construction.

### 7.4 Reference-Path Construction

| Method | Description |
|---|---|
| `GetEgoRefPaths(...)` | Builds up to `NUM_REF_PATH` (3) reference paths for the ego vehicle by traversing successors from the current edge, extending backward (`PATH_LOOKBACK_DISTANCE`) and forward (`LOOKAHEAD_MIN_DISTANCE`). Populates `Path` objects with coordinates, curvature, and speed data. |
| `UpdateEgoRefPaths(...)` | Incrementally updates reference paths when the ego transitions between edges (lane to connector or vice versa). |
| `ExtendPath(edge_id, successor_id, ...)` | Extends a `Path` by appending the smoothed centerline of a successor edge. |
| `SmoothPath(edge_id, edge_name)` | Retrieves or computes a smoothed centerline `Path` for an edge, caching the result. |
| `UpdateEgoRefPathsTrafficInfo(...)` | Updates traffic-light Frenet s-coordinates on each reference path. |

### 7.5 Spatial Queries

| Method | Description |
|---|---|
| `GetEdgeIdByPosition(x, y)` | Returns lane/connector IDs at a given position (Python call) |
| `IsPointInLaneById(lane_id, x, y)` | Point-in-polygon test for a specific lane (Python call) |
| `FindNearestEdgeAndUpdate(...)` | Finds the nearest edge to a position, preferring route edges and target-path edges |
| `IsOnRoute(edge_id)` | Checks `GLOBAL_ROUTE_EDGE_ID` membership |

---

## 8. Occupancy Map (`occupancy_map.hpp` / `occupancy_map.cpp`)

### 8.1 Purpose

`OccupancyMap` maintains a set of pre-processed polygons representing the drivable area around the ego vehicle, enabling fast batched point-in-polygon queries for trajectory evaluation.

### 8.2 Polygon Categories

| Category | Storage | Description |
|---|---|---|
| `route_edges_` | `unordered_map<string, PreparedPolygon>` | Polygons of lane/connector edges on the global route |
| `not_on_route_roadblocks_and_carparks_` | same | Off-route roadblocks and car parks |
| `intersections_` | same | Intersection areas |

Each polygon is stored as a GEOS `PreparedPolygon` (pre-processed for fast containment queries).

### 8.3 Incremental Update

`UpdateDrivableMapObjects(ego_x, ego_y, ...)` calls a Python method that returns:
- **Inserted tokens + coordinates:** New polygons to add.
- **Deleted tokens:** Polygons to remove.

This differential update avoids rebuilding the entire polygon set every frame. The update is triggered only when the ego moves beyond `DRIVABLE_MAP_UPDATE_DISTANCE` (30 m) from the last update position.

### 8.4 Batch Containment Query

`ContainsPointsInDrivableAreaBatch(...)` checks multiple ego-vehicle points (center, rear axle, 6 corners) against all polygon categories in a single call, producing per-point SIMD bitmasks:

| Output Mask | Meaning |
|---|---|
| `on_non_drivable_area_mask` | Center point not in any route-edge polygon |
| `left/right_corners_on_non_drivable_area_mask` | Left/right corners outside drivable area |
| `on_coming_traffic_mask` | On an edge with opposing traffic direction |
| `on_intersection_mask` | Inside an intersection polygon |
| `on_multiple_lane_mask` | Overlapping multiple lane polygons |
| `on_different_path_lanes_mask` | On a lane that differs from the reference path |
| `on_non_route_mask` | On a non-route roadblock or car park |

These masks feed directly into the reward function as penalty signals.

### 8.5 Scalar Queries

| Method | Description |
|---|---|
| `ContainsPointInRouteEdges(x, y)` | Single-point containment in route edges |
| `ContainsPointInRoadblocksAndCarparks(x, y)` | Single-point containment in off-route areas |
| `ContainsPointInIntersections(x, y)` | Single-point containment in intersections |

---

## 9. Build Integration

The utils module is compiled as an OBJECT library in `src/utils/CMakeLists.txt`:

```cmake
add_library(vec_qmdp_utils OBJECT
    geometry_utils.cpp
    global_utils.cpp
    map_utils.cpp
    math_utils.cpp
    occupancy_map.cpp
    params.cpp
    path_utils.cpp
)
```

Object files are linked into the parent `vec_qmdp` static library and transitively into the `vec_qmdp_closed_planner` shared library. The `POSITION_INDEPENDENT_CODE` property is set for shared-library compatibility.

Dependencies: Eigen (math), Boost.Python + Python (map interface), GEOS (polygon operations), VAMP (SIMD vector types).
