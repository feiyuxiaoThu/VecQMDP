# VecQMDP Tree Search -- Design Document

## Overview

VecQMDP is a high-performance, SIMD-vectorised belief tree search engine for online POMDP planning. It maintains **per-scenario search trees** and leverages AVX2/NEON gather/scatter instructions to evaluate UCB scores, propagate returns, and track convergence across all scenarios simultaneously.

The engine is split into two backends that share the same algorithmic building blocks (UCB selection, back-propagation, early termination) but differ in how tree memory is managed:

| Aspect | **Static VecQMDP** | **Dynamic DynVecQMDP** |
|---|---|---|
| Header | `include/planning/vec_qmdp_static.hpp` | `include/planning/vec_qmdp_dynamic.hpp` |
| Tree allocation | Full balanced tree pre-allocated at construction | Nodes allocated on demand via `DynSoAPool` |
| Memory | O(A^H x S) where A = actions, H = height, S = scenarios | O(visited_nodes) -- typically O(max_iters x A) |
| Parent formula | Implicit: parent_rel = (child_rel - 1) / A | Explicit `parent_idx` field per node |
| Best for | Small action space, shallow depth (A^H fits in memory) | Large action space or deep trees (A^H intractable) |
| Example domain | Autonomous driving (`VecQMDP_AD`, A=9, H=10) | Rock Sample (`RockSampleVecQMDP`, A=13, H=90) |

Both backends are **abstract base classes**. A concrete domain must derive from one and implement:

1. **`computeNodeCandidateActions(node_idx)`** -- return the priority-ordered list of valid actions at a node.
2. **`expandNodesBatch(...)`** -- simulate one forward step + rollout for a batch of selected nodes.
3. **`makeWorkerInstance()`** -- factory for single-threaded clones used by the parallel infrastructure.

---

## 1. Static VecQMDP (`VecQMDP`)

### 1.1 Tree Layout

The tree is an implicit A-ary heap stored in flat, SIMD-aligned arrays. For a tree of height H with branching factor A:

```
tree_node_size = sum(k=0..H) A^k = (A^(H+1) - 1) / (A - 1)
```

Every scenario s in [0, S) owns a separate copy of this tree. The **global index** for a node with relative index r in scenario s is:

```
g(r, s) = r + s * tree_node_size
```

Children of a node at relative index r occupy indices [A*r + 1, A*r + A].

### 1.2 Core Algorithms

#### 1.2.1 `exploreNodes` -- SIMD-Parallel UCB Traversal

`exploreNodes` selects one leaf (or unvisited) node per scenario by descending the tree via UCB1. All S scenarios are processed in SIMD-width batches of `IVectorT::num_scalars` (typically 8 for AVX2).

**Key optimisations:**

1. **Precomputed UCB tables.** `ucb_coeff_table_[n] = C * sqrt(ln(n))` and `inv_sqrt_table_[n] = 1/sqrt(n)` are built once via `initUCBTables()`. No transcendental functions are evaluated during search.
2. **SIMD gather.** Per candidate action, visit counts and Q-values for all 8 scenarios are loaded with a single `_mm256_i32gather` instruction, issuing 8 cache-line requests in parallel.
3. **SIMD UCB arithmetic.** `UCB(c) = Q(c) + ucb_coeff[N(parent)] * inv_sqrt[N(c)]` is computed as a fused multiply-add across 8 lanes.
4. **Precomputed scenario offsets.** `scen_offsets_[s] = s * tree_node_size` avoids per-iteration multiplication.

**Correctness invariant:** All active (not-yet-terminated) scenarios are at the same tree depth at the start of every loop iteration. This ensures the candidate-action list obtained from a single representative scenario is valid for all active scenarios.

**Pseudocode:**

```
for each SIMD batch of scenarios:
    current_rel <- 0  (root)
    while any scenario is undone:
        cands <- getNodeCandidateActions(representative_scenario)
        for each action a in cands:
            child_rel <- current_rel * A + 1 + a
            SIMD-gather visit_count[child] and Q[child]
            compute UCB across all 8 lanes
            for each lane:
                if OOB or unvisited -> mark done, record action
                if visited and UCB > best -> track best child
        descend to best child
```

#### 1.2.2 `exploreNodesVote` -- Majority-Vote Depth Synchronisation

When scenarios are heterogeneous (e.g., different exogenous-agent trajectories in autonomous driving), independent UCB probes may reach **different depths** across scenarios. `exploreNodesVote` aligns them via a three-phase protocol:

**Phase 1 -- Standard UCB probe (no penalty).**
Each scenario independently descends its tree using standard UCB until it finds an unvisited child or hits a tree boundary.

**Phase 2 -- Majority vote.**
The depth reached by each scenario is tallied. The **target sync depth** is the depth with the most votes (ties broken in favour of the deeper level; leaf depth is a last resort).

**Phase 3 -- Re-alignment.**
Scenarios whose probe depth differs from the target are re-run with a **depth-sync UCB penalty** on the selection score:

```
UCB_sync(c) = UCB(c) - lambda_depth * |eff_depth(c) - d_target|
```

where `eff_depth(c)` is clamped to `[min_depth(c), max_depth(c)]` (the reachable depth range tracked by `backupDepth`), and `lambda_depth` is `depth_sync_lambda_`.

This load-balancing mechanism ensures that `expandNodesBatch` receives nodes at a uniform depth across scenarios, enabling efficient SIMD-batched state transitions.

#### 1.2.3 `exploreNodesHomogenous` / `backPropagateHomogenous` -- Simplified Search

For domains where **search overhead dominates expansion cost** (e.g., Rock Sample, where the SIMD state transition is very cheap), a simplified homogenous search mode is available under `#define ENABLE_HOMOGENOUS_SEARCH`.

**Key idea:** Only scenario-0's subtree is explored via UCB. The resulting tree path (sequence of relative node indices and actions) is **shared identically across all S scenarios**. Each scenario then independently expands and evaluates its own state at the shared node location.

- **`exploreNodesHomogenous(action_idx)`** performs scalar UCB descent on scenario 0 only. Returns the relative node index; the caller derives scenario s's global index as `g = rel + s * tree_node_size`.
- **`backPropagateHomogenous(node_idx, rollout_values)`** propagates all S scenarios' returns along the **shared path** in SIMD batches. Per level, it:
  1. Gathers `node_rewards` for all BATCH scenarios via SIMD load.
  2. Accumulates into the running per-scenario return.
  3. Masked `count++` and Q running-mean update (mask = not-yet-pruned).
  4. Updates cumulative pruned bitmask: prune if cumulative value < threshold OR reward < threshold.
  5. Max-backup blend over all children of parent: gather child counts and Q-values, build max/found masks, select-blend.
  6. Scatters updated counts and Q-values back.

This trades per-scenario search optimality for dramatically reduced selection overhead: one scalar UCB traversal replaces S independent SIMD traversals. For Rock Sample (`RockSampleStaticVecQMDP`), where `stepBatch` costs ~10 ns/scenario but UCB descent costs ~50 ns/node, homogenous search yields a significant net speedup for small trees.

#### 1.2.4 `backPropagate` -- Scalar Per-Scenario Back-Propagation

After expanding a node at global index g in scenario s, the rollout return v is propagated up to the root:

```
cur <- expanded_node
while cur > root:
    v <- v + reward[cur]
    visit_count[cur]++
    Q[cur] <- Q[cur] + (v - Q[cur]) / visit_count[cur]         // running mean
    Q[cur] <- max(Q[cur], initial_rollout[cur])                 // persistent max-backup
    if v < threshold or reward[cur] < threshold:
        prune; break
    max_child_q <- max over siblings' Q-values (visited only)
    v <- (1 - lambda_max) * max_child_q + lambda_max * v       // max-backup blend
    cur <- parent(cur)    // parent = (cur - 1) / num_actions
root: visit_count[root]++; update Q[root]
```

- **Persistent max-backup:** The first rollout value at each node is stored as `initial_rollout`. The Q-value is always clamped to be >= this initial estimate, preventing regression from unlucky subsequent rollouts.
- **Max-backup blending:** At each ancestor, the propagated value is blended with the maximum Q-value among all visited siblings, controlled by `max_backup_lambda_` (typically 0.001). This stabilises value estimates without the full cost of a minimax backup.
- **Pruning:** If the cumulative return or node reward falls below `pruned_threshold_`, propagation halts early and only the root visit count is incremented (preventing the root from appearing unexplored).

#### 1.2.5 `backupDepth` -- Depth-Gap Tracking

Maintains per-node `[min_depth, max_depth]` intervals that record the shallowest and deepest leaves reachable from each node. These are used by `exploreNodesVote` to compute the depth-sync penalty.

After expanding a child at depth d:

```
min_depth[child] <- d
max_depth[child] <- d
cur <- child
while cur > root:
    par <- parent(cur)
    new_min <- min over active children's min_depth
    new_max <- max over active children's max_depth
    if parent not fully expanded:
        clamp by parent's own depth
    if min_depth[par] == new_min and max_depth[par] == new_max:
        break   // no change -> ancestors are already up to date
    min_depth[par] <- new_min
    max_depth[par] <- new_max
    cur <- par
```

### 1.3 Example: Autonomous Driving (`VecQMDP_AD`)

`VecQMDP_AD` (`include/planning/vec_qmdp_ad.hpp`) instantiates the static VecQMDP for on-road motion planning:

| Parameter | Value |
|---|---|
| Reference paths | 3 (left, center, right) |
| Lateral offsets per path | 3 (-1 m, 0 m, +1 m) |
| **Total actions** | 3 x 3 = 9 |
| Action duration | 2.0 s |
| Time step (simulation) | 0.2 s |
| **Tree height** | 10 (= 2.0 s / 0.2 s) |

**Action-transition table:** Actions are constrained by lane adjacency -- a vehicle on the center path can switch to left or right, but cannot jump two lanes. `STATIC_TRANSITION_TABLE[current_action]` provides the priority-ordered list of reachable target actions (center-out expansion: same action first, then +/-1, +/-2, ...).

**Search loop:**

```cpp
while (budget_remaining) {
    auto nodes = exploreNodesVote(target_action_idxs);    // depth-synced UCB
    decomposeActionBatch(target_action_idxs, paths, offsets);
    expandNodesBatch(nodes, target_action_idxs, paths, offsets, iter);
    // expandNodesBatch internally calls:
    //   context_qmdp.StepBatch(...)   -> SIMD forward simulation
    //   backPropagate(node, s, value) -> per-scenario value propagation
    //   backupDepth(node, s)          -> depth-gap update
}
```

Multi-threaded search is supported via `parallelBeliefTreeSearch`, which dispatches independent `beliefTreeSearch` calls to worker instances and averages per-action Q-values.

---

## 2. Dynamic DynVecQMDP (`DynVecQMDP`)

### 2.1 Motivation

For domains with large action spaces and/or deep trees, the static pre-allocated tree becomes intractable. For example, Rock Sample with A=13 actions and H=90 depth would require 13^90 (approx 10^100) nodes per scenario -- clearly infeasible.

DynVecQMDP solves this by allocating nodes **only when visited**, reducing memory from O(A^H) to O(max_iters x A) -- typically ~40,000 nodes per thread for a 5000-iteration search budget.

### 2.2 DynSoAPool -- Chunked Structure-of-Arrays Memory Pool

`DynSoAPool` (`src/planning/dynamic_soa_pool.hpp`) is a thread-private, lock-free memory pool composed of fixed-size SoA chunks:

- **Chunk size:** `DYN_CHUNK_CAPACITY = 32768` slots (2^15, for fast index decomposition via bit shift/mask).
- **Per-chunk layout:** Each `DynSoAChunk` is a 64-byte-aligned struct containing parallel arrays for all node fields:
  - Value/visit: `q_values[]`, `rewards[]`, `visit_counts[]`, `initial_rollout[]`
  - Depth: `depth[]`, `rollout_len[]`, `min_depth[]`, `max_depth[]`
  - Topology: `child_base_idx[]`, `parent_idx[]`, `action_to_reach[]`
  - Control: `active_flags[]`, `curr_action_idx[]`
- **Global index:** `g = chunk_id * 32768 + slot`. Decomposition uses bit shift and mask (no division).
- **Allocation:** A single counter `total_allocated_` is bumped; when the current chunk fills, a new one is appended. Existing chunk pointers are stable (no relocation on growth).
- **Reset:** Logical only -- `total_allocated_ = 0`. Chunks are retained for reuse without heap traffic.
- **Pre-warming:** 16 chunks (512K slots, ~24 MB) are allocated at construction to avoid `grow()` calls during the early search phase.

### 2.3 Explicit Tree Topology

Unlike the static backend, parent-child relationships are stored explicitly:

- **`child_base_idx[g]`**: Global index of the first child of node g. Children for action a are at `child_base + a`. Set to `DYN_INVALID_IDX` (-1) for unexpanded nodes.
- **`parent_idx[g]`**: Global index of the parent node. `DYN_INVALID_IDX` for root nodes.

**`allocateChildBlock(parent_g)`** allocates `num_actions_` consecutive slots, initialises their tree fields (depth, rollout length, parent pointer, action label), and writes `child_base_idx` on the parent. The derived class must then initialise domain-specific state at the returned child indices.

### 2.4 Core Algorithms

#### 2.4.1 `exploreNodes` -- Standard UCB Search

Identical logic to the static variant but operates on pool global indices instead of implicit heap positions. For each scenario:

1. Start at `scenario_roots_[s]`.
2. Descend via UCB: at each node, check `child_base_idx` -- if `DYN_INVALID_IDX`, the node has not been expanded yet; stop.
3. Among allocated children, find unvisited ones first (return immediately), then select the best UCB child.
4. Return a `DynExploreResult` per scenario containing `{parent_global_idx, parent_depth, action_idx, scenario_idx}`.

#### 2.4.2 `exploreNodesVote` -- Majority-Vote Depth Synchronisation

Same three-phase protocol as the static variant:

1. **Phase 1:** Standard UCB probe per scenario.
2. **Phase 2:** Majority vote to determine the target sync depth.
3. **Phase 3:** Re-align mismatched scenarios with a depth-sync penalty on UCB.

The return type is `std::vector<DynExploreResult>`, which carries the global pool index of each selected parent. This allows the derived class to **SIMD-gather domain state** directly from the pool without reconstructing node paths.

#### 2.4.3 `exploreNodesHomogenous` / `backPropagateHomogenous`

Available under `#define ENABLE_HOMOGENOUS_SEARCH`. Analogous to the static variant, but adapted for the dynamic pool:

- **`exploreNodesHomogenous(action_idx)`** traverses scenario-0's tree via UCB and maintains a **level-indexed node map** `homogenous_node_map_[level][s]` that tracks each scenario's global pool index at each depth level. When descending through a visited child, it synchronises the map for all scenarios:

    ```
    for each scenario s:
        homogenous_node_map_[level][s] = child_base(parent_at_level[s]) + taken_action
    ```

    Returns the current level (not a node index), which `backPropagateHomogenous` uses to walk back up.

- **`backPropagateHomogenous(child_level, rollout_values)`** traverses from `child_level` to the root, updating all scenarios at each level using the node-map. Per level:
  1. Accumulate reward into each scenario's working value.
  2. Update visit count and Q-value (incremental running mean).
  3. Apply persistent max-backup: `Q = max(Q, initial_rollout)`.
  4. Check pruning; if triggered, bump root counts and exit.
  5. Max-backup blending: blend the maximum sibling Q-value into the propagated return.

#### 2.4.4 `backPropagate` -- Explicit Parent-Chain Traversal

Follows `parent_idx` pointers instead of the implicit heap formula. Otherwise identical to the static version:

```
cur <- child_g
while cur != DYN_INVALID_IDX:
    v <- reward[cur] + gamma * v                    // discount factor applied
    visit_count[cur]++
    Q[cur] <- running_mean_update(Q[cur], v)
    Q[cur] <- max(Q[cur], initial_rollout[cur])     // persistent max-backup
    if pruned: break
    max-backup blend with siblings
    cur <- parent_idx[cur]
```

Note: The dynamic variant applies the discount factor gamma explicitly during back-propagation (`reward + gamma * v`), whereas the static variant expects the caller to pre-discount.

#### 2.4.5 `backupDepth` -- Depth-Gap Tracking

Identical semantics to the static variant, but traverses `parent_idx` pointers and reads sibling information via `child_base_idx`.

### 2.5 Example: Rock Sample (`RockSampleVecQMDP`)

`RockSampleVecQMDP` (`examples/rock_sample/src/rock_sample_vecqmdp/rock_sample_dynamic_vecqmdp.hpp`) instantiates DynVecQMDP for the classic Rock Sample POMDP:

| Parameter | Value |
|---|---|
| Actions | 6 (North, East, South, West, Sample, Sense) -- up to 5 + num_rocks for larger instances |
| Tree height | 90 |
| Scenarios per worker | Configurable (8-3000 typical) |
| Memory (static) | 6^90 approx 10^70 nodes -- **impossible** |
| Memory (dynamic) | ~5000 iters x 6 actions = 30K nodes -- **~1.5 MB** |

**Domain state encoding (Data-Oriented Design):**

```
robot_pos_flat_[global_pool_idx] = y * grid_size + x    (int)
rock_bits_flat_[global_pool_idx] = bitmask               (bit i = rock i is good)
```

These plain `std::vector<int>` arrays grow in lockstep with the pool via `ensureDomainCapacity()`.

**SIMD-batched expansion:**

`expandNodesBatch` processes all `scenario_size_` scenarios in SIMD batches of `IVectorT::num_scalars` (8 for AVX2):

1. **Allocate:** Call `allocateChildBlock(parent_g)` for parents without children.
2. **Gather:** `IVectorT::gather` loads 8 parent states (`robot_pos`, `rock_bits`) in two SIMD instructions.
3. **`stepBatch`:** Computes all 8 transitions in parallel -- movement, sampling, sensing -- producing new positions, rock bits, active flags, and rewards.
4. **Rollout:** `rolloutEast` (always-east lower bound) or `rolloutENT` (explore-nearest-in-thresholded-state) computes the discounted future return per lane.
5. **Scatter:** A scalar loop writes child state and rewards to their (distinct) global pool indices.
6. **Back-propagate:** `backPropagate(child_g, rollout_value)` and `backupDepth(child_g)`.

---

## 3. Common Infrastructure

### 3.1 Parallel Search

Both backends provide a thread-pool-based parallel search mechanism:

1. **`initParallelInfrastructure(num_threads)`** creates N independent worker instances via `makeWorkerInstance()` and starts a shared `ThreadPool`.
2. **`dispatchParallelSearch(search_fn)`** dispatches the search function to all workers in parallel, collects per-action Q-value sums and visit counts, and computes the per-action average:

```
Q_agg(a) = sum(t=1..N) sum(s) Q_t(a, s) / sum(t=1..N) count_t(a)
```

Each worker operates on its own tree (static or dynamic), with its own scenario set -- no shared mutable state, no locks.

### 3.2 Early Termination

Both backends track per-scenario convergence and support automatic early termination when the best action stabilises:

- **Check interval:** Every `early_term_check_interval_` iterations (default 10).
- **Minimum expand calls:** At least `early_term_min_expand_calls_` (default 50) must occur before checking.
- **Stability criterion:** The best action must remain unchanged for `early_term_stable_iters_` consecutive checks (default 5), with at least `early_term_min_best_visits_` visits.
- **Q-value convergence:** The relative change in the best Q-value must fall below `early_term_q_change_thr_` (default 0.01).
- Termination requires **all scenarios** to satisfy these criteria simultaneously.

### 3.3 Hyperparameters

| Parameter | Setter | Description |
|---|---|---|
| `discount_factor_` | `setDiscountFactor(v)` | Discount factor gamma for return accumulation |
| `exploration_constant_` | `setExplorationConstant(v)` | UCB exploration coefficient C |
| `max_backup_lambda_` | `setMaxBackupLambda(v)` | Blend weight for max-backup (0 = pure max, 1 = pure Monte Carlo) |
| `depth_sync_lambda_` | `setDepthSyncLambda(v)` | Penalty coefficient for depth-sync in `exploreNodesVote` |
| `pruned_threshold_` | `setPrunedThreshold(v)` | Cumulative return below which a scenario is pruned |
| `max_planning_time_ms_` | `setMaxPlanningTimeMs(v)` | Wall-clock budget for the search loop |

---

## 4. Choosing a Backend

Use **Static VecQMDP** when:
- A^H is small enough to pre-allocate (e.g., 9^10 -- large but feasible with sparse visitation).
- The implicit heap addressing yields better cache locality for shallow trees.
- You need SIMD-parallel UCB across scenarios (`exploreNodes`).

Use **Dynamic DynVecQMDP** when:
- A^H is astronomically large (e.g., 13^90).
- Memory must be proportional to the actual search budget, not the theoretical tree size.
- The domain benefits from explicit parent pointers (e.g., non-uniform branching, domain state gathered by global index).

Use **Homogenous Search** (`exploreNodesHomogenous` / `backPropagateHomogenous`) when:
- The search/selection overhead dominates the expansion cost.
- Approximate planning optimality is acceptable (all scenarios share one search path).
- The domain state transition is very cheap (e.g., discrete grid-world POMDPs like Rock Sample).
