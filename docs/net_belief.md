This document provides a technical overview of the `NetBelief` component, which is responsible for managing environmental uncertainty and agent filtering within the autonomous driving framework.

---

## 1. Functionality Overview

The `NetBelief` class manages the **belief space** of the ego vehicle by integrating and processing multi-modal predictions for surrounding agents (exo-vehicles).

### Key Capabilities:
* **Multi-modal Belief Representation:** It stores and interfaces with multi-modal trajectory predictions (positions, velocities, headings, and probabilities) for all observed agents. This data represents the different possible future behaviors of surrounding traffic, forming the basis for probabilistic planning.
* **Path Context Integration:** It maintains a set of ego reference paths (`ego_ref_paths_`) used to establish the spatial context for the belief update.
* **SIMD-Accelerated Processing:** Many operations, including coordinate transformations and collision checks, utilize SIMD (Single Instruction, Multiple Data) types to ensure high-performance execution.

---

## 2. Agent Filtering Logic (`filterDiscardedAgents`)

The `filterDiscardedAgents` function is a critical optimization step. It prunes the set of observed agents to identify those most relevant to the ego vehicle's decision-making, significantly reducing downstream computational load.

The filtering logic follows a multi-stage process:

### Phase 1: Frenet-based Spatial Filtering
The algorithm first evaluates the spatial relationship between agents and the ego vehicle's potential reference paths using Frenet coordinates ($s$ for longitudinal, $l$ for lateral distance).

* **Reference Path Projection:** Agents are projected onto all available ego reference paths.
* **Lateral Distance Cutoff:** Agents are discarded if their lateral distance ($|l|$) from the reference paths exceeds specific thresholds (e.g., `FILTER_OTHER_LANE_LATERAL_CUTOFF`).
* **Lane-Level Relevance:** The system prioritizes:
    * The **closest leading vehicle** on the ego vehicle's current reference path.
    * The **closest following vehicle** (if it poses a specific threat).
    * Agents on adjacent lanes that are within a safe lateral margin.

### Phase 2: Future Trajectory Intersection Filtering
If an agent passes the initial spatial filter, the system performs a deeper temporal check using its **Most Likely (ML) prediction mode**.

* **Trajectory-Path Intersection:** The system simulates the agent's most probable future trajectory and checks for intersections with the ego vehicle's **current** reference path.
* **Dynamic Bounding Box Check:** At each future time step, the agent's predicted position and expanded bounding box are compared against the ego's reference path.
* **SIMD Acceleration:** This intersection check is fully vectorized (using `NearestBatch` and batch intersection masks), allowing the system to check multiple future time steps simultaneously for an agent.
* **Result:** Only agents whose future trajectories physically cross or significantly encroach upon the ego's path (within a longitudinal and lateral safety threshold) are kept in the active belief set.

### Phase 3: Collision and Validity Management
Finally, the function handles edge cases and physical constraints:
* **Collision Exclusion:** Any agent that has already collided with the ego vehicle (tracked via `collided_agents_tokens_`) is automatically discarded.
* **Static Motion Inference:** The presence of nearby agents is used to determine if the ego should remain in a "static motion" state (e.g., waiting at a red light or behind stationary traffic).