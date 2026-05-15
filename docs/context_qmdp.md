This document provides a technical overview of the `ContextQMDP` module, which defines the interaction between the autonomous vehicle (Ego) and its environment, specifically focusing on control command calculation, state transitions, and trajectory evaluation.

---

## 1. Functionality Overview

The `ContextQMDP` module is the core planning and interaction engine of the `vec_qmdp` framework. It implements a high-performance, SIMD-accelerated model for autonomous driving that simulates how the vehicle responds to surrounding traffic and map constraints.

### Key Capabilities:
* **Vectorized Interaction Model:** Uses SIMD instructions to simulate hundreds of "scenarios" (belief states) in parallel, allowing the planner to handle environmental uncertainty efficiently.
* **Integrated Control:** Combines longitudinal control (**Intelligent Driver Model - IDM**) and lateral control (**Stanley Controller**) to generate realistic vehicle behavior.
* **Spatial-Temporal Indexing:** Integrates with `STRtree` and the **Separating Axis Theorem (SAT)** for rapid, rigorous collision detection.
* **Dual-Resolution Planning:** Supports both coarse-grained tree search (for high-level decisions) and fine-grained trajectory optimization.

---

## 2. Core Planning and State Transition Functions

### `StepBatch`
This function serves as the **State Transition** mechanism for the QMDP tree search. It simulates vehicle progress over a **2.0-second horizon** with a **0.2-second time step**.

**Internal Execution Logic:**
1.  **Reference Line Identification:** Determines which reference path (left, middle, or right) the Ego vehicle is currently associated with for each scenario.
2.  **Behavioral Decision:** Checks whether the scenario dictates Lane Following (LF) or Lane Change (LC).
3.  **Proximity Search:** Searches for the leading vehicle. In LC scenarios, it checks for lead/following vehicles on both the current and target lanes.
4.  **Feasibility Check:** Invokes `LCAllowanceCheckBatch` to determine if a lane change is physically safe. If unsafe, it reverts to Lane Follow (LF) logic.
5.  **Control Calculation:** Computes acceleration (via IDM) and steering (via Stanley) based on the target path and lead vehicle.
6.  **Reward Accumulation:** Computes rewards/penalties for the step (e.g., progress, safety, comfort).

### `generateProposalTrajectoryLFBatch` & `generateProposalTrajectoryLCBatch`
These functions are used for **Trajectory Optimization**. They generate "Proposal Trajectories" which represent the movement along reference lines. 

* **Logic:** The calculation logic is similar to `StepBatch` (searching lead vehicles, computing IDM/Stanley, and updating states).
* **Resolution:** Unlike the tree search, these functions use a **0.1-second time step** to match the resolution of the published trajectory.
* **Context:** They generate a family of candidate paths with varying lateral offsets to find the optimal trajectory through the environment.

---

## 3. Environmental Sensing and Interaction

### `FindLeadOrFollowVehicleCloseToEgoReferencePathBatch`
Responsible for identifying relevant traffic participants in the **Frenet Frame** relative to the Ego's reference path.
* **Longitudinal ($s$):** Determines if the agent is ahead (Lead) or behind (Follow) the Ego.
* **Lateral ($l$):** Checks if the agent's lateral distance from the reference line is small enough to be considered "in-path" or on a specific adjacent lane.

### `FindLeadVehicleIntersectedBatch`
A temporal-spatial check that looks ahead through a future time window. It determines if an exo-vehicle's predicted trajectory will intersect with the Ego's intended path at any point in the future, even if they are not currently in the same lane.

---

## 4. Decision Logic and Safety Mechanics

### `LCAllowanceCheckBatch`
The primary logic for Lane Change (LC) safety. It evaluates:
* **Lateral Offset:** The current distance from the target path.
* **Heading Deviation:** The angular difference between the vehicle and the path.
* **Dynamic Gaps:** Uses IDM to check if a lane change would force the new lead or following vehicle on the target lane to brake too hard.

### `calculateLateralMotionParams`
Calculates distance compensation needed for smooth transitions. Based on the current lateral distance to the target reference line, it adds a "compensation distance" to the target $s$-coordinate to ensure the vehicle reaches the target lane with a natural forward displacement.

---

## 5. Collision Detection and Evaluation

### `HasCollisionBatch`
A high-performance collision detection routine that operates in two stages:
1.  **Broad Phase (`STRtree`):** Performs a vectorized AABB query to quickly identify a small set of potential collision candidates.
2.  **Narrow Phase (SAT):** For the candidates found, it performs a rigorous **Separating Axis Theorem (SAT)** check to determine if the 2D oriented bounding boxes (OBB) of the Ego and the agent actually overlap.

### `crossScenarioEvaluationBatch`
Evaluates the quality of a generated proposal trajectory across multiple simulated scenarios. It aggregates penalties for:
* **Drivable Area Compliance:** Checks if any part of the vehicle leaves the legal road area.
* **Speed & Progress:** Penalizes deviations from the target speed or insufficient progress.
* **Safety:** Incorporates collision data and Time-to-Collision (TTC) metrics to ensure the trajectory is robust against environmental uncertainty.