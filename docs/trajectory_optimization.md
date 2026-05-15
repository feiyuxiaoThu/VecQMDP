This summary provides an overview of the `TrajectoryOptimization` class, which implements the vectorized trajectory generation and refinement pipeline for the motion planner.

---

# Trajectory Optimization

## 1. Overview
The `TrajectoryOptimization` module is responsible for transforming high-level behavioral decisions into executable, smooth, and safe vehicle trajectories. It utilizes a **vectorized architecture** to evaluate a large batch of candidate trajectories across multiple probabilistic scenarios simultaneously.

The optimization pipeline follows three primary stages:
1.  **Proposal Generation:** The planner generates a set of "proposals" based on selected reference paths. To increase the flexibility and search space of the planner, **lateral offsets** are applied to the reference trajectories. This creates a diverse set of spatial paths and corresponding velocity profiles (e.g., varying degrees of shifting within a lane or across lanes).
2.  **Vectorized LQR Tracking:** Once proposals are generated, a C++ based **vectorized LQR tracker** is used to simulate how the vehicle would actually follow these paths. This step accounts for vehicle kinematics and dynamics, turning geometric proposals into physically realizable trajectories.
3.  **Cross-Scenario Evaluation:** The resulting trajectories are scored against an occupancy map and multiple sampled "belief scenarios" (potential future behaviors of other agents). The trajectory that maximizes the expected reward while maintaining safety is selected as the final output.

---

## 2. Key Functions Detailed

### `determineReferencePath`
This function evaluates the feasibility of following a specific target path (typically during a lane change).
* **Logic:** It identifies leading and following vehicles on the target path relative to the ego vehicle's projected progress.
* **Safety Check:** It calculates the gaps to these vehicles and uses the `LCAllowanceCheckSerial` logic to determine if a lane change is safe and permissible.
* **Decision:** If the target path is deemed unsafe or blocked, the planner can decide to revert to a "Lane Following" (LF) trajectory on the current path instead of a "Lane Change" (LC) trajectory.

### `checkAndGenerateEmergencyBrake`
This function acts as a safety "governor" that can override the optimized trajectory with an emergency braking profile if a critical situation is detected.
* **TTC Trigger:** It monitors the `time_to_infraction_`. If the time to a potential collision falls below critical thresholds (e.g., `EMERGENCY_TTC_CRITICAL`), it calculates a hard braking response.
* **Physical Property Analysis:** It examines the acceleration and velocity of the best-evaluated trajectory. If the planner detects a "hard-brake" pattern in the optimal results at low speeds or extreme decelerations, it generates a dedicated braking trajectory using a PD-controller approach to ensure a safe stop.

### `determineTrajectoryMode`
Before generating trajectories, this function decides the operational mode:
* **Lane Following (LF):** Used when the ego vehicle intends to stay in its current lane or has already successfully merged into the target lane.
* **Lane Change (LC):** Used when the vehicle is actively transitioning between paths. It checks if the ego vehicle is already "on target" based on lateral distance thresholds to prevent unnecessary steering once a merge is nearly complete.

### `importanceSampleScenarios`
To handle uncertainty in other agents' behaviors, this function performs **Importance Sampling** from the prediction belief.
* **Diversity:** It samples multiple potential "modes" (trajectories) for each external agent.
* **Weighting:** It calculates importance weights to ensure that high-probability scenarios have a greater impact on the final trajectory evaluation, while still accounting for low-probability but high-risk events.

---

## 3. Implementation Details
* **Data Layout:** Uses flattened arrays (`exo_xs_flat_`, etc.) and SIMD-friendly structures to ensure that evaluation across hundreds of scenarios remains computationally efficient.
* **Spatial Indexing:** Employs **STRtrees** (Spatial Index Trees) to perform fast collision checking and distance calculations between the ego vehicle and other agents across all scenarios.