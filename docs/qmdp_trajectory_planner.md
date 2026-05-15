This summary provides an overview of the `QMDPTrajectoryPlanner` class, which serves as the core orchestration layer for the motion planning system.

---

# QMDPTrajectoryPlanner

## Overview
The `QMDPTrajectoryPlanner` is the primary entry point for a belief-aware autonomous driving motion planner. It implements a **QMDP (Quick-MDP)** approach, which bridges high-level decision-making under uncertainty with low-level geometric trajectory optimization. 

The planner's primary responsibility is to evaluate a set of discrete high-level actions (such as lane changes or lateral offsets) across multiple probabilistic scenarios and then refine the best-evaluated action into a smooth, collision-free trajectory.

## Key Functionalities

### 1. Planning Entry Point (`planTrajectory`)
This is the main execution pipeline. It manages the lifecycle of a single planning frame:
* Receives the current **EgoState**, environment **Belief**, and available **Reference Paths**.
* Triggers the parallel search to determine the best high-level intent.
* Triggers the parallel optimization to generate the final executable trajectory.
* Handles special cases like **Static Motion** (e.g., when the vehicle is stationary or in a fallback state).

### 2. Parallel Belief Tree Search
The planner utilizes `VecQMDP_AD` to perform a multi-threaded search across various scenarios sampled from the environment belief.
* **Evaluation:** It calculates Q-values (action values) for different candidate paths and offsets.
* **Efficiency:** The search is parallelized to process multiple "belief scenarios" simultaneously, ensuring real-time performance in complex environments.

### 3. Best-Action Selection (`getBestAction`)
Once the search is complete, the planner aggregates the results to select the optimal high-level action:
* **Preference Logic:** It incorporates a "current path preference bonus" to prevent unnecessary oscillating lane changes (hysteresis).
* **Safety Fallbacks:** If the search indicates that all primary actions lead to potential collisions, the planner includes logic to fall back to the previous best action or a default "stay-in-lane" behavior.

### 4. Parallel Trajectory Optimization
After deciding on a high-level goal (e.g., "move to the left lane"), the planner executes a fine-grained optimization:
* **Two-Phase Optimization:** 1.  **Generation:** Creates candidate trajectory proposals (e.g., using LQR or polynomial curves).
    2.  **Evaluation:** Scores these trajectories against an occupancy map and predicted agent behaviors.
* **Thread Management:** It uses a dedicated `ThreadPool` and manages the **Python Global Interpreter Lock (GIL)** to allow C++ worker threads to run concurrently while interacting with Python-based trajectory generators.

## Architecture & Design
* **Concurrency:** Utilizes `std::future` and a custom `ThreadPool` to maximize CPU utilization during the heavy compute phases of tree search and trajectory scoring.
* **Modularity:** Separates high-level decision-making (`VecQMDP_AD`) from geometric path refinement (`TrajectoryOptimization`).
* **Python Integration:** Designed to work within a hybrid C++/Python ecosystem, handling data conversion from NumPy arrays and managing cross-language execution safety.