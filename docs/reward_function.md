This summary provides an overview of the `reward_functions.hpp` file, which defines the evaluation logic used by the `QMDPTrajectoryPlanner` to score candidate trajectories and high-level actions.

---

# VecQMDP Reward and Penalty Functions

## 1. Overview
The `reward_functions.hpp` file contains the core evaluation logic for the motion planning system. While the planner orchestrates parallel search and trajectory optimization, this file defines the **cost functions** that allow those processes to distinguish between "good" and "bad" driving behaviors.

To maintain high performance during parallelized scenario search, these functions are implemented as **inline SIMD-templated functions**. This architecture allows the planner to evaluate rewards for a whole batch of scenarios (e.g., different probabilistic agent behaviors) simultaneously using vector instructions.

---

## 2. Reward Function Categories

The rewards are categorized to balance the trade-off between progress, safety, and legal compliance.

### A. Efficiency (Speed and Progress)
These functions encourage the vehicle to maintain a target speed and make progress toward the destination.
* **Movement Penalty:** Penalties are applied if the ego vehicle's velocity deviates significantly from the maximum allowable speed (`MAX_VEL`).
* **Goal Penalty:** This function penalizes the vehicle for ending a planning horizon too far from the intended Frenet-frame goal. It uses a dynamic factor that increases as the vehicle approaches the terminal point of the route.
* **Action Penalty:** To prevent "oscillating" decisions, a penalty is applied whenever the planner switches between different candidate paths or lateral offsets.

### B. Safety
Safety functions represent the highest priority and typically carry the heaviest weights (e.g., `CRASH_PENALTY`).
* **Collision Penalty:** A direct penalty applied when a collision is detected between the ego vehicle and an external agent (exo-vehicle).
* **Collision Classification:** The code includes a complex `CollisionType` system that classifies impacts into categories like `STOPPED_EGO`, `REAR_COLLISION`, or `ACTIVE_FRONT_COLLISION`.
* **No-At-Fault Logic:** Implements a nuPlan-inspired metric to determine "at-fault" status. It can ignore or reduce penalties for collisions where the ego vehicle is stationary or hit from behind by another moving agent.
* **Lateral Offset Guidance:** A linear penalty that guides the vehicle away from non-drivable areas by encouraging minimal lateral adjustments to return to the center of the lane.

### C. Traffic Rules and Compliance
These functions ensure the vehicle adheres to road geometry and traffic laws.
* **Direction Compliance:** Penalizes the vehicle for driving against the flow of traffic (oncoming traffic). It uses a cumulative speed threshold to distinguish between minor deviations and major violations.
* **Non-Drivable Area Penalty:** Penalizes the vehicle for entering areas designated as off-road or non-drivable.
* **Route Compliance (`OnNonRoute`):** Penalizes the vehicle for selecting paths that are not part of the assigned navigation route.
* **Curvature Scaling:** Many traffic rule penalties (like being on the wrong lane) are scaled by the path's curvature, applying higher penalties in sharp turns where precision is more critical.

---

## 3. Implementation Details
* **SIMD Batching:** Most functions accept a template type `T` (representing a vector of floats) and `U` (a vector of integers/masks), allowing the system to process 8 or 16 scenarios in a single CPU cycle.
* **Geometric Primitives:** Includes optimized line-rectangle intersection tests to determine exactly which part of the vehicle (front, rear, or side) made contact during a collision.