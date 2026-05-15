This document provides a technical overview of the `State` module, which manages the physical attributes and spatial relationships of both the self-vehicle (Ego) and surrounding agents (Exo) within the `vec_qmdp` framework.

---

## 1. Functionality Overview

The `State` module serves as the primary data interface and coordinate transformation engine for the planning system. It is composed of three main structures:

* **`EgoState`**: Manages the autonomous vehicle's internal state, including position, velocity, acceleration, heading, and steering metrics. It also includes logic to determine proximity to junctions based on map topology and current speed.
* **`ObservedExoState`**: Acts as a container for all surrounding agents (vehicles and pedestrians) detected in the environment. It facilitates the transition of data from Python/Numpy formats into high-performance C++ SoA (Structure of Arrays) layouts.
* **`ExoStates`**: A utility class containing static, SIMD-accelerated methods for batch processing agent trajectories, specifically for coordinate transformations and spatial index construction.

The module relies heavily on **SIMD (Single Instruction, Multiple Data)** instructions to process agents in batches of 8, ensuring that coordinate conversions and collision-index building do not become a bottleneck during high-frequency planning.

---

## 2. Key Processing Functions

### `GetFrenetPointsBatch`
This function is responsible for converting agent coordinates from the Cartesian global frame $(X, Y)$ to the Frenet frame $(s, l)$ relative to a specific reference path.

* **SIMD Acceleration**: It processes batches of agents simultaneously using `vamp::FloatVectorWidth` (typically 8).
* **Coordinate Transformation**: It performs batch lookups for the nearest points on the path and computes the longitudinal position ($s$) and lateral offset ($l$) using vectorized vector math.
* **Velocity Adjustment Logic**: A unique feature of this function is the modification of agent velocities based on their heading relative to the reference path:
    * **Reversal**: If the difference between the agent's heading and the path heading ($\Delta\theta$) exceeds $2/3\pi$, the velocity is reversed ($v = -v$), indicating the agent is moving against the path direction.
    * **Stopping**: If $\Delta\theta$ is between $1/3\pi$ and $2/3\pi$, the velocity is set to `EXO_FRENET_STOPPED_VELOCITY`.
    * **The $0.051$ Constant**: `EXO_FRENET_STOPPED_VELOCITY` is specifically set to **$0.051\text{ m/s}$**. This non-zero value is used to distinguish agents that are "plannably stopped" (e.g., a car waiting at a junction) from truly static obstacles or parked cars that have a velocity of exactly $0.0$.

### `buildSTRtreesFrenetBatch`
This function constructs spatial indices (`STRtree`) for agent predictions across the entire planning horizon.

* **Temporal Hierarchy**: It builds a separate `STRtree` for every time step within the prediction window and for every potential reference path.
* **Projected Bounding Boxes**: Instead of using raw agent dimensions, it calculates the **projected radius** of the agent's bounding box in Frenet space ($s$-$l$). This projection accounts for the agent's heading relative to the path curvature.
* **Safety Margins**: When inserting into the tree, it applies `EXO_STRTREE_SAFETY_MARGIN` to the lateral bounds to ensure conservative spatial querying during the QMDP sampling process.
* **Efficiency**: The function clears and reuses existing tree objects to avoid frequent memory allocations, maintaining high performance during real-time execution.