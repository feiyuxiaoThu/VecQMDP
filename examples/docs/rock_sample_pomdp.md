# Rock Sample: POMDP Problem Formulation

## 1. Problem Overview

Rock Sample is a canonical benchmark problem for POMDP planning algorithms, originally proposed by Smith & Simmons (2004). It models a rover exploration scenario on a 2-D grid where the rover must decide which rocks to collect and when to exit the map, under uncertainty about which rocks are scientifically valuable.

The key challenge is that the rover can sense rocks from a distance, but the sensor is noisy — accuracy degrades with distance. The agent must reason under partial observability to decide whether to approach and sample a rock, or move on.

---

## 2. State Space

### 2.1 Factored State Representation

The state \( s \in \mathcal{S} \) is a tuple:

$$s = (\text{pos}_{\text{rob}},\ \text{status}_{0},\ \text{status}_{1},\ \ldots,\ \text{status}_{K-1})$$

where:
- \(\text{pos}_{\text{rob}} \in \{0, 1, \ldots, N-1\}^2\) is the robot position on an \(N \times N\) grid.
- \(\text{status}_k \in \{0, 1\}\) is the goodness of rock \(k\): `1` = good (valuable), `0` = bad (worthless).
- \(K\) is the total number of rocks.

### 2.2 State Encoding

States are encoded as a single integer `state_id`:

$$\text{state\_id} = \text{pos\_index} \times 2^K + \text{rock\_bits}$$

- **`rock_bits`**: bits 0 through \(K-1\), where bit \(k\) = 1 means rock \(k\) is good.
- **`pos_index`**: the linearized grid index \(= y \times N + x\).

Decoding:
```
pos_index = state_id >> K
rock_k_is_good = CheckFlag(state_id, k)   // i.e., (state_id >> k) & 1
```

### 2.3 Total State Count

$$|\mathcal{S}| = N^2 \times 2^K + 1$$

The final "+1" is a dummy **terminal state** reached when the robot exits the east boundary.

### 2.4 Standard Configurations

| Configuration | Grid | Rocks | \(|\mathcal{S}|\) |
|---|---|---|---|
| RockSample(4,4) | 4×4 | 4 | 4×4×16 + 1 = 257 |
| RockSample(5,5) | 5×5 | 5 | 5×5×32 + 1 = 801 |
| RockSample(5,7) | 5×5 | 7 | 5×5×128 + 1 = 3201 |
| RockSample(7,8) | 7×7 | 8 | 7×7×256 + 1 = 12545 |
| RockSample(11,11) | 11×11 | 11 | 11×11×2048 + 1 = 247809 |

**RockSample(7,8) rock layout** (from `Init_7_8`):

```
Rock 0: (2,0),  Rock 1: (0,1),  Rock 2: (3,1),  Rock 3: (6,3)
Rock 4: (2,4),  Rock 5: (3,4),  Rock 6: (5,5),  Rock 7: (1,6)
Start: (0,3)
```

---

## 3. Action Space

The total number of actions is \(K + 5\):

$$\mathcal{A} = \underbrace{\{\text{NORTH},\ \text{EAST},\ \text{SOUTH},\ \text{WEST}\}}_{\text{4 movement}} \cup \underbrace{\{\text{SAMPLE}\}}_{\text{1 sample}} \cup \underbrace{\{\text{CHECK}_0, \ldots, \text{CHECK}_{K-1}\}}_{\text{K sensing}}$$

### 3.1 Action Indices

```
NORTH   = 0  (Compass::NORTH)
EAST    = 1  (Compass::EAST)
SOUTH   = 2  (Compass::SOUTH)
WEST    = 3  (Compass::WEST)
SAMPLE  = 4  (E_SAMPLE)
CHECK_k = 5 + k   for k = 0, ..., K-1
```

Action \(a \leq 3\) is a movement, \(a = 4\) is sample, and \(a > 4\) is a sensing action for rock \(a - 5\).

---

## 4. Transition Model

The transition function \( T(s, a, s') = P(s' \mid s, a) \) is **deterministic** for all actions.

### 4.1 Movement Actions (\(a < \text{SAMPLE}\))

The robot moves one step in direction \(a\):

$$\text{pos}' = \text{pos} + \Delta_a$$

- If \(\text{pos}'.x = N\): the robot exits the map → transition to the terminal state.
- If \(\text{pos}'.x < 0\), \(\text{pos}'.y < 0\), or \(\text{pos}'.y \geq N\): **illegal move**, stay in place.
- Rock statuses are unchanged.

### 4.2 Sample Action (\(a = \text{SAMPLE}\))

Let \(k = \text{rock at current position}\):
- If no rock is present: illegal, state unchanged.
- If rock \(k\) is present: rock \(k\) status is set to bad (bit \(k\) cleared):

$$\text{status}'_k = 0,\quad \text{all other components unchanged}$$

Sampling is **irreversible** — a sampled rock stays bad regardless of its original quality.

### 4.3 Sensing Actions (\(a > \text{SAMPLE}\))

Sensing does not change state:
$$s' = s$$

### 4.4 Transition Summary

$$T(s, a, s') = \begin{cases} 1 & s' = \text{NextState}(s, a) \\ 0 & \text{otherwise} \end{cases}$$

The transition is fully deterministic (one-hot distribution).

---

## 5. Observation Model

### 5.1 Observation Space

$$\mathcal{O} = \{E\_NONE,\ E\_GOOD,\ E\_BAD\} = \{2,\ 1,\ 0\}$$

- **E_NONE** (2): no information (emitted after movement or sample actions).
- **E_GOOD** (1): sensor reports the targeted rock is good.
- **E_BAD** (0): sensor reports the targeted rock is bad.

### 5.2 Sensor Efficiency

The sensor accuracy decays with the **Euclidean distance** \(d\) between the robot and the target rock:

$$\eta(d) = \frac{1 + 2^{-d / d_{1/2}}}{2}$$

where \(d_{1/2}\) is the **half-efficiency distance** (set to 20 in the standard implementation). At distance 0, \(\eta = 1.0\) (perfect). As \(d \to \infty\), \(\eta \to 0.5\) (random guess).

### 5.3 Observation Probabilities

For **movement and sample** actions (\(a \leq \text{SAMPLE}\)):

$$P(o \mid s', a) = \mathbf{1}[o = E\_NONE]$$

For **sensing** action targeting rock \(k\) (\(a = \text{SAMPLE} + 1 + k\)):

$$P(o \mid s', a) = \begin{cases} \eta(d) & \text{if } o = \text{status}_k(s') \text{ (correct reading)} \\ 1 - \eta(d) & \text{if } o \neq \text{status}_k(s') \text{ (incorrect reading)} \end{cases}$$

where \(d = \|pos_{\text{rob}}(s') - pos_k\|_2\).

In code (`ObsProb` in `rock_sample_despot.cpp`):

```cpp
double efficiency = (1 + pow(2, -distance / half_efficiency_distance_)) * 0.5;
return ((GetRock(&rockstate, rock) & 1) == obs) ? efficiency : (1 - efficiency);
```

---

## 6. Reward Model

The reward function \(R(s, a)\) is deterministic:

| Action | Condition | Reward |
|---|---|---|
| Movement (N/S/W) | Legal move (stays in grid) | 0 |
| Movement (N/S/W) | Illegal move (out of bounds) | **−100** |
| Movement (EAST) | Exits east boundary | **+10** |
| Movement (EAST) | Stays in grid | 0 |
| SAMPLE | Rock at position is good | **+10** |
| SAMPLE | Rock at position is bad | **−10** |
| SAMPLE | No rock at position | **−100** |
| Sensing (CHECK\_k) | Always | 0 |

Key observations:
- Rewards incentivize collecting good rocks (+10), penalize bad sampling (−10), and reward exiting the map (+10).
- Large penalties (−100) discourage illegal actions (moving into walls or sampling empty cells).
- Sensing is free (reward = 0), which allows the agent to reduce uncertainty at no cost.

In code (`Reward` in `base_rock_sample.cpp`):

```cpp
if (a < E_SAMPLE) {
    // movement
    if (grid_.Inside(rob_pos + dir))  return 0;
    return rob_pos.x == N ? 10 : -100;
} else if (a == E_SAMPLE) {
    return (rock >= 0) ? (valuable ? 10 : -10) : -100;
} else {
    return 0;  // sensing
}
```

---

## 7. Initial Belief

### 7.1 Robot Position

The robot always starts at a fixed position on the **west side** of the grid:
- Default: `start_pos = Coord(0, size/2)` (left column, middle row).
- For named configurations, fixed starting positions are used (e.g., `(0,3)` for 7×8).

### 7.2 Rock Goodness

Each rock's initial goodness is **independent and uniformly random**:

$$P(\text{status}_k = 1) = P(\text{status}_k = 0) = 0.5 \quad \forall k$$

### 7.3 Particle Representation

The initial belief is represented as a **uniform particle set** over all \(2^K\) rock configurations, all at the start position:

$$b_0 = \text{Uniform over } \{ (pos_{\text{start}},\ \text{rock\_bits}) : \text{rock\_bits} \in \{0,\ldots,2^K-1\} \}$$

Each particle has weight \(1 / 2^K\).

---

## 8. Objective

Maximize the expected **discounted cumulative reward**:

$$\max_\pi\ \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \right]$$

with discount factor \(\gamma \in (0, 1)\). The episode terminates when the robot exits the east boundary.

---

## 9. Value Bounds for POMDP Solvers

### 9.1 Upper Bounds

Three upper bound strategies are implemented:

| Name | Description |
|---|---|
| `UB1` | Assumes all good rocks can be sampled immediately: \(V \leq 10 \sum_{k} \text{good}_k \cdot \gamma^k\) |
| `UB2` | Discounts each good rock by Manhattan distance from robot, plus exit reward |
| `MDP` (default) | Optimal MDP value (ignores partial observability, assumes full state knowledge) |
| `APPROX` | Greedy nearest-good-rock traversal upper bound |

### 9.2 Lower Bounds

| Name | Description |
|---|---|
| `EAST` (default) | Always go east: guaranteed to exit and get +10 |
| `ENT` | Explore Nearest in Thresholded state: greedily head to nearest rock with positive expected value |
| `MMAP` | Uses the maximum marginal a posteriori state to compute an optimistic sampling plan |

The **EAST** lower bound gives:
$$V_{\text{east}} = 10 \cdot \gamma^{N - x_{\text{rob}}}$$

---

## 10. MDP Relaxation

Because sensing does not change state, the full observability (MDP) relaxation only considers the 5 non-sensing actions: NORTH, EAST, SOUTH, WEST, SAMPLE. The optimal MDP policy is computed via **value iteration**:

$$V^*(s) = \max_{a \in \{0,\ldots,4\}} \left[ R(s,a) + \gamma V^*(T(s,a)) \right]$$

convergence threshold: \(\|V_{t+1} - V_t\|_1 < 0.001\).

This MDP value serves as the primary upper bound for online POMDP solvers (DESPOT, POMCP).

---

## 11. Implementations in This Repository

The rock sample problem is implemented across several solver backends:

| Directory | Solver | Key Files |
|---|---|---|
| `src/base/` | Shared base model | `base_rock_sample.h`, `base_rock_sample.cpp` |
| `src/rock_sample_despot/` | DESPOT | `rock_sample_despot.h`, `rock_sample_despot.cpp` |
| `src/rock_sample_dynamic_vecqmdp/` | VecQMDP | `rock_sample_dynamic_vecqmdp.hpp`, `rock_sample_dynamic_vecqmdp.cpp` |
| `src/fvrs/` | FVRS | `fvrs.h`, `fvrs.cpp` |

The `BaseRockSample` class provides the shared state encoding, transition/reward functions, belief management, and bounds. Solver-specific subclasses override `Step()`, `ObsProb()`, `NumActions()`, and `PrintObs()`.

---

## 12. References

- Smith, T. & Simmons, R. (2004). *Heuristic Search Value Iteration for POMDPs*. UAI.
- Ye, N., Somani, A., Hsu, D., & Lee, W. S. (2017). *DESPOT: Online POMDP Planning with Regularization*. JAIR.
