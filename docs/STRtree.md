## 1. Functionality Overview

The `STRtree` (Sorted Tile Recursive Tree) is a specialized R-tree variant designed for static or semi-static 2D spatial data. It is primarily used to store **Axis-Aligned Bounding Boxes (AABBs)** and perform fast intersection queries.

### Key Features:
* **SIMD Acceleration:** Utilizes AVX2/NEON intrinsics and the **Google Highway library** (`VQSort`) for parallelized sorting and AABB comparisons.
* **SoA (Structure of Arrays) Layout:** Stores coordinates (`min_x`, `max_x`, etc.) in separate contiguous arrays to maximize cache locality and allow direct loading into SIMD registers.
* **STR Algorithm:** Implements the Sort-Tile-Recursive algorithm, which produces a 100% fill rate in leaf nodes, minimizing the total number of nodes and tree depth.
* **Low Overhead:** Uses a fixed-memory layout to avoid dynamic allocations during the tree-building phase.

---

## 2. Tree Building and Vectorized Construction

The `build()` function transforms raw input data into a hierarchical spatial index.

### The Build Workflow:
1.  **STR Sorting (`str_sort_and_group`):** * The leaf nodes are first sorted by their center X-coordinates using `hwy::VQSort`.
    * The data is divided into "vertical slices." Within each slice, nodes are sorted by their center Y-coordinates.
2.  **Vectorized AABB Computation:**
    * Once leaves are ordered, the `compute_parent_aabb` function calculates the bounding box of parent nodes.
    * It processes **8 nodes at a time** using SIMD instructions (e.g., `hmin` and `hmax` operations) to determine the collective `min` and `max` bounds for the parent layer.

### Why use `use_dual_subtree_`?
The implementation employs a unique **Dual Subtree** layout to optimize memory and traversal efficiency based on a node capacity of **8** (matching the 256-bit SIMD lane width for floats).

* **The Constraint:** A single-tree hierarchy with a branching factor of 8 grows exponentially:
    * **Level 1 (Root):** 1 node.
    * **Level 2 (Internal):** Up to 8 nodes.
    * **Level 3 (Leaf):** Up to $8 \times 8 = 64$ nodes.
* **The Problem:** The system is designed to handle up to **96 leaf nodes**. A single 2-level tree is too small (max 64), while a 3-level tree is overkill, supporting up to 512 nodes and increasing query latency.
* **The Solution:** By using **Dual Subtrees** ($8 \times 8 \times 2$):
    * **Subtree 1** handles a full block of **64** leaves.
    * **Subtree 2** handles the remaining **32** leaves.
    * This keeps the tree depth shallow (2 levels) for all 96 elements, significantly reducing the number of bounding box checks required during a query.

---

## 3. Querying and Vectorized Intersection

The `queryBatch` and `querySubtreeBatch` functions handle spatial range queries, identifying which stored polygons potentially intersect with a target AABB.

### Vectorized Query Mechanism:
The query process is designed to be almost entirely branchless and utilizes heavy SIMD optimization:

1.  **`intersectsBatch`:** This function loads 8 node bounding boxes into SIMD registers simultaneously. It performs 8 parallel intersection tests against the target AABB using vectorized comparison operators (`>=` and `<=`).
2.  **Bitmask Extraction:**
    * The result of the SIMD comparison is converted into an 8-bit integer mask using `_mm256_movemask_ps`. 
    * Each bit in this mask represents whether one of the 8 nodes intersected the target.
3.  **Efficient Traversal:**
    * Instead of iterating through every node with an `if` statement, the code uses the **`__builtin_ctz`** (Count Trailing Zeros) CPU instruction.
    * `ctz` identifies the index of the first "set" bit in the mask, allowing the algorithm to "jump" directly to intersecting nodes and skip non-intersecting ones instantly.
4.  **Two-Tier Check:**
    * The query first checks the **Parent Layer** (8 boxes). 
    * If a parent box intersects, it proceeds to check the **Leaf Layer** associated with that parent (another 8 boxes).
    * This reduces the worst-case search complexity from $O(N)$ to $O(\log_8 N)$.

### Result Sorting:
After identifying candidates, the system uses a manually unrolled **insertion sort** for small result sets (threshold $\le 3$) or `std::sort` for larger sets. This ensures that the returned polygon indices are always ordered for deterministic downstream processing.