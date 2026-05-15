/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file STRtree.hpp
 * @brief SIMD-accelerated STR-tree for fast 2D spatial intersection queries using SoA layout.
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <utils/aligned_allocator.hpp>
#include <utils/geometry_utils.hpp>
#include <utils/global_utils.hpp>
#include <vamp/collision/math.hh>
#include <vamp/vector.hh>
#include <vector>

namespace vec_qmdp
{
    static constexpr std::size_t FloatVectorWidth = vamp::FloatVectorWidth;
    namespace collision
    {
        // Declare AlignedAllocator
        using AlignedVectorFloat = utils::AlignedVectorFloat;
        using AlignedVectorInt = utils::AlignedVectorInt;

        // STRtree (Sorted Tile Recursive Tree) for spatial indexing with SIMD support
        class STRtree
        {
          public:
            using FVectorT_1 = utils::FVectorT_1;
            using IVectorT_1 = utils::IVectorT_1;
            using Point = utils::Point; // 2D point
            // using AABB = std::pair<Point, Point>; // min, max 2Dpoints

            // Constructor with node capacity
            explicit STRtree(int node_capacity = utils::STRTREE_NODE_CAPACITY);

            // Default move operations (efficient)
            STRtree(STRtree &&) noexcept = default;
            STRtree &operator=(STRtree &&) noexcept = default;

            // Disable copy (to avoid accidental expensive copies)
            STRtree(const STRtree &) = delete;
            STRtree &operator=(const STRtree &) = delete;

            // Insert a polygon's AABB with its index
            void insert(float min_x, float min_y, float max_x, float max_y, float center_x, float center_y,
                        int polygon_index);

            // Insert multiple AABBs at once using SIMD
            void insertBatch(const FVectorT_1 &min_x, const FVectorT_1 &min_y, const FVectorT_1 &max_x,
                             const FVectorT_1 &max_y, const FVectorT_1 &center_x, const FVectorT_1 &center_y,
                             const IVectorT_1 &polygon_indices, int valid_count = FVectorT_1::num_scalars);

            // Build the tree structure
            void build(int thread_id = -1);

            // Clear and prepare for reuse (faster than recreating)
            void clear();

            // Query polygons that might intersect with target AABB
            template <int CANDIDATE_SIZE>
            std::array<int, CANDIDATE_SIZE> queryBatch(float min_x, float min_y, float max_x, float max_y,
                                                       int &candidate_size, bool print = false) const
            {
                static thread_local std::array<int, CANDIDATE_SIZE> results{utils::DUMMY_TIME_STEPS - 1};
                results.fill(utils::DUMMY_TIME_STEPS - 1);
                candidate_size = 0;

                if (leaf_count_ == 0)
                    return results;

                // Pre-compute intersection results for both subtrees to avoid branch mispredictions
                const bool intersects_1 = intersects(min_x, min_y, max_x, max_y, sub_root_1_min_x_, sub_root_1_min_y_,
                                                     sub_root_1_max_x_, sub_root_1_max_y_);
                const bool intersects_2 =
                    use_dual_subtree_ && intersects(min_x, min_y, max_x, max_y, sub_root_2_min_x_, sub_root_2_min_y_,
                                                    sub_root_2_max_x_, sub_root_2_max_y_);

                // Parallel mode: use optimized variant
                if (intersects_1)
                {
                    querySubtreeBatch<CANDIDATE_SIZE>(0, min_x, min_y, max_x, max_y, results, candidate_size, print);
                }

                if (intersects_2)
                {
                    querySubtreeBatch<CANDIDATE_SIZE>(1, min_x, min_y, max_x, max_y, results, candidate_size, print);
                }

                // Optimized sort: use insertion sort for small datasets, eliminating unnecessary checks
                if (candidate_size <= utils::STRTREE_UNROLLED_SORT_THRESHOLD && candidate_size > 1)
                {
                    // Manually unrolled insertion sort to avoid loop overhead
                    if (results[0] > results[1])
                        std::swap(results[0], results[1]);
                    if (candidate_size == utils::STRTREE_UNROLLED_SORT_THRESHOLD)
                    {
                        int key = results[2];
                        if (key < results[1])
                        {
                            results[2] = results[1];
                            if (key < results[0])
                            {
                                results[1] = results[0];
                                results[0] = key;
                            }
                            else
                            {
                                results[1] = key;
                            }
                        }
                    }
                }
                else
                {
                    std::sort(results.begin(), results.begin() + candidate_size);
                }

                return results;
            }

            // Add getter methods to access internal data
            // const STRnode& get_root() const { return root_; }
            const AlignedVectorFloat &get_tests_min_x() const { return tests_min_x_; }
            const AlignedVectorFloat &get_tests_min_y() const { return tests_min_y_; }
            const AlignedVectorFloat &get_tests_max_x() const { return tests_max_x_; }
            const AlignedVectorFloat &get_tests_max_y() const { return tests_max_y_; }

          private:
            // Buffers for inserting nodes
            AlignedVectorFloat tmp_min_x_, tmp_min_y_, tmp_max_x_, tmp_max_y_, tmp_center_x_, tmp_center_y_;
            AlignedVectorInt   tmp_polygon_idxs_;
            AlignedVectorInt   tmp_sort_indices_;
            AlignedVectorFloat temp_keys_y_;

            // Test buffer storing AABB bounds for each node
            AlignedVectorFloat tests_min_x_, tests_min_y_, tests_max_x_, tests_max_y_;
            AlignedVectorInt   polygon_idxs_; // polygon index array

            // STRnode storage
            float sub_root_1_min_x_, sub_root_1_min_y_, sub_root_1_max_x_,
                sub_root_1_max_y_; // root node of the first subtree in the dual-subtree layout
            float sub_root_2_min_x_, sub_root_2_min_y_, sub_root_2_max_x_,
                sub_root_2_max_y_; // root node of the second subtree in the dual-subtree layout

            // Leaf node count (operate directly on tests arrays; no separate nodes_ array required)
            size_t leaf_count_{0};

            // Configuration
            const int node_capacity_{utils::STRTREE_NODE_CAPACITY};
            size_t    internal_nodes_num_{0};
            size_t    total_nodes_num_{0};
            size_t    vector_size_{FVectorT_1::num_scalars};
            size_t    max_node_num_{utils::STRTREE_MAX_NODE_NUM};

            // Dual-subtree configuration
            static constexpr size_t MAX_SINGLE_SUBTREE_NODES = utils::STRTREE_SUBTREE_LEAF_SIZE;
            static constexpr size_t SUBTREE_INTERNAL_SIZE = utils::STRTREE_SUBTREE_INTERNAL_SIZE;
            static constexpr size_t SUBTREE_LEAF_SIZE = utils::STRTREE_SUBTREE_LEAF_SIZE;
            static constexpr size_t SUBTREE_TOTAL_SIZE = utils::STRTREE_SUBTREE_TOTAL_SIZE;
            static constexpr size_t TOTAL_LEAF_SIZE = utils::STRTREE_TOTAL_LEAF_SIZE;
            static constexpr size_t TOTAL_TREE_SIZE = utils::STRTREE_TOTAL_TREE_SIZE;
            bool                    use_dual_subtree_{false};
            size_t                  subtree_1_parent_count_{0};
            size_t                  subtree_2_parent_count_{0};
            size_t                  subtree_1_leaf_count_{0}; // actual leaf count of subtree 1
            size_t                  subtree_2_leaf_count_{0}; // actual leaf count of subtree 2
            size_t                  write_offset_subtree1_{SUBTREE_INTERNAL_SIZE};
            size_t                  write_offset_subtree2_{SUBTREE_TOTAL_SIZE + SUBTREE_INTERNAL_SIZE};

            // Helper functions
            void str_sort_and_group();                                                       // STR sort and group
            void compute_parent_aabb(size_t parent_idx, size_t leaf_idx, size_t leaf_count); // SIMD compute parent AABB

            bool intersects(float min_x, float min_y, float max_x, float max_y, float target_min_x, float target_min_y,
                            float target_max_x, float target_max_y) const;
            // bool intersects(const AABB& box1, const AABB& box2) const;

            FVectorT_1 intersectsBatch(float min_x, float min_y, float max_x, float max_y,
                                       const size_t &batch_start) const;

            // Aggressive optimized version: fully vectorized to reduce branching
            template <int CANDIDATE_SIZE>
            void querySubtreeBatch(size_t subtree_id, float min_x, float min_y, float max_x, float max_y,
                                   std::array<int, CANDIDATE_SIZE> &results, int &candidate_size,
                                   bool print = false) const
            {
                const int tests_offset = subtree_id * SUBTREE_TOTAL_SIZE;
                const int leaf_offset = subtree_id * SUBTREE_LEAF_SIZE;
                const int actual_leaf_count = (subtree_id == 0) ? subtree_1_leaf_count_ : subtree_2_leaf_count_;
                const int parent_size = subtree_id == 0 ? subtree_1_parent_count_ : subtree_2_parent_count_;

                // 1. Batch-check the parent layer
                const FVectorT_1 parent_intersects = intersectsBatch(min_x, min_y, max_x, max_y, tests_offset);

                if (parent_intersects.none())
                    return;

                const int leaf_tests_offset = tests_offset + SUBTREE_INTERNAL_SIZE;

                // 2. Pre-compute parent mask to reduce redundant checks
                // Extract mask using bitwise operations to avoid repeated array accesses
                unsigned int parent_mask = 0;
                // Under AVX2, movemask extracts all comparison results in one instruction
                parent_mask = _mm256_movemask_ps(parent_intersects.data[0]);

                // 3. Quickly traverse intersecting parents using bitwise operations
                while (parent_mask != 0 && candidate_size < CANDIDATE_SIZE)
                {
                    // Find the lowest set bit (i.e., the next intersecting parent)
                    const int parent_idx = __builtin_ctz(parent_mask);
                    parent_mask &= (parent_mask - 1); // Clear the lowest set bit

                    const int leaf_idx_base = parent_idx * node_capacity_;
                    if (leaf_idx_base >= actual_leaf_count)
                        break;

                    // SIMD batch-check leaves
                    const FVectorT_1 leaf_intersects =
                        intersectsBatch(min_x, min_y, max_x, max_y, leaf_tests_offset + leaf_idx_base);

                    if (leaf_intersects.none())
                        continue;

                    // Extract leaf mask
                    unsigned int leaf_mask = 0;
                    leaf_mask = _mm256_movemask_ps(leaf_intersects.data[0]);

                    // Quickly extract all intersecting leaf indices
                    const int valid_leaf_count = std::min(node_capacity_, actual_leaf_count - leaf_idx_base);
                    while (leaf_mask != 0 && candidate_size < CANDIDATE_SIZE)
                    {
                        const int leaf_idx = __builtin_ctz(leaf_mask);
                        if (leaf_idx >= valid_leaf_count)
                            break;

                        leaf_mask &= (leaf_mask - 1);
                        results[candidate_size++] = polygon_idxs_[leaf_offset + leaf_idx_base + leaf_idx];
                    }

                    if (candidate_size >= CANDIDATE_SIZE)
                        return;
                }
            }
        };

    } // namespace collision
} // namespace vec_qmdp