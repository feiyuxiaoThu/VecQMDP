/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <collision/STRtree.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#ifdef ARCH_X86_64
#include <immintrin.h>
#elif defined(ARCH_AARCH64)
#include <arm_neon.h>
#endif
#include <atomic>

// Highway SIMD sort library
#include "hwy/base.h"                // hwy::K32V32
#include "hwy/contrib/sort/vqsort.h" // hwy::VQSort, hwy::SortAscending, hwy::SortDescending

// Convert float to a uint32 that preserves IEEE-754 total order (handles negatives).
// Positive floats: flip sign bit → 0x80000000..0xFFFFFFFF
// Negative floats: flip all bits  → 0x00000000..0x7FFFFFFF
static inline uint32_t float_to_sort_key(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(bits >> 31)) | 0x80000000u;
    return bits ^ mask;
}

// Inverse of float_to_sort_key.
static inline float sort_key_to_float(uint32_t key)
{
    uint32_t mask = ((key >> 31) - 1u) | 0x80000000u;
    uint32_t bits = key ^ mask;
    float    f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Sort (keys[], values[]) together by key using highway VQSort.
// Both arrays are rearranged in-place to reflect the sorted order.
void perform_sort(float *keys, int *values, size_t size, bool descending = false)
{
    std::vector<hwy::K32V32> pairs(size);
    for (size_t i = 0; i < size; ++i)
    {
        pairs[i].key = float_to_sort_key(keys[i]);
        pairs[i].value = static_cast<uint32_t>(values[i]);
    }
    if (descending)
    {
        hwy::VQSort(pairs.data(), size, hwy::SortDescending{});
    }
    else
    {
        hwy::VQSort(pairs.data(), size, hwy::SortAscending{});
    }
    for (size_t i = 0; i < size; ++i)
    {
        keys[i] = sort_key_to_float(pairs[i].key);
        values[i] = static_cast<int>(pairs[i].value);
    }
}

namespace vec_qmdp
{
    namespace collision
    {
        STRtree::STRtree(int node_capacity) : node_capacity_(node_capacity)
        {
            // Preallocate array space (use tests array directly, no need for nodes array)
            tmp_min_x_.resize(max_node_num_);
            tmp_min_y_.resize(max_node_num_);
            tmp_max_x_.resize(max_node_num_);
            tmp_max_y_.resize(max_node_num_);
            tmp_center_x_.resize(max_node_num_);
            tmp_center_y_.resize(max_node_num_);
            tmp_polygon_idxs_.resize(max_node_num_);
            tmp_sort_indices_.resize(max_node_num_);
            temp_keys_y_.resize(max_node_num_);

            tests_min_x_.resize(TOTAL_TREE_SIZE);
            tests_min_y_.resize(TOTAL_TREE_SIZE);
            tests_max_x_.resize(TOTAL_TREE_SIZE);
            tests_max_y_.resize(TOTAL_TREE_SIZE);
            polygon_idxs_.resize(TOTAL_LEAF_SIZE);
        }

        void STRtree::clear()
        {
            // Clear but retain allocated memory capacity
            leaf_count_ = 0;
        }

        void STRtree::insert(float min_x, float min_y, float max_x, float max_y, float center_x, float center_y,
                             int polygon_index)
        {
            // Optimization: insert directly at indices [0, leaf_count_), defer offset to build phase
            tmp_min_x_[leaf_count_] = min_x;
            tmp_min_y_[leaf_count_] = min_y;
            tmp_max_x_[leaf_count_] = max_x;
            tmp_max_y_[leaf_count_] = max_y;
            tmp_center_x_[leaf_count_] = center_x;
            tmp_center_y_[leaf_count_] = center_y;
            tmp_polygon_idxs_[leaf_count_] = polygon_index;
            ++leaf_count_;
        }

        void STRtree::insertBatch(const FVectorT_1 &min_x, const FVectorT_1 &min_y, const FVectorT_1 &max_x,
                                  const FVectorT_1 &max_y, const FVectorT_1 &center_x, const FVectorT_1 &center_y,
                                  const IVectorT_1 &polygon_indices, int valid_count)
        {
            min_x.to_array(tmp_min_x_.data() + leaf_count_);
            min_y.to_array(tmp_min_y_.data() + leaf_count_);
            max_x.to_array(tmp_max_x_.data() + leaf_count_);
            max_y.to_array(tmp_max_y_.data() + leaf_count_);
            center_x.to_array(tmp_center_x_.data() + leaf_count_);
            center_y.to_array(tmp_center_y_.data() + leaf_count_);
            polygon_indices.to_array(tmp_polygon_idxs_.data() + leaf_count_);

            leaf_count_ += valid_count;
        }

        void STRtree::build(int thread_id)
        {
            // If leaf_count_ is empty, return directly
            if (leaf_count_ == 0)
                return;

            // 1. Determine whether to use dual subtree or single subtree
            if (leaf_count_ > SUBTREE_LEAF_SIZE)
            {
                use_dual_subtree_ = true;
                subtree_1_leaf_count_ = SUBTREE_LEAF_SIZE;
                subtree_2_leaf_count_ = leaf_count_ - SUBTREE_LEAF_SIZE;
            }
            else
            {
                use_dual_subtree_ = false;
                subtree_1_leaf_count_ = leaf_count_;
                subtree_2_leaf_count_ = 0;
            }

            // 3. STR sorting and grouping of leaf nodes
            {
                str_sort_and_group();
            }

            {
                // 2. Compute parent nodes (using SIMD optimization)
                if (use_dual_subtree_)
                {
                    // Dual subtree: subtree 1 handles first 64 leaves
                    compute_parent_aabb(0, SUBTREE_INTERNAL_SIZE, subtree_1_leaf_count_);

                    // Subtree 2 handles remaining leaves
                    compute_parent_aabb(SUBTREE_TOTAL_SIZE, SUBTREE_TOTAL_SIZE + SUBTREE_INTERNAL_SIZE,
                                        subtree_2_leaf_count_);
                }
                else
                {
                    // Single subtree
                    compute_parent_aabb(0, SUBTREE_INTERNAL_SIZE, leaf_count_);
                }

                // 3. Build root and sub_root bounds (handle case with fewer than 8 parents)
                FVectorT_1 min_x_v, min_y_v, max_x_v, max_y_v;

                if (use_dual_subtree_)
                {
                    subtree_1_parent_count_ = utils::STRTREE_NODE_CAPACITY;
                    subtree_2_parent_count_ = (subtree_2_leaf_count_ + node_capacity_ - 1) / node_capacity_;

                    // Subtree 1 bounds (full 8 parents)
                    min_x_v = FVectorT_1::load_contiguous(tests_min_x_.data(), 0);
                    min_y_v = FVectorT_1::load_contiguous(tests_min_y_.data(), 0);
                    max_x_v = FVectorT_1::load_contiguous(tests_max_x_.data(), 0);
                    max_y_v = FVectorT_1::load_contiguous(tests_max_y_.data(), 0);

                    sub_root_1_min_x_ = min_x_v.hmin();
                    sub_root_1_min_y_ = min_y_v.hmin();
                    sub_root_1_max_x_ = max_x_v.hmax();
                    sub_root_1_max_y_ = max_y_v.hmax();

                    // Subtree 2 bounds
                    // Fewer than 8 elements: scalar operations (max 96 elements, second subtree max 32, will not be
                    // full)
                    sub_root_2_min_x_ = tests_min_x_[SUBTREE_TOTAL_SIZE];
                    sub_root_2_min_y_ = tests_min_y_[SUBTREE_TOTAL_SIZE];
                    sub_root_2_max_x_ = tests_max_x_[SUBTREE_TOTAL_SIZE];
                    sub_root_2_max_y_ = tests_max_y_[SUBTREE_TOTAL_SIZE];

                    for (size_t i = 1; i < subtree_2_parent_count_; ++i)
                    {
                        sub_root_2_min_x_ = std::min(sub_root_2_min_x_, tests_min_x_[SUBTREE_TOTAL_SIZE + i]);
                        sub_root_2_min_y_ = std::min(sub_root_2_min_y_, tests_min_y_[SUBTREE_TOTAL_SIZE + i]);
                        sub_root_2_max_x_ = std::max(sub_root_2_max_x_, tests_max_x_[SUBTREE_TOTAL_SIZE + i]);
                        sub_root_2_max_y_ = std::max(sub_root_2_max_y_, tests_max_y_[SUBTREE_TOTAL_SIZE + i]);
                    }
                }
                else
                {
                    subtree_1_parent_count_ = (leaf_count_ + node_capacity_ - 1) / node_capacity_;
                    subtree_2_parent_count_ = 0;

                    // Single subtree
                    if (subtree_1_parent_count_ == SUBTREE_INTERNAL_SIZE)
                    {
                        // Full 8 parents: use SIMD
                        min_x_v = FVectorT_1::load_contiguous(tests_min_x_.data(), 0);
                        min_y_v = FVectorT_1::load_contiguous(tests_min_y_.data(), 0);
                        max_x_v = FVectorT_1::load_contiguous(tests_max_x_.data(), 0);
                        max_y_v = FVectorT_1::load_contiguous(tests_max_y_.data(), 0);

                        sub_root_1_min_x_ = min_x_v.hmin();
                        sub_root_1_min_y_ = min_y_v.hmin();
                        sub_root_1_max_x_ = max_x_v.hmax();
                        sub_root_1_max_y_ = max_y_v.hmax();
                    }
                    else
                    {
                        // Fewer than 8 parents: scalar operations
                        sub_root_1_min_x_ = tests_min_x_[0];
                        sub_root_1_min_y_ = tests_min_y_[0];
                        sub_root_1_max_x_ = tests_max_x_[0];
                        sub_root_1_max_y_ = tests_max_y_[0];

                        for (size_t i = 1; i < subtree_1_parent_count_; ++i)
                        {
                            sub_root_1_min_x_ = std::min(sub_root_1_min_x_, tests_min_x_[i]);
                            sub_root_1_min_y_ = std::min(sub_root_1_min_y_, tests_min_y_[i]);
                            sub_root_1_max_x_ = std::max(sub_root_1_max_x_, tests_max_x_[i]);
                            sub_root_1_max_y_ = std::max(sub_root_1_max_y_, tests_max_y_[i]);
                        }
                    }
                }
            }
        }

        // STR sort and group: optimized version, no offset calculation needed
        void STRtree::str_sort_and_group()
        {
            {
                // Use int instead of size_t to avoid 64-bit index overhead
                std::iota(tmp_sort_indices_.begin(), tmp_sort_indices_.begin() + leaf_count_, 0);

                // 1. Sort by X axis: use SIMD-accelerated keyvalue_qsort
                if (leaf_count_ <= static_cast<size_t>(node_capacity_))
                {
                    // For very small arrays, insertion sort is still fastest (avoids function call overhead)
                    for (size_t i = 1; i < leaf_count_; ++i)
                    {
                        int    key = tmp_sort_indices_[i];
                        float  key_x = tmp_center_x_[key];
                        size_t j = i;
                        while (j > 0 && tmp_center_x_[tmp_sort_indices_[j - 1]] > key_x)
                        {
                            tmp_sort_indices_[j] = tmp_sort_indices_[j - 1];
                            --j;
                        }
                        tmp_sort_indices_[j] = key;
                    }
                }
                else
                {
                    perform_sort(tmp_center_x_.data(), tmp_sort_indices_.data(), leaf_count_,
                                 false // ascending order
                    );
                }

                // 2. Compute slice count (key to STR algorithm)
                size_t slice_count = (size_t)std::ceil(std::sqrt((float)leaf_count_ / (float)node_capacity_));
                size_t items_per_slice = (size_t)std::ceil((float)leaf_count_ / (float)slice_count);

                // 3. Sort by y within each x-slice (also using SIMD acceleration)
                for (size_t i = 0; i < leaf_count_; i += items_per_slice)
                {
                    size_t end_idx = std::min(i + items_per_slice, leaf_count_);
                    size_t slice_size = end_idx - i;

                    if (slice_size <= static_cast<size_t>(node_capacity_))
                    {
                        // Insertion sort (faster for small arrays)
                        for (size_t k = i + 1; k < end_idx; ++k)
                        {
                            int    key = tmp_sort_indices_[k];
                            float  key_y = tmp_center_y_[key];
                            size_t j = k;
                            while (j > i && tmp_center_y_[tmp_sort_indices_[j - 1]] > key_y)
                            {
                                tmp_sort_indices_[j] = tmp_sort_indices_[j - 1];
                                --j;
                            }
                            tmp_sort_indices_[j] = key;
                        }
                    }
                    else
                    {
                        size_t       k = 0;
                        const size_t simd_end = (slice_size / FloatVectorWidth) * FloatVectorWidth;

                        for (; k < simd_end; k += FloatVectorWidth)
                        {

                            IVectorT_1 idx_vec = IVectorT_1::load_contiguous_unaligned(tmp_sort_indices_.data(), i + k);

                            FVectorT_1 center_y_vec = FVectorT_1::gather(tmp_center_y_.data(), idx_vec);

                            center_y_vec.to_array(temp_keys_y_.data() + k);
                        }

                        // Handle remaining elements (scalar)
                        for (; k < slice_size; ++k)
                        {
                            temp_keys_y_[k] = tmp_center_y_[tmp_sort_indices_[i + k]];
                        }

                        perform_sort(temp_keys_y_.data(), tmp_sort_indices_.data() + i, slice_size,
                                     false // ascending order
                        );
                    }
                }
            }

            // 5. Reorder by indices and write directly to the correct positions in the tests array
            {
                if (use_dual_subtree_)
                {
                    // Dual subtree mode: handle each subtree separately
                    // Subtree 1: tmp_sort_indices_[0, 64) -> tests[SUBTREE_INTERNAL_SIZE, ...]
                    size_t       i = 0;
                    const size_t simd_end_1 = (SUBTREE_LEAF_SIZE / FloatVectorWidth) * FloatVectorWidth;

                    for (; i < simd_end_1; i += FloatVectorWidth)
                    {
                        IVectorT_1 idx_vec = IVectorT_1::load_contiguous(tmp_sort_indices_.data(), i);
                        FVectorT_1 min_x_vec = FVectorT_1::gather(tmp_min_x_.data(), idx_vec);
                        FVectorT_1 min_y_vec = FVectorT_1::gather(tmp_min_y_.data(), idx_vec);
                        FVectorT_1 max_x_vec = FVectorT_1::gather(tmp_max_x_.data(), idx_vec);
                        FVectorT_1 max_y_vec = FVectorT_1::gather(tmp_max_y_.data(), idx_vec);

                        min_x_vec.to_array(tests_min_x_.data() + write_offset_subtree1_ + i);
                        min_y_vec.to_array(tests_min_y_.data() + write_offset_subtree1_ + i);
                        max_x_vec.to_array(tests_max_x_.data() + write_offset_subtree1_ + i);
                        max_y_vec.to_array(tests_max_y_.data() + write_offset_subtree1_ + i);
                    }

                    // Handle remaining elements of subtree 1
                    for (; i < SUBTREE_LEAF_SIZE; ++i)
                    {
                        int src_idx = tmp_sort_indices_[i];
                        tests_min_x_[write_offset_subtree1_ + i] = tmp_min_x_[src_idx];
                        tests_min_y_[write_offset_subtree1_ + i] = tmp_min_y_[src_idx];
                        tests_max_x_[write_offset_subtree1_ + i] = tmp_max_x_[src_idx];
                        tests_max_y_[write_offset_subtree1_ + i] = tmp_max_y_[src_idx];
                    }

                    // Subtree 2: tmp_sort_indices_[64, leaf_count_) -> tests[SUBTREE_TOTAL_SIZE+SUBTREE_INTERNAL_SIZE,
                    // ...]
                    i = SUBTREE_LEAF_SIZE;
                    const size_t simd_end_2 =
                        SUBTREE_LEAF_SIZE + ((leaf_count_ - SUBTREE_LEAF_SIZE) / FloatVectorWidth) * FloatVectorWidth;

                    for (; i < simd_end_2; i += FloatVectorWidth)
                    {
                        IVectorT_1 idx_vec = IVectorT_1::load_contiguous(tmp_sort_indices_.data(), i);
                        FVectorT_1 min_x_vec = FVectorT_1::gather(tmp_min_x_.data(), idx_vec);
                        FVectorT_1 min_y_vec = FVectorT_1::gather(tmp_min_y_.data(), idx_vec);
                        FVectorT_1 max_x_vec = FVectorT_1::gather(tmp_max_x_.data(), idx_vec);
                        FVectorT_1 max_y_vec = FVectorT_1::gather(tmp_max_y_.data(), idx_vec);

                        size_t write_idx = write_offset_subtree2_ + (i - SUBTREE_LEAF_SIZE);
                        min_x_vec.to_array(tests_min_x_.data() + write_idx);
                        min_y_vec.to_array(tests_min_y_.data() + write_idx);
                        max_x_vec.to_array(tests_max_x_.data() + write_idx);
                        max_y_vec.to_array(tests_max_y_.data() + write_idx);
                    }

                    // Handle remaining elements of subtree 2
                    for (; i < leaf_count_; ++i)
                    {
                        int    src_idx = tmp_sort_indices_[i];
                        size_t write_idx = write_offset_subtree2_ + (i - SUBTREE_LEAF_SIZE);
                        tests_min_x_[write_idx] = tmp_min_x_[src_idx];
                        tests_min_y_[write_idx] = tmp_min_y_[src_idx];
                        tests_max_x_[write_idx] = tmp_max_x_[src_idx];
                        tests_max_y_[write_idx] = tmp_max_y_[src_idx];
                    }
                }
                else
                {
                    // Single subtree mode: write directly
                    size_t       i = 0;
                    const size_t simd_end = (leaf_count_ / FloatVectorWidth) * FloatVectorWidth;

                    for (; i < simd_end; i += FloatVectorWidth)
                    {
                        IVectorT_1 idx_vec = IVectorT_1::load_contiguous(tmp_sort_indices_.data(), i);
                        FVectorT_1 min_x_vec = FVectorT_1::gather(tmp_min_x_.data(), idx_vec);
                        FVectorT_1 min_y_vec = FVectorT_1::gather(tmp_min_y_.data(), idx_vec);
                        FVectorT_1 max_x_vec = FVectorT_1::gather(tmp_max_x_.data(), idx_vec);
                        FVectorT_1 max_y_vec = FVectorT_1::gather(tmp_max_y_.data(), idx_vec);

                        min_x_vec.to_array(tests_min_x_.data() + write_offset_subtree1_ + i);
                        min_y_vec.to_array(tests_min_y_.data() + write_offset_subtree1_ + i);
                        max_x_vec.to_array(tests_max_x_.data() + write_offset_subtree1_ + i);
                        max_y_vec.to_array(tests_max_y_.data() + write_offset_subtree1_ + i);
                    }

                    // Handle remaining elements
                    for (; i < leaf_count_; ++i)
                    {
                        int src_idx = tmp_sort_indices_[i];
                        tests_min_x_[write_offset_subtree1_ + i] = tmp_min_x_[src_idx];
                        tests_min_y_[write_offset_subtree1_ + i] = tmp_min_y_[src_idx];
                        tests_max_x_[write_offset_subtree1_ + i] = tmp_max_x_[src_idx];
                        tests_max_y_[write_offset_subtree1_ + i] = tmp_max_y_[src_idx];
                    }
                }

                // polygon_idxs are also written via direct gather
                size_t       i = 0;
                const size_t simd_end = (leaf_count_ / FloatVectorWidth) * FloatVectorWidth;
                for (; i < simd_end; i += FloatVectorWidth)
                {
                    IVectorT_1 idx_vec = IVectorT_1::load_contiguous(tmp_sort_indices_.data(), i);
                    IVectorT_1 poly_vec = IVectorT_1::gather(tmp_polygon_idxs_.data(), idx_vec);
                    poly_vec.to_array(polygon_idxs_.data() + i);
                }
                for (; i < leaf_count_; ++i)
                {
                    polygon_idxs_[i] = tmp_polygon_idxs_[tmp_sort_indices_[i]];
                }
            }
        }

        // SIMD computation of parent AABB (handles cases with fewer than 8 elements)
        void STRtree::compute_parent_aabb(size_t parent_idx, size_t leaf_idx, size_t leaf_count)
        {
            size_t parent_count = leaf_count / node_capacity_;
            size_t remaining_num = leaf_count % node_capacity_;

            // Parallel mode: use SIMD for groups of exactly 8 elements
            for (size_t i = 0; i < parent_count; ++i, leaf_idx += node_capacity_, ++parent_idx)
            {
                FVectorT_1 child_min_x = FVectorT_1::load_contiguous(tests_min_x_.data(), leaf_idx);
                FVectorT_1 child_min_y = FVectorT_1::load_contiguous(tests_min_y_.data(), leaf_idx);
                FVectorT_1 child_max_x = FVectorT_1::load_contiguous(tests_max_x_.data(), leaf_idx);
                FVectorT_1 child_max_y = FVectorT_1::load_contiguous(tests_max_y_.data(), leaf_idx);

                tests_min_x_[parent_idx] = child_min_x.hmin();
                tests_min_y_[parent_idx] = child_min_y.hmin();
                tests_max_x_[parent_idx] = child_max_x.hmax();
                tests_max_y_[parent_idx] = child_max_y.hmax();
            }

            // For fewer than 8 elements: use scalar operations
            if (remaining_num > 0)
            {
                float min_x = tests_min_x_[leaf_idx];
                float min_y = tests_min_y_[leaf_idx];
                float max_x = tests_max_x_[leaf_idx];
                float max_y = tests_max_y_[leaf_idx];

                size_t end_idx = leaf_idx + remaining_num;
                for (++leaf_idx; leaf_idx < end_idx; ++leaf_idx)
                {
                    min_x = std::min(min_x, tests_min_x_[leaf_idx]);
                    min_y = std::min(min_y, tests_min_y_[leaf_idx]);
                    max_x = std::max(max_x, tests_max_x_[leaf_idx]);
                    max_y = std::max(max_y, tests_max_y_[leaf_idx]);
                }
                tests_min_x_[parent_idx] = min_x;
                tests_min_y_[parent_idx] = min_y;
                tests_max_x_[parent_idx] = max_x;
                tests_max_y_[parent_idx] = max_y;
            }
        }

        bool STRtree::intersects(float min_x, float min_y, float max_x, float max_y, float target_min_x,
                                 float target_min_y, float target_max_x, float target_max_y) const
        {
            // Check whether two AABBs intersect
            return !(max_x < target_min_x || min_x > target_max_x || max_y < target_min_y || min_y > target_max_y);
        }

        STRtree::FVectorT_1 STRtree::intersectsBatch(float min_x, float min_y, float max_x, float max_y,
                                                     const size_t &batch_start) const
        {
            // Prefetch the next batch of data into cache (assuming sequential queries)
            __builtin_prefetch(&tests_min_x_[batch_start + FloatVectorWidth], 0, 3);
            __builtin_prefetch(&tests_max_x_[batch_start + FloatVectorWidth], 0, 3);
            __builtin_prefetch(&tests_min_y_[batch_start + FloatVectorWidth], 0, 3);
            __builtin_prefetch(&tests_max_y_[batch_start + FloatVectorWidth], 0, 3);

            // Use temporary variables to allow the compiler to better schedule load instructions
            // Modern CPUs can execute multiple memory loads in parallel
            const FVectorT_1 node_max_x = FVectorT_1::load_contiguous(tests_max_x_.data(), batch_start);
            const FVectorT_1 node_min_x = FVectorT_1::load_contiguous(tests_min_x_.data(), batch_start);
            const FVectorT_1 node_max_y = FVectorT_1::load_contiguous(tests_max_y_.data(), batch_start);
            const FVectorT_1 node_min_y = FVectorT_1::load_contiguous(tests_min_y_.data(), batch_start);

            // Separate comparison operations to exploit CPU instruction-level parallelism
            const FVectorT_1 cmp_x = (node_max_x >= min_x) & (node_min_x <= max_x);
            const FVectorT_1 cmp_y = (node_max_y >= min_y) & (node_min_y <= max_y);

            return cmp_x & cmp_y;
        }

    } // namespace collision
} // namespace vec_qmdp
