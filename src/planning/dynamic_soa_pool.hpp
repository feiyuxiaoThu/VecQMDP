/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file dynamic_soa_pool.hpp
 * @brief Thread-private dynamic Structure-of-Arrays memory pool for VecQMDP tree nodes.
 *
 * Replaces the implicit balanced-tree addressing scheme
 *   child_rel = num_actions * parent_rel + 1 + action
 * with explicit topology stored in a child_base_idx field per node.
 * This limits memory consumption to nodes actually visited during search,
 * making deep trees (depth >= 90) tractable.
 *
 * Layout
 * ------
 *  - Nodes are allocated in fixed-size Chunks of DYN_CHUNK_CAPACITY slots.
 *  - Each Chunk is an SIMD-aligned SoA block covering all BeliefTreeSearch bookkeeping
 *    fields (q_values, visit_counts, depth, topology links, …).
 *  - A global index g encodes: chunk_id = g / DYN_CHUNK_CAPACITY,
 *    slot = g % DYN_CHUNK_CAPACITY.
 *  - The pool is thread-private; no synchronisation primitives are needed.
 *  - Memory is retained across reset() calls to avoid repeated heap traffic.
 *
 * SIMD compatibility
 * ------------------
 *  Within a single chunk all arrays are contiguous and 64-byte aligned,
 *  supporting AVX-512 loads/stores.  Cross-chunk access requires scalar
 *  element-wise reads or gather instructions using the global index.
 */
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        // =========================================================================
        // Constants
        // =========================================================================

        /// Number of node slots per chunk.  Must be a power of two for fast
        /// index decomposition via bit-shift / bit-mask.
        /// 32768 (2^15) reduces heap allocations from ~9 800 to ~305 when
        /// the search visits 10^7 nodes, at ~1.5 MB per chunk.
        static constexpr int32_t DYN_CHUNK_CAPACITY = 32768;

        /// Sentinel value: no children allocated / no parent (root node).
        static constexpr int32_t DYN_INVALID_IDX = -1;

        // =========================================================================
        // SoA Chunk: SIMD-aligned storage for DYN_CHUNK_CAPACITY nodes
        // =========================================================================

        /**
         * @brief One fixed-size block of node storage in Structure-of-Arrays layout.
         *
         * Every field array is individually 64-byte aligned so that any sub-range
         * starting at a 64-byte boundary can be fed directly to AVX-512 instructions.
         * The struct itself is also 64-byte aligned to prevent false sharing when
         * multiple chunks reside in adjacent heap pages.
         */
        struct alignas(64) DynSoAChunk
        {
            // ---- BeliefTreeSearch value/visit bookkeeping ----
            alignas(64) float q_values[DYN_CHUNK_CAPACITY];        ///< running mean Q
            alignas(64) float rewards[DYN_CHUNK_CAPACITY];         ///< immediate reward stored at node
            alignas(64) int32_t visit_counts[DYN_CHUNK_CAPACITY];  ///< N(node)
            alignas(64) float initial_rollout[DYN_CHUNK_CAPACITY]; ///< Initial rollout value (persistent lower bound)

            // ---- Depth metadata ----
            alignas(64) int32_t depth[DYN_CHUNK_CAPACITY];       ///< depth in tree (root = 0)
            alignas(64) int32_t rollout_len[DYN_CHUNK_CAPACITY]; ///< tree_height - depth - 1
            alignas(64) int32_t min_depth[DYN_CHUNK_CAPACITY];   ///< min reachable leaf depth from node
            alignas(64) int32_t max_depth[DYN_CHUNK_CAPACITY];   ///< max reachable leaf depth from node

            // ---- Explicit topology (replaces implicit formula) ----
            alignas(64) int32_t
                child_base_idx[DYN_CHUNK_CAPACITY]; ///< global idx of first child; DYN_INVALID_IDX if unexpanded
            alignas(64) int32_t parent_idx[DYN_CHUNK_CAPACITY];      ///< global idx of parent; DYN_INVALID_IDX for root
            alignas(64) int32_t action_to_reach[DYN_CHUNK_CAPACITY]; ///< action taken from parent to this node

            // ---- Control flags ----
            alignas(64) int32_t active_flags[DYN_CHUNK_CAPACITY];    ///< 0xFFFFFFFF = active, 0 = terminal
            alignas(64) int32_t curr_action_idx[DYN_CHUNK_CAPACITY]; ///< last selected action at this node

            /// Zero-initialise a single slot, then set structural fields.
            inline void init_slot(int32_t slot, int32_t depth_val, int32_t rollout_len_val, int32_t parent_g,
                                  int32_t action, int32_t tree_height) noexcept
            {
                q_values[slot] = 0.0f;
                rewards[slot] = 0.0f;
                visit_counts[slot] = 0;
                initial_rollout[slot] = std::numeric_limits<float>::lowest();
                depth[slot] = depth_val;
                rollout_len[slot] = rollout_len_val;
                min_depth[slot] = tree_height + 1;
                max_depth[slot] = -1;
                child_base_idx[slot] = DYN_INVALID_IDX;
                parent_idx[slot] = parent_g;
                action_to_reach[slot] = action;
                active_flags[slot] = 0xFFFFFFFF;
                curr_action_idx[slot] = 0;
            }
        };

        // =========================================================================
        // DynSoAPool: thread-private pool of DynSoAChunks
        // =========================================================================

        /**
         * @brief Thread-private memory pool built from DynSoAChunk blocks.
         *
         * Allocation is lock-free (a single integer counter) because the pool is
         * owned and accessed exclusively by one thread.  When the current chunk is
         * exhausted a new one is appended; existing chunk pointers remain stable,
         * so there is no need to copy data on growth.
         *
         * Usage pattern per BeliefTreeSearch round:
         *   1. pool.reset()            — discard logical contents, keep chunks.
         *   2. pool.allocate(S)        — allocate S root nodes (one per scenario).
         *   3. pool.init_node(g, …)   — initialise each root.
         *   4. [search loop]
         *      pool.allocate(A)        — A = num_actions child slots per expansion.
         *      pool.init_node(g, …)   — initialise each child.
         */
        class DynSoAPool
        {
          public:
            /// @param initial_chunks  Number of chunks to allocate upfront.
            ///        At DYN_CHUNK_CAPACITY=32768 each chunk is ~1.5 MB.
            ///        16 chunks pre-warm ~512 K nodes (24 MB) and avoid grow()
            ///        calls during the early phase of a 10^7-node search.
            explicit DynSoAPool(int32_t initial_chunks = 16)
            {
                // Reserve enough pointer slots to cover a 10^7-node search without
                // reallocating the chunks_ vector (10^7 / 32768 ≈ 305 chunks).
                chunks_.reserve(static_cast<size_t>(initial_chunks) + 512);
                for (int32_t i = 0; i < initial_chunks; ++i)
                    grow();
            }

            // ----------------------------------------------------------------
            // Allocation
            // ----------------------------------------------------------------

            /**
             * @brief Allocate @p count consecutive node slots.
             * @return Global index of the first allocated slot.
             *
             * Complexity: O(1) amortised (chunk growth is rare and O(1) amortised).
             * Thread safety: none — call only from the owning thread.
             */
            int32_t allocate(int32_t count)
            {
                if (count <= 0)
                    return DYN_INVALID_IDX;

                const int32_t start_g = total_allocated_;
                const int32_t end_g = start_g + count - 1;
                const int32_t need_chks = end_g / DYN_CHUNK_CAPACITY + 1;
                while (static_cast<int32_t>(chunks_.size()) < need_chks)
                    grow();

                total_allocated_ += count;
                return start_g;
            }

            /**
             * @brief Discard all allocations (logical reset).
             *
             * Chunks are retained so their memory can be reused in the next round
             * without heap traffic.  The first allocate() call after reset() will
             * overwrite old data via init_node().
             */
            void reset() noexcept { total_allocated_ = 0; }

            // ----------------------------------------------------------------
            // Node initialisation
            // ----------------------------------------------------------------

            inline void init_node(int32_t g, int32_t depth_val, int32_t rollout_len_val, int32_t parent_g,
                                  int32_t action, int32_t tree_height) noexcept
            {
                ch(g).init_slot(sl(g), depth_val, rollout_len_val, parent_g, action, tree_height);
            }

            // ----------------------------------------------------------------
            // Metrics
            // ----------------------------------------------------------------

            int32_t total_nodes() const noexcept { return total_allocated_; }
            int32_t total_chunks() const noexcept { return static_cast<int32_t>(chunks_.size()); }

            /// Approximate pool memory footprint in bytes.
            size_t memory_bytes() const noexcept { return chunks_.size() * sizeof(DynSoAChunk); }

            // ----------------------------------------------------------------
            // Mutable field accessors  (indexed by global node index g)
            // ----------------------------------------------------------------

            inline float   &q_value(int32_t g) noexcept { return ch(g).q_values[sl(g)]; }
            inline float   &reward(int32_t g) noexcept { return ch(g).rewards[sl(g)]; }
            inline int32_t &visit_count(int32_t g) noexcept { return ch(g).visit_counts[sl(g)]; }
            inline float   &initial_rollout(int32_t g) noexcept { return ch(g).initial_rollout[sl(g)]; }
            inline int32_t &depth(int32_t g) noexcept { return ch(g).depth[sl(g)]; }
            inline int32_t &rollout_len(int32_t g) noexcept { return ch(g).rollout_len[sl(g)]; }
            inline int32_t &min_depth(int32_t g) noexcept { return ch(g).min_depth[sl(g)]; }
            inline int32_t &max_depth(int32_t g) noexcept { return ch(g).max_depth[sl(g)]; }
            inline int32_t &child_base(int32_t g) noexcept { return ch(g).child_base_idx[sl(g)]; }
            inline int32_t &parent(int32_t g) noexcept { return ch(g).parent_idx[sl(g)]; }
            inline int32_t &action_taken(int32_t g) noexcept { return ch(g).action_to_reach[sl(g)]; }
            inline int32_t &active_flag(int32_t g) noexcept { return ch(g).active_flags[sl(g)]; }
            inline int32_t &curr_action(int32_t g) noexcept { return ch(g).curr_action_idx[sl(g)]; }

            // ----------------------------------------------------------------
            // Const field accessors
            // ----------------------------------------------------------------

            inline float   q_value(int32_t g) const noexcept { return ch(g).q_values[sl(g)]; }
            inline float   reward(int32_t g) const noexcept { return ch(g).rewards[sl(g)]; }
            inline int32_t visit_count(int32_t g) const noexcept { return ch(g).visit_counts[sl(g)]; }
            inline float   initial_rollout(int32_t g) const noexcept { return ch(g).initial_rollout[sl(g)]; }
            inline int32_t depth(int32_t g) const noexcept { return ch(g).depth[sl(g)]; }
            inline int32_t rollout_len(int32_t g) const noexcept { return ch(g).rollout_len[sl(g)]; }
            inline int32_t min_depth(int32_t g) const noexcept { return ch(g).min_depth[sl(g)]; }
            inline int32_t max_depth(int32_t g) const noexcept { return ch(g).max_depth[sl(g)]; }
            inline int32_t child_base(int32_t g) const noexcept { return ch(g).child_base_idx[sl(g)]; }
            inline int32_t parent(int32_t g) const noexcept { return ch(g).parent_idx[sl(g)]; }
            inline int32_t action_taken(int32_t g) const noexcept { return ch(g).action_to_reach[sl(g)]; }
            inline int32_t active_flag(int32_t g) const noexcept { return ch(g).active_flags[sl(g)]; }

            // ----------------------------------------------------------------
            // Chunk-level pointer access for within-chunk SIMD operations
            // ----------------------------------------------------------------

            /// Pointer to q_values array starting at slot sl(g) within the chunk.
            inline float       *q_values_ptr(int32_t g) noexcept { return ch(g).q_values + sl(g); }
            inline const float *q_values_ptr(int32_t g) const noexcept { return ch(g).q_values + sl(g); }

            inline int32_t *visit_count_ptr(int32_t g) noexcept { return ch(g).visit_counts + sl(g); }
            inline int32_t *min_depth_ptr(int32_t g) noexcept { return ch(g).min_depth + sl(g); }
            inline int32_t *max_depth_ptr(int32_t g) noexcept { return ch(g).max_depth + sl(g); }
            inline int32_t *active_flag_ptr(int32_t g) noexcept { return ch(g).active_flags + sl(g); }

            /// Base pointer for the entire q_values array of chunk c.
            /// Used for cross-chunk gather: ptr = chunk_q_values_base(c) + slot.
            inline const float *chunk_q_values_base(int32_t c) const noexcept
            {
                return chunks_[static_cast<size_t>(c)]->q_values;
            }
            inline const int32_t *chunk_visits_base(int32_t c) const noexcept
            {
                return chunks_[static_cast<size_t>(c)]->visit_counts;
            }

          private:
            std::vector<std::unique_ptr<DynSoAChunk>> chunks_;
            int32_t                                   total_allocated_{0};

            void grow() { chunks_.push_back(std::make_unique<DynSoAChunk>()); }

            // Decompose global index g into chunk and slot.
            // DYN_CHUNK_CAPACITY = 32768 = 2^15, so bit-ops are valid.
            inline DynSoAChunk       &ch(int32_t g) noexcept { return *chunks_[static_cast<size_t>(g >> 15)]; }
            inline const DynSoAChunk &ch(int32_t g) const noexcept { return *chunks_[static_cast<size_t>(g >> 15)]; }
            static inline int32_t     sl(int32_t g) noexcept { return g & (DYN_CHUNK_CAPACITY - 1); }
        };

    } // namespace planning
} // namespace vec_qmdp
