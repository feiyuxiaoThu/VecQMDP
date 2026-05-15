/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file aligned_allocator.hpp
 * @brief STL-compatible aligned memory allocator for SIMD operations.
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>

namespace vec_qmdp
{
    namespace utils
    {
        template <typename T, std::size_t Alignment = 32> struct AlignedAllocator
        {
            using value_type = T;

            AlignedAllocator() noexcept = default;
            template <typename U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

            template <typename U> struct rebind
            {
                using other = AlignedAllocator<U, Alignment>;
            };

            [[nodiscard]] value_type *allocate(std::size_t n)
            {
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(value_type))
                    throw std::bad_array_new_length();

                static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
                static_assert(Alignment >= alignof(value_type), "Alignment must be at least alignof(T)");

                void *ptr = nullptr;
                if (posix_memalign(&ptr, Alignment, n * sizeof(value_type)) != 0)
                    throw std::bad_alloc();

// Use std::cout instead of LOG_DS to avoid circular dependency
#ifdef DEBUG_ALLOCATOR
                std::cout << "AlignedAllocator allocated " << n << " elements at " << ptr << std::endl;
#endif
                return reinterpret_cast<value_type *>(ptr);
            }

            void deallocate(value_type *ptr, std::size_t n) noexcept
            {
// Use std::cout instead of LOG_DS to avoid circular dependency
#ifdef DEBUG_ALLOCATOR
                std::cout << "AlignedAllocator deallocating " << n << " elements at " << ptr << std::endl;
#endif
                free(ptr);
            }
        };

        // Equality operators (STL compatibility)
        template <typename T1, typename T2, std::size_t Alignment>
        bool operator==(const AlignedAllocator<T1, Alignment> &, const AlignedAllocator<T2, Alignment> &) noexcept
        {
            return true;
        }

        template <typename T1, typename T2, std::size_t Alignment>
        bool operator!=(const AlignedAllocator<T1, Alignment> &, const AlignedAllocator<T2, Alignment> &) noexcept
        {
            return false;
        }

    } // namespace utils
} // namespace vec_qmdp