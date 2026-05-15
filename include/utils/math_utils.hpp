/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file math_utils.hpp
 * @brief Scalar and SIMD math helpers (distance, angle wrapping, clipping).
 */

#pragma once

#include <Eigen/Dense>
#include <utils/global_utils.hpp>
#include <vamp/vector.hh>
#include <vector>

namespace vec_qmdp
{
    namespace utils
    {
        template <typename T> inline std::pair<T, T> div_mod(const T &a, const T &b)
        {
            return std::make_pair(a / b, a % b);
        }

        // Helper function to calculate squared distance
        inline float SquaredDistance(float x1, float y1, float x2, float y2)
        {
            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
        }

        // Helper function to calculate distance
        inline float Distance(float x1, float y1, float x2, float y2)
        {
            return std::sqrt(SquaredDistance(x1, y1, x2, y2));
        }

        /**
         * @brief Normalize angle to [-PI, PI).
         * @param angle the original value of the angle.
         * @return The normalized value of the angle.
         */
        inline float NormalizeAngle(float angle)
        {
            float a = std::fmod(angle + M_PI, 2.0 * M_PI);
            if (a < 0.0)
                a += (2.0 * M_PI);
            return a - M_PI;
        }

        /**
         * @brief Normalize angle to [-PI, PI).
         * @param angle the original value of the angle.
         * @return The normalized value of the angle.
         */
        template <typename T> inline T NormalizeAngleSIMD(const T &angle)
        {
            // Normalize angle differences
            T theta_diffs = angle + M_PI;              // shift to [0, 2π]
            theta_diffs = theta_diffs % (2.0f * M_PI); // mod to [0, 2π]
            theta_diffs = theta_diffs - M_PI;          // shift back to [-π, π]
            return theta_diffs;
        }

        inline bool InFront(float x1, float y1, float theta1, float x2, float y2, float threshold = M_PI_2)
        {
            float angle = std::atan2(y2 - y1, x2 - x1);
            float abs_theta_diff = std::fabs(NormalizeAngle(theta1 - angle));
            return abs_theta_diff < threshold;
        }

        inline bool InBehind(float x1, float y1, float theta1, float x2, float y2, float threshold = M_PI_2)
        {
            float angle = std::atan2(y2 - y1, x2 - x1);
            float abs_theta_diff = std::fabs(NormalizeAngle(theta1 - angle));
            return abs_theta_diff >= threshold;
        }

        /**
         * @brief Check if the angle between a vector and reference direction is within threshold
         *
         * @tparam T Floating point type (float, double, etc.)
         * @param vec_x X component of the vector
         * @param vec_y Y component of the vector
         * @param ref_cos Cosine of reference angle
         * @param ref_sin Sine of reference angle
         * @param threshold_rad Angle threshold in radians，should be lower than 90°
         * @return T Boolean mask (true if within threshold, false otherwise)
         */
        template <typename T>
        inline T IsAngleWithinThreshold(const T &vec_x, const T &vec_y,     // vector components
                                        const T &ref_cos, const T &ref_sin, // reference direction (cos, sin)
                                        float threshold_rad)                // threshold (radians)
        {
            float cos_thres = std::cos(threshold_rad);
            float cos_sqr = cos_thres * cos_thres;

            // Dot product
            T dot = vec_x * ref_cos + vec_y * ref_sin;
            T dist_sqr = vec_x * vec_x + vec_y * vec_y;

            // Logic:
            // 1) dot > 0 (ensures vectors point roughly the same way, angle < 90°)
            // 2) dot^2 / dist^2 > cos^2 (equivalently cos^2(theta) > cos^2(threshold))
            return (dot > 0.0f) & (dot * dot > dist_sqr * cos_sqr);
        }

        template <typename T> inline void clipEdgeBatch(const T &p, const T &q, T &valid, T &t_min, T &t_max)
        {
            // Parallel case
            T parallel = (p.abs() < 1e-8f);
            T outside = parallel & (q < 0.0f);
            valid = valid & (~outside);

            if (valid.none())
            {
                return;
            }

            // Non-parallel case: update t_min / t_max
            T r = q / T::select(p >= 0.0f, p.max(1e-8f), p.min(-1e-8f));
            T entering = (p < 0.0f) & (~parallel);
            T leaving = (p > 0.0f) & (~parallel);

            t_min = T::select(entering, t_min.max(r), t_min);
            t_max = T::select(leaving, t_max.min(r), t_max);
        }

        inline void clipEdgeSerial(float p, float q, bool &valid, float &t_min, float &t_max)
        {
            // Parallel case
            bool parallel = (std::abs(p) < 1e-8f);
            bool outside = parallel && (q < 0.0f);
            if (outside)
            {
                valid = false;
                return;
            }

            // Non-parallel case: update t_min / t_max
            float r = q / (p >= 0.0f ? std::max(p, 1e-8f) : std::min(p, -1e-8f));
            bool  entering = (p < 0.0f) && (!parallel);
            bool  leaving = (p > 0.0f) && (!parallel);

            if (entering)
            {
                t_min = std::max(t_min, r);
            }
            if (leaving)
            {
                t_max = std::min(t_max, r);
            }
        }
    } // namespace utils
} // namespace vec_qmdp