/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file path_utils.hpp
 * @brief Reference-path representation with SIMD-batched nearest-point search.
 */

#pragma once

#include <cstdint>
#include <utils/aligned_allocator.hpp>
#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace utils
    {
        class Path
        {
          public:
            Path();
            ~Path();

            // Resize path data arrays
            void resize(size_t size);

            // Find nearest point for a single point (using binary search)
            int NearestSerial(float x, float y, float heading = 0.0, bool for_ego = false, bool print = false) const;
            int NearestSerial(float x, float y, float heading, float &min_dist, bool for_ego = false,
                              bool print = false) const;

            // These two are only for the ego-agent
            // Find nearest point for a single point
            int NearestBatch(float x, float y, float heading);
            int NearestBatch(float x, float y, float heading, float &min_dist);

            // Find nearest points for a batch of points
            AlignedVectorInt NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, int valid_num,
                                          bool print = false);

            template <typename T, typename U>
            AlignedVectorInt NearestBatch(const T &xs, const T &ys, const U &valid_mask)
            {
                if (valid_mask.none())
                    return AlignedVectorInt(T::num_scalars, 0);

                int refline_size = xs_.size();

                AlignedVectorInt closest_idxs(T::num_scalars, 0);

                // SIMD vectors to store begin_idx and end_idx
                U begin_idxs, end_idxs;
                FindNearestSegmentBatch(xs, ys, begin_idxs, end_idxs);

                constexpr int NUM_WIDTH_1 = FVectorT_1::num_scalars_per_row;

                while ((end_idxs - begin_idxs).hmax() >= NUM_WIDTH_1)
                {
                    U mid_idxs = (begin_idxs + end_idxs) / 2;

                    // Compute distances to three points
                    T delta_begin_xs = T::gather(xs_.data(), begin_idxs) - xs;
                    T delta_begin_ys = T::gather(ys_.data(), begin_idxs) - ys;
                    T delta_mid_xs = T::gather(xs_.data(), mid_idxs) - xs;
                    T delta_mid_ys = T::gather(ys_.data(), mid_idxs) - ys;
                    T delta_end_xs = T::gather(xs_.data(), end_idxs) - xs;
                    T delta_end_ys = T::gather(ys_.data(), end_idxs) - ys;

                    T distance_to_begin = delta_begin_xs * delta_begin_xs + delta_begin_ys * delta_begin_ys;
                    T distance_to_mid = delta_mid_xs * delta_mid_xs + delta_mid_ys * delta_mid_ys;
                    T distance_to_end = delta_end_xs * delta_end_xs + delta_end_ys * delta_end_ys;

                    // Three-point comparison logic
                    U begin_is_min = ((distance_to_begin <= distance_to_mid) & (distance_to_begin <= distance_to_end))
                                         .template as<U>();
                    U end_is_min = ((distance_to_end <= distance_to_mid) & (distance_to_end <= distance_to_begin))
                                       .template as<U>();
                    U mid_is_min = ~(begin_is_min | end_is_min);

                    if (mid_is_min.none())
                    {
                        begin_idxs = U::select(begin_is_min, begin_idxs, mid_idxs);
                        end_idxs = U::select(begin_is_min, mid_idxs, end_idxs);
                    }
                    else
                    {
                        // For cases where mid point is closest, subdivide further
                        U quarter_left = (begin_idxs + mid_idxs) / 2;
                        U quarter_right = (mid_idxs + end_idxs) / 2;

                        T delta_qleft_xs = T::gather(xs_.data(), quarter_left) - xs;
                        T delta_qleft_ys = T::gather(ys_.data(), quarter_left) - ys;
                        T delta_qright_xs = T::gather(xs_.data(), quarter_right) - xs;
                        T delta_qright_ys = T::gather(ys_.data(), quarter_right) - ys;

                        T distance_to_qleft = delta_qleft_xs * delta_qleft_xs + delta_qleft_ys * delta_qleft_ys;
                        T distance_to_qright = delta_qright_xs * delta_qright_xs + delta_qright_ys * delta_qright_ys;

                        U choose_left_quarter = (distance_to_qleft <= distance_to_qright).template as<U>();
                        U mid_choose_left = mid_is_min & choose_left_quarter;
                        U mid_choose_right = mid_is_min & ~choose_left_quarter;

                        // Update indices
                        begin_idxs = U::select(begin_is_min | mid_choose_left, begin_idxs, mid_idxs);
                        end_idxs = U::select(begin_is_min | mid_choose_left, mid_idxs, end_idxs);
                    }
                }

                // **SIMD: compute final nearest points**
                begin_idxs = begin_idxs.min(refline_size - NUM_WIDTH_1);

                auto begin_idxs_array = begin_idxs.to_array();
                auto end_idxs_array = end_idxs.to_array();

                for (size_t i = 0; i < T::num_scalars; ++i)
                {
                    const auto pair_idx = utils::div_mod<size_t>(i, T::num_scalars_per_row);
                    if (!valid_mask[pair_idx])
                    {
                        continue;
                    }

                    int begin_idx = begin_idxs_array[i];

                    FVectorT_1 delta_xs = FVectorT_1::load_contiguous_unaligned(xs_.data(), begin_idx) - xs[pair_idx];
                    FVectorT_1 delta_ys = FVectorT_1::load_contiguous_unaligned(ys_.data(), begin_idx) - ys[pair_idx];

                    FVectorT_1 dists = delta_xs * delta_xs + delta_ys * delta_ys;

                    const auto dists_array = dists.to_array();
                    int        min_index = 0;
                    float      min_dist = utils::MAX_VALUE;
                    for (int j = 0; j < end_idxs_array[i] - begin_idx; ++j)
                    {
                        if (dists_array[j] < min_dist)
                        {
                            min_dist = dists_array[j];
                            min_index = j;
                        }
                    }
                    closest_idxs[i] = min_index + begin_idx;
                }

                return closest_idxs;
            }

            // Find nearest points for a batch of points (using binary search)
            template <typename T, typename U>
            U NearestBatch(const T &xs, const T &ys, const U &valid_mask, T &min_dists)
            {
                if (valid_mask.none())
                    return U::fill(0);

                int refline_size = xs_.size();

                AlignedVectorInt   closest_idxs(T::num_scalars, 0);
                AlignedVectorFloat tmp_min_dists(T::num_scalars, utils::MAX_VALUE);

                // SIMD vectors to store begin_idx and end_idx
                U begin_idxs, end_idxs;
                FindNearestSegmentBatch(xs, ys, begin_idxs, end_idxs);

                constexpr int NUM_WIDTH_1 = FVectorT_1::num_scalars_per_row;

                while ((end_idxs - begin_idxs).hmax() >= NUM_WIDTH_1)
                {
                    U mid_idxs = (begin_idxs + end_idxs) / 2;

                    // Compute distances to three points
                    T delta_begin_xs = T::gather(xs_.data(), begin_idxs) - xs;
                    T delta_begin_ys = T::gather(ys_.data(), begin_idxs) - ys;
                    T delta_mid_xs = T::gather(xs_.data(), mid_idxs) - xs;
                    T delta_mid_ys = T::gather(ys_.data(), mid_idxs) - ys;
                    T delta_end_xs = T::gather(xs_.data(), end_idxs) - xs;
                    T delta_end_ys = T::gather(ys_.data(), end_idxs) - ys;

                    T distance_to_begin = delta_begin_xs * delta_begin_xs + delta_begin_ys * delta_begin_ys;
                    T distance_to_mid = delta_mid_xs * delta_mid_xs + delta_mid_ys * delta_mid_ys;
                    T distance_to_end = delta_end_xs * delta_end_xs + delta_end_ys * delta_end_ys;

                    // Three-point comparison logic
                    U begin_is_min = ((distance_to_begin <= distance_to_mid) & (distance_to_begin <= distance_to_end))
                                         .template as<U>();
                    U end_is_min = ((distance_to_end <= distance_to_mid) & (distance_to_end <= distance_to_begin))
                                       .template as<U>();
                    U mid_is_min = ~(begin_is_min | end_is_min);

                    if (mid_is_min.none())
                    {
                        begin_idxs = U::select(begin_is_min, begin_idxs, mid_idxs);
                        end_idxs = U::select(begin_is_min, mid_idxs, end_idxs);
                    }
                    else
                    {
                        // For cases where mid point is closest, subdivide further
                        U quarter_left = (begin_idxs + mid_idxs) / 2;
                        U quarter_right = (mid_idxs + end_idxs) / 2;

                        T delta_qleft_xs = T::gather(xs_.data(), quarter_left) - xs;
                        T delta_qleft_ys = T::gather(ys_.data(), quarter_left) - ys;
                        T delta_qright_xs = T::gather(xs_.data(), quarter_right) - xs;
                        T delta_qright_ys = T::gather(ys_.data(), quarter_right) - ys;

                        T distance_to_qleft = delta_qleft_xs * delta_qleft_xs + delta_qleft_ys * delta_qleft_ys;
                        T distance_to_qright = delta_qright_xs * delta_qright_xs + delta_qright_ys * delta_qright_ys;

                        U choose_left_quarter = (distance_to_qleft <= distance_to_qright).template as<U>();
                        U mid_choose_left = mid_is_min & choose_left_quarter;
                        U mid_choose_right = mid_is_min & ~choose_left_quarter;

                        // Update indices
                        begin_idxs = U::select(begin_is_min | mid_choose_left, begin_idxs, mid_idxs);
                        end_idxs = U::select(begin_is_min | mid_choose_left, mid_idxs, end_idxs);
                    }
                }

                // **SIMD: compute final nearest points**
                begin_idxs = begin_idxs.min(refline_size - NUM_WIDTH_1);

                auto begin_idxs_array = begin_idxs.to_array();
                auto end_idxs_array = end_idxs.to_array();

                for (size_t i = 0; i < T::num_scalars; ++i)
                {
                    const auto pair_idx = utils::div_mod<size_t>(i, T::num_scalars_per_row);
                    if (~valid_mask[pair_idx])
                    {
                        continue;
                    }

                    int        begin_idx = begin_idxs_array[i];
                    FVectorT_1 delta_xs = FVectorT_1::load_contiguous_unaligned(xs_.data(), begin_idx) - xs[pair_idx];
                    FVectorT_1 delta_ys = FVectorT_1::load_contiguous_unaligned(ys_.data(), begin_idx) - ys[pair_idx];

                    FVectorT_1 dists = delta_xs * delta_xs + delta_ys * delta_ys;

                    const auto dists_array = dists.to_array();
                    int        min_index = 0;
                    float      min_dist = utils::MAX_VALUE;
                    for (int j = 0; j < end_idxs_array[i] - begin_idx; ++j)
                    {
                        if (dists_array[j] < min_dist)
                        {
                            min_dist = dists_array[j];
                            min_index = j;
                        }
                    }
                    tmp_min_dists[i] = min_dist;
                    closest_idxs[i] = min_index + begin_idx;
                }

                min_dists = T(tmp_min_dists.data()).sqrt();

                return U(closest_idxs.data());
            }

            void NearestSerial(float x, float y, int &previous_nearest_idx, const int &step_direction) const;

            // Find nearest points for a batch of points given the previous nearest idxs
            void NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &previous_nearest_idxs,
                              const IVectorT_1 &step_directions) const;

            void NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &previous_nearest_idxs,
                              const IVectorT_1 &step_directions, IVectorT_1 valid_mask) const;

            void NearestSerial(float x, float y, int &previous_nearest_idx, float &min_dist,
                               const int &step_direction) const;

            // Find nearest points for a batch of points given the previous nearest idxs
            template <typename T, typename U>
            void NearestBatch(const T &xs, const T &ys, U &previous_nearest_idxs, T &min_dists,
                              const U &step_directions, U valid_mask) const
            {
                if (valid_mask.none())
                    return;

                int refline_size = xs_.size();

                // Initialize counter and minimum distance vectors
                previous_nearest_idxs =
                    (previous_nearest_idxs - utils::TRACEBACK_STEPS * step_directions).clamp(0, refline_size - 1);
                U min_idxs = previous_nearest_idxs; // Initialize to current index

                // Helper: gather(xs, idx) - x; return dx*dx + dy*dy
                min_dists = DistSqBatch<T, U>(xs, ys, xs_, ys_, previous_nearest_idxs);

                // Set safety upper bound (e.g. 25) to cover extreme 20+ step cases
                // Rely on internal break for early exit
                constexpr int SAFETY_MAX_STEPS = 25;
                // Create comparison threshold
                constexpr int PATIENCE_LIMIT = 3;

                U patience_counts = U::fill(0);

                for (int i = 0; i < SAFETY_MAX_STEPS; ++i)
                {
                    // Probe next step
                    // Update current index
                    U next_idxs = previous_nearest_idxs + step_directions;

                    // Update validity mask
                    valid_mask = valid_mask & (next_idxs >= 0) & (next_idxs < refline_size);

                    // === Core optimization: early termination check ===
                    // Exit immediately if no lane needs further searching
                    if (valid_mask.none())
                        break;

                    // Only update lanes still searching to avoid out-of-bounds access outside valid_mask (clamp
                    // provides protection)
                    previous_nearest_idxs = U::select(valid_mask, next_idxs, previous_nearest_idxs);

                    // Compute next step distance (most expensive operation: 2 gathers)
                    T next_dists = DistSqBatch<T, U>(xs, ys, xs_, ys_, previous_nearest_idxs);

                    // Check if improved
                    // Only compare for lanes still searching
                    U is_improved = (next_dists < min_dists).template as<U>() & valid_mask;

                    // Update best state
                    // If a closer point is found, update min_dists and best_idxs
                    min_dists = T::select(is_improved.template as<T>(), next_dists, min_dists);
                    min_idxs = U::select(is_improved, previous_nearest_idxs, min_idxs);

                    // Update patience counter
                    // If improved: reset counter to 0
                    // Otherwise: increment counter by 1
                    // Note: select logic is (mask, true_val, false_val)
                    patience_counts = U::select(is_improved, U::fill(0), patience_counts + 1);

                    // F. Update search state
                    // Stop searching if patience >= limit
                    U patience_exhausted = (patience_counts >= PATIENCE_LIMIT);
                    valid_mask = valid_mask & (~patience_exhausted);

                    // G. Early termination check
                    if (valid_mask.none())
                        break;
                }

                min_dists = min_dists.sqrt();

                previous_nearest_idxs = min_idxs;
            }

            // Find interpolated nearest points for a batch of points given the previous nearest idxs
            void InterpolatedNearest(const float &x, const float &y, const int &nearest_idx, float &interpolated_x,
                                     float &interpolated_y, float &interpolated_theta,
                                     float &interpolated_path_index) const;

            template <typename T, typename U>
            void InterpolatedNearestBatch(const T &xs, const T &ys, const U &nearest_idxs, T &interpolated_xs,
                                          T &interpolated_ys, T &interpolated_thetas, T &interpolated_path_idxs) const
            {
                U prev_nearest_idxs = nearest_idxs.min(xs_.size() - 2);
                U next_nearest_idxs = prev_nearest_idxs + 1;

                // Direct gather operations for previous points
                T xs_prev = T::gather(xs_.data(), prev_nearest_idxs);
                T ys_prev = T::gather(ys_.data(), prev_nearest_idxs);
                T thetas_prev = T::gather(thetas_.data(), prev_nearest_idxs);

                // Compute segment vectors and query vectors in one go
                T dx = T::gather(xs_.data(), next_nearest_idxs) - xs_prev;
                T dy = T::gather(ys_.data(), next_nearest_idxs) - ys_prev;
                T qdx = xs - xs_prev;
                T qdy = ys - ys_prev;

                // Compute projection parameter t directly
                T t = (qdx * dx + qdy * dy) / (dx * dx + dy * dy + 1e-6f);

                // Compute final interpolated values directly
                interpolated_xs = xs_prev + t * dx;
                interpolated_ys = ys_prev + t * dy;
                interpolated_thetas = thetas_prev + t * utils::NormalizeAngleSIMD<T>(
                                                            T::gather(thetas_.data(), next_nearest_idxs) - thetas_prev);
                interpolated_path_idxs = prev_nearest_idxs.template convert<T>() + t;
            }

            template <typename T, typename U>
            void InterpolatedNearestBatch(const T &interpolated_path_idxs, const U &nearest_idxs, T &interpolated_xs,
                                          T &interpolated_ys, T &interpolated_thetas) const
            {
                U prev_nearest_idxs = nearest_idxs.min(xs_.size() - 2);
                U next_nearest_idxs = prev_nearest_idxs + 1;

                // Direct gather operations for previous points
                T xs_prev = T::gather(xs_.data(), prev_nearest_idxs);
                T ys_prev = T::gather(ys_.data(), prev_nearest_idxs);
                T thetas_prev = T::gather(thetas_.data(), prev_nearest_idxs);

                // Compute segment vectors and query vectors in one go
                T dx = T::gather(xs_.data(), next_nearest_idxs) - xs_prev;
                T dy = T::gather(ys_.data(), next_nearest_idxs) - ys_prev;

                T t = interpolated_path_idxs - prev_nearest_idxs.template convert<T>();

                // Compute final interpolated values directly
                interpolated_xs = xs_prev + t * dx;
                interpolated_ys = ys_prev + t * dy;
                interpolated_thetas = thetas_prev + t * utils::NormalizeAngleSIMD<T>(
                                                            T::gather(thetas_.data(), next_nearest_idxs) - thetas_prev);
            }

            float PointOnLeft(float x, float y, const int &nearest_index) const;

            float PointOnLeft(float x, float y, float theta_r, const int &nearest_index) const;

            template <typename T, typename U> T PointOnLeft(const T &xs, const T &ys, const U &nearest_idxs) const
            {
                T theta_r = T::gather(thetas_.data(), nearest_idxs);

                // T dx = xs - T::gather(xs_.data(), nearest_idxs);
                // T dy = ys - T::gather(ys_.data(), nearest_idxs);

                return T::select(theta_r.cos() * (ys - T::gather(ys_.data(), nearest_idxs)) -
                                         theta_r.sin() * (xs - T::gather(xs_.data(), nearest_idxs)) >
                                     0.0f,
                                 1.0f, -1.0f);
            }

            template <typename T, typename U>
            T PointOnLeft(const T &xs, const T &ys, const T &theta_r, const U &nearest_idxs) const
            {
                return T::select(theta_r.cos() * (ys - T::gather(ys_.data(), nearest_idxs)) -
                                         theta_r.sin() * (xs - T::gather(xs_.data(), nearest_idxs)) >
                                     0.0f,
                                 1.0f, -1.0f);
            }

            // Helper: find the nearest segment range based on anchor segment
            std::pair<int, int> FindNearestSegmentSingle(float x, float y, float heading = 0.0, bool for_ego = false,
                                                         bool print = false) const;

            void FindNearestSegmentBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &begin_idxs,
                                         IVectorT_1 &end_idxs, bool print = false) const;

            template <typename T, typename U>
            void FindNearestSegmentBatch(const T &xs, const T &ys, U &begin_idxs, U &end_idxs) const
            {
                if (anchor_idxs_.empty())
                {
                    // If no anchor segments exist, use the entire path
                    begin_idxs = U::fill(0);
                    end_idxs = U::fill(xs_.size() - 1);
                    return;
                }

                // Initialize minimum distance and best segment index
                T min_proj_dists = T::fill(utils::MAX_VALUE);
                U best_segments = U::fill(0);

                AlignedVectorInt begin_idxs_array(U::num_scalars, 0);
                AlignedVectorInt end_idxs_array(U::num_scalars, xs_.size() - 1);

                // Iterate over all segments, find the one with minimum projection distance for each point
                for (int i = 0; i < anchor_idxs_.size(); ++i)
                {
                    // Compute segment vector
                    float dx = anchor_dx_[i];
                    float dy = anchor_dy_[i];
                    float segment_length_sq = anchor_len_sq_[i];

                    if (segment_length_sq < 1e-6f) // Handle degenerate case
                    {
                        continue;
                    }

                    // Compute vector from segment start to query point
                    T qx = xs - anchor_xs_[i].first;
                    T qy = ys - anchor_ys_[i].first;

                    // Compute projection parameter t (vectorized)
                    T t = (qx * dx + qy * dy) / segment_length_sq;
                    t = t.clamp(0.0f, 1.0f); // Clamp to [0,1] range

                    // Compute projected point
                    T proj_x = anchor_xs_[i].first + t * dx;
                    T proj_y = anchor_ys_[i].first + t * dy;

                    // Compute distance to projected point
                    T delta_x = xs - proj_x;
                    T delta_y = ys - proj_y;
                    T proj_dists = delta_x * delta_x + delta_y * delta_y;

                    // Update minimum distance and best segment
                    T is_better = proj_dists < min_proj_dists;
                    min_proj_dists = T::select(is_better, proj_dists, min_proj_dists);
                    best_segments = U::select(is_better.template as<U>(), i, best_segments);
                }

                auto best_segments_array = best_segments.to_array();
                for (int i = 0; i < best_segments_array.size(); ++i)
                {
                    begin_idxs_array[i] = anchor_idxs_[best_segments_array[i]].first;
                    end_idxs_array[i] = anchor_idxs_[best_segments_array[i]].second;
                }

                // Set begin_idxs and end_idxs based on best segment indices
                begin_idxs = U(begin_idxs_array.data());
                end_idxs = U(end_idxs_array.data());
            }

            inline float GetFrenetS(const int &idx) const { return idx * utils::PATH_POINT_INTERVAL; }

            inline FVectorT_1 GetFrenetS(const IVectorT_1 &nearest_idxs) const
            {
                return nearest_idxs.template convert<FVectorT_1>() * utils::PATH_POINT_INTERVAL;
            }

            inline float GetX(const int &idx) const { return xs_[idx]; }

            inline FVectorT_qmdp GetXBatch(const IVectorT_qmdp &nearest_idxs) const
            {
                return FVectorT_qmdp::gather(xs_.data(), nearest_idxs);
            }

            inline FVectorT_traj GetXBatch(const IVectorT_traj &nearest_idxs) const
            {
                return FVectorT_traj::gather(xs_.data(), nearest_idxs);
            }

            inline AlignedVectorFloat &GetXs() { return xs_; }

            inline float GetY(const int &idx) const { return ys_[idx]; }

            inline FVectorT_qmdp GetYBatch(const IVectorT_qmdp &nearest_idxs) const
            {
                return FVectorT_qmdp::gather(ys_.data(), nearest_idxs);
            }

            inline FVectorT_traj GetYBatch(const IVectorT_traj &nearest_idxs) const
            {
                return FVectorT_traj::gather(ys_.data(), nearest_idxs);
            }

            inline AlignedVectorFloat &GetYs() { return ys_; }

            inline float GetTheta(const int &idx) const { return thetas_[idx]; }

            inline FVectorT_qmdp GetThetaBatch(const IVectorT_qmdp &nearest_idxs) const
            {
                return FVectorT_qmdp::gather(thetas_.data(), nearest_idxs);
            }

            inline FVectorT_traj GetThetaBatch(const IVectorT_traj &nearest_idxs) const
            {
                return FVectorT_traj::gather(thetas_.data(), nearest_idxs);
            }

            inline AlignedVectorFloat &GetThetas() { return thetas_; }

            inline float GetKappa(const int &idx) const { return kappas_[idx]; }

            inline AlignedVectorFloat &GetKappas() { return kappas_; }

            inline float GetCurvature(const int &idx) const { return kappas_[idx]; }

            // Get path point curvature
            template <typename T, typename U> inline T GetCurvature(const U &nearest_idxs) const
            {
                return T::gather(kappas_.data(), nearest_idxs);
            }

            inline std::pair<FVectorT_1, FVectorT_1>
            GetMaxCurvatureAndMinDesiredSpeedBatch(const IVectorT_1 &start_nearest_idxs, const FVectorT_1 &speeds,
                                                   FVectorT_1 &out_curr_curvatures, IVectorT_1 active_mask) const
            {
                // 1. Precompute end-point index
                IVectorT_1 end_nearest_idxs =
                    start_nearest_idxs + utils::LOOKAHEAD_TIME_LENGTH_SIZE_OFFSET +
                    (speeds * utils::LOOKAHEAD_TIME_LENGTH_SIZE).template convert<IVectorT_1>();

                // 2. Gather starting segment index via precomputed point-to-segment map
                IVectorT_1 curr_seg_idxs = IVectorT_1::gather(point_to_segment_idx_.data(), start_nearest_idxs);

                // 3. Output curvature at the current point
                out_curr_curvatures = FVectorT_1::gather(max_signed_curvature_vec_.data(), curr_seg_idxs);

                // Masked loop over segments
                FVectorT_1 final_max_curvs = FVectorT_1::fill(0.0f);
                FVectorT_1 final_min_speeds = FVectorT_1::fill(utils::MAX_VEL);

                // Safety bound to prevent out-of-range access
                IVectorT_1 max_valid_idx = IVectorT_1::fill(max_curvature_idx_vec_.size() - 1);

                while (active_mask.any())
                {
                    // A. Clamp indices to valid range
                    IVectorT_1 safe_read_idxs = curr_seg_idxs.min(max_valid_idx);

                    // B. Gather current segment data
                    FVectorT_1 seg_max_curv = FVectorT_1::gather(max_curvature_vec_.data(), safe_read_idxs);
                    FVectorT_1 seg_min_speed = FVectorT_1::gather(min_desired_speed_vec_.data(), safe_read_idxs);
                    IVectorT_1 seg_boundary = IVectorT_1::gather(max_curvature_idx_vec_.data(), safe_read_idxs);

                    // C. Update results (only for active lanes)
                    final_max_curvs = FVectorT_1::select(active_mask.template as<FVectorT_1>(),
                                                         final_max_curvs.max(seg_max_curv), final_max_curvs);

                    final_min_speeds = FVectorT_1::select(active_mask.template as<FVectorT_1>(),
                                                          final_min_speeds.min(seg_min_speed), final_min_speeds);

                    // D. Check exit conditions: reached target position or end of array
                    IVectorT_1 should_break = (end_nearest_idxs <= seg_boundary) | (curr_seg_idxs >= max_valid_idx);

                    // E. Update mask
                    active_mask = active_mask & (~should_break);

                    // F. Advance index (all lanes; inactive ones are masked in next iteration)
                    curr_seg_idxs = curr_seg_idxs + 1;
                }

                return {final_max_curvs, final_min_speeds};
            }

            template <typename T, typename U>
            inline std::pair<T, T> GetMaxCurvatureAndMinDesiredSpeedBatch(const U &start_nearest_idxs, const T &speeds,
                                                                          U active_mask) const
            {
                // 1. Precompute end-point index
                U end_nearest_idxs = start_nearest_idxs + utils::LOOKAHEAD_TIME_LENGTH_SIZE_OFFSET +
                                     (speeds * utils::LOOKAHEAD_TIME_LENGTH_SIZE).template convert<U>();

                // 2. Gather starting segment index via precomputed point-to-segment map
                U curr_seg_idxs = U::gather(point_to_segment_idx_.data(), start_nearest_idxs);

                // Masked loop over segments
                T final_max_curvs = T::fill(0.0f);
                T final_min_speeds = T::fill(utils::MAX_VEL);

                // Safety bound to prevent out-of-range access
                U max_valid_idx = U::fill(max_curvature_idx_vec_.size() - 1);

                while (active_mask.any())
                {
                    // A. Clamp indices to valid range
                    U safe_read_idxs = curr_seg_idxs.min(max_valid_idx);

                    // B. Gather current segment data
                    T seg_max_curv = T::gather(max_curvature_vec_.data(), safe_read_idxs);
                    T seg_min_speed = T::gather(min_desired_speed_vec_.data(), safe_read_idxs);
                    U seg_boundary = U::gather(max_curvature_idx_vec_.data(), safe_read_idxs);

                    // C. Update results (only for active lanes)
                    final_max_curvs =
                        T::select(active_mask.template as<T>(), final_max_curvs.max(seg_max_curv), final_max_curvs);

                    final_min_speeds =
                        T::select(active_mask.template as<T>(), final_min_speeds.min(seg_min_speed), final_min_speeds);

                    // D. Check exit conditions: reached target position or end of array
                    U should_break = (end_nearest_idxs <= seg_boundary) | (curr_seg_idxs >= max_valid_idx);

                    // E. Update mask
                    active_mask = active_mask & (~should_break);

                    // F. Advance index (all lanes; inactive ones are masked in next iteration)
                    curr_seg_idxs = curr_seg_idxs + 1;
                }

                return {final_max_curvs, final_min_speeds};
            }

            template <typename T, typename U>
            inline std::pair<T, T> GetMaxCurvatureAndMinDesiredSpeedBatch(const U &start_nearest_idxs, const T &speeds,
                                                                          T &out_curr_curvatures, U active_mask) const
            {
                // 1. Precompute end-point index
                U end_nearest_idxs = start_nearest_idxs + utils::LOOKAHEAD_TIME_LENGTH_SIZE_OFFSET +
                                     (speeds * utils::LOOKAHEAD_TIME_LENGTH_SIZE).template convert<U>();

                // 2. Gather starting segment index via precomputed point-to-segment map
                U curr_seg_idxs = U::gather(point_to_segment_idx_.data(), start_nearest_idxs);

                // 3. Output curvature at the current point
                out_curr_curvatures = T::gather(max_signed_curvature_vec_.data(), curr_seg_idxs);

                // Masked loop over segments
                T final_max_curvs = T::fill(0.0f);
                T final_min_speeds = T::fill(utils::MAX_VEL);

                // Safety bound to prevent out-of-range access
                U max_valid_idx = U::fill(max_curvature_idx_vec_.size() - 1);

                while (active_mask.any())
                {
                    // A. Clamp indices to valid range
                    U safe_read_idxs = curr_seg_idxs.min(max_valid_idx);

                    // B. Gather current segment data
                    T seg_max_curv = T::gather(max_curvature_vec_.data(), safe_read_idxs);
                    T seg_min_speed = T::gather(min_desired_speed_vec_.data(), safe_read_idxs);
                    U seg_boundary = U::gather(max_curvature_idx_vec_.data(), safe_read_idxs);

                    // C. Update results (only for active lanes)
                    final_max_curvs =
                        T::select(active_mask.template as<T>(), final_max_curvs.max(seg_max_curv), final_max_curvs);

                    final_min_speeds =
                        T::select(active_mask.template as<T>(), final_min_speeds.min(seg_min_speed), final_min_speeds);

                    // D. Check exit conditions: reached target position or end of array
                    U should_break = (end_nearest_idxs <= seg_boundary) | (curr_seg_idxs >= max_valid_idx);

                    // E. Update mask
                    active_mask = active_mask & (~should_break);

                    // F. Advance index (all lanes; inactive ones are masked in next iteration)
                    curr_seg_idxs = curr_seg_idxs + 1;
                }

                return {final_max_curvs, final_min_speeds};
            }

            inline void GetNearestEdgeIdx(int &nearest_edge_idx, int nearest_idx) const
            {
                // Find the curvature segment corresponding to start_nearest_idx
                for (; nearest_edge_idx < comprised_ref_path_idxs_.size() - 1; ++nearest_edge_idx)
                {
                    if (nearest_idx < comprised_ref_path_idxs_[nearest_edge_idx + 1])
                    {
                        break;
                    }
                }
            }

            inline size_t GetSize() const { return xs_.size(); }

            inline float GetPathLen() const { return path_len_; }

            inline std::string GetLaneId() const { return lane_id_; }

            inline std::string GetLaneName() const { return lane_name_; }

            inline bool IsStraight() const { return is_straight_; }

            inline void SetRank(int rank) { rank_ = rank; }

            inline int GetRank() const { return rank_; }

            // inline float GetSpeedLimitation(int edge_idx) const
            // {
            //     return edge_idx < comprised_ref_path_speed_limits_.size() ?
            //     comprised_ref_path_speed_limits_[edge_idx] : utils::MAX_VEL;
            // }

            inline void UpdateComprisedEdges(const std::string &edge_id, size_t start_idx)
            {
                comprised_ref_path_ids_.emplace_back(edge_id);
                comprised_ref_path_idxs_.emplace_back(start_idx);
            }

            inline void UpdateMaxCurvatureAndMinDesiredSpeed(float max_signed_curvature, float max_curvature,
                                                             float min_desired_speed, size_t end_idx)
            {
                max_signed_curvature_vec_.emplace_back(max_signed_curvature);
                max_curvature_vec_.emplace_back(max_curvature);
                min_desired_speed_vec_.emplace_back(min_desired_speed);
                max_curvature_idx_vec_.emplace_back(end_idx);

                // Batch fill using insert (faster than for-loop push_back; can leverage memset or vectorized store)
                point_to_segment_idx_.insert(point_to_segment_idx_.end(), end_idx - point_to_segment_idx_.size() + 1,
                                             static_cast<int>(max_curvature_idx_vec_.size()) - 1);
            }

            inline void PrintPath() const
            {
                // LOG_DS << "PrintPath " << lane_id_;
                for (size_t i = 0; i < xs_.size(); ++i)
                {
                    LOG_DS << "[" << i << "] " << xs_[i] << " " << ys_[i];
                }
            }

          private:
            // Helper: compute squared distance
            template <typename T, typename U>
            static inline T DistSqBatch(const T &px, const T &py, const AlignedVectorFloat &path_xs,
                                        const AlignedVectorFloat &path_ys, const U &idxs)
            {
                T dx = T::gather(path_xs.data(), idxs) - px;
                T dy = T::gather(path_ys.data(), idxs) - py;
                return dx * dx + dy * dy;
            }

          public:
            AlignedVectorFloat xs_;
            AlignedVectorFloat ys_;
            AlignedVectorFloat thetas_;
            AlignedVectorFloat kappas_;
            float              red_light_point_s_; // Traffic light point Frenet s-coordinate

            std::vector<float> max_signed_curvature_vec_; // Max signed curvature per segment
            std::vector<float> max_curvature_vec_;        // Max curvature per segment
            std::vector<float> min_desired_speed_vec_;    // Min desired speed per segment
            std::vector<int>   max_curvature_idx_vec_;    // End-point index per segment
            std::vector<int>   point_to_segment_idx_;     // Point-to-segment mapping (length = path size)

            std::vector<std::string> comprised_ref_path_ids_;  // Edge IDs composing this path
            std::vector<size_t>      comprised_ref_path_idxs_; // Start indices of edges composing this path
            // std::vector<float> miss_goal_penalty_vec_; // Miss-goal penalty per edge
            // std::vector<float> comprised_ref_path_speed_limits_; // Speed limit per edge

            std::vector<std::pair<float, float>> anchor_xs_;     // Anchor x per segment
            std::vector<std::pair<float, float>> anchor_ys_;     // Anchor y per segment
            std::vector<float>                   anchor_thetas_; // Anchor theta per segment
            std::vector<float>                   anchor_dx_;     // Segment direction vector x
            std::vector<float>                   anchor_dy_;     // Segment direction vector y
            std::vector<float>                   anchor_len_sq_; // Squared segment length
            std::vector<std::pair<int, int>>     anchor_idxs_;   // Anchor index range per segment

            std::string lane_id_;
            std::string lane_name_;
            bool        is_straight_;
            float
                miss_goal_penalty_; // Miss-goal penalty for this path; for composite paths, takes the max across edges
            float goal_frenet_s_;
            float path_len_;
            float speed_limit_;
            bool  not_on_route_;
            bool  loop_flag_;
            int   rank_;                   // ordered from left to right, begin from 0
            float weighted_avg_r_squared_; // Weighted average R-squared value based on segment lengths
        };
    } // namespace utils
} // namespace vec_qmdp