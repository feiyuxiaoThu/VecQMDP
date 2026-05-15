/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <iostream>
#include <utils/path_utils.hpp>

namespace vec_qmdp
{
    namespace utils
    {
        Path::Path()
        {
            // LOG_DS << "Path constructor called";
            red_light_point_s_ = utils::MAX_VALUE;
            not_on_route_ = false;
            loop_flag_ = false;
            rank_ = 0;
            goal_frenet_s_ = 0.0f;
        }

        Path::~Path()
        {
            // LOG_DS << "Path destructor called";
        }

        void Path::resize(size_t size)
        {
            xs_.resize(size);
            ys_.resize(size);
            thetas_.resize(size);
            kappas_.resize(size);
            point_to_segment_idx_.reserve(size);
        }

        // Find nearest anchor segment
        std::pair<int, int> Path::FindNearestSegmentSingle(float x, float y, float heading, bool for_ego,
                                                           bool print) const
        {
            if (anchor_idxs_.empty())
            {
                // If no anchor segments, use the entire path
                return std::make_pair(0, static_cast<int>(xs_.size()) - 1);
            }

            float min_proj_dist = utils::MAX_VALUE;
            int   best_segment = 0;

            // Iterate over all segments, find the one with minimum projection distance
            for (size_t i = 0; i < anchor_idxs_.size(); ++i)
            {
                // Skip if the heading difference between point and segment is too large
                if (for_ego && utils::NormalizeAngle(heading - anchor_thetas_[i]) > M_PI_2)
                {
                    continue;
                }

                // Compute projection distance from point to segment
                float dx = anchor_dx_[i];
                float dy = anchor_dy_[i];
                float segment_length_sq = anchor_len_sq_[i];

                if (segment_length_sq < 1e-6f) // Handle degenerate case
                {
                    continue;
                }

                float qx = x - anchor_xs_[i].first;
                float qy = y - anchor_ys_[i].first;

                // Compute projection parameter t
                float t = (qx * dx + qy * dy) / segment_length_sq;
                t = std::max(0.0f, std::min(1.0f, t)); // Clamp to [0,1] range

                // Compute the projected point
                float proj_x = anchor_xs_[i].first + t * dx;
                float proj_y = anchor_ys_[i].first + t * dy;

                // Compute distance to the projected point
                float proj_dist = SquaredDistance(x, y, proj_x, proj_y);

                if (print)
                {
                    std::cout << "i: " << i << " anchor_xs_[i]: " << anchor_xs_[i].first << " " << anchor_xs_[i].second
                              << std::endl;
                    std::cout << "i: " << i << " anchor_ys_[i]: " << anchor_ys_[i].first << " " << anchor_ys_[i].second
                              << std::endl;
                    std::cout << "i: " << i << " anchor_idxs_[i]: " << anchor_idxs_[i].first << " "
                              << anchor_idxs_[i].second << std::endl;
                    std::cout << "i: " << i << " proj_dist: " << proj_dist << std::endl;
                }

                if (proj_dist < min_proj_dist)
                {
                    min_proj_dist = proj_dist;
                    best_segment = i;
                }
            }

            return anchor_idxs_[best_segment];
        }

        void Path::FindNearestSegmentBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &begin_idxs,
                                           IVectorT_1 &end_idxs, bool print) const
        {
            if (anchor_idxs_.empty())
            {
                // If no anchor segments exist, use the entire path
                begin_idxs = IVectorT_1::fill(0);
                end_idxs = IVectorT_1::fill(xs_.size() - 1);
                return;
            }

            // Initialize minimum distance and best segment index
            FVectorT_1 min_proj_dists = FVectorT_1::fill(utils::MAX_VALUE);
            IVectorT_1 best_segments = IVectorT_1::fill(0);

            AlignedVectorInt begin_idxs_array(IVectorT_1::num_scalars, 0);
            AlignedVectorInt end_idxs_array(IVectorT_1::num_scalars, xs_.size() - 1);

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
                FVectorT_1 qx = xs - anchor_xs_[i].first;
                FVectorT_1 qy = ys - anchor_ys_[i].first;

                // Compute projection parameter t (vectorized)
                FVectorT_1 t = (qx * dx + qy * dy) / segment_length_sq;
                t = t.clamp(0.0f, 1.0f); // Clamp to [0,1] range

                // Compute projected point
                FVectorT_1 proj_x = anchor_xs_[i].first + t * dx;
                FVectorT_1 proj_y = anchor_ys_[i].first + t * dy;

                // Compute distance to projected point
                FVectorT_1 delta_x = xs - proj_x;
                FVectorT_1 delta_y = ys - proj_y;
                FVectorT_1 proj_dists = delta_x * delta_x + delta_y * delta_y;

                // Update minimum distance and best segment
                FVectorT_1 is_better = proj_dists < min_proj_dists;
                min_proj_dists = FVectorT_1::select(is_better, proj_dists, min_proj_dists);
                best_segments = IVectorT_1::select(is_better.template as<IVectorT_1>(), i, best_segments);
            }

            auto best_segments_array = best_segments.to_array();
            for (int i = 0; i < best_segments_array.size(); ++i)
            {
                begin_idxs_array[i] = anchor_idxs_[best_segments_array[i]].first;
                end_idxs_array[i] = anchor_idxs_[best_segments_array[i]].second;
            }

            // Set begin_idxs and end_idxs based on best segment indices
            begin_idxs = IVectorT_1(begin_idxs_array.data());
            end_idxs = IVectorT_1(end_idxs_array.data());
        }

        // *Unknown initial point, find the nearest point on the refline*
        // __attribute__((optimize("O1")))
        int Path::NearestSerial(float x, float y, float heading, bool for_ego, bool print) const
        {
            int closest_idx = 0;

            auto nearest_segment_idx = FindNearestSegmentSingle(x, y, heading, for_ego, print);
            int  begin_idx = nearest_segment_idx.first;
            int  end_idx = nearest_segment_idx.second;

            if (print)
            {
                std::cout << "begin_idx: " << begin_idx << " end_idx: " << end_idx << std::endl;
            }

            // Improved binary search with three-point comparison
            while (end_idx - begin_idx >= FloatVectorWidth)
            {
                int mid_idx = (begin_idx + end_idx) / 2;

                float distance_to_begin = SquaredDistance(x, y, xs_[begin_idx], ys_[begin_idx]);
                float distance_to_mid = SquaredDistance(x, y, xs_[mid_idx], ys_[mid_idx]);
                float distance_to_end = SquaredDistance(x, y, xs_[end_idx], ys_[end_idx]);

                if (distance_to_begin <= distance_to_mid && distance_to_begin <= distance_to_end)
                {
                    // Begin point is closest, search first half
                    end_idx = mid_idx;
                }
                else if (distance_to_end <= distance_to_mid && distance_to_end <= distance_to_begin)
                {
                    // End point is closest, search second half
                    begin_idx = mid_idx;
                }
                else
                {
                    // Mid point is closest, need finer search
                    // Check a small range near mid point to determine search direction
                    int quarter_left = (begin_idx + mid_idx) / 2;
                    int quarter_right = (mid_idx + end_idx) / 2;

                    float distance_to_qleft = SquaredDistance(x, y, xs_[quarter_left], ys_[quarter_left]);
                    float distance_to_qright = SquaredDistance(x, y, xs_[quarter_right], ys_[quarter_right]);

                    if (distance_to_qleft <= distance_to_qright)
                    {
                        end_idx = mid_idx;
                    }
                    else
                    {
                        begin_idx = mid_idx;
                    }
                }
            }

            // Use the serial machenism
            float min_dist = utils::MAX_VALUE;
            for (int i = begin_idx; i <= end_idx; ++i)
            {
                float distance = SquaredDistance(x, y, xs_[i], ys_[i]);
                if (distance < min_dist)
                {
                    min_dist = distance;
                    closest_idx = i;
                }
            }

            return closest_idx;
        }

        // __attribute__((optimize("O1")))
        int Path::NearestSerial(float x, float y, float heading, float &min_dist, bool for_ego, bool print) const
        {
            int closest_idx = -1;

            auto nearest_segment_idx = FindNearestSegmentSingle(x, y, heading, for_ego);
            int  begin_idx = nearest_segment_idx.first;
            int  end_idx = nearest_segment_idx.second;

            // Improved binary search with three-point comparison
            while (end_idx - begin_idx >= FloatVectorWidth)
            {
                int mid_idx = (begin_idx + end_idx) / 2;

                float distance_to_begin = SquaredDistance(x, y, xs_[begin_idx], ys_[begin_idx]);
                float distance_to_mid = SquaredDistance(x, y, xs_[mid_idx], ys_[mid_idx]);
                float distance_to_end = SquaredDistance(x, y, xs_[end_idx], ys_[end_idx]);

                if (distance_to_begin <= distance_to_mid && distance_to_begin <= distance_to_end)
                {
                    // Begin point is closest, search first half
                    end_idx = mid_idx;
                }
                else if (distance_to_end <= distance_to_mid && distance_to_end <= distance_to_begin)
                {
                    // End point is closest, search second half
                    begin_idx = mid_idx;
                }
                else
                {
                    // Mid point is closest, need finer search
                    // Check a small range near mid point to determine search direction
                    int quarter_left = (begin_idx + mid_idx) / 2;
                    int quarter_right = (mid_idx + end_idx) / 2;

                    float distance_to_qleft = SquaredDistance(x, y, xs_[quarter_left], ys_[quarter_left]);
                    float distance_to_qright = SquaredDistance(x, y, xs_[quarter_right], ys_[quarter_right]);

                    if (distance_to_qleft <= distance_to_qright)
                    {
                        end_idx = mid_idx;
                    }
                    else
                    {
                        begin_idx = mid_idx;
                    }
                }
            }

            // Use the serial machenism
            min_dist = utils::MAX_VALUE;
            for (int i = begin_idx; i <= end_idx; ++i)
            {
                float distance = SquaredDistance(x, y, xs_[i], ys_[i]);
                if (distance < min_dist)
                {
                    min_dist = distance;
                    closest_idx = i;
                }
            }

            return closest_idx;
        }

        int Path::NearestBatch(float x, float y, float heading)
        {
            int refline_size = xs_.size();
            int closest_idx = -1;

            auto nearest_segment_idx = FindNearestSegmentSingle(x, y, heading, true);
            int  begin_idx = nearest_segment_idx.first;
            int  end_idx = nearest_segment_idx.second;

            // Improved binary search with three-point comparison
            while (end_idx - begin_idx >= FloatVectorWidth)
            {
                int mid_idx = (begin_idx + end_idx) / 2;

                float distance_to_begin = SquaredDistance(x, y, xs_[begin_idx], ys_[begin_idx]);
                float distance_to_mid = SquaredDistance(x, y, xs_[mid_idx], ys_[mid_idx]);
                float distance_to_end = SquaredDistance(x, y, xs_[end_idx], ys_[end_idx]);

                if (distance_to_begin <= distance_to_mid && distance_to_begin <= distance_to_end)
                {
                    // Begin point is closest, search first half
                    end_idx = mid_idx;
                }
                else if (distance_to_end <= distance_to_mid && distance_to_end <= distance_to_begin)
                {
                    // End point is closest, search second half
                    begin_idx = mid_idx;
                }
                else
                {
                    // Mid point is closest, need finer search
                    // Check a small range near mid point to determine search direction
                    int quarter_left = (begin_idx + mid_idx) / 2;
                    int quarter_right = (mid_idx + end_idx) / 2;

                    float distance_to_quarter_left = SquaredDistance(x, y, xs_[quarter_left], ys_[quarter_left]);
                    float distance_to_quarter_right = SquaredDistance(x, y, xs_[quarter_right], ys_[quarter_right]);

                    if (distance_to_quarter_left <= distance_to_quarter_right)
                    {
                        end_idx = mid_idx;
                    }
                    else
                    {
                        begin_idx = mid_idx;
                    }
                }
            }

            // Use the parallel machenism of SIMD
            begin_idx = std::min(refline_size - FloatVectorWidth, begin_idx);

            // Load multiple data points at once using load_contiguous
            FVectorT_1 delta_xs = FVectorT_1::load_contiguous_unaligned(xs_.data(), begin_idx) - x;
            FVectorT_1 delta_ys = FVectorT_1::load_contiguous_unaligned(ys_.data(), begin_idx) - y;

            FVectorT_1 dists = delta_xs * delta_xs + delta_ys * delta_ys;
            const auto dists_array = dists.to_array();

            int   min_index = 0;
            float min_dist = utils::MAX_VALUE;
            for (int j = 0; j < end_idx - begin_idx; ++j)
            {
                if (dists_array[j] < min_dist)
                {
                    min_dist = dists_array[j];
                    min_index = j;
                }
            }
            return min_index + begin_idx;
        }

        int Path::NearestBatch(float x, float y, float heading, float &min_dist)
        {
            int refline_size = xs_.size();
            int closest_idx = -1;

            auto nearest_segment_idx = FindNearestSegmentSingle(x, y, heading, true);
            int  begin_idx = nearest_segment_idx.first;
            int  end_idx = nearest_segment_idx.second;

            // Improved binary search with three-point comparison
            while (end_idx - begin_idx >= FloatVectorWidth)
            {
                int mid_idx = (begin_idx + end_idx) / 2;

                float distance_to_begin = SquaredDistance(x, y, xs_[begin_idx], ys_[begin_idx]);
                float distance_to_mid = SquaredDistance(x, y, xs_[mid_idx], ys_[mid_idx]);
                float distance_to_end = SquaredDistance(x, y, xs_[end_idx], ys_[end_idx]);

                if (distance_to_begin <= distance_to_mid && distance_to_begin <= distance_to_end)
                {
                    // Begin point is closest, search first half
                    end_idx = mid_idx;
                }
                else if (distance_to_end <= distance_to_mid && distance_to_end <= distance_to_begin)
                {
                    // End point is closest, search second half
                    begin_idx = mid_idx;
                }
                else
                {
                    // Mid point is closest, need finer search
                    // Check a small range near mid point to determine search direction
                    int quarter_left = (begin_idx + mid_idx) / 2;
                    int quarter_right = (mid_idx + end_idx) / 2;

                    float distance_to_quarter_left = SquaredDistance(x, y, xs_[quarter_left], ys_[quarter_left]);
                    float distance_to_quarter_right = SquaredDistance(x, y, xs_[quarter_right], ys_[quarter_right]);

                    if (distance_to_quarter_left <= distance_to_quarter_right)
                    {
                        end_idx = mid_idx;
                    }
                    else
                    {
                        begin_idx = mid_idx;
                    }
                }
            }

            // Use the parallel machenism of SIMD
            begin_idx = std::min(refline_size - FloatVectorWidth, begin_idx);

            // Load multiple data points at once using load_contiguous
            FVectorT_1 delta_xs = FVectorT_1::load_contiguous_unaligned(xs_.data(), begin_idx) - x;
            FVectorT_1 delta_ys = FVectorT_1::load_contiguous_unaligned(ys_.data(), begin_idx) - y;

            FVectorT_1 dists = delta_xs * delta_xs + delta_ys * delta_ys;
            const auto dists_array = dists.to_array();

            int min_index = 0;
            min_dist = utils::MAX_VALUE;
            for (int j = 0; j < end_idx - begin_idx; ++j)
            {
                if (dists_array[j] < min_dist)
                {
                    min_dist = dists_array[j];
                    min_index = j;
                }
            }
            return min_index + begin_idx;
        }

        AlignedVectorInt Path::NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, int valid_num, bool print)
        {
            int refline_size = xs_.size();

            AlignedVectorInt closest_idxs(FVectorT_1::num_scalars, 0);

            // Determine search range using anchor segments
            IVectorT_1 begin_idxs, end_idxs;
            FindNearestSegmentBatch(xs, ys, begin_idxs, end_idxs, print);

            constexpr int NUM_WIDTH_1 = FVectorT_1::num_scalars_per_row;

            while ((end_idxs - begin_idxs).hmax() >= NUM_WIDTH_1)
            {
                IVectorT_1 mid_idxs = (begin_idxs + end_idxs) / 2;

                // Compute distances to three points
                FVectorT_1 delta_begin_xs = FVectorT_1::gather(xs_.data(), begin_idxs) - xs;
                FVectorT_1 delta_begin_ys = FVectorT_1::gather(ys_.data(), begin_idxs) - ys;
                FVectorT_1 delta_mid_xs = FVectorT_1::gather(xs_.data(), mid_idxs) - xs;
                FVectorT_1 delta_mid_ys = FVectorT_1::gather(ys_.data(), mid_idxs) - ys;
                FVectorT_1 delta_end_xs = FVectorT_1::gather(xs_.data(), end_idxs) - xs;
                FVectorT_1 delta_end_ys = FVectorT_1::gather(ys_.data(), end_idxs) - ys;

                FVectorT_1 distance_to_begin = delta_begin_xs * delta_begin_xs + delta_begin_ys * delta_begin_ys;
                FVectorT_1 distance_to_mid = delta_mid_xs * delta_mid_xs + delta_mid_ys * delta_mid_ys;
                FVectorT_1 distance_to_end = delta_end_xs * delta_end_xs + delta_end_ys * delta_end_ys;

                // Three-point comparison logic
                IVectorT_1 begin_is_min =
                    ((distance_to_begin <= distance_to_mid) & (distance_to_begin <= distance_to_end))
                        .template as<IVectorT_1>();
                IVectorT_1 end_is_min = ((distance_to_end <= distance_to_mid) & (distance_to_end <= distance_to_begin))
                                            .template as<IVectorT_1>();
                IVectorT_1 mid_is_min = ~(begin_is_min | end_is_min);

                if (mid_is_min.none())
                {
                    begin_idxs = IVectorT_1::select(begin_is_min, begin_idxs, mid_idxs);
                    end_idxs = IVectorT_1::select(begin_is_min, mid_idxs, end_idxs);
                }
                else
                {
                    // For cases where mid point is closest, subdivide further
                    IVectorT_1 quarter_left = (begin_idxs + mid_idxs) / 2;
                    IVectorT_1 quarter_right = (mid_idxs + end_idxs) / 2;

                    FVectorT_1 delta_qleft_xs = FVectorT_1::gather(xs_.data(), quarter_left) - xs;
                    FVectorT_1 delta_qleft_ys = FVectorT_1::gather(ys_.data(), quarter_left) - ys;
                    FVectorT_1 delta_qright_xs = FVectorT_1::gather(xs_.data(), quarter_right) - xs;
                    FVectorT_1 delta_qright_ys = FVectorT_1::gather(ys_.data(), quarter_right) - ys;

                    FVectorT_1 distance_to_qleft = delta_qleft_xs * delta_qleft_xs + delta_qleft_ys * delta_qleft_ys;
                    FVectorT_1 distance_to_qright =
                        delta_qright_xs * delta_qright_xs + delta_qright_ys * delta_qright_ys;

                    IVectorT_1 choose_left_quarter =
                        (distance_to_qleft <= distance_to_qright).template as<IVectorT_1>();
                    IVectorT_1 mid_choose_left = mid_is_min & choose_left_quarter;
                    IVectorT_1 mid_choose_right = mid_is_min & ~choose_left_quarter;

                    // Update indices
                    begin_idxs = IVectorT_1::select(begin_is_min | mid_choose_left, begin_idxs, mid_idxs);
                    end_idxs = IVectorT_1::select(begin_is_min | mid_choose_left, mid_idxs, end_idxs);
                }
            }

            // **SIMD: compute final nearest points**
            begin_idxs = begin_idxs.min(refline_size - NUM_WIDTH_1);

            for (int i = 0; i < valid_num; ++i)
            {
                int        begin_idx = begin_idxs[{0, i}];
                FVectorT_1 delta_xs = FVectorT_1::load_contiguous_unaligned(xs_.data(), begin_idx) - xs[{0, i}];
                FVectorT_1 delta_ys = FVectorT_1::load_contiguous_unaligned(ys_.data(), begin_idx) - ys[{0, i}];

                FVectorT_1 dists = delta_xs * delta_xs + delta_ys * delta_ys;

                const auto dists_array = dists.to_array();
                int        min_index = 0;
                float      min_dist = utils::MAX_VALUE;
                for (int j = 0; j < end_idxs[{0, i}] - begin_idx; ++j)
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

        // *Known initial point, find the nearest point on the refline*
        void Path::NearestSerial(float x, float y, int &previous_nearest_idx, const int &step_direction) const
        {
            int refline_size = xs_.size();

            // Clamp previous_nearest_idx to valid range

            // Initialize counter and minimum distance
            int   counts = 0;
            float min_dist = utils::MAX_VALUE;
            previous_nearest_idx =
                std::max(0, std::min(refline_size - 1, previous_nearest_idx - utils::TRACEBACK_STEPS * step_direction));
            int min_idx = previous_nearest_idx; // Initialize to current index

            // Patience threshold for early termination
            int count_limit = 3;

            bool valid_mask = true;

            while (valid_mask) // Continue while any point still needs processing
            {
                // Fetch coordinates at current index and compute distance
                float delta_xs = xs_[previous_nearest_idx] - x;
                float delta_ys = ys_[previous_nearest_idx] - y;
                float distance = delta_xs * delta_xs + delta_ys * delta_ys;

                // Update minimum distance and index
                if (distance < min_dist)
                {
                    min_dist = distance;
                    min_idx = previous_nearest_idx;
                    counts = 0;
                }
                else
                {
                    ++counts;
                }

                // Update current index
                previous_nearest_idx = previous_nearest_idx + step_direction;

                // Update validity mask
                if (previous_nearest_idx < 0 || previous_nearest_idx >= refline_size || counts >= count_limit)
                {
                    valid_mask = false;
                }
            }

            previous_nearest_idx = min_idx;
        }

        // __attribute__((optimize("O1")))
        void Path::NearestSerial(float x, float y, int &previous_nearest_idx, float &min_dist,
                                 const int &step_direction) const
        {
            int refline_size = xs_.size();

            // Clamp previous_nearest_idx to valid range
            int curr_idx =
                std::max(0, std::min(refline_size - 1, previous_nearest_idx - utils::TRACEBACK_STEPS * step_direction));

            // Initialize counter and minimum distance
            int counts = 0;
            int min_idx = curr_idx; // Initialize to current index

            // Create comparison threshold
            int count_limit = 3;

            bool valid_mask = true;

            min_dist = utils::MAX_VALUE;

            while (valid_mask) // Continue while any point still needs processing
            {
                // Fetch coordinates at current index and compute distance
                float delta_xs = xs_[curr_idx] - x;
                float delta_ys = ys_[curr_idx] - y;
                float distances = delta_xs * delta_xs + delta_ys * delta_ys;

                // Update minimum distance and index
                if (distances < min_dist)
                {
                    min_dist = distances;
                    min_idx = curr_idx;
                    counts = 0;
                }
                else
                {
                    counts++;
                }

                // Update current index
                curr_idx = curr_idx + step_direction;

                // Update validity mask
                valid_mask = valid_mask & (counts < count_limit) & (curr_idx >= 0) & (curr_idx < refline_size);
            }

            min_dist = sqrt(min_dist);

            previous_nearest_idx = min_idx;
        }

        void Path::NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &previous_nearest_idxs,
                                const IVectorT_1 &step_directions) const
        {
            int refline_size = xs_.size();

            // Initialize counter and minimum distance vectors
            previous_nearest_idxs =
                (previous_nearest_idxs - utils::TRACEBACK_STEPS * step_directions).clamp(0, refline_size - 1);
            IVectorT_1 min_idxs = previous_nearest_idxs; // Initialize to current index

            FVectorT_1 min_dists = DistSqBatch<FVectorT_1, IVectorT_1>(xs, ys, xs_, ys_, previous_nearest_idxs);

            IVectorT_1 patience_counts = IVectorT_1::fill(0);

            // Initialize validity mask
            IVectorT_1 valid_mask = IVectorT_1::fill(0xFFFFFFFF);

            for (int i = 0; i < utils::NEAREST_SEARCH_MAX_STEPS; ++i)
            {
                // Probe next step
                // Update current index
                IVectorT_1 next_idxs = previous_nearest_idxs + step_directions;

                // Update validity mask
                valid_mask = valid_mask & (next_idxs >= 0) & (next_idxs < refline_size);

                // === Core optimization: early termination check ===
                // Exit immediately if no lane needs further searching
                if (valid_mask.none())
                    break;

                // Only update lanes still searching to avoid out-of-bounds access outside valid_mask (clamp provides
                // protection)
                previous_nearest_idxs = IVectorT_1::select(valid_mask, next_idxs, previous_nearest_idxs);

                // Compute next step distance (most expensive operation: 2 gathers)
                FVectorT_1 next_dists = DistSqBatch<FVectorT_1, IVectorT_1>(xs, ys, xs_, ys_, previous_nearest_idxs);

                // Check if improved
                // Only compare for lanes still searching
                IVectorT_1 is_improved = (next_dists < min_dists).template as<IVectorT_1>() & valid_mask;

                // Update best state
                // If a closer point is found, update min_dists and best_idxs
                min_dists = FVectorT_1::select(is_improved.template as<FVectorT_1>(), next_dists, min_dists);
                min_idxs = IVectorT_1::select(is_improved, previous_nearest_idxs, min_idxs);

                // Update patience counter
                // If improved: reset counter to 0
                // Otherwise: increment counter by 1
                // Note: select logic is (mask, true_val, false_val)
                patience_counts = IVectorT_1::select(is_improved, IVectorT_1::fill(0), patience_counts + 1);

                // F. Update search state
                // Stop searching if patience >= limit
                IVectorT_1 patience_exhausted = (patience_counts >= utils::NEAREST_SEARCH_PATIENCE_LIMIT);
                valid_mask = valid_mask & (~patience_exhausted);

                // G. Early termination check
                if (valid_mask.none())
                    break;
            }

            previous_nearest_idxs = min_idxs;
        }

        void Path::NearestBatch(const FVectorT_1 &xs, const FVectorT_1 &ys, IVectorT_1 &previous_nearest_idxs,
                                const IVectorT_1 &step_directions, IVectorT_1 valid_mask) const
        {
            if (valid_mask.none())
                return;

            int refline_size = xs_.size();

            // Initialize counter and minimum distance vectors
            previous_nearest_idxs =
                (previous_nearest_idxs - utils::TRACEBACK_STEPS * step_directions).clamp(0, refline_size - 1);
            IVectorT_1 min_idxs = previous_nearest_idxs; // Initialize to current index

            FVectorT_1 min_dists = DistSqBatch<FVectorT_1, IVectorT_1>(xs, ys, xs_, ys_, previous_nearest_idxs);

            IVectorT_1 patience_counts = IVectorT_1::fill(0);

            for (int i = 0; i < utils::NEAREST_SEARCH_MAX_STEPS; ++i)
            {
                // Probe next step
                // Update current index
                IVectorT_1 next_idxs = previous_nearest_idxs + step_directions;

                // Update validity mask
                valid_mask = valid_mask & (next_idxs >= 0) & (next_idxs < refline_size);

                // === Core optimization: early termination check ===
                // Exit immediately if no lane needs further searching
                if (valid_mask.none())
                    break;

                // Only update lanes still searching to avoid out-of-bounds access outside valid_mask (clamp provides
                // protection)
                previous_nearest_idxs = IVectorT_1::select(valid_mask, next_idxs, previous_nearest_idxs);

                // Compute next step distance (most expensive operation: 2 gathers)
                FVectorT_1 next_dists = DistSqBatch<FVectorT_1, IVectorT_1>(xs, ys, xs_, ys_, previous_nearest_idxs);

                // Check if improved
                // Only compare for lanes still searching
                IVectorT_1 is_improved = (next_dists < min_dists).template as<IVectorT_1>() & valid_mask;

                // Update best state
                // If a closer point is found, update min_dists and best_idxs
                min_dists = FVectorT_1::select(is_improved.template as<FVectorT_1>(), next_dists, min_dists);
                min_idxs = IVectorT_1::select(is_improved, previous_nearest_idxs, min_idxs);

                // Update patience counter
                // If improved: reset counter to 0
                // Otherwise: increment counter by 1
                // Note: select logic is (mask, true_val, false_val)
                patience_counts = IVectorT_1::select(is_improved, IVectorT_1::fill(0), patience_counts + 1);

                // F. Update search state
                // Stop searching if patience >= limit
                IVectorT_1 patience_exhausted = (patience_counts >= utils::NEAREST_SEARCH_PATIENCE_LIMIT);
                valid_mask = valid_mask & (~patience_exhausted);

                // G. Early termination check
                if (valid_mask.none())
                    break;
            }

            previous_nearest_idxs = min_idxs;
        }

        // Find interpolated nearest points for a batch of points given the previous nearest idxs
        void Path::InterpolatedNearest(const float &x, const float &y, const int &nearest_idx, float &interpolated_x,
                                       float &interpolated_y, float &interpolated_theta,
                                       float &interpolated_path_index) const
        {
            int   prev_nearest_idx = nearest_idx - 1;
            int   next_nearest_idx = prev_nearest_idx + 1;
            float dx = xs_[next_nearest_idx] - xs_[prev_nearest_idx];
            float dy = ys_[next_nearest_idx] - ys_[prev_nearest_idx];
            float qdx = x - xs_[prev_nearest_idx];
            float qdy = y - ys_[prev_nearest_idx];

            float t = (qdx * dx + qdy * dy) / (dx * dx + dy * dy + 1e-6f);
            interpolated_x = xs_[prev_nearest_idx] + t * dx;
            interpolated_y = ys_[prev_nearest_idx] + t * dy;
            interpolated_theta = thetas_[prev_nearest_idx] +
                                 t * utils::NormalizeAngle(thetas_[next_nearest_idx] - thetas_[prev_nearest_idx]);
            interpolated_path_index = prev_nearest_idx + t;
        }

        float Path::PointOnLeft(float x, float y, const int &nearest_index) const
        {
            float theta_r = thetas_[nearest_index];
            return cos(theta_r) * (y - ys_[nearest_index]) - (sin(theta_r)) * (x - xs_[nearest_index]) > 0.0f ? 1.0f
                                                                                                              : -1.0f;
        }

        float Path::PointOnLeft(float x, float y, float theta_r, const int &nearest_index) const
        {
            return cos(theta_r) * (y - ys_[nearest_index]) - (sin(theta_r)) * (x - xs_[nearest_index]) > 0.0f ? 1.0f
                                                                                                              : -1.0f;
        }
    } // namespace utils
} // namespace vec_qmdp
