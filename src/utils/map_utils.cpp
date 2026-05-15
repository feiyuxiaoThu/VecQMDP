/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <set>
#include <utils/map_utils.hpp>
#include <utils/params.hpp>

namespace vec_qmdp
{
    namespace utils
    {
        MapUtils::MapUtils() {}

        MapUtils::~MapUtils() { ResetPath(); }

        void UpdateAnchorSegments(std::shared_ptr<Path> extended_path, std::shared_ptr<Path> pred_path, float *dst_x,
                                  float *dst_y, int dst_offset, int copy_size)
        {
            int begin_idx = 0;
            // Store anchor segment
            if (pred_path->IsStraight())
            {
                // std::cout << "pred_path is straight begin idx: " << dst_offset << " end idx: " << dst_offset +
                // copy_size - 1 << std::endl;
                extended_path->anchor_xs_.emplace_back(std::make_pair(dst_x[0], dst_x[copy_size - 1]));
                extended_path->anchor_ys_.emplace_back(std::make_pair(dst_y[0], dst_y[copy_size - 1]));
                extended_path->anchor_dx_.emplace_back(dst_x[copy_size - 1] - dst_x[0]);
                extended_path->anchor_dy_.emplace_back(dst_y[copy_size - 1] - dst_y[0]);
                extended_path->anchor_thetas_.emplace_back(extended_path->GetTheta(dst_offset));
                extended_path->anchor_len_sq_.emplace_back(
                    extended_path->anchor_dx_.back() * extended_path->anchor_dx_.back() +
                    extended_path->anchor_dy_.back() * extended_path->anchor_dy_.back());
                extended_path->anchor_idxs_.emplace_back(std::make_pair(dst_offset, dst_offset + copy_size - 1));
            }
            else
            {
                // split it in segments
                int begin_idx = 0;
                for (; begin_idx < copy_size - CURVATURE_SEGMENT_LENGTH_SIZE;
                     begin_idx += CURVATURE_SEGMENT_LENGTH_SIZE)
                {
                    int end_idx = begin_idx + CURVATURE_SEGMENT_LENGTH_SIZE;
                    // std::cout << "begin idx: " << begin_idx + dst_offset << " end idx: " << end_idx + dst_offset <<
                    // std::endl;
                    extended_path->anchor_xs_.emplace_back(std::make_pair(dst_x[begin_idx], dst_x[end_idx]));
                    extended_path->anchor_ys_.emplace_back(std::make_pair(dst_y[begin_idx], dst_y[end_idx]));
                    extended_path->anchor_dx_.emplace_back(dst_x[end_idx] - dst_x[begin_idx]);
                    extended_path->anchor_dy_.emplace_back(dst_y[end_idx] - dst_y[begin_idx]);
                    extended_path->anchor_thetas_.emplace_back(extended_path->GetTheta(begin_idx + dst_offset));
                    extended_path->anchor_len_sq_.emplace_back(
                        extended_path->anchor_dx_.back() * extended_path->anchor_dx_.back() +
                        extended_path->anchor_dy_.back() * extended_path->anchor_dy_.back());
                    extended_path->anchor_idxs_.emplace_back(
                        std::make_pair(begin_idx + dst_offset, end_idx + dst_offset));
                }

                // Prepare remaining segments
                // std::cout << "remaining begin idx: " << begin_idx + dst_offset << " end idx: " << copy_size - 1 +
                // dst_offset << std::endl;
                extended_path->anchor_xs_.emplace_back(std::make_pair(dst_x[begin_idx], dst_x[copy_size - 1]));
                extended_path->anchor_ys_.emplace_back(std::make_pair(dst_y[begin_idx], dst_y[copy_size - 1]));
                extended_path->anchor_dx_.emplace_back(dst_x[copy_size - 1] - dst_x[begin_idx]);
                extended_path->anchor_dy_.emplace_back(dst_y[copy_size - 1] - dst_y[begin_idx]);
                extended_path->anchor_thetas_.emplace_back(extended_path->GetTheta(begin_idx + dst_offset));
                extended_path->anchor_len_sq_.emplace_back(
                    extended_path->anchor_dx_.back() * extended_path->anchor_dx_.back() +
                    extended_path->anchor_dy_.back() * extended_path->anchor_dy_.back());
                extended_path->anchor_idxs_.emplace_back(
                    std::make_pair(begin_idx + dst_offset, copy_size - 1 + dst_offset));
            }
        }

        /**
         * Get the neighbor-filtered successor ID, avoiding lane skipping and merge issues
         * @param edge_id Current edge ID
         * @param is_left Whether it is the left neighbor (true for left, false for right)
         * @param curr_successors Successors of the current edge
         * @return Filtered successor ID
         */
        std::string GetFilteredNeighborSuccessorId(MapUtils *map_utils, const std::string &edge_id, bool is_left,
                                                   const std::vector<std::string> &curr_successors)
        {
            if (edge_id.empty() || !map_utils->GLOBAL_ROUTE_EDGE_ID.count(edge_id))
                return "";

            const auto &neighbor_successors = map_utils->GetRouteSuccessorIdsById(edge_id);
            if (neighbor_successors.empty())
                return "";

            // When curr edge has only one successor, find the best-matching neighbor successor
            if (curr_successors.size() == 1)
            {
                const auto &curr_next_successors = map_utils->GetRouteSuccessorIdsById(curr_successors[0]);
                if (!curr_next_successors.empty())
                {
                    std::string target_next_successor =
                        is_left ? map_utils->GetLeftNeighborIdById(curr_next_successors[0])
                                : map_utils->GetRightNeighborIdById(curr_next_successors[0]);

                    // Find the neighbor_successor that leads to target_next_successor
                    for (const auto &successor : neighbor_successors)
                    {
                        const auto &tmp_successors = map_utils->GetRouteSuccessorIdsById(successor);
                        if (!tmp_successors.empty() && tmp_successors[0] == target_next_successor)
                        {
                            return successor;
                        }
                    }
                }
            }
            // When curr edge has multiple successors, avoid selecting one that merges into the neighbor lane
            else if (curr_successors.size() > 1)
            {
                // Collect all next successors of the curr edge
                std::set<std::string> curr_next_successors_set;
                for (const auto &successor : curr_successors)
                {
                    const auto &next_successors = map_utils->GetRouteSuccessorIdsById(successor);
                    if (!next_successors.empty())
                    {
                        curr_next_successors_set.insert(next_successors[0]);
                    }
                }

                // Select a neighbor successor that does not merge into the curr lane
                for (const auto &successor : neighbor_successors)
                {
                    const auto &tmp_successors = map_utils->GetRouteSuccessorIdsById(successor);
                    if (!tmp_successors.empty() &&
                        curr_next_successors_set.find(tmp_successors[0]) == curr_next_successors_set.end())
                    {
                        return successor;
                    }
                }
            }

            // Default: return the first successor
            return neighbor_successors[0];
        }

        /**
         * Get the merge-filtered current lane successor ID, avoiding merges into neighbor lanes
         * @param curr_successors All successors of the current lane
         * @param left_successor_id Successor ID of the left neighbor
         * @param right_successor_id Successor ID of the right neighbor
         * @return Filtered successor ID
         */
        std::string GetFilteredCurrentSuccessorId(MapUtils *map_utils, const std::vector<std::string> &curr_successors,
                                                  const std::string &left_successor_id,
                                                  const std::string &right_successor_id)
        {
            if (curr_successors.size() <= 1)
                return curr_successors.empty() ? "" : curr_successors[0];

            // Get next successors of the left and right neighbors
            std::string left_next_successor_id = "";
            std::string right_next_successor_id = "";

            if (!left_successor_id.empty())
            {
                const auto &left_next_successors = map_utils->GetRouteSuccessorIdsById(left_successor_id);
                if (!left_next_successors.empty())
                {
                    left_next_successor_id = left_next_successors[0];
                }
            }

            if (!right_successor_id.empty())
            {
                const auto &right_next_successors = map_utils->GetRouteSuccessorIdsById(right_successor_id);
                if (!right_next_successors.empty())
                {
                    right_next_successor_id = right_next_successors[0];
                }
            }

            // Select a successor that does not merge into a neighbor lane
            for (const auto &successor : curr_successors)
            {
                const auto &tmp_next_successors = map_utils->GetRouteSuccessorIdsById(successor);
                if (tmp_next_successors.size() > 0)
                {
                    if (tmp_next_successors[0] == left_next_successor_id ||
                        tmp_next_successors[0] == right_next_successor_id)
                    {
                        continue;
                    }
                }
                return successor;
            }

            // If all successors would merge, return the first one
            return curr_successors[0];
        }

        /**
         * Check whether a successor should be replaced by a neighbor
         * When a neighbor's successor's successor matches the successor's successor, prefer the neighbor.
         * Also checks r_squared values; if the neighbor path's r_squared is too small (indicating a curve), skip
         * filtering.
         * @param map_utils Pointer to MapUtils
         * @param edge_id Current edge ID
         * @param successor_id Successor ID to check
         * @param left_neighbor Pre-computed left neighbor ID
         * @param right_neighbor Pre-computed right neighbor ID
         * @param left_successors Pre-computed left neighbor successors
         * @param right_successors Pre-computed right neighbor successors
         * @return True if the successor should be replaced by a neighbor
         */
        bool ShouldReplaceSuccessorWithNeighbor(MapUtils *map_utils, const std::string &edge_id,
                                                const std::string &successor_id, const std::string &left_neighbor,
                                                const std::string              &right_neighbor,
                                                const std::vector<std::string> &left_successors,
                                                const std::vector<std::string> &right_successors)
        {
            if (edge_id.empty() || successor_id.empty())
                return false;

            // Get the successor's next successors
            const auto &successor_next_ids = map_utils->GetRouteSuccessorIdsById(successor_id);
            if (successor_next_ids.empty())
                return false;

            std::string successor_next_id = successor_next_ids[0];

            // Get the successor's r_squared value for comparison
            // std::shared_ptr<Path> successor_path = map_utils->GetRefLine(successor_id);
            // float successor_r_squared = successor_path ? successor_path->weighted_avg_r_squared_ : 1.0f;

            // Check the left neighbor
            if (!left_neighbor.empty() && map_utils->GLOBAL_ROUTE_EDGE_ID.count(left_neighbor))
            {
                for (const auto &left_successor : left_successors)
                {
                    const auto &left_successor_next_ids = map_utils->GetRouteSuccessorIdsById(left_successor);
                    if (!left_successor_next_ids.empty() && left_successor_next_ids[0] == successor_next_id)
                    {
                        // Check r_squared: if the left neighbor's r_squared is too small (indicating a curve), skip
                        // filtering
                        std::shared_ptr<Path> left_successor_path = map_utils->GetRefLine(left_successor);
                        if (left_successor_path)
                        {
                            float left_r_squared = left_successor_path->weighted_avg_r_squared_;
                            if (left_r_squared < utils::NEIGHBOR_RSQUARED_MIN_THRESHOLD)
                            {
                                continue; // Do not use this neighbor for filtering
                            }
                        }
                        return true;
                    }
                }
            }

            // Check the right neighbor
            if (!right_neighbor.empty() && map_utils->GLOBAL_ROUTE_EDGE_ID.count(right_neighbor))
            {
                for (const auto &right_successor : right_successors)
                {
                    const auto &right_successor_next_ids = map_utils->GetRouteSuccessorIdsById(right_successor);
                    if (!right_successor_next_ids.empty() && right_successor_next_ids[0] == successor_next_id)
                    {
                        // Check r_squared: if the right neighbor's r_squared is too small (indicating a curve), skip
                        // filtering
                        std::shared_ptr<Path> right_successor_path = map_utils->GetRefLine(right_successor);
                        if (right_successor_path)
                        {
                            float right_r_squared = right_successor_path->weighted_avg_r_squared_;
                            if (right_r_squared < utils::NEIGHBOR_RSQUARED_MIN_THRESHOLD)
                            {
                                continue; // Do not use this neighbor for filtering
                            }
                        }
                        return true;
                    }
                }
            }

            return false;
        }

        /**
         * Re-align theta values for smooth transition between path segments
         * @param prev_theta Theta value at the end of the previous segment
         * @param next_thetas Pointer to the theta array of the next segment
         * @param size Size of the next segment's theta array
         * @param N Number of points for smooth correction
         */
        void ReAlignHeadings(float prev_theta, float *next_thetas, int size, int N = 20)
        {
            if (size <= 0 || !next_thetas)
                return;

            N = std::min(N, size); // Ensure N does not exceed array size

            // Compute the angular difference
            float delta = utils::NormalizeAngle(prev_theta - next_thetas[0]);

            // Gradually correct over the first N points
            for (int i = 0; i < N; ++i)
            {
                // Construct a smooth decay weight (from 1 to 0)
                float weight = 1.0f - static_cast<float>(i) / static_cast<float>(N - 1);
                if (i == N - 1)
                    weight = 0.0f; // Ensure weight is 0 at the last point

                next_thetas[i] = utils::NormalizeAngle(next_thetas[i] + delta * weight);
            }
        }

        /**
         * Create a forward and backward extended Path for the given edge_id
         * @param edge_id Current edge ID
         * @param successor_id Specified successor ID
         * @param lookback_points_size Number of points for backward extension
         * @param lookahead_points_size Number of points for forward extension, including current path size
         * @param for_lane Path type flag:
         *                 - true (lane): lookback includes all predecessors
         *                 - false (lane_connector): lookback includes current edge and predecessors
         * @return Pointer to the extended Path, or nullptr on failure
         */
        std::shared_ptr<Path> MapUtils::ExtendPath(const std::string &edge_id, const std::string &successor_id,
                                                   std::vector<std::string> &refline_edge_ids,
                                                   std::vector<std::string> &refline_edge_names,
                                                   int lookback_points_size, int lookahead_points_size, bool for_lane)
        {
            // LOG_DS << "ExtendPath called";
            // std::cout << "edge_id: " << edge_id << " successor_id: " << successor_id << std::endl;
            if (edge_id.empty())
                return nullptr;

            // Get the current edge's path
            std::shared_ptr<Path> curr_path = GetRefLine(edge_id);
            if (!curr_path)
                return nullptr;

            // Create a new Path object
            std::shared_ptr<Path> extended_path = std::make_shared<Path>();
            extended_path->lane_id_ = edge_id;
            // std::cout << "curr_path token: " << curr_path->lane_id_ << " " << curr_path->miss_goal_penalty_ <<
            // std::endl; extended_path->miss_goal_penalty_ = curr_path->miss_goal_penalty_;
            int extended_path_size = lookback_points_size + lookahead_points_size;
            utils::PATH_SIZE = extended_path_size;
            std::set<std::string> handled_edge_ids;

            // std::cout << "extended_path_size: " << extended_path_size << std::endl;

            extended_path->resize(extended_path_size);
            extended_path->path_len_ = extended_path_size * utils::PATH_POINT_INTERVAL;

            // ======= Backward extension (predecessor) =======
            int pred_offset = (for_lane) ? lookback_points_size
                                         : std::max(lookback_points_size - static_cast<int>(curr_path->GetSize()), 0);
            int tmp_pred_offset = pred_offset;

            // std::cout << "pred_offset: " << pred_offset << std::endl;

            std::string pred_id = edge_id;

            // Keep fetching predecessors until the required number of points is met or no more predecessors
            while (tmp_pred_offset > 0)
            {
                auto predecessors = GetRoutePredecessorIdsById(pred_id);
                // std::cout << "predecessors size: " << predecessors.size() << std::endl;
                if (predecessors.empty())
                    break;

                // Get predecessor
                pred_id = predecessors[0];
                if (!extended_path->loop_flag_ && handled_edge_ids.find(pred_id) != handled_edge_ids.end())
                    extended_path->loop_flag_ = true;
                handled_edge_ids.insert(pred_id);

                // std::cout << "pred_id: " << pred_id << std::endl;
                std::shared_ptr<Path> pred_path = GetRefLine(pred_id);
                if (!pred_path)
                    break;

                // Take points from the end of the predecessor, working backwards
                int path_size = pred_path->GetSize();
                int copy_size = std::min(path_size, tmp_pred_offset);

                // Get source pointers (reverse copy)
                int          src_offset = path_size - copy_size;
                const float *src_x = pred_path->GetXs().data() + src_offset;
                const float *src_y = pred_path->GetYs().data() + src_offset;
                const float *src_theta = pred_path->GetThetas().data() + src_offset;
                const float *src_kappa = pred_path->GetKappas().data() + src_offset;

                // Get destination pointers
                int    dst_offset = tmp_pred_offset - copy_size;
                float *dst_x = extended_path->GetXs().data() + dst_offset;
                float *dst_y = extended_path->GetYs().data() + dst_offset;
                float *dst_theta = extended_path->GetThetas().data() + dst_offset;
                float *dst_kappa = extended_path->GetKappas().data() + dst_offset;

                // Perform memcpy
                memcpy(dst_x, src_x, copy_size * sizeof(float));
                memcpy(dst_y, src_y, copy_size * sizeof(float));
                memcpy(dst_theta, src_theta, copy_size * sizeof(float));
                memcpy(dst_kappa, src_kappa, copy_size * sizeof(float));

                // Store anchor segment
                UpdateAnchorSegments(extended_path, pred_path, dst_x, dst_y, dst_offset, copy_size);

                tmp_pred_offset -= copy_size;
            }

            // If points are still insufficient, extend using the last available point.
            if (tmp_pred_offset > 0 && tmp_pred_offset < lookback_points_size)
            {
                // Extend using the first valid point.
                float first_x = extended_path->GetX(tmp_pred_offset);
                float first_y = extended_path->GetY(tmp_pred_offset);
                float first_theta = extended_path->GetTheta(tmp_pred_offset);
                float first_kappa = extended_path->GetKappa(tmp_pred_offset);
                float dx = utils::PATH_POINT_INTERVAL * std::cos(first_theta);
                float dy = utils::PATH_POINT_INTERVAL * std::sin(first_theta);

                // Extend with a fixed interval and fix the loop condition to avoid unsigned integer overflow.
                for (size_t i = tmp_pred_offset; i > 0; --i)
                {
                    size_t idx = i - 1;
                    extended_path->GetXs()[idx] = first_x - dx;
                    extended_path->GetYs()[idx] = first_y - dy;
                    extended_path->GetThetas()[idx] = first_theta;
                    extended_path->GetKappas()[idx] = first_kappa;
                    first_x -= dx;
                    first_y -= dy;
                }

                extended_path->anchor_xs_.emplace_back(
                    std::make_pair(extended_path->GetX(0), extended_path->GetX(tmp_pred_offset)));
                extended_path->anchor_ys_.emplace_back(
                    std::make_pair(extended_path->GetY(0), extended_path->GetY(tmp_pred_offset)));
                extended_path->anchor_dx_.emplace_back(extended_path->GetX(tmp_pred_offset) - extended_path->GetX(0));
                extended_path->anchor_dy_.emplace_back(extended_path->GetY(tmp_pred_offset) - extended_path->GetY(0));
                extended_path->anchor_thetas_.emplace_back(first_theta);
                extended_path->anchor_len_sq_.emplace_back(
                    extended_path->anchor_dx_.back() * extended_path->anchor_dx_.back() +
                    extended_path->anchor_dy_.back() * extended_path->anchor_dy_.back());
                extended_path->anchor_idxs_.emplace_back(std::make_pair(0, tmp_pred_offset));
            }

            // ======= Copy points from the current edge =======
            int curr_path_size = curr_path->GetSize();
            int curr_offset = pred_offset, copy_size, src_offset;

            if (for_lane)
            {
                // When for_lane is true, the entire lookback comes from predecessors
                // and the current edge is copied from the beginning.
                copy_size = std::min(curr_path_size, extended_path_size - curr_offset);
                src_offset = 0;
            }
            else
            {
                copy_size = std::min(lookback_points_size, curr_path_size);
                src_offset = curr_path_size - copy_size;
            }

            // std::cout << "curr_offset: " << curr_offset << " copy_size: " << copy_size << " src_offset: " <<
            // src_offset << std::endl;

            // curr_path->PrintPath();

            if (!extended_path->loop_flag_ && handled_edge_ids.find(curr_path->GetLaneId()) != handled_edge_ids.end())
                extended_path->loop_flag_ = true;
            handled_edge_ids.insert(curr_path->GetLaneId());

            // Get source pointers.
            const float *src_x = curr_path->GetXs().data() + src_offset;
            const float *src_y = curr_path->GetYs().data() + src_offset;
            const float *src_theta = curr_path->GetThetas().data() + src_offset;
            const float *src_kappa = curr_path->GetKappas().data() + src_offset;

            // Get destination pointers.
            float *dst_x = extended_path->GetXs().data() + curr_offset;
            float *dst_y = extended_path->GetYs().data() + curr_offset;
            float *dst_theta = extended_path->GetThetas().data() + curr_offset;
            float *dst_kappa = extended_path->GetKappas().data() + curr_offset;

            // Use memcpy to speed up copying.
            memcpy(dst_x, src_x, copy_size * sizeof(float));
            memcpy(dst_y, src_y, copy_size * sizeof(float));
            memcpy(dst_theta, src_theta, copy_size * sizeof(float));
            memcpy(dst_kappa, src_kappa, copy_size * sizeof(float));

            // Smooth the heading on the current edge so that it connects smoothly with the predecessor path.
            if (curr_offset > 0 && copy_size > 0)
            {
                float prev_theta = extended_path->GetTheta(curr_offset - 1);
                ReAlignHeadings(prev_theta, dst_theta, copy_size, 20);
            }

            // Update the edge IDs and starting indices that compose the path.
            extended_path->UpdateComprisedEdges(curr_path->GetLaneId(), curr_offset);
            refline_edge_ids.emplace_back(curr_path->GetLaneId());
            refline_edge_names.emplace_back(curr_path->GetLaneName());

            // Update miss-goal penalty.
            extended_path->miss_goal_penalty_ = curr_path->miss_goal_penalty_;
            if (IsTerminated(curr_path->GetLaneId()))
                extended_path->goal_frenet_s_ = (curr_offset + copy_size) * utils::PATH_POINT_INTERVAL;

            // ======= Update maximum curvature information for the composed path (predecessor + current) =======
            for (size_t i = 0; i < curr_path->max_curvature_vec_.size(); ++i)
            {
                int original_idx = curr_path->max_curvature_idx_vec_[i];

                // 1. Skip segments that end before src_offset in the original path.
                if (original_idx < src_offset)
                    continue;

                // 2. Compute the adjusted index in the extended path.
                int adjusted_idx = original_idx - src_offset + curr_offset;

                // 3. Key fix: clamp the index.
                // Regardless of how long this segment is in the original path, in the extended path
                // its responsibility must end before the successor section starts.
                // This ensures that the predecessor portion is fully covered while not overwriting
                // data that belongs to the successor.
                int current_end_limit = curr_offset + copy_size - 1;

                // If the current segment ends within the copied range or crosses it, process it.
                // Use min to ensure we never write out of bounds.
                int final_idx = std::min(adjusted_idx, current_end_limit);

                extended_path->UpdateMaxCurvatureAndMinDesiredSpeed(curr_path->max_signed_curvature_vec_[i],
                                                                    curr_path->max_curvature_vec_[i],
                                                                    curr_path->min_desired_speed_vec_[i], final_idx);

                // If this segment already reaches the end of the copied range,
                // later segments can be skipped since point_to_segment_idx_ is filled sequentially.
                if (adjusted_idx >= current_end_limit)
                    break;
            }

            // Update red-light position on the extended path.
            if (HasRedTrafficLight(curr_path->GetLaneId()))
            {
                extended_path->red_light_point_s_ =
                    std::min(extended_path->red_light_point_s_, curr_offset * utils::PATH_POINT_INTERVAL);
            }

            // Store anchor segments from the current edge.
            UpdateAnchorSegments(extended_path, curr_path, dst_x, dst_y, curr_offset, copy_size);

            curr_offset += copy_size;

            // ======= Extend forward using successors =======
            std::string           curr_successor_id = successor_id;
            std::shared_ptr<Path> successor_path = GetRefLine(curr_successor_id);
            while (curr_offset < extended_path_size)
            {
                if (!successor_path)
                    break;

                // successor_path->PrintPath();

                // Take points from the beginning of the successor in forward order.
                size_t path_size = successor_path->GetSize();
                size_t copy_size = std::min<size_t>(path_size, extended_path_size - curr_offset);

                // Get source pointers for forward copy.
                const float *src_x = successor_path->GetXs().data();
                const float *src_y = successor_path->GetYs().data();
                const float *src_theta = successor_path->GetThetas().data();
                const float *src_kappa = successor_path->GetKappas().data();

                // Get destination pointers.
                float *dst_x = extended_path->GetXs().data() + curr_offset;
                float *dst_y = extended_path->GetYs().data() + curr_offset;
                float *dst_theta = extended_path->GetThetas().data() + curr_offset;
                float *dst_kappa = extended_path->GetKappas().data() + curr_offset;

                // Perform memcpy copy.
                memcpy(dst_x, src_x, copy_size * sizeof(float));
                memcpy(dst_y, src_y, copy_size * sizeof(float));
                memcpy(dst_theta, src_theta, copy_size * sizeof(float));
                memcpy(dst_kappa, src_kappa, copy_size * sizeof(float));

                // Smooth the successor heading so that it connects smoothly with the previous path segment.
                if (curr_offset > 0 && copy_size > 0)
                {
                    float prev_theta = extended_path->GetTheta(curr_offset - 1);
                    ReAlignHeadings(prev_theta, dst_theta, copy_size, 20);
                }

                // Update the edge IDs and starting indices that compose the extended path.
                extended_path->UpdateComprisedEdges(successor_path->GetLaneId(), curr_offset);
                if (!extended_path->loop_flag_ &&
                    handled_edge_ids.find(successor_path->GetLaneId()) != handled_edge_ids.end())
                    extended_path->loop_flag_ = true;
                handled_edge_ids.insert(successor_path->GetLaneId());

                refline_edge_ids.emplace_back(successor_path->GetLaneId());
                refline_edge_names.emplace_back(successor_path->GetLaneName());

                // Update penalty.
                if (successor_path->miss_goal_penalty_ < 10.0)
                    extended_path->miss_goal_penalty_ =
                        std::max(extended_path->miss_goal_penalty_, successor_path->miss_goal_penalty_);
                if (extended_path->goal_frenet_s_ == 0.0f && IsTerminated(successor_path->GetLaneId()))
                    extended_path->goal_frenet_s_ = (curr_offset + copy_size) * utils::PATH_POINT_INTERVAL;

                // ======= Update maximum curvature information for successor segments in the composed path =======
                for (size_t i = 0; i < successor_path->max_curvature_vec_.size(); ++i)
                {
                    // Compute the mapped index in the extended path.
                    int adjusted_idx = successor_path->max_curvature_idx_vec_[i] + curr_offset;

                    // 1. Bounds check and clamping.
                    // The extended path has length extended_path_size, so any index beyond that
                    // must be clamped to the last valid point.
                    if (adjusted_idx >= extended_path_size - 1)
                    {
                        adjusted_idx = extended_path_size - 1;

                        extended_path->UpdateMaxCurvatureAndMinDesiredSpeed(
                            successor_path->max_signed_curvature_vec_[i], successor_path->max_curvature_vec_[i],
                            successor_path->min_desired_speed_vec_[i], adjusted_idx);

                        // 2. Once we have filled up to the end of the path,
                        //    later successor segments no longer contribute and we can stop here.
                        break;
                    }

                    // Normal update inside valid range.
                    extended_path->UpdateMaxCurvatureAndMinDesiredSpeed(
                        successor_path->max_signed_curvature_vec_[i], successor_path->max_curvature_vec_[i],
                        successor_path->min_desired_speed_vec_[i], adjusted_idx);
                }

                // Update red-light position based on successor edges.
                if (HasRedTrafficLight(successor_path->GetLaneId()))
                {
                    extended_path->red_light_point_s_ =
                        std::min(extended_path->red_light_point_s_, curr_offset * utils::PATH_POINT_INTERVAL);
                }

                // Store anchor segments for the successor path.
                // std::cout << "saving anchor segment path: " << successor_path->lane_id_ << " curr_offset: " <<
                // curr_offset << " copy_size: " << copy_size << std::endl;
                UpdateAnchorSegments(extended_path, successor_path, dst_x, dst_y, curr_offset, copy_size);

                curr_offset += copy_size;

                // std::cout << "curr_offset: " << curr_offset << " copy size: " << copy_size << std::endl;

                const auto &successors = GetRouteSuccessorIdsById(curr_successor_id);
                if (successors.empty())
                    break;

                curr_successor_id = successors[0];
                successor_path = GetRefLine(curr_successor_id);
            }

            // If there are still not enough points, extend using the last point.
            if (curr_offset < extended_path_size && curr_offset != lookback_points_size)
            {
                // Extend using the last valid point.
                size_t last_point_idx = curr_offset - 1;
                float  last_x = extended_path->GetX(last_point_idx);
                float  last_y = extended_path->GetY(last_point_idx);
                float  last_theta = extended_path->GetTheta(last_point_idx);
                float  last_kappa = extended_path->GetKappa(last_point_idx);
                float  dx = utils::PATH_POINT_INTERVAL * std::cos(last_theta);
                float  dy = utils::PATH_POINT_INTERVAL * std::sin(last_theta);

                // Extend with a fixed spatial interval.
                for (size_t i = curr_offset; i < extended_path_size; ++i)
                {
                    extended_path->GetXs()[i] = last_x + dx;
                    extended_path->GetYs()[i] = last_y + dy;
                    extended_path->GetThetas()[i] = last_theta;
                    extended_path->GetKappas()[i] = last_kappa;
                    last_x += dx;
                    last_y += dy;
                }

                if (!extended_path->max_curvature_vec_.empty())
                {
                    extended_path->UpdateMaxCurvatureAndMinDesiredSpeed(
                        extended_path->max_signed_curvature_vec_.back(), extended_path->max_curvature_vec_.back(),
                        extended_path->min_desired_speed_vec_.back(),
                        extended_path_size - 1 // Fill until the last index.
                    );
                }
                else
                {
                    // Fallback default values when no curvature segments are available.
                    extended_path->UpdateMaxCurvatureAndMinDesiredSpeed(0.f, 0.f, utils::FALLBACK_MIN_DESIRED_SPEED,
                                                                        extended_path_size - 1);
                }

                extended_path->anchor_xs_.emplace_back(
                    std::make_pair(extended_path->GetX(curr_offset), extended_path->GetX(extended_path_size - 1)));
                extended_path->anchor_ys_.emplace_back(
                    std::make_pair(extended_path->GetY(curr_offset), extended_path->GetY(extended_path_size - 1)));
                extended_path->anchor_dx_.emplace_back(extended_path->GetX(extended_path_size - 1) -
                                                       extended_path->GetX(curr_offset));
                extended_path->anchor_dy_.emplace_back(extended_path->GetY(extended_path_size - 1) -
                                                       extended_path->GetY(curr_offset));
                extended_path->anchor_thetas_.emplace_back(last_theta);
                extended_path->anchor_len_sq_.emplace_back(
                    extended_path->anchor_dx_.back() * extended_path->anchor_dx_.back() +
                    extended_path->anchor_dy_.back() * extended_path->anchor_dy_.back());
                extended_path->anchor_idxs_.emplace_back(std::make_pair(curr_offset, extended_path_size - 1));
            }

            return extended_path;
        }

        /**
         * Prepare reference paths for the ego vehicle (extended paths for the current edge).
         *
         * For LANE edges:
         * 1. If the distance to the end of the lane is greater than LEAST_DIST2JUNCTION, lane changes are allowed.
         *    - Return: [left neighbor path, current lane path, right neighbor path] (use nullptr when a neighbor does
         * not exist).
         * 2. If the distance to the end of the lane is less than or equal to LEAST_DIST2JUNCTION, successors are
         * preferred.
         *    - Successors are ordered from left to right, at most three paths.
         *    - If there are fewer than three successors, consider left and right neighbor paths that are still on the
         * route.
         *
         * For LANE_CONNECTOR edges:
         * 1. Find the predecessor and build paths based on the predecessor.
         * 2. If fewer than three paths can be built from the predecessor, consider left and right neighbor paths that
         * are on the route.
         * 3. If there is no predecessor, use the current connector and its successors.
         *
         * Note: `ref_paths` is ordered as [left neighbor path < current path < right neighbor path].
         *
         * @param x Current ego x position.
         * @param y Current ego y position.
         * @param v Current ego speed.
         * @param curr_edge_id Current edge ID.
         * @param ref_paths Reference paths for the ego vehicle.
         * @param extra_ref_paths Extra reference paths considered when filtering other vehicles.
         * @param edge_name Edge type: "LANE" or "LANE_CONNECTOR".
         */
        void MapUtils::GetEgoRefPaths(float x, float y, float v, int &curr_ref_path_idx,
                                      std::vector<std::shared_ptr<Path>> &ref_paths,
                                      std::vector<std::shared_ptr<Path>> &extra_ref_paths,
                                      std::vector<std::string>           &refline_edge_ids,
                                      std::vector<std::string> &refline_edge_names, const std::string &last_edge_id,
                                      const std::string &curr_edge_id, const std::string &edge_name,
                                      const std::vector<std::string> &candidate_edge_token, bool &is_close_to_junction)
        {
            // LOG_DS << "GetEgoRefPaths called";

            // Pre-compute commonly used information to avoid repeated work.
            const auto &curr_successors = GetRouteSuccessorIdsById(curr_edge_id);
            std::string successor_id = curr_successors.empty() ? "" : curr_successors[0];

            // Pre-compute neighbor IDs and their successors.
            std::string              left_neighbor = GetLeftNeighborIdById(curr_edge_id);
            std::string              right_neighbor = GetRightNeighborIdById(curr_edge_id);
            std::vector<std::string> left_successors, right_successors;

            if (!left_neighbor.empty())
            {
                left_successors = GetRouteSuccessorIdsById(left_neighbor);
            }
            if (!right_neighbor.empty())
            {
                right_successors = GetRouteSuccessorIdsById(right_neighbor);
            }

            // Pre-compute successors of the current successor (used for later filtering).
            std::string curr_next_successor_id = "";
            if (curr_successors.size() == 1 && !successor_id.empty())
            {
                const auto &tmp_next_successor_ids = GetRouteSuccessorIdsById(successor_id);
                if (!tmp_next_successor_ids.empty())
                {
                    curr_next_successor_id = tmp_next_successor_ids[0];
                }
            }

            // Default backward and forward extension distances.
            float                 lookback_distance = utils::PATH_LOOKBACK_DISTANCE; // Extend 40 m backward.
            std::shared_ptr<Path> curr_path = GetRefLine(curr_edge_id);
            std::shared_ptr<Path> next_path = GetRefLine(successor_id);
            float                 curr_path_len = curr_path ? curr_path->GetPathLen() : 0.0f;
            float                 next_path_len = next_path ? next_path->GetPathLen() : 0.0f;
            float                 lookahead_distance =
                curr_path_len + next_path_len +
                std::max(utils::VELOCITY_TO_DISTANCE_TIME_HORIZON * v, utils::LOOKAHEAD_MIN_DISTANCE);

            // Convert distances to number of points.
            size_t lookback_points_size = static_cast<size_t>(lookback_distance / utils::PATH_POINT_INTERVAL);
            size_t lookahead_points_size = static_cast<size_t>(lookahead_distance / utils::PATH_POINT_INTERVAL);

            // Initialize the reference path slots.
            for (int i = 0; i < ref_paths.size(); ++i)
            {
                ref_paths[i] = nullptr;
            }

            // Clear candidate reference paths.
            extra_ref_paths.clear();

            if (curr_edge_id.empty())
            {
                curr_ref_path_idx = 0;
                return;
            }

            // Shared logic: sort successors and build paths.
            auto buildSortedPaths = [&](const std::string &start_id, const std::vector<std::string> &connections,
                                        int &curr_ref_path_idx, bool is_lane, const std::string &neighbor_left = "",
                                        const std::string              &neighbor_right = "",
                                        const std::vector<std::string> &left_succs = std::vector<std::string>(),
                                        const std::vector<std::string> &right_succs = std::vector<std::string>())
            {
                if (connections.empty())
                {
                    ref_paths[0] = ExtendPath(start_id, "", refline_edge_ids, refline_edge_names, lookback_points_size,
                                              lookahead_points_size, is_lane);
                    curr_ref_path_idx = 0;
                    return;
                }

                std::vector<std::pair<size_t, std::string>> next_ids;
                for (size_t i = 0; i < connections.size(); ++i)
                {
                    auto next = GetSuccessorIdsById(connections[i]);
                    if (!next.empty())
                    {
                        // Filter out connections that would introduce a loop.
                        if (next[0] == last_edge_id || next[0] == start_id)
                        {
                            continue;
                        }

                        // Check whether this successor should be replaced by a neighbor.
                        if (ShouldReplaceSuccessorWithNeighbor(this, start_id, connections[i], neighbor_left,
                                                               neighbor_right, left_succs, right_succs))
                        {
                            continue;
                        }

                        next_ids.emplace_back(std::make_pair(i, next[0]));
                    }
                    else
                    {
                        next_ids.emplace_back(std::make_pair(i, ""));
                    }
                }

                // Backup policy: if all successors are filtered out, still select at least one.
                if (next_ids.empty())
                {
                    if (!is_lane)
                    {
                        for (size_t i = 0; i < 3; ++i)
                        {
                            if (connections[i] == curr_edge_id)
                            {
                                ref_paths[0] =
                                    ExtendPath(start_id, connections[i], refline_edge_ids, refline_edge_names,
                                               lookback_points_size, lookahead_points_size, is_lane);
                                break;
                            }
                        }
                    }
                    if (ref_paths[0] == nullptr)
                    {
                        ref_paths[0] = ExtendPath(start_id, connections[0], refline_edge_ids, refline_edge_names,
                                                  lookback_points_size, lookahead_points_size, is_lane);
                    }
                    curr_ref_path_idx = 0;
                }
                else
                {
                    std::sort(next_ids.begin(), next_ids.end(),
                              [this](const std::pair<size_t, std::string> &a, const std::pair<size_t, std::string> &b)
                              { return ComputeLeftDepth(a.second) < ComputeLeftDepth(b.second); });

                    for (size_t i = 0; i < next_ids.size() && i < 3; ++i)
                    {
                        auto idx = next_ids[i].first;
                        ref_paths[i] = ExtendPath(start_id, connections[idx], refline_edge_ids, refline_edge_names,
                                                  lookback_points_size, lookahead_points_size, is_lane);

                        if (!GLOBAL_ROUTE_EDGE_ID.count(connections[idx]))
                        {
                            ref_paths[i]->not_on_route_ = true;
                        }
                    }

                    if (!is_lane)
                    {
                        bool found = false;
                        if (next_ids.size() == 3)
                        {
                            std::string tmp_connection = connections[next_ids[1].first];
                            if (std::find(candidate_edge_token.begin(), candidate_edge_token.end(), tmp_connection) !=
                                candidate_edge_token.end())
                            {
                                curr_ref_path_idx = 1;
                                found = true;
                            }
                        }
                        if (!found)
                        {
                            for (size_t i = 0; i < next_ids.size() && i < 3; ++i)
                            {
                                auto idx = next_ids[i].first;
                                if (connections[idx] == curr_edge_id)
                                {
                                    curr_ref_path_idx = i;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found)
                        {
                            curr_ref_path_idx = std::min((int)next_ids.size(), 3) / 2;
                        }
                    }
                    else
                    {
                        curr_ref_path_idx = std::min((int)next_ids.size(), 3) / 2;
                    }
                }
            };

            // Helper function to add left and right neighbor paths.
            auto addNeighborPaths =
                [&](const std::string &center_id, int &curr_ref_path_idx, int valid_paths, bool is_lane,
                    const std::vector<std::string> &center_successors = std::vector<std::string>(),
                    const std::string &cached_left_neighbor = "", const std::string &cached_right_neighbor = "",
                    const std::vector<std::string> &cached_left_successors = std::vector<std::string>(),
                    const std::vector<std::string> &cached_right_successors = std::vector<std::string>())
            {
                // Use cached neighbor information when available, otherwise recompute it.
                std::string left_neighbor =
                    cached_left_neighbor.empty() ? GetLeftNeighborIdById(center_id) : cached_left_neighbor;
                std::string right_neighbor =
                    cached_right_neighbor.empty() ? GetRightNeighborIdById(center_id) : cached_right_neighbor;

                // Collect valid neighbor information.
                struct NeighborInfo
                {
                    std::string neighbor_id;
                    std::string successor_id;
                    float       miss_goal_penalty;
                    bool        is_left;
                    bool        not_on_route;
                };

                std::vector<NeighborInfo> valid_neighbors;

                // Use cached successors when available, otherwise recompute them.
                std::vector<std::string> left_neighbor_successors =
                    cached_left_successors.empty() ? (!left_neighbor.empty() ? GetRouteSuccessorIdsById(left_neighbor)
                                                                             : std::vector<std::string>())
                                                   : cached_left_successors;
                std::vector<std::string> right_neighbor_successors =
                    cached_right_successors.empty()
                        ? (!right_neighbor.empty() ? GetRouteSuccessorIdsById(right_neighbor)
                                                   : std::vector<std::string>())
                        : cached_right_successors;

                // Check the left neighbor using the filtering logic.
                if (!left_neighbor.empty())
                {
                    std::string left_successor_id =
                        GetFilteredNeighborSuccessorId(this, left_neighbor, true, center_successors);
                    if (!left_successor_id.empty())
                    {
                        valid_neighbors.push_back({left_neighbor, left_successor_id, ActualMissGoalNum(left_neighbor),
                                                   true, !GLOBAL_ROUTE_EDGE_ID.count(left_successor_id)});
                    }
                }

                // Check the right neighbor using the filtering logic.
                if (!right_neighbor.empty())
                {
                    std::string right_successor_id =
                        GetFilteredNeighborSuccessorId(this, right_neighbor, false, center_successors);
                    if (!right_successor_id.empty())
                    {
                        valid_neighbors.push_back({right_neighbor, right_successor_id,
                                                   ActualMissGoalNum(right_neighbor), false,
                                                   !GLOBAL_ROUTE_EDGE_ID.count(right_successor_id)});
                    }
                }

                // Sort neighbors by ActualMissGoalNum; smaller values are preferred.
                std::sort(valid_neighbors.begin(), valid_neighbors.end(),
                          [](const NeighborInfo &a, const NeighborInfo &b)
                          { return a.miss_goal_penalty < b.miss_goal_penalty; });

                // Add neighbor paths according to the sorted order.
                for (const auto &neighbor : valid_neighbors)
                {
                    auto neighbor_path =
                        ExtendPath(neighbor.neighbor_id, neighbor.successor_id, refline_edge_ids, refline_edge_names,
                                   lookback_points_size, lookahead_points_size, is_lane);

                    neighbor_path->not_on_route_ = neighbor.not_on_route;

                    if (valid_paths < 3)
                    {
                        // When valid_paths < 3, insert the neighbor into the main ref_paths array.
                        if (neighbor.is_left)
                        {
                            // Insert a left neighbor by shifting existing paths to the right.
                            if (ref_paths[0] != nullptr)
                            {
                                ref_paths[2] = ref_paths[1];
                                ref_paths[1] = ref_paths[0];
                            }
                            ref_paths[0] = neighbor_path;
                            if (curr_ref_path_idx < 2)
                            {
                                curr_ref_path_idx = 1;
                            }
                        }
                        else
                        {
                            // Insert a right neighbor at the last available slot.
                            ref_paths[valid_paths] = neighbor_path;
                        }
                        ++valid_paths;
                    }
                    else
                    {
                        // When valid_paths >= 3, push additional neighbors into extra_ref_paths.
                        // Mark the Path rank to distinguish left/right neighbors; 0: left, 4: right.
                        neighbor_path->SetRank(neighbor.is_left ? 0 : 4);
                        extra_ref_paths.push_back(neighbor_path);
                    }
                }
            };

            // Handle LANE_CONNECTOR edges.
            if (edge_name == "LANE_CONNECTOR")
            {
                auto predecessors = GetRoutePredecessorIdsById(curr_edge_id);
                // std::cout << "curr_edge_id: " << curr_edge_id << " predecessors.size(): " << predecessors.size() <<
                // std::endl;
                if (!predecessors.empty())
                {
                    auto        predecessor = predecessors[0];
                    const auto &successors = GetRouteSuccessorIdsById(predecessor);
                    // Pre-compute neighbor information for the predecessor.
                    std::string              pred_left = GetLeftNeighborIdById(predecessor);
                    std::string              pred_right = GetRightNeighborIdById(predecessor);
                    std::vector<std::string> pred_left_successors, pred_right_successors;
                    if (!pred_left.empty())
                    {
                        pred_left_successors = GetRouteSuccessorIdsById(pred_left);
                    }
                    if (!pred_right.empty())
                    {
                        pred_right_successors = GetRouteSuccessorIdsById(pred_right);
                    }

                    buildSortedPaths(predecessor, successors, curr_ref_path_idx, false, pred_left, pred_right,
                                     pred_left_successors, pred_right_successors);

                    // Check whether neighbor paths should be added.
                    int path_count = 0;
                    for (const auto &path : ref_paths)
                    {
                        if (path != nullptr)
                        {
                            path_count++;
                        }
                    }
                    addNeighborPaths(predecessor, curr_ref_path_idx, path_count, false, successors, pred_left,
                                     pred_right, pred_left_successors, pred_right_successors);
                }
                else
                {
                    if (curr_successors.empty())
                    {
                        ref_paths[0] = ExtendPath(curr_edge_id, "", refline_edge_ids, refline_edge_names,
                                                  lookback_points_size, lookahead_points_size, true);
                    }
                    else
                    {
                        ref_paths[0] =
                            ExtendPath(curr_edge_id, curr_successors[0], refline_edge_ids, refline_edge_names,
                                       lookback_points_size, lookahead_points_size, true);
                    }
                    curr_ref_path_idx = 0;
                }
            }
            else // Handle LANE edges.
            {
                // Compute the distance from the current position to the end of the lane.
                float dist_to_end = DistanceToEndOfEdge(x, y, curr_edge_id);
                is_close_to_junction = dist_to_end <= utils::LEAST_DIST2JUNCTION;

                // Get the unique successor on this lane, if any.
                std::string curr_next_successor_id = "";
                if (curr_successors.size() == 1)
                {
                    const auto &tmp_next_successor_ids = GetRouteSuccessorIdsById(successor_id);
                    if (tmp_next_successor_ids.size() > 0)
                    {
                        curr_next_successor_id = tmp_next_successor_ids[0];
                    }
                }

                // Handle LANE case.
                if (!is_close_to_junction)
                {
                    // Lane-change is allowed: collect left neighbor, current lane, and right neighbor using
                    // pre-computed values.

                    // Use pre-computed information to get successor IDs for left/right neighbors.
                    std::string left_successor_id =
                        GetFilteredNeighborSuccessorId(this, left_neighbor, true, curr_successors);
                    std::string right_successor_id =
                        GetFilteredNeighborSuccessorId(this, right_neighbor, false, curr_successors);

                    // Get the extended path for the left neighbor.
                    if (!left_successor_id.empty()) // && GLOBAL_ROUTE_EDGE_ID.count(left_neighbor))
                    {
                        std::shared_ptr<Path> left_path =
                            ExtendPath(left_neighbor, left_successor_id, refline_edge_ids, refline_edge_names,
                                       lookback_points_size, lookahead_points_size, true);
                        if (!GLOBAL_ROUTE_EDGE_ID.count(left_successor_id))
                            left_path->not_on_route_ = true;
                        ref_paths[0] = left_path;
                    }

                    // Get the extended path for the right neighbor.
                    if (!right_successor_id.empty()) // && GLOBAL_ROUTE_EDGE_ID.count(right_neighbor))
                    {
                        std::shared_ptr<Path> right_path =
                            ExtendPath(right_neighbor, right_successor_id, refline_edge_ids, refline_edge_names,
                                       lookback_points_size, lookahead_points_size, true);
                        if (!GLOBAL_ROUTE_EDGE_ID.count(right_successor_id))
                            right_path->not_on_route_ = true;
                        ref_paths[2] = right_path;
                    }

                    // Get the extended path for the current lane, using filtering logic to avoid merging into neighbor
                    // lanes.
                    std::string filtered_successor_id =
                        GetFilteredCurrentSuccessorId(this, curr_successors, left_successor_id, right_successor_id);
                    if (!filtered_successor_id.empty())
                    {
                        successor_id = filtered_successor_id;
                    }

                    std::shared_ptr<Path> curr_path =
                        ExtendPath(curr_edge_id, successor_id, refline_edge_ids, refline_edge_names,
                                   lookback_points_size, lookahead_points_size, true);
                    if (!GLOBAL_ROUTE_EDGE_ID.count(successor_id))
                        curr_path->not_on_route_ = true;
                    ref_paths[1] = curr_path;

                    curr_ref_path_idx = 1;
                }
                else
                {
                    // Close to junction: prioritize successors.
                    if (curr_successors.empty())
                    {
                        ref_paths[0] = ExtendPath(curr_edge_id, "", refline_edge_ids, refline_edge_names,
                                                  lookback_points_size, lookahead_points_size, true);
                        curr_ref_path_idx = 0;
                        return;
                    }

                    buildSortedPaths(curr_edge_id, curr_successors, curr_ref_path_idx, true, left_neighbor,
                                     right_neighbor, left_successors, right_successors);

                    // Check whether neighbor paths should be added.
                    int path_count = 0;
                    for (const auto &path : ref_paths)
                    {
                        if (path != nullptr)
                        {
                            path_count++;
                        }
                    }

                    addNeighborPaths(curr_edge_id, curr_ref_path_idx, path_count, true, curr_successors, left_neighbor,
                                     right_neighbor, left_successors, right_successors);
                }
            }

            // ===== Filter out paths that contain loops =====
            // Any path with a loop is set to nullptr and curr_ref_path_idx is updated accordingly.
            int valid_count = 0;
            for (int i = 0; i < ref_paths.size(); ++i)
            {
                if (ref_paths[i])
                    ++valid_count;
            }
            if (valid_count > 1)
            {
                for (int i = 0; i < ref_paths.size(); ++i)
                {
                    if (ref_paths[i] && ref_paths[i]->loop_flag_)
                    {
                        ref_paths[i] = nullptr;
                        // If curr_ref_path_idx points to a filtered path, move it to the next valid path.
                        if (curr_ref_path_idx == i)
                        {
                            // Prefer valid paths on the right side first.
                            for (int j = i + 1; j < ref_paths.size(); ++j)
                            {
                                if (ref_paths[j] != nullptr)
                                {
                                    curr_ref_path_idx = j;
                                    break;
                                }
                            }
                            // If no valid path exists on the right, search on the left.
                            if (curr_ref_path_idx == i)
                            {
                                for (int j = i - 1; j >= 0; --j)
                                {
                                    if (ref_paths[j] != nullptr)
                                    {
                                        curr_ref_path_idx = j;
                                        break;
                                    }
                                }
                            }
                            // If no valid path exists at all, fall back to index 0.
                            if (curr_ref_path_idx == i)
                            {
                                curr_ref_path_idx = 0;
                            }
                        }
                    }
                }
            }

            // ===== Unified rank assignment logic =====
            // 1. For the paths that share successors of the same edge, their rank should be the same, taking the
            // minimum among them.
            // 2. Then re-normalize the remaining ranks: the left side ranks are kept,
            //    and right side ranks are shifted down by the same amount.
            //    Example: 0, 1, 2, 3, 4 where 1, 2, 3 belong to successors of the same path
            //    becomes 0, 1, 1, 1, 2.
            int         current_rank = 1;
            std::string prev_edge_id;
            bool        first_valid = true;

            for (int i = 0; i < ref_paths.size(); ++i)
            {
                if (!ref_paths[i])
                    continue; // Skip null pointers.

                if (ref_paths[i]->goal_frenet_s_ > 0.0f)
                    utils::approaching_terminal_point = true;

                std::string edge_id = ref_paths[i]->GetLaneId();

                if (first_valid)
                {
                    // First valid path.
                    current_rank = 1;
                    prev_edge_id = edge_id;
                    first_valid = false;
                }
                else
                {
                    // If the current edge differs from the previous edge, increase the rank;
                    // otherwise reuse the previous rank.
                    if (edge_id != prev_edge_id)
                    {
                        ++current_rank;
                        prev_edge_id = edge_id;
                    }
                }

                ref_paths[i]->SetRank(current_rank);
            }

            for (int i = 0; i < extra_ref_paths.size(); ++i)
            {
                if (extra_ref_paths[i] == nullptr)
                    continue;

                // If this is a right-side neighbor.
                if (extra_ref_paths[i]->GetRank() == 4)
                {
                    extra_ref_paths[i]->SetRank(current_rank + 1);
                }
            }

            // if one of the paths is terminated, then remaining frenet_s should be assigned with large values
            if (utils::approaching_terminal_point)
            {
                for (int i = 0; i < ref_paths.size(); ++i)
                {
                    if (!ref_paths[i])
                        continue; // Skip null pointers.

                    if (ref_paths[i]->goal_frenet_s_ == 0.0f)
                        ref_paths[i]->goal_frenet_s_ = 10000.0f;
                }
            }

            return;
        }

        void MapUtils::UpdateEgoRefPaths(std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                         std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths,
                                         int &curr_ref_path_idx, std::vector<std::string> &refline_edge_ids,
                                         std::vector<std::string> &refline_edge_names, const std::string &curr_edge_id,
                                         const std::string &edge_name)
        {
            if (curr_ref_path_idx == 1)
                return;

            // get predecessor
            std::string predecessor_id = ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_[0];

            // get neighbor
            bool        find_left = (curr_ref_path_idx == 0);
            std::string neighbor_id =
                find_left ? GetLeftNeighborIdById(predecessor_id) : GetRightNeighborIdById(predecessor_id);

            if (neighbor_id == "")
                return;

            // check if the neighbor refline is in ego_extra_ref_paths
            for (int i = 0; i < ego_extra_ref_paths.size(); ++i)
            {
                if (ego_extra_ref_paths[i]->GetLaneId() == neighbor_id)
                {
                    if (find_left)
                    {
                        ego_ref_paths[2] = ego_ref_paths[1];
                        ego_ref_paths[1] = ego_ref_paths[0];
                        ego_ref_paths[0] = ego_extra_ref_paths[i];
                    }
                    else
                    {
                        ego_ref_paths[0] = ego_ref_paths[1];
                        ego_ref_paths[1] = ego_ref_paths[2];
                        ego_ref_paths[2] = ego_extra_ref_paths[i];
                    }
                    ego_extra_ref_paths.erase(ego_extra_ref_paths.begin() + i);
                    curr_ref_path_idx = 1;
                    return;
                }
            }

            // if not, then we need to refind a new neighbor
            std::vector<std::string> neighbor_successors = GetRouteSuccessorIdsById(neighbor_id);
            if (neighbor_successors.empty())
                return;

            std::string curr_next_successor_id = ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_.size() > 2
                                                     ? ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_[2]
                                                     : "";
            std::string neighbor_next_edge_id = find_left ? GetLeftNeighborIdById(curr_next_successor_id)
                                                          : GetRightNeighborIdById(curr_next_successor_id);

            std::string neighbor_connector_id = "";
            for (const auto &successor : neighbor_successors)
            {
                const auto &tmp_successors = GetRouteSuccessorIdsById(successor);
                if (!tmp_successors.empty() && tmp_successors[0] == neighbor_next_edge_id)
                {
                    neighbor_connector_id = successor;
                    break;
                }
            }

            if (neighbor_connector_id == "")
            {
                neighbor_connector_id = neighbor_successors[0];
            }

            size_t lookback_points_size =
                static_cast<size_t>(utils::PATH_LOOKBACK_DISTANCE / utils::PATH_POINT_INTERVAL);
            if (find_left)
            {
                ego_ref_paths[2] = ego_ref_paths[1];
                ego_ref_paths[1] = ego_ref_paths[0];
                ego_ref_paths[0] = ExtendPath(neighbor_id, neighbor_connector_id, refline_edge_ids, refline_edge_names,
                                              lookback_points_size, utils::PATH_SIZE - lookback_points_size, false);
            }
            else
            {
                ego_ref_paths[0] = ego_ref_paths[1];
                ego_ref_paths[1] = ego_ref_paths[2];
                ego_ref_paths[2] = ExtendPath(neighbor_id, neighbor_connector_id, refline_edge_ids, refline_edge_names,
                                              lookback_points_size, utils::PATH_SIZE - lookback_points_size, false);
            }
            curr_ref_path_idx = 1;

            // ===== Re-assign ranks for ego reference paths using the unified rank logic =====
            int         current_rank = 1;
            std::string prev_edge_id;
            bool        first_valid = true;

            for (int i = 0; i < ego_ref_paths.size(); ++i)
            {
                if (!ego_ref_paths[i])
                    continue; // Skip null pointers.

                if (ego_ref_paths[i]->goal_frenet_s_ > 0.0f)
                    utils::approaching_terminal_point = true;

                std::string edge_id = ego_ref_paths[i]->GetLaneId();

                if (first_valid)
                {
                    // First valid path.
                    current_rank = 1;
                    prev_edge_id = edge_id;
                    first_valid = false;
                }
                else
                {
                    // If the current edge differs from the previous edge, increase the rank;
                    // otherwise reuse the previous rank.
                    if (edge_id != prev_edge_id)
                    {
                        ++current_rank;
                        prev_edge_id = edge_id;
                    }
                }

                ego_ref_paths[i]->SetRank(current_rank);
            }

            // If one of the paths reaches a terminal point, assign a large frenet_s to the remaining paths.
            if (utils::approaching_terminal_point)
            {
                for (int i = 0; i < ego_ref_paths.size(); ++i)
                {
                    if (!ego_ref_paths[i])
                        continue; // Skip null pointers.

                    if (ego_ref_paths[i]->goal_frenet_s_ == 0.0f)
                        ego_ref_paths[i]->goal_frenet_s_ = utils::TERMINAL_POINT_LARGE_FRENET_S;
                }
            }
        }

        void MapUtils::UpdateEgoRefPathsTrafficInfo(const std::vector<std::shared_ptr<Path>> &ego_ref_paths)
        {
            for (size_t i = 0; i < ego_ref_paths.size(); ++i)
            {
                if (ego_ref_paths[i] == nullptr)
                    continue;

                ego_ref_paths[i]->red_light_point_s_ = utils::MAX_VALUE;

                auto &comprised_ref_path_ids = ego_ref_paths[i]->comprised_ref_path_ids_;
                auto &comprised_ref_path_idxs = ego_ref_paths[i]->comprised_ref_path_idxs_;
                for (size_t j = 0; j < comprised_ref_path_ids.size(); ++j)
                {
                    if (HasRedTrafficLight(comprised_ref_path_ids[j]))
                    {
                        ego_ref_paths[i]->red_light_point_s_ = comprised_ref_path_idxs[j] * utils::PATH_POINT_INTERVAL;
                        break;
                    }
                }
            }
        }
    } // namespace utils
} // namespace vec_qmdp