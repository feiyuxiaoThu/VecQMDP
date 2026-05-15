/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file state.hpp
 * @brief Physical attributes and spatial relationships of ego and exo agents.
 */

#pragma once

#include <boost/python.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <collision/STRtree.hpp>
#include <utils/aligned_allocator.hpp>
#include <utils/global_utils.hpp>
#include <utils/map_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>
#include <vamp/vector.hh>
#include <vector>

namespace vec_qmdp
{
    namespace core
    {
        namespace py = boost::python;

        constexpr std::size_t FloatVectorWidth = vamp::FloatVectorWidth;
        // using AABB = collision::STRtree::AABB;
        using Point = collision::STRtree::Point;
        using Path = utils::Path;
        using FVectorT_1 = utils::FVectorT_1;
        using IVectorT_1 = utils::IVectorT_1;
        using AlignedVectorFloat = utils::AlignedVectorFloat;
        using AlignedVectorInt = utils::AlignedVectorInt;

        class EgoState
        {
          public:
            EgoState();
            ~EgoState();

            inline float       x() const { return x_; }
            inline float       y() const { return y_; }
            inline float       v() const { return v_; }
            inline float       a() const { return a_; }
            inline float       theta() const { return theta_; }
            inline float       steering_angle() const { return steering_angle_; }
            inline float       steering_rate() const { return steering_rate_; }
            inline std::string edge_token() const { return edge_token_; }
            inline std::string edge_name() const { return edge_name_; }

            void UpdateEgoState(const py::object &ego_info);

            bool IsCloseToJunction(const std::shared_ptr<utils::MapUtils> &map_utils) const;

          public:
            uint64_t    timestamp_;
            float       x_, y_, v_, a_, theta_, steering_angle_, steering_rate_;
            std::string edge_token_;
            std::string edge_name_;

          private:
            friend class boost::serialization::access;

            template <class Archive> void serialize(Archive &ar, const unsigned int version)
            {
                ar & timestamp_;
                ar & x_;
                ar & y_;
                ar & v_;
                ar & a_;
                ar & theta_;
                ar & steering_angle_;
                ar & steering_rate_;
                ar & edge_token_;
                ar & edge_name_;
            }
        };

        class ObservedExoState
        {
          public:
            ObservedExoState();
            ~ObservedExoState();

            void reset();

            void updateObservedExoState(const py::dict                           &agents_info,
                                        const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                        const std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths);

            void updateValidExoState();

            void GetFrenetPointsBatch(const std::shared_ptr<Path> &path, std::size_t path_idx);

          public:
            size_t                          num_vehicles_;       // actual number of observed vehicles
            std::vector<bool>               agent_vehicle_mask_; // size: max_vehicle_size, true: vehicle, false: others
            std::vector<bool>               agent_ped_mask_; // size: max_vehicle_size, true: pedestrian, false: others
            AlignedVectorFloat              obs_xs_;         // other agents' x coordinates [v] -> flat array
            AlignedVectorFloat              obs_ys_;         // other agents' y coordinates [v] -> flat array
            AlignedVectorFloat              obs_vs_;         // other agents' velocities [v] -> flat array
            AlignedVectorFloat              obs_thetas_;     // other agents' headings [v] -> flat array
            AlignedVectorFloat              obs_original_bb_extent_xs_; // agent AABB half-widths [v] -> flat array
            AlignedVectorFloat              obs_original_bb_extent_ys_; // agent AABB half-lengths [v] -> flat array
            AlignedVectorFloat              obs_expanded_bb_extent_xs_; // expanded AABB half-widths [v] -> flat array
            AlignedVectorFloat              obs_expanded_bb_extent_ys_; // expanded AABB half-lengths [v] -> flat array
            std::vector<AlignedVectorInt>   obs_nearest_idxs_;          // [path_idx][vehicle_idx]
            std::vector<AlignedVectorFloat> obs_frenet_s_;              // [path_idx][vehicle_idx]
            std::vector<AlignedVectorFloat> obs_frenet_l_;              // [path_idx][vehicle_idx]
            std::vector<bool>               agent_valid_mask_; // size: max_vehicle_size, true: valid, false: discarded
            std::vector<bool>               agent_dynamic_mask_; // size: max_vehicle_size, true: dynamic, false: static

            std::vector<size_t> valid_agent_idxs_;
            AlignedVectorFloat  valid_obs_original_bb_extent_xs_;
            AlignedVectorFloat  valid_obs_original_bb_extent_ys_;
            AlignedVectorFloat  valid_obs_expanded_bb_extent_xs_;
            AlignedVectorFloat  valid_obs_expanded_bb_extent_ys_;

            std::vector<std::string> collided_agents_;

            // for debug
            std::vector<std::string> tokens_;
            std::vector<int>         agent_types_;

          private:
            friend class boost::serialization::access;

            template <class Archive> void serialize(Archive &ar, const unsigned int version)
            {
                ar & num_vehicles_;
                ar & agent_vehicle_mask_;
                ar & obs_xs_;
                ar & obs_ys_;
                ar & obs_vs_;
                ar & obs_thetas_;
                ar & obs_original_bb_extent_xs_;
                ar & obs_original_bb_extent_ys_;
                ar & obs_nearest_idxs_;
                ar & obs_frenet_s_;
                ar & obs_frenet_l_;
                ar & agent_valid_mask_;
                ar & valid_agent_idxs_;
                ar & valid_obs_original_bb_extent_xs_;
                ar & valid_obs_original_bb_extent_ys_;
                ar & collided_agents_;
                ar & tokens_;
            }
        };

        class ExoStates
        {
          public:
            using STRtree = collision::STRtree;

            ExoStates();
            ~ExoStates() = default;

            // Batch For QMDP Sampling
            static void GetFrenetPointsBatch(const std::vector<std::shared_ptr<Path>> &paths, int curr_path_idx,
                                             std::vector<AlignedVectorInt> &nearest_idxs,
                                             AlignedVectorFloat &exo_thetas_flat_, AlignedVectorFloat &exo_xs_flat_,
                                             AlignedVectorFloat &exo_ys_flat_, AlignedVectorFloat &exo_vs_flat_,
                                             std::vector<AlignedVectorFloat> &exo_ss_flat_,
                                             std::vector<AlignedVectorFloat> &exo_ls_flat_, const size_t &offset,
                                             const size_t &max_num_vehicles, const size_t &valid_time_size,
                                             const size_t &global_time_size, const size_t &valid_num)
            {

                // Allocate local buffers to store transposed batch data
                // Size: Time(48) * Batch(8) * sizeof(float) = ~1.5KB (fits in L1)
                // Layout: [Time][Batch] -> t0[v0..v7], t1[v0..v7]...
                alignas(32) float local_xs[utils::MAX_PRED_TIME_STEPS * 8];
                alignas(32) float local_ys[utils::MAX_PRED_TIME_STEPS * 8];
                alignas(32) float local_vs[utils::MAX_PRED_TIME_STEPS * 8]; // For velocities if needed (V)
                alignas(32) float local_thetas[utils::MAX_PRED_TIME_STEPS * 8];

                // Iterate over all paths
                for (size_t p = 0; p < paths.size(); ++p)
                {
                    const std::shared_ptr<Path> &path = paths[p];
                    if (path == nullptr)
                        continue;

                    bool is_curr_path = (p == curr_path_idx);

                    // Iterate over vehicle batches
                    for (size_t v_batch = 0; v_batch < valid_num; v_batch += FloatVectorWidth)
                    {
                        int remaining = static_cast<int>(valid_num - v_batch);

                        // Iterate up to 8 vehicles (or remaining)
                        for (int i = 0; i < std::min((int)FloatVectorWidth, remaining); ++i)
                        {
                            int global_v_idx = (v_batch + i) * global_time_size + offset;

                            // Original data layout: [Vehicle][Time]
                            for (int t = 0; t < valid_time_size; ++t)
                            {
                                // Transpose and store to local: [Time][Batch]
                                // local_idx = t * 8 + i
                                local_xs[t * 8 + i] = exo_xs_flat_[global_v_idx + t];
                                local_ys[t * 8 + i] = exo_ys_flat_[global_v_idx + t];
                                // Optionally transpose theta/v here if needed
                                if (t == 0)
                                {                                                     // t=0 used to compute step_dir
                                    local_thetas[i] = exo_thetas_flat_[global_v_idx]; // Note: confirm whether theta is
                                                                                      // per-time or scalar
                                }
                                if (is_curr_path)
                                {
                                    // theta/v logic for current path
                                    local_thetas[t * 8 + i] = exo_thetas_flat_[global_v_idx + t];
                                    local_vs[t * 8 + i] = exo_vs_flat_[global_v_idx + t];
                                }
                            }
                        }

                        // Create mask to handle tails when remaining < FloatVectorWidth
                        // For a full batch the mask is all-ones; compiler may optimize masked store to a plain store
                        IVectorT_1 batch_mask = IVectorT_1::create_tail_mask(remaining);

                        // Base indices
                        IVectorT_1 curr_v_vec = IVectorT_1::iota(v_batch) * global_time_size + offset;

                        // Load initial nearest indices (Aligned Load)
                        // nearest_idxs[p] is stored contiguously and can be loaded directly
                        IVectorT_1 prev_idxs = IVectorT_1::load_contiguous(nearest_idxs[p].data(), v_batch);

                        // Initial step direction computation (done at t=0 or updated each frame)
                        // For performance, compute at t=0 and reuse or apply a light update
                        FVectorT_1 agent_th = FVectorT_1::load_contiguous(local_thetas, 0);
                        FVectorT_1 path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);
                        FVectorT_1 th_diff = utils::NormalizeAngleSIMD<FVectorT_1>(agent_th - path_th).abs();
                        IVectorT_1 step_dir =
                            IVectorT_1::select((th_diff < utils::PI_1_2).template as<IVectorT_1>(), 1, -1);

                        // Time Loop
                        for (size_t t = 0; t < valid_time_size; ++t, curr_v_vec = curr_v_vec + 1)
                        {
                            FVectorT_1 cur_xs = FVectorT_1::load_contiguous(local_xs, t * 8);
                            FVectorT_1 cur_ys = FVectorT_1::load_contiguous(local_ys, t * 8);

                            if (t > 0)
                            {
                                // Call the path's NearestBatch routine
                                path->NearestBatch(cur_xs, cur_ys, prev_idxs, step_dir, batch_mask);

                                // Update path theta (used for subsequent Frenet and velocity checks)
                                path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);

                                // Update step_dir (optional; update each frame if path curvature is large)
                                if (is_curr_path)
                                {
                                    // Only the current path requires precise angle difference for velocity reversal
                                    agent_th = FVectorT_1::load_contiguous(local_thetas, t * 8);
                                    th_diff = utils::NormalizeAngleSIMD<FVectorT_1>(agent_th - path_th).abs();
                                }
                            }

                            // === Frenet computation (fully vectorized) ===
                            FVectorT_1 p_xs = FVectorT_1::gather(path->xs_.data(), prev_idxs);
                            FVectorT_1 p_ys = FVectorT_1::gather(path->ys_.data(), prev_idxs);

                            FVectorT_1 dx = cur_xs - p_xs;
                            FVectorT_1 dy = cur_ys - p_ys;

                            // nx = -sin, ny = cos
                            FVectorT_1 nx = -path_th.sin();
                            FVectorT_1 ny = path_th.cos();

                            FVectorT_1 l = (dx * nx + dy * ny).sign() * (dx * dx + dy * dy).sqrt();

                            // Assume GetFrenetS internally gathers s distances
                            // FVectorT_1 s = path->GetFrenetSBatch(prev_idxs);
                            // Original GetFrenetS might be scalar; use a gather-friendly variant
                            FVectorT_1 s = path->GetFrenetS(prev_idxs);

                            // === Store results (vectorized) ===
                            // Compute storage offset: idx = t * max_num_vehicles + v_batch
                            size_t store_offset = t * max_num_vehicles + v_batch;

                            // Masked store (handle boundary tails)
                            // store nearest_idxs
                            prev_idxs.to_array(nearest_idxs[p].data() + store_offset);
                            // store lateral offset l
                            l.to_array(exo_ls_flat_[p].data() + store_offset + offset);
                            // store longitudinal position s
                            s.to_array(exo_ss_flat_[p].data() + store_offset + offset);

                            // === Only adjust speed for the current path (vectorized) ===
                            if (is_curr_path)
                            {
                                // Logic: if diff > 2/3 PI -> v = -v; else if diff > 1/3 PI -> v = 0
                                FVectorT_1 old_v = FVectorT_1::load_contiguous(local_vs, t * 8);

                                // Construct masks
                                IVectorT_1 reverse_mask = (th_diff > utils::PI_2_3).template as<IVectorT_1>();
                                IVectorT_1 stop_mask = (th_diff > utils::PI_1_3).template as<IVectorT_1>();

                                // Select new velocity
                                FVectorT_1 new_v =
                                    FVectorT_1::select(reverse_mask.template as<FVectorT_1>(), -old_v, old_v);

                                IVectorT_1 real_stop_mask = stop_mask & (~reverse_mask);
                                new_v = FVectorT_1::select(real_stop_mask.template as<FVectorT_1>(), 0.0f, new_v);

                                // Scatter store back (curr_v_vec may be strided).
                                // If global_time_size == 1 it's contiguous and mask_store can be used;
                                // otherwise use scatter.
                                new_v.scatter(exo_vs_flat_.data(), curr_v_vec);
                            }
                        }
                    }
                }
            }

            // Batch For Importance Sampling (Generating Trajectory Proposal)
            static void GetFrenetPointsBatch(int thread_id, const std::shared_ptr<Path> &path,
                                             AlignedVectorInt &nearest_idxs_, AlignedVectorFloat &exo_thetas_flat_,
                                             AlignedVectorFloat &exo_xs_flat_, AlignedVectorFloat &exo_ys_flat_,
                                             AlignedVectorFloat &exo_vs_flat_, AlignedVectorFloat &exo_ss_flat_,
                                             AlignedVectorFloat &exo_ls_flat_, const size_t &offset,
                                             const size_t &max_num_vehicles, const size_t &valid_time_size,
                                             const size_t &global_time_size, const size_t &valid_num)
            {
                if (path == nullptr)
                    return;

                // Iterate over each vehicle
                for (size_t v_batch = 0; v_batch < valid_num; v_batch += FloatVectorWidth)
                {
                    int remaining = static_cast<int>(valid_num - v_batch);

                    IVectorT_1 batch_mask = IVectorT_1::create_tail_mask(remaining);

                    // Get current vehicles' indices
                    IVectorT_1 curr_v_vec = IVectorT_1::iota(v_batch) * global_time_size + offset;

                    // Initial nearest point indices were computed during sampling
                    IVectorT_1 prev_idxs = IVectorT_1::load_contiguous(nearest_idxs_.data(), v_batch);

                    FVectorT_1 agent_th = FVectorT_1::gather(exo_thetas_flat_.data(), curr_v_vec);
                    FVectorT_1 path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);

                    // Compute angle difference and normalize to [-π, π]
                    FVectorT_1 th_diff = utils::NormalizeAngleSIMD<FVectorT_1>(agent_th - path_th).abs();

                    // Determine search direction based on angle difference
                    IVectorT_1 step_dir =
                        IVectorT_1::select((th_diff < utils::PI_1_2).template as<IVectorT_1>(), 1, -1);

                    for (size_t t = 0; t < valid_time_size; ++t, curr_v_vec = curr_v_vec + 1)
                    {
                        // Fetch vehicle indices for the current time step
                        FVectorT_1 cur_xs = FVectorT_1::gather(exo_xs_flat_.data(), curr_v_vec);
                        FVectorT_1 cur_ys = FVectorT_1::gather(exo_ys_flat_.data(), curr_v_vec);

                        // Batch nearest-point lookup
                        if (t > 0)
                        {
                            path->NearestBatch(cur_xs, cur_ys, prev_idxs, step_dir, batch_mask);
                            path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);

                            agent_th = FVectorT_1::gather(exo_thetas_flat_.data(), curr_v_vec);
                            th_diff = utils::NormalizeAngleSIMD<FVectorT_1>(agent_th - path_th)
                                          .abs(); // Compute angle difference and normalize to [-π, π]
                        }

                        // Store Frenet coordinates for the current batch
                        FVectorT_1 p_xs = FVectorT_1::gather(path->xs_.data(), prev_idxs);
                        FVectorT_1 p_ys = FVectorT_1::gather(path->ys_.data(), prev_idxs);

                        // Compute vector from vehicle position to path point
                        FVectorT_1 dx = cur_xs - p_xs;
                        FVectorT_1 dy = cur_ys - p_ys;

                        // Compute normal vector for the path point
                        FVectorT_1 nx = -path_th.sin(); // cos(theta + pi/2)
                        FVectorT_1 ny = path_th.cos();  // sin(theta + pi/2)

                        // Compute lateral offset (l) and longitudinal position (s)
                        FVectorT_1 l = (dx * nx + dy * ny).sign() * (dx * dx + dy * dy).sqrt();
                        FVectorT_1 s = path->GetFrenetS(prev_idxs);

                        // === Result storage (fully vectorized store) ===
                        // Compute storage offset: idx = t * max_num_vehicles + v_batch
                        size_t store_offset = t * max_num_vehicles + v_batch;

                        // Masked Store (handle boundary cases)
                        // Store nearest_idxs
                        prev_idxs.to_array(nearest_idxs_.data() + store_offset);
                        // Store l
                        l.to_array(exo_ls_flat_.data() + store_offset + offset);
                        // Store s
                        s.to_array(exo_ss_flat_.data() + store_offset + offset);

                        // Logic: if diff > 2/3 PI -> v = -v; else if diff > 1/3 PI -> v = 0
                        FVectorT_1 old_v = FVectorT_1::gather(exo_vs_flat_.data(), curr_v_vec);

                        // Construct masks
                        IVectorT_1 reverse_mask = (th_diff > utils::PI_2_3).template as<IVectorT_1>();
                        IVectorT_1 stop_mask = (th_diff > utils::PI_1_3).template as<IVectorT_1>();
                        IVectorT_1 speed_gt_0_5_mask =
                            (old_v > utils::VELOCITY_ALMOST_ZERO_THRESHOLD).template as<IVectorT_1>();

                        // Select new velocity
                        FVectorT_1 new_v = FVectorT_1::select(reverse_mask.template as<FVectorT_1>(), -old_v, old_v);
                        // Note: stop has lower priority than reverse (original logic: if ... else if)
                        // Therefore process stop after reverse (exclude reverse from stop mask)
                        // Original: if (>2/3) ... else if (>1/3 && v > 0.05f) ...
                        // So stop_mask should actually be (>1/3) & (<= 2/3) & (v > 0.05f)
                        IVectorT_1 real_stop_mask = stop_mask & speed_gt_0_5_mask & (~reverse_mask);
                        new_v = FVectorT_1::select(real_stop_mask.template as<FVectorT_1>(),
                                                   utils::EXO_FRENET_STOPPED_VELOCITY, new_v);

                        // Scatter write-back (because curr_v_vec is strided)
                        // If global_time_size == 1 it's contiguous and mask_store can be used;
                        // otherwise use scatter.
                        new_v.scatter(exo_vs_flat_.data(), curr_v_vec);
                    }
                }
            }

            static void GetFrenetPointsBatch(const std::shared_ptr<Path> &path, AlignedVectorInt &nearest_idxs_,
                                             AlignedVectorFloat &exo_thetas_flat_, AlignedVectorFloat &exo_xs_flat_,
                                             AlignedVectorFloat &exo_ys_flat_, AlignedVectorFloat &exo_ss_flat_,
                                             AlignedVectorFloat &exo_ls_flat_, const size_t &offset,
                                             const size_t &max_num_vehicles, const size_t &valid_time_size,
                                             const size_t &global_time_size, const size_t &valid_num)
            {
                if (path == nullptr)
                    return;

                // Iterate over each vehicle
                for (size_t v_batch = 0; v_batch < valid_num; v_batch += FloatVectorWidth)
                {
                    int        remaining = static_cast<int>(valid_num - v_batch);
                    IVectorT_1 batch_mask = IVectorT_1::create_tail_mask(remaining);

                    // Get current vehicles' indices
                    IVectorT_1 curr_v_vec = IVectorT_1::iota(v_batch) * global_time_size + offset;

                    // Initial nearest point indices were computed during sampling
                    IVectorT_1 prev_idxs = IVectorT_1::load_contiguous(nearest_idxs_.data(), v_batch);

                    FVectorT_1 agent_th = FVectorT_1::gather(exo_thetas_flat_.data(), curr_v_vec);
                    FVectorT_1 path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);

                    // Compute angle difference and normalize to [-π, π]
                    FVectorT_1 th_diff = utils::NormalizeAngleSIMD<FVectorT_1>(agent_th - path_th).abs();

                    // Determine search direction based on angle difference
                    IVectorT_1 step_dir =
                        IVectorT_1::select((th_diff < utils::PI_1_2).template as<IVectorT_1>(), 1, -1);

                    for (size_t t = 0; t < valid_time_size; ++t, curr_v_vec = curr_v_vec + 1)
                    {
                        // Fetch vehicle indices for the current time step
                        FVectorT_1 cur_xs = FVectorT_1::gather(exo_xs_flat_.data(), curr_v_vec);
                        FVectorT_1 cur_ys = FVectorT_1::gather(exo_ys_flat_.data(), curr_v_vec);

                        // Batch nearest-point lookup
                        if (t > 0)
                        {
                            path->NearestBatch(cur_xs, cur_ys, prev_idxs, step_dir, batch_mask);
                            path_th = FVectorT_1::gather(path->thetas_.data(), prev_idxs);
                        }

                        // Store Frenet coordinates for the current batch
                        // Load path point coordinates and headings
                        FVectorT_1 p_xs = FVectorT_1::gather(path->xs_.data(), prev_idxs);
                        FVectorT_1 p_ys = FVectorT_1::gather(path->ys_.data(), prev_idxs);

                        // Compute vector from vehicle position to path point
                        FVectorT_1 dx = cur_xs - p_xs;
                        FVectorT_1 dy = cur_ys - p_ys;

                        // Compute normal vector for the path point
                        FVectorT_1 nx = -path_th.sin(); // cos(theta + pi/2)
                        FVectorT_1 ny = path_th.cos();  // sin(theta + pi/2)

                        // Compute lateral offset (l) and longitudinal position (s)
                        FVectorT_1 l = (dx * nx + dy * ny).sign() * (dx * dx + dy * dy).sqrt();
                        FVectorT_1 s = path->GetFrenetS(prev_idxs);

                        // === Result storage (fully vectorized) ===
                        // Compute storage offset: idx = t * max_num_vehicles + v_batch
                        size_t store_offset = t * max_num_vehicles + v_batch;

                        // Masked store (handle boundary cases)
                        // store nearest_idxs
                        prev_idxs.to_array(nearest_idxs_.data() + store_offset);
                        // store l
                        l.to_array(exo_ls_flat_.data() + store_offset + offset);
                        // store s
                        s.to_array(exo_ss_flat_.data() + store_offset + offset);
                    }
                }
            }

            // Batch For QMDP Sampling (build STRtree Frenet coordinates for each time step)
            static void buildSTRtreesFrenetBatch(const std::vector<std::shared_ptr<Path>> &paths,
                                                 AlignedVectorFloat &exo_cos_thetas, AlignedVectorFloat &exo_sin_thetas,
                                                 AlignedVectorFloat                    &exo_bb_extent_xs,
                                                 AlignedVectorFloat                    &exo_bb_extent_ys,
                                                 const std::vector<AlignedVectorFloat> &exo_ss_flat_,
                                                 const std::vector<AlignedVectorFloat> &exo_ls_flat_,
                                                 std::vector<AlignedVectorFloat>       &exo_ls_projected_radius_flat_,
                                                 const std::vector<AlignedVectorInt>   &nearest_idxs_,
                                                 std::vector<std::vector<std::shared_ptr<STRtree>>> &strtrees,
                                                 const size_t &offset, const size_t &offset_strtree,
                                                 const size_t &max_num_vehicles, const size_t &valid_time_size,
                                                 const size_t &global_time_size, const size_t &valid_num)
            {
                int current_v_batch, batch_valid_num;

                IVectorT_1 nearest_idxs, v_batch_idxs, idxs;

                FVectorT_1 s, l, thetas_cos, thetas_sin, path_th, path_thetas_cos, path_thetas_sin, bb_extent_xs,
                    bb_extent_ys, proj_radius_ss, proj_radius_ls, min_ss, max_ss, min_ls, max_ls;

                std::shared_ptr<STRtree> tree;

                for (size_t p = 0; p < paths.size(); ++p)
                {
                    const std::shared_ptr<Path> &path = paths[p];
                    if (path == nullptr)
                        continue;

                    for (size_t t = 0; t < valid_time_size; ++t)
                    {
                        {
                            tree = strtrees[p][t + offset_strtree];
                            tree->clear();
                        }

                        current_v_batch = t * max_num_vehicles + offset;

                        {
                            for (size_t v_batch = 0; v_batch < valid_num;
                                 v_batch += FloatVectorWidth, current_v_batch += FloatVectorWidth)
                            {
                                batch_valid_num = static_cast<int>(std::min(FloatVectorWidth, valid_num - v_batch));

                                s = FVectorT_1::load_contiguous(exo_ss_flat_[p].data(), current_v_batch);
                                l = FVectorT_1::load_contiguous(exo_ls_flat_[p].data(), current_v_batch);
                                nearest_idxs =
                                    IVectorT_1::load_contiguous(nearest_idxs_[p].data(), current_v_batch - offset);

                                v_batch_idxs = IVectorT_1::iota(v_batch);
                                idxs = v_batch_idxs * global_time_size + t + offset;
                                thetas_cos = FVectorT_1::gather(exo_cos_thetas.data(), idxs);
                                thetas_sin = FVectorT_1::gather(exo_sin_thetas.data(), idxs);

                                path_th = FVectorT_1::gather(path->thetas_.data(), nearest_idxs);
                                path_thetas_cos = path_th.cos();
                                path_thetas_sin = path_th.sin();

                                bb_extent_xs = FVectorT_1::load_contiguous(exo_bb_extent_xs.data(), v_batch);
                                bb_extent_ys = FVectorT_1::load_contiguous(exo_bb_extent_ys.data(), v_batch);

                                proj_radius_ss =
                                    (thetas_cos * path_thetas_cos + thetas_sin * path_thetas_sin).abs() * bb_extent_ys +
                                    (-thetas_sin * path_thetas_cos + thetas_cos * path_thetas_sin).abs() * bb_extent_xs;
                                proj_radius_ls =
                                    (-thetas_cos * path_thetas_sin + thetas_sin * path_thetas_cos).abs() *
                                        bb_extent_ys +
                                    (thetas_sin * path_thetas_sin + thetas_cos * path_thetas_cos).abs() * bb_extent_xs;

                                min_ss = s - proj_radius_ss;
                                max_ss = s + proj_radius_ss;
                                min_ls = l - proj_radius_ls - utils::EXO_STRTREE_SAFETY_MARGIN;
                                max_ls = l + proj_radius_ls + utils::EXO_STRTREE_SAFETY_MARGIN;

                                proj_radius_ls.to_array(exo_ls_projected_radius_flat_[p].data() + current_v_batch);
                                tree->insertBatch(min_ss, min_ls, max_ss, max_ls, s, l, v_batch_idxs, batch_valid_num);
                            }
                        }

                        {
                            tree->build();
                        }
                    }
                }
            }

            // Batch For Importance Sampling (build STRtree Frenet coordinates for each time step)
            static void buildSTRtreesFrenetBatch(
                int thread_id, const std::shared_ptr<Path> &path, AlignedVectorFloat &exo_cos_thetas,
                AlignedVectorFloat &exo_sin_thetas, AlignedVectorFloat &exo_original_bb_extent_xs,
                AlignedVectorFloat &exo_original_bb_extent_ys, AlignedVectorFloat &exo_expanded_bb_extent_xs,
                AlignedVectorFloat &exo_expanded_bb_extent_ys, AlignedVectorFloat &exo_ss_flat_,
                AlignedVectorFloat &exo_ls_flat_, AlignedVectorFloat &exo_ls_projected_radius_flat_,
                AlignedVectorInt &nearest_idxs_,
                // std::vector<std::shared_ptr<STRtree>> &original_strtrees,
                std::vector<std::shared_ptr<STRtree>> &expanded_strtrees, const size_t &offset,
                const size_t &offset_strtree, const size_t &max_num_vehicles, const size_t &valid_time_size,
                const size_t &global_time_size, const size_t &valid_num, bool print = false)
            // const std::vector<std::string> &tokens)
            {

                if (path == nullptr)
                    return;

                int current_v_batch, batch_valid_num;

                IVectorT_1 nearest_idxs, v_batch_idxs, idxs;

                FVectorT_1 frenet_s, frenet_l, thetas_cos, thetas_sin, path_th, path_thetas_cos, path_thetas_sin,
                    original_bb_extent_xs, original_bb_extent_ys, expanded_bb_extent_xs, expanded_bb_extent_ys,
                    original_proj_radius_l, expanded_proj_radius_s, expanded_proj_radius_l, expanded_min_ss,
                    expanded_max_ss, expanded_min_ls, expanded_max_ls;

                std::shared_ptr<STRtree> expanded_tree;

                for (size_t t = 0; t < valid_time_size; ++t)
                {
                    {
                        // Optimization: reuse existing tree objects instead of creating new ones each time
                        expanded_tree = expanded_strtrees[t + offset_strtree];
                        expanded_tree->clear(); // clear but keep memory
                    }

                    // Create vehicle base index for this time step
                    current_v_batch = t * max_num_vehicles + offset;

                    {
                        // Process vehicles in batches
                        for (size_t v_batch = 0; v_batch < valid_num;
                             v_batch += FloatVectorWidth, current_v_batch += FloatVectorWidth)
                        {
                            // Determine number of valid vehicles in the current batch
                            batch_valid_num = static_cast<int>(std::min(FloatVectorWidth, valid_num - v_batch));

                            // Load vehicle Frenet coordinates and nearest point indices (time * vehicle layout)
                            frenet_s = FVectorT_1::load_contiguous(exo_ss_flat_.data(), current_v_batch);
                            frenet_l = FVectorT_1::load_contiguous(exo_ls_flat_.data(), current_v_batch);
                            nearest_idxs = IVectorT_1::load_contiguous(nearest_idxs_.data(), current_v_batch - offset);

                            // Load vehicle orientations (vehicle * time layout)
                            v_batch_idxs = IVectorT_1::iota(v_batch);
                            idxs = v_batch_idxs * global_time_size + t + offset;
                            thetas_cos = FVectorT_1::gather(exo_cos_thetas.data(), idxs);
                            thetas_sin = FVectorT_1::gather(exo_sin_thetas.data(), idxs);

                            // Load path orientations at nearest indices
                            path_th = FVectorT_1::gather(path->thetas_.data(), nearest_idxs);
                            path_thetas_cos = path_th.cos();
                            path_thetas_sin = path_th.sin();

                            // Load vehicle bounding box extents
                            original_bb_extent_xs =
                                FVectorT_1::load_contiguous(exo_original_bb_extent_xs.data(), v_batch);
                            original_bb_extent_ys =
                                FVectorT_1::load_contiguous(exo_original_bb_extent_ys.data(), v_batch);
                            expanded_bb_extent_xs =
                                FVectorT_1::load_contiguous(exo_expanded_bb_extent_xs.data(), v_batch);
                            expanded_bb_extent_ys =
                                FVectorT_1::load_contiguous(exo_expanded_bb_extent_ys.data(), v_batch);

                            original_proj_radius_l =
                                (-thetas_cos * path_thetas_sin + thetas_sin * path_thetas_cos).abs() *
                                    original_bb_extent_ys +
                                (thetas_sin * path_thetas_sin + thetas_cos * path_thetas_cos).abs() *
                                    original_bb_extent_xs;
                            expanded_proj_radius_s =
                                (thetas_cos * path_thetas_cos + thetas_sin * path_thetas_sin).abs() *
                                    expanded_bb_extent_ys +
                                (-thetas_sin * path_thetas_cos + thetas_cos * path_thetas_sin).abs() *
                                    expanded_bb_extent_xs;
                            expanded_proj_radius_l =
                                (-thetas_cos * path_thetas_sin + thetas_sin * path_thetas_cos).abs() *
                                    expanded_bb_extent_ys +
                                (thetas_sin * path_thetas_sin + thetas_cos * path_thetas_cos).abs() *
                                    expanded_bb_extent_xs;

                            expanded_min_ss = frenet_s - expanded_proj_radius_s;
                            expanded_max_ss = frenet_s + expanded_proj_radius_s;
                            expanded_min_ls = frenet_l - expanded_proj_radius_l - utils::EXO_STRTREE_SAFETY_MARGIN;
                            expanded_max_ls = frenet_l + expanded_proj_radius_l + utils::EXO_STRTREE_SAFETY_MARGIN;

                            original_proj_radius_l.to_array(exo_ls_projected_radius_flat_.data() + current_v_batch);
                            expanded_tree->insertBatch(expanded_min_ss, expanded_min_ls, expanded_max_ss,
                                                       expanded_max_ls, frenet_s, frenet_l, v_batch_idxs,
                                                       batch_valid_num);
                        }
                    }

                    {
                        expanded_tree->build();
                    }
                }
            }
        };

    } // namespace core
} // namespace vec_qmdp