/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file global_utils.hpp
 * @brief SIMD type aliases, thread pool, logging, performance profiling, and thread-local workspaces.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono> // include chrono for time calculations
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map> // include unordered_map for storing performance data
#include <unordered_set> // include unordered_set for storing auxiliary sets
#include <utils/aligned_allocator.hpp>
#include <utils/params.hpp>
#include <vamp/vector.hh>
#include <vector>

namespace vec_qmdp
{
    namespace utils
    {
        constexpr int FloatVectorWidth = vamp::FloatVectorWidth;
        using AlignedVectorFloat = std::vector<float, AlignedAllocator<float>>;
        using AlignedVectorInt = std::vector<int, AlignedAllocator<int>>;
        using AlignedVectorBool = std::vector<bool, AlignedAllocator<bool>>;

        using FVectorT_1 = vamp::FloatVector<FloatVectorWidth, 1>;
        using IVectorT_1 = vamp::IntVector<FloatVectorWidth, 1>;

        using FVectorT_12 = vamp::FloatVector<FloatVectorWidth, 12>;
        using IVectorT_12 = vamp::IntVector<FloatVectorWidth, 12>;

        // qmdp simd vector
        using FVectorT_qmdp = vamp::FloatVector<NUM_SCENARIOS_PER_THREAD, 1>;
        using IVectorT_qmdp = vamp::IntVector<NUM_SCENARIOS_PER_THREAD, 1>;

        // trajectory optimization simd vector
        using FVectorT_traj_1 = vamp::FloatVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, 1>;
        using IVectorT_traj_1 = vamp::IntVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, 1>;
        using FVectorT_traj = vamp::FloatVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, utils::LATERAL_OFFSETS_NUM>;
        using IVectorT_traj = vamp::IntVector<NUM_SCENARIOS_TRAJ_OPT_PER_THREAD, utils::LATERAL_OFFSETS_NUM>;

        // Thread-safe print function: accept a string and atomically print it to the console
        inline void PrintSafe(int thread_id, const std::string &msg)
        {
            // if (thread_id != 0)
            //     return;
            std::stringstream ss_final;

            // Automatically prepend a [Tn] prefix
            ss_final << "[T" << thread_id << "] " << msg;

            static std::mutex           cout_mutex;
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << ss_final.str() << std::flush;
        }

        template <typename VecT> void print_vector(const std::string &name, const VecT &vec, int thread_id = 0)
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4); // fixed format with 4 decimal places
            ss << name << " = [ ";

            auto arr = vec.to_array();
            for (size_t i = 0; i < VecT::num_scalars; ++i)
            {
                ss << std::setw(8) << arr[i] << ", ";
                if ((i + 1) % 8 == 0 && (i + 1) < VecT::num_scalars)
                {
                    ss << "\n      ";
                }
            }
            ss << "]" << std::endl;

            utils::PrintSafe(thread_id, ss.str());
        }

        // Workspaces for Thread-Local Storage
        struct StepBatchWorkspace
        {
            // Scalars / Fixed size vectors
            IVectorT_qmdp step_end_time_idxs, curr_path_is_left, curr_path_is_middle, curr_path_is_right;
            IVectorT_qmdp target_path_is_left, target_path_is_middle, target_path_is_right;
            IVectorT_qmdp left_path_mask, middle_path_mask, right_path_mask;
            IVectorT_qmdp left_path_nearest_idxs, middle_path_nearest_idxs, right_path_nearest_idxs;
            IVectorT_qmdp ego_curr_path_idxs, should_switch_path, ego_curr_path_nearest_idxs,
                ego_target_path_nearest_idxs;
            FVectorT_qmdp ego_curr_lateral_offsets;

            FVectorT_qmdp ego_cos_thetas, ego_sin_thetas;
            FVectorT_qmdp left_path_min_distance, middle_path_min_distance, right_path_min_distance,
                target_path_min_distance;
            FVectorT_qmdp ego_curr_path_ls, future_curvatures, ego_desired_vs, ego_curr_path_heading;

            FVectorT_qmdp curv_0, curv_1, curv_2;
            FVectorT_qmdp speed_0, speed_1, speed_2;

            FVectorT_qmdp f_curv_0, f_curv_1, f_curv_2;
            FVectorT_qmdp des_v_0, des_v_1, des_v_2;

            FVectorT_qmdp lateral_offsets, curr_path_thetas_sin, curr_path_thetas_cos, ego_proj_radius_ls;
            FVectorT_qmdp v_lateral, relative_distance, leading_exo_vs;
            FVectorT_qmdp ref_point_xs, ref_point_ys, ref_point_thetas;
            FVectorT_qmdp cal_accs, cal_steerings;
            FVectorT_qmdp curr_path_min_distance, points_on_path_left;
            FVectorT_qmdp ego_curr_path_ss;
            FVectorT_qmdp min_s_values, min_l_values, max_s_values, max_l_values;
            FVectorT_qmdp ego_rear_xs, ego_rear_ys;
            FVectorT_qmdp active_flags_float, collision_flags_float;
            FVectorT_qmdp collision_penalty, movement_penalty, miss_goal_rewards, action_penalties, rollout_step_reward;
            FVectorT_qmdp miss_goal_pen, goal_diff;

            IVectorT_qmdp consider_LC_mask, consider_left_LC_mask, consider_right_LC_mask;
            IVectorT_qmdp LC_allowance_flags, LF_lead_allowance_flags, collision_flags;

            // Dynamic size vectors (need resize)
            AlignedVectorFloat final_relative_distance_vec, final_exo_vs_vec, ref_point_xs_vec, ref_point_ys_vec,
                ref_point_thetas_vec, points_on_path_left_vec;
            AlignedVectorInt has_collision;

            // Pairs of vectors
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LF_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LC_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LC_following_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LF_lead_vehicles_future_path_intersected;

            StepBatchWorkspace(size_t size)
            {
                final_relative_distance_vec.resize(size);
                final_exo_vs_vec.resize(size);
                ref_point_xs_vec.resize(size);
                ref_point_ys_vec.resize(size);
                ref_point_thetas_vec.resize(size);
                points_on_path_left_vec.resize(size);
                has_collision.resize(size);

                LF_lead_vehicles_close_to_ego_reference_path.first.resize(size);
                LF_lead_vehicles_close_to_ego_reference_path.second.resize(size);
                LC_lead_vehicles_close_to_ego_reference_path.first.resize(size);
                LC_lead_vehicles_close_to_ego_reference_path.second.resize(size);
                LC_following_vehicles_close_to_ego_reference_path.first.resize(size);
                LC_following_vehicles_close_to_ego_reference_path.second.resize(size);
                LF_lead_vehicles_future_path_intersected.first.resize(size);
                LF_lead_vehicles_future_path_intersected.second.resize(size);
            }

            void reset(size_t size) {}
        };

        struct GenerateProposalLFWorkspace
        {
            IVectorT_traj path_nearest_idxs;
            FVectorT_traj actual_lateral_distances_abs, ego_vs, ego_as, interpolated_path_idxs;
            FVectorT_traj path_curr_curvatures, ego_desired_vs, relative_distance, leading_exo_vs, v_lateral,
                compensated_distance, cal_accs, curvature_scale_factor, ego_avg_vs_ref, interpolated_xs,
                interpolated_ys, interpolated_thetas;

            std::pair<AlignedVectorFloat, AlignedVectorFloat> lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> lead_vehicles_future_path_intersected;

            GenerateProposalLFWorkspace(size_t size)
            {
                lead_vehicles_close_to_ego_reference_path.first.resize(size);
                lead_vehicles_close_to_ego_reference_path.second.resize(size);
                lead_vehicles_future_path_intersected.first.resize(size);
                lead_vehicles_future_path_intersected.second.resize(size);
            }

            void reset(size_t size) {}
        };

        struct GenerateProposalLCWorkspace
        {
            IVectorT_traj curr_path_nearest_idxs, on_target_path_mask, assumed_exactly_on_target_path_mask;
            FVectorT_traj ego_cos_thetas, ego_sin_thetas, actual_lateral_distances_abs, interpolated_target_path_idxs;

            IVectorT_traj step_directions, tmp_target_path_nearest_idxs, should_switch_path, newly_on_target,
                consider_LC_mask, LC_allowance_flags, LF_lead_allowance_flags, in_lane_change_flag_simd,
                tmp_nearest_path_idxs;
            FVectorT_traj init_path_min_distance, target_path_min_distance, curr_path_thetas, init_path_ls,
                target_path_ls, real_target_path_distance, distance_diff, path_curr_curvatures, path_curvatures,
                ego_desired_vs, lateral_offsets_for_LF, v_lateral, distance_compensation, relative_distance,
                leading_exo_vs, ref_point_xs, ref_point_ys, ref_point_thetas, cal_accs, target_path_xs, target_path_ys,
                target_path_thetas, curr_lateral_offsets, curvature_scale_factor, ego_avg_vs_ref, interpolated_xs,
                interpolated_ys, interpolated_thetas, ego_offset_xs, ego_offset_ys, cal_steerings;

            std::pair<AlignedVectorFloat, AlignedVectorFloat> LF_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LC_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LC_following_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> LF_lead_vehicles_future_path_intersected;

            GenerateProposalLCWorkspace(size_t size)
            {
                LF_lead_vehicles_close_to_ego_reference_path.first.resize(size);
                LF_lead_vehicles_close_to_ego_reference_path.second.resize(size);
                LC_lead_vehicles_close_to_ego_reference_path.first.resize(size);
                LC_lead_vehicles_close_to_ego_reference_path.second.resize(size);
                LC_following_vehicles_close_to_ego_reference_path.first.resize(size);
                LC_following_vehicles_close_to_ego_reference_path.second.resize(size);
                LF_lead_vehicles_future_path_intersected.first.resize(size);
                LF_lead_vehicles_future_path_intersected.second.resize(size);
            }

            void reset(size_t size) { step_directions = IVectorT_traj::fill(1); }
        };

        struct TrackerWorkspace
        {
            // Force memory alignment for Eigen types
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            // Compile-time constants
            static constexpr int NS =
                static_cast<int>(utils::NUM_DISPLACEMENTS_EXPECTED); // Number of trajectory displacements (20 expected)
            static constexpr int      Batch = static_cast<int>(utils::PROPOSAL_BATCH_SIZE);
            static constexpr int      TH = static_cast<int>(utils::TRACKING_HORIZON);
            static constexpr int      Lts = static_cast<int>(utils::NUM_LATERAL_STATES);
            static constexpr uint32_t MS = utils::MAX_EXPECTED_TIME_STEPS;
            static constexpr float    DT = utils::DISCRETIZATION_TIME;
            static constexpr float    DT_2 = utils::DISCRETIZATION_TIME_SQUARE;
            static constexpr float    DT_3 = utils::DISCRETIZATION_TIME_CUBE;
            static constexpr float    DT_4 = utils::DISCRETIZATION_TIME_QUAD;

            // Static precomputed matrices
            // Regularization matrix R^T*R
            static inline const Eigen::Matrix<float, NS, NS> RtR = []()
            {
                Eigen::Matrix<float, NS, NS> m = Eigen::Matrix<float, NS, NS>::Identity() * 1e-4f;
                m(0, 0) = 0.0f;

                // Cross-boundary coupling terms
                m(NS - 2, NS - 1) = -1e-4f;
                m(NS - 1, NS - 2) = -1e-4f;
                return m;
            }();

            // Precompute A^T*A + RtR for velocity solving, where A is the NS x NS matrix that maps curvature rates to
            // heading displacements (size NS x NS) (A^T*A)_{0,0}=NS·dt², (A^T*A)_{0,j}=(NS-j)·dt·dt₂,
            // (A^T*A)_{j,k}=(NS-max(j,k))·dt₂²
            static inline const Eigen::Matrix<float, NS, NS> AtA_plus_RtR = []()
            {
                Eigen::Matrix<float, NS, NS> M;
                M.setZero();
                M(0, 0) = static_cast<float>(NS) * DT_2;
                for (size_t j = 1; j < NS; ++j)
                {
                    float val = static_cast<float>(NS - j) * DT_3;
                    M(0, j) = val;
                    M(j, 0) = val;
                }
                for (size_t j = 1; j < NS; ++j)
                {
                    for (size_t k = 1; k < NS; ++k)
                    {
                        M(j, k) = static_cast<float>(NS - std::max(j, k)) * DT_4;
                    }
                }
                M += RtR;
                return M;
            }();

            // regularization matrix Q (size NS x NS)
            static inline const Eigen::Matrix<float, NS, NS> r_m_q = []()
            {
                Eigen::Matrix<float, NS, NS> Q =
                    Eigen::Matrix<float, NS, NS>::Identity() * utils::CURVATURE_RATE_PENALTY;
                Q(0, 0) = utils::INITIAL_CURVATURE_PENALTY;
                return Q;
            }();

            // diagonal q_vec of the regularization matrix Q
            static inline const Eigen::Matrix<float, NS, 1> q_diag = []()
            {
                Eigen::Matrix<float, NS, 1> v = Eigen::Matrix<float, NS, 1>::Constant(utils::CURVATURE_RATE_PENALTY);
                v(0) = utils::INITIAL_CURVATURE_PENALTY;
                return v;
            }();

            // Precomputed LDLT decomposition of AtA_plus_RtR for velocity solving
            static inline const auto                  vel_ldlt = AtA_plus_RtR.ldlt();
            Eigen::LDLT<Eigen::Matrix<float, NS, NS>> ldlt_solver;

            // Instance members (dynamically sized by batch_size)
            // Output command vectors
            FVectorT_traj              accel_cmds = FVectorT_traj(FVectorT_traj::fill(0.0f));
            FVectorT_traj              reference_velocities = FVectorT_traj(FVectorT_traj::fill(0.0f));
            std::vector<FVectorT_traj> initial_steering_angle = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> initial_steering_rate = std::vector<FVectorT_traj>(MS);

            // Displacement vectors (NS-sized)
            std::vector<FVectorT_traj> dx_displacements = std::vector<FVectorT_traj>(NS);
            std::vector<FVectorT_traj> dy_displacements = std::vector<FVectorT_traj>(NS);
            std::vector<FVectorT_traj> heading_displacements = std::vector<FVectorT_traj>(NS);

            // Trajectory state vectors (MS-sized)
            std::vector<FVectorT_traj> ego_xs_rear_axle = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> ego_ys_rear_axle = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> ego_thetas_proposal_cos = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> ego_thetas_proposal_sin = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> ego_thetas_traj_cos = std::vector<FVectorT_traj>(MS);
            std::vector<FVectorT_traj> ego_thetas_traj_sin = std::vector<FVectorT_traj>(MS);

            // Heading least-squares variables (NS-only)
            // Eigen::Matrix<float, NS, NS> A_head = Eigen::Matrix<float, NS, NS>::Zero();
            Eigen::Matrix<float, NS, NS, Eigen::RowMajor> A_head =
                Eigen::Matrix<float, NS, NS, Eigen::RowMajor>::Zero();
            Eigen::Matrix<float, NS, 1>  rhs_head = Eigen::Matrix<float, NS, 1>::Zero();
            Eigen::Matrix<float, NS, 1>  x_params_head = Eigen::Matrix<float, NS, 1>::Zero();
            Eigen::Matrix<float, NS, NS> chol_head_v2 = Eigen::Matrix<float, NS, NS>::Zero();

            // Velocity least-squares scratch matrices (NS x Batch)
            Eigen::Matrix<float, NS, Eigen::Dynamic> Aty_all =
                Eigen::Matrix<float, NS, Eigen::Dynamic>::Zero(NS, Batch);
            Eigen::Matrix<float, NS, Eigen::Dynamic> x_all = Eigen::Matrix<float, NS, Eigen::Dynamic>::Zero(NS, Batch);

            // Lateral control matrices (Lts-only)
            Eigen::Matrix<float, Lts, Lts> A_lateral = Eigen::Matrix<float, Lts, Lts>::Identity();
            Eigen::Matrix<float, Lts, 1>   B_lateral = Eigen::Matrix<float, Lts, 1>::Zero();
            Eigen::Matrix<float, Lts, 1>   g_lateral = Eigen::Matrix<float, Lts, 1>::Zero();
            Eigen::Matrix<float, Lts, 1>   initial_state = Eigen::Matrix<float, Lts, 1>::Zero();
            Eigen::Matrix<float, Lts, 1>   state_error = Eigen::Matrix<float, Lts, 1>::Zero();
            Eigen::Matrix<float, Lts, 1>   input_matrix_lateral = Eigen::Matrix<float, Lts, 1>::Zero();
            Eigen::Matrix<float, Lts, 1>   reference_state = Eigen::Matrix<float, Lts, 1>::Zero();

            // Steering rate command output
            AlignedVectorFloat steering_rate_cmds = AlignedVectorFloat(Batch, 0.0f);

            // Displacement and heading arrays (NS x batch_size)
            Eigen::Array<float, NS, Eigen::Dynamic>  dx = Eigen::Array<float, NS, Eigen::Dynamic>::Zero(NS, Batch);
            Eigen::Array<float, NS, Eigen::Dynamic>  dy = Eigen::Array<float, NS, Eigen::Dynamic>::Zero(NS, Batch);
            Eigen::Matrix<float, NS, Eigen::Dynamic> dHeading =
                Eigen::Matrix<float, NS, Eigen::Dynamic>::Zero(NS, Batch);
            Eigen::Array<float, NS, Eigen::Dynamic> cosT = Eigen::Array<float, NS, Eigen::Dynamic>::Zero(NS, Batch);
            Eigen::Array<float, NS, Eigen::Dynamic> sinT = Eigen::Array<float, NS, Eigen::Dynamic>::Zero(NS, Batch);

            // Reference curvature profile cache (batch_size x TH)
            Eigen::Array<float, Eigen::Dynamic, TH> reference_curvature_profiles =
                Eigen::Array<float, Eigen::Dynamic, TH>::Zero(Batch, TH);

            // Batch velocity and curvature profiles (batch_size x NS)
            Eigen::Array<float, Eigen::Dynamic, NS> velocity_profile =
                Eigen::Array<float, Eigen::Dynamic, NS>::Zero(Batch, NS);
            Eigen::Array<float, Eigen::Dynamic, NS> curvature_profile =
                Eigen::Array<float, Eigen::Dynamic, NS>::Zero(Batch, NS);

            // Initial conditions per batch entry (batch_size x 1)
            Eigen::Array<float, Eigen::Dynamic, 1> initial_velocity =
                Eigen::Array<float, Eigen::Dynamic, 1>::Zero(Batch, 1);
            Eigen::Array<float, Eigen::Dynamic, 1> initial_curvature =
                Eigen::Array<float, Eigen::Dynamic, 1>::Zero(Batch, 1);

            // Acceleration and curvature-rate profiles (batch_size x (NS-1))
            Eigen::Array<float, Eigen::Dynamic, NS - 1> acceleration_profile =
                Eigen::Array<float, Eigen::Dynamic, NS - 1>::Zero(Batch, NS - 1);
            Eigen::Array<float, Eigen::Dynamic, NS - 1> curvature_rate_profile =
                Eigen::Array<float, Eigen::Dynamic, NS - 1>::Zero(Batch, NS - 1);

            TrackerWorkspace(int batch_size)
            {
                // Resize dynamic matrices only when batch_size differs from the default
                if (batch_size != Batch)
                {
                    steering_rate_cmds.resize(batch_size, 0.0f);

                    dx.resize(NS, batch_size);
                    dx.setZero();
                    dy.resize(NS, batch_size);
                    dy.setZero();
                    dHeading.resize(NS, batch_size);
                    dHeading.setZero();
                    cosT.resize(NS, batch_size);
                    cosT.setZero();
                    sinT.resize(NS, batch_size);
                    sinT.setZero();

                    reference_curvature_profiles.resize(batch_size, TH);
                    reference_curvature_profiles.setZero();

                    velocity_profile.resize(batch_size, NS);
                    velocity_profile.setZero();
                    curvature_profile.resize(batch_size, NS);
                    curvature_profile.setZero();

                    initial_velocity.resize(batch_size, 1);
                    initial_velocity.setZero();
                    initial_curvature.resize(batch_size, 1);
                    initial_curvature.setZero();

                    acceleration_profile.resize(batch_size, NS - 1);
                    acceleration_profile.setZero();
                    curvature_rate_profile.resize(batch_size, NS - 1);
                    curvature_rate_profile.setZero();

                    Aty_all.resize(NS, batch_size);
                    Aty_all.setZero();
                    x_all.resize(NS, batch_size);
                    x_all.setZero();
                }

                // Initialize lateral control input matrix
                input_matrix_lateral(static_cast<int>(utils::STEERING_ANGLE_IDX), 0) = utils::DISCRETIZATION_TIME;
            }

            void reset()
            {
                // Reset batch-dependent Eigen matrices (preserving current dimensions)
                dx.setZero();
                dy.setZero();
                dHeading.setZero();
                cosT.setZero();
                sinT.setZero();
                velocity_profile.setZero();
                curvature_profile.setZero();
                reference_curvature_profiles.setZero();
                initial_velocity.setZero();
                initial_curvature.setZero();
                acceleration_profile.setZero();
                curvature_rate_profile.setZero();

                // Reset batch-independent variables
                rhs_head.setZero();
                x_params_head.setZero();
                initial_state.setZero();
                state_error.setZero();
                chol_head_v2.setZero();
                Aty_all.setZero();
                x_all.setZero();
            }

            // Vectorized A^T*y for all batches: s(i,b) = cos·dx + sin·dy, then suffix sum
            inline void solveVelocityBatch()
            {
                // Project displacements onto the heading direction
                Aty_all = (cosT * dx + sinT * dy).matrix();

                // Backward suffix sum to accumulate path integrals
                for (int i = static_cast<int>(NS) - 2; i >= 0; --i)
                {
                    Aty_all.row(i) += Aty_all.row(i + 1);
                }

                // Scale by time step
                Aty_all.row(0) *= DT;
                Aty_all.bottomRows(NS - 1) *= DT_2;

                // Solve using the precomputed static LDLT decomposition
                x_all = vel_ldlt.solve(Aty_all);

                // Extract results
                initial_velocity = x_all.row(0).transpose().array();
                acceleration_profile = x_all.bottomRows(NS - 1).transpose().array();
            }
        };

        struct CrossScenarioEvaluationWorkspace
        {
            FVectorT_traj path_min_dists, path_left_flags, ego_path_ss, ego_path_ls;
            FVectorT_traj precomputed_non_on_drivable_area_penalties, ego_oncoming_culmulative_vs;
            FVectorT_traj value_array_sum;
            IVectorT_traj path_nearest_idxs;

            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> left_corners_on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> right_corners_on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_coming_traffic_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_intersection_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_multiple_lane_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_different_path_lanes_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> on_non_route_masks;

            std::array<FVectorT_traj, utils::PROPOSAL_TRAJECTORY_SIZE> ego_cos_thetas_traj;
            std::array<FVectorT_traj, utils::PROPOSAL_TRAJECTORY_SIZE> ego_sin_thetas_traj;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     precomputed_penalties;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_rear_axle_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_rear_axle_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_front_left_corner_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_front_left_corner_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_front_right_corner_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     ego_front_right_corner_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN>     nuplan_metric_penalties;

            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> query_aabb_min_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> query_aabb_min_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> query_aabb_max_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> query_aabb_max_ys;

            FVectorT_traj corners_xs[6];
            FVectorT_traj corners_ys[6];

            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> exo_inactive_flags;
            AlignedVectorInt                                      has_original_collision;
            AlignedVectorInt                                      has_expanded_collision;
            AlignedVectorInt                                      collided_agent_idxs;

            AlignedVectorFloat collided_exo_xs_v;
            AlignedVectorFloat collided_exo_ys_v;
            AlignedVectorFloat collided_exo_vs_v;
            AlignedVectorFloat collided_exo_thetas_v;
            AlignedVectorFloat collided_exo_cos_thetas_v;
            AlignedVectorFloat collided_exo_sin_thetas_v;
            AlignedVectorFloat collided_exo_expanded_bb_extent_xs_v;
            AlignedVectorFloat collided_exo_expanded_bb_extent_ys_v;
            AlignedVectorFloat collided_min_distances_v;

            CrossScenarioEvaluationWorkspace(size_t batch_size)
            {
                exo_inactive_flags.resize(batch_size);
                has_original_collision.resize(batch_size);
                has_expanded_collision.resize(batch_size);
                collided_agent_idxs.resize(batch_size);
                collided_exo_xs_v.resize(batch_size);
                collided_exo_ys_v.resize(batch_size);
                collided_exo_vs_v.resize(batch_size);
                collided_exo_thetas_v.resize(batch_size);
                collided_exo_cos_thetas_v.resize(batch_size);
                collided_exo_sin_thetas_v.resize(batch_size);
                collided_exo_expanded_bb_extent_xs_v.resize(batch_size);
                collided_exo_expanded_bb_extent_ys_v.resize(batch_size);
                collided_min_distances_v.resize(batch_size);
            }

            void reset(size_t batch_size)
            {
                // Initialize FVectorT_traj variables to zero/default
                precomputed_non_on_drivable_area_penalties = FVectorT_traj::fill(0.0f);
                ego_oncoming_culmulative_vs = FVectorT_traj::fill(0.0f);
                value_array_sum = FVectorT_traj::fill(0.0f);
            }
        };

        struct OccupancyMapWorkspace
        {
            static constexpr int    corner_category_num = 6;
            static constexpr size_t VECTOR_SIZE = 16;
            static constexpr size_t BUFFER_SIZE = FVectorT_1::num_scalars_per_row;

            std::vector<std::vector<std::vector<std::string>>> corners_area_ids;
            std::vector<AlignedVectorBool>                     corners_area_on_drivable_area_mask;

            AlignedVectorInt on_non_drivable_area_mask_v;
            AlignedVectorInt left_corners_on_non_drivable_area_mask_v;
            AlignedVectorInt right_corners_on_non_drivable_area_mask_v;
            AlignedVectorInt on_coming_traffic_mask_v;
            AlignedVectorInt on_intersection_mask_v;
            AlignedVectorInt on_multiple_lane_mask_v;
            AlignedVectorInt on_different_path_lanes_mask_v;
            AlignedVectorInt on_non_route_mask_v;

            std::vector<float>                            min_ys;
            std::vector<float>                            max_ys;
            std::vector<std::vector<float>>               different_lateral_offset_min_ys;
            std::vector<std::vector<float>>               different_lateral_offset_max_ys;
            std::array<float, FVectorT_traj::num_scalars> corners_xs_arr[corner_category_num];
            std::array<float, FVectorT_traj::num_scalars> corners_ys_arr[corner_category_num];
            std::array<float, FVectorT_traj::num_scalars> center_xs_arr;
            std::array<float, FVectorT_traj::num_scalars> center_ys_arr;
            std::array<float, FVectorT_traj::num_scalars> rear_axle_xs_arr;
            std::array<float, FVectorT_traj::num_scalars> rear_axle_ys_arr;

            // Pre-allocated SIMD buffers to avoid dynamic allocation
            int                     buffer_index, vec_index;
            AlignedVectorFloat      simd_segments_x1_buffer;
            AlignedVectorFloat      simd_segments_y1_buffer;
            AlignedVectorFloat      simd_segments_x2_buffer;
            AlignedVectorFloat      simd_segments_y2_buffer;
            AlignedVectorInt        simd_valid_mask_buffer;
            std::vector<FVectorT_1> simd_segments_x1_vec;
            std::vector<FVectorT_1> simd_segments_y1_vec;
            std::vector<FVectorT_1> simd_segments_x2_vec;
            std::vector<FVectorT_1> simd_segments_y2_vec;
            std::vector<FVectorT_1> simd_valid_mask_vec;

            OccupancyMapWorkspace()
            {
                corners_area_ids.resize(corner_category_num,
                                        std::vector<std::vector<std::string>>(FVectorT_traj::num_scalars));
                corners_area_on_drivable_area_mask.resize(corner_category_num,
                                                          AlignedVectorBool(FVectorT_traj::num_scalars, false));

                on_non_drivable_area_mask_v.resize(FVectorT_traj::num_scalars);
                left_corners_on_non_drivable_area_mask_v.resize(FVectorT_traj::num_scalars);
                right_corners_on_non_drivable_area_mask_v.resize(FVectorT_traj::num_scalars);
                on_coming_traffic_mask_v.resize(FVectorT_traj::num_scalars);
                on_intersection_mask_v.resize(FVectorT_traj::num_scalars);
                on_multiple_lane_mask_v.resize(FVectorT_traj::num_scalars);
                on_different_path_lanes_mask_v.resize(FVectorT_traj::num_scalars);
                on_non_route_mask_v.resize(FVectorT_traj::num_scalars);

                min_ys.resize(corner_category_num);
                max_ys.resize(corner_category_num);
                different_lateral_offset_min_ys.resize(corner_category_num,
                                                       std::vector<float>(FVectorT_traj::num_rows));
                different_lateral_offset_max_ys.resize(corner_category_num,
                                                       std::vector<float>(FVectorT_traj::num_rows));

                buffer_index = 0;
                vec_index = 0;
                simd_segments_x1_buffer.resize(BUFFER_SIZE, 0.0f);
                simd_segments_y1_buffer.resize(BUFFER_SIZE, 0.0f);
                simd_segments_x2_buffer.resize(BUFFER_SIZE, 0.0f);
                simd_segments_y2_buffer.resize(BUFFER_SIZE, 0.0f);
                simd_valid_mask_buffer.resize(BUFFER_SIZE, 0);
                simd_segments_x1_vec.resize(VECTOR_SIZE);
                simd_segments_y1_vec.resize(VECTOR_SIZE);
                simd_segments_x2_vec.resize(VECTOR_SIZE);
                simd_segments_y2_vec.resize(VECTOR_SIZE);
                simd_valid_mask_vec.resize(VECTOR_SIZE);
            }

            void reset()
            {
                for (auto &outer_vec : corners_area_ids)
                {
                    for (auto &inner_vec : outer_vec)
                    {
                        inner_vec.clear(); // clear strings but keep capacity
                    }
                }

                for (auto &outer_vec : corners_area_on_drivable_area_mask)
                {
                    std::fill(outer_vec.begin(), outer_vec.end(), false);
                }

                std::fill(on_non_drivable_area_mask_v.begin(), on_non_drivable_area_mask_v.end(), 0);
                std::fill(left_corners_on_non_drivable_area_mask_v.begin(),
                          left_corners_on_non_drivable_area_mask_v.end(), 0xFFFFFFFF);
                std::fill(right_corners_on_non_drivable_area_mask_v.begin(),
                          right_corners_on_non_drivable_area_mask_v.end(), 0xFFFFFFFF);
                std::fill(on_coming_traffic_mask_v.begin(), on_coming_traffic_mask_v.end(), 0xFFFFFFFF);
                std::fill(on_intersection_mask_v.begin(), on_intersection_mask_v.end(), 0);
                std::fill(on_multiple_lane_mask_v.begin(), on_multiple_lane_mask_v.end(), 0);
                std::fill(on_different_path_lanes_mask_v.begin(), on_different_path_lanes_mask_v.end(), 0);
                std::fill(on_non_route_mask_v.begin(), on_non_route_mask_v.end(), 0);
            }
        };

        // Thread pool
        class ThreadPool
        {
          public:
            ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
            ~ThreadPool();

            template <typename F, typename... Args>
            auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
            {
                using return_type = typename std::result_of<F(Args...)>::type;

                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...));

                std::future<return_type> result = task->get_future();

                // Wrap the task in a function object
                std::function<void()> wrapper = [task]() { (*task)(); };
                enqueueTask(wrapper);

                return result;
            }

          private:
            // Add a non-template method to handle task enqueueing
            void enqueueTask(std::function<void()> task);

            class Impl;
            std::unique_ptr<Impl> pImpl;
        };

        void enableSIMD();
        bool isSIMDAvailable();
        int  getSIMDWidth();

        using Clock = std::chrono::steady_clock;
        using TimePoint = Clock::time_point;
        using DurationMs = std::chrono::duration<double, std::micro>;

        // Fast timestamp type (using CPU cycle counts)
        using FastTimeStamp = TimePoint;

        // Inline access to current time point (replacement for fast_now())
        inline FastTimeStamp fast_now()
        {
            // Use std::chrono::steady_clock to ensure monotonic time
            return Clock::now();
        }

        /**
         * @brief Quickly stop timing and accumulate the duration.
         * @param start_time Start time point (FastTimeStamp/TimePoint).
         * @param total_time_micros_ref Reference to accumulate total microseconds.
         */
        inline void fast_end(FastTimeStamp start_time, double &total_time_micros_ref)
        {
            // 1. Capture end time point
            TimePoint end_time = fast_now();

            // 2. Compute duration
            auto duration = end_time - start_time;

            // 3. Convert to DurationMs (double, microseconds) using duration_cast and accumulate
            // Note: renamed total_time_ms_ref to total_time_micros_ref to match units
            double duration_micros = std::chrono::duration_cast<DurationMs>(duration).count();
            total_time_micros_ref += duration_micros;
        }

        /**
         * @brief Quickly stop timing, accumulate duration and increment count.
         * @param start_time Start time point (FastTimeStamp/TimePoint).
         * @param total_time_micros_ref Reference to accumulate total microseconds.
         * @param count_ref Reference to increment call count.
         */
        inline void fast_end(FastTimeStamp start_time, double &total_time_micros_ref, long long &count_ref)
        {
            // 1. Capture end time point
            TimePoint end_time = fast_now();

            // 2. Compute duration
            auto duration = end_time - start_time;

            // 3. Convert to DurationMs (double, microseconds) using duration_cast and accumulate
            double duration_micros = std::chrono::duration_cast<DurationMs>(duration).count();
            total_time_micros_ref += duration_micros;

            // 4. Increment count
            ++count_ref;
        }

        // Legacy chrono interface (compatibility)
        inline TimePoint now()
        {
            // This is a very low-overhead operation: calls the clock once.
            return Clock::now();
        }

        inline void end(TimePoint start_time, double &total_time_ms_ref, long long &count_ref)
        {
            // 1. Capture end time
            auto end_time = Clock::now();

            // 2. Compute duration (microseconds)
            double duration_ms = DurationMs(end_time - start_time).count();

            // 3. Calibrate duration: subtract timer overhead (end() function only)
            // For extreme precision, G_TIMER_OVERHEAD_MS should include only the post-clock-call logic in end().
            // In practice we often subtract the full (now() + end()) overhead; use the previous calibration value here.
            double corrected_duration_ms = std::max(0.0, duration_ms);

            // 4. Accumulate into the referenced variable
            total_time_ms_ref += corrected_duration_ms;
            ++count_ref;
        }

        // Forward declaration
        struct PerformanceData;

        // Performance analysis structures / variables
        struct PerformanceData
        {
            double    total_time_ms = 0.0;
            long long call_count = 0;
            // Use std::map for automatic ordering to ease log inspection
            std::map<std::string, std::shared_ptr<PerformanceData>> sub_functions;
        };

        // 1. Global aggregated data (all threads will merge here)
        extern std::map<std::string, PerformanceData> g_global_perf_data;
        extern std::mutex                             g_perf_mutex; // Mutex protecting global performance data

        // 2. Thread-local data (each thread owns its data; no locking required, very fast)
        extern thread_local std::map<std::string, PerformanceData> t_local_perf_data;

        // Print performance profiling data
        void commitPerformanceData();
        void printPerformanceData();

        class FastScopedTimer
        {
          public:
            // Type aliases to simplify code
            using Clock = std::chrono::high_resolution_clock;
            using TimePoint = Clock::time_point;

            // parent_function defaults to empty, indicating a top-level function
            FastScopedTimer(const std::string &function_name, const std::string &parent_function = "", int count = 1)
                : function_name_(function_name), parent_function_(parent_function),
                  //   start_cycles_(fast_now()), // Ensure fast_now() is available
                  start_time_(Clock::now()), count_(count)
            {
            }

            ~FastScopedTimer()
            {
                // 1. Capture end time
                TimePoint end_time = Clock::now();

                // 2. Compute duration
                auto duration = end_time - start_time_;

                // 3. Convert to milliseconds (double) using duration_cast
                // Use std::chrono::duration<double, std::milli> to ensure a floating-point millisecond result
                double duration_ms =
                    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();

                // Core change: only operate on t_local_perf_data
                if (parent_function_.empty())
                {
                    // Top-level function
                    t_local_perf_data[function_name_].total_time_ms += duration_ms;
                    t_local_perf_data[function_name_].call_count += count_;
                }
                else
                {
                    // Child function
                    // Note: if the parent hasn't been destructed yet, this will create an empty parent entry; this is
                    // safe
                    auto &parent = t_local_perf_data[parent_function_];

                    // Find or create child function node
                    if (parent.sub_functions.find(function_name_) == parent.sub_functions.end())
                    {
                        parent.sub_functions[function_name_] = std::make_shared<PerformanceData>();
                    }
                    parent.sub_functions[function_name_]->total_time_ms += duration_ms;
                    parent.sub_functions[function_name_]->call_count += count_;
                }
            }

          private:
            std::string function_name_;
            std::string parent_function_;
            TimePoint   start_time_; // stores std::chrono::time_point
            int         count_;
        };

        enum class LogLevel
        {
            DEBUG,   // for detailed debug information
            INFO,    // for informational messages
            WARNING, // for warnings
            ERROR    // for error messages
        };

        // Configure logging system
        void initializeLogger(const std::string &log_file, LogLevel level);

        // Basic logging function - version unaffected by O3 optimization
        void log(LogLevel level, const std::string &message, bool append_newline = true);

        // Advanced logging template supporting stream-style output; may be removed in optimized builds
        template <LogLevel L> class Logger
        {
          private:
            std::ostringstream os_;
            bool               append_newline_;

            // Compile-time check for whether the build is in optimized mode
            constexpr static bool isOptimized()
            {
#if defined(NDEBUG) || defined(__OPTIMIZE__)
                return (L == LogLevel::DEBUG || L == LogLevel::INFO);
#else
                return false;
#endif
            }

          public:
            explicit Logger(bool append_newline = true) : append_newline_(append_newline)
            {
                os_ << std::fixed << std::setprecision(4);
            }
            ~Logger()
            {
                if constexpr (!isOptimized())
                {
                    log(L, os_.str(), append_newline_);
                }
            }

            // If in optimized build, DEBUG/INFO stream operations may be optimized away
            template <typename T> Logger &operator<<(const T &value)
            {
                if constexpr (!isOptimized())
                {
                    os_ << value;
                }
                return *this;
            }
        };

// Below are compile-time optimized logging macros; under O3 optimization they will be skipped
// Basic logging macros - support string messages
#if defined(NDEBUG) || defined(__OPTIMIZE__)
#define VECQMDP_LOG_DEBUG(message)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#define VECQMDP_LOG_INFO(message)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#else
#define VECQMDP_LOG_DEBUG(message) vec_qmdp::utils::log(vec_qmdp::utils::LogLevel::DEBUG, message)
#define VECQMDP_LOG_INFO(message) vec_qmdp::utils::log(vec_qmdp::utils::LogLevel::INFO, message)
#endif

// WARNING and ERROR level logs are preserved even under O3 optimization
#define VECQMDP_LOG_WARNING(message) vec_qmdp::utils::log(vec_qmdp::utils::LogLevel::WARNING, message)
#define VECQMDP_LOG_ERROR(message) vec_qmdp::utils::log(vec_qmdp::utils::LogLevel::ERROR, message)

// Stream-style logging macros - more flexible output
#define VECQMDP_LOG_DEBUG_STREAM vec_qmdp::utils::Logger<vec_qmdp::utils::LogLevel::DEBUG>()
#define VECQMDP_LOG_INFO_STREAM vec_qmdp::utils::Logger<vec_qmdp::utils::LogLevel::INFO>()
#define VECQMDP_LOG_WARNING_STREAM vec_qmdp::utils::Logger<vec_qmdp::utils::LogLevel::WARNING>()
#define VECQMDP_LOG_ERROR_STREAM vec_qmdp::utils::Logger<vec_qmdp::utils::LogLevel::ERROR>()

// Short alias macros for more concise calls
#define LOG_D VECQMDP_LOG_DEBUG
#define LOG_I VECQMDP_LOG_INFO
#define LOG_W VECQMDP_LOG_WARNING
#define LOG_E VECQMDP_LOG_ERROR

// Short alias stream macros
#define LOG_DS VECQMDP_LOG_DEBUG_STREAM
#define LOG_IS VECQMDP_LOG_INFO_STREAM
#define LOG_WS VECQMDP_LOG_WARNING_STREAM
#define LOG_ES VECQMDP_LOG_ERROR_STREAM

        struct PerformanceMetrics
        {
            double computation_time;
            double memory_usage;
            int    num_threads_used;
            int    num_simd_operations;
        };

        PerformanceMetrics getPerformanceMetrics();
        void               resetPerformanceMetrics();

        struct pair_hash
        {
            size_t operator()(const std::pair<std::string, std::string> &p) const
            {
                return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
            }
        };
    } // namespace utils
} // namespace vec_qmdp