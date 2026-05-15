/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file tracker.hpp
 * @brief LQR-based trajectory tracking controller for the kinematic bicycle model.
 */

#pragma once

#include <cmath>
#include <core/net_belief.hpp>
#include <core/state.hpp>
#include <cstddef>
#include <memory>
#include <sys/resource.h>
#include <utils/aligned_allocator.hpp>
#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>
#include <vamp/vector/eigen.hh>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        /*==================== Tracker Class Definition ====================*/

        /**
         * @brief Tracker implements an LQR-based trajectory tracking controller for the kinematic bicycle model.
         *
         * The tracker operates on a batch of reference trajectories. Based on nuplan-devkit implementation.
         * Link: https://github.com/motional/nuplan-devkit.
         *
         * We decouple the system into longitudinal and lateral subsystems and linearize using
         * small-angle approximation. Then we solve two consecutive LQR sub-problems to find
         * the acceleration and steering rate inputs.
         *
         * Longitudinal subsystem model:
         *   State: velocity
         *   Input: acceleration
         *   Kinematics (continuous time):
         *      velocity_dot = acceleration
         *
         * Lateral subsystem model (after linearization / small-angle approximation):
         *   State: [lateral_error, heading_error, steering_angle]
         *   Input: steering_rate
         *   Kinematics (continuous time):
         *      lateral_error  = velocity * heading_error
         *      heading_error  = velocity * (steering_angle / wheelbase_length - curvature)
         *      steering_angle = steering_rate
         *
         * Continuous-time dynamics are discretized using Euler integration with zero-order hold on inputs.
         * For stopped references, a simplified stopping P-controller is used instead of the LQR controller.
         *
         * The final control inputs passed to the kinematic model are:
         *   - acceleration
         *   - steering_rate
         */

        class Tracker
        {
          public:
            using FVectorT_traj = utils::FVectorT_traj;
            using IVectorT_traj = utils::IVectorT_traj;
            using AlignedVectorFloat = utils::AlignedVectorFloat;
            using AlignedVectorInt = utils::AlignedVectorInt;

            /**
             * @brief Constructor; initializes LQR controller parameters.
             */
            Tracker();
            virtual ~Tracker();

            /**
             * @brief Track proposal trajectories using LQR to produce actual trajectories
             * @param ego_state current ego vehicle state
             * @param ego_xs_proposal ego x-coordinate proposal trajectory
             * @param ego_ys_proposal ego y-coordinate proposal trajectory
             * @param ego_vs_proposal ego velocity proposal trajectory
             * @param ego_as_proposal ego acceleration proposal trajectory
             * @param ego_thetas_proposal ego heading angle proposal trajectory
             * @param ego_xs_traj ego x-coordinate tracked trajectory (output)
             * @param ego_ys_traj ego y-coordinate tracked trajectory (output)
             * @param ego_vs_traj ego velocity tracked trajectory (output)
             * @param ego_as_traj ego acceleration tracked trajectory (output)
             * @param ego_thetas_traj ego heading angle tracked trajectory (output)
             */

            void trackTrajectoryWithWorkspace(
                const std::vector<FVectorT_traj> &ego_xs_proposal, const std::vector<FVectorT_traj> &ego_ys_proposal,
                const std::vector<FVectorT_traj> &ego_vs_proposal, const std::vector<FVectorT_traj> &ego_as_proposal,
                const std::vector<FVectorT_traj> &ego_thetas_proposal, std::vector<FVectorT_traj> &ego_xs_traj,
                std::vector<FVectorT_traj> &ego_ys_traj, std::vector<FVectorT_traj> &ego_vs_traj,
                std::vector<FVectorT_traj> &ego_as_traj, std::vector<FVectorT_traj> &ego_thetas_traj,
                const float &ego_initial_x, const float &ego_initial_y, const float &ego_initial_v,
                const float &ego_current_a, const float &ego_initial_theta, const float &ego_initial_steering_angle,
                const float &ego_initial_steering_rate);

            void trackTrajectorySerial(const std::vector<AlignedVectorFloat> &ego_xs_proposal,
                                       const std::vector<AlignedVectorFloat> &ego_ys_proposal,
                                       const std::vector<AlignedVectorFloat> &ego_vs_proposal,
                                       const std::vector<AlignedVectorFloat> &ego_as_proposal,
                                       const std::vector<AlignedVectorFloat> &ego_thetas_proposal,
                                       std::vector<AlignedVectorFloat>       &ego_xs_traj,
                                       std::vector<AlignedVectorFloat>       &ego_ys_traj,
                                       std::vector<AlignedVectorFloat>       &ego_vs_traj,
                                       std::vector<AlignedVectorFloat>       &ego_as_traj,
                                       std::vector<AlignedVectorFloat> &ego_thetas_traj, const float &ego_initial_x,
                                       const float &ego_initial_y, const float &ego_initial_v,
                                       const float &ego_current_a, const float &ego_initial_theta,
                                       const float &ego_initial_steering_angle, const float &ego_initial_steering_rate);

            void printMemoryUsage();

          private:
            // LQR parameters
            constexpr static float q_longitudinal_ = utils::Q_LONGITUDINAL; // Longitudinal subsystem Q matrix weight
            constexpr static float r_longitudinal_ = utils::R_LONGITUDINAL; // Longitudinal subsystem R matrix weight
            constexpr static float qrq_longitudinal_ = utils::QRQ_LONGITUDINAL; // Longitudinal subsystem coefficient
            constexpr static float q_lateral_1_ = utils::Q_LATERAL_1;           // Lateral subsystem Q matrix weight (1)
            constexpr static float q_lateral_2_ = utils::Q_LATERAL_2;           // Lateral subsystem Q matrix weight (2)
            constexpr static float q_lateral_3_ = utils::Q_LATERAL_3;           // Lateral subsystem Q matrix weight (3)
            constexpr static float r_lateral_ = utils::R_LATERAL;               // Lateral subsystem R matrix weight
            constexpr static float discretization_time_ =
                utils::DISCRETIZATION_TIME; // Time interval for discretizing continuous-time dynamics
            constexpr static int tracking_horizon_ =
                static_cast<int>(utils::TRACKING_HORIZON);              // Number of discrete time steps for LQR
            constexpr static float jerk_penalty_ = utils::JERK_PENALTY; // Jerk penalty weight
            constexpr static float curvature_rate_penalty_ =
                utils::CURVATURE_RATE_PENALTY; // Curvature rate penalty weight
            constexpr static float stopping_proportional_gain_ =
                utils::STOPPING_PROPORTIONAL_GAIN; // P-controller proportional gain for stopping
            constexpr static float stopping_velocity_ =
                utils::STOPPING_VELOCITY; // Velocity threshold below which the vehicle is considered stopped (bypass
                                          // LQR)
            constexpr static float initial_curvature_penalty_ =
                utils::INITIAL_CURVATURE_PENALTY;                                    // Initial curvature penalty weight
            constexpr static size_t num_lateral_states_ = utils::NUM_LATERAL_STATES; // Number of lateral states
            constexpr static size_t lateral_error_idx_ = utils::LATERAL_ERROR_IDX;   // Lateral error index
            constexpr static size_t heading_error_idx_ = utils::HEADING_ERROR_IDX;   // Heading error index
            constexpr static size_t steering_angle_idx_ = utils::STEERING_ANGLE_IDX; // Steering angle index
            constexpr static float  horizon_dt_ = tracking_horizon_ * discretization_time_;
            constexpr static float  dt_2_ = utils::DISCRETIZATION_TIME_SQUARE;

            // Vehicle parameters
            constexpr static float wheelbase_length_ = utils::WHEEL_BASE; // Distance between front and rear axles
            constexpr static float max_acceleration_ = utils::MAX_ACC;    // Maximum allowed acceleration
            constexpr static float max_deceleration_ = utils::MAX_DEC; // Maximum allowed deceleration (positive value)
            constexpr static float max_steering_angle_ = utils::MAX_STEERING_ANGLE; // Maximum allowed steering angle
            constexpr static float max_steering_rate_ = utils::MAX_STEERING_RATE;   // Maximum allowed steering rate
            constexpr static float vehicle_rear_axle_to_center_ =
                utils::VEHICLE_REAR_AXLE_TO_CENTER; // Distance between rear axle and vehicle center

            // Time parameters and reference states
            inline static const Eigen::RowVectorXf ones_tracking_horizon_ = Eigen::RowVectorXf::Ones(tracking_horizon_);
            inline static const Eigen::RowVectorXf time_factor_ =
                Eigen::RowVectorXf::LinSpaced(tracking_horizon_, 0.0f, (tracking_horizon_ - 1) * discretization_time_);

            inline static const Eigen::ArrayXf ones_tracking_horizon_array_ = Eigen::ArrayXf::Ones(tracking_horizon_);
            inline static const Eigen::ArrayXf time_factor_array_ =
                Eigen::ArrayXf::LinSpaced(tracking_horizon_, 0.0f, (tracking_horizon_ - 1) * discretization_time_);
            constexpr static float discretization_time_over_wheelbase_length_ =
                discretization_time_ / wheelbase_length_;
            inline static const Eigen::VectorXf reference_state_ = Eigen::VectorXf::Zero(num_lateral_states_);

            // Regularization matrix R (size approx. (M-2) x (M))
            inline static const Eigen::Matrix<float, 40, 40> RtR_fixed_40_ = []()
            {
                Eigen::Matrix<float, 40, 40> matrix = Eigen::Matrix<float, 40, 40>::Identity() * 1e-4f;
                matrix(0, 0) = 0.0f;
                matrix(38, 39) = -1e-4f;
                matrix(39, 38) = -1e-4f;
                return matrix;
            }();

            inline static const Eigen::Matrix<float, 20, 20> RtR_fixed_20_ = []()
            {
                Eigen::Matrix<float, 20, 20> matrix = Eigen::Matrix<float, 20, 20>::Identity() * 1e-4f;
                matrix(0, 0) = 0.0f;
                matrix(18, 19) = -1e-4f;
                matrix(19, 18) = -1e-4f;
                return matrix;
            }();

            // Lateral Q matrix and R matrix
            // Q: The state tracking 2-norm cost matrix.
            // R: The input 2-norm cost matrix.
            inline static const Eigen::MatrixXf Q_lateral_ = []
            {
                Eigen::MatrixXf m = Eigen::MatrixXf::Zero(num_lateral_states_, num_lateral_states_);
                m.diagonal() << q_lateral_1_, q_lateral_2_, q_lateral_3_;
                return m;
            }();

            inline static const Eigen::MatrixXf R_lateral_ = []
            {
                Eigen::MatrixXf m = Eigen::MatrixXf::Identity(1, 1) * r_lateral_;
                return m;
            }();

            /**
             * @brief Update ego vehicle state using bicycle model
             * @param ego_xs ego x coordinates
             * @param ego_ys ego y coordinates
             * @param ego_vs ego velocities
             * @param ego_accs ego accelerations
             * @param ego_thetas ego heading angles
             * @param ego_thetas_cos cosines of ego headings
             * @param ego_thetas_sin sines of ego headings
             * @param accs acceleration commands
             * @param steerings steering angle commands
             */
            template <typename T>
            void updateState(T &ego_xs, T &ego_ys, T &ego_vs, T &ego_as, T &ego_thetas, T &ego_thetas_cos,
                             T &ego_thetas_sin, const T &cal_accs, const T &cal_steerings, T &steering_angle,
                             T &steering_rate);

            template <typename T>
            void updateStateSerial(T &ego_xs, T &ego_ys, T &ego_vs, T &ego_as, T &ego_thetas, T &ego_thetas_cos,
                                   T &ego_thetas_sin, const T &cal_accs, const T &cal_steerings, T &steering_angle,
                                   T &steering_rate);

            // Template for VAMP Vector types (with to_array method)
            template <typename T> void printFvectorT(const std::string &name, const T &vec);

            // helper functions for debugging
            void printMatrixInfo(const std::string &name, const Eigen::MatrixXf &matrix);
            void printVectorInfo(const std::string &name, const Eigen::VectorXf &vector);
        };
    } // namespace planning
} // namespace vec_qmdp