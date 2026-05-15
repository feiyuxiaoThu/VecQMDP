#include <chrono>
#include <planning/tracker.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        Tracker::Tracker()
        {
// Validate parameters
#if DEBUG_TRACKER
            assert(discretization_time_ > 0.0f && "Discretization time must be positive");
            assert(tracking_horizon_ > 0 && "Tracking horizon must be positive");
            assert(stopping_velocity_ >= 0.0f && "Stopping velocity cannot be negative");
            assert(wheelbase_length_ > 0.0f && "Wheelbase length must be positive");
            assert(num_lateral_states_ > 0 && "Number of lateral states must be positive");
            assert(lateral_error_idx_ > 0 && heading_error_idx_ > 0 && steering_angle_idx_ > 0 &&
                   "State indices cannot be negative");
            assert(lateral_error_idx_ <= num_lateral_states_ && heading_error_idx_ <= num_lateral_states_ &&
                   steering_angle_idx_ <= num_lateral_states_ &&
                   "State indices must be less than number of lateral states");
            assert(max_steering_angle_ > 0.0f && "Max steering angle must be positive");
            assert(max_steering_rate_ > 0.0f && "Max steering rate must be positive");
            assert(max_acceleration_ > 0.0f && "Max acceleration must be positive");
            assert(max_deceleration_ < 0.0f && "Max deceleration must be negative");
#endif
        }

        Tracker::~Tracker() {}

        // print memory usage
        void Tracker::printMemoryUsage()
        {
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            std::cout << "Memory usage: " << usage.ru_maxrss << " KB" << std::endl;
        }

        template <typename T>
        void Tracker::updateState(T &ego_xs, T &ego_ys, T &ego_vs, T &ego_as, T &ego_thetas, T &ego_thetas_cos,
                                  T &ego_thetas_sin, const T &cal_accs, const T &cal_steerings, T &steering_angle,
                                  T &steering_rate)
        {
            // constants definition
            constexpr float STEP_INTERVAL = utils::TIME_STEP;
            constexpr float WHEEL_BASE = utils::WHEEL_BASE;
            constexpr float ACCEL_TIME_CONSTANT = utils::ACCEL_TIME_CONSTANT;
            constexpr float STEERING_ANGLE_TIME_CONSTANT = utils::STEERING_ANGLE_TIME_CONSTANT;
            constexpr float MIN_PI = utils::MIN_PI;
            constexpr float MAX_STEERING_ANGLE = utils::PI_1_3;
            constexpr float ACCEL_CONSTANT_DIVIDER = STEP_INTERVAL / (STEP_INTERVAL + ACCEL_TIME_CONSTANT);
            constexpr float STEERING_CONSTANT_DIVIDER = STEP_INTERVAL / (STEP_INTERVAL + STEERING_ANGLE_TIME_CONSTANT);

            // update commands
            T updated_accel_x = ACCEL_CONSTANT_DIVIDER * (cal_accs - ego_as) + ego_as;
            T updated_steering_rate = STEERING_CONSTANT_DIVIDER * cal_steerings;

            // update state
            ego_as = updated_accel_x;
            ego_thetas =
                utils::NormalizeAngleSIMD<T>(ego_thetas + ego_vs * steering_angle.tan() / WHEEL_BASE * STEP_INTERVAL);
            ego_xs = ego_xs + ego_vs * ego_thetas_cos * STEP_INTERVAL;
            ego_ys = ego_ys + ego_vs * ego_thetas_sin * STEP_INTERVAL;
            ego_thetas.sincos(ego_thetas_sin, ego_thetas_cos);
            ego_vs = ego_vs + updated_accel_x * STEP_INTERVAL;
            steering_angle = (steering_angle + updated_steering_rate * STEP_INTERVAL)
                                 .max(-MAX_STEERING_ANGLE)
                                 .min(MAX_STEERING_ANGLE);
            steering_rate = updated_steering_rate;
        }

        template <typename T>
        void Tracker::updateStateSerial(T &ego_xs, T &ego_ys, T &ego_vs, T &ego_as, T &ego_thetas, T &ego_thetas_cos,
                                        T &ego_thetas_sin, const T &cal_accs, const T &cal_steerings, T &steering_angle,
                                        T &steering_rate)
        {
            // constants definition
            constexpr float STEP_INTERVAL = utils::TIME_STEP;
            constexpr float WHEEL_BASE = utils::WHEEL_BASE;
            constexpr float ACCEL_TIME_CONSTANT = utils::ACCEL_TIME_CONSTANT;
            constexpr float STEERING_ANGLE_TIME_CONSTANT = utils::STEERING_ANGLE_TIME_CONSTANT;
            constexpr float MIN_PI = utils::MIN_PI; // -pi
            constexpr float MAX_STEERING_ANGLE = utils::PI_1_3;
            constexpr float ACCEL_CONSTANT_DIVIDER = STEP_INTERVAL / (STEP_INTERVAL + ACCEL_TIME_CONSTANT);
            constexpr float STEERING_CONSTANT_DIVIDER = STEP_INTERVAL / (STEP_INTERVAL + STEERING_ANGLE_TIME_CONSTANT);

            // update commands
            T updated_accel_x = ACCEL_CONSTANT_DIVIDER * (cal_accs - ego_as) + ego_as;
            T updated_steering_rate = STEERING_CONSTANT_DIVIDER * cal_steerings;

            // update state
            ego_as = updated_accel_x;
            ego_thetas =
                utils::NormalizeAngle(ego_thetas + ego_vs * std::tan(steering_angle) / WHEEL_BASE * STEP_INTERVAL);
            ego_xs = ego_xs + ego_vs * ego_thetas_cos * STEP_INTERVAL;
            ego_ys = ego_ys + ego_vs * ego_thetas_sin * STEP_INTERVAL;
            ego_thetas_cos = std::cos(ego_thetas);
            ego_thetas_sin = std::sin(ego_thetas);
            ego_vs = ego_vs + updated_accel_x * STEP_INTERVAL;
            steering_angle =
                std::max(std::min(steering_angle + updated_steering_rate * STEP_INTERVAL, MAX_STEERING_ANGLE),
                         -MAX_STEERING_ANGLE);
            steering_rate = updated_steering_rate;
        }

        void Tracker::trackTrajectoryWithWorkspace(
            const std::vector<FVectorT_traj> &ego_xs_proposal, const std::vector<FVectorT_traj> &ego_ys_proposal,
            const std::vector<FVectorT_traj> &ego_vs_proposal, const std::vector<FVectorT_traj> &ego_as_proposal,
            const std::vector<FVectorT_traj> &ego_thetas_proposal, std::vector<FVectorT_traj> &ego_xs_traj,
            std::vector<FVectorT_traj> &ego_ys_traj, std::vector<FVectorT_traj> &ego_vs_traj,
            std::vector<FVectorT_traj> &ego_as_traj, std::vector<FVectorT_traj> &ego_thetas_traj,
            const float &ego_initial_x, const float &ego_initial_y, const float &ego_initial_v,
            const float &ego_current_a, const float &ego_initial_theta, const float &ego_initial_steering_angle,
            const float &ego_initial_steering_rate)
        {
            // local variables
            constexpr size_t       expected_traj_size = utils::MAX_EXPECTED_TIME_STEPS;
            constexpr size_t       num_displacements = expected_traj_size - 1;
            constexpr float        vehicle_rear_axle_to_center = vehicle_rear_axle_to_center_; // 1.461f
            constexpr size_t       batch_size = FVectorT_traj::num_scalars;
            constexpr size_t       tracking_horizon = tracking_horizon_;
            constexpr float        dt = discretization_time_;
            constexpr float        dt_2 = dt_2_;
            constexpr float        stopping_velocity = stopping_velocity_;
            constexpr float        stopping_proportional_gain = stopping_proportional_gain_;
            constexpr float        qrq_longitudinal = qrq_longitudinal_;
            constexpr Eigen::Index lateral_error_idx = lateral_error_idx_;
            constexpr Eigen::Index heading_error_idx = heading_error_idx_;
            constexpr Eigen::Index steering_angle_idx = steering_angle_idx_;
            constexpr float discretization_time_over_wheelbase_length = discretization_time_over_wheelbase_length_;
            const auto     &ones_tracking_horizon_array = ones_tracking_horizon_array_;
            const auto     &time_factor_array = time_factor_array_;
            constexpr Eigen::Index num_lateral_states = num_lateral_states_;

            // create reusable vectors and matrices for Eigen operations
            static thread_local utils::TrackerWorkspace tracker_ws(utils::PROPOSAL_BATCH_SIZE);
            auto                                       &ws = tracker_ws;
            ws.reset();

            // create reusable vectors and matrices for Eigen operations
            auto       &accel_cmds = ws.accel_cmds;
            auto       &steering_rate_cmds = ws.steering_rate_cmds;
            auto       &dx_displacements = ws.dx_displacements;
            auto       &dy_displacements = ws.dy_displacements;
            auto       &heading_displacements = ws.heading_displacements;
            auto       &ego_xs_rear_axle_proposal = ws.ego_xs_rear_axle;
            auto       &ego_ys_rear_axle_proposal = ws.ego_ys_rear_axle;
            auto       &ego_thetas_proposal_cos = ws.ego_thetas_proposal_cos;
            auto       &ego_thetas_proposal_sin = ws.ego_thetas_proposal_sin;
            auto       &dx_matrix = ws.dx;
            auto       &dy_matrix = ws.dy;
            auto       &heading_displacements_matrix = ws.dHeading;
            auto       &theta_cos_matrix = ws.cosT;
            auto       &theta_sin_matrix = ws.sinT;
            auto       &A_head = ws.A_head;
            auto       &Aty_head = ws.rhs_head;
            auto       &ch_p_m_head = ws.chol_head_v2;
            auto       &velocity_profile_matrix = ws.velocity_profile;
            auto       &curvature_profile_matrix = ws.curvature_profile;
            auto       &reference_curvature_profiles = ws.reference_curvature_profiles;
            auto       &reference_velocities = ws.reference_velocities;
            auto       &acceleration_profile = ws.acceleration_profile;
            auto       &curvature_rate_profile = ws.curvature_rate_profile;
            auto       &A_lateral = ws.A_lateral;
            auto       &B_lateral = ws.B_lateral;
            auto       &g_lateral = ws.g_lateral;
            auto       &initial_state = ws.initial_state;
            auto       &state_error = ws.state_error;
            auto       &x_params_head = ws.x_params_head;
            auto       &ego_thetas_traj_cos = ws.ego_thetas_traj_cos;
            auto       &ego_thetas_traj_sin = ws.ego_thetas_traj_sin;
            auto       &ego_steering_angle_traj = ws.initial_steering_angle;
            auto       &ego_steering_rate_traj = ws.initial_steering_rate;
            const auto &input_matrix_lateral = ws.input_matrix_lateral;
            const auto &reference_state = ws.reference_state;
            const auto &Q = ws.r_m_q;
            const auto &q_vec = ws.q_diag;
            auto       &ldlt_solver = ws.ldlt_solver;

            // compute initial states
            ego_thetas_proposal[0].sincos(ego_thetas_proposal_sin[0], ego_thetas_proposal_cos[0]);
            ego_xs_rear_axle_proposal[0] =
                ego_xs_proposal[0] - vehicle_rear_axle_to_center * ego_thetas_proposal_cos[0];
            ego_ys_rear_axle_proposal[0] =
                ego_ys_proposal[0] - vehicle_rear_axle_to_center * ego_thetas_proposal_sin[0];

            // compute trajectorys' displacements
            for (size_t i = 1; i < expected_traj_size; ++i)
            {
                ego_thetas_proposal[i].sincos(ego_thetas_proposal_sin[i], ego_thetas_proposal_cos[i]);
                ego_xs_rear_axle_proposal[i] =
                    ego_xs_proposal[i] - vehicle_rear_axle_to_center * ego_thetas_proposal_cos[i];
                ego_ys_rear_axle_proposal[i] =
                    ego_ys_proposal[i] - vehicle_rear_axle_to_center * ego_thetas_proposal_sin[i];

                dx_displacements[i - 1] = ego_xs_rear_axle_proposal[i] - ego_xs_rear_axle_proposal[i - 1];
                dy_displacements[i - 1] = ego_ys_rear_axle_proposal[i] - ego_ys_rear_axle_proposal[i - 1];
                heading_displacements[i - 1] =
                    utils::NormalizeAngleSIMD<FVectorT_traj>(ego_thetas_proposal[i] - ego_thetas_proposal[i - 1]);

                // copy data to Eigen matrix using vamp::EigenFloatVectorMap for better performance
                dx_matrix.row(i - 1) =
                    vamp::EigenFloatVectorMap<FVectorT_traj::num_scalars>(dx_displacements[i - 1].to_array().data());
                dy_matrix.row(i - 1) =
                    vamp::EigenFloatVectorMap<FVectorT_traj::num_scalars>(dy_displacements[i - 1].to_array().data());
                heading_displacements_matrix.row(i - 1) = vamp::EigenFloatVectorMap<FVectorT_traj::num_scalars>(
                    heading_displacements[i - 1].to_array().data());
                theta_cos_matrix.row(i - 1) = vamp::EigenFloatVectorMap<FVectorT_traj::num_scalars>(
                    (ego_thetas_proposal_cos[i - 1]).to_array().data());
                theta_sin_matrix.row(i - 1) = vamp::EigenFloatVectorMap<FVectorT_traj::num_scalars>(
                    (ego_thetas_proposal_sin[i - 1]).to_array().data());
            }

            ws.solveVelocityBatch();
            auto &initial_curvature = ws.initial_curvature;

            // compute the velocity profiles
            velocity_profile_matrix.col(0) = ws.initial_velocity;
            acceleration_profile *= dt;
            for (int j = 0; j < num_displacements - 1; ++j)
            {
                velocity_profile_matrix.block(0, j + 1, batch_size, 1) =
                    velocity_profile_matrix.block(0, j, batch_size, 1) +
                    acceleration_profile.block(0, j, batch_size, 1);
            }

            // compute initial curvature + curvature rate least squares solution and extract results
            for (size_t b = 0; b < batch_size; ++b)
            {
                // reset A_eigen
                A_head.setZero();
                A_head.triangularView<Eigen::Lower>().setOnes();

                // column for k0
                A_head.col(0) = velocity_profile_matrix.row(b) * dt;

                // optimize the loop by using vectorized operations
                for (int i = 1; i < num_displacements; ++i)
                {
                    A_head.row(i).segment(1, i).array() *= velocity_profile_matrix(b, i) * dt_2;
                }

                // solve (A^T*A + Q) * x = A^T * y
                ch_p_m_head.setZero();
                ch_p_m_head.diagonal() = q_vec;
                const auto At_head = A_head.transpose();
                ch_p_m_head.selfadjointView<Eigen::Lower>().rankUpdate(At_head);
                Aty_head.noalias() = At_head * heading_displacements_matrix.col(b);
                ldlt_solver.compute(ch_p_m_head);
                x_params_head = ldlt_solver.solve(Aty_head);
                initial_curvature(b) = x_params_head(0);
                curvature_rate_profile.row(b) = x_params_head.tail(num_displacements - 1);
            }

            // compute the curvature profile
            curvature_profile_matrix.col(0) = initial_curvature;
            curvature_rate_profile *= dt;
            for (int j = 0; j < num_displacements - 1; ++j)
            {
                curvature_profile_matrix.block(0, j + 1, batch_size, 1) =
                    curvature_profile_matrix.block(0, j, batch_size, 1) +
                    curvature_rate_profile.block(0, j, batch_size, 1);
            }

            // initialize trajectory arrays
            ego_xs_traj.resize(expected_traj_size);
            ego_ys_traj.resize(expected_traj_size);
            ego_vs_traj.resize(expected_traj_size);
            ego_as_traj.resize(expected_traj_size);
            ego_thetas_traj.resize(expected_traj_size);
            ego_vs_traj[0] = FVectorT_traj::fill(ego_initial_v);
            ego_as_traj[0] = FVectorT_traj::fill(ego_current_a);
            ego_thetas_traj[0] = FVectorT_traj::fill(ego_initial_theta);
            const auto ego_thetas_traj_cos_initial = std::cos(ego_initial_theta);
            ego_thetas_traj_cos[0] = FVectorT_traj::fill(ego_thetas_traj_cos_initial);
            ego_xs_traj[0] =
                FVectorT_traj::fill(ego_initial_x - vehicle_rear_axle_to_center * ego_thetas_traj_cos_initial);
            const auto ego_thetas_traj_sin_initial = std::sin(ego_initial_theta);
            ego_thetas_traj_sin[0] = FVectorT_traj::fill(ego_thetas_traj_sin_initial);
            ego_ys_traj[0] =
                FVectorT_traj::fill(ego_initial_y - vehicle_rear_axle_to_center * ego_thetas_traj_sin_initial);
            ego_steering_angle_traj[0] = FVectorT_traj::fill(ego_initial_steering_angle);
            ego_steering_rate_traj[0] = FVectorT_traj::fill(ego_initial_steering_rate);

            // main tracking loop
            for (size_t t = 0; t < num_displacements; ++t)
            {
                // compute initial states errors
                const auto &initial_velocities = ego_vs_traj[t]; // initial velocity v_x
                const auto &lateral_errors =
                    -(ego_xs_traj[t] - ego_xs_rear_axle_proposal[t]) * ego_thetas_proposal_sin[t] +
                    (ego_ys_traj[t] - ego_ys_rear_axle_proposal[t]) * ego_thetas_proposal_cos[t]; // lateral error
                const auto &heading_errors_normalized = utils::NormalizeAngleSIMD<FVectorT_traj>(
                    ego_thetas_traj[t] - ego_thetas_proposal[t]); // heading error

                // index for reference vel (N steps ahead)
                const auto reference_idx = std::min((t + tracking_horizon), (num_displacements - 1));

                // calculate effective reference length
                const auto reference_length = reference_idx - t;

                // fill reference velocity [m/s] and curvature profile [rad] to track
                reference_velocities = utils::FVectorT_traj(velocity_profile_matrix.col(reference_idx).data());
                reference_curvature_profiles.block(0, 0, batch_size, reference_length) =
                    curvature_profile_matrix.block(0, t, batch_size, reference_length);
                if (reference_length < tracking_horizon)
                {
                    const size_t remaining = tracking_horizon - reference_length;

                    // get last column of curvature profile and copy it to the remaining columns
                    // Keep constant steering curvature at the end of the proposal horizon to prevent LQR lateral
                    // whiplash
                    reference_curvature_profiles.block(0, reference_length, batch_size, remaining) =
                        (curvature_profile_matrix.col(reference_idx)).replicate(1, remaining); // batch_size x remaining
                }

                // compute stop masks
                const auto should_stop_mask = ((reference_velocities.less_equal(stopping_velocity)) &
                                               (initial_velocities.less_equal(stopping_velocity)));
                const auto should_not_stop_mask = ~should_stop_mask;

                // 2. apply regular controller
                if (should_not_stop_mask.any())
                {
                    // 2.1 longitudinal control
                    accel_cmds = FVectorT_traj::select(
                        should_not_stop_mask, (-(initial_velocities - reference_velocities) * qrq_longitudinal),
                        accel_cmds);

                    // 2.2 lateral control
                    for (size_t b = 0; b < batch_size; ++b)
                    {
                        const auto idx = utils::div_mod<size_t>(b, FloatVectorWidth);
                        if (should_not_stop_mask[idx] != 0)
                        {
                            const auto velocity_profile = initial_velocities[idx] * ones_tracking_horizon_array +
                                                          accel_cmds[idx] * time_factor_array;

                            // reset A, B, and g
                            A_lateral.setIdentity();
                            B_lateral.setZero();
                            g_lateral.setZero();

                            for (Eigen::Index index_step = 0; index_step < tracking_horizon; ++index_step)
                            {
                                const float velocity = velocity_profile(index_step);
                                const float curvature = reference_curvature_profiles(b, index_step);

                                // Precompute the two non-zero elements of state_matrix_at_step
                                // state_matrix_at_step is identity matrix with only two non-zero off-diagonal elements:
                                const float a_01 = velocity * dt;
                                const float a_12 = velocity * discretization_time_over_wheelbase_length;

                                // construct affine term at this step
                                const float g_heading = -velocity * curvature * dt;

                                // Optimized update: Since state_matrix_at_step = I + sparse_matrix,
                                // we can directly update affected rows/columns instead of full matrix multiplication
                                // For A_lateral: A_new = (I + M) * A_old = A_old + M * A_old
                                {
                                    // Store the affected rows before modification
                                    const auto A_row_heading = A_lateral.row(heading_error_idx).eval();
                                    const auto A_row_steering = A_lateral.row(steering_angle_idx).eval();

                                    // Update lateral_error_idx row: add a_01 * heading_error_idx row
                                    A_lateral.row(lateral_error_idx).noalias() += a_01 * A_row_heading;

                                    // Update heading_error_idx row: add a_12 * steering_angle_idx row
                                    A_lateral.row(heading_error_idx).noalias() += a_12 * A_row_steering;
                                }

                                // For B_lateral: B_new = (I + M) * B_old + input = B_old + M * B_old + input
                                {
                                    const float B_heading = B_lateral(heading_error_idx);
                                    const float B_steering = B_lateral(steering_angle_idx);

                                    // Update lateral_error_idx element
                                    B_lateral(lateral_error_idx) += a_01 * B_heading;

                                    // Update heading_error_idx element
                                    B_lateral(heading_error_idx) += a_12 * B_steering;

                                    // Update steering_angle_idx element (add input_matrix_lateral value)
                                    B_lateral(steering_angle_idx) += input_matrix_lateral(steering_angle_idx);
                                }

                                // For g_lateral: g_new = (I + M) * g_old + affine = g_old + M * g_old + affine
                                {
                                    const float g_heading_old = g_lateral(heading_error_idx);

                                    // Update lateral_error_idx element
                                    g_lateral(lateral_error_idx) += a_01 * g_heading_old;

                                    // Update heading_error_idx element
                                    g_lateral(heading_error_idx) += a_12 * g_lateral(steering_angle_idx) + g_heading;
                                }
                            }

                            // solve the LQR problem at this step
                            initial_state << lateral_errors[idx], heading_errors_normalized[idx],
                                ego_steering_angle_traj[t][idx];
                            state_error = A_lateral * initial_state + g_lateral - reference_state;

                            // apply angle differences for specified indices
                            state_error(heading_error_idx) = utils::NormalizeAngle(state_error(heading_error_idx));
                            state_error(steering_angle_idx) = utils::NormalizeAngle(state_error(steering_angle_idx));

                            // scalar LQR solve (Q_lateral is diagonal, B is 3×1, result is scalar)
                            const float B0 = B_lateral(0), B1 = B_lateral(1), B2 = B_lateral(2);
                            const float numerator = B0 * q_lateral_1_ * state_error(0) +
                                                    B1 * q_lateral_2_ * state_error(1) +
                                                    B2 * q_lateral_3_ * state_error(2);
                            const float denominator =
                                B0 * B0 * q_lateral_1_ + B1 * B1 * q_lateral_2_ + B2 * B2 * q_lateral_3_ + r_lateral_;
                            steering_rate_cmds[b] = -numerator / denominator;
                        }
                        else
                        {
                            steering_rate_cmds[b] = 0.0f;
                        }
                    }
                }

                // 1. apply stopping controller
                if (should_stop_mask.any())
                {
                    accel_cmds = FVectorT_traj::select(
                        should_stop_mask, (-stopping_proportional_gain * (initial_velocities - reference_velocities)),
                        accel_cmds);
                }

                // update and store trajectory states
                auto ego_xs_traj_t = ego_xs_traj[t];
                auto ego_ys_traj_t = ego_ys_traj[t];
                auto ego_vs_traj_t = ego_vs_traj[t];
                auto ego_as_traj_t = ego_as_traj[t];
                auto ego_thetas_traj_t = ego_thetas_traj[t];
                auto ego_thetas_traj_t_cos = ego_thetas_traj_cos[t];
                auto ego_thetas_traj_t_sin = ego_thetas_traj_sin[t];
                auto ego_steering_angle_traj_t = ego_steering_angle_traj[t];
                auto ego_steering_rate_traj_t = ego_steering_rate_traj[t];

                // update vehicle position
                ego_xs_traj[t] = ego_xs_traj[t] + vehicle_rear_axle_to_center * ego_thetas_traj_t_cos;
                ego_ys_traj[t] = ego_ys_traj[t] + vehicle_rear_axle_to_center * ego_thetas_traj_t_sin;

                // clamp acceleration to physical limits if needed
                // accel_cmds.clamp(max_deceleration_, max_acceleration_);

                // updateStateBatchLQR<FVectorT_traj>(ego_xs_traj_t, ego_ys_traj_t, ego_vs_traj_t, ego_as_traj_t,
                // ego_thetas_traj_t, ego_thetas_traj_t_cos, ego_thetas_traj_t_sin, accel_cmds,
                // FVectorT_traj(steering_rate_cmds.data()));
                updateState<FVectorT_traj>(ego_xs_traj_t, ego_ys_traj_t, ego_vs_traj_t, ego_as_traj_t,
                                           ego_thetas_traj_t, ego_thetas_traj_t_cos, ego_thetas_traj_t_sin, accel_cmds,
                                           FVectorT_traj(steering_rate_cmds.data()), ego_steering_angle_traj_t,
                                           ego_steering_rate_traj_t);

                // store trajectory using direct assignment
                accel_cmds = FVectorT_traj::fill(0.0f);
                std::fill(steering_rate_cmds.begin(), steering_rate_cmds.end(), 0.0f);
                ego_xs_traj[t + 1] = ego_xs_traj_t;
                ego_ys_traj[t + 1] = ego_ys_traj_t;
                ego_vs_traj[t + 1] = ego_vs_traj_t;
                ego_as_traj[t + 1] = ego_as_traj_t;
                ego_thetas_traj[t + 1] = ego_thetas_traj_t;
                ego_thetas_traj_cos[t + 1] = ego_thetas_traj_t_cos;
                ego_thetas_traj_sin[t + 1] = ego_thetas_traj_t_sin;
                ego_steering_angle_traj[t + 1] = ego_steering_angle_traj_t;
                ego_steering_rate_traj[t + 1] = ego_steering_rate_traj_t;
            }

            // update vehicle position
            ego_xs_traj[num_displacements] =
                ego_xs_traj[num_displacements] + vehicle_rear_axle_to_center * ego_thetas_traj_cos[num_displacements];
            ego_ys_traj[num_displacements] =
                ego_ys_traj[num_displacements] + vehicle_rear_axle_to_center * ego_thetas_traj_sin[num_displacements];
        }

        __attribute__((optimize("O1"))) void Tracker::trackTrajectorySerial(
            const std::vector<AlignedVectorFloat> &ego_xs_proposal,
            const std::vector<AlignedVectorFloat> &ego_ys_proposal,
            const std::vector<AlignedVectorFloat> &ego_vs_proposal,
            const std::vector<AlignedVectorFloat> &ego_as_proposal,
            const std::vector<AlignedVectorFloat> &ego_thetas_proposal, std::vector<AlignedVectorFloat> &ego_xs_traj,
            std::vector<AlignedVectorFloat> &ego_ys_traj, std::vector<AlignedVectorFloat> &ego_vs_traj,
            std::vector<AlignedVectorFloat> &ego_as_traj, std::vector<AlignedVectorFloat> &ego_thetas_traj,
            const float &ego_initial_x, const float &ego_initial_y, const float &ego_initial_v,
            const float &ego_current_a, const float &ego_initial_theta, const float &ego_initial_steering_angle,
            const float &ego_initial_steering_rate)
        {
            // local variables
            constexpr size_t       expected_traj_size = utils::MAX_EXPECTED_TIME_STEPS;
            constexpr size_t       num_displacements = expected_traj_size - 1;
            constexpr float        vehicle_rear_axle_to_center = vehicle_rear_axle_to_center_; // 1.461f
            const size_t           batch_size = ego_xs_proposal[0].size();
            constexpr size_t       tracking_horizon = tracking_horizon_;
            constexpr float        dt = discretization_time_;
            constexpr float        dt_2 = dt_2_;
            constexpr float        stopping_velocity = stopping_velocity_;
            constexpr float        stopping_proportional_gain = stopping_proportional_gain_;
            constexpr float        qrq_longitudinal = qrq_longitudinal_;
            constexpr Eigen::Index lateral_error_idx = lateral_error_idx_;
            constexpr Eigen::Index heading_error_idx = heading_error_idx_;
            constexpr Eigen::Index steering_angle_idx = steering_angle_idx_;
            constexpr float      discretization_time_over_wheelbase_length = discretization_time_over_wheelbase_length_;
            const Eigen::ArrayXf ones_tracking_horizon_array = ones_tracking_horizon_array_;
            const Eigen::ArrayXf time_factor_array = time_factor_array_;
            const size_t         num_lateral_states = num_lateral_states_;

            // initialize output command vectors
            float accel_cmd = 0.0f;
            float steering_rate_cmd = 0.0f;

            // create reusable vectors and matrices for Eigen operations
            std::vector<AlignedVectorFloat> ego_xs_rear_axle_proposal(expected_traj_size,
                                                                      AlignedVectorFloat(batch_size));
            std::vector<AlignedVectorFloat> ego_ys_rear_axle_proposal(expected_traj_size,
                                                                      AlignedVectorFloat(batch_size));
            Eigen::MatrixXf                 dx_matrix(num_displacements, batch_size);
            Eigen::MatrixXf                 dy_matrix(num_displacements, batch_size);
            Eigen::MatrixXf                 heading_displacements_matrix(num_displacements, batch_size);
            Eigen::MatrixXf                 ego_thetas_proposal_cos(expected_traj_size, batch_size);
            Eigen::MatrixXf                 ego_thetas_proposal_sin(expected_traj_size, batch_size);
            Eigen::VectorXf                 Aty(num_displacements);
            Eigen::MatrixXf                 K(num_displacements, num_displacements);
            Eigen::MatrixXf A_lateral = Eigen::MatrixXf::Identity(num_lateral_states, num_lateral_states);
            Eigen::MatrixXf B_lateral = Eigen::MatrixXf::Zero(num_lateral_states, 1);
            Eigen::VectorXf g_lateral = Eigen::VectorXf::Zero(num_lateral_states);
            Eigen::MatrixXf state_matrix_at_step = Eigen::MatrixXf::Identity(num_lateral_states, num_lateral_states);
            Eigen::VectorXf affine_term = Eigen::VectorXf::Zero(num_lateral_states);
            Eigen::VectorXf initial_state(num_lateral_states);
            Eigen::VectorXf reference_state = reference_state_;
            Eigen::VectorXf state_error = Eigen::VectorXf::Zero(num_lateral_states);
            Eigen::MatrixXf Q_lateral_eigen = Q_lateral_;
            Eigen::MatrixXf R_lateral_eigen = R_lateral_;
            Eigen::MatrixXf A_heading = Eigen::MatrixXf::Zero(num_displacements, num_displacements);
            Eigen::VectorXf Aty_heading(num_displacements);
            Eigen::MatrixXf K_heading(num_displacements, num_displacements);
            Eigen::VectorXf x_params_heading(num_displacements);
            Eigen::VectorXf initial_curvature(batch_size);
            Eigen::MatrixXf velocity_profiles(batch_size, num_displacements);
            Eigen::MatrixXf curvature_rate_profile(batch_size, num_displacements - 1);
            Eigen::MatrixXf reference_curvature_profiles(batch_size, tracking_horizon);
            Eigen::VectorXf x_vars(num_displacements);
            Eigen::VectorXf initial_velocity(batch_size);
            Eigen::MatrixXf acceleration_profile(batch_size, num_displacements - 1);
            Eigen::MatrixXf curvature_profile(batch_size, num_displacements);
            Eigen::MatrixXf input_matrix_lateral = Eigen::MatrixXf::Zero(num_lateral_states, 1);
            input_matrix_lateral(steering_angle_idx, 0) = dt;

            // construct regularization matrix Q (size M x M)
            const Eigen::MatrixXf Q_matrix = [num_displacements]()
            {
                Eigen::MatrixXf Q =
                    Eigen::MatrixXf::Identity(num_displacements, num_displacements) * curvature_rate_penalty_;
                Q(0, 0) = initial_curvature_penalty_;
                return Q;
            }();

            // construct y vector for this batch item: [dx_0, dy_0, dx_1, dy_1, ..., dx_M-1, dy_M-1]
            Eigen::VectorXf y_eigen(2 * num_displacements);

            // construct design matrix A for this batch item (size 2M x M)
            Eigen::MatrixXf A_eigen = Eigen::MatrixXf::Zero(2 * num_displacements, num_displacements);

            // construct regularization matrix R (size approx. (M-2) x (M))
            const Eigen::MatrixXf RtR = [num_displacements]()
            {
                Eigen::MatrixXf matrix = Eigen::MatrixXf::Identity(num_displacements, num_displacements) * 1e-4f;
                matrix(0, 0) = 0.0f;
                matrix(num_displacements - 2, num_displacements - 1) = -1e-4f;
                matrix(num_displacements - 1, num_displacements - 2) = -1e-4f;
                return matrix;
            }();

            // compute initial states and trajectory displacements
            for (size_t b = 0; b < batch_size; ++b)
            {
                // initial state at t=0
                const float theta_0 = ego_thetas_proposal[0][b];
                const float cos_0 = std::cos(theta_0);
                const float sin_0 = std::sin(theta_0);
                ego_thetas_proposal_cos(0, b) = cos_0;
                ego_thetas_proposal_sin(0, b) = sin_0;
                const float rear_x_0 = ego_xs_proposal[0][b] - vehicle_rear_axle_to_center * cos_0;
                const float rear_y_0 = ego_ys_proposal[0][b] - vehicle_rear_axle_to_center * sin_0;
                ego_xs_rear_axle_proposal[0][b] = rear_x_0;
                ego_ys_rear_axle_proposal[0][b] = rear_y_0;

                // compute displacements for remaining steps
                float prev_rear_x = rear_x_0;
                float prev_rear_y = rear_y_0;
                float prev_theta = theta_0;

                for (size_t i = 1; i < expected_traj_size; ++i)
                {
                    const float theta_i = ego_thetas_proposal[i][b];
                    const float cos_i = std::cos(theta_i);
                    const float sin_i = std::sin(theta_i);
                    ego_thetas_proposal_cos(i, b) = cos_i;
                    ego_thetas_proposal_sin(i, b) = sin_i;

                    const float rear_x_i = ego_xs_proposal[i][b] - vehicle_rear_axle_to_center * cos_i;
                    const float rear_y_i = ego_ys_proposal[i][b] - vehicle_rear_axle_to_center * sin_i;
                    ego_xs_rear_axle_proposal[i][b] = rear_x_i;
                    ego_ys_rear_axle_proposal[i][b] = rear_y_i;

                    // compute displacements
                    const size_t disp_idx = i - 1;
                    dx_matrix(disp_idx, b) = rear_x_i - prev_rear_x;
                    dy_matrix(disp_idx, b) = rear_y_i - prev_rear_y;
                    heading_displacements_matrix(disp_idx, b) = utils::NormalizeAngle(theta_i - prev_theta);

                    prev_rear_x = rear_x_i;
                    prev_rear_y = rear_y_i;
                    prev_theta = theta_i;
                }
            }

            // compute initial velocity + acceleration least squares solution and extract results
            for (size_t b = 0; b < batch_size; ++b)
            {
                // reset A_eigen once at the start of each batch
                A_eigen.setZero();

                for (size_t i = 0; i < num_displacements; ++i)
                {
                    // fill y vector with dx and dy displacement components
                    y_eigen[2 * i] = dx_matrix(i, b);
                    y_eigen[2 * i + 1] = dy_matrix(i, b);

                    // fill the cosine value of the heading angle to the x-component position of all displacements (even
                    // index) fill the sine value of the heading angle to the y-component position of all displacements
                    // (odd index)
                    const float theta_cos = ego_thetas_proposal_cos(i, b);
                    const float theta_sin = ego_thetas_proposal_sin(i, b);

                    A_eigen(2 * i, 0) = theta_cos * dt;
                    A_eigen(2 * i + 1, 0) = theta_sin * dt;

                    // acceleration column for a0, ..., ai
                    if (i > 0)
                    {
                        const float     cos_val = theta_cos * dt_2;
                        const float     sin_val = theta_sin * dt_2;
                        Eigen::Vector2f base_vector(cos_val, sin_val);
                        A_eigen.block<2, Eigen::Dynamic>(2 * i, 1, 2, i).colwise() = base_vector;
                    }
                }

                // solve (A^T*A + jerk_penalty * R^T*R) * x = A^T * y
                const auto At = A_eigen.transpose();
                Aty.noalias() = At * y_eigen;
                K.noalias() = At * A_eigen + RtR;

                // solve K * x = Aty
                auto chol = K.ldlt();
                if (chol.info() == Eigen::Success)
                {
                    x_vars = chol.solve(Aty);
                }
                else
                {
                    x_vars = K.colPivHouseholderQr().solve(Aty);
                }

                initial_velocity(b) = x_vars(0);
                acceleration_profile.row(b) = x_vars.tail(num_displacements - 1);
            }

            // compute the velocity profiles
            velocity_profiles.col(0) = initial_velocity;
            const auto scaled_derivs_acceleration = acceleration_profile * dt;
            for (size_t j = 0; j < num_displacements - 1; ++j)
            {
                velocity_profiles.col(j + 1) = velocity_profiles.col(j) + scaled_derivs_acceleration.col(j);
            }

            // compute initial curvature + curvature rate least squares solution and extract results
            for (size_t b = 0; b < batch_size; ++b)
            {
                // reset A_heading once at the start
                A_heading.setZero();
                A_heading.triangularView<Eigen::Lower>().setOnes();

                // column for k0
                A_heading.col(0) = velocity_profiles.row(b) * dt;

                // optimize the loop by using vectorized operations
                for (int i = 1; i < num_displacements; ++i)
                {
                    A_heading.row(i).segment(1, i).array() *= velocity_profiles(b, i) * dt_2;
                }

                // solve (A^T*A + Q) * x = A^T * y
                const auto At_heading = A_heading.transpose();
                Aty_heading.noalias() = At_heading * heading_displacements_matrix.col(b);
                K_heading.noalias() = At_heading * A_heading + Q_matrix;
                auto chol = K_heading.ldlt();

                // choose the solver
                if (chol.info() == Eigen::Success)
                {
                    x_params_heading = chol.solve(Aty_heading);
                }
                else
                {
                    x_params_heading = K_heading.colPivHouseholderQr().solve(Aty_heading);
                }

                initial_curvature(b) = x_params_heading(0);
                curvature_rate_profile.row(b) = x_params_heading.tail(num_displacements - 1);
            }

            // compute the curvature profile
            curvature_profile.col(0) = initial_curvature;
            const auto scaled_derivs_curvature = curvature_rate_profile * dt;
            for (size_t j = 0; j < num_displacements - 1; ++j)
            {
                curvature_profile.col(j + 1) = curvature_profile.col(j) + scaled_derivs_curvature.col(j);
            }

            // Initialize trajectory arrays
            std::vector<AlignedVectorFloat> ego_thetas_traj_cos(expected_traj_size, AlignedVectorFloat(batch_size));
            std::vector<AlignedVectorFloat> ego_thetas_traj_sin(expected_traj_size, AlignedVectorFloat(batch_size));
            std::vector<AlignedVectorFloat> ego_steering_angle_traj(expected_traj_size, AlignedVectorFloat(batch_size));
            std::vector<AlignedVectorFloat> ego_steering_rate_traj(expected_traj_size, AlignedVectorFloat(batch_size));
            ego_xs_traj.resize(expected_traj_size, AlignedVectorFloat(batch_size));
            ego_ys_traj.resize(expected_traj_size, AlignedVectorFloat(batch_size));
            ego_vs_traj.resize(expected_traj_size, AlignedVectorFloat(batch_size));
            ego_as_traj.resize(expected_traj_size, AlignedVectorFloat(batch_size));
            ego_thetas_traj.resize(expected_traj_size, AlignedVectorFloat(batch_size));

            // compute initial theta cos/sin once (same for all batches)
            const float ego_thetas_traj_cos_initial = std::cos(ego_initial_theta);
            const float ego_thetas_traj_sin_initial = std::sin(ego_initial_theta);
            const float ego_xs_traj_initial = ego_initial_x - vehicle_rear_axle_to_center * ego_thetas_traj_cos_initial;
            const float ego_ys_traj_initial = ego_initial_y - vehicle_rear_axle_to_center * ego_thetas_traj_sin_initial;

            // initialize trajectory arrays for all batches
            for (size_t b = 0; b < batch_size; ++b)
            {
                ego_vs_traj[0][b] = ego_initial_v;
                ego_as_traj[0][b] = ego_current_a;
                ego_thetas_traj[0][b] = ego_initial_theta;
                ego_thetas_traj_cos[0][b] = ego_thetas_traj_cos_initial;
                ego_thetas_traj_sin[0][b] = ego_thetas_traj_sin_initial;
                ego_steering_angle_traj[0][b] = ego_initial_steering_angle;
                ego_steering_rate_traj[0][b] = ego_initial_steering_rate;
                ego_xs_traj[0][b] = ego_xs_traj_initial;
                ego_ys_traj[0][b] = ego_ys_traj_initial;
            }

            // main tracking loop
            for (size_t t = 0; t < num_displacements; ++t)
            {
                for (size_t b = 0; b < batch_size; ++b)
                {
                    // compute initial states errors
                    const auto &initial_velocity = ego_vs_traj[t][b];
                    const auto &lateral_error =
                        -(ego_xs_traj[t][b] - ego_xs_rear_axle_proposal[t][b]) * ego_thetas_proposal_sin(t, b) +
                        (ego_ys_traj[t][b] - ego_ys_rear_axle_proposal[t][b]) * ego_thetas_proposal_cos(t, b);
                    const auto &heading_error_normalized =
                        utils::NormalizeAngle(ego_thetas_traj[t][b] - ego_thetas_proposal[t][b]);

                    // index for reference vel (N steps ahead)
                    const auto reference_idx = std::min((t + tracking_horizon), (num_displacements - 1));

                    // calculate effective reference length
                    const auto reference_length = reference_idx - t;

                    // fill reference velocity [m/s] and curvature profile [rad] to track
                    const auto &reference_velocities = velocity_profiles.col(reference_idx);
                    reference_curvature_profiles.block(0, 0, batch_size, reference_length) =
                        curvature_profile.block(0, t, batch_size, reference_length);
                    if (reference_length < tracking_horizon)
                    {
                        const size_t remaining = tracking_horizon - reference_length;

                        // get last column of curvature profile and copy it to the remaining columns
                        reference_curvature_profiles.block(0, reference_length, batch_size, remaining) =
                            (curvature_profile.col(reference_idx)).replicate(1, remaining);
                    }

                    if (reference_velocities(b) <= stopping_velocity && initial_velocity <= stopping_velocity)
                    {
                        // 1. apply stopping controller
                        accel_cmd = -stopping_proportional_gain * (initial_velocity - reference_velocities(b));
                        steering_rate_cmd = 0.0f;
                    }
                    else
                    {
                        // 2. apply regular controller
                        // 2.1 longitudinal control
                        accel_cmd = -(initial_velocity - reference_velocities(b)) * qrq_longitudinal;
                        const auto velocity_profile =
                            initial_velocity * ones_tracking_horizon_array + accel_cmd * time_factor_array;

                        // reset A, B, and g (only reset off-diagonal elements that change)
                        A_lateral.setIdentity();
                        B_lateral.setZero();
                        g_lateral.setZero();

                        for (Eigen::Index index_step = 0; index_step < tracking_horizon; ++index_step)
                        {
                            const float velocity = velocity_profile(index_step);
                            const float curvature = reference_curvature_profiles(b, index_step);
                            const float a_01 = velocity * dt; // lateral_error_idx -> heading_error_idx
                            const float a_12 =
                                velocity *
                                discretization_time_over_wheelbase_length; // heading_error_idx -> steering_angle_idx

                            // construct affine term at this step
                            const float g_heading = -velocity * curvature * dt;

                            // state_matrix_at_step = I + sparse_matrix,
                            // For A_lateral: A_new = (I + M) * A_old = A_old + M * A_old
                            {
                                const auto A_row_heading = A_lateral.row(heading_error_idx);
                                const auto A_row_steering = A_lateral.row(steering_angle_idx);

                                // Update lateral_error_idx row: add a_01 * heading_error_idx row
                                A_lateral.row(lateral_error_idx).noalias() += a_01 * A_row_heading;

                                // Update heading_error_idx row: add a_12 * steering_angle_idx row
                                A_lateral.row(heading_error_idx).noalias() += a_12 * A_row_steering;
                            }

                            // For B_lateral: B_new = (I + M) * B_old + input = B_old + M * B_old + input
                            {
                                const float B_heading = B_lateral(heading_error_idx);
                                const float B_steering = B_lateral(steering_angle_idx);

                                // Update lateral_error_idx element
                                B_lateral(lateral_error_idx) += a_01 * B_heading;

                                // Update heading_error_idx element
                                B_lateral(heading_error_idx) += a_12 * B_steering;

                                // Update steering_angle_idx element (add input_matrix_lateral value)
                                B_lateral(steering_angle_idx) += input_matrix_lateral(steering_angle_idx);
                            }

                            // For g_lateral: g_new = (I + M) * g_old + affine = g_old + M * g_old + affine
                            {
                                const float g_heading_old = g_lateral(heading_error_idx);

                                // Update lateral_error_idx element
                                g_lateral(lateral_error_idx) += a_01 * g_heading_old;

                                // Update heading_error_idx element
                                g_lateral(heading_error_idx) += a_12 * g_lateral(steering_angle_idx) + g_heading;
                            }
                        }

                        // solve the LQR problem at this step
                        initial_state << lateral_error, heading_error_normalized, ego_steering_angle_traj[t][b];
                        state_error = A_lateral * initial_state + g_lateral - reference_state;

                        // apply angle differences for specified indices
                        state_error(heading_error_idx) = utils::NormalizeAngle(state_error(heading_error_idx));
                        state_error(steering_angle_idx) = utils::NormalizeAngle(state_error(steering_angle_idx));

                        const auto BtQ = B_lateral.transpose() * Q_lateral_eigen;

                        // calculate the term Inv = (B^T * Q * B + R) (1 x 1)
                        steering_rate_cmd =
                            -1.0f * ((BtQ * state_error)(0, 0)) / ((BtQ * B_lateral + R_lateral_eigen)(0, 0));
                    }

                    // update the ego state
                    auto ego_xs_traj_t = ego_xs_traj[t][b];
                    auto ego_ys_traj_t = ego_ys_traj[t][b];
                    auto ego_vs_traj_t = ego_vs_traj[t][b];
                    auto ego_as_traj_t = ego_as_traj[t][b];
                    auto ego_theta_traj_t = ego_thetas_traj[t][b];
                    auto ego_theta_traj_t_cos = ego_thetas_traj_cos[t][b];
                    auto ego_theta_traj_t_sin = ego_thetas_traj_sin[t][b];
                    auto ego_steering_angle_traj_t = ego_steering_angle_traj[t][b];
                    auto ego_steering_rate_traj_t = ego_steering_rate_traj[t][b];

                    // update vehicle position
                    ego_xs_traj[t][b] += vehicle_rear_axle_to_center * ego_theta_traj_t_cos;
                    ego_ys_traj[t][b] += vehicle_rear_axle_to_center * ego_theta_traj_t_sin;
                    updateStateSerial<float>(ego_xs_traj_t, ego_ys_traj_t, ego_vs_traj_t, ego_as_traj_t,
                                             ego_theta_traj_t, ego_theta_traj_t_cos, ego_theta_traj_t_sin, accel_cmd,
                                             steering_rate_cmd, ego_steering_angle_traj_t, ego_steering_rate_traj_t);

                    // store trajectory using direct assignment
                    ego_xs_traj[t + 1][b] = ego_xs_traj_t;
                    ego_ys_traj[t + 1][b] = ego_ys_traj_t;
                    ego_vs_traj[t + 1][b] = ego_vs_traj_t;
                    ego_as_traj[t + 1][b] = ego_as_traj_t;
                    ego_thetas_traj[t + 1][b] = ego_theta_traj_t;
                    ego_thetas_traj_cos[t + 1][b] = ego_theta_traj_t_cos;
                    ego_thetas_traj_sin[t + 1][b] = ego_theta_traj_t_sin;
                    ego_steering_angle_traj[t + 1][b] = ego_steering_angle_traj_t;
                    ego_steering_rate_traj[t + 1][b] = ego_steering_rate_traj_t;

                    // update vehicle position at the last step
                    if (t == num_displacements - 1)
                    {
                        ego_xs_traj[t + 1][b] += vehicle_rear_axle_to_center * ego_theta_traj_t_cos;
                        ego_ys_traj[t + 1][b] += vehicle_rear_axle_to_center * ego_theta_traj_t_sin;
                    }

                    // Reset commands for next iteration (will be set in next loop iteration)
                    accel_cmd = 0.0f;
                    steering_rate_cmd = 0.0f;
                }
            }
        }

        template <typename T> void Tracker::printFvectorT(const std::string &name, const T &vec)
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(5);
            ss << name << " [";
            auto arr = vec.to_array();
            for (size_t i = 0; i < T::num_scalars; ++i)
            {
                ss << std::setw(5) << arr[i] << " ";
            }
            ss << "]";
            std::cout << ss.str() << std::endl;
        }

        template void Tracker::updateState<Tracker::FVectorT_traj>(
            Tracker::FVectorT_traj &ego_xs, Tracker::FVectorT_traj &ego_ys, Tracker::FVectorT_traj &ego_vs,
            Tracker::FVectorT_traj &ego_as, Tracker::FVectorT_traj &ego_thetas, Tracker::FVectorT_traj &ego_thetas_cos,
            Tracker::FVectorT_traj &ego_thetas_sin, const Tracker::FVectorT_traj &cal_accs,
            const Tracker::FVectorT_traj &cal_steerings, Tracker::FVectorT_traj &steering_angle,
            Tracker::FVectorT_traj &steering_rate);

        template void Tracker::printFvectorT<Tracker::FVectorT_traj>(const std::string            &name,
                                                                     const Tracker::FVectorT_traj &vec);

        // helper function for Eigen matrix info
        void Tracker::printMatrixInfo(const std::string &name, const Eigen::MatrixXf &matrix)
        {
            std::cout << name << " dimensions: [" << matrix.rows() << " x " << matrix.cols() << "]" << std::endl;

            // print first few elements if matrix is not empty
            if (matrix.size() > 0)
            {
                std::cout << name << " first element: " << matrix(0, 0) << std::endl;

                // print top-left 2x2 corner if matrix is large enough
                if (matrix.rows() >= 2 && matrix.cols() >= 2)
                {
                    std::cout << name << " top-left 2x2:\n" << matrix.block(0, 0, 2, 2) << std::endl;
                }
            }
        }

        // helper function for Eigen vector info
        void Tracker::printVectorInfo(const std::string &name, const Eigen::VectorXf &vector)
        {
            std::cout << name << " size: " << vector.size() << std::endl;

            // print first few elements if vector is not empty
            if (vector.size() > 0)
            {
                std::cout << name << " first element: " << vector(0) << std::endl;

                // print first 3 elements if vector is large enough
                if (vector.size() >= 3)
                {
                    std::cout << name << " first 3 elements: " << vector.head(3).transpose() << std::endl;
                }
            }
        }
    } // namespace planning
} // namespace vec_qmdp