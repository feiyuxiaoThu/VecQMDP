/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <core/state.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace vec_qmdp
{
    namespace core
    {

        EgoState::EgoState() {}

        EgoState::~EgoState() {}

        void EgoState::UpdateEgoState(const py::object &ego_info)
        {
            x_ = py::extract<float>(ego_info.attr("X"));
            y_ = py::extract<float>(ego_info.attr("Y"));
            v_ = py::extract<float>(ego_info.attr("velocityX"));
            a_ = py::extract<float>(ego_info.attr("accX"));
            theta_ = py::extract<float>(ego_info.attr("heading"));
            steering_angle_ = py::extract<float>(ego_info.attr("steering"));
            steering_rate_ = py::extract<float>(ego_info.attr("steering_rate"));
            timestamp_ = py::extract<uint64_t>(ego_info.attr("timestamp"));
        }

        bool EgoState::IsCloseToJunction(const std::shared_ptr<utils::MapUtils> &map_utils_ptr) const
        {
            if (v_ >= utils::JUNCTION_PROXIMITY_MAX_SPEED || edge_name_ == "LANE_CONNECTOR" || edge_token_ == "")
            {
                return false;
            }

            // check if the car is on a straight lane
            const auto &curr_path = map_utils_ptr->GetRefLine(edge_token_, "LANE");
            if (!curr_path->IsStraight() ||
                map_utils_ptr->DistanceToEndOfEdge(x_, y_, edge_token_) > utils::JUNCTION_EDGE_END_DISTANCE_THRESHOLD)
            {
                return false;
            }

            // check if the next edge is a long lane connector
            std::string successor_id = "";
            const auto &successors = map_utils_ptr->GetRouteSuccessorIdsById(edge_token_);
            if (!successors.empty())
            {
                successor_id = successors[0];
                const auto &next_path = map_utils_ptr->GetRefLine(successor_id, "LANE_CONNECTOR");
                if (next_path->GetPathLen() < utils::JUNCTION_MIN_CONNECTOR_LENGTH)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            return true;
        }

        ObservedExoState::ObservedExoState() { reset(); }

        void ObservedExoState::reset()
        {
            num_vehicles_ = 0;
            agent_vehicle_mask_.resize(utils::MAX_OBS_VEHICLES, false);
            agent_ped_mask_.resize(utils::MAX_OBS_VEHICLES, false);

            obs_xs_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_ys_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_vs_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_thetas_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_original_bb_extent_xs_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_original_bb_extent_ys_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_expanded_bb_extent_xs_.resize(utils::MAX_OBS_VEHICLES, 0.0f);
            obs_expanded_bb_extent_ys_.resize(utils::MAX_OBS_VEHICLES, 0.0f);

            obs_nearest_idxs_.resize(utils::MAX_NUM_REFLINES, AlignedVectorInt(utils::MAX_OBS_VEHICLES, 0.0f));
            obs_frenet_s_.resize(utils::MAX_NUM_REFLINES, AlignedVectorFloat(utils::MAX_OBS_VEHICLES, 0.0f));
            obs_frenet_l_.resize(utils::MAX_NUM_REFLINES, AlignedVectorFloat(utils::MAX_OBS_VEHICLES, 0.0f));

            agent_valid_mask_.resize(utils::MAX_OBS_VEHICLES, false);
            agent_dynamic_mask_.resize(utils::MAX_OBS_VEHICLES, false);

            valid_obs_original_bb_extent_xs_.resize(utils::MAX_SIM_VEHICLES, 0.0f);
            valid_obs_original_bb_extent_ys_.resize(utils::MAX_SIM_VEHICLES, 0.0f);
            valid_obs_expanded_bb_extent_xs_.resize(utils::MAX_SIM_VEHICLES, 0.0f);
            valid_obs_expanded_bb_extent_ys_.resize(utils::MAX_SIM_VEHICLES, 0.0f);
        }

        ObservedExoState::~ObservedExoState() {}

        void ObservedExoState::updateObservedExoState(const py::dict                           &agents_info,
                                                      const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                                      const std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths)
        {
            reset();

            // Iterate over all agents
            py::list keys = agents_info.keys();

            tokens_.clear();
            agent_types_.clear();
            for (size_t i = 0; i < len(keys); ++i)
            {
                if (num_vehicles_ >= utils::MAX_OBS_VEHICLES)
                    break;

                py::str           key = py::extract<py::str>(keys[i]);
                const py::object &obj = agents_info[key];

                // For debug
                tokens_.emplace_back(py::extract<std::string>(obj.attr("token")));
                // std::cout << "Update C++ agent: " << tokens_.back() << std::endl;
                agent_types_.emplace_back(py::extract<int>(obj.attr("type")));

                agent_vehicle_mask_[num_vehicles_] = py::extract<int>(obj.attr("type")) == 0;
                agent_ped_mask_[num_vehicles_] = py::extract<int>(obj.attr("type")) == 1;
                obs_xs_[num_vehicles_] = py::extract<float>(obj.attr("X"));
                obs_ys_[num_vehicles_] = py::extract<float>(obj.attr("Y"));
                obs_vs_[num_vehicles_] = py::extract<float>(obj.attr("speed"));
                obs_thetas_[num_vehicles_] = py::extract<float>(obj.attr("heading"));
                obs_original_bb_extent_xs_[num_vehicles_] = py::extract<float>(obj.attr("extentX"));
                obs_original_bb_extent_ys_[num_vehicles_] = py::extract<float>(obj.attr("extentY"));
                obs_expanded_bb_extent_xs_[num_vehicles_] = py::extract<float>(obj.attr("extentX_expanded"));
                obs_expanded_bb_extent_ys_[num_vehicles_] = py::extract<float>(obj.attr("extentY_expanded"));
                agent_dynamic_mask_[num_vehicles_] = py::extract<bool>(obj.attr("dynamic"));

                ++num_vehicles_;
            }

            LOG_IS << "num_vehicles_: " << num_vehicles_;

            // Batch compute Frenet coordinates of exo vehicles on reference paths at the current timestep
            for (size_t i = 0; i < ego_ref_paths.size(); ++i)
            {
                const auto &path = ego_ref_paths[i];
                if (path)
                {
                    GetFrenetPointsBatch(path, i);
                }
            }

            for (size_t i = 0; i < ego_extra_ref_paths.size(); ++i)
            {
                const auto &path = ego_extra_ref_paths[i];
                if (path)
                {
                    GetFrenetPointsBatch(path, i + ego_ref_paths.size());
                }
            }
        }

        void ObservedExoState::updateValidExoState()
        {
            valid_agent_idxs_.clear();
            size_t           valid_num = 0;
            size_t           max_obs_num = std::min(static_cast<size_t>(utils::MAX_OBS_VEHICLES), num_vehicles_);
            size_t           notice_vehicle_idx = 0;
            std::vector<int> tmp_noticed_agent_idxs;
            for (size_t i = 0, notice_vehicle_idx = 0; i < max_obs_num && valid_num < utils::MAX_SIM_VEHICLES; ++i)
            {
                bool agent_valid = false;
                if (notice_vehicle_idx < utils::NOTICE_VEHICLE_IDXS.size() &&
                    i == utils::NOTICE_VEHICLE_IDXS[notice_vehicle_idx])
                {
                    ++notice_vehicle_idx;
                    agent_valid = true;
                    tmp_noticed_agent_idxs.emplace_back(valid_num);
                }

                if (agent_valid || agent_valid_mask_[i])
                {
                    if (agent_ped_mask_[i] && obs_vs_[i] > utils::PEDESTRIAN_EXPANDED_BB_SPEED_THRESHOLD)
                    {
                        valid_obs_original_bb_extent_xs_[valid_num] = obs_expanded_bb_extent_xs_[i];
                        valid_obs_original_bb_extent_ys_[valid_num] = obs_expanded_bb_extent_ys_[i];
                    }
                    else
                    {
                        valid_obs_original_bb_extent_xs_[valid_num] = obs_original_bb_extent_xs_[i];
                        valid_obs_original_bb_extent_ys_[valid_num] = obs_original_bb_extent_ys_[i];
                    }
                    valid_obs_expanded_bb_extent_xs_[valid_num] = obs_expanded_bb_extent_xs_[i];
                    valid_obs_expanded_bb_extent_ys_[valid_num] = obs_expanded_bb_extent_ys_[i];
                    valid_agent_idxs_.emplace_back(i);
                    ++valid_num;
                }
            }
            num_vehicles_ = valid_num;
            utils::NOTICE_VEHICLE_IDXS = tmp_noticed_agent_idxs;
        }

        void ObservedExoState::GetFrenetPointsBatch(const std::shared_ptr<Path> &path, std::size_t path_idx)
        {
            int batch_size = FloatVectorWidth; // SIMD vector width

            for (size_t v_batch = 0; v_batch < num_vehicles_; v_batch += batch_size)
            {
                // Determine number of valid vehicles in the current batch
                int valid_num = std::min(batch_size, static_cast<int>(num_vehicles_ - v_batch));

                // Step 1: process nearest points at t=0 for the current batch
                FVectorT_1 xs_t0 = FVectorT_1::load_contiguous(obs_xs_.data(), v_batch);
                FVectorT_1 ys_t0 = FVectorT_1::load_contiguous(obs_ys_.data(), v_batch);

                // Batch lookup of nearest points at t=0
                AlignedVectorInt nearest_idxs_vec = path->NearestBatch(xs_t0, ys_t0, valid_num);
                IVectorT_1       nearest_idxs = IVectorT_1(nearest_idxs_vec.data());

                // Get path point coordinates and angles
                FVectorT_1 path_xs = FVectorT_1::gather(path->GetXs().data(), nearest_idxs);
                FVectorT_1 path_ys = FVectorT_1::gather(path->GetYs().data(), nearest_idxs);

                // Calculate vector between vehicle position and path point
                FVectorT_1 dx = xs_t0 - path_xs;
                FVectorT_1 dy = ys_t0 - path_ys;

                // Calculate path point normal vector
                FVectorT_1 thetas = FVectorT_1::gather(path->thetas_.data(), nearest_idxs);
                FVectorT_1 nx = -thetas.sin(); // cos(theta + pi/2)
                FVectorT_1 ny = thetas.cos();  // sin(theta + pi/2)

                // Calculate lateral offset (l) and longitudinal position (s)
                FVectorT_1 frenet_ss = path->GetFrenetS(nearest_idxs);
                FVectorT_1 frenet_ls = (dx * nx + dy * ny).sign() * (dx * dx + dy * dy).sqrt();

                // Store results
                for (int i = 0; i < valid_num; ++i)
                {
                    uint32_t idx = v_batch + i;
                    auto     pair_idx = std::make_pair(0, i);
                    obs_nearest_idxs_[path_idx][idx] = nearest_idxs[pair_idx];
                    obs_frenet_s_[path_idx][idx] = frenet_ss[pair_idx];
                    obs_frenet_l_[path_idx][idx] = frenet_ls[pair_idx];
                }
            }
        }
    } // namespace core
} // namespace vec_qmdp