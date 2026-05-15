/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include "utils/params.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace vec_qmdp
{
    namespace utils
    {
        // DEBUG
        int         iteration = 0;
        std::string scenario_token = "";

        float offset_x = 0.0f;
        float offset_y = 0.0f;
        bool  approaching_terminal_point = false;

        float PATH_SIZE = 100.0f;

        std::vector<int> NOTICE_VEHICLE_IDXS;

        float max_q_value = 0.0f;
        float weighted_q_value = 0.0f;
    } // namespace utils
} // namespace vec_qmdp