/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file geometry_utils.hpp
 * @brief Basic geometry primitives (Point, Polygon) and coordinate transformations.
 */

#pragma once

#include <memory>
#include <vector>

namespace vec_qmdp
{
    namespace utils
    {
        struct Point
        {
            float x, y;

            Point() : x(0), y(0) {}

            Point(float x_, float y_) : x(x_), y(y_) {}
        };

        struct Polygon
        {
            std::vector<Point> vertices;
        };

    } // namespace utils
} // namespace vec_qmdp