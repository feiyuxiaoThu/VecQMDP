/**********************************************************************
 *
 * GEOS - Geometry Engine Open Source
 * http://geos.osgeo.org
 *
 * Copyright (C) 2001-2002 Vivid Solutions Inc.
 *
 * This is free software; you can redistribute and/or modify it under
 * the terms of the GNU Lesser General Public Licence as published
 * by the Free Software Foundation.
 * See the COPYING file for more information.
 *
 **********************************************************************
 *
 * Last port: geom/prep/PreparedPolygon.java rev 1.7 (JTS-1.10)
 *
 **********************************************************************/

#include <geos/algorithm/locate/IndexedPointInAreaLocator.h>
#include <geos/algorithm/locate/PointOnGeometryLocator.h>
#include <geos/algorithm/locate/SimplePointInAreaLocator.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateSequence.h>
#include <geos/geom/LineString.h>
#include <geos/geom/Polygon.h>
#include <geos/geom/prep/PreparedPolygon.h>
#include <geos/geom/prep/PreparedPolygonContains.h>
#include <geos/geom/prep/PreparedPolygonContainsProperly.h>
#include <geos/geom/prep/PreparedPolygonCovers.h>
#include <geos/geom/prep/PreparedPolygonDistance.h>
#include <geos/geom/prep/PreparedPolygonIntersects.h>
#include <geos/geom/prep/PreparedPolygonPredicate.h>
#include <geos/geom/util/ComponentCoordinateExtracter.h>
#include <geos/geom/util/LinearComponentExtracter.h>
#include <geos/noding/FastSegmentSetIntersectionFinder.h>
#include <geos/noding/SegmentStringUtil.h>
#include <geos/operation/predicate/RectangleContains.h>
#include <geos/operation/predicate/RectangleIntersects.h>
#include <geos/util.h>

// SIMD includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vamp/vector.hh>
#include <vector>

namespace geos
{
namespace geom
{
namespace prep
{
using SegmentView = algorithm::locate::IndexedPointInAreaLocator::SegmentView;

PreparedPolygon::PreparedPolygon(const geom::Geometry *geom)
    : BasicPreparedGeometry(geom)
{
    isRectangle = getGeometry().isRectangle();
    indexedPtInAreaLocator = getIndexedPointInAreaLocator();
    indexedPtInAreaLocator->updateIndex();
}

PreparedPolygon::~PreparedPolygon()
{
    for (std::size_t i = 0, ni = segStrings.size(); i < ni; i++)
    {
        delete segStrings[i];
    }
}

noding::FastSegmentSetIntersectionFinder *
PreparedPolygon::getIntersectionFinder() const
{
    if (!segIntFinder)
    {
        noding::SegmentStringUtil::extractSegmentStrings(&getGeometry(),
                                                         segStrings);
        segIntFinder.reset(
            new noding::FastSegmentSetIntersectionFinder(&segStrings));
    }
    return segIntFinder.get();
}

algorithm::locate::PointOnGeometryLocator *
PreparedPolygon::getPointLocator() const
{
    // If we are only going to locate a single point, it's faster to do a
    // brute-force SimplePointInAreaLocator instead of an
    // IndexedPointInAreaLocator. There's a reasonable chance we will only use
    // this locator once (for example, if we get here through
    // Geometry::intersects). So we create a simple locator for the first usage
    // and switch to an indexed locator when it is clear we're in a multiple-use
    // scenario.
    if (!ptOnGeomLoc)
    {
        ptOnGeomLoc =
            detail::make_unique<algorithm::locate::SimplePointInAreaLocator>(
                &getGeometry());
        return ptOnGeomLoc.get();
    }
    else if (!indexedPtOnGeomLoc)
    {
        indexedPtOnGeomLoc =
            detail::make_unique<algorithm::locate::IndexedPointInAreaLocator>(
                getGeometry());
    }

    return indexedPtOnGeomLoc.get();
}

algorithm::locate::IndexedPointInAreaLocator *
PreparedPolygon::getIndexedPointInAreaLocator()
{
    if (!indexedPtOnGeomLoc)
    {
        indexedPtOnGeomLoc =
            detail::make_unique<algorithm::locate::IndexedPointInAreaLocator>(
                getGeometry());
    }
    return dynamic_cast<algorithm::locate::IndexedPointInAreaLocator *>(
        indexedPtOnGeomLoc.get());
}

bool PreparedPolygon::contains(const geom::Geometry *g) const
{
    // short-circuit test
    if (!envelopeCoversSerial(g))
    {
        return false;
    }

    // optimization for rectangles
    if (isRectangle)
    {
        geom::Geometry const &geom = getGeometry();
        geom::Polygon const &poly =
            *detail::down_cast<geom::Polygon const *>(&geom);

        return operation::predicate::RectangleContains::contains(poly, *g);
    }

    return PreparedPolygonContains::contains(this, g);
}

bool PreparedPolygon::containsProperly(const geom::Geometry *g) const
{
    // short-circuit test
    if (!envelopeCoversSerial(g))
    {
        return false;
    }

    return PreparedPolygonContainsProperly::containsProperly(this, g);
}

bool PreparedPolygon::covers(const geom::Geometry *g) const
{
    // short-circuit test
    if (!envelopeCoversSerial(g))
    {
        return false;
    }

    // optimization for rectangle arguments
    if (isRectangle)
    {
        return true;
    }

    return PreparedPolygonCovers::covers(this, g);
}

bool PreparedPolygon::intersects(const geom::Geometry *g) const
{
    geos::util::ensureNoCurvedComponents(g);

    // envelope test
    if (!envelopesIntersect(g))
    {
        return false;
    }

    // optimization for rectangles
    if (isRectangle)
    {
        geom::Geometry const &geom = getGeometry();
        geom::Polygon const &poly = dynamic_cast<geom::Polygon const &>(geom);

        return operation::predicate::RectangleIntersects::intersects(poly, *g);
    }

    return PreparedPolygonIntersects::intersects(this, g);
}

/* public */
operation::distance::IndexedFacetDistance *
PreparedPolygon::getIndexedFacetDistance() const
{
    if (!indexedDistance)
    {
        indexedDistance.reset(
            new operation::distance::IndexedFacetDistance(&getGeometry()));
    }
    return indexedDistance.get();
}

double PreparedPolygon::distance(const geom::Geometry *g) const
{
    return PreparedPolygonDistance::distance(*this, g);
}

bool PreparedPolygon::isWithinDistance(const geom::Geometry *g, double d) const
{
    return PreparedPolygonDistance(*this).isWithinDistance(g, d);
}

// SIMD version of RayCrossingCounter for multiple segments
int PreparedPolygon::countSegmentsBatch(float point_x, float point_y,
                                        const FVectorT_1 &segments_x1,
                                        const FVectorT_1 &segments_y1,
                                        const FVectorT_1 &segments_x2,
                                        const FVectorT_1 &segments_y2,
                                        FVectorT_1 active_mask,
                                        bool &isOnSegment, bool print) const
{
    // Check if segments are strictly to the left of the test point
    active_mask =
        active_mask & ((segments_x1 >= point_x) | (segments_x2 >= point_x));

    if (active_mask.none())
    {
        return 0;
    }

    // Check if point equals segment endpoints
    isOnSegment =
        (active_mask & (segments_x2 == point_x) & (segments_y2 == point_y))
            .any();

    if (isOnSegment)
    {
        return 0;
    }

    // For horizontal segments, check if the point is on the segment.
    // Otherwise, horizontal segments are not counted.
    isOnSegment =
        (active_mask & (segments_y1 == point_y) & (segments_y2 == point_y) &
         ((segments_x1 - point_x) * (segments_x2 - point_x) <= 0))
            .any();

    if (isOnSegment)
    {
        return 0;
    }

    // Evaluate all non-horizontal segments which cross a horizontal ray
    // to the right of the test pt.
    // To avoid float-counting shared vertices, we use the convention that
    // - an upward edge includes its starting endpoint, and excludes its
    //   final endpoint
    // - a downward edge excludes its starting endpoint, and includes its
    //   final endpoint
    active_mask =
        active_mask & ((segments_y1 > point_y) ^ (segments_y2 > point_y));

    if (active_mask.none())
    {
        return 0;
    }

    // For an upward edge, orientationIndex will be positive when p1->p2
    // crosses ray. Conversely, downward edges should have negative sign.
    auto det = (segments_x2 - segments_x1) * (point_y - segments_y1) -
               (segments_y2 - segments_y1) * (point_x - segments_x1);
    isOnSegment = (active_mask & (det == 0.0f)).any();

    if (isOnSegment)
    {
        return 0;
    }

    active_mask = active_mask & (FVectorT_1::select(segments_y2 < segments_y1,
                                                    -det, det) > 0.0f);

    int crossing_count = 0;
    for (int i = 0; i < vamp::FloatVectorWidth; ++i)
    {
        if (active_mask[{0, i}])
        {
            ++crossing_count;
        }
    }
    return crossing_count;
}

// Optimized segment caching for spatial queries
void PreparedPolygon::cacheSegmentsInBbox(
    float min_y, float max_y, const int &VECTOR_SIZE, int &buffer_index,
    int &vec_index, AlignedVectorFloat &simd_segments_x1_buffer,
    AlignedVectorFloat &simd_segments_y1_buffer,
    AlignedVectorFloat &simd_segments_x2_buffer,
    AlignedVectorFloat &simd_segments_y2_buffer,
    AlignedVectorInt &simd_valid_mask_buffer,
    std::vector<FVectorT_1> &simd_segments_x1_vec,
    std::vector<FVectorT_1> &simd_segments_y1_vec,
    std::vector<FVectorT_1> &simd_segments_x2_vec,
    std::vector<FVectorT_1> &simd_segments_y2_vec,
    std::vector<FVectorT_1> &simd_valid_mask_vec, bool print_debug) const
{
    vec_index = 0;

    // Query all segments in the bounding box once
    indexedPtInAreaLocator->index->query(
        min_y, max_y,
        [&](const SegmentView &segment)
        {
            if (vec_index == VECTOR_SIZE)
            {
                return;
            }
            simd_segments_x1_buffer[buffer_index] = segment.p0().x;
            simd_segments_y1_buffer[buffer_index] = segment.p0().y;
            simd_segments_x2_buffer[buffer_index] = segment.p1().x;
            simd_segments_y2_buffer[buffer_index] = segment.p1().y;
            simd_valid_mask_buffer[buffer_index] = 0xFFFFFFFF;

            ++buffer_index;
            if (buffer_index == vamp::FloatVectorWidth)
            {
                simd_segments_x1_vec[vec_index] =
                    FVectorT_1(simd_segments_x1_buffer.data());
                simd_segments_y1_vec[vec_index] =
                    FVectorT_1(simd_segments_y1_buffer.data());
                simd_segments_x2_vec[vec_index] =
                    FVectorT_1(simd_segments_x2_buffer.data());
                simd_segments_y2_vec[vec_index] =
                    FVectorT_1(simd_segments_y2_buffer.data());
                simd_valid_mask_vec[vec_index] =
                    IVectorT_1(simd_valid_mask_buffer.data())
                        .template as<FVectorT_1>();
                std::fill(simd_valid_mask_buffer.begin(),
                          simd_valid_mask_buffer.end(), 0);

                buffer_index = 0;
                ++vec_index;
            }
        });

    if (buffer_index != 0 && vec_index < VECTOR_SIZE)
    {
        simd_segments_x1_vec[vec_index] =
            FVectorT_1(simd_segments_x1_buffer.data());
        simd_segments_y1_vec[vec_index] =
            FVectorT_1(simd_segments_y1_buffer.data());
        simd_segments_x2_vec[vec_index] =
            FVectorT_1(simd_segments_x2_buffer.data());
        simd_segments_y2_vec[vec_index] =
            FVectorT_1(simd_segments_y2_buffer.data());
        simd_valid_mask_vec[vec_index] =
            IVectorT_1(simd_valid_mask_buffer.data()).template as<FVectorT_1>();
        std::fill(simd_valid_mask_buffer.begin(), simd_valid_mask_buffer.end(),
                  0);

        buffer_index = 0;
        ++vec_index;
    }
}

// Optimized crossing count using pre-cached segments
int PreparedPolygon::countCrossingsBatch(
    float point_x, float point_y, int vec_index,
    const std::vector<FVectorT_1> &simd_segments_x1_vec,
    const std::vector<FVectorT_1> &simd_segments_y1_vec,
    const std::vector<FVectorT_1> &simd_segments_x2_vec,
    const std::vector<FVectorT_1> &simd_segments_y2_vec,
    const std::vector<FVectorT_1> &simd_valid_mask_vec, bool print) const
{
    int total_crossings = 0;
    bool on_segment = false;

    // Process segments in SIMD batches
    for (int i = 0; i < vec_index; ++i)
    {
        int batch_crossings = countSegmentsBatch(
            point_x, point_y, simd_segments_x1_vec[i], simd_segments_y1_vec[i],
            simd_segments_x2_vec[i], simd_segments_y2_vec[i],
            simd_valid_mask_vec[i], on_segment);
        if (on_segment)
            return -1;
        total_crossings += batch_crossings;
    }

    return total_crossings;
}

void PreparedPolygon::containsPointsInRouteEdgesBatchOptimized(
    const FVectorT_traj &center_xs, const FVectorT_traj &center_ys, const FVectorT_traj *point_xs,
    const FVectorT_traj *point_ys, const std::array<float, FVectorT_traj::num_scalars> &center_xs_arr,
    const std::array<float, FVectorT_traj::num_scalars> &center_ys_arr,
    const std::array<float, FVectorT_traj::num_scalars> *point_xs_arr,
    const std::array<float, FVectorT_traj::num_scalars> *point_ys_arr,
    const std::vector<float> &min_ys, const std::vector<float> &max_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_min_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_max_ys,
    const int &corner_category_num, const std::string &polygon_id,
    std::vector<std::vector<std::vector<std::string>>> &corners_area_ids,
    std::vector<AlignedVectorBool> &corners_area_on_drivable_area_mask,
    AlignedVectorInt &on_coming_traffic_mask, const int &VECTOR_SIZE,
    int &buffer_index, int &vec_index,
    AlignedVectorFloat &simd_segments_x1_buffer,
    AlignedVectorFloat &simd_segments_y1_buffer,
    AlignedVectorFloat &simd_segments_x2_buffer,
    AlignedVectorFloat &simd_segments_y2_buffer,
    AlignedVectorInt &simd_valid_mask_buffer,
    std::vector<FVectorT_1> &simd_segments_x1_vec,
    std::vector<FVectorT_1> &simd_segments_y1_vec,
    std::vector<FVectorT_1> &simd_segments_x2_vec,
    std::vector<FVectorT_1> &simd_segments_y2_vec,
    std::vector<FVectorT_1> &simd_valid_mask_vec, bool print_debug) const
{
    // quick envelope check
    for (int corner_type = 0; corner_type < corner_category_num; ++corner_type)
    {
        FVectorT_traj point_in_envelop_mask;
        if (!this->template envelopeCoversBatch<FVectorT_traj>(point_xs[corner_type],
                                                   point_ys[corner_type],
                                                   point_in_envelop_mask))
        {
            continue;
        }

        auto point_in_envelop_mask_arr = point_in_envelop_mask.to_array();

        // Cache all relevant segments once
        bool query_different_lateral_offset = false;
        if (std::fabs(max_ys[corner_type] - min_ys[corner_type]) >=
            vec_qmdp::utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
        {
            query_different_lateral_offset = true;
        }
        else
        {
            cacheSegmentsInBbox(
                min_ys[corner_type], max_ys[corner_type], VECTOR_SIZE,
                buffer_index, vec_index, simd_segments_x1_buffer,
                simd_segments_y1_buffer, simd_segments_x2_buffer,
                simd_segments_y2_buffer, simd_valid_mask_buffer,
                simd_segments_x1_vec, simd_segments_y1_vec,
                simd_segments_x2_vec, simd_segments_y2_vec, simd_valid_mask_vec,
                print_debug);
        }

        for (int row = 0; row < FVectorT_traj::num_rows; ++row)
        {
            if (query_different_lateral_offset)
            {
                cacheSegmentsInBbox(
                    different_lateral_offset_min_ys[corner_type][row],
                    different_lateral_offset_max_ys[corner_type][row],
                    VECTOR_SIZE, buffer_index, vec_index,
                    simd_segments_x1_buffer, simd_segments_y1_buffer,
                    simd_segments_x2_buffer, simd_segments_y2_buffer,
                    simd_valid_mask_buffer, simd_segments_x1_vec,
                    simd_segments_y1_vec, simd_segments_x2_vec,
                    simd_segments_y2_vec, simd_valid_mask_vec, print_debug);
            }

            for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
            {
                int idx = i + row * FVectorT_traj::num_scalars_per_row;
                // Quick envelope check
                if (!point_in_envelop_mask_arr[idx])
                {
                    continue;
                }

                int crossings = countCrossingsBatch(
                    point_xs_arr[corner_type][idx],
                    point_ys_arr[corner_type][idx], vec_index,
                    simd_segments_x1_vec, simd_segments_y1_vec,
                    simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec);

                if (crossings == -1 || crossings % 2 == 1)
                {
                    corners_area_ids[corner_type][idx].emplace_back(polygon_id);
                    corners_area_on_drivable_area_mask[corner_type][idx] = true;
                }
            }
        }
    }

    // quick envelope check
    FVectorT_traj point_in_envelop_mask;
    if (!this->template envelopeCoversBatch<FVectorT_traj>(center_xs, center_ys,
                                               point_in_envelop_mask))
    {
        return;
    }

    auto point_in_envelop_mask_arr = point_in_envelop_mask.to_array();
    float min_center_y = center_ys.hmin();
    float max_center_y = center_ys.hmax();
    bool query_different_lateral_offset = false;
    if (std::fabs(max_center_y - min_center_y) >=
        vec_qmdp::utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
    {
        query_different_lateral_offset = true;
    }
    else
    {
        cacheSegmentsInBbox(
            min_center_y, max_center_y, VECTOR_SIZE, buffer_index, vec_index,
            simd_segments_x1_buffer, simd_segments_y1_buffer,
            simd_segments_x2_buffer, simd_segments_y2_buffer,
            simd_valid_mask_buffer, simd_segments_x1_vec, simd_segments_y1_vec,
            simd_segments_x2_vec, simd_segments_y2_vec, simd_valid_mask_vec);
    }

    // process on coming traffic
    for (int row = 0; row < FVectorT_traj::num_rows; ++row)
    {
        if (query_different_lateral_offset)
        {
            cacheSegmentsInBbox(
                center_ys.row(row).hmin(), center_ys.row(row).hmax(),
                VECTOR_SIZE, buffer_index, vec_index, simd_segments_x1_buffer,
                simd_segments_y1_buffer, simd_segments_x2_buffer,
                simd_segments_y2_buffer, simd_valid_mask_buffer,
                simd_segments_x1_vec, simd_segments_y1_vec,
                simd_segments_x2_vec, simd_segments_y2_vec,
                simd_valid_mask_vec);
        }

        for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
        {
            int idx = i + row * FVectorT_traj::num_scalars_per_row;
            if (on_coming_traffic_mask[idx])
            {
                if (!corners_area_ids[0][idx].empty() &&
                    !corners_area_ids[1][idx].empty() &&
                    !corners_area_ids[2][idx].empty() &&
                    !corners_area_ids[3][idx].empty())
                {
                    on_coming_traffic_mask[idx] = 0;
                    continue;
                }

                if (!point_in_envelop_mask_arr[idx])
                {
                    continue;
                }

                int crossings = countCrossingsBatch(
                    center_xs_arr[idx], center_ys_arr[idx], vec_index,
                    simd_segments_x1_vec, simd_segments_y1_vec,
                    simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec);

                if (crossings == -1 || crossings % 2 == 1)
                {
                    on_coming_traffic_mask[idx] = 0;
                }
            }
        }
    }
}

void PreparedPolygon::containsPointsInNotOnRouteBatchOptimized(
    const FVectorT_traj *point_xs, const FVectorT_traj *point_ys,
    const std::array<float, FVectorT_traj::num_scalars> *point_xs_arr,
    const std::array<float, FVectorT_traj::num_scalars> *point_ys_arr,
    const std::vector<float> &min_ys, const std::vector<float> &max_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_min_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_max_ys,
    const int &corner_category_num,
    std::vector<AlignedVectorBool> &corners_area_on_drivable_area_mask,
    const int &VECTOR_SIZE, int &buffer_index, int &vec_index,
    AlignedVectorFloat &simd_segments_x1_buffer,
    AlignedVectorFloat &simd_segments_y1_buffer,
    AlignedVectorFloat &simd_segments_x2_buffer,
    AlignedVectorFloat &simd_segments_y2_buffer,
    AlignedVectorInt &simd_valid_mask_buffer,
    std::vector<FVectorT_1> &simd_segments_x1_vec,
    std::vector<FVectorT_1> &simd_segments_y1_vec,
    std::vector<FVectorT_1> &simd_segments_x2_vec,
    std::vector<FVectorT_1> &simd_segments_y2_vec,
    std::vector<FVectorT_1> &simd_valid_mask_vec) const
{
    // quick envelope check
    for (int corner_type = 0; corner_type < corner_category_num; ++corner_type)
    {
        FVectorT_traj point_in_envelop_mask;
        if (!this->template envelopeCoversBatch<FVectorT_traj>(point_xs[corner_type],
                                                   point_ys[corner_type],
                                                   point_in_envelop_mask))
        {
            continue;
        }

        auto point_in_envelop_mask_arr = point_in_envelop_mask.to_array();

        // Cache all relevant segments once
        bool query_different_lateral_offset = false;
        if (std::fabs(max_ys[corner_type] - min_ys[corner_type]) >=
            vec_qmdp::utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
        {
            query_different_lateral_offset = true;
        }
        else
        {
            cacheSegmentsInBbox(
                min_ys[corner_type], max_ys[corner_type], VECTOR_SIZE,
                buffer_index, vec_index, simd_segments_x1_buffer,
                simd_segments_y1_buffer, simd_segments_x2_buffer,
                simd_segments_y2_buffer, simd_valid_mask_buffer,
                simd_segments_x1_vec, simd_segments_y1_vec,
                simd_segments_x2_vec, simd_segments_y2_vec,
                simd_valid_mask_vec);
        }

        for (int row = 0; row < FVectorT_traj::num_rows; ++row)
        {
            if (query_different_lateral_offset)
            {
                cacheSegmentsInBbox(
                    different_lateral_offset_min_ys[corner_type][row],
                    different_lateral_offset_max_ys[corner_type][row],
                    VECTOR_SIZE, buffer_index, vec_index,
                    simd_segments_x1_buffer, simd_segments_y1_buffer,
                    simd_segments_x2_buffer, simd_segments_y2_buffer,
                    simd_valid_mask_buffer, simd_segments_x1_vec,
                    simd_segments_y1_vec, simd_segments_x2_vec,
                    simd_segments_y2_vec, simd_valid_mask_vec);
            }

            for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
            {
                int idx = i + row * FVectorT_traj::num_scalars_per_row;
                if (!corners_area_on_drivable_area_mask[corner_type][idx] &&
                    point_in_envelop_mask_arr[idx])
                {
                    int crossings = countCrossingsBatch(
                        point_xs_arr[corner_type][idx],
                        point_ys_arr[corner_type][idx], vec_index,
                        simd_segments_x1_vec, simd_segments_y1_vec,
                        simd_segments_x2_vec, simd_segments_y2_vec,
                        simd_valid_mask_vec);
                    if (crossings == -1 || crossings % 2 == 1)
                    {
                        corners_area_on_drivable_area_mask[corner_type][idx] =
                            true;
                    }
                }
            }
        }
    }
}

void PreparedPolygon::containsPointsInIntersectionBatchOptimized(
    const FVectorT_traj &rear_axle_xs, const FVectorT_traj &rear_axle_ys, const FVectorT_traj *point_xs,
    const FVectorT_traj *point_ys,
    const std::array<float, FVectorT_traj::num_scalars> &rear_axle_xs_arr,
    const std::array<float, FVectorT_traj::num_scalars> &rear_axle_ys_arr,
    const std::array<float, FVectorT_traj::num_scalars> *point_xs_arr,
    const std::array<float, FVectorT_traj::num_scalars> *point_ys_arr,
    const std::vector<float> &min_ys, const std::vector<float> &max_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_min_ys,
    const std::vector<std::vector<float>> &different_lateral_offset_max_ys,
    const int &corner_category_num,
    std::vector<AlignedVectorBool> &corners_area_on_drivable_area_mask,
    AlignedVectorInt &on_intersection_mask, const int &VECTOR_SIZE,
    int &buffer_index, int &vec_index,
    AlignedVectorFloat &simd_segments_x1_buffer,
    AlignedVectorFloat &simd_segments_y1_buffer,
    AlignedVectorFloat &simd_segments_x2_buffer,
    AlignedVectorFloat &simd_segments_y2_buffer,
    AlignedVectorInt &simd_valid_mask_buffer,
    std::vector<FVectorT_1> &simd_segments_x1_vec,
    std::vector<FVectorT_1> &simd_segments_y1_vec,
    std::vector<FVectorT_1> &simd_segments_x2_vec,
    std::vector<FVectorT_1> &simd_segments_y2_vec,
    std::vector<FVectorT_1> &simd_valid_mask_vec) const
{
    // quick envelope check
    for (int corner_type = 0; corner_type < corner_category_num; ++corner_type)
    {
        FVectorT_traj point_in_envelop_mask;
        if (!this->template envelopeCoversBatch<FVectorT_traj>(point_xs[corner_type],
                                                   point_ys[corner_type],
                                                   point_in_envelop_mask))
        {
            continue;
        }

        auto point_in_envelop_mask_arr = point_in_envelop_mask.to_array();

        // Cache all relevant segments once
        bool query_different_lateral_offset = false;
        if (std::fabs(max_ys[corner_type] - min_ys[corner_type]) >=
            vec_qmdp::utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
        {
            query_different_lateral_offset = true;
        }
        else
        {
            cacheSegmentsInBbox(
                min_ys[corner_type], max_ys[corner_type], VECTOR_SIZE,
                buffer_index, vec_index, simd_segments_x1_buffer,
                simd_segments_y1_buffer, simd_segments_x2_buffer,
                simd_segments_y2_buffer, simd_valid_mask_buffer,
                simd_segments_x1_vec, simd_segments_y1_vec,
                simd_segments_x2_vec, simd_segments_y2_vec,
                simd_valid_mask_vec);
        }

        // process on route lane-lane_connectors
        for (int row = 0; row < FVectorT_traj::num_rows; ++row)
        {
            if (query_different_lateral_offset)
            {
                cacheSegmentsInBbox(
                    different_lateral_offset_min_ys[corner_type][row],
                    different_lateral_offset_max_ys[corner_type][row],
                    VECTOR_SIZE, buffer_index, vec_index,
                    simd_segments_x1_buffer, simd_segments_y1_buffer,
                    simd_segments_x2_buffer, simd_segments_y2_buffer,
                    simd_valid_mask_buffer, simd_segments_x1_vec,
                    simd_segments_y1_vec, simd_segments_x2_vec,
                    simd_segments_y2_vec, simd_valid_mask_vec);
            }

            for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
            {
                int idx = i + row * FVectorT_traj::num_scalars_per_row;
                if (!corners_area_on_drivable_area_mask[corner_type][idx] &&
                    point_in_envelop_mask_arr[idx])
                {
                    int crossings = countCrossingsBatch(
                        point_xs_arr[corner_type][idx],
                        point_ys_arr[corner_type][idx], vec_index,
                        simd_segments_x1_vec, simd_segments_y1_vec,
                        simd_segments_x2_vec, simd_segments_y2_vec,
                        simd_valid_mask_vec);
                    if (crossings == -1 || crossings % 2 == 1)
                    {
                        corners_area_on_drivable_area_mask[corner_type][idx] =
                            true;
                    }
                }
            }
        }
    }

    // quick envelope check
    FVectorT_traj point_in_envelop_mask;
    if (!this->template envelopeCoversBatch<FVectorT_traj>(rear_axle_xs, rear_axle_ys,
                                               point_in_envelop_mask))
    {
        return;
    }
    auto point_in_envelop_mask_arr = point_in_envelop_mask.to_array();
    float min_rear_axle_y = rear_axle_ys.hmin();
    float max_rear_axle_y = rear_axle_ys.hmax();
    bool query_different_lateral_offset = false;
    if (std::fabs(max_rear_axle_y - min_rear_axle_y) >=
        vec_qmdp::utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
    {
        query_different_lateral_offset = true;
    }
    else
    {
        cacheSegmentsInBbox(
            min_rear_axle_y, max_rear_axle_y, VECTOR_SIZE, buffer_index,
            vec_index, simd_segments_x1_buffer, simd_segments_y1_buffer,
            simd_segments_x2_buffer, simd_segments_y2_buffer,
            simd_valid_mask_buffer, simd_segments_x1_vec, simd_segments_y1_vec,
            simd_segments_x2_vec, simd_segments_y2_vec, simd_valid_mask_vec);
    }

    // process on intersection
    for (int row = 0; row < FVectorT_traj::num_rows; ++row)
    {
        if (query_different_lateral_offset)
        {
            cacheSegmentsInBbox(
                rear_axle_ys.row(row).hmin(), rear_axle_ys.row(row).hmax(),
                VECTOR_SIZE, buffer_index, vec_index, simd_segments_x1_buffer,
                simd_segments_y1_buffer, simd_segments_x2_buffer,
                simd_segments_y2_buffer, simd_valid_mask_buffer,
                simd_segments_x1_vec, simd_segments_y1_vec,
                simd_segments_x2_vec, simd_segments_y2_vec,
                simd_valid_mask_vec);
        }
        for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
        {
            int idx = i + row * FVectorT_traj::num_scalars_per_row;
            if (!on_intersection_mask[idx] && point_in_envelop_mask_arr[idx])
            {
                int crossings = countCrossingsBatch(
                    rear_axle_xs_arr[idx], rear_axle_ys_arr[idx], vec_index,
                    simd_segments_x1_vec, simd_segments_y1_vec,
                    simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec);

                if (crossings == -1 || crossings % 2 == 1)
                {
                    on_intersection_mask[idx] = 0xFFFFFFFF;
                }
            }
        }
    }
}

} // namespace prep
} // namespace geom
} // namespace geos
