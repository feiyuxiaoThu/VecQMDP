/**********************************************************************
 *
 * GEOS - Geometry Engine Open Source
 * http://geos.osgeo.org
 *
 * Copyright (C) 2020 Sandro Santilli <strk@kbt.io>
 * Copyright (C) 2006 Refractions Research Inc.
 *
 * This is free software; you can redistribute and/or modify it under
 * the terms of the GNU Lesser General Public Licence as published
 * by the Free Software Foundation.
 * See the COPYING file for more information.
 *
 *
 **********************************************************************
 *
 * Last port: geom/prep/PreparedPolygon.java rev 1.7 (JTS-1.10)
 *
 **********************************************************************/

#pragma once

#include <geos/algorithm/locate/IndexedPointInAreaLocator.h>
#include <geos/geom/prep/BasicPreparedGeometry.h> // for inheritance
#include <geos/noding/SegmentString.h>
#include <geos/operation/distance/IndexedFacetDistance.h>
#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <vamp/vector.hh>

#include <memory>

namespace geos
{
namespace noding
{
class FastSegmentSetIntersectionFinder;
}
namespace algorithm
{
namespace locate
{
class PointOnGeometryLocator;
}
} // namespace algorithm
} // namespace geos

namespace geos
{
namespace geom
{ // geos::geom
namespace prep
{ // geos::geom::prep

/**
 * \brief
 * A prepared version of {@link Polygon} or {@link MultiPolygon} geometries.
 *
 * @author mbdavis
 *
 */
class PreparedPolygon : public BasicPreparedGeometry
{

  public:
    using FVectorT_1 = vec_qmdp::utils::FVectorT_1;
    using IVectorT_1 = vec_qmdp::utils::IVectorT_1;
    using FVectorT_traj = vec_qmdp::utils::FVectorT_traj;
    using IVectorT_traj = vec_qmdp::utils::IVectorT_traj;
    using AlignedVectorFloat = vec_qmdp::utils::AlignedVectorFloat;
    using AlignedVectorInt = vec_qmdp::utils::AlignedVectorInt;
    using AlignedVectorBool = vec_qmdp::utils::AlignedVectorBool;
    using SegmentView =
        algorithm::locate::IndexedPointInAreaLocator::SegmentView;

  private:
    bool isRectangle;
    mutable std::unique_ptr<noding::FastSegmentSetIntersectionFinder>
        segIntFinder;
    mutable std::unique_ptr<algorithm::locate::PointOnGeometryLocator>
        ptOnGeomLoc;
    mutable std::unique_ptr<algorithm::locate::PointOnGeometryLocator>
        indexedPtOnGeomLoc;
    mutable noding::SegmentString::ConstVect segStrings;
    mutable std::unique_ptr<operation::distance::IndexedFacetDistance>
        indexedDistance;
    algorithm::locate::IndexedPointInAreaLocator *indexedPtInAreaLocator;

    // Inherited from BasicPreparedGeometry, but need explicit declaration for
    // clarity
    using BasicPreparedGeometry::envelopeCoversSerial;

    /**
     * SIMD version of RayCrossingCounter for checking multiple segments
     * simultaneously
     * @param point_x x coordinate of test point
     * @param point_y y coordinate of test point
     * @param segments_x1 SIMD vector of segment start x coordinates
     * @param segments_y1 SIMD vector of segment start y coordinates
     * @param segments_x2 SIMD vector of segment end x coordinates
     * @param segments_y2 SIMD vector of segment end y coordinates
     * @param segment_count number of valid segments in the vectors
     * @return crossing count for determining point location
     */
    int countSegmentsBatch(float point_x, float point_y,
                           const FVectorT_1 &segments_x1,
                           const FVectorT_1 &segments_y1,
                           const FVectorT_1 &segments_x2,
                           const FVectorT_1 &segments_y2,
                           FVectorT_1 active_mask, bool &isOnSegment,
                           bool print = false) const;

  protected:
  public:
    PreparedPolygon(const geom::Geometry *geom);
    ~PreparedPolygon() override;

    noding::FastSegmentSetIntersectionFinder *getIntersectionFinder() const;
    algorithm::locate::PointOnGeometryLocator *getPointLocator() const;
    algorithm::locate::IndexedPointInAreaLocator *
    getIndexedPointInAreaLocator();
    operation::distance::IndexedFacetDistance *getIndexedFacetDistance() const;

    bool contains(const geom::Geometry *g) const override;
    bool containsProperly(const geom::Geometry *g) const override;
    bool covers(const geom::Geometry *g) const override;
    bool intersects(const geom::Geometry *g) const override;
    double distance(const geom::Geometry *g) const override;
    bool isWithinDistance(const geom::Geometry *g, double d) const override;

    inline bool envelopeCoversSerial(float point_x, float point_y) const
    {
        const geom::Envelope *env = getGeometry().getEnvelopeInternal();

        bool point_in_envelop_mask =
            (point_x >= static_cast<float>(env->getMinX()) &&
             point_x <= static_cast<float>(env->getMaxX()) &&
             point_y >= static_cast<float>(env->getMinY()) &&
             point_y <= static_cast<float>(env->getMaxY()));

        return point_in_envelop_mask;
    }

    template <typename T>
    inline bool envelopeCoversBatch(const T &point_xs, const T &point_ys,
                                    T &point_in_envelop_mask) const
    {
        const geom::Envelope *env = getGeometry().getEnvelopeInternal();

        point_in_envelop_mask =
            (point_xs >= static_cast<float>(env->getMinX()) &
             point_xs <= static_cast<float>(env->getMaxX()) &
             point_ys >= static_cast<float>(env->getMinY()) &
             point_ys <= static_cast<float>(env->getMaxY()));

        return point_in_envelop_mask.any();
    }

    // Optimized batch processing methods
    void cacheSegmentsInBbox(float min_y, float max_y, const int &VECTOR_SIZE,
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
                             std::vector<FVectorT_1> &simd_valid_mask_vec,
                             bool print_debug = false) const;

    int countCrossingsBatch(float point_x, float point_y, int vec_index,
                            const std::vector<FVectorT_1> &simd_segments_x1_vec,
                            const std::vector<FVectorT_1> &simd_segments_y1_vec,
                            const std::vector<FVectorT_1> &simd_segments_x2_vec,
                            const std::vector<FVectorT_1> &simd_segments_y2_vec,
                            const std::vector<FVectorT_1> &simd_valid_mask_vec,
                            bool print = false) const;

    // Batch processing with optimal memory layout
    void containsPointsInRouteEdgesBatchOptimized(
        const FVectorT_traj &center_xs, const FVectorT_traj &center_ys, const FVectorT_traj *point_xs,
        const FVectorT_traj *point_ys,
        const std::array<float, FVectorT_traj::num_scalars> &center_xs_arr,
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
        std::vector<FVectorT_1> &simd_valid_mask_vec, bool print_debug) const;

    void containsPointsInNotOnRouteBatchOptimized(
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
        std::vector<FVectorT_1> &simd_valid_mask_vec) const;

    void containsPointsInIntersectionBatchOptimized(
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
        std::vector<FVectorT_1> &simd_valid_mask_vec) const;

    // Simplified SIMD ray crossing with minimal branches
    int countCrossingsSimplified(float point_x, float point_y,
                                 const float *seg_x1, const float *seg_y1,
                                 const float *seg_x2, const float *seg_y2,
                                 size_t num_segments) const;
};

} // namespace prep
} // namespace geom
} // namespace geos
