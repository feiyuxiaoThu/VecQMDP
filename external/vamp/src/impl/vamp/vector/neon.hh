#pragma once

#include <initializer_list>
#if not defined(__ARM_NEON)
#error "Tried to compile NEON intrinsics on non-ARM platform!"
#endif

#include <cstdint>

#include <vamp/vector/interface.hh>

#include <arm_neon.h>
#include <limits>

namespace vamp
{
    template <>
    struct SIMDVector<float32x4_t>
    {
        using VectorT = float32x4_t;
        using ScalarT = float32_t;
        static constexpr std::size_t VectorWidth = 4;
        static constexpr std::size_t Alignment = 16;

        template <unsigned int = 0>
        inline static auto constant(ScalarT v) noexcept -> VectorT
        {
            return vdupq_n_f32(v);
        }

        template <unsigned int = 0>
        inline static auto iota(ScalarT start) noexcept -> VectorT
        {
            static constexpr ScalarT offsets[4] = {0.0f, 1.0f, 2.0f, 3.0f};
            const ScalarT values[4] = {
                start + offsets[0],
                start + offsets[1],
                start + offsets[2],
                start + offsets[3]
            };
            return vld1q_f32(values);
        }

        template <unsigned int = 0>
        inline static auto load(const ScalarT *const f) noexcept -> VectorT
        {
            return vld1q_f32(f);
        }

        template <unsigned int = 0>
        inline static auto store(ScalarT *f, VectorT v) noexcept -> void
        {
            return vst1q_f32(f, v);
        }

        template <unsigned int = 0>
        inline static auto store_unaligned(ScalarT *f, VectorT v) noexcept -> void
        {
            return vst1q_f32(f, v);
        }

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            return v[idx];
        }

        // C++ is so dumb. We have to do this (unless someone has a cleverer idea) because (1) vdupq_laneq_f32
        // is a macro and the preprocessor hates commas and (2) you can't use the usual parenthesis trick f or
        // commas with parameter packs, apparently
        template <std::size_t idx>
        inline static constexpr auto broadcast_dispatch(VectorT v) noexcept -> VectorT
        {
            return vdupq_laneq_f32(v, idx);
        }

        template <std::size_t... I>
        inline static constexpr auto
        broadcast_lookup(VectorT v, std::size_t lane, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            std::initializer_list<int>(
                {(lane == I ? (ret = broadcast_dispatch<std::integral_constant<int, I>{}>(v)), 0 : 0)...});
            return ret;
        }

        template <unsigned int = 0>
        inline static constexpr auto broadcast(VectorT v, std::size_t lane) noexcept -> VectorT
        {
            return broadcast_lookup(v, lane, std::make_index_sequence<VectorWidth>());
        }

        // NOTE: Dummy parameter because otherwise we get constexpr errors with set1_ps...
        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(l)));  // maybe a reverse is needed
        }

        template <unsigned int = 0>
        inline static constexpr auto neg(VectorT l) noexcept -> VectorT
        {
            return vreinterpretq_f32_s32(veorq_s32(
                vreinterpretq_s32_f32(l),
                vreinterpretq_s32_f32(vdupq_n_f32(-0.0))));  // maybe a reverse is needed
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vaddq_f32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vsubq_f32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vmulq_f32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vcleq_f32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vcgeq_f32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vceqq_f32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(l, r)));
        }

        // NOTE: Dummy parameter because otherwise we get constexpr errors with set1_ps...
        template <unsigned int = 0>
        inline static auto floor(VectorT v) noexcept -> VectorT
        {
            return vrndmq_f32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto div(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vdivq_f32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto rcp(VectorT l) noexcept -> VectorT
        {
            auto s = vrecpeq_f32(l);
            auto p = vrecpsq_f32(l, s);
            return vmulq_f32(s, p);
        }

        template <unsigned int = 0>
        inline static auto mask(VectorT v) noexcept -> unsigned int
        {
            auto MSB = vsliq_n_u32(vdupq_n_u32(0), vreinterpretq_u32_f32(v), 16);
            auto sumtwo = vreinterpret_u32_u16(
                vpadd_u16(vreinterpret_u16_u32(vget_low_u32(MSB)), vreinterpret_u16_u32(vget_high_u32(MSB))));
            auto attempt = vreinterpret_u16_u32(sumtwo);
            auto attempt2 = vreinterpret_u8_u16(attempt);
            auto reorg = vshrn_n_u16(vreinterpretq_u16_u8(vcombine_u8(attempt2, attempt2)), 8);
            return vget_lane_u32(vreinterpret_u32_u8(reorg), 0);
            // IT MAY NEED A REVERSE vrev32_u8
            // vget_lane_u32(vreinterpret_u32_u8(vrev32_u8(reorg)), 0);
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return vmovq_n_f32(0.0f);
        }

        template <unsigned int = 0>
        inline static auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            auto andlr = vandq_u32(vreinterpretq_u32_f32(l), vreinterpretq_u32_f32(r));
            auto horizor = vorr_u32(vget_low_u32(andlr), vget_high_u32(andlr));
            uint32x2_t mask = {0x80000000, 0x80000000};
            auto test = vand_u32(horizor, mask);
            return (vget_lane_u32(test, 0) || vget_lane_u32(test, 1)) == 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto abs(VectorT v) noexcept -> VectorT
        {
            return vabsq_f32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(l), vreinterpretq_u32_f32(r)));
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(l), vreinterpretq_u32_f32(r)));
        }

        template <std::size_t... I>
        inline static constexpr auto
        lshift_lookup(VectorT v, ScalarT shift, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            std::initializer_list<int>(
                {(shift == I ? (ret = lshift_dispatch<std::integral_constant<int, I>{}>(v)), 0 : 0)...});
            return ret;
        }

        template <unsigned int i>
        inline static constexpr auto lshift_dispatch(VectorT v) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(v), i));
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, ScalarT i) noexcept -> VectorT
        {
            return lshift_lookup(v, i, std::make_index_sequence<32>());
        }

        template <std::size_t... I>
        inline static constexpr auto
        rshift_lookup(VectorT v, ScalarT shift, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            std::initializer_list<int>(
                {(shift == I + 1 ? (ret = rshift_dispatch<std::integral_constant<int, I + 1>{}>(v)),
                  0 :
                                   0)...});
            return ret;
        }

        template <unsigned int i>
        inline static constexpr auto rshift_dispatch(VectorT v) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(v), i));
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, ScalarT i) noexcept -> VectorT
        {
            return rshift_lookup(v, i, std::make_index_sequence<32>());
        }

        template <unsigned int = 0>
        inline static constexpr auto sqrt(VectorT v) noexcept -> VectorT
        {
            return vsqrtq_f32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto clamp(VectorT v, VectorT lower, VectorT upper) noexcept -> VectorT
        {
            return vminq_f32(vmaxq_f32(v, lower), upper);
        }

        template <unsigned int = 0>
        inline static constexpr auto max(VectorT v, VectorT other) noexcept -> VectorT
        {
            return vmaxq_f32(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto hsum(VectorT v) noexcept -> float
        {
            return vaddvq_f32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto min(VectorT v, VectorT other) noexcept -> VectorT
        {
            return vminq_f32(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmin(VectorT v) noexcept -> float
        {
            float32x2_t min1 = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
            float32x2_t min2 = vpmin_f32(min1, min1);
            return vget_lane_f32(min2, 0);
        }

        template <unsigned int = 0>
        inline static constexpr auto blend(VectorT a, VectorT b, VectorT blend_mask) noexcept -> VectorT
        {
            return vbslq_f32(vreinterpretq_u32_f32(blend_mask), b, a);
        }

        // HACK: We only ever use this for trim() and pack_and_pad, and ARM makes it hard to go from
        // a scalar mask to an appropriate vector mask for vbslq. So, we special-case for the values
        // we could possibly get
        template <unsigned int blend_mask>
        inline static constexpr auto blend_constant(VectorT a, VectorT b) noexcept -> VectorT
        {
            if constexpr (blend_mask == 8)
            {
                return vbslq_f32(vcombine_u32(vcreate_u32(0l), vcreate_u32(0xffffffff00000000)), b, a);
            }
            else if constexpr (blend_mask == 12)
            {
                return vbslq_f32(vcombine_u32(vcreate_u32(0l), vcreate_u32(0xffffffffffffffff)), b, a);
            }
            else if constexpr (blend_mask == 14)
            {
                return vbslq_f32(
                    vcombine_u32(vcreate_u32(0xffffffff00000000), vcreate_u32(0xffffffffffffffff)), b, a);
            }
            else
            {
                static_assert(always_false<blend_mask>, "blend_mask not in allowed value set!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            return vcvtq_s32_f32(v);
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            return vcvtq_f32_s32(v);
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            return vreinterpretq_s32_f32(v);
        }

        template <typename OtherVectorT>
        inline auto map_to_range(OtherVectorT v) -> VectorT
        {
            const auto v_1 = vandq_s32(v, vdupq_n_s32(1));
            const auto v1_f = vcvtq_f32_s32(v_1);
            const auto v_scaled = vaddq_f32(vcvtq_f32_s32(v), v1_f);
            return vmulq_f32(
                v_scaled, vdupq_n_f32(1.f / static_cast<float>(std::numeric_limits<unsigned int>::max())));
        }

        template <typename = void>
        inline static auto gather(int32x4_t idxs, const ScalarT *base) noexcept -> VectorT
        {
            // Pretty sure there isn't a better way to do a 32-bit lookup table...
            float32x4_t result = vdupq_n_f32(0);
            result = vsetq_lane_f32(base[vgetq_lane_s32(idxs, 0)], result, 0);
            result = vsetq_lane_f32(base[vgetq_lane_s32(idxs, 1)], result, 1);
            result = vsetq_lane_f32(base[vgetq_lane_s32(idxs, 2)], result, 2);
            result = vsetq_lane_f32(base[vgetq_lane_s32(idxs, 3)], result, 3);
            return result;
        }

        template <typename = void>
        inline static constexpr auto
        gather_select(int32x4_t idxs, VectorT mask, VectorT alternative, const ScalarT *base) noexcept
            -> VectorT
        {
            auto overlay = gather(idxs, base);
            return blend(overlay, alternative, mask);
        }

        template <unsigned int = 0>
        inline static constexpr auto select(VectorT mask, VectorT when_true, VectorT when_false) noexcept -> VectorT
        {
            return vbslq_f32(vreinterpretq_u32_f32(mask), when_true, when_false);
        }

        template <unsigned int = 0>
        inline static constexpr auto mod(VectorT v, VectorT divisor) noexcept -> VectorT
        {
            auto quotient = vrndmq_f32(vdivq_f32(v, divisor));
            return vsubq_f32(v, vmulq_f32(quotient, divisor));
        }

        template <unsigned int = 0>
        inline static constexpr auto sign(VectorT v) noexcept -> VectorT
        {
            const auto zero = vdupq_n_f32(0.0f);
            const auto pos_one = vdupq_n_f32(1.0f);
            const auto neg_one = vdupq_n_f32(-1.0f);

            auto gt_mask = vcgtq_f32(v, zero); // v > 0
            auto lt_mask = vcltq_f32(v, zero); // v < 0

            // bitwise AND preserves the float value; reinterpret+mul would yield NaN
            auto pos_values = vreinterpretq_f32_u32(vandq_u32(gt_mask, vreinterpretq_u32_f32(pos_one)));
            auto neg_values = vreinterpretq_f32_u32(vandq_u32(lt_mask, vreinterpretq_u32_f32(neg_one)));

            return vaddq_f32(pos_values, neg_values);
        }

        template <unsigned int = 0>
        inline static constexpr auto pow_vector(VectorT v, float exponent) noexcept -> VectorT
        {
            alignas(16) float temp[4];
            vst1q_f32(temp, v);
            
            for (int i = 0; i < 4; ++i) {
                temp[i] = std::pow(temp[i], exponent);
            }
            
            return vld1q_f32(temp);
        }

        template <unsigned int = 0>
        inline static constexpr auto atan2_approx(VectorT y, VectorT x) noexcept -> VectorT
        {
            const auto zero    = vdupq_n_f32(0.0f);
            const auto pi      = vdupq_n_f32(3.14159265359f);
            const auto pi_half = vdupq_n_f32(1.57079632679f);

            const auto C1 = vdupq_n_f32(-0.0464964749f);
            const auto C2 = vdupq_n_f32( 0.15931422f);
            const auto C3 = vdupq_n_f32(-0.327622764f);

            auto abs_y  = vabsq_f32(y);
            auto abs_x  = vabsq_f32(x);

            auto min_xy = vminq_f32(abs_x, abs_y);
            auto max_xy = vmaxq_f32(abs_x, abs_y);
            auto a      = vdivq_f32(min_xy, max_xy);
            auto s      = vmulq_f32(a, a);

            // polynomial: r = ((C1*s + C2)*s + C3)*s*a + a
            auto r = vfmaq_f32(C2, C1, s);   // C1*s + C2
            r = vfmaq_f32(C3, r, s);           // (prev)*s + C3
            r = vmulq_f32(r, s);
            r = vmulq_f32(r, a);
            r = vaddq_f32(r, a);

            // adjust for octant
            auto y_gt_x = vcgtq_f32(abs_y, abs_x);
            r = vbslq_f32(y_gt_x, vsubq_f32(pi_half, r), r);

            auto x_neg = vcltq_f32(x, zero);
            auto y_neg = vcltq_f32(y, zero);

            // x<0, y>=0  ->  r = pi - r
            auto mask1 = vbicq_u32(x_neg, y_neg);
            r = vbslq_f32(mask1, vsubq_f32(pi, r), r);

            // x<0, y<0   ->  r = r - pi
            auto mask2 = vandq_u32(x_neg, y_neg);
            r = vbslq_f32(mask2, vsubq_f32(r, pi), r);

            // x>=0, y<0  ->  r = -r
            auto mask3 = vbicq_u32(y_neg, x_neg);
            r = vbslq_f32(mask3, vsubq_f32(zero, r), r);

            return r;
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vcltq_f32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(vcgtq_f32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto xor_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(l), vreinterpretq_u32_f32(r)));
        }

        template <unsigned int = 0>
        inline static auto load_unaligned(const ScalarT *const f) noexcept -> VectorT
        {
            return vld1q_f32(f);
        }

        template <unsigned int = 0>
        inline static auto create_tail_mask(int remainder) noexcept -> VectorT
        {
            alignas(16) static constexpr int32_t indices[4] = {0, 1, 2, 3};
            const int32x4_t v_indices   = vld1q_s32(indices);
            const int32x4_t v_remainder = vdupq_n_s32(remainder);
            return vreinterpretq_f32_u32(vcgtq_s32(v_remainder, v_indices));
        }

        template <unsigned int = 0>
        inline static constexpr auto hmax(VectorT v) noexcept -> float
        {
            float32x2_t max1 = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
            float32x2_t max2 = vpmax_f32(max1, max1);
            return vget_lane_f32(max2, 0);
        }

        template <typename OtherVectorT>
        inline static constexpr auto convert(VectorT v) noexcept -> OtherVectorT
        {
            return vcvtq_s32_f32(v);
        }

        template <typename = void>
        inline static auto scatter(VectorT values, ScalarT *base, int32x4_t idxs) noexcept -> void
        {
            alignas(16) ScalarT  vals[4];
            alignas(16) int32_t  inds[4];
            vst1q_f32(vals, values);
            vst1q_s32(inds, idxs);
            base[inds[0]] = vals[0];
            base[inds[1]] = vals[1];
            base[inds[2]] = vals[2];
            base[inds[3]] = vals[3];
        }

        inline static void sincos(VectorT x, VectorT& out_sin, VectorT& out_cos) noexcept
        {
            const float32x4_t PI      = vdupq_n_f32(3.14159265359f);
            const float32x4_t HALF_PI = vdupq_n_f32(1.57079632679f);
            const float32x4_t TWO_PI  = vdupq_n_f32(6.28318530718f);

            // compute x_cos = (x + pi/2) wrapped into [-pi, pi)
            float32x4_t v      = vaddq_f32(x, HALF_PI);
            uint32x4_t  smask  = vcgeq_f32(v, PI);
            float32x4_t x_cos  = vsubq_f32(v, vreinterpretq_f32_u32(
                                     vandq_u32(smask, vreinterpretq_u32_f32(TWO_PI))));

            float32x4_t x_sin     = x;
            float32x4_t x_sin_abs = vabsq_f32(x_sin);
            float32x4_t x_cos_abs = vabsq_f32(x_cos);

            const float32x4_t C_s_vsq = vdupq_n_f32(-0.478637850138f);
            const float32x4_t C_s_v   = vdupq_n_f32( 1.503684069359f);
            const float32x4_t C_d1    = vdupq_n_f32( 0.665200679751f);
            const float32x4_t C_d2    = vdupq_n_f32( 0.140024078368f);
            const float32x4_t C_d3    = vdupq_n_f32( 0.011596870476f);

            // v_sq = x * |x|
            float32x4_t v_sq_sin = vmulq_f32(x_sin, x_sin_abs);
            float32x4_t v_sq_cos = vmulq_f32(x_cos, x_cos_abs);

            // p = v_sq * C_s_vsq + x * C_s_v
            float32x4_t p_sin = vfmaq_f32(vmulq_f32(x_sin, C_s_v), v_sq_sin, C_s_vsq);
            float32x4_t p_cos = vfmaq_f32(vmulq_f32(x_cos, C_s_v), v_sq_cos, C_s_vsq);

            float32x4_t abs_p_sin = vabsq_f32(p_sin);
            float32x4_t abs_p_cos = vabsq_f32(p_cos);
            float32x4_t p2_sin    = vmulq_f32(p_sin, abs_p_sin);
            float32x4_t p2_cos    = vmulq_f32(p_cos, abs_p_cos);
            float32x4_t p3_sin    = vmulq_f32(p2_sin, abs_p_sin);
            float32x4_t p3_cos    = vmulq_f32(p2_cos, abs_p_cos);

            // out = p3*C_d3 + p2*C_d2 + p*C_d1
            out_sin = vfmaq_f32(vfmaq_f32(vmulq_f32(p_sin, C_d1), p2_sin, C_d2), p3_sin, C_d3);
            out_cos = vfmaq_f32(vfmaq_f32(vmulq_f32(p_cos, C_d1), p2_cos, C_d2), p3_cos, C_d3);
        }

    };

    template <>
    struct SIMDVector<int32x4_t>
    {
        using VectorT = int32x4_t;
        using ScalarT = int32_t;
        static constexpr std::size_t VectorWidth = 4;
        static constexpr std::size_t Alignment = 16;

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            return ((int *)(&v))[idx];
        }

        template <unsigned int = 0>
        inline static constexpr auto constant(ScalarT v) noexcept -> VectorT
        {
            return vdupq_n_s32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vsubq_s32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vaddq_s32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vmulq_s32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vmvnq_u32(vreinterpretq_u32_s32(l)));  // maybe a reverse is needed
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vceqq_s32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vcgtq_s32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(l), vreinterpretq_u32_s32(r)));
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vorrq_u32(vreinterpretq_u32_s32(l), vreinterpretq_u32_s32(r)));
        }

        template <std::size_t... I>
        inline static constexpr auto
        lshift_lookup(VectorT v, ScalarT shift, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            std::initializer_list<int>(
                {(shift == I ? (ret = lshift_dispatch<std::integral_constant<int, I>{}>(v)), 0 : 0)...});
            return ret;
        }

        template <ScalarT i>
        inline static constexpr auto lshift_dispatch(VectorT v) noexcept -> VectorT
        {
            return vshlq_n_s32(v, i);
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, ScalarT i) noexcept -> VectorT
        {
            return lshift_lookup(v, i, std::make_index_sequence<32>());
        }

        template <std::size_t... I>
        inline static constexpr auto
        rshift_lookup(VectorT v, ScalarT shift, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            std::initializer_list<int>(
                {(shift == I + 1 ? (ret = rshift_dispatch<std::integral_constant<int, I + 1>{}>(v)),
                  0 :
                                   0)...});
            return ret;
        }

        template <ScalarT i>
        inline static constexpr auto rshift_dispatch(VectorT v) noexcept -> VectorT
        {
            return vshrq_n_s32(v, i);
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, ScalarT i) noexcept -> VectorT
        {
            return rshift_lookup(v, i, std::make_index_sequence<32>());
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return vmovq_n_s32(0);
        }

        template <unsigned int = 0>
        inline static auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            auto andlr = vandq_u32(vreinterpretq_u32_s32(l), vreinterpretq_u32_s32(r));
            auto horizor = vorr_u32(vget_low_u32(andlr), vget_high_u32(andlr));
            uint32x2_t mask = {0x80000000, 0x80000000};
            auto test = vand_u32(horizor, mask);
            return (vget_lane_u32(test, 0) || vget_lane_u32(test, 1)) == 0;
        }

        template <unsigned int = 0>
        inline static auto load(const ScalarT *const i) noexcept -> VectorT
        {
            return vld1q_s32((const int32_t *const)i);
        }

        template <unsigned int = 0>
        inline static auto store(ScalarT *i, VectorT v) noexcept -> void
        {
            return vst1q_s32(i, v);
        }

        template <unsigned int = 0>
        inline static auto store_unaligned(ScalarT *i, VectorT v) noexcept -> void
        {
            return vst1q_s32(i, v);
        }

        template <unsigned int = 0>
        inline static constexpr auto blend(VectorT a, VectorT b, VectorT blend_mask) noexcept -> VectorT
        {
            return vbslq_s32(vreinterpretq_u32_s32(blend_mask), b, a);
        }

        template <unsigned int = 0>
        inline static auto mask(VectorT v) noexcept -> unsigned int
        {
            auto MSB = vsliq_n_u32(vdupq_n_u32(0), vreinterpretq_u32_s32(v), 16);
            auto sumtwo = vreinterpret_u32_u16(
                vpadd_u16(vreinterpret_u16_u32(vget_low_u32(MSB)), vreinterpret_u16_u32(vget_high_u32(MSB))));
            auto attempt = vreinterpret_u16_u32(sumtwo);
            auto attempt2 = vreinterpret_u8_u16(attempt);
            auto reorg = vshrn_n_u16(vreinterpretq_u16_u8(vcombine_u8(attempt2, attempt2)), 8);
            return vget_lane_u32(vreinterpret_u32_u8(reorg), 0);
            // IT MAY NEED A REVERSE vrev32_u8
            // vget_lane_u32(vreinterpret_u32_u8(vrev32_u8(reorg)), 0);
        }

        template <typename = void>
        inline static constexpr auto gather(int32x4_t idxs, const ScalarT *base) noexcept -> VectorT
        {
            // Pretty sure there isn't a better way to do a 32-bit lookup table...
            int32x4_t result = vdupq_n_s32(0);
            result = vsetq_lane_s32(base[vgetq_lane_s32(idxs, 0)], result, 0);
            result = vsetq_lane_s32(base[vgetq_lane_s32(idxs, 1)], result, 1);
            result = vsetq_lane_s32(base[vgetq_lane_s32(idxs, 2)], result, 2);
            result = vsetq_lane_s32(base[vgetq_lane_s32(idxs, 3)], result, 3);
            return result;
        }

        template <typename = void>
        inline static constexpr auto
        gather_select(int32x4_t idxs, VectorT mask, VectorT alternative, const ScalarT *base) noexcept
            -> VectorT
        {
            auto overlay = gather(idxs, base);
            return blend(overlay, alternative, mask);
        }

        template <unsigned int = 0>
        inline static constexpr auto select(VectorT mask, VectorT when_true, VectorT when_false) noexcept -> VectorT
        {
            return vbslq_s32(vreinterpretq_u32_s32(mask), when_true, when_false);
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, float32x4_t>)
            {
                return vcvtq_f32_s32(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, float32x4_t>)
            {
                return vcvtq_s32_f32(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, float32x4_t>)
            {
                return vreinterpretq_f32_s32(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <unsigned int = 0>
        inline static constexpr auto min(VectorT v, VectorT other) noexcept -> VectorT
        {
            return vminq_s32(v, other);
        }
        
        template <unsigned int = 0>
        inline static constexpr auto hmin(VectorT v) noexcept -> ScalarT
        {
            int32x2_t min1 = vpmin_s32(vget_low_s32(v), vget_high_s32(v));
            int32x2_t min2 = vpmin_s32(min1, min1);
            return vget_lane_s32(min2, 0);
        }

        template <unsigned int = 0>
        inline static constexpr auto sign(VectorT v) noexcept -> VectorT
        {
            const auto zero = vdupq_n_s32(0);
            const auto pos_one = vdupq_n_s32(1);
            const auto neg_one = vdupq_n_s32(-1);
            
            auto gt_mask = vcgtq_s32(v, zero); // v > 0
            auto lt_mask = vcltq_s32(v, zero); // v < 0
            
            auto pos_values = vandq_s32(vreinterpretq_s32_u32(gt_mask), pos_one);
            auto neg_values = vandq_s32(vreinterpretq_s32_u32(lt_mask), neg_one);
            
            return vorrq_s32(pos_values, neg_values);
        }

        template <unsigned int = 0>
        inline static constexpr auto iota(ScalarT start) noexcept -> VectorT
        {
            const int32_t vals[4] = {start, start + 1, start + 2, start + 3};
            return vld1q_s32(vals);
        }

        template <unsigned int = 0>
        inline static auto create_tail_mask(int remainder) noexcept -> VectorT
        {
            alignas(16) static constexpr int32_t indices[4] = {0, 1, 2, 3};
            const int32x4_t v_indices   = vld1q_s32(indices);
            const int32x4_t v_remainder = vdupq_n_s32(remainder);
            return vreinterpretq_s32_u32(vcgtq_s32(v_remainder, v_indices));
        }

        template <unsigned int = 0>
        inline static auto load_unaligned(const ScalarT *const i) noexcept -> VectorT
        {
            return vld1q_s32((const int32_t *const)i);
        }

        template <unsigned int = 0>
        inline static constexpr auto xor_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(l), vreinterpretq_u32_s32(r)));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(l, r)));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vcgeq_s32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vcltq_s32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return vreinterpretq_s32_u32(vcleq_s32(l, r));
        }

        template <unsigned int = 0>
        inline static constexpr auto div(VectorT l, VectorT r) noexcept -> VectorT
        {
            float32x4_t l_float   = vcvtq_f32_s32(l);
            float32x4_t r_float   = vcvtq_f32_s32(r);
            uint32x4_t  r_is_zero = vceqq_f32(r_float, vdupq_n_f32(0.0f));
            float32x4_t r_safe    = vbslq_f32(r_is_zero, vdupq_n_f32(1.0f), r_float);
            float32x4_t result    = vdivq_f32(l_float, r_safe);
            return vcvtq_s32_f32(result);  // truncate toward zero (matches _mm256_cvttps_epi32)
        }

        template <typename OtherVectorT>
        inline static constexpr auto convert(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, float32x4_t>)
            {
                return vcvtq_f32_s32(v);
            }
            else
            {
                static_assert(sizeof(OtherVectorT) == 0, "Invalid convert type!");
            }
        }

        template <typename = void>
        inline static auto scatter(VectorT values, ScalarT *base, int32x4_t idxs) noexcept -> void
        {
            alignas(16) int32_t vals[4];
            alignas(16) int32_t inds[4];
            vst1q_s32(vals, values);
            vst1q_s32(inds, idxs);
            base[inds[0]] = vals[0];
            base[inds[1]] = vals[1];
            base[inds[2]] = vals[2];
            base[inds[3]] = vals[3];
        }

        template <unsigned int = 0>
        inline static constexpr auto max(VectorT v, VectorT other) noexcept -> VectorT
        {
            return vmaxq_s32(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto clamp(VectorT v, VectorT lower, VectorT upper) noexcept -> VectorT
        {
            return vminq_s32(vmaxq_s32(v, lower), upper);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmax(VectorT v) noexcept -> ScalarT
        {
            int32x2_t max1 = vpmax_s32(vget_low_s32(v), vget_high_s32(v));
            int32x2_t max2 = vpmax_s32(max1, max1);
            return vget_lane_s32(max2, 0);
        }

        template <unsigned int = 0>
        inline static constexpr auto any(VectorT v) noexcept -> bool
        {
            return mask(v) != 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto all(VectorT v) noexcept -> bool
        {
            return vminvq_u32(vreinterpretq_u32_s32(v)) == 0xFFFFFFFFu;
        }

        template <unsigned int = 0>
        inline static constexpr auto none(VectorT v) noexcept -> bool
        {
            return mask(v) == 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto mod(VectorT v, VectorT divisor) noexcept -> VectorT
        {
            float32x4_t v_float   = vcvtq_f32_s32(v);
            float32x4_t d_float   = vcvtq_f32_s32(divisor);
            uint32x4_t  d_is_zero = vceqq_f32(d_float, vdupq_n_f32(0.0f));
            float32x4_t d_safe    = vbslq_f32(d_is_zero, vdupq_n_f32(1.0f), d_float);
            float32x4_t quotient  = vrndmq_f32(vdivq_f32(v_float, d_safe));  // floor
            float32x4_t product   = vmulq_f32(quotient, d_safe);
            float32x4_t result    = vsubq_f32(v_float, product);
            return vcvtq_s32_f32(result);
        }

        template <unsigned int = 0>
        inline static constexpr auto neg(VectorT l) noexcept -> VectorT
        {
            return vnegq_s32(l);
        }

        inline static auto hsum(VectorT v) noexcept -> ScalarT
        {
            return vaddvq_s32(v);
        }

    };
}  // namespace vamp
