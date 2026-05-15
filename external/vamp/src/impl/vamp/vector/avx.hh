#pragma once

#if not defined(__x86_64__)
#error "Tried to compile x86 intrinsics on non-x86 platform!"
#endif

#include <vamp/vector/interface.hh>

#include <immintrin.h>
#include <limits>
#include <cfloat>

namespace vamp
{
    template <>
    struct SIMDVector<__m256>
    {
        using VectorT = __m256;
        using ScalarT = float;
        static constexpr std::size_t VectorWidth = 8;
        static constexpr std::size_t Alignment = 32;
        inline static __m256 EPSILON_VECTOR = _mm256_set1_ps(FLT_MIN);

        template <unsigned int = 0>
        inline static constexpr auto constant(ScalarT v) noexcept -> VectorT
        {
            return _mm256_set1_ps(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto iota(ScalarT start) noexcept -> VectorT
        {
            return _mm256_set_ps(
                start + 7.0f,
                start + 6.0f,
                start + 5.0f,
                start + 4.0f,
                start + 3.0f,
                start + 2.0f,
                start + 1.0f,
                start
            );
        }

        template <unsigned int = 0>
        inline static auto create_tail_mask(int remainder) noexcept -> VectorT
        {
            alignas(32) static constexpr int indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            const __m256i v_indices = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
        
            const __m256i v_remainder = _mm256_set1_epi32(remainder);

            return _mm256_castsi256_ps(_mm256_cmpgt_epi32(v_remainder, v_indices));
        }

        template <unsigned int = 0>
        inline static constexpr auto load(const ScalarT *const f) noexcept -> VectorT
        {
            return _mm256_load_ps(f);
        }

        template <unsigned int = 0>
        inline static constexpr auto load_unaligned(const ScalarT *const f) noexcept -> VectorT
        {
            return _mm256_loadu_ps(f);
        }

        template <unsigned int = 0>
        inline static constexpr auto store(ScalarT *f, VectorT v) noexcept -> void
        {
            _mm256_store_ps(f, v);
        }

        template <unsigned int = 0>
        inline static constexpr auto store_unaligned(ScalarT *f, VectorT v) noexcept -> void
        {
            _mm256_storeu_ps(f, v);
        }

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            return v[idx];
        }

        template <unsigned int = 0>
        inline static constexpr auto broadcast(VectorT v, int idx) noexcept -> VectorT
        {
            return _mm256_permutevar8x32_ps(v, _mm256_set1_epi32(idx));
        }

        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            return _mm256_xor_ps(
                l, _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(l), _mm256_castps_si256(l))));
        }

        // NOTE: Dummy parameter because otherwise we get constexpr errors with set1_ps...
        template <unsigned int = 0>
        inline static constexpr auto neg(VectorT l) noexcept -> VectorT
        {
            return _mm256_xor_ps(l, _mm256_set1_ps(-0.0));
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_add_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_sub_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_mul_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_LE_OQ);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_LT_OQ);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_GE_OQ);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_GT_OQ);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_EQ_OQ);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmp_ps(l, r, _CMP_NEQ_OQ);
        }

        // NOTE: Dummy parameter because otherwise we get constexpr errors with set1_ps...
        template <unsigned int = 0>
        inline static auto floor(VectorT v) noexcept -> VectorT
        {
            return _mm256_floor_ps(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto div(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_div_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto rcp(VectorT l) noexcept -> VectorT
        {
            return _mm256_rcp_ps(l);
        }

        template <unsigned int = 0>
        inline static constexpr auto mask(VectorT v) noexcept -> unsigned int
        {
            return _mm256_movemask_ps(v);
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return _mm256_setzero_ps();
        }

        template <unsigned int = 0>
        inline static constexpr auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            return _mm256_testz_ps(l, r);
        }

        // NOTE: Dummy parameter because otherwise we get constexpr errors with set1_ps...
        template <unsigned int = 0>
        inline static constexpr auto abs(VectorT v) noexcept -> VectorT
        {
            const auto abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
            return _mm256_and_ps(v, abs_mask);
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_and_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_or_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto xor_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_xor_ps(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto sqrt(VectorT v) noexcept -> VectorT
        {
            const VectorT v_clamped = _mm256_max_ps(v, EPSILON_VECTOR);

            const VectorT rsqrt_v = _mm256_rsqrt_ps(v_clamped);

            return _mm256_mul_ps(v_clamped, rsqrt_v);
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(v), i));
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(v), i));
        }

        template <unsigned int = 0>
        inline static constexpr auto clamp(VectorT v, VectorT lower, VectorT upper) noexcept -> VectorT
        {
            return _mm256_min_ps(_mm256_max_ps(v, lower), upper);
        }

        template <unsigned int = 0>
        inline static constexpr auto max(VectorT v, VectorT other) noexcept -> VectorT
        {
            return _mm256_max_ps(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto hsum(VectorT v) noexcept -> ScalarT
        {
            auto vhigh = _mm256_extractf128_ps(v, 1);
            auto vlow = _mm256_castps256_ps128(v);
            auto sum_1 = _mm_add_ps(vhigh, vlow);
            auto shuf_1 = _mm_castpd_ps(_mm_permute_pd(_mm_castps_pd(sum_1), 0b01));
            auto sum_2 = _mm_add_ps(sum_1, shuf_1);
            auto shuf_2 = _mm_movehdup_ps(sum_2);
            auto sum_3 = _mm_add_ps(sum_2, shuf_2);
            return _mm_cvtss_f32(sum_3);
        }

        template <unsigned int = 0>
        inline static constexpr auto min(VectorT v, VectorT other) noexcept -> VectorT
        {
            return _mm256_min_ps(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmin(VectorT v) noexcept -> ScalarT
        {
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            __m128 min1 = _mm_min_ps(vlow, vhigh);
            
            __m128 min2 = _mm_permute_ps(min1, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 min3 = _mm_min_ps(min1, min2);
            __m128 min4 = _mm_permute_ps(min3, _MM_SHUFFLE(0, 1, 0, 1));
            __m128 min5 = _mm_min_ps(min3, min4);
            
            return _mm_cvtss_f32(min5);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmax(VectorT v) noexcept -> ScalarT
        {
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            __m128 max1 = _mm_max_ps(vlow, vhigh);
            
            __m128 max2 = _mm_permute_ps(max1, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 max3 = _mm_max_ps(max1, max2);
            __m128 max4 = _mm_permute_ps(max3, _MM_SHUFFLE(0, 1, 0, 1));
            __m128 max5 = _mm_max_ps(max3, max4);
            
            return _mm_cvtss_f32(max5);
        }

        template <unsigned int = 0>
        inline static constexpr auto mod(VectorT v, VectorT divisor) noexcept -> VectorT
        {
            auto quotient = _mm256_floor_ps(_mm256_div_ps(v, divisor));
            return _mm256_sub_ps(v, _mm256_mul_ps(quotient, divisor));
        }

        template <unsigned int = 0>
        inline static constexpr auto blend(VectorT a, VectorT b, VectorT blend_mask) noexcept -> VectorT
        {
            return _mm256_blendv_ps(a, b, blend_mask);
        }

        template <unsigned int blend_mask>
        inline static constexpr auto blend_constant(VectorT a, VectorT b) noexcept -> VectorT
        {
            return _mm256_blend_ps(a, b, blend_mask);
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256i>)
            {
                return _mm256_cvtps_epi32(v);
            }
            else if constexpr (std::is_same_v<OtherVectorT, VectorT>)
            {
                return v;
            }
            else
            {
                static_assert("Invalid cast-to type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256i>)
            {
                return _mm256_cvtepi32_ps(v);
            }
            else
            {
                static_assert("Invalid cast-from type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256i>)
            {
                return _mm256_castps_si256(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto convert(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256i>)
            {
                return _mm256_cvttps_epi32(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static auto map_to_range(OtherVectorT v) -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256i>)
            {
                const auto v_1 = _mm256_and_si256(v, _mm256_set1_epi32(1));
                const auto v1_f = _mm256_cvtepi32_ps(v_1);
                const auto v_scaled = _mm256_add_ps(_mm256_cvtepi32_ps(v), v1_f);
                return _mm256_mul_ps(
                    v_scaled,
                    _mm256_set1_ps(1.F / static_cast<float>(std::numeric_limits<unsigned int>::max())));
            }
            else
            {
                static_assert("Invalid range-map type!");
            }
        }

        template <typename = void>
        inline static constexpr auto gather(__m256i idxs, const ScalarT *base) noexcept -> VectorT
        {
            return _mm256_i32gather_ps(base, idxs, sizeof(ScalarT));
        }

        template <typename = void>
        inline static constexpr auto
        gather_select(__m256i idxs, VectorT mask, VectorT alternative, const ScalarT *base) noexcept
            -> VectorT
        {
            return _mm256_mask_i32gather_ps(alternative, base, idxs, mask, sizeof(ScalarT));
        }

        template <typename = void>
        inline static constexpr auto scatter(VectorT values, ScalarT *base, __m256i idxs) noexcept -> void
        {
            alignas(32) float vals[8];
            alignas(32) int indices[8];
            _mm256_store_ps(vals, values);
            _mm256_store_si256(reinterpret_cast<__m256i*>(indices), idxs);
            
            // Unrolled for clarity (Compiler does this anyway)
            base[indices[0]] = vals[0];
            base[indices[1]] = vals[1];
            base[indices[2]] = vals[2];
            base[indices[3]] = vals[3];
            base[indices[4]] = vals[4];
            base[indices[5]] = vals[5];
            base[indices[6]] = vals[6];
            base[indices[7]] = vals[7];
        }

        template <unsigned int = 0>
        inline static constexpr auto select(VectorT mask, VectorT when_true, VectorT when_false) noexcept -> VectorT
        {
            return _mm256_blendv_ps(when_false, when_true, mask);
        }

        template <unsigned int = 0>
        inline static constexpr auto sign(VectorT v) noexcept -> VectorT
        {
            const auto zero = _mm256_setzero_ps();
            const auto pos_one = _mm256_set1_ps(1.0f);
            const auto neg_one = _mm256_set1_ps(-1.0f);
            
            auto gt_mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ); // v > 0
            auto lt_mask = _mm256_cmp_ps(v, zero, _CMP_LT_OQ); // v < 0
            
            auto result = _mm256_and_ps(gt_mask, pos_one);
            auto neg_values = _mm256_and_ps(lt_mask, neg_one);
            
            return _mm256_or_ps(result, neg_values);
        }

        template <unsigned int = 0>
        inline static constexpr auto pow_vector(VectorT v, float exponent) noexcept -> VectorT
        {
            alignas(32) float temp[8];
            _mm256_store_ps(temp, v);
            
            for (int i = 0; i < 8; ++i) {
                temp[i] = std::pow(temp[i], exponent);
            }
            
            return _mm256_load_ps(temp);
        }

        template <unsigned int = 0>
        inline static constexpr auto atan2_approx(VectorT y, VectorT x) noexcept -> VectorT
        {
            const auto zero = _mm256_setzero_ps();
            const auto pi = _mm256_set1_ps((float)M_PI);
            const auto pi_half = _mm256_set1_ps(M_PI_2);
        
            const auto C1 = _mm256_set1_ps(-0.0464964749f);
            const auto C2 = _mm256_set1_ps( 0.15931422f);
            const auto C3 = _mm256_set1_ps(-0.327622764f);
        
            auto abs_y = abs(y);
            auto abs_x = abs(x);
        
            auto min_xy = _mm256_min_ps(abs_x, abs_y);
            auto max_xy = _mm256_max_ps(abs_x, abs_y);
            auto a = _mm256_div_ps(min_xy, max_xy);
            auto s = _mm256_mul_ps(a, a);
        
            auto r = _mm256_fmadd_ps(C1, s, C2);   // C1*s + C2
            r = _mm256_fmadd_ps(r, s, C3);           // (prev)*s + C3
            r = _mm256_mul_ps(r, s);
            r = _mm256_mul_ps(r, a);
            r = _mm256_add_ps(r, a);
        
            auto y_gt_x = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
            r = _mm256_blendv_ps(r, _mm256_sub_ps(pi_half, r), y_gt_x);
        
            auto x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
            auto y_neg = _mm256_cmp_ps(y, zero, _CMP_LT_OQ);
        
            // x<0, y>=0 -> r = pi - r
            auto mask1 = _mm256_andnot_ps(y_neg, x_neg);
            r = _mm256_blendv_ps(r, _mm256_sub_ps(pi, r), mask1);
        
            // x<0, y<0 -> r = r - pi
            auto mask2 = _mm256_and_ps(x_neg, y_neg);
            r = _mm256_blendv_ps(r, _mm256_sub_ps(r, pi), mask2);

            // x>=0, y<0 -> r = -r
            auto mask3 = _mm256_andnot_ps(x_neg, y_neg);
            r = _mm256_blendv_ps(r, _mm256_sub_ps(zero, r), mask3);
        
            return r;
        }

        inline static const __m256 SIGN_MASK = _mm256_set1_ps(-0.0f);

        inline static void sincos(VectorT x, VectorT& out_sin, VectorT& out_cos) noexcept
        {
            const __m256 PI = _mm256_set1_ps(3.14159265359f);
            const __m256 HALF_PI = _mm256_set1_ps(1.57079632679f);
            const __m256 TWO_PI = _mm256_set1_ps(6.28318530718f);
            
            __m256 v = _mm256_add_ps(x, HALF_PI);
            __m256 mask = _mm256_cmp_ps(v, PI, _CMP_GE_OQ);
            __m256 x_cos = _mm256_sub_ps(v, _mm256_and_ps(mask, TWO_PI));

            __m256 x_sin = x;
            __m256 x_sin_abs = _mm256_andnot_ps(SIGN_MASK, x_sin);
            __m256 x_cos_abs = _mm256_andnot_ps(SIGN_MASK, x_cos);

            const __m256 C_s_vsq = _mm256_set1_ps(-0.478637850138f);
            const __m256 C_s_v = _mm256_set1_ps( 1.503684069359f);
            const __m256 C_d1 = _mm256_set1_ps( 0.665200679751f);
            const __m256 C_d2 = _mm256_set1_ps( 0.140024078368f);
            const __m256 C_d3 = _mm256_set1_ps( 0.011596870476f);

            __m256 v_sq_sin = _mm256_mul_ps(x_sin, x_sin_abs);
            __m256 v_sq_cos = _mm256_mul_ps(x_cos, x_cos_abs);

            // x * |x|
            __m256 p_sin = _mm256_fmadd_ps(v_sq_sin, C_s_vsq, _mm256_mul_ps(x_sin, C_s_v));
            __m256 p_cos = _mm256_fmadd_ps(v_sq_cos, C_s_vsq, _mm256_mul_ps(x_cos, C_s_v));

            // s_vsq + s_v (FMA)
            __m256 abs_p_sin = _mm256_andnot_ps(SIGN_MASK, p_sin);
            __m256 abs_p_cos = _mm256_andnot_ps(SIGN_MASK, p_cos);
            __m256 p2_sin = _mm256_mul_ps(p_sin, abs_p_sin);
            __m256 p2_cos = _mm256_mul_ps(p_cos, abs_p_cos);
            __m256 p3_sin = _mm256_mul_ps(p2_sin, abs_p_sin);
            __m256 p3_cos = _mm256_mul_ps(p2_cos, abs_p_cos);

            // out = p3 * C_d3 + p2 * C_d2 + p * C_d1
            out_sin = _mm256_fmadd_ps(p3_sin, C_d3, _mm256_fmadd_ps(p2_sin, C_d2, _mm256_mul_ps(p_sin, C_d1)));
            out_cos = _mm256_fmadd_ps(p3_cos, C_d3, _mm256_fmadd_ps(p2_cos, C_d2, _mm256_mul_ps(p_cos, C_d1)));
        }

    };

    template <>
    struct SIMDVector<__m256i>
    {
        using VectorT = __m256i;
        using ScalarT = int;
        static constexpr std::size_t VectorWidth = 8;
        static constexpr std::size_t Alignment = 32;

        template <unsigned int = 0>
        inline static constexpr auto constant(ScalarT v) noexcept -> VectorT
        {
            return _mm256_set1_epi32(v);
        }

        template <unsigned int = 0>
        inline static constexpr auto iota(ScalarT start) noexcept -> VectorT
        {
            return _mm256_set_epi32(
                start + 7,
                start + 6,
                start + 5,
                start + 4,
                start + 3,
                start + 2,
                start + 1,
                start
            );
        }

        template <unsigned int = 0>
        inline static auto create_tail_mask(int remainder) noexcept -> VectorT
        {
            alignas(32) static constexpr int indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            const __m256i v_indices = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
        
            // v_remainder = [remainder, remainder, ..., remainder]
            const __m256i v_remainder = _mm256_set1_epi32(remainder);
        
            // result = (v_remainder > v_indices) ? 0xFFFFFFFF : 0
            return _mm256_cmpgt_epi32(v_remainder, v_indices);
        }

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            // Awful, but so is extracting an int in AVX2
            switch (idx)
            {
                case 0:
                    return _mm256_extract_epi32(v, 0);
                case 1:
                    return _mm256_extract_epi32(v, 1);
                case 2:
                    return _mm256_extract_epi32(v, 2);
                case 3:
                    return _mm256_extract_epi32(v, 3);
                case 4:
                    return _mm256_extract_epi32(v, 4);
                case 5:
                    return _mm256_extract_epi32(v, 5);
                case 6:
                    return _mm256_extract_epi32(v, 6);
                case 7:
                    return _mm256_extract_epi32(v, 7);
                default:
                    return 0;
            };
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_sub_epi32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_add_epi32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            // NOTE: Kinda slow. Note footgun with overflow to 64 bits, as well as footgun with
            // "normal" intmul
            return _mm256_mullo_epi32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto div(VectorT l, VectorT r) noexcept -> VectorT
        {
            __m256 l_float = _mm256_cvtepi32_ps(l);
            __m256 r_float = _mm256_cvtepi32_ps(r);
            
            __m256 r_safe = _mm256_blendv_ps(
                r_float, 
                _mm256_set1_ps(1.0f), 
                _mm256_cmp_ps(r_float, _mm256_setzero_ps(), _CMP_EQ_OQ)
            );
            
            __m256 result_float = _mm256_div_ps(l_float, r_safe);

            return _mm256_cvttps_epi32(result_float);
        }

        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            return _mm256_xor_si256(l, _mm256_cmpeq_epi32(l, l));
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_and_si256(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_or_si256(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto xor_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_xor_si256(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmpeq_epi32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_xor_si256(_mm256_cmpeq_epi32(l, r), _mm256_cmpeq_epi32(l, l));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmpgt_epi32(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_xor_si256(_mm256_cmpgt_epi32(r, l), _mm256_cmpeq_epi32(l, l));
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_cmpgt_epi32(r, l);
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return _mm256_xor_si256(_mm256_cmpgt_epi32(l, r), _mm256_cmpeq_epi32(l, l));
        }

        template <unsigned int = 0>
        inline static constexpr auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            return _mm256_testz_si256(l, r);
        }

        template <unsigned int = 0>
        inline static constexpr auto load(const ScalarT *const i) noexcept -> VectorT
        {
            return _mm256_loadu_si256((const __m256i *const)i);
        }

        template <unsigned int = 0>
        inline static constexpr auto load_unaligned(const ScalarT *const i) noexcept -> VectorT
        {
            return _mm256_loadu_si256((const __m256i *const)i);
        }

        template <unsigned int = 0>
        inline static constexpr auto store(ScalarT *f, VectorT v) noexcept -> void
        {
            _mm256_store_si256(reinterpret_cast<VectorT *>(f), v);
        }

        template <unsigned int = 0>
        inline static constexpr auto store_unaligned(ScalarT *f, VectorT v) noexcept -> void
        {
            _mm256_storeu_si256(reinterpret_cast<VectorT *>(f), v);
        }

        template <unsigned int = 0>
        inline static constexpr auto mask(VectorT v) noexcept -> unsigned int
        {
            // HACK: This will create more FP port contention. We could use _mm256_movemask_epi8, but this has
            // poor latency and would require a change to the logic for all_true
            return _mm256_movemask_ps(_mm256_castsi256_ps(v));
        }

        // TODO: Figure out how to support vector shifting; currently causes a template deduction error
        // inline static constexpr auto shift_left(VectorT v, VectorT i) noexcept -> VectorT
        // {
        //     return _mm256_sllv_epi32(v, i);
        // }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return _mm256_slli_epi32(v, i);
        }

        // inline static constexpr auto shift_right(VectorT v, VectorT i) noexcept -> VectorT
        // {
        //     return _mm256_srlv_epi32(v, i);
        // }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return _mm256_srli_epi32(v, i);
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return _mm256_setzero_si256();
        }

        template <typename = void>
        inline static constexpr auto gather(__m256i idxs, const ScalarT *base) noexcept -> VectorT
        {
            return _mm256_i32gather_epi32(base, idxs, sizeof(ScalarT));
        }

        template <typename = void>
        inline static constexpr auto
        gather_select(__m256i idxs, VectorT mask, VectorT alternative, const ScalarT *base) noexcept
            -> VectorT
        {
            return _mm256_mask_i32gather_epi32(alternative, base, idxs, mask, sizeof(ScalarT));
        }

        template <typename = void>
        inline static constexpr auto scatter(VectorT values, ScalarT *base, __m256i idxs) noexcept -> void
        {
            alignas(32) int vals[8];
            alignas(32) int indices[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals), values);
            _mm256_store_si256(reinterpret_cast<__m256i*>(indices), idxs);
            
            // Unrolled for clarity (Compiler does this anyway)
            base[indices[0]] = vals[0];
            base[indices[1]] = vals[1];
            base[indices[2]] = vals[2];
            base[indices[3]] = vals[3];
            base[indices[4]] = vals[4];
            base[indices[5]] = vals[5];
            base[indices[6]] = vals[6];
            base[indices[7]] = vals[7];
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256>)
            {
                return _mm256_cvtepi32_ps(v);
            }
            else if constexpr (std::is_same_v<OtherVectorT, VectorT>)
            {
                return v;
            }
            else
            {
                static_assert("Invalid cast-to type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            {
                if constexpr (std::is_same_v<OtherVectorT, __m256>)
                {
                    return _mm256_cvtps_epi32(v);
                }
                else
                {
                    static_assert("Invalid cast-from type!");
                }
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256>)
            {
                return _mm256_castsi256_ps(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto convert(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, __m256>)
            {
                return _mm256_cvtepi32_ps(v);
            }
            else
            {
                static_assert("Invalid cast-as type!");
            }
        }

        template <unsigned int = 0>
        inline static constexpr auto select(VectorT mask, VectorT when_true, VectorT when_false) noexcept -> VectorT
        {
            return _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(when_false),
                _mm256_castsi256_ps(when_true),
                _mm256_castsi256_ps(mask)));
        }

        template <unsigned int = 0>
        inline static constexpr auto min(VectorT v, VectorT other) noexcept -> VectorT
        {
            return _mm256_min_epi32(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto max(VectorT v, VectorT other) noexcept -> VectorT
        {
            return _mm256_max_epi32(v, other);
        }

        template <unsigned int = 0>
        inline static constexpr auto clamp(VectorT v, VectorT lower, VectorT upper) noexcept -> VectorT
        {
            return _mm256_min_epi32(_mm256_max_epi32(v, lower), upper);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmin(VectorT v) noexcept -> ScalarT
        {
            __m256 v_float = _mm256_cvtepi32_ps(v);

            __m128 vlow = _mm256_castps256_ps128(v_float);
            __m128 vhigh = _mm256_extractf128_ps(v_float, 1);
            __m128 min1 = _mm_min_ps(vlow, vhigh);
            __m128 min2 = _mm_permute_ps(min1, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 min3 = _mm_min_ps(min1, min2);
            __m128 min4 = _mm_permute_ps(min3, _MM_SHUFFLE(0, 1, 0, 1));
            __m128 min5 = _mm_min_ps(min3, min4);

            return (int)_mm_cvtss_f32(min5);
        }

        template <unsigned int = 0>
        inline static constexpr auto hmax(VectorT v) noexcept -> ScalarT
        {
            __m256 v_float = _mm256_cvtepi32_ps(v);
            
            __m128 vlow = _mm256_castps256_ps128(v_float);
            __m128 vhigh = _mm256_extractf128_ps(v_float, 1);
            __m128 max1 = _mm_max_ps(vlow, vhigh);
            __m128 max2 = _mm_permute_ps(max1, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 max3 = _mm_max_ps(max1, max2);
            __m128 max4 = _mm_permute_ps(max3, _MM_SHUFFLE(0, 1, 0, 1));
            __m128 max5 = _mm_max_ps(max3, max4);
            
            return (int)_mm_cvtss_f32(max5);
        }

        template <unsigned int = 0>
        inline static constexpr auto sign(VectorT v) noexcept -> VectorT
        {
            const auto zero = _mm256_setzero_si256();
            const auto pos_one = _mm256_set1_epi32(1);
            const auto neg_one = _mm256_set1_epi32(-1);
            
            auto gt_mask = _mm256_cmpgt_epi32(v, zero); // v > 0
            auto lt_mask = _mm256_cmpgt_epi32(zero, v); // 0 > v
            
            auto pos_values = _mm256_and_si256(gt_mask, pos_one);
            auto neg_values = _mm256_and_si256(lt_mask, neg_one);
            
            return _mm256_or_si256(pos_values, neg_values);
        }

        template <unsigned int = 0>
        inline static constexpr auto any(VectorT v) noexcept -> bool
        {
            return _mm256_testz_si256(v, v) == 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto all(VectorT v) noexcept -> bool
        {
            auto all_ones = _mm256_cmpeq_epi32(v, v);
            return _mm256_testc_si256(v, all_ones) != 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto none(VectorT v) noexcept -> bool
        {
            return _mm256_testz_si256(v, v) != 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto mod(VectorT v, VectorT divisor) noexcept -> VectorT
        {
            __m256 v_float = _mm256_cvtepi32_ps(v);
            __m256 divisor_float = _mm256_cvtepi32_ps(divisor);
            
            __m256 divisor_safe = _mm256_blendv_ps(
                divisor_float, 
                _mm256_set1_ps(1.0f), 
                _mm256_cmp_ps(divisor_float, _mm256_setzero_ps(), _CMP_EQ_OQ)
            );
            
            __m256 quotient = _mm256_floor_ps(_mm256_div_ps(v_float, divisor_safe));
            __m256 product = _mm256_mul_ps(quotient, divisor_safe);
            __m256 result_float = _mm256_sub_ps(v_float, product);
            
            return _mm256_cvtps_epi32(result_float);
        }

        // Unary negation: -x  (two's complement, element-wise)
        template <unsigned int = 0>
        inline static constexpr auto neg(VectorT l) noexcept -> VectorT
        {
            return _mm256_sub_epi32(_mm256_setzero_si256(), l);
        }

        // Horizontal sum of 8 x int32 lanes
        inline static auto hsum(VectorT v) noexcept -> ScalarT
        {
            // Pair-wise horizontal add: 8 → 4 → 2 (within each 128-bit lane)
            __m256i t = _mm256_hadd_epi32(v, v);   // [a+b, c+d, a+b, c+d | e+f, g+h, e+f, g+h]
            t         = _mm256_hadd_epi32(t, t);   // [a+b+c+d, ..., e+f+g+h, ...]
            // Extract the sums from the two 128-bit lanes and add
            return _mm256_extract_epi32(t, 0) + _mm256_extract_epi32(t, 4);
        }
    };
}  // namespace vamp
