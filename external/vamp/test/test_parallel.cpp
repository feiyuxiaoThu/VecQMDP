#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <chrono>

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
constexpr double PI_OVER_2 = PI / 2;

// Polynomial coefficients for sine approximation
constexpr double P0sin = -1.66666666666666307295e-1;
constexpr double P1sin = 8.33333333332211858878e-3;
constexpr double P2sin = -1.98412698295895385996e-4;
constexpr double P3sin = 2.75573136213857245213e-6;
constexpr double P4sin = -2.50507477628578072866e-8;
constexpr double P5sin = 1.58962301576546568060e-10;

inline void PrintM256d(__m256d vec)
{
    double values[4];
    _mm256_storeu_pd(values, vec);
    std::cout << "Values: " << values[0] << ", "
              << values[1] << ", "
              << values[2] << ", "
              << values[3] << std::endl;
}

inline __m256d _mm256_fmod_pd(__m256d x, __m256d y) {
    __m256d quotient = _mm256_div_pd(x, y);
    quotient = _mm256_floor_pd(quotient);
    __m256d product = _mm256_mul_pd(quotient, y);
    return _mm256_sub_pd(x, product);
}

inline __m256d normalize(__m256d x) {
    __m256d two_pi = _mm256_set1_pd(TWO_PI);
    x = _mm256_fmod_pd(_mm256_add_pd(x, _mm256_set1_pd(PI)), two_pi);
    x = _mm256_sub_pd(x, _mm256_set1_pd(PI));
    return x;
}

inline __m256d _mm256_sin_pd(__m256d x) {
    x = normalize(x); // Normalize x to [-π, π]
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x3 = _mm256_mul_pd(x2, x);
    __m256d x5 = _mm256_mul_pd(x3, x2);
    __m256d x7 = _mm256_mul_pd(x5, x2);
    __m256d x9 = _mm256_mul_pd(x7, x2);
    __m256d x11 = _mm256_mul_pd(x9, x2);
    __m256d x13 = _mm256_mul_pd(x11, x2);

    const __m256d coeff0 = _mm256_set1_pd(P0sin);
    const __m256d coeff1 = _mm256_set1_pd(P1sin);
    const __m256d coeff2 = _mm256_set1_pd(P2sin);
    const __m256d coeff3 = _mm256_set1_pd(P3sin);
    const __m256d coeff4 = _mm256_set1_pd(P4sin);
    const __m256d coeff5 = _mm256_set1_pd(P5sin);

    __m256d result = _mm256_add_pd(x, _mm256_mul_pd(coeff0, x3));
    result = _mm256_add_pd(result, _mm256_mul_pd(coeff1, x5));
    result = _mm256_add_pd(result, _mm256_mul_pd(coeff2, x7));
    result = _mm256_add_pd(result, _mm256_mul_pd(coeff3, x9));
    result = _mm256_add_pd(result, _mm256_mul_pd(coeff4, x11));
    result = _mm256_add_pd(result, _mm256_mul_pd(coeff5, x13));

    return result;
}

inline double scalar_sin(double x) {
    x = fmod(x + PI, TWO_PI) - PI; // Normalize to [-π, π]
    double x2 = x * x;
    double x3 = x2 * x;
    double x5 = x3 * x2;
    double x7 = x5 * x2;
    double x9 = x7 * x2;
    double x11 = x9 * x2;
    double x13 = x11 * x2;

    return x + P0sin * x3 + P1sin * x5 + P2sin * x7 + P3sin * x9 + P4sin * x11 + P5sin * x13;
}

int main() {
    const int N = 10000; // Number of iterations
    double input[N];
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<double>(i) * (PI / 180.0); // Example inputs in radians
    }

    // AVX test
    auto start_avx = std::chrono::high_resolution_clock::now();
    __m256d result1 = _mm256_set1_pd(0.0);
    for (int i = 0; i < N; i += 4) {
        __m256d vec = _mm256_loadu_pd(&input[i]);
        __m256d result = _mm256_sin_pd(vec);
        result1 = _mm256_add_pd(result1, result);
        // Optionally store or use result
    }
    // std::cout << result1[0] << std::endl;
    auto end_avx = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_avx = end_avx - start_avx;

    // Scalar test
    auto start_scalar = std::chrono::high_resolution_clock::now();
    double result2 = 0;
    for (int i = 0; i < N; ++i) {
        result2 += sin(i);
        // double result = scalar_sin(input[i]);
        // Optionally store or use result
    }
    // std::cout << result2 << std::endl;
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_scalar = end_scalar - start_scalar;

    std::cout << "AVX Duration: " << duration_avx.count() << " seconds" << std::endl;
    std::cout << "Scalar Duration: " << duration_scalar.count() << " seconds" << std::endl;

    return 0;
}

// g++ -O3 -mavx -march=native -std=c++17 -o test_parallel test_parallel.cpp
// g++ -O3 -mavx -march=native -ftree-vectorizer-verbose=2 -std=c++17 test_parallel.cpp -o test_parallel
