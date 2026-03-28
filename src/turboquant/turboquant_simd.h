// TurboQuant SIMD Optimizations
// AVX2/AVX-512 vectorized operations for rotation and quantization

#ifndef TURBOQUANT_SIMD_H
#define TURBOQUANT_SIMD_H

#include <cstdint>
#include <vector>

// Check for SIMD support
#if defined(__AVX512F__)
#define TURBOQUANT_AVX512_AVAILABLE 1
#elif defined(__AVX2__)
#define TURBOQUANT_AVX2_AVAILABLE 1
#endif

namespace turboquant {
namespace simd {

// Runtime CPU feature detection
bool has_avx2();
bool has_avx512();

// Vectorized rotation matrix application
// Q is a dim x dim rotation matrix
// input/output are dim-dimensional vectors
void apply_rotation_scalar(const float* Q, const float* input, float* output, int dim);
void apply_rotation_avx2(const float* Q, const float* input, float* output, int dim);
void apply_rotation_avx512(const float* Q, const float* input, float* output, int dim);

// Auto-dispatch based on CPU capabilities
void apply_rotation_auto(const float* Q, const float* input, float* output, int dim);

} // namespace simd
} // namespace turboquant

#endif // TURBOQUANT_SIMD_H
