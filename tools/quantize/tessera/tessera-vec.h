#pragma once

//
// tessera-vec.h
//
// Thin vector-math shim. On macOS dispatches to Accelerate/vDSP;
// on Linux to OpenBLAS cblas_s*; fallback is naive scalar loops.
// Only the surface actually used by the Tessera quantizer is ported.
//

#include <cstdint>
#include <cstddef>

// reductions (return scalar)
float ts_vec_meanv(const float * x, int64_t n);
float ts_vec_measqv(const float * x, int64_t n);   // mean of squares
float ts_vec_maxv(const float * x, int64_t n);
float ts_vec_minv(const float * x, int64_t n);
float ts_vec_sve(const float * x, int64_t n);      // sum
float ts_vec_dotpr(const float * a, const float * b, int64_t n);
float ts_vec_norm2(const float * x, int64_t n);    // sqrt(dotpr(x,x))

// elementwise (write into out, may alias x)
void ts_vec_vsmul(const float * x, float scalar, float * out, int64_t n);
void ts_vec_vmul(const float * a, const float * b, float * out, int64_t n);
void ts_vec_vadd(const float * a, const float * b, float * out, int64_t n);
void ts_vec_vsub(const float * a, const float * b, float * out, int64_t n);
void ts_vec_scale(float * x, float scalar, int64_t n);  // in-place
float ts_vec_maxabs(const float * x, int64_t n);        // max |x|
float ts_vec_meanabs(const float * x, int64_t n);       // mean(|x|)

// matrix ops (row-major, M rows x K cols)
void ts_mat_scale_cols(const float * W, const float * scale, float * out,
                       int64_t rows, int64_t cols);  // out[r,c] = W[r,c] * scale[c]
void ts_mat_mul(const float * A, const float * B, float * C,
                int64_t M, int64_t K, int64_t N);  // C = A @ B
void ts_mat_mul_at(const float * A, const float * B, float * C,
                   int64_t M, int64_t K, int64_t N);  // C = A^T @ B
