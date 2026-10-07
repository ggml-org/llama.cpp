//******************************************************************************
// Solve Triangular F32 Kernel
// Forward substitution: solve AX = B where A is lower-triangular.
//
// src0 (A): [n, n, B1, B2]  lower-triangular matrix
// src1 (B): [k, n, B1, B2]  right-hand side
// dst  (X): [k, n, B1, B2]  solution
//
// For each column j (parallelized across threads):
//   For i = 0..n-1:
//     X[i,j] = (B[i,j] - dot(A[i,0..i-1], X[0..i-1,j])) / A[i,i]
//
// Lower-triangular, left-side, non-unit variant implemented.
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_solve_tri_params {
    struct ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    struct ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    struct ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

int entry_point(struct ggml_et_solve_tri_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor * src0 = &params->src0;  // A
    struct ggml_tensor * src1 = &params->src1;  // B
    struct ggml_tensor * dst  = &params->dst;   // X

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * A_data = (const float *) src0->data;
    const float * B_data = (const float *) src1->data;
    float *       X_data = (float *) dst->data;

    if (!A_data || !B_data || !X_data) {
        return -1;
    }

    const int64_t n   = src0->ne[1];  // A is n×n
    const int64_t k   = src1->ne[0];  // number of RHS columns
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    // Strides in bytes
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    const int64_t cols_per_cl    = 16;
    const int64_t num_col_groups = k / cols_per_cl;
    const int64_t total_work     = num_col_groups * ne2 * ne3;

    for (int64_t work = thread_id; work < total_work; work += num_threads) {
        const int64_t cg = work % num_col_groups;
        const int64_t i2 = (work / num_col_groups) % ne2;
        const int64_t i3 = work / (num_col_groups * ne2);

        const int64_t j_start = cg * cols_per_cl;
        const int64_t j_end   = j_start + cols_per_cl;

        const float * A_batch = (const float *) ((const char *) A_data + i2 * nb02 + i3 * nb03);
        const float * B_batch = (const float *) ((const char *) B_data + i2 * nb12 + i3 * nb13);
        float *       X_batch = (float *) ((char *) X_data + i2 * nb2 + i3 * nb3);

        for (int64_t j = j_start; j < j_end; j++) {
            for (int64_t i = 0; i < n; i++) {
                const float * A_row = (const float *) ((const char *) A_batch + i * nb01);
                float *       X_row = (float *) ((char *) X_batch + i * nb1);
                const float * B_row = (const float *) ((const char *) B_batch + i * nb11);

                float sum = 0.0f;
                for (int64_t t = 0; t < i; t++) {
                    const float * X_t = (const float *) ((const char *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    return 0;
}
