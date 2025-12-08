#define GGML_COMMON_DECL_CPP
#include "ggml-ifairy-lut.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

// Scalar reference implementation for iFairy 3-weight LUT path.
// Integration into routing is pending; functions here produce correct scalar outputs.

void ggml_ifairy_lut_init(void) {
    // No global initialization needed yet.
}

void ggml_ifairy_lut_free(void) {
    // No global teardown needed yet.
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if (getenv("GGML_IFAIRY_LUT") && strcmp(getenv("GGML_IFAIRY_LUT"), "0") == 0) {
        return false;
    }

    if (src0->type != GGML_TYPE_IFAIRY || src1->type != GGML_TYPE_IFAIRY_Q16) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    // small-N path (matvec/small batch)
    if (src1->ne[1] > 4) {
        return false;
    }
    return true;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads) {
    (void) src0;
    (void) src1;
    (void) dst;
    (void) n_threads;
    // TODO: compute real workspace when routing is enabled.
    return 0;
}

bool ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out) {
    if (index_tensor_out) {
        *index_tensor_out = NULL;
    }

    (void) tensor;

    // TODO: call ggml_ifairy_3w_encode and create shadow tensor.
    return false;
}

void ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    (void) m; // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    const int64_t K  = k;
    const int64_t K3 = (K + 2) / 3 * 3;
    const int64_t groups = K3 / 3;

    // canonical patterns for idx' = 0..15 (u1, u2)
    static const int8_t u_wr[16] = { 1, 1, 1, 1, 0, 0, 0, 0,-1,-1,-1,-1, 0, 0, 0, 0};
    static const int8_t u_wi[16] = { 0, 1, 0,-1, 1, 1, 1, 1, 0, 1, 0,-1,-1,-1,-1,-1};
    static const int8_t v_wr[16] = { 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0};
    static const int8_t v_wi[16] = { 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1};

    for (int col = 0; col < n; ++col) {
        const uint8_t * act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) act_col_bytes;
        float * scales_out = (float *) lut_scales + (size_t) col * 2;
        // Use first block as per-tensor activation scales (matches current pipeline).
        scales_out[0] = GGML_FP16_TO_FP32(act_blocks[0].d_real);
        scales_out[1] = GGML_FP16_TO_FP32(act_blocks[0].d_imag);

        int8_t * lut_out = (int8_t *) lut_buf + (size_t) col * (size_t) groups * 32;

        for (int64_t g = 0; g < groups; ++g) {
            const int64_t idx0 = g * 3 + 0;
            const int64_t idx1 = g * 3 + 1;
            const int64_t idx2 = g * 3 + 2;

            const int64_t blk0 = idx0 / QK_K;
            const int64_t blk1 = idx1 / QK_K;
            const int64_t blk2 = idx2 / QK_K;
            const int off0 = idx0 % QK_K;
            const int off1 = idx1 % QK_K;
            const int off2 = idx2 % QK_K;

            int xr0 = 0, xi0 = 0;
            int xr1 = 0, xi1 = 0;
            int xr2 = 0, xi2 = 0;

            if (idx0 < K) { xr0 = (int8_t) act_blocks[blk0].x_real[off0]; xi0 = (int8_t) act_blocks[blk0].x_imag[off0]; }
            if (idx1 < K) { xr1 = (int8_t) act_blocks[blk1].x_real[off1]; xi1 = (int8_t) act_blocks[blk1].x_imag[off1]; }
            if (idx2 < K) { xr2 = (int8_t) act_blocks[blk2].x_real[off2]; xi2 = (int8_t) act_blocks[blk2].x_imag[off2]; }

            int8_t * real_tbl = lut_out + (size_t) g * 32 + 0;
            int8_t * imag_tbl = lut_out + (size_t) g * 32 + 16;

            for (int idx = 0; idx < 16; ++idx) {
                const int wr1 = u_wr[idx];
                const int wi1 = u_wi[idx];
                const int wr2 = v_wr[idx];
                const int wi2 = v_wi[idx];

                int real = xr0;
                int imag = xi0;

                real += wr1 * xr1 - wi1 * xi1;
                imag += wr1 * xi1 + wi1 * xr1;

                real += wr2 * xr2 - wi2 * xi2;
                imag += wr2 * xi2 + wi2 * xr2;

                if (real > 127) real = 127;
                if (real < -127) real = -127;
                if (imag > 127) imag = 127;
                if (imag < -127) imag = -127;

                real_tbl[idx] = (int8_t) real;
                imag_tbl[idx] = (int8_t) imag;
            }
        }
    }
}

void ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_stride) {
    if (!indexes || !lut || !lut_scales || !dst || !qweights) {
        return;
    }

    const int64_t K = k;
    const int64_t K3 = (K + 2) / 3 * 3;
    const int64_t groups = K3 / 3;

    // Use first block as per-tensor weight scales (aligns with current ifairy quantization).
    const block_ifairy * w_blocks = (const block_ifairy *) qweights;
    const float w_scale_r = GGML_FP16_TO_FP32(w_blocks[0].d_real);
    const float w_scale_i = GGML_FP16_TO_FP32(w_blocks[0].d_imag);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const int8_t * lut_base = (const int8_t *) lut + (size_t) col * (size_t) groups * 32;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;
            const float * scales = (const float *) lut_scales + (size_t) col * 2;

            int32_t acc_r = 0;
            int32_t acc_i = 0;

            for (int64_t g = 0; g < groups; ++g) {
                const uint8_t code = idx_row[g];
                const uint8_t idx = code & 0x0f;
                const bool do_swap = (code & 0x80u) != 0;
                const bool neg_r   = (code & 0x20u) != 0;
                const bool neg_i   = (code & 0x40u) != 0;

                const int8_t val_r = lut_base[(size_t) g * 32 + idx];
                const int8_t val_i = lut_base[(size_t) g * 32 + 16 + idx];

                int vr = val_r;
                int vi = val_i;
                if (do_swap) {
                    int tmp = vr;
                    vr = vi;
                    vi = tmp;
                }
                if (neg_r) vr = -vr;
                if (neg_i) vi = -vi;

                acc_r += vr;
                acc_i += vi;
            }

            const float act_scale_r = scales[0];
            const float act_scale_i = scales[1];
            float out_r = (float) acc_r * act_scale_r * w_scale_r;
            float out_i = (float) acc_i * act_scale_i * w_scale_i;

            float * out_ptr = (float *) ((uint8_t *) dst + (size_t) col * dst_stride);
            out_ptr[(size_t) row * 2 + 0] = out_r;
            out_ptr[(size_t) row * 2 + 1] = out_i;
        }
    }
}

void ggml_ifairy_lut_mul_mat_scalar(int m, int k, int n, const void * qweights, const void * act, size_t act_stride, float * dst) {
    if (!qweights || !act || !dst) {
        return;
    }

    const int64_t K = k;
    const int64_t K3 = (K + 2) / 3 * 3;
    const int64_t groups = K3 / 3;

    // workspace: indexes + LUT + scales
    const size_t index_bytes = (size_t) m * (size_t) groups;
    const size_t lut_bytes   = (size_t) n * (size_t) groups * 32;
    const size_t scale_bytes = (size_t) n * 2 * sizeof(float);
    const size_t total_bytes = index_bytes + lut_bytes + scale_bytes;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return;
    }
    uint8_t * buf = (uint8_t *) ptr;
    memset(buf, 0, total_bytes);
    uint8_t * indexes = buf;
    int8_t * lut = (int8_t *) (buf + index_bytes);
    float * scales = (float *) (buf + index_bytes + lut_bytes);

    // build indexes per row
    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes);

    // preprocess activations -> LUT per column
    ggml_ifairy_lut_preprocess(m, k, n, act, act_stride, scales, lut);

    // qgemm
    ggml_ifairy_lut_qgemm(m, k, n, qweights, indexes, lut, scales, dst, (size_t) m * 2 * sizeof(float));

    free(buf);
}
