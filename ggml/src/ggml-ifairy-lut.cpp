#define GGML_COMMON_DECL_CPP
#include "ggml-ifairy-lut.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#ifndef GGML_FP16_TO_FP32
#define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <mutex>

static std::vector<ifairy_lut_extra *> g_ifairy_lut_extras;
static std::mutex g_ifairy_lut_mutex;

static inline uint8_t ggml_ifairy_read_code_local(const block_ifairy * row_blocks, int64_t idx) {
    const int64_t block_idx    = idx / QK_K;
    const int64_t idx_in_block = idx - block_idx * QK_K;
    const int     chunk        = (int) (idx_in_block >> 6);          // 0..3
    const int     lane         = (int) (idx_in_block & 0x0f);        // 0..15
    const int     part         = (int) ((idx_in_block >> 4) & 0x3);  // 0..3
    const uint8_t packed       = row_blocks[block_idx].qs[chunk * 16 + lane];
    return (packed >> (2 * part)) & 0x3;
}

// Scalar reference implementation for iFairy 3-weight LUT path.
// Integration into routing is pending; functions here produce correct scalar outputs.

void ggml_ifairy_lut_init(void) {
    // No global initialization needed yet.
}

void ggml_ifairy_lut_free(void) {
    // free any extras allocated by transform_tensor
    std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
    for (auto * e : g_ifairy_lut_extras) {
        if (e) {
            if (e->indexes && e->index_tensor == NULL) {
                ggml_aligned_free(e->indexes, e->size);
            }
            delete e;
        }
    }
    g_ifairy_lut_extras.clear();
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const bool dbg = getenv("GGML_IFAIRY_LUT_DEBUG") && strcmp(getenv("GGML_IFAIRY_LUT_DEBUG"), "0") != 0;
    if (getenv("GGML_IFAIRY_LUT") && strcmp(getenv("GGML_IFAIRY_LUT"), "0") == 0) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: disabled by env GGML_IFAIRY_LUT=0\n");
        }
        return false;
    }

    if (src0->type != GGML_TYPE_IFAIRY || (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_IFAIRY_Q16)) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: type mismatch src0=%s src1=%s dst=%s\n",
                          ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type));
        }
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: dst type not F32 (%s)\n", ggml_type_name(dst->type));
        }
        return false;
    }
    // require logical K aligned to block
    if (src0->ne[0] % QK_K != 0 || src1->ne[0] != src0->ne[0]) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: K misaligned K0=%lld K1=%lld QK_K=%d\n",
                          (long long) src0->ne[0], (long long) src1->ne[0], QK_K);
        }
        return false;
    }
    if (dbg) {
        const long long s0n0 = (long long) src0->ne[0];
        const long long s0n1 = (long long) src0->ne[1];
        const long long s0n2 = (long long) src0->ne[2];
        const long long s0n3 = (long long) src0->ne[3];
        const long long s1n0 = (long long) src1->ne[0];
        const long long s1n1 = (long long) src1->ne[1];
        const long long s1n2 = (long long) src1->ne[2];
        const long long s1n3 = (long long) src1->ne[3];
        const long long dn0  = (long long) dst->ne[0];
        const long long dn1  = (long long) dst->ne[1];
        const long long dn2  = (long long) dst->ne[2];
        const long long dn3  = (long long) dst->ne[3];
        GGML_LOG_WARN("ifairy_lut: can_mul_mat=true src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] dst_nb=[%zu,%zu,%zu,%zu]\n",
                      s0n0, s0n1, s0n2, s0n3,
                      s1n0, s1n1, s1n2, s1n3,
                      dn0,  dn1,  dn2,  dn3,
                      dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        fprintf(stderr,
                "[ifairy_lut_dbg] src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] dst_nb=[%zu,%zu,%zu,%zu]\n",
                s0n0, s0n1, s0n2, s0n3,
                s1n0, s1n1, s1n2, s1n3,
                dn0,  dn1,  dn2,  dn3,
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        fflush(stderr);
    }
    return true;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_ifairy_lut_can_mul_mat(src0, src1, dst)) {
        return 0;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src1->ne[1];
    const int64_t blocks_per_col = K / QK_K;
    const int64_t groups = blocks_per_col * ((QK_K - 1) / 3); // drop each 256th elem
    size_t quant_bytes = 0;
    if (src1->type == GGML_TYPE_F32) {
        quant_bytes = GGML_PAD((size_t) N * (size_t) blocks_per_col * sizeof(block_ifairy_q16), 64);
    }
    const size_t lut_bytes = (size_t) N * (size_t) groups * 32;
    const size_t scale_bytes = (size_t) N * (size_t) groups * 2 * sizeof(float);
    const size_t tmp_bytes = (size_t) N * sizeof(float);
    const size_t per_thread = GGML_PAD(lut_bytes + scale_bytes + tmp_bytes, 64);
    return quant_bytes + per_thread * (size_t) n_threads;
}

bool ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out) {
    if (!tensor || tensor->type != GGML_TYPE_IFAIRY) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return false;
    }

    ifairy_lut_extra * extra = (ifairy_lut_extra *) tensor->extra;
    if (extra && extra->indexes) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return true;
    }

    const int64_t k = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    if (k % QK_K != 0 || rows <= 0) {
        return false;
    }

    const struct ggml_ifairy_3w_index_info info = ggml_ifairy_3w_get_index_info(k);
    const size_t index_bytes = ggml_ifairy_3w_index_buffer_size(&info, rows);
    uint8_t * buf = (uint8_t *) ggml_aligned_malloc(index_bytes);
    if (!buf) {
        return false;
    }

    const bool ok = ggml_ifairy_3w_encode((const block_ifairy *) tensor->data, k, rows, buf, index_bytes);
    if (!ok) {
        ggml_aligned_free(buf, index_bytes);
        return false;
    }

    extra = new ifairy_lut_extra;
    extra->indexes = buf;
    extra->size    = index_bytes;
    extra->index_tensor = NULL;
    tensor->extra  = extra;

    {
        std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
        g_ifairy_lut_extras.push_back(extra);
    }

    if (index_tensor_out) {
        *index_tensor_out = NULL;
    }
    return true;
}

void ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    (void) m; // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    const int64_t K  = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K - 1) / 3;
    const int64_t groups = blocks * groups_per_block;

    // canonical patterns for idx' = 0..15 (u1, u2)
    static const int8_t u_wr[16] = { 1, 1, 1, 1, 0, 0, 0, 0,-1,-1,-1,-1, 0, 0, 0, 0};
    static const int8_t u_wi[16] = { 0, 1, 0,-1, 1, 1, 1, 1, 0, 1, 0,-1,-1,-1,-1,-1};
    static const int8_t v_wr[16] = { 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0};
    static const int8_t v_wi[16] = { 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1, 0, 1, 0,-1};

    for (int col = 0; col < n; ++col) {
        const uint8_t * act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) act_col_bytes;
        float * scales_out = (float *) lut_scales + (size_t) col * (size_t) groups * 2;

        int8_t * lut_out = (int8_t *) lut_buf + (size_t) col * (size_t) groups * 32;

        for (int64_t g = 0; g < groups; ++g) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const int64_t idx0 = blk * QK_K + intra * 3 + 0;
            const int64_t idx1 = blk * QK_K + intra * 3 + 1;
            const int64_t idx2 = blk * QK_K + intra * 3 + 2;

            const int blk0 = (int) blk;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off0 = (int) (idx0 - blk * QK_K);
            const int off1 = (int) (idx1 - blk * QK_K);
            const int off2 = (int) (idx2 - blk * QK_K);

            int xr0 = 0, xi0 = 0;
            int xr1 = 0, xi1 = 0;
            int xr2 = 0, xi2 = 0;

            if (idx0 < K) { xr0 = (int8_t) act_blocks[blk0].x_real[off0]; xi0 = (int8_t) act_blocks[blk0].x_imag[off0]; }
            if (idx1 < K) { xr1 = (int8_t) act_blocks[blk1].x_real[off1]; xi1 = (int8_t) act_blocks[blk1].x_imag[off1]; }
            if (idx2 < K) { xr2 = (int8_t) act_blocks[blk2].x_real[off2]; xi2 = (int8_t) act_blocks[blk2].x_imag[off2]; }

            int8_t * real_tbl = lut_out + (size_t) g * 32 + 0;
            int8_t * imag_tbl = lut_out + (size_t) g * 32 + 16;

            float act_scale_r = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
            float act_scale_i = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
            scales_out[g * 2 + 0] = act_scale_r;
            scales_out[g * 2 + 1] = act_scale_i;

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

void ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict) {
    if (!indexes || !dst || !qweights) {
        return;
    }

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K - 1) / 3;
    const int64_t groups = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;

    const char * cmp_env = getenv("GGML_IFAIRY_LUT_COMPARE");
    const bool do_cmp = cmp_env && strcmp(cmp_env, "0") != 0;
    const int cmp_limit = cmp_env && strcmp(cmp_env, "1") != 0 ? (int) strtol(cmp_env, NULL, 10) : 4; // default small
    static int cmp_prints = 0;

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
        for (int col = 0; col < n; ++col) {
            const int8_t * lut_base = (const int8_t *) lut + (size_t) col * (size_t) groups * 32;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) groups * 2;
            const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride) : NULL;

            float acc_r = 0.0f;
            float acc_i = 0.0f;

            for (int64_t g = 0; g < groups; ++g) {
                const uint8_t code = idx_row[g];
                const uint8_t idx = code & 0x0f;
                const bool do_swap = (code & 0x80u) != 0;
                const bool neg_r   = (code & 0x20u) != 0;
                const bool neg_i   = (code & 0x40u) != 0;

                const int64_t blk   = g / groups_per_block;
                const int64_t intra = g - blk * groups_per_block;
                const int idx0 = (int) (blk * QK_K + intra * 3 + 0);
                const int idx1 = (int) (blk * QK_K + intra * 3 + 1);
                const int idx2 = (int) (blk * QK_K + intra * 3 + 2);

                if (!strict) {
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

                    const int w_blk = (int) blk;
                    const float w_scale_r = GGML_FP16_TO_FP32(w_row[w_blk].d_real);
                    const float w_scale_i = GGML_FP16_TO_FP32(w_row[w_blk].d_imag);

                    const float act_scale_r = scales[g * 2 + 0];
                    const float act_scale_i = scales[g * 2 + 1];

                    acc_r += (float) vr * act_scale_r * w_scale_r;
                    acc_i += (float) vi * act_scale_i * w_scale_i;
                } else {
                    // strict: reconstruct contribution directly from quantized weights/activations
                    const int blk0 = idx0 / QK_K;
                    const int blk1 = idx1 / QK_K;
                    const int blk2 = idx2 / QK_K;
                    const int off0 = idx0 - blk0 * QK_K;
                    const int off1 = idx1 - blk1 * QK_K;
                    const int off2 = idx2 - blk2 * QK_K;

                    int xr0 = 0, xi0 = 0;
                    int xr1 = 0, xi1 = 0;
                    int xr2 = 0, xi2 = 0;

                    if (idx0 < K) { xr0 = (int8_t) act_blocks[blk0].x_real[off0]; xi0 = (int8_t) act_blocks[blk0].x_imag[off0]; }
                    if (idx1 < K) { xr1 = (int8_t) act_blocks[blk1].x_real[off1]; xi1 = (int8_t) act_blocks[blk1].x_imag[off1]; }
                    if (idx2 < K) { xr2 = (int8_t) act_blocks[blk2].x_real[off2]; xi2 = (int8_t) act_blocks[blk2].x_imag[off2]; }

                    const float a0r = (idx0 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk0].d_real) : 0.0f;
                    const float a0i = (idx0 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk0].d_imag) : 0.0f;
                    const float a1r = (idx1 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk1].d_real) : 0.0f;
                    const float a1i = (idx1 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk1].d_imag) : 0.0f;
                    const float a2r = (idx2 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk2].d_real) : 0.0f;
                    const float a2i = (idx2 < K && act_blocks) ? GGML_FP16_TO_FP32(act_blocks[blk2].d_imag) : 0.0f;

                    const uint8_t c0 = (idx0 < K) ? ggml_ifairy_read_code_local(w_row, idx0) : 0;
                    const uint8_t c1 = (idx1 < K) ? ggml_ifairy_read_code_local(w_row, idx1) : 0;
                    const uint8_t c2 = (idx2 < K) ? ggml_ifairy_read_code_local(w_row, idx2) : 0;

                    auto accum_one = [&](uint8_t c, float ar, float ai, const block_ifairy & wb, float & rr, float & ii) {
                        if (c <= 1) {
                            const float w = (c == 1 ? 1.0f : -1.0f) * GGML_FP16_TO_FP32(wb.d_real);
                            rr += w * ar;
                            ii += w * ai;
                        } else {
                            const float w = (c == 3 ? 1.0f : -1.0f) * GGML_FP16_TO_FP32(wb.d_imag);
                            rr += -w * ai;
                            ii +=  w * ar;
                        }
                    };

                    float real = 0.0f;
                    float imag = 0.0f;
                    if (idx0 < K) accum_one(c0, (float) xr0 * a0r, (float) xi0 * a0i, w_row[blk0], real, imag);
                    if (idx1 < K) accum_one(c1, (float) xr1 * a1r, (float) xi1 * a1i, w_row[blk1], real, imag);
                    if (idx2 < K) accum_one(c2, (float) xr2 * a2r, (float) xi2 * a2i, w_row[blk2], real, imag);

                    acc_r += real;
                    acc_i += imag;
                }
            }

            const float out_r = acc_r;
            const float out_i = acc_i;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (do_cmp && cmp_prints < cmp_limit && row < 2 && col < 2 && act_blocks) {
                double base_r = 0.0, base_i = 0.0;
                // reference similar to vec_dot_ifairy_q16_K (per-weight scales, no LUT drop)
                for (int idx = 0; idx < k; ++idx) {
                    const int blk = idx / QK_K;
                    const int off = idx - blk * QK_K;
                    const uint8_t code = ggml_ifairy_read_code_local(w_row, idx);
                    const float wr_scale = GGML_FP16_TO_FP32(w_row[blk].d_real);
                    const float wi_scale = GGML_FP16_TO_FP32(w_row[blk].d_imag);
                    float wr = 0.0f, wi = 0.0f;
                    if (code <= 1) {
                        wr = (code == 1 ? wr_scale : -wr_scale);
                    } else {
                        wi = (code == 3 ? wi_scale : -wi_scale);
                    }
                    const int8_t ar_q = act_blocks[blk].x_real[off];
                    const int8_t ai_q = act_blocks[blk].x_imag[off];
                    const float ar = GGML_FP16_TO_FP32(act_blocks[blk].d_real) * ar_q;
                    const float ai = GGML_FP16_TO_FP32(act_blocks[blk].d_imag) * ai_q;
                    base_r += (double) wr * (double) ar - (double) wi * (double) ai;
                    base_i += (double) wr * (double) ai + (double) wi * (double) ar;
                }
                const float diff_r = out_r - base_r;
                const float diff_i = out_i - base_i;
                fprintf(stderr, "[ifairy_lut_cmp] row=%d col=%d out=(%.6f,%.6f) base=(%.6f,%.6f) diff=(%.6f,%.6f)\n",
                        row, col, out_r, out_i, base_r, base_i, diff_r, diff_i);
                ++cmp_prints;
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                out_ptr[0] = out_r;
                out_ptr[1] = out_i;
            }
        }
    }
}

static void ggml_ifairy_lut_mul_mat_scalar_internal(int m, int k, int n, const void * qweights, const void * act, size_t act_stride,
                                                    const uint8_t * indexes, int8_t * lut, float * scales, float * dst, size_t dst_col_stride) {
    if (!qweights || !act || !dst || !indexes || !lut || !scales) {
        return;
    }

    const bool strict = getenv("GGML_IFAIRY_LUT_VALIDATE_STRICT") && strcmp(getenv("GGML_IFAIRY_LUT_VALIDATE_STRICT"), "0") != 0;

    // preprocess activations -> LUT per column
    ggml_ifairy_lut_preprocess(m, k, n, act, act_stride, scales, lut);
    const size_t dst_row_stride = 2 * sizeof(float);
    ggml_ifairy_lut_qgemm(m, k, n, qweights, indexes, lut, scales, act, act_stride, dst, dst_col_stride, dst_row_stride, false, strict);
}

void ggml_ifairy_lut_mul_mat_scalar(int m, int k, int n, const void * qweights, const void * act, size_t act_stride, float * dst) {
    if (!qweights || !act || !dst) {
        return;
    }

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K - 1) / 3;
    const int64_t groups = blocks * groups_per_block;

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

    ggml_ifairy_lut_mul_mat_scalar_internal(m, k, n, qweights, act, act_stride, indexes, lut, scales, dst, (size_t) m * 2 * sizeof(float));

    free(buf);
}
