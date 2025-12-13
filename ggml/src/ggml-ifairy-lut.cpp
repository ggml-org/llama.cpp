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

static inline bool ggml_ifairy_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && strcmp(env, "0") != 0;
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
    const bool dbg = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    const char * enabled_env = getenv("GGML_IFAIRY_LUT");
    if (enabled_env && strcmp(enabled_env, "0") == 0) {
        if (dbg) { GGML_LOG_WARN("ifairy_lut: disabled by env GGML_IFAIRY_LUT=0\n"); }
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
    if (dbg) { GGML_LOG_INFO("ifairy_lut: can_mul_mat=true\n"); }
    return true;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_ifairy_lut_can_mul_mat(src0, src1, dst)) {
        return 0;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src1->ne[1];
    const int64_t blocks_per_col = K / QK_K;
    const int64_t groups = blocks_per_col * ((QK_K + 2) / 3); // 85 triplets + 1 tail group per block
    size_t quant_bytes = 0;
    if (src1->type == GGML_TYPE_F32) {
        quant_bytes = GGML_PAD((size_t) N * (size_t) blocks_per_col * sizeof(block_ifairy_q16), 64);
    }
    const size_t lut_bytes = (size_t) N * (size_t) groups * (size_t) (4 * 64) * sizeof(int16_t);
    const size_t scale_bytes = (size_t) N * (size_t) groups * 2 * sizeof(float);
    const size_t shared_bytes = GGML_PAD(lut_bytes + scale_bytes, 64);
    const size_t tmp_bytes = GGML_PAD((size_t) N * sizeof(float), 64);
    return quant_bytes + shared_bytes + tmp_bytes * (size_t) n_threads;
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
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const uint8_t * act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) act_col_bytes;
        float * scales_out = (float *) lut_scales + (size_t) col * (size_t) groups * 2;

        int16_t * lut_out = (int16_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * (size_t) (4 * 64) * sizeof(int16_t));

        for (int64_t g = 0; g < groups; ++g) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const bool tail = intra == groups_per_block - 1;
            const int64_t base_off = tail ? (QK_K - 1) : intra * 3;
            const int64_t idx0 = blk * QK_K + base_off + 0;

            const int blk0 = (int) blk;
            const int off0 = (int) base_off;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off1 = (int) (base_off + 1);
            const int off2 = (int) (base_off + 2);

            int xr0 = 0, xi0 = 0;
            int xr1 = 0, xi1 = 0;
            int xr2 = 0, xi2 = 0;

            if (idx0 < K) { xr0 = (int8_t) act_blocks[blk0].x_real[off0]; xi0 = (int8_t) act_blocks[blk0].x_imag[off0]; }
            if (!tail) {
                xr1 = (int8_t) act_blocks[blk1].x_real[off1]; xi1 = (int8_t) act_blocks[blk1].x_imag[off1];
                xr2 = (int8_t) act_blocks[blk2].x_real[off2]; xi2 = (int8_t) act_blocks[blk2].x_imag[off2];
            }

            float act_scale_r = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
            float act_scale_i = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
            scales_out[g * 2 + 0] = act_scale_r;
            scales_out[g * 2 + 1] = act_scale_i;

            // Build tables for all 64 (c0,c1,c2) patterns, matching ggml_vec_dot_ifairy_q16_K_generic:
            //   sum_ac = Σ xr * wr
            //   sum_ad = Σ xi * wr
            //   sum_bc = Σ xr * wi
            //   sum_bd = Σ xi * wi
            // where code -> (wr,wi):
            //   0 -> (-1,0), 1 -> (1,0), 2 -> (0,-1), 3 -> (0,1)
            int16_t * tbl = lut_out + (size_t) g * (4 * 64);
            for (int pat = 0; pat < 64; ++pat) {
                const uint8_t c0 = (uint8_t) (pat & 3);
                const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                int wr0 = 0, wi0 = 0;
                int wr1 = 0, wi1 = 0;
                int wr2 = 0, wi2 = 0;

                switch (c0) { case 0: wr0 = -1; break; case 1: wr0 =  1; break; case 2: wi0 = -1; break; case 3: wi0 =  1; break; }
                switch (c1) { case 0: wr1 = -1; break; case 1: wr1 =  1; break; case 2: wi1 = -1; break; case 3: wi1 =  1; break; }
                switch (c2) { case 0: wr2 = -1; break; case 1: wr2 =  1; break; case 2: wi2 = -1; break; case 3: wi2 =  1; break; }

                const int sum_ac = xr0 * wr0 + xr1 * wr1 + xr2 * wr2;
                const int sum_ad = xi0 * wr0 + xi1 * wr1 + xi2 * wr2;
                const int sum_bc = xr0 * wi0 + xr1 * wi1 + xr2 * wi2;
                const int sum_bd = xi0 * wi0 + xi1 * wi1 + xi2 * wi2;

                tbl[0 * 64 + pat] = (int16_t) sum_ac;
                tbl[1 * 64 + pat] = (int16_t) sum_ad;
                tbl[2 * 64 + pat] = (int16_t) sum_bc;
                tbl[3 * 64 + pat] = (int16_t) sum_bd;
            }
        }
    }
}

void ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict) {
    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
        const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

        const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
        const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

        for (int col = 0; col < n; ++col) {
            const int16_t * lut_base = (const int16_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * (size_t) (4 * 64) * sizeof(int16_t));
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) groups * 2;
            const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride) : NULL;

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;
            for (int64_t g = 0; g < groups; ++g) {
                const float act_scale_r = scales[g * 2 + 0];
                const float act_scale_i = scales[g * 2 + 1];

                const uint8_t pat = (uint8_t) (idx_row[g] & 0x3f);

                const int16_t sum_ac = lut_base[(size_t) g * (4 * 64) + 0 * 64 + pat];
                const int16_t sum_ad = lut_base[(size_t) g * (4 * 64) + 1 * 64 + pat];
                const int16_t sum_bc = lut_base[(size_t) g * (4 * 64) + 2 * 64 + pat];
                const int16_t sum_bd = lut_base[(size_t) g * (4 * 64) + 3 * 64 + pat];

                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act_blocks != NULL);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int blk = 0; blk < (int) blocks; ++blk) {
                    const uint8_t * GGML_RESTRICT w_ptr   = w_row[blk].qs;
                    const int8_t  * GGML_RESTRICT x_r_ptr = (const int8_t *) act_blocks[blk].x_real;
                    const int8_t  * GGML_RESTRICT x_i_ptr = (const int8_t *) act_blocks[blk].x_imag;

                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0: wr = -1; wi =  0; break;
                            case 1: wr =  1; wi =  0; break;
                            case 2: wr =  0; wi = -1; break;
                            case 3: wr =  0; wi =  1; break;
                        }

                        const int xr = (int) x_r_ptr[j];
                        const int xi = (int) x_i_ptr[j];

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
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
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    // workspace: indexes + LUT + scales
    const size_t index_bytes_raw = (size_t) m * (size_t) groups;
    const size_t index_bytes = GGML_PAD(index_bytes_raw, 64);
    const size_t lut_bytes   = (size_t) n * (size_t) groups * (size_t) (4 * 64) * sizeof(int16_t);
    const size_t scale_bytes = (size_t) n * (size_t) groups * 2 * sizeof(float);
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
    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);

    ggml_ifairy_lut_mul_mat_scalar_internal(m, k, n, qweights, act, act_stride, indexes, lut, scales, dst, (size_t) m * 2 * sizeof(float));

    free(buf);
}
