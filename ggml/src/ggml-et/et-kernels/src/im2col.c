//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

static inline void im2col_store_elem(void * dst_base, enum ggml_type dst_type, int64_t idx, float value) {
    if (dst_type == GGML_TYPE_F32) {
        ((float *) dst_base)[idx] = value;
    } else {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(value);
    }
}

static inline float im2col_load_src_elem(const void * src_base, enum ggml_type src_type, int64_t idx) {
    if (src_type == GGML_TYPE_F32) {
        return ((const float *) src_base)[idx];
    }

    return fp16_to_fp32(((const uint16_t *) src_base)[idx]);
}

int entry_point(struct ggml_et_binary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * dst  = &params->dst;

    if (!src1->data || !dst->data) {
        return -1;
    }

    if (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        return -1;
    }

    const int32_t s0    = ((const int32_t *) dst->op_params)[0];
    const int32_t s1    = ((const int32_t *) dst->op_params)[1];
    const int32_t p0    = ((const int32_t *) dst->op_params)[2];
    const int32_t p1    = ((const int32_t *) dst->op_params)[3];
    const int32_t d0    = ((const int32_t *) dst->op_params)[4];
    const int32_t d1    = ((const int32_t *) dst->op_params)[5];
    const int32_t is_2d = ((const int32_t *) dst->op_params)[6];

    const int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    const int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    const int64_t IH = is_2d ? src1->ne[1] : 1;
    const int64_t IW = src1->ne[0];

    const int64_t KH = is_2d ? src0->ne[1] : 1;
    const int64_t KW = src0->ne[0];

    const int64_t OH         = is_2d ? dst->ne[2] : 1;
    const int64_t OW         = dst->ne[1];
    const int64_t row_elems  = dst->ne[0];
    const int64_t total_rows = OW * OH * N;

    const size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    const size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    const size_t dst_row_stride   = dst->nb[1];
    const size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    const size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    const int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) sizeof(float) : (int64_t) sizeof(uint16_t);
    const int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    const int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    for (int64_t grp = thread_id; grp < total_groups; grp += num_threads) {
        const int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        if (row_end > total_rows) {
            row_end = total_rows;
        }

        for (int64_t row = row_start; row < row_end; ++row) {
            const int64_t in  = row / (OH * OW);
            const int64_t rem = row % (OH * OW);
            const int64_t ioh = rem / OW;
            const int64_t iow = rem % OW;

            void * dst_row = (char *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            for (int64_t iic = 0; iic < IC; ++iic) {
                const void * src_channel = (const char *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                for (int64_t ikh = 0; ikh < KH; ++ikh) {
                    for (int64_t ikw = 0; ikw < KW; ++ikw) {
                        const int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        const int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        const int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        if (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } else {
                            const int64_t src_idx = iih * IW + iiw;
                            const float   value   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, value);
                        }
                    }
                }
            }
        }
    }

    return 0;
}
