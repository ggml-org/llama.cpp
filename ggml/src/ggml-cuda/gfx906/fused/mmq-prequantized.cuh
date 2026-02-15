#pragma once

// Pre-quantized MMQ path - accepts already-quantized Q8_1 data

#include "../../mmq.cuh"

static void mmq_switch_type_prequantized(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:   mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);   break;
        case GGML_TYPE_Q4_1:   mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);   break;
        case GGML_TYPE_Q5_0:   mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);   break;
        case GGML_TYPE_Q5_1:   mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);   break;
        case GGML_TYPE_Q8_0:   mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);   break;
        case GGML_TYPE_MXFP4:  mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);  break;
        case GGML_TYPE_Q2_K:   mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);   break;
        case GGML_TYPE_Q3_K:   mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);   break;
        case GGML_TYPE_Q4_K:   mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);   break;
        case GGML_TYPE_Q5_K:   mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);   break;
        case GGML_TYPE_Q6_K:   mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);   break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_S:  mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);  break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream); break;
        case GGML_TYPE_IQ3_S:  mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);  break;
        case GGML_TYPE_IQ1_S:  mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);  break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream); break;
        default: GGML_ABORT("unsupported type");
    }
}

// MUL_MAT with pre-quantized Q8_1 src1 data
static void ggml_cuda_mul_mat_q_prequantized(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const void * src1_q8_1,
    ggml_tensor * dst,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13
) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(src0->nb[0] == ts_src0);
    GGML_ASSERT(dst->nb[0]  == ts_dst);

    const char * src0_d = (const char *) src0->data;
    float * dst_d = (float *) dst->data;

    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne1  = dst->ne[1];

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  = dst->nb[1]  / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  = dst->nb[2]  / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  = dst->nb[3]  / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;

    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1, nullptr, nullptr, dst_d,
        ne00, ne01, ne1, s01, ne11, s1,
        ne02, ne12, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne1};

    mmq_switch_type_prequantized(ctx, args, stream);
}
