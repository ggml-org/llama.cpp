#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_mul_mat_q_id(
        ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * const * dsts, int n_mats) {
    GGML_ASSERT(n_mats == 1 || n_mats == 2);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const ggml_tensor * src0 = src0s[0];
    ggml_tensor * dst = dsts[0];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] % src1->nb[1] == 0);
    GGML_ASSERT(dst->nb[2] % dst->nb[1] == 0);
    GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = src1->ne[2];
    const int64_t ne_get_rows = n_tokens*n_expert_used;
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const bool dedup_bcast = src1->ne[1] == 1 && n_expert_used > 1;
    const bool use_native_fp4 = blackwell_mma_available(cc) &&
        (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(!use_native_fp4 || n_mats == 1);
    const size_t y_block_size = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), src0->ne[2] + 1);

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = src1->nb[2] / src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        src0->ne[2], n_tokens, n_expert_used, src1->ne[1], si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
    CUDA_CHECK(cudaGetLastError());

    size_t nbytes_src1_q8_1 = n_tokens*n_expert_used*ne10_padded * y_block_size/y_values_per_block;
    for (int i = 0; i < n_mats; ++i) {
        const ggml_tensor * src0_i = src0s[i];
        ggml_tensor * dst_i = dsts[i];
        GGML_ASSERT(dst_i->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_are_same_shape(src0, src0_i));
        GGML_ASSERT(ggml_are_same_shape(dst, dst_i));
        GGML_ASSERT(mmq_get_q8_1_ds_layout(src0->type) == mmq_get_q8_1_ds_layout(src0_i->type));
        GGML_ASSERT(src0_i->ne[0] == src1->ne[0]);
        GGML_ASSERT(dst_i->ne[1] == n_expert_used);

        const bool fallback = src0_i->ne[1] % 128 != 0;
        nbytes_src1_q8_1 = std::max(nbytes_src1_q8_1,
            n_tokens*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0_i->type, fallback, cc, src1->ne[1]) * sizeof(block_q8_1_mmq));
    }

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne_get_rows);
    }
    const int64_t s11 = src1->nb[1] / ggml_type_size(src1->type);
    const int64_t s12_src = src1->nb[2] / ggml_type_size(src1->type);
    const int64_t s13_src = src1->nb[3] / ggml_type_size(src1->type);
    if (use_native_fp4) {
        static constexpr size_t align_float8 = 32;
        const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
        if (dedup_bcast) {
            quantize_scatter_mmq_fp4_cuda((const float *) src1->data, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr,
                src0->type, use_aligned_float8, src1->ne[0], /*stride_token=*/s12_src, ne10_padded,
                n_tokens, ne_get_rows, n_expert_used, stream);
        } else {
            quantize_mmq_fp4_cuda((const float *) src1->data, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr,
                src0->type, use_aligned_float8, src1->ne[0], s11, s12_src, s13_src,
                ne10_padded, ne_get_rows, 1, 1, stream);
        }
    } else if (dedup_bcast) {
        quantize_scatter_mmq_q8_1_cuda((const float *) src1->data, ids_src1.get(), src1_q8_1.get(), src0->type,
            src1->ne[0], /*stride_token=*/s12_src, ne10_padded, n_tokens, ne_get_rows, n_expert_used, stream);
    } else {
        quantize_mmq_q8_1_cuda((const float *) src1->data, ids_src1.get(), src1_q8_1.get(), src0->type,
            src1->ne[0], s11, s12_src, s13_src, ne10_padded, ne_get_rows, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12_q = use_native_fp4 ?
        src1->ne[1] * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
        src1->ne[1] * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_q = n_tokens*s12_q;
    for (int i = 0; i < n_mats; ++i) {
        const ggml_tensor * src0_i = src0s[i];
        ggml_tensor * dst_i = dsts[i];
        const size_t ts_src0 = ggml_type_size(src0_i->type);
        const size_t ts_dst = ggml_type_size(dst_i->type);
        const int64_t s01 = src0_i->nb[1] / ts_src0;
        const int64_t s02 = src0_i->nb[2] / ts_src0;
        const int64_t s03 = src0_i->nb[3] / ts_src0;
        const int64_t s1 = dst_i->nb[1] / ts_dst;
        const int64_t s2 = dst_i->nb[2] / ts_dst;
        const int64_t s3 = dst_i->nb[3] / ts_dst;
        const mmq_args args = {
            (const char *) src0_i->data, src0_i->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), (float *) dst_i->data,
            src1_scale.ptr,
            src0_i->ne[0], src0_i->ne[1], ne_get_rows, s01, ne_get_rows, s1,
            src0_i->ne[2], src0_i->ne[2], s02, s12_q, s2,
            src0_i->ne[3], src1->ne[3], s03, s13_q, s3,
            n_tokens};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    const ggml_tensor * src0s[] = { src0 };
    ggml_tensor * dsts[] = { dst };
    ggml_cuda_mul_mat_q_id(ctx, src0s, src1, ids, dsts, 1);
}

void ggml_cuda_mul_mat_q_pair(ggml_backend_cuda_context & ctx, ggml_tensor * dst0, ggml_tensor * dst1) {
    GGML_ASSERT(dst0->src[1] == dst1->src[1]);
    GGML_ASSERT(dst0->src[2] == dst1->src[2]);
    const ggml_tensor * src0s[] = { dst0->src[0], dst1->src[0] };
    ggml_tensor * dsts[] = { dst0, dst1 };
    ggml_cuda_mul_mat_q_id(ctx, src0s, dst0->src[1], dst0->src[2], dsts, 2);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
