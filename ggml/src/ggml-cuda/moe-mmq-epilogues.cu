#include "mmq.cuh"
#include "moe-mmq-epilogues.cuh"
#include "unary.cuh"

static __global__ void moe_mmq_add_gate_up_bias(const float *   src,
                                                const float *   bias,
                                                const int32_t * ids,
                                                float *         dst,
                                                int64_t         width,
                                                int64_t         n_expert_used,
                                                int64_t         n_tokens,
                                                int64_t         ids_stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = width * n_expert_used * n_tokens;

    if (i >= n) {
        return;
    }

    const int64_t i_col   = i % width;
    const int64_t i_row   = i / width;
    const int64_t i_route = i_row % n_expert_used;
    const int64_t i_token = i_row / n_expert_used;
    const int32_t expert  = ids[i_route + i_token * ids_stride];
    dst[i]                = __fadd_rn(src[i], bias[i_col + (int64_t) expert * width]);
}

static __global__ void moe_mmq_swiglu_oai_staged(const float * gate_up, float * dst, int64_t n_ff, int64_t n_rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_ff * n_rows;

    if (i >= n) {
        return;
    }

    const int64_t i_ff   = i % n_ff;
    const int64_t i_row  = i / n_ff;
    const int64_t stride = 2 * n_ff;
    dst[i] = ggml_cuda_op_swiglu_oai_single(gate_up[i_ff + i_row * stride], gate_up[i_ff + n_ff + i_row * stride]);
}

static __global__ void moe_mmq_swiglu_oai_fused(const float *   gate_up,
                                                const float *   bias,
                                                const int32_t * ids,
                                                float *         dst,
                                                int64_t         n_ff,
                                                int64_t         n_expert_used,
                                                int64_t         n_tokens,
                                                int64_t         ids_stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_ff * n_expert_used * n_tokens;

    if (i >= n) {
        return;
    }

    const int64_t i_ff    = i % n_ff;
    const int64_t i_route = (i / n_ff) % n_expert_used;
    const int64_t i_token = i / (n_ff * n_expert_used);
    const int64_t i_row   = i_route + i_token * n_expert_used;
    const int32_t expert  = ids[i_route + i_token * ids_stride];
    const int64_t stride  = 2 * n_ff;
    const float   gate    = __fadd_rn(gate_up[i_ff + i_row * stride], bias[i_ff + (int64_t) expert * stride]);
    const float   up = __fadd_rn(gate_up[i_ff + n_ff + i_row * stride], bias[i_ff + n_ff + (int64_t) expert * stride]);
    dst[i]           = ggml_cuda_op_swiglu_oai_single(gate, up);
}

static __device__ __forceinline__ uint8_t moe_mmq_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr int fp4_e2m1_emax = 2;
    const int     shared_exp    = __float2int_rn(log2f(amax)) - fp4_e2m1_emax;
    return (uint8_t) max(0, min(254, shared_exp + 127));
}

static __global__ void moe_mmq_swiglu_oai_quant_mxfp4(const float *   gate_up,
                                                      const float *   bias,
                                                      const int32_t * ids,
                                                      const int32_t * ids_dst,
                                                      void *          dst,
                                                      int64_t         n_ff,
                                                      int64_t         ne0,
                                                      int64_t         n_rows,
                                                      int64_t         n_expert_used,
                                                      int64_t         ids_stride) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;

    const int     warp_id    = threadIdx.y;
    const int     lane_id    = threadIdx.x;
    const int     nwarps     = blockDim.y;
    const int64_t sorted_row = blockIdx.x;
    const int64_t route_row  = ids_dst[sorted_row];
    const int64_t i_route    = route_row % n_expert_used;
    const int64_t i_token    = route_row / n_expert_used;
    const int32_t expert     = ids[i_route + i_token * ids_stride];
    const int64_t warp_start = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start >= ne0) {
        return;
    }

    const int64_t k_block        = warp_start / QK_FP4_MMQ;
    const int64_t quad           = (warp_start % QK_FP4_MMQ) / vals_per_warp;
    const int     group          = lane_id / 4;
    const int     lane_in_group  = lane_id % 4;
    const int     base           = group * 2;
    const int64_t gate_up_stride = 2 * n_ff;

    ggml_cuda_pdl_sync();

    uint8_t scales[2];
    char2   packed[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i_ff  = warp_start + b * vals_per_scale + lane_id;
        float         value = 0.0f;
        if (i_ff < n_ff) {
            const float gate =
                __fadd_rn(gate_up[i_ff + route_row * gate_up_stride], bias[i_ff + (int64_t) expert * gate_up_stride]);
            const float up = __fadd_rn(gate_up[i_ff + n_ff + route_row * gate_up_stride],
                                       bias[i_ff + n_ff + (int64_t) expert * gate_up_stride]);
            value          = __fadd_rn(ggml_cuda_op_swiglu_oai_single(gate, up), 0.0f);
        }

        float amax = fabsf(value);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e       = moe_mmq_e8m0_scale(amax);
        scales[b]             = e;
        const float inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float     scaled = value * inv_scale;
        const float     v0     = __shfl_sync(0xFFFFFFFF, scaled, base, WARP_SIZE);
        const float     v1     = __shfl_sync(0xFFFFFFFF, scaled, base + 16, WARP_SIZE);
        const float     v2     = __shfl_sync(0xFFFFFFFF, scaled, base + 1, WARP_SIZE);
        const float     v3     = __shfl_sync(0xFFFFFFFF, scaled, base + 17, WARP_SIZE);
        __nv_fp4x4_e2m1 fp4_packed(make_float4(v0, v1, v2, v3));
        packed[b] = *(char2 *) &fp4_packed;
#else
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        char2         q_packed;
        q_packed.x =
            (__shfl_sync(0xFFFFFFFF, q, base + 16, WARP_SIZE) << 4) | __shfl_sync(0xFFFFFFFF, q, base, WARP_SIZE);
        q_packed.y =
            (__shfl_sync(0xFFFFFFFF, q, base + 17, WARP_SIZE) << 4) | __shfl_sync(0xFFFFFFFF, q, base + 1, WARP_SIZE);
        packed[b] = q_packed;
#endif
    }

    block_fp4_mmq * yb  = (block_fp4_mmq *) dst + k_block * n_rows + sorted_row;
    char2 *         yqs = (char2 *) yb->qs;
    if (lane_in_group == 0) {
        yqs[quad * 16 + group]     = packed[0];
        yqs[quad * 16 + 8 + group] = packed[1];
    }
    if (lane_id == 0) {
        yb->d4[quad] = ((uint32_t) scales[1] << 8) | scales[0];
    }
}

static __global__ void moe_mmq_add_down_bias(const float *   src,
                                             const float *   bias,
                                             const int32_t * ids,
                                             float *         dst,
                                             int64_t         n_embd,
                                             int64_t         n_expert_used,
                                             int64_t         n_tokens,
                                             int64_t         ids_stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_embd * n_expert_used * n_tokens;

    if (i >= n) {
        return;
    }

    const int64_t i_embd  = i % n_embd;
    const int64_t i_row   = i / n_embd;
    const int64_t i_route = i_row % n_expert_used;
    const int64_t i_token = i_row / n_expert_used;
    const int32_t expert  = ids[i_route + i_token * ids_stride];
    dst[i]                = __fadd_rn(src[i], bias[i_embd + (int64_t) expert * n_embd]);
}

static __global__ void moe_mmq_apply_weights(const float * src,
                                             const float * weights,
                                             float *       dst,
                                             int64_t       n_embd,
                                             int64_t       n_rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_embd * n_rows;
    if (i < n) {
        dst[i] = __fmul_rn(src[i], weights[i / n_embd]);
    }
}

static __global__ void moe_mmq_reduce(const float * src,
                                      float *       dst,
                                      int64_t       n_embd,
                                      int64_t       n_expert_used,
                                      int64_t       n_tokens) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_embd * n_tokens;

    if (i >= n) {
        return;
    }

    const int64_t i_embd  = i % n_embd;
    const int64_t i_token = i / n_embd;
    float         sum     = 0.0f;
    for (int64_t route = 0; route < n_expert_used; ++route) {
        sum = __fadd_rn(sum, src[i_embd + (route + i_token * n_expert_used) * n_embd]);
    }
    dst[i] = sum;
}

static __global__ void moe_mmq_w2_epilogue(const float *   down,
                                           const float *   bias,
                                           const int32_t * ids,
                                           const float *   weights,
                                           float *         dst,
                                           int64_t         n_embd,
                                           int64_t         n_expert_used,
                                           int64_t         n_tokens,
                                           int64_t         ids_stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_embd * n_tokens;

    if (i >= n) {
        return;
    }

    const int64_t i_embd  = i % n_embd;
    const int64_t i_token = i / n_embd;
    float         sum     = 0.0f;
    for (int64_t route = 0; route < n_expert_used; ++route) {
        const int64_t i_row  = route + i_token * n_expert_used;
        const int32_t expert = ids[route + i_token * ids_stride];
        const float   value  = __fadd_rn(down[i_embd + i_row * n_embd], bias[i_embd + (int64_t) expert * n_embd]);
        sum                  = __fadd_rn(sum, __fmul_rn(value, weights[i_row]));
    }
    dst[i] = sum;
}

static int moe_mmq_blocks(int64_t n) {
    constexpr int threads = 256;
    return (int) ((n + threads - 1) / threads);
}

void ggml_cuda_moe_mmq_w13_epilogue_staged(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream) {
    constexpr int threads = 256;
    moe_mmq_add_gate_up_bias<<<moe_mmq_blocks(ggml_nelements(args.gate_up_biased)), threads, 0, stream>>>(
        (const float *) args.gate_up_dst->data, (const float *) args.gate_up_bias->data,
        (const int32_t *) args.ids->data, (float *) args.gate_up_biased->data, args.gate_up_biased->ne[0],
        args.ids->ne[0], args.ids->ne[1], ids_stride);
    CUDA_CHECK(cudaGetLastError());

    moe_mmq_swiglu_oai_staged<<<moe_mmq_blocks(ggml_nelements(args.activation)), threads, 0, stream>>>(
        (const float *) args.gate_up_biased->data, (float *) args.activation->data, args.activation->ne[0],
        args.ids->ne[0] * args.ids->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_mmq_w13_epilogue_fused(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream) {
    constexpr int threads = 256;
    moe_mmq_swiglu_oai_fused<<<moe_mmq_blocks(ggml_nelements(args.activation)), threads, 0, stream>>>(
        (const float *) args.gate_up_dst->data, (const float *) args.gate_up_bias->data,
        (const int32_t *) args.ids->data, (float *) args.activation->data, args.activation->ne[0], args.ids->ne[0],
        args.ids->ne[1], ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_mmq_w13_epilogue_quantize(const ggml_cuda_moe_mmq_args & args,
                                              const int32_t *                ids_dst,
                                              void *                         activation_q,
                                              int64_t                        activation_q_ne0,
                                              int64_t                        ids_stride,
                                              cudaStream_t                   stream) {
    constexpr int nwarps         = 8;
    constexpr int vals_per_block = nwarps * 2 * 32;
    const int64_t n_rows         = args.ids->ne[0] * args.ids->ne[1];
    const dim3    blocks(n_rows, (activation_q_ne0 + vals_per_block - 1) / vals_per_block, 1);
    const dim3    threads(WARP_SIZE, nwarps, 1);
    moe_mmq_swiglu_oai_quant_mxfp4<<<blocks, threads, 0, stream>>>(
        (const float *) args.gate_up_dst->data, (const float *) args.gate_up_bias->data,
        (const int32_t *) args.ids->data, ids_dst, activation_q, args.activation->ne[0], activation_q_ne0, n_rows,
        args.ids->ne[0], ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_mmq_w2_epilogue_staged(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream) {
    constexpr int threads = 256;
    moe_mmq_add_down_bias<<<moe_mmq_blocks(ggml_nelements(args.down_biased)), threads, 0, stream>>>(
        (const float *) args.down_dst->data, (const float *) args.down_bias->data, (const int32_t *) args.ids->data,
        (float *) args.down_biased->data, args.down_biased->ne[0], args.ids->ne[0], args.ids->ne[1], ids_stride);
    CUDA_CHECK(cudaGetLastError());

    moe_mmq_apply_weights<<<moe_mmq_blocks(ggml_nelements(args.weighted)), threads, 0, stream>>>(
        (const float *) args.down_biased->data, (const float *) args.weights->data, (float *) args.weighted->data,
        args.weighted->ne[0], args.ids->ne[0] * args.ids->ne[1]);
    CUDA_CHECK(cudaGetLastError());

    moe_mmq_reduce<<<moe_mmq_blocks(ggml_nelements(args.dst)), threads, 0, stream>>>(
        (const float *) args.weighted->data, (float *) args.dst->data, args.dst->ne[0], args.ids->ne[0],
        args.ids->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_mmq_w2_epilogue_fused(
    const ggml_cuda_moe_mmq_args & args, int64_t ids_stride, cudaStream_t stream) {
    constexpr int threads = 256;
    moe_mmq_w2_epilogue<<<moe_mmq_blocks(ggml_nelements(args.dst)), threads, 0, stream>>>(
        (const float *) args.down_dst->data, (const float *) args.down_bias->data, (const int32_t *) args.ids->data,
        (const float *) args.weights->data, (float *) args.dst->data, args.dst->ne[0], args.ids->ne[0], args.ids->ne[1],
        ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_mmq_reduce_weighted(const ggml_cuda_moe_mmq_args & args, cudaStream_t stream) {
    constexpr int threads = 256;
    moe_mmq_reduce<<<moe_mmq_blocks(ggml_nelements(args.dst)), threads, 0, stream>>>(
        (const float *) args.weighted->data, (float *) args.dst->data, args.dst->ne[0], args.ids->ne[0],
        args.ids->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
