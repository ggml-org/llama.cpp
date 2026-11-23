#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr, const float * res_ptr, const float res_scale,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // the residual has the layout of dst
    const float * r_block = res_ptr ? (const float *) ((const char *) res_ptr + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0) : nullptr;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        if (r_block) {
            sumf = (sumf + r_block[i * stride_y + tid]) * res_scale;
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const float * __restrict__ res, const float res_scale,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // the residual has the layout of dst
    const float * r_block = res ? (const float *) ((const char *) res + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 +
                                                   bidy * split_d_inner * dst_nb0) : nullptr;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        if (r_block) {
            sumf = (sumf + r_block[i * stride_y + tid]) * res_scale;
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const float * res, const float res_scale,
                              const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, res, res_scale, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, res, res_scale, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 2:  launch_kernel(std::integral_constant<int, 2 >{}); break;
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst,
                           ggml_tensor * res_add_node, ggml_tensor * res_scale_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;
    const bool fuse_res  = res_scale_dst != nullptr;

    // a residual add and scale (SSM_CONV -> RESHAPE -> ADD -> SCALE) is fused alone
    GGML_ASSERT(!fuse_res || (!fuse_bias && !fuse_silu && res_add_node));

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst or res_scale_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : fuse_res ? res_scale_dst : dst;

    // the residual is the ADD operand that is not the reshaped conv output
    const struct ggml_tensor * res = fuse_res ? (res_add_node->src[1]->view_src == dst || res_add_node->src[1] == dst ?
                                                 res_add_node->src[0] : res_add_node->src[1]) : nullptr;
    const float res_scale = fuse_res ? ggml_get_op_params_f32(res_scale_dst, 0) : 1.0f;
    if (fuse_res) {
        GGML_ASSERT(res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_is_contiguous(out));
        GGML_ASSERT(ggml_nelements(res) == ggml_nelements(out));
    }

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = dst->ne[1];                 // tokens per sequence
    const int64_t n_s = dst->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    const float * res_d  = fuse_res ? (const float *) res->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, res_d, res_scale, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        // with a fused residual, out is the SCALE node, contiguous in the [nr, n_t, n_s] layout of dst
        const struct ggml_tensor * lay = fuse_res ? dst : out;
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, res_d, res_scale, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, lay->nb[0], lay->nb[1],
                          lay->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// one token of a short convolution with its state: the window is the n_conv - 1 columns of history and the new
// column x, out = (sum(window*w) + res)*scale, and the state keeps the last n_conv - 1 columns of the window
template <int d_conv>
static __global__ void ssm_conv_step_f32(
        const float * __restrict__ past, const int64_t past_s1, const char * __restrict__ x, const int64_t x_s,
        const float * __restrict__ w, const int64_t w_s1, const float * res, const float scale,
        float * dst, float * __restrict__ state, const int nc, const float * res2, float * dst2) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nc) {
        return;
    }

    float win[d_conv];
#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        win[k] = past[c*past_s1 + k];
    }
    win[d_conv - 1] = *(const float *) (x + c*x_s);

    float sumf = 0.0f;
#pragma unroll
    for (int k = 0; k < d_conv; ++k) {
        sumf += win[k] * w[c*w_s1 + k];
    }
    const float y = (sumf + res[c]) * scale;

#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        state[c*(d_conv - 1) + k] = win[k + 1];
    }
    // with a residual add fused after the step, only its sum is written (res2 + y, the ADD's operand order)
    if (dst2) {
        dst2[c] = res2[c] + y;
    } else {
        dst[c] = y;
    }
}

void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, ggml_tensor * cpy,
        const ggml_tensor * conv, const ggml_tensor * add, ggml_tensor * scale, const ggml_tensor * res2, ggml_tensor * add2) {
    const ggml_tensor * past  = concat->src[0];   // [n_conv - 1, nc]
    const ggml_tensor * x     = concat->src[1];   // [1, nc]
    const ggml_tensor * w     = conv->src[1];     // [n_conv, nc]
    const ggml_tensor * state = cpy->src[1];      // n_conv - 1 floats per channel
    const ggml_tensor * res   = add->src[0]->op == GGML_OP_RESHAPE && add->src[0]->src[0] == conv ? add->src[1] : add->src[0];

    const int nc     = (int) w->ne[1];
    const int n_conv = (int) w->ne[0];

    const int64_t past_s1 = past->nb[1] / sizeof(float);
    const int64_t w_s1    = w->nb[1] / sizeof(float);
    const float   s       = ggml_get_op_params_f32(scale, 0);

    const int block = 256;
    const dim3 grid((nc + block - 1) / block);
    cudaStream_t stream = ctx.stream();

    const float * past_d  = (const float *) past->data;
    const char  * x_d     = (const char *) x->data;
    const float * w_d     = (const float *) w->data;
    const float * res_d   = (const float *) res->data;
    float       * dst_d   = (float *) scale->data;
    float       * state_d = (float *) state->data;
    const float * res2_d  = add2 ? (const float *) res2->data : nullptr;
    float       * dst2_d  = add2 ? (float *) add2->data : nullptr;

    switch (n_conv) {
        case 2: ssm_conv_step_f32<2><<<grid, block, 0, stream>>>(past_d, past_s1, x_d, x->nb[1], w_d, w_s1, res_d, s, dst_d, state_d, nc, res2_d, dst2_d); break;
        case 3: ssm_conv_step_f32<3><<<grid, block, 0, stream>>>(past_d, past_s1, x_d, x->nb[1], w_d, w_s1, res_d, s, dst_d, state_d, nc, res2_d, dst2_d); break;
        case 4: ssm_conv_step_f32<4><<<grid, block, 0, stream>>>(past_d, past_s1, x_d, x->nb[1], w_d, w_s1, res_d, s, dst_d, state_d, nc, res2_d, dst2_d); break;
        default: GGML_ABORT("unsupported conv width %d", n_conv);
    }
}

// the MoE weighted sum (moe_weighted_reduction_f32 for one token, in its order) feeding the conv step as both its new
// window column and its residual: m = sum_e (experts[e, c]*scale_e)*w_e, out = (sum(window*w) + m)*scale, the state
// keeps the window's last n_conv - 1 columns, and with a residual ADD after the step, res2 + out is written instead
template <int d_conv>
static __global__ void ssm_conv_step_moe_f32(
        const float * __restrict__ experts, const float * __restrict__ expert_scale, const float * __restrict__ weights,
        const int n_used, float * __restrict__ moe_dst,
        const float * __restrict__ past, const int64_t past_s1, const float * __restrict__ w, const int64_t w_s1,
        const float scale, float * dst, float * __restrict__ state, const int nc, const float * res2, float * dst2) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nc) {
        return;
    }
    const float first_scale = expert_scale != nullptr ? expert_scale[0] : 1.0f;
    float m = (experts[c] * first_scale) * weights[0];
    for (int e = 1; e < n_used; ++e) {
        const float sc = expert_scale != nullptr ? expert_scale[e] : 1.0f;
        m += (experts[(int64_t) e*nc + c] * sc) * weights[e];
    }
    moe_dst[c] = m;

    float win[d_conv];
#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        win[k] = past[c*past_s1 + k];
    }
    win[d_conv - 1] = m;

    float sumf = 0.0f;
#pragma unroll
    for (int k = 0; k < d_conv; ++k) {
        sumf += win[k] * w[c*w_s1 + k];
    }
    const float y = (sumf + m) * scale;

#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        state[c*(d_conv - 1) + k] = win[k + 1];
    }
    if (dst2) {
        dst2[c] = res2[c] + y;
    } else {
        dst[c] = y;
    }
}

void ggml_cuda_op_ssm_conv_step_moe(ggml_backend_cuda_context & ctx,
        const ggml_tensor * experts, const ggml_tensor * expert_scale, const ggml_tensor * weights, ggml_tensor * moe_dst,
        const ggml_tensor * concat, ggml_tensor * cpy, const ggml_tensor * conv, ggml_tensor * scale,
        const ggml_tensor * res2, ggml_tensor * add2) {
    const ggml_tensor * past  = concat->src[0];
    const ggml_tensor * w     = conv->src[1];
    const ggml_tensor * state = cpy->src[1];

    const int nc     = (int) w->ne[1];
    const int n_conv = (int) w->ne[0];
    const int n_used = (int) experts->ne[1];

    const int block = 256;
    const dim3 grid((nc + block - 1) / block);
    cudaStream_t stream = ctx.stream();

    const float * e_d   = (const float *) experts->data;
    const float * es_d  = expert_scale ? (const float *) expert_scale->data : nullptr;
    const float * wt_d  = (const float *) weights->data;
    float       * md_d  = (float *) moe_dst->data;
    const float * p_d   = (const float *) past->data;
    const float * w_d   = (const float *) w->data;
    const float   s     = ggml_get_op_params_f32(scale, 0);
    float       * dst_d = (float *) scale->data;
    float       * st_d  = (float *) state->data;
    const float * r2_d  = add2 ? (const float *) res2->data : nullptr;
    float       * d2_d  = add2 ? (float *) add2->data : nullptr;
    const int64_t past_s1 = past->nb[1] / sizeof(float);
    const int64_t w_s1    = w->nb[1] / sizeof(float);

    switch (n_conv) {
        case 2: ssm_conv_step_moe_f32<2><<<grid, block, 0, stream>>>(e_d, es_d, wt_d, n_used, md_d, p_d, past_s1, w_d, w_s1, s, dst_d, st_d, nc, r2_d, d2_d); break;
        case 3: ssm_conv_step_moe_f32<3><<<grid, block, 0, stream>>>(e_d, es_d, wt_d, n_used, md_d, p_d, past_s1, w_d, w_s1, s, dst_d, st_d, nc, r2_d, d2_d); break;
        case 4: ssm_conv_step_moe_f32<4><<<grid, block, 0, stream>>>(e_d, es_d, wt_d, n_used, md_d, p_d, past_s1, w_d, w_s1, s, dst_d, st_d, nc, r2_d, d2_d); break;
        default: GGML_ABORT("unsupported conv width %d", n_conv);
    }
}
