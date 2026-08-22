#include "moe-ffn.cuh"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "topk-moe.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"

#include <cstring>

static __global__ void moe_ffn_iota(int32_t * dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = i;
    }
}

// ids_dst maps a sorted slot to the flat weight index it*n_expert_used + iex, which is also
// the index into the weights tensor; split it into the token row and its routing weight
static __global__ void moe_ffn_sorted_scales(
        const int32_t * __restrict__ ids_dst, const float * __restrict__ weights,
        int32_t * __restrict__ ids_token, float * __restrict__ scales_sorted,
        const int n, const int n_expert_used) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int slot = ids_dst[i];
    ids_token[i]     = slot/n_expert_used;
    scales_sorted[i] = weights[slot];
}

// gate and up live in one buffer with a per-column stride, so merged and split layouts differ
// only in the base pointers and the stride
static __global__ void moe_ffn_swiglu(
        const float * __restrict__ gate, const float * __restrict__ up, float * __restrict__ act,
        const int n_ff, const int64_t stride, const int64_t n) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int64_t c = idx / n_ff;
    const int64_t i = idx - c*n_ff;
    const float   g = gate[c*stride + i];
    act[idx] = g/(1.0f + expf(-g)) * up[c*stride + i];
}

static __global__ void moe_ffn_scale_rows(float * __restrict__ act, const float * __restrict__ row_scales, const int n_ff) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_ff) {
        act[(int64_t) blockIdx.y*n_ff + i] *= row_scales[blockIdx.y];
    }
}

static __global__ void moe_ffn_reduce(
        const float * __restrict__ experts, const float * __restrict__ weights, float * __restrict__ dst,
        const int n_embd, const int n_expert_used) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_embd) {
        return;
    }
    const int64_t t = blockIdx.y;

    const float * experts_t = experts + t*(int64_t) n_expert_used*n_embd;
    const float * weights_t = weights + t*n_expert_used;

    float sum = 0.0f;
    for (int j = 0; j < n_expert_used; ++j) {
        sum += weights_t[j]*experts_t[(int64_t) j*n_embd + i];
    }
    dst[t*n_embd + i] = sum;
}

// Describe a pool buffer as a tensor so the existing mul_mat_vec_q entry point can be reused.
static ggml_tensor moe_ffn_view(ggml_type type, void * data, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1_override = 0) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type  = type;
    t.data  = data;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = nb1_override ? nb1_override : t.nb[0]*ne0;
    t.nb[2] = t.nb[1]*ne1;
    t.nb[3] = t.nb[2]*ne2;
    return t;
}

// Decode-sized batches: the per-expert GEMMs are vector-shaped, where the tiled mmq kernels lose
// badly to mmvq. Mirrors what ggml_cuda_mul_mat_id does for the same range.
static void moe_ffn_small_batch(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * x, const ggml_tensor * up_exps, const ggml_tensor * gate_exps,
        const ggml_tensor * down_exps, int32_t * ids_data, const float * weights,
        int64_t n_embd, int64_t n_ff, int64_t n_expert, int n_expert_used, int64_t n_tokens, bool merged) {
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int64_t ne_sorted = n_tokens*n_expert_used;

    ggml_tensor ids_t = moe_ffn_view(GGML_TYPE_I32, ids_data, n_expert_used, n_tokens, 1,
                                    n_expert*sizeof(int32_t));
    ggml_tensor x_t   = moe_ffn_view(GGML_TYPE_F32, x->data, n_embd, 1, n_tokens);

    ggml_cuda_pool_alloc<float> act(ctx.pool(), n_ff*ne_sorted);
    ggml_cuda_pool_alloc<float> experts(ctx.pool(), n_embd*ne_sorted);

    // gate first, then up, matching the merged tensor layout
    ggml_tensor gate_half = *up_exps;
    ggml_tensor up_half   = *up_exps;
    if (merged) {
        gate_half.ne[1] = n_ff;
        up_half.ne[1]   = n_ff;
        up_half.data    = (char *) up_exps->data + n_ff*up_exps->nb[1];
    }
    const ggml_tensor * gate_w = merged ? &gate_half : gate_exps;
    const ggml_tensor * up_w   = merged ? &up_half   : up_exps;

    ggml_tensor a = moe_ffn_view(GGML_TYPE_F32, act.get(), n_ff, n_expert_used, n_tokens);

    // mmvq folds the gate GEMV and the swiglu into the up GEMV, but only for one destination
    // column, i.e. a single token; the same restriction gates the unfused graph path
    if (n_tokens == 1 && cc > GGML_CUDA_CC_PASCAL) {
        ggml_cuda_mm_fusion_args_host fusion = {};
        fusion.gate   = gate_w;
        fusion.glu_op = GGML_GLU_OP_SWIGLU;
        ggml_cuda_mul_mat_vec_q(ctx, up_w, &x_t, &ids_t, &a, &fusion);
    } else {
        ggml_cuda_pool_alloc<float> upgate(ctx.pool(), 2*n_ff*ne_sorted);
        ggml_tensor og = moe_ffn_view(GGML_TYPE_F32, upgate.get(),                 n_ff, n_expert_used, n_tokens);
        ggml_tensor ou = moe_ffn_view(GGML_TYPE_F32, upgate.get() + n_ff*ne_sorted, n_ff, n_expert_used, n_tokens);
        ggml_cuda_mul_mat_vec_q(ctx, gate_w, &x_t, &ids_t, &og);
        ggml_cuda_mul_mat_vec_q(ctx, up_w,   &x_t, &ids_t, &ou);

        const int64_t n = n_ff*ne_sorted;
        const int block = 256;
        moe_ffn_swiglu<<<(n + block - 1)/block, block, 0, stream>>>(
            (const float *) og.data, (const float *) ou.data, act.get(), n_ff, n_ff, n);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        ggml_tensor o = moe_ffn_view(GGML_TYPE_F32, experts.get(), n_embd, n_expert_used, n_tokens);
        ggml_cuda_mul_mat_vec_q(ctx, down_exps, &a, &ids_t, &o);
    }

    {
        const int block = 256;
        const dim3 grid((n_embd + block - 1)/block, n_tokens, 1);
        moe_ffn_reduce<<<grid, block, 0, stream>>>(experts.get(), weights, (float *) dst->data, n_embd, n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_moe_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * gate_inp  = dst->src[1];
    const ggml_tensor * up_exps   = dst->src[2];
    const ggml_tensor * gate_exps = dst->src[3];
    const ggml_tensor * down_exps = dst->src[4];

    const bool merged = gate_exps == nullptr;

    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(gate_inp->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(merged || up_exps->type == gate_exps->type);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(gate_inp));
    GGML_ASSERT(ggml_is_contiguous(up_exps));
    GGML_ASSERT(merged || ggml_is_contiguous(gate_exps));
    GGML_ASSERT(ggml_is_contiguous(down_exps));

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t n_expert = gate_inp->ne[1];
    const int64_t n_ff     = merged ? up_exps->ne[1]/2 : up_exps->ne[1];

    const int n_expert_used = ggml_get_op_params_i32(dst, 0);

    const int64_t ne_sorted = n_tokens*n_expert_used;

    if (n_tokens == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const ggml_type type_up   = up_exps->type;
    const ggml_type type_down = down_exps->type;

    // router logits = gate_inp^T @ x -> [n_expert, n_tokens]
    ggml_cuda_pool_alloc<float> logits(ctx.pool(), n_expert*n_tokens);
    if (n_tokens <= MMVF_MAX_BATCH_SIZE &&
        ggml_cuda_should_use_mmvf(gate_inp->type, cc, gate_inp->ne, gate_inp->nb, n_tokens)) {
        // at decode this is a GEMV; the dedicated kernel beats a cuBLAS GEMM call, and it keeps
        // the logits in plain fp32 so top-k tie-breaking matches the unfused path
        ggml_tensor x_t = moe_ffn_view(GGML_TYPE_F32, x->data, n_embd, n_tokens, 1);
        ggml_tensor l_t = moe_ffn_view(GGML_TYPE_F32, logits.get(), n_expert, n_tokens, 1);
        ggml_cuda_mul_mat_vec_f(ctx, gate_inp, &x_t, nullptr, &l_t);
    } else {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        // full fp32: with TF32 the top-k selection can flip between nearly-tied experts
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_PEDANTIC_MATH));
        CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            n_expert, n_tokens, n_embd,
            &alpha, (const float *) gate_inp->data, gate_inp->nb[1]/sizeof(float),
                    (const float *) x->data,        x->nb[1]/sizeof(float),
            &beta,  logits.get(), n_expert));
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));
    }

    // fused softmax + top-k + weight normalization; ids are written with a row stride of n_expert
    ggml_cuda_pool_alloc<float>   weights(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids(ctx.pool(), n_expert*n_tokens);

    // matches the clamp in llm_graph_context::build_moe_ffn
    ggml_cuda_topk_moe_softmax_norm(ctx, logits.get(), weights.get(), ids.get(), n_tokens, n_expert, n_expert_used, 6.103515625e-5f);

    if (n_tokens <= MMVQ_MAX_BATCH_SIZE && ggml_is_quantized(type_up) && ggml_is_quantized(type_down) &&
        n_expert_used <= get_mmvq_mmid_max_batch(type_up, cc) &&
        n_expert_used <= get_mmvq_mmid_max_batch(type_down, cc)) {
        moe_ffn_small_batch(ctx, dst, x, up_exps, gate_exps, down_exps, ids.get(), weights.get(),
            n_embd, n_ff, n_expert, n_expert_used, n_tokens, merged);
        return;
    }

    // expert-sorted row mapping, shared by all three GEMMs
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> ids_iota(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_expert + 1);

    ggml_cuda_launch_mm_ids_helper(ids.get(), ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_expert, n_tokens, n_expert_used, /*nchannels_y=*/1, /*si1=*/n_expert, /*sis1=*/1,
        /*write_inverse =*/ true, stream);
    CUDA_CHECK(cudaGetLastError());

    {
        const int block = 256;
        moe_ffn_iota<<<(ne_sorted + block - 1)/block, block, 0, stream>>>(ids_iota.get(), ne_sorted);
        CUDA_CHECK(cudaGetLastError());
    }

    // quantize x once in expert-sorted order, shared by the up and gate GEMM
    const int64_t n_embd_padded = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t n_ff_padded   = GGML_PAD(n_ff,   MATRIX_ROW_PADDING);

    // gate and up outputs always share one buffer with gate first, so the swiglu-quantize
    // reads both halves with a single row stride regardless of how the weights are stored
    const int64_t stride_upgate = 2*n_ff;

    const bool fb_upgate = n_ff   % 128 != 0;
    const bool fb_down   = n_embd % 128 != 0;

    // Blackwell consumes fp4 weights with fp4 activations instead of q8_1; up/gate and down are
    // decided separately because a model may mix formats (e.g. Q4_K experts with Q5_K down)
    const bool fp4_up   = blackwell_mma_available(cc) && (type_up   == GGML_TYPE_MXFP4 || type_up   == GGML_TYPE_NVFP4);
    const bool fp4_down = blackwell_mma_available(cc) && (type_down == GGML_TYPE_MXFP4 || type_down == GGML_TYPE_NVFP4);

    const size_t yb_up   = fp4_up   ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t yv_up   = fp4_up   ? QK_FP4_MMQ            : QK8_1_MMQ;
    const size_t yb_down = fp4_down ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t yv_down = fp4_down ? QK_FP4_MMQ            : QK8_1_MMQ;

    const size_t nbytes_x_q = ne_sorted*n_embd_padded*yb_up/yv_up +
        ggml_cuda_mmq_get_J_max(type_up, fb_upgate, cc, n_tokens)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> x_q(ctx.pool(), nbytes_x_q);

    // NVFP4 activations are W4A4: the quantizer emits a per-row scale that the GEMM applies
    ggml_cuda_pool_alloc<float> x_scale(ctx.pool());
    if (fp4_up && type_up == GGML_TYPE_NVFP4) {
        x_scale.alloc(ne_sorted);
    }

    {
        // every expert of a token consumes the same activation row, so quantize once per token and
        // scatter to that token's compact rows via the inverse map
        const int64_t s1 = x->nb[1]/sizeof(float);
        if (fp4_up) {
            const bool use_aligned_float8 = ggml_cuda_is_aligned(x, 32);
            quantize_scatter_mmq_fp4_cuda((const float *) x->data, ids_src1.get(), x_q.get(), x_scale.ptr, type_up,
                use_aligned_float8, n_embd, /*stride_token=*/s1, n_embd_padded, n_tokens, ne_sorted, n_expert_used, stream);
        } else {
            quantize_scatter_mmq_q8_1_cuda((const float *) x->data, ids_src1.get(), x_q.get(), type_up,
                n_embd, /*stride_token=*/s1, n_embd_padded, n_tokens, ne_sorted, n_expert_used, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<int32_t> ids_token(ctx.pool(), ne_sorted);
    ggml_cuda_pool_alloc<float>   scales_sorted(ctx.pool(), ne_sorted);
    {
        const int block = 256;
        moe_ffn_sorted_scales<<<(ne_sorted + block - 1)/block, block, 0, stream>>>(
            ids_dst.get(), weights.get(), ids_token.get(), scales_sorted.get(), ne_sorted, n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<float> upgate_s(ctx.pool(), stride_upgate*ne_sorted);

    // strides of the quantized activations; unused by the kernel when ids_dst is present
    const int64_t s12_x_q = fp4_up ? n_embd_padded*sizeof(block_fp4_mmq)/(QK_FP4_MMQ*sizeof(int))
                                   : n_embd_padded*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_x_q = n_tokens*s12_x_q;

    const size_t ts_upgate = ggml_type_size(type_up);

    // up/gate GEMM outputs stay in expert-sorted order (identity ids_dst)
    auto launch_upgate = [&](const ggml_tensor * exps, float * out, int64_t nrows) {
        const mmq_args args = {
            (const char *) exps->data, type_up, (const int *) x_q.get(), ids_iota.get(), expert_bounds.get(), out,
            /*y_scale =*/ x_scale.ptr,
            n_embd, nrows, ne_sorted, (int64_t)(exps->nb[1]/ts_upgate), ne_sorted, stride_upgate,
            n_expert, n_expert, (int64_t)(exps->nb[2]/ts_upgate), s12_x_q, 0,
            1, 1, (int64_t)(exps->nb[2]/ts_upgate)*n_expert, s13_x_q, 0,
            n_tokens, /*atomic_acc =*/ false};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    };

    if (merged) {
        launch_upgate(up_exps, upgate_s.get(), stride_upgate);
    } else {
        launch_upgate(gate_exps, upgate_s.get(),        n_ff);
        launch_upgate(up_exps,   upgate_s.get() + n_ff, n_ff);
    }

    const size_t nbytes_act_q = ne_sorted*n_ff_padded*yb_down/yv_down +
        ggml_cuda_mmq_get_J_max(type_down, fb_down, cc, n_tokens)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> act_q(ctx.pool(), nbytes_act_q);

    ggml_cuda_pool_alloc<float> act_scale(ctx.pool());
    if (fp4_down && type_down == GGML_TYPE_NVFP4) {
        act_scale.alloc(ne_sorted);
    }

    // silu(gate)*up and the routing weight are applied while quantizing; neither is written out.
    // TODO: upstream's fp4 quantize reads x in two passes, so the fused variant only exists for
    // q8_1 for now; fp4 materialises the product first.
    if (fp4_down) {
        ggml_cuda_pool_alloc<float> act(ctx.pool(), n_ff*ne_sorted);
        {
            const int64_t n = n_ff*ne_sorted;
            const int block = 256;
            moe_ffn_swiglu<<<(n + block - 1)/block, block, 0, stream>>>(
                upgate_s.get(), upgate_s.get() + n_ff, act.get(), n_ff, stride_upgate, n);
            CUDA_CHECK(cudaGetLastError());
        }
        moe_ffn_scale_rows<<<dim3((n_ff + 255)/256, ne_sorted, 1), 256, 0, stream>>>(
            act.get(), scales_sorted.get(), n_ff);
        CUDA_CHECK(cudaGetLastError());
        const bool use_aligned_float8 = false;
        quantize_mmq_fp4_cuda(act.get(), nullptr, act_q.get(), act_scale.ptr, type_down, use_aligned_float8,
            n_ff, n_ff, n_ff*ne_sorted, n_ff*ne_sorted, n_ff_padded, ne_sorted, 1, 1, stream);
    } else {
        quantize_mmq_q8_1_swiglu_cuda(upgate_s.get() + n_ff, upgate_s.get(), act_q.get(), type_down,
            n_ff, stride_upgate, n_ff_padded, ne_sorted, scales_sorted.get(), stream);
    }
    CUDA_CHECK(cudaGetLastError());

    // the down GEMM accumulates weight*result directly into the token rows of dst
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const int64_t s12_act_q = fp4_down ? n_ff_padded*sizeof(block_fp4_mmq)/(QK_FP4_MMQ*sizeof(int))
                                       : n_ff_padded*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_act_q = n_tokens*s12_act_q;

    const size_t ts_down = ggml_type_size(type_down);

    const mmq_args args_down = {
        (const char *) down_exps->data, type_down, (const int *) act_q.get(), ids_token.get(), expert_bounds.get(), (float *) dst->data,
        /*y_scale =*/ act_scale.ptr,
        n_ff, n_embd, ne_sorted, (int64_t)(down_exps->nb[1]/ts_down), ne_sorted, n_embd,
        n_expert, n_expert, (int64_t)(down_exps->nb[2]/ts_down), s12_act_q, 0,
        1, 1, (int64_t)(down_exps->nb[2]/ts_down)*n_expert, s13_act_q, 0,
        n_tokens, /*atomic_acc =*/ true};
    ggml_cuda_mul_mat_q_switch_type(ctx, args_down, stream);
}
