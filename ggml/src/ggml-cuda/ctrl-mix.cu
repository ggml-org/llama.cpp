#include "ctrl-mix.cuh"

template <typename T> static __device__ __forceinline__ float ctrl_mix_load(const T * x, int64_t i);
template <> __device__ __forceinline__ float ctrl_mix_load<float>(const float * x, int64_t i) { return x[i]; }
template <> __device__ __forceinline__ float ctrl_mix_load<half>(const half * x, int64_t i) { return __half2float(x[i]); }
template <> __device__ __forceinline__ float ctrl_mix_load<nv_bfloat16>(const nv_bfloat16 * x, int64_t i) {
    return __uint_as_float(((uint32_t) reinterpret_cast<const uint16_t *>(x)[i]) << 16);
}

// one warp per output row: the two dot products over k, then the gated mix
template <typename T>
static __global__ void ctrl_mix_kernel(
        const T * __restrict__ wf, const T * __restrict__ wg, const int64_t w_s1, const float * __restrict__ g, const int k,
        const float * cur, const float * msg, float * dst, const int nrows) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row  = blockIdx.x*(blockDim.x/warp_size) + threadIdx.x/warp_size;
    const int lane = threadIdx.x % warp_size;
    if (row >= nrows) {
        return;
    }

    float f  = 0.0f;
    float gg = 0.0f;
    for (int c = lane; c < k; c += warp_size) {
        const float gc = g[c];
        f  += ctrl_mix_load(wf, row*w_s1 + c)*gc;
        gg += ctrl_mix_load(wg, row*w_s1 + c)*gc;
    }
    f  = warp_reduce_sum<warp_size>(f);
    gg = warp_reduce_sum<warp_size>(gg);

    if (lane == 0) {
        const float x = cur[row];
        const float m = msg[row];
        dst[row] = (x - tanhf(f)*x) + m*tanhf(gg);
    }
}

static bool ctrl_mix_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static void ctrl_mix_span(const ggml_tensor * t, int64_t & beg, int64_t & end) {
    beg = (int64_t) t->data;
    end = beg + ggml_type_size(t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        end += (t->ne[d] - 1)*t->nb[d];
    }
}

int ggml_cuda_match_ctrl_mix(const ggml_cgraph * cgraph, int i) {
    // the compute nodes of the pattern, view nodes may sit between them
    const ggml_op ops[8] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
    int idx[8];
    int n = 0;
    for (int k = i; k < cgraph->n_nodes && n < 8; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (ctrl_mix_is_view(t) && n > 0) {
            continue;
        }
        if (t->op != ops[n]) {
            return -1;
        }
        idx[n++] = k;
    }
    if (n < 8) {
        return -1;
    }
    const ggml_tensor * mf    = cgraph->nodes[idx[0]];
    const ggml_tensor * mg    = cgraph->nodes[idx[1]];
    const ggml_tensor * tf    = cgraph->nodes[idx[2]];
    const ggml_tensor * mulf  = cgraph->nodes[idx[3]];
    const ggml_tensor * sub   = cgraph->nodes[idx[4]];
    const ggml_tensor * tg    = cgraph->nodes[idx[5]];
    const ggml_tensor * mulg  = cgraph->nodes[idx[6]];
    const ggml_tensor * add   = cgraph->nodes[idx[7]];

    const ggml_tensor * wf = mf->src[0];
    const ggml_tensor * wg = mg->src[0];
    const ggml_tensor * g  = mf->src[1];

    if (mg->src[1] != g || g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) || ggml_nrows(g) != 1 ||
            (wf->type != GGML_TYPE_F32 && wf->type != GGML_TYPE_F16 && wf->type != GGML_TYPE_BF16) ||
            wg->type != wf->type || !ggml_are_same_shape(wf, wg) || wf->nb[1] != wg->nb[1] || !ggml_is_contiguous(wf) ||
            !ggml_is_contiguous(wg) || wf->ne[2] != 1 || wf->ne[3] != 1 || mf->type != GGML_TYPE_F32 || mg->type != GGML_TYPE_F32) {
        return -1;
    }
    if (ggml_get_unary_op(tf) != GGML_UNARY_OP_TANH || ggml_get_unary_op(tg) != GGML_UNARY_OP_TANH ||
            tf->src[0] != mf || tg->src[0] != mg) {
        return -1;
    }
    const ggml_tensor * cur = mulf->src[0] == tf ? mulf->src[1] : mulf->src[0];
    const ggml_tensor * msg = mulg->src[0] == tg ? mulg->src[1] : mulg->src[0];
    if ((mulf->src[0] != tf && mulf->src[1] != tf) || (mulg->src[0] != tg && mulg->src[1] != tg) ||
            sub->src[0] != cur || sub->src[1] != mulf || !((add->src[0] == sub && add->src[1] == mulg) || (add->src[0] == mulg && add->src[1] == sub))) {
        return -1;
    }
    const int64_t nrows = wf->ne[1];
    for (const ggml_tensor * t : { cur, msg, (const ggml_tensor *) add }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != nrows) {
            return -1;
        }
    }
    // the intermediate results stay inside the controller
    for (int k = 0; k < 7; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (ggml_node_get_use_count(cgraph, idx[k]) != 1 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return -1;
        }
    }
    for (int k = 0; k < 8; ++k) {
        if (!(cgraph->nodes[idx[k]]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return -1;
        }
    }
    // the output is read element-wise from cur and msg, but g is read by every row
    int64_t ob, oe, gb, ge;
    ctrl_mix_span(add, ob, oe);
    ctrl_mix_span(g, gb, ge);
    if (ob < ge && gb < oe) {
        return -1;
    }
    return idx[7];
}

void ggml_cuda_op_ctrl_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * mf   = cgraph->nodes[i];
    ggml_tensor       * add  = cgraph->nodes[last];
    const ggml_tensor * wf   = mf->src[0];
    const ggml_tensor * g    = mf->src[1];
    const ggml_tensor * sub  = add->src[0]->op == GGML_OP_SUB ? add->src[0] : add->src[1];
    const ggml_tensor * mulg = add->src[0]->op == GGML_OP_SUB ? add->src[1] : add->src[0];
    const ggml_tensor * cur  = sub->src[0];
    const ggml_tensor * tg   = mulg->src[0]->op == GGML_OP_UNARY ? mulg->src[0] : mulg->src[1];
    const ggml_tensor * msg  = mulg->src[0] == tg ? mulg->src[1] : mulg->src[0];
    const ggml_tensor * wg   = tg->src[0]->src[0];

    const int k     = (int) wf->ne[0];
    const int nrows = (int) wf->ne[1];
    const int64_t w_s1 = wf->nb[1]/ggml_type_size(wf->type);

    constexpr int block_size = 256;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int rows_per_block = block_size/warp_size;
    const dim3 grid((nrows + rows_per_block - 1)/rows_per_block);
    cudaStream_t stream = ctx.stream();

    const float * g_d   = (const float *) g->data;
    const float * cur_d = (const float *) cur->data;
    const float * msg_d = (const float *) msg->data;
    float       * dst_d = (float *) add->data;

    switch (wf->type) {
        case GGML_TYPE_F32:
            ctrl_mix_kernel<float><<<grid, block_size, 0, stream>>>((const float *) wf->data, (const float *) wg->data, w_s1, g_d, k, cur_d, msg_d, dst_d, nrows);
            break;
        case GGML_TYPE_F16:
            ctrl_mix_kernel<half><<<grid, block_size, 0, stream>>>((const half *) wf->data, (const half *) wg->data, w_s1, g_d, k, cur_d, msg_d, dst_d, nrows);
            break;
        case GGML_TYPE_BF16:
            ctrl_mix_kernel<nv_bfloat16><<<grid, block_size, 0, stream>>>((const nv_bfloat16 *) wf->data, (const nv_bfloat16 *) wg->data, w_s1, g_d, k, cur_d, msg_d, dst_d, nrows);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
