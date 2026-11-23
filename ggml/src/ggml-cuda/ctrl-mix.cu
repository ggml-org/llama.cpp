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

int ggml_cuda_match_ctrl_mix(const ggml_cgraph * cgraph, int i, bool check_memory) {
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
    // cur and msg are read element-wise, g by every row
    if (check_memory && (ggml_cuda_fusion_overlap(add, g) || !ggml_cuda_fusion_elementwise_ok(add, cur) ||
                         !ggml_cuda_fusion_elementwise_ok(add, msg))) {
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

// a warp per row: v = x*w, out = v/max(|v|, eps)*scale
static __global__ void mul_l2_norm_f32(const float * x, const float * w, float * dst, const int ncols, const int nrows,
        const float eps, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int row  = blockIdx.x*(blockDim.x/warp_size) + threadIdx.x/warp_size;
    const int lane = threadIdx.x % warp_size;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + (int64_t) row*ncols;
    const float * wr = w + (int64_t) row*ncols;
    float       * dr = dst + (int64_t) row*ncols;

    float sum = 0.0f;
    for (int c = lane; c < ncols; c += warp_size) {
        const float v = xr[c]*wr[c];
        sum += v*v;
    }
    sum = warp_reduce_sum<warp_size>(sum);
    const float s = rsqrtf(fmaxf(sum, eps*eps));

    for (int c = lane; c < ncols; c += warp_size) {
        dr[c] = ((xr[c]*wr[c])*s)*scale;
    }
}

int ggml_cuda_match_mul_l2_norm(const ggml_cgraph * cgraph, int i, bool check_memory) {
    if (i + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * mul = cgraph->nodes[i];
    const ggml_tensor * l2  = cgraph->nodes[i + 1];
    // a warp per row: for the few rows of a generated token, larger batches keep the separate kernels
    if (mul->op != GGML_OP_MUL || l2->op != GGML_OP_L2_NORM || l2->src[0] != mul || mul->type != GGML_TYPE_F32 ||
            ggml_nrows(mul) > 1024 ||
            l2->type != GGML_TYPE_F32 || ggml_node_get_use_count(cgraph, i) != 1 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return -1;
    }
    for (const ggml_tensor * t : { (const ggml_tensor *) mul->src[0], (const ggml_tensor *) mul->src[1], mul, l2 }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !ggml_are_same_shape(t, mul)) {
            return -1;
        }
    }
    int last = i + 1;
    if (i + 2 < cgraph->n_nodes) {
        const ggml_tensor * sc = cgraph->nodes[i + 2];
        if (sc->op == GGML_OP_SCALE && sc->src[0] == l2 && ggml_get_op_params_f32(sc, 1) == 0.0f &&
                ggml_is_contiguous(sc) && sc->type == GGML_TYPE_F32 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                !(l2->flags & GGML_TENSOR_FLAG_OUTPUT) && (sc->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            last = i + 2;
        }
    }
    // a row is read whole before it is written, by one warp: the output may only be exactly an input
    const ggml_tensor * out = cgraph->nodes[last];
    if (check_memory && (!ggml_cuda_fusion_elementwise_ok(out, mul->src[0]) || !ggml_cuda_fusion_elementwise_ok(out, mul->src[1]))) {
        return -1;
    }
    return last;
}

void ggml_cuda_op_mul_l2_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * mul = cgraph->nodes[i];
    const ggml_tensor * l2  = cgraph->nodes[i + 1];
    ggml_tensor       * out = cgraph->nodes[last];

    const int ncols = (int) mul->ne[0];
    const int nrows = (int) ggml_nrows(mul);
    const float eps   = ggml_get_op_params_f32(l2, 0);
    const float scale = last > i + 1 ? ggml_get_op_params_f32(out, 0) : 1.0f;

    constexpr int block_size = 256;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int rows_per_block = block_size/warp_size;
    const dim3 grid((nrows + rows_per_block - 1)/rows_per_block);
    mul_l2_norm_f32<<<grid, block_size, 0, ctx.stream()>>>((const float *) mul->src[0]->data, (const float *) mul->src[1]->data,
        (float *) out->data, ncols, nrows, eps, scale);
}

// out[h, i] = f16((<r_h, table_rel[i]> - sinks[h]) + mask[i]), stored as [n_kv, 1, n_head]
static __global__ void rel_bias_f32_f16(const float * __restrict__ r, const float * __restrict__ table, const int64_t table_s1,
        const float * __restrict__ sinks, const int32_t * __restrict__ rel, const float * __restrict__ mask,
        half * __restrict__ dst, const int n_rel, const int n_kv) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (i >= n_kv) {
        return;
    }
    const float * rh = r + (int64_t) h*n_rel;
    const float * ti = table + (int64_t) rel[i]*table_s1;
    float dot = 0.0f;
    for (int k = 0; k < n_rel; ++k) {
        dot += rh[k]*ti[k];
    }
    dst[(int64_t) h*n_kv + i] = __float2half((dot - sinks[h]) + mask[i]);
}

int ggml_cuda_match_rel_bias(const ggml_cgraph * cgraph, int i, bool check_memory) {
    const ggml_op ops[6] = { GGML_OP_MUL_MAT, GGML_OP_SUB, GGML_OP_GET_ROWS, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY };
    int idx[6];
    int n = 0;
    for (int k = i; k < cgraph->n_nodes && n < 6; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (ctrl_mix_is_view(t) && n > 0) {
            continue;
        }
        if (t->op != ops[n]) {
            return -1;
        }
        idx[n++] = k;
    }
    if (n < 6) {
        return -1;
    }
    const ggml_tensor * mm   = cgraph->nodes[idx[0]];
    const ggml_tensor * sub  = cgraph->nodes[idx[1]];
    const ggml_tensor * gr   = cgraph->nodes[idx[2]];
    const ggml_tensor * cont = cgraph->nodes[idx[3]];
    const ggml_tensor * add  = cgraph->nodes[idx[4]];
    const ggml_tensor * cpy  = cgraph->nodes[idx[5]];

    const ggml_tensor * r     = mm->src[0];   // [n_rel, n_head]
    const ggml_tensor * table = mm->src[1];   // [n_rel, n_extent + 1]
    const ggml_tensor * sinks = sub->src[1];  // [n_head, 1]
    const ggml_tensor * rel   = gr->src[1];   // I32 [n_kv, 1]
    const ggml_tensor * kqb   = cont->src[0]; // permuted view of gr, [n_kv, 1, n_head]
    const ggml_tensor * mask  = add->src[1];  // [n_kv, 1]

    const int64_t n_rel  = r->ne[0];
    const int64_t n_head = r->ne[1];
    const int64_t n_kv   = rel->ne[0];

    if (r->type != GGML_TYPE_F32 || table->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || r->ne[2] != 1 ||
            table->nb[0] != sizeof(float) || table->ne[0] != n_rel || table->ne[2] != 1 || mm->ne[0] != n_head) {
        return -1;
    }
    if (sub->src[0] != mm || sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || ggml_nelements(sinks) != n_head ||
            sinks->ne[0] != n_head) {
        return -1;
    }
    if (gr->src[0] != sub || rel->type != GGML_TYPE_I32 || !ggml_is_contiguous(rel) || ggml_nelements(rel) != n_kv) {
        return -1;
    }
    // the bias reaches the add through views of the gathered rows only, as [n_kv, 1, n_head]
    const ggml_tensor * v = kqb;
    while (v != gr && ctrl_mix_is_view(v)) {
        v = v->src[0];
    }
    if (v != gr || kqb->ne[0] != n_kv || kqb->ne[1] != 1 || kqb->ne[2] != n_head || kqb->ne[3] != 1 ||
            kqb->nb[0] != gr->nb[1] || kqb->nb[2] != gr->nb[0]) {
        return -1;
    }
    if (add->src[0] != cont || !ggml_are_same_shape(cont, kqb) || mask->type != GGML_TYPE_F32 || mask->ne[0] != n_kv ||
            ggml_nelements(mask) != n_kv || mask->nb[0] != sizeof(float)) {
        return -1;
    }
    if (cpy->src[0] != add || cpy->type != GGML_TYPE_F16 || !ggml_is_contiguous(cpy) || !ggml_are_same_shape(cpy, add) ||
            !ggml_is_contiguous(cpy->src[1])) {
        return -1;
    }
    for (int k = 0; k < 5; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (ggml_node_get_use_count(cgraph, idx[k]) != 1 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return -1;
        }
    }
    for (int k = 0; k < 6; ++k) {
        if (!(cgraph->nodes[idx[k]]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return -1;
        }
    }
    // every input is read by many threads
    for (const ggml_tensor * src : { r, table, sinks, rel, mask }) {
        if (check_memory && ggml_cuda_fusion_overlap(cpy->src[1], src)) {
            return -1;
        }
    }
    return idx[5];
}

void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * mm   = cgraph->nodes[i];
    ggml_tensor       * cpy  = cgraph->nodes[last];
    const ggml_tensor * add  = cpy->src[0];
    const ggml_tensor * sub  = mm;
    // walk back from the add to the SUB and GET_ROWS
    const ggml_tensor * v = add->src[0];
    while (v->op != GGML_OP_GET_ROWS) {
        v = v->src[0];
    }
    const ggml_tensor * gr = v;
    sub = gr->src[0];

    const ggml_tensor * r     = mm->src[0];
    const ggml_tensor * table = mm->src[1];
    const ggml_tensor * sinks = sub->src[1];
    const ggml_tensor * rel   = gr->src[1];
    const ggml_tensor * mask  = add->src[1];

    const int n_rel  = (int) r->ne[0];
    const int n_head = (int) r->ne[1];
    const int n_kv   = (int) rel->ne[0];

    constexpr int block_size = 256;
    const dim3 grid((n_kv + block_size - 1)/block_size, n_head);
    rel_bias_f32_f16<<<grid, block_size, 0, ctx.stream()>>>((const float *) r->data, (const float *) table->data,
        table->nb[1]/sizeof(float), (const float *) sinks->data, (const int32_t *) rel->data, (const float *) mask->data,
        (half *) cpy->src[1]->data, n_rel, n_kv);
}

// prompt batches: dst[s, h, t, i] = f16(table[tok, rel[tok, i], h] + mask[s, t, i]) with tok = s*n_tps + t, where
// table is the [n_head, n_ext + 1, n_tokens] F32 table the rows are gathered from. Each thread reads its distance
// and mask value once and writes all heads; the gathered, transposed and summed F32 tensors are never written.
static __global__ void rel_bias_batch_f16(
        const float * __restrict__ table, const int64_t table_s1, const int64_t table_s2,
        const int32_t * __restrict__ rel, const int64_t rel_s1,
        const float * __restrict__ mask, const int64_t mask_s1, const int64_t mask_s3,
        half * __restrict__ dst, const int n_kv, const int n_tps, const int n_head) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    if (i >= n_kv) {
        return;
    }
    const int64_t tok = (int64_t) s*n_tps + t;
    const float   m   = mask[s*mask_s3 + t*mask_s1 + i];

    const int64_t h_stride = (int64_t) n_tps*n_kv;
    half * d = dst + ((int64_t) s*n_head*n_tps + t)*n_kv + i;
    if (m == -INFINITY) {
        // masked out: -inf whatever the bias, without reading it
        const half ninf = __float2half(-INFINITY);
        for (int h = 0; h < n_head; ++h) {
            d[h*h_stride] = ninf;
        }
        return;
    }
    const float * th = table + tok*table_s2 + (int64_t) rel[tok*rel_s1 + i]*table_s1;
    for (int h = 0; h < n_head; ++h) {
        d[h*h_stride] = __float2half(th[h] + m);
    }
}

int ggml_cuda_match_rel_bias_batch(const ggml_cgraph * cgraph, int i, bool check_memory) {
    // GET_ROWS(table, rel) -> RESHAPE -> PERMUTE -> CONT -> ADD(mask) -> CPY(F16), the views skipped
    const ggml_op ops[4] = { GGML_OP_GET_ROWS, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY };
    int idx[4];
    int n = 0;
    for (int k = i; k < cgraph->n_nodes && n < 4; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (ctrl_mix_is_view(t) && n > 0) {
            continue;
        }
        if (t->op != ops[n]) {
            return -1;
        }
        idx[n++] = k;
    }
    if (n < 4) {
        return -1;
    }
    const ggml_tensor * gr   = cgraph->nodes[idx[0]];
    const ggml_tensor * cont = cgraph->nodes[idx[1]];
    const ggml_tensor * add  = cgraph->nodes[idx[2]];
    const ggml_tensor * cpy  = cgraph->nodes[idx[3]];

    const ggml_tensor * table = gr->src[0];   // [n_head, n_ext + 1, n_tokens]
    const ggml_tensor * rel   = gr->src[1];   // I32 [n_kv, n_tokens]
    const ggml_tensor * kqb   = cont->src[0]; // permuted view of gr, [n_kv, n_tps, n_head, n_stream]
    const ggml_tensor * mask  = add->src[1];  // [n_kv, n_tps, 1, n_stream]

    const int64_t n_head   = table->ne[0];
    const int64_t n_tokens = table->ne[2];
    const int64_t n_kv     = rel->ne[0];

    if (table->type != GGML_TYPE_F32 || table->nb[0] != sizeof(float) || table->ne[3] != 1) {
        return -1;
    }
    if (rel->type != GGML_TYPE_I32 || rel->nb[0] != sizeof(int32_t) || rel->ne[1] != n_tokens || rel->ne[2] != 1 ||
            rel->ne[3] != 1) {
        return -1;
    }
    if (gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) || gr->ne[0] != n_head || gr->ne[1] != n_kv ||
            gr->ne[2] != n_tokens || gr->ne[3] != 1) {
        return -1;
    }
    // the gathered rows reach the CONT through views only, permuted to [n_kv, n_tps, n_head, n_stream]
    const ggml_tensor * v = kqb;
    while (v != gr && ctrl_mix_is_view(v)) {
        v = v->src[0];
    }
    const int64_t n_tps    = kqb->ne[1];
    const int64_t n_stream = kqb->ne[3];
    if (v != gr || kqb->ne[0] != n_kv || kqb->ne[2] != n_head || n_tps*n_stream != n_tokens ||
            kqb->nb[0] != gr->nb[1] || kqb->nb[1] != gr->nb[2] || kqb->nb[2] != gr->nb[0] ||
            kqb->nb[3] != gr->nb[2]*n_tps) {
        return -1;
    }
    if (cont->type != GGML_TYPE_F32 || !ggml_is_contiguous(cont) || !ggml_are_same_shape(cont, kqb)) {
        return -1;
    }
    if (add->src[0] != cont || !ggml_are_same_shape(add, cont) || mask->type != GGML_TYPE_F32 ||
            mask->nb[0] != sizeof(float) || mask->ne[0] != n_kv || mask->ne[1] != n_tps || mask->ne[2] != 1 ||
            mask->ne[3] != n_stream) {
        return -1;
    }
    if (cpy->src[0] != add || cpy->type != GGML_TYPE_F16 || !ggml_is_contiguous(cpy) || !ggml_are_same_shape(cpy, add) ||
            !ggml_is_contiguous(cpy->src[1])) {
        return -1;
    }
    for (int k = 0; k < 3; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (ggml_node_get_use_count(cgraph, idx[k]) != 1 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return -1;
        }
    }
    for (int k = 0; k < 4; ++k) {
        if (!(cgraph->nodes[idx[k]]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return -1;
        }
    }
    // the table and the mask are read by many threads, the distances by the thread that writes each column
    for (const ggml_tensor * src : { table, rel, mask }) {
        if (check_memory && ggml_cuda_fusion_overlap(cpy->src[1], src)) {
            return -1;
        }
    }
    return idx[3];
}

void ggml_cuda_op_rel_bias_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * gr    = cgraph->nodes[i];
    ggml_tensor       * cpy   = cgraph->nodes[last];
    const ggml_tensor * add   = cpy->src[0];
    const ggml_tensor * cont  = add->src[0];
    const ggml_tensor * table = gr->src[0];
    const ggml_tensor * rel   = gr->src[1];
    const ggml_tensor * mask  = add->src[1];

    const int n_head   = (int) table->ne[0];
    const int n_kv     = (int) rel->ne[0];
    const int n_tps    = (int) cont->ne[1];
    const int n_stream = (int) cont->ne[3];

    constexpr int block_size = 256;
    const dim3 grid((n_kv + block_size - 1)/block_size, n_tps, n_stream);
    rel_bias_batch_f16<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) table->data, table->nb[1]/sizeof(float), table->nb[2]/sizeof(float),
        (const int32_t *) rel->data, rel->nb[1]/sizeof(int32_t),
        (const float *) mask->data, mask->nb[1]/sizeof(float), mask->nb[3]/sizeof(float),
        (half *) cpy->src[1]->data, n_kv, n_tps, n_head);
}

void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * rel   = dst->src[1];
    const ggml_tensor * mask  = dst->src[2];

    const int n_kv     = (int) dst->ne[0];
    const int n_tps    = (int) dst->ne[1];
    const int n_head   = (int) dst->ne[2];
    const int n_stream = (int) dst->ne[3];

    constexpr int block_size = 256;
    const dim3 grid((n_kv + block_size - 1)/block_size, n_tps, n_stream);
    rel_bias_batch_f16<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) table->data, table->nb[1]/sizeof(float), table->nb[2]/sizeof(float),
        (const int32_t *) rel->data, rel->nb[1]/sizeof(int32_t),
        (const float *) mask->data, mask->nb[1]/sizeof(float), mask->nb[3]/sizeof(float),
        (half *) dst->data, n_kv, n_tps, n_head);
}
