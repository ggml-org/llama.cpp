#include "depth-mix.cuh"
#include "ctrl-mix.cuh"

// out[:, q] = sum_j softmax_j(scale*<s_j, query_q>) s_j over the states s_j: the n_blocks stored (already
// normalized) blocks, then rms_norm(cur). One pass gathers the sum of squares of cur and all dot products (those of
// rms_norm(cur) are those of cur times the norm factor), one reduction sums them, a second pass mixes.
template <int block_size>
static __global__ void depth_mix_f32(
        const float * __restrict__ blocks, const int64_t blocks_s1, const int n_blocks,
        const float * __restrict__ cur, const float * __restrict__ query, const int64_t query_s1, const int nq,
        const float eps, const float scale, float * __restrict__ dst, const int64_t dst_s1, const int ne,
        float * dst2, const float * w2, const float eps2) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps   = block_size/warp_size;
    constexpr int max_vals  = 1 + DEPTH_MIX_MAX_STATES*DEPTH_MIX_MAX_QUERIES;

    __shared__ float red[n_warps][max_vals];
    __shared__ float p[DEPTH_MIX_MAX_STATES*DEPTH_MIX_MAX_QUERIES];
    __shared__ float rs_shared;

    const int tid      = threadIdx.x;
    const int lane     = tid % warp_size;
    const int warp     = tid / warp_size;
    const int n_states = n_blocks + 1;

    // vals[0] = sum of squares of cur, vals[1 + j*nq + q] = <state j, query q> with cur unnormalized
    float vals[max_vals];
#pragma unroll
    for (int v = 0; v < max_vals; ++v) {
        vals[v] = 0.0f;
    }

    for (int e = tid; e < ne; e += block_size) {
        float qe[DEPTH_MIX_MAX_QUERIES];
#pragma unroll
        for (int q = 0; q < DEPTH_MIX_MAX_QUERIES; ++q) {
            qe[q] = q < nq ? query[q*query_s1 + e] : 0.0f;
        }
        const float c = cur[e];
        vals[0] += c*c;
#pragma unroll
        for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
            if (j < n_states) {
                const float x = j < n_blocks ? blocks[j*blocks_s1 + e] : c;
#pragma unroll
                for (int q = 0; q < DEPTH_MIX_MAX_QUERIES; ++q) {
                    vals[1 + j*DEPTH_MIX_MAX_QUERIES + q] += x*qe[q];
                }
            }
        }
    }

#pragma unroll
    for (int v = 0; v < max_vals; ++v) {
        const float r = warp_reduce_sum<warp_size>(vals[v]);
        if (lane == 0) {
            red[warp][v] = r;
        }
    }
    __syncthreads();

    if (tid < max_vals) {
        float r = 0.0f;
#pragma unroll
        for (int w = 0; w < n_warps; ++w) {
            r += red[w][tid];
        }
        red[0][tid] = r;
    }
    __syncthreads();

    if (tid < nq) {
        const float rs = rsqrtf(red[0][0]/ne + eps);
        if (tid == 0) {
            rs_shared = rs;
        }
        float mx = -INFINITY;
        for (int j = 0; j < n_states; ++j) {
            const float d = red[0][1 + j*DEPTH_MIX_MAX_QUERIES + tid]*(j < n_blocks ? 1.0f : rs);
            p[j*DEPTH_MIX_MAX_QUERIES + tid] = d*scale;
            mx = fmaxf(mx, d*scale);
        }
        float sum = 0.0f;
        for (int j = 0; j < n_states; ++j) {
            const float v = expf(p[j*DEPTH_MIX_MAX_QUERIES + tid] - mx);
            p[j*DEPTH_MIX_MAX_QUERIES + tid] = v;
            sum += v;
        }
        const float inv = 1.0f/sum;
        for (int j = 0; j < n_states; ++j) {
            p[j*DEPTH_MIX_MAX_QUERIES + tid] *= inv;
        }
    }
    __syncthreads();

    const float rs = rs_shared;
    for (int e = tid; e < ne; e += block_size) {
        float acc[DEPTH_MIX_MAX_QUERIES] = {0.0f};
#pragma unroll
        for (int j = 0; j < DEPTH_MIX_MAX_STATES; ++j) {
            if (j < n_states) {
                const float x = j < n_blocks ? blocks[j*blocks_s1 + e] : cur[e]*rs;
#pragma unroll
                for (int q = 0; q < DEPTH_MIX_MAX_QUERIES; ++q) {
                    acc[q] += p[j*DEPTH_MIX_MAX_QUERIES + q]*x;
                }
            }
        }
#pragma unroll
        for (int q = 0; q < DEPTH_MIX_MAX_QUERIES; ++q) {
            if (q < nq) {
                dst[q*dst_s1 + e] = acc[q];
            }
        }
    }

    // the RMS_NORM [-> MUL] that follows (one query): the vectorized rms_norm kernel with 1024 threads, one float4
    // per thread and its block_reduce, so the result is bit-identical (the host checks that rms_norm would take
    // exactly that path)
    if (dst2) {
        __syncthreads();
        __shared__ float s_sum[32];
        const int n4 = ne/4;
        float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float tmp = 0.0f;
        if (tid < n4) {
            xv = ((const float4 *) dst)[tid];
            tmp += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w;
        }
        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
        const float sc = rsqrtf(tmp / ne + eps2);
        if (tid < n4) {
            float4 r = make_float4(sc*xv.x, sc*xv.y, sc*xv.z, sc*xv.w);
            if (w2) {
                const float4 m = ((const float4 *) w2)[tid];
                r.x *= m.x;
                r.y *= m.y;
                r.z *= m.z;
                r.w *= m.w;
            }
            ((float4 *) dst2)[tid] = r;
        }
    }
}

bool ggml_cuda_match_depth_mix(const ggml_cgraph * cgraph, int i, bool check_memory) {
    if (i + 6 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * tr   = cgraph->nodes[i + 1];
    const ggml_tensor * ct   = cgraph->nodes[i + 2];
    const ggml_tensor * cs   = cgraph->nodes[i + 3];
    const ggml_tensor * m1   = cgraph->nodes[i + 4];
    const ggml_tensor * sm   = cgraph->nodes[i + 5];
    const ggml_tensor * m2   = cgraph->nodes[i + 6];

    if (norm->op != GGML_OP_RMS_NORM || tr->op != GGML_OP_TRANSPOSE || ct->op != GGML_OP_CONCAT || cs->op != GGML_OP_CONCAT ||
            m1->op != GGML_OP_MUL_MAT || sm->op != GGML_OP_SOFT_MAX || m2->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * cur    = norm->src[0];
    const ggml_tensor * blocks = cs->src[0];
    const ggml_tensor * query  = m1->src[1];
    const int64_t ne = cur->ne[0];

    // one token, F32
    if (norm->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) || ggml_nelements(cur) != ne ||
            tr->src[0] != norm || ct->src[1] != tr || ggml_get_op_params_i32(ct, 0) != 0 ||
            cs->src[1] != norm || ggml_get_op_params_i32(cs, 0) != 1 || cs->ne[2] != 1 || cs->ne[3] != 1 ||
            blocks->type != GGML_TYPE_F32 || blocks->ne[0] != ne || blocks->nb[0] != sizeof(float) ||
            blocks->ne[1] + 1 > DEPTH_MIX_MAX_STATES || ct->src[0]->ne[0] != blocks->ne[1]) {
        return false;
    }
    if (m1->src[0] != cs || query->type != GGML_TYPE_F32 || query->ne[0] != ne || query->nb[0] != sizeof(float) ||
            query->ne[1] > DEPTH_MIX_MAX_QUERIES || query->ne[2] != 1 || query->ne[3] != 1) {
        return false;
    }
    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    if (sm->src[0] != m1 || sm->src[1] != nullptr || sm->src[2] != nullptr || max_bias != 0.0f) {
        return false;
    }
    if (m2->src[0] != ct || m2->src[1] != sm || m2->type != GGML_TYPE_F32 || m2->ne[0] != ne || m2->ne[1] != query->ne[1] ||
            m2->ne[2] != 1 || m2->nb[0] != sizeof(float)) {
        return false;
    }
    // the intermediate results stay inside the mix
    if (ggml_node_get_use_count(cgraph, i) != 2 || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
            ggml_node_get_use_count(cgraph, i + 2) != 1 || ggml_node_get_use_count(cgraph, i + 3) != 1 ||
            ggml_node_get_use_count(cgraph, i + 4) != 1 || ggml_node_get_use_count(cgraph, i + 5) != 1) {
        return false;
    }
    for (int k = 0; k <= 6; ++k) {
        const ggml_tensor * t = cgraph->nodes[i + k];
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE) || (k < 6 && (t->flags & GGML_TENSOR_FLAG_OUTPUT))) {
            return false;
        }
    }
    // the output is written while the blocks, the query and cur are read
    for (const ggml_tensor * src : { blocks, query, cur }) {
        if (check_memory && ggml_cuda_fusion_overlap(m2, src)) {
            return false;
        }
    }
    return true;
}

static bool depth_mix_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static bool depth_mix_aligned(const ggml_tensor * t) {
    return ((uintptr_t) t->data) % 16 == 0;
}

int ggml_cuda_match_depth_mix_norm(const ggml_cgraph * cgraph, int i, bool check_memory) {
    if (!ggml_cuda_match_depth_mix(cgraph, i, check_memory)) {
        return -1;
    }
    const ggml_tensor * m2 = cgraph->nodes[i + 6];
    const int64_t ne = m2->ne[0];
    // one query, and a width for which rms_norm runs its vectorized 1024-thread kernel (513..1024 float4s)
    if (m2->ne[1] != 1 || ne % 4 != 0 || ne/4 <= 512 || ne/4 > 1024) {
        return -1;
    }
    int k = i + 7;
    while (k < cgraph->n_nodes && depth_mix_is_view(cgraph->nodes[k])) {
        ++k;
    }
    if (k >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * rn = cgraph->nodes[k];
    const ggml_tensor * v  = rn->src[0];
    while (v && v != m2 && depth_mix_is_view(v)) {
        v = v->src[0];
    }
    if (rn->op != GGML_OP_RMS_NORM || v != m2 || rn->src[0]->data != m2->data || rn->src[0]->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(rn->src[0]) || ggml_nelements(rn->src[0]) != ne || rn->src[0]->ne[0] != ne ||
            rn->type != GGML_TYPE_F32 || !ggml_is_contiguous(rn) || !(rn->flags & GGML_TENSOR_FLAG_COMPUTE)) {
        return -1;
    }
    int last = k;
    int k2 = k + 1;
    while (k2 < cgraph->n_nodes && depth_mix_is_view(cgraph->nodes[k2])) {
        ++k2;
    }
    if (k2 < cgraph->n_nodes) {
        const ggml_tensor * mul = cgraph->nodes[k2];
        if (mul->op == GGML_OP_MUL && mul->src[0] == rn && mul->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(mul->src[1]) && ggml_nelements(mul->src[1]) == ne && mul->src[1]->ne[0] == ne &&
                mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_are_same_shape(mul, rn) &&
                depth_mix_aligned(mul->src[1]) && ggml_node_get_use_count(cgraph, k) == 1 &&
                !(rn->flags & GGML_TENSOR_FLAG_OUTPUT) && (mul->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            last = k2;
        }
    }
    const ggml_tensor * out = cgraph->nodes[last];
    if (!depth_mix_aligned(m2) || !depth_mix_aligned(out)) {
        return -1;
    }
    if (check_memory) {
        const ggml_tensor * norm  = cgraph->nodes[i];
        const ggml_tensor * cs    = cgraph->nodes[i + 3];
        const ggml_tensor * m1    = cgraph->nodes[i + 4];
        // written after a barrier, each thread at the float4 it read: only the mix output (or the weight) may be
        // exactly the same memory
        for (const ggml_tensor * src : { (const ggml_tensor *) cs->src[0], (const ggml_tensor *) m1->src[1],
                                         (const ggml_tensor *) norm->src[0] }) {
            if (ggml_cuda_fusion_overlap(out, src)) {
                return -1;
            }
        }
        if (!ggml_cuda_fusion_elementwise_ok(out, m2) || (last != k && !ggml_cuda_fusion_elementwise_ok(out, cgraph->nodes[last]->src[1]))) {
            return -1;
        }
    }
    return last;
}

void ggml_cuda_op_depth_mix_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    const ggml_tensor * out = cgraph->nodes[last];
    const ggml_tensor * rn  = out->op == GGML_OP_MUL ? out->src[0] : out;
    ggml_cuda_op_depth_mix(ctx, cgraph, i, (float *) out->data, out->op == GGML_OP_MUL ? (const float *) out->src[1]->data : nullptr,
                           ggml_get_op_params_f32(rn, 0));
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, float * dst2, const float * w2, float eps2) {
    const ggml_tensor * norm   = cgraph->nodes[i];
    const ggml_tensor * cs     = cgraph->nodes[i + 3];
    const ggml_tensor * m1     = cgraph->nodes[i + 4];
    const ggml_tensor * sm     = cgraph->nodes[i + 5];
    ggml_tensor       * m2     = cgraph->nodes[i + 6];
    const ggml_tensor * cur    = norm->src[0];
    const ggml_tensor * blocks = cs->src[0];
    const ggml_tensor * query  = m1->src[1];

    const int   ne    = (int) cur->ne[0];
    const float eps   = ggml_get_op_params_f32(norm, 0);
    const float scale = ggml_get_op_params_f32(sm, 0);

    constexpr int block_size = 1024;
    depth_mix_f32<block_size><<<1, block_size, 0, ctx.stream()>>>(
        (const float *) blocks->data, blocks->nb[1]/sizeof(float), (int) blocks->ne[1],
        (const float *) cur->data, (const float *) query->data, query->nb[1]/sizeof(float), (int) query->ne[1],
        eps, scale, (float *) m2->data, m2->nb[1]/sizeof(float), ne, dst2, w2, eps2);
}
