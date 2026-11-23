#include "situ.cuh"
#include "ctrl-mix.cuh"

namespace {

bool is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

bool is_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

bool is_scale(const ggml_tensor * t) {
    return t->op == GGML_OP_SCALE && ggml_get_op_params_f32(t, 1) == 0.0f;
}

// the 9 compute nodes of the pattern, views skipped
bool match_nodes(const ggml_cgraph * cgraph, int i, int idx[9]) {
    int n = 0;
    for (int k = i; k < cgraph->n_nodes && n < 9; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (is_view(t) && n > 0) {
            continue;
        }
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return false;
        }
        idx[n++] = k;
    }
    if (n < 9) {
        return false;
    }
    const ggml_tensor * t[9];
    for (int k = 0; k < 9; ++k) {
        t[k] = cgraph->nodes[idx[k]];
    }
    const ggml_tensor * g = t[0]->src[0];
    const ggml_tensor * u = t[5]->src[0];
    if (!is_scale(t[0]) || !is_unary(t[1], GGML_UNARY_OP_TANH) || !is_scale(t[2]) ||
            !is_unary(t[3], GGML_UNARY_OP_SIGMOID) || t[4]->op != GGML_OP_MUL ||
            !is_scale(t[5]) || !is_unary(t[6], GGML_UNARY_OP_TANH) || !is_scale(t[7]) || t[8]->op != GGML_OP_MUL) {
        return false;
    }
    if (t[1]->src[0] != t[0] || t[2]->src[0] != t[1] || t[3]->src[0] != g ||
            !((t[4]->src[0] == t[2] && t[4]->src[1] == t[3]) || (t[4]->src[0] == t[3] && t[4]->src[1] == t[2])) ||
            t[6]->src[0] != t[5] || t[7]->src[0] != t[6] ||
            !((t[8]->src[0] == t[4] && t[8]->src[1] == t[7]) || (t[8]->src[0] == t[7] && t[8]->src[1] == t[4]))) {
        return false;
    }
    // F32 throughout, g and u of one shape with a unit element stride, the computed tensors contiguous
    if (g->type != GGML_TYPE_F32 || u->type != GGML_TYPE_F32 || g->nb[0] != sizeof(float) || u->nb[0] != sizeof(float) ||
            !ggml_are_same_shape(g, u) || g->ne[3] != 1) {
        return false;
    }
    for (int k = 0; k < 9; ++k) {
        if (t[k]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t[k]) || !ggml_are_same_shape(t[k], g)) {
            return false;
        }
    }
    // everything but the last node is read inside the pattern only
    for (int k = 0; k < 8; ++k) {
        if (ggml_node_get_use_count(cgraph, idx[k]) != 1 || (t[k]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
    }
    return true;
}

__global__ void situ_f32(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ dst,
        const int64_t k, const int ne0, const int ne1, const int64_t g1, const int64_t g2, const int64_t u1, const int64_t u2,
        const float gs, const float gc, const float us, const float uc) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int     i0 = (int) (i % ne0);
    const int64_t r  = i / ne0;
    const int     i1 = (int) (r % ne1);
    const int64_t i2 = r / ne1;

    // the expressions of the softcap, unary and binary kernels this replaces, in their order
    const float gv  = g[i2*g2 + i1*g1 + i0];
    const float uv  = u[i2*u2 + i1*u1 + i0];
    const float act = tanhf(gs * gv) * gc;
    const float sig = 1.0f / (1.0f + expf(-gv));
    const float up  = tanhf(us * uv) * uc;
    dst[i] = (act * sig) * up;
}

} // namespace

int ggml_cuda_match_situ(const ggml_cgraph * cgraph, int i, bool check_memory) {
    int idx[9];
    if (!match_nodes(cgraph, i, idx)) {
        return -1;
    }
    // element-wise, but the inputs are strided: the output must not overlap them at all
    const ggml_tensor * out = cgraph->nodes[idx[8]];
    if (check_memory && (ggml_cuda_fusion_overlap(out, cgraph->nodes[idx[0]]->src[0]) ||
                         ggml_cuda_fusion_overlap(out, cgraph->nodes[idx[5]]->src[0]))) {
        return -1;
    }
    return idx[8];
}

void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    int idx[9];
    GGML_ASSERT(match_nodes(cgraph, i, idx) && idx[8] == last);

    const ggml_tensor * g   = cgraph->nodes[idx[0]]->src[0];
    const ggml_tensor * u   = cgraph->nodes[idx[5]]->src[0];
    ggml_tensor       * dst = cgraph->nodes[idx[8]];

    const int64_t k = ggml_nelements(dst);
    constexpr int block_size = 256;
    situ_f32<<<(k + block_size - 1)/block_size, block_size, 0, ctx.stream()>>>(
        (const float *) g->data, (const float *) u->data, (float *) dst->data, k, (int) g->ne[0], (int) g->ne[1],
        g->nb[1]/sizeof(float), g->nb[2]/sizeof(float), u->nb[1]/sizeof(float), u->nb[2]/sizeof(float),
        ggml_get_op_params_f32(cgraph->nodes[idx[0]], 0), ggml_get_op_params_f32(cgraph->nodes[idx[2]], 0),
        ggml_get_op_params_f32(cgraph->nodes[idx[5]], 0), ggml_get_op_params_f32(cgraph->nodes[idx[7]], 0));
}
