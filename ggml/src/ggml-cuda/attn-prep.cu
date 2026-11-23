#include "attn-prep.cuh"
#include "ctrl-mix.cuh"

namespace {

struct attn_prep_nodes {
    int q_mul = -1, q_l2 = -1, q_tau = -1;
    int k_l2  = -1, v_rms = -1, k_set = -1, v_set = -1;
    int r_mul = -1, r_l2 = -1, r_scale = -1, r_tau = -1;
    int last  = -1;
};

struct attn_prep_params {
    const float * q; const float * wq; float * qo;
    const float * k; float * ko; char * kc; int64_t kc_s1; const void * k_idx; bool k_idx64;
    const float * v; float * vo; char * vc; int64_t vc_s1; const void * v_idx; bool v_idx64;
    const float * r; const float * wr; float * ro;
    const float * tau_q; const float * tau_r;
    float eps_q, eps_k, eps_v, eps_r, scale_r;
    int nq, nk, nv, nr;
    int dq, dk, dv, dr;
};

bool is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// t, or the tensor t is a chain of views of
const ggml_tensor * view_root(const ggml_tensor * t) {
    while (t && is_view(t)) {
        t = t->src[0];
    }
    return t;
}

bool f32_rows(const ggml_tensor * t, int64_t d) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == d;
}

// a one-element F32 per-token factor (tau)
bool is_scalar_factor(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_nelements(t) == 1;
}

// one token, one row index, a contiguous F16 destination row as wide as the stored tensor
bool set_rows_ok(const ggml_tensor * set, const ggml_tensor * norm) {
    const ggml_tensor * src = set->src[0];
    const ggml_tensor * idx = set->src[1];
    return view_root(src) == norm && ggml_is_contiguous(src) && ggml_nelements(src) == ggml_nelements(norm) &&
        src->ne[0] == set->ne[0] && ggml_nrows(src) == 1 &&
        set->type == GGML_TYPE_F16 && set->nb[0] == sizeof(half) &&
        (idx->type == GGML_TYPE_I64 || idx->type == GGML_TYPE_I32) && ggml_nelements(idx) == 1;
}

// the index of node t in [i, last], or -1
int node_index(const ggml_cgraph * cgraph, int i, int last, const ggml_tensor * t) {
    for (int k = i; k <= last; ++k) {
        if (cgraph->nodes[k] == t) {
            return k;
        }
    }
    return -1;
}

// norm (node k) reaches nothing but the SET_ROWS set, through views that nothing else reads: its F32 result need not
// be written at all
bool only_stored(const ggml_cgraph * cgraph, int i, int last, int k, const ggml_tensor * set) {
    const ggml_tensor * norm = cgraph->nodes[k];
    if (norm->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    for (const ggml_tensor * t = set->src[0]; ; t = t->src[0]) {
        const int kt = node_index(cgraph, i, last, t);
        if (kt < 0 || ggml_node_get_use_count(cgraph, kt) != 1 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        if (t == norm) {
            return true;
        }
    }
}

bool intermediate(const ggml_cgraph * cgraph, int k) {
    const ggml_tensor * t = cgraph->nodes[k];
    return ggml_node_get_use_count(cgraph, k) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

bool match(const ggml_cgraph * cgraph, int i, attn_prep_nodes & m) {
    // the compute nodes from i on, views skipped
    int idx[11];
    int n = 0;
    for (int k = i; k < cgraph->n_nodes && n < 11; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (is_view(t) && n > 0) {
            continue;
        }
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return false;
        }
        idx[n++] = k;
    }
    auto node = [&](int j) -> const ggml_tensor * { return j < n ? cgraph->nodes[idx[j]] : nullptr; };
    auto op_is = [&](int j, ggml_op op) { const ggml_tensor * t = node(j); return t && t->op == op; };

    int j = 0;
    // Q: MUL -> L2_NORM [-> MUL tau]
    if (!op_is(j, GGML_OP_MUL) || !op_is(j + 1, GGML_OP_L2_NORM) || node(j + 1)->src[0] != node(j)) {
        return false;
    }
    m.q_mul = idx[j]; m.q_l2 = idx[j + 1]; j += 2;
    if (op_is(j, GGML_OP_MUL) && node(j)->src[0] == cgraph->nodes[m.q_l2] && is_scalar_factor(node(j)->src[1])) {
        m.q_tau = idx[j++];
    }
    // K: L2_NORM, V: RMS_NORM
    if (!op_is(j, GGML_OP_L2_NORM) || !op_is(j + 1, GGML_OP_RMS_NORM)) {
        return false;
    }
    m.k_l2 = idx[j]; m.v_rms = idx[j + 1]; j += 2;
    // SET_ROWS K, SET_ROWS V
    if (!op_is(j, GGML_OP_SET_ROWS) || !op_is(j + 1, GGML_OP_SET_ROWS) ||
            !set_rows_ok(node(j), cgraph->nodes[m.k_l2]) || !set_rows_ok(node(j + 1), cgraph->nodes[m.v_rms])) {
        return false;
    }
    m.k_set = idx[j]; m.v_set = idx[j + 1]; j += 2;
    // R: MUL -> L2_NORM -> SCALE [-> MUL tau]
    if (!op_is(j, GGML_OP_MUL) || !op_is(j + 1, GGML_OP_L2_NORM) || !op_is(j + 2, GGML_OP_SCALE) ||
            node(j + 1)->src[0] != node(j) || node(j + 2)->src[0] != node(j + 1) ||
            ggml_get_op_params_f32(node(j + 2), 1) != 0.0f) {
        return false;
    }
    m.r_mul = idx[j]; m.r_l2 = idx[j + 1]; m.r_scale = idx[j + 2]; j += 3;
    if (op_is(j, GGML_OP_MUL) && node(j)->src[0] == cgraph->nodes[m.r_scale] && is_scalar_factor(node(j)->src[1])) {
        m.r_tau = idx[j++];
    }
    if ((m.q_tau < 0) != (m.r_tau < 0)) {
        return false;
    }
    m.last = idx[j - 1];

    // shapes: Q and R with a weight of their own shape, K and V rows of the head size, everything F32 and contiguous
    const ggml_tensor * qm = cgraph->nodes[m.q_mul];
    const ggml_tensor * rm = cgraph->nodes[m.r_mul];
    const ggml_tensor * kn = cgraph->nodes[m.k_l2];
    const ggml_tensor * vn = cgraph->nodes[m.v_rms];
    for (const ggml_tensor * t : { qm, (const ggml_tensor *) qm->src[0], (const ggml_tensor *) qm->src[1],
                                   (const ggml_tensor *) cgraph->nodes[m.q_l2] }) {
        if (!f32_rows(t, qm->ne[0]) || !ggml_are_same_shape(t, qm)) {
            return false;
        }
    }
    for (const ggml_tensor * t : { rm, (const ggml_tensor *) rm->src[0], (const ggml_tensor *) rm->src[1],
                                   (const ggml_tensor *) cgraph->nodes[m.r_l2], (const ggml_tensor *) cgraph->nodes[m.r_scale] }) {
        if (!f32_rows(t, rm->ne[0]) || !ggml_are_same_shape(t, rm)) {
            return false;
        }
    }
    for (const ggml_tensor * t : { kn, (const ggml_tensor *) kn->src[0], vn, (const ggml_tensor *) vn->src[0] }) {
        if (!f32_rows(t, kn->ne[0])) {
            return false;
        }
    }
    if (!ggml_are_same_shape(kn, kn->src[0]) || !ggml_are_same_shape(vn, vn->src[0]) || !ggml_are_same_shape(kn, vn) ||
            ggml_nrows(qm) > 1024 || ggml_nrows(rm) > 1024 || ggml_nrows(kn) > 1024) {
        return false;
    }
    for (int t : { m.q_tau, m.r_tau }) {
        const ggml_tensor * ref = t == m.q_tau ? qm : rm;
        if (t >= 0 && (!f32_rows(cgraph->nodes[t], ref->ne[0]) || !ggml_are_same_shape(cgraph->nodes[t], ref))) {
            return false;
        }
    }
    // the steps inside a chain are read by the next step only
    for (int t : { m.q_mul, m.r_mul, m.r_l2 }) {
        if (!intermediate(cgraph, t)) {
            return false;
        }
    }
    if ((m.q_tau >= 0 && !intermediate(cgraph, m.q_l2)) || (m.r_tau >= 0 && !intermediate(cgraph, m.r_scale))) {
        return false;
    }
    return true;
}

// the cache row a SET_ROWS writes, read on the device (the index is a graph input)
__device__ __forceinline__ half * cache_row(char * base, const int64_t s1, const void * idx, const bool idx64) {
    const int64_t r = idx64 ? *(const int64_t *) idx : (int64_t) *(const int32_t *) idx;
    return (half *) (base + r*s1);
}

__global__ void attn_prep_f32(const attn_prep_params p) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x;
    int row = blockIdx.x;

    if (row < p.nq) {
        const float * x = p.q  + (int64_t) row*p.dq;
        const float * w = p.wq + (int64_t) row*p.dq;
        float sum = 0.0f;
        for (int c = lane; c < p.dq; c += warp_size) {
            const float v = x[c]*w[c];
            sum += v*v;
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float s = rsqrtf(fmaxf(sum, p.eps_q*p.eps_q));
        float * d = p.qo + (int64_t) row*p.dq;
        for (int c = lane; c < p.dq; c += warp_size) {
            const float o = s*(x[c]*w[c]);
            d[c] = p.tau_q ? o*p.tau_q[0] : o;
        }
        return;
    }
    row -= p.nq;
    if (row < p.nk) {
        const float * x = p.k + (int64_t) row*p.dk;
        float sum = 0.0f;
        for (int c = lane; c < p.dk; c += warp_size) {
            sum += x[c]*x[c];
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float s = rsqrtf(fmaxf(sum, p.eps_k*p.eps_k));
        float * d  = p.ko + (int64_t) row*p.dk;
        half  * dc = cache_row(p.kc, p.kc_s1, p.k_idx, p.k_idx64) + (int64_t) row*p.dk;
        for (int c = lane; c < p.dk; c += warp_size) {
            const float o = s*x[c];
            if (p.ko) {
                d[c] = o;
            }
            dc[c] = __float2half(o);
        }
        return;
    }
    row -= p.nk;
    if (row < p.nv) {
        const float * x = p.v + (int64_t) row*p.dv;
        float sum = 0.0f;
        if (p.dv % 4 == 0 && p.dv/4 <= warp_size) {
            // the summation order of the vectorized rms_norm kernel (four consecutive values per lane), so that the
            // result is bit-identical to the unfused op
            if (lane < p.dv/4) {
                const float4 x4 = ((const float4 *) x)[lane];
                sum = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + x4.w*x4.w;
            }
        } else {
            for (int c = lane; c < p.dv; c += warp_size) {
                sum += x[c]*x[c];
            }
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float s = rsqrtf(sum/p.dv + p.eps_v);
        float * d  = p.vo + (int64_t) row*p.dv;
        half  * dc = cache_row(p.vc, p.vc_s1, p.v_idx, p.v_idx64) + (int64_t) row*p.dv;
        for (int c = lane; c < p.dv; c += warp_size) {
            const float o = s*x[c];
            if (p.vo) {
                d[c] = o;
            }
            dc[c] = __float2half(o);
        }
        return;
    }
    row -= p.nv;
    if (row < p.nr) {
        const float * x = p.r  + (int64_t) row*p.dr;
        const float * w = p.wr + (int64_t) row*p.dr;
        float sum = 0.0f;
        for (int c = lane; c < p.dr; c += warp_size) {
            const float v = x[c]*w[c];
            sum += v*v;
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float s = rsqrtf(fmaxf(sum, p.eps_r*p.eps_r));
        float * d = p.ro + (int64_t) row*p.dr;
        for (int c = lane; c < p.dr; c += warp_size) {
            const float o = (s*(x[c]*w[c]))*p.scale_r;
            d[c] = p.tau_r ? o*p.tau_r[0] : o;
        }
    }
}

} // namespace

int ggml_cuda_match_attn_prep(const ggml_cgraph * cgraph, int i, bool check_memory) {
    attn_prep_nodes m;
    if (!match(cgraph, i, m)) {
        return -1;
    }
    if (check_memory) {
        const ggml_tensor * qm = cgraph->nodes[m.q_mul];
        const ggml_tensor * rm = cgraph->nodes[m.r_mul];
        const ggml_tensor * kn = cgraph->nodes[m.k_l2];
        const ggml_tensor * vn = cgraph->nodes[m.v_rms];
        const ggml_tensor * qo = cgraph->nodes[m.q_tau >= 0 ? m.q_tau : m.q_l2];
        const ggml_tensor * ro = cgraph->nodes[m.r_tau >= 0 ? m.r_tau : m.r_scale];
        const ggml_tensor * ks = cgraph->nodes[m.k_set];
        const ggml_tensor * vs = cgraph->nodes[m.v_set];
        const ggml_tensor * ins[] = { qm->src[0], qm->src[1], rm->src[0], rm->src[1], kn->src[0], vn->src[0],
                                      ks->src[1], vs->src[1],
                                      m.q_tau >= 0 ? (const ggml_tensor *) cgraph->nodes[m.q_tau]->src[1] : nullptr,
                                      m.r_tau >= 0 ? (const ggml_tensor *) cgraph->nodes[m.r_tau]->src[1] : nullptr };
        // each output row is written by the warp that read its own inputs: an output may be exactly its own
        // element-wise input, never anything else that is read
        const ggml_tensor * own[][3] = { { qo, qm->src[0], qm->src[1] }, { ro, rm->src[0], rm->src[1] },
                                         { kn, kn->src[0], nullptr }, { vn, vn->src[0], nullptr } };
        for (auto & o : own) {
            for (const ggml_tensor * in : ins) {
                if (!in) {
                    continue;
                }
                const bool mine = in == o[1] || in == o[2];
                if (mine ? !ggml_cuda_fusion_elementwise_ok(o[0], in) : ggml_cuda_fusion_overlap(o[0], in)) {
                    return -1;
                }
            }
        }
        // the outputs are written at the same time by different warps: the ones written must not overlap each other
        // (the allocator may give R the memory K and V die in, since unfused those die before R is computed)
        {
            const bool k_out = !only_stored(cgraph, i, m.last, m.k_l2, ks);
            const bool v_out = !only_stored(cgraph, i, m.last, m.v_rms, vs);
            const ggml_tensor * outs[] = { qo, ro, k_out ? kn : nullptr, v_out ? vn : nullptr, ks, vs };
            for (int a = 0; a < 6; ++a) {
                for (int b = a + 1; b < 6; ++b) {
                    if (outs[a] && outs[b] && ggml_cuda_fusion_overlap(outs[a], outs[b])) {
                        return -1;
                    }
                }
            }
        }
        // the cache rows: whole cache views, which must not overlap anything read here
        for (const ggml_tensor * c : { ks, vs }) {
            for (const ggml_tensor * in : ins) {
                if (in && ggml_cuda_fusion_overlap(c, in)) {
                    return -1;
                }
            }
        }
    }
    return m.last;
}

void ggml_cuda_op_attn_prep(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last) {
    attn_prep_nodes m;
    GGML_ASSERT(match(cgraph, i, m) && m.last == last);

    const ggml_tensor * qm = cgraph->nodes[m.q_mul];
    const ggml_tensor * rm = cgraph->nodes[m.r_mul];
    ggml_tensor       * kn = cgraph->nodes[m.k_l2];
    ggml_tensor       * vn = cgraph->nodes[m.v_rms];
    ggml_tensor       * qo = cgraph->nodes[m.q_tau >= 0 ? m.q_tau : m.q_l2];
    ggml_tensor       * ro = cgraph->nodes[m.r_tau >= 0 ? m.r_tau : m.r_scale];
    const ggml_tensor * ks = cgraph->nodes[m.k_set];
    const ggml_tensor * vs = cgraph->nodes[m.v_set];

    attn_prep_params p;
    p.q = (const float *) qm->src[0]->data; p.wq = (const float *) qm->src[1]->data; p.qo = (float *) qo->data;
    p.k = (const float *) kn->src[0]->data; p.ko = only_stored(cgraph, i, last, m.k_l2, ks) ? nullptr : (float *) kn->data;
    p.v = (const float *) vn->src[0]->data; p.vo = only_stored(cgraph, i, last, m.v_rms, vs) ? nullptr : (float *) vn->data;
    p.r = (const float *) rm->src[0]->data; p.wr = (const float *) rm->src[1]->data; p.ro = (float *) ro->data;
    p.tau_q = m.q_tau >= 0 ? (const float *) cgraph->nodes[m.q_tau]->src[1]->data : nullptr;
    p.tau_r = m.r_tau >= 0 ? (const float *) cgraph->nodes[m.r_tau]->src[1]->data : nullptr;
    p.eps_q   = ggml_get_op_params_f32(cgraph->nodes[m.q_l2], 0);
    p.eps_k   = ggml_get_op_params_f32(kn, 0);
    p.eps_v   = ggml_get_op_params_f32(vn, 0);
    p.eps_r   = ggml_get_op_params_f32(cgraph->nodes[m.r_l2], 0);
    p.scale_r = ggml_get_op_params_f32(cgraph->nodes[m.r_scale], 0);
    p.dq = (int) qm->ne[0]; p.nq = (int) ggml_nrows(qm);
    p.dk = (int) kn->ne[0]; p.nk = (int) ggml_nrows(kn);
    p.dv = (int) vn->ne[0]; p.nv = (int) ggml_nrows(vn);
    p.dr = (int) rm->ne[0]; p.nr = (int) ggml_nrows(rm);
    p.kc = (char *) ks->data; p.kc_s1 = ks->nb[1]; p.k_idx = ks->src[1]->data; p.k_idx64 = ks->src[1]->type == GGML_TYPE_I64;
    p.vc = (char *) vs->data; p.vc_s1 = vs->nb[1]; p.v_idx = vs->src[1]->data; p.v_idx64 = vs->src[1]->type == GGML_TYPE_I64;

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    attn_prep_f32<<<p.nq + p.nk + p.nv + p.nr, warp_size, 0, ctx.stream()>>>(p);
}
