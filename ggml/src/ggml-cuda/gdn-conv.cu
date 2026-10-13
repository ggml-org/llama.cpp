#include "gdn-conv.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cstdlib>

struct gdn_conv_match {
    int                 conv_idx = -1;
    const ggml_tensor * x        = nullptr; // [C, T] F32 contiguous, the root of the transposed view
    const ggml_tensor * state    = nullptr; // [3, C] F32 contiguous
    const ggml_tensor * concat   = nullptr; // [T + 3, C] F32 contiguous
    const ggml_tensor * w        = nullptr; // [4, C] F32 contiguous
    ggml_tensor *       out      = nullptr; // the silu node
    int64_t C = 0, T = 0;
    int64_t tail_from = 0; // first concat column that the snapshot copies read
    bool    in_place  = false; // the conv output is x's memory
};

static bool gdn_conv_enabled() {
    static const bool enabled = getenv("GGML_CUDA_DISABLE_CONV_FUSION") == nullptr;
    return enabled;
}

// columns [tail_from, T + 3) of every channel row of the concat
static __global__ void gdn_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int tail_from, const int row_stride) {
    const int     ncols = T + 3 - tail_from;
    const int64_t idx   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) {
        return;
    }
    const int c = (int) (idx / ncols);
    const int j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = j < 3 ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c];
}

// In place (y is x's memory) the tiles over t would race: tile k reads the 3 rows of x before t0 that tile k - 1 writes.
// So those rows are first saved per tile into concat columns [3k, 3k + 3) (below the tail, which nothing reads), and
// each tile reads its window's first 3 columns from there: halo[c * row_stride + 3k + r] holds x row t0 - 3 + r.
static __global__ void gdn_conv_save_halo(const float * __restrict__ x, float * __restrict__ halo, const int C, const int T,
                                          const int tt, const int ntiles, const int row_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y + 1; // tile 0 starts from the state
    if (c >= C || k >= ntiles) {
        return;
    }
    const int t0 = k * tt;
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        halo[(size_t) c * row_stride + 3 * k + r] = x[(size_t) (t0 - 3 + r) * C + c];
    }
}

// y[t, c] = silu(sum_j concat[c, t + j] * w[c, j]) with the concat read from the state, x and (in place) the saved
// halos. A block covers BS channels and tt steps; within its tile a thread reads row t + 1 before writing row t of its
// own column, so y may be x.
template <int BS>
static __global__ void __launch_bounds__(BS) gdn_conv_direct(const float * __restrict__ state, const float * x,
                                                            const float * __restrict__ halo, const int halo_stride,
                                                            const float * __restrict__ w, float * y,
                                                            const int C, const int T, const int tt, const float b) {
    const int c  = blockIdx.x * BS + threadIdx.x;
    const int k  = blockIdx.y;
    const int t0 = k * tt;
    if (c >= C) {
        return;
    }
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wr[j] = w[c * 4 + j];
    }
    float xw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int jp = t0 + j; // concat column
        if (jp < 3) {
            xw[j] = state[c * 3 + jp];
        } else if (halo != nullptr && j < 3) {
            xw[j] = halo[(size_t) c * halo_stride + 3 * k + j];
        } else {
            xw[j] = x[(size_t) (jp - 3) * C + c];
        }
    }
    const int tend = min(T, t0 + tt);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += xw[j] * wr[j];
        }
        sumf += b; // the unfused conv adds its (zero) bias at run time
        // the next row of this tile (the first row of the next tile belongs to its block)
        const float next = t + 1 < tend ? x[(size_t) (t + 1) * C + c] : 0.0f;
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(sumf);
        xw[0] = xw[1];
        xw[1] = xw[2];
        xw[2] = xw[3];
        xw[3] = next;
    }
}

// i: the CONCAT node
static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, gdn_conv_match & m) {
    if (!gdn_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) {
        return false;
    }
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || tr->op != GGML_OP_TRANSPOSE || !tr->view_src) {
        return false;
    }
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) {
        return false;
    }
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) {
        return false;
    }
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) {
        return false;
    }
    // prefill ubatches, and on RDNA4 also the 2..255-token verify batches (their concat is otherwise a generic copy)
    const bool rdna4 = GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
    if ((T < 256 && !(rdna4 && T >= 2)) || C % 256 != 0) {
        return false;
    }

    // the concat's own node is replaced by the tail write; the evaluator skips it together with the next node, a view
    if (cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) {
        return false;
    }

    // between the concat and the conv: only 3-column views of the concat and the snapshot copies of them
    int64_t tail_from = T + 3;
    int     conv      = -1;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) {
            conv = n;
            break;
        }
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            reads = reads || t->src[s] == cc || t->src[s]->view_src == cc;
        }
        if (!reads) {
            continue;
        }
        if (t->op == GGML_OP_VIEW) {
            if (t->ne[0] != 3 || t->ne[1] != C || t->nb[1] != cc->nb[1] || t->view_offs % sizeof(float)) {
                return false;
            }
            const int64_t col = (int64_t) (t->view_offs / sizeof(float));
            if (col < 0 || col + 3 > T + 3) {
                return false;
            }
            tail_from = std::min(tail_from, col);
            continue;
        }
        if (t->op == GGML_OP_CPY) {
            const ggml_tensor * s0 = t->src[0];
            if (!s0 || s0->op != GGML_OP_VIEW || s0->view_src != cc || s0->ne[0] != 3 || s0->ne[1] != C || s0->ne[2] != 1) {
                return false;
            }
            continue;
        }
        return false;
    }
    if (conv < 0 || conv + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w  = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) {
        return false;
    }
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || !ggml_is_contiguous(cv)) {
        return false;
    }
    ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv ||
            su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) {
        return false;
    }
    // the raw conv output is never written, so nothing else may read it
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            if (t->src[s] == cv || t->src[s]->view_src == cv) {
                return false;
            }
        }
    }
    // The conv runs at the concat, where x is still live; in the unfused graph x is dead after the concat, so the
    // allocator may give its memory to the conv output (seen: the silu output is exactly x) or to tensors created
    // between the concat and the conv. Writing the output early is safe when its region holds nothing else then:
    // it is x itself (the in-place kernel) or disjoint from x, the state and the concat, and no node between the
    // concat and the conv reads or writes it.
    const char * o0 = (const char *) su->data;
    const char * o1 = o0 + ggml_nbytes(su);
    const auto overlaps = [&](const ggml_tensor * t) {
        const char * a0 = (const char *) t->data;
        return a0 != nullptr && a0 < o1 && o0 < a0 + ggml_nbytes(t);
    };
    const bool in_place = su->data == x->data && ggml_nbytes(su) == ggml_nbytes(x);
    if ((overlaps(x) && !in_place) || overlaps(st) || overlaps(cc) || overlaps(w)) {
        return false;
    }
    // in place, tile k's halo goes in concat columns [3k, 3k + 3): they must lie below the tail the copies read
    constexpr int64_t TT = 32;
    if (in_place && 3 * ((T + TT - 1) / TT) > tail_from) {
        return false;
    }
    for (int n = i + 1; n < conv; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->view_src == nullptr && overlaps(t)) {
            return false;
        }
        for (int s2 = 0; s2 < GGML_MAX_SRC && t->src[s2]; ++s2) {
            if (overlaps(t->src[s2])) {
                return false;
            }
        }
    }

    m.conv_idx  = conv;
    m.x         = x;
    m.state     = st;
    m.concat    = cc;
    m.w         = w;
    m.out       = su;
    m.C         = C;
    m.T         = T;
    m.tail_from = tail_from;
    m.in_place  = in_place;
    return true;
}

int ggml_cuda_try_fuse_gdn_conv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    gdn_conv_match m;
    if (node->op == GGML_OP_CONCAT) {
        if (!gdn_conv_check(cgraph, i, m)) {
            return 0;
        }
        const int ncols = (int) (m.T + 3 - m.tail_from);
        if (ncols > 0) {
            const int64_t n = m.C * ncols;
            gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>(
                (const float *) m.state->data, (const float *) m.x->data, (float *) m.concat->data,
                (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
            CUDA_CHECK(cudaGetLastError());
        }
        constexpr int BS = 256;
        constexpr int TT = 32;
        const int ntiles     = (int) ((m.T + TT - 1) / TT);
        const int row_stride = (int) (m.concat->nb[1] / sizeof(float));
        float *   halo       = nullptr;
        if (m.in_place && ntiles > 1) {
            // halos go in concat columns [3, 3 * ntiles), below the tail (checked by gdn_conv_check)
            halo = (float *) m.concat->data;
            const dim3 hgrid((unsigned) (m.C / BS), (unsigned) (ntiles - 1));
            gdn_conv_save_halo<<<hgrid, BS, 0, ctx.stream()>>>((const float *) m.x->data, halo, (int) m.C, (int) m.T, TT,
                                                               ntiles, row_stride);
            CUDA_CHECK(cudaGetLastError());
        }
        const dim3 grid((unsigned) (m.C / BS), (unsigned) ntiles);
        gdn_conv_direct<BS><<<grid, BS, 0, ctx.stream()>>>(
            (const float *) m.state->data, (const float *) m.x->data, halo, row_stride, (const float *) m.w->data,
            (float *) m.out->data, (int) m.C, (int) m.T, TT, 0.0f);
        CUDA_CHECK(cudaGetLastError());
        return 1; // the concat and the view after it; the other views and the copies run as usual
    }
    if (node->op == GGML_OP_SSM_CONV && gdn_conv_enabled()) {
        // already computed at the concat: skip the conv and the silu
        const ggml_tensor * cc = node->src[0];
        for (int j = i - 1; j >= 0 && j >= i - 64; --j) {
            if (cgraph->nodes[j] == cc) {
                return gdn_conv_check(cgraph, j, m) && m.conv_idx == i ? 1 : 0;
            }
        }
    }
    return 0;
}
