#include "ssm_conv.hpp"
#include "common.hpp"
#include "element_wise.hpp"
#include "fusion.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>

using namespace sycl;

// One output element of the conv. DC is d_conv as a compile-time constant (0 keeps the
// runtime loop); unfused callers pass literal false/nullptr so the epilogue folds away.
template <int DC>
static __dpct_inline__ void ssm_conv_element(
    size_t idx,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // src is token-contiguous per channel, dst is channel-contiguous per token,
    // so indexing token-fastest coalesces the d_conv loads.
    const int token   = static_cast<int>(idx % n_t);
    const int channel = static_cast<int>((idx / n_t) % d_inner);
    const int seq     = static_cast<int>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

    const float *s = src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

    float sumf = 0.0f;
    if constexpr (DC > 0) {
#pragma unroll
        for (int i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } else {
        for (int i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // fused bias add: the ADD node broadcasts a 1-D channel bias over tokens
    if (bias != nullptr) {
        sumf += bias[channel];
    }

    const size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
        static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
        static_cast<size_t>(channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false keeps apply_silu/bias out of the kernel capture list, so the unfused launch
// takes the pre-fusion argument list; matters at n_t == 1, where the op is launch-bound.
template <int DC, bool FUSED>
static void kernel_ssm_conv_impl(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    if constexpr (FUSED) {
        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, apply_silu, bias);
                }
            );
        });
    } else {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                                         src_stride_inner, src_stride_seq, dst_stride_token,
                                         dst_stride_seq, false, nullptr);
                }
            );
        });
    }
}

// SLM transpose tile: coalesces both the loads and the stores. The +1 pad makes the row
// stride 33, coprime with 32 banks, so both phases are bank-conflict-free.
template <int DC, int TT, int TC, int WG>
static __dpct_inline__ void ssm_conv_tile(
    nd_item<1> it, local_accessor<float, 1> tile, const float *src_data, const float *weights,
    float *dst_data, int n_t, int nt_tiles, int nc_tiles, int src_stride_inner,
    int src_stride_seq, int dst_stride_token, int dst_stride_seq, bool apply_silu,
    const float *bias
) {
    const int    lid = static_cast<int>(it.get_local_id(0));
    const size_t g   = it.get_group(0);
    const int    tt  = static_cast<int>(g % nt_tiles);
    const int    ct  = static_cast<int>((g / nt_tiles) % nc_tiles);
    const int    seq = static_cast<int>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
    const int    t0 = tt * TT, c0 = ct * TC;

    const int ti = lid % TT;
    const int cj = lid / TT;
#pragma unroll
    for (int r = 0; r < TC / (WG / TT); ++r) {
        const int c   = cj + r * (WG / TT);
        const int tok = t0 + ti;
        float sumf = 0.0f;
        if (tok < n_t) {
            const float *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                           + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
            const float *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
            for (int i = 0; i < DC; ++i) sumf += s[i] * cw[i];
            if (bias != nullptr) sumf += bias[c0 + c];
            if (apply_silu) sumf = op_silu(sumf);
        }
        tile[c * (TT + 1) + ti] = sumf;
    }
    it.barrier(access::fence_space::local_space);

    const int cc = lid % TC;
    const int tj = lid / TC;
#pragma unroll
    for (int r = 0; r < TT / (WG / TC); ++r) {
        const int t   = tj + r * (WG / TC);
        const int tok = t0 + t;
        if (tok < n_t) {
            dst_data[static_cast<size_t>(seq) * dst_stride_seq
                     + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                = tile[cc * (TT + 1) + t];
        }
    }
}

// Same FUSED split as kernel_ssm_conv_impl. The fused instantiation keeps the runtime
// apply_silu/bias branches: at n_t >= 32 they are amortized over the whole tile.
template <int DC, bool FUSED>
static void kernel_ssm_conv_tiled(
    queue &q, const float *src_data, const float *weights, float *dst_data,
    int d_inner, int n_t, int n_s, int src_stride_inner, int src_stride_seq,
    int dst_stride_token, int dst_stride_seq, bool apply_silu, const float *bias
) {
    constexpr int TT = 32, TC = 32, WG = 256;
    const int nt_tiles = (n_t + TT - 1) / TT;
    const int nc_tiles = d_inner / TC;
    const size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    if constexpr (FUSED) {
        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, apply_silu, bias);
            });
        });
    } else {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler &h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
                ssm_conv_tile<DC, TT, TC, WG>(it, tile, src_data, weights, dst_data, n_t, nt_tiles,
                                              nc_tiles, src_stride_inner, src_stride_seq,
                                              dst_stride_token, dst_stride_seq, false, nullptr);
            });
        });
    }
}

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    bool apply_silu,
    const float *bias
) {
    // Only the fused instantiations carry apply_silu/bias as kernel arguments; the plain
    // ssm_conv launch keeps the argument list it had before the fusion landed.
    const bool fused = apply_silu || bias != nullptr;

    // d_inner must be a multiple of 32 so the channel tiles are exact; the transpose is only
    // worth it for n_t >= 32. d_conv == 4 is the only window with a DC-specialized kernel.
    if (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        if (fused) {
            kernel_ssm_conv_tiled<4, true>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                           src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        } else {
            kernel_ssm_conv_tiled<4, false>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                            src_stride_inner, src_stride_seq, dst_stride_token,
                                            dst_stride_seq, apply_silu, bias);
        }
        return;
    }

    if (d_conv == 4) {
        if (fused) {
            kernel_ssm_conv_impl<4, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                          ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                          dst_stride_seq, apply_silu, bias);
        } else {
            kernel_ssm_conv_impl<4, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                           ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                           dst_stride_seq, apply_silu, bias);
        }
        return;
    }

    if (fused) {
        kernel_ssm_conv_impl<0, true>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                      ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                      dst_stride_seq, apply_silu, bias);
    } else {
        kernel_ssm_conv_impl<0, false>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                                       ncs, src_stride_inner, src_stride_seq, dst_stride_token,
                                       dst_stride_seq, apply_silu, bias);
    }
}

inline void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr, const float * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        const bool apply_silu = silu_dst != nullptr;
        float *dst_data       = static_cast<float *>((silu_dst ? silu_dst : dst)->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq,
            apply_silu,
            bias
        );

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    // the fused kernel reads only the ADD's bias operand; the ADD result is never written
    const float * bias = nullptr;
    if (add != nullptr) {
        const ggml_tensor * bias_t = (add->src[0] == dst) ? add->src[1] : add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}

static const ggml_tensor * conv_window_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

static bool conv_window_f32_strides(const ggml_tensor * t) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->nb[d] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

bool ggml_sycl_match_conv_window(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_conv_window_match & m) {
    if (!g_ggml_sycl_fuse_conv_window || !g_ggml_sycl_enable_fusion) {
        return false;
    }

    const ggml_tensor * concat = cgraph->nodes[node_idx];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
        (concat->flags & GGML_TENSOR_FLAG_OUTPUT) || concat->ne[3] != 1 || !ggml_is_contiguous(concat)) {
        return false;
    }

    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    if (state->type != GGML_TYPE_F32 || xt->type != GGML_TYPE_F32 || !conv_window_f32_strides(state) ||
        !conv_window_f32_strides(xt)) {
        return false;
    }

    // the kernel is written for d_conv == 4: three state columns
    const int64_t state_cols = state->ne[0];
    const int64_t channels   = concat->ne[1];
    const int64_t n_seqs     = concat->ne[2];
    if (state_cols != 3 || channels <= 0 || channels > INT_MAX - 255 ||
        xt->ne[0] <= 0 || xt->ne[0] > INT_MAX - 7 || n_seqs <= 0 || n_seqs > INT_MAX) {
        return false;
    }

    ggml_sycl_conv_window_match r;
    r.concat = concat;

    const ggml_tensor * cache   = nullptr;
    const ggml_tensor * pending = nullptr;  // a cont still waiting for its cpy
    int                 n_views = 0;

    // Check intervening nodes after the cache is known.
    const int end = std::min(cgraph->n_nodes, node_idx + 64);
    for (int j = node_idx + 1; j < end && r.conv == nullptr; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_VIEW && n->src[0] == concat) {
            if ((n->flags & GGML_TENSOR_FLAG_OUTPUT) || n->ne[0] != state_cols || n->ne[1] != channels || n->ne[2] != n_seqs || n->ne[3] != 1 ||
                n->nb[1] != concat->nb[1] || n->nb[2] != concat->nb[2] || n->view_offs % sizeof(float) != 0 ||
                ggml_node_get_use_count(cgraph, j) != 1) {
                return false;
            }
            n_views++;
        } else if (n->op == GGML_OP_CONT && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == concat) {
            if (pending != nullptr || r.n_slots == GGML_SYCL_CONV_WINDOW_MAX_SLOTS ||
                !ggml_node_has_n_uses(cgraph, j, 1)) {
                return false;
            }
            pending = n;
        } else if (n->op == GGML_OP_CPY && ((pending != nullptr && n->src[0] == pending) ||
                   (pending == nullptr && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == concat))) {
            if (r.n_slots == GGML_SYCL_CONV_WINDOW_MAX_SLOTS) {
                return false;
            }
            // dst is a [state_cols * channels, n_seqs] view of the cache rows
            const ggml_tensor * dst = n->src[1];
            if (dst->type != GGML_TYPE_F32 || dst->ne[0] != state_cols * channels || dst->ne[1] != n_seqs ||
                dst->ne[2] != 1 || dst->ne[3] != 1 || dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[1] < dst->ne[0] * sizeof(float)) {
                return false;
            }
            if (cache == nullptr) {
                cache = conv_window_base(dst);
            }
            // one seq stride for all slots, and the tail must stay inside one channel row
            const ggml_tensor * tail = pending != nullptr ? pending->src[0] : n->src[0];
            const int64_t s_idx = tail->view_offs / sizeof(float);
            if (conv_window_base(dst) != cache || (r.n_slots > 0 && dst->nb[1] != r.cpy[0]->src[1]->nb[1]) ||
                s_idx + state_cols > concat->ne[0]) {
                return false;
            }
            for (int p = 0; p < r.n_slots; ++p) {
                const ggml_tensor * prev = r.cpy[p]->src[1];
                if (dst->view_offs >= prev->view_offs ? dst->view_offs - prev->view_offs < ggml_nbytes(prev) :
                                                      prev->view_offs - dst->view_offs < ggml_nbytes(dst)) {
                    return false;
                }
            }
            r.cont[r.n_slots] = pending;
            r.cpy[r.n_slots]  = n;
            r.n_slots++;
            pending = nullptr;
        } else if (n->op == GGML_OP_SSM_CONV && n->src[0] == concat) {
            r.conv     = n;
            r.conv_idx = j;
        }
    }

    if (r.conv == nullptr || r.n_slots == 0 || pending != nullptr || n_views != r.n_slots ||
        ggml_node_get_use_count(cgraph, node_idx) != n_views + 1) {
        return false;
    }

    const ggml_tensor * weights = r.conv->src[1];
    if (weights->type != GGML_TYPE_F32 || weights->ne[0] != state_cols + 1 || !ggml_is_contiguous(weights) ||
        r.conv->type != GGML_TYPE_F32 || conv_window_base(weights) == cache) {
        return false;
    }

    // the window reads state and x, the kernel writes the cache: they must not overlap
    const ggml_tensor * state_base = conv_window_base(state);
    const ggml_tensor * x_base     = conv_window_base(xt);
    if (state_base == cache || x_base == cache) {
        return false;
    }

    // Delayed cache copies must not cross a reader or an input write.
    for (int j = node_idx + 1; j < r.conv_idx; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) {
            continue;
        }
        if (std::find(r.cont, r.cont + r.n_slots, n) != r.cont + r.n_slots ||
            std::find(r.cpy, r.cpy + r.n_slots, n) != r.cpy + r.n_slots) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = n->src[s];
            if (src == nullptr) {
                continue;
            }
            const ggml_tensor * b = conv_window_base(src);
            if (b == cache || b == concat || std::find(r.cont, r.cont + r.n_slots, b) != r.cont + r.n_slots) {
                return false;
            }
        }
        const ggml_tensor * b = conv_window_base(n);
        if (b == cache || b == state_base || b == x_base) {
            return false;
        }
    }

    if (ggml_sycl_can_fuse(cgraph, r.conv_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        r.add      = cgraph->nodes[r.conv_idx + 1];
        r.silu     = cgraph->nodes[r.conv_idx + 2];
        r.last_idx = r.conv_idx + 2;
    } else if (ggml_sycl_can_fuse(cgraph, r.conv_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        r.silu     = cgraph->nodes[r.conv_idx + 1];
        r.last_idx = r.conv_idx + 1;
    } else {
        return false;
    }
    if (r.silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(r.silu)) {
        return false;
    }

    if (r.add != nullptr) {
        const ggml_tensor * bias = r.add->src[0] == r.conv ? r.add->src[1] : r.add->src[0];
        if (conv_window_base(bias) == cache) {
            return false;
        }
    }

    m = r;
    return true;
}

struct conv_window_slots {
    float * dst[GGML_SYCL_CONV_WINDOW_MAX_SLOTS];
    int     s_idx[GGML_SYCL_CONV_WINDOW_MAX_SLOTS];
    int     n;
};

// Each work-item slides a four-value window over up to eight tokens.
// Token max(0, w - 3) writes cache column w, so each cache value has one writer.
static void kernel_ssm_conv_window(queue & q, const float * state, const float * x, const float * weights,
                                   const float * bias, float * dst, conv_window_slots slots, int d_inner, int n_t,
                                   int n_s, int64_t st_col, int64_t st_ch, int64_t st_seq, int64_t x_tok,
                                   int64_t x_ch, int64_t x_seq, int64_t cache_seq) {
    constexpr int DC = 4;
    constexpr int NC = DC - 1;
    constexpr int TT = 8;

    // a 3D range: 64-bit div/mod of a flat index is emulated and costs a lot at the prefill shape
    const int wg    = 256;
    const int c_pad = (d_inner + wg - 1) / wg * wg;
    const int n_tb  = (n_t + TT - 1) / TT;

    q.parallel_for(nd_range<3>(range<3>(n_s, n_tb, c_pad), range<3>(1, 1, wg)), [=](nd_item<3> item) {
        const int c  = static_cast<int>(item.get_global_id(2));
        const int t0 = static_cast<int>(item.get_global_id(1)) * TT;
        const int s  = static_cast<int>(item.get_global_id(0));
        if (c >= d_inner) {
            return;
        }
        const int t1 = sycl::min(t0 + TT, n_t);

        const float * sp = state + s * st_seq + c * st_ch;
        const float * xp = x + s * x_seq + c * x_ch;

        float win[DC];
#pragma unroll
        for (int k = 0; k < NC; ++k) {
            const int w = t0 + k;
            win[k + 1] = w < NC ? sp[w * st_col] : xp[(w - NC) * x_tok];
        }

        float cw[DC];
#pragma unroll
        for (int k = 0; k < DC; ++k) {
            cw[k] = weights[static_cast<size_t>(c) * DC + k];
        }
        const float b = bias != nullptr ? bias[c] : 0.0f;

        for (int t = t0; t < t1; ++t) {
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                win[k] = win[k + 1];
            }
            win[NC] = xp[t * x_tok];

            float sumf = 0.0f;
#pragma unroll
            for (int k = 0; k < DC; ++k) {
                sumf += win[k] * cw[k];
            }
            if (bias != nullptr) {
                sumf += b;
            }
            dst[(static_cast<size_t>(s) * n_t + t) * d_inner + c] = op_silu(sumf);

            // constant win[] indices keep the window in registers
            for (int p = 0; p < slots.n; ++p) {
                float * cd = slots.dst[p] + s * cache_seq + static_cast<size_t>(c) * NC;
#pragma unroll
                for (int k = 0; k < DC; ++k) {
                    const int w = t + k;
                    const int j = w - slots.s_idx[p];
                    if (j >= 0 && j < NC && sycl::max(w - NC, 0) == t) {
                        cd[j] = win[k];
                    }
                }
            }
        }
    });
}

void ggml_sycl_ssm_conv_window(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_window_match & m) {
    const ggml_tensor * state = m.concat->src[0];
    const ggml_tensor * xt    = m.concat->src[1];

    const int d_inner = m.concat->ne[1];
    const int n_t     = xt->ne[0];
    const int n_s     = m.concat->ne[2];

    GGML_ASSERT(m.conv->ne[0] == d_inner && m.conv->ne[1] == n_t && m.conv->ne[2] == n_s);

    conv_window_slots slots = {};
    slots.n = m.n_slots;
    for (int p = 0; p < m.n_slots; ++p) {
        const ggml_tensor * tail = m.cpy[p]->src[0];
        if (tail->op == GGML_OP_CONT) {
            tail = tail->src[0];
        }
        slots.dst[p]   = static_cast<float *>(m.cpy[p]->src[1]->data);
        slots.s_idx[p] = static_cast<int>(tail->view_offs / sizeof(float));
    }

    const float * bias = nullptr;
    if (m.add != nullptr) {
        const ggml_tensor * bias_t = (m.add->src[0] == m.conv) ? m.add->src[1] : m.add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }

    constexpr int64_t fs = sizeof(float);
    kernel_ssm_conv_window(*ctx.stream(), static_cast<const float *>(state->data),
                           static_cast<const float *>(xt->data), static_cast<const float *>(m.conv->src[1]->data),
                           bias, static_cast<float *>(m.silu->data), slots, d_inner, n_t, n_s, state->nb[0] / fs,
                           state->nb[1] / fs, state->nb[2] / fs, xt->nb[0] / fs, xt->nb[1] / fs, xt->nb[2] / fs,
                           m.cpy[0]->src[1]->nb[1] / fs);
}
