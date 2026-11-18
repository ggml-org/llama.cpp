#include "htp-lower.h"

#include "htp-common.h"
#include "htp-kparams.h"
#include "htp-opnode.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

bool htp_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (ne[i] == 1) continue;
        if (s == 0) { s = nb[i]; next = s * ne[i]; continue; }
        if (nb[i] != next) return false;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    return true;
}

bool htp_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    size_t s1, sd;
    return (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && htp_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && htp_rows_stride(dst->ne, dst->nb, &sd);
}

bool htp_matmul_add_can_collapse(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst
) {
    if (!htp_matmul_can_collapse(src0, src1, dst)) {
        return false;
    }
    if (!src2) {
        return true;
    }
    const int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    if (src2_nrows == 1) {
        return true;
    }
    size_t s2;
    return src2->nb[0] == ggml_type_size(src2->type) &&
           ggml_are_same_shape(src2, dst) &&
           htp_rows_stride(src2->ne, src2->nb, &s2);
}

ggml_tensor htp_tensor_collapse_rows(const struct ggml_tensor * t) {
    size_t stride = 0;
    htp_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    return c;
}

// Collapse contiguous dims of elementwise ops so that rows fill at least one HVX vector
static void htp_collapse_dims(struct ggml_tensor ** tensors, size_t n_tensors, size_t elem_size, uint32_t n_hvx) {
    if (n_tensors == 0 || !tensors[0]) {
        return;
    }
    for (size_t t = 0; t < n_tensors; t++) {
        if (!tensors[t] || tensors[t]->nb[0] != elem_size) {
            return;
        }
        if (t > 0 && !ggml_are_same_shape(tensors[0], tensors[t])) {
            return;
        }
    }

    struct ggml_tensor * t0 = tensors[0];

    while (t0->ne[1] > 1 || t0->ne[2] > 1 || t0->ne[3] > 1) {
        const size_t row_size = (size_t) t0->ne[0] * elem_size;
        bool rows_contig = true;
        for (size_t t = 0; t < n_tensors; t++) {
            if (tensors[t]->nb[1] != (size_t) tensors[t]->ne[0] * elem_size) {
                rows_contig = false;
                break;
            }
        }
        bool all_ne1_one = true;
        for (size_t t = 0; t < n_tensors; t++) {
            if (tensors[t]->ne[1] != 1) {
                all_ne1_one = false;
                break;
            }
        }
        const bool can_merge = all_ne1_one ||
                               (rows_contig && (t0->ne[0] == 1 || (row_size < 128 && row_size * t0->ne[1] <= 2048)));
        if (!can_merge) {
            break;
        }
        for (size_t t = 0; t < n_tensors; t++) {
            tensors[t]->ne[0] *= tensors[t]->ne[1];
            tensors[t]->ne[1]  = tensors[t]->ne[2];
            tensors[t]->ne[2]  = tensors[t]->ne[3];
            tensors[t]->ne[3]  = 1;
            tensors[t]->nb[1]  = tensors[t]->nb[2];
            tensors[t]->nb[2]  = tensors[t]->nb[3];
            tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
        }
    }

    // ne1 too big to merge in full: merge it in chunks of k rows (k divides ne1)
    const size_t row_size = (size_t) t0->ne[0] * elem_size;
    bool rows_contig = true;
    for (size_t t = 0; t < n_tensors; t++) {
        if (tensors[t]->nb[1] != row_size) {
            rows_contig = false;
            break;
        }
    }
    if (t0->ne[1] > 1 && row_size < 128 && rows_contig) {
        int64_t k = 2048 / row_size;
        while (k > 1 && t0->ne[1] % k != 0) {
            k--;
        }
        if (k > 1) {
            for (size_t t = 0; t < n_tensors; t++) {
                tensors[t]->ne[0] *= k;
                tensors[t]->ne[1] /= k;
                tensors[t]->nb[1] *= k;
                tensors[t]->nb[2]  = tensors[t]->nb[1] * tensors[t]->ne[1];
                tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
            }
        }
    }

    // nrows too small to saturate HVX units: split ne0 into k rows (k divides ne0)
    const size_t   row_size_curr = (size_t) t0->ne[0] * elem_size;
    const uint32_t nrows         = (uint32_t) (t0->ne[1] * t0->ne[2] * t0->ne[3]);
    rows_contig = true;
    for (size_t t = 0; t < n_tensors; t++) {
        if (tensors[t]->ne[1] != 1 && tensors[t]->nb[1] != row_size_curr) {
            rows_contig = false;
            break;
        }
    }
    if (nrows < n_hvx && rows_contig && row_size_curr >= 4096) {
        const size_t vec_elems = 128 / elem_size;
        int64_t k = std::max<int64_t>(row_size_curr / 16384, (n_hvx * 4 + nrows - 1) / nrows);
        k = std::min<int64_t>(k, row_size_curr / 2048);
        while (k > 1 && t0->ne[0] % (k * vec_elems) != 0) {
            k--;
        }
        if (k > 1) {
            for (size_t t = 0; t < n_tensors; t++) {
                tensors[t]->ne[0] /= k;
                tensors[t]->ne[1] *= k;
                tensors[t]->nb[1]  = tensors[t]->ne[0] * elem_size;
                tensors[t]->nb[2]  = tensors[t]->nb[1] * tensors[t]->ne[1];
                tensors[t]->nb[3]  = tensors[t]->nb[2] * tensors[t]->ne[2];
            }
        }
    }
}

bool htp_dsv4_hc_pre_can_collapse(
    const struct ggml_tensor * x,
    const struct ggml_tensor * weights,
    const struct ggml_tensor * dst
) {
    if (x->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || weights->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (x->ne[1] != 4 || dst->ne[0] != x->ne[0]) {
        return false;
    }

    const bool gated = ggml_get_op_params_i32(dst, 1) != 0;
    const int64_t x_tokens = x->ne[2] * x->ne[3];
    const int64_t d_tokens = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t w_tokens = gated ? (weights->ne[2] * weights->ne[3]) : (weights->ne[1] * weights->ne[2] * weights->ne[3]);

    if (x_tokens != d_tokens || x_tokens != w_tokens) {
        return false;
    }

    if (x->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        (gated ? (weights->ne[3] == 1) : (weights->ne[2] == 1 && weights->ne[3] == 1))) {
        return false;
    }

    if (x->ne[3] > 1 && x->nb[3] != x->ne[2] * x->nb[2]) return false;

    if (dst->ne[2] > 1 && dst->nb[2] != dst->ne[1] * dst->nb[1]) return false;
    if (dst->ne[3] > 1 && dst->nb[3] != dst->ne[2] * dst->nb[2]) return false;

    if (gated) {
        if (weights->ne[3] > 1 && weights->nb[3] != weights->ne[2] * weights->nb[2]) return false;
    } else {
        if (weights->ne[2] > 1 && weights->nb[2] != weights->ne[1] * weights->nb[1]) return false;
        if (weights->ne[3] > 1 && weights->nb[3] != weights->ne[2] * weights->nb[2]) return false;
    }

    return true;
}

void htp_dsv4_hc_pre_collapse_tensors(
    const struct ggml_tensor * x,
    const struct ggml_tensor * weights,
    const struct ggml_tensor * dst,
    struct ggml_tensor * x_c,
    struct ggml_tensor * w_c,
    struct ggml_tensor * d_c
) {
    const int64_t n_tokens = x->ne[2] * x->ne[3];
    const bool gated = ggml_get_op_params_i32(dst, 1) != 0;

    *x_c = *x;
    x_c->ne[2] = n_tokens;
    x_c->ne[3] = 1;
    x_c->nb[3] = x_c->ne[2] * x_c->nb[2];

    *d_c = *dst;
    d_c->ne[1] = n_tokens;
    d_c->ne[2] = 1;
    d_c->ne[3] = 1;
    d_c->nb[2] = d_c->ne[1] * d_c->nb[1];
    d_c->nb[3] = d_c->nb[2];

    *w_c = *weights;
    if (gated) {
        w_c->ne[2] = n_tokens;
        w_c->ne[3] = 1;
        w_c->nb[3] = w_c->ne[2] * w_c->nb[2];
    } else {
        w_c->ne[1] = n_tokens;
        w_c->ne[2] = 1;
        w_c->ne[3] = 1;
        w_c->nb[2] = w_c->ne[1] * w_c->nb[1];
        w_c->nb[3] = w_c->nb[2];
    }
}

bool htp_dsv4_hc_post_can_collapse(
    const struct ggml_tensor * x,
    const struct ggml_tensor * residual,
    const struct ggml_tensor * post,
    const struct ggml_tensor * comb,
    const struct ggml_tensor * dst
) {
    if (x->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (comb && comb->type != GGML_TYPE_F32) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || residual->nb[0] != sizeof(float) || post->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (comb && comb->nb[0] != sizeof(float)) {
        return false;
    }
    if (residual->ne[0] != x->ne[0] || residual->ne[1] != 4 || dst->ne[0] != x->ne[0] || dst->ne[1] != 4 || post->ne[0] != 4) {
        return false;
    }
    if (comb && (comb->ne[0] != 4 || comb->ne[1] != 4)) {
        return false;
    }

    const int64_t x_tokens = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t r_tokens = residual->ne[2] * residual->ne[3];
    const int64_t p_tokens = post->ne[1] * post->ne[2] * post->ne[3];
    const int64_t d_tokens = dst->ne[2] * dst->ne[3];
    if (x_tokens != r_tokens || x_tokens != p_tokens || x_tokens != d_tokens) {
        return false;
    }
    if (comb && (comb->ne[2] * comb->ne[3] != x_tokens)) {
        return false;
    }

    if (x->ne[2] == 1 && x->ne[3] == 1 &&
        residual->ne[3] == 1 &&
        post->ne[2] == 1 && post->ne[3] == 1 &&
        dst->ne[3] == 1 &&
        (!comb || comb->ne[3] == 1)) {
        return false;
    }

    if (x->ne[2] > 1 && x->nb[2] != x->ne[1] * x->nb[1]) return false;
    if (x->ne[3] > 1 && x->nb[3] != x->ne[2] * x->nb[2]) return false;

    if (residual->ne[3] > 1 && residual->nb[3] != residual->ne[2] * residual->nb[2]) return false;
    if (dst->ne[3] > 1 && dst->nb[3] != dst->ne[2] * dst->nb[2]) return false;

    if (post->ne[2] > 1 && post->nb[2] != post->ne[1] * post->nb[1]) return false;
    if (post->ne[3] > 1 && post->nb[3] != post->ne[2] * post->nb[2]) return false;

    if (comb && comb->ne[3] > 1 && comb->nb[3] != comb->ne[2] * comb->nb[2]) return false;

    return true;
}

void htp_dsv4_hc_post_collapse_tensors(
    const struct ggml_tensor * x,
    const struct ggml_tensor * residual,
    const struct ggml_tensor * post,
    const struct ggml_tensor * comb,
    const struct ggml_tensor * dst,
    struct ggml_tensor * x_c,
    struct ggml_tensor * r_c,
    struct ggml_tensor * p_c,
    struct ggml_tensor * c_c,
    struct ggml_tensor * d_c
) {
    const int64_t n_tokens = x->ne[1] * x->ne[2] * x->ne[3];

    *x_c = *x;
    x_c->ne[1] = n_tokens;
    x_c->ne[2] = 1;
    x_c->ne[3] = 1;
    x_c->nb[2] = x_c->ne[1] * x_c->nb[1];
    x_c->nb[3] = x_c->nb[2];

    *r_c = *residual;
    r_c->ne[2] = n_tokens;
    r_c->ne[3] = 1;
    r_c->nb[3] = r_c->ne[2] * r_c->nb[2];

    *d_c = *dst;
    d_c->ne[2] = n_tokens;
    d_c->ne[3] = 1;
    d_c->nb[3] = d_c->ne[2] * d_c->nb[2];

    *p_c = *post;
    p_c->ne[1] = n_tokens;
    p_c->ne[2] = 1;
    p_c->ne[3] = 1;
    p_c->nb[2] = p_c->ne[1] * p_c->nb[1];
    p_c->nb[3] = p_c->nb[2];

    if (comb && c_c) {
        *c_c = *comb;
        c_c->ne[2] = n_tokens;
        c_c->ne[3] = 1;
        c_c->nb[3] = c_c->ne[2] * c_c->nb[2];
    }
}

bool htp_lightning_indexer_can_collapse(
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * w,
    const struct ggml_tensor * m,
    const struct ggml_tensor * dst
) {
    // 2D K normalization: ne[1] is n_kv, ne[2] is 1
    const bool can_collapse_k = (k->ne[1] > 1 && k->ne[2] == 1);

    // Stream-to-token collapse across q, w, dst (and m)
    bool can_collapse_stream = false;
    if (q->ne[3] > 1 && k->ne[3] == 1) {
        const bool q_ok = (q->nb[3] == (size_t) q->ne[2] * q->nb[2]);
        const bool dst_ok = (dst->ne[3] == q->ne[3] && dst->ne[2] == 1 && dst->ne[1] == q->ne[2] &&
                             dst->nb[3] == (size_t) dst->ne[1] * dst->nb[1]);
        const bool w_ok = (w->ne[3] == q->ne[3] && w->ne[2] == 1 && w->ne[1] == q->ne[2] &&
                           w->nb[3] == (size_t) w->ne[1] * w->nb[1]) ||
                          (w->ne[3] == 1 && w->ne[1] == q->ne[2] * q->ne[3]);
        const bool m_ok = (m->ne[3] == q->ne[3] && m->ne[2] == 1 && m->ne[1] == q->ne[2] &&
                           m->nb[3] == (size_t) m->ne[1] * m->nb[1]) ||
                          (m->ne[3] == 1 && m->ne[1] == q->ne[2] * q->ne[3]);
        can_collapse_stream = q_ok && dst_ok && w_ok && m_ok;
    }

    return can_collapse_k || can_collapse_stream;
}

void htp_lightning_indexer_collapse_tensors(
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * w,
    const struct ggml_tensor * m,
    const struct ggml_tensor * dst,
    struct ggml_tensor * q_c,
    struct ggml_tensor * k_c,
    struct ggml_tensor * w_c,
    struct ggml_tensor * m_c,
    struct ggml_tensor * dst_c
) {
    *q_c   = *q;
    *k_c   = *k;
    *w_c   = *w;
    *m_c   = *m;
    *dst_c = *dst;

    if (k->ne[1] > 1 && k->ne[2] == 1) {
        k_c->ne[2] = k->ne[1];
        k_c->ne[1] = 1;
        k_c->nb[1] = k->nb[1];
        k_c->nb[2] = k->nb[1];
        k_c->nb[3] = k->nb[2];
    }

    if (q->ne[3] > 1 && k->ne[3] == 1) {
        const int64_t n_tokens_collapsed = (int64_t) q->ne[2] * q->ne[3];

        q_c->ne[2] = n_tokens_collapsed;
        q_c->ne[3] = 1;
        q_c->nb[3] = (size_t) q_c->ne[2] * q_c->nb[2];

        if (w->ne[3] == q->ne[3]) {
            w_c->ne[1] = n_tokens_collapsed;
            w_c->ne[3] = 1;
            w_c->nb[2] = (size_t) w_c->ne[1] * w_c->nb[1];
            w_c->nb[3] = w_c->nb[2];
        }

        dst_c->ne[1] = n_tokens_collapsed;
        dst_c->ne[3] = 1;
        dst_c->nb[2] = (size_t) dst_c->ne[1] * dst_c->nb[1];
        dst_c->nb[3] = dst_c->nb[2];

        if (m->ne[3] == q->ne[3] && m->ne[1] == q->ne[2]) {
            m_c->ne[1] = n_tokens_collapsed;
            m_c->ne[3] = 1;
            m_c->nb[2] = (size_t) m_c->ne[1] * m_c->nb[1];
            m_c->nb[3] = m_c->nb[2];
        }
    }
}

void htp_lower_opnode(const struct ggml_hexagon_session * sess, htp_opnode & node, struct ggml_tensor * scratch) {
    const uint32_t n_hvx = htp_session_n_hvx(sess);
    const size_t   n_in  = node.inputs.size();
    const size_t   n_out = node.outputs.size();

    ggml_tensor in_mat[HTP_OP_MAX_INPUTS]   = {};
    ggml_tensor out_mat[HTP_OP_MAX_OUTPUTS] = {};
    bool        in_dirty[HTP_OP_MAX_INPUTS]   = {};
    bool        out_dirty[HTP_OP_MAX_OUTPUTS] = {};

    for (size_t i = 0; i < n_in; ++i) {
        if (node.inputs[i]) {
            in_dirty[i] = ggml_hexagon_tensor_is_lowered_f16(node.inputs[i]);
            in_mat[i]   = htp_tensor_lower_f16(node.inputs[i]);
        }
    }
    for (size_t i = 0; i < n_out; ++i) {
        out_dirty[i] = ggml_hexagon_tensor_is_lowered_f16(node.outputs[i]);
        out_mat[i]   = htp_tensor_lower_f16(node.outputs[i]);
    }

    if (htp_op_is_unary(node.opcode) && htp_unary_is_elementwise(node.opcode)) {
        if (n_in >= 1 && n_out >= 1 &&
            ggml_are_same_shape(&in_mat[0], &out_mat[0])) {
            const int64_t old_ne0   = in_mat[0].ne[0];
            const int64_t old_ne1   = in_mat[0].ne[1];
            const size_t  elem_size = ggml_type_size(in_mat[0].type);
            struct ggml_tensor * tensors[2] = { &in_mat[0], &out_mat[0] };
            htp_collapse_dims(tensors, 2, elem_size, n_hvx);
            if (in_mat[0].ne[0] != old_ne0 || in_mat[0].ne[1] != old_ne1) {
                in_dirty[0]  = true;
                out_dirty[0] = true;
            }
        }
    } else if (htp_op_is_glu(node.opcode)) {
        if (n_in >= 2 && n_out >= 1 &&
            ggml_are_same_shape(&in_mat[0], &out_mat[0]) &&
            ggml_are_same_shape(&in_mat[1], &out_mat[0])) {
            const int64_t old_ne0   = in_mat[0].ne[0];
            const int64_t old_ne1   = in_mat[0].ne[1];
            const size_t  elem_size = ggml_type_size(in_mat[0].type);
            struct ggml_tensor * tensors[3] = { &in_mat[0], &in_mat[1], &out_mat[0] };
            htp_collapse_dims(tensors, 3, elem_size, n_hvx);
            if (in_mat[0].ne[0] != old_ne0 || in_mat[0].ne[1] != old_ne1) {
                in_dirty[0]  = true;
                in_dirty[1]  = true;
                out_dirty[0] = true;
            }
        }
    } else if (node.opcode == HTP_OP_MUL_MAT) {
        if (n_in >= 2 && n_out >= 1 &&
            htp_matmul_can_collapse(&in_mat[0], &in_mat[1], &out_mat[0])) {
            in_mat[1]    = htp_tensor_collapse_rows(&in_mat[1]);
            out_mat[0]   = htp_tensor_collapse_rows(&out_mat[0]);
            in_dirty[1]  = true;
            out_dirty[0] = true;
        }
    } else if (node.opcode == HTP_OP_MUL_MAT_ADD) {
        if (n_in >= 3 && n_out >= 1 &&
            htp_matmul_add_can_collapse(&in_mat[0], &in_mat[1], &in_mat[2], &out_mat[0])) {
            in_mat[1]    = htp_tensor_collapse_rows(&in_mat[1]);
            out_mat[0]   = htp_tensor_collapse_rows(&out_mat[0]);
            in_dirty[1]  = true;
            out_dirty[0] = true;
            if (in_mat[2].ne[1] * in_mat[2].ne[2] * in_mat[2].ne[3] > 1) {
                in_mat[2]   = htp_tensor_collapse_rows(&in_mat[2]);
                in_dirty[2] = true;
            }
        }
    } else if (node.opcode == HTP_OP_MUL_MAT_NX) {
        const size_t n_w = n_out;
        if (n_w > 0 && n_in == n_w + 1) {
            bool can_collapse = true;
            for (size_t i = 0; i < n_w; ++i) {
                if (!htp_matmul_can_collapse(&in_mat[i], &in_mat[n_w], &out_mat[i])) {
                    can_collapse = false;
                    break;
                }
            }
            if (can_collapse) {
                in_mat[n_w]   = htp_tensor_collapse_rows(&in_mat[n_w]);
                in_dirty[n_w] = true;
                for (size_t i = 0; i < n_w; ++i) {
                    out_mat[i]   = htp_tensor_collapse_rows(&out_mat[i]);
                    out_dirty[i] = true;
                }
            }
        }
    } else if (node.opcode == HTP_OP_UPSCALE) {
        if (n_in >= 1 && n_out >= 1) {
            if ((in_mat[0].ne[3] > 1 || out_mat[0].ne[3] > 1) &&
                in_mat[0].ne[2] == out_mat[0].ne[2] && in_mat[0].ne[3] == out_mat[0].ne[3] &&
                in_mat[0].nb[3] == (size_t) in_mat[0].ne[2] * in_mat[0].nb[2] &&
                out_mat[0].nb[3] == (size_t) out_mat[0].ne[2] * out_mat[0].nb[2]) {
                in_mat[0].ne[2]  *= in_mat[0].ne[3];
                in_mat[0].ne[3]   = 1;
                in_mat[0].nb[3]   = (size_t) in_mat[0].ne[2] * in_mat[0].nb[2];
                out_mat[0].ne[2] *= out_mat[0].ne[3];
                out_mat[0].ne[3]  = 1;
                out_mat[0].nb[3]  = (size_t) out_mat[0].ne[2] * out_mat[0].nb[2];
                in_dirty[0]       = true;
                out_dirty[0]      = true;
            }
        }
    } else if (node.opcode == HTP_OP_DSV4_HC_PRE) {
        if (n_in >= 2 && n_out >= 1 &&
            htp_dsv4_hc_pre_can_collapse(&in_mat[0], &in_mat[1], &out_mat[0])) {
            struct ggml_tensor x_c, w_c, d_c;
            htp_dsv4_hc_pre_collapse_tensors(&in_mat[0], &in_mat[1], &out_mat[0], &x_c, &w_c, &d_c);
            in_mat[0]    = x_c;
            in_mat[1]    = w_c;
            out_mat[0]   = d_c;
            in_dirty[0]  = true;
            in_dirty[1]  = true;
            out_dirty[0] = true;
        }
    } else if (node.opcode == HTP_OP_DSV4_HC_POST) {
        if (n_in >= 3 && n_out >= 1) {
            const struct ggml_tensor * comb = (n_in > 3 && node.inputs[3]) ? &in_mat[3] : nullptr;
            if (htp_dsv4_hc_post_can_collapse(&in_mat[0], &in_mat[1], &in_mat[2], comb, &out_mat[0])) {
                struct ggml_tensor x_c, r_c, p_c, c_c, d_c;
                htp_dsv4_hc_post_collapse_tensors(&in_mat[0], &in_mat[1], &in_mat[2], comb, &out_mat[0],
                                                  &x_c, &r_c, &p_c, comb ? &c_c : nullptr, &d_c);
                in_mat[0]    = x_c;
                in_mat[1]    = r_c;
                in_mat[2]    = p_c;
                out_mat[0]   = d_c;
                in_dirty[0]  = true;
                in_dirty[1]  = true;
                in_dirty[2]  = true;
                out_dirty[0] = true;
                if (comb) {
                    in_mat[3]   = c_c;
                    in_dirty[3] = true;
                }
            }
        }
    } else if (node.opcode == HTP_OP_LIGHTNING_INDEXER) {
        if (n_in >= 4 && n_out >= 1 &&
            htp_lightning_indexer_can_collapse(&in_mat[0], &in_mat[1], &in_mat[2], &in_mat[3], &out_mat[0])) {
            struct ggml_tensor q_c, k_c, w_c, m_c, d_c;
            htp_lightning_indexer_collapse_tensors(&in_mat[0], &in_mat[1], &in_mat[2], &in_mat[3], &out_mat[0],
                                                   &q_c, &k_c, &w_c, &m_c, &d_c);
            in_mat[0]    = q_c;
            in_mat[1]    = k_c;
            in_mat[2]    = w_c;
            in_mat[3]    = m_c;
            out_mat[0]   = d_c;
            in_dirty[0]  = true;
            in_dirty[1]  = true;
            in_dirty[2]  = true;
            in_dirty[3]  = true;
            out_dirty[0] = true;
        }
    }

    size_t n_scratch = 0;
    auto commit_tensor = [&](const ggml_tensor & m) -> const ggml_tensor * {
        if (scratch) {
            scratch[n_scratch] = m;
            return &scratch[n_scratch++];
        }
        return node.add_dummy(m);
    };

    const ggml_tensor * orig_in[HTP_OP_MAX_INPUTS] = {};
    for (size_t i = 0; i < n_in; ++i) {
        orig_in[i] = node.inputs[i];
        if (in_dirty[i]) {
            node.inputs[i] = commit_tensor(in_mat[i]);
        }
    }
    for (size_t j = 0; j < n_out; ++j) {
        if (!out_dirty[j]) {
            continue;
        }
        const ggml_tensor * reused = nullptr;
        for (size_t i = 0; i < n_in; ++i) {
            if (orig_in[i] == node.outputs[j] && in_dirty[i] &&
                in_mat[i].type == out_mat[j].type &&
                ggml_are_same_shape(&in_mat[i], &out_mat[j]) &&
                ggml_are_same_stride(&in_mat[i], &out_mat[j])) {
                reused = node.inputs[i];
                break;
            }
        }
        node.outputs[j] = reused ? reused : commit_tensor(out_mat[j]);
    }
}

static bool htp_lower_is_view(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE;
}

static bool htp_lower_is_root_candidate(const ggml_tensor * root, const std::unordered_set<const ggml_tensor *> & produced) {
    return produced.count(root) && root->type == GGML_TYPE_F32 && root->buffer &&
           ggml_backend_buffer_get_usage(root->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           (root->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM)) == 0;
}

static bool htp_lower_is_partial_write(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_SET || tensor->op == GGML_OP_SET_ROWS) {
        return true;
    }
    return tensor->op == GGML_OP_CPY && !ggml_is_contiguous(tensor);
}

static bool htp_lower_safe_class(
    const ggml_cgraph * graph,
    const ggml_tensor * root,
    const std::vector<const ggml_tensor *> & tensors,
    const std::unordered_map<const ggml_tensor *, int> & node_idx,
    const std::unordered_map<const ggml_tensor *, int> & internal_uses
) {
    int total_internal_uses = 0;
    for (const ggml_tensor * tensor : tensors) {
        if (tensor->type != GGML_TYPE_F32) {
            return false;
        }
        if (tensor != root && !htp_lower_is_view(tensor)) {
            return false;
        }
        const ptrdiff_t offset = (const uint8_t *) tensor->data - (const uint8_t *) root->data;
        if (offset % 4 != 0 || (tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->nb[d] % 4 != 0) {
                return false;
            }
        }
        auto it = node_idx.find(tensor);
        if (it == node_idx.end() || htp_lower_is_partial_write(tensor)) {
            return false;
        }
        const int total_uses = ggml_node_get_use_count(graph, it->second);
        auto uit = internal_uses.find(tensor);
        const int uses = (uit != internal_uses.end()) ? uit->second : 0;
        if (uses != total_uses) {
            return false;
        }
        total_internal_uses += uses;
    }
    return total_internal_uses >= 1;
}

static bool htp_lower_inplace_types_ok(const htp_opnode & node) {
    for (const ggml_tensor * inp : node.inputs) {
        if (!inp) {
            continue;
        }
        const ggml_type inp_type = htp_effective_type(inp);
        if (inp_type != GGML_TYPE_F32 && inp_type != GGML_TYPE_F16) {
            continue;
        }
        for (const ggml_tensor * out : node.outputs) {
            if (!out) {
                continue;
            }
            const ggml_type out_type = htp_effective_type(out);
            if (out_type != GGML_TYPE_F32 && out_type != GGML_TYPE_F16) {
                continue;
            }
            if (inp_type != out_type && htp_tensors_overlap(inp, out)) {
                return false;
            }
        }
    }
    return true;
}

static bool htp_lower_is_mm_opcode(uint32_t opcode) {
    return opcode == HTP_OP_MUL_MAT || opcode == HTP_OP_MUL_MAT_ADD ||
           opcode == HTP_OP_MUL_MAT_ID || opcode == HTP_OP_MUL_MAT_NX ||
           opcode == HTP_OP_MUL_MAT_ID_NX;
}

static const ggml_tensor * htp_lower_orig_mm_for_output(const htp_opnode & node, const ggml_tensor * tensor) {
    if (!htp_lower_is_mm_opcode(node.opcode)) {
        return nullptr;
    }
    if (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX) {
        for (size_t p = 0; p < node.outputs.size(); ++p) {
            if (node.outputs[p] == tensor) {
                if (p == 0) {
                    return node.node;
                }
                return (p - 1 < node.fused.size()) ? node.fused[p - 1] : nullptr;
            }
        }
        return nullptr;
    }
    return node.node;
}

static bool htp_lower_mm_blocks_output_f16(const ggml_tensor * orig_mm) {
    if (!orig_mm) {
        return false;
    }
    if (ggml_prec_get_acc(orig_mm) == GGML_PREC_F32) {
        return true;
    }
    const ggml_prec src1_prec = ggml_prec_get_src(orig_mm, 1);
    return src1_prec == GGML_PREC_BF16 || src1_prec == GGML_PREC_F32;
}

static bool htp_lower_respect_prec_hints(
    const htp_opnode & owner,
    const ggml_tensor * r,
    const std::vector<const ggml_tensor *> & cls,
    const std::vector<htp_opnode> & nodes,
    const std::unordered_map<const ggml_tensor *, size_t> & owner_idx,
    const std::unordered_map<const ggml_tensor *, std::vector<size_t>> & root_users
) {
    if (htp_lower_is_mm_opcode(owner.opcode)) {
        const ggml_tensor * orig_mm = htp_lower_orig_mm_for_output(owner, r);
        if (htp_lower_mm_blocks_output_f16(orig_mm)) {
            return false;
        }
    }
    if (owner.opcode == HTP_OP_SOFTMAX && !owner.inputs.empty() && owner.inputs[0]) {
        const ggml_tensor * in_root = htp_tensor_root(owner.inputs[0]);
        auto oit = owner_idx.find(in_root);
        if (oit != owner_idx.end()) {
            const htp_opnode & node = nodes[oit->second];
            if (htp_lower_is_mm_opcode(node.opcode) &&
                htp_lower_mm_blocks_output_f16(htp_lower_orig_mm_for_output(node, in_root))) {
                return false;
            }
        }
    }
    auto uit = root_users.find(r);
    if (uit == root_users.end()) {
        return true;
    }
    for (size_t idx : uit->second) {
        const htp_opnode & node = nodes[idx];
        if (!htp_lower_is_mm_opcode(node.opcode)) {
            continue;
        }
        const size_t act_idx = (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX)
                                   ? node.outputs.size()
                                   : 1;
        if (act_idx >= node.inputs.size()) {
            continue;
        }
        const ggml_tensor * act = node.inputs[act_idx];
        bool uses_cls = false;
        for (const ggml_tensor * tensor : cls) {
            if (act == tensor) {
                uses_cls = true;
                break;
            }
        }
        if (!uses_cls && (!act || htp_tensor_root(act) != r)) {
            continue;
        }
        auto blocks_src1 = [](const ggml_tensor * orig_mm) -> bool {
            if (!orig_mm) {
                return false;
            }
            const ggml_prec src1_prec = ggml_prec_get_src(orig_mm, 1);
            return src1_prec == GGML_PREC_BF16 || src1_prec == GGML_PREC_F32;
        };
        if (blocks_src1(node.node)) {
            return false;
        }
        if (node.opcode == HTP_OP_MUL_MAT_NX || node.opcode == HTP_OP_MUL_MAT_ID_NX) {
            for (const ggml_tensor * f : node.fused) {
                if (f && (f->op == GGML_OP_MUL_MAT || f->op == GGML_OP_MUL_MAT_ID) && blocks_src1(f)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool htp_lower_is_mm_root(
    const std::vector<htp_opnode> & nodes,
    const std::unordered_map<const ggml_tensor *, size_t> & owner_idx,
    const ggml_tensor * root
) {
    auto it = owner_idx.find(root);
    return it != owner_idx.end() && htp_lower_is_mm_opcode(nodes[it->second].opcode);
}

static bool htp_lower_is_fa_root(
    const std::vector<htp_opnode> & nodes,
    const std::unordered_map<const ggml_tensor *, size_t> & owner_idx,
    const ggml_tensor * root
) {
    auto it = owner_idx.find(root);
    return it != owner_idx.end() && nodes[it->second].opcode == HTP_OP_FLASH_ATTN_EXT;
}

static bool htp_lower_is_rms_root(
    const std::vector<htp_opnode> & nodes,
    const std::unordered_map<const ggml_tensor *, size_t> & owner_idx,
    const ggml_tensor * root
) {
    auto it = owner_idx.find(root);
    if (it == owner_idx.end()) {
        return false;
    }
    const uint32_t opcode = nodes[it->second].opcode;
    return opcode == HTP_OP_RMS_NORM || opcode == HTP_OP_RMS_NORM_MUL ||
           opcode == HTP_OP_RMS_NORM_MUL_ADD;
}

static bool htp_lower_safe_gdn_output(
    const htp_opnode & owner,
    const ggml_tensor * root,
    const std::vector<htp_opnode> & nodes,
    const std::unordered_map<const ggml_tensor *, std::vector<size_t>> & root_users
) {
    if (owner.outputs.size() != 2 || root != owner.outputs[0] || owner.inputs.size() < 3) {
        return false;
    }
    auto uit = root_users.find(root);
    if (uit == root_users.end()) {
        return false;
    }
    const ggml_tensor * v = owner.inputs[2];
    const size_t tail_off = (size_t) v->ne[0] * v->ne[1] * v->ne[2] * v->ne[3] * sizeof(float);
    bool has_consumer = false;
    for (size_t idx : uit->second) {
        const htp_opnode & node = nodes[idx];
        for (const ggml_tensor * input : node.inputs) {
            if (!input || htp_tensor_root(input) != root) {
                continue;
            }
            const ptrdiff_t offset = (const uint8_t *) input->data - (const uint8_t *) root->data;
            if (offset < 0 || (size_t) offset + ggml_nbytes(input) > tail_off) {
                return false;
            }
            has_consumer = true;
        }
    }
    return has_consumer;
}

void htp_lower_reset_graph(const struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_hexagon_tensor_clear_lowered(node->src[j]);
        }
        ggml_hexagon_tensor_clear_lowered(node);
    }
}

void htp_pack_tensor_lowered_f16(struct ggml_tensor * t, const void * src, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(offset % sizeof(float) == 0 && size % sizeof(float) == 0);
    const struct ggml_tensor * root = htp_tensor_root(t);
    uint8_t * base = (uint8_t *) root->data + ((const uint8_t *) t->data - (const uint8_t *) root->data) / 2;
    ggml_fp16_t * dst = (ggml_fp16_t *) (base + offset / 2);
    const int64_t n = (int64_t) (size / sizeof(float));
    if ((uintptr_t) src % alignof(float) == 0) {
        ggml_fp32_to_fp16_row((const float *) src, dst, n);
    } else {
        std::vector<float> tmp((size_t) n);
        memcpy(tmp.data(), src, size);
        ggml_fp32_to_fp16_row(tmp.data(), dst, n);
    }
}

void htp_unpack_tensor_lowered_f16(void * dst, const struct ggml_tensor * t, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(offset % sizeof(float) == 0 && size % sizeof(float) == 0);
    const struct ggml_tensor * root = htp_tensor_root(t);
    const uint8_t * base = (const uint8_t *) root->data + ((const uint8_t *) t->data - (const uint8_t *) root->data) / 2;
    const ggml_fp16_t * src = (const ggml_fp16_t *) (base + offset / 2);
    const int64_t n = (int64_t) (size / sizeof(float));
    if ((uintptr_t) dst % alignof(float) == 0) {
        ggml_fp16_to_fp32_row(src, (float *) dst, n);
    } else {
        std::vector<float> tmp((size_t) n);
        ggml_fp16_to_fp32_row(src, tmp.data(), n);
        memcpy(dst, tmp.data(), size);
    }
}

void htp_lower_graph(const struct ggml_hexagon_session * sess, const struct ggml_cgraph * graph, std::vector<htp_opnode> & nodes) {
    if (opt_oplower <= 0) {
        for (htp_opnode & node : nodes) {
            htp_lower_opnode(sess, node);
        }
        return;
    }

    std::vector<const ggml_tensor *> roots;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> classes;
    std::unordered_set<const ggml_tensor *> seen;
    std::unordered_map<const ggml_tensor *, int> node_idx;
    std::unordered_map<const ggml_tensor *, int> internal_uses;
    std::unordered_set<const ggml_tensor *> produced;

    auto add_to_class = [&](const ggml_tensor * tensor) {
        if (!tensor || !seen.insert(tensor).second) {
            return;
        }
        ggml_hexagon_tensor_clear_flags(const_cast<ggml_tensor *>(tensor), HTP_TENSOR_LOWERED_F16);
        const ggml_tensor * root = htp_tensor_root(tensor);
        auto & cls = classes[root];
        if (cls.empty()) {
            roots.push_back(root);
        }
        cls.push_back(tensor);
    };

    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * tensor = graph->nodes[i];
        node_idx[tensor] = i;
        produced.insert(tensor);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (tensor->src[j]) {
                ++internal_uses[tensor->src[j]];
            }
            add_to_class(tensor->src[j]);
        }
        add_to_class(tensor);
    }

    std::unordered_map<const ggml_tensor *, size_t>              owner_idx;
    std::unordered_map<const ggml_tensor *, std::vector<size_t>> root_users;

    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        const htp_opnode & node = nodes[idx];
        auto add_user = [&](const ggml_tensor * t) {
            if (!t) {
                return;
            }
            auto & vec = root_users[htp_tensor_root(t)];
            if (vec.empty() || vec.back() != idx) {
                vec.push_back(idx);
            }
        };
        for (const ggml_tensor * inp : node.inputs) {
            add_user(inp);
        }
        for (const ggml_tensor * out : node.outputs) {
            if (out) {
                owner_idx.emplace(out, idx);
                add_user(out);
            }
        }
    }

    size_t n_candidates    = 0;
    size_t n_lowered_roots = 0;
    size_t n_lowered_mm    = 0;
    size_t bytes_saved     = 0;

    auto find_owner = [&](const ggml_tensor * r) -> htp_opnode * {
        auto it = owner_idx.find(r);
        return (it != owner_idx.end()) ? &nodes[it->second] : nullptr;
    };

    std::unordered_map<const ggml_tensor *, bool> root_safe_class;
    for (const ggml_tensor * root : roots) {
        auto it = classes.find(root);
        const bool safe = it != classes.end() &&
                          htp_lower_is_root_candidate(root, produced) &&
                          htp_lower_safe_class(graph, root, it->second, node_idx, internal_uses);
        root_safe_class[root] = safe;
        if (safe && find_owner(root)) {
            ++n_candidates;
        }
    }

    auto is_f16_tensor = [&](const ggml_tensor * t) -> bool {
        return t && (htp_effective_type(t) == GGML_TYPE_F16 || ggml_hexagon_tensor_is_lowered_f16(htp_tensor_root(t)));
    };

    auto is_weight_or_bcast = [](const ggml_tensor * inp, const ggml_tensor * dst) -> bool {
        if (!inp) {
            return true;
        }
        if ((inp->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return true;
        }
        if (inp->buffer && ggml_backend_buffer_get_usage(inp->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return true;
        }
        return ggml_nelements(inp) < ggml_nelements(dst);
    };

    // Under OPLOWER=1, keep cross-layer residual spine merges in F32 when the residual input is F32.
    auto is_residual_spine_stop = [&](const htp_opnode & owner, const ggml_tensor * r) -> bool {
        if (opt_oplower != 1) {
            return false;
        }
        if (owner.opcode == HTP_OP_ADD) {
            for (const ggml_tensor * inp : owner.inputs) {
                if (inp && !is_f16_tensor(inp) && !is_weight_or_bcast(inp, r)) {
                    return true;
                }
            }
        } else if (owner.opcode == HTP_OP_RMS_NORM_MUL_ADD || owner.opcode == HTP_OP_MUL_MAT_ADD) {
            if (owner.inputs.size() >= 3 && owner.inputs[2]) {
                const ggml_tensor * res = owner.inputs[2];
                if (!is_f16_tensor(res) && !is_weight_or_bcast(res, r)) {
                    return true;
                }
            }
        }
        return false;
    };

    auto can_lower = [&](const ggml_tensor * r) -> bool {
        auto it = classes.find(r);
        if (it == classes.end() || ggml_hexagon_tensor_is_lowered_f16(r)) {
            return false;
        }
        auto sit = root_safe_class.find(r);
        if (sit == root_safe_class.end() || !sit->second) {
            return false;
        }
        htp_opnode * owner = find_owner(r);
        if (!owner) {
            return false;
        }
        if (!htp_lower_respect_prec_hints(*owner, r, it->second, nodes, owner_idx, root_users)) {
            return false;
        }
        if (is_residual_spine_stop(*owner, r)) {
            return false;
        }
        if (owner->opcode == HTP_OP_GATED_DELTA_NET && !htp_lower_safe_gdn_output(*owner, r, nodes, root_users)) {
            return false;
        }
        return true;
    };

    auto is_selected_root = [&](const ggml_tensor * r) -> bool {
        if (htp_lower_is_mm_root(nodes, owner_idx, r) || htp_lower_is_fa_root(nodes, owner_idx, r) ||
            (opt_oplower >= 2 && htp_lower_is_rms_root(nodes, owner_idx, r))) {
            return true;
        }
        if (opt_oplower >= 1) {
            for (const ggml_tensor * tensor : classes[r]) {
                auto oit = owner_idx.find(tensor);
                if (oit != owner_idx.end()) {
                    for (const ggml_tensor * input : nodes[oit->second].inputs) {
                        if (input && ggml_hexagon_tensor_is_lowered_f16(input)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    };

    struct saved_node {
        size_t       idx;
        htp_opnode * node;
        bool         dirty;
        int32_t      kernel_params[HTP_OP_MAX_KERN_PARAMS];
    };

    std::vector<int> touched_pos(nodes.size(), -1);

    for (const ggml_tensor * root : roots) {
        if (!can_lower(root) || !is_selected_root(root)) {
            continue;
        }

        std::vector<const ggml_tensor *> tx_roots;
        std::vector<saved_node>          touched;

        auto apply_root = [&](const ggml_tensor * r) -> bool {
            if (ggml_hexagon_tensor_is_lowered_f16(r)) {
                return false;
            }
            tx_roots.push_back(r);
            for (const ggml_tensor * tensor : classes[r]) {
                ggml_hexagon_tensor_set_flags(const_cast<ggml_tensor *>(tensor), HTP_TENSOR_LOWERED_F16);
            }

            auto uit = root_users.find(r);
            if (uit != root_users.end()) {
                for (size_t idx : uit->second) {
                    const int pos = touched_pos[idx];
                    if (pos < 0) {
                        touched_pos[idx] = (int) touched.size();
                        saved_node s;
                        s.idx   = idx;
                        s.node  = &nodes[idx];
                        s.dirty = true;
                        memcpy(s.kernel_params, nodes[idx].kernel_params, sizeof(s.kernel_params));
                        touched.push_back(s);
                    } else {
                        touched[pos].dirty = true;
                    }
                }
            }
            return true;
        };

        apply_root(root);

        bool solved = false;
        while (true) {
            solved = true;
            bool expanded = false;
            for (size_t i = 0; i < touched.size(); ++i) {
                if (!touched[i].dirty) {
                    continue;
                }
                htp_opnode * tnode = touched[i].node;
                if (htp_lower_inplace_types_ok(*tnode) && htp_kparams_solve(sess, *tnode)) {
                    touched[i].dirty = false;
                    continue;
                }
                solved = false;
                for (const ggml_tensor * out : tnode->outputs) {
                    const ggml_tensor * out_root = out ? htp_tensor_root(out) : nullptr;
                    if (out_root && can_lower(out_root) && apply_root(out_root)) {
                        expanded = true;
                    }
                }
                for (const ggml_tensor * inp : tnode->inputs) {
                    const ggml_tensor * inp_root = inp ? htp_tensor_root(inp) : nullptr;
                    if (inp_root && can_lower(inp_root) &&
                        (is_selected_root(inp_root) || htp_lower_is_rms_root(nodes, owner_idx, inp_root)) &&
                        apply_root(inp_root)) {
                        expanded = true;
                    }
                }
                if (expanded) {
                    break;
                }
            }
            if (solved || !expanded) {
                break;
            }
        }

        for (const saved_node & s : touched) {
            touched_pos[s.idx] = -1;
        }

        if (!solved) {
            for (const ggml_tensor * r : tx_roots) {
                for (const ggml_tensor * tensor : classes[r]) {
                    ggml_hexagon_tensor_clear_flags(const_cast<ggml_tensor *>(tensor), HTP_TENSOR_LOWERED_F16);
                }
            }
            for (saved_node & s : touched) {
                memcpy(s.node->kernel_params, s.kernel_params, sizeof(s.kernel_params));
            }
            continue;
        }

        for (const ggml_tensor * r : tx_roots) {
            ++n_lowered_roots;
            bytes_saved += ggml_nbytes(r) / 2;
            if (htp_lower_is_mm_root(nodes, owner_idx, r)) {
                ++n_lowered_mm;
            }
        }
    }

    for (htp_opnode & node : nodes) {
        htp_lower_opnode(sess, node);
    }

    if (n_lowered_roots > 0 || opt_oplower > 0) {
        HEX_VERBOSE("ggml-hex: %s lower oplower=%d roots=%zu/%zu mm_out=%zu saved=%zuB\n",
                    htp_session_name(sess), opt_oplower, n_lowered_roots, n_candidates, n_lowered_mm, bytes_saved);
    }
}
