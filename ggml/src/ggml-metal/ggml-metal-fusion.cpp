#include "ggml-metal-fusion.h"

#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// ---- helpers -------------------------------------------------------------

// follow the view/reshape chain to the underlying tensor
static const ggml_tensor * ggml_metal_fusion_view_src(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->n_ops >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->n_ops == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < fusion->n_ops; j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// MUL_MAT + UNARY (sigmoid/silu): dense mat-vec output followed directly by a supported unary op
static bool ggml_metal_fusion_check_mul_mat_unary(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * mm   = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (mm->op != GGML_OP_MUL_MAT || un->op != GGML_OP_UNARY || un->src[1]) {
        return false;
    }

    if (ggml_metal_fusion_view_src(un->src[0]) != mm) {
        return false;
    }

    const ggml_unary_op un_op = ggml_get_unary_op(un);
    if (un_op != GGML_UNARY_OP_SIGMOID && un_op != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (mm->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    // only mat-vec paths for now
    if (mm->src[1]->type != GGML_TYPE_F32 || mm->src[1]->ne[1] > 8) {
        return false;
    }

    const ggml_type wt = mm->src[0]->type;
    if (wt != GGML_TYPE_F32 && wt != GGML_TYPE_F16 && wt != GGML_TYPE_BF16 && wt != GGML_TYPE_Q8_0) {
        return false;
    }

    return true;
}

// MUL_MAT + ADD (bias) + UNARY (softplus): dense mat-vec output with a 1D bias and softplus
static bool ggml_metal_fusion_check_mul_mat_add_unary(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * mm   = nodes[0];
    const ggml_tensor * add  = nodes[1];
    const ggml_tensor * un   = nodes[2];
    const ggml_tensor * bias = add->src[1];

    if (mm->op != GGML_OP_MUL_MAT || add->op != GGML_OP_ADD ||
        un->op != GGML_OP_UNARY || un->src[0] != add || !bias) {
        return false;
    }

    if (ggml_metal_fusion_view_src(add->src[0]) != mm) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }

    if (mm->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(add) || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
        bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1 ||
        bias->ne[0] != mm->ne[0]) {
        return false;
    }

    // only mat-vec paths for now
    if (mm->src[1]->type != GGML_TYPE_F32 || mm->src[1]->ne[1] > 8) {
        return false;
    }

    const ggml_type wt = mm->src[0]->type;
    if (wt != GGML_TYPE_F32 && wt != GGML_TYPE_F16 && wt != GGML_TYPE_BF16 && wt != GGML_TYPE_Q8_0) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->n_ops >= 2);

    for (int j = 1; j < fusion->n_ops; j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_ASSERT(fusion->n_ops >= 3);

    const int n_ops = fusion->n_ops;

    const bool with_norm  = n_ops >= 6;
    const bool with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    static const ggml_op ops_topk_moe_raw[] = {
        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
    };
    static const ggml_op ops_topk_moe_scale_raw[] = {
        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
    };
    static const ggml_op ops_topk_moe_norm_raw[] = {
        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
        GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
    };
    static const ggml_op ops_topk_moe_norm_scale_raw[] = {
        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
        GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
    };

    const ggml_op * expected = nullptr;
    int             expected_n = 0;

    if (with_norm && with_scale) {
        expected    = ops_topk_moe_norm_scale_raw;
        expected_n  = 11;
    } else if (with_norm) {
        expected    = ops_topk_moe_norm_raw;
        expected_n  = 10;
    } else if (with_scale) {
        expected    = ops_topk_moe_scale_raw;
        expected_n  = 6;
    } else {
        expected    = ops_topk_moe_raw;
        expected_n  = 5;
    }

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != expected_n) {
        return false;
    }

    std::vector<ggml_op> raw_ops(raw_count);
    std::vector<int>     raw_idxs(raw_count);
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        raw_ops[i]  = gf->nodes[raw_start + i]->op;
    }
    if (!std::equal(expected, expected + expected_n, raw_ops.begin())) {
        return false;
    }

    const ggml_tensor * softmax        = gf->nodes[raw_start];
    const ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    const ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    const ggml_tensor * ids            = gf->nodes[raw_start + 3];
    const ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    const ggml_tensor * out            = gf->nodes[raw_end];
    const ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    float scale   = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    ((const int32_t *) softmax->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) softmax->op_params) + 1, sizeof(max_bias));
    if (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        return false;
    }

    if (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    if (n_expert <= 0 || n_tokens <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    if (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        const ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        const ggml_tensor * div              = gf->nodes[raw_start + 8];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            return false;
        }

        if (with_scale) {
            const ggml_tensor * scale_node = gf->nodes[raw_start + 10];
            if (scale_node->src[0] != out_reshaped) {
                return false;
            }
        }
    } else if (with_scale) {
        const ggml_tensor * scale_node = gf->nodes[raw_start + 5];
        if (scale_node->src[0] != get_rows) {
            return false;
        }
    }

    const int outputs[2] = { raw_start + 3, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs.data(), raw_count, raw_ops.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + 3 > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views < 2 || n_views > GGML_METAL_MOE_REDUCE_MAX_EXPERTS) {
        return false;
    }

    const int n_expert_used = n_views;
    const int raw_count     = 1 + n_expert_used + (n_expert_used - 1);

    if (node_idx + raw_count > gf->n_nodes) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    std::vector<ggml_op> raw_ops(raw_count);
    std::vector<int>     raw_idxs(raw_count);
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        raw_ops[i]  = gf->nodes[node_idx + i]->op;
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    if (n_embd <= 0 || n_tokens <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs.data(), raw_count, raw_ops.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_cgraph            * gf,
        const int                    * node_idxs,
              int                      idx,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], &match)) {
        return false;
    }

    if (fusion->n_ops != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + fusion->n_ops - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }
    }

    return true;
}

// ---- patterns ------------------------------------------------------------

static const ggml_op ops_norm_mul[]         = { GGML_OP_NORM, GGML_OP_MUL };
static const ggml_op ops_norm_mul_add[]     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_norm_scale[]       = { GGML_OP_NORM, GGML_OP_SCALE };
static const ggml_op ops_rms_norm_mul[]     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const ggml_op ops_rms_norm_mul_add[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_rms_norm_scale[]   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const ggml_op ops_add_2[] = { GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_3[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_4[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_5[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_6[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_7[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_snake[] = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const ggml_op ops_gdn_cache[] = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const ggml_op ops_topk_moe[] = {
    GGML_OP_SOFT_MAX, GGML_OP_ARGSORT, GGML_OP_GET_ROWS
};
static const ggml_op ops_topk_moe_scale[] = {
    GGML_OP_SOFT_MAX, GGML_OP_ARGSORT, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const ggml_op ops_topk_moe_norm[] = {
    GGML_OP_SOFT_MAX, GGML_OP_ARGSORT, GGML_OP_GET_ROWS,
    GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV
};
static const ggml_op ops_topk_moe_norm_scale[] = {
    GGML_OP_SOFT_MAX, GGML_OP_ARGSORT, GGML_OP_GET_ROWS,
    GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_SCALE
};

static const ggml_op ops_mul_mat_unary[]     = { GGML_OP_MUL_MAT, GGML_OP_UNARY };
static const ggml_op ops_mul_mat_add_unary[] = { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_UNARY };

static const ggml_op ops_moe_reduce_2[] = { GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_3[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_4[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_5[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_6[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_7[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_reduce_8[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };

static const ggml_metal_fusion ggml_metal_fusions[] = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     3, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, 3, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_2,            2, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_3,            3, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_4,            4, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_5,            5, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_6,            6, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_7,            7, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,        ops_snake,            5, false, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,    ops_gdn_cache,        2, true,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,     ops_topk_moe,          3, true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,     ops_topk_moe_scale,    4, true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,     ops_topk_moe_norm,     6, true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,     ops_topk_moe_norm_scale, 7, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, 2, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, 3, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, 4, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, 5, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, 6, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, 7, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, 8, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MUL_MAT_UNARY,     ops_mul_mat_unary,     2, true, ggml_metal_fusion_check_mul_mat_unary },
    { GGML_METAL_FUSION_MUL_MAT_ADD_UNARY, ops_mul_mat_add_unary, 3, true, ggml_metal_fusion_check_mul_mat_add_unary },
};

const ggml_metal_fusion * ggml_metal_fusion_all(int * n) {
    *n = (int) sizeof(ggml_metal_fusions) / sizeof(ggml_metal_fusions[0]);

    return ggml_metal_fusions;
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < fusion->n_ops; j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const struct ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const struct ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    int idx = -1;
    for (int i = 0; i < n; i++) {
        if (&all[i] == fusion) {
            idx = i;
            break;
        }
    }

    if (idx >= 0 && idx < (int) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(struct ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(struct ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    finfo->labels.clear();
    finfo->counts.assign(n, 0);
    finfo->labels.reserve(n);

    for (int i = 0; i < n; i++) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&all[i]));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(struct ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(struct ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (int i = 0; i < n; i++) {
        const ggml_metal_fusion * fusion = &all[i];

        // only look for a longer match than the current best
        if (fusion->n_ops <= best) {
            continue;
        }
        if (idx + fusion->n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < fusion->n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion->ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion->unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < fusion->n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // all current fusions are single-output elision chains, so the last node is the only output
            // TODO: multi-output fusions: store pattern-relative offsets in the table and translate them here
            int outputs_buf[1];
            outputs_buf[0] = node_idxs[idx + fusion->n_ops - 1];

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->n_ops, fusion->ops, outputs_buf, 1)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion->check && !fusion->check(fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        best = fusion->n_ops;
        res = fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // an empty/view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op) || ggml_is_empty(gf->nodes[idx])) {
        return 1;
    }

    // collect the non-empty node indices starting at idx
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
