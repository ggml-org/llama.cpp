#include "htp-fusion.h"
#include "htp-kparams.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>

static inline bool htp_same_tensor_data(const ggml_tensor * a, const ggml_tensor * b) {
    return a && b && (a == b || a->data == b->data);
}

static inline bool htp_same_shape_and_contiguity(const ggml_tensor * a, const ggml_tensor * b) {
    return ggml_are_same_shape(a, b) && (ggml_is_contiguous(a) == ggml_is_contiguous(b));
}

static inline bool htp_inplace_overlap_unsafe(const ggml_tensor * dst, const ggml_tensor * in) {
    return htp_tensors_overlap(dst, in) && (dst->data != in->data || !ggml_are_same_stride(dst, in));
}

static inline bool htp_inputs_overlap_unsafe(const htp_opnode & prev, const ggml_tensor * dst, const ggml_tensor * extra = nullptr) {
    if (extra && htp_inplace_overlap_unsafe(dst, extra)) return true;
    for (const auto * in : prev.inputs) {
        if (htp_inplace_overlap_unsafe(dst, in)) return true;
    }
    return false;
}

static inline bool htp_match_consumer(
    const htp_opnode & prev,
    const htp_opnode & cur,
    bool commutative,
    const ggml_tensor ** other_src = nullptr
) {
    if (prev.outputs.size() != 1) {
        return false;
    }
    const ggml_tensor * prev_out = prev.dst();
    if (!ggml_hexagon_tensor_is_fuseable(prev_out)) {
        return false;
    }
    if (htp_same_tensor_data(cur.src0(), prev_out)) {
        if (other_src) *other_src = cur.src1();
        return true;
    }
    if (commutative && htp_same_tensor_data(cur.src1(), prev_out)) {
        if (other_src) *other_src = cur.src0();
        return true;
    }
    return false;
}

static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return htp_mm_kernel_is_hmx_2d(kparams->kernel_type);
    }
    return ggml_hexagon_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return htp_mm_kernel_is_hmx_2d(kparams->kernel_type);
    }
    return ggml_hexagon_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

// --- Fusion rules ---

static bool fuse_allreduce_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (opt_ar_select != 2 || prev.opcode != HTP_OP_ALLREDUCE) return false;

    const auto * ar_kparams = prev.kparams<htp_allreduce_kernel_params>();
    const uint32_t rank    = (uint32_t) ar_kparams->rank;
    const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;

    const ggml_tensor * ar_local = prev.inputs[rank];
    if (!ggml_hexagon_tensor_is_fuseable(ar_local)) return false;

    if (cur.inputs.size() < 2) return false;
    const ggml_tensor * cur_src0   = cur.inputs[0];
    const ggml_tensor * cur_src1   = cur.inputs[1];
    const ggml_tensor * add_dst    = cur.dst();
    const ggml_tensor * res_tensor = nullptr;
    if (htp_same_tensor_data(cur_src0, ar_local)) {
        res_tensor = cur_src1;
    } else if (htp_same_tensor_data(cur_src1, ar_local)) {
        res_tensor = cur_src0;
    } else {
        return false;
    }

    if (ar_local->type != res_tensor->type || ar_local->type != add_dst->type) return false;

    const bool is_same_shape = ggml_are_same_shape(ar_local, res_tensor);
    const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && ggml_is_vector(res_tensor));
    if (!is_same_shape && !is_row_bcast) return false;

    if (is_same_shape && !ggml_are_same_stride(ar_local, res_tensor)) {
        return false;
    }
    if (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) return false;

    const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

    struct htp_allreduce_kernel_params kparams;
    if (!htp_kparams_solve_allreduce(sess, add_dst, rank, n_ranks, true, is_row_bcast, is_shard_ok, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", htp_session_name(sess));
        return false;
    }

    const bool scatter_ok = kparams.n_dsts > 1;
    kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    for (uint32_t r = 0; r < n_ranks; r++) {
        const ggml_tensor * ar_src = prev.inputs[r];
        if (!htp_tensors_overlap(add_dst, ar_src)) continue;
        if (scatter_ok && r == rank) continue;
        HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", htp_session_name(sess), r);
        return false;
    }

    prev.opcode = HTP_OP_ALLREDUCE_ADD;
    prev.inputs.push_back(res_tensor);
    if (kparams.mode != HTP_ALLREDUCE_SHARDED_FANOUT) {
        prev.outputs = { add_dst };
    }
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_rms_norm_mul(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_RMS_NORM) return false;

    const ggml_tensor * weight = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &weight)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) ||
        src0->ne[0] != weight->ne[0] || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, weight)) return false;

    const bool is_row_bcast  = ggml_is_vector(weight);
    const bool is_dim1_bcast = (weight->ne[1] == src0->ne[1] && ggml_is_matrix(weight));
    const bool is_same_shape = ggml_are_same_shape(src0, weight);
    if (!is_row_bcast && !is_dim1_bcast && !is_same_shape) return false;

    struct htp_unary_kernel_params kparams;
    if (!htp_kparams_solve_rms_norm_mul(sess, src0, weight, dst, &kparams)) {
        HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                    htp_session_name(sess), kparams.vtcm_size, htp_session_vtcm_size(sess));
        return false;
    }

    prev.opcode  = HTP_OP_RMS_NORM_MUL;
    prev.inputs  = { src0, weight };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

// post-norm residual: dst = rms_norm(x) * w + r
static bool fuse_rms_norm_mul_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_RMS_NORM_MUL) return false;

    const ggml_tensor * r = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &r)) return false;

    const ggml_tensor * src0   = prev.src0();
    const ggml_tensor * weight = prev.inputs[1];
    const ggml_tensor * dst    = cur.dst();
    if ((r->type != GGML_TYPE_F32 && r->type != GGML_TYPE_F16) ||
        (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16)) return false;
    if (!ggml_are_same_shape(src0, r) || (ggml_is_contiguous(src0) != ggml_is_contiguous(r))) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, r)) return false;

    struct htp_unary_kernel_params kparams;
    if (!htp_kparams_solve_rms_norm_mul_add(sess, src0, weight, r, dst, &kparams)) return false;

    prev.opcode = HTP_OP_RMS_NORM_MUL_ADD;
    prev.inputs.push_back(r);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_rms_norm_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_RMS_NORM && prev.opcode != HTP_OP_RMS_NORM_MUL) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float scale = 0.0f;
    float bias  = 0.0f;
    memcpy(&scale, (const float *) cur.params + 0, sizeof(float));
    memcpy(&bias,  (const float *) cur.params + 1, sizeof(float));
    if (bias != 0.0f || scale == 0.0f) return false;

    float prev_scale = 0.0f;
    memcpy(&prev_scale, &prev.params[1], sizeof(float));
    if (prev_scale != 0.0f) {
        scale *= prev_scale;
    }
    memcpy(&prev.params[1], &scale, sizeof(float));

    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_mul_mat_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &src2)) return false;
    if ((src2->type != GGML_TYPE_F32 && src2->type != GGML_TYPE_F16) ||
        !ggml_is_contiguous_to_2(src2)) {
        return false;
    }

    const ggml_tensor * src0 = prev.node->src[0];
    const ggml_tensor * src1 = prev.node->src[1];
    const ggml_tensor * dst  = cur.node;
    if (!ggml_are_same_shape(prev.node, dst)) return false;

    // Require matching ne[0], valid broadcast dims (1 or equal), and no unsafe overlap.
    if (src2->ne[0] != dst->ne[0] ||
        (src2->ne[1] != 1 && src2->ne[1] != dst->ne[1]) ||
        (src2->ne[2] != 1 && src2->ne[2] != dst->ne[2]) ||
        (src2->ne[3] != 1 && src2->ne[3] != dst->ne[3])) {
        return false;
    }
    if (htp_tensors_overlap(dst, src0) || htp_tensors_overlap(dst, src1) || htp_inplace_overlap_unsafe(dst, src2)) {
        return false;
    }

    const bool prev_collapsed = htp_matmul_can_collapse(src0, src1, prev.node);
    const bool cur_collapsed  = htp_matmul_add_can_collapse(src0, src1, src2, dst);
    if (prev_collapsed != cur_collapsed) {
        return false;
    }

    const auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    struct htp_mm_kernel_params kparams;
    if (!htp_kparams_solve_matmul_add(sess, src0, src1, src2, dst, &kparams)) {
        return false;
    }

    const int  src1_nrows = (int) ggml_nrows(src1);
    const bool is_batched = (!ggml_is_matrix(src0) || !ggml_is_matrix(src1));
    if (kparams.n_hmx == 0 && (src1_nrows != 1 || is_batched)) return false;

    if (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
        if (kparams.m_chunk < orig_kparams->m_chunk ||
            kparams.n_act_hvx < orig_kparams->n_act_hvx) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, hvx %d->%d)\n",
                        htp_session_name(sess), orig_kparams->m_chunk, kparams.m_chunk,
                        orig_kparams->n_chunk, kparams.n_chunk,
                        orig_kparams->n_act_hvx, kparams.n_act_hvx);
            return false;
        }
    }

    prev.opcode  = HTP_OP_MUL_MAT_ADD;
    prev.inputs  = { src0, src1, src2 };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_mul_mat_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT && prev.opcode != HTP_OP_MUL_MAT_NX) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0     = prev.inputs[0];
    const ggml_tensor * x_orig = prev.node->src[1];
    const ggml_tensor * w_in   = cur.node->src[0];
    const ggml_tensor * x_in   = cur.node->src[1];
    const ggml_tensor * d_in   = cur.node;

    auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    const auto * cur_kparams = cur.kparams<htp_mm_kernel_params>();

    if (x_in != x_orig || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
        !ggml_is_matrix(w0) || !ggml_is_matrix(w_in) ||
        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
        orig_kparams->n_hmx != cur_kparams->n_hmx ||
        orig_kparams->kernel_type != cur_kparams->kernel_type) {
        return false;
    }

    const bool collapse = htp_matmul_can_collapse(w0, x_orig, prev.node) && htp_matmul_can_collapse(w_in, x_in, d_in);
    if (!ggml_is_matrix(x_orig) && !collapse) return false;

    const ggml_tensor * prev_w        = prev.inputs[0];
    ggml_type           prev_dst_type = prev.outputs[0]->type;
    for (size_t i = 1; i < curr_n; ++i) {
        if (prev.inputs[i]->ne[1] > prev_w->ne[1]) prev_w = prev.inputs[i];
        if (ggml_type_size(prev.outputs[i]->type) > ggml_type_size(prev_dst_type)) {
            prev_dst_type = prev.outputs[i]->type;
        }
    }
    const ggml_tensor * w_for_solve  = (w_in->ne[1] > prev_w->ne[1]) ? w_in : prev_w;
    const ggml_type     max_dst_type = (ggml_type_size(d_in->type) > ggml_type_size(prev_dst_type)) ? d_in->type : prev_dst_type;

    if (prev.opcode == HTP_OP_MUL_MAT_NX && w_for_solve == prev_w &&
        ggml_type_size(max_dst_type) == ggml_type_size(prev_dst_type)) {
        orig_kparams->n_weights = curr_n + 1;
    } else {
        struct htp_mm_kernel_params kparams;
        const ggml_tensor x_collapsed = collapse ? htp_tensor_collapse_rows(x_orig) : *x_orig;
        ggml_tensor       d_collapsed = collapse ? htp_tensor_collapse_rows(prev.outputs[0]) : *prev.outputs[0];
        d_collapsed.type = max_dst_type;
        if (!htp_kparams_solve_mmnx(sess, w_for_solve, &x_collapsed, &d_collapsed, curr_n + 1, &kparams)) return false;
        if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) return false;

        prev.set_kparams(kparams);
        prev.opcode = HTP_OP_MUL_MAT_NX;
    }

    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    return true;
}

static bool htp_is_fuseable_mmnx_glu_op(uint32_t op) {
    return op == HTP_OP_GLU_SWIGLU || op == HTP_OP_GLU_GEGLU ||
           op == HTP_OP_GLU_GEGLU_QUICK || op == HTP_OP_GLU_REGLU;
}

static bool fuse_mul_mat_nx_glu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT_NX || prev.outputs.size() != 2 || prev.inputs.size() != 3) {
        return false;
    }
    if (!htp_is_fuseable_mmnx_glu_op(cur.opcode) || cur.inputs.size() != 2 || cur.outputs.size() != 1 || !cur.src1()) {
        return false;
    }
    if (prev.fused.size() + 1 + cur.fused.size() > HTP_OP_MAX_FUSED) {
        return false;
    }

    const auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    if (!orig_kparams->n_hmx || !htp_mm_kernel_is_hmx_2d(orig_kparams->kernel_type)) {
        return false;
    }

    const ggml_tensor * out0 = prev.outputs[0];
    const ggml_tensor * out1 = prev.outputs[1];
    if (!ggml_hexagon_tensor_is_fuseable(out0) || !ggml_hexagon_tensor_is_fuseable(out1) ||
        (out0->flags & GGML_TENSOR_FLAG_OUTPUT) || (out1->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_prec_get_acc(out0) == GGML_PREC_F32 || ggml_prec_get_acc(out1) == GGML_PREC_F32) {
        return false;
    }

    const ggml_tensor * w_gate   = nullptr;
    const ggml_tensor * w_up     = nullptr;
    const ggml_tensor * out_gate = nullptr;
    const ggml_tensor * out_up   = nullptr;
    if (htp_same_tensor_data(cur.src0(), out0) && htp_same_tensor_data(cur.src1(), out1)) {
        w_gate   = prev.inputs[0];
        w_up     = prev.inputs[1];
        out_gate = out0;
        out_up   = out1;
    } else if (htp_same_tensor_data(cur.src0(), out1) && htp_same_tensor_data(cur.src1(), out0)) {
        w_gate   = prev.inputs[1];
        w_up     = prev.inputs[0];
        out_gate = out1;
        out_up   = out0;
    } else {
        return false;
    }

    const ggml_tensor * act = prev.inputs[2];
    const ggml_tensor * dst = cur.dst();
    if (w_gate->ne[1] != w_up->ne[1] || w_gate->ne[0] != w_up->ne[0] ||
        !ggml_is_contiguous(out_gate) || !ggml_is_contiguous(out_up) || !ggml_is_contiguous(dst) ||
        out_gate->type != cur.src0()->type || out_up->type != cur.src1()->type ||
        !ggml_are_same_shape(out_gate, cur.src0()) || !ggml_are_same_stride(out_gate, cur.src0()) ||
        !ggml_are_same_shape(out_up,   cur.src1()) || !ggml_are_same_stride(out_up,   cur.src1()) ||
        !ggml_are_same_shape(out_gate, dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (htp_tensors_overlap(dst, w_gate) || htp_tensors_overlap(dst, w_up) || htp_tensors_overlap(dst, act)) {
        return false;
    }

    const bool collapse = htp_matmul_can_collapse(w_gate, act, dst) && htp_matmul_can_collapse(w_up, act, dst);
    if (!ggml_is_matrix(act) && !collapse) {
        return false;
    }

    const ggml_tensor x_collapsed = collapse ? htp_tensor_collapse_rows(act) : *act;
    const ggml_tensor d_collapsed = collapse ? htp_tensor_collapse_rows(dst) : *dst;

    struct htp_mm_kernel_params kparams;
    if (!htp_kparams_solve_mmnx_glu(sess, w_gate, w_up, &x_collapsed, &d_collapsed, (uint8_t) cur.opcode, &kparams)) {
        return false;
    }

    prev.opcode  = HTP_OP_MUL_MAT_NX_GLU;
    prev.inputs  = { w_gate, w_up, act };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    for (const ggml_tensor * f : cur.fused) {
        prev.fused.push_back(f);
    }
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_mul_mat_id_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL_MAT_ID && prev.opcode != HTP_OP_MUL_MAT_ID_NX) return false;

    const uint32_t curr_n = (uint32_t) prev.outputs.size();
    if (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) return false;

    const ggml_tensor * w0     = prev.inputs[0];
    const ggml_tensor * x      = prev.inputs[curr_n];
    const ggml_tensor * ids    = prev.inputs[curr_n + 1];
    const ggml_tensor * w_in   = cur.src0();
    const ggml_tensor * x_in   = cur.src1();
    const ggml_tensor * ids_in = cur.node->src[2];
    const ggml_tensor * d_in   = cur.dst();

    auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    const auto * cur_kparams = cur.kparams<htp_mm_kernel_params>();

    if (x_in != x || ids_in != ids || w_in->type != w0->type ||
        w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
        orig_kparams->n_hmx != cur_kparams->n_hmx) {
        return false;
    }

    struct htp_mm_kernel_params kparams;
    if (d_in->nb[1] == prev.node->nb[1]) {
        kparams = *orig_kparams;
        kparams.n_weights = curr_n + 1;
    } else if (!htp_kparams_solve_mmidnx(sess, w0, x, d_in, curr_n + 1, &kparams)) {
        return false;
    }
    if (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) return false;

    prev.opcode = HTP_OP_MUL_MAT_ID_NX;
    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_gdn_cpy(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_GATED_DELTA_NET || prev.outputs.size() != 1) return false;

    const ggml_tensor * gdn_out  = prev.dst();
    const ggml_tensor * cpy_node = cur.node;
    const ggml_tensor * cpy_src  = cur.src0();
    const ggml_tensor * cpy_dst  = cur.dst();

    if (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) return false;
    if ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_node->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;

    const ggml_tensor * v = prev.node->src[2];
    if (!v) return false;

    const int64_t S_v       = v->ne[0];
    const int64_t H         = v->ne[1];
    const int64_t n_tokens  = v->ne[2];
    const int64_t n_seqs    = v->ne[3];
    const int64_t K         = ggml_get_op_params_i32(prev.node, 0);
    const size_t  tail_off  = (size_t) S_v * H * n_tokens * n_seqs * sizeof(float);
    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    if (cpy_src->op != GGML_OP_VIEW || !htp_same_tensor_data(cpy_src->view_src, gdn_out) ||
        cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
        return false;
    }

    if (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->ne[2] != n_written || !ggml_is_contiguous(cpy_dst)) {
        return false;
    }

    prev.outputs.push_back(cpy_dst);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_ssm_conv_silu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_SSM_CONV) return false;

    auto * kparams = prev.kparams<htp_ssm_conv_kernel_params>();
    if (kparams->silu) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * conv_dst = prev.dst();
    const ggml_tensor * dst      = cur.dst();
    if (!ggml_are_same_shape(conv_dst, cur.src0()) || !ggml_are_same_stride(conv_dst, cur.src0())) return false;
    if (!ggml_are_same_shape(conv_dst, dst)) return false;
    if (htp_tensors_overlap(dst, prev.src0()) || htp_tensors_overlap(dst, prev.src1())) return false;

    htp_opnode cand = prev;
    cand.outputs[0] = dst;
    cand.fused.push_back(cur.node);
    cand.kparams<htp_ssm_conv_kernel_params>()->silu = 1;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    return true;
}

static bool fuse_repeat_get_rows(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_REPEAT) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * repeat_out  = prev.dst();
    const ggml_tensor * repeat_src0 = prev.src0();
    const ggml_tensor * gr_src1     = cur.src1();
    const ggml_tensor * gr_dst      = cur.dst();

    if (repeat_src0->ne[0] != repeat_out->ne[0] || repeat_src0->ne[1] != repeat_out->ne[1]) return false;
    if (repeat_src0->type != repeat_out->type) return false;

    struct htp_get_rows_kernel_params kparams;
    if (!htp_kparams_solve_repeat_get_rows(sess, repeat_src0, gr_src1, gr_dst, &kparams) || kparams.vtcm_size == 0) return false;

    prev.opcode  = HTP_OP_GET_ROWS;
    prev.inputs  = { repeat_src0, gr_src1 };
    prev.outputs = { gr_dst };
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static inline bool htp_is_scale_bias_unary(uint32_t opcode) {
    switch (opcode) {
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_SILU:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_GELU_ERF:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
            return true;
        default:
            return false;
    }
}

static bool fuse_scale_unary(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_SCALE || !prev.fused.empty() || !htp_is_scale_bias_unary(cur.opcode)) return false;
    if (cur.kparams<htp_unary_kernel_params>()->has_scale_bias) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        dst->type != src0->type || !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float in_scale = 0.0f;
    float in_bias  = 0.0f;
    memcpy(&in_scale, (const float *) prev.params + 0, sizeof(float));
    memcpy(&in_bias,  (const float *) prev.params + 1, sizeof(float));

    htp_opnode cand = prev;
    cand.opcode  = cur.opcode;
    cand.outputs = { dst };
    memcpy(cand.params, cur.params, sizeof(cand.params));

    auto * kp          = cand.kparams<htp_unary_kernel_params>();
    kp->has_scale_bias = 1;
    kp->in_scale       = in_scale;
    kp->in_bias        = in_bias;
    kp->out_scale      = 1.0f;
    kp->out_bias       = 0.0f;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_unary_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (!htp_is_scale_bias_unary(prev.opcode) || prev.fused.size() >= HTP_OP_MAX_FUSED) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        dst->type != src0->type || !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst)) return false;

    float s_out = 0.0f;
    float b_out = 0.0f;
    memcpy(&s_out, (const float *) cur.params + 0, sizeof(float));
    memcpy(&b_out, (const float *) cur.params + 1, sizeof(float));

    const auto * prev_kp  = prev.kparams<htp_unary_kernel_params>();
    const float in_scale  = prev_kp->has_scale_bias ? prev_kp->in_scale  : 1.0f;
    const float in_bias   = prev_kp->has_scale_bias ? prev_kp->in_bias   : 0.0f;
    const float out_scale = (prev_kp->has_scale_bias ? prev_kp->out_scale : 1.0f) * s_out;
    const float out_bias  = (prev_kp->has_scale_bias ? prev_kp->out_bias  : 0.0f) * s_out + b_out;

    htp_opnode cand = prev;
    cand.outputs = { dst };

    auto * kp          = cand.kparams<htp_unary_kernel_params>();
    kp->has_scale_bias = 1;
    kp->in_scale       = in_scale;
    kp->in_bias        = in_bias;
    kp->out_scale      = out_scale;
    kp->out_bias       = out_bias;
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    return true;
}

static bool fuse_unary_mul_glu(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    htp_op_code glu_op = HTP_OP_INVALID;
    switch (prev.opcode) {
        case HTP_OP_UNARY_SILU:       glu_op = HTP_OP_GLU_SWIGLU;      break;
        case HTP_OP_UNARY_GELU:       glu_op = HTP_OP_GLU_GEGLU;       break;
        case HTP_OP_UNARY_GELU_QUICK: glu_op = HTP_OP_GLU_GEGLU_QUICK; break;
        case HTP_OP_UNARY_GELU_ERF:   glu_op = HTP_OP_GLU_GEGLU_ERF;   break;
        case HTP_OP_UNARY_RELU:       glu_op = HTP_OP_GLU_REGLU;       break;
        case HTP_OP_UNARY_SIGMOID:    glu_op = HTP_OP_GLU_SIGMOID;     break;
        default: return false;
    }
    if (prev.kparams<htp_unary_kernel_params>()->has_scale_bias) return false;

    const ggml_tensor * up = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &up)) return false;

    const ggml_tensor * gate = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (htp_inputs_overlap_unsafe(prev, dst, up)) return false;

    htp_opnode cand = prev;
    cand.opcode  = glu_op;
    cand.inputs  = { gate, up };
    cand.outputs = { dst };
    if (!htp_kparams_solve(sess, cand)) return false;

    prev = std::move(cand);
    prev.fused.push_back(cur.node);
    memset(prev.params, 0, sizeof(prev.params));
    return true;
}

static bool fuse_mul_mul_scalar(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != HTP_OP_MUL || prev.inputs.size() != 2) return false;

    const auto * k0 = prev.kparams<htp_binary_kernel_params>();
    const auto * k1 = cur.kparams<htp_binary_kernel_params>();
    if (k0->kernel_type != HTP_BINARY_KERNEL_SCALAR || k1->kernel_type != HTP_BINARY_KERNEL_SCALAR) return false;
    if (k0->src1_strided || k1->src1_strided) return false;

    const ggml_tensor * src2 = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false, &src2)) return false;

    const ggml_tensor * src1 = prev.src1();
    const ggml_tensor * dst  = cur.dst();
    if (prev.src0()->type != src1->type || prev.dst()->type != src1->type || src2->type != src1->type || dst->type != src1->type) return false;
    if (!ggml_are_same_shape(src1, src2) || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, src2)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_kernel_params kparams = *k0;
    kparams.src2_size       = kparams.src1_size;
    kparams.rows_per_buffer = 0;
    if (kparams.src1_size > 65536 || kparams.src1_size + kparams.src2_size > vtcm_size) return false;

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    if (L.rows_per_buffer == 0 || L.total_bytes > vtcm_size) return false;

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = L.total_bytes;

    prev.inputs.push_back(src2);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

static bool fuse_binary_nx(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (prev.opcode != cur.opcode) return false;

    const uint32_t curr_n = (uint32_t) prev.inputs.size();
    if (curr_n < 2 || curr_n >= HTP_OP_MAX_INPUTS) return false;

    const auto * k0 = prev.kparams<htp_binary_kernel_params>();
    const auto * k1 = cur.kparams<htp_binary_kernel_params>();
    const bool k0_ok = (k0->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k0->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k0->is_scalar);
    const bool k1_ok = (k1->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) ||
                       (k1->kernel_type == HTP_BINARY_KERNEL_CHUNKED && !k1->is_scalar);
    if (!k0_ok || !k1_ok) return false;

    const bool is_comm = (cur.opcode == HTP_OP_ADD || cur.opcode == HTP_OP_MUL);
    const ggml_tensor * new_src = nullptr;
    if (!htp_match_consumer(prev, cur, is_comm, &new_src)) return false;

    const ggml_tensor * src0 = prev.inputs[0];
    const ggml_tensor * dst  = cur.dst();
    if (prev.dst()->type != src0->type || new_src->type != src0->type || dst->type != src0->type) return false;
    for (const auto * in : prev.inputs) {
        if (in->type != src0->type) return false;
    }
    if (!ggml_are_same_shape(src0, prev.inputs[1]) ||
        !ggml_are_same_shape(src0, new_src) ||
        !htp_same_shape_and_contiguity(prev.dst(), dst)) {
        return false;
    }
    if (htp_inputs_overlap_unsafe(prev, dst, new_src)) return false;

    const size_t vtcm_size = htp_session_vtcm_size(sess);
    struct htp_binary_kernel_params kparams = *k0;
    struct htp_binary_vtcm_layout L;
    memset(&L, 0, sizeof(L));

    const uint32_t n_srcs = curr_n + 1;
    if (k0->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE || k1->kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
        if (k0->kernel_type != HTP_BINARY_KERNEL_SAME_SHAPE) {
            kparams = *k1;
        }
        kparams.n_srcs          = (uint8_t) n_srcs;
        kparams.rows_per_buffer = 0;

        auto check_split = [&](const ggml_tensor * t) {
            if (src0->ne[2] > 1 && t->nb[2] != t->ne[1] * t->nb[1]) kparams.split_at_ne01 = 1;
            if (src0->ne[3] > 1 && t->nb[3] != t->ne[2] * t->nb[2]) kparams.split_at_ne02 = 1;
        };
        for (const auto * in : prev.inputs) check_split(in);
        check_split(new_src);
        check_split(dst);

        htp_binary_vtcm_layout_build(&L, &kparams, vtcm_size);
    }

    if (L.rows_per_buffer == 0 || L.total_bytes == 0 || L.total_bytes > vtcm_size) {
        bool all_contig = ggml_is_contiguous(new_src) && ggml_is_contiguous(dst);
        for (const auto * in : prev.inputs) {
            all_contig = all_contig && ggml_is_contiguous(in);
        }
        if (!all_contig) return false;

        const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        const uint32_t elem_size   = (uint32_t) ggml_type_size(src0->type);
        const uint32_t max_hvx     = htp_session_n_hvx(sess);
        if (!htp_binary_solve_chunked(&L, &kparams, total_elems, elem_size, elem_size, elem_size, max_hvx, n_srcs, false, vtcm_size)) {
            return false;
        }
    }

    kparams.rows_per_buffer = L.rows_per_buffer;
    kparams.vtcm_size       = (uint32_t) L.total_bytes;

    prev.inputs.push_back(new_src);
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
    prev.set_kparams(kparams);
    return true;
}

bool htp_fusion_can_stack(const struct ggml_tensor * a) {
    if (!a || !a->src[0] || !a->src[1]) return false;
    switch (a->op) {
        case GGML_OP_MUL_MAT:
            if (!htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX)) return false;
            if (htp_is_hadamard(a)) return false;
            if (a->src[0]->ne[2] != 1 || a->src[0]->ne[3] != 1) return false;
            return ggml_is_quantized(a->src[0]->type) ||
                   (a->src[0]->type == GGML_TYPE_F16 && a->src[0]->ne[0] % 32 == 0);
        case GGML_OP_MUL_MAT_ID:
            if (!htp_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX)) return false;
            return a->src[2] != nullptr && ggml_is_quantized(a->src[0]->type);
        default:
            return false;
    }
}

bool htp_fusion_can_stack(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (!htp_fusion_can_stack(b) || a->op != b->op || a->src[1] != b->src[1]) {
        return false;
    }
    if (a->op == GGML_OP_MUL_MAT_ID && a->src[2] != b->src[2]) {
        return false;
    }
    const ggml_tensor * w0 = a->src[0];
    const ggml_tensor * w1 = b->src[0];
    return w0->type == w1->type && w0->ne[0] == w1->ne[0] && w0->ne[2] == w1->ne[2];
}

int htp_fusion_max(const struct ggml_cgraph * graph, int idx) {
    const int n = graph->n_nodes;
    const ggml_op op0 = graph->nodes[idx]->op;
    if (op0 != GGML_OP_ADD && op0 != GGML_OP_NORM && op0 != GGML_OP_RMS_NORM) {
        return 1;
    }

    enum ggml_op ops[HTP_OP_MAX_FUSED];
    ops[0] = op0;

    int f = idx + 1;
    while (f < n && f < idx + HTP_OP_MAX_FUSED) {
        const ggml_op op = graph->nodes[f]->op;
        if (op != GGML_OP_ADD && op != GGML_OP_MUL && op != GGML_OP_SCALE &&
            op != GGML_OP_NORM && op != GGML_OP_RMS_NORM) {
            break;
        }
        ops[f - idx] = op;
        f++;
    }

    f -= idx;
    for (; f > 1; f--) {
        if (ggml_can_fuse(graph, idx, ops, f)) {
            break;
        }
    }
    return f;
}

void htp_fusion_tag_graph(struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor * node = graph->nodes[i];
        ggml_hexagon_tensor_clear_flags(node, HTP_TENSOR_FUSEABLE);

        const ggml_op op = node->op;
        if ((op == GGML_OP_RMS_NORM || op == GGML_OP_SCALE || op == GGML_OP_REPEAT ||
             op == GGML_OP_UNARY    || op == GGML_OP_ADD   || op == GGML_OP_SUB    ||
             op == GGML_OP_MUL      || op == GGML_OP_DIV   || op == GGML_OP_MUL_MAT_ID ||
             op == GGML_OP_SSM_CONV ||
             (op == GGML_OP_MUL_MAT && !htp_is_hadamard(node))) &&
            ggml_node_has_n_uses(graph, i, 1)) {
            ggml_hexagon_tensor_set_flags(node, HTP_TENSOR_FUSEABLE);
        }
    }
}

bool htp_fusion_try_fuse(const struct ggml_hexagon_session * sess, struct htp_opnode & prev, const struct htp_opnode & cur) {
    if (!opt_opfusion) return false;
    if (!cur.fused.empty()) return false;

    switch (cur.opcode) {
        case HTP_OP_ADD:
            return (htp_fusion_enabled(HTP_FUSE_ALLREDUCE_ADD) && fuse_allreduce_add(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_MUL_MAT_ADD)   && fuse_mul_mat_add(sess, prev, cur))   ||
                   (htp_fusion_enabled(HTP_FUSE_RMS_NORM_MUL_ADD) && fuse_rms_norm_mul_add(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_BINARY_NX)     && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_MUL:
            return (htp_fusion_enabled(HTP_FUSE_RMS_NORM_MUL)   && fuse_rms_norm_mul(sess, prev, cur))   ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_MUL_GLU)  && fuse_unary_mul_glu(sess, prev, cur))  ||
                   (htp_fusion_enabled(HTP_FUSE_MUL_MUL_SCALAR) && fuse_mul_mul_scalar(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_BINARY_NX)      && fuse_binary_nx(sess, prev, cur));
        case HTP_OP_SUB:
        case HTP_OP_DIV:
            return htp_fusion_enabled(HTP_FUSE_BINARY_NX) && fuse_binary_nx(sess, prev, cur);
        case HTP_OP_SCALE:
            return (htp_fusion_enabled(HTP_FUSE_RMS_NORM_SCALE) && fuse_rms_norm_scale(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_SCALE)    && fuse_unary_scale(sess, prev, cur));
        case HTP_OP_UNARY_SILU:
            return (htp_fusion_enabled(HTP_FUSE_SSM_CONV_SILU) && fuse_ssm_conv_silu(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_UNARY_SCALE)   && fuse_scale_unary(sess, prev, cur));
        case HTP_OP_UNARY_EXP:
        case HTP_OP_UNARY_SIGMOID:
        case HTP_OP_UNARY_GELU:
        case HTP_OP_UNARY_GELU_QUICK:
        case HTP_OP_UNARY_GELU_ERF:
        case HTP_OP_UNARY_SOFTPLUS:
        case HTP_OP_UNARY_TANH:
            return htp_fusion_enabled(HTP_FUSE_UNARY_SCALE) && fuse_scale_unary(sess, prev, cur);
        case HTP_OP_GET_ROWS:
            return htp_fusion_enabled(HTP_FUSE_REPEAT_GET_ROWS) && fuse_repeat_get_rows(sess, prev, cur);
        case HTP_OP_CPY:
            return htp_fusion_enabled(HTP_FUSE_GDN_CPY) && fuse_gdn_cpy(sess, prev, cur);
        case HTP_OP_MUL_MAT:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX) && fuse_mul_mat_nx(sess, prev, cur);
        case HTP_OP_MUL_MAT_ID:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX) && fuse_mul_mat_id_nx(sess, prev, cur);
        case HTP_OP_GLU_SWIGLU:
        case HTP_OP_GLU_GEGLU:
        case HTP_OP_GLU_GEGLU_QUICK:
        case HTP_OP_GLU_REGLU:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX_GLU) && fuse_mul_mat_nx_glu(sess, prev, cur);
        default:
            return false;
    }
}

static void htp_fuse_gdn_state(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (nodes[c].opcode != HTP_OP_GATED_DELTA_NET || nodes[c].inputs.size() != 6 ||
            nodes[c].fused.size() >= HTP_OP_MAX_FUSED) {
            continue;
        }
        const ggml_tensor * v = nodes[c].inputs[2];
        const ggml_tensor * s = nodes[c].inputs[5];

        const ggml_tensor * base_s = s;
        while (base_s->view_src &&
               (base_s->op == GGML_OP_RESHAPE || (base_s->op == GGML_OP_VIEW && base_s->view_offs == 0)) &&
               ggml_is_contiguous(base_s)) {
            base_s = base_s->view_src;
        }
        if (base_s->flags & GGML_TENSOR_FLAG_OUTPUT) continue;

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode == HTP_OP_GET_ROWS && nodes[i].fused.empty() &&
                nodes[i].outputs.size() == 1 && nodes[i].dst() == base_s) {
                p = i;
                break;
            }
        }
        if (p == c) continue;

        const ggml_tensor * gr_src0 = nodes[p].src0();
        const ggml_tensor * gr_src1 = nodes[p].src1();
        const int64_t n_seqs = v->ne[3];
        if (htp_tensors_overlap(nodes[c].outputs[0], gr_src0) || htp_tensors_overlap(nodes[c].outputs[0], gr_src1)) {
            continue;
        }
        if (nodes[c].outputs.size() > 1 && nodes[c].outputs[1]) {
            const ggml_tensor * dst_cache = nodes[c].outputs[1];
            if (htp_tensors_overlap(dst_cache, gr_src1)) continue;
            if (htp_tensors_overlap(gr_src0, dst_cache)) {
                const uintptr_t src0_ptr = (uintptr_t) htp_tensor_data_ptr(gr_src0);
                const uintptr_t dst_ptr  = (uintptr_t) htp_tensor_data_ptr(dst_cache);
                const bool slot_aligned  = (n_seqs == 1) &&
                                           (dst_cache->type == gr_src0->type) &&
                                           (dst_ptr >= src0_ptr) &&
                                           (gr_src0->nb[1] > 0) &&
                                           ((dst_ptr - src0_ptr) % gr_src0->nb[1] == 0) &&
                                           (dst_cache->ne[2] == 1 || gr_src0->nb[1] == dst_cache->nb[2]);
                if (!slot_aligned) continue;
            }
        }

        bool safe = true;
        for (size_t k = 0; k < nodes.size() && safe; ++k) {
            if (k == p) continue;
            if (k != c) {
                for (const ggml_tensor * in : nodes[k].inputs) {
                    for (const ggml_tensor * t = in; t != nullptr; t = t->view_src) {
                        if (t == base_s) { safe = false; break; }
                    }
                    if (!safe) break;
                }
            }
            if (k > p && k < c) {
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, gr_src0) || htp_tensors_overlap(out, gr_src1) ||
                                htp_tensors_overlap(out, base_s))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) continue;

        htp_opnode cand = nodes[c];
        cand.inputs[5] = gr_src0;
        cand.inputs.push_back(gr_src1);
        cand.fused.insert(cand.fused.begin(), nodes[p].node);
        if (!htp_kparams_solve(sess, cand)) continue;

        nodes[c] = std::move(cand);
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

static bool htp_unwrap_transpose_01(const ggml_tensor * t, const ggml_tensor ** out_src) {
    if (!t || !t->view_src || t->view_offs != 0) {
        return false;
    }
    if (t->op == GGML_OP_TRANSPOSE) {
        *out_src = t->view_src;
        return true;
    }
    if (t->op == GGML_OP_PERMUTE) {
        if (ggml_get_op_params_i32(t, 0) == 1 &&
            ggml_get_op_params_i32(t, 1) == 0 &&
            ggml_get_op_params_i32(t, 2) == 2 &&
            ggml_get_op_params_i32(t, 3) == 3) {
            *out_src = t->view_src;
            return true;
        }
    }
    return false;
}

static bool htp_tensor_is_view_of(const ggml_tensor * in, const ggml_tensor * target) {
    if (!in || !target) {
        return false;
    }
    for (const ggml_tensor * t = in; t != nullptr; t = t->view_src) {
        if (t == target) {
            return true;
        }
    }
    return false;
}

static bool htp_tensor_reads_from(const ggml_tensor * in, const ggml_tensor * target) {
    if (!in || !target) {
        return false;
    }
    if (htp_tensor_is_view_of(in, target)) {
        return true;
    }
    return htp_tensors_overlap(in, target);
}

static void htp_fuse_concat_ssm_conv(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (nodes[c].opcode != HTP_OP_SSM_CONV || nodes[c].inputs.size() != 2 ||
            nodes[c].outputs.size() != 1 || nodes[c].fused.size() >= HTP_OP_MAX_FUSED) {
            continue;
        }

        const ggml_tensor * conv_x      = nodes[c].src0();
        const ggml_tensor * conv_weight = nodes[c].src1();
        const ggml_tensor * conv_dst    = nodes[c].dst();
        if (conv_x->flags & GGML_TENSOR_FLAG_OUTPUT) {
            continue;
        }

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode == HTP_OP_CONCAT && nodes[i].fused.empty() &&
                nodes[i].outputs.size() == 1 && nodes[i].dst() == conv_x &&
                ggml_get_op_params_i32(nodes[i].node, 0) == 0) {
                p = i;
                break;
            }
        }
        if (p == c) {
            continue;
        }

        const ggml_tensor * cat_state = nodes[p].src0();
        const ggml_tensor * cat_xT    = nodes[p].src1();
        const ggml_tensor * cat_x     = nullptr;
        if (!htp_unwrap_transpose_01(cat_xT, &cat_x)) {
            continue;
        }

        const int64_t d_conv  = conv_weight->ne[0];
        const int64_t d_inner = conv_weight->ne[1];
        const int64_t n_t     = conv_dst->ne[1];
        const int64_t n_s     = conv_dst->ne[2];
        const int64_t S       = d_conv - 1;

        size_t q          = nodes.size();
        size_t n_other    = 0;
        for (size_t k = 0; k < nodes.size(); ++k) {
            if (k == p || k == c) {
                continue;
            }
            bool uses_conv_x = false;
            for (const ggml_tensor * in : nodes[k].inputs) {
                if (htp_tensor_is_view_of(in, conv_x)) {
                    uses_conv_x = true;
                    break;
                }
            }
            if (uses_conv_x) {
                q = k;
                ++n_other;
            }
        }
        if (n_other > 1) {
            continue;
        }

        const bool has_cpy = (n_other == 1);
        const ggml_tensor * cpy_dst = nullptr;
        if (has_cpy) {
            if (q <= p || nodes[q].opcode != HTP_OP_CPY || !nodes[q].fused.empty() ||
                nodes[q].outputs.size() != 1 || (nodes[q].node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }
            if (nodes[c].fused.size() + 2 > HTP_OP_MAX_FUSED) {
                continue;
            }
            const ggml_tensor * cpy_src = nodes[q].src0();
            cpy_dst = nodes[q].dst();
            const size_t conv_x_es = ggml_type_size(conv_x->type);
            if (cpy_src->op != GGML_OP_VIEW ||
                !htp_same_tensor_data(cpy_src->view_src, conv_x) ||
                cpy_src->ne[0] != S || cpy_src->ne[1] != d_inner || cpy_src->ne[2] != n_s || cpy_src->ne[3] != 1 ||
                cpy_src->nb[0] != conv_x_es || cpy_src->nb[1] != conv_x->nb[1] ||
                (n_s > 1 && cpy_src->nb[2] != conv_x->nb[2]) ||
                cpy_src->view_offs != (size_t) n_t * conv_x_es) {
                continue;
            }
        }

        const bool inplace_x_dst = htp_same_tensor_data(conv_dst, cat_x) &&
                                   conv_dst->type == cat_x->type &&
                                   ggml_are_same_shape(conv_dst, cat_x) &&
                                   ggml_are_same_stride(conv_dst, cat_x);
        if (htp_tensors_overlap(conv_dst, cat_state) ||
            (htp_tensors_overlap(conv_dst, cat_x) && !inplace_x_dst) ||
            htp_tensors_overlap(conv_dst, conv_weight)) {
            continue;
        }
        if (has_cpy) {
            if (htp_tensors_overlap(conv_dst, cpy_dst) ||
                htp_tensors_overlap(cpy_dst, cat_x) ||
                htp_tensors_overlap(cpy_dst, conv_weight)) {
                continue;
            }
            if (htp_tensors_overlap(cpy_dst, cat_state)) {
                const bool inplace_state = (n_s == 1) &&
                                           htp_same_tensor_data(cpy_dst, cat_state) &&
                                           cpy_dst->type == cat_state->type;
                if (!inplace_state) {
                    continue;
                }
            }
        }

        bool safe = true;
        for (size_t k = p + 1; k < c && safe; ++k) {
            if (has_cpy && k == q) {
                continue;
            }
            for (const ggml_tensor * out : nodes[k].outputs) {
                if (!out) {
                    continue;
                }
                if (htp_tensors_overlap(out, cat_state) ||
                    htp_tensors_overlap(out, cat_x) ||
                    htp_tensors_overlap(out, conv_x) ||
                    htp_tensors_overlap(out, conv_weight) ||
                    (has_cpy && htp_tensors_overlap(out, cpy_dst))) {
                    safe = false;
                    break;
                }
            }
        }
        if (has_cpy && safe) {
            const size_t lo = (q < c) ? (q + 1) : (c + 1);
            const size_t hi = (q < c) ? c       : q;
            for (size_t k = lo; k < hi && safe; ++k) {
                for (const ggml_tensor * in : nodes[k].inputs) {
                    if (htp_tensor_reads_from(in, cpy_dst)) {
                        safe = false;
                        break;
                    }
                }
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, cpy_dst) || (q > c && htp_tensors_overlap(out, conv_x)))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) {
            continue;
        }

        htp_opnode cand = nodes[c];
        cand.inputs = { cat_state, conv_weight, cat_x };
        if (has_cpy) {
            cand.outputs = { conv_dst, cpy_dst };
            cand.fused.insert(cand.fused.begin(), nodes[q].node);
        }
        cand.fused.insert(cand.fused.begin(), nodes[p].node);
        if (!htp_kparams_solve(sess, cand)) {
            continue;
        }

        nodes[c] = std::move(cand);
        if (has_cpy) {
            nodes.erase(nodes.begin() + (ptrdiff_t) q);
            if (q < c) {
                --c;
            }
        }
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

static void htp_fuse_mul_mat_nx_glu(const struct ggml_hexagon_session * sess, std::vector<struct htp_opnode> & nodes) {
    for (size_t c = 0; c < nodes.size(); ++c) {
        if (!htp_is_fuseable_mmnx_glu_op(nodes[c].opcode) || nodes[c].inputs.size() != 2 ||
            nodes[c].outputs.size() != 1 || !nodes[c].src1()) {
            continue;
        }

        const ggml_tensor * s0 = nodes[c].src0();
        const ggml_tensor * s1 = nodes[c].src1();

        size_t p = c;
        for (size_t i = 0; i < c; ++i) {
            if (nodes[i].opcode != HTP_OP_MUL_MAT_NX || nodes[i].outputs.size() != 2 || nodes[i].inputs.size() != 3) {
                continue;
            }
            const ggml_tensor * o0 = nodes[i].outputs[0];
            const ggml_tensor * o1 = nodes[i].outputs[1];
            if ((htp_same_tensor_data(s0, o0) && htp_same_tensor_data(s1, o1)) ||
                (htp_same_tensor_data(s0, o1) && htp_same_tensor_data(s1, o0))) {
                p = i;
                break;
            }
        }
        if (p == c) {
            continue;
        }

        const ggml_tensor * w0  = nodes[p].inputs[0];
        const ggml_tensor * w1  = nodes[p].inputs[1];
        const ggml_tensor * act = nodes[p].inputs[2];
        const ggml_tensor * o0  = nodes[p].outputs[0];
        const ggml_tensor * o1  = nodes[p].outputs[1];

        bool safe = true;
        for (size_t k = 0; k < nodes.size() && safe; ++k) {
            if (k == p || k == c) {
                continue;
            }
            for (const ggml_tensor * in : nodes[k].inputs) {
                if (htp_tensor_reads_from(in, o0) || htp_tensor_reads_from(in, o1)) {
                    safe = false;
                    break;
                }
            }
            if (safe && k > p && k < c) {
                for (const ggml_tensor * out : nodes[k].outputs) {
                    if (out && (htp_tensors_overlap(out, w0) || htp_tensors_overlap(out, w1) ||
                                htp_tensors_overlap(out, act) || htp_tensors_overlap(out, o0) ||
                                htp_tensors_overlap(out, o1))) {
                        safe = false;
                        break;
                    }
                }
            }
        }
        if (!safe) {
            continue;
        }

        htp_opnode cand = nodes[p];
        if (!fuse_mul_mat_nx_glu(sess, cand, nodes[c])) {
            continue;
        }

        nodes[c] = std::move(cand);
        nodes.erase(nodes.begin() + (ptrdiff_t) p);
        --c;
    }
}

void htp_fusion_optimize_nodes(const struct ggml_hexagon_session * sess, const struct ggml_cgraph * graph, std::vector<struct htp_opnode> & nodes) {
    GGML_UNUSED(graph);

    if (htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX_GLU)) {
        htp_fuse_mul_mat_nx_glu(sess, nodes);
    }
    if (htp_fusion_enabled(HTP_FUSE_GDN_STATE)) {
        htp_fuse_gdn_state(sess, nodes);
    }
    if (htp_fusion_enabled(HTP_FUSE_CONCAT_SSM_CONV)) {
        htp_fuse_concat_ssm_conv(sess, nodes);
    }
}
