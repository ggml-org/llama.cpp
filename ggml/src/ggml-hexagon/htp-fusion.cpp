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
    if (!htp_tensor_is_fuseable(prev_out)) {
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
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }
    return htp_tensor_is_tiled(src0) && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW);
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    if (kparams->n_hmx) {
        return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }
    return htp_tensor_is_tiled(src0) &&
           (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
}

// --- Fusion rules ---

static bool fuse_allreduce_add(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    if (opt_ar_select != 2 || prev.opcode != HTP_OP_ALLREDUCE) return false;

    const auto * ar_kparams = prev.kparams<htp_allreduce_kernel_params>();
    const uint32_t rank    = (uint32_t) ar_kparams->rank;
    const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;

    const ggml_tensor * ar_local = prev.inputs[rank];
    if (!htp_tensor_is_fuseable(ar_local)) return false;

    const ggml_tensor * add_dst    = cur.dst();
    const ggml_tensor * res_tensor = nullptr;
    if (htp_same_tensor_data(cur.src0(), ar_local)) {
        res_tensor = cur.src1();
    } else if (htp_same_tensor_data(cur.src1(), ar_local)) {
        res_tensor = cur.src0();
    } else {
        return false;
    }

    if (ar_local->type != res_tensor->type) return false;

    const bool is_same_shape = ggml_are_same_shape(ar_local, res_tensor);
    const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] &&
                                                  res_tensor->ne[1] == 1 && res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);
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
    if (src0->type != GGML_TYPE_F32 || src0->ne[0] != weight->ne[0] || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
    if (htp_inputs_overlap_unsafe(prev, dst, weight)) return false;

    const bool is_row_bcast  = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
    const bool is_same_shape = ggml_are_same_shape(src0, weight);
    if (!is_row_bcast && !is_same_shape) return false;

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

static bool fuse_rms_norm_scale(const ggml_hexagon_session * sess, htp_opnode & prev, const htp_opnode & cur) {
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_RMS_NORM && prev.opcode != HTP_OP_RMS_NORM_MUL) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (src0->type != GGML_TYPE_F32 || !htp_same_shape_and_contiguity(prev.dst(), dst)) return false;
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
    if (src2->type != GGML_TYPE_F32) return false;

    const ggml_tensor * src0 = prev.src0();
    const ggml_tensor * src1 = prev.src1();
    const ggml_tensor * dst  = cur.dst();
    if (!ggml_are_same_shape(prev.dst(), dst)) return false;

    const auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    struct htp_mm_kernel_params kparams;
    if (!htp_kparams_solve_matmul_add(sess, src0, src1, src2, dst, &kparams) ||
        orig_kparams->collapse != kparams.collapse) {
        return false;
    }

    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (kparams.n_hmx == 0 && src1_nrows != 1) return false;

    if (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
        if (kparams.m_chunk < orig_kparams->m_chunk ||
            kparams.n_chunk < orig_kparams->n_chunk ||
            kparams.n_act_threads < orig_kparams->n_act_threads) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                        htp_session_name(sess), orig_kparams->m_chunk, kparams.m_chunk,
                        orig_kparams->n_chunk, kparams.n_chunk,
                        orig_kparams->n_act_threads, kparams.n_act_threads);
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

    const ggml_tensor * w0   = prev.inputs[0];
    const ggml_tensor * x    = prev.inputs[curr_n];
    const ggml_tensor * w_in = cur.src0();
    const ggml_tensor * x_in = cur.src1();
    const ggml_tensor * d_in = cur.dst();

    auto * orig_kparams = prev.kparams<htp_mm_kernel_params>();
    const auto * cur_kparams = cur.kparams<htp_mm_kernel_params>();

    if (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
        w0->ne[2] != 1 || w0->ne[3] != 1 || w_in->ne[2] != 1 || w_in->ne[3] != 1 ||
        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
        orig_kparams->n_hmx != cur_kparams->n_hmx) {
        return false;
    }

    const bool collapse = orig_kparams->collapse && htp_matmul_can_collapse(w_in, x, d_in);
    if ((x->ne[2] > 1 || x->ne[3] > 1) && !collapse) return false;

    if (prev.opcode == HTP_OP_MUL_MAT_NX) {
        orig_kparams->n_weights = curr_n + 1;
    } else {
        struct htp_mm_kernel_params kparams;
        if (collapse) {
            const ggml_tensor x_collapsed = htp_tensor_collapse_rows(x);
            if (!htp_kparams_solve_mmnx(sess, w0, &x_collapsed, curr_n + 1, &kparams)) return false;
            kparams.collapse = 1;
        } else if (!htp_kparams_solve_mmnx(sess, w0, x, curr_n + 1, &kparams)) {
            return false;
        }
        if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) return false;

        prev.set_kparams(kparams);
        prev.opcode = HTP_OP_MUL_MAT_NX;
    }

    prev.inputs.insert(prev.inputs.begin() + curr_n, w_in);
    prev.outputs.push_back(d_in);
    prev.fused.push_back(cur.node);
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
        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
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
    GGML_UNUSED(sess);
    if (prev.opcode != HTP_OP_SSM_CONV) return false;

    auto * kparams = prev.kparams<htp_ssm_conv_kernel_params>();
    if (kparams->silu) return false;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ false)) return false;

    const ggml_tensor * conv_dst = prev.dst();
    const ggml_tensor * dst      = cur.dst();
    if (!ggml_are_same_layout(conv_dst, dst)) return false;
    if (htp_tensors_overlap(dst, prev.src0()) || htp_tensors_overlap(dst, prev.src1())) return false;

    kparams->silu   = 1;
    prev.outputs[0] = dst;
    prev.fused.push_back(cur.node);
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

    const ggml_tensor * up = nullptr;
    if (!htp_match_consumer(prev, cur, /*commutative=*/ true, &up)) return false;

    const ggml_tensor * gate = prev.src0();
    const ggml_tensor * dst  = cur.dst();
    if (htp_inputs_overlap_unsafe(prev, dst, up)) return false;

    struct htp_glu_kernel_params kparams;
    if (!htp_kparams_solve_unary_mul_glu(sess, glu_op, gate, up, dst, &kparams)) return false;

    prev.opcode  = glu_op;
    prev.inputs  = { gate, up };
    prev.outputs = { dst };
    prev.fused.push_back(cur.node);
    memset(prev.params, 0, sizeof(prev.params));
    prev.set_kparams(kparams);
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
        const uint32_t max_threads = htp_session_n_threads(sess);
        if (!htp_binary_solve_chunked(&L, &kparams, total_elems, elem_size, max_threads, n_srcs, false, vtcm_size)) {
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
            return ggml_is_quantized(a->src[0]->type);
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
        auto * extra = (htp_tensor_extra *) graph->nodes[i]->extra;
        if (!extra) continue;

        extra->flags &= ~HTP_TENSOR_FUSEABLE;

        const ggml_op op = graph->nodes[i]->op;
        if ((op == GGML_OP_RMS_NORM || op == GGML_OP_SCALE || op == GGML_OP_REPEAT ||
             op == GGML_OP_UNARY    || op == GGML_OP_ADD   || op == GGML_OP_SUB    ||
             op == GGML_OP_MUL      || op == GGML_OP_DIV   || op == GGML_OP_MUL_MAT_ID ||
             op == GGML_OP_SSM_CONV ||
             (op == GGML_OP_MUL_MAT && !htp_is_hadamard(graph->nodes[i]))) &&
            ggml_node_has_n_uses(graph, i, 1)) {
            extra->flags |= HTP_TENSOR_FUSEABLE;
        }
    }
}

bool htp_fusion_try_fuse(const struct ggml_hexagon_session * sess, struct htp_opnode & prev, const struct htp_opnode & cur) {
    if (!opt_opfusion) return false;

    switch (cur.opcode) {
        case HTP_OP_ADD:
            return (htp_fusion_enabled(HTP_FUSE_ALLREDUCE_ADD) && fuse_allreduce_add(sess, prev, cur)) ||
                   (htp_fusion_enabled(HTP_FUSE_MUL_MAT_ADD)   && fuse_mul_mat_add(sess, prev, cur))   ||
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
            return htp_fusion_enabled(HTP_FUSE_RMS_NORM_SCALE) && fuse_rms_norm_scale(sess, prev, cur);
        case HTP_OP_UNARY_SILU:
            return htp_fusion_enabled(HTP_FUSE_SSM_CONV_SILU) && fuse_ssm_conv_silu(sess, prev, cur);
        case HTP_OP_GET_ROWS:
            return htp_fusion_enabled(HTP_FUSE_REPEAT_GET_ROWS) && fuse_repeat_get_rows(sess, prev, cur);
        case HTP_OP_CPY:
            return htp_fusion_enabled(HTP_FUSE_GDN_CPY) && fuse_gdn_cpy(sess, prev, cur);
        case HTP_OP_MUL_MAT:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_NX) && fuse_mul_mat_nx(sess, prev, cur);
        case HTP_OP_MUL_MAT_ID:
            return htp_fusion_enabled(HTP_FUSE_MUL_MAT_ID_NX) && fuse_mul_mat_id_nx(sess, prev, cur);
        default:
            return false;
    }
}
