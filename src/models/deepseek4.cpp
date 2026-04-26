#include "models.h"

#include "llama-memory-deepseek4.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

namespace {

static bool deepseek4_is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static void deepseek4_fill_hadamard(std::vector<float> & data, int64_t n) {
    GGML_ASSERT(deepseek4_is_power_of_2(n));

    data.assign(n*n, 0.0f);
    data[0] = 1.0f / std::sqrt(float(n));

    for (int64_t s = 1; s < n; s *= 2) {
        for (int64_t i = 0; i < s; ++i) {
            for (int64_t j = 0; j < s; ++j) {
                const float v = data[i*n + j];
                data[(i + s)*n + j    ] =  v;
                data[i*n       + j + s] =  v;
                data[(i + s)*n + j + s] = -v;
            }
        }
    }
}

class llm_build_deepseek4_inputs : public llm_graph_input_i {
public:
    explicit llm_build_deepseek4_inputs(uint32_t n_swa) : n_swa(n_swa) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->n_tokens >= 1);
        const int32_t pos = ubatch->pos ? ubatch->pos[0] : 0;

        if (attn_cache_idx && attn_cache_idx->buffer) {
            const int32_t cache_idx = pos % (int32_t) n_swa;
            ggml_backend_tensor_set(attn_cache_idx, &cache_idx, 0, sizeof(cache_idx));
        }

        if (comp_pos_r4 && comp_pos_r4->buffer) {
            const int32_t pos_r4 = std::max<int32_t>(0, pos + 1 - 4);
            ggml_backend_tensor_set(comp_pos_r4, &pos_r4, 0, sizeof(pos_r4));
        }

        if (comp_pos_r128 && comp_pos_r128->buffer) {
            const int32_t pos_r128 = std::max<int32_t>(0, pos + 1 - 128);
            ggml_backend_tensor_set(comp_pos_r128, &pos_r128, 0, sizeof(pos_r128));
        }

        if (comp_cache_idx_r4 && comp_cache_idx_r4->buffer) {
            const int32_t comp_cache_idx = n_swa + pos / 4;
            ggml_backend_tensor_set(comp_cache_idx_r4, &comp_cache_idx, 0, sizeof(comp_cache_idx));
        }

        if (indexer_cache_idx_r4 && indexer_cache_idx_r4->buffer) {
            const int32_t indexer_cache_idx = pos / 4;
            ggml_backend_tensor_set(indexer_cache_idx_r4, &indexer_cache_idx, 0, sizeof(indexer_cache_idx));
        }

        if (comp_cache_idx_r128 && comp_cache_idx_r128->buffer) {
            const int32_t comp_cache_idx = n_swa + pos / 128;
            ggml_backend_tensor_set(comp_cache_idx_r128, &comp_cache_idx, 0, sizeof(comp_cache_idx));
        }

        if (comp_slot_idx_r4 && comp_slot_idx_r4->buffer) {
            const int32_t comp_slot_idx = 4 + (pos % 4);
            ggml_backend_tensor_set(comp_slot_idx_r4, &comp_slot_idx, 0, sizeof(comp_slot_idx));
        }

        if (comp_slot_idx_r128 && comp_slot_idx_r128->buffer) {
            const int32_t comp_slot_idx = pos % 128;
            ggml_backend_tensor_set(comp_slot_idx_r128, &comp_slot_idx, 0, sizeof(comp_slot_idx));
        }

        if (indexer_hadamard && indexer_hadamard->buffer) {
            const int64_t n = indexer_hadamard->ne[0];
            GGML_ASSERT(indexer_hadamard->ne[1] == n);
            if (indexer_hadamard_data.empty()) {
                deepseek4_fill_hadamard(indexer_hadamard_data, n);
            }
            ggml_backend_tensor_set(indexer_hadamard, indexer_hadamard_data.data(), 0, ggml_nbytes(indexer_hadamard));
        }
    }

    ggml_tensor * attn_cache_idx = nullptr;
    ggml_tensor * comp_pos_r4 = nullptr;
    ggml_tensor * comp_pos_r128 = nullptr;
    ggml_tensor * comp_cache_idx_r4 = nullptr;
    ggml_tensor * comp_cache_idx_r128 = nullptr;
    ggml_tensor * indexer_cache_idx_r4 = nullptr;
    ggml_tensor * comp_slot_idx_r4 = nullptr;
    ggml_tensor * comp_slot_idx_r128 = nullptr;
    ggml_tensor * indexer_hadamard = nullptr;

    std::vector<float> indexer_hadamard_data;

    const uint32_t n_swa;
};

} // namespace

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(model.arch == LLM_ARCH_DEEPSEEK4);
    GGML_ASSERT(n_tokens >= 1);

    const auto * mctx_cur = dynamic_cast<const llama_memory_deepseek4_context *>(mctx);
    GGML_ASSERT(mctx_cur != nullptr);
    GGML_ASSERT(hparams.n_swa > 0);

    const bool reserve_only = n_tokens != 1;
    const llama_pos start_pos = reserve_only ? 0 : ubatch.pos[0];
    const int64_t work_tokens = reserve_only ? 1 : n_tokens;
    GGML_ASSERT(start_pos >= 0);
    GGML_ASSERT((uint32_t) start_pos < mctx_cur->get_n_ctx_seq());

    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t rope_dim = hparams.n_rot();
    const int64_t nope_dim = head_dim - rope_dim;
    const int64_t total_q_dim = head_dim * n_head;
    const int64_t hc_mult = model.hc_head_base ? model.hc_head_base->ne[0] : 0;
    GGML_ASSERT(hc_mult > 0);
    GGML_ASSERT(nope_dim >= 0);

    auto inp_ds4 = std::make_unique<llm_build_deepseek4_inputs>(hparams.n_swa);
    inp_ds4->attn_cache_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->attn_cache_idx);
    ggml_set_name(inp_ds4->attn_cache_idx, "deepseek4_attn_cache_idx");
    inp_ds4->comp_pos_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_pos_r4);
    ggml_set_name(inp_ds4->comp_pos_r4, "deepseek4_comp_pos_r4");
    inp_ds4->comp_pos_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_pos_r128);
    ggml_set_name(inp_ds4->comp_pos_r128, "deepseek4_comp_pos_r128");
    inp_ds4->comp_cache_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_cache_idx_r4);
    ggml_set_name(inp_ds4->comp_cache_idx_r4, "deepseek4_comp_cache_idx_r4");
    inp_ds4->comp_cache_idx_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_cache_idx_r128);
    ggml_set_name(inp_ds4->comp_cache_idx_r128, "deepseek4_comp_cache_idx_r128");
    inp_ds4->indexer_cache_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->indexer_cache_idx_r4);
    ggml_set_name(inp_ds4->indexer_cache_idx_r4, "deepseek4_indexer_cache_idx_r4");
    inp_ds4->comp_slot_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_slot_idx_r4);
    ggml_set_name(inp_ds4->comp_slot_idx_r4, "deepseek4_comp_slot_idx_r4");
    inp_ds4->comp_slot_idx_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_ds4->comp_slot_idx_r128);
    ggml_set_name(inp_ds4->comp_slot_idx_r128, "deepseek4_comp_slot_idx_r128");
    if (hparams.indexer_head_size > 0 &&
            hparams.indexer_top_k > 0 &&
            uint64_t(cparams.n_ctx_seq) > uint64_t(hparams.indexer_top_k) * 4u) {
        inp_ds4->indexer_hadamard = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.indexer_head_size, hparams.indexer_head_size);
        ggml_set_input(inp_ds4->indexer_hadamard);
        ggml_set_name(inp_ds4->indexer_hadamard, "deepseek4_indexer_hadamard");
    }
    auto * deepseek4_inputs = static_cast<llm_build_deepseek4_inputs *>(res->add_input(std::move(inp_ds4)));

    auto scalar_view = [&](ggml_tensor * tensor, int64_t idx) -> ggml_tensor * {
        return ggml_view_1d(ctx0, tensor, 1, idx * tensor->nb[0]);
    };

    auto vector_slice = [&](ggml_tensor * tensor, int64_t offset, int64_t len) -> ggml_tensor * {
        return ggml_view_1d(ctx0, tensor, len, offset * tensor->nb[0]);
    };

    auto matrix_slice = [&](ggml_tensor * tensor, int64_t offset, int64_t rows, int64_t cols) -> ggml_tensor * {
        return ggml_view_2d(ctx0, tensor, rows, cols, rows * tensor->nb[0], offset * tensor->nb[0]);
    };

    auto matrix_block = [&](ggml_tensor * tensor, int64_t row_offset, int64_t col_offset, int64_t rows, int64_t cols) -> ggml_tensor * {
        return ggml_view_2d(ctx0, tensor, rows, cols, tensor->nb[1], row_offset * tensor->nb[0] + col_offset * tensor->nb[1]);
    };

    auto reshape_3d_checked = [&](ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, const char * tag, int il = -1) -> ggml_tensor * {
        const int64_t expected = ne0 * ne1 * ne2;
        if (ggml_nelements(tensor) != expected) {
            GGML_ABORT(
                    "deepseek4: reshape_3d mismatch in %s layer %d pos %d"
                    " ne=%" PRId64 " expected=%" PRId64 " target=(%" PRId64 ",%" PRId64 ",%" PRId64 ") tensor=%s",
                    tag, il, (int) start_pos, ggml_nelements(tensor), expected, ne0, ne1, ne2,
                    tensor->name[0] ? tensor->name : "<unnamed>");
        }
        return ggml_reshape_3d(ctx0, tensor, ne0, ne1, ne2);
    };

    auto add_eps = [&](ggml_tensor * tensor, float eps) -> ggml_tensor * {
        return ggml_clamp(ctx0, tensor, eps, INFINITY);
    };

    auto mul_mat_checked = [&](ggml_tensor * a, ggml_tensor * b, const char * tag) -> ggml_tensor * {
        if (ggml_is_transposed(a)) {
            GGML_ABORT("deepseek4: transposed lhs in %s (%s)", tag, a->name[0] ? a->name : "<unnamed>");
        }
        if (b->nb[0] != ggml_type_size(b->type)) {
            GGML_ABORT(
                "deepseek4: mul_mat rhs layout in %s (%s) nb0=%zu nb1=%zu",
                tag, b->name[0] ? b->name : "<unnamed>", b->nb[0], b->nb[1]);
        }
        return ggml_mul_mat(ctx0, a, b);
    };

    auto repeat_checked = [&](ggml_tensor * src, ggml_tensor * dst, const char * tag) -> ggml_tensor * {
        if (src->nb[0] != sizeof(float)) {
            GGML_ABORT(
                "deepseek4: repeat source layout in %s (%s) nb0=%zu nb1=%zu",
                tag, src->name[0] ? src->name : "<unnamed>", src->nb[0], src->nb[1]);
        }
        if (dst->nb[0] != sizeof(float)) {
            GGML_ABORT(
                "deepseek4: repeat destination layout in %s (%s) nb0=%zu nb1=%zu",
                tag, dst->name[0] ? dst->name : "<unnamed>", dst->nb[0], dst->nb[1]);
        }
        return ggml_repeat(ctx0, src, dst);
    };

    auto sum_rows_checked = [&](ggml_tensor * src, const char * tag) -> ggml_tensor * {
        if (src->nb[0] != sizeof(float)) {
            GGML_ABORT(
                "deepseek4: sum_rows source layout in %s (%s) nb0=%zu nb1=%zu",
                tag, src->name[0] ? src->name : "<unnamed>", src->nb[0], src->nb[1]);
        }
        return ggml_sum_rows(ctx0, src);
    };

    auto affine = [&](ggml_tensor * tensor, ggml_tensor * scale, ggml_tensor * bias) -> ggml_tensor * {
        ggml_tensor * scale_r = repeat_checked(scale, tensor, "affine.scale");
        ggml_tensor * out = ggml_mul(ctx0, tensor, scale_r);
        return ggml_add(ctx0, out, bias);
    };

    auto weighted_sum_hc = [&](ggml_tensor * x_hc, ggml_tensor * weights) -> ggml_tensor * {
        ggml_tensor * x_mat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, x_hc, n_embd, hc_mult));
        ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x_mat));
        return mul_mat_checked(x_t, weights, "weighted_sum_hc");
    };

    auto sinkhorn = [&](ggml_tensor * comb) -> ggml_tensor * {
        if (comb->type == GGML_TYPE_F32 &&
                comb->ne[0] == 4 && comb->ne[1] == 4 && comb->ne[2] == 1 && comb->ne[3] == 1) {
            return ggml_sinkhorn_4x4(ctx0, comb);
        }

        comb = ggml_soft_max(ctx0, comb);
        comb = add_eps(comb, 1e-6f);

        ggml_tensor * col_sum = sum_rows_checked(ggml_cont(ctx0, ggml_transpose(ctx0, comb)), "sinkhorn.col_sum");
        col_sum = add_eps(col_sum, 1e-6f);
        comb = ggml_div(ctx0, comb, repeat_checked(ggml_cont(ctx0, ggml_transpose(ctx0, col_sum)), comb, "sinkhorn.col_sum"));

        for (int i = 1; i < 20; ++i) {
            ggml_tensor * row_sum = sum_rows_checked(comb, "sinkhorn.row_sum");
            row_sum = add_eps(row_sum, 1e-6f);
            comb = ggml_div(ctx0, comb, repeat_checked(row_sum, comb, "sinkhorn.row_sum"));

            col_sum = sum_rows_checked(ggml_cont(ctx0, ggml_transpose(ctx0, comb)), "sinkhorn.col_sum_iter");
            col_sum = add_eps(col_sum, 1e-6f);
            comb = ggml_div(ctx0, comb, repeat_checked(ggml_cont(ctx0, ggml_transpose(ctx0, col_sum)), comb, "sinkhorn.col_sum_iter"));
        }

        return comb;
    };

    auto hc_pre = [&](ggml_tensor * x_hc, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base, int il) {
        ggml_tensor * x_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, x_hc, n_embd * hc_mult, work_tokens));
        ggml_tensor * x_norm = ggml_rms_norm(ctx0, x_flat, hparams.f_norm_rms_eps);
        cb(x_norm, "hc_norm", il);

        ggml_tensor * mixes = mul_mat_checked(hc_fn, x_norm, "hc_pre.mixes");
        cb(mixes, "hc_mixes", il);

        ggml_tensor * pre = vector_slice(mixes, 0, hc_mult);
        ggml_tensor * post = vector_slice(mixes, hc_mult, hc_mult);
        ggml_tensor * comb = matrix_slice(mixes, 2 * hc_mult, hc_mult, hc_mult);

        pre = affine(pre, scalar_view(hc_scale, 0), vector_slice(hc_base, 0, hc_mult));
        pre = ggml_sigmoid(ctx0, pre);
        pre = add_eps(pre, 1e-6f);
        cb(pre, "hc_pre", il);

        post = affine(post, scalar_view(hc_scale, 1), vector_slice(hc_base, hc_mult, hc_mult));
        post = ggml_sigmoid(ctx0, post);
        post = ggml_scale(ctx0, post, 2.0f);
        cb(post, "hc_post_w", il);

        comb = affine(comb, scalar_view(hc_scale, 2), matrix_slice(hc_base, 2 * hc_mult, hc_mult, hc_mult));
        comb = sinkhorn(comb);
        cb(comb, "hc_comb", il);

        ggml_tensor * y = weighted_sum_hc(x_hc, pre);
        cb(y, "hc_reduce", il);

        return std::make_tuple(y, post, comb);
    };

    auto hc_post = [&](ggml_tensor * x_single, ggml_tensor * residual_hc, ggml_tensor * post, ggml_tensor * comb, int il) -> ggml_tensor * {
        ggml_tensor * residual = ggml_cont(ctx0, ggml_reshape_2d(ctx0, residual_hc, n_embd, hc_mult));
        ggml_tensor * residual_t = ggml_cont(ctx0, ggml_transpose(ctx0, residual));
        ggml_tensor * mixed_t = mul_mat_checked(comb, residual_t, "hc_post.mixed");
        ggml_tensor * mixed = ggml_cont(ctx0, ggml_transpose(ctx0, mixed_t));

        ggml_tensor * x_repeat = repeat_checked(x_single, residual, "hc_post.x");
        ggml_tensor * post_repeat = repeat_checked(ggml_cont(ctx0, ggml_transpose(ctx0, post)), residual, "hc_post.post");

        ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, x_repeat, post_repeat), mixed);
        cb(out, "hc_expand", il);

        return reshape_3d_checked(out, n_embd, hc_mult, work_tokens, "hc_post.out", il);
    };

    auto hc_head = [&](ggml_tensor * x_hc, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base) -> ggml_tensor * {
        ggml_tensor * x_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, x_hc, n_embd * hc_mult, work_tokens));
        ggml_tensor * x_norm = ggml_rms_norm(ctx0, x_flat, hparams.f_norm_rms_eps);
        ggml_tensor * mixes = mul_mat_checked(hc_fn, x_norm, "hc_head.mixes");
        ggml_tensor * pre = affine(mixes, scalar_view(hc_scale, 0), hc_base);
        pre = ggml_sigmoid(ctx0, pre);
        pre = add_eps(pre, 1e-6f);
        return weighted_sum_hc(x_hc, pre);
    };

    auto build_grouped_out = [&](ggml_tensor * attn_out, const llama_layer & layer, int il) -> ggml_tensor * {
        const int64_t group_dim = layer.attn_out_a->ne[0];
        const int64_t n_groups = total_q_dim / group_dim;
        const int64_t o_rank = layer.attn_out_b->ne[0] / n_groups;

        GGML_ASSERT(group_dim > 0);
        GGML_ASSERT(n_groups > 0);
        GGML_ASSERT(layer.attn_out_b->ne[0] == n_groups * o_rank);

        ggml_tensor * grouped = nullptr;
        for (int64_t g = 0; g < n_groups; ++g) {
            ggml_tensor * xg = ggml_view_2d(ctx0, attn_out, group_dim, work_tokens, attn_out->nb[1], g * group_dim * attn_out->nb[0]);
            ggml_tensor * wg = ggml_view_2d(ctx0, layer.attn_out_a, group_dim, o_rank, layer.attn_out_a->nb[1], g * o_rank * layer.attn_out_a->nb[1]);
            ggml_tensor * og = mul_mat_checked(wg, xg, "build_grouped_out.group");
            cb(og, "attn_group_out", il);
            grouped = grouped ? ggml_concat(ctx0, grouped, og, 0) : og;
        }

        ggml_tensor * out = mul_mat_checked(layer.attn_out_b, grouped, "build_grouped_out.out");
        cb(out, "attn_out_proj", il);
        return out;
    };

    auto build_expert_mix = [&](ggml_tensor * cur_ffn, ggml_tensor * selected_experts, ggml_tensor * weights, const llama_layer & layer, int il) -> ggml_tensor * {
        ggml_tensor * cur_experts_in = reshape_3d_checked(cur_ffn, n_embd, 1, work_tokens, "build_expert_mix.cur_ffn", il);
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;

        if (layer.ffn_gate_up_exps) {
            ggml_tensor * gate_up = build_lora_mm_id(layer.ffn_gate_up_exps, cur_experts_in, selected_experts);
            cb(gate_up, "ffn_moe_gate_up", il);

            const int64_t n_ff = gate_up->ne[0] / 2;
            gate = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            up = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        } else {
            gate = build_lora_mm_id(layer.ffn_gate_exps, cur_experts_in, selected_experts);
            up = build_lora_mm_id(layer.ffn_up_exps, cur_experts_in, selected_experts);
            cb(gate, "ffn_moe_gate", il);
            cb(up, "ffn_moe_up", il);
        }

        const float swiglu_limit = hparams.swiglu_clamp_exp[il];
        if (swiglu_limit > 1e-6f) {
            gate = ggml_clamp(ctx0, gate, -INFINITY, swiglu_limit);
            up   = ggml_clamp(ctx0, up,   -swiglu_limit, swiglu_limit);
            cb(gate, "ffn_moe_gate_clamped", il);
            cb(up,   "ffn_moe_up_clamped",   il);
        }

        ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
        cb(act, "ffn_moe_swiglu", il);

        ggml_tensor * experts = build_lora_mm_id(layer.ffn_down_exps, act, selected_experts);
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_down", il);

        ggml_tensor * views[LLAMA_MAX_EXPERTS] = { nullptr };
        for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
            views[i] = ggml_view_2d(ctx0, experts, n_embd, work_tokens, experts->nb[2], i * experts->nb[1]);
        }

        ggml_tensor * out = views[0];
        for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
            out = ggml_add(ctx0, out, views[i]);
        }

        cb(out, "ffn_moe_out", il);
        return out;
    };

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_tokens = res->get_inp_tokens();
    if (reserve_only) {
        inpL = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, 1, inpL->nb[1], 0));
        inp_pos = ggml_view_1d(ctx0, inp_pos, 1, 0);
        if (inp_tokens) {
            inp_tokens = ggml_view_1d(ctx0, inp_tokens, 1, 0);
        }
    }
    GGML_UNUSED(inp_tokens);

    auto build_moe_v4 = [&](ggml_tensor * cur_ffn, const llama_layer & layer, int il) -> ggml_tensor * {
        ggml_tensor * scores = build_lora_mm(layer.ffn_gate_inp, cur_ffn);
        scores = ggml_softplus(ctx0, scores);
        scores = ggml_sqrt(ctx0, scores);
        cb(scores, "ffn_scores", il);

        ggml_tensor * selection = scores;
        if (layer.ffn_gate_tid2eid) {
            ggml_tensor * hash_selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            ggml_tensor * score3d = reshape_3d_checked(scores, 1, n_expert, work_tokens, "build_moe_v4.scores_hash", il);
            ggml_tensor * selected_scores = ggml_get_rows(ctx0, score3d, hash_selected);
            selection = ggml_set_rows(ctx0, ggml_fill(ctx0, score3d, -INFINITY), selected_scores, hash_selected);
            selection = ggml_reshape_2d(ctx0, selection, n_expert, work_tokens);
            cb(selection, "ffn_hash_scores", il);
        } else if (layer.ffn_exp_probs_b) {
            selection = ggml_add(ctx0, scores, layer.ffn_exp_probs_b);
            cb(selection, "ffn_biased_scores", il);
        }

        ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection, n_expert_used);
        cb(selected_experts, "ffn_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0, reshape_3d_checked(scores, 1, n_expert, work_tokens, "build_moe_v4.scores", il), selected_experts);
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, work_tokens);
        ggml_tensor * weights_sum = sum_rows_checked(weights, "build_moe_v4.weights_sum");
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        if (hparams.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        }
        weights = reshape_3d_checked(weights, 1, n_expert_used, work_tokens, "build_moe_v4.weights", il);
        cb(weights, "ffn_weights", il);

        return build_expert_mix(cur_ffn, selected_experts, weights, layer, il);
    };

    auto build_attn_v4 = [&](ggml_tensor * cur_attn, const llama_layer & layer, int il) -> ggml_tensor * {
        const int64_t comp_ratio = layer.attn_compress_ape ? layer.attn_compress_ape->ne[1] : 0;
        const float layer_freq_base = layer.attn_compress_ape ? hparams.rope_freq_base_train_swa : hparams.rope_freq_base_train;
        const float layer_freq_scale = layer.attn_compress_ape ? hparams.rope_freq_scale_train_swa : 1.0f;
        const float layer_ext_factor = layer.attn_compress_ape ? 1.0f : 0.0f;
        const float layer_attn_factor = layer.attn_compress_ape && layer_freq_scale != 1.0f ?
            1.0f / (1.0f + 0.1f * std::log(1.0f / layer_freq_scale)) : 1.0f;
        const float layer_beta_fast = layer.attn_compress_ape ? hparams.yarn_beta_fast : 0.0f;
        const float layer_beta_slow = layer.attn_compress_ape ? hparams.yarn_beta_slow : 0.0f;
        const int32_t layer_n_ctx_orig = layer.attn_compress_ape ? hparams.n_ctx_orig_yarn : 0;

        ggml_tensor * q_base = mul_mat_checked(layer.wq_a, cur_attn, "build_attn_v4.wq_a");
        q_base = build_norm(q_base, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * q = mul_mat_checked(layer.wq_b, q_base, "build_attn_v4.wq_b");
        q = reshape_3d_checked(q, head_dim, n_head, work_tokens, "build_attn_v4.q", il);
        q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);
        cb(q, "q_proj", il);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, nope_dim, n_head, work_tokens, q->nb[1], q->nb[2], 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, rope_dim, n_head, work_tokens, q->nb[1], q->nb[2], nope_dim * q->nb[0]);
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
        ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(q_states, "q_states", il);

        ggml_tensor * kv = mul_mat_checked(layer.attn_kv_latent, cur_attn, "build_attn_v4.kv_latent");
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv = reshape_3d_checked(kv, head_dim, 1, work_tokens, "build_attn_v4.kv", il);
        cb(kv, "kv_latent", il);

        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, nope_dim, 1, work_tokens, kv->nb[1], kv->nb[2], 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv, rope_dim, 1, work_tokens, kv->nb[1], kv->nb[2], nope_dim * kv->nb[0]);
        k_nope = ggml_fp8_act_quant(ctx0, ggml_cont(ctx0, k_nope));
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
        ggml_tensor * k_states = ggml_concat(ctx0, k_nope, k_pe, 0);
        ggml_tensor * k_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, k_states, head_dim, work_tokens));

        const auto & state = mctx_cur->get_layer(il);
        ggml_tensor * updated_cache = ggml_set_rows(ctx0, state.attn_kv, k_flat, deepseek4_inputs->attn_cache_idx);
        ggml_tensor * updated_attn_comp_kv_state = state.attn_comp_kv_state;
        ggml_tensor * updated_attn_comp_score_state = state.attn_comp_score_state;
        ggml_tensor * updated_indexer_kv = state.indexer_kv;
        ggml_tensor * updated_indexer_comp_kv_state = state.indexer_comp_kv_state;
        ggml_tensor * updated_indexer_comp_score_state = state.indexer_comp_score_state;

        if (comp_ratio > 0) {
            GGML_ASSERT(state.attn_comp_kv_state != nullptr);
            GGML_ASSERT(state.attn_comp_score_state != nullptr);

            const int64_t comp_dim = layer.attn_compress_ape->ne[0];
            const int64_t comp_slots = comp_dim / head_dim;
            const bool overlap = comp_slots > 1;
            const bool should_compress = ((start_pos + 1) % comp_ratio) == 0;

            ggml_tensor * comp_kv = mul_mat_checked(layer.attn_compress_kv, cur_attn, "build_attn_v4.comp_kv");
            ggml_tensor * comp_score = mul_mat_checked(layer.attn_compress_gate, cur_attn, "build_attn_v4.comp_score");
            comp_kv = ggml_cont(ctx0, ggml_cast(ctx0, comp_kv, GGML_TYPE_F32));
            comp_score = ggml_cont(ctx0, ggml_cast(ctx0, comp_score, GGML_TYPE_F32));

            ggml_tensor * ape_row = matrix_block(layer.attn_compress_ape, 0, start_pos % comp_ratio, comp_dim, 1);
            comp_score = ggml_cont(ctx0, ggml_add(ctx0, comp_score, ape_row));
            cb(comp_score, "attn_comp_score", il);

            ggml_tensor * comp_slot_idx = nullptr;
            if (comp_ratio == 4) {
                comp_slot_idx = deepseek4_inputs->comp_slot_idx_r4;
            } else if (comp_ratio == 128) {
                comp_slot_idx = deepseek4_inputs->comp_slot_idx_r128;
            } else {
                GGML_ABORT("deepseek4: unsupported compress ratio %" PRId64, comp_ratio);
            }

            updated_attn_comp_kv_state = ggml_set_rows(ctx0, state.attn_comp_kv_state, comp_kv, comp_slot_idx);
            updated_attn_comp_score_state = ggml_set_rows(ctx0, state.attn_comp_score_state, comp_score, comp_slot_idx);

            if (should_compress) {
                ggml_tensor * comp_kv_slots = nullptr;
                ggml_tensor * comp_score_slots = nullptr;

                if (overlap) {
                    ggml_tensor * kv_prev = matrix_block(updated_attn_comp_kv_state, 0, 0, head_dim, comp_ratio);
                    ggml_tensor * kv_cur = matrix_block(updated_attn_comp_kv_state, head_dim, comp_ratio, head_dim, comp_ratio);
                    ggml_tensor * score_prev = matrix_block(updated_attn_comp_score_state, 0, 0, head_dim, comp_ratio);
                    ggml_tensor * score_cur = matrix_block(updated_attn_comp_score_state, head_dim, comp_ratio, head_dim, comp_ratio);

                    comp_kv_slots = ggml_concat(ctx0, kv_prev, kv_cur, 1);
                    comp_score_slots = ggml_concat(ctx0, score_prev, score_cur, 1);

                    ggml_tensor * carry_kv = matrix_block(updated_attn_comp_kv_state, 0, comp_ratio, comp_dim, comp_ratio);
                    ggml_tensor * carry_score = matrix_block(updated_attn_comp_score_state, 0, comp_ratio, comp_dim, comp_ratio);
                    // HF seeds the next overlapping current window with the just-compressed window; new tokens overwrite it slot by slot.
                    updated_attn_comp_kv_state = ggml_concat(ctx0, carry_kv, carry_kv, 1);
                    updated_attn_comp_score_state = ggml_concat(ctx0, carry_score, carry_score, 1);
                } else {
                    comp_kv_slots = updated_attn_comp_kv_state;
                    comp_score_slots = updated_attn_comp_score_state;
                }

                ggml_tensor * comp_kv_seq = ggml_cont(ctx0, ggml_transpose(ctx0, comp_kv_slots));
                ggml_tensor * comp_score_seq = ggml_cont(ctx0, ggml_transpose(ctx0, comp_score_slots));
                ggml_tensor * comp_weights = ggml_soft_max(ctx0, comp_score_seq);
                ggml_tensor * comp_weighted = ggml_mul(ctx0, comp_kv_seq, comp_weights);
                ggml_tensor * comp_flat = sum_rows_checked(comp_weighted, "build_attn_v4.comp_sum");
                comp_flat = ggml_cont(ctx0, ggml_transpose(ctx0, comp_flat));
                comp_flat = build_norm(comp_flat, layer.attn_compress_norm, nullptr, LLM_NORM_RMS, il);
                if (ggml_nelements(comp_flat) != head_dim) {
                    GGML_ABORT(
                            "deepseek4: comp_flat reshape mismatch at layer %d pos %d ratio %" PRId64
                            " ne=%" PRId64 " expected=%" PRId64,
                            il, (int) start_pos, comp_ratio, ggml_nelements(comp_flat), head_dim);
                }

                ggml_tensor * comp_states = reshape_3d_checked(comp_flat, head_dim, 1, 1, "build_attn_v4.comp_states", il);
                ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp_states, nope_dim, 1, 1, comp_states->nb[1], comp_states->nb[2], 0);
                ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp_states, rope_dim, 1, 1, comp_states->nb[1], comp_states->nb[2], nope_dim * comp_states->nb[0]);
                comp_nope = ggml_fp8_act_quant(ctx0, ggml_cont(ctx0, comp_nope));

                ggml_tensor * comp_pos = nullptr;
                ggml_tensor * comp_cache_idx = nullptr;
                if (comp_ratio == 4) {
                    comp_pos = deepseek4_inputs->comp_pos_r4;
                    comp_cache_idx = deepseek4_inputs->comp_cache_idx_r4;
                } else if (comp_ratio == 128) {
                    comp_pos = deepseek4_inputs->comp_pos_r128;
                    comp_cache_idx = deepseek4_inputs->comp_cache_idx_r128;
                } else {
                    GGML_ABORT("deepseek4: unsupported compress ratio %" PRId64, comp_ratio);
                }

                comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                        layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                comp_states = ggml_concat(ctx0, comp_nope, comp_pe, 0);
                comp_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, comp_states, head_dim, 1));
                cb(comp_flat, "attn_comp_cache", il);

                updated_cache = ggml_set_rows(ctx0, updated_cache, comp_flat, comp_cache_idx);
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_attn_comp_kv_state, state.attn_comp_kv_state));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_attn_comp_score_state, state.attn_comp_score_state));
        }

        const bool indexer_reaches_topk =
            hparams.indexer_top_k > 0 &&
            comp_ratio > 0 &&
            uint64_t(cparams.n_ctx_seq) > uint64_t(hparams.indexer_top_k) * uint64_t(comp_ratio);
        const bool has_indexer =
            comp_ratio == 4 &&
            indexer_reaches_topk &&
            layer.indexer_proj != nullptr &&
            layer.indexer_attn_q_b != nullptr &&
            layer.indexer_compress_ape != nullptr &&
            layer.indexer_compress_norm != nullptr &&
            layer.indexer_compress_kv != nullptr &&
            layer.indexer_compress_gate != nullptr &&
            state.indexer_kv != nullptr &&
            state.indexer_comp_kv_state != nullptr &&
            state.indexer_comp_score_state != nullptr &&
            deepseek4_inputs->indexer_hadamard != nullptr;

        if (has_indexer) {
            const int64_t indexer_head_dim = hparams.indexer_head_size;
            const int64_t indexer_nope_dim = indexer_head_dim - rope_dim;
            GGML_ASSERT(indexer_nope_dim >= 0);

            const int64_t indexer_comp_dim = layer.indexer_compress_ape->ne[0];
            const int64_t indexer_comp_slots = indexer_comp_dim / indexer_head_dim;
            const bool indexer_overlap = indexer_comp_slots > 1;
            const bool should_compress = ((start_pos + 1) % comp_ratio) == 0;

            ggml_tensor * indexer_comp_kv = mul_mat_checked(layer.indexer_compress_kv, cur_attn, "build_attn_v4.indexer_comp_kv");
            ggml_tensor * indexer_comp_score = mul_mat_checked(layer.indexer_compress_gate, cur_attn, "build_attn_v4.indexer_comp_score");
            indexer_comp_kv = ggml_cont(ctx0, ggml_cast(ctx0, indexer_comp_kv, GGML_TYPE_F32));
            indexer_comp_score = ggml_cont(ctx0, ggml_cast(ctx0, indexer_comp_score, GGML_TYPE_F32));

            ggml_tensor * indexer_ape_row = matrix_block(layer.indexer_compress_ape, 0, start_pos % comp_ratio, indexer_comp_dim, 1);
            indexer_comp_score = ggml_cont(ctx0, ggml_add(ctx0, indexer_comp_score, indexer_ape_row));
            cb(indexer_comp_score, "indexer_comp_score", il);

            updated_indexer_comp_kv_state = ggml_set_rows(ctx0, state.indexer_comp_kv_state, indexer_comp_kv, deepseek4_inputs->comp_slot_idx_r4);
            updated_indexer_comp_score_state = ggml_set_rows(ctx0, state.indexer_comp_score_state, indexer_comp_score, deepseek4_inputs->comp_slot_idx_r4);

            if (should_compress) {
                ggml_tensor * indexer_comp_kv_slots = nullptr;
                ggml_tensor * indexer_comp_score_slots = nullptr;

                if (indexer_overlap) {
                    ggml_tensor * kv_prev = matrix_block(updated_indexer_comp_kv_state, 0, 0, indexer_head_dim, comp_ratio);
                    ggml_tensor * kv_cur = matrix_block(updated_indexer_comp_kv_state, indexer_head_dim, comp_ratio, indexer_head_dim, comp_ratio);
                    ggml_tensor * score_prev = matrix_block(updated_indexer_comp_score_state, 0, 0, indexer_head_dim, comp_ratio);
                    ggml_tensor * score_cur = matrix_block(updated_indexer_comp_score_state, indexer_head_dim, comp_ratio, indexer_head_dim, comp_ratio);

                    indexer_comp_kv_slots = ggml_concat(ctx0, kv_prev, kv_cur, 1);
                    indexer_comp_score_slots = ggml_concat(ctx0, score_prev, score_cur, 1);

                    ggml_tensor * carry_kv = matrix_block(updated_indexer_comp_kv_state, 0, comp_ratio, indexer_comp_dim, comp_ratio);
                    ggml_tensor * carry_score = matrix_block(updated_indexer_comp_score_state, 0, comp_ratio, indexer_comp_dim, comp_ratio);
                    // HF seeds the next overlapping current window with the just-compressed window; new tokens overwrite it slot by slot.
                    updated_indexer_comp_kv_state = ggml_concat(ctx0, carry_kv, carry_kv, 1);
                    updated_indexer_comp_score_state = ggml_concat(ctx0, carry_score, carry_score, 1);
                } else {
                    indexer_comp_kv_slots = updated_indexer_comp_kv_state;
                    indexer_comp_score_slots = updated_indexer_comp_score_state;
                }

                ggml_tensor * indexer_comp_kv_seq = ggml_cont(ctx0, ggml_transpose(ctx0, indexer_comp_kv_slots));
                ggml_tensor * indexer_comp_score_seq = ggml_cont(ctx0, ggml_transpose(ctx0, indexer_comp_score_slots));
                ggml_tensor * indexer_comp_weights = ggml_soft_max(ctx0, indexer_comp_score_seq);
                ggml_tensor * indexer_comp_weighted = ggml_mul(ctx0, indexer_comp_kv_seq, indexer_comp_weights);
                ggml_tensor * indexer_comp_flat = sum_rows_checked(indexer_comp_weighted, "build_attn_v4.indexer_comp_sum");
                indexer_comp_flat = ggml_cont(ctx0, ggml_transpose(ctx0, indexer_comp_flat));
                indexer_comp_flat = build_norm(indexer_comp_flat, layer.indexer_compress_norm, nullptr, LLM_NORM_RMS, il);

                ggml_tensor * indexer_comp_states = reshape_3d_checked(indexer_comp_flat, indexer_head_dim, 1, 1, "build_attn_v4.indexer_comp_states", il);
                ggml_tensor * indexer_comp_nope = ggml_view_3d(ctx0, indexer_comp_states, indexer_nope_dim, 1, 1, indexer_comp_states->nb[1], indexer_comp_states->nb[2], 0);
                ggml_tensor * indexer_comp_pe = ggml_view_3d(ctx0, indexer_comp_states, rope_dim, 1, 1, indexer_comp_states->nb[1], indexer_comp_states->nb[2], indexer_nope_dim * indexer_comp_states->nb[0]);
                indexer_comp_pe = ggml_rope_ext(ctx0, indexer_comp_pe, deepseek4_inputs->comp_pos_r4, nullptr, rope_dim, rope_type,
                        layer_n_ctx_orig, layer_freq_base, layer_freq_scale, layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                indexer_comp_states = ggml_concat(ctx0, indexer_comp_nope, indexer_comp_pe, 0);
                indexer_comp_flat = ggml_cont(ctx0, ggml_reshape_2d(ctx0, indexer_comp_states, indexer_head_dim, 1));
                indexer_comp_flat = ggml_mul_mat(ctx0, deepseek4_inputs->indexer_hadamard, indexer_comp_flat);
                indexer_comp_flat = ggml_fp4_act_quant(ctx0, ggml_cont(ctx0, indexer_comp_flat));
                cb(indexer_comp_flat, "indexer_comp_cache", il);

                updated_indexer_kv = ggml_set_rows(ctx0, updated_indexer_kv, indexer_comp_flat, deepseek4_inputs->indexer_cache_idx_r4);
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_comp_kv_state, state.indexer_comp_kv_state));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_comp_score_state, state.indexer_comp_score_state));
            if (should_compress) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_kv, state.indexer_kv));
            }
        }

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_cache, state.attn_kv));

        const int64_t n_kv = std::min<int64_t>(start_pos + 1, hparams.n_swa);
        ggml_tensor * kv_prefix = ggml_view_2d(ctx0, updated_cache, head_dim, n_kv, updated_cache->nb[1], 0);
        kv_prefix = ggml_cast(ctx0, kv_prefix, GGML_TYPE_F32);
        int64_t n_comp_attn = comp_ratio > 0 ? (start_pos + 1) / comp_ratio : 0;
        if (comp_ratio > 0) {
            const int64_t n_comp = (start_pos + 1) / comp_ratio;
            if (n_comp > 0) {
                ggml_tensor * comp_prefix = ggml_view_2d(ctx0, updated_cache, head_dim, n_comp, updated_cache->nb[1], hparams.n_swa * updated_cache->nb[1]);
                if (has_indexer && hparams.indexer_top_k > 0 && n_comp > hparams.indexer_top_k) {
                    const int64_t indexer_head_dim = hparams.indexer_head_size;
                    const int64_t indexer_nope_dim = indexer_head_dim - rope_dim;

                    ggml_tensor * indexer_q = mul_mat_checked(layer.indexer_attn_q_b, q_base, "build_attn_v4.indexer_q");
                    indexer_q = reshape_3d_checked(indexer_q, indexer_head_dim, hparams.indexer_n_head, work_tokens, "build_attn_v4.indexer_q_3d", il);
                    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, indexer_nope_dim, hparams.indexer_n_head, work_tokens, indexer_q->nb[1], indexer_q->nb[2], 0);
                    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, rope_dim, hparams.indexer_n_head, work_tokens, indexer_q->nb[1], indexer_q->nb[2], indexer_nope_dim * indexer_q->nb[0]);
                    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, rope_dim, rope_type,
                            layer_n_ctx_orig, layer_freq_base, layer_freq_scale, layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
                    indexer_q = ggml_cont(ctx0, ggml_reshape_2d(ctx0, indexer_q, indexer_head_dim, hparams.indexer_n_head));
                    indexer_q = ggml_mul_mat(ctx0, deepseek4_inputs->indexer_hadamard, indexer_q);
                    indexer_q = ggml_fp4_act_quant(ctx0, ggml_cont(ctx0, indexer_q));
                    cb(indexer_q, "indexer_q", il);

                    ggml_tensor * indexer_kv_prefix = ggml_view_2d(ctx0, updated_indexer_kv, indexer_head_dim, n_comp, updated_indexer_kv->nb[1], 0);
                    ggml_tensor * index_scores = ggml_mul_mat(ctx0, indexer_kv_prefix, indexer_q);
                    index_scores = ggml_relu(ctx0, index_scores);

                    ggml_tensor * index_weights = mul_mat_checked(layer.indexer_proj, cur_attn, "build_attn_v4.indexer_weights");
                    const float index_scale = 1.0f / std::sqrt(float(indexer_head_dim)) / std::sqrt(float(hparams.indexer_n_head));
                    index_weights = ggml_scale(ctx0, index_weights, index_scale);
                    index_weights = ggml_reshape_2d(ctx0, index_weights, 1, hparams.indexer_n_head);
                    index_scores = ggml_mul(ctx0, index_scores, index_weights);
                    index_scores = ggml_cont(ctx0, ggml_transpose(ctx0, index_scores));
                    index_scores = sum_rows_checked(index_scores, "build_attn_v4.index_scores");
                    index_scores = ggml_reshape_2d(ctx0, index_scores, n_comp, 1);
                    cb(index_scores, "index_scores", il);

                    ggml_tensor * selected_comp = ggml_argsort_top_k(ctx0, index_scores, hparams.indexer_top_k);
                    cb(selected_comp, "index_topk", il);
                    comp_prefix = ggml_get_rows(ctx0, comp_prefix, selected_comp);
                    n_comp_attn = hparams.indexer_top_k;
                }
                comp_prefix = ggml_cast(ctx0, comp_prefix, GGML_TYPE_F32);
                kv_prefix = ggml_concat(ctx0, kv_prefix, comp_prefix, 1);
            }
        }
        const int64_t n_kv_total = n_kv + n_comp_attn;
        ggml_tensor * kv_states = reshape_3d_checked(kv_prefix, head_dim, 1, n_kv_total, "build_attn_v4.kv_states", il);

        ggml_tensor * out = build_attn_mha(
                q_states,
                kv_states,
                kv_states,
                nullptr,
                nullptr,
                layer.attn_sinks,
                nullptr,
                1.0f / sqrtf(float(head_dim)),
                il);

        out = reshape_3d_checked(out, head_dim, n_head, work_tokens, "build_attn_v4.out", il);

        ggml_tensor * o_nope = ggml_view_3d(ctx0, out, nope_dim, n_head, work_tokens, out->nb[1], out->nb[2], 0);
        ggml_tensor * o_pe = ggml_view_3d(ctx0, out, rope_dim, n_head, work_tokens, out->nb[1], out->nb[2], nope_dim * out->nb[0]);
        if (cparams.flash_attn) {
            o_nope = ggml_cont(ctx0, o_nope);
            o_pe = ggml_cont(ctx0, o_pe);
        }
        o_pe = ggml_rope_ext_back(ctx0, o_pe, inp_pos, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);

        out = ggml_concat(ctx0, o_nope, o_pe, 0);
        out = ggml_cont(ctx0, ggml_reshape_2d(ctx0, out, total_q_dim, work_tokens));
        cb(out, "attn_out", il);

        return build_grouped_out(out, layer, il);
    };

    ggml_tensor * hc_target = ggml_new_tensor_3d(ctx0, inpL->type, n_embd, hc_mult, work_tokens);
    ggml_tensor * inpL_hc = repeat_checked(reshape_3d_checked(inpL, n_embd, 1, work_tokens, "inpL_hc"), hc_target, "inpL_hc");

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL_hc;

        auto [attn_in, attn_post_w, attn_comb] = hc_pre(inpL_hc, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        attn_in = build_norm(attn_in, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_in, "attn_norm", il);

        ggml_tensor * attn_out = build_attn_v4(attn_in, layer, il);
        inpL_hc = hc_post(attn_out, residual, attn_post_w, attn_comb, il);

        residual = inpL_hc;

        auto [ffn_in, ffn_post_w, ffn_comb] = hc_pre(inpL_hc, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        ffn_in = build_norm(ffn_in, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(ffn_in, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_v4(ffn_in, layer, il);
        ggml_tensor * shared_out = build_ffn(ffn_in,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU,
                LLM_FFN_PAR,
                il);
        cb(shared_out, "ffn_shared", il);

        ggml_tensor * ffn_out = ggml_add(ctx0, moe_out, shared_out);
        cb(ffn_out, "ffn_out", il);

        inpL_hc = hc_post(ffn_out, residual, ffn_post_w, ffn_comb, il);
    }

    ggml_tensor * cur = hc_head(inpL_hc, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = mul_mat_checked(model.output, cur, "output");
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
