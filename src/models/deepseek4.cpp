#include "models.h"

#include "llama-impl.h"
#include "llama-memory-deepseek4.h"
#include "../llama-deepseek4-hot.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace {

static bool deepseek4_is_power_of_2(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static bool deepseek4_batch_log_enabled() {
    const char * value = std::getenv("LLAMA_DEEPSEEK4_BATCH_LOG");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static bool deepseek4_batch_prefill_enabled() {
    // Default-on: batched prefill is ~7x faster than single-token at long
    // context with no measurable correctness regression on the NMSE smoke
    // tests. Set LLAMA_DEEPSEEK4_BATCH_PREFILL=0 to fall back to the
    // single-token path (used as an escape hatch if a downstream model
    // shows quality regressions from the shared top-k indexer aggregation).
    const char * value = std::getenv("LLAMA_DEEPSEEK4_BATCH_PREFILL");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

static bool deepseek4_indexer_collapse_q() {
    // Default OFF: in batched prefill, the indexer keeps each query's score
    // separate and only aggregates AT THE END (sum across n_head AND
    // work_tokens) for a single ubatch-shared top-k. This is mathematically
    // closer to the original per-token model and works correctly at long
    // context (65K+ retrieval-style prompts).
    //
    // Set LLAMA_DEEPSEEK4_INDEXER_COLLAPSE_Q=1 to opt into the approximate
    // path that sums indexer_q across queries BEFORE the score mul_mat.
    // That cuts the score tensor from [n_comp, n_head, work_tokens] down
    // to [n_comp, n_head] (fits in tight VRAM at large ub) and is faster
    // per ubatch, but breaks long-context quality because the relu
    // non-linearity in scoring means relu(sum_q (kv*q)) != sum_q
    // relu(kv*q) -- the model attends to wrong KV slots.
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_DEEPSEEK4_INDEXER_COLLAPSE_Q");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool deepseek4_indexer_per_query() {
    // Inverse of deepseek4_indexer_collapse_q (kept for code clarity at
    // call sites). Per-query is the default; collapse-Q is opt-in.
    return !deepseek4_indexer_collapse_q();
}

static bool deepseek4_hot_dispatch_enabled() {
    // Default OFF until the prompt-content-sensitive crash on certain expert
    // ID patterns is resolved. Set DS4_HOT_DISPATCH=1 to opt in.
    static const bool enabled = []() {
        const char * value = std::getenv("DS4_HOT_DISPATCH");
        if (value == nullptr) return false;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
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
        const uint32_t n_tokens = ubatch->n_tokens;

        auto set_i32_input = [&](ggml_tensor * tensor, auto fn) {
            if (!tensor || !tensor->buffer) {
                return;
            }

            i32_data.resize(tensor->ne[0]);
            for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                const int32_t p = ubatch->pos ? ubatch->pos[std::min<int64_t>(i, n_tokens - 1)] : 0;
                i32_data[i] = fn(p);
            }
            ggml_backend_tensor_set(tensor, i32_data.data(), 0, ggml_nbytes(tensor));
        };

        set_i32_input(attn_cache_idx, [&](int32_t p) { return p % (int32_t) n_swa; });

        set_i32_input(comp_pos_r4, [](int32_t p) { return std::max<int32_t>(0, p + 1 - 4); });

        set_i32_input(comp_pos_r128, [](int32_t p) { return std::max<int32_t>(0, p + 1 - 128); });

        set_i32_input(comp_cache_idx_r4, [&](int32_t p) { return (int32_t) n_swa + p / 4; });

        set_i32_input(indexer_cache_idx_r4, [](int32_t p) { return p / 4; });

        set_i32_input(comp_cache_idx_r128, [&](int32_t p) { return (int32_t) n_swa + p / 128; });

        set_i32_input(comp_slot_idx_r4, [](int32_t p) { return 4 + (p % 4); });

        set_i32_input(comp_slot_idx_r128, [](int32_t p) { return p % 128; });

        for (size_t mi = 0; mi < kq_masks.size(); ++mi) {
            ggml_tensor * mask = kq_masks[mi];
            if (!mask || !mask->buffer) {
                continue;
            }

            const int64_t n_kv_padded = mask->ne[0];
            const int64_t n_q         = mask->ne[1];
            // n_kv_total[mi] is the actual (unpadded) size; slots in
            // [n_kv_total, n_kv_padded) are padding and stay at -INFINITY.
            const int64_t n_kv_actual = (mi < kq_mask_n_kv_total.size()) ? kq_mask_n_kv_total[mi] : n_kv_padded;
            f32_data.assign(ggml_nelements(mask), -INFINITY);
            for (int64_t iq = 0; iq < n_q; ++iq) {
                const int32_t q_pos = ubatch->pos ? ubatch->pos[std::min<int64_t>(iq, n_tokens - 1)] : 0;
                for (int64_t ikv = 0; ikv < n_kv_actual; ++ikv) {
                    if (ikv >= (int64_t) n_swa || ikv <= q_pos) {
                        f32_data[iq*n_kv_padded + ikv] = 0.0f;
                    }
                }
            }
            ggml_backend_tensor_set(mask, f32_data.data(), 0, ggml_nbytes(mask));
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
    std::vector<ggml_tensor *> kq_masks;
    std::vector<int64_t>       kq_mask_n_kv_total;
    // Cache shared kq_mask tensors keyed by (n_kv_total, work_tokens) so all
    // V4 layers with the same comp_ratio reuse a single graph input. Without
    // this we hit GGML_SCHED_MAX_SPLIT_INPUTS (30) at >30 layers.
    std::map<std::pair<int64_t,int64_t>, ggml_tensor *> kq_mask_by_shape;

    std::vector<int32_t> i32_data;
    std::vector<float> f32_data;
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

    const bool batch_prefill = deepseek4_batch_prefill_enabled() && n_outputs != n_tokens;
    const bool reserve_only = n_tokens != 1 && !batch_prefill;
    const llama_pos start_pos = reserve_only ? 0 : ubatch.pos[0];
    const int64_t work_tokens = reserve_only ? 1 : n_tokens;
    if (deepseek4_batch_log_enabled()) {
        std::fprintf(stderr, "%s: n_tokens=%" PRId64 " reserve_only=%d work_tokens=%" PRId64 " start_pos=%d\n",
                __func__, n_tokens, reserve_only ? 1 : 0, work_tokens, (int) start_pos);
    }
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
    inp_ds4->attn_cache_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->attn_cache_idx);
    ggml_set_name(inp_ds4->attn_cache_idx, "deepseek4_attn_cache_idx");
    inp_ds4->comp_pos_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->comp_pos_r4);
    ggml_set_name(inp_ds4->comp_pos_r4, "deepseek4_comp_pos_r4");
    inp_ds4->comp_pos_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->comp_pos_r128);
    ggml_set_name(inp_ds4->comp_pos_r128, "deepseek4_comp_pos_r128");
    inp_ds4->comp_cache_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->comp_cache_idx_r4);
    ggml_set_name(inp_ds4->comp_cache_idx_r4, "deepseek4_comp_cache_idx_r4");
    inp_ds4->comp_cache_idx_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->comp_cache_idx_r128);
    ggml_set_name(inp_ds4->comp_cache_idx_r128, "deepseek4_comp_cache_idx_r128");
    inp_ds4->indexer_cache_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->indexer_cache_idx_r4);
    ggml_set_name(inp_ds4->indexer_cache_idx_r4, "deepseek4_indexer_cache_idx_r4");
    inp_ds4->comp_slot_idx_r4 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
    ggml_set_input(inp_ds4->comp_slot_idx_r4);
    ggml_set_name(inp_ds4->comp_slot_idx_r4, "deepseek4_comp_slot_idx_r4");
    inp_ds4->comp_slot_idx_r128 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, work_tokens);
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

    auto compression_ape_rows = [&](ggml_tensor * ape, int64_t comp_dim, int64_t comp_ratio) -> ggml_tensor * {
        const int64_t start_mod = start_pos % comp_ratio;

        // Fast path: the entire ubatch fits within one compression window.
        if (start_mod + work_tokens <= comp_ratio) {
            return matrix_block(ape, 0, start_mod, comp_dim, work_tokens);
        }

        // General multi-window slice: decompose the ubatch into
        //   - start_remaining tokens from the current partial window
        //   - any number of complete windows
        //   - end_remaining tokens from the final partial window
        // and concat the corresponding ape slices together. The number
        // of concat ops is bounded by ceil(work_tokens / comp_ratio) + 1.
        ggml_tensor * out = nullptr;
        int64_t consumed = 0;

        if (start_mod > 0) {
            const int64_t start_remaining = comp_ratio - start_mod;
            out = matrix_block(ape, 0, start_mod, comp_dim, start_remaining);
            consumed = start_remaining;
        }

        while (consumed < work_tokens) {
            const int64_t remaining = work_tokens - consumed;
            const int64_t slice_len = std::min<int64_t>(remaining, comp_ratio);
            ggml_tensor * cur = matrix_block(ape, 0, 0, comp_dim, slice_len);
            out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
            consumed += slice_len;
        }

        return out;
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

    auto reshape_2d_checked = [&](ggml_tensor * tensor, int64_t ne0, int64_t ne1, const char * tag, int il = -1) -> ggml_tensor * {
        const int64_t expected = ne0 * ne1;
        if (ggml_nelements(tensor) != expected) {
            GGML_ABORT(
                    "deepseek4: reshape_2d mismatch in %s layer %d pos %d"
                    " ne=%" PRId64 " expected=%" PRId64 " target=(%" PRId64 ",%" PRId64 ") tensor=%s"
                    " shape=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")",
                    tag, il, (int) start_pos, ggml_nelements(tensor), expected, ne0, ne1,
                    tensor->name[0] ? tensor->name : "<unnamed>",
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        }
        return ggml_reshape_2d(ctx0, tensor, ne0, ne1);
    };

    auto add_eps = [&](ggml_tensor * tensor, float eps) -> ggml_tensor * {
        return ggml_clamp(ctx0, tensor, eps, INFINITY);
    };

    auto cont_if_needed = [&](ggml_tensor * tensor) -> ggml_tensor * {
        return ggml_is_contiguous(tensor) ? tensor : ggml_cont(ctx0, tensor);
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
        ggml_tensor * out = ggml_mul(ctx0, tensor, scale);
        return ggml_add(ctx0, out, bias);
    };

    auto weighted_sum_hc = [&](ggml_tensor * x_hc, ggml_tensor * weights) -> ggml_tensor * {
        if (work_tokens > 1 && x_hc->ne[0] == n_embd && x_hc->ne[1] == hc_mult && x_hc->ne[2] == work_tokens &&
                weights->ne[0] == hc_mult && weights->ne[1] == work_tokens) {
            return ggml_hc_weighted_sum(ctx0, x_hc, weights);
        }

        if (x_hc->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 &&
                x_hc->ne[0] == n_embd && x_hc->ne[1] == hc_mult && x_hc->ne[2] == 1 && x_hc->ne[3] == 1 &&
                weights->ne[0] == hc_mult && weights->ne[1] == 1 && weights->ne[2] == 1 && weights->ne[3] == 1) {
            return ggml_hc_weighted_sum(ctx0, x_hc, weights);
        }

        ggml_tensor * x_mat = cont_if_needed(reshape_2d_checked(x_hc, n_embd, hc_mult, "weighted_sum_hc.x_hc"));
        ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x_mat));
        return mul_mat_checked(x_t, weights, "weighted_sum_hc");
    };

    auto sinkhorn = [&](ggml_tensor * comb) -> ggml_tensor * {
        if (comb->type == GGML_TYPE_F32 &&
                comb->ne[0] == 4 && comb->ne[1] == 4) {
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
        ggml_tensor * x_flat = cont_if_needed(reshape_2d_checked(x_hc, n_embd * hc_mult, work_tokens, "hc_pre.x_flat", il));
        ggml_tensor * x_norm = ggml_rms_norm(ctx0, x_flat, hparams.f_norm_rms_eps);
        cb(x_norm, "hc_norm", il);

        ggml_tensor * mixes = mul_mat_checked(hc_fn, x_norm, "hc_pre.mixes");
        cb(mixes, "hc_mixes", il);

        ggml_tensor * pre = vector_slice(mixes, 0, hc_mult);
        ggml_tensor * post = vector_slice(mixes, hc_mult, hc_mult);
        ggml_tensor * comb = matrix_slice(mixes, 2 * hc_mult, hc_mult, hc_mult);
        if (work_tokens > 1) {
            pre  = ggml_view_2d(ctx0, mixes, hc_mult, work_tokens, mixes->nb[1], 0);
            post = ggml_view_2d(ctx0, mixes, hc_mult, work_tokens, mixes->nb[1], hc_mult * mixes->nb[0]);
            comb = ggml_view_3d(ctx0, mixes, hc_mult, hc_mult, work_tokens,
                    hc_mult * mixes->nb[0], mixes->nb[1], 2 * hc_mult * mixes->nb[0]);
        }

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
        if (work_tokens > 1) {
            ggml_tensor * residual_t = ggml_cont(ctx0, ggml_permute(ctx0, residual_hc, 1, 0, 2, 3));
            ggml_tensor * mixed_t = mul_mat_checked(comb, residual_t, "hc_post.mixed_batched");
            ggml_tensor * mixed = ggml_cont(ctx0, ggml_permute(ctx0, mixed_t, 1, 0, 2, 3));

            ggml_tensor * x_repeat = repeat_checked(reshape_3d_checked(x_single, n_embd, 1, work_tokens, "hc_post.x_batched", il),
                    residual_hc, "hc_post.x_batched");
            ggml_tensor * post_repeat = repeat_checked(reshape_3d_checked(post, 1, hc_mult, work_tokens, "hc_post.post_batched", il),
                    residual_hc, "hc_post.post_batched");

            ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, x_repeat, post_repeat), mixed);
            cb(out, "hc_expand", il);
            return out;
        }

        ggml_tensor * residual = cont_if_needed(reshape_2d_checked(residual_hc, n_embd, hc_mult, "hc_post.residual", il));
        ggml_tensor * residual_t = ggml_cont(ctx0, ggml_transpose(ctx0, residual));
        ggml_tensor * mixed_t = mul_mat_checked(comb, residual_t, "hc_post.mixed");
        ggml_tensor * mixed = ggml_cont(ctx0, ggml_transpose(ctx0, mixed_t));

        ggml_tensor * x_repeat = repeat_checked(x_single, residual, "hc_post.x");
        ggml_tensor * post_t = reshape_2d_checked(post, 1, hc_mult, "hc_post.post", il);

        ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, x_repeat, post_t), mixed);
        cb(out, "hc_expand", il);

        return reshape_3d_checked(out, n_embd, hc_mult, work_tokens, "hc_post.out", il);
    };

    auto hc_head = [&](ggml_tensor * x_hc, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base) -> ggml_tensor * {
        ggml_tensor * x_flat = cont_if_needed(reshape_2d_checked(x_hc, n_embd * hc_mult, work_tokens, "hc_head.x_flat"));
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
        const int64_t mix_tokens = cur_ffn->ne[1];
        ggml_tensor * cur_experts_in = reshape_3d_checked(cur_ffn, n_embd, 1, mix_tokens, "build_expert_mix.cur_ffn", il);
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;

        // Phase 2: hot-expert dual dispatch.
        // If a hot-expert profile was loaded (DS4_HOT_PROFILE_JSON) and this
        // layer has hot tensors pinned on GPU, route the K hot experts through
        // the GPU-resident subset and only run the cold picks on the CPU
        // tensor (with a sentinel cold expert in place of any hot picks).
        const ds4_hot::layer_hot_state * hot =
            ds4_hot::instance().is_active() ? ds4_hot::instance().get(il) : nullptr;
        // Warmup/reserve graphs sometimes pass a selected_experts with
        // ne[0] = n_expert instead of n_picks. In that case our per-pick
        // arithmetic would assert in ggml_mul, so fall back to the single
        // path code below.
        const bool dispatch_dual = hot && hot->ready_for_dispatch()
                                   && deepseek4_hot_dispatch_enabled()
                                   && selected_experts->ne[0] == hot->n_picks;

        if (dispatch_dual) {
            const int64_t n_picks_local  = selected_experts->ne[0];
            const int64_t n_tokens_local = selected_experts->ne[1];

            // Ensure selected_experts is contiguous before reshape (defensive).
            ggml_tensor * sel_cont = ggml_cont(ctx0, selected_experts);
            ggml_tensor * sel_flat = ggml_reshape_1d(ctx0, sel_cont, n_picks_local * n_tokens_local);

            // Lookup tables produce float values per pick (in [P*T] flat).
            // Reshape each to [P, T] for the per-pick arithmetic and final
            // mul_mat_id IDs cast.
            ggml_tensor * hot_remap_flat   = ggml_get_rows(ctx0, hot->hot_remap_table,   sel_flat);
            ggml_tensor * cold_remap_flat  = ggml_get_rows(ctx0, hot->cold_remap_table,  sel_flat);
            ggml_tensor * is_hot_flat      = ggml_get_rows(ctx0, hot->is_hot_mask,       sel_flat);
            ggml_tensor * is_cold_flat     = ggml_get_rows(ctx0, hot->is_cold_mask,      sel_flat);

            ggml_tensor * hot_remap  = ggml_reshape_2d(ctx0, hot_remap_flat,  n_picks_local, n_tokens_local);
            ggml_tensor * cold_remap = ggml_reshape_2d(ctx0, cold_remap_flat, n_picks_local, n_tokens_local);
            ggml_tensor * is_hot     = ggml_reshape_2d(ctx0, is_hot_flat,     n_picks_local, n_tokens_local);
            ggml_tensor * is_cold    = ggml_reshape_2d(ctx0, is_cold_flat,    n_picks_local, n_tokens_local);

            // Construct per-pick unique IDs:
            //   hot_ids  = hot_remap  + is_cold * hot_pick_arange    (broadcasts [P,1] -> [P,T])
            //   cold_ids = cold_remap + is_hot  * cold_pick_sentinel
            // hot_pick_arange    = [0, 1, ..., P-1] so cold picks land in [K, K+P) (the dummy zero-weighted experts).
            // cold_pick_sentinel = [cold_ids[0], ..., cold_ids[P-1]] so hot picks each get a different cold sentinel.
            ggml_tensor * hot_offset  = ggml_mul(ctx0, is_cold, hot->hot_pick_arange);    // [P, T] f32
            ggml_tensor * cold_offset = ggml_mul(ctx0, is_hot,  hot->cold_pick_sentinel); // [P, T] f32

            ggml_tensor * hot_ids_f  = ggml_add(ctx0, hot_remap,  hot_offset);
            ggml_tensor * cold_ids_f = ggml_add(ctx0, cold_remap, cold_offset);
            ggml_tensor * hot_ids    = ggml_cast(ctx0, hot_ids_f,  GGML_TYPE_I32);
            ggml_tensor * cold_ids   = ggml_cast(ctx0, cold_ids_f, GGML_TYPE_I32);

            // For the cold-path output mask, we still need [1, P, T] f32 to
            // broadcast against the [n_embd, P, T] expert outputs.
            ggml_tensor * is_cold_3d = ggml_reshape_3d(ctx0, is_cold, 1, n_picks_local, n_tokens_local);

            const float swiglu_limit = hparams.swiglu_clamp_exp[il];

            // Diagnostic mode (DS4_HOT_DISPATCH=cold): only run cold path with
            // cold_ids; no hot contribution. The mask still zeros out hot
            // positions so the output is partial (hot picks contribute 0) but
            // we can verify the cold-with-remap path doesn't crash.
            const char * mode = std::getenv("DS4_HOT_DISPATCH_MODE");
            const bool cold_only = mode && std::strcmp(mode, "cold") == 0;
            const bool hot_only  = mode && std::strcmp(mode, "hot")  == 0;

            ggml_tensor * out_h = nullptr;
            ggml_tensor * out_c = nullptr;

            // === HOT path on GPU (K real hot experts + P dummy zero-weighted experts) ===
            // No output mask needed: cold-pick positions hit dummy experts
            // (positions K..K+P-1) which are zero-initialized, so their
            // contribution is naturally 0. For hot picks, hot_ids points at
            // the right real expert in [0, K).
            if (!cold_only) {
                ggml_tensor * gate_h = nullptr;
                ggml_tensor * up_h   = nullptr;
                // Diagnostic: DS4_HOT_USE_FULL_WEIGHTS=1 forces hot path to use the
                // CPU-resident full-N tensor instead of the GPU-resident K-subset.
                // hot_ids values in [0, K) are still valid for the full tensor.
                const bool use_full = std::getenv("DS4_HOT_USE_FULL_WEIGHTS") != nullptr;
                ggml_tensor * w_gate = use_full ? layer.ffn_gate_exps : hot->hot_gate_exps;
                ggml_tensor * w_up   = use_full ? layer.ffn_up_exps   : hot->hot_up_exps;
                ggml_tensor * w_down = use_full ? layer.ffn_down_exps : hot->hot_down_exps;

                if (hot->hot_gate_up_exps && !use_full) {
                    ggml_tensor * gate_up_h = build_lora_mm_id(hot->hot_gate_up_exps, cur_experts_in, hot_ids);
                    cb(gate_up_h, "ffn_moe_hot_gate_up", il);
                    const int64_t n_ff = gate_up_h->ne[0] / 2;
                    gate_h = ggml_view_3d(ctx0, gate_up_h, n_ff, gate_up_h->ne[1], gate_up_h->ne[2],
                                           gate_up_h->nb[1], gate_up_h->nb[2], 0);
                    up_h   = ggml_view_3d(ctx0, gate_up_h, n_ff, gate_up_h->ne[1], gate_up_h->ne[2],
                                           gate_up_h->nb[1], gate_up_h->nb[2], n_ff * gate_up_h->nb[0]);
                } else {
                    gate_h = build_lora_mm_id(w_gate, cur_experts_in, hot_ids);
                    up_h   = build_lora_mm_id(w_up,   cur_experts_in, hot_ids);
                    cb(gate_h, "ffn_moe_hot_gate", il);
                    cb(up_h,   "ffn_moe_hot_up",   il);
                }

                if (swiglu_limit > 1e-6f) {
                    gate_h = ggml_clamp(ctx0, gate_h, -INFINITY, swiglu_limit);
                    up_h   = ggml_clamp(ctx0, up_h,   -swiglu_limit, swiglu_limit);
                }
                ggml_tensor * act_h    = ggml_swiglu_split(ctx0, gate_h, up_h);
                ggml_tensor * down_h   = build_lora_mm_id(w_down, act_h, hot_ids);
                out_h                  = ggml_mul(ctx0, down_h, weights);
                cb(out_h, "ffn_moe_hot_out", il);
            }

            // === COLD path on CPU (full original tensor with hot picks redirected to per-pick cold sentinels) ===
            // Per-pick cold sentinels avoid the same-expert-multiple-times
            // problem on CPU mul_mat_id. The output mask zeros out hot-pick
            // positions (we still need it because the cold sentinels are
            // real cold experts producing real outputs).
            if (!hot_only) {
                ggml_tensor * gate_c = nullptr;
                ggml_tensor * up_c   = nullptr;
                if (layer.ffn_gate_up_exps) {
                    ggml_tensor * gate_up_c = build_lora_mm_id(layer.ffn_gate_up_exps, cur_experts_in, cold_ids);
                    cb(gate_up_c, "ffn_moe_cold_gate_up", il);
                    const int64_t n_ff = gate_up_c->ne[0] / 2;
                    gate_c = ggml_view_3d(ctx0, gate_up_c, n_ff, gate_up_c->ne[1], gate_up_c->ne[2],
                                           gate_up_c->nb[1], gate_up_c->nb[2], 0);
                    up_c   = ggml_view_3d(ctx0, gate_up_c, n_ff, gate_up_c->ne[1], gate_up_c->ne[2],
                                           gate_up_c->nb[1], gate_up_c->nb[2], n_ff * gate_up_c->nb[0]);
                } else {
                    gate_c = build_lora_mm_id(layer.ffn_gate_exps, cur_experts_in, cold_ids);
                    up_c   = build_lora_mm_id(layer.ffn_up_exps,   cur_experts_in, cold_ids);
                    cb(gate_c, "ffn_moe_cold_gate", il);
                    cb(up_c,   "ffn_moe_cold_up",   il);
                }

                if (swiglu_limit > 1e-6f) {
                    gate_c = ggml_clamp(ctx0, gate_c, -INFINITY, swiglu_limit);
                    up_c   = ggml_clamp(ctx0, up_c,   -swiglu_limit, swiglu_limit);
                }
                ggml_tensor * act_c  = ggml_swiglu_split(ctx0, gate_c, up_c);
                ggml_tensor * down_c = build_lora_mm_id(layer.ffn_down_exps, act_c, cold_ids);
                out_c                = ggml_mul(ctx0, down_c, weights);
                out_c                = ggml_mul(ctx0, out_c, is_cold_3d);
                cb(out_c, "ffn_moe_cold_out", il);
            }

            // === Combine ===
            ggml_tensor * experts;
            if (out_h && out_c) {
                experts = ggml_add(ctx0, out_h, out_c);
            } else if (out_h) {
                experts = out_h;
            } else {
                experts = out_c;
            }
            cb(experts, "ffn_moe_dual_combined", il);

            ggml_tensor * experts_by_id = ggml_cont(ctx0, ggml_permute(ctx0, experts, 1, 0, 2, 3));
            ggml_tensor * out_dual = sum_rows_checked(experts_by_id, "build_expert_mix.sum");
            out_dual = reshape_3d_checked(out_dual, 1, n_embd, mix_tokens, "build_expert_mix.sum_out", il);
            out_dual = reshape_2d_checked(out_dual, n_embd, mix_tokens, "build_expert_mix.out", il);
            cb(out_dual, "ffn_moe_out", il);
            return out_dual;
        }

        // === Default single-path (unchanged) ===
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

        ggml_tensor * experts_by_id = ggml_cont(ctx0, ggml_permute(ctx0, experts, 1, 0, 2, 3));
        ggml_tensor * out = sum_rows_checked(experts_by_id, "build_expert_mix.sum");
        out = reshape_3d_checked(out, 1, n_embd, mix_tokens, "build_expert_mix.sum_out", il);
        out = reshape_2d_checked(out, n_embd, mix_tokens, "build_expert_mix.out", il);

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

    auto build_moe_v4 = [&](ggml_tensor * cur_ffn, ggml_tensor * inp_tokens_local, const llama_layer & layer, int il) -> ggml_tensor * {
        const int64_t moe_tokens = cur_ffn->ne[1];
        ggml_tensor * scores = build_lora_mm(layer.ffn_gate_inp, cur_ffn);
        scores = ggml_softplus(ctx0, scores);
        scores = ggml_sqrt(ctx0, scores);
        cb(scores, "ffn_scores", il);

        ggml_tensor * selection = scores;
        if (layer.ffn_gate_tid2eid) {
            ggml_tensor * hash_selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens_local);
            ggml_tensor * score3d = reshape_3d_checked(scores, 1, n_expert, moe_tokens, "build_moe_v4.scores_hash", il);
            ggml_tensor * selected_scores = ggml_get_rows(ctx0, score3d, hash_selected);
            selection = ggml_set_rows(ctx0, ggml_fill(ctx0, score3d, -INFINITY), selected_scores, hash_selected);
            selection = reshape_2d_checked(selection, n_expert, moe_tokens, "build_moe_v4.selection", il);
            cb(selection, "ffn_hash_scores", il);
        } else if (layer.ffn_exp_probs_b) {
            selection = ggml_add(ctx0, scores, layer.ffn_exp_probs_b);
            cb(selection, "ffn_biased_scores", il);
        }

        ggml_tensor * selected_experts = ggml_top_k(ctx0, selection, n_expert_used);
        cb(selected_experts, "ffn_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0, reshape_3d_checked(scores, 1, n_expert, moe_tokens, "build_moe_v4.scores", il), selected_experts);
        weights = reshape_2d_checked(weights, n_expert_used, moe_tokens, "build_moe_v4.weights_2d", il);
        ggml_tensor * weights_sum = sum_rows_checked(weights, "build_moe_v4.weights_sum");
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        if (hparams.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        }
        weights = reshape_3d_checked(weights, 1, n_expert_used, moe_tokens, "build_moe_v4.weights", il);
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
        k_nope = ggml_fp8_act_quant(ctx0, cont_if_needed(k_nope));
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
        ggml_tensor * k_states = ggml_concat(ctx0, k_nope, k_pe, 0);
        ggml_tensor * k_flat = cont_if_needed(reshape_2d_checked(k_states, head_dim, work_tokens, "build_attn_v4.k_flat", il));

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
            const bool should_compress = ((start_pos + work_tokens) % comp_ratio) == 0;
            const bool multiwindow_r4 =
                comp_ratio == 4 && overlap && work_tokens > comp_ratio &&
                (start_pos % comp_ratio) == 0 && (work_tokens % comp_ratio) == 0;

            ggml_tensor * comp_kv = mul_mat_checked(layer.attn_compress_kv, cur_attn, "build_attn_v4.comp_kv");
            ggml_tensor * comp_score = mul_mat_checked(layer.attn_compress_gate, cur_attn, "build_attn_v4.comp_score");
            // mul_mat always produces a contiguous F32 output, so the
            // cast/cont wrappers we used to use here are no-ops that just
            // add CPY nodes to the prefill graph (43 layers x 2 nodes
            // per ubatch). Drop them.

            ggml_tensor * ape_row = compression_ape_rows(layer.attn_compress_ape, comp_dim, comp_ratio);
            comp_score = ggml_add(ctx0, comp_score, ape_row);
            cb(comp_score, "attn_comp_score", il);

            ggml_tensor * comp_slot_idx = nullptr;
            if (comp_ratio == 4) {
                comp_slot_idx = deepseek4_inputs->comp_slot_idx_r4;
            } else if (comp_ratio == 128) {
                comp_slot_idx = deepseek4_inputs->comp_slot_idx_r128;
            } else {
                GGML_ABORT("deepseek4: unsupported compress ratio %" PRId64, comp_ratio);
            }

            if (!multiwindow_r4) {
                updated_attn_comp_kv_state = ggml_set_rows(ctx0, state.attn_comp_kv_state, comp_kv, comp_slot_idx);
                updated_attn_comp_score_state = ggml_set_rows(ctx0, state.attn_comp_score_state, comp_score, comp_slot_idx);
            }

            if (should_compress) {
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

                if (multiwindow_r4) {
                    // Batched compression for prefill: instead of looping
                    // n_comp_windows = work_tokens/comp_ratio times and emitting
                    // O(n_comp_windows) graph nodes per layer per ubatch (~2.7K
                    // per layer at ub=512), do all windows in one set of ops.
                    //
                    // The original loop builds two head_dim-tall slabs per
                    // window (kv_prev and kv_cur) where comp_kv stacks two
                    // logical "slots" along dim 0 (comp_dim = 2*head_dim).
                    // We build each slab as a 3D batched tensor and then
                    // concat them along dim 1 to recover the [head_dim, 2r]
                    // per-window matrix.
                    const int64_t n = work_tokens / comp_ratio;
                    const int64_t r = comp_ratio;
                    const size_t  type_size = ggml_type_size(GGML_TYPE_F32);
                    const size_t  col_stride = comp_dim * type_size;

                    // prev slab: iw=0 from state, iw>=1 from comp_kv first half
                    ggml_tensor * state_first_kv = ggml_view_3d(ctx0, state.attn_comp_kv_state,
                            head_dim, r, 1, col_stride, r * col_stride, 0);
                    ggml_tensor * state_first_score = ggml_view_3d(ctx0, state.attn_comp_score_state,
                            head_dim, r, 1, col_stride, r * col_stride, 0);
                    ggml_tensor * comp_kv_prev_strided = (n > 1) ? ggml_view_3d(ctx0, comp_kv,
                            head_dim, r, n - 1, col_stride, r * col_stride, 0) : nullptr;
                    ggml_tensor * comp_score_prev_strided = (n > 1) ? ggml_view_3d(ctx0, comp_score,
                            head_dim, r, n - 1, col_stride, r * col_stride, 0) : nullptr;
                    ggml_tensor * prev_kv_b    = comp_kv_prev_strided    ? ggml_concat(ctx0, state_first_kv,    comp_kv_prev_strided,    2) : state_first_kv;
                    ggml_tensor * prev_score_b = comp_score_prev_strided ? ggml_concat(ctx0, state_first_score, comp_score_prev_strided, 2) : state_first_score;

                    // cur slab: comp_kv second half across all n windows
                    ggml_tensor * cur_kv_b    = ggml_view_3d(ctx0, comp_kv,
                            head_dim, r, n, col_stride, r * col_stride, head_dim * type_size);
                    ggml_tensor * cur_score_b = ggml_view_3d(ctx0, comp_score,
                            head_dim, r, n, col_stride, r * col_stride, head_dim * type_size);

                    // [head_dim, 2r, n]
                    ggml_tensor * batched_kv_slots    = ggml_concat(ctx0, prev_kv_b,    cur_kv_b,    1);
                    ggml_tensor * batched_score_slots = ggml_concat(ctx0, prev_score_b, cur_score_b, 1);

                    // permute (1, 0, 2, 3): [head_dim, 2r, n] -> [2r, head_dim, n]
                    ggml_tensor * batched_kv_seq    = ggml_cont(ctx0, ggml_permute(ctx0, batched_kv_slots, 1, 0, 2, 3));
                    ggml_tensor * batched_score_seq = ggml_cont(ctx0, ggml_permute(ctx0, batched_score_slots, 1, 0, 2, 3));

                    ggml_tensor * batched_weights  = ggml_soft_max(ctx0, batched_score_seq);
                    ggml_tensor * batched_weighted = ggml_mul(ctx0, batched_kv_seq, batched_weights);
                    ggml_tensor * batched_flat     = sum_rows_checked(batched_weighted, "build_attn_v4.comp_sum_b");
                    // [1, head_dim, n] -> [head_dim, n]
                    batched_flat = cont_if_needed(reshape_2d_checked(batched_flat, head_dim, n, "build_attn_v4.comp_flat_b", il));
                    batched_flat = build_norm(batched_flat, layer.attn_compress_norm, nullptr, LLM_NORM_RMS, il);

                    // split nope/pe along dim 0
                    ggml_tensor * batched_states = reshape_3d_checked(batched_flat, head_dim, 1, n, "build_attn_v4.comp_states_b", il);
                    ggml_tensor * batched_nope = ggml_view_3d(ctx0, batched_states, nope_dim, 1, n,
                            batched_states->nb[1], batched_states->nb[2], 0);
                    ggml_tensor * batched_pe   = ggml_view_3d(ctx0, batched_states, rope_dim, 1, n,
                            batched_states->nb[1], batched_states->nb[2], nope_dim * batched_states->nb[0]);
                    batched_nope = ggml_fp8_act_quant(ctx0, cont_if_needed(batched_nope));

                    // positions / cache indices: stride-r view picks up
                    // the (r-1)-th token of each window.
                    const size_t i32 = ggml_type_size(GGML_TYPE_I32);
                    ggml_tensor * batched_pos = ggml_view_2d(ctx0, comp_pos, 1, n, r * i32, (r - 1) * i32);
                    batched_pos = ggml_reshape_1d(ctx0, ggml_cont(ctx0, batched_pos), n);
                    ggml_tensor * batched_cache_idx = ggml_view_2d(ctx0, comp_cache_idx, 1, n, r * i32, (r - 1) * i32);
                    batched_cache_idx = ggml_reshape_1d(ctx0, ggml_cont(ctx0, batched_cache_idx), n);

                    batched_pe = ggml_rope_ext(ctx0, batched_pe, batched_pos, nullptr, rope_dim, rope_type,
                            layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                            layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                    batched_states = ggml_concat(ctx0, batched_nope, batched_pe, 0);
                    batched_flat = cont_if_needed(reshape_2d_checked(batched_states, head_dim, n, "build_attn_v4.comp_flat_b2", il));
                    cb(batched_flat, "attn_comp_cache_b", il);

                    updated_cache = ggml_set_rows(ctx0, updated_cache, batched_flat, batched_cache_idx);

                    // overlap state seeding: final window is comp_kv[:, (n-1)*r:n*r]
                    ggml_tensor * final_carry_kv    = matrix_block(comp_kv,    0, (n - 1) * r, comp_dim, r);
                    ggml_tensor * final_carry_score = matrix_block(comp_score, 0, (n - 1) * r, comp_dim, r);
                    updated_attn_comp_kv_state    = ggml_concat(ctx0, final_carry_kv,    final_carry_kv,    1);
                    updated_attn_comp_score_state = ggml_concat(ctx0, final_carry_score, final_carry_score, 1);
                } else {
                    ggml_tensor * comp_kv_slots = nullptr;
                    ggml_tensor * comp_score_slots = nullptr;
                    ggml_tensor * final_carry_kv = nullptr;
                    ggml_tensor * final_carry_score = nullptr;

                    if (overlap) {
                        ggml_tensor * kv_prev = matrix_block(updated_attn_comp_kv_state, 0, 0, head_dim, comp_ratio);
                        ggml_tensor * kv_cur = matrix_block(updated_attn_comp_kv_state, head_dim, comp_ratio, head_dim, comp_ratio);
                        ggml_tensor * score_prev = matrix_block(updated_attn_comp_score_state, 0, 0, head_dim, comp_ratio);
                        ggml_tensor * score_cur = matrix_block(updated_attn_comp_score_state, head_dim, comp_ratio, head_dim, comp_ratio);

                        comp_kv_slots = ggml_concat(ctx0, kv_prev, kv_cur, 1);
                        comp_score_slots = ggml_concat(ctx0, score_prev, score_cur, 1);
                        final_carry_kv = matrix_block(updated_attn_comp_kv_state, 0, comp_ratio, comp_dim, comp_ratio);
                        final_carry_score = matrix_block(updated_attn_comp_score_state, 0, comp_ratio, comp_dim, comp_ratio);
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
                    comp_nope = ggml_fp8_act_quant(ctx0, cont_if_needed(comp_nope));

                    const int64_t token_in_ubatch = work_tokens - 1;
                    ggml_tensor * comp_pos_i = ggml_view_1d(ctx0, comp_pos, 1, token_in_ubatch * comp_pos->nb[0]);
                    ggml_tensor * comp_cache_idx_i = ggml_view_1d(ctx0, comp_cache_idx, 1, token_in_ubatch * comp_cache_idx->nb[0]);

                    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos_i, nullptr, rope_dim, rope_type, layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                            layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                    comp_states = ggml_concat(ctx0, comp_nope, comp_pe, 0);
                    comp_flat = cont_if_needed(reshape_2d_checked(comp_states, head_dim, 1, "build_attn_v4.comp_flat", il));
                    cb(comp_flat, "attn_comp_cache", il);

                    updated_cache = ggml_set_rows(ctx0, updated_cache, comp_flat, comp_cache_idx_i);

                    if (overlap) {
                        // HF seeds the next overlapping current window with the just-compressed window; new tokens overwrite it slot by slot.
                        updated_attn_comp_kv_state = ggml_concat(ctx0, final_carry_kv, final_carry_kv, 1);
                        updated_attn_comp_score_state = ggml_concat(ctx0, final_carry_score, final_carry_score, 1);
                    }
                }
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
            const bool should_compress = ((start_pos + work_tokens) % comp_ratio) == 0;
            const bool multiwindow_r4 =
                indexer_overlap && work_tokens > comp_ratio &&
                (start_pos % comp_ratio) == 0 && (work_tokens % comp_ratio) == 0;

            ggml_tensor * indexer_comp_kv = mul_mat_checked(layer.indexer_compress_kv, cur_attn, "build_attn_v4.indexer_comp_kv");
            ggml_tensor * indexer_comp_score = mul_mat_checked(layer.indexer_compress_gate, cur_attn, "build_attn_v4.indexer_comp_score");

            ggml_tensor * indexer_ape_row = compression_ape_rows(layer.indexer_compress_ape, indexer_comp_dim, comp_ratio);
            indexer_comp_score = ggml_add(ctx0, indexer_comp_score, indexer_ape_row);
            cb(indexer_comp_score, "indexer_comp_score", il);

            if (!multiwindow_r4) {
                updated_indexer_comp_kv_state = ggml_set_rows(ctx0, state.indexer_comp_kv_state, indexer_comp_kv, deepseek4_inputs->comp_slot_idx_r4);
                updated_indexer_comp_score_state = ggml_set_rows(ctx0, state.indexer_comp_score_state, indexer_comp_score, deepseek4_inputs->comp_slot_idx_r4);
            }

            if (should_compress) {
                ggml_tensor * indexer_comp_pos = deepseek4_inputs->comp_pos_r4;
                ggml_tensor * indexer_cache_idx = deepseek4_inputs->indexer_cache_idx_r4;

                if (multiwindow_r4) {
                    // See attn-side compression for the strided-view explanation.
                    const int64_t n = work_tokens / comp_ratio;
                    const int64_t r = comp_ratio;
                    const size_t  type_size = ggml_type_size(GGML_TYPE_F32);
                    const size_t  col_stride = indexer_comp_dim * type_size;

                    ggml_tensor * state_first_kv = ggml_view_3d(ctx0, state.indexer_comp_kv_state,
                            indexer_head_dim, r, 1, col_stride, r * col_stride, 0);
                    ggml_tensor * state_first_score = ggml_view_3d(ctx0, state.indexer_comp_score_state,
                            indexer_head_dim, r, 1, col_stride, r * col_stride, 0);
                    ggml_tensor * comp_kv_prev_strided = (n > 1) ? ggml_view_3d(ctx0, indexer_comp_kv,
                            indexer_head_dim, r, n - 1, col_stride, r * col_stride, 0) : nullptr;
                    ggml_tensor * comp_score_prev_strided = (n > 1) ? ggml_view_3d(ctx0, indexer_comp_score,
                            indexer_head_dim, r, n - 1, col_stride, r * col_stride, 0) : nullptr;
                    ggml_tensor * prev_kv_b    = comp_kv_prev_strided    ? ggml_concat(ctx0, state_first_kv,    comp_kv_prev_strided,    2) : state_first_kv;
                    ggml_tensor * prev_score_b = comp_score_prev_strided ? ggml_concat(ctx0, state_first_score, comp_score_prev_strided, 2) : state_first_score;

                    ggml_tensor * cur_kv_b    = ggml_view_3d(ctx0, indexer_comp_kv,
                            indexer_head_dim, r, n, col_stride, r * col_stride, indexer_head_dim * type_size);
                    ggml_tensor * cur_score_b = ggml_view_3d(ctx0, indexer_comp_score,
                            indexer_head_dim, r, n, col_stride, r * col_stride, indexer_head_dim * type_size);

                    ggml_tensor * batched_kv_slots    = ggml_concat(ctx0, prev_kv_b,    cur_kv_b,    1);
                    ggml_tensor * batched_score_slots = ggml_concat(ctx0, prev_score_b, cur_score_b, 1);

                    ggml_tensor * batched_kv_seq    = ggml_cont(ctx0, ggml_permute(ctx0, batched_kv_slots, 1, 0, 2, 3));
                    ggml_tensor * batched_score_seq = ggml_cont(ctx0, ggml_permute(ctx0, batched_score_slots, 1, 0, 2, 3));

                    ggml_tensor * batched_weights  = ggml_soft_max(ctx0, batched_score_seq);
                    ggml_tensor * batched_weighted = ggml_mul(ctx0, batched_kv_seq, batched_weights);
                    ggml_tensor * batched_flat     = sum_rows_checked(batched_weighted, "build_attn_v4.indexer_comp_sum_b");
                    batched_flat = cont_if_needed(reshape_2d_checked(batched_flat, indexer_head_dim, n, "build_attn_v4.indexer_comp_flat_b", il));
                    batched_flat = build_norm(batched_flat, layer.indexer_compress_norm, nullptr, LLM_NORM_RMS, il);

                    ggml_tensor * batched_states = reshape_3d_checked(batched_flat, indexer_head_dim, 1, n, "build_attn_v4.indexer_comp_states_b", il);
                    ggml_tensor * batched_nope = ggml_view_3d(ctx0, batched_states, indexer_nope_dim, 1, n,
                            batched_states->nb[1], batched_states->nb[2], 0);
                    ggml_tensor * batched_pe   = ggml_view_3d(ctx0, batched_states, rope_dim, 1, n,
                            batched_states->nb[1], batched_states->nb[2], indexer_nope_dim * batched_states->nb[0]);

                    const size_t i32 = ggml_type_size(GGML_TYPE_I32);
                    ggml_tensor * batched_pos = ggml_view_2d(ctx0, indexer_comp_pos, 1, n, r * i32, (r - 1) * i32);
                    batched_pos = ggml_reshape_1d(ctx0, ggml_cont(ctx0, batched_pos), n);
                    ggml_tensor * batched_cache_idx = ggml_view_2d(ctx0, indexer_cache_idx, 1, n, r * i32, (r - 1) * i32);
                    batched_cache_idx = ggml_reshape_1d(ctx0, ggml_cont(ctx0, batched_cache_idx), n);

                    batched_pe = ggml_rope_ext(ctx0, batched_pe, batched_pos, nullptr, rope_dim, rope_type,
                            layer_n_ctx_orig, layer_freq_base, layer_freq_scale,
                            layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                    batched_states = ggml_concat(ctx0, batched_nope, batched_pe, 0);
                    batched_flat = cont_if_needed(reshape_2d_checked(batched_states, indexer_head_dim, n, "build_attn_v4.indexer_comp_flat_b2", il));
                    batched_flat = ggml_mul_mat(ctx0, deepseek4_inputs->indexer_hadamard, batched_flat);
                    batched_flat = ggml_fp4_act_quant(ctx0, cont_if_needed(batched_flat));
                    cb(batched_flat, "indexer_comp_cache_b", il);

                    updated_indexer_kv = ggml_set_rows(ctx0, updated_indexer_kv, batched_flat, batched_cache_idx);

                    ggml_tensor * final_carry_kv    = matrix_block(indexer_comp_kv,    0, (n - 1) * r, indexer_comp_dim, r);
                    ggml_tensor * final_carry_score = matrix_block(indexer_comp_score, 0, (n - 1) * r, indexer_comp_dim, r);
                    updated_indexer_comp_kv_state    = ggml_concat(ctx0, final_carry_kv,    final_carry_kv,    1);
                    updated_indexer_comp_score_state = ggml_concat(ctx0, final_carry_score, final_carry_score, 1);
                } else {
                    ggml_tensor * indexer_comp_kv_slots = nullptr;
                    ggml_tensor * indexer_comp_score_slots = nullptr;
                    ggml_tensor * final_carry_kv = nullptr;
                    ggml_tensor * final_carry_score = nullptr;

                    if (indexer_overlap) {
                        ggml_tensor * kv_prev = matrix_block(updated_indexer_comp_kv_state, 0, 0, indexer_head_dim, comp_ratio);
                        ggml_tensor * kv_cur = matrix_block(updated_indexer_comp_kv_state, indexer_head_dim, comp_ratio, indexer_head_dim, comp_ratio);
                        ggml_tensor * score_prev = matrix_block(updated_indexer_comp_score_state, 0, 0, indexer_head_dim, comp_ratio);
                        ggml_tensor * score_cur = matrix_block(updated_indexer_comp_score_state, indexer_head_dim, comp_ratio, indexer_head_dim, comp_ratio);

                        indexer_comp_kv_slots = ggml_concat(ctx0, kv_prev, kv_cur, 1);
                        indexer_comp_score_slots = ggml_concat(ctx0, score_prev, score_cur, 1);
                        final_carry_kv = matrix_block(updated_indexer_comp_kv_state, 0, comp_ratio, indexer_comp_dim, comp_ratio);
                        final_carry_score = matrix_block(updated_indexer_comp_score_state, 0, comp_ratio, indexer_comp_dim, comp_ratio);
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

                    const int64_t token_in_ubatch = work_tokens - 1;
                    ggml_tensor * indexer_comp_pos_i = ggml_view_1d(ctx0, indexer_comp_pos, 1, token_in_ubatch * indexer_comp_pos->nb[0]);
                    ggml_tensor * indexer_cache_idx_i = ggml_view_1d(ctx0, indexer_cache_idx, 1, token_in_ubatch * indexer_cache_idx->nb[0]);

                    indexer_comp_pe = ggml_rope_ext(ctx0, indexer_comp_pe, indexer_comp_pos_i, nullptr, rope_dim, rope_type,
                            layer_n_ctx_orig, layer_freq_base, layer_freq_scale, layer_ext_factor, layer_attn_factor, layer_beta_fast, layer_beta_slow);
                    indexer_comp_states = ggml_concat(ctx0, indexer_comp_nope, indexer_comp_pe, 0);
                    indexer_comp_flat = cont_if_needed(reshape_2d_checked(indexer_comp_states, indexer_head_dim, 1, "build_attn_v4.indexer_comp_flat", il));
                    indexer_comp_flat = ggml_mul_mat(ctx0, deepseek4_inputs->indexer_hadamard, indexer_comp_flat);
                    indexer_comp_flat = ggml_fp4_act_quant(ctx0, cont_if_needed(indexer_comp_flat));
                    cb(indexer_comp_flat, "indexer_comp_cache", il);

                    updated_indexer_kv = ggml_set_rows(ctx0, updated_indexer_kv, indexer_comp_flat, indexer_cache_idx_i);

                    if (indexer_overlap) {
                        // HF seeds the next overlapping current window with the just-compressed window; new tokens overwrite it slot by slot.
                        updated_indexer_comp_kv_state = ggml_concat(ctx0, final_carry_kv, final_carry_kv, 1);
                        updated_indexer_comp_score_state = ggml_concat(ctx0, final_carry_score, final_carry_score, 1);
                    }
                }
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_comp_kv_state, state.indexer_comp_kv_state));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_comp_score_state, state.indexer_comp_score_state));
            if (should_compress) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_indexer_kv, state.indexer_kv));
            }
        }

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, updated_cache, state.attn_kv));

        const int64_t n_kv = std::min<int64_t>(start_pos + work_tokens, hparams.n_swa);
        ggml_tensor * kv_prefix = ggml_view_2d(ctx0, updated_cache, head_dim, n_kv, updated_cache->nb[1], 0);
        kv_prefix = ggml_cast(ctx0, kv_prefix, GGML_TYPE_F32);
        int64_t n_comp_attn = comp_ratio > 0 ? (start_pos + work_tokens) / comp_ratio : 0;
        if (comp_ratio > 0) {
            const int64_t n_comp = (start_pos + work_tokens) / comp_ratio;
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

                    if (work_tokens > 1 && !deepseek4_indexer_per_query()) {
                        // Batched prefill ubatch-shared top-k: collapse the
                        // work_tokens axis BEFORE the score mul_mat by summing
                        // indexer_q across queries. This avoids materializing
                        // a per-query [n_comp, n_head, work_tokens] score
                        // tensor (which OOMs at ub=512 + long context where
                        // n_comp * 64 * 512 * 4 bytes per layer x 21 r=4
                        // layers blows past 10 GB on the GPUs). The scoring
                        // becomes mul_mat(kv [128, n_comp], sum_q [128, 64])
                        // -> [n_comp, 64], which is the same shape as the
                        // existing decode (work_tokens=1) path. The
                        // approximation is small in practice for short
                        // prompts but at very long context (65K+ retrieval-
                        // style prompts) it can cause the model to attend
                        // to wrong KV slots because relu(sum_q kv*q) !=
                        // sum_q relu(kv*q). Set
                        // LLAMA_DEEPSEEK4_INDEXER_PER_QUERY=1 to keep the
                        // exact per-query path (more accurate, more VRAM,
                        // requires a smaller -ub on tight VRAM hosts).
                        indexer_q = ggml_cont(ctx0, ggml_permute(ctx0, indexer_q, 1, 2, 0, 3));
                        indexer_q = sum_rows_checked(indexer_q, "build_attn_v4.indexer_q_sum_b");
                        indexer_q = cont_if_needed(reshape_2d_checked(indexer_q, indexer_head_dim, hparams.indexer_n_head, "build_attn_v4.indexer_q_collapsed", il));
                    } else if (work_tokens > 1) {
                        // Per-query path: keep work_tokens as a separate
                        // dim through the score mul_mat. Used when
                        // LLAMA_DEEPSEEK4_INDEXER_PER_QUERY=1 is set.
                        indexer_q = cont_if_needed(reshape_3d_checked(indexer_q, indexer_head_dim, hparams.indexer_n_head, work_tokens, "build_attn_v4.indexer_q_b", il));
                    } else {
                        indexer_q = cont_if_needed(reshape_2d_checked(indexer_q, indexer_head_dim, hparams.indexer_n_head, "build_attn_v4.indexer_q", il));
                    }
                    indexer_q = ggml_mul_mat(ctx0, deepseek4_inputs->indexer_hadamard, indexer_q);
                    indexer_q = ggml_fp4_act_quant(ctx0, cont_if_needed(indexer_q));
                    cb(indexer_q, "indexer_q", il);

                    ggml_tensor * indexer_kv_prefix = ggml_view_2d(ctx0, updated_indexer_kv, indexer_head_dim, n_comp, updated_indexer_kv->nb[1], 0);
                    // After the work_tokens collapse this is [n_comp, n_head]
                    // (same as decode). In per-query mode it is
                    // [n_comp, n_head, work_tokens].
                    ggml_tensor * index_scores = ggml_mul_mat(ctx0, indexer_kv_prefix, indexer_q);
                    index_scores = ggml_relu(ctx0, index_scores);

                    ggml_tensor * index_weights = mul_mat_checked(layer.indexer_proj, cur_attn, "build_attn_v4.indexer_weights");
                    const float index_scale = 1.0f / std::sqrt(float(indexer_head_dim)) / std::sqrt(float(hparams.indexer_n_head));
                    index_weights = ggml_scale(ctx0, index_weights, index_scale);
                    if (work_tokens > 1 && !deepseek4_indexer_per_query()) {
                        // index_weights starts as [indexer_n_head, work_tokens];
                        // collapse the work_tokens axis the same way as the
                        // queries so the per-head weighting stays consistent
                        // with the collapsed scores.
                        index_weights = ggml_cont(ctx0, ggml_transpose(ctx0, index_weights));
                        index_weights = sum_rows_checked(index_weights, "build_attn_v4.index_weights_sum_b");
                        index_weights = reshape_2d_checked(index_weights, 1, hparams.indexer_n_head, "build_attn_v4.index_weights_collapsed", il);
                    } else if (work_tokens > 1) {
                        // Per-query: keep weights aligned with scores [.., n_head, work_tokens]
                        index_weights = reshape_3d_checked(index_weights, 1, hparams.indexer_n_head, work_tokens, "build_attn_v4.index_weights_b", il);
                    } else {
                        index_weights = reshape_2d_checked(index_weights, 1, hparams.indexer_n_head, "build_attn_v4.index_weights", il);
                    }
                    index_scores = ggml_mul(ctx0, index_scores, index_weights);
                    if (work_tokens > 1 && deepseek4_indexer_per_query()) {
                        // Aggregate per-query scores into a single ubatch-wide
                        // top-k. Sum across both indexer_n_head and the
                        // work_tokens axis so every query in the ubatch
                        // shares one selected prefix.
                        // Shape evolves [n_comp, n_head, work_tokens] ->
                        //               [n_comp, n_head*work_tokens] ->
                        //               [n_head*work_tokens, n_comp] ->
                        //               [1, n_comp] -> [n_comp, 1]
                        index_scores = cont_if_needed(reshape_2d_checked(index_scores, n_comp, hparams.indexer_n_head * work_tokens, "build_attn_v4.index_scores_flat", il));
                        index_scores = ggml_cont(ctx0, ggml_transpose(ctx0, index_scores));
                        index_scores = sum_rows_checked(index_scores, "build_attn_v4.index_scores_sum");
                        index_scores = reshape_2d_checked(index_scores, n_comp, 1, "build_attn_v4.index_scores_perq", il);
                    } else {
                        index_scores = ggml_cont(ctx0, ggml_transpose(ctx0, index_scores));
                        index_scores = sum_rows_checked(index_scores, "build_attn_v4.index_scores");
                        index_scores = reshape_2d_checked(index_scores, n_comp, 1, "build_attn_v4.index_scores", il);
                    }
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
        // FA kernels for K[0]=512 require K->ne[1] (= n_kv_total after permute)
        // to be a multiple of FATTN_KQ_STRIDE, and require a non-null mask.
        // Without this, the auto-FA reservation graph (which runs with
        // work_tokens=1 and arbitrary n_kv_total) reports unsupported on CUDA,
        // the scheduler places the FA tensor on CPU, and auto-FA disables
        // FA globally for the entire context. Pad both kv_states and the
        // mask to the next multiple of 256 so FA stays on GPU; the padded
        // K/V slots are masked out with -INFINITY and contribute nothing.
        constexpr int64_t kq_pad = 256;
        const int64_t n_kv_total_padded = ((n_kv_total + kq_pad - 1) / kq_pad) * kq_pad;
        const int64_t kv_pad = n_kv_total_padded - n_kv_total;
        if (kv_pad > 0) {
            kv_prefix = ggml_pad(ctx0, kv_prefix, 0, (int) kv_pad, 0, 0);
        }
        ggml_tensor * kv_states = reshape_3d_checked(kv_prefix, head_dim, 1, n_kv_total_padded, "build_attn_v4.kv_states", il);

        ggml_tensor * kq_mask = nullptr;
        {
            const auto key = std::make_pair(n_kv_total_padded, (int64_t) work_tokens);
            auto it = deepseek4_inputs->kq_mask_by_shape.find(key);
            if (it != deepseek4_inputs->kq_mask_by_shape.end()) {
                kq_mask = it->second;
            } else {
                kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_total_padded, work_tokens);
                ggml_set_input(kq_mask);
                ggml_format_name(kq_mask, "deepseek4_kq_mask_%lldx%lld", (long long) n_kv_total_padded, (long long) work_tokens);
                deepseek4_inputs->kq_masks.push_back(kq_mask);
                deepseek4_inputs->kq_mask_n_kv_total.push_back(n_kv_total);
                deepseek4_inputs->kq_mask_by_shape[key] = kq_mask;
            }
        }

        // Flash attention requires the mask in F16; the dense path takes F32.
        ggml_tensor * kq_mask_arg = cparams.flash_attn ? ggml_cast(ctx0, kq_mask, GGML_TYPE_F16) : kq_mask;

        ggml_tensor * out = build_attn_mha(
                q_states,
                kv_states,
                kv_states,
                nullptr,
                kq_mask_arg,
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
        out = cont_if_needed(reshape_2d_checked(out, total_q_dim, work_tokens, "build_attn_v4.out_2d", il));
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

        ggml_tensor * moe_out = build_moe_v4(ffn_in, inp_tokens, layer, il);
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
