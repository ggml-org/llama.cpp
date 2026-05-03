#include "models.h"

#include "ggml-backend.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct dsv4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

struct dsv4_state_pair {
    ggml_tensor * kv;
    ggml_tensor * score;
};

struct dsv4_decode_compressor {
    ggml_tensor * kv_state;
    ggml_tensor * score_state;
    ggml_tensor * kv_comp;
};

struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

enum class dsv4_mask_kind {
    RAW_WINDOW,
    COMPRESS_CAUSAL,
    ATTN_STATIC,
};

struct dsv4_mask_entry {
    ggml_tensor   * tensor = nullptr;
    dsv4_mask_kind kind;
    int64_t         n_raw = 0;
    int64_t         n_comp = 0;
    int64_t         window = 0;
    int64_t         ratio = 0;
};

class dsv4_graph_inputs : public llm_graph_input_i {
public:
    ggml_tensor * add_mask(
            ggml_context  * ctx,
            dsv4_mask_kind kind,
            int64_t        n0,
            int64_t        n1,
            int64_t        n_raw,
            int64_t        n_comp,
            int64_t        window,
            int64_t        ratio,
            const char   * name) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, 1, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        masks.push_back({ t, kind, n_raw, n_comp, window, ratio });
        return t;
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (const auto & mask : masks) {
            GGML_ASSERT(mask.tensor != nullptr);
            if (mask.tensor->buffer == nullptr) {
                continue;
            }

            const int64_t n0 = mask.tensor->ne[0];
            const int64_t n1 = mask.tensor->ne[1];

            std::vector<float> data(n0*n1, -INFINITY);

            switch (mask.kind) {
                case dsv4_mask_kind::RAW_WINDOW:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_CAUSAL:
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, ubatch);
                    break;
                case dsv4_mask_kind::ATTN_STATIC:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    fill_compress_causal(data, n0, n1, mask.ratio, mask.n_raw, ubatch);
                    break;
            }

            ggml_backend_tensor_set(mask.tensor, data.data(), 0, data.size()*sizeof(float));
        }
    }

private:
    static void fill_raw_window(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              window,
            const llama_ubatch * ubatch) {
        GGML_ASSERT((int64_t) ubatch->n_tokens == n1);

        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;

            for (int64_t ik = 0; ik < std::min<int64_t>(n0, ubatch->n_tokens); ++ik) {
                const llama_pos p0 = ubatch->pos ? ubatch->pos[ik] : (llama_pos) ik;

                if (p0 > p1) {
                    continue;
                }

                if (window > 0 && p1 - p0 >= window) {
                    continue;
                }

                data[iq*n0 + ik] = 0.0f;
            }
        }
    }

    static void fill_compress_causal(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              ratio,
            int64_t              offset,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(ratio > 0);

        const int64_t n_comp = n0 - offset;
        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;
            const int64_t n_visible = (p1 + 1) / ratio;

            for (int64_t ic = 0; ic < std::min<int64_t>(n_comp, n_visible); ++ic) {
                data[iq*n0 + offset + ic] = 0.0f;
            }
        }
    }

    std::vector<dsv4_mask_entry> masks;
};

struct dsv4_rope_cfg {
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
};

static ggml_tensor * dsv4_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    return ggml_view_2d(ctx, scale, 1, 1, scale->nb[0], idx * scale->nb[0]);
}

static ggml_tensor * dsv4_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_mul_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale(ctx, x, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    ggml_tensor * t = ggml_arange(ctx, (float) begin, (float) end, 1.0f);
    return ggml_cast(ctx, t, GGML_TYPE_I32);
}

static ggml_tensor * dsv4_new_filled_2d(ggml_context * ctx, int64_t n0, int64_t n1, float value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static ggml_tensor * dsv4_new_filled_3d(ggml_context * ctx, int64_t n0, int64_t n1, int64_t n2, float value) {
    return ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2), value);
}

static dsv4_state_layout dsv4_make_state_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows  = coff * compress_ratio;
    return { width, rows, width * rows };
}

static ggml_tensor * dsv4_view_cols(
        ggml_context * ctx,
        ggml_tensor  * x,
        int64_t        n0,
        int64_t        n1,
        int64_t        off0,
        int64_t        off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static ggml_tensor * dsv4_view_state_segment(
        ggml_context * ctx,
        ggml_tensor  * state,
        int64_t        offset,
        int64_t        width,
        int64_t        rows) {
    return ggml_view_2d(ctx, state, width, rows, width*state->nb[0], offset*state->nb[0]);
}

static void dsv4_store_state_segment(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * src,
        ggml_tensor  * dst,
        int64_t        state_size,
        int64_t        head,
        int64_t        offset) {
    const int64_t n = ggml_nelements(src);
    src = ggml_cont(ctx, src);
    src = ggml_reshape_1d(ctx, src, n);

    ggml_tensor * view = ggml_view_1d(ctx, dst, n, (head*state_size + offset)*ggml_element_size(dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, view));
}

static void dsv4_store_cache_rows(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * cache,
        ggml_tensor  * src,
        int64_t        row_start,
        int64_t        n_rows) {
    if (n_rows <= 0) {
        return;
    }

    src = ggml_cont(ctx, src);
    src = ggml_reshape_2d(ctx, src, cache->ne[0], n_rows);

    ggml_tensor * rows = dsv4_arange_i32(ctx, row_start, row_start + n_rows);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, src, rows));
}

static dsv4_rope_cfg dsv4_make_rope_cfg(
        const llama_hparams & hparams,
        const llama_cparams  & cparams,
        uint32_t              compress_ratio) {
    if (compress_ratio == 0) {
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            cparams.yarn_beta_fast,
            cparams.yarn_beta_slow,
        };
    }

    float attn_factor = 1.0f;
    if (cparams.yarn_ext_factor != 0.0f && cparams.rope_freq_scale > 0.0f) {
        // DeepSeek V4 uses YaRN-style frequency interpolation for compressed RoPE,
        // but the reference implementation does not apply YaRN's magnitude scale.
        attn_factor /= 1.0f + 0.1f * std::log(1.0f / cparams.rope_freq_scale);
    }

    return {
        (int32_t) cparams.n_ctx_orig_yarn,
        hparams.compress_rope_freq_base > 0.0f ? hparams.compress_rope_freq_base : cparams.rope_freq_base,
        cparams.rope_freq_scale,
        cparams.yarn_ext_factor,
        attn_factor,
        cparams.yarn_beta_fast,
        cparams.yarn_beta_slow,
    };
}

static ggml_tensor * dsv4_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    return ggml_view_2d(ctx, base, n, 1, base->nb[0], off * base->nb[0]);
}

static ggml_tensor * dsv4_apply_rope_tail(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * inp_pos,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_tokens,
        int64_t        n_rot,
        int            rope_type,
        int32_t        n_ctx_orig,
        float          freq_base,
        float          freq_scale,
        float          ext_factor,
        float          attn_factor,
        float          beta_fast,
        float          beta_slow,
        bool           inverse) {
    GGML_ASSERT(x->ne[0] == n_embd_head);
    GGML_ASSERT(x->ne[1] == n_head);
    GGML_ASSERT(x->ne[2] == n_tokens);

    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext     (ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }

    const int64_t n_nope = n_embd_head - n_rot;
    GGML_ASSERT(n_nope > 0);

    return ggml_dsv4_rope_tail(ctx, x, inp_pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, inverse);
}

static dsv4_hc_mix dsv4_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int            sinkhorn_iters,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat); // [mix_hc, n_tokens]
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

static ggml_tensor * dsv4_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == n_hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == n_hc);
    GGML_ASSERT(comb->ne[1] == n_hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static ggml_tensor * dsv4_hc_head(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;

    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat); // [hc, n_tokens]
    pre = ggml_mul(ctx, pre, dsv4_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dsv4_view_base(ctx, hc_base, n_hc, 0));
    pre = dsv4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);

    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dsv4_grouped_out(
        ggml_context * ctx,
        ggml_tensor  * o,
        ggml_tensor  * wo_a,
        ggml_tensor  * wo_b,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_groups,
        int64_t        o_lora_rank,
        int64_t        n_tokens) {
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim   = n_embd_head * group_heads;

    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids); // [o_lora_rank, n_groups, n_tokens]
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);

    return ggml_mul_mat(ctx, wo_b, low);
}

static ggml_tensor * dsv4_softmax_pool_ratio(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score) {
    score = ggml_soft_max(ctx, score);
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * dsv4_shift_overlap_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        float          pad_value) {
    const int64_t n_embd  = x->ne[0];
    const int64_t ratio   = x->ne[1];
    const int64_t n_comp  = x->ne[2];

    ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1,
            x->nb[1], x->nb[2], 0);
    ggml_tensor * pad = ggml_fill(ctx, ggml_cont(ctx, first), pad_value);

    if (n_comp == 1) {
        return pad;
    }

    ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1,
            x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

static ggml_tensor * dsv4_build_compressor_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * pos,
        int64_t              n_embd_head,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_comp = n_tokens / compress_ratio;
    GGML_ASSERT(n_comp > 0);

    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv = coff * n_embd_head;
    const int64_t cutoff = n_comp * compress_ratio;

    ggml_tensor * kv = ggml_mul_mat(ctx, wkv, x);       // [coff*head_dim, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);  // [coff*head_dim, n_tokens]

    kv = ggml_view_3d(ctx, kv, n_kv, compress_ratio, n_comp,
            kv->nb[1],
            kv->nb[1] * compress_ratio,
            0);
    score = ggml_view_3d(ctx, score, n_kv, compress_ratio, n_comp,
            score->nb[1],
            score->nb[1] * compress_ratio,
            0);
    GGML_ASSERT(cutoff <= n_tokens);

    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    if (coff == 1) {
        kv = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));       // [ratio, head_dim, n_comp]
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);                // [head_dim, n_comp]
    } else {
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], n_embd_head * kv->nb[0]);
        ggml_tensor * score_prev = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        ggml_tensor * score_curr = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], n_embd_head * score->nb[0]);

        kv_prev    = dsv4_shift_overlap_state(ctx, kv_prev,    0.0f);
        score_prev = dsv4_shift_overlap_state(ctx, score_prev, -INFINITY);

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        kv    = ggml_concat(ctx, kv_prev,    kv_curr,    0); // [2*ratio, head_dim, n_comp]
        score = ggml_concat(ctx, score_prev, score_curr, 0);
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);        // [head_dim, n_comp]
    }

    kv = ggml_rms_norm(ctx, kv, norm_eps);
    kv = ggml_mul(ctx, kv, norm);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, 1, n_comp);

    kv = dsv4_apply_rope_tail(ctx, kv, pos,
            n_embd_head, 1, n_comp, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    return kv;
}

static dsv4_state_pair dsv4_build_compressor_prefill_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * wkv,
        ggml_tensor  * wgate,
        ggml_tensor  * ape,
        int64_t        head_dim,
        int64_t        n_tokens,
        int64_t        compress_ratio) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    const int64_t cutoff    = (n_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_tokens - cutoff;

    ggml_tensor * kv    = ggml_mul_mat(ctx, wkv,    x); // [width, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate,  x);
    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    if (compress_ratio == 4) {
        ggml_tensor * kv_prev    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_prev = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            kv_prev = ggml_view_2d(ctx, kv, layout.width, compress_ratio, kv->nb[1], (cutoff - compress_ratio)*kv->nb[1]);
            score_prev = ggml_view_2d(ctx, score, layout.width, compress_ratio, score->nb[1], (cutoff - compress_ratio)*score->nb[1]);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        ggml_tensor * kv_curr    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_curr = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
            ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev,    kv_curr,    1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    ggml_tensor * kv_state    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
    ggml_tensor * score_state = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
        ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}

static ggml_tensor * dsv4_pool_decode_state(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps) {
    const int64_t n_rows = kv->ne[1];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, 1);

    ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, norm_eps);
    pooled = ggml_mul(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, 1);

    return dsv4_apply_rope_tail(ctx, pooled, pos,
            head_dim, 1, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps);

static dsv4_decode_compressor dsv4_build_compressor_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;

    ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv, x);       // [width, 1]
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

    return dsv4_build_compressor_decode_projected(ctx,
            kv_cur, sc_cur,
            prev_kv_state, prev_score_state,
            norm,
            head_dim, n_rot, pos, compress_ratio,
            rope_type, rope_cfg, norm_eps);
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;
    const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = (pos + 1) % compress_ratio == 0;

    ggml_tensor * row_idx = dsv4_arange_i32(ctx, row, row + 1);
    ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur, row_idx);
    ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur, row_idx);
    ggml_tensor * kv_comp = nullptr;

    if (should_compress) {
        ggml_tensor * kv_pool;
        ggml_tensor * score_pool;

        if (compress_ratio == 4) {
            ggml_tensor * kv_prev = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, 0,        0);
            ggml_tensor * kv_curr = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, head_dim, compress_ratio);
            ggml_tensor * sc_prev = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, 0,        0);
            ggml_tensor * sc_curr = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

            kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
            score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);

            ggml_tensor * shifted_kv    = dsv4_view_cols(ctx, kv_state,    layout.width, compress_ratio, 0, compress_ratio);
            ggml_tensor * shifted_score = dsv4_view_cols(ctx, score_state, layout.width, compress_ratio, 0, compress_ratio);
            kv_state    = ggml_concat(ctx, shifted_kv,    shifted_kv,    1);
            score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
        } else {
            kv_pool    = kv_state;
            score_pool = score_state;
        }

        ggml_tensor * comp_pos = dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
                head_dim, n_rot, rope_type, rope_cfg, norm_eps);
    }

    return { kv_state, score_state, kv_comp };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_chunk(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x); // [width, n_tokens]
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : (llama_pos) i;
        const int64_t pos_mod = pos % compress_ratio;

        ggml_tensor * kv_cur = ggml_view_2d(ctx, kv_all, layout.width, 1, kv_all->nb[1], i*kv_all->nb[1]);
        ggml_tensor * sc_cur = ggml_view_2d(ctx, sc_all, layout.width, 1, sc_all->nb[1], i*sc_all->nb[1]);
        sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_projected(ctx,
                kv_cur,
                sc_cur,
                kv_state,
                score_state,
                norm,
                head_dim,
                n_rot,
                pos,
                compress_ratio,
                rope_type,
                rope_cfg,
                norm_eps);

        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        if (dec.kv_comp != nullptr) {
            kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        }
    }

    return { kv_state, score_state, kv_comp };
}

static ggml_tensor * dsv4_build_indexer_scores_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        ggml_tensor        * causal_mask,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_tokens,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    ggml_tensor * k = ggml_permute(ctx, index_kv, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);                     // [head_dim, n_tokens, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q);            // [n_comp, n_tokens, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x);      // [n_heads, n_tokens]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3);         // [1, n_tokens, n_heads]

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, n_tokens]
    score = ggml_sum_rows(ctx, score);                            // [1, n_comp, n_tokens]
    score = ggml_reshape_2d(ctx, score, index_kv->ne[2], n_tokens);

    return ggml_add(ctx, score, causal_mask);
}

static ggml_tensor * dsv4_build_indexer_scores_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_comp,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, 1);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    ggml_tensor * k = ggml_reshape_3d(ctx, index_kv, n_index_head_size, 1, n_comp);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, 1, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q); // [n_comp, 1, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x); // [n_heads, 1]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, 1);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3); // [1, 1, n_heads]

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, 1]
    score = ggml_sum_rows(ctx, score);
    return ggml_reshape_2d(ctx, score, n_comp, 1);
}

static ggml_tensor * dsv4_build_compressed_mask_from_topk(
        ggml_context * ctx,
        ggml_tensor  * scores,
        ggml_tensor  * topk) {
    const int64_t n_comp   = scores->ne[0];
    const int64_t n_tokens = scores->ne[1];

    ggml_tensor * scores_rows = ggml_reshape_3d(ctx, scores, 1, scores->ne[0], scores->ne[1]);
    ggml_tensor * selected_scores = ggml_get_rows(ctx, scores_rows, topk); // [1, top_k, n_tokens]
    ggml_tensor * valid = ggml_step(ctx, dsv4_add_scalar(ctx, selected_scores, 1.0e30f));
    ggml_tensor * values = dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, valid, -1.0f), 1.0e9f);

    ggml_tensor * mask = dsv4_new_filled_3d(ctx, 1, n_comp, n_tokens, -INFINITY);
    mask = ggml_set_rows(ctx, mask, values, topk);
    return ggml_reshape_2d(ctx, mask, n_comp, n_tokens);
}

static ggml_tensor * dsv4_cache_view_3d(ggml_context * ctx, ggml_tensor * cache, int64_t n_rows) {
    ggml_tensor * view = ggml_view_2d(ctx, cache, cache->ne[0], n_rows, cache->nb[1], 0);
    return ggml_reshape_3d(ctx, view, cache->ne[0], 1, n_rows);
}

} // namespace

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
	llm_graph_context(params) {

    const int64_t n_hc        = hparams.n_hc;
    const int64_t n_lora_q    = hparams.n_lora_q;
    const int64_t n_lora_o    = hparams.n_lora_o;
    const int64_t n_out_group = hparams.n_attn_out_groups;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_lora_o > 0);
    GGML_ASSERT(n_out_group > 0);
    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_mem  = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_mem->get_attn();
    auto * inp_rs   = inp_mem->get_recr();
    const auto * mctx_dsv4 = inp_mem->mctx;
    dsv4_graph_inputs * inp_dsv4 = nullptr;
    auto get_dsv4_inputs = [&]() {
        if (inp_dsv4 == nullptr) {
            auto inputs = std::make_unique<dsv4_graph_inputs>();
            inp_dsv4 = inputs.get();
            res->add_input(std::move(inputs));
        }
        return inp_dsv4;
    };

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;

        if (compress_ratio != 0) {
            if (compress_ratio != 4 && compress_ratio != 128) {
                throw std::runtime_error("DeepSeek V4 unsupported attention compression ratio " + std::to_string(compress_ratio));
            }
            // The hybrid memory splitter emits one sequence set per ubatch
            // for compressed DeepSeek V4 attention.
            GGML_ASSERT(ubatch.n_seqs == 1);
        }

        ggml_tensor * residual = inpL;
        dsv4_hc_mix mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * cur = mix.x;
        cb(cur, "hc_attn_pre", il);
        cb(mix.mixes, "hc_attn_pre_mixes", il);
        cb(mix.pre, "hc_attn_pre_weights", il);
        cb(mix.post, "hc_attn_pre_post_weights", il);
        cb(mix.comb, "hc_attn_pre_comb", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(qr, "q_lora", il);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "q_lora_norm", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
        cb(q, "Qnorm", il);
        q = dsv4_apply_rope_tail(ctx0, q, inp_pos,
                n_embd_head_k, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(q, "Qcur", il);
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
        cb(kv, "KVnorm", il);
        kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos,
                n_embd_head_k, 1, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(kv, "KVrope", il);
        kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
        cb(kv, "KVcur", il);

        const auto * mctx_swa = inp_attn->mctx->get_swa();
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

        if (compress_ratio == 0) {
            ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
            k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]);
            cur = build_attn_mha(q, k_cache, k_cache, nullptr, inp_attn->get_kq_mask_swa(),
                    layer.attn_sinks, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        } else {
            ggml_tensor * k_all = kv;
            ggml_tensor * v_all = kv;
            ggml_tensor * attn_mask = nullptr;
            const llama_seq_id seq_id = ubatch.seq_id[0][0];
            auto store_attn_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                }
            };
            auto store_index_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                }
            };
            const int64_t state_size = hparams.n_embd_r();
            const dsv4_state_layout attn_state_layout = dsv4_make_state_layout(compress_ratio, n_embd_head_k);

            ggml_tensor * prev_kv_state_all = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_sc_state_all = build_rs(inp_rs, inp_rs->mctx->get_s_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_attn_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows);
            ggml_tensor * prev_attn_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows);

            const int64_t n_comp = n_tokens / compress_ratio;
            if (is_prefill) {
                dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        n_embd_head_k,
                        n_tokens,
                        compress_ratio);
                dsv4_store_state_segment(ctx0, gf, state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                if (compress_ratio == 4) {
                    const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                    dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            hparams.indexer_head_size,
                            n_tokens,
                            compress_ratio);
                    dsv4_store_state_segment(ctx0, gf, index_state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    GGML_ASSERT(attn_state_layout.elems + index_state_layout.elems <= state_size);
                }
            }

            if (is_prefill && n_comp > 0) {
                ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
                comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                ggml_tensor * kv_comp = dsv4_build_compressor_prefill(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        layer.attn_compressor_norm,
                        comp_pos,
                        n_embd_head_k, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
                cb(kv_comp, "KVcompress", il);

                store_attn_cache_rows(kv_comp, 0, n_comp);

                k_all = ggml_concat(ctx0, kv, kv_comp, 2);
                v_all = k_all;

                if (compress_ratio == 4) {
                    ggml_tensor * raw_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::RAW_WINDOW,
                            n_tokens, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_raw_window_mask");
                    ggml_tensor * index_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::COMPRESS_CAUSAL,
                            n_comp, n_tokens,
                            0, n_comp, 0, compress_ratio,
                            "dsv4_indexer_causal_mask");

                    ggml_tensor * index_kv = dsv4_build_compressor_prefill(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            layer.indexer_compressor_norm,
                            comp_pos,
                            hparams.indexer_head_size, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                    cb(index_kv, "indexer_KVcompress", il);

                    store_index_cache_rows(index_kv, 0, n_comp);

                    ggml_tensor * index_scores = dsv4_build_indexer_scores_prefill(ctx0,
                            cur, qr, index_kv,
                            layer.indexer_attn_q_b,
                            layer.indexer_proj,
                            inp_pos,
                            index_mask,
                            hparams.indexer_n_head,
                            hparams.indexer_head_size,
                            n_tokens,
                            n_rot,
                            rope_type,
                            rope_cfg);
                    cb(index_scores, "indexer_scores", il);

                    const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp);
                    ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                    cb(topk, "indexer_topk", il);

                    ggml_tensor * comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                    cb(comp_mask, "dsv4_attn_compress_mask", il);

                    attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                } else {
                    attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::ATTN_STATIC,
                            n_tokens + n_comp, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_static_mask");
                }
            } else {
                attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                        dsv4_mask_kind::RAW_WINDOW,
                        n_tokens, n_tokens,
                        n_tokens, 0, hparams.n_swa, compress_ratio,
                        "dsv4_attn_raw_window_mask");
            }

            if (!is_prefill) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                const int64_t n_comp_before  = first_pos / compress_ratio;
                const int64_t n_comp_visible = (last_pos + 1) / compress_ratio;
                const int64_t n_comp_cache = mctx_dsv4->get_dsv4_n_comp(il);
                GGML_ASSERT(n_comp_visible <= n_comp_cache);

                dsv4_decode_compressor dec = n_tokens == 1
                    ? dsv4_build_compressor_decode(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            n_embd_head_k,
                            n_rot,
                            first_pos,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps)
                    : dsv4_build_compressor_decode_chunk(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            ubatch,
                            n_embd_head_k,
                            n_rot,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps);

                dsv4_store_state_segment(ctx0, gf, dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                if (dec.kv_comp != nullptr) {
                    dec.kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    store_attn_cache_rows(dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                }

                ggml_tensor * k_raw = mctx_swa->get_k(ctx0, il);
                k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head_k, 1, k_raw->ne[2]);
                k_all = k_raw;
                v_all = k_raw;
                attn_mask = inp_attn->self_kq_mask_swa;

                if (n_comp_visible > 0) {
                    ggml_tensor * kv_comp_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_visible);
                    k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);
                    v_all = k_all;

                    ggml_tensor * comp_mask = nullptr;
                    if (compress_ratio == 4) {
                        const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                        ggml_tensor * prev_index_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                        ggml_tensor * prev_index_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                        dsv4_decode_compressor index_dec = n_tokens == 1
                            ? dsv4_build_compressor_decode(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    first_pos,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps)
                            : dsv4_build_compressor_decode_chunk(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    ubatch,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps);

                        dsv4_store_state_segment(ctx0, gf, index_dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                        dsv4_store_state_segment(ctx0, gf, index_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);

                        if (index_dec.kv_comp != nullptr) {
                            store_index_cache_rows(index_dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                        }

                        if (n_tokens == 1 && n_comp_visible <= hparams.indexer_top_k) {
                            comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_visible, n_tokens,
                                    0, n_comp_visible, 0, compress_ratio,
                                    "dsv4_attn_compress_mask");
                        } else {
                            ggml_tensor * index_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_visible);
                            index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, n_comp_visible);
                            ggml_tensor * index_scores = n_tokens == 1
                                ? dsv4_build_indexer_scores_decode(ctx0,
                                        cur, qr, index_cache,
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_comp_visible,
                                        n_rot,
                                        rope_type,
                                        rope_cfg)
                                : dsv4_build_indexer_scores_prefill(ctx0,
                                        cur, qr, dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_visible),
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                n_comp_visible, n_tokens,
                                                0, n_comp_visible, 0, compress_ratio,
                                                "dsv4_indexer_decode_causal_mask"),
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_tokens,
                                        n_rot,
                                        rope_type,
                                        rope_cfg);
                            cb(index_scores, "indexer_scores", il);

                            const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_visible);
                            ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                            cb(topk, "indexer_topk", il);

                            comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                        }
                    } else {
                        comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                n_comp_visible, n_tokens,
                                0, n_comp_visible, 0, compress_ratio,
                                "dsv4_attn_compress_mask");
                    }

                    attn_mask = ggml_concat(ctx0, attn_mask, comp_mask, 0);
                }
            }

            ggml_tensor * attn_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
            cur = build_attn_mha(q, k_all, v_all, nullptr, attn_mask_cnv, layer.attn_sinks, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        }
        cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
        cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos,
                n_embd_head_v, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
        cur = dsv4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
        cb(cur, "attn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        cb(cur, "hc_ffn_pre", il);
        cb(mix.mixes, "hc_ffn_pre_mixes", il);
        cb(mix.pre, "hc_ffn_pre_weights", il);
        cb(mix.post, "hc_ffn_pre_post_weights", il);
        cb(mix.comb, "hc_ffn_pre_comb", il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        ggml_tensor * selected = nullptr;
        if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup) {
            selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected, "ffn_moe_hash_topk", il);
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected);
        cb(moe_out, "ffn_moe_out", il);
        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_ffn_post", il);
    }
    if (inp_out_ids) {
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
    }

    ggml_tensor * cur = dsv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
