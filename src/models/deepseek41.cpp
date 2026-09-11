#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced. V4 computes and
//    consumes in the same sublayer.
// 2. There is no learned hyper-connection head. Nothing is left to collapse the copies with in V4,
//    which is why it needs one; here the last layer's FFN mix is still unused and does the job.
//    These two facts explain each other, and the file ships no output_hc_* tensors.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// TODO: the sparse attention is still V4's and does not match this model. V4.1 compresses KV on
// four source layers and shares one stream, derives index keys from that latent instead of from a
// second compressor, and filters with a two-level candidate mask. See build_attention() below.

// mean over the hyper-connection copies; deepseek4.cpp keeps its own copy of this
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] == (uint32_t) il) {
            return (int) e;
        }
    }
    return -1;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, engram_n_layer);
    if (engram_n_layer == 0 || engram_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", engram_n_layer));
    }
    ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_layer_ids);

    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
    ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
    ml.get_key(LLM_KV_ENGRAM_PAD_ID,         engram_pad_id);

    if (hparams.engram_n_head == 0 || hparams.engram_max_ngram_size < 2) {
        throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
    }

    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] >= hparams.n_layer()) {
            throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
        }
    }

    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS, engram_multipliers);
    ml.get_arr(LLM_KV_ENGRAM_PRIMES,      engram_primes);
    ml.get_arr(LLM_KV_ENGRAM_OFFSETS,     engram_offsets);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,   engram_token_map);

    // the hash indexes straight into these, so a short array would read past the end
    const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;

    if (engram_multipliers.size() != (size_t) engram_n_layer * hparams.engram_max_ngram_size) {
        throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (engram_primes.size() != (size_t) engram_n_layer * n_bucket ||
        engram_offsets.size() != engram_primes.size()) {
        throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (uint64_t p : engram_primes) {
        if (p == 0) {
            throw std::runtime_error("engram prime of zero would divide by zero in the hash");
        }
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t o_groups      = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult       = hparams.dsv4_hc_mult;
    const int64_t hc_dim        = hc_mult * n_embd;
    const int64_t hc_mix_dim    = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    if ((size_t) n_vocab > engram_token_map.size()) {
        throw std::runtime_error(format("engram token map has %zu entries, too few for %" PRId64 " tokens",
                                        engram_token_map.size(), n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups),
        // so reshape at load and keep the graph free of it
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it.
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);

        const int eg = engram_index(i);
        if (eg >= 0) {
            const int64_t n_cols = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
            const int64_t key_len = hparams.engram_key_length;

            // The table has hundreds of millions of rows and is far too large to hold in memory,
            // but each token only touches n_cols of them, so read those rows on demand.
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();
            const auto * embd_w = ml.get_weight(embd_name.c_str());
            if (embd_w == nullptr) {
                throw std::runtime_error(format("%s is missing", embd_name.c_str()));
            }
            const int64_t n_rows = embd_w->tensor->ne[1];

            // a row index is a bucket offset plus a hash, so the last bucket has to end inside
            uint64_t max_row = 0;
            for (int64_t b = 0; b < n_cols; ++b) {
                const size_t k = (size_t) eg*n_cols + b;
                max_row = std::max(max_row, engram_offsets[k] + engram_primes[k]);
            }
            if ((int64_t) max_row > n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the engram buckets (%" PRIu64 ")",
                                                embd_name.c_str(), n_rows, max_row));
            }

            layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {key_len, n_rows}, TENSOR_READ_LAZY);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, 0);
        }

    }

    // Work out which layer publishes the stream each layer reads. Only a source carries a
    // compressor, only an index key owner carries indexer_attn_k, and only an index source
    // carries indexer_attn_q_b, so the file itself says which layer plays which role.
    hparams.dsv41_kv_source.fill(-1);
    hparams.dsv41_index_key_source.fill(-1);
    hparams.dsv41_topk_source.fill(-1);

    int32_t last_kv_source    = -1;
    int32_t last_key_owner    = -1;
    int32_t last_index_source = -1;

    for (int i = 0; i < n_layer; ++i) {
        const auto & layer = layers[i];

        if (layer.attn_comp_wkv)    { last_kv_source    = i; }
        if (layer.indexer_attn_k)   { last_key_owner    = i; }
        if (layer.indexer_attn_q_b) { last_index_source = i; }

        if (hparams.dsv4_compress_ratios[i] == 0) {
            // pure sliding window, no compressed stream to read
            continue;
        }

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            throw std::runtime_error(format("layer %d reads a compressed stream before any layer publishes one", i));
        }

        // the row layout of a stream follows the ratio it was compressed at, so a reader that
        // disagrees with its source would index into rows that stand for different positions
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_kv_source]) {
            throw std::runtime_error(format("layer %d compresses at ratio %u but reads layer %d, compressed at %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_kv_source, hparams.dsv4_compress_ratios[last_kv_source]));
        }

        hparams.dsv41_kv_source[i]        = last_kv_source;
        hparams.dsv41_index_key_source[i] = last_key_owner;
        hparams.dsv41_topk_source[i]      = last_index_source;
    }

    // a compressor with no gate only makes sense where there is nothing to pool
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_is_kv_source(i) && !layers[i].attn_comp_wgate && hparams.dsv4_compress_ratios[i] != 1) {
            throw std::runtime_error(format("layer %d compresses %u tokens per row but has no pooling gate",
                                            i, hparams.dsv4_compress_ratios[i]));
        }
    }

    // TODO: remove once the graph reads the compressed stream. Until then the attention here is
    // V4's, which recognises neither of this model's ratios and would silently drop the long
    // range half of attention while still producing fluent text. Refuse rather than mislead.
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_kv_source[i] >= 0) {
            throw std::runtime_error("DeepSeek-V4.1 sparse attention is not implemented yet");
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek41 & pmodel,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg) : pmodel(pmodel), mctx(mctx), eg(eg) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (pmodel.hparams.engram_max_ngram_size - 1) * pmodel.hparams.engram_n_head;
        return rows->ne[0] == n_cols * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_cols * n_tokens]

    const llama_model_deepseek41 & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult = pmodel.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime = pmodel.engram_primes.data()  + (size_t) eg*n_cols;
    const uint64_t * offset = pmodel.engram_offsets.data() + (size_t) eg*n_cols;

    // an image arrives as an embd batch, so ubatch->token is null; the reference gives those
    // positions no engram contribution at all, which the padding token stands in for here
    const int32_t pad = (int32_t) pmodel.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= pmodel.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) pmodel.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        std::vector<uint64_t> ctx(n_gram);
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(pmodel, mctx_cur->get_raw(), pmodel.engram_index(il));

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, model.layers[il].engram_k);
    ggml_tensor * q = grouped_norm(x,   model.layers[il].engram_q);

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    // The reference uses copysign, which treats +0 as positive where ggml_sgn gives 0, so a dot
    // product of exactly zero gates 0.5 here against 0.50025 there. qwen4exp's PLE gate is built
    // the same way.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream
        if (pmodel.engram_index(il) >= 0) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
