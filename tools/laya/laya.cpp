#include "laya.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ======================================================================
// Reference (PyTorch): laya/common.py -> DecisionModel
//   h        = encoder(input_ids, attention_mask).last_hidden_state
//   h        = h + type_emb(qtype)[:, None, :]
//   for layer in head.layers: h = layer(h, src_key_padding_mask=pad)
//   m        = gather(h, marker_pos)
//   logits   = scorer(m).squeeze(-1); logits.masked_fill(~marker_mask, -1e4)
//   p        = softmax(logits.detach(), -1)
//   k        = marker_mask.sum(-1).clamp(min=2)
//   ent      = -(p * log(p)).sum(-1) / log(k)
//   top2     = p.topk(2, -1).values
//   feats    = [top2[:,0], top2[:,0]-top2[:,1], ent, k/255]
//   pooled   = h[:, 0]
//   act_logits = act_head(cat([pooled, feats], -1))
//
// The encoder is a ModernBERT (mmBERT) body: bidirectional attention with a
// per-layer dense/sliding window pattern, GeGLU FFN, RoPE (NEOX), norm_first
// residual blocks. The decision head is 2 norm_first TransformerEncoderLayers
// (full self-attention + ReLU FFN, like PyTorch nn.TransformerEncoderLayer),
// a scorer and an action head.
// ======================================================================

struct laya_encoder_layer {
    ggml_tensor * attn_norm = nullptr; // layer 0 is identity (may be null)
    ggml_tensor * wqkv      = nullptr;
    ggml_tensor * wo        = nullptr;
    ggml_tensor * ffn_up    = nullptr;
    ggml_tensor * ffn_down  = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
};

struct laya_head_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * wqkv        = nullptr;
    ggml_tensor * wqkv_b      = nullptr;
    ggml_tensor * wo          = nullptr;
    ggml_tensor * wo_b        = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_norm_b  = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_up_b    = nullptr;
    ggml_tensor * ffn_down    = nullptr;
    ggml_tensor * ffn_down_b  = nullptr;
};

struct laya_model {
    laya_hparams hparams;

    ggml_context_ptr ctx_meta;   // tensor definitions from the gguf header
    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;   // data context holding the real tensors

    ggml_backend_t         backend = nullptr;
    ggml_backend_buffer_ptr buf;

    // self-contained tokenizer data (mmBERT BPE)
    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string> & p) const {
            return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, pair_hash> merge_rank;
    int32_t bos_id  = 2;
    int32_t eos_id  = 1;
    int32_t sep_id  = 1;
    int32_t mask_id = 4;
    int32_t unk_id  = 3;
    int32_t pad_id  = 0;

    ggml_tensor * tok_embd   = nullptr;
    ggml_tensor * tok_norm   = nullptr;
    ggml_tensor * output_norm = nullptr;

    std::vector<laya_encoder_layer> layers;
    ggml_tensor * type_emb = nullptr;
    std::vector<laya_head_layer> head_layers;

    ggml_tensor * scorer_0   = nullptr; ggml_tensor * scorer_0_b = nullptr;
    ggml_tensor * scorer_1   = nullptr; ggml_tensor * scorer_1_b = nullptr;
    ggml_tensor * scorer_3   = nullptr; ggml_tensor * scorer_3_b = nullptr;
    ggml_tensor * act_head_0 = nullptr; ggml_tensor * act_head_0_b = nullptr;
    ggml_tensor * act_head_2 = nullptr; ggml_tensor * act_head_2_b = nullptr;
};

struct laya_context {
    const laya_model * model = nullptr;

    ggml_backend_t         backend     = nullptr;
    ggml_backend_t         backend_cpu = nullptr;
    ggml_backend_sched_ptr sched;

    int n_threads = 1;
};

static void laya_log(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "laya: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// ---- GGUF helpers ------------------------------------------------------

static int64_t gguf_find_key_or(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("missing GGUF key: ") + key);
    }
    return id;
}

static uint32_t gguf_get_u32(const gguf_context * ctx, const char * key, uint32_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_u32(ctx, id);
}

static int32_t gguf_get_i32(const gguf_context * ctx, const char * key, int32_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
        default: throw std::runtime_error(std::string("unexpected type for GGUF key: ") + key);
    }
}

static float gguf_get_f32(const gguf_context * ctx, const char * key, float def = 0.0f) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_f32(ctx, id);
}

static bool gguf_get_bool(const gguf_context * ctx, const char * key, bool def = false) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_bool(ctx, id);
}

static std::vector<float> gguf_get_arr_f32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, id) != GGUF_TYPE_FLOAT32) {
        return {};
    }
    const size_t n = gguf_get_arr_n(ctx, id);
    const float * data = (const float *) gguf_get_arr_data(ctx, id);
    return std::vector<float>(data, data + n);
}

// ---- model loading ------------------------------------------------------

laya_model * laya_model_load_from_file(const char * fname) {
    std::unique_ptr<laya_model> model(new laya_model());

    struct ggml_context * meta = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    model->ctx_gguf.reset(gguf_init_from_file(fname, params));
    if (!model->ctx_gguf.get()) {
        throw std::runtime_error(std::string("failed to load laya model from ") + fname + " (does the file exist?)");
    }
    model->ctx_meta.reset(meta);

    const gguf_context * ctx_gguf = model->ctx_gguf.get();

    // architecture check
    {
        const int64_t id = gguf_find_key(ctx_gguf, "general.architecture");
        if (id < 0 || std::string(gguf_get_val_str(ctx_gguf, id)) != "laya") {
            throw std::runtime_error("not a laya GGUF (general.architecture != \"laya\")");
        }
    }

    auto & hp = model->hparams;
    hp.n_embd          = (int32_t) gguf_get_u32(ctx_gguf, "laya.embedding_length");
    hp.n_layer         = (int32_t) gguf_get_u32(ctx_gguf, "laya.block_count");
    hp.n_head          = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.head_count");
    hp.n_ff            = (int32_t) gguf_get_u32(ctx_gguf, "laya.feed_forward_length");
    hp.n_vocab         = (int32_t) gguf_get_u32(ctx_gguf, "laya.vocab_size");
    hp.n_qtype         = (int32_t) gguf_get_u32(ctx_gguf, "laya.n_qtype", 3);
    hp.marker_token_id = (int32_t) gguf_get_u32(ctx_gguf, "laya.marker_token_id");
    hp.max_len         = (int32_t) gguf_get_u32(ctx_gguf, "laya.max_len", 1024);
    hp.head_max_len    = (int32_t) gguf_get_u32(ctx_gguf, "laya.head_max_len", 256);
    hp.n_act           = (int32_t) gguf_get_u32(ctx_gguf, "laya.act_classes", 2);
    hp.n_head_layers   = (int32_t) gguf_get_u32(ctx_gguf, "laya.head_layers", 2);
    hp.n_swa           = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.sliding_window", 0);
    hp.swa_pattern     = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.sliding_window_pattern", 0);
    hp.norm_eps        = gguf_get_f32(ctx_gguf, "laya.attention.layer_norm_rms_epsilon", 1e-5f);
    hp.rope_freq_base    = gguf_get_f32(ctx_gguf, "laya.rope.freq_base", 10000.0f);
    hp.rope_freq_base_swa = gguf_get_f32(ctx_gguf, "laya.rope.freq_base_swa", hp.rope_freq_base);

    if (hp.n_embd == 0 || hp.n_layer == 0 || hp.n_head == 0) {
        throw std::runtime_error("invalid laya hparams (missing encoder dims)");
    }
    if (hp.n_embd % hp.n_head != 0) {
        throw std::runtime_error("n_embd not divisible by n_head");
    }
    hp.n_embd_head = hp.n_embd / hp.n_head;

    // temperature array, padded to n_qtype with the last value (or 1.0)
    hp.temperature = gguf_get_arr_f32(ctx_gguf, "laya.temperature");
    if (hp.temperature.empty()) {
        hp.temperature.assign(std::max(1, hp.n_qtype), 1.0f);
    }
    while ((int32_t) hp.temperature.size() < hp.n_qtype) {
        hp.temperature.push_back(hp.temperature.back());
    }

    // ---- tokenizer data (mmBERT BPE) ----
    {
        auto get_special = [&](const char * key, int32_t def) {
            const int64_t id = gguf_find_key(ctx_gguf, key);
            return id < 0 ? def : (int32_t) gguf_get_val_u32(ctx_gguf, id);
        };

        model->bos_id  = get_special("tokenizer.ggml.bos_token_id", 2);
        model->eos_id  = get_special("tokenizer.ggml.eos_token_id", 1);
        model->sep_id  = get_special("tokenizer.ggml.seperator_token_id", model->eos_id);
        model->mask_id = get_special("tokenizer.ggml.mask_token_id", 4);
        model->unk_id  = get_special("tokenizer.ggml.unknown_token_id", 3);
        model->pad_id  = get_special("tokenizer.ggml.padding_token_id", 0);

        const int64_t tid = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
        if (tid < 0) {
            throw std::runtime_error("missing tokenizer.ggml.tokens in GGUF");
        }
        model->token_to_id.reserve(gguf_get_arr_n(ctx_gguf, tid));
        for (int64_t i = 0; i < gguf_get_arr_n(ctx_gguf, tid); ++i) {
            model->token_to_id.emplace(gguf_get_arr_str(ctx_gguf, tid, i), (int32_t) i);
        }

        const int64_t mid = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
        if (mid >= 0) {
            const int64_t n_merges = gguf_get_arr_n(ctx_gguf, mid);
            model->merge_rank.reserve((size_t) n_merges);
            for (int64_t i = 0; i < n_merges; ++i) {
                const std::string word = gguf_get_arr_str(ctx_gguf, mid, i);
                const size_t pos = word.find(' ', 1);
                if (pos == std::string::npos) {
                    continue;
                }
                model->merge_rank.emplace(
                        std::make_pair(word.substr(0, pos), word.substr(pos + 1)), (uint32_t) i);
            }
        }
    }

    // ---- create data context and duplicate tensors ----
    {
        struct ggml_init_params p = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        model->ctx_data.reset(ggml_init(p));
        if (!model->ctx_data.get()) {
            throw std::runtime_error("failed to init ggml data context");
        }
    }

    std::map<std::string, size_t> tensor_offset;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        tensor_offset[gguf_get_tensor_name(ctx_gguf, i)] =
            gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
    }

    std::vector<ggml_tensor *> tensors_to_load;
    std::vector<ggml_tensor *> tensors_data;

    // weights stay in their native GGUF type (F16 for the reference F16 GGUF,
    // quantized for the k-quant models); ggml mul_mat dequantizes internally.
    auto get_tensor = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        ggml_tensor * cur = ggml_get_tensor(meta, name.c_str());
        if (!cur) {
            if (required) {
                throw std::runtime_error("missing tensor: " + name);
            }
            return nullptr;
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(model->ctx_data.get(), cur);
        ggml_set_name(data_tensor, cur->name);
        tensors_to_load.push_back(cur);
        tensors_data.push_back(data_tensor);
        return data_tensor;
    };

    // encoder
    model->tok_embd   = get_tensor("token_embd.weight");
    model->tok_norm   = get_tensor("token_embd_norm.weight", false);
    model->output_norm = get_tensor("output_norm.weight", false);

    model->layers.resize(hp.n_layer);
    for (int32_t il = 0; il < hp.n_layer; ++il) {
        auto & layer = model->layers[il];
        char buf[64];
        snprintf(buf, sizeof(buf), "blk.%d.", il);
        const std::string p = buf;

        // layer 0 uses an identity attn pre-norm (ModernBERT); the GGUF ships a
        // blk.0.attn_norm.weight but it must NOT be applied (matches the
        // reference, where layer 0 has no attn_norm parameter at all).
        layer.attn_norm = il == 0 ? nullptr : get_tensor(p + "attn_norm.weight");
        layer.wqkv      = get_tensor(p + "attn_qkv.weight");
        layer.wo        = get_tensor(p + "attn_output.weight");
        layer.ffn_up    = get_tensor(p + "ffn_up.weight");
        layer.ffn_down  = get_tensor(p + "ffn_down.weight");
        layer.ffn_norm  = get_tensor(p + "ffn_norm.weight");
    }

    // decision head
    model->type_emb = get_tensor("type_emb.weight");

    model->head_layers.resize(hp.n_head_layers);
    for (int32_t il = 0; il < hp.n_head_layers; ++il) {
        auto & layer = model->head_layers[il];
        char buf[64];
        snprintf(buf, sizeof(buf), "head.%d.", il);
        const std::string p = buf;

        layer.attn_norm   = get_tensor(p + "attn_norm.weight");
        layer.attn_norm_b = get_tensor(p + "attn_norm.bias");
        layer.wqkv        = get_tensor(p + "attn_qkv.weight");
        layer.wqkv_b      = get_tensor(p + "attn_qkv.bias");
        layer.wo          = get_tensor(p + "attn_output.weight");
        layer.wo_b        = get_tensor(p + "attn_output.bias");
        layer.ffn_norm    = get_tensor(p + "ffn_norm.weight");
        layer.ffn_norm_b  = get_tensor(p + "ffn_norm.bias");
        layer.ffn_up      = get_tensor(p + "ffn_up.weight");
        layer.ffn_up_b    = get_tensor(p + "ffn_up.bias");
        layer.ffn_down    = get_tensor(p + "ffn_down.weight");
        layer.ffn_down_b  = get_tensor(p + "ffn_down.bias");
    }

    // scorer: LayerNorm(d) -> Linear(d, d) -> GELU -> Linear(d, 1)
    model->scorer_0   = get_tensor("scorer.0.weight");
    model->scorer_0_b = get_tensor("scorer.0.bias");
    model->scorer_1   = get_tensor("scorer.1.weight");
    model->scorer_1_b = get_tensor("scorer.1.bias");
    model->scorer_3   = get_tensor("scorer.3.weight");
    model->scorer_3_b = get_tensor("scorer.3.bias");

    // act head: Linear(d+4, 256) -> GELU -> Linear(256, n_act)
    model->act_head_0   = get_tensor("act_head.0.weight");
    model->act_head_0_b = get_tensor("act_head.0.bias");
    model->act_head_2   = get_tensor("act_head.2.weight");
    model->act_head_2_b = get_tensor("act_head.2.bias");

    // backend
    model->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!model->backend) {
        throw std::runtime_error("failed to initialize CPU backend");
    }

    // alloc weights and read tensor data from file
    {
        std::ifstream fin(fname, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("failed to open " + std::string(fname));
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model->backend);
        model->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_data.get(), buft));
        ggml_backend_buffer_set_usage(model->buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<uint8_t> read_buf;
        for (size_t i = 0; i < tensors_to_load.size(); ++i) {
            ggml_tensor * cur = tensors_data[i];
            GGML_ASSERT(cur && "tensor not found in ctx_data");
            auto it_off = tensor_offset.find(cur->name);
            GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
            fin.seekg(it_off->second, std::ios::beg);
            if (!fin) {
                throw std::runtime_error("failed to seek for tensor " + std::string(cur->name));
            }
            const size_t num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(buft)) {
                fin.read((char *) cur->data, num_bytes);
            } else {
                read_buf.resize(num_bytes);
                fin.read((char *) read_buf.data(), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
    }

    return model.release();
}

void laya_model_free(laya_model * model) {
    if (!model) {
        return;
    }
    if (model->backend) {
        ggml_backend_free(model->backend);
    }
    delete model;
}

const laya_hparams & laya_model_hparams(const laya_model * model) {
    return model->hparams;
}

int32_t laya_vocab_bos (const laya_model * model) { return model->bos_id; }
int32_t laya_vocab_sep (const laya_model * model) { return model->sep_id; }
int32_t laya_vocab_mask(const laya_model * model) { return model->mask_id; }

// ---- self-contained tokenizer (HF fast tokenizer port) ----
// The reference checkpoint's tokenizer is a Metaspace pre-tokenizer
// (replacement U+2581, prepend_scheme="always") on top of a byte-level BPE
// with the mmBERT vocabulary. llama.cpp's built-in GPT-2 pre-tokenizer cannot
// reproduce it for this vocab, so this is implemented here against the GGUF
// tokenizer data.

static const char LAYA_SPACE[] = "\xe2\x96\x81"; // U+2581 (▁)

static uint32_t laya_cpt_from_utf8(const char * s, size_t n, size_t & len) {
    const uint8_t * u = (const uint8_t *) s;
    if (n == 0) { len = 0; return 0; }
    if (u[0] < 0x80) { len = 1; return u[0]; }
    if ((u[0] & 0xE0) == 0xC0 && n >= 2) { len = 2; return ((u[0] & 0x1F) << 6) | (u[1] & 0x3F); }
    if ((u[0] & 0xF0) == 0xE0 && n >= 3) { len = 3; return ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F); }
    if ((u[0] & 0xF8) == 0xF0 && n >= 4) { len = 4; return ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) | ((u[2] & 0x3F) << 6) | (u[3] & 0x3F); }
    len = 1;
    return 0xFFFD; // replacement char for invalid bytes
}

// greedy byte-level BPE over the codepoint symbols of one word
static void laya_bpe_encode(const laya_model * model, const std::string & word, std::vector<int32_t> & out) {
    const auto & token_to_id = model->token_to_id;
    const auto & merge_rank  = model->merge_rank;

    std::vector<std::string> syms;
    size_t i = 0;
    while (i < word.size()) {
        size_t len = 0;
        const uint32_t cpt = laya_cpt_from_utf8(word.c_str() + i, word.size() - i, len);
        const std::string ch = word.substr(i, len);
        i += len;

        if (token_to_id.find(ch) != token_to_id.end()) {
            syms.push_back(ch);
        } else {
            // byte fallback
            char buf[16];
            for (unsigned char b : ch) {
                snprintf(buf, sizeof(buf), "<0x%02X>", (int) b);
                syms.push_back(buf);
            }
        }
    }

    while (syms.size() > 1) {
        // find the adjacent pair with the lowest merge rank
        int64_t best_rank = -1;
        size_t  best_i = 0;
        for (size_t k = 0; k + 1 < syms.size(); ++k) {
            auto it = merge_rank.find(std::make_pair(syms[k], syms[k + 1]));
            if (it != merge_rank.end() && (best_rank < 0 || (int64_t) it->second < best_rank)) {
                best_rank = (int64_t) it->second;
                best_i = k;
            }
        }
        if (best_rank < 0) {
            break;
        }
        const std::string a = syms[best_i];
        const std::string b = syms[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(syms.size());
        size_t k = 0;
        while (k < syms.size()) {
            if (k + 1 < syms.size() && syms[k] == a && syms[k + 1] == b) {
                merged.push_back(a + b);
                k += 2;
            } else {
                merged.push_back(syms[k]);
                ++k;
            }
        }
        syms.swap(merged);
    }

    for (const auto & s : syms) {
        auto it = token_to_id.find(s);
        if (it == token_to_id.end()) {
            throw std::runtime_error("laya_tokenize: produced unknown symbol: " + s);
        }
        out.push_back(it->second);
    }
}

std::vector<int32_t> laya_tokenize(const laya_model * model, const std::string & text) {
    std::vector<int32_t> out;
    if (text.empty()) {
        return out;
    }

    // 1. normalizer: Replace(" ", "\u2581")
    // 2. Metaspace: prepend "\u2581" if not already present
    // 3. split on "\u2581"; every piece after a marker is a word (leading marker kept)
    std::string norm;
    norm.reserve(text.size() + 3);
    for (char c : text) {
        if (c == ' ') {
            norm += LAYA_SPACE;
        } else {
            norm += c;
        }
    }
    if (norm.compare(0, 3, LAYA_SPACE) != 0) {
        norm = LAYA_SPACE + norm;
    }

    size_t pos = 0;
    while (pos < norm.size()) {
        // at pos we are at a word boundary (the leading marker)
        size_t next = norm.find(LAYA_SPACE, pos + 3);
        if (next == std::string::npos) {
            next = norm.size();
        }
        const std::string word = norm.substr(pos, next - pos);
        laya_bpe_encode(model, word, out);
        pos = next;
    }

    return out;
}

laya_context * laya_init(const laya_model * model, int n_threads) {
    std::unique_ptr<laya_context> ctx(new laya_context());
    ctx->model = model;

    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!ctx->backend_cpu) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    ctx->backend = ctx->backend_cpu;

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx->backend);
    ctx->sched.reset(ggml_backend_sched_new(&ctx->backend, &buft, 1, 8192, false, true));
    if (!ctx->sched.get()) {
        throw std::runtime_error("failed to initialize backend scheduler");
    }

    ctx->n_threads = std::max(1, n_threads);
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    return ctx.release();
}

void laya_free(laya_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}

// ---- graph helpers ------------------------------------------------------

static ggml_tensor * laya_norm(ggml_context * ctx0, ggml_tensor * cur,
                               ggml_tensor * w, ggml_tensor * b, float eps) {
    cur = ggml_norm(ctx0, cur, eps);
    if (w) {
        cur = ggml_mul(ctx0, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx0, cur, b);
    }
    return cur;
}

// bidirectional multi-head attention without KV cache.
// q/k/v: [n_embd_head, n_head, n_tokens]; mask: [n_tokens, n_tokens, 1, 1]
// returns [n_embd, n_tokens]
static ggml_tensor * laya_attn(ggml_context * ctx0,
                               ggml_tensor * Qcur, ggml_tensor * Kcur, ggml_tensor * Vcur,
                               ggml_tensor * kq_mask, float kq_scale) {
    ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);   // [n_embd_head, n_tokens, n_head, 1]
    ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);              // [n_tokens, n_tokens, n_head, 1]
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));             // [n_tokens, n_embd_head, n_head, 1]

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);            // [n_embd_head, n_tokens, n_head, 1]
    ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);  // [n_embd_head, n_head, n_tokens, 1]
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]); // [n_embd, n_tokens]

    return cur;
}

// fused QKV projection + split into per-head views.
// qkv: [3*n_embd, n_tokens] (already projected); b applied by the caller
// returns contiguous per-head tensors [n_embd_head, n_head, n_tokens]
static std::array<ggml_tensor *, 3> laya_qkv_views(ggml_context * ctx0,
        ggml_tensor * qkv, int64_t n_embd_head, int64_t n_head, int64_t n_tokens) {
    const int64_t n_embd = n_embd_head * n_head;

    // extract contiguous Q / K / V blocks, then reshape into per-head views.
    // (a strided view of the fused projection feeds RoPE/attention with
    //  non-contiguous strides; materializing the blocks keeps every
    //  downstream op contiguous and matches the reference numerically)
    auto block = [&](int64_t offset) -> ggml_tensor * {
        ggml_tensor * t = ggml_view_2d(ctx0, qkv, n_embd, n_tokens, qkv->nb[1], offset * ggml_row_size(qkv->type, n_embd));
        t = ggml_cont(ctx0, t);
        t = ggml_reshape_3d(ctx0, t, n_embd_head, n_head, n_tokens);
        return t;
    };

    ggml_tensor * Qcur = block(0);
    ggml_tensor * Kcur = block(1);
    ggml_tensor * Vcur = block(2);

    return { Qcur, Kcur, Vcur };
}

// RoPE (GPT-NeoX) on [n_embd_head, n_head, n_tokens] tensors.
// ggml_rope_ext returns a new tensor, so the inputs are replaced in-place.
static void laya_rope(ggml_context * ctx0,
                      ggml_tensor ** Qcur, ggml_tensor ** Kcur,
                      ggml_tensor * inp_pos,
                      int n_dims, float freq_base, int n_ctx_orig) {
    *Qcur = ggml_rope_ext(ctx0, *Qcur, inp_pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                          freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    *Kcur = ggml_rope_ext(ctx0, *Kcur, inp_pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                          freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// is layer `il` a sliding-window layer? pattern: dense_first (layer 0 dense,
// then every `pattern`-th layer dense)
static bool laya_is_swa(int32_t il, int32_t pattern) {
    if (pattern <= 0) {
        return false;
    }
    return il % pattern != 0;
}

// build the dense / swa attention masks, matching the PyTorch reference:
//   dense: 0.0 within the same sequence, -inf across sequences
//   swa:   additionally mask |p_query - p_key| > n_swa/2 + 1
static void laya_build_masks(const laya_batch & batch, int32_t n_swa,
                             std::vector<float> & mask, std::vector<float> & mask_swa) {
    const int64_t n = batch.n_tokens;
    mask.assign((size_t) n * n, -INFINITY);
    mask_swa.assign((size_t) n * n, -INFINITY);

    const int32_t half = n_swa / 2 + 1; // reference: config.sliding_window = local_attention//2; +1 inclusive

    for (int64_t i1 = 0; i1 < n; ++i1) {
        const int32_t s1 = batch.seq_id[i1];
        const int32_t p1 = batch.positions[i1];
        for (int64_t i0 = 0; i0 < n; ++i0) {
            if (batch.seq_id[i0] != s1) {
                continue;
            }
            const int32_t p0 = batch.positions[i0];
            mask[i1*n + i0] = 0.0f;
            if (half <= 0 || std::abs(p1 - p0) <= half) {
                mask_swa[i1*n + i0] = 0.0f;
            }
        }
    }
}

// ---- graph building -----------------------------------------------------

struct laya_graph {
    ggml_context_ptr ctx;
    ggml_cgraph * gf = nullptr;

    ggml_tensor * inp_tokens = nullptr;
    ggml_tensor * inp_pos    = nullptr;
    ggml_tensor * inp_qtype  = nullptr;
    ggml_tensor * kq_mask    = nullptr;
    ggml_tensor * kq_mask_swa = nullptr;
    ggml_tensor * marker_pos  = nullptr;
    ggml_tensor * marker_mask = nullptr;
    ggml_tensor * seq_start   = nullptr;

    ggml_tensor * logits     = nullptr;
    ggml_tensor * act_logits = nullptr;

};

static laya_graph laya_graph_build(const laya_model * model, const laya_batch & batch,
                                   ggml_tensor * out_logits, ggml_tensor * out_act) {
    const auto & hp = model->hparams;
    const int64_t n_tokens = batch.n_tokens;
    const int64_t n_seqs   = batch.n_seqs;

    struct ggml_init_params params = {
        /*.mem_size =*/ ggml_tensor_overhead() * 16384,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };

    laya_graph g;
    g.ctx.reset(ggml_init(params));
    ggml_context * ctx0 = g.ctx.get();
    g.gf = ggml_new_graph_custom(ctx0, 8192, false);

    const float kq_scale = 1.0f / sqrtf((float) hp.n_embd_head);
    const int   n_ctx_orig = 8192;

    // ---- inputs ----
    g.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_tokens, "inp_tokens");
    ggml_set_input(g.inp_tokens);

    g.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_pos, "inp_pos");
    ggml_set_input(g.inp_pos);

    g.inp_qtype = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_qtype, "inp_qtype");
    ggml_set_input(g.inp_qtype);

    g.kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(g.kq_mask, "kq_mask");
    ggml_set_input(g.kq_mask);

    g.kq_mask_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(g.kq_mask_swa, "kq_mask_swa");
    ggml_set_input(g.kq_mask_swa);

    g.marker_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(g.marker_pos, "marker_pos");
    ggml_set_input(g.marker_pos);

    g.marker_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(g.marker_mask, "marker_mask");
    ggml_set_input(g.marker_mask);

    g.seq_start = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
    ggml_set_name(g.seq_start, "seq_start");
    ggml_set_input(g.seq_start);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ---- input embeddings + embedding norm ----
    // the token embedding rows are stored in F16; the reference runs the whole
    // encoder in F32, so upcast the lookup to F32 before the embedding norm to
    // avoid F16 activation error (the near-one-hot attention amplifies it).
    inpL = ggml_cast(ctx0, ggml_get_rows(ctx0, model->tok_embd, g.inp_tokens), GGML_TYPE_F32); // [n_embd, n_tokens]
    if (model->tok_norm) {
        inpL = laya_norm(ctx0, inpL, model->tok_norm, nullptr, hp.norm_eps);
    }

    // ---- encoder (ModernBERT) ----
    for (int32_t il = 0; il < hp.n_layer; ++il) {
        const auto & layer = model->layers[il];
        const bool is_swa = hp.n_swa > 0 && laya_is_swa(il, hp.swa_pattern);
        const float freq_base = is_swa ? hp.rope_freq_base_swa : hp.rope_freq_base;

        cur = inpL;
        if (layer.attn_norm) {
            cur = laya_norm(ctx0, inpL, layer.attn_norm, nullptr, hp.norm_eps);
        }

        ggml_tensor * qkv_full = ggml_mul_mat(ctx0, layer.wqkv, cur); // [3*n_embd, n_tokens]
        auto qkv = laya_qkv_views(ctx0, qkv_full, hp.n_embd_head, hp.n_head, n_tokens);
        laya_rope(ctx0, &qkv[0], &qkv[1], g.inp_pos, hp.n_embd_head, freq_base, n_ctx_orig);

        cur = laya_attn(ctx0, qkv[0], qkv[1], qkv[2],
                        is_swa ? g.kq_mask_swa : g.kq_mask, kq_scale);
        cur = ggml_mul_mat(ctx0, layer.wo, cur);           // attn output projection (Wo)

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);

        cur = laya_norm(ctx0, ffn_inp, layer.ffn_norm, nullptr, hp.norm_eps);
        cur = ggml_mul_mat(ctx0, layer.ffn_up, cur);      // [2*n_ff, n_tokens]
        cur = ggml_geglu_erf(ctx0, cur);                  // GeGLU with erf GELU, matching PyTorch
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);    // [n_embd, n_tokens]

        inpL = ggml_add(ctx0, cur, ffn_inp);
    }

    if (model->output_norm) {
        inpL = laya_norm(ctx0, inpL, model->output_norm, nullptr, hp.norm_eps);
    }

    // ---- type embedding ----
    // h = h + type_emb[qtype] broadcast over every token
    inpL = ggml_add(ctx0, inpL, ggml_get_rows(ctx0, model->type_emb, g.inp_qtype));
    ggml_set_name(inpL, "type_emb_out");

    // ---- decision head transformer layers (norm_first, full self-attn) ----
    for (int32_t il = 0; il < hp.n_head_layers; ++il) {
        const auto & layer = model->head_layers[il];

        // attn pre-norm
        cur = laya_norm(ctx0, inpL, layer.attn_norm, layer.attn_norm_b, hp.norm_eps);

        ggml_tensor * qkv_full = ggml_mul_mat(ctx0, layer.wqkv, cur);
        if (layer.wqkv_b) {
            qkv_full = ggml_add(ctx0, qkv_full, layer.wqkv_b);
        }
        auto qkv = laya_qkv_views(ctx0, qkv_full, hp.n_embd_head, hp.n_head, n_tokens);
        // full self-attention, no RoPE, pad mask only
        cur = laya_attn(ctx0, qkv[0], qkv[1], qkv[2], g.kq_mask, kq_scale);

        cur = ggml_mul_mat(ctx0, layer.wo, cur);
        if (layer.wo_b) {
            cur = ggml_add(ctx0, cur, layer.wo_b);
        }

        ggml_tensor * res_inp = ggml_add(ctx0, cur, inpL);

        // ffn pre-norm + FFN (ReLU, like nn.TransformerEncoderLayer default)
        cur = laya_norm(ctx0, res_inp, layer.ffn_norm, layer.ffn_norm_b, hp.norm_eps);
        cur = ggml_mul_mat(ctx0, layer.ffn_up, cur);      // [4*d, n_tokens]
        if (layer.ffn_up_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        }
        cur = ggml_relu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);    // [d, n_tokens]
        if (layer.ffn_down_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        }

        inpL = ggml_add(ctx0, cur, res_inp);
    }

    // ---- gather hidden states at marker positions ----
    // ggml_get_rows only indexes the ne1 (row) dimension, so flatten the
    // [n_markers_max, n_seqs] position tensor to a single row-vector of
    // absolute token indices, then restore the 3D layout afterwards.
    // marker_pos flat index = m + n_markers_max * s, matching a row-major
    // [n_markers_max, n_seqs] input.
    ggml_tensor * marker_pos_flat = ggml_reshape_2d(ctx0, g.marker_pos, LAYA_MAX_MARKERS * n_seqs, 1);
    ggml_tensor * markers = ggml_get_rows(ctx0, inpL, marker_pos_flat);  // [n_embd, n_markers_max * n_seqs]
    markers = ggml_reshape_3d(ctx0, markers, hp.n_embd, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(markers, "markers");

    // ---- scorer: LayerNorm -> Linear -> GELU -> Linear(1) ----
    cur = laya_norm(ctx0, markers, model->scorer_0, model->scorer_0_b, hp.norm_eps);
    cur = ggml_mul_mat(ctx0, model->scorer_1, cur);
    if (model->scorer_1_b) {
        cur = ggml_add(ctx0, cur, model->scorer_1_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model->scorer_3, cur);       // [1, n_markers, n_seqs]
    if (model->scorer_3_b) {
        cur = ggml_add(ctx0, cur, model->scorer_3_b);
    }

    ggml_tensor * logits = ggml_reshape_2d(ctx0, cur, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(logits, "logits");

    // mask invalid markers with -1e4: logits = where(mask, logits, -1e4)
    {
        ggml_tensor * mask_f = ggml_cast(ctx0, g.marker_mask, GGML_TYPE_F32);
        ggml_tensor * ones   = ggml_scale_bias(ctx0, mask_f, 0.0f, 1.0f);
        logits = ggml_add(ctx0,
                ggml_mul(ctx0, logits, mask_f),
                ggml_scale(ctx0, ggml_sub(ctx0, ones, mask_f), -1e4f));
        ggml_set_name(logits, "logits_masked");
    }

    // probabilities over markers (ne0)
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);

    // ---- act head features ----
    // k = marker_mask.sum(-1).clamp(min=2) -> [1, n_seqs]
    ggml_tensor * k_f = ggml_cast(ctx0, g.marker_mask, GGML_TYPE_F32);
    ggml_tensor * k = ggml_clamp(ctx0, ggml_sum_rows(ctx0, k_f), 2.0f, FLT_MAX);

    // ent = -(p * log(p.clamp_min(1e-9))).sum(-1) / log(k)
    ggml_tensor * p_clamped = ggml_clamp(ctx0, probs, 1e-9f, FLT_MAX);
    ggml_tensor * ent = ggml_div(ctx0,
            ggml_neg(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, ggml_log(ctx0, p_clamped)))),
            ggml_log(ctx0, k));

    // top-2 probability values per question (argsort desc along the marker dim)
    ggml_tensor * topk_idx = ggml_argsort_top_k(ctx0, probs, 2);       // I32 [2, n_seqs]
    ggml_tensor * top1_idx = ggml_view_1d(ctx0, topk_idx, n_seqs, 0);
    ggml_tensor * top2_idx = ggml_view_1d(ctx0, topk_idx, n_seqs, 2*sizeof(int32_t));

    ggml_tensor * probs4 = ggml_reshape_4d(ctx0, probs, 1, LAYA_MAX_MARKERS, n_seqs, 1);
    ggml_tensor * b1 = ggml_reshape_4d(ctx0, top1_idx, 1, n_seqs, 1, 1);
    ggml_tensor * b2 = ggml_reshape_4d(ctx0, top2_idx, 1, n_seqs, 1, 1);

    ggml_tensor * top1 = ggml_view_1d(ctx0, ggml_get_rows(ctx0, probs4, b1), n_seqs, 0);
    ggml_tensor * top2 = ggml_view_1d(ctx0, ggml_get_rows(ctx0, probs4, b2), n_seqs, 0);

    // feats = [top1, top1-top2, ent, k/255] -> [4, n_seqs]
    ggml_tensor * f0 = ggml_reshape_2d(ctx0, top1, 1, n_seqs);
    ggml_tensor * f1 = ggml_reshape_2d(ctx0, ggml_sub(ctx0, top1, top2), 1, n_seqs);
    ggml_tensor * f2 = ggml_reshape_2d(ctx0, ent, 1, n_seqs);
    ggml_tensor * f3 = ggml_reshape_2d(ctx0, ggml_scale(ctx0, k, 1.0f/255.0f), 1, n_seqs);

    ggml_tensor * feats = ggml_concat(ctx0,
            ggml_concat(ctx0, ggml_concat(ctx0, f0, f1, 0), f2, 0), f3, 0);

    // pooled = h[:, seq_start] -> [n_embd, n_seqs]
    ggml_tensor * pooled = ggml_get_rows(ctx0, inpL, g.seq_start);

    // act input: concat(pooled, feats) -> [n_embd+4, n_seqs]
    ggml_tensor * act_in = ggml_concat(ctx0, pooled, feats, 0);

    // act head: Linear(d+4, 256) -> GELU -> Linear(256, n_act)
    cur = ggml_mul_mat(ctx0, model->act_head_0, act_in);
    if (model->act_head_0_b) {
        cur = ggml_add(ctx0, cur, model->act_head_0_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model->act_head_2, cur);
    if (model->act_head_2_b) {
        cur = ggml_add(ctx0, cur, model->act_head_2_b);
    }

    g.act_logits = cur;
    ggml_set_name(g.act_logits, "act_logits");

    g.logits = logits;
    // copy the results into pre-allocated output tensors: the scheduler may
    // reuse the compute buffers of intermediate nodes, so reading the graph
    // tensors directly after compute is not safe. ggml_cpy keeps the values
    // in the dedicated output buffer.
    if (out_logits && out_act) {
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx0, logits, out_logits));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx0, cur, out_act));
    }


    return g;
}

int laya_encode(laya_context * ctx, const laya_batch & batch, laya_result & result) {
    const laya_model * model = ctx->model;

    if (batch.n_tokens <= 0 || batch.n_seqs <= 0) {
        return 1;
    }

    laya_graph g;

    // dedicated output buffers (not reused by the scheduler)
    ggml_context_ptr out_ctx;
    ggml_backend_buffer_ptr out_buf;
    ggml_tensor * out_logits = nullptr;
    ggml_tensor * out_act    = nullptr;
    {
        struct ggml_init_params op = {
            /*.mem_size =*/ 8 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        out_ctx.reset(ggml_init(op));
        out_logits = ggml_new_tensor_2d(out_ctx.get(), GGML_TYPE_F32, LAYA_MAX_MARKERS, batch.n_seqs);
        out_act    = ggml_new_tensor_2d(out_ctx.get(), GGML_TYPE_F32, model->hparams.n_act, batch.n_seqs);
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx->backend);
        out_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(out_ctx.get(), buft));
    }

    ggml_backend_sched_reset(ctx->sched.get());
    g = laya_graph_build(model, batch, out_logits, out_act);

    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), g.gf)) {
        laya_log("%s: failed to allocate compute graph\n", __func__);
        return 1;
    }

    // debug: verify input buffers
    {
        ggml_tensor * inps[] = { g.inp_tokens, g.inp_pos, g.inp_qtype, g.kq_mask, g.kq_mask_swa,
                                 g.marker_pos, g.marker_mask, g.seq_start };
        for (auto * t : inps) {
            if (!t->buffer) {
                laya_log("%s: input tensor '%s' has no buffer\n", __func__, t->name);
                return 1;
            }
        }
    }

    // set inputs
    auto set_input = [&](ggml_tensor * t, const void * data) {
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    };

    set_input(g.inp_tokens, batch.tokens);
    set_input(g.inp_pos,    batch.positions);
    set_input(g.inp_qtype,  batch.qtype);
    set_input(g.marker_pos,  batch.marker_pos);
    set_input(g.marker_mask, batch.marker_mask);
    set_input(g.seq_start,   batch.seq_start);

    std::vector<float> mask;
    std::vector<float> mask_swa;
    laya_build_masks(batch, model->hparams.n_swa, mask, mask_swa);
    set_input(g.kq_mask, mask.data());
    set_input(g.kq_mask_swa, mask_swa.data());

    const auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), g.gf);
    if (status != GGML_STATUS_SUCCESS) {
        laya_log("%s: graph compute failed with status %d\n", __func__, (int) status);
        return 1;
    }


    result.n_markers_max = LAYA_MAX_MARKERS;
    result.n_seqs        = batch.n_seqs;
    result.n_act         = model->hparams.n_act;

    result.logits.resize((size_t) LAYA_MAX_MARKERS * batch.n_seqs);
    result.act_logits.resize((size_t) model->hparams.n_act * batch.n_seqs);

    ggml_backend_tensor_get(out_logits, result.logits.data(), 0, ggml_nbytes(out_logits));
    ggml_backend_tensor_get(out_act, result.act_logits.data(), 0, ggml_nbytes(out_act));

    return 0;
}
