#include "models.h"

#include "../llama-batch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr llama_token T5GEMMA2_PAD_TOKEN = 0;

template <typename T>
void fill_t5gemma2_encoder_mask(
        T * data,
        const llama_ubatch * ubatch,
        uint32_t sliding_window,
        bool local) {
    const int64_t n_tokens = ubatch->n_tokens;
    std::fill(data, data + n_tokens*n_tokens, llama_cast<T>(-INFINITY));

    for (int64_t iq = 0; iq < n_tokens; ++iq) {
        const llama_seq_id seq_q = ubatch->seq_id[iq][0];
        const llama_pos pos_q = ubatch->pos[iq];
        for (int64_t ik = 0; ik < n_tokens; ++ik) {
            if (ubatch->seq_id[ik][0] != seq_q ||
                ubatch->token[ik] == T5GEMMA2_PAD_TOKEN) {
                continue;
            }
            const llama_pos pos_k = ubatch->pos[ik];
            const llama_pos q_minus_k = pos_q - pos_k;
            const llama_pos half_window = (llama_pos) sliding_window/2;
            if (local &&
                (q_minus_k < -half_window || q_minus_k >= half_window)) {
                continue;
            }
            data[iq*n_tokens + ik] = llama_cast<T>(0.0f);
        }
    }
}

class llm_graph_input_t5gemma2_encoder_attn : public llm_graph_input_i {
public:
    explicit llm_graph_input_t5gemma2_encoder_attn(uint32_t sliding_window) :
        sliding_window(sliding_window) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token != nullptr);
        GGML_ASSERT(full_mask != nullptr && local_mask != nullptr);

        if (full_mask->type == GGML_TYPE_F16) {
            if (full_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(full_mask->buffer));
                fill_t5gemma2_encoder_mask(
                    (ggml_fp16_t *) full_mask->data, ubatch, sliding_window, false);
            }
            if (local_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(local_mask->buffer));
                fill_t5gemma2_encoder_mask(
                    (ggml_fp16_t *) local_mask->data, ubatch, sliding_window, true);
            }
        } else {
            if (full_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(full_mask->buffer));
                fill_t5gemma2_encoder_mask(
                    (float *) full_mask->data, ubatch, sliding_window, false);
            }
            if (local_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(local_mask->buffer));
                fill_t5gemma2_encoder_mask(
                    (float *) local_mask->data, ubatch, sliding_window, true);
            }
        }
    }

    uint32_t sliding_window;
    ggml_tensor * full_mask  = nullptr;
    ggml_tensor * local_mask = nullptr;
};

class llm_graph_input_t5gemma2_chunk_pool : public llm_graph_input_i {
public:
    explicit llm_graph_input_t5gemma2_chunk_pool(uint32_t chunk_size) :
        chunk_size(chunk_size) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token != nullptr);
        GGML_ASSERT(weights != nullptr);
        GGML_ASSERT(weights->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_backend_buffer_is_host(weights->buffer));

        const int64_t n_tokens = weights->ne[0];
        const int64_t n_chunks = weights->ne[1];
        float * data = (float *) weights->data;
        std::fill(data, data + n_tokens*n_chunks, 0.0f);

        for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
            const int64_t begin = chunk*chunk_size;
            const int64_t end = std::min<int64_t>(begin + chunk_size, n_tokens);
            int64_t valid = 0;
            for (int64_t i = begin; i < end; ++i) {
                valid += ubatch->token[i] != T5GEMMA2_PAD_TOKEN;
            }
            if (valid == 0) {
                continue;
            }
            const float scale = 1.0f/valid;
            for (int64_t i = begin; i < end; ++i) {
                if (ubatch->token[i] != T5GEMMA2_PAD_TOKEN) {
                    data[chunk*n_tokens + i] = scale;
                }
            }
        }
    }

    uint32_t chunk_size;
    ggml_tensor * weights = nullptr;
};

template <typename T>
void fill_t5gemma2_merged_mask(
        T * data,
        const llama_ubatch * ubatch,
        const llama_cross * cross,
        int64_t n_cross,
        uint32_t sliding_window,
        bool local) {
    const int64_t n_self = ubatch->n_tokens;
    const int64_t n_kv = n_self + n_cross;
    std::fill(data, data + n_self*n_kv, llama_cast<T>(-INFINITY));

    for (int64_t iq = 0; iq < n_self; ++iq) {
        const llama_seq_id seq_q = ubatch->seq_id[iq][0];
        const llama_pos pos_q = ubatch->pos[iq];

        for (int64_t ik = 0; ik < n_self; ++ik) {
            if (ubatch->seq_id[ik][0] != seq_q ||
                ubatch->token[ik] == T5GEMMA2_PAD_TOKEN) {
                continue;
            }
            const llama_pos delta = pos_q - ubatch->pos[ik];
            if (delta < 0 || (local && delta >= (llama_pos) sliding_window)) {
                continue;
            }
            data[iq*n_kv + ik] = llama_cast<T>(0.0f);
        }

        for (int64_t ik = 0; ik < n_cross; ++ik) {
            if (cross->seq_ids_enc.size() <= (size_t) ik) {
                continue;
            }
            if (cross->seq_ids_enc[ik].find(seq_q) != cross->seq_ids_enc[ik].end()) {
                data[iq*n_kv + n_self + ik] = llama_cast<T>(0.0f);
            }
        }
    }
}

class llm_graph_input_t5gemma2_merged_attn : public llm_graph_input_i {
public:
    llm_graph_input_t5gemma2_merged_attn(
            const llama_cross * cross,
            uint32_t sliding_window,
            int64_t n_cross) :
        cross(cross),
        sliding_window(sliding_window),
        n_cross(n_cross) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token != nullptr);
        GGML_ASSERT(full_mask != nullptr && local_mask != nullptr);

        if (full_mask->type == GGML_TYPE_F16) {
            if (full_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(full_mask->buffer));
                fill_t5gemma2_merged_mask(
                    (ggml_fp16_t *) full_mask->data,
                    ubatch, cross, n_cross, sliding_window, false);
            }
            if (local_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(local_mask->buffer));
                fill_t5gemma2_merged_mask(
                    (ggml_fp16_t *) local_mask->data,
                    ubatch, cross, n_cross, sliding_window, true);
            }
        } else {
            if (full_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(full_mask->buffer));
                fill_t5gemma2_merged_mask(
                    (float *) full_mask->data,
                    ubatch, cross, n_cross, sliding_window, false);
            }
            if (local_mask->buffer) {
                GGML_ASSERT(ggml_backend_buffer_is_host(local_mask->buffer));
                fill_t5gemma2_merged_mask(
                    (float *) local_mask->data,
                    ubatch, cross, n_cross, sliding_window, true);
            }
        }
    }

    const llama_cross * cross;
    uint32_t sliding_window;
    int64_t n_cross;
    ggml_tensor * full_mask  = nullptr;
    ggml_tensor * local_mask = nullptr;
};

void validate_t5gemma2_tensor_type(const ggml_tensor * tensor, bool allow_quantized) {
    const ggml_type type = tensor->type;
    const bool floating =
        type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
    if (!floating && !(allow_quantized && ggml_is_quantized(type))) {
        throw std::runtime_error(format(
            "T5Gemma2 tensor '%s' has unsupported dtype %s",
            ggml_get_name(tensor), ggml_type_name(type)));
    }
}

} // namespace

void llama_model_t5gemma2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT,          hparams.dec_n_layer);
    uint32_t decoder_start_token_id = 0;
    ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, decoder_start_token_id);
    hparams.dec_start_token_id = decoder_start_token_id;
    ml.get_key(LLM_KV_EMBEDDING_SCALE,              hparams.f_embedding_scale);
    ml.get_key(LLM_KV_ATTENTION_SCALE,              hparams.f_attention_scale);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,     hparams.n_swa);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(
        LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
        hparams.is_swa_impl,
        hparams.n_layer());
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa);
    hparams.rope_freq_scale_train_swa = 1.0f;

    std::string hidden_activation;
    ml.get_key(LLM_KV_HIDDEN_ACT, hidden_activation);
    if (hidden_activation != "gelu_pytorch_tanh") {
        throw std::runtime_error(
            "T5Gemma2 requires hidden_activation=gelu_pytorch_tanh, got " + hidden_activation);
    }
    hparams.llm_ffn_op = LLM_FFN_GELU;

    if (hparams.n_layer() == 0 || hparams.dec_n_layer != hparams.n_layer()) {
        throw std::runtime_error(format(
            "T5Gemma2 v1 requires equal non-zero encoder/decoder layer counts, got %u/%u",
            hparams.n_layer(), hparams.dec_n_layer));
    }
    if (hparams.n_head() == 0 || hparams.n_head_kv() == 0 ||
        hparams.n_head() % hparams.n_head_kv() != 0) {
        throw std::runtime_error(format(
            "T5Gemma2 requires n_head to be divisible by n_head_kv, got %u/%u",
            hparams.n_head(), hparams.n_head_kv()));
    }
    if (hparams.n_embd_head_k_full != hparams.n_embd_head_v_full ||
        hparams.n_rot_full != hparams.n_embd_head_k_full) {
        throw std::runtime_error(format(
            "T5Gemma2 requires equal QK/V head dimensions and full-head RoPE, got %u/%u/%u",
            hparams.n_embd_head_k_full, hparams.n_embd_head_v_full, hparams.n_rot_full));
    }
    if (hparams.n_swa == 0 || hparams.n_swa % 2 != 0 ||
        hparams.f_embedding_scale <= 0.0f ||
        hparams.f_attention_scale <= 0.0f) {
        throw std::runtime_error(
            "T5Gemma2 requires a positive even sliding-window and positive embedding and attention scales");
    }
    if (!std::isfinite(hparams.f_norm_rms_eps) || hparams.f_norm_rms_eps <= 0.0f) {
        throw std::runtime_error("T5Gemma2 requires a positive finite RMS epsilon");
    }

    if (hparams.n_layer() == 18 && hparams.n_embd == 640 &&
        hparams.n_ff() == 2048 && hparams.n_head() == 4 &&
        hparams.n_head_kv() == 1 && hparams.n_swa == 512) {
        type = LLM_TYPE_360M;
    } else if (hparams.n_layer() == 26 && hparams.n_embd == 1152 &&
               hparams.n_ff() == 6912 && hparams.n_head() == 4 &&
               hparams.n_head_kv() == 1 && hparams.n_swa == 512) {
        type = LLM_TYPE_1_7B;
    } else if (hparams.n_layer() == 34 && hparams.n_embd == 2560 &&
               hparams.n_ff() == 10240 && hparams.n_head() == 8 &&
               hparams.n_head_kv() == 4 && hparams.n_swa == 1024) {
        type = LLM_TYPE_7B;
    } else {
        type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_t5gemma2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.dec_start_token_id < 0 ||
        (uint64_t) hparams.dec_start_token_id >= (uint64_t) n_vocab) {
        throw std::runtime_error(format(
            "T5Gemma2 decoder_start_token_id %d is outside vocabulary size %lld",
            hparams.dec_start_token_id, (long long) n_vocab));
    }

    const int64_t n_embd_q = n_head * n_embd_head_k;

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), {n_embd, n_vocab}, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM,  "weight"), {n_embd}, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM,  "weight"), {n_embd}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(
            tn(LLM_TENSOR_TOKEN_EMBD, "weight"),
            {n_embd, n_vocab},
            TENSOR_DUPLICATED);
    }
    validate_t5gemma2_tensor_type(tok_embd, true);
    validate_t5gemma2_tensor_type(output_norm_enc, false);
    validate_t5gemma2_tensor_type(output_norm, false);
    validate_t5gemma2_tensor_type(output, true);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm_enc      = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_q_norm_enc    = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm_enc    = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_post_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.ffn_norm_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_enc        = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down_enc      = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_DEC_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);

        const ggml_tensor * norms[] = {
            layer.attn_norm_enc,
            layer.attn_q_norm_enc,
            layer.attn_k_norm_enc,
            layer.attn_post_norm_enc,
            layer.ffn_norm_enc,
            layer.ffn_post_norm_enc,
            layer.attn_norm,
            layer.attn_q_norm,
            layer.attn_k_norm,
            layer.attn_post_norm,
            layer.ffn_norm,
            layer.ffn_post_norm,
        };
        for (const ggml_tensor * tensor : norms) {
            validate_t5gemma2_tensor_type(tensor, false);
        }

        const ggml_tensor * matrices[] = {
            layer.wq_enc,
            layer.wk_enc,
            layer.wv_enc,
            layer.wo_enc,
            layer.ffn_gate_enc,
            layer.ffn_up_enc,
            layer.ffn_down_enc,
            layer.wq,
            layer.wk,
            layer.wv,
            layer.wo,
            layer.ffn_gate,
            layer.ffn_up,
            layer.ffn_down,
        };
        for (const ggml_tensor * tensor : matrices) {
            validate_t5gemma2_tensor_type(tensor, true);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_t5gemma2::build_arch_graph(
        const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid T5Gemma2 graph type");
    }
}

template <>
llama_model_t5gemma2::graph<true>::graph(
        const llama_model & model,
        const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(cparams.encoder_chunk_size > 0);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    // Transformers materializes the non-persistent scale buffer in the model
    // load dtype. The BF16 checkpoint therefore uses the BF16-rounded sqrt(d).
    const float hf_embedding_scale = ggml_bf16_to_fp32(
        ggml_fp32_to_bf16(hparams.f_embedding_scale));
    inpL = ggml_scale(
        ctx0, inpL, hf_embedding_scale/hparams.f_embedding_scale);
    // The checkpoint was trained with BF16 activations. Keep that execution
    // contract after the physical weights are block-quantized.
    const bool round_bf16 =
        ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    auto round_activation = [&](ggml_tensor * tensor) {
        return round_bf16
            ? ggml_cast(ctx0, ggml_cast(ctx0, tensor, GGML_TYPE_BF16), GGML_TYPE_F32)
            : tensor;
    };
    auto round_native = [&](ggml_tensor * tensor) {
        return cparams.encoder_chunk_size == 1 ? round_activation(tensor) : tensor;
    };
    // Match the HF residual dtype at block boundaries for encoder sequences
    // that extend beyond this model profile's local-attention window.
    auto round_long_encoder = [&](ggml_tensor * tensor) {
        return n_tokens > hparams.n_swa ? round_activation(tensor) : tensor;
    };
    const bool promote_cpu_weights =
        model.tok_embd->type == GGML_TYPE_BF16 &&
        ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
        loras->empty();
    auto mm = [&](ggml_tensor * weight, ggml_tensor * input) {
        return promote_cpu_weights && weight->type == GGML_TYPE_BF16
            ? ggml_mul_mat(ctx0, ggml_cast(ctx0, weight, GGML_TYPE_F32), input)
            : build_lora_mm(weight, input);
    };
    inpL = round_activation(inpL);
    cb(inpL, "enc_inp_scaled", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto inp_attn =
        std::make_unique<llm_graph_input_t5gemma2_encoder_attn>(hparams.n_swa);
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    inp_attn->full_mask = ggml_new_tensor_4d(
        ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    inp_attn->local_mask = ggml_new_tensor_4d(
        ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp_attn->full_mask);
    ggml_set_input(inp_attn->local_mask);
    auto * attn_inputs =
        (llm_graph_input_t5gemma2_encoder_attn *) res->add_input(std::move(inp_attn));

    for (uint32_t il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(
            inpL, model.layers[il].attn_norm_enc, nullptr, LLM_NORM_RMS, il);
        cur = round_native(cur);
        cb(cur, "enc_attn_norm", il);

        ggml_tensor * Qcur = round_native(mm(model.layers[il].wq_enc, cur));
        ggml_tensor * Kcur = round_native(mm(model.layers[il].wk_enc, cur));
        ggml_tensor * Vcur = round_native(mm(model.layers[il].wv_enc, cur));
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

        Qcur = build_norm(
            Qcur, model.layers[il].attn_q_norm_enc, nullptr, LLM_NORM_RMS, il);
        Qcur = round_native(Qcur);
        Kcur = build_norm(
            Kcur, model.layers[il].attn_k_norm_enc, nullptr, LLM_NORM_RMS, il);
        Kcur = round_native(Kcur);
        cb(Qcur, "enc_q_norm", il);
        cb(Kcur, "enc_k_norm", il);

        Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow);
        Qcur = round_native(Qcur);
        Kcur = round_native(Kcur);
        cb(Qcur, "enc_q_rope", il);
        cb(Kcur, "enc_k_rope", il);
        cb(Vcur, "enc_v", il);

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_tensor * mask = hparams.is_swa(il)
            ? attn_inputs->local_mask
            : attn_inputs->full_mask;
        cur = build_attn_mha(
            Qcur, Kcur, Vcur, nullptr, mask, nullptr, nullptr,
            n_tokens, hparams.f_attention_scale, il);
        cur = round_native(cur);
        cb(cur, "enc_attn_pre_o", il);
        cur = round_native(mm(model.layers[il].wo_enc, cur));
        cur = build_norm(
            cur, model.layers[il].attn_post_norm_enc, nullptr, LLM_NORM_RMS, il);
        cur = round_native(cur);
        cur = ggml_add(ctx0, cur, residual);
        cur = round_native(round_long_encoder(cur));
        cb(cur, "enc_sa_out", il);

        residual = cur;
        cur = build_norm(
            cur, model.layers[il].ffn_norm_enc, nullptr, LLM_NORM_RMS, il);
        cur = round_native(cur);
        ggml_tensor * ffn_up = round_native(mm(model.layers[il].ffn_up_enc, cur));
        ggml_tensor * ffn_gate = round_native(mm(model.layers[il].ffn_gate_enc, cur));
        ffn_gate = round_native(ggml_gelu(ctx0, ffn_gate));
        cur = round_native(ggml_mul(ctx0, ffn_gate, ffn_up));
        cur = round_native(mm(model.layers[il].ffn_down_enc, cur));
        cur = build_norm(
            cur, model.layers[il].ffn_post_norm_enc, nullptr, LLM_NORM_RMS, il);
        cur = round_native(cur);
        cur = ggml_add(ctx0, cur, residual);
        cur = round_native(round_long_encoder(cur));
        cb(cur, "enc_layer_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(
        inpL, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cur = round_native(cur);
    cur = round_native(round_long_encoder(cur));
    cb(cur, "enc_final_unpooled", -1);

    if (cparams.encoder_chunk_size == 1) {
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    const int64_t n_chunks =
        (n_tokens + cparams.encoder_chunk_size - 1)/cparams.encoder_chunk_size;
    auto inp_pool = std::make_unique<llm_graph_input_t5gemma2_chunk_pool>(
        cparams.encoder_chunk_size);
    inp_pool->weights = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, n_tokens, n_chunks);
    ggml_set_input(inp_pool->weights);
    auto * pool_input =
        (llm_graph_input_t5gemma2_chunk_pool *) res->add_input(std::move(inp_pool));

    cur = ggml_mul_mat(
        ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, cur)),
        pool_input->weights);
    cur = round_activation(cur);
    cb(cur, "enc_final_pooled", -1);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

template <>
llama_model_t5gemma2::graph<false>::graph(
        const llama_model & model,
        const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(cparams.encoder_chunk_size > 0);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    // Keep decoder embedding scaling identical to the encoder/HF buffer.
    const float hf_embedding_scale = ggml_bf16_to_fp32(
        ggml_fp32_to_bf16(hparams.f_embedding_scale));
    inpL = ggml_scale(
        ctx0, inpL, hf_embedding_scale/hparams.f_embedding_scale);
    // The checkpoint was trained with BF16 activations. Keep that execution
    // contract after the physical weights are block-quantized.
    const bool round_bf16 =
        ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    auto round_activation = [&](ggml_tensor * tensor) {
        return round_bf16
            ? ggml_cast(ctx0, ggml_cast(ctx0, tensor, GGML_TYPE_BF16), GGML_TYPE_F32)
            : tensor;
    };
    const bool promote_cpu_weights =
        model.tok_embd->type == GGML_TYPE_BF16 &&
        ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
        loras->empty();
    auto mm = [&](ggml_tensor * weight, ggml_tensor * input) {
        return promote_cpu_weights && weight->type == GGML_TYPE_BF16
            ? ggml_mul_mat(ctx0, ggml_cast(ctx0, weight, GGML_TYPE_F32), input)
            : build_lora_mm(weight, input);
    };
    inpL = round_activation(inpL);
    cb(inpL, "dec_inp_scaled", -1);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * embd_enc = build_inp_cross_embd();
    const int64_t n_enc = embd_enc->ne[1];

    auto inp_attn = std::make_unique<llm_graph_input_t5gemma2_merged_attn>(
        cross, hparams.n_swa, n_enc);
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    inp_attn->full_mask = ggml_new_tensor_4d(
        ctx0, type_mask, n_tokens + n_enc, n_tokens, 1, 1);
    inp_attn->local_mask = ggml_new_tensor_4d(
        ctx0, type_mask, n_tokens + n_enc, n_tokens, 1, 1);
    ggml_set_input(inp_attn->full_mask);
    ggml_set_input(inp_attn->local_mask);
    auto * attn_inputs =
        (llm_graph_input_t5gemma2_merged_attn *) res->add_input(std::move(inp_attn));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (uint32_t il = 0; il < hparams.dec_n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(
            inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = round_activation(cur);
        cb(cur, "dec_attn_norm", il);

        ggml_tensor * Qcur = round_activation(mm(model.layers[il].wq, cur));
        ggml_tensor * Kself = round_activation(mm(model.layers[il].wk, cur));
        ggml_tensor * Vself = round_activation(mm(model.layers[il].wv, cur));
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
        Kself = ggml_reshape_3d(ctx0, Kself, n_embd_head_k, n_head_kv, n_tokens);
        Vself = ggml_reshape_3d(ctx0, Vself, n_embd_head_v, n_head_kv, n_tokens);

        Qcur = build_norm(
            Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        Kself = build_norm(
            Kself, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Qcur = round_activation(Qcur);
        Kself = round_activation(Kself);
        Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow);
        Kself = ggml_rope_ext(
            ctx0, Kself, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
            ext_factor, attn_factor, beta_fast, beta_slow);
        Qcur = round_activation(Qcur);
        Kself = round_activation(Kself);
        cb(Qcur, "dec_q_rope", il);
        cb(Kself, "dec_self_k_rope", il);

        ggml_tensor * Kcross = round_activation(mm(model.layers[il].wk, embd_enc));
        ggml_tensor * Vcross = round_activation(mm(model.layers[il].wv, embd_enc));
        Kcross = ggml_reshape_3d(ctx0, Kcross, n_embd_head_k, n_head_kv, n_enc);
        Vcross = ggml_reshape_3d(ctx0, Vcross, n_embd_head_v, n_head_kv, n_enc);
        Kcross = build_norm(
            Kcross, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        Kcross = round_activation(Kcross);
        cb(Kcross, "dec_cross_k_norm", il);

        ggml_tensor * Kcur = ggml_concat(ctx0, Kself, Kcross, 2);
        ggml_tensor * Vcur = ggml_concat(ctx0, Vself, Vcross, 2);
        cb(Kcur, "dec_merged_k", il);
        cb(Vcur, "dec_merged_v", il);

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_tensor * mask = hparams.is_swa(il)
            ? attn_inputs->local_mask
            : attn_inputs->full_mask;
        cur = build_attn_mha(
            Qcur, Kcur, Vcur, nullptr, mask, nullptr, nullptr,
            n_tokens + n_enc, hparams.f_attention_scale, il);
        cur = round_activation(cur);
        cb(cur, "dec_attn_pre_o", il);
        cur = round_activation(mm(model.layers[il].wo, cur));
        cb(cur, "dec_attn_o", il);
        cur = build_norm(
            cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = round_activation(cur);
        cur = ggml_add(ctx0, cur, residual);
        cur = round_activation(cur);
        cb(cur, "dec_sa_out", il);

        residual = cur;
        cur = build_norm(
            cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = round_activation(cur);
        ggml_tensor * ffn_up = round_activation(mm(model.layers[il].ffn_up, cur));
        ggml_tensor * ffn_gate = round_activation(mm(model.layers[il].ffn_gate, cur));
        ffn_gate = ggml_gelu(ctx0, ffn_gate);
        ffn_gate = round_activation(ffn_gate);
        cur = round_activation(ggml_mul(ctx0, ffn_gate, ffn_up));
        cur = round_activation(mm(model.layers[il].ffn_down, cur));
        cur = build_norm(
            cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = round_activation(cur);
        cur = ggml_add(ctx0, cur, residual);
        cur = round_activation(cur);
        cb(cur, "dec_layer_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(
        inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cur = round_activation(cur);
    cb(cur, "dec_final_hidden", -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    res->t_embd = cur;
    if (cparams.embeddings) {
        ggml_build_forward_expand(gf, cur);
        return;
    }
    cur = promote_cpu_weights && model.output->type == GGML_TYPE_BF16
        ? ggml_mul_mat(ctx0, ggml_cast(ctx0, model.output, GGML_TYPE_F32), cur)
        : build_lora_mm(model.output, cur, model.output_s);
    cur = round_activation(cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_t5gemma2::graph<true>;
template struct llama_model_t5gemma2::graph<false>;
