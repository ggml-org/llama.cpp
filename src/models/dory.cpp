#include "models.h"

#include "gguf.h"

#include <cmath>
#include <cstring>

// Dory1.0 alternates attention and MLP modules at even and odd blk.N indices.
// Recurrent modules share weights across loop passes but use separate KV cache slots.
// Reference: src/transformers/models/dory/modeling_dory.py.

// Read dory.* metadata directly: the string-key get_key overload is only
// instantiated for enum llm_kv keys, so calling it here would not link.

namespace {

uint32_t dory_meta_u32(llama_model_loader & ml, const char * key, bool required, uint32_t fallback) {
    const int id = gguf_find_key(ml.metadata, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("dory: key not found in model: %s", key));
        }
        return fallback;
    }
    if (gguf_get_kv_type(ml.metadata, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(format("dory: key %s is not a uint32", key));
    }
    return gguf_get_val_u32(ml.metadata, id);
}

float dory_meta_f32(llama_model_loader & ml, const char * key, bool required, float fallback) {
    const int id = gguf_find_key(ml.metadata, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("dory: key not found in model: %s", key));
        }
        return fallback;
    }
    if (gguf_get_kv_type(ml.metadata, id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(format("dory: key %s is not a float32", key));
    }
    return gguf_get_val_f32(ml.metadata, id);
}

// returns an empty string when the key is absent and not required
std::string dory_meta_str(llama_model_loader & ml, const char * key, bool required) {
    const int id = gguf_find_key(ml.metadata, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("dory: key not found in model: %s", key));
        }
        return std::string();
    }
    if (gguf_get_kv_type(ml.metadata, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("dory: key %s is not a string", key));
    }
    return std::string(gguf_get_val_str(ml.metadata, id));
}

// L2Norm projects onto the unit sphere: x / sqrt(sum_rows(x^2) + 1e-6), not RMS.
// ggml_l2_norm clamps the denominator instead, so use the reference form.
// ggml_scale_bias supplies the additive epsilon without a constant tensor.
ggml_tensor * dory_l2_norm(ggml_context * ctx0, ggml_tensor * x) {
    static constexpr float eps = 1e-6f;

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_sqr(ctx0, x));   // [1, nrows]
    s = ggml_scale_bias(ctx0, s, 1.0f, eps);                    // sum + eps
    s = ggml_sqrt(ctx0, s);
    return ggml_div(ctx0, x, s);
}

// GGUF scale vectors are BF16; ggml binary ops do not mix BF16 with F32 activations.
ggml_tensor * dory_f32(ggml_context * ctx0, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32);
}

ggml_tensor * dory_residual_rate(ggml_context * ctx0, ggml_tensor * alpha, float lambda_scale) {
    return ggml_abs(ctx0, ggml_scale(ctx0, dory_f32(ctx0, alpha), lambda_scale));
}

// Interpolate from the normalized input: h_norm + rate * (cur - h_norm).
ggml_tensor * dory_residual(ggml_context * ctx0, ggml_tensor * h_norm, ggml_tensor * cur, ggml_tensor * rate) {
    // ggml takes the result shape from the first operand, so the token-shaped
    // difference must precede the per-channel rate.
    ggml_tensor * delta = ggml_sub(ctx0, cur, h_norm);
    return ggml_add(ctx0, h_norm, ggml_mul(ctx0, delta, rate));
}

} // namespace

// Map extra KV cache slots to their shared physical recurrent module.
int llama_model_dory::layer_phys(int il) const {
    return il < n_phys ? il : (int) rec_start + (il - n_phys) % (int) (rec_end - rec_start);
}

void llama_model_dory::load_arch_hparams(llama_model_loader & ml) {
    // block_count counts physical modules: input + recurrent + output.
    n_phys = (int) hparams.n_layer_all;

    n_input = dory_meta_u32(ml, "dory.input_layer_count",     true, 0);
    const uint32_t n_recurrent = dory_meta_u32(ml, "dory.recurrent_layer_count", true, 0);
    const uint32_t n_output    = dory_meta_u32(ml, "dory.output_layer_count",    true, 0);
    const uint32_t n_loops     = dory_meta_u32(ml, "dory.num_loops",             true, 0);

    rec_start = n_input;
    rec_end   = n_input + n_recurrent;

    if (n_loops < 1) {
        throw std::runtime_error("dory: num_loops must be >= 1");
    }
    if (rec_start == 0 || rec_start >= rec_end || rec_end > (uint32_t) n_phys) {
        throw std::runtime_error(format("dory: invalid recurrent group [%u, %u) for %d physical modules",
                rec_start, rec_end, n_phys));
    }
    if (rec_start + n_recurrent + n_output != (uint32_t) n_phys) {
        throw std::runtime_error(format("dory: input(%u) + recurrent(%u) + output(%u) modules != block_count(%d)",
                n_input, n_recurrent, n_output, n_phys));
    }
    if (n_phys % 2 != 0) {
        throw std::runtime_error(format("dory: block_count(%d) must be even (attention/MLP alternate)", n_phys));
    }

    // GGUF scale tensors are raw learned parameters; the graph applies the nGPT folds.
    // Dory has no LayerNorm/RMSNorm, so the eps hparams carry init_std and alpha_init.
    hparams.f_norm_eps     = dory_meta_f32(ml, "dory.init_std",   true,  0.0f);
    hparams.f_norm_rms_eps = dory_meta_f32(ml, "dory.alpha_init", true,  0.0f);
    if (!(hparams.f_norm_eps > 0.0f)) {
        throw std::runtime_error(format("dory: init_std must be > 0, got %g", hparams.f_norm_eps));
    }

    // Sliding modules use freq_base_2; full-attention modules use freq_base.
    hparams.rope_freq_base_train_swa = dory_meta_f32(ml, "dory.rope.freq_base_2", false,
            hparams.rope_freq_base_train_swa);

    // Standard causal sliding window.
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa == 0) {
        throw std::runtime_error("dory: sliding_window must be > 0");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // layer_pattern has one entry per attention module, mapped to even physical indices:
    // 'S' = sliding, '*' = full.
    const std::string pattern = dory_meta_str(ml, "dory.layer_pattern", false);
    const uint32_t n_attn_module = (uint32_t) n_phys / 2;
    if (!pattern.empty()) {
        std::vector<char> entries;
        for (size_t i = 0; i < pattern.size(); ++i) {
            const char c = pattern[i];
            if (c == '-' ) {
                continue;
            }
            entries.push_back(c);
        }
        if (entries.size() != n_attn_module) {
            throw std::runtime_error(format("dory: layer_pattern has %zu entries, expected %u",
                    entries.size(), n_attn_module));
        }
        for (uint32_t ia = 0; ia < n_attn_module; ++ia) {
            if (entries[ia] != 'S' && entries[ia] != '*') {
                throw std::runtime_error(format("dory: layer_pattern entry %u is '%c', expected 'S' or '*'",
                        ia, entries[ia]));
            }
            hparams.is_swa_impl[2*ia] = entries[ia] == 'S' ? 1 : 0;
        }
    } else {
        // Default to full attention on every fourth attention module.
        for (uint32_t ia = 0; ia < n_attn_module; ++ia) {
            hparams.is_swa_impl[2*ia] = ia % 4 == 3 ? 0 : 1;
        }
    }

    if (!hparams.is_swa_any()) {
        throw std::runtime_error("dory: no sliding-window module");
    }

    // Expand loop passes into KV cache slots; weights remain shared.
    const size_t n_slot = (size_t) n_phys + (size_t) (n_loops - 1)*(rec_end - rec_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("dory: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    hparams.n_layer_all = (uint32_t) n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]    = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il] = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]      = hparams.n_ff_arr[p];
        hparams.is_swa_impl[il]   = hparams.is_swa_impl[p];
    }

    switch (n_phys) {
        case 72: type = LLM_TYPE_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_dory::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,       "weight"), {n_embd, n_vocab}, 0);
    // Raw per-vocab logit scale; the graph applies logit_scale_factor.
    output_s = create_tensor(tn(LLM_TENSOR_OUTPUT_SCALE, "weight"), {n_vocab},         0);

    const int64_t n_embd_q = n_embd_head_k * n_head;

    // Only physical modules have tensors in the file.
    for (int i = 0; i < n_phys; ++i) {
        auto & layer = layers[i];

        if (i % 2 == 0) {
            // attention module
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_q},     0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd},     0);

            // gate_proj of the output gate, [n_embd -> n_embd_q]
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_q}, 0);

            // attn_ssmax_s is sqk and attn_q_norm is s_gate; both are raw per-channel scales.
            layer.attn_ssmax_s = create_tensor(tn(LLM_TENSOR_ATTN_QK_SCALE,   "weight", i), {n_embd_q}, 0);
            layer.attn_q_norm  = create_tensor(tn(LLM_TENSOR_ATTN_GATE_SCALE, "weight", i), {n_embd_q}, 0);

            layer.attn_resid_alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", i), {n_embd}, 0);
        } else {
            // MLP module
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);

            // suv_gate/suv_up scales, not generic weight scales; never pass them to build_lora_mm.
            layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SCALE, "weight", i), {n_ff}, 0);
            layer.ffn_up_s   = create_tensor(tn(LLM_TENSOR_FFN_UP_SCALE,   "weight", i), {n_ff}, 0);

            layer.ffn_resid_alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", i), {n_embd}, 0);
        }
    }

    // Loop slots share physical weights and own only their KV cache.
    for (int il = n_phys; il < (int) hparams.n_layer_all; ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_dory::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_dory::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_dory &>(model);

    // Recover the loop count from the expanded KV cache slots.
    const int n_recurrent = (int) (m.rec_end - m.rec_start);
    const int n_loops     = 1 + (int) (hparams.n_layer_all - m.n_phys) / n_recurrent;
    // The eps hparams carry dory.init_std and dory.alpha_init.
    const float init_std  = norm_eps;
    const float alpha     = norm_rms_eps;

    const float sqk_scale    = sqrtf((float) n_embd_head_k) / init_std;
    const float s_gate_scale = powf((float) n_embd, 0.25f) / init_std;
    const float suv_scale    = sqrtf((float) n_embd);
    const float lambda_scale = alpha / init_std;
    const float logit_factor = 1.0f / init_std;

    // Attention uses head_dim^-0.5 after q was scaled by (sqk * sqk_scale)^2.
    const float kq_scale = 1.0f / sqrtf((float) n_embd_head_k);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // il selects the KV cache slot, attention mask, and RoPE base;
    // ip selects the shared physical weights.
    auto build_module = [&](int il, int ip, ggml_tensor * inp) -> ggml_tensor * {
        if (il >= m.n_phys) {
            GGML_ASSERT(m.layer_phys(il) == ip);
        } else {
            GGML_ASSERT(il == ip);
        }

        const auto & layer = model.layers[ip];

        ggml_tensor * h = dory_l2_norm(ctx0, inp);
        cb(h, "module_inp_norm", il);

        ggml_tensor * cur;
        ggml_tensor * rate;

        if (ip % 2 == 0) {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, h);
            ggml_tensor * Kcur = build_lora_mm(layer.wk, h);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, h);
            ggml_tensor * Gcur = build_lora_mm(layer.wqkv_gate, h);   // [n_embd_q, n_tokens], not reshaped

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            // Unit-normalize q/k over head_dim before RoPE.
            Qcur = dory_l2_norm(ctx0, Qcur);
            Kcur = dory_l2_norm(ctx0, Kcur);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // q = q * (sqk * sqk_scale)^2, per (head, channel).
            {
                ggml_tensor * qk_scale = ggml_scale(ctx0, dory_f32(ctx0, layer.attn_ssmax_s), sqk_scale);
                qk_scale = ggml_reshape_2d(ctx0, qk_scale, n_embd_head_k, n_head);
                qk_scale = ggml_sqr(ctx0, qk_scale);
                Qcur = ggml_mul(ctx0, Qcur, qk_scale);
                cb(Qcur, "Qcur_scaled", il);
            }

            // Output projection is applied after gating and attn_sub_norm.
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // gate = sigmoid(gate_proj(x) * (s_gate * s_gate_scale)^2); the scale is squared.
            {
                ggml_tensor * gate_scale = ggml_scale(ctx0, dory_f32(ctx0, layer.attn_q_norm), s_gate_scale);
                gate_scale = ggml_sqr(ctx0, gate_scale);
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, Gcur, gate_scale));
                cb(gate, "attn_gate", il);
                cur = ggml_mul(ctx0, cur, gate);
                cb(cur, "attn_gated", il);
            }

            cur = dory_l2_norm(ctx0, cur);            // attn_sub_norm
            cur = build_lora_mm(layer.wo, cur);       // o_proj
            cur = dory_l2_norm(ctx0, cur);            // post_attention_layernorm
            cb(cur, "attn_out_norm", il);

            rate = dory_residual_rate(ctx0, layer.attn_resid_alpha, lambda_scale);
        } else {
            ggml_tensor * gate = build_lora_mm(layer.ffn_gate, h);
            gate = ggml_mul(ctx0, gate, ggml_scale(ctx0, dory_f32(ctx0, layer.ffn_gate_s), suv_scale));
            ggml_tensor * up = build_lora_mm(layer.ffn_up, h);
            up = ggml_mul(ctx0, up, ggml_scale(ctx0, dory_f32(ctx0, layer.ffn_up_s), suv_scale));

            cur = ggml_swiglu_split(ctx0, gate, up);  // silu(gate) * up
            cb(cur, "ffn_swiglu", il);

            cur = dory_l2_norm(ctx0, cur);            // ffn_sub_norm
            cur = build_lora_mm(layer.ffn_down, cur);
            cur = dory_l2_norm(ctx0, cur);            // post_feedforward_layernorm
            cb(cur, "ffn_out", il);

            rate = dory_residual_rate(ctx0, layer.ffn_resid_alpha, lambda_scale);
        }

        ggml_tensor * out = dory_residual(ctx0, h, cur, rate);
        cb(out, "l_out", il);
        return out;
    };

    // Input phase, applied once.
    for (int ip = 0; ip < (int) m.rec_start; ++ip) {
        inpL = build_module(ip, ip, inpL);
    }

    // Each loop restarts from the input phase output.
    ggml_tensor * input_hidden = dory_l2_norm(ctx0, inpL);
    cb(input_hidden, "input_hidden", -1);

    for (int pass = 0; pass < n_loops; ++pass) {
        // Reinject before every pass, including the first:
        // hidden = L2Norm(hidden) + input_hidden.
        // Pass 0 therefore sees 2 * input_hidden.
        inpL = ggml_add(ctx0, dory_l2_norm(ctx0, inpL), input_hidden);
        cb(inpL, "loop_reinject", pass);

        for (int ip = (int) m.rec_start; ip < (int) m.rec_end; ++ip) {
            const int il = pass == 0
                ? ip
                : m.n_phys + (pass - 1)*n_recurrent + (ip - (int) m.rec_start);
            inpL = build_module(il, ip, inpL);
        }
    }

    // Output phase, applied once.
    for (int ip = (int) m.rec_end; ip < m.n_phys; ++ip) {
        inpL = build_module(ip, ip, inpL);
    }

    if (inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    // Final L2Norm.
    ggml_tensor * cur = dory_l2_norm(ctx0, inpL);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // logits = (h @ W) * (logit_scale * logit_scale_factor)
    cur = build_lora_mm(model.output, cur);
    cur = ggml_mul(ctx0, cur, ggml_scale(ctx0, dory_f32(ctx0, model.output_s), logit_factor));
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
