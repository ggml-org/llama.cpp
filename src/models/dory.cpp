#include "models.h"

#include "gguf.h"

#include <cmath>
#include <cstring>

// dory: NVIDIA Dory1.0-4B, "Looped (latent-recurrent) normalized-GPT".
//
// A 72-module stack (16 input / 40 recurrent / 16 output). Modules alternate
// attention (even blk.N) and MLP (odd blk.N); there are 36 of each, and every
// module carries its own `residual_alpha`. The recurrent group is applied
// `num_loops` times (GGUF: 4) with the input-phase output reinjected before each
// pass, and each pass writes to its OWN KV cache.
//
// The loop is modelled as extra KV cache SLOTS after the physical modules
// (see docs/dory-port-contract.md, the inkling2-loop idiom): weights are shared
// through layer_phys(), caches are not.
//
// Arithmetic is taken from the reference, src/transformers/models/dory/
// modeling_dory.py (a copy lives at docs/modeling_dory.py). Line numbers below
// refer to that file.

// ---------------------------------------------------------------------------
// metadata helpers
//
// The dory.* keys have no LLM_KV enum. The loader's
// `get_key(const std::string &, T&, bool)` overload is declared in
// llama-model-loader.h but its definition lives in llama-model-loader.cpp and
// is only explicitly instantiated for the `enum llm_kv` form, so calling it with
// a literal key from this TU does not link. Reading the gguf metadata directly
// (llama_model_loader::metadata is public) is the linkable equivalent.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// nGPT ops
// ---------------------------------------------------------------------------

// L2Norm(x) = x * rsqrt(sum(x^2) + eps), eps = 1e-6: the projection onto the unit
// sphere, NOT RMS normalisation (modeling_dory.py:59-63). `ggml_l2_norm` guards
// the denominator instead (1/max(sqrt(sum), eps)), which is a different function
// for small sums, so the reference form is built explicitly here:
//   x / sqrt(sum_rows(x^2) + 1e-6)
// ggml_scale_bias is `s*a + b`, i.e. the additive eps without needing a constant
// tensor (this ggml has no scalar-constant constructor).
ggml_tensor * dory_l2_norm(ggml_context * ctx0, ggml_tensor * x) {
    static constexpr float eps = 1e-6f;

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_sqr(ctx0, x));   // [1, nrows]
    s = ggml_scale_bias(ctx0, s, 1.0f, eps);                    // sum + eps
    s = ggml_sqrt(ctx0, s);
    return ggml_div(ctx0, x, s);
}

// The per-channel scale vectors are stored as BF16 in the GGUF (they are 1-D
// tensors, so the quantiser left them at 16 bits), and ggml's binary ops do not
// mix bf16 with f32 activations: they are read as f32 here.
ggml_tensor * dory_f32(ggml_context * ctx0, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32);
}

// rate = |lambda * lambda_scale|, one value per hidden channel
// (modeling_dory.py:386, 391)
ggml_tensor * dory_residual_rate(ggml_context * ctx0, ggml_tensor * alpha, float lambda_scale) {
    return ggml_abs(ctx0, ggml_scale(ctx0, dory_f32(ctx0, alpha), lambda_scale));
}

// the residual of a module is an INTERPOLATION towards its output, and its base
// is the NORMALISED input, not the module input:
//   h = h_norm + rate * (f(h_norm) - h_norm)      (modeling_dory.py:387, 392)
ggml_tensor * dory_residual(ggml_context * ctx0, ggml_tensor * h_norm, ggml_tensor * cur, ggml_tensor * rate) {
    // note the operand order: ggml takes the result shape from the first operand,
    // so the [n_embd, n_tokens] difference has to come before the [n_embd] rate
    ggml_tensor * delta = ggml_sub(ctx0, cur, h_norm);
    return ggml_add(ctx0, h_norm, ggml_mul(ctx0, delta, rate));
}

} // namespace

// ---------------------------------------------------------------------------
// slot <-> physical module
// ---------------------------------------------------------------------------

int llama_model_dory::layer_phys(int il) const {
    return il < n_phys ? il : (int) rec_start + (il - n_phys) % (int) (rec_end - rec_start);
}

// ---------------------------------------------------------------------------
// hparams
// ---------------------------------------------------------------------------

void llama_model_dory::load_arch_hparams(llama_model_loader & ml) {
    // block_count counts PHYSICAL modules: 16 input + 40 recurrent + 16 output
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

    // the nGPT fold constants. Every `*_scale` tensor in the GGUF is the RAW
    // learned parameter, verified byte-exact against the BF16 reference
    // parameters (see the report): the port applies the fold itself.
    //   sqk_scale        = sqrt(head_dim)      / initializer_range
    //   s_gate_scale     = hidden_size^(1/4)   / initializer_range
    //   suv_scale        = sqrt(hidden_size)
    //   lambda_scale     = ngpt_alpha_init     / initializer_range
    //   logit_scale_factor= 1                  / initializer_range
    // dory has no LayerNorm/RMSNorm (its L2Norm eps is the fixed 1e-6), so the
    // two eps hparams are free to carry initializer_range and ngpt_alpha_init
    // into the graph; the graph reads them as norm_eps / norm_rms_eps.
    hparams.f_norm_eps     = dory_meta_f32(ml, "dory.init_std",   true,  0.0f);
    hparams.f_norm_rms_eps = dory_meta_f32(ml, "dory.alpha_init", true,  0.0f);
    if (!(hparams.f_norm_eps > 0.0f)) {
        throw std::runtime_error(format("dory: init_std must be > 0, got %g", hparams.f_norm_eps));
    }

    // dual RoPE: sliding modules use freq_base_2 (1e4), full modules freq_base
    // (1e7); llama_model::get_rope_freq_base picks between them by is_swa(il)
    hparams.rope_freq_base_train_swa = dory_meta_f32(ml, "dory.rope.freq_base_2", false,
            hparams.rope_freq_base_train_swa);

    // sliding window (1024), standard causal window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa == 0) {
        throw std::runtime_error("dory: sliding_window must be > 0");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // dory.layer_pattern is one character per ATTENTION module (36 of them, the
    // 72 physical modules alternate attention/MLP): 'S' = sliding, '*' = full.
    // Verified against config.json layer_types: full_attention falls on the
    // attention modules 3, 7, ... 35, i.e. physical modules 6, 14, ... 70.
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
        // config.json rule: full_attention on every 4th attention module
        for (uint32_t ia = 0; ia < n_attn_module; ++ia) {
            hparams.is_swa_impl[2*ia] = ia % 4 == 3 ? 0 : 1;
        }
    }

    if (!hparams.is_swa_any()) {
        throw std::runtime_error("dory: no sliding-window module");
    }

    // expand the loop into KV cache slots, exactly as inkling2-loop does: the
    // file holds the physical modules, the extra slots reuse them
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

// ---------------------------------------------------------------------------
// tensors
// ---------------------------------------------------------------------------

void llama_model_dory::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,       "weight"), {n_embd, n_vocab}, 0);
    // per-vocab logit scale (DoryLMHead.logit_scale); the fold factor is applied
    // in the graph, so this tensor holds the raw learned scale
    output_s = create_tensor(tn(LLM_TENSOR_OUTPUT_SCALE, "weight"), {n_vocab},         0);

    const int64_t n_embd_q = n_embd_head_k * n_head;

    // only the PHYSICAL modules are in the file (n_layer == n_layer_all == 192
    // here, the slots are copies made below)
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

            // raw per-channel scales (the fold is applied in the graph):
            //   attn_qk_scale   -> sqk      (multiplies q after rope, squared)
            //   attn_gate_scale -> s_gate   (inside the sigmoid, squared)
            layer.attn_ssmax_s = create_tensor(tn(LLM_TENSOR_ATTN_QK_SCALE,   "weight", i), {n_embd_q}, 0);
            layer.attn_q_norm  = create_tensor(tn(LLM_TENSOR_ATTN_GATE_SCALE, "weight", i), {n_embd_q}, 0);

            // per-hidden-channel interpolation rate (nGPT residual)
            layer.attn_resid_alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", i), {n_embd}, 0);
        } else {
            // MLP module
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);

            // raw per-branch scales (suv_gate / suv_up, folded with suv_scale in
            // the graph). They are NOT the generic per-tensor weight scales, so
            // they are never passed to build_lora_mm.
            layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SCALE, "weight", i), {n_ff}, 0);
            layer.ffn_up_s   = create_tensor(tn(LLM_TENSOR_FFN_UP_SCALE,   "weight", i), {n_ff}, 0);

            layer.ffn_resid_alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", i), {n_embd}, 0);
        }
    }

    // the loop slots share the physical weights and own only their KV cache
    for (int il = n_phys; il < (int) hparams.n_layer_all; ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

std::unique_ptr<llm_graph_context> llama_model_dory::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_dory::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_dory &>(model);

    // number of loop passes, recovered from the slot expansion:
    //   n_layer_all = n_phys + (n_loops - 1) * |recurrent group|
    const int n_recurrent = (int) (m.rec_end - m.rec_start);
    const int n_loops     = 1 + (int) (hparams.n_layer_all - m.n_phys) / n_recurrent;
    // nGPT fold constants (all `*_scale` tensors are raw, see load_arch_hparams)
    const float init_std  = norm_eps;       // dory.init_std     (initializer_range)
    const float alpha     = norm_rms_eps;   // dory.alpha_init   (ngpt_alpha_init)

    const float sqk_scale    = sqrtf((float) n_embd_head_k) / init_std;   // DoryAttention.sqk_scale
    const float s_gate_scale = powf((float) n_embd, 0.25f) / init_std;    // DoryAttention.s_gate_scale
    const float suv_scale    = sqrtf((float) n_embd);                     // DoryMLP.suv_scale
    const float lambda_scale = alpha / init_std;                          // DoryDecoderLayer.lambda_scale
    const float logit_factor = 1.0f / init_std;                           // DoryLMHead.logit_scale_factor

    // attention scaling is head_dim^-0.5 (DoryAttention.scaling), applied to the
    // q.k dot product; q itself was already rescaled by (sqk*sqk_scale)^2
    const float kq_scale = 1.0f / sqrtf((float) n_embd_head_k);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // one physical module = one op. `il` is the SLOT (it owns the KV cache and
    // selects the sliding/full mask + rope base); `ip` is the PHYSICAL module
    // (it owns the weights).
    auto build_module = [&](int il, int ip, ggml_tensor * inp) -> ggml_tensor * {
        if (il >= m.n_phys) {
            GGML_ASSERT(m.layer_phys(il) == ip);
        } else {
            GGML_ASSERT(il == ip);
        }

        const auto & layer = model.layers[ip];

        // input layernorm (modeling_dory.py:377): inside the attention module
        ggml_tensor * h = dory_l2_norm(ctx0, inp);
        cb(h, "module_inp_norm", il);

        ggml_tensor * cur;
        ggml_tensor * rate;

        if (ip % 2 == 0) {
            // ---------------- attention module (DoryAttention.forward) --------
            ggml_tensor * Qcur = build_lora_mm(layer.wq, h);
            ggml_tensor * Kcur = build_lora_mm(layer.wk, h);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, h);
            ggml_tensor * Gcur = build_lora_mm(layer.wqkv_gate, h);   // [n_embd_q, n_tokens], not reshaped

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            // unit-norm q/k over head_dim, BEFORE rope (modeling_dory.py:293-294)
            Qcur = dory_l2_norm(ctx0, Qcur);
            Kcur = dory_l2_norm(ctx0, Kcur);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // q = q * (sqk * sqk_scale)^2, per (head, channel) (modeling_dory.py:299-300)
            {
                ggml_tensor * qk_scale = ggml_scale(ctx0, dory_f32(ctx0, layer.attn_ssmax_s), sqk_scale);
                qk_scale = ggml_reshape_2d(ctx0, qk_scale, n_embd_head_k, n_head);
                qk_scale = ggml_sqr(ctx0, qk_scale);
                Qcur = ggml_mul(ctx0, Qcur, qk_scale);
                cb(Qcur, "Qcur_scaled", il);
            }

            // no output projection here: the attn_sub_norm sits between
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr, Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // gate = sigmoid(gate_proj(x) * (s_gate * s_gate_scale)^2)
            // (modeling_dory.py:322). Note: the SCALE is squared, not the product.
            {
                ggml_tensor * gate_scale = ggml_scale(ctx0, dory_f32(ctx0, layer.attn_q_norm), s_gate_scale);
                gate_scale = ggml_sqr(ctx0, gate_scale);
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, Gcur, gate_scale));
                cb(gate, "attn_gate", il);
                cur = ggml_mul(ctx0, cur, gate);
                cb(cur, "attn_gated", il);
            }

            cur = dory_l2_norm(ctx0, cur);            // attn_sub_norm (line 323)
            cur = build_lora_mm(layer.wo, cur);       // o_proj
            cur = dory_l2_norm(ctx0, cur);            // post_attention_layernorm (line 385)
            cb(cur, "attn_out_norm", il);

            rate = dory_residual_rate(ctx0, layer.attn_resid_alpha, lambda_scale);
        } else {
            // ---------------- MLP module (DoryMLP.forward) --------------------
            ggml_tensor * gate = build_lora_mm(layer.ffn_gate, h);
            gate = ggml_mul(ctx0, gate, ggml_scale(ctx0, dory_f32(ctx0, layer.ffn_gate_s), suv_scale));
            ggml_tensor * up = build_lora_mm(layer.ffn_up, h);
            up = ggml_mul(ctx0, up, ggml_scale(ctx0, dory_f32(ctx0, layer.ffn_up_s), suv_scale));

            cur = ggml_swiglu_split(ctx0, gate, up);  // silu(gate) * up
            cb(cur, "ffn_swiglu", il);

            cur = dory_l2_norm(ctx0, cur);            // ffn_sub_norm (line 345)
            cur = build_lora_mm(layer.ffn_down, cur);
            cur = dory_l2_norm(ctx0, cur);            // post_feedforward_layernorm (line 390)
            cb(cur, "ffn_out", il);

            rate = dory_residual_rate(ctx0, layer.ffn_resid_alpha, lambda_scale);
        }

        // the residual is an interpolation whose base is the NORMALISED input
        ggml_tensor * out = dory_residual(ctx0, h, cur, rate);
        cb(out, "l_out", il);
        return out;
    };

    // input phase, applied once
    for (int ip = 0; ip < (int) m.rec_start; ++ip) {
        inpL = build_module(ip, ip, inpL);
    }

    // the loops restart from the input section's output (modeling_dory.py:541)
    ggml_tensor * input_hidden = dory_l2_norm(ctx0, inpL);
    cb(input_hidden, "input_hidden", -1);

    for (int pass = 0; pass < n_loops; ++pass) {
        // reinjection, once per pass and before ANY slot of that pass, INCLUDING
        // the first one (modeling_dory.py:542-543):
        //   for loop_idx in range(n_loops):
        //       hidden_states = norm(hidden_states) + input_hidden_states
        // so at pass 0 the recurrent group already sees 2 * input_hidden
        inpL = ggml_add(ctx0, dory_l2_norm(ctx0, inpL), input_hidden);
        cb(inpL, "loop_reinject", pass);

        for (int ip = (int) m.rec_start; ip < (int) m.rec_end; ++ip) {
            const int il = pass == 0
                ? ip
                : m.n_phys + (pass - 1)*n_recurrent + (ip - (int) m.rec_start);
            inpL = build_module(il, ip, inpL);
        }
    }

    // output phase, applied once
    for (int ip = (int) m.rec_end; ip < m.n_phys; ++ip) {
        inpL = build_module(ip, ip, inpL);
    }

    if (inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    // final L2Norm (modeling_dory.py:566)
    ggml_tensor * cur = dory_l2_norm(ctx0, inpL);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // logits = (h @ W) * (logit_scale * logit_scale_factor) (modeling_dory.py:404-406)
    cur = build_lora_mm(model.output, cur);
    cur = ggml_mul(ctx0, cur, ggml_scale(ctx0, dory_f32(ctx0, model.output_s), logit_factor));
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
