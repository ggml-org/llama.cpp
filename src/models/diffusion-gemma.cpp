#include "models.h"
#include "gemma4-common.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// DiffusionGemma: block text-diffusion MoE on a Gemma-4 backbone. A single no-cache bidirectional
// forward over [prompt | canvas] reproduces the two-pass (causal encoder prefill + bidirectional
// decoder denoise, zero self-conditioning) result. Three things are region-aware, split at
// P = n_tokens - canvas_length (canvas = the last canvas_length positions):
//   1. input embeddings: prompt = embed*sqrt(n_embd); canvas = rmsnorm_noscale(same)
//   2. per-layer scalar: prompt = encoder scalar; canvas = decoder scalar
//   3. attention mask: prompt causal over prompt only; canvas bidirectional over all prompt+canvas
// The Gemma-4 backbone is identical to gemma4 (shared via gemma4-common.h).

// Region-aware additive mask for the unified [prompt | canvas] forward. Prompt queries are causal
// (SWA-clipped in sliding layers); canvas queries are bidirectional. Canvas->prompt reach: global
// layers see all prompt, sliding layers only the last (n_swa-1) prompt positions.
class llm_graph_input_attn_diffusion : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion(const llama_hparams & hparams, const llama_cparams & cparams,
                                   int64_t n_prompt) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt) {}
    ~llm_graph_input_attn_diffusion() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t P        = n_prompt;

        // swa clips keys outside the sliding window, but only for prompt (causal) queries.
        const auto fill = [&](auto * data, bool swa) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n_tokens * n_tokens, llama_cast<T>(-INFINITY));
            for (int64_t q = 0; q < n_tokens; ++q) {
                const bool q_is_canvas = q >= P;
                const uint64_t row = q * n_tokens;
                // canvas->prompt sliding bound: last (n_swa-1) prompt positions (<= 0 for short prompts)
                const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;
                for (int64_t k = 0; k < n_tokens; ++k) {
                    const bool k_is_canvas = k >= P;
                    bool allow;
                    if (q_is_canvas) {
                        if (swa) {
                            // sliding: last (n_swa-1) prompt + all canvas
                            allow = k_is_canvas || (k >= canvas_prompt_lo);
                        } else {
                            allow = true; // global: all prompt + canvas
                        }
                    } else {
                        // prompt query: causal over earlier prompt, never canvas
                        allow = (!k_is_canvas) && (k <= q);
                    }
                    if (allow && swa && !q_is_canvas &&
                        llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, k, q)) {
                        allow = false;
                    }
                    if (allow) {
                        data[row + k] = llama_cast<T>(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data, false);
        } else {
            fill((float *) self_kq_mask->data, false);
        }

        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) {
                fill((ggml_fp16_t *) self_kq_mask_swa->data, true);
            } else {
                fill((float *) self_kq_mask_swa->data, true);
            }
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    int64_t n_prompt;
};

// Self-conditioning input: uploads the previous step's raw logits [n_vocab, C] for the canvas embedding.
class llm_graph_input_sc : public llm_graph_input_i {
public:
    llm_graph_input_sc(const float * src, int64_t n_vocab, int64_t C) :
        src(src), n_vocab(n_vocab), C(C) {}
    ~llm_graph_input_sc() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (sc_logits && src) {
            GGML_ASSERT(ggml_nelements(sc_logits) == n_vocab * C);
            ggml_backend_tensor_set(sc_logits, src, 0, (size_t) n_vocab * C * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    ggml_tensor * sc_logits = nullptr;
    const float * src;
    int64_t       n_vocab;
    int64_t       C;
};

// Decode-phase mask (prompt-KV caching): canvas queries over [cached prompt (first P) | fresh canvas
// (last C)], rectangular [P+C, C]. Global sees all prompt; sliding the last (n_swa-1) prompt.
class llm_graph_input_attn_diffusion_decode : public llm_graph_input_attn_no_cache {
public:
    llm_graph_input_attn_diffusion_decode(const llama_hparams & hparams, const llama_cparams & cparams,
                                          int64_t n_prompt, int64_t n_canvas) :
        llm_graph_input_attn_no_cache(hparams, cparams), n_prompt(n_prompt), n_canvas(n_canvas) {}
    ~llm_graph_input_attn_diffusion_decode() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        const int64_t P    = n_prompt;
        const int64_t C    = n_canvas;
        const int64_t n_kv = P + C;
        const int64_t canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1;

        const auto fill = [&](auto * data, bool swa) {
            using T = std::remove_reference_t<decltype(*data)>;
            std::fill(data, data + n_kv * C, llama_cast<T>(-INFINITY));
            for (int64_t q = 0; q < C; ++q) {            // canvas query (position P+q)
                const uint64_t row = q * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) {     // key: k<P prompt (pos k), else canvas
                    bool allow;
                    if (k < P) {
                        allow = swa ? (k >= canvas_prompt_lo) : true;
                    } else {
                        allow = true;                     // bidirectional over the canvas
                    }
                    if (allow) {
                        data[row + k] = llama_cast<T>(0.0f);
                    }
                }
            }
        };

        GGML_ASSERT(self_kq_mask && ggml_backend_buffer_is_host(self_kq_mask->buffer));
        if (self_kq_mask->type == GGML_TYPE_F16) {
            fill((ggml_fp16_t *) self_kq_mask->data, false);
        } else {
            fill((float *) self_kq_mask->data, false);
        }
        if (self_kq_mask_swa) {
            GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
            if (self_kq_mask_swa->type == GGML_TYPE_F16) {
                fill((ggml_fp16_t *) self_kq_mask_swa->data, true);
            } else {
                fill((float *) self_kq_mask_swa->data, true);
            }
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override { return false; }

    int64_t n_prompt;
    int64_t n_canvas;
};

void llama_model_diffusion_gemma::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    // bidirectional decoder; the forward fills its own region-aware mask
    hparams.causal_attn = false;

    hparams.f_attention_scale = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // canvas_length splits the forward (P = n_tokens - canvas_length); must be positive
    ml.get_key(std::string("diffusion.canvas_length"), canvas_length, true);
    if (canvas_length <= 0) {
        throw std::runtime_error("DiffusionGemma requires a positive diffusion.canvas_length");
    }

    switch (hparams.n_layer()) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_diffusion_gemma::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("DiffusionGemma requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("DiffusionGemma requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // lm_head is tied to the (decoder) token embeddings
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // self-conditioning gated MLP (optional; unused in the zero-SC exactness forward)
    sc_pre_norm = create_tensor(tn(LLM_TENSOR_SC_PRE_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    sc_gate     = create_tensor(tn(LLM_TENSOR_SC_GATE,     "weight"), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
    sc_up       = create_tensor(tn(LLM_TENSOR_SC_UP,       "weight"), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
    sc_down     = create_tensor(tn(LLM_TENSOR_SC_DOWN,     "weight"), {n_ff, n_embd}, TENSOR_NOT_REQUIRED);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // global layers have no v_proj -> value reuses k_proj
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // per-layer scalars: decoder (canvas) + encoder (prompt)
        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE,     "weight", i), {1u}, 0);
        layer.enc_out_scale = create_tensor(tn(LLM_TENSOR_ENC_LAYER_OUT_SCALE, "weight", i), {1u}, 0);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // dense gated MLP = the shared expert
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router + experts
        layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale",  i), {n_embd}, 0);

        layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
        layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
        layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);
        if (layer.ffn_gate_up_exps == nullptr) {
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        }
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        // per-expert scale (router.per_expert_scale) is loaded as ffn_down_exps_s
    }
}

// fwd decl: lazily build the transposed/dequantized SC soft-embedding weight (defined below)
static void dg_ensure_sc_embT(const llama_model_diffusion_gemma & m);
// fwd decl: lazily (re)allocate the device-resident prev-step canvas-logits buffer for device SC (defined below)
static void dg_ensure_sc_dev(const llama_model_diffusion_gemma & m, int64_t C);

std::unique_ptr<llm_graph_context> llama_model_diffusion_gemma::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_diffusion_gemma::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    const auto & dmodel = (const llama_model_diffusion_gemma &) model;
    const auto   phase  = dmodel.pkv_phase;
    const bool is_prefill = (phase == llama_model_diffusion_gemma::PKV_PREFILL);
    const bool is_decode  = (phase == llama_model_diffusion_gemma::PKV_DECODE);

    const int64_t canvas_length = dmodel.canvas_length;
    // Region split P|C. UNIFIED: prompt = first (n_tokens - canvas_length), canvas = last canvas_length
    // (P=0 for tiny warmup graphs). PREFILL: all prompt. DECODE: all canvas, P = cached prompt length.
    int64_t P, C;
    if (is_decode) {
        P = dmodel.pkv_P;
        C = n_tokens;
    } else if (is_prefill) {
        P = n_tokens;
        C = 0;
    } else {
        P = (canvas_length > 0 && n_tokens > canvas_length) ? (n_tokens - canvas_length) : 0;
        C = n_tokens - P;
    }

    // guard the prompt-KV store is allocated and large enough (misuse fails loudly, not OOB)
    if (is_prefill || is_decode) {
        const int64_t need = is_prefill ? n_tokens : P;
        GGML_ASSERT(!dmodel.pkv_k.empty() && !dmodel.pkv_v.empty() && dmodel.pkv_cap >= need &&
                    "DiffusionGemma prompt-KV store not allocated/sized for this phase");
    }

    // Canvas input embedding = rms_norm_noscale(embed*sqrt(n_embd) [+ self-conditioning]). Shared by
    // UNIFIED canvas rows and the DECODE batch. Zero SC -> exactness forward.
    auto dg_canvas_embed = [&](ggml_tensor * canvas) -> ggml_tensor * {
        // build the SC subgraph whenever SC is enabled (reserve covers it; upload only when a buffer is set)
        if (dmodel.sc_enabled) {
            const int64_t Cc      = canvas->ne[1];
            const int64_t n_vocab = model.tok_embd->ne[1];
            canvas = ggml_cont(ctx0, canvas);

            // previous step's raw logits [n_vocab, Cc]
            ggml_tensor * sc_logits;
            if (dmodel.sc_device_resident) {
                // device-resident: read the persistent sc_dev buffer the prior step's lm_head wrote into
                // (see the ggml_cpy after t_logits below). No host input/upload. Values are identical to
                // the host path, so the SC math and the forward stay bit-for-bit the same.
                dg_ensure_sc_dev(dmodel, Cc);
                sc_logits = dmodel.sc_dev;
            } else {
                auto inp_sc = std::make_unique<llm_graph_input_sc>(dmodel.sc_logits_ptr, n_vocab, Cc);
                inp_sc->sc_logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, Cc);
                ggml_set_input(inp_sc->sc_logits);
                sc_logits = inp_sc->sc_logits;
                res->add_input(std::move(inp_sc));
            }

            // raw/temperature, then softmax over vocab (fp32)
            ggml_tensor * probs = ggml_soft_max(ctx0, ggml_scale(ctx0, sc_logits, dmodel.sc_temp_inv));
            // soft_emb = (softmax @ embed_tokens) * embed_scale. sc_embT is the embedding transposed/
            // dequantized once on-device (dg_ensure_sc_embT) so the per-step cost is one matmul vs a weight.
            dg_ensure_sc_embT(dmodel);
            ggml_tensor * soft = ggml_mul_mat(ctx0, dmodel.sc_embT, probs);             // [n_embd, Cc]
            soft = ggml_scale(ctx0, soft, sqrtf((float) n_embd));
            // SC gated MLP: pre_norm (plain weight) -> down( gelu_tanh(gate) * up )
            ggml_tensor * normed = build_norm(soft, model.sc_pre_norm, nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * g  = ggml_gelu(ctx0, ggml_mul_mat(ctx0, model.sc_gate, normed)); // [n_ff, Cc]
            ggml_tensor * u  = ggml_mul_mat(ctx0, model.sc_up, normed);                    // [n_ff, Cc]
            ggml_tensor * sc_sig = ggml_mul_mat(ctx0, model.sc_down, ggml_mul(ctx0, g, u)); // [n_embd, Cc]
            sc_sig = ggml_scale(ctx0, sc_sig, dmodel.sc_use); // runtime {0,1} gate (0 == first step)
            canvas = ggml_add(ctx0, canvas, sc_sig);
            canvas = ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps); // post_norm, no scale
        } else {
            canvas = ggml_rms_norm(ctx0, canvas, hparams.f_norm_rms_eps); // no scale (zero-SC)
        }
        return canvas;
    };

    inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_scale(ctx0, inpL, sqrtf((float) n_embd)); // embed_scale = sqrt(n_embd) (ScaledWordEmbedding)
    cb(inpL, "inp_scaled", -1);

    if (is_prefill) {
        // prompt-only (encoder): scaled embedding feeds the layers directly
    } else if (is_decode) {
        // canvas-only (decoder)
        inpL = dg_canvas_embed(inpL);
    } else if (P > 0 && P < n_tokens) {
        ggml_tensor * prompt = ggml_view_2d(ctx0, inpL, n_embd, P, inpL->nb[1], 0);
        ggml_tensor * canvas = ggml_view_2d(ctx0, inpL, n_embd, C, inpL->nb[1], P * inpL->nb[1]);
        canvas = dg_canvas_embed(canvas);
        inpL = ggml_concat(ctx0, ggml_cont(ctx0, prompt), ggml_cont(ctx0, canvas), 1);
    } else {
        // pure-canvas (no prompt) path
        inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    }
    cb(inpL, "inp_region", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // region-aware no-cache mask. DECODE: rectangular [P+C keys, C queries] over cached prompt + fresh
    // canvas K,V. UNIFIED/PREFILL: square [n_tokens, n_tokens] (PREFILL has P=n_tokens, all causal rows).
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    llm_graph_input_attn_no_cache * inp_attn = nullptr;
    if (is_decode) {
        const int64_t n_kv = P + C;
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion_decode>(hparams, cparams, P, C);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, C, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_kv, C, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            uptr->self_kq_mask_swa_cnv = uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    } else {
        auto uptr = std::make_unique<llm_graph_input_attn_diffusion>(hparams, cparams, P);
        uptr->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(uptr->self_kq_mask);
        uptr->self_kq_mask_cnv = uptr->self_kq_mask;
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            uptr->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
            ggml_set_input(uptr->self_kq_mask_swa);
            uptr->self_kq_mask_swa_cnv = uptr->self_kq_mask_swa;
        }
        inp_attn = (llm_graph_input_attn_no_cache *) res->add_input(std::move(uptr));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));
        const int64_t n_head_kv = hparams.n_head_kv(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Q/K/V + q/k-norm + partial rope: shared Gemma-4 backbone (gemma4-common.h)
        ggml_tensor * Qcur = gemma4_build_q(*this, model, cur, inp_pos, il);
        llm_graph_qkv kv   = gemma4_build_kv(*this, model, cur, inp_pos, il);
        ggml_tensor * Kcur = kv.k;
        ggml_tensor * Vcur = kv.v;

        if (is_prefill) {
            // PREFILL: persist this layer's prompt K,V (F32) into the store for the block's DECODE steps
            ggml_tensor * sk = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv, n_tokens,
                                            dmodel.pkv_k[il]->nb[1], dmodel.pkv_k[il]->nb[2], 0);
            ggml_tensor * sv = ggml_view_3d(ctx0, dmodel.pkv_v[il], n_embd_head, n_head_kv, n_tokens,
                                            dmodel.pkv_v[il]->nb[1], dmodel.pkv_v[il]->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, sk));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, sv));
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        } else if (is_decode) {
            // DECODE: prepend cached prompt K,V (first P) to the fresh canvas K,V
            ggml_tensor * pk = ggml_view_3d(ctx0, dmodel.pkv_k[il], n_embd_head, n_head_kv, P,
                                            dmodel.pkv_k[il]->nb[1], dmodel.pkv_k[il]->nb[2], 0);
            ggml_tensor * pv = ggml_view_3d(ctx0, dmodel.pkv_v[il], n_embd_head, n_head_kv, P,
                                            dmodel.pkv_v[il]->nb[1], dmodel.pkv_v[il]->nb[2], 0);
            ggml_tensor * Kfull = ggml_concat(ctx0, pk, Kcur, 2);
            ggml_tensor * Vfull = ggml_concat(ctx0, pv, Vcur, 2);
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                             Qcur, Kfull, Vfull, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        } else {
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                             hparams.f_attention_scale, il);
        }

        // unlike AR gemma4, no inp_out_ids row-selection mid-stack: every canvas row's logits are needed

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // dense-MLP + 128-expert MoE + ffn_post_norm + residual: shared Gemma-4 backbone (gemma4-common.h)
        cur = gemma4_build_ffn_moe(*this, model, attn_out, il);

        // region-aware per-layer scalar: prompt * encoder scalar, canvas * decoder scalar
        if (is_prefill) {
            cur = ggml_mul(ctx0, cur, model.layers[il].enc_out_scale);
        } else if (is_decode) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        } else if (P > 0 && P < n_tokens) {
            ggml_tensor * prompt = ggml_view_2d(ctx0, cur, n_embd, P, cur->nb[1], 0);
            ggml_tensor * canvas = ggml_view_2d(ctx0, cur, n_embd, C, cur->nb[1], P * cur->nb[1]);
            prompt = ggml_mul(ctx0, ggml_cont(ctx0, prompt), model.layers[il].enc_out_scale);
            canvas = ggml_mul(ctx0, ggml_cont(ctx0, canvas), model.layers[il].out_scale);
            cur = ggml_concat(ctx0, prompt, canvas, 1);
        } else {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
        }
        cb(cur, "out_scaled", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // device SC: persist this step's canvas logits (last C rows) into sc_dev for the next step to read on-
    // device, no host round-trip. The cpy is downstream of the SC read in the DAG, so topo order runs
    // read-before-write on the single buffer (no hazard). F32->F32 = the exact bytes the host path uploads.
    if (dmodel.sc_enabled && dmodel.sc_device_resident && C > 0) {
        dg_ensure_sc_dev(dmodel, C);
        const int64_t n_out   = cur->ne[1];
        const int64_t n_vocab = cur->ne[0];
        GGML_ASSERT(n_out >= C && "device SC expects the canvas rows to be present in the output logits");
        ggml_tensor * canvas_logits = ggml_view_2d(ctx0, cur, n_vocab, C,
                                                   cur->nb[1], (n_out - C) * cur->nb[1]);
        ggml_tensor * sc_cpy = ggml_cpy(ctx0, canvas_logits, dmodel.sc_dev);
        ggml_build_forward_expand(gf, sc_cpy);
    }
}

// Public API: set per-request self-conditioning (no-op on other models). sc_logits is borrowed and
// must stay valid through the next llama_decode (when the graph uploads it).
void llama_diffusion_set_sc(struct llama_model * model, const float * sc_logits,
                            float use_sc, float temp_inv, bool enabled) {
    auto * dm = dynamic_cast<llama_model_diffusion_gemma *>(model);
    if (!dm) {
        return;
    }
    dm->sc_logits_ptr = sc_logits;
    dm->sc_use        = use_sc;
    dm->sc_temp_inv   = temp_inv;
    dm->sc_enabled    = enabled;
}

// Public API: opt into device-resident self-conditioning (no-op on other models). See llama.h.
void llama_diffusion_set_device_sc(struct llama_model * model, bool enabled) {
    auto * dm = dynamic_cast<llama_model_diffusion_gemma *>(model);
    if (!dm) {
        return;
    }
    dm->sc_device_resident = enabled;
}

// Debug only: copy the device SC buffer (sc_dev) to host; returns floats copied (0 if none). Verifies
// sc_dev == the canvas logits the host path would upload. Off the hot path.
size_t llama_diffusion_debug_get_sc_dev(const struct llama_model * model, float * dst, size_t max_elems) {
    const auto * dm = dynamic_cast<const llama_model_diffusion_gemma *>(model);
    if (!dm || dm->sc_dev == nullptr || dst == nullptr) {
        return 0;
    }
    const size_t n = std::min(max_elems, (size_t) ggml_nelements(dm->sc_dev));
    ggml_backend_tensor_get(dm->sc_dev, dst, 0, n * sizeof(float));
    return n;
}

// Stage-1 device sampling entry. Fetches the CUDA backend's dense sampler via the backend-reg proc address
// (keeps the llama<->ggml-cuda link at the existing backend boundary) and runs it on sc_dev. Returns false
// for non-DiffusionGemma / no sc_dev / non-CUDA builds so the caller falls back to the host path.
typedef bool (*dg_cuda_sample_fn)(struct ggml_tensor *, const float *, int *, float *, int *, int, float);

bool llama_diffusion_device_sample(const struct llama_model * model, const float * u, int * argmax,
                                   float * entropy, int * sampled, int n_tokens, float inv_temp) {
    const auto * dm = dynamic_cast<const llama_model_diffusion_gemma *>(model);
    if (!dm || dm->sc_dev == nullptr || !u || !argmax || !entropy || !sampled || n_tokens <= 0) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        return false;
    }
    static dg_cuda_sample_fn fn =
        (dg_cuda_sample_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diffusion_sample");
    if (!fn) {
        return false;
    }
    return fn(dm->sc_dev, u, argmax, entropy, sampled, n_tokens, inv_temp);
}

llama_model_diffusion_gemma::~llama_model_diffusion_gemma() {
    if (pkv_buf) { ggml_backend_buffer_free(pkv_buf); pkv_buf = nullptr; }
    if (pkv_ctx) { ggml_free(pkv_ctx); pkv_ctx = nullptr; }
    if (sc_embT_buf) { ggml_backend_buffer_free(sc_embT_buf); sc_embT_buf = nullptr; }
    if (sc_embT_ctx) { ggml_free(sc_embT_ctx); sc_embT_ctx = nullptr; }
    if (sc_dev_buf)  { ggml_backend_buffer_free(sc_dev_buf);  sc_dev_buf  = nullptr; }
    if (sc_dev_ctx)  { ggml_free(sc_dev_ctx);  sc_dev_ctx  = nullptr; }
}

// Build the SC soft-embedding weight once: tok_embd dequantized + transposed to [n_vocab, n_embd] F16
// in a device weights buffer, so the per-step SC matmul runs on-device instead of on the CPU.
static void dg_ensure_sc_embT(const llama_model_diffusion_gemma & m) {
    if (m.sc_embT != nullptr) {
        return;
    }
    ggml_tensor * src = m.tok_embd;
    GGML_ASSERT(src != nullptr);
    const int64_t n_embd  = src->ne[0];
    const int64_t n_vocab = src->ne[1];

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    m.sc_embT_ctx = ggml_init(ip);
    GGML_ASSERT(m.sc_embT_ctx != nullptr);
    m.sc_embT = ggml_new_tensor_2d(m.sc_embT_ctx, GGML_TYPE_F16, n_vocab, n_embd);
    ggml_set_name(m.sc_embT, "sc_embT");

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
    m.sc_embT_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.sc_embT_ctx, buft);
    GGML_ASSERT(m.sc_embT_buf != nullptr);
    ggml_backend_buffer_set_usage(m.sc_embT_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // dequantize each row on the host and scatter the transpose into F16
    const ggml_type st       = src->type;
    const size_t    row_size = ggml_row_size(st, n_embd);
    std::vector<char> host_src((size_t) row_size * n_vocab);
    ggml_backend_tensor_get(src, host_src.data(), 0, host_src.size());

    std::vector<ggml_fp16_t> dstT((size_t) n_vocab * n_embd);
    const ggml_type_traits * tr = ggml_get_type_traits(st);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
    auto worker = [&](int64_t v0, int64_t v1) {
        std::vector<float> tmp(n_embd);
        for (int64_t v = v0; v < v1; ++v) {
            const char * row = host_src.data() + (size_t) v * row_size;
            if (st == GGML_TYPE_F32) {
                std::memcpy(tmp.data(), row, (size_t) n_embd * sizeof(float));
            } else {
                tr->to_float(row, tmp.data(), n_embd);
            }
            for (int64_t e = 0; e < n_embd; ++e) {
                dstT[(size_t) e * n_vocab + v] = ggml_fp32_to_fp16(tmp[e]);
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + nth - 1) / nth;
    for (unsigned t = 0; t < nth; ++t) {
        const int64_t v0 = (int64_t) t * chunk;
        const int64_t v1 = std::min(v0 + chunk, n_vocab);
        if (v0 < v1) {
            pool.emplace_back(worker, v0, v1);
        }
    }
    for (auto & th : pool) {
        th.join();
    }

    ggml_backend_tensor_set(m.sc_embT, dstT.data(), 0, dstT.size() * sizeof(ggml_fp16_t));
}

// Lazily (re)allocate the device prev-step canvas-logits buffer [n_vocab, C] F32 (grow-only) on layer-0's
// buft. Zero-init so step 0 (SC gated off) reads finite values: soft_max(0)=uniform x 0 gate, no NaN.
static void dg_ensure_sc_dev(const llama_model_diffusion_gemma & m, int64_t C) {
    const int64_t n_vocab = m.tok_embd->ne[1];
    if (m.sc_dev != nullptr && m.sc_dev_C >= C) {
        return;
    }
    if (m.sc_dev_buf) { ggml_backend_buffer_free(m.sc_dev_buf); m.sc_dev_buf = nullptr; }
    if (m.sc_dev_ctx) { ggml_free(m.sc_dev_ctx); m.sc_dev_ctx = nullptr; }

    ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, /*.no_alloc =*/ true };
    m.sc_dev_ctx = ggml_init(ip);
    GGML_ASSERT(m.sc_dev_ctx != nullptr);
    m.sc_dev = ggml_new_tensor_2d(m.sc_dev_ctx, GGML_TYPE_F32, n_vocab, C);
    ggml_set_name(m.sc_dev, "sc_dev");

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev)
                                          : ggml_backend_cpu_buffer_type();
    m.sc_dev_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.sc_dev_ctx, buft);
    GGML_ASSERT(m.sc_dev_buf != nullptr);
    ggml_backend_buffer_clear(m.sc_dev_buf, 0);  // step-0 safety (see above)
    m.sc_dev_C = C;
}

// Lazily (re)allocate the device-resident F32 prompt-KV store (per-layer K,V, grow-only) for a prompt
// of length P, on layer-0's buffer type (single-GPU; cross-device would need a per-buft context map).
static void dg_ensure_pkv_store(const llama_model_diffusion_gemma & m, int64_t P) {
    if (m.pkv_buf != nullptr && m.pkv_cap >= P) {
        return;
    }
    if (m.pkv_buf) { ggml_backend_buffer_free(m.pkv_buf); m.pkv_buf = nullptr; }
    if (m.pkv_ctx) { ggml_free(m.pkv_ctx); m.pkv_ctx = nullptr; }
    m.pkv_k.clear();
    m.pkv_v.clear();

    const int     n_layer = (int) m.hparams.n_layer();
    const int64_t cap     = P;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (2 * n_layer + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    m.pkv_ctx = ggml_init(ip);
    GGML_ASSERT(m.pkv_ctx != nullptr);
    m.pkv_k.resize(n_layer);
    m.pkv_v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const int64_t hd  = m.hparams.n_embd_head_k(il);
        const int64_t nkv = m.hparams.n_head_kv(il);
        m.pkv_k[il] = ggml_new_tensor_3d(m.pkv_ctx, GGML_TYPE_F32, hd, nkv, cap);
        m.pkv_v[il] = ggml_new_tensor_3d(m.pkv_ctx, GGML_TYPE_F32, hd, nkv, cap);
        ggml_format_name(m.pkv_k[il], "pkv_k_l%d", il);
        ggml_format_name(m.pkv_v[il], "pkv_v_l%d", il);
    }

    ggml_backend_dev_t dev = m.dev_layer(0);
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev)
                                          : ggml_backend_cpu_buffer_type();
    m.pkv_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.pkv_ctx, buft);
    GGML_ASSERT(m.pkv_buf != nullptr);
    m.pkv_cap = cap;
}

// Public API: select the prompt-KV-caching phase for the next llama_decode (no-op otherwise). UNIFIED =
// no-cache forward; PREFILL writes the prompt K,V store (P = prompt length); DECODE reads it.
void llama_diffusion_set_phase(struct llama_model * model, int phase, int32_t P) {
    auto * dm = dynamic_cast<llama_model_diffusion_gemma *>(model);
    if (!dm) {
        return;
    }
    dm->pkv_phase = (llama_model_diffusion_gemma::pkv_phase_t) phase;
    dm->pkv_P     = P;
    if (phase != llama_model_diffusion_gemma::PKV_UNIFIED && P > 0) {
        dg_ensure_pkv_store(*dm, P);
    }
}
