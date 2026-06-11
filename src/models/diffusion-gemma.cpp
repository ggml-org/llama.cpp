#include "models.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"

// diffusion_gemma reuses the gemma4 decoder block (tensor layout + per-layer math) but runs
// as a bidirectional (non-causal) block-diffusion denoiser over a canvas, with KV-cache reuse:
// the prompt / previously-finalized canvases form a causal, read-only prefix in the unified
// sliding-window KV cache, and each denoising step decodes only the current canvas against it
// (self-conditioned, bidirectional), rolling back its own K/V afterwards.
//
// Two graph variants are provided (see build_arch_graph): a single phase-branching graph, and
// a separate encoder/decoder pair (DG_SEPARATE_ENC_DEC). Both reuse the gemma4 transformer
// body and differ only in input-embedding handling (encoder: plain; decoder: self-conditioned).

void llama_model_diffusion_gemma::load_arch_hparams(llama_model_loader & ml) {
    // reuse the gemma4 hparam loading (sliding window pattern, dual head dims, MoE, rope,
    // softcapping, layer types, ...)
    llama_model_gemma4::load_arch_hparams(ml);

    // the diffusion decoder attends bidirectionally
    hparams.causal_attn = false;
}

void llama_model_diffusion_gemma::load_arch_tensors(llama_model_loader & ml) {
    // load the shared gemma4 tensors (token embd, attention, dual dense+MoE FFN, norms,
    // per-layer layer_scalar, output)
    llama_model_gemma4::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    // self_conditioning is a gated MLP at hidden_size -> intermediate_size -> hidden_size
    const int64_t n_ff_sc = n_ff;

    self_cond_norm = create_tensor(tn(LLM_TENSOR_SC_PRE_NORM, "weight"), {n_embd}, 0);
    self_cond_gate = create_tensor(tn(LLM_TENSOR_SC_GATE,     "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_up   = create_tensor(tn(LLM_TENSOR_SC_UP,       "weight"), {n_embd,  n_ff_sc}, 0);
    self_cond_down = create_tensor(tn(LLM_TENSOR_SC_DOWN,     "weight"), {n_ff_sc, n_embd},  0);

    for (int i = 0; i < n_layer; ++i) {
        create_tensor(tn(LLM_TENSOR_ENC_LAYER_OUT_SCALE, "weight", i), {1u}, 0);
    }
}

llama_model_diffusion_gemma::~llama_model_diffusion_gemma() {
    if (tok_embd_gpu_buf) {
        ggml_backend_buffer_free(tok_embd_gpu_buf);
    }
    if (tok_embd_gpu_ctx) {
        ggml_free(tok_embd_gpu_ctx);
    }
    if (tok_embd_t_buf) {
        ggml_backend_buffer_free(tok_embd_t_buf);
    }
    if (tok_embd_t_ctx) {
        ggml_free(tok_embd_t_ctx);
    }
}

// Place the token embedding the diffusion decoder needs on an offloaded (GPU) backend.
//
// Primary path (sparse gather): store tok_embd {n_embd, n_vocab} as F16 in tok_embd_gpu on-device.
// The decoder graph gathers the canvas token rows and the self-conditioning top-k rows from this
// tensor. This keeps token-id driven embedding lookup on device: ~1.47 GiB resident, no per-decode
// token-id D2H for CPU row selection, no per-decode embedding H2D. F16 (not the native Q4_K)
// because CUDA get_rows has no Q4_K/Q6_K kernel -- a quantized gather would fall back to CPU every
// step (a large regression). F16 halves the VRAM vs the F32 dense path.
//
// Fallback path (dense matmul): if the F16 copy can't be allocated, precompute the transposed F32
// embedding {n_vocab, n_embd} so the dense `probs @ token_embd` matmul (build_input) still runs
// on-device. This costs ~2.75 GiB and is only used when the gather copy is unavailable.
void llama_model_diffusion_gemma::load_arch_post(llama_model_loader & ml) {
    GGML_UNUSED(ml);

    if (!tok_embd || !tok_embd->buffer) {
        return;
    }

    const int64_t n_embd_t  = tok_embd->ne[0];
    const int64_t n_vocab_t = tok_embd->ne[1];

    // Choose an offloaded (non-host) backend taken from a layer weight; fall back to token_embd's
    // own buffer (CPU-only runs). Leaving the self-cond embedding host-resident would force the
    // scheduler to stream it across PCIe every forward, which dominated the per-step time.
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tok_embd->buffer);
    for (const auto & layer : layers) {
        ggml_tensor * t = layer.wq ? layer.wq : (layer.wk ? layer.wk : layer.ffn_down);
        if (t && t->buffer) {
            ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(t->buffer);
            if (!ggml_backend_buft_is_host(b)) { buft = b; break; }
        }
    }

    // Dequantize the embedding to F32 once (host). It's needed by both paths below, and crucially
    // CUDA ggml_get_rows supports F16/F32/BF16 but NOT Q4_K/Q6_K (see ggml-cuda supports_op) -- a
    // gather from the native quantized type silently falls back to CPU every step. So the gather
    // copy must be F16/F32; we use F16 to halve its VRAM (~1.47 GiB vs 2.75 GiB F32).
    const int64_t n_elem = n_embd_t * n_vocab_t;
    const auto * tt = ggml_get_type_traits(tok_embd->type);
    if (!tt || !tt->to_float) {
        LLAMA_LOG_WARN("%s: cannot dequantize token embedding type %s; self-conditioning will use "
                       "the per-decode transpose fallback\n", __func__, ggml_type_name(tok_embd->type));
        return;
    }

    std::vector<uint8_t> raw(ggml_nbytes(tok_embd));
    ggml_backend_tensor_get(tok_embd, raw.data(), 0, raw.size());

    std::vector<float> src((size_t) n_elem);
    tt->to_float(raw.data(), src.data(), n_elem);

    // --- primary: F16 on-device copy for the sparse gather (get_rows runs on the GPU) ---
    {
        std::vector<ggml_fp16_t> half((size_t) n_elem);
        ggml_fp32_to_fp16_row(src.data(), half.data(), n_elem);

        ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        tok_embd_gpu_ctx = ggml_init(ip);
        tok_embd_gpu = ggml_new_tensor_2d(tok_embd_gpu_ctx, GGML_TYPE_F16, n_embd_t, n_vocab_t);
        ggml_set_name(tok_embd_gpu, "token_embd.gpu.f16");

        tok_embd_gpu_buf = ggml_backend_alloc_ctx_tensors_from_buft(tok_embd_gpu_ctx, buft);
        if (tok_embd_gpu_buf) {
            ggml_backend_tensor_set(tok_embd_gpu, half.data(), 0, ggml_nbytes(tok_embd_gpu));

            LLAMA_LOG_INFO("%s: placed diffusion token embedding {%lld, %lld} F16 (%.2f GiB) on %s (gather, k=%lld)\n",
                           __func__, (long long) n_embd_t, (long long) n_vocab_t,
                           ggml_nbytes(tok_embd_gpu) / (1024.0 * 1024.0 * 1024.0),
                           ggml_backend_buffer_name(tok_embd_gpu->buffer),
                           (long long) N_SC_TOPK);
            return;
        }
        LLAMA_LOG_WARN("%s: failed to allocate on-device gather embedding; falling back to dense matmul\n", __func__);
        ggml_free(tok_embd_gpu_ctx);
        tok_embd_gpu_ctx = nullptr;
        tok_embd_gpu = nullptr;
    }

    // --- fallback: transposed F32 embedding for the dense matmul path ---
    std::vector<float> dst((size_t) n_elem);
    for (int64_t e = 0; e < n_embd_t; ++e) {
        for (int64_t v = 0; v < n_vocab_t; ++v) {
            dst[(size_t) e * n_vocab_t + v] = src[(size_t) v * n_embd_t + e];
        }
    }

    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    tok_embd_t_ctx = ggml_init(ip);
    tok_embd_t = ggml_new_tensor_2d(tok_embd_t_ctx, GGML_TYPE_F32, n_vocab_t, n_embd_t);
    ggml_set_name(tok_embd_t, "token_embd_t.f32");

    tok_embd_t_buf = ggml_backend_alloc_ctx_tensors_from_buft(tok_embd_t_ctx, buft);
    if (!tok_embd_t_buf) {
        LLAMA_LOG_WARN("%s: failed to allocate transposed embedding buffer; falling back\n", __func__);
        ggml_free(tok_embd_t_ctx);
        tok_embd_t_ctx = nullptr;
        tok_embd_t = nullptr;
        return;
    }
    ggml_backend_tensor_set(tok_embd_t, dst.data(), 0, ggml_nbytes(tok_embd_t));

    LLAMA_LOG_INFO("%s: precomputed transposed F32 token embedding {%lld, %lld} (%.2f GiB) on %s (dense fallback)\n",
                   __func__, (long long) n_vocab_t, (long long) n_embd_t,
                   ggml_nbytes(tok_embd_t) / (1024.0 * 1024.0 * 1024.0),
                   ggml_backend_buffer_name(tok_embd_t->buffer));
}

std::unique_ptr<llm_graph_context> llama_model_diffusion_gemma::build_arch_graph(const llm_graph_params & params) const {
    const bool is_decoder = params.diffusion && params.diffusion->decoder_phase;

    // Variant B ("separate encoder and decoder block", shared weights): opt-in via
    // DG_SEPARATE_ENC_DEC. Two distinct graphs are built per phase. Functionally identical
    // to Variant A here (the checkpoint shares encoder/decoder weights); the split mirrors
    // the HF two-stack structure and generalizes to a checkpoint with distinct weights.
    if (getenv("DG_SEPARATE_ENC_DEC")) {
        if (is_decoder) {
            return std::make_unique<graph_decoder>(*this, params);
        }
        return std::make_unique<graph_encoder>(*this, params);
    }

    // Variant A ("single encoder/decoder block"): one graph that branches on the phase.
    return std::make_unique<graph>(*this, params);
}

// Scaled input embeddings. In the decoder phase, apply the self-conditioning transform:
//   inpL = post_norm(scaled_embed + sc_mlp(pre_norm(soft))),
//   soft = (probs @ token_embd) * sqrt(n_embd)   [probs = previous step's softmax, 0 on step 1]
ggml_tensor * llama_model_diffusion_gemma::graph_base::build_input(bool is_decoder) {
    const auto & dmodel = static_cast<const llama_model_diffusion_gemma &>(model);

    ggml_tensor * inpL = build_inp_embd(dmodel.tok_embd_gpu ? dmodel.tok_embd_gpu : model.tok_embd);

    // scaled word embeddings (sqrt(hidden_size)); raw embeddings input is not scaled
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    if (is_decoder) {
        ggml_tensor * soft; // soft-embedding {n_embd, n_tokens}: blend of the previous step's
                            // predicted token embeddings, scaled by sqrt(n_embd)
        if (dmodel.tok_embd_gpu) {
            // Sparse gather path (Option-2): the previous step's top-k token ids+probs are fed per
            // position; gather just those k embedding rows and blend them, instead of the dense
            // full-vocab `probs @ token_embd` matmul. Gather width is fixed (N_SC_TOPK) so the
            // graph shape is constant; unused slots carry prob 0 (the CLI zero-pads).
            const int64_t k = llama_model_diffusion_gemma::N_SC_TOPK;
            auto * inp = build_inp_diffusion_self_cond_topk(k);
            ggml_tensor * ids   = inp->ids;                                                // I32 {k*n_tokens}
            ggml_tensor * probs = inp->probs;                                              // F32 {k, n_tokens}

            // gather: {n_embd, n_vocab} x {k*n_tokens} ids -> {n_embd, k*n_tokens} -> {n_embd, k, n_tokens}
            ggml_tensor * emb = ggml_get_rows(ctx0, dmodel.tok_embd_gpu, ids);             // {n_embd, k*n_tokens}
            emb = ggml_reshape_3d(ctx0, emb, n_embd, k, n_tokens);                          // {n_embd, k, n_tokens}
            // weight each gathered row by its prob (broadcast over n_embd): {n_embd, k, n_tokens}
            ggml_tensor * w = ggml_mul(ctx0, emb, ggml_reshape_3d(ctx0, probs, 1, k, n_tokens));
            // sum over k: bring k to ne[0] then sum_rows -> {1, n_embd, n_tokens} -> {n_embd, n_tokens}
            w = ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3));                         // {k, n_embd, n_tokens}
            w = ggml_sum_rows(ctx0, w);                                                     // {1, n_embd, n_tokens}
            soft = ggml_reshape_2d(ctx0, w, n_embd, n_tokens);                              // {n_embd, n_tokens}
        } else {
            // Dense fallback: soft = (probs @ token_embd). mul_mat contracts ne[0], so token_embd
            // needs vocab as ne[0]; prefer the transposed F32 embedding from load_arch_post, else
            // dequantize+transpose every decode (a quantized tensor can't be transposed directly).
            ggml_tensor * probs   = build_inp_diffusion_self_cond(model.tok_embd->ne[1]); // {n_vocab, n_tokens}
            ggml_tensor * embed_t = dmodel.tok_embd_t;                                     // {n_vocab, n_embd}
            if (!embed_t) {
                ggml_tensor * embed_f = ggml_cast(ctx0, model.tok_embd, GGML_TYPE_F32);     // {n_embd, n_vocab}
                embed_t = ggml_cont(ctx0, ggml_transpose(ctx0, embed_f));                  // {n_vocab, n_embd}
            }
            soft = ggml_mul_mat(ctx0, embed_t, probs);                                     // {n_embd, n_tokens}
        }
        soft = ggml_scale(ctx0, soft, sqrtf((float) n_embd));
        cb(soft, "self_cond_soft_embd", -1);
        ggml_tensor * scn = build_norm(soft, dmodel.self_cond_norm, nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * sc  = build_ffn(scn,
                dmodel.self_cond_up,   nullptr, nullptr,
                dmodel.self_cond_gate, nullptr, nullptr,
                dmodel.self_cond_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, -1);
        inpL = ggml_rms_norm(ctx0, ggml_add(ctx0, inpL, sc), hparams.f_norm_rms_eps); // scale-less post_norm
        cb(inpL, "self_cond_input", -1);
    }

    return inpL;
}

// Variant A: single graph, phase chosen at runtime from the diffusion cond.
llama_model_diffusion_gemma::graph::graph(const llama_model & model, const llm_graph_params & params) :
        graph_base(model, params) {
    build_transformer(build_input(diffusion && diffusion->decoder_phase));
}

// Variant B: separate encoder / decoder graphs (shared weight tensors).
llama_model_diffusion_gemma::graph_encoder::graph_encoder(const llama_model & model, const llm_graph_params & params) :
        graph_base(model, params) {
    build_transformer(build_input(/*is_decoder=*/false));
}

llama_model_diffusion_gemma::graph_decoder::graph_decoder(const llama_model & model, const llm_graph_params & params) :
        graph_base(model, params) {
    build_transformer(build_input(/*is_decoder=*/true));
}

// Run the reused gemma4 decoder block over the input embeddings and emit logits.
void llama_model_diffusion_gemma::graph_base::build_transformer(ggml_tensor * inpL) {
    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();

    // Reuse the unified sliding-window KV cache: the canvas (decoder phase) reads the
    // cached prompt/previous-canvas prefix; encoder-phase tokens write (commit) their KV.
    // The prompt/canvas attention pattern is selected by cparams.causal_attn, toggled by
    // the caller (encoder: causal; decoder: bidirectional + full cross to the prefix).
    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        // full_attention layers use rope_freqs for proportional rope
        ggml_tensor * freq_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        // attention norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention (QK norm + scale-less V norm, mirrors Gemma4Attention)
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        // KV-sharing layers (n_kv_shared_layers) do not own a cache slot: they reuse an
        // earlier layer's cached K/V (wk/wv/k_norm are absent). Mirror gemma4: compute and
        // store K/V only for has_kv layers, otherwise pass nullptr (no store).
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);
            // global (full-attention) layers have no v_proj: V = K (before norms)
            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                    : Kcur;
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps); // scale-less v_norm
            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                                 freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, model.layers[il].wo_s,
                             Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        } else {
            // reuse the cached K/V of an earlier layer
            cur = build_attn(inp_attn, model.layers[il].wo, nullptr, model.layers[il].wo_s,
                             Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward: dense MLP (shared expert) + routed MoE, summed (mirrors gemma4)
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            ggml_tensor * cur_mlp = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);
            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp, model.layers[il].ffn_post_norm_1, nullptr, LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            ggml_tensor * cur_moe = build_norm(attn_out, model.layers[il].ffn_pre_norm_2, nullptr, LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // router operates on attn_out (scale-less norm * 1/sqrt(n_embd) * router scale)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp);
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe, model.layers[il].ffn_post_norm_2, nullptr, LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
