#include "models.h"

// MSA-4B (Memory Sparse Attention — arXiv:2603.23516)
// HuggingFace: EverMind-AI/MSA-4B
//
// Architecture: 36-layer Qwen3-4B backbone
//   Layer  0-17: standard GQA attention
//   Layer 18-35: MSA attention (router Q/K + sparse top-k KV over a 100M-token memory bank)
//
// Degraded-mode implementation:
//   The router projection weights (attn_router_q / attn_router_k) are loaded from the
//   GGUF but not connected to the compute graph. The model runs standard GQA for all
//   36 layers, identical to Qwen3-4B. Inference is fully correct within the local
//   KV-cache window; only the long-range memory retrieval is inactive.
//
// The constructor simply replicates the Qwen3 graph — no code duplication because
// we re-implement it here identically rather than inheriting (llm_build_qwen3 has
// no virtual dispatch; its constructor builds the whole graph inline).

llm_build_msa::llm_build_msa(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // build_qkv returns 3D tensors (n_embd_head, n_head, n_tokens).
            // ggml_rms_norm on a 3D tensor can produce non-contiguous output
            // whose nb[0] stride fails the CUDA broadcast alignment check when
            // multiplied by the 1D norm weight. Flatten to 2D → norm → reshape
            // back to 3D to guarantee contiguous strides through the multiply.
            {
                const int64_t n_tokens_local = Qcur->ne[2];

                // Q norm
                ggml_tensor * Qflat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, Qcur),
                                                      n_embd_head, n_head * n_tokens_local);
                Qflat = build_norm(Qflat, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                Qcur  = ggml_reshape_3d(ctx0, Qflat, n_embd_head, n_head, n_tokens_local);
                cb(Qcur, "Qcur_normed", il);

                // K norm
                ggml_tensor * Kflat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, Kcur),
                                                      n_embd_head, n_head_kv * n_tokens_local);
                Kflat = build_norm(Kflat, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                Kcur  = ggml_reshape_3d(ctx0, Kflat, n_embd_head, n_head_kv, n_tokens_local);
                cb(Kcur, "Kcur_normed", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Note: for MSA layers (il >= first_msa_layer), model.layers[il].wq_a and
            // model.layers[il].wk_a hold the router projections. They are loaded but
            // not used here — standard KV attention is applied to all layers.
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}