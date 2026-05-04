#include "models.h"

void llama_model_t5::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

    uint32_t dec_start_token_id;
    if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
        hparams.dec_start_token_id = dec_start_token_id;
    }

    hparams.dec_n_layer = hparams.n_layer;
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer, false);

    switch (hparams.n_layer) {
        case 6:  type = LLM_TYPE_60M;  break; // t5-small
        case 8:  type = LLM_TYPE_80M;  break; // flan-t5-small
        case 12:
            switch (hparams.n_ff()) {
                case 3072: type = LLM_TYPE_220M; break; // t5-base
                case 2048: type = LLM_TYPE_250M; break; // flan-t5-base
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 24:
            switch (hparams.n_ff()) {
                case 4096:  type = LLM_TYPE_770M; break; // t5-large
                case 2816:  type = LLM_TYPE_780M; break; // flan-t5-large
                case 16384: type = LLM_TYPE_3B;   break; // t5-3b
                case 5120:  type = LLM_TYPE_3B;   break; // flan-t5-xl
                case 65536: type = LLM_TYPE_11B;  break; // t5-11b
                case 10240: type = LLM_TYPE_11B;  break; // flan-t5-xxl
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
   }
}

void llama_model_t5::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // n_layer:     number of encoder_layers
    // dec_n_layer: number of decoder_layers
    const int dec_n_layer = hparams.dec_n_layer;
    if (dec_n_layer > n_layer) {
        layers.resize(dec_n_layer);
    }

    // load encoder layers
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }

    // load decoder layers
    for (int i = 0; i < dec_n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM,  "weight", i), {n_embd}, 0);
        layer.attn_rel_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

        layer.attn_norm_cross  = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "weight", i), {n_embd}, 0);
        // this tensor seems to be unused in HF transformers implementation
        layer.attn_rel_b_cross = create_tensor(
            tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);

        layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_t5::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
llama_model_t5::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * embd_enc       = build_inp_cross_embd();
    ggml_tensor * pos_bucket_dec = build_inp_pos_bucket_dec();

    const int64_t n_outputs_enc = embd_enc->ne[1];

    auto * inp_attn_self  = build_attn_inp_kv();
    auto * inp_attn_cross = build_attn_inp_cross();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t dec_n_layer = hparams.dec_n_layer;

    for (int il = 0; il < dec_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
            ggml_tensor * kq_b = build_pos_bias(pos_bucket_dec, attn_rel_b);

            cur = build_attn(inp_attn_self,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f, il);
            cb(cur, "kqv_out", il);
        }
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "cross_inp", il);

        ggml_tensor * inpCA = cur;

        // norm
        cur = build_norm(cur,
                model.layers[il].attn_norm_cross, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm_cross", il);

        // cross-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_outputs_enc);

            cur = build_attn(inp_attn_cross,
                    model.layers[il].wo_cross, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
            cb(cur, "kqv_out", il);

            //ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            //ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            //ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            //cb(kq, "kq", il);

            //kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
            //cb(kq, "kq_soft_max_ext", il);

            //ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
            //cb(v, "v", il);

            //ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
            //cb(kqv, "kqv", il);

            //ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            //cb(kqv_merged, "kqv_merged", il);

            //cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            //cb(cur, "kqv_merged_cont", il);

            //ggml_build_forward_expand(gf, cur);

            //cur = build_lora_mm(model.layers[il].wo_cross, cur);
            //cb(cur, "kqv_out", il);
        }
        if (il == dec_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // T5 uses relu, flan-T5 uses gelu-gated
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    il);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

template <>
llama_model_t5::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * pos_bucket_enc = build_inp_pos_bucket_enc();

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm_enc, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
            ggml_tensor * kq_b = build_pos_bias(pos_bucket_enc, attn_rel_b);

            cur = build_attn(inp_attn,
                    model.layers[il].wo_enc, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f, il);
            cb(cur, "kqv_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm_enc, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // T5 uses relu, flan-T5 uses gelu-gated
            cur = build_ffn(cur,
                    model.layers[il].ffn_up_enc,   NULL, NULL,
                    model.layers[il].ffn_gate_enc, NULL, NULL,
                    model.layers[il].ffn_down_enc, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
                    il);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = build_norm(cur,
            model.output_norm_enc, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
