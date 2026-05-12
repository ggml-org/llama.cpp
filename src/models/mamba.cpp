#include "models.h"

void llama_model_mamba::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_DT_B_C_RMS,     hparams.ssm_dt_b_c_rms, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24:
            switch (hparams.n_embd) {
                case 768: type = LLM_TYPE_SMALL; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 48:
            switch (hparams.n_embd) {
                case 1024: type = LLM_TYPE_MEDIUM; break;
                case 1536: type = LLM_TYPE_LARGE; break;
                case 2048: type = LLM_TYPE_XL; break;
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        case 64:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_3B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mamba::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;

    // only an expansion factor of 2 is supported for now
    if (2 * n_embd != d_inner) {
        throw std::runtime_error("only an expansion factor of 2 is supported for now");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed, duplicated to allow offloading
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner}, 0);

        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner}, 0);

        layer.ssm_x = create_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state}, 0);

        layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner}, 0);
        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

        // no "weight" suffix for these
        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner}, 0);
        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {d_inner}, 0);

        // out_proj
        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mamba::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mamba::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_mamba_base(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (model.arch == LLM_ARCH_MAMBA2) {
            cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
        } else {
            cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpL);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    // final rmsnorm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
