#include "models.h"

void llama_model_mamba2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

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
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 64:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_3B; break;
                case 4096: type = LLM_TYPE_7B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mamba2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t n_group = hparams.ssm_n_group;
    const int64_t d_in_proj = 2*d_inner + 2*n_group*d_state + n_head;

    // only an expansion factor of 2 is supported for now
    GGML_ASSERT(2 * n_embd == d_inner);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // norm
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, 0);

        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_head}, 0);

        // no "weight" suffix for these
        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_head}, 0);

        layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

        // out_proj
        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mamba2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

