#include "models.h"

namespace {
class llm_graph_input_pos_sensenova_u1 : public llm_graph_input_pos {
public:
    llm_graph_input_pos_sensenova_u1() : llm_graph_input_pos(4) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch->pos && pos) {
            const int64_t n_tokens = ubatch->n_tokens;

            if (ubatch->token) {
                std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd, 0);
                for (int i = 0; i < n_tokens; ++i) {
                    pos_data[i] = ubatch->pos[i];
                }
                ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
            } else {
                ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
            }
        }
    }
};
}

void llama_model_sensenova_u1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key("sensenova_u1.rope.freq_base_spatial", rope_freq_base_spatial);
    if (hparams.n_embd_head_k() % 4 != 0 || hparams.n_embd_head_v() != hparams.n_embd_head_k()) {
        throw std::runtime_error("SenseNova U1 requires equal Q/K/V head sizes divisible by four");
    }
    type = hparams.n_layer() == 42 ? LLM_TYPE_8B : LLM_TYPE_UNKNOWN;
}

void llama_model_sensenova_u1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_sensenova_u1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_sensenova_u1::graph::graph(const llama_model_sensenova_u1 & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const bool bf16 = model.output->type == GGML_TYPE_BF16;
    auto linear_input = [&](ggml_tensor * x) {
        return bf16 ? ggml_cast(ctx0, x, GGML_TYPE_BF16) : x;
    };
    auto normalize = [&](ggml_tensor * x, ggml_tensor * weight) {
        x = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);

        return ggml_mul(ctx0, x, weight);
    };

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * pos[3]{};
    auto inp = std::make_unique<llm_graph_input_pos_sensenova_u1>();
    auto & inp_pos = inp->pos;
    inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(inp_pos);
    res->add_input(std::move(inp));

    for (int axis = 0; axis < 3; ++axis) {
        pos[axis] = ggml_view_1d(ctx0, inp_pos, n_tokens, axis*n_tokens*sizeof(int32_t));
    }

    auto normalize_and_rotate = [&](ggml_tensor * x, ggml_tensor * weight) {
        const int64_t heads = x->ne[1];
        x = ggml_reshape_4d(ctx0, x, n_embd_head/2, 2, heads, n_tokens);
        x = normalize(x, ggml_reshape_2d(ctx0, weight, n_embd_head/2, 2));
        x = ggml_reshape_3d(ctx0, x, n_embd_head, heads, n_tokens);

        // Normalize temporal and spatial halves separately; rotate time, height, then width.
        for (int axis = 0; axis < 3; ++axis) {
            const int dims = axis == 0 ? n_embd_head/2 : n_embd_head/4;
            const int offset = axis == 0 ? 0 : n_embd_head/2 + (axis - 1)*n_embd_head/4;
            const float base = axis == 0 ? freq_base : model.rope_freq_base_spatial;
            x = ggml_rope_ext(ctx0, x, pos[axis], nullptr, dims, GGML_ROPE_TYPE_NEOX,
                    n_ctx_orig, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_rope_set_offset(x, offset);
        }
        return x;
    };

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = normalize(inpL, model.layers[il].attn_norm);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], linear_input(cur),
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = normalize_and_rotate(Qcur, model.layers[il].attn_q_norm);
            Kcur = normalize_and_rotate(Kcur, model.layers[il].attn_k_norm);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = normalize(ffn_inp, model.layers[il].ffn_norm);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(linear_input(cur),
            model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = normalize(cur, model.output_norm);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, linear_input(cur), model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
