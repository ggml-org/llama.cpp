#include "models.h"

#include <map>

// one transformer stack of the tiny-voca TTS model, bidirectional, no KV cache
// - the text model is the phoneme encoder, its input rows are built by the audio pipeline (token + stress + voice)
// - mtmd runs the prosody and acoustic stacks of the mmproj with this arch too
// block: x + attn(LN(x)), then x + up/down(gelu(dwconv(LN(x)))), LN at the end

void llama_model_tiny_voca::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_tiny_voca::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    GGML_UNUSED(ml);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

    // no output head, the logits are unused; reuse the embedding table so a sampler can still run
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

        // depthwise conv over time, 7 taps
        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {7, n_embd}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i),   {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);
    }
}

// row of each tap of the depthwise conv: for token t and tap k, the token of the same sequence at pos(t) + k - pad,
// or the zero row n_tokens past the edges, so sequences that share a ubatch do not mix
class llm_graph_input_tiny_voca_conv : public llm_graph_input_i {
public:
    llm_graph_input_tiny_voca_conv(int d_conv) : d_conv(d_conv) {}
    virtual ~llm_graph_input_tiny_voca_conv() = default;

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_tokens = ubatch->n_tokens;
        const int     pad      = (d_conv - 1) / 2;
        GGML_ASSERT(ubatch->pos && ubatch->seq_id);
        std::map<std::pair<llama_seq_id, llama_pos>, int32_t> row_of;
        for (int64_t i = 0; i < n_tokens; ++i) {
            row_of[{ ubatch->seq_id[i][0], ubatch->pos[i] }] = (int32_t) i;
        }
        std::vector<int32_t> rows((size_t) n_tokens * d_conv);
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int k = 0; k < d_conv; ++k) {
                const auto it = row_of.find({ ubatch->seq_id[i][0], ubatch->pos[i] + k - pad });
                rows[i * d_conv + k] = it == row_of.end() ? (int32_t) n_tokens : it->second;
            }
        }
        ggml_backend_tensor_set(idx, rows.data(), 0, ggml_nbytes(idx));
    }

    ggml_tensor * idx = nullptr; // I32 [d_conv * n_tokens]
    const int d_conv;
};

std::unique_ptr<llm_graph_context> llama_model_tiny_voca::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_tiny_voca::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int d_conv = (int) model.layers[0].ssm_conv1d->ne[0];
    auto inp_conv = std::make_unique<llm_graph_input_tiny_voca_conv>(d_conv);
    inp_conv->idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) d_conv * n_tokens);
    ggml_set_input(inp_conv->idx);
    ggml_tensor * conv_idx = inp_conv->idx;
    res->add_input(std::move(inp_conv));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        cur = build_norm(inpL, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_rope(Qcur, inp_pos, il);
            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, layer.wo_b, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        // same-length depthwise conv over the positions of each sequence: gather the taps, weight them, sum
        {
            ggml_tensor * x = ggml_pad(ctx0, cur, 0, 1, 0, 0);                    // [n_embd, n_tokens + 1], last row 0
            x = ggml_get_rows(ctx0, x, conv_idx);                                 // [n_embd, d_conv * n_tokens]
            x = ggml_reshape_3d(ctx0, x, n_embd, d_conv, n_tokens);
            ggml_tensor * w = ggml_cont(ctx0, ggml_transpose(ctx0, layer.ssm_conv1d)); // [n_embd, d_conv]
            x = ggml_mul(ctx0, x, ggml_reshape_3d(ctx0, w, n_embd, d_conv, 1));
            x = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3))); // [1, n_embd, n_tokens]
            cur = ggml_add(ctx0, ggml_reshape_2d(ctx0, x, n_embd, n_tokens), layer.ssm_conv1d_b);
            cb(cur, "dwconv", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   NULL,
                NULL,           NULL,             NULL,
                layer.ffn_down, layer.ffn_down_b, NULL, NULL,
                LLM_FFN_GELU_ERF, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
