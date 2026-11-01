#include "models.h"

#include "llama-ext.h"

#include <cmath>

void llama_model_vllm_decision2::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_hparams(ml);

    // used by the head
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    // the output is one score per token, see llama_batch_ext_set_decision_order()
    hparams.n_embd_out_impl = 1;
}

void llama_model_vllm_decision2::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen3::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const auto * w_key = ml.get_weight(tn(LLM_TENSOR_DECISION_KEY, "weight").str().c_str());
    if (w_key == nullptr) {
        throw std::runtime_error("the decision head is missing");
    }
    const int64_t n_embd_h = w_key->tensor->ne[1];

    option_norm   = create_tensor(tn(LLM_TENSOR_DECISION_OPTION_NORM, "weight"), {n_embd}, 0);
    option_norm_b = create_tensor(tn(LLM_TENSOR_DECISION_OPTION_NORM, "bias"),   {n_embd}, 0);
    query_norm    = create_tensor(tn(LLM_TENSOR_DECISION_QUERY_NORM,  "weight"), {n_embd}, 0);
    query_norm_b  = create_tensor(tn(LLM_TENSOR_DECISION_QUERY_NORM,  "bias"),   {n_embd}, 0);

    key          = create_tensor(tn(LLM_TENSOR_DECISION_KEY,        "weight"), {n_embd, n_embd_h}, 0);
    query        = create_tensor(tn(LLM_TENSOR_DECISION_QUERY,      "weight"), {n_embd, n_embd_h}, 0);
    option_mlp   = create_tensor(tn(LLM_TENSOR_DECISION_OPTION_MLP, "weight"), {n_embd, n_embd_h}, 0);
    option_mlp_b = create_tensor(tn(LLM_TENSOR_DECISION_OPTION_MLP, "bias"),   {n_embd_h}, 0);
    query_mlp    = create_tensor(tn(LLM_TENSOR_DECISION_QUERY_MLP,  "weight"), {n_embd, n_embd_h}, 0);
    scorer_out   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT, "weight"), {n_embd_h, 1}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_vllm_decision2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// output rows read by the head: the last token of each option span and the last token of the question span
struct decision2_rows {
    std::vector<int32_t> options;
    int32_t query = 0;
    bool valid = false;
};

// see llama_batch_ext_set_decision_order()
// if the batch has no usable order, returns one option at row 0
static decision2_rows decision2_get_rows(const llama_ubatch & ubatch) {
    decision2_rows res;

    const int32_t n_tokens = ubatch.n_tokens;
    bool ok = ubatch.decision_order != nullptr && ubatch.n_seqs_unq == 1;

    // output row of each token, -1 if the token has no output
    std::vector<int32_t> rows(n_tokens, -1);
    for (int32_t i = 0, n = 0; i < n_tokens; i++) {
        if (ubatch.output[i]) {
            rows[i] = n++;
        }
    }

    int32_t n_questions = 0;
    for (int32_t i = 0; ok && i < n_tokens;) {
        const int32_t order = ubatch.decision_order[i];
        int32_t end = i + 1;
        while (end < n_tokens && ubatch.decision_order[end] == order) {
            end++;
        }
        switch (order) {
            case LLAMA_DECISION_ORDER_NONE:
                break;
            case LLAMA_DECISION_ORDER_QUESTION_NOUL:
            case LLAMA_DECISION_ORDER_QUESTION_CHOICE:
            case LLAMA_DECISION_ORDER_QUESTION_SCORE:
                res.query = rows[end - 1];
                n_questions++;
                break;
            case LLAMA_DECISION_ORDER_OPTION:
                res.options.push_back(rows[end - 1]);
                break;
            default:
                ok = false;
        }
        ok = ok && (order == LLAMA_DECISION_ORDER_NONE || rows[end - 1] >= 0);
        i = end;
    }

    res.valid = ok && n_questions == 1 && !res.options.empty();
    if (!res.valid) {
        res.options.assign(1, 0);
        res.query = 0;
    }
    return res;
}

class llama_model_vllm_decision2::input_decision : public llm_graph_input_i {
public:
    input_decision(const llama_ubatch & ubatch) : n_options(decision2_get_rows(ubatch).options.size()) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto rows = decision2_get_rows(*ubatch);
        GGML_ASSERT(rows.options.size() == n_options);

        // the scores are NaN if the batch has a decision order that cannot be used
        const float status_data = rows.valid || ubatch->decision_order == nullptr ? 0.0f : NAN;

        ggml_backend_tensor_set(options, rows.options.data(), 0, ggml_nbytes(options));
        ggml_backend_tensor_set(query,   &rows.query,         0, ggml_nbytes(query));
        ggml_backend_tensor_set(status,  &status_data,        0, ggml_nbytes(status));
    }

    bool can_reuse(const llm_graph_params & params) override {
        // the values are computed again in set_input(), only the shapes must match
        return decision2_get_rows(params.ubatch).options.size() == n_options;
    }

    ggml_tensor * options = nullptr; // I32 [n_options]
    ggml_tensor * query   = nullptr; // I32 [1]
    ggml_tensor * status  = nullptr; // F32 [1], added to the scores: 0, or NaN on invalid input

    const size_t n_options;
};

llama_model_vllm_decision2::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llama_model_qwen3::graph(params, no_build()), model(static_cast<const llama_model_vllm_decision2 &>(model_base)) {
    auto inp_decision_ptr = std::make_unique<input_decision>(ubatch);
    auto * inp_decision   = inp_decision_ptr.get();
    res->add_input(std::move(inp_decision_ptr));

    ggml_tensor * cur = build_backbone(model);

    // the head is always evaluated, so that the graph has the same nodes for every batch
    cur = build_head(cur, inp_decision);

    // row i of the output is the score of option i
    cur = ggml_pad(ctx0, cur, 0, n_outputs - cur->ne[1], 0, 0);
    cur = ggml_add(ctx0, cur, inp_decision->status);
    cb(cur, "result_decision", -1);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_vllm_decision2::graph::build_head(ggml_tensor * hidden, input_decision * inp) {
    inp->options = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_options);
    inp->query   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    inp->status  = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { inp->options, inp->query, inp->status }) {
        ggml_set_input(t);
    }

    ggml_tensor * options = ggml_get_rows(ctx0, hidden, inp->options); // [n_embd, n_options]
    ggml_tensor * query   = ggml_get_rows(ctx0, hidden, inp->query);   // [n_embd, 1]

    options = build_norm(options, model.option_norm, model.option_norm_b, LLM_NORM, -1);
    query   = build_norm(query,   model.query_norm,  model.query_norm_b,  LLM_NORM, -1);

    // scaled dot product of the option key and the query
    ggml_tensor * k = ggml_mul_mat(ctx0, model.key,   options); // [n_embd_h, n_options]
    ggml_tensor * q = ggml_mul_mat(ctx0, model.query, query);   // [n_embd_h, 1]
    ggml_tensor * bilinear = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    bilinear = ggml_scale(ctx0, bilinear, 1.0f / sqrtf((float) k->ne[0])); // [1, n_options]

    ggml_tensor * mlp = ggml_add(ctx0, ggml_mul_mat(ctx0, model.option_mlp, options), model.option_mlp_b);
    mlp = ggml_add(ctx0, mlp, ggml_mul_mat(ctx0, model.query_mlp, query));
    mlp = ggml_gelu_erf(ctx0, mlp);
    ggml_tensor * nonlinear = ggml_mul_mat(ctx0, model.scorer_out, mlp); // [1, n_options]

    ggml_tensor * scores = ggml_add(ctx0, bilinear, nonlinear);
    cb(scores, "decision_scores", -1);
    return scores;
}
