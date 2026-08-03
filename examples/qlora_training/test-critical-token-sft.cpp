#ifdef NDEBUG
#undef NDEBUG
#endif

#include "critical-token-sft.h"

#include "ggml-backend.h"
#include "ggml-opt.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

static bool close_to(double a, double b) {
    return std::abs(a - b) < 1e-5;
}

static ggml_opt_optimizer_params test_optimizer_params(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);
    result.sgd.alpha = 0.1f;
    result.sgd.wd = 0.0f;
    return result;
}

struct graph_result {
    float loss;
    std::vector<float> selected;
    std::vector<float> effective;
    std::vector<float> logits_after;
};

static graph_result run_graph(
        const std::vector<float> & probabilities,
        const std::vector<bool>  & active,
        const std::vector<float> & span_weights,
        const std::vector<float> & reward_weights,
        bool                       critical_enabled,
        bool                       confidence_enabled,
        float                      critical_weight = 3.0f,
        float                      threshold = 0.25f,
        bool                       linear = false,
        int32_t                    max_tokens = -1,
        float                      warmup_scale = 1.0f,
        bool                       backward = false) {
    assert(probabilities.size() == active.size());
    assert(probabilities.size() == span_weights.size());
    assert(probabilities.size() == reward_weights.size());
    const int64_t nrows = probabilities.size();
    const int64_t vocab = 2;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend);
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params static_params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(static_params);
    ggml_tensor * logits = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, vocab, nrows);
    ggml_set_param(logits);
    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors(ctx_static, backend);

    std::vector<float> logits_before(vocab * nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        const float p = probabilities[row];
        logits_before[vocab * row] = std::log(p / (1.0f - p));
        logits_before[vocab * row + 1] = 0.0f;
    }
    ggml_backend_tensor_set(logits, logits_before.data(), 0, logits_before.size() * sizeof(float));

    ggml_init_params compute_params = {
        /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 3 * ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(compute_params);
    ggml_tensor * outputs = ggml_scale(ctx_compute, logits, 1.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(graph, outputs);

    ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD;
    opt_params.get_opt_pars = test_optimizer_params;
    opt_params.critical_token_weighting = critical_enabled;
    opt_params.critical_confidence_weighting = confidence_enabled;
    opt_params.critical_token_weight = critical_weight;
    opt_params.critical_confidence_threshold = threshold;
    opt_params.critical_weight_linear = linear;
    ggml_opt_context_t opt = ggml_opt_init(opt_params);
    ggml_opt_set_critical_max_tokens(opt, max_tokens);
    ggml_opt_prepare_alloc(opt, ctx_compute, graph, logits, outputs);
    ggml_opt_alloc(opt, backward);

    ggml_tensor * labels = ggml_opt_labels(opt);
    ggml_set_zero(labels);
    for (int64_t row = 0; row < nrows; ++row) {
        if (!active[row]) {
            continue;
        }
        const float value = critical_enabled ? 1.0f : (float) nrows / std::count(active.begin(), active.end(), true);
        ggml_backend_tensor_set(labels, &value, row * vocab * sizeof(float), sizeof(float));
    }
    if (critical_enabled) {
        ggml_backend_tensor_set(ggml_opt_critical_span_weights(opt), span_weights.data(), 0, nrows * sizeof(float));
        ggml_backend_tensor_set(ggml_opt_critical_reward_weights(opt), reward_weights.data(), 0, nrows * sizeof(float));
        ggml_backend_tensor_set(ggml_opt_critical_warmup_scale(opt), &warmup_scale, 0, sizeof(float));
    }

    ggml_opt_result_t opt_result = ggml_opt_result_init();
    ggml_opt_eval(opt, opt_result);
    graph_result result;
    ggml_backend_tensor_get(ggml_opt_loss(opt), &result.loss, 0, sizeof(float));
    result.logits_after.resize(logits_before.size());
    ggml_backend_tensor_get(logits, result.logits_after.data(), 0, result.logits_after.size() * sizeof(float));
    if (critical_enabled) {
        result.selected.resize(nrows);
        result.effective.resize(nrows);
        ggml_backend_tensor_get(ggml_opt_critical_selected(opt), result.selected.data(), 0, nrows * sizeof(float));
        ggml_backend_tensor_get(ggml_opt_critical_effective_weights(opt), result.effective.data(), 0, nrows * sizeof(float));
    }

    ggml_opt_result_free(opt_result);
    ggml_opt_free(opt);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    return result;
}

int main() {
    {
        const std::vector<float> losses  = { 1.0f, 2.0f, 4.0f };
        const std::vector<float> weights = { 1.0f, 3.0f, 1.0f };
        const std::vector<bool> active   = { true, true, true };
        assert(close_to(critical_token_weighted_mean(losses, weights, active), 2.2));
    }
    {
        const std::vector<float> losses  = { 100.0f, 2.0f, 100.0f, 4.0f };
        const std::vector<float> weights = { 9.0f, 3.0f, 9.0f, 1.0f };
        const std::vector<bool> active   = { false, true, false, true };
        assert(close_to(critical_token_weighted_mean(losses, weights, active), 2.5));
    }
    {
        const std::string response = "alpha beta gamma";
        const std::vector<critical_span> spans = {
            { 6, 10, 2.0f },
            { 8, 14, 4.0f },
        };
        const std::vector<critical_token_range> ranges = {
            { 0, 5 }, { 6, 10 }, { 11, 16 },
        };
        std::vector<float> weights;
        std::string error;
        assert(critical_token_apply_spans(response.size(), spans, ranges, weights, error));
        assert(weights == std::vector<float>({ 1.0f, 4.0f, 4.0f }));
    }
    {
        const std::string response =
            "\xEC\xA0\x95\xEB\x8B\xB5\xEC\x9D\x80 \xEC\x84\x9C\xEC\x9A\xB8\xEC\x9E\x85\xEB\x8B\x88\xEB\x8B\xA4.";
        const std::string seoul = "\xEC\x84\x9C\xEC\x9A\xB8";
        const size_t seoul_start = response.find(seoul);
        assert(seoul_start == 10);
        const std::vector<critical_span> spans = {
            { seoul_start, seoul_start + seoul.size(), 3.0f },
        };
        const std::vector<critical_token_range> ranges = {
            { 0, 3 }, { 3, 9 }, { 9, 10 }, { 10, 13 }, { 13, 16 }, { 16, response.size() },
        };
        std::vector<float> weights;
        std::string error;
        assert(critical_token_apply_spans(response.size(), spans, ranges, weights, error));
        assert(weights == std::vector<float>({ 1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 1.0f }));
    }
    {
        const std::vector<float> probabilities = { 0.1f, 0.2f, 0.05f, 0.05f, 0.8f };
        const std::vector<bool> active = { true, true, true, true, false };
        const auto selected = critical_token_select_lowest_confidence(probabilities, active, 0.25f, 0.5f);
        assert(selected == std::vector<size_t>({ 2, 3 }));
    }
    {
        std::ifstream input(std::string(CRITICAL_TOKEN_SFT_SOURCE_DIR) + "/critical-token-sft-data.jsonl");
        std::string line;
        size_t nrecords = 0;
        while (std::getline(input, line)) {
            const nlohmann::json record = nlohmann::json::parse(line);
            std::vector<critical_span> spans;
            std::string error;
            assert(critical_token_parse_spans(record, record["response"], 3.0f, spans, error));
            ++nrecords;
        }
        assert(nrecords == 4);

        std::ifstream invalid_input(std::string(CRITICAL_TOKEN_SFT_SOURCE_DIR) + "/critical-token-sft-invalid.jsonl");
        assert(std::getline(invalid_input, line));
        const nlohmann::json invalid = nlohmann::json::parse(line);
        std::vector<critical_span> spans;
        std::string error;
        assert(!critical_token_parse_spans(invalid, invalid["response"], 3.0f, spans, error));
        assert(error.find("critical_spans[0]") != std::string::npos);
    }
    {
        assert(close_to(critical_token_confidence_weight(0.1f, 0.25f, 3.0f, false), 3.0));
        assert(close_to(critical_token_confidence_weight(0.1f, 0.25f, 3.0f, true), 2.2));
        assert(close_to(critical_token_confidence_weight(0.3f, 0.25f, 3.0f, true), 1.0));
        assert(close_to(critical_token_warmup_scale(40, 100), 0.4));
        assert(close_to(critical_token_warmup_scale(140, 100), 1.0));
        assert(close_to(critical_token_warmup_scale(140, 0), 1.0));
    }
    {
        const std::vector<float> p = { std::exp(-1.0f), std::exp(-2.0f), std::exp(-4.0f) };
        const std::vector<bool> active = { true, true, true };
        const graph_result result = run_graph(p, active, { 1.0f, 3.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, true, false);
        assert(close_to(result.loss, 2.2));
    }
    {
        const std::vector<float> p = { 0.2f, 0.3f, 0.1f, 0.2f };
        const std::vector<bool> active = { true, true, false, true };
        const graph_result legacy = run_graph(p, active, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, false, false);
        const graph_result disabled_equivalent = run_graph(p, active, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, true, false);
        assert(close_to(legacy.loss, disabled_equivalent.loss));
        assert(disabled_equivalent.effective == std::vector<float>({ 1, 1, 0, 1 }));

        const graph_result legacy_update = run_graph(p, active, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, false, false,
                3.0f, 0.25f, false, -1, 1.0f, true);
        const graph_result weighted_update = run_graph(p, active, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, true, false,
                3.0f, 0.25f, false, -1, 1.0f, true);
        for (size_t i = 0; i < legacy_update.logits_after.size(); ++i) {
            assert(close_to(legacy_update.logits_after[i], weighted_update.logits_after[i]));
        }
    }
    {
        const std::vector<float> p = { 0.2f, 0.3f, 0.1f, 0.05f };
        const std::vector<bool> active = { true, true, false, true };
        const graph_result confidence = run_graph(p, active, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, true, true);
        assert(confidence.selected == std::vector<float>({ 1, 0, 0, 1 }));
        assert(confidence.effective == std::vector<float>({ 3, 1, 0, 3 }));

        const graph_result hybrid = run_graph(p, active, { 4, 2, 0, 1 }, { 1, 1, 0, 1 }, true, true);
        assert(hybrid.effective == std::vector<float>({ 4, 2, 0, 3 }));

        const graph_result linear = run_graph(p, active, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, true, true,
                3.0f, 0.25f, true);
        assert(close_to(linear.effective[0], 1.4));
        assert(close_to(linear.effective[3], 2.6));
    }
    {
        const std::vector<float> p = { 0.1f, 0.2f, 0.05f, 0.05f };
        const std::vector<bool> active = { true, true, true, true };
        const graph_result capped = run_graph(p, active, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, true, true, 3.0f, 0.25f, false, 2);
        assert(capped.selected == std::vector<float>({ 0, 0, 1, 1 }));
    }
    {
        const std::vector<float> p = { 0.2f, 0.2f, 0.2f };
        const std::vector<bool> active = { true, true, true };
        const graph_result ordinary = run_graph(p, active, { 1, 1, 1 }, { 1, 1, 1 }, true, false);
        const graph_result all_critical = run_graph(p, active, { 3, 3, 3 }, { 1, 1, 1 }, true, false);
        assert(close_to(ordinary.loss, all_critical.loss));

        const graph_result warmed_reward = run_graph(p, active, { 3, 1, 1 }, { 0.5f, 1, 1 }, true, false,
                3.0f, 0.25f, false, -1, 0.5f);
        assert(warmed_reward.effective == std::vector<float>({ 1, 1, 1 }));
    }
    {
        const std::vector<float> shifted_weights = { 0, 0, 1, 4, 1, 0, 0 };
        const std::vector<float> first_window(shifted_weights.begin(), shifted_weights.begin() + 5);
        const std::vector<float> stride_window(shifted_weights.begin() + 2, shifted_weights.begin() + 7);
        assert(first_window == std::vector<float>({ 0, 0, 1, 4, 1 }));
        assert(stride_window == std::vector<float>({ 1, 4, 1, 0, 0 }));
    }
    {
        const std::vector<float> p = { 0.2f, 0.2f };
        const std::vector<bool> active = { true, true };
        const graph_result gradient = run_graph(p, active, { 3, 1 }, { 1, 1 }, true, false, 3.0f, 0.25f, false, -1, 1.0f, true);
        const float delta_critical = gradient.logits_after[0] - std::log(0.2f / 0.8f);
        const float delta_ordinary = gradient.logits_after[2] - std::log(0.2f / 0.8f);
        assert(close_to(delta_critical / delta_ordinary, 3.0));
    }

    return 0;
}
