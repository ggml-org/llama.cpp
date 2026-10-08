#include "backend-context.h"
#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/common/dispatch-activation-publication.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/graph-executor.h"
#include "testing_suite.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void restore_environment_value(const char * name, bool had_value, const std::string & value) {
    if (had_value) {
        REQUIRE(setenv(name, value.c_str(), 1) == 0);
    } else {
        REQUIRE(unsetenv(name) == 0);
    }
}

static std::string expected_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static constexpr float   kQwenRmsNormEps        = 0.000001f;
static constexpr int64_t kQwenFlashHeadSize     = 128;
static constexpr int64_t kQwenRouterExpertCount = 128;
static constexpr int64_t kQwenRouterRouteCount  = 8;
static constexpr int64_t kQwenHiddenSize        = 2048;
static constexpr int64_t kQwenMoeIntermediate   = 768;
static constexpr int64_t kQwenVocabularyCount   = 151936;
static constexpr int64_t kGemmaHiddenSize       = 3840;
static constexpr int64_t kGemmaPromptTokenCount = 18;
static constexpr float   kGemmaRmsNormEps       = 0.000001f;

static ggml::hrx::Value make_test_value(ggml::hrx::ValueId        id,
                                        ggml::hrx::ValueStorageId storage,
                                        ggml::hrx::ValueId        storage_root,
                                        ggml::hrx::ValueId        alias_source,
                                        size_t                    storage_offset,
                                        size_t                    storage_byte_count,
                                        ggml_type                 type,
                                        int64_t                   element_count) {
    ggml::hrx::Value value   = {};
    value.id                 = id;
    value.kind               = ggml::hrx::ValueKind::Transient;
    value.storage            = storage;
    value.storage_root       = storage_root;
    value.alias_source       = alias_source;
    value.storage_offset     = storage_offset;
    value.storage_byte_count = storage_byte_count;
    value.type               = type;
    value.ne                 = { element_count, 1, 1, 1 };
    value.nb                 = { ggml_type_size(type), ggml_type_size(type) * static_cast<size_t>(element_count),
                                 ggml_type_size(type) * static_cast<size_t>(element_count),
                                 ggml_type_size(type) * static_cast<size_t>(element_count) };
    value.element_count      = element_count;
    value.byte_count         = ggml_row_size(type, element_count);
    value.contiguous         = true;
    return value;
}

static std::vector<float> make_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 29) - 14) * 0.125f;
    }
    return data;
}

static std::vector<float> make_weight(int64_t hidden_size) {
    std::vector<float> data(hidden_size);
    for (int64_t i = 0; i < hidden_size; ++i) {
        data[i] = 0.5f + static_cast<float>(i % 17) * 0.03125f;
    }
    return data;
}

static const char * expected_plain_rmsnorm_binary_kernel(int64_t hidden_size, int64_t token_count) {
    return token_count < 6 && hidden_size >= 2048 && hidden_size <= 5120 ?
               "loom_libs:ggml_rmsnorm_binary_f32_lowtoken_striped" :
               "loom_libs:ggml_rmsnorm_binary_f32";
}

static std::vector<float> make_router_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data(hidden_size * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 41) - 20) * 0.01f;
    }
    return data;
}

static std::vector<float> make_router_weight(int64_t hidden_size, int64_t expert_count) {
    std::vector<float> data(hidden_size * expert_count);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t column = 0; column < hidden_size; ++column) {
            data[expert * hidden_size + column] = static_cast<float>(((expert + column) % 31) - 15) * 0.0025f;
        }
    }
    return data;
}

static std::vector<float> make_router_logits(int64_t token_count) {
    std::vector<float> data(kQwenRouterExpertCount * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>(i % kQwenRouterRouteCount);
    }
    return data;
}

static std::vector<float> make_flash_query(int64_t token_count, int64_t head_size = kQwenFlashHeadSize) {
    std::vector<float> data(static_cast<size_t>(head_size) * static_cast<size_t>(token_count));
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 37) - 18) * 0.01f;
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_key_value(int64_t token_count,
                                                     int     offset,
                                                     int64_t head_size = kQwenFlashHeadSize) {
    std::vector<ggml_fp16_t> data(static_cast<size_t>(head_size) * static_cast<size_t>(token_count));
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        const float value = static_cast<float>(((i + offset) % 31) - 15) * 0.015f;
        data[i]           = ggml_fp32_to_fp16(value);
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_mask(int64_t query_token_count, int64_t key_value_token_count) {
    std::vector<ggml_fp16_t> data(query_token_count * key_value_token_count);
    for (int64_t query = 0; query < query_token_count; ++query) {
        for (int64_t key = 0; key < key_value_token_count; ++key) {
            const float value                         = key <= query + 1 ? 0.0f : -10000.0f;
            data[query * key_value_token_count + key] = ggml_fp32_to_fp16(value);
        }
    }
    return data;
}

static std::vector<float> rmsnorm_mul_reference(const std::vector<float> & input,
                                                const std::vector<float> & weight,
                                                int64_t                    hidden_size,
                                                int64_t                    token_count) {
    std::vector<float> output(input.size());
    for (int64_t token = 0; token < token_count; ++token) {
        float sum_squares = 0.0f;
        for (int64_t column = 0; column < hidden_size; ++column) {
            const float value = input[token * hidden_size + column];
            sum_squares += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_size) + kQwenRmsNormEps);
        for (int64_t column = 0; column < hidden_size; ++column) {
            output[token * hidden_size + column] = input[token * hidden_size + column] * scale * weight[column];
        }
    }
    return output;
}

static std::vector<float> rmsnorm_reference(const std::vector<float> & input,
                                            int64_t                    hidden_size,
                                            int64_t                    token_count,
                                            float                      epsilon) {
    std::vector<float> output(input.size());
    for (int64_t token = 0; token < token_count; ++token) {
        float sum_squares = 0.0f;
        for (int64_t column = 0; column < hidden_size; ++column) {
            const float value = input[token * hidden_size + column];
            sum_squares += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_size) + epsilon);
        for (int64_t column = 0; column < hidden_size; ++column) {
            output[token * hidden_size + column] = input[token * hidden_size + column] * scale;
        }
    }
    return output;
}

static std::vector<float> router_projection_reference(const std::vector<float> & input,
                                                      const std::vector<float> & weight,
                                                      int64_t                    hidden_size,
                                                      int64_t                    expert_count,
                                                      int64_t                    token_count) {
    std::vector<float> output(expert_count * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            float sum = 0.0f;
            for (int64_t column = 0; column < hidden_size; ++column) {
                sum += input[token * hidden_size + column] * weight[expert * hidden_size + column];
            }
            output[token * expert_count + expert] = sum;
        }
    }
    return output;
}

static std::vector<float> router_top8_weights_reference(const std::vector<float> & logits, int64_t token_count) {
    std::vector<float> output(kQwenRouterRouteCount * token_count);
    for (int64_t token = 0; token < token_count; ++token) {
        bool    used[kQwenRouterExpertCount]    = {};
        int64_t selected[kQwenRouterRouteCount] = {};
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            int64_t best_expert = -1;
            float   best_value  = -std::numeric_limits<float>::infinity();
            for (int64_t expert = 0; expert < kQwenRouterExpertCount; ++expert) {
                const float value = logits[token * kQwenRouterExpertCount + expert];
                if (!used[expert] &&
                    (best_expert < 0 || value > best_value || (value == best_value && expert < best_expert))) {
                    best_value  = value;
                    best_expert = expert;
                }
            }
            selected[route]   = best_expert;
            used[best_expert] = true;
        }

        float max_selected = -std::numeric_limits<float>::infinity();
        for (const int64_t expert : selected) {
            max_selected = std::max(max_selected, logits[token * kQwenRouterExpertCount + expert]);
        }
        float sum = 0.0f;
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            const float value = std::exp(logits[token * kQwenRouterExpertCount + selected[route]] - max_selected);
            output[token * kQwenRouterRouteCount + route] = value;
            sum += value;
        }
        for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
            output[token * kQwenRouterRouteCount + route] /= sum;
        }
    }
    return output;
}

static std::vector<float> flash_attention_reference(const std::vector<float> &       query,
                                                    const std::vector<ggml_fp16_t> & key,
                                                    const std::vector<ggml_fp16_t> & value,
                                                    const std::vector<ggml_fp16_t> & mask,
                                                    int64_t                          query_token_count,
                                                    int64_t                          key_value_token_count,
                                                    int64_t                          qk_head_size = kQwenFlashHeadSize,
                                                    int64_t                          value_head_size = -1,
                                                    float                            scale           = 0.0f) {
    const int64_t      actual_value_head_size = value_head_size > 0 ? value_head_size : qk_head_size;
    std::vector<float> output(static_cast<size_t>(query_token_count) * static_cast<size_t>(actual_value_head_size));
    const float        actual_scale = scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(qk_head_size)) : scale;
    for (int64_t query_token = 0; query_token < query_token_count; ++query_token) {
        std::vector<float> scores(key_value_token_count);
        float              max_score = -std::numeric_limits<float>::infinity();
        for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
            float dot = 0.0f;
            for (int64_t channel = 0; channel < qk_head_size; ++channel) {
                dot += query[query_token * qk_head_size + channel] *
                       ggml_fp16_to_fp32(key[key_token * qk_head_size + channel]);
            }
            const float score =
                dot * actual_scale + ggml_fp16_to_fp32(mask[query_token * key_value_token_count + key_token]);
            scores[key_token] = score;
            max_score         = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (int64_t channel = 0; channel < actual_value_head_size; ++channel) {
            float weighted_sum = 0.0f;
            for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
                const float probability = scores[key_token] / sum;
                weighted_sum += probability * ggml_fp16_to_fp32(value[key_token * actual_value_head_size + channel]);
            }
            output[query_token * actual_value_head_size + channel] = weighted_sum;
        }
    }
    return output;
}

static ggml_tensor * build_rmsnorm_mul_graph(ggml_context * ctx,
                                             ggml_tensor *  input,
                                             ggml_tensor *  weight,
                                             float          eps = kQwenRmsNormEps) {
    ggml_tensor * rms = ggml_rms_norm(ctx, input, eps);
    REQUIRE(rms != nullptr);
    ggml_tensor * output = ggml_mul(ctx, rms, weight);
    REQUIRE(output != nullptr);
    return output;
}

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      ggml_tensor *  query,
                                                      ggml_tensor *  key,
                                                      ggml_tensor *  value,
                                                      ggml_tensor *  mask,
                                                      int64_t        head_size = kQwenFlashHeadSize,
                                                      float          scale     = 0.0f) {
    const float   actual_scale = scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(head_size)) : scale;
    ggml_tensor * output       = ggml_flash_attn_ext(ctx, query, key, value, mask, actual_scale, 0.0f, 0.0f);
    REQUIRE(output != nullptr);
    return output;
}

static ggml_tensor * build_qwen_router_top8_graph(ggml_context * ctx,
                                                  ggml_tensor *  logits,
                                                  ggml_tensor ** route_ids = nullptr) {
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    REQUIRE(probs != nullptr);
    ggml_tensor * probs_reshaped = ggml_reshape_3d(ctx, probs, 1, kQwenRouterExpertCount, logits->ne[1]);
    REQUIRE(probs_reshaped != nullptr);
    ggml_tensor * argsort = ggml_argsort(ctx, probs, GGML_SORT_ORDER_DESC);
    REQUIRE(argsort != nullptr);
    ggml_tensor * topk = ggml_view_2d(ctx, argsort, kQwenRouterRouteCount, logits->ne[1], argsort->nb[1], 0);
    REQUIRE(topk != nullptr);
    if (route_ids != nullptr) {
        *route_ids = topk;
    }
    ggml_tensor * selected = ggml_get_rows(ctx, probs_reshaped, topk);
    REQUIRE(selected != nullptr);
    ggml_tensor * selected_reshaped = ggml_reshape_2d(ctx, selected, kQwenRouterRouteCount, logits->ne[1]);
    REQUIRE(selected_reshaped != nullptr);
    ggml_tensor * sum = ggml_sum_rows(ctx, selected_reshaped);
    REQUIRE(sum != nullptr);
    ggml_tensor * clamped_sum = ggml_clamp(ctx, sum, 1.0e-7f, std::numeric_limits<float>::infinity());
    REQUIRE(clamped_sum != nullptr);
    ggml_tensor * normalized = ggml_div(ctx, selected_reshaped, clamped_sum);
    REQUIRE(normalized != nullptr);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, kQwenRouterRouteCount, logits->ne[1]);
    REQUIRE(output != nullptr);
    return output;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static std::vector<std::string> scheduled_kernel_sequence(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics)) {
        for (const std::string & error : scheduler.plan().status.errors()) {
            std::fprintf(stderr, "scheduler error: %s\n", error.c_str());
        }
        std::fprintf(stderr, "unsupported: %s\n", diagnostics.unsupported_message.c_str());
        for (const ggml::hrx::DispatchRegistrationAttempt & attempt : diagnostics.match.attempts) {
            std::fprintf(stderr, "  attempt %s matched=%d\n", attempt.name.c_str(), attempt.matched ? 1 : 0);
            if (!attempt.covered_nodes.empty()) {
                std::fprintf(stderr, "    covered:");
                for (size_t node : attempt.covered_nodes) {
                    std::fprintf(stderr, " %zu", node);
                }
                std::fprintf(stderr, "\n");
            }
            for (const std::string & error : attempt.errors) {
                std::fprintf(stderr, "    %s\n", error.c_str());
            }
        }
        std::abort();
    }
    REQUIRE(scheduler.plan().valid());

    std::vector<std::string> names;
    names.reserve(scheduler.plan().dispatches.size());
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        names.push_back(kernel_name_for_id(dispatch.kernel.kernel_id));
    }
    return names;
}

static ggml::hrx::KernelSpecialization scheduled_kernel_specialization(ggml_cgraph * graph,
                                                                        const char *  kernel_name) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    REQUIRE(scheduler.plan().valid());
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == kernel_name) {
            return dispatch.kernel;
        }
    }
    REQUIRE(false);
    return {};
}

static size_t producer_index_for_tensor(const ggml::hrx::Graph & graph, const ggml_tensor * tensor) {
    const ggml::hrx::Value * value = graph.values().find_tensor(tensor);
    REQUIRE(value != nullptr);
    const ggml::hrx::GraphNode * producer = graph.index().producer(value->id);
    REQUIRE(producer != nullptr);
    size_t index = 0;
    REQUIRE(graph.index().node_index(producer, index));
    return index;
}

static ggml::hrx::ValueId next_plan_value(const ggml::hrx::Graph & graph, const ggml::hrx::CommandPlan & plan) {
    return ggml::hrx::ValueId(
        static_cast<int32_t>(graph.values().size() + plan.transients.size() + plan.completion_counter_requests.size()));
}

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes) {
    for (ggml::hrx::Dispatch & dispatch : match.initialization_dispatches) {
        plan.initialization_dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::Dispatch & dispatch : match.dispatches) {
        plan.dispatches.push_back(std::move(dispatch));
    }
    for (ggml::hrx::CommandPlanTransient & transient : match.transients) {
        plan.transients.push_back(std::move(transient));
    }
    for (ggml::hrx::CommandPlanConstantInitialization & initialization : match.constant_initializations) {
        plan.constant_initializations.push_back(std::move(initialization));
    }
    for (ggml::hrx::CommandPlanCompletionCounterRequest & request : match.completion_counter_requests) {
        plan.completion_counter_requests.push_back(std::move(request));
    }
    REQUIRE(plan.metadata.append(std::move(match.metadata), plan.status));
    plan.status.append(match.status);
    for (size_t covered_node : match.covered_nodes) {
        REQUIRE(covered_node < covered_nodes.size());
        REQUIRE(!covered_nodes[covered_node]);
        covered_nodes[covered_node] = true;
    }
}

static void match_dispatch_at_index(const ggml::hrx::Graph &            graph,
                                    const ggml::hrx::DispatchRegistry & registry,
                                    ggml::hrx::CommandPlan &            plan,
                                    std::vector<bool> &                 covered_nodes,
                                    size_t                              root_index,
                                    ggml::hrx::DispatchMatch &          match) {
    REQUIRE(root_index < graph.nodes().size());
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes()[root_index], root_index, covered_nodes, plan, next_plan_value(graph, plan), &registry,
    };
    ggml::hrx::DispatchMatchDiagnostics diagnostics;
    if (!registry.match(context, match, &diagnostics)) {
        std::fprintf(stderr, "manual matcher failed for node %zu %s\n", root_index,
                     ggml_op_name(graph.nodes()[root_index].op));
        const ggml::hrx::GraphNode & node = graph.nodes()[root_index];
        for (size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
            const ggml::hrx::Value * input = graph.values().find(node.inputs[input_index]);
            if (input != nullptr) {
                std::fprintf(stderr,
                             "  input %zu value=%d type=%d ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                             "] nb=[%zu,%zu,%zu,%zu]\n",
                             input_index, input->id.value, static_cast<int>(input->type), input->ne[0], input->ne[1],
                             input->ne[2], input->ne[3], input->nb[0], input->nb[1], input->nb[2], input->nb[3]);
            }
        }
        const ggml::hrx::Value * output = graph.values().find(node.output);
        if (output != nullptr) {
            std::fprintf(stderr,
                         "  output value=%d type=%d ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                         "] nb=[%zu,%zu,%zu,%zu]\n",
                         output->id.value, static_cast<int>(output->type), output->ne[0], output->ne[1], output->ne[2],
                         output->ne[3], output->nb[0], output->nb[1], output->nb[2], output->nb[3]);
        }
        for (const ggml::hrx::CommandPlanAlternateValue & alternate : plan.metadata.alternate_values()) {
            std::fprintf(stderr, "  alternate graph=%d value=%d type=%d bytes=%zu name=%s\n",
                         alternate.graph_value.value, alternate.alternate_value.value, static_cast<int>(alternate.type),
                         alternate.byte_count, alternate.name.c_str());
        }
        for (const ggml::hrx::DispatchRegistrationAttempt & attempt : diagnostics.attempts) {
            std::fprintf(stderr, "  attempt %s matched=%d\n", attempt.name.c_str(), attempt.matched ? 1 : 0);
            for (const std::string & error : attempt.errors) {
                std::fprintf(stderr, "    %s\n", error.c_str());
            }
        }
        std::abort();
    }
    append_match_to_plan(plan, match, covered_nodes);
    REQUIRE(plan.valid());
}

static void require_kernel_subsequence(const std::vector<std::string> & sequence,
                                       const std::vector<std::string> & expected) {
    size_t sequence_index = 0;
    for (const std::string & name : expected) {
        while (sequence_index < sequence.size() && sequence[sequence_index] != name) {
            ++sequence_index;
        }
        if (sequence_index >= sequence.size()) {
            std::fprintf(stderr, "missing expected kernel: %s\nscheduled kernels:\n", name.c_str());
            for (const std::string & scheduled : sequence) {
                std::fprintf(stderr, "  %s\n", scheduled.c_str());
            }
            std::abort();
        }
        ++sequence_index;
    }
}

static void run_alternate_value_alias_lookup_checks() {
    constexpr int64_t element_count   = 2048;
    const size_t      full_bytes      = ggml_row_size(GGML_TYPE_F32, element_count);
    const size_t      f16_bytes       = ggml_row_size(GGML_TYPE_F16, element_count);
    const size_t      q8_bytes        = ggml_row_size(GGML_TYPE_Q8_1, element_count);
    const size_t      symmetric_bytes = 1536;

    ggml::hrx::Graph       graph;
    ggml::hrx::Status      status;
    ggml::hrx::CommandPlan plan;

    const ggml::hrx::ValueId        root(0);
    const ggml::hrx::ValueId        full_alias(1);
    const ggml::hrx::ValueId        partial_alias(2);
    const ggml::hrx::ValueId        reordered_alias(3);
    const ggml::hrx::ValueId        unrelated_same_storage(4);
    const ggml::hrx::ValueId        different_shape_alias(5);
    const ggml::hrx::ValueId        incompatible_q8_shape_alias(6);
    const ggml::hrx::ValueId        f16_alternate(99);
    const ggml::hrx::ValueId        q8_alternate(100);
    const ggml::hrx::ValueStorageId storage(0);

    status = graph.values().add_snapshot_storage({ storage, root, full_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(root, storage, root, ggml::hrx::ValueId(), 0, full_bytes, GGML_TYPE_F32, element_count));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(full_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(partial_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count / 2));
    REQUIRE(status.success());
    ggml::hrx::Value reordered =
        make_test_value(reordered_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count);
    reordered.contiguous = false;
    status               = graph.values().add_snapshot_value(std::move(reordered));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_test_value(unrelated_same_storage, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count));
    REQUIRE(status.success());
    ggml::hrx::Value different_shape =
        make_test_value(different_shape_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count);
    different_shape.ne = { element_count / 2, 2, 1, 1 };
    different_shape.nb = { sizeof(float), static_cast<size_t>(element_count / 2) * sizeof(float), full_bytes,
                           full_bytes };
    status             = graph.values().add_snapshot_value(std::move(different_shape));
    REQUIRE(status.success());
    ggml::hrx::Value incompatible_q8_shape =
        make_test_value(incompatible_q8_shape_alias, storage, root, root, 0, full_bytes, GGML_TYPE_F32, element_count);
    incompatible_q8_shape.ne = { 64, element_count / 64, 1, 1 };
    incompatible_q8_shape.nb = { sizeof(float), 64 * sizeof(float), full_bytes, full_bytes };
    status = graph.values().add_snapshot_value(std::move(incompatible_q8_shape));
    REQUIRE(status.success());

    graph.add_node(GGML_OP_RESHAPE, full_alias, { root });
    graph.add_node(GGML_OP_VIEW, partial_alias, { root });
    graph.add_node(GGML_OP_TRANSPOSE, reordered_alias, { root });
    graph.add_node(GGML_OP_CPY, unrelated_same_storage, { root });
    graph.add_node(GGML_OP_RESHAPE, different_shape_alias, { root });
    graph.add_node(GGML_OP_RESHAPE, incompatible_q8_shape_alias, { root });
    status = graph.build_index();
    REQUIRE(status.success());

    REQUIRE(plan.metadata.append_alternate_value({ root, q8_alternate, GGML_TYPE_Q8_1, q8_bytes, "q8" }, status));
    REQUIRE(plan.metadata.append_alternate_value({ root, f16_alternate, GGML_TYPE_F16, f16_bytes, "f16" }, status));
    REQUIRE(plan.metadata.append_alternate_value(
        { root, ggml::hrx::ValueId(104), GGML_TYPE_COUNT, symmetric_bytes, "symmetric-i4" }, status));

    const ggml::hrx::CommandPlanAlternateValue * exact =
        ggml::hrx::find_alternate_value(graph, plan, root, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(exact != nullptr);
    REQUIRE(exact->alternate_value == q8_alternate);

    const ggml::hrx::CommandPlanAlternateValue * through_full_alias =
        ggml::hrx::find_alternate_value(graph, plan, full_alias, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(through_full_alias != nullptr);
    REQUIRE(through_full_alias->alternate_value == q8_alternate);

    const ggml::hrx::CommandPlanAlternateValue * through_partial_alias =
        ggml::hrx::find_alternate_value(graph, plan, partial_alias, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(through_partial_alias == nullptr);

    const ggml::hrx::CommandPlanAlternateValue * through_reordered_alias =
        ggml::hrx::find_alternate_value(graph, plan, reordered_alias, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(through_reordered_alias == nullptr);

    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, unrelated_same_storage, GGML_TYPE_Q8_1, q8_bytes) == nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, different_shape_alias, GGML_TYPE_Q8_1, q8_bytes) != nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, incompatible_q8_shape_alias, GGML_TYPE_Q8_1, q8_bytes) ==
            nullptr);

    const ggml::hrx::CommandPlanAlternateValue * f16_exact =
        ggml::hrx::find_alternate_value(graph, plan, root, GGML_TYPE_F16, f16_bytes);
    REQUIRE(f16_exact != nullptr);
    REQUIRE(f16_exact->alternate_value == f16_alternate);
    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, different_shape_alias, GGML_TYPE_F16, f16_bytes) != nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, full_alias, GGML_TYPE_COUNT, symmetric_bytes) != nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(graph, plan, different_shape_alias, GGML_TYPE_COUNT, symmetric_bytes) ==
            nullptr);

    ggml::hrx::Status conflict_status;
    REQUIRE(!plan.metadata.append_alternate_value(
        { root, ggml::hrx::ValueId(101), GGML_TYPE_Q8_1, q8_bytes, "q8-conflict" }, conflict_status));

    ggml::hrx::Status        generated_status;
    const ggml::hrx::ValueId k16_value(102);
    REQUIRE(plan.metadata.append_generated_resource(
        { root, ggml::hrx::GeneratedResourceRole::F16K16Major, k16_value, f16_bytes, {} }, generated_status));
    const ggml::hrx::CommandPlanGeneratedResource * generated_through_alias = ggml::hrx::find_generated_resource(
        graph, plan, full_alias, ggml::hrx::GeneratedResourceRole::F16K16Major, f16_bytes);
    REQUIRE(generated_through_alias != nullptr);
    REQUIRE(generated_through_alias->generated_value == k16_value);
    REQUIRE(ggml::hrx::find_generated_resource(graph, plan, partial_alias,
                                               ggml::hrx::GeneratedResourceRole::F16K16Major, f16_bytes) == nullptr);
    REQUIRE(ggml::hrx::find_generated_resource(graph, plan, reordered_alias,
                                               ggml::hrx::GeneratedResourceRole::F16K16Major, f16_bytes) == nullptr);
    REQUIRE(ggml::hrx::find_generated_resource(graph, plan, different_shape_alias,
                                               ggml::hrx::GeneratedResourceRole::F16K16Major, f16_bytes) == nullptr);
}

static bool publication_accepts_add(const ggml::hrx::DispatchMatchContext &,
                                    const ggml::hrx::GraphNode & consumer,
                                    const ggml::hrx::Value &     input) {
    return consumer.op == GGML_OP_ADD &&
           std::find(consumer.inputs.begin(), consumer.inputs.end(), input.id) != consumer.inputs.end();
}

static bool publication_accepts_mul(const ggml::hrx::DispatchMatchContext &,
                                    const ggml::hrx::GraphNode & consumer,
                                    const ggml::hrx::Value &     input) {
    return consumer.op == GGML_OP_MUL &&
           std::find(consumer.inputs.begin(), consumer.inputs.end(), input.id) != consumer.inputs.end();
}

static bool publication_accepts_mul_output_13(const ggml::hrx::DispatchMatchContext & context,
                                              const ggml::hrx::GraphNode &            consumer,
                                              const ggml::hrx::Value &                input) {
    return consumer.output == ggml::hrx::ValueId(13) && publication_accepts_mul(context, consumer, input);
}

static bool publication_accepts_mul_output_15(const ggml::hrx::DispatchMatchContext & context,
                                              const ggml::hrx::GraphNode &            consumer,
                                              const ggml::hrx::Value &                input) {
    return consumer.output == ggml::hrx::ValueId(15) && publication_accepts_mul(context, consumer, input);
}

static void run_activation_publication_contract_checks() {
    constexpr int64_t row_size  = 2048;
    constexpr int64_t row_count = 2;
    constexpr int64_t elements  = row_size * row_count;
    const size_t      f32_bytes = ggml_row_size(GGML_TYPE_F32, elements);

    auto make_activation = [&](ggml::hrx::ValueId id, ggml::hrx::ValueStorageId storage, ggml::hrx::ValueId root,
                               ggml::hrx::ValueId alias, int64_t element_count = elements) {
        ggml::hrx::Value value = make_test_value(id, storage, root, alias, 0, f32_bytes, GGML_TYPE_F32, element_count);
        if (element_count == elements) {
            value.ne = { row_size, row_count, 1, 1 };
            value.nb = { sizeof(float), static_cast<size_t>(row_size) * sizeof(float), f32_bytes, f32_bytes };
        }
        return value;
    };

    ggml::hrx::Graph         graph;
    ggml::hrx::Status        status;
    const ggml::hrx::ValueId root(0);
    const ggml::hrx::ValueId reshaped(1);
    const ggml::hrx::ValueId partial(2);
    const ggml::hrx::ValueId transposed(3);
    const ggml::hrx::ValueId different_shape(8);
    const ggml::hrx::ValueId incompatible_q8_shape(10);
    const ggml::hrx::ValueId shape_back(12);
    const ggml::hrx::ValueId mismatched_alias_source(14);
    status = graph.values().add_snapshot_storage({ ggml::hrx::ValueStorageId(0), root, f32_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_activation(root, ggml::hrx::ValueStorageId(0), root, ggml::hrx::ValueId()));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(make_activation(reshaped, ggml::hrx::ValueStorageId(0), root, root));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_activation(partial, ggml::hrx::ValueStorageId(0), root, root, elements / 2));
    REQUIRE(status.success());
    ggml::hrx::Value reordered = make_activation(transposed, ggml::hrx::ValueStorageId(0), root, root);
    reordered.contiguous       = false;
    status                     = graph.values().add_snapshot_value(std::move(reordered));
    REQUIRE(status.success());

    for (int32_t value = 4; value < 8; ++value) {
        const ggml::hrx::ValueStorageId storage(value - 3);
        const ggml::hrx::ValueId        id(value);
        status = graph.values().add_snapshot_storage({ storage, id, f32_bytes });
        REQUIRE(status.success());
        status = graph.values().add_snapshot_value(make_activation(id, storage, id, ggml::hrx::ValueId()));
        REQUIRE(status.success());
    }
    ggml::hrx::Value different = make_activation(different_shape, ggml::hrx::ValueStorageId(0), root, root);
    different.ne               = { row_size / 2, row_count * 2, 1, 1 };
    different.nb = { sizeof(float), static_cast<size_t>(row_size / 2) * sizeof(float), f32_bytes, f32_bytes };
    status       = graph.values().add_snapshot_value(std::move(different));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_storage({ ggml::hrx::ValueStorageId(5), ggml::hrx::ValueId(9), f32_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(make_activation(ggml::hrx::ValueId(9), ggml::hrx::ValueStorageId(5),
                                                               ggml::hrx::ValueId(9), ggml::hrx::ValueId()));
    REQUIRE(status.success());
    ggml::hrx::Value incompatible = make_activation(incompatible_q8_shape, ggml::hrx::ValueStorageId(0), root, root);
    incompatible.ne               = { 64, elements / 64, 1, 1 };
    incompatible.nb               = { sizeof(float), 64 * sizeof(float), f32_bytes, f32_bytes };
    status                        = graph.values().add_snapshot_value(std::move(incompatible));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_storage({ ggml::hrx::ValueStorageId(6), ggml::hrx::ValueId(11), f32_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(make_activation(ggml::hrx::ValueId(11), ggml::hrx::ValueStorageId(6),
                                                               ggml::hrx::ValueId(11), ggml::hrx::ValueId()));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(
        make_activation(shape_back, ggml::hrx::ValueStorageId(0), root, different_shape));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_storage({ ggml::hrx::ValueStorageId(7), ggml::hrx::ValueId(13), f32_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(make_activation(ggml::hrx::ValueId(13), ggml::hrx::ValueStorageId(7),
                                                               ggml::hrx::ValueId(13), ggml::hrx::ValueId()));
    REQUIRE(status.success());
    ggml::hrx::Value mismatched = make_activation(
        mismatched_alias_source, ggml::hrx::ValueStorageId(0), root, root);
    mismatched.ne = { row_size / 2, row_count * 2, 1, 1 };
    mismatched.nb = { sizeof(float), static_cast<size_t>(row_size / 2) * sizeof(float), f32_bytes, f32_bytes };
    status = graph.values().add_snapshot_value(std::move(mismatched));
    REQUIRE(status.success());
    status = graph.values().add_snapshot_storage({ ggml::hrx::ValueStorageId(8), ggml::hrx::ValueId(15), f32_bytes });
    REQUIRE(status.success());
    status = graph.values().add_snapshot_value(make_activation(ggml::hrx::ValueId(15), ggml::hrx::ValueStorageId(8),
                                                               ggml::hrx::ValueId(15), ggml::hrx::ValueId()));
    REQUIRE(status.success());

    graph.add_node(GGML_OP_RESHAPE, reshaped, { root });
    graph.add_node(GGML_OP_VIEW, partial, { root });
    graph.add_node(GGML_OP_TRANSPOSE, transposed, { root });
    graph.add_node(GGML_OP_ADD, ggml::hrx::ValueId(4), { root, root });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(5), { reshaped, reshaped });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(6), { partial, partial });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(7), { transposed, transposed });
    graph.add_node(GGML_OP_RESHAPE, different_shape, { root });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(9), { different_shape, different_shape });
    graph.add_node(GGML_OP_RESHAPE, incompatible_q8_shape, { root });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(11), { incompatible_q8_shape, incompatible_q8_shape });
    graph.add_node(GGML_OP_RESHAPE, shape_back, { different_shape });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(13), { shape_back, shape_back });
    graph.add_node(GGML_OP_RESHAPE, mismatched_alias_source, { different_shape });
    graph.add_node(GGML_OP_MUL, ggml::hrx::ValueId(15), { mismatched_alias_source, mismatched_alias_source });
    status = graph.build_index();
    REQUIRE(status.success());

    ggml::hrx::DispatchRegistryBuilder publication_registry_builder;
    publication_registry_builder.add_activation_consumer({
        "test.q8.mul",
        GGML_OP_MUL,
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationConsumerUse::BandwidthLimited,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_mul,
    });
    publication_registry_builder.add_activation_consumer({
        "test.f16-row.add",
        GGML_OP_ADD,
        ggml::hrx::DispatchActivationInputF16Row,
        ggml::hrx::DispatchActivationConsumerUse::Row,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_add,
    });
    const ggml::hrx::DispatchRegistry publication_registry = publication_registry_builder.build();
    const std::vector<bool>               covered(graph.nodes().size(), false);
    ggml::hrx::CommandPlan                plan;
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(100), &publication_registry,
    };
    const ggml::hrx::Value * produced = graph.values().find(root);
    REQUIRE(produced != nullptr);

    const ggml::hrx::CommonActivationPublicationCapabilities direct_only = {
        ggml::hrx::DispatchActivationInputQ8_1X4,
    };
    const ggml::hrx::CommonActivationPublicationPlan direct_only_plan =
        ggml::hrx::common_select_activation_publication_plan(context, *produced, direct_only);
    REQUIRE(!direct_only_plan.matched());
    REQUIRE(direct_only_plan.fallback_reason ==
            ggml::hrx::CommonActivationPublicationFallbackReason::NoQualifiedConsumer);

    const ggml::hrx::CommonActivationPublicationCapabilities q8_capabilities = {
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationInputQ8_1X4,
    };
    const ggml::hrx::CommonActivationPublicationPlan q8_plan =
        ggml::hrx::common_select_activation_publication_plan(context, *produced, q8_capabilities);
    REQUIRE(q8_plan.matched());
    const ggml::hrx::CommonActivationPublicationDemand q8_demand = q8_plan.preferred();
    REQUIRE(q8_demand.format == ggml::hrx::CommonActivationPublicationFormat::Q8_1X4);
    REQUIRE(q8_demand.consumer_value->id == reshaped);
    REQUIRE(q8_demand.alias_depth == 1);
    REQUIRE(q8_demand.mixed_fanout);

    const ggml::hrx::CommonActivationPublicationCapabilities f16_capabilities = {
        ggml::hrx::DispatchActivationInputF16Row,
    };
    const ggml::hrx::CommonActivationPublicationPlan f16_plan =
        ggml::hrx::common_select_activation_publication_plan(context, *produced, f16_capabilities);
    REQUIRE(f16_plan.matched());
    const ggml::hrx::CommonActivationPublicationDemand f16_demand = f16_plan.preferred();
    REQUIRE(f16_demand.format == ggml::hrx::CommonActivationPublicationFormat::F16Row);
    REQUIRE(f16_demand.consumer_value->id == root);
    REQUIRE(f16_demand.alias_depth == 0);
    REQUIRE(f16_demand.mixed_fanout);

    ggml::hrx::DispatchRegistryBuilder direct_k16_registry_builder;
    direct_k16_registry_builder.add_activation_consumer({
        "test.k16.mul",
        GGML_OP_MUL,
        ggml::hrx::DispatchActivationInputF16K16Major,
        ggml::hrx::DispatchActivationConsumerUse::Tiled,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_mul,
    });
    const ggml::hrx::DispatchRegistry direct_k16_registry = direct_k16_registry_builder.build();
    const ggml::hrx::DispatchMatchContext direct_k16_context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(100), &direct_k16_registry,
    };
    const ggml::hrx::CommonActivationPublicationCapabilities mixed_alias_capabilities = {
        ggml::hrx::DispatchActivationInputQ8_1X4 | ggml::hrx::DispatchActivationInputF16K16Major,
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationProducerRequirementNone,
    };
    const ggml::hrx::CommonActivationPublicationDemand direct_k16_through_q8_alias =
        ggml::hrx::common_select_activation_publication_plan(
            direct_k16_context, *produced, mixed_alias_capabilities).publications.front();
    REQUIRE(!direct_k16_through_q8_alias.matched());

    ggml::hrx::DispatchRegistryBuilder shape_back_registry_builder;
    shape_back_registry_builder.add_activation_consumer({
        "test.k16.shape-back",
        GGML_OP_MUL,
        ggml::hrx::DispatchActivationInputF16K16Major,
        ggml::hrx::DispatchActivationConsumerUse::Tiled,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_mul_output_13,
    });
    const ggml::hrx::DispatchRegistry shape_back_registry = shape_back_registry_builder.build();
    const ggml::hrx::DispatchMatchContext shape_back_context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(100), &shape_back_registry,
    };
    REQUIRE(!ggml::hrx::common_select_activation_publication_plan(
                 shape_back_context, *produced, mixed_alias_capabilities).matched());

    ggml::hrx::CommonActivationPublicationCapabilities multiple_capabilities = {
        ggml::hrx::DispatchActivationInputQ8_1X4 | ggml::hrx::DispatchActivationInputF16Row,
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchActivationInputQ8_1X4 | ggml::hrx::DispatchActivationInputF16Row,
    };
    ggml::hrx::DispatchRegistryBuilder mismatched_alias_registry_builder;
    mismatched_alias_registry_builder.add_activation_consumer({
        "test.q8.mismatched-alias-source",
        GGML_OP_MUL,
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationConsumerUse::BandwidthLimited,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_mul_output_15,
    });
    const ggml::hrx::DispatchRegistry mismatched_alias_registry = mismatched_alias_registry_builder.build();
    const ggml::hrx::DispatchMatchContext mismatched_alias_context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(100), &mismatched_alias_registry,
    };
    REQUIRE(!ggml::hrx::common_select_activation_publication_plan(
                 mismatched_alias_context, *produced, multiple_capabilities).matched());

    const ggml::hrx::CommonActivationPublicationPlan multiple_plan =
        ggml::hrx::common_select_activation_publication_plan(context, *produced, multiple_capabilities);
    REQUIRE(multiple_plan.publication_count == 2);
    REQUIRE(multiple_plan.publications[0].format == ggml::hrx::CommonActivationPublicationFormat::Q8_1X4);
    REQUIRE(multiple_plan.publications[1].format == ggml::hrx::CommonActivationPublicationFormat::F16Row);
    ggml::hrx::DispatchRegistryBuilder q8_only_registry_builder;
    q8_only_registry_builder.add_activation_consumer({
        "test.q8.only",
        GGML_OP_MUL,
        ggml::hrx::DispatchActivationInputQ8_1X4,
        ggml::hrx::DispatchActivationConsumerUse::BandwidthLimited,
        ggml::hrx::DispatchActivationProducerRequirementNone,
        ggml::hrx::DispatchSource::Common,
        publication_accepts_mul,
    });
    const ggml::hrx::DispatchRegistry q8_only_registry = q8_only_registry_builder.build();
    const ggml::hrx::DispatchMatchContext q8_only_context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(100), &q8_only_registry,
    };
    const ggml::hrx::CommonActivationPublicationPlan companion_plan =
        ggml::hrx::common_select_activation_publication_plan(q8_only_context, *produced, multiple_capabilities);
    REQUIRE(companion_plan.publication_count == 2);
    REQUIRE(companion_plan.publications[0].format == ggml::hrx::CommonActivationPublicationFormat::Q8_1X4);
    REQUIRE(companion_plan.publications[1].format == ggml::hrx::CommonActivationPublicationFormat::F16Row);
    REQUIRE(companion_plan.publications[1].consumer == companion_plan.publications[0].consumer);
    multiple_capabilities.co_publication_formats = ggml::hrx::DispatchActivationInputNone;
    REQUIRE(ggml::hrx::common_select_activation_publication_plan(context, *produced, multiple_capabilities)
                .publication_count == 1);

    const ggml::hrx::CommonActivationPublicationCapabilities disabled_capabilities;
    const ggml::hrx::CommonActivationPublicationPlan disabled_plan =
        ggml::hrx::common_select_activation_publication_plan(context, *produced, disabled_capabilities);
    REQUIRE(!disabled_plan.matched());
    REQUIRE(disabled_plan.fallback_reason ==
            ggml::hrx::CommonActivationPublicationFallbackReason::NoEnabledCandidates);

    ggml::hrx::DispatchMatch match;
    match.completion_counter_requests.push_back({ ggml::hrx::ValueId(100), "existing.counter", 1 });
    const ggml::hrx::CommonActivationPublicationDemand symmetric_demand = {
        ggml::hrx::CommonActivationPublicationFormat::SymmetricI4K32,
        produced,
        &graph.nodes()[3],
    };
    ggml::hrx::CommonActivationPublication symmetric;
    ggml::hrx::CommonActivationPublication q8;
    ggml::hrx::CommonActivationPublication f16;
    ggml::hrx::CommonActivationPublication k16;
    REQUIRE(ggml::hrx::common_reserve_activation_publication(context, match, *produced, symmetric_demand,
                                                             "test.symmetric_i4", symmetric));
    REQUIRE(ggml::hrx::common_append_activation_publication(match, symmetric, "test.symmetric_i4"));
    REQUIRE(ggml::hrx::common_reserve_activation_publication(context, match, *produced, q8_demand, "test.q8", q8));
    REQUIRE(ggml::hrx::common_append_activation_publication(match, q8, "test.q8"));
    REQUIRE(ggml::hrx::common_reserve_activation_publication(context, match, *produced, f16_demand, "test.f16", f16));
    REQUIRE(ggml::hrx::common_append_activation_publication(match, f16, "test.f16"));
    const ggml::hrx::CommonActivationPublicationDemand k16_demand = {
        ggml::hrx::CommonActivationPublicationFormat::F16K16Major,
        q8_demand.consumer_value,
        q8_demand.consumer,
    };
    REQUIRE(ggml::hrx::common_reserve_activation_publication(context, match, *produced, k16_demand, "test.k16", k16));
    REQUIRE(ggml::hrx::common_append_activation_publication(match, k16, "test.k16"));

    ggml::hrx::DispatchMatch              invalid_id_match;
    const ggml::hrx::DispatchMatchContext invalid_id_context = {
        graph, &graph.nodes().front(), 0, covered, plan, ggml::hrx::ValueId(-1),
    };
    ggml::hrx::CommonActivationPublication invalid_id_publication;
    REQUIRE(!ggml::hrx::common_reserve_activation_publication(invalid_id_context, invalid_id_match, *produced,
                                                              f16_demand, "test.invalid_id", invalid_id_publication));
    REQUIRE(invalid_id_match.transients.empty());

    ggml::hrx::DispatchMatch conflict_match;
    REQUIRE(ggml::hrx::common_append_activation_publication(conflict_match, q8, "test.q8"));
    ggml::hrx::CommonActivationPublication conflicting_q8 = q8;
    conflicting_q8.alternate_value                        = ggml::hrx::ValueId(200);
    REQUIRE(!ggml::hrx::common_append_activation_publication(conflict_match, conflicting_q8, "test.q8.conflict"));
    REQUIRE(!conflict_match.status.success());

    REQUIRE(match.status.success());
    REQUIRE(match.transients.size() == 4);
    REQUIRE(symmetric.alternate_value == ggml::hrx::ValueId(101));
    REQUIRE(q8.alternate_value == ggml::hrx::ValueId(102));
    REQUIRE(f16.alternate_value == ggml::hrx::ValueId(103));
    REQUIRE(k16.alternate_value == ggml::hrx::ValueId(104));
    REQUIRE(symmetric.payload_bytes == static_cast<size_t>(elements) / 2);
    REQUIRE(symmetric.scales_offset % 256 == 0);
    REQUIRE(symmetric.sums_offset % 256 == 0);
    REQUIRE(q8.byte_count == static_cast<size_t>(row_count) * ggml_row_size(GGML_TYPE_Q8_1, row_size));
    REQUIRE(f16.byte_count == static_cast<size_t>(elements) * sizeof(ggml_fp16_t));
    REQUIRE(symmetric.binding(symmetric.scales_offset, symmetric.scales_bytes).value == symmetric.alternate_value);

    ggml::hrx::CommandPlan recorded;
    recorded.metadata = std::move(match.metadata);
    const std::vector<ggml::hrx::CommandPlanActivationPublicationDiagnostic> & diagnostics =
        recorded.metadata.activation_publication_diagnostics();
    REQUIRE(diagnostics.size() == 4);
    REQUIRE(diagnostics[1].source_value == root);
    REQUIRE(diagnostics[1].alternate_value == q8.alternate_value);
    REQUIRE(diagnostics[1].consumer_value == reshaped);
    REQUIRE(diagnostics[1].requested_format == "q8-1-x4");
    REQUIRE(diagnostics[1].publication_name == "test.q8");
    REQUIRE(diagnostics[1].publication_stage == "published");
    REQUIRE(diagnostics[1].fallback_reason == "none");
    REQUIRE(diagnostics[1].alias_depth == 1);
    REQUIRE(diagnostics[1].mixed_fanout);
    REQUIRE(ggml::hrx::find_alternate_value(graph, recorded, reshaped, GGML_TYPE_Q8_1, q8.byte_count) != nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(graph, recorded, reshaped, GGML_TYPE_F16, f16.byte_count) != nullptr);
    REQUIRE(ggml::hrx::find_generated_resource(graph, recorded, reshaped, ggml::hrx::GeneratedResourceRole::F16K16Major,
                                               k16.byte_count) != nullptr);
}

static std::vector<float> make_pattern_f32(size_t element_count, int seed, float scale = 0.01f) {
    std::vector<float> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        const int value = static_cast<int>((i * 17 + static_cast<size_t>(seed) * 29) % 97) - 48;
        data[i]         = static_cast<float>(value) * scale;
    }
    return data;
}

static std::vector<float> make_gemma_scaled_embedding_input(int64_t hidden_size, int64_t token_count) {
    std::vector<float> data            = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 29, 0.0005f);
    const float        embedding_scale = std::sqrt(static_cast<float>(hidden_size));
    for (float & value : data) {
        value *= embedding_scale;
    }
    return data;
}

static std::vector<int32_t> make_i32_mod_data(size_t element_count, int32_t modulo) {
    std::vector<int32_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = static_cast<int32_t>(i % static_cast<size_t>(modulo));
    }
    return data;
}

static std::vector<int64_t> make_i64_mod_data(size_t element_count, int64_t modulo) {
    std::vector<int64_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = static_cast<int64_t>(i % static_cast<size_t>(modulo));
    }
    return data;
}

static std::vector<ggml_fp16_t> make_pattern_f16(size_t element_count, int seed, float scale = 0.01f) {
    const std::vector<float> f32 = make_pattern_f32(element_count, seed, scale);
    std::vector<ggml_fp16_t> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        data[i] = ggml_fp32_to_fp16(f32[i]);
    }
    return data;
}

static std::vector<uint8_t> make_quantized_rows(ggml_type type, int64_t row_length, int64_t row_count, int seed) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    REQUIRE(traits != nullptr);
    ggml_quantize_init(type);
    const size_t         row_size = ggml_row_size(type, row_length);
    std::vector<uint8_t> data(static_cast<size_t>(row_count) * row_size);
    if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS) {
        std::vector<float> source(static_cast<size_t>(row_length * row_count));
        std::vector<float> importance(static_cast<size_t>(row_length), 1.0f);
        for (int64_t r = 0; r < row_count; ++r) {
            for (int64_t c = 0; c < row_length; ++c) {
                const int value = static_cast<int>((r * 13 + c * 7 + seed * 31) % 101) - 50;
                source[static_cast<size_t>(r * row_length + c)] = static_cast<float>(value) * 0.005f;
            }
        }
        const size_t bytes_written = type == GGML_TYPE_IQ2_XXS ?
                                         quantize_iq2_xxs(source.data(), data.data(), row_count, row_length,
                                                          importance.data()) :
                                         quantize_iq2_xs(source.data(), data.data(), row_count, row_length,
                                                         importance.data());
        REQUIRE(bytes_written == data.size());
        return data;
    }
    REQUIRE(traits->from_float_ref != nullptr);
    std::vector<float>   row(static_cast<size_t>(row_length));
    for (int64_t r = 0; r < row_count; ++r) {
        for (int64_t c = 0; c < row_length; ++c) {
            const int value             = static_cast<int>((r * 13 + c * 7 + seed * 31) % 101) - 50;
            row[static_cast<size_t>(c)] = static_cast<float>(value) * 0.005f;
        }
        traits->from_float_ref(row.data(), data.data() + static_cast<size_t>(r) * row_size, row_length);
    }
    return data;
}

static std::vector<uint8_t> make_iq1_rows(ggml_type type, int64_t row_length, int64_t row_count, int seed) {
    REQUIRE(type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M);
    REQUIRE(row_length % 256 == 0);
    const size_t row_size   = ggml_row_size(type, row_length);
    const size_t block_size = type == GGML_TYPE_IQ1_S ? 50 : 56;
    REQUIRE(row_size == static_cast<size_t>(row_length / 256) * block_size);
    std::vector<uint8_t> data(static_cast<size_t>(row_count) * row_size);
    for (int64_t row = 0; row < row_count; ++row) {
        uint8_t * row_data = data.data() + static_cast<size_t>(row) * row_size;
        for (int64_t block = 0; block < row_length / 256; ++block) {
            uint8_t * dst = row_data + static_cast<size_t>(block) * block_size;
            if (type == GGML_TYPE_IQ1_S) {
                const ggml_fp16_t d = ggml_fp32_to_fp16(0.015625f);
                std::memcpy(dst, &d, sizeof(d));
                for (int i = 0; i < 32; ++i) {
                    dst[2 + i] = static_cast<uint8_t>((row * 13 + block * 29 + i * 37 + seed) & 0xff);
                }
                for (int group = 0; group < 8; ++group) {
                    uint16_t qh = 0;
                    for (int entry = 0; entry < 4; ++entry) {
                        qh |= static_cast<uint16_t>(((row + block + group + entry + seed) & 7) << (3 * entry));
                    }
                    qh |= static_cast<uint16_t>(((row + block + group) & 7) << 12);
                    qh |= static_cast<uint16_t>(((row + group) & 1) << 15);
                    std::memcpy(dst + 34 + 2 * group, &qh, sizeof(qh));
                }
            } else {
                for (int i = 0; i < 32; ++i) {
                    dst[i] = static_cast<uint8_t>((row * 17 + block * 31 + i * 41 + seed) & 0xff);
                }
                for (int i = 0; i < 16; ++i) {
                    const uint8_t low  = static_cast<uint8_t>((row + block + i + seed) & 7);
                    const uint8_t high = static_cast<uint8_t>((row + block + i + seed + 3) & 7);
                    dst[32 + i] = static_cast<uint8_t>(low | ((i & 1) ? 0x08 : 0) | (high << 4) |
                                                        ((i & 2) ? 0x80 : 0));
                }
                const uint16_t base_bits = ggml_fp32_to_fp16(0.015625f);
                for (int word = 0; word < 4; ++word) {
                    const uint16_t local_scales = static_cast<uint16_t>(1 | (2 << 3) | (3 << 6) | (4 << 9));
                    const uint16_t base_nibble  = static_cast<uint16_t>((base_bits >> (4 * word)) & 0xf);
                    const uint16_t scales       = static_cast<uint16_t>(local_scales | (base_nibble << 12));
                    std::memcpy(dst + 48 + 2 * word, &scales, sizeof(scales));
                }
            }
        }
    }
    return data;
}

static std::vector<uint8_t> make_matmul_weight_bytes(ggml_type type, int64_t row_length, int64_t row_count, int seed) {
    if (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        return make_iq1_rows(type, row_length, row_count, seed);
    }
    if (type == GGML_TYPE_F32) {
        (void) seed;
        const std::vector<float> weights(static_cast<size_t>(row_length * row_count), 0.00390625f);
        std::vector<uint8_t>     bytes(weights.size() * sizeof(float));
        std::memcpy(bytes.data(), weights.data(), bytes.size());
        return bytes;
    }

    if (type == GGML_TYPE_F16) {
        (void) seed;
        const std::vector<ggml_fp16_t> weights(static_cast<size_t>(row_length * row_count),
                                               ggml_fp32_to_fp16(0.00390625f));
        std::vector<uint8_t>           bytes(weights.size() * sizeof(ggml_fp16_t));
        std::memcpy(bytes.data(), weights.data(), bytes.size());
        return bytes;
    }

    if (type == GGML_TYPE_BF16) {
        (void) seed;
        const std::vector<ggml_bf16_t> weights(static_cast<size_t>(row_length * row_count),
                                               ggml_fp32_to_bf16(0.00390625f));
        std::vector<uint8_t>           bytes(weights.size() * sizeof(ggml_bf16_t));
        std::memcpy(bytes.data(), weights.data(), bytes.size());
        return bytes;
    }

    return make_quantized_rows(type, row_length, row_count, seed);
}

static void set_tensor_bytes(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t byte_count) {
    REQUIRE(tensor != nullptr);
    REQUIRE(ggml_nbytes(tensor) == byte_count);
    ggml_backend_tensor_set(tensor, data, 0, byte_count);
    ggml_backend_synchronize(backend);
}

static void set_tensor_pair_bytes(ggml_backend_t cpu_backend,
                                  ggml_tensor *  cpu_tensor,
                                  ggml_backend_t hrx_backend,
                                  ggml_tensor *  hrx_tensor,
                                  const void *   data,
                                  size_t         byte_count) {
    set_tensor_bytes(cpu_backend, cpu_tensor, data, byte_count);
    set_tensor_bytes(hrx_backend, hrx_tensor, data, byte_count);
}

static std::vector<float> get_f32_tensor(ggml_backend_t backend, ggml_tensor * tensor) {
    REQUIRE(tensor != nullptr);
    const size_t       element_count = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<float> data(element_count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(element_count);
        ggml_backend_tensor_get(tensor, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < element_count; ++i) {
            data[i] = ggml_fp16_to_fp32(f16[i]);
        }
    } else {
        REQUIRE(false);
    }
    ggml_backend_synchronize(backend);
    return data;
}

static void require_close(const std::vector<float> & actual,
                          const std::vector<float> & expected,
                          float                      abs_tolerance,
                          float                      rel_tolerance = 0.0f) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff    = std::fabs(actual[i] - expected[i]);
        const float allowed = abs_tolerance + rel_tolerance * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || diff > allowed) {
            std::fprintf(stderr, "value mismatch at %zu: actual=%g expected=%g diff=%g allowed=%g\n", i, actual[i],
                         expected[i], diff, allowed);
            std::abort();
        }
    }
}

static ggml_backend_t init_cpu_backend() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(backend != nullptr);
    return backend;
}

struct AttentionPostprocessGraph {
    ggml_tensor * input               = nullptr;
    ggml_tensor * query_weight        = nullptr;
    ggml_tensor * key_weight          = nullptr;
    ggml_tensor * value_weight        = nullptr;
    ggml_tensor * query_norm_weight   = nullptr;
    ggml_tensor * key_norm_weight     = nullptr;
    ggml_tensor * positions           = nullptr;
    ggml_tensor * inverse_frequencies = nullptr;
    ggml_tensor * key_cache           = nullptr;
    ggml_tensor * value_cache         = nullptr;
    ggml_tensor * key_cache_indices   = nullptr;
    ggml_tensor * value_cache_indices = nullptr;
    ggml_tensor * attention_mask      = nullptr;
    ggml_tensor * query_reshape       = nullptr;
    ggml_tensor * query_output        = nullptr;
    ggml_tensor * key_output          = nullptr;
    ggml_tensor * value_output        = nullptr;
};

static AttentionPostprocessGraph build_attention_postprocess_graph(ggml_context * ctx,
                                                                   int64_t        token_count,
                                                                   int64_t        query_head_count,
                                                                   int64_t        key_value_head_count,
                                                                   int64_t        cache_row_count) {
    AttentionPostprocessGraph graph;
    const int64_t             query_size     = query_head_count * kQwenFlashHeadSize;
    const int64_t             key_value_size = key_value_head_count * kQwenFlashHeadSize;

    graph.input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    graph.query_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, query_size);
    graph.key_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, key_value_size);
    graph.value_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, key_value_size);
    REQUIRE(graph.input != nullptr);
    REQUIRE(graph.query_weight != nullptr);
    REQUIRE(graph.key_weight != nullptr);
    REQUIRE(graph.value_weight != nullptr);

    ggml_tensor * query_raw = ggml_mul_mat(ctx, graph.query_weight, graph.input);
    ggml_tensor * key_raw   = ggml_mul_mat(ctx, graph.key_weight, graph.input);
    ggml_tensor * value_raw = ggml_mul_mat(ctx, graph.value_weight, graph.input);
    REQUIRE(query_raw != nullptr);
    REQUIRE(key_raw != nullptr);
    REQUIRE(value_raw != nullptr);

    ggml_tensor * query_reshape = ggml_reshape_3d(ctx, query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * key_reshape   = ggml_reshape_3d(ctx, key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    ggml_tensor * value_reshape =
        ggml_reshape_3d(ctx, value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(query_reshape != nullptr);
    REQUIRE(key_reshape != nullptr);
    REQUIRE(value_reshape != nullptr);
    graph.query_reshape = query_reshape;

    graph.query_norm_weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    graph.key_norm_weight     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    graph.positions           = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    graph.inverse_frequencies = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize / 2);
    REQUIRE(graph.query_norm_weight != nullptr);
    REQUIRE(graph.key_norm_weight != nullptr);
    REQUIRE(graph.positions != nullptr);
    REQUIRE(graph.inverse_frequencies != nullptr);

    ggml_tensor * query_norm = ggml_rms_norm(ctx, query_reshape, kQwenRmsNormEps);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, graph.query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    graph.query_output = ggml_rope_ext(ctx, query_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                       GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(graph.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, key_reshape, kQwenRmsNormEps);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, graph.key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = ggml_rope_ext(ctx, key_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                           GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows   = ggml_reshape_2d(ctx, key_rope, key_value_size, token_count);
    ggml_tensor * value_cache_rows = ggml_reshape_2d(ctx, value_reshape, key_value_size, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);

    graph.key_cache           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    graph.value_cache         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    graph.key_cache_indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    graph.value_cache_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    REQUIRE(graph.key_cache != nullptr);
    REQUIRE(graph.value_cache != nullptr);
    REQUIRE(graph.key_cache_indices != nullptr);
    REQUIRE(graph.value_cache_indices != nullptr);

    graph.key_output   = ggml_set_rows(ctx, graph.key_cache, key_cache_rows, graph.key_cache_indices);
    graph.value_output = ggml_set_rows(ctx, graph.value_cache, value_cache_rows, graph.value_cache_indices);
    REQUIRE(graph.key_output != nullptr);
    REQUIRE(graph.value_output != nullptr);
    return graph;
}

static ggml_tensor * append_qwen_full_cache_flash_attention_consumer(ggml_context *              ctx,
                                                                     AttentionPostprocessGraph & graph,
                                                                     int64_t                     token_count,
                                                                     int64_t                     query_head_count,
                                                                     int64_t                     key_value_head_count,
                                                                     int64_t                     cache_row_count) {
    ggml_tensor * query_layout =
        ggml_reshape_3d(ctx, graph.query_output, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * query_permute = ggml_permute(ctx, query_layout, 0, 2, 1, 3);
    ggml_tensor * key_cache_layout =
        ggml_reshape_3d(ctx, graph.key_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * key_permute = ggml_permute(ctx, key_cache_layout, 0, 2, 1, 3);
    ggml_tensor * value_cache_layout =
        ggml_reshape_3d(ctx, graph.value_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * value_permute = ggml_permute(ctx, value_cache_layout, 0, 2, 1, 3);
    graph.attention_mask        = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cache_row_count, token_count);
    REQUIRE(query_layout != nullptr);
    REQUIRE(query_permute != nullptr);
    REQUIRE(key_cache_layout != nullptr);
    REQUIRE(key_permute != nullptr);
    REQUIRE(value_cache_layout != nullptr);
    REQUIRE(value_permute != nullptr);
    REQUIRE(graph.attention_mask != nullptr);
    return build_qwen_flash_attention_graph(ctx, query_permute, key_permute, value_permute, graph.attention_mask);
}

struct QwenFlashAttentionLayoutGraph {
    ggml_tensor * query  = nullptr;
    ggml_tensor * key    = nullptr;
    ggml_tensor * value  = nullptr;
    ggml_tensor * mask   = nullptr;
    ggml_tensor * output = nullptr;
};

static QwenFlashAttentionLayoutGraph build_qwen_flash_attention_layout_graph(
    ggml_context * ctx,
    int64_t        query_token_count,
    int64_t        key_value_token_count,
    int64_t        qk_head_size    = kQwenFlashHeadSize,
    int64_t        value_head_size = kQwenFlashHeadSize,
    float          scale           = 0.0f) {
    QwenFlashAttentionLayoutGraph graph;
    constexpr int64_t             query_head_count     = 32;
    constexpr int64_t             key_value_head_count = 4;

    ggml_tensor * query_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, qk_head_size, query_head_count, query_token_count);
    ggml_tensor * key_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, qk_head_size, key_value_head_count, key_value_token_count);
    ggml_tensor * value_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, value_head_size, key_value_head_count, key_value_token_count);
    REQUIRE(query_storage != nullptr);
    REQUIRE(key_storage != nullptr);
    REQUIRE(value_storage != nullptr);

    graph.query = ggml_permute(ctx, query_storage, 0, 2, 1, 3);
    graph.key   = ggml_permute(ctx, key_storage, 0, 2, 1, 3);
    graph.value = ggml_permute(ctx, value_storage, 0, 2, 1, 3);
    graph.mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
    REQUIRE(graph.query != nullptr);
    REQUIRE(graph.key != nullptr);
    REQUIRE(graph.value != nullptr);
    REQUIRE(graph.mask != nullptr);

    graph.output =
        build_qwen_flash_attention_graph(ctx, graph.query, graph.key, graph.value, graph.mask, qk_head_size, scale);
    return graph;
}

static AttentionPostprocessGraph build_decode_attention_qkv_graph(ggml_context * ctx) {
    constexpr int64_t         token_count          = 1;
    constexpr int64_t         query_head_count     = 32;
    constexpr int64_t         key_value_head_count = 4;
    constexpr int64_t         cache_row_count      = 1024;
    AttentionPostprocessGraph graph =
        build_attention_postprocess_graph(ctx, token_count, query_head_count, key_value_head_count, cache_row_count);

    ggml_tensor * hidden_state     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor * attention_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(hidden_state != nullptr);
    REQUIRE(attention_weight != nullptr);
    ggml_tensor * attention_rms = ggml_rms_norm(ctx, hidden_state, kQwenRmsNormEps);
    REQUIRE(attention_rms != nullptr);
    graph.input = ggml_mul(ctx, attention_rms, attention_weight);
    REQUIRE(graph.input != nullptr);

    const int64_t key_value_size = key_value_head_count * kQwenFlashHeadSize;
    ggml_tensor * query_raw      = ggml_mul_mat(ctx, graph.query_weight, graph.input);
    ggml_tensor * key_raw        = ggml_mul_mat(ctx, graph.key_weight, graph.input);
    ggml_tensor * value_raw      = ggml_mul_mat(ctx, graph.value_weight, graph.input);
    REQUIRE(query_raw != nullptr);
    REQUIRE(key_raw != nullptr);
    REQUIRE(value_raw != nullptr);

    ggml_tensor * query_reshape = ggml_reshape_3d(ctx, query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * key_reshape   = ggml_reshape_3d(ctx, key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    ggml_tensor * value_reshape =
        ggml_reshape_3d(ctx, value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(query_reshape != nullptr);
    REQUIRE(key_reshape != nullptr);
    REQUIRE(value_reshape != nullptr);

    ggml_tensor * query_norm = ggml_rms_norm(ctx, query_reshape, kQwenRmsNormEps);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, graph.query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    graph.query_output = ggml_rope_ext(ctx, query_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                       GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(graph.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, key_reshape, kQwenRmsNormEps);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, graph.key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = ggml_rope_ext(ctx, key_mul, graph.positions, graph.inverse_frequencies, kQwenFlashHeadSize,
                                           GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows   = ggml_reshape_2d(ctx, key_rope, key_value_size, token_count);
    ggml_tensor * value_cache_rows = ggml_reshape_2d(ctx, value_reshape, key_value_size, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);
    graph.key_output   = ggml_set_rows(ctx, graph.key_cache, key_cache_rows, graph.key_cache_indices);
    graph.value_output = ggml_set_rows(ctx, graph.value_cache, value_cache_rows, graph.value_cache_indices);
    REQUIRE(graph.key_output != nullptr);
    REQUIRE(graph.value_output != nullptr);
    return graph;
}

struct RoutedMoeGraph {
    ggml_tensor * logits       = nullptr;
    ggml_tensor * input        = nullptr;
    ggml_tensor * gate_weight  = nullptr;
    ggml_tensor * up_weight    = nullptr;
    ggml_tensor * down_weight  = nullptr;
    ggml_tensor * hidden_state = nullptr;
    ggml_tensor * norm_weight  = nullptr;
    ggml_tensor * output       = nullptr;
};

static RoutedMoeGraph build_routed_moe_graph(ggml_context * ctx,
                                             ggml_type      down_weight_type,
                                             bool           include_next_rmsnorm) {
    RoutedMoeGraph    graph;
    constexpr int64_t token_count = 1;
    graph.logits                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(graph.logits != nullptr);
    ggml_tensor * route_ids     = nullptr;
    ggml_tensor * route_weights = build_qwen_router_top8_graph(ctx, graph.logits, &route_ids);
    REQUIRE(route_weights != nullptr);
    REQUIRE(route_ids != nullptr);

    graph.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenHiddenSize, 1, token_count);
    graph.gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    graph.up_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    graph.down_weight =
        ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize, kQwenRouterExpertCount);
    graph.hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    REQUIRE(graph.input != nullptr);
    REQUIRE(graph.gate_weight != nullptr);
    REQUIRE(graph.up_weight != nullptr);
    REQUIRE(graph.down_weight != nullptr);
    REQUIRE(graph.hidden_state != nullptr);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, graph.gate_weight, graph.input, route_ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, graph.up_weight, graph.input, route_ids);
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    ggml_tensor * glu = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
    REQUIRE(glu != nullptr);
    ggml_tensor * down = ggml_mul_mat_id(ctx, graph.down_weight, glu, route_ids);
    REQUIRE(down != nullptr);
    ggml_tensor * weighted = ggml_mul(ctx, down, route_weights);
    REQUIRE(weighted != nullptr);

    std::vector<ggml_tensor *> route_views;
    route_views.reserve(kQwenRouterRouteCount);
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, kQwenHiddenSize, token_count, weighted->nb[2],
                                          static_cast<size_t>(route) * weighted->nb[1]);
        REQUIRE(view != nullptr);
        route_views.push_back(view);
    }

    ggml_tensor * reduced = route_views.front();
    for (size_t i = 1; i < route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, route_views[i]);
        REQUIRE(reduced != nullptr);
    }

    ggml_tensor * residual = ggml_add(ctx, graph.hidden_state, reduced);
    REQUIRE(residual != nullptr);
    graph.output = residual;

    if (include_next_rmsnorm) {
        graph.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
        REQUIRE(graph.norm_weight != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
        REQUIRE(rms != nullptr);
        graph.output = ggml_mul(ctx, rms, graph.norm_weight);
        REQUIRE(graph.output != nullptr);
    }

    return graph;
}

static void run_rmsnorm_support_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    const ggml::hrx::GraphSupportResult support = executor.can_execute(*graph);
    REQUIRE(support.supported);
    REQUIRE(support.status.success());

    ggml_tensor * wrong_eps_output = build_rmsnorm_mul_graph(ctx, input, weight, 1.0e-5f);
    ggml_cgraph * wrong_eps_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_eps_graph != nullptr);
    ggml_build_forward_expand(wrong_eps_graph, wrong_eps_output);
    const ggml::hrx::GraphSupportResult wrong_eps_support = executor.can_execute(*wrong_eps_graph);
    REQUIRE(wrong_eps_support.supported);
    REQUIRE(wrong_eps_support.status.success());
    require_kernel_subsequence(scheduled_kernel_sequence(wrong_eps_graph), { "loom_libs:ggml_rmsnorm_binary_f32" });

    ggml_tensor * standalone_output = ggml_rms_norm(ctx, input, 1.0e-5f);
    REQUIRE(standalone_output != nullptr);
    ggml_cgraph * standalone_graph = ggml_new_graph(ctx);
    REQUIRE(standalone_graph != nullptr);
    ggml_build_forward_expand(standalone_graph, standalone_output);
    const ggml::hrx::GraphSupportResult standalone_support = executor.can_execute(*standalone_graph);
    REQUIRE(standalone_support.supported);
    REQUIRE(standalone_support.status.success());
    require_kernel_subsequence(scheduled_kernel_sequence(standalone_graph), { "loom_libs:ggml_rmsnorm_f32" });

    ggml_tensor * q8_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, 1);
    ggml_tensor * q8_rhs    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor * q8_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount);
    REQUIRE(q8_input != nullptr);
    REQUIRE(q8_rhs != nullptr);
    REQUIRE(q8_weight != nullptr);
    ggml_tensor * q8_rms    = ggml_rms_norm(ctx, q8_input, 1.0e-5f);
    ggml_tensor * q8_binary = ggml_add(ctx, q8_rms, q8_rhs);
    ggml_tensor * q8_output = ggml_mul_mat(ctx, q8_weight, q8_binary);
    REQUIRE(q8_output != nullptr);
    ggml_cgraph * q8_graph = ggml_new_graph(ctx);
    REQUIRE(q8_graph != nullptr);
    ggml_build_forward_expand(q8_graph, q8_output);
    require_kernel_subsequence(scheduled_kernel_sequence(q8_graph),
                               { "loom_libs:ggml_rmsnorm_binary_f32", "loom_libs:ggml_mul_mat_vector_q6_f32_f32" });

    ggml_tensor * wrong_type_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
    ggml_tensor * wrong_type_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    REQUIRE(wrong_type_input != nullptr);
    REQUIRE(wrong_type_weight != nullptr);
    ggml_tensor * wrong_type_output = build_rmsnorm_mul_graph(ctx, wrong_type_input, wrong_type_weight);
    ggml_cgraph * wrong_type_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_type_graph != nullptr);
    ggml_build_forward_expand(wrong_type_graph, wrong_type_output);
    const ggml::hrx::GraphSupportResult wrong_type_support = executor.can_execute(*wrong_type_graph);
    REQUIRE(!wrong_type_support.supported);

    ggml_tensor * wrong_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    REQUIRE(wrong_weight != nullptr);
    ggml_tensor * wrong_weight_output = build_rmsnorm_mul_graph(ctx, input, wrong_weight);
    ggml_cgraph * wrong_weight_graph  = ggml_new_graph(ctx);
    REQUIRE(wrong_weight_graph != nullptr);
    ggml_build_forward_expand(wrong_weight_graph, wrong_weight_output);
    const ggml::hrx::GraphSupportResult wrong_weight_support = executor.can_execute(*wrong_weight_graph);
    REQUIRE(!wrong_weight_support.supported);

    ggml_free(ctx);
}

static void run_rmsnorm_mul_case(int64_t hidden_size, int64_t token_count) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(hidden_size * token_count * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * output = build_rmsnorm_mul_graph(ctx, input, weight);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { expected_plain_rmsnorm_binary_kernel(hidden_size, token_count) });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_input(hidden_size, token_count);
    const std::vector<float> weight_data = make_weight(hidden_size);
    const std::vector<float> expected    = rmsnorm_mul_reference(input_data, weight_data, hidden_size, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-4f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_rmsnorm_binary_add_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    constexpr int64_t hidden_size = 256;
    constexpr int64_t token_count = 4;
    constexpr float   epsilon     = 1.0e-5f;

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * cpu_rhs   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, hidden_size);
    ggml_tensor * hrx_input = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * hrx_rhs   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, hidden_size);
    REQUIRE(cpu_input != nullptr);
    REQUIRE(cpu_rhs != nullptr);
    REQUIRE(hrx_input != nullptr);
    REQUIRE(hrx_rhs != nullptr);

    ggml_tensor * cpu_rms    = ggml_rms_norm(cpu_ctx, cpu_input, epsilon);
    ggml_tensor * hrx_rms    = ggml_rms_norm(hrx_ctx, hrx_input, epsilon);
    ggml_tensor * cpu_output = ggml_add(cpu_ctx, cpu_rms, cpu_rhs);
    ggml_tensor * hrx_output = ggml_add(hrx_ctx, hrx_rms, hrx_rhs);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input = make_pattern_f32(hidden_size * token_count, 17, 0.02f);
    const std::vector<float> rhs   = make_pattern_f32(hidden_size, 18, 0.03f);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_rhs, hrx_backend, hrx_rhs, rhs.data(), rhs.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gemma_post_proj_cpu_reference_case(int64_t token_count) {
    constexpr int64_t hidden_size = 640;
    constexpr float   epsilon     = 1.0e-6f;
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    struct GraphTensors {
        ggml_tensor * input;
        ggml_tensor * weight;
        ggml_tensor * residual;
        ggml_tensor * raw;
        ggml_tensor * raw_side;
        ggml_tensor * output;
    };

    auto build = [&](ggml_context * ctx) {
        GraphTensors g = {};
        g.input         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
        g.weight        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        g.residual      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
        g.raw           = ggml_add(ctx, g.input, g.residual);
        ggml_tensor * normalized = ggml_rms_norm(ctx, g.raw, epsilon);
        ggml_tensor * scaled     = ggml_mul(ctx, normalized, g.weight);
        g.output        = ggml_add(ctx, scaled, g.residual);
        g.raw_side      = ggml_mul(ctx, g.raw, g.residual);
        REQUIRE(g.output != nullptr);
        REQUIRE(g.raw_side != nullptr);
        return g;
    };
    GraphTensors cpu = build(cpu_ctx);
    GraphTensors hrx = build(hrx_ctx);
    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu.output);
    ggml_build_forward_expand(cpu_graph, cpu.raw_side);
    ggml_build_forward_expand(hrx_graph, hrx.output);
    ggml_build_forward_expand(hrx_graph, hrx.raw_side);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_mul_add_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);
    const std::vector<float> input    = make_pattern_f32(hidden_size * token_count, 37, 0.02f);
    const std::vector<float> residual = make_pattern_f32(hidden_size * token_count, 43, 0.04f);
    const std::vector<float> weight   = make_weight(hidden_size);
    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.residual, hrx_backend, hrx.residual, residual.data(),
                          residual.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.weight, hrx_backend, hrx.weight, weight.data(),
                          weight.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx.output), get_f32_tensor(cpu_backend, cpu.output), 5.0e-4f);
    require_close(get_f32_tensor(hrx_backend, hrx.raw_side), get_f32_tensor(cpu_backend, cpu.raw_side), 1.0e-5f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gemma_post_proj_matcher_negatives() {
    constexpr int64_t hidden_size = 640;
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);

    for (const int variant : { 0, 1, 2, 3, 4 }) {
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 3);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 3);
        ggml_tensor * raw = ggml_add(ctx, input, rhs);
        ggml_tensor * residual = variant == 2 ? ggml_view_2d(ctx, raw, hidden_size, 3, raw->nb[1], 0) :
                                 variant == 3 ? ggml_mul(ctx, raw, rhs) :
                                                rhs;
        ggml_tensor * rms = ggml_rms_norm(ctx, raw, 1.0e-6f);
        ggml_tensor * scaled = variant == 4 ? ggml_add(ctx, rms, weight) : ggml_mul(ctx, rms, weight);
        ggml_tensor * output = ggml_add(ctx, scaled, residual);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        if (variant == 1) {
            ggml_build_forward_expand(graph, ggml_mul(ctx, scaled, rhs));
        }
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        const size_t rms_index = producer_index_for_tensor(imported.graph, rms);
        const size_t raw_index = producer_index_for_tensor(imported.graph, raw);
        ggml::hrx::CommandPlan plan;
        std::vector<bool> covered(imported.graph.nodes().size(), false);
        auto kernel = [&]() {
            ggml::hrx::DispatchMatch match;
            const ggml::hrx::DispatchMatchContext context = {
                imported.graph, &imported.graph.nodes()[rms_index],    rms_index, covered,
                plan,           next_plan_value(imported.graph, plan),
            };
            REQUIRE(registry->match(context, match));
            REQUIRE(match.dispatches.size() == 1);
            return kernel_name_for_id(match.dispatches[0].kernel.kernel_id);
        };
        REQUIRE(kernel() == "loom_libs:ggml_rmsnorm_binary_f32");
        covered[raw_index] = true;
        if (variant == 3) {
            REQUIRE(kernel() == "loom_libs:ggml_rmsnorm_binary_f32");
            covered[producer_index_for_tensor(imported.graph, residual)] = true;
        }
        if (variant == 2) {
            const size_t view_index = producer_index_for_tensor(imported.graph, residual);
            covered[view_index] = true;
            const ggml::hrx::Value * raw_value = imported.graph.values().find_tensor(raw);
            const ggml::hrx::Value * residual_value = imported.graph.values().find_tensor(residual);
            REQUIRE(raw_value != nullptr);
            REQUIRE(residual_value != nullptr);
            REQUIRE(raw_value->storage_root == residual_value->storage_root);
        }
        REQUIRE(kernel() == (variant == 0 || variant == 3 ? "loom_libs:ggml_rmsnorm_mul_add_f32" :
                                                        "loom_libs:ggml_rmsnorm_binary_f32"));
        ggml_free(ctx);
    }
}

static void run_rmsnorm_gate_cpu_reference_case(int64_t hidden_size,
                                               int64_t head_count,
                                               int64_t token_count,
                                               ggml_unary_op gate_op) {
    ggml_backend_t cpu = init_cpu_backend();
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    REQUIRE(hrx != nullptr);
    const int64_t elements = hidden_size * head_count * token_count;
    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(elements * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * cpu_ctx = ggml_init(params);
    ggml_context * hrx_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr && hrx_ctx != nullptr);

    struct Graph {
        ggml_tensor * input;
        ggml_tensor * weight;
        ggml_tensor * gate;
        ggml_tensor * output;
        ggml_cgraph * graph;
    };

    auto build = [&](ggml_context * ctx) {
        Graph g;
        g.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, head_count, token_count);
        g.weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        g.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size * head_count, token_count);
        ggml_tensor * normalized = build_rmsnorm_mul_graph(ctx, g.input, g.weight);
        ggml_tensor * reshaped_gate = ggml_reshape_3d(ctx, g.gate, hidden_size, head_count, token_count);
        g.output = ggml_mul(ctx, normalized, ggml_unary(ctx, reshaped_gate, gate_op));
        g.graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(g.graph, g.output);
        return g;
    };
    Graph a = build(cpu_ctx);
    Graph b = build(hrx_ctx);
    require_kernel_subsequence(scheduled_kernel_sequence(b.graph), { "loom_libs:ggml_rmsnorm_gate_f32_publish" });
    ggml_backend_buffer_t a_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
    ggml_backend_buffer_t b_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx);
    REQUIRE(a_buffer != nullptr && b_buffer != nullptr);
    const auto input = make_pattern_f32(elements, 23, 0.05f);
    const auto weight = make_pattern_f32(hidden_size, 11, 0.075f);
    const auto gate = make_pattern_f32(elements, 29, 0.125f);
    set_tensor_pair_bytes(cpu, a.input, hrx, b.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu, a.weight, hrx, b.weight, weight.data(), weight.size() * sizeof(float));
    set_tensor_pair_bytes(cpu, a.gate, hrx, b.gate, gate.data(), gate.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(cpu, a.graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx, b.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);
    ggml_backend_synchronize(hrx);
    const auto actual = get_f32_tensor(hrx, b.output);
    const auto cpu_output = get_f32_tensor(cpu, a.output);
    if (gate_op == GGML_UNARY_OP_GELU) {
        // The CPU GELU route uses an F16 lookup table. Keep the tighter HRX
        // check against the defining function instead of inheriting its rounding.
        auto expected = rmsnorm_mul_reference(input, weight, hidden_size, head_count * token_count);
        for (size_t i = 0; i < expected.size(); ++i) {
            const double x = gate[i];
            const double gelu = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
            expected[i] *= static_cast<float>(gelu);
        }
        require_close(actual, expected, 2.0e-5f);
        require_close(cpu_output, expected, 2.0e-5f, 5.0e-4f);
    } else {
        require_close(actual, cpu_output, 2.0e-5f);
    }

    ggml_backend_buffer_free(a_buffer);
    ggml_backend_buffer_free(b_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
}

static void run_rmsnorm_cpu_reference_case(int64_t hidden_size = 256,
                                           int64_t token_count = 4,
                                           float   epsilon     = 1.0e-5f) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(hidden_size * token_count * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * hrx_input = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
    REQUIRE(cpu_input != nullptr);
    REQUIRE(hrx_input != nullptr);

    ggml_tensor * cpu_output = ggml_rms_norm(cpu_ctx, cpu_input, epsilon);
    ggml_tensor * hrx_output = ggml_rms_norm(hrx_ctx, hrx_input, epsilon);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_f32" });
    REQUIRE(scheduled_kernel_specialization(hrx_graph, "loom_libs:ggml_rmsnorm_f32").workload_specialization ==
            ggml::hrx::WorkloadSpecialization::Dynamic);

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input    = make_pattern_f32(hidden_size * token_count, 19, 0.02f);
    const std::vector<float> expected = rmsnorm_reference(input, hidden_size, token_count, epsilon);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(cpu_backend, cpu_output), expected, 5.0e-4f);
    require_close(get_f32_tensor(hrx_backend, hrx_output), expected, 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_strided_rmsnorm_cpu_reference_case(int64_t token_count) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    constexpr int64_t hidden_size  = 256;
    constexpr int64_t input_stride = 512;
    constexpr float   epsilon      = 1.0e-5f;
    constexpr size_t  input_offset = 4 * sizeof(float);
    const size_t      storage_elements =
        input_offset / sizeof(float) + static_cast<size_t>((token_count - 1) * input_stride + hidden_size);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_storage = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * cpu_input =
        ggml_view_2d(cpu_ctx, cpu_storage, hidden_size, token_count, input_stride * sizeof(float), input_offset);
    ggml_tensor * cpu_output  = ggml_rms_norm(cpu_ctx, cpu_input, epsilon);
    ggml_tensor * hrx_storage = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * hrx_input =
        ggml_view_2d(hrx_ctx, hrx_storage, hidden_size, token_count, input_stride * sizeof(float), input_offset);
    ggml_tensor * hrx_output = ggml_rms_norm(hrx_ctx, hrx_input, epsilon);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> rows = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 53, 0.015f);
    std::vector<float>       storage(storage_elements, -19.0f);
    for (int64_t token = 0; token < token_count; ++token) {
        std::memcpy(storage.data() + input_offset / sizeof(float) + static_cast<size_t>(token * input_stride),
                    rows.data() + static_cast<size_t>(token * hidden_size),
                    static_cast<size_t>(hidden_size) * sizeof(float));
    }
    const std::vector<float> expected = rmsnorm_reference(rows, hidden_size, token_count, epsilon);
    set_tensor_pair_bytes(cpu_backend, cpu_storage, hrx_backend, hrx_storage, storage.data(),
                          storage.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(cpu_backend, cpu_output), expected, 5.0e-4f);
    require_close(get_f32_tensor(hrx_backend, hrx_output), expected, 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_strided_rmsnorm_binary_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    constexpr int64_t hidden_size  = 256;
    constexpr int64_t token_count  = 64;
    constexpr int64_t input_stride = 288;
    constexpr float   epsilon      = 1.0e-5f;
    const size_t      storage_elements = static_cast<size_t>((token_count - 1) * input_stride + hidden_size);

    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_storage = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * cpu_input =
        ggml_view_2d(cpu_ctx, cpu_storage, hidden_size, token_count, input_stride * sizeof(float), 0);
    ggml_tensor * cpu_scale   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, hidden_size);
    ggml_tensor * cpu_output  = ggml_mul(cpu_ctx, ggml_rms_norm(cpu_ctx, cpu_input, epsilon), cpu_scale);
    ggml_tensor * hrx_storage = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * hrx_input =
        ggml_view_2d(hrx_ctx, hrx_storage, hidden_size, token_count, input_stride * sizeof(float), 0);
    ggml_tensor * hrx_scale  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, hidden_size);
    ggml_tensor * hrx_output = ggml_mul(hrx_ctx, ggml_rms_norm(hrx_ctx, hrx_input, epsilon), hrx_scale);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_binary_strided_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> rows  = make_pattern_f32(hidden_size * token_count, 54, 0.015f);
    const std::vector<float> scale = make_pattern_f32(hidden_size, 55, 0.03f);
    std::vector<float>       storage(storage_elements, -19.0f);
    for (int64_t token = 0; token < token_count; ++token) {
        std::memcpy(storage.data() + static_cast<size_t>(token * input_stride),
                    rows.data() + static_cast<size_t>(token * hidden_size),
                    static_cast<size_t>(hidden_size) * sizeof(float));
    }
    set_tensor_pair_bytes(cpu_backend, cpu_storage, hrx_backend, hrx_storage, storage.data(),
                          storage.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_scale, hrx_backend, hrx_scale, scale.data(), scale.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_strided_cont_cpu_reference_case(int64_t token_count) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    constexpr int64_t width        = 64;
    constexpr int64_t row_count    = 40;
    constexpr int64_t token_stride = 4096;
    constexpr size_t  view_offset  = 8 * sizeof(float);
    const size_t      storage_elements =
        view_offset / sizeof(float) + static_cast<size_t>((token_count - 1) * token_stride + width * row_count);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_storage = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * cpu_view    = ggml_view_3d(cpu_ctx, cpu_storage, width, row_count, token_count, width * sizeof(float),
                                             token_stride * sizeof(float), view_offset);
    ggml_tensor * cpu_output  = ggml_cont(cpu_ctx, cpu_view);
    ggml_tensor * hrx_storage = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * hrx_view    = ggml_view_3d(hrx_ctx, hrx_storage, width, row_count, token_count, width * sizeof(float),
                                             token_stride * sizeof(float), view_offset);
    ggml_tensor * hrx_output  = ggml_cont(hrx_ctx, hrx_view);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_copy_strided_source_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> rows = make_pattern_f32(static_cast<size_t>(width * row_count * token_count), 59, 0.01f);
    std::vector<float>       storage(storage_elements, -37.0f);
    for (int64_t token = 0; token < token_count; ++token) {
        std::memcpy(storage.data() + view_offset / sizeof(float) + static_cast<size_t>(token * token_stride),
                    rows.data() + static_cast<size_t>(token * width * row_count),
                    static_cast<size_t>(width * row_count) * sizeof(float));
    }
    set_tensor_pair_bytes(cpu_backend, cpu_storage, hrx_backend, hrx_storage, storage.data(),
                          storage.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(cpu_backend, cpu_output), rows, 0.0f);
    require_close(get_f32_tensor(hrx_backend, hrx_output), rows, 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_rmsnorm_mul_cpu_reference_case(int64_t hidden_size,
                                               int64_t token_count,
                                               float   epsilon,
                                               bool    adversarial = false) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(hidden_size * token_count * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * cpu_weight = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, hidden_size);
    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor * hrx_weight = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, hidden_size);
    REQUIRE(cpu_input != nullptr);
    REQUIRE(cpu_weight != nullptr);
    REQUIRE(hrx_input != nullptr);
    REQUIRE(hrx_weight != nullptr);

    ggml_tensor * cpu_output = build_rmsnorm_mul_graph(cpu_ctx, cpu_input, cpu_weight, epsilon);
    ggml_tensor * hrx_output = build_rmsnorm_mul_graph(hrx_ctx, hrx_input, hrx_weight, epsilon);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph),
                               { expected_plain_rmsnorm_binary_kernel(hidden_size, token_count) });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    std::vector<float> input = make_pattern_f32(hidden_size * token_count, 23, 0.02f);
    if (adversarial) {
        for (size_t i = 0; i < input.size(); ++i) {
            switch (i % 8) {
                case 0: input[i] = 0.0f; break;
                case 1: input[i] = 1.0e-6f; break;
                case 2: input[i] = -1.0e-6f; break;
                case 3: input[i] = 8.0f; break;
                case 4: input[i] = -8.0f; break;
                case 5: input[i] = 0.125f; break;
                case 6: input[i] = -0.25f; break;
                default: input[i] = 1.5f; break;
            }
        }
    }
    const std::vector<float> weight = make_weight(hidden_size);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(),
                          weight.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gemma_scaled_rmsnorm_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * cpu_ctx = ggml_init(params);
    ggml_context * hrx_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    ggml_tensor * hrx_input = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    REQUIRE(cpu_input != nullptr);
    REQUIRE(hrx_input != nullptr);

    ggml_tensor * cpu_output = ggml_rms_norm(cpu_ctx, cpu_input, kGemmaRmsNormEps);
    ggml_tensor * hrx_output = ggml_rms_norm(hrx_ctx, hrx_input, kGemmaRmsNormEps);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input = make_gemma_scaled_embedding_input(kGemmaHiddenSize, kGemmaPromptTokenCount);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gemma_scaled_rmsnorm_mul_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * cpu_ctx = ggml_init(params);
    ggml_context * hrx_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    ggml_tensor * cpu_weight = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, kGemmaHiddenSize);
    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    ggml_tensor * hrx_weight = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, kGemmaHiddenSize);
    REQUIRE(cpu_input != nullptr);
    REQUIRE(cpu_weight != nullptr);
    REQUIRE(hrx_input != nullptr);
    REQUIRE(hrx_weight != nullptr);

    ggml_tensor * cpu_output = build_rmsnorm_mul_graph(cpu_ctx, cpu_input, cpu_weight, kGemmaRmsNormEps);
    ggml_tensor * hrx_output = build_rmsnorm_mul_graph(hrx_ctx, hrx_input, hrx_weight, kGemmaRmsNormEps);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rmsnorm_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input  = make_gemma_scaled_embedding_input(kGemmaHiddenSize, kGemmaPromptTokenCount);
    const std::vector<float> weight = make_weight(kGemmaHiddenSize);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(),
                          weight.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_scheduled_hrx_scale_rmsnorm_case() {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    ggml_backend_t cpu_backend = init_cpu_backend();
    REQUIRE(hrx_backend != nullptr);

    ggml_backend_t       backends[] = { hrx_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    REQUIRE(sched != nullptr);

    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    REQUIRE(input != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, input, hrx_backend);

    ggml_tensor * scaled = ggml_scale(ctx, input, std::sqrt(static_cast<float>(kGemmaHiddenSize)));
    ggml_tensor * output = ggml_rms_norm(ctx, scaled, kGemmaRmsNormEps);
    REQUIRE(scaled != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    const ggml::hrx::KernelSpecialization scale_specialization =
        scheduled_kernel_specialization(graph, "loom_libs:ggml_scale_f32");
    REQUIRE(scale_specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);

    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, scaled) == hrx_backend);
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, output) == hrx_backend);

    const std::vector<float> input_data =
        make_pattern_f32(static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount), 31, 0.0005f);
    std::vector<float> scaled_input    = input_data;
    const float        embedding_scale = std::sqrt(static_cast<float>(kGemmaHiddenSize));
    for (float & value : scaled_input) {
        value *= embedding_scale;
    }
    const std::vector<float> expected =
        rmsnorm_reference(scaled_input, kGemmaHiddenSize, kGemmaPromptTokenCount, kGemmaRmsNormEps);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    require_close(actual, expected, 5.0e-4f);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(hrx_backend);
    ggml_backend_free(cpu_backend);
}

static void run_scheduled_hrx_rmsnorm_scale_case() {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    ggml_backend_t cpu_backend = init_cpu_backend();
    REQUIRE(hrx_backend != nullptr);

    ggml_backend_t       backends[] = { hrx_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    REQUIRE(sched != nullptr);

    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    REQUIRE(input != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, input, hrx_backend);

    ggml_tensor * normalized = ggml_rms_norm(ctx, input, kGemmaRmsNormEps);
    ggml_tensor * output     = ggml_scale(ctx, normalized, 0.5f);
    REQUIRE(normalized != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, normalized) == hrx_backend);
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, output) == hrx_backend);

    const std::vector<float> input_data =
        make_pattern_f32(static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount), 37, 0.0005f);
    std::vector<float> expected =
        rmsnorm_reference(input_data, kGemmaHiddenSize, kGemmaPromptTokenCount, kGemmaRmsNormEps);
    for (float & value : expected) {
        value *= 0.5f;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    require_close(actual, expected, 5.0e-4f);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(hrx_backend);
    ggml_backend_free(cpu_backend);
}

static void run_scheduled_hrx_rmsnorm_view_scale_case() {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    ggml_backend_t cpu_backend = init_cpu_backend();
    REQUIRE(hrx_backend != nullptr);

    ggml_backend_t       backends[] = { hrx_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    REQUIRE(sched != nullptr);

    ggml_init_params params = {};
    params.mem_size = static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount * sizeof(float) * 8 + 1024 * 1024);
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kGemmaHiddenSize, kGemmaPromptTokenCount);
    REQUIRE(input != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, input, hrx_backend);

    ggml_tensor * normalized = ggml_rms_norm(ctx, input, kGemmaRmsNormEps);
    ggml_tensor * view       = ggml_reshape_2d(ctx, normalized, kGemmaHiddenSize, kGemmaPromptTokenCount);
    ggml_tensor * output     = ggml_scale(ctx, view, 0.25f);
    REQUIRE(normalized != nullptr);
    REQUIRE(view != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    REQUIRE(ggml_backend_sched_alloc_graph(sched, graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, normalized) == hrx_backend);
    REQUIRE(ggml_backend_sched_get_tensor_backend(sched, output) == hrx_backend);

    const std::vector<float> input_data =
        make_pattern_f32(static_cast<size_t>(kGemmaHiddenSize * kGemmaPromptTokenCount), 41, 0.0005f);
    std::vector<float> expected =
        rmsnorm_reference(input_data, kGemmaHiddenSize, kGemmaPromptTokenCount, kGemmaRmsNormEps);
    for (float & value : expected) {
        value *= 0.25f;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    require_close(actual, expected, 5.0e-4f);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(hrx_backend);
    ggml_backend_free(cpu_backend);
}

static void run_external_view_input_rmsnorm_case() {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    static constexpr int64_t kHeadSize  = 256;
    static constexpr int64_t kHeadCount = 16;
    const int64_t            row_count  = kHeadCount * kGemmaPromptTokenCount;

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(kHeadSize * row_count * sizeof(float) * 4 + 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * root = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadSize, kHeadCount, kGemmaPromptTokenCount, 1);
    REQUIRE(root != nullptr);
    ggml_tensor * input_view = ggml_view_4d(ctx, root, kHeadSize, kHeadCount, kGemmaPromptTokenCount, 1, root->nb[1],
                                            root->nb[2], root->nb[3], 0);
    REQUIRE(input_view != nullptr);
    ggml_tensor * output = ggml_rms_norm(ctx, input_view, kGemmaRmsNormEps);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_graph_add_node(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "loom_libs:ggml_rmsnorm_f32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, hrx_backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input    = make_pattern_f32(static_cast<size_t>(kHeadSize * row_count), 29, 0.0025f);
    const std::vector<float> expected = rmsnorm_reference(input, kHeadSize, row_count, kGemmaRmsNormEps);

    ggml_backend_tensor_set(root, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(hrx_backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, output), expected, 5.0e-4f);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(hrx_backend);
}

static void run_split_local_view_alias_import_case() {
    static constexpr int64_t kHeadSize       = 256;
    static constexpr int64_t kKeyValueHeads  = 8;
    static constexpr int64_t kKeyValueHidden = kHeadSize * kKeyValueHeads;
    static constexpr int64_t kRootHeads      = kKeyValueHeads + 1;
    const size_t             split_offset    = static_cast<size_t>(kHeadSize * sizeof(float));

    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size =
        static_cast<size_t>(kHeadSize * kRootHeads * kGemmaPromptTokenCount * sizeof(float) * 4 + 1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * root = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadSize, kRootHeads, kGemmaPromptTokenCount, 1);
    REQUIRE(root != nullptr);
    ggml_tensor * split_input = ggml_view_4d(ctx, root, kHeadSize, kKeyValueHeads, kGemmaPromptTokenCount, 1,
                                             root->nb[1], root->nb[2], root->nb[3], split_offset);
    REQUIRE(split_input != nullptr);
    ggml_tensor * flattened =
        ggml_view_2d(ctx, split_input, kKeyValueHidden, kGemmaPromptTokenCount, split_input->nb[2], 0);
    REQUIRE(flattened != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_graph_add_node(graph, flattened);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, imported.graph.nodes().front()));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    REQUIRE(scheduler.plan().dispatches.empty());

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, hrx_backend);
    REQUIRE(buffer != nullptr);

    const size_t             root_element_count = static_cast<size_t>(kHeadSize * kRootHeads * kGemmaPromptTokenCount);
    const std::vector<float> root_data          = make_pattern_f32(root_element_count, 37, 0.001f);
    ggml_backend_tensor_set(root, root_data.data(), 0, root_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(hrx_backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(hrx_backend);

    std::vector<float> actual(static_cast<size_t>(kKeyValueHidden * kGemmaPromptTokenCount));
    ggml_backend_tensor_get(flattened, actual.data(), 0, actual.size() * sizeof(float));
    const float * expected = root_data.data() + split_offset / sizeof(float);
    require_close(actual, std::vector<float>(expected, expected + actual.size()), 0.0f);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(hrx_backend);
}

static void run_router_projection_case(int64_t token_count) {
    static constexpr int64_t kHiddenSize  = 2048;
    static constexpr int64_t kExpertCount = 128;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(
        (kHiddenSize * token_count + kHiddenSize * kExpertCount + kExpertCount * token_count) * sizeof(float) * 4 +
        1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, kExpertCount);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHiddenSize, token_count);
    REQUIRE(weight != nullptr);
    REQUIRE(input != nullptr);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> input_data  = make_router_input(kHiddenSize, token_count);
    const std::vector<float> weight_data = make_router_weight(kHiddenSize, kExpertCount);
    const std::vector<float> expected =
        router_projection_reference(input_data, weight_data, kHiddenSize, kExpertCount, token_count);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 1.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_router_top8_case(int64_t token_count) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size    = static_cast<size_t>(kQwenRouterExpertCount * token_count * sizeof(float) * 16 + 1024 * 1024);
    params.no_alloc    = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(logits != nullptr);
    ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen3_moe_router_top8_f32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float> logits_data = make_router_logits(token_count);
    const std::vector<float> expected    = router_top8_weights_reference(logits_data, token_count);

    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 1.0e-5f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_qwen_flash_attention_case() {
    static constexpr int64_t kQueryTokenCount    = 2;
    static constexpr int64_t kKeyValueTokenCount = 4;

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, kQueryTokenCount, 1);
    ggml_tensor * key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, kKeyValueTokenCount, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kKeyValueTokenCount, kQueryTokenCount);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    REQUIRE(mask != nullptr);
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, query, key, value, mask);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph),
                               { "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8" });
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }));
    const ggml::hrx::Value * output_value = imported.graph.values().find_tensor(output);
    REQUIRE(output_value != nullptr);
    REQUIRE(scheduler.plan().metadata.find_alternate_value(
                output_value->id, GGML_TYPE_Q8_1,
                static_cast<size_t>(kQueryTokenCount) * ggml_row_size(GGML_TYPE_Q8_1, kQwenFlashHeadSize)) == nullptr);
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
    ggml_tensor * common_query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, 16, 1);
    ggml_tensor * common_key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, 16, 1);
    ggml_tensor * common_value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, 16, 1);
    ggml_tensor * common_mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 16);
    REQUIRE(common_query != nullptr);
    REQUIRE(common_key != nullptr);
    REQUIRE(common_value != nullptr);
    REQUIRE(common_mask != nullptr);
    ggml_tensor * common_output =
        build_qwen_flash_attention_graph(ctx, common_query, common_key, common_value, common_mask);
    ggml_cgraph * common_graph = ggml_new_graph(ctx);
    REQUIRE(common_graph != nullptr);
    ggml_build_forward_expand(common_graph, common_output);
    require_kernel_subsequence(scheduled_kernel_sequence(common_graph),
                               { "loom_libs:ggml_flash_attention_f32_f16_wmma" });
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float>       query_data = make_flash_query(kQueryTokenCount);
    const std::vector<ggml_fp16_t> key_data   = make_flash_key_value(kKeyValueTokenCount, 3);
    const std::vector<ggml_fp16_t> value_data = make_flash_key_value(kKeyValueTokenCount, 11);
    const std::vector<ggml_fp16_t> mask_data  = make_flash_mask(kQueryTokenCount, kKeyValueTokenCount);
    const std::vector<float>       expected =
        flash_attention_reference(query_data, key_data, value_data, mask_data, kQueryTokenCount, kKeyValueTokenCount);

    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size() * sizeof(float));
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_common_flash_attention_cpu_reference_case(int64_t      head_size,
                                                          int64_t      query_token_count,
                                                          int64_t      key_value_token_count,
                                                          float        scale,
                                                          const char * expected_kernel,
                                                          int64_t      value_head_size = -1) {
    if (value_head_size < 0) {
        value_head_size = head_size;
    }
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, query_token_count, 1);
    ggml_tensor * key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_size, key_value_token_count, 1);
    ggml_tensor * value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, value_head_size, key_value_token_count, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    REQUIRE(mask != nullptr);
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, query, key, value, mask, head_size, scale);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { expected_kernel });
    const ggml::hrx::WorkloadSpecialization expected_specialization =
        query_token_count == 256 && key_value_token_count == 256 ? ggml::hrx::WorkloadSpecialization::Exact :
                                                                   ggml::hrx::WorkloadSpecialization::Dynamic;
    REQUIRE(scheduled_kernel_specialization(graph, expected_kernel).workload_specialization ==
            expected_specialization);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float>       query_data = make_flash_query(query_token_count, head_size);
    const std::vector<ggml_fp16_t> key_data   = make_flash_key_value(key_value_token_count, 3, head_size);
    const std::vector<ggml_fp16_t> value_data = make_flash_key_value(key_value_token_count, 11, value_head_size);
    const std::vector<ggml_fp16_t> mask_data  = make_flash_mask(query_token_count, key_value_token_count);
    const std::vector<float>       expected =
        flash_attention_reference(query_data, key_data, value_data, mask_data, query_token_count, key_value_token_count,
                                  head_size, value_head_size, scale);

    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size() * sizeof(float));
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
}

static void run_asymmetric_flash_attention_cpu_reference_case() {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);

    constexpr int64_t query_token_count     = 23;
    constexpr int64_t key_value_token_count = 256;
    constexpr int64_t qk_head_size          = 96;
    constexpr int64_t value_head_size       = 64;
    const float       scale                 = 1.0f / std::sqrt(static_cast<float>(qk_head_size));

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, qk_head_size, query_token_count, 1);
    ggml_tensor * key   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, qk_head_size, key_value_token_count, 1);
    ggml_tensor * value = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, value_head_size, key_value_token_count, 1);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    REQUIRE(mask != nullptr);
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, query, key, value, mask, qk_head_size, scale);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "loom_libs:ggml_flash_attention_f32_f16_wmma" });
    REQUIRE(scheduled_kernel_specialization(graph, "loom_libs:ggml_flash_attention_f32_f16_wmma")
                .workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<float>       query_data = make_flash_query(query_token_count, qk_head_size);
    const std::vector<ggml_fp16_t> key_data   = make_flash_key_value(key_value_token_count, 3, qk_head_size);
    const std::vector<ggml_fp16_t> value_data = make_flash_key_value(key_value_token_count, 11, value_head_size);
    const std::vector<ggml_fp16_t> mask_data  = make_flash_mask(query_token_count, key_value_token_count);
    const std::vector<float>       expected =
        flash_attention_reference(query_data, key_data, value_data, mask_data, query_token_count, key_value_token_count,
                                  qk_head_size, value_head_size, scale);

    ggml_backend_tensor_set(query, query_data.data(), 0, query_data.size() * sizeof(float));
    ggml_backend_tensor_set(key, key_data.data(), 0, key_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(value, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        REQUIRE(diff <= 5.0e-2f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
}

static void run_qwen_full_cache_prefill_flash_attention_scheduling_case(int64_t token_count) {
    constexpr int64_t kQueryHeadCount    = 32;
    constexpr int64_t kKeyValueHeadCount = 4;
    constexpr int64_t kFullCacheRowCount = 40960;
    const size_t      active_mask_byte_count =
        static_cast<size_t>(token_count) * static_cast<size_t>(token_count) * sizeof(ggml_fp16_t);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    AttentionPostprocessGraph attention =
        build_attention_postprocess_graph(ctx, token_count, kQueryHeadCount, kKeyValueHeadCount, kFullCacheRowCount);
    ggml_tensor * flash_output = append_qwen_full_cache_flash_attention_consumer(
        ctx, attention, token_count, kQueryHeadCount, kKeyValueHeadCount, kFullCacheRowCount);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, attention.key_output);
    ggml_build_forward_expand(graph, attention.value_output);
    ggml_build_forward_expand(graph, flash_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);

    std::vector<bool>        covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan   plan;
    ggml::hrx::DispatchMatch postprocess_match;
    match_dispatch_at_index(imported.graph, *registry, plan, covered_nodes,
                            producer_index_for_tensor(imported.graph, attention.query_reshape), postprocess_match);
    ggml::hrx::DispatchMatch flash_match;
    match_dispatch_at_index(imported.graph, *registry, plan, covered_nodes,
                            producer_index_for_tensor(imported.graph, flash_output), flash_match);

    const ggml::hrx::Value * mask_value = imported.graph.values().find_tensor(attention.attention_mask);
    REQUIRE(mask_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * compact_mask =
        ggml::hrx::find_alternate_value(plan, mask_value->id, GGML_TYPE_F16, active_mask_byte_count);
    REQUIRE(compact_mask != nullptr);

    const ggml::hrx::Dispatch * metadata_dispatch = nullptr;
    for (const ggml::hrx::Dispatch & dispatch : plan.initialization_dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen_attention_metadata") {
            metadata_dispatch = &dispatch;
        }
    }
    REQUIRE(metadata_dispatch != nullptr);
    REQUIRE(metadata_dispatch->kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(metadata_dispatch->kernel.integer_parameters.at("context_capacity") == token_count);
    REQUIRE(metadata_dispatch->bindings.size() == 5);
    REQUIRE(metadata_dispatch->bindings[4].value == compact_mask->alternate_value);
    REQUIRE(metadata_dispatch->bindings[4].length == active_mask_byte_count);

    const ggml::hrx::Dispatch * flash_dispatch = nullptr;
    for (const ggml::hrx::Dispatch & dispatch : plan.dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_flash_attention_f32_f16_wmma") {
            flash_dispatch = &dispatch;
        }
    }
    REQUIRE(flash_dispatch != nullptr);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("query_token_count") == token_count);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("key_value_token_count") == token_count);
    REQUIRE(flash_dispatch->bindings.size() == 6);
    REQUIRE(flash_dispatch->kernel.compile_parameters.at("ggml.flash_attention.apply_gate") == "0");
    REQUIRE(flash_dispatch->bindings[3].value == compact_mask->alternate_value);
    REQUIRE(flash_dispatch->bindings[3].length == active_mask_byte_count);

    ggml_free(ctx);
}

static void run_qwen_decode_split_flash_attention_scheduling_case(int64_t query_token_count,
                                                                  int64_t key_value_token_count,
                                                                  bool    use_common_dispatch = false,
                                                                  int64_t qk_head_size        = kQwenFlashHeadSize,
                                                                  int64_t value_head_size     = kQwenFlashHeadSize,
                                                                  float   scale               = 0.0f) {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    if (use_common_dispatch) {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
    } else {
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }

    constexpr int64_t query_head_count     = 32;
    constexpr int64_t key_value_head_count = 4;
    const int64_t     query_hidden_size    = query_head_count * qk_head_size;
    const int64_t     output_hidden_size   = query_head_count * value_head_size;
    const size_t      q8_row_bytes         = ggml_row_size(GGML_TYPE_Q8_1, output_hidden_size);
    const size_t      q8_output_bytes      = static_cast<size_t>(query_token_count) * q8_row_bytes;
    int64_t key_value_capacity = 64;
    while (key_value_capacity < key_value_token_count) {
        key_value_capacity *= 2;
    }
    const int64_t     key_value_blocks     = key_value_capacity / 64;
    const size_t      partial_scalar_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16) * sizeof(float);
    const size_t partial_output_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16 * value_head_size) * sizeof(ggml_fp16_t);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    QwenFlashAttentionLayoutGraph attention = build_qwen_flash_attention_layout_graph(
        ctx, query_token_count, key_value_token_count, qk_head_size, value_head_size, scale);
    ggml_tensor * attention_output =
        ggml_reshape_2d(ctx, attention.output, output_hidden_size, query_token_count);
    ggml_tensor * consumer_weight =
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, output_hidden_size, kQwenHiddenSize);
    ggml_tensor * consumer_output = ggml_mul_mat(ctx, consumer_weight, attention_output);
    REQUIRE(attention_output != nullptr);
    REQUIRE(consumer_weight != nullptr);
    REQUIRE(consumer_output != nullptr);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, consumer_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == static_cast<size_t>(query_token_count + 1));
    REQUIRE(plan.transients.size() == 4);
    REQUIRE(plan.completion_counter_requests.size() == 1);

    REQUIRE(plan.transients[0].size == partial_scalar_bytes);
    REQUIRE(plan.transients[1].size == partial_scalar_bytes);
    REQUIRE(plan.transients[2].size == partial_output_bytes);
    REQUIRE(plan.transients[3].size == q8_output_bytes);
    REQUIRE(plan.completion_counter_requests[0].count == key_value_head_count);

    const ggml::hrx::Value * query_value       = imported.graph.values().find_tensor(attention.query);
    const ggml::hrx::Value * mask_value        = imported.graph.values().find_tensor(attention.mask);
    const ggml::hrx::Value * output_value      = imported.graph.values().find_tensor(attention.output);
    const ggml::hrx::Value * publication_value = imported.graph.values().find_tensor(attention_output);
    REQUIRE(query_value != nullptr);
    REQUIRE(mask_value != nullptr);
    REQUIRE(output_value != nullptr);
    REQUIRE(publication_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * q8_alternate =
        plan.metadata.find_alternate_value(publication_value->id, GGML_TYPE_Q8_1, q8_output_bytes);
    REQUIRE(q8_alternate != nullptr);
    REQUIRE(q8_alternate->alternate_value == plan.transients[3].value);

    const size_t query_row_bytes  = static_cast<size_t>(query_hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = static_cast<size_t>(output_hidden_size) * sizeof(float);
    for (int64_t row = 0; row < query_token_count; ++row) {
        const ggml::hrx::Dispatch & dispatch = plan.dispatches[static_cast<size_t>(row)];
        const char * expected_kernel         = "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8";
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel);
        REQUIRE(dispatch.kernel.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
        REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.query_head_count") ==
                std::to_string(query_head_count));
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.key_value_head_count") ==
                std::to_string(key_value_head_count));
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.qk_head_size") ==
                std::to_string(qk_head_size));
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.value_head_size") ==
                std::to_string(value_head_size));
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.attention_scale") ==
                expected_config_value(scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(qk_head_size)) : scale));
        REQUIRE(dispatch.kernel.compile_parameters.at("ggml.flash_attention.decode.key_value_token_capacity") ==
                std::to_string(key_value_capacity));
        REQUIRE(dispatch.bindings.size() == 10);
        REQUIRE(dispatch.bindings[0].value == query_value->id);
        REQUIRE(dispatch.bindings[0].offset == static_cast<size_t>(row) * query_value->nb[1]);
        REQUIRE(dispatch.bindings[0].length == query_row_bytes);
        REQUIRE(dispatch.bindings[3].value == mask_value->id);
        REQUIRE(dispatch.bindings[3].offset == static_cast<size_t>(row) * mask_value->nb[1]);
        REQUIRE(dispatch.bindings[3].length == mask_row_bytes);
        REQUIRE(dispatch.bindings[8].value == output_value->id);
        REQUIRE(dispatch.bindings[8].offset == static_cast<size_t>(row) * output_value->nb[2]);
        REQUIRE(dispatch.bindings[8].length == output_row_bytes);
        REQUIRE(dispatch.bindings[9].value == q8_alternate->alternate_value);
        REQUIRE(dispatch.bindings[9].offset == static_cast<size_t>(row) * q8_row_bytes);
        REQUIRE(dispatch.bindings[9].length == q8_row_bytes);
    }

    ggml_free(ctx);
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
}

static void run_qwen_decode_attention_output_next_q8_scheduling_case(bool include_get_rows_selectors) {
    constexpr int64_t query_token_count     = 1;
    constexpr int64_t key_value_token_count = 512;
    constexpr int64_t attention_hidden_size = 4096;
    const size_t      attention_q8_bytes    = ggml_row_size(GGML_TYPE_Q8_1, attention_hidden_size);
    const size_t      next_q8_bytes         = ggml_row_size(GGML_TYPE_Q8_1, kQwenHiddenSize);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    QwenFlashAttentionLayoutGraph attention =
        build_qwen_flash_attention_layout_graph(ctx, query_token_count, key_value_token_count);
    ggml_tensor * attention_output = ggml_reshape_2d(ctx, attention.output, attention_hidden_size, query_token_count);
    ggml_tensor * output_weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, attention_hidden_size, kQwenHiddenSize);
    ggml_tensor * projection       = ggml_mul_mat(ctx, output_weight, attention_output);
    ggml_tensor * residual_input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, query_token_count);
    ggml_tensor * selected_projection = projection;
    ggml_tensor * selected_residual   = residual_input;
    ggml_tensor * row_indices         = nullptr;
    if (include_get_rows_selectors) {
        row_indices         = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        selected_projection = ggml_get_rows(ctx, projection, row_indices);
        selected_residual   = ggml_get_rows(ctx, residual_input, row_indices);
        REQUIRE(row_indices != nullptr);
        REQUIRE(selected_projection != nullptr);
        REQUIRE(selected_residual != nullptr);
    }
    ggml_tensor * residual    = ggml_add(ctx, selected_projection, selected_residual);
    ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor * rms         = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
    ggml_tensor * normalized  = ggml_mul(ctx, rms, norm_weight);
    ggml_tensor * next_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenHiddenSize);
    ggml_tensor * next_output = ggml_mul_mat(ctx, next_weight, normalized);
    REQUIRE(attention_output != nullptr);
    REQUIRE(output_weight != nullptr);
    REQUIRE(projection != nullptr);
    REQUIRE(residual_input != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(norm_weight != nullptr);
    REQUIRE(rms != nullptr);
    REQUIRE(normalized != nullptr);
    REQUIRE(next_weight != nullptr);
    REQUIRE(next_output != nullptr);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, next_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 3);
    REQUIRE(plan.completion_counter_requests.size() == 2);

    const auto projection_it = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
        return kernel_name_for_id(dispatch.kernel.kernel_id) ==
               "qwen3_moe:qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8";
    });
    REQUIRE(projection_it != plan.dispatches.end());
    const ggml::hrx::Dispatch & projection_dispatch = *projection_it;
    REQUIRE(kernel_name_for_id(projection_dispatch.kernel.kernel_id) ==
            "qwen3_moe:qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8");
    REQUIRE(projection_dispatch.kernel.integer_parameters.at("token_count") == query_token_count);
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") ==
            std::to_string(attention_hidden_size));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") ==
            std::to_string(kQwenHiddenSize));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "1");
    REQUIRE(projection_dispatch.bindings.size() == 7);

    const ggml::hrx::Value * flash_output_value   = imported.graph.values().find_tensor(attention_output);
    const ggml::hrx::Value * residual_input_value = imported.graph.values().find_tensor(residual_input);
    const ggml::hrx::Value * residual_value       = imported.graph.values().find_tensor(residual);
    const ggml::hrx::Value * normalized_value     = imported.graph.values().find_tensor(normalized);
    REQUIRE(flash_output_value != nullptr);
    REQUIRE(residual_input_value != nullptr);
    REQUIRE(residual_value != nullptr);
    REQUIRE(normalized_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * attention_q8 =
        ggml::hrx::find_alternate_value(imported.graph, plan, flash_output_value->id, GGML_TYPE_Q8_1,
                                        attention_q8_bytes);
    const ggml::hrx::CommandPlanAlternateValue * next_q8 =
        ggml::hrx::find_alternate_value(imported.graph, plan, normalized_value->id, GGML_TYPE_Q8_1, next_q8_bytes);
    REQUIRE(attention_q8 != nullptr);
    REQUIRE(next_q8 != nullptr);
    REQUIRE(projection_dispatch.bindings[0].value == attention_q8->alternate_value);
    REQUIRE(projection_dispatch.bindings[2].value == residual_value->id);
    REQUIRE(projection_dispatch.bindings[4].value == normalized_value->id);
    REQUIRE(projection_dispatch.bindings[6].value == next_q8->alternate_value);

    const ggml::hrx::Value * aliased_residual = imported.graph.values().find(residual_value->id);
    REQUIRE(aliased_residual != nullptr);
    REQUIRE(aliased_residual->alias_source == residual_input_value->id);

    ggml_free(ctx);
}

static void run_add_f32_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params cpu_params = {};
    cpu_params.mem_size         = 256 * 1024;
    cpu_params.no_alloc         = true;
    ggml_init_params hrx_params = cpu_params;
    ggml_context *   cpu_ctx    = ggml_init(cpu_params);
    ggml_context *   hrx_ctx    = ggml_init(hrx_params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t element_count = 257;
    ggml_tensor *     cpu_a         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     cpu_b         = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     cpu_output    = ggml_add(cpu_ctx, cpu_a, cpu_b);
    ggml_tensor *     hrx_a         = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     hrx_b         = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     hrx_output    = ggml_add(hrx_ctx, hrx_a, hrx_b);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> a = make_pattern_f32(element_count, 1, 0.125f);
    const std::vector<float> b = make_pattern_f32(element_count, 2, 0.25f);
    set_tensor_pair_bytes(cpu_backend, cpu_a, hrx_backend, hrx_a, a.data(), a.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_b, hrx_backend, hrx_b, b.data(), b.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dynamic_generic_ops_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t element_count = 257;
    ggml_tensor * cpu_input = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * hrx_input = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * cpu_unary = ggml_sqr(cpu_ctx, cpu_input);
    ggml_tensor * hrx_unary = ggml_sqr(hrx_ctx, hrx_input);
    ggml_tensor * cpu_copy_target = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * hrx_copy_target = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, element_count);
    ggml_tensor * cpu_copy = ggml_cpy(cpu_ctx, cpu_input, cpu_copy_target);
    ggml_tensor * hrx_copy = ggml_cpy(hrx_ctx, hrx_input, hrx_copy_target);
    ggml_tensor * cpu_concat_lhs = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, 64, 7);
    ggml_tensor * cpu_concat_rhs = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, 32, 7);
    ggml_tensor * hrx_concat_lhs = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, 64, 7);
    ggml_tensor * hrx_concat_rhs = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, 32, 7);
    ggml_tensor * cpu_concat = ggml_concat(cpu_ctx, cpu_concat_lhs, cpu_concat_rhs, 0);
    ggml_tensor * hrx_concat = ggml_concat(hrx_ctx, hrx_concat_lhs, hrx_concat_rhs, 0);
    REQUIRE(cpu_unary != nullptr && hrx_unary != nullptr);
    REQUIRE(cpu_copy != nullptr && hrx_copy != nullptr);
    REQUIRE(cpu_concat != nullptr && hrx_concat != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr && hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_unary);
    ggml_build_forward_expand(cpu_graph, cpu_copy);
    ggml_build_forward_expand(cpu_graph, cpu_concat);
    ggml_build_forward_expand(hrx_graph, hrx_unary);
    ggml_build_forward_expand(hrx_graph, hrx_copy);
    ggml_build_forward_expand(hrx_graph, hrx_concat);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph),
                               { "loom_libs:ggml_unary_f32", "loom_libs:ggml_copy_f32",
                                 "loom_libs:ggml_concat_dim0_f32" });
    for (const char * kernel_name :
         { "loom_libs:ggml_unary_f32", "loom_libs:ggml_copy_f32", "loom_libs:ggml_concat_dim0_f32" }) {
        const ggml::hrx::KernelSpecialization specialization =
            scheduled_kernel_specialization(hrx_graph, kernel_name);
        REQUIRE(specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
    }

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr && hrx_buffer != nullptr);
    const std::vector<float> input = make_pattern_f32(element_count, 73, 0.03125f);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    const std::vector<float> concat_lhs = make_pattern_f32(64 * 7, 79, 0.015625f);
    const std::vector<float> concat_rhs = make_pattern_f32(32 * 7, 83, 0.0234375f);
    set_tensor_pair_bytes(cpu_backend, cpu_concat_lhs, hrx_backend, hrx_concat_lhs, concat_lhs.data(),
                          concat_lhs.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_concat_rhs, hrx_backend, hrx_concat_rhs, concat_rhs.data(),
                          concat_rhs.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_unary), get_f32_tensor(cpu_backend, cpu_unary), 0.0f);
    require_close(get_f32_tensor(hrx_backend, hrx_copy), get_f32_tensor(cpu_backend, cpu_copy), 0.0f);
    require_close(get_f32_tensor(hrx_backend, hrx_concat), get_f32_tensor(cpu_backend, cpu_concat), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_view_mul_f32_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t hidden_size = 2048;
    constexpr int64_t token_count = 2;
    ggml_tensor *     cpu_source  = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, hidden_size, 3, token_count);
    ggml_tensor *     hrx_source  = ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, hidden_size, 3, token_count);
    REQUIRE(cpu_source != nullptr);
    REQUIRE(hrx_source != nullptr);
    ggml_tensor * cpu_lhs = ggml_view_2d(cpu_ctx, cpu_source, hidden_size, token_count, cpu_source->nb[2], 0);
    ggml_tensor * cpu_rhs =
        ggml_view_2d(cpu_ctx, cpu_source, hidden_size, token_count, cpu_source->nb[2], 2 * cpu_source->nb[1]);
    ggml_tensor * hrx_lhs = ggml_view_2d(hrx_ctx, hrx_source, hidden_size, token_count, hrx_source->nb[2], 0);
    ggml_tensor * hrx_rhs =
        ggml_view_2d(hrx_ctx, hrx_source, hidden_size, token_count, hrx_source->nb[2], 2 * hrx_source->nb[1]);
    REQUIRE(cpu_lhs != nullptr);
    REQUIRE(cpu_rhs != nullptr);
    REQUIRE(hrx_lhs != nullptr);
    REQUIRE(hrx_rhs != nullptr);
    ggml_tensor * cpu_output = ggml_mul(cpu_ctx, cpu_lhs, cpu_rhs);
    ggml_tensor * hrx_output = ggml_mul(hrx_ctx, hrx_lhs, hrx_rhs);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> source = make_pattern_f32(hidden_size * 3 * token_count, 19, 0.015625f);
    set_tensor_pair_bytes(cpu_backend, cpu_source, hrx_backend, hrx_source, source.data(),
                          source.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_glu_split_f32_cpu_reference_case(ggml_glu_op glu_op) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t hidden_size = 256;
    constexpr int64_t token_count = 3;
    ggml_tensor *     cpu_gate    = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor *     cpu_up      = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor *     cpu_output  = ggml_glu_split(cpu_ctx, cpu_gate, cpu_up, glu_op);
    ggml_tensor *     hrx_gate    = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor *     hrx_up      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
    ggml_tensor *     hrx_output  = ggml_glu_split(hrx_ctx, hrx_gate, hrx_up, glu_op);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> gate = make_pattern_f32(hidden_size * token_count, 21, 0.02f);
    const std::vector<float> up   = make_pattern_f32(hidden_size * token_count, 22, 0.03f);
    set_tensor_pair_bytes(cpu_backend, cpu_gate, hrx_backend, hrx_gate, gate.data(), gate.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_up, hrx_backend, hrx_up, up.data(), up.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_glu_packed_f32_cpu_reference_case(ggml_glu_op glu_op,
                                                  bool        swapped,
                                                  int64_t     hidden_size = 256,
                                                  int64_t     token_count = 3) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(16 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, 2 * hidden_size, token_count);
    ggml_tensor * cpu_output = ggml_glu(cpu_ctx, cpu_input, glu_op, swapped);
    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, 2 * hidden_size, token_count);
    ggml_tensor * hrx_output = ggml_glu(hrx_ctx, hrx_input, glu_op, swapped);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_binary_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input =
        make_pattern_f32(static_cast<size_t>(2 * hidden_size * token_count), swapped ? 24 : 23, 0.02f);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_gather_add_f32_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t hidden_size        = 256;
    constexpr int64_t source_token_count = 11;
    constexpr int64_t output_token_count = 5;
    ggml_tensor *     cpu_a              = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor *     cpu_b              = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor *     cpu_ids            = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor *     cpu_rows_a         = ggml_get_rows(cpu_ctx, cpu_a, cpu_ids);
    ggml_tensor *     cpu_rows_b         = ggml_get_rows(cpu_ctx, cpu_b, cpu_ids);
    ggml_tensor *     cpu_output         = ggml_add(cpu_ctx, cpu_rows_a, cpu_rows_b);

    ggml_tensor * hrx_a      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * hrx_b      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * hrx_ids    = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * hrx_rows_a = ggml_get_rows(hrx_ctx, hrx_a, hrx_ids);
    ggml_tensor * hrx_rows_b = ggml_get_rows(hrx_ctx, hrx_b, hrx_ids);
    ggml_tensor * hrx_output = ggml_add(hrx_ctx, hrx_rows_a, hrx_rows_b);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "hrx:ggml_gather_add_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   a   = make_pattern_f32(hidden_size * source_token_count, 3, 0.05f);
    const std::vector<float>   b   = make_pattern_f32(hidden_size * source_token_count, 4, 0.075f);
    const std::vector<int32_t> ids = { 9, 3, 7, 1, 5 };
    set_tensor_pair_bytes(cpu_backend, cpu_a, hrx_backend, hrx_a, a.data(), a.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_b, hrx_backend, hrx_b, b.data(), b.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_scale_add_f32_cpu_reference_case(bool strided_residual) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t width          = 257;
    constexpr int64_t token_count    = 3;
    constexpr int64_t residual_width = 320;
    ggml_tensor *     cpu_input      = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, width, token_count);
    ggml_tensor *     hrx_input      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, width, token_count);
    ggml_tensor *     cpu_residual_storage =
        ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, strided_residual ? residual_width : width, token_count);
    ggml_tensor * hrx_residual_storage =
        ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, strided_residual ? residual_width : width, token_count);
    ggml_tensor * cpu_residual = strided_residual ? ggml_view_2d(cpu_ctx, cpu_residual_storage, width, token_count,
                                                                 cpu_residual_storage->nb[1], 0) :
                                                    cpu_residual_storage;
    ggml_tensor * hrx_residual = strided_residual ? ggml_view_2d(hrx_ctx, hrx_residual_storage, width, token_count,
                                                                 hrx_residual_storage->nb[1], 0) :
                                                    hrx_residual_storage;
    ggml_tensor * cpu_scaled   = ggml_scale_bias(cpu_ctx, cpu_input, 0.177800179f, -0.125f);
    ggml_tensor * hrx_scaled   = ggml_scale_bias(hrx_ctx, hrx_input, 0.177800179f, -0.125f);
    ggml_tensor * cpu_output   = ggml_add(cpu_ctx, cpu_scaled, cpu_residual);
    ggml_tensor * hrx_output   = ggml_add(hrx_ctx, hrx_scaled, hrx_residual);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_scale_add_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input = make_pattern_f32(width * token_count, 31, 0.03125f);
    const std::vector<float> residual =
        make_pattern_f32((strided_residual ? residual_width : width) * token_count, 37, 0.046875f);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_residual_storage, hrx_backend, hrx_residual_storage, residual.data(),
                          residual.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 0.0f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_get_rows_f32_cpu_reference_case(ggml_type weight_type,
                                                int64_t   hidden_size      = kQwenHiddenSize,
                                                int64_t   vocabulary_count = 64,
                                                int64_t   token_count      = 7) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, hidden_size, vocabulary_count);
    ggml_tensor * cpu_ids    = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * cpu_output = ggml_get_rows(cpu_ctx, cpu_weight, cpu_ids);
    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, hidden_size, vocabulary_count);
    ggml_tensor * hrx_ids    = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * hrx_output = ggml_get_rows(hrx_ctx, hrx_weight, hrx_ids);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_get_rows_f32" });
    const ggml::hrx::KernelSpecialization specialization =
        scheduled_kernel_specialization(hrx_graph, "loom_libs:ggml_get_rows_f32");
    REQUIRE(specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
    int64_t expected_token_capacity = 1;
    while (expected_token_capacity < token_count) {
        expected_token_capacity *= 2;
    }
    REQUIRE(specialization.compile_parameters.at("ggml.get_rows_f32.token_capacity") ==
            std::to_string(expected_token_capacity));

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_matmul_weight_bytes(weight_type, hidden_size, vocabulary_count, 5);
    std::vector<int32_t>       ids(static_cast<size_t>(token_count));
    for (int64_t i = 0; i < token_count; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 17 + 3) % vocabulary_count);
    }
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 5.0e-4f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_get_rows_q8_1_zero_weight_case() {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t vocabulary_count = 64;
    constexpr int64_t hidden_size      = kQwenHiddenSize;
    constexpr int64_t token_count      = 7;
    ggml_tensor *     weight           = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q8_1, hidden_size, vocabulary_count);
    ggml_tensor *     ids              = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor *     output           = ggml_get_rows(hrx_ctx, weight, ids);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(hrx_ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "loom_libs:ggml_get_rows_f32" });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(buffer != nullptr);

    const std::vector<uint8_t> weight_bytes(ggml_nbytes(weight), 0);
    const std::vector<int32_t> id_values = { 3, 17, 29, 41, 53, 7, 19 };
    set_tensor_bytes(hrx_backend, weight, weight_bytes.data(), weight_bytes.size());
    set_tensor_bytes(hrx_backend, ids, id_values.data(), id_values.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(hrx_backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, output),
                  std::vector<float>(static_cast<size_t>(hidden_size * token_count), 0.0f), 0.0f, 0.0f);

    ggml_backend_buffer_free(buffer);
    ggml_free(hrx_ctx);
    ggml_backend_free(hrx_backend);
}

static void run_get_rows_scale_f32_cpu_reference_case(ggml_type weight_type = GGML_TYPE_F32,
                                                      int64_t   hidden_size = 256,
                                                      int64_t   row_count   = 8,
                                                      int64_t   token_count = 6) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    const float scale = weight_type == GGML_TYPE_Q5_1 ? 25.2982216f : 0.177800179f;
    ggml_tensor * cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, hidden_size, row_count);
    ggml_tensor * cpu_ids    = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * cpu_rows   = ggml_get_rows(cpu_ctx, cpu_weight, cpu_ids);
    ggml_tensor * cpu_output = ggml_scale(cpu_ctx, cpu_rows, scale);
    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, hidden_size, row_count);
    ggml_tensor * hrx_ids    = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * hrx_rows   = ggml_get_rows(hrx_ctx, hrx_weight, hrx_ids);
    ggml_tensor * hrx_output = ggml_scale(hrx_ctx, hrx_rows, scale);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);
    ggml_set_output(cpu_rows);
    ggml_set_output(hrx_rows);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_get_rows_scale_f32" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_matmul_weight_bytes(weight_type, hidden_size, row_count, 9);
    const std::vector<int32_t> ids    = { 0, static_cast<int32_t>(row_count - 1), 2,
                                          2, static_cast<int32_t>(row_count - 2), 1 };
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int32_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    const float tolerance = weight_type == GGML_TYPE_F32 ? 0.0f : 5.0e-4f;
    require_close(get_f32_tensor(hrx_backend, hrx_rows), get_f32_tensor(cpu_backend, cpu_rows), tolerance);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), tolerance);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static auto allocate_test_weight(ggml_backend_t backend, ggml_tensor * weight) {
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, weight), alignment);
    auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_buft_alloc_buffer(buft, size), ggml_backend_buffer_free);
    REQUIRE(buffer != nullptr);
    ggml_tallocr allocator = ggml_tallocr_new(buffer.get());
    REQUIRE(ggml_tallocr_alloc(&allocator, weight) == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

static void run_dense_matmul_cpu_reference_case(ggml_type    weight_type,
                                                const char * expected_kernel,
                                                int64_t      token_count,
                                                int64_t      output_size,
                                                int64_t      input_size = kQwenHiddenSize,
                                                bool         normalize_input = false,
                                                bool         square_output = false,
                                                ggml_type    cache_type = GGML_TYPE_COUNT) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor *     cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, output_size);
    ggml_tensor *     cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, output_size);
    ggml_tensor *     hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_norm_w = normalize_input ? ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, input_size) : nullptr;
    ggml_tensor *     hrx_norm_w = normalize_input ? ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, input_size) : nullptr;
    ggml_tensor *     cpu_rhs    = normalize_input ? build_rmsnorm_mul_graph(cpu_ctx, cpu_input, cpu_norm_w) : cpu_input;
    ggml_tensor *     hrx_rhs    = normalize_input ? build_rmsnorm_mul_graph(hrx_ctx, hrx_input, hrx_norm_w) : hrx_input;
    ggml_tensor *     cpu_output = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_rhs);
    ggml_tensor *     hrx_output = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_rhs);
    if (square_output) {
        cpu_output = ggml_sqr(cpu_ctx, cpu_output);
        hrx_output = ggml_sqr(hrx_ctx, hrx_output);
    }
    ggml_tensor * cpu_indices = nullptr;
    ggml_tensor * hrx_indices = nullptr;
    ggml_tensor * cpu_cache = nullptr;
    ggml_tensor * hrx_cache = nullptr;
    if (cache_type != GGML_TYPE_COUNT) {
        REQUIRE(token_count == 1);
        cpu_indices = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, token_count);
        hrx_indices = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, token_count);
        cpu_cache = ggml_new_tensor_2d(cpu_ctx, cache_type, output_size, 17);
        hrx_cache = ggml_new_tensor_2d(hrx_ctx, cache_type, output_size, 17);
        cpu_output = ggml_set_rows(cpu_ctx, cpu_cache, cpu_output, cpu_indices);
        hrx_output = ggml_set_rows(hrx_ctx, hrx_cache, hrx_output, hrx_indices);
    }
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    auto hrx_weight_buffer = allocate_test_weight(hrx_backend, hrx_weight);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    const auto sequence = scheduled_kernel_sequence(hrx_graph);
    require_kernel_subsequence(sequence, { expected_kernel });
    const bool packed_q4_input = weight_type == GGML_TYPE_Q4_K;
    const bool iq4_xs_weight = weight_type == GGML_TYPE_IQ4_XS;
    const bool prefill_tokens = token_count >= 256;
    const bool packed_iq4_xs_input = iq4_xs_weight && prefill_tokens;
    const bool packed_input = packed_q4_input || packed_iq4_xs_input || square_output;
    const bool normalized_packed_input = normalize_input && packed_input;
    if (normalized_packed_input) {
        require_kernel_subsequence(sequence, { "loom_libs:ggml_rmsnorm_binary_q8_1_x4_publish", expected_kernel });
        REQUIRE(std::find(sequence.begin(), sequence.end(), "qwen3_moe:ggml_quantize_q8_1_x4_f32") == sequence.end());
    } else if (normalize_input) {
        REQUIRE(std::find(sequence.begin(), sequence.end(), "loom_libs:ggml_rmsnorm_binary_q8_1_x4_publish") == sequence.end());
    }

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight         = make_matmul_weight_bytes(weight_type, input_size, output_size, 6);
    const bool                 weight_is_f16  = weight_type == GGML_TYPE_F16;
    const bool                 weight_is_bf16 = weight_type == GGML_TYPE_BF16;
    const bool                 weight_is_f32  = weight_type == GGML_TYPE_F32;
    const bool                 weight_is_dense_float = weight_is_f16 || weight_is_bf16 || weight_is_f32;
    std::vector<float>         input                 = weight_is_dense_float ?
                                                           std::vector<float>(static_cast<size_t>(input_size * token_count), 0.00390625f) :
                                  make_pattern_f32(input_size * token_count, 7, 0.01f);
    if (cache_type != GGML_TYPE_COUNT) {
        for (size_t i = 0; i < input.size(); ++i) {
            const int value = i % 32 == 0 ? 127 : static_cast<int>((i * 29) % 255) - 127;
            input[i] = static_cast<float>(value) * 0.0078125f;
        }
    }
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    if (normalize_input) {
        const std::vector<float> norm_w = make_weight(input_size);
        set_tensor_pair_bytes(cpu_backend, cpu_norm_w, hrx_backend, hrx_norm_w, norm_w.data(),
                              norm_w.size() * sizeof(float));
    }
    if (cache_type != GGML_TYPE_COUNT) {
        const int64_t index = 7;
        set_tensor_pair_bytes(cpu_backend, cpu_indices, hrx_backend, hrx_indices, &index, sizeof(index));
        const std::vector<uint8_t> cache(ggml_nbytes(cpu_cache), 0);
        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(), cache.size());
    }

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    const float abs_tolerance = cache_type != GGML_TYPE_COUNT ? 2.0e-3f : 1.0f;
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), abs_tolerance,
                  cache_type != GGML_TYPE_COUNT ? 1.0e-4f : 3.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_prefill_value_cache_repeated_indices_case() {
    static constexpr const char * disable_env = "GGML_HRX_DISABLE_PREFILL_V_CACHE_FUSION";
    const char *                  old_env     = std::getenv(disable_env);
    const bool                    had_env     = old_env != nullptr;
    const std::string             old_value   = had_env ? old_env : "";
    REQUIRE(unsetenv(disable_env) == 0);

    constexpr int64_t input_size       = 1536;
    constexpr int64_t output_size      = 256;
    constexpr int64_t token_count      = 64;
    constexpr int64_t vocabulary_count = 128;
    constexpr int64_t cache_row_count  = 512;

    ggml_backend_t fused_backend = ggml_backend_hrx_init(0);
    ggml_backend_t split_backend = ggml_backend_hrx_init(0);
    REQUIRE(fused_backend != nullptr);
    REQUIRE(split_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * fused_ctx = ggml_init(params);
    ggml_context * split_ctx = ggml_init(params);
    REQUIRE(fused_ctx != nullptr);
    REQUIRE(split_ctx != nullptr);

    auto build_graph = [](ggml_context * ctx) {
        ggml_tensor * embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, input_size, vocabulary_count);
        ggml_tensor * tokens    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * norm      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_size);
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, input_size, output_size);
        ggml_tensor * indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * cache     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, output_size, cache_row_count);
        ggml_tensor * rows      = ggml_get_rows(ctx, embedding, tokens);
        ggml_tensor * input     = build_rmsnorm_mul_graph(ctx, rows, norm);
        ggml_tensor * projected = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * output    = ggml_set_rows(ctx, cache, projected, indices);
        return std::array<ggml_tensor *, 7>{ embedding, tokens, norm, weight, indices, cache, output };
    };

    const auto fused = build_graph(fused_ctx);
    const auto split = build_graph(split_ctx);
    auto       fused_embedding_buffer  = allocate_test_weight(fused_backend, fused[0]);
    auto       fused_projection_buffer = allocate_test_weight(fused_backend, fused[3]);
    auto       split_embedding_buffer  = allocate_test_weight(split_backend, split[0]);
    auto       split_projection_buffer = allocate_test_weight(split_backend, split[3]);

    ggml_cgraph * fused_graph = ggml_new_graph(fused_ctx);
    ggml_cgraph * split_graph = ggml_new_graph(split_ctx);
    REQUIRE(fused_graph != nullptr);
    REQUIRE(split_graph != nullptr);
    ggml_build_forward_expand(fused_graph, fused[6]);
    ggml_build_forward_expand(split_graph, split[6]);
    require_kernel_subsequence(scheduled_kernel_sequence(fused_graph),
        { "loom_libs:ggml_get_rows_rmsnorm_binary_q8_1_x4_f16",
          "loom_libs:llm_attention_v_matmul_set_rows_tiled_f32_f32" });
    REQUIRE(setenv(disable_env, "1", 1) == 0);
    require_kernel_subsequence(scheduled_kernel_sequence(split_graph),
                               { "loom_libs:ggml_get_rows_rmsnorm_binary_q8_1_x4_f16",
                                 "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned",
                                 "loom_libs:ggml_set_rows" });

    ggml_backend_buffer_t fused_buffer = ggml_backend_alloc_ctx_tensors(fused_ctx, fused_backend);
    ggml_backend_buffer_t split_buffer = ggml_backend_alloc_ctx_tensors(split_ctx, split_backend);
    REQUIRE(fused_buffer != nullptr);
    REQUIRE(split_buffer != nullptr);

    const std::vector<uint8_t> embedding = make_matmul_weight_bytes(GGML_TYPE_Q4_K, input_size, vocabulary_count, 5);
    const std::vector<uint8_t> projection = make_matmul_weight_bytes(GGML_TYPE_Q6_K, input_size, output_size, 11);
    const std::vector<float>   norm       = make_weight(input_size);
    const std::vector<int32_t> tokens(static_cast<size_t>(token_count), 3);
    std::vector<int64_t>       indices(static_cast<size_t>(token_count));
    for (int64_t token = 0; token < token_count; ++token) {
        indices[static_cast<size_t>(token)] = (token * 37 + 11) % 31;
    }
    const std::vector<ggml_fp16_t> cache(static_cast<size_t>(output_size * cache_row_count), ggml_fp32_to_fp16(-2.5f));

    set_tensor_pair_bytes(fused_backend, fused[0], split_backend, split[0], embedding.data(), embedding.size());
    set_tensor_pair_bytes(fused_backend, fused[1], split_backend, split[1], tokens.data(),
                          tokens.size() * sizeof(int32_t));
    set_tensor_pair_bytes(fused_backend, fused[2], split_backend, split[2], norm.data(), norm.size() * sizeof(float));
    set_tensor_pair_bytes(fused_backend, fused[3], split_backend, split[3], projection.data(), projection.size());
    set_tensor_pair_bytes(fused_backend, fused[4], split_backend, split[4], indices.data(),
                          indices.size() * sizeof(int64_t));
    set_tensor_pair_bytes(fused_backend, fused[5], split_backend, split[5], cache.data(),
                          cache.size() * sizeof(ggml_fp16_t));

    REQUIRE(unsetenv(disable_env) == 0);
    REQUIRE(ggml_backend_graph_compute(fused_backend, fused_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(fused_backend);
    REQUIRE(setenv(disable_env, "1", 1) == 0);
    REQUIRE(ggml_backend_graph_compute(split_backend, split_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(split_backend);
    require_close(get_f32_tensor(fused_backend, fused[6]), get_f32_tensor(split_backend, split[6]), 0.0f);

    ggml_backend_buffer_free(fused_buffer);
    ggml_backend_buffer_free(split_buffer);
    ggml_free(fused_ctx);
    ggml_free(split_ctx);
    ggml_backend_free(fused_backend);
    ggml_backend_free(split_backend);
    restore_environment_value(disable_env, had_env, old_value);
}

static void run_dense_matmul_unary_cpu_reference_case(ggml_type    weight_type,
                                                      const char * expected_kernel,
                                                      int64_t      token_count,
                                                      int64_t      output_size,
                                                      int64_t      input_size = kQwenHiddenSize) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor *     cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, output_size);
    ggml_tensor *     cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_matmul = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);
    ggml_tensor *     cpu_output = ggml_sqr(cpu_ctx, cpu_matmul);
    ggml_tensor *     hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, output_size);
    ggml_tensor *     hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     hrx_matmul = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_input);
    ggml_tensor *     hrx_output = ggml_sqr(hrx_ctx, hrx_matmul);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight(ggml_nbytes(cpu_weight), 0);
    const std::vector<float>   input = make_pattern_f32(input_size * token_count, 7, 0.01f);
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0e-1f, 3.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_swiglu_cpu_reference_case(ggml_type    gate_weight_type,
                                                       ggml_type    up_weight_type,
                                                       const char * expected_kernel,
                                                       int64_t      token_count,
                                                       int64_t      output_size,
                                                       ggml_glu_op  glu_op     = GGML_GLU_OP_SWIGLU,
                                                       int64_t      input_size = kQwenHiddenSize) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(48 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_gate_weight = ggml_new_tensor_2d(cpu_ctx, gate_weight_type, input_size, output_size);
    ggml_tensor * cpu_up_weight   = ggml_new_tensor_2d(cpu_ctx, up_weight_type, input_size, output_size);
    ggml_tensor * cpu_input       = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * cpu_gate        = ggml_mul_mat(cpu_ctx, cpu_gate_weight, cpu_input);
    ggml_tensor * cpu_up          = ggml_mul_mat(cpu_ctx, cpu_up_weight, cpu_input);
    ggml_tensor * cpu_output      = ggml_glu_split(cpu_ctx, cpu_gate, cpu_up, glu_op);
    ggml_tensor * hrx_gate_weight = ggml_new_tensor_2d(hrx_ctx, gate_weight_type, input_size, output_size);
    ggml_tensor * hrx_up_weight   = ggml_new_tensor_2d(hrx_ctx, up_weight_type, input_size, output_size);
    ggml_tensor * hrx_input       = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * hrx_gate        = ggml_mul_mat(hrx_ctx, hrx_gate_weight, hrx_input);
    ggml_tensor * hrx_up          = ggml_mul_mat(hrx_ctx, hrx_up_weight, hrx_input);
    ggml_tensor * hrx_output      = ggml_glu_split(hrx_ctx, hrx_gate, hrx_up, glu_op);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    auto hrx_gate_buffer = allocate_test_weight(hrx_backend, hrx_gate_weight);
    auto hrx_up_buffer   = allocate_test_weight(hrx_backend, hrx_up_weight);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> gate_weight = make_matmul_weight_bytes(gate_weight_type, input_size, output_size, 6);
    const std::vector<uint8_t> up_weight   = make_matmul_weight_bytes(up_weight_type, input_size, output_size, 11);
    const std::vector<float>   input(static_cast<size_t>(input_size * token_count), 0.00390625f);
    set_tensor_pair_bytes(cpu_backend, cpu_gate_weight, hrx_backend, hrx_gate_weight, gate_weight.data(),
                          gate_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_up_weight, hrx_backend, hrx_up_weight, up_weight.data(), up_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0e-4f, 1.0e-3f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_packed_glu_cpu_reference_case(ggml_type    weight_type,
                                                           const char * expected_kernel,
                                                           int64_t      token_count,
                                                           int64_t      output_size,
                                                           ggml_glu_op  glu_op     = GGML_GLU_OP_SWIGLU,
                                                           bool         swapped    = false,
                                                           int64_t      input_size = 256) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(64 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, 2 * output_size);
    ggml_tensor * cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * cpu_packed = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);
    ggml_tensor * cpu_output = ggml_glu(cpu_ctx, cpu_packed, glu_op, swapped);
    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, 2 * output_size);
    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * hrx_packed = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_input);
    ggml_tensor * hrx_output = ggml_glu(hrx_ctx, hrx_packed, glu_op, swapped);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_matmul_weight_bytes(weight_type, input_size, 2 * output_size, 15);
    const std::vector<float>   input(static_cast<size_t>(input_size * token_count), 0.00390625f);
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0f, 3.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_binary_cpu_reference_case(ggml_type    lhs_weight_type,
                                                       ggml_type    rhs_weight_type,
                                                       const char * expected_kernel,
                                                       int64_t      token_count,
                                                       int64_t      output_size) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(48 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t input_size     = kQwenHiddenSize;
    ggml_tensor *     cpu_lhs_weight = ggml_new_tensor_2d(cpu_ctx, lhs_weight_type, input_size, output_size);
    ggml_tensor *     cpu_rhs_weight = ggml_new_tensor_2d(cpu_ctx, rhs_weight_type, input_size, output_size);
    ggml_tensor *     cpu_input      = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_lhs        = ggml_mul_mat(cpu_ctx, cpu_lhs_weight, cpu_input);
    ggml_tensor *     cpu_rhs        = ggml_mul_mat(cpu_ctx, cpu_rhs_weight, cpu_input);
    ggml_tensor *     cpu_output     = ggml_sub(cpu_ctx, cpu_lhs, cpu_rhs);
    ggml_tensor *     hrx_lhs_weight = ggml_new_tensor_2d(hrx_ctx, lhs_weight_type, input_size, output_size);
    ggml_tensor *     hrx_rhs_weight = ggml_new_tensor_2d(hrx_ctx, rhs_weight_type, input_size, output_size);
    ggml_tensor *     hrx_input      = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     hrx_lhs        = ggml_mul_mat(hrx_ctx, hrx_lhs_weight, hrx_input);
    ggml_tensor *     hrx_rhs        = ggml_mul_mat(hrx_ctx, hrx_rhs_weight, hrx_input);
    ggml_tensor *     hrx_output     = ggml_sub(hrx_ctx, hrx_lhs, hrx_rhs);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    auto hrx_lhs_buffer = allocate_test_weight(hrx_backend, hrx_lhs_weight);
    auto hrx_rhs_buffer = allocate_test_weight(hrx_backend, hrx_rhs_weight);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> lhs_weight = make_matmul_weight_bytes(lhs_weight_type, input_size, output_size, 6);
    const std::vector<uint8_t> rhs_weight = make_matmul_weight_bytes(rhs_weight_type, input_size, output_size, 11);
    const std::vector<float>   input(static_cast<size_t>(input_size * token_count), 0.00390625f);
    set_tensor_pair_bytes(cpu_backend, cpu_lhs_weight, hrx_backend, hrx_lhs_weight, lhs_weight.data(),
                          lhs_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_rhs_weight, hrx_backend, hrx_rhs_weight, rhs_weight.data(),
                          rhs_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0e-4f, 1.0e-3f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_vector_q8_publish_cpu_reference_case() {
    constexpr int64_t input_size        = 256;
    constexpr int64_t intermediate_size = 256;
    constexpr int64_t output_size       = 128;
    constexpr int64_t token_count       = 1;

    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor * cpu_first_weight  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, intermediate_size);
    ggml_tensor * cpu_second_weight = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q4_K, intermediate_size, output_size);
    ggml_tensor * cpu_input         = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * cpu_hidden        = ggml_mul_mat(cpu_ctx, cpu_first_weight, cpu_input);
    ggml_tensor * cpu_output        = ggml_mul_mat(cpu_ctx, cpu_second_weight, cpu_hidden);
    ggml_tensor * hrx_first_weight  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, intermediate_size);
    ggml_tensor * hrx_second_weight = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q4_K, intermediate_size, output_size);
    ggml_tensor * hrx_input         = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * hrx_hidden        = ggml_mul_mat(hrx_ctx, hrx_first_weight, hrx_input);
    ggml_tensor * hrx_output        = ggml_mul_mat(hrx_ctx, hrx_second_weight, hrx_hidden);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    auto hrx_first_weight_buffer  = allocate_test_weight(hrx_backend, hrx_first_weight);
    auto hrx_second_weight_buffer = allocate_test_weight(hrx_backend, hrx_second_weight);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph),
        { "loom_libs:ggml_mul_mat_vector_f32_f32", "qwen3_moe:ggml_quantize_q8_1_x4_f32",
          "loom_libs:ggml_mul_mat_vector_f32_f32" });

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*hrx_graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());

    const ggml::hrx::Value * hidden_value = imported.graph.values().find_tensor(hrx_hidden);
    REQUIRE(hidden_value != nullptr);
    const size_t q8_byte_count = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, intermediate_size);
    const ggml::hrx::CommandPlanAlternateValue * q8_alternate =
        plan.metadata.find_alternate_value(hidden_value->id, GGML_TYPE_Q8_1, q8_byte_count);
    REQUIRE(q8_alternate != nullptr);

    const ggml::hrx::Dispatch * second_vector = nullptr;
    int                         vector_count  = 0;
    for (const ggml::hrx::Dispatch & dispatch : plan.dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_vector_f32_f32") {
            ++vector_count;
            if (vector_count == 2) {
                second_vector = &dispatch;
                break;
            }
        }
    }
    REQUIRE(second_vector != nullptr);
    REQUIRE(second_vector->bindings.size() == 4);
    REQUIRE(second_vector->bindings[0].value == q8_alternate->alternate_value);
    REQUIRE(second_vector->kernel.compile_parameters.at("ggml.matmul.vector.activation_format") ==
            std::to_string(GGML_TYPE_Q8_1));

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> first_weight = make_matmul_weight_bytes(GGML_TYPE_F32, input_size, intermediate_size, 6);
    const std::vector<uint8_t> second_weight =
        make_matmul_weight_bytes(GGML_TYPE_Q4_K, intermediate_size, output_size, 11);
    const std::vector<float> input = make_pattern_f32(input_size * token_count, 7, 0.01f);
    set_tensor_pair_bytes(cpu_backend, cpu_first_weight, hrx_backend, hrx_first_weight, first_weight.data(),
                          first_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_second_weight, hrx_backend, hrx_second_weight, second_weight.data(),
                          second_weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0f, 5.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_postops_cpu_reference_case(ggml_type    weight_type,
                                                        const char * expected_kernel,
                                                        int64_t      token_count,
                                                        int64_t      output_size,
                                                        bool         include_bias,
                                                        bool         include_residual,
                                                        bool         include_next_rmsnorm,
                                                        int64_t      input_size = kQwenHiddenSize) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(96 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    ggml_tensor *     cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, output_size);
    ggml_tensor *     cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_bias   = include_bias ? ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, output_size) : nullptr;
    ggml_tensor *     cpu_residual =
        include_residual ? ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, output_size, token_count) : nullptr;
    ggml_tensor * cpu_norm_weight =
        include_next_rmsnorm ? ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, output_size) : nullptr;
    ggml_tensor * cpu_projection = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);
    ggml_tensor * cpu_postops    = cpu_projection;
    if (include_bias) {
        cpu_postops = ggml_add(cpu_ctx, cpu_postops, cpu_bias);
    }
    if (include_residual) {
        cpu_postops = ggml_add(cpu_ctx, cpu_postops, cpu_residual);
    }
    ggml_tensor * cpu_output = cpu_postops;
    if (include_next_rmsnorm) {
        ggml_tensor * cpu_rms        = ggml_rms_norm(cpu_ctx, cpu_postops, kQwenRmsNormEps);
        ggml_tensor * cpu_normalized = ggml_mul(cpu_ctx, cpu_rms, cpu_norm_weight);
        cpu_output                   = ggml_add(cpu_ctx, cpu_normalized, cpu_postops);
    }

    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, output_size);
    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * hrx_bias   = include_bias ? ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, output_size) : nullptr;
    ggml_tensor * hrx_residual =
        include_residual ? ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, output_size, token_count) : nullptr;
    ggml_tensor * hrx_norm_weight =
        include_next_rmsnorm ? ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, output_size) : nullptr;
    ggml_tensor * hrx_projection = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_input);
    ggml_tensor * hrx_postops    = hrx_projection;
    if (include_bias) {
        hrx_postops = ggml_add(hrx_ctx, hrx_postops, hrx_bias);
    }
    if (include_residual) {
        hrx_postops = ggml_add(hrx_ctx, hrx_postops, hrx_residual);
    }
    ggml_tensor * hrx_output = hrx_postops;
    if (include_next_rmsnorm) {
        ggml_tensor * hrx_rms        = ggml_rms_norm(hrx_ctx, hrx_postops, kQwenRmsNormEps);
        ggml_tensor * hrx_normalized = ggml_mul(hrx_ctx, hrx_rms, hrx_norm_weight);
        hrx_output                   = ggml_add(hrx_ctx, hrx_normalized, hrx_postops);
    }
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    if (include_next_rmsnorm) {
        require_kernel_subsequence(
            scheduled_kernel_sequence(hrx_graph),
            { expected_kernel, "loom_libs:ggml_rmsnorm_binary_f32", "loom_libs:ggml_binary_f32" });
    } else {
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { expected_kernel });
    }

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight(ggml_nbytes(cpu_weight), 0);
    const std::vector<float>   input(static_cast<size_t>(input_size * token_count), 0.00390625f);
    const std::vector<float>   bias     = make_pattern_f32(static_cast<size_t>(output_size), 13, 0.02f);
    const std::vector<float>   residual = make_pattern_f32(static_cast<size_t>(output_size * token_count), 23, 0.02f);
    const std::vector<float>   norm_weight = make_weight(output_size);
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    if (include_bias) {
        set_tensor_pair_bytes(cpu_backend, cpu_bias, hrx_backend, hrx_bias, bias.data(), bias.size() * sizeof(float));
    }
    if (include_residual) {
        set_tensor_pair_bytes(cpu_backend, cpu_residual, hrx_backend, hrx_residual, residual.data(),
                              residual.size() * sizeof(float));
    }
    if (include_next_rmsnorm) {
        set_tensor_pair_bytes(cpu_backend, cpu_norm_weight, hrx_backend, hrx_norm_weight, norm_weight.data(),
                              norm_weight.size() * sizeof(float));
    }

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    if (!include_next_rmsnorm) {
        require_close(get_f32_tensor(hrx_backend, hrx_postops), get_f32_tensor(cpu_backend, cpu_postops), 1.0f,
                      3.0e-2f);
    }
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 1.0f, 5.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_dense_matmul_zero_weight_case(ggml_type    weight_type,
                                              const char * expected_kernel,
                                              int64_t      token_count = 2) {
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = static_cast<size_t>(32 * 1024 * 1024);
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t input_size  = kQwenHiddenSize;
    constexpr int64_t output_size = 128;
    ggml_tensor *     weight      = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
    ggml_tensor *     input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     output      = ggml_mul_mat(ctx, weight, input);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    require_kernel_subsequence(scheduled_kernel_sequence(graph), { expected_kernel });

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, hrx_backend);
    REQUIRE(buffer != nullptr);

    const std::vector<uint8_t> weight_bytes(ggml_nbytes(weight), 0);
    const std::vector<float>   input_data = make_pattern_f32(input_size * token_count, 7, 0.01f);
    const std::vector<float>   expected(static_cast<size_t>(output_size * token_count), 0.0f);
    set_tensor_bytes(hrx_backend, weight, weight_bytes.data(), weight_bytes.size());
    set_tensor_bytes(hrx_backend, input, input_data.data(), input_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(hrx_backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, output), expected, 0.0f, 0.0f);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(hrx_backend);
}

static void run_endpoint_rmsnorm_q6k_q8_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 32 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t token_count = 1;
    ggml_tensor *     cpu_input   = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor *     cpu_norm_w  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor *     cpu_weight  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount);
    ggml_tensor *     cpu_norm    = build_rmsnorm_mul_graph(cpu_ctx, cpu_input, cpu_norm_w);
    ggml_tensor *     cpu_output  = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_norm);

    ggml_tensor * hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, kQwenHiddenSize, token_count);
    ggml_tensor * hrx_norm_w = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, kQwenHiddenSize);
    ggml_tensor * hrx_weight = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount);
    ggml_tensor * hrx_norm   = build_rmsnorm_mul_graph(hrx_ctx, hrx_input, hrx_norm_w);
    ggml_tensor * hrx_output = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_norm);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(
        scheduled_kernel_sequence(hrx_graph),
        { "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4", "qwen3_moe:ggml_linear_q6k_q8_1_x4" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   input  = make_pattern_f32(kQwenHiddenSize * token_count, 8, 0.01f);
    const std::vector<float>   norm_w = make_weight(kQwenHiddenSize);
    const std::vector<uint8_t> weight = make_quantized_rows(GGML_TYPE_Q6_K, kQwenHiddenSize, kQwenVocabularyCount, 9);
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_norm_w, hrx_backend, hrx_norm_w, norm_w.data(),
                          norm_w.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 6.0e-1f, 3.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_lfm_head_rope_cpu_reference_case(int64_t head_count, int64_t token_count, bool cache_output) {
    constexpr int64_t head_size = 64;
    constexpr float   epsilon   = 1.0e-5f;
    const int64_t     row_size  = head_size * head_count;
    const int64_t     cache_rows = token_count + 64;

    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    struct HeadGraph {
        ggml_tensor * input;
        ggml_tensor * weight;
        ggml_tensor * positions;
        ggml_tensor * cache;
        ggml_tensor * row_ids;
        ggml_tensor * output;
    };

    auto build = [&](ggml_context * ctx) {
        HeadGraph g = {};
        g.input     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        g.weight    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size);
        g.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * normalized = ggml_rms_norm(ctx, g.input, epsilon);
        ggml_tensor * scaled     = ggml_mul(ctx, normalized, g.weight);
        ggml_tensor * rope = ggml_rope_ext(ctx, scaled, g.positions, nullptr, head_size, GGML_ROPE_TYPE_NEOX, 0,
                                          10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        g.output = rope;
        if (cache_output) {
            g.cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, row_size, cache_rows);
            g.row_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
            ggml_tensor * rows = ggml_reshape_2d(ctx, rope, row_size, token_count);
            ggml_tensor * view = ggml_view_2d(ctx, rows, row_size, token_count, rows->nb[1], 0);
            g.output = ggml_set_rows(ctx, g.cache, view, g.row_ids);
        }
        REQUIRE(g.output != nullptr);
        return g;
    };

    HeadGraph cpu = build(cpu_ctx);
    HeadGraph hrx = build(hrx_ctx);
    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu.output);
    ggml_build_forward_expand(hrx_graph, hrx.output);
    const std::vector<std::string> kernels = scheduled_kernel_sequence(hrx_graph);
    if (cache_output) {
        REQUIRE(std::find(kernels.begin(), kernels.end(), "loom_libs:ggml_rmsnorm_mul_rope_f32") == kernels.end());
        require_kernel_subsequence(kernels, { "loom_libs:ggml_rope_f32", "loom_libs:ggml_set_rows" });
    } else {
        require_kernel_subsequence(kernels, { "loom_libs:ggml_rmsnorm_mul_rope_f32" });
    }

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> input  = make_pattern_f32(static_cast<size_t>(row_size * token_count), 47, 0.03f);
    const std::vector<float> weight = make_weight(head_size);
    std::vector<int32_t> positions(static_cast<size_t>(token_count));
    std::vector<int64_t> row_ids(static_cast<size_t>(token_count));
    for (int64_t i = 0; i < token_count; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(3 + 2 * i);
        row_ids[static_cast<size_t>(i)]   = (17 * i + 5) % cache_rows;
    }
    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.weight, hrx_backend, hrx.weight, weight.data(),
                          weight.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.positions, hrx_backend, hrx.positions, positions.data(),
                          positions.size() * sizeof(int32_t));
    if (cache_output) {
        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(row_size * cache_rows), ggml_fp32_to_fp16(-3.0f));
        set_tensor_pair_bytes(cpu_backend, cpu.cache, hrx_backend, hrx.cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu.row_ids, hrx_backend, hrx.row_ids, row_ids.data(),
                              row_ids.size() * sizeof(int64_t));
    }

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    const float tolerance = cache_output ? 1.0e-3f : 1.0e-4f;
    require_close(get_f32_tensor(hrx_backend, hrx.output), get_f32_tensor(cpu_backend, cpu.output), tolerance,
                  tolerance);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_lfm_head_rope_availability_checks() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 1);
    ggml_tensor * weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * pos_f32  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * positions = ggml_cast(ctx, pos_f32, GGML_TYPE_I32);
    ggml_tensor * rms      = ggml_rms_norm(ctx, input, 1.0e-5f);
    ggml_tensor * scaled   = ggml_mul(ctx, rms, weight);
    ggml_tensor * output    = ggml_rope_ext(ctx, scaled, positions, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                                            0.0f, 1.0f, 0.0f, 0.0f);
    REQUIRE(output != nullptr);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry({ "gfx1151" });
    REQUIRE(registry != nullptr);
    const size_t rms_index = producer_index_for_tensor(imported.graph, rms);
    const size_t pos_index = producer_index_for_tensor(imported.graph, positions);
    ggml::hrx::CommandPlan plan;
    std::vector<bool> covered(imported.graph.nodes().size(), false);
    auto matched_kernel = [&]() {
        ggml::hrx::DispatchMatch match;
        const ggml::hrx::DispatchMatchContext context = {
            imported.graph, &imported.graph.nodes()[rms_index],    rms_index, covered,
            plan,           next_plan_value(imported.graph, plan),
        };
        REQUIRE(registry->match(context, match));
        REQUIRE(match.dispatches.size() == 1);
        return kernel_name_for_id(match.dispatches[0].kernel.kernel_id);
    };
    REQUIRE(matched_kernel() == "loom_libs:ggml_rmsnorm_f32");
    covered[pos_index] = true;
    REQUIRE(matched_kernel() == "loom_libs:ggml_rmsnorm_mul_rope_f32");

    ggml_free(ctx);
}

static void run_rope_set_rows_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    constexpr int64_t head_size       = 8;
    constexpr int64_t head_count      = 2;
    constexpr int64_t token_count     = 3;
    constexpr int64_t cache_row_count = 8;
    constexpr int64_t hidden_size     = head_size * head_count;

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_input = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * cpu_pos   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * cpu_freq  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * cpu_out = ggml_rope_ext(cpu_ctx, cpu_input, cpu_pos, cpu_freq, head_size, GGML_ROPE_TYPE_NEOX, 0,
                                              10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * hrx_input = ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * hrx_pos   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * hrx_freq  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * hrx_out = ggml_rope_ext(hrx_ctx, hrx_input, hrx_pos, hrx_freq, head_size, GGML_ROPE_TYPE_NEOX, 0,
                                              10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rope_f32" });
        const ggml::hrx::KernelSpecialization specialization =
            scheduled_kernel_specialization(hrx_graph, "loom_libs:ggml_rope_f32");
        REQUIRE(specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
        REQUIRE(specialization.compile_parameters.at("ggml.rope_f32.token_capacity") == "4");
        REQUIRE(specialization.compile_parameters.count("ggml.rope_f32.input_span") == 0);
        REQUIRE(specialization.integer_parameters.at("input_span") == hidden_size * token_count);

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<float>   input = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 41, 0.03f);
        const std::vector<int32_t> pos   = { 0, 1, 7 };
        const std::vector<float>   freq  = { 1.0f, 0.5f, 0.25f, 0.125f };
        set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(),
                              input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_pos, hrx_backend, hrx_pos, pos.data(), pos.size() * sizeof(int32_t));
        set_tensor_pair_bytes(cpu_backend, cpu_freq, hrx_backend, hrx_freq, freq.data(), freq.size() * sizeof(float));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-4f, 1.0e-4f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 8 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        constexpr int64_t partial_head_size  = 128;
        constexpr int64_t partial_n_dims     = 96;
        constexpr int64_t partial_head_count = 3;
        constexpr int64_t partial_tokens     = 4;
        ggml_tensor *     cpu_input =
            ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, partial_head_size, partial_head_count, partial_tokens);
        ggml_tensor * cpu_pos  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, partial_tokens);
        ggml_tensor * cpu_freq = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, partial_n_dims / 2);
        ggml_tensor * cpu_out  = ggml_rope_ext(cpu_ctx, cpu_input, cpu_pos, cpu_freq, partial_n_dims,
                                               GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * hrx_input =
            ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, partial_head_size, partial_head_count, partial_tokens);
        ggml_tensor * hrx_pos  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, partial_tokens);
        ggml_tensor * hrx_freq = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, partial_n_dims / 2);
        ggml_tensor * hrx_out  = ggml_rope_ext(hrx_ctx, hrx_input, hrx_pos, hrx_freq, partial_n_dims,
                                               GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rope_f32" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<float> input =
            make_pattern_f32(static_cast<size_t>(partial_head_size * partial_head_count * partial_tokens), 47, 0.01f);
        const std::vector<int32_t> pos = { 0, 1, 13, 29 };
        std::vector<float>         freq(static_cast<size_t>(partial_n_dims / 2));
        for (size_t i = 0; i < freq.size(); ++i) {
            freq[i] = 1.0f + 0.01f * static_cast<float>(i);
        }
        set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(),
                              input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_pos, hrx_backend, hrx_pos, pos.data(), pos.size() * sizeof(int32_t));
        set_tensor_pair_bytes(cpu_backend, cpu_freq, hrx_backend, hrx_freq, freq.data(), freq.size() * sizeof(float));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-4f, 1.0e-4f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    auto run_view_set_rows_case = [&](int64_t view_token_count) {
        ggml_init_params params = {};
        params.mem_size         = 16 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        constexpr int64_t phi_hidden_size     = 1024;
        constexpr int64_t phi_cache_row_count = 256;
        constexpr int64_t phi_row_stride      = 5120;
        constexpr size_t  row_offset          = 8 * sizeof(float);
        const size_t      storage_elements =
            row_offset / sizeof(float) + static_cast<size_t>((view_token_count - 1) * phi_row_stride + phi_hidden_size);

        ggml_tensor * cpu_cache = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, phi_hidden_size, phi_cache_row_count);
        ggml_tensor * cpu_store = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, storage_elements);
        ggml_tensor * cpu_rows  = ggml_view_2d(cpu_ctx, cpu_store, phi_hidden_size, view_token_count,
                                               phi_row_stride * sizeof(float), row_offset);
        ggml_tensor * cpu_ids   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, view_token_count);
        ggml_tensor * cpu_out   = ggml_set_rows(cpu_ctx, cpu_cache, cpu_rows, cpu_ids);
        ggml_tensor * hrx_cache = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, phi_hidden_size, phi_cache_row_count);
        ggml_tensor * hrx_store = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, storage_elements);
        ggml_tensor * hrx_rows  = ggml_view_2d(hrx_ctx, hrx_store, phi_hidden_size, view_token_count,
                                               phi_row_stride * sizeof(float), row_offset);
        ggml_tensor * hrx_ids   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, view_token_count);
        ggml_tensor * hrx_out   = ggml_set_rows(hrx_ctx, hrx_cache, hrx_rows, hrx_ids);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_set_rows" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(phi_hidden_size * phi_cache_row_count),
                                             ggml_fp32_to_fp16(-2.5f));
        std::vector<float>             storage(storage_elements, -17.0f);
        const std::vector<float>       rows =
            make_pattern_f32(static_cast<size_t>(phi_hidden_size * view_token_count), 52, 0.004f);
        for (int64_t token = 0; token < view_token_count; ++token) {
            std::memcpy(storage.data() + row_offset / sizeof(float) + static_cast<size_t>(token * phi_row_stride),
                        rows.data() + static_cast<size_t>(token * phi_hidden_size),
                        static_cast<size_t>(phi_hidden_size) * sizeof(float));
        }
        std::vector<int64_t> ids(static_cast<size_t>(view_token_count));
        for (int64_t i = 0; i < view_token_count; ++i) {
            ids[static_cast<size_t>(i)] = (i * 17 + 3) % phi_cache_row_count;
        }

        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_store, hrx_backend, hrx_store, storage.data(),
                              storage.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int64_t));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-3f, 1.0e-3f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    };

    run_view_set_rows_case(2);
    run_view_set_rows_case(14);

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_input = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * cpu_pos   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * cpu_freq  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * cpu_out   = ggml_rope_ext(cpu_ctx, cpu_input, cpu_pos, cpu_freq, head_size, GGML_ROPE_TYPE_NORMAL,
                                                0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * hrx_input = ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * hrx_pos   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * hrx_freq  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * hrx_out   = ggml_rope_ext(hrx_ctx, hrx_input, hrx_pos, hrx_freq, head_size, GGML_ROPE_TYPE_NORMAL,
                                                0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rope_f32" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<float>   input = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 45, 0.03f);
        const std::vector<int32_t> pos   = { 0, 1, 7 };
        const std::vector<float>   freq  = { 1.0f, 0.5f, 0.25f, 0.125f };
        set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(),
                              input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_pos, hrx_backend, hrx_pos, pos.data(), pos.size() * sizeof(int32_t));
        set_tensor_pair_bytes(cpu_backend, cpu_freq, hrx_backend, hrx_freq, freq.data(), freq.size() * sizeof(float));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-4f, 1.0e-4f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_cache = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * cpu_rows  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, hidden_size, token_count);
        ggml_tensor * cpu_ids   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * cpu_out   = ggml_set_rows(cpu_ctx, cpu_cache, cpu_rows, cpu_ids);
        ggml_tensor * hrx_cache = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * hrx_rows  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, hidden_size, token_count);
        ggml_tensor * hrx_ids   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * hrx_out   = ggml_set_rows(hrx_ctx, hrx_cache, hrx_rows, hrx_ids);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_set_rows" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(hidden_size * cache_row_count),
                                             ggml_fp32_to_fp16(-4.0f));
        const std::vector<ggml_fp16_t> rows =
            make_pattern_f16(static_cast<size_t>(hidden_size * token_count), 44, 0.04f);
        const std::vector<int64_t> ids = { 7, 0, 5 };
        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_rows, hrx_backend, hrx_rows, rows.data(),
                              rows.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int64_t));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-3f, 1.0e-3f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_cache = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * cpu_rows  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, token_count);
        ggml_tensor * cpu_ids   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * cpu_out   = ggml_set_rows(cpu_ctx, cpu_cache, cpu_rows, cpu_ids);
        ggml_tensor * hrx_cache = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * hrx_rows  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, hidden_size, token_count);
        ggml_tensor * hrx_ids   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * hrx_out   = ggml_set_rows(hrx_ctx, hrx_cache, hrx_rows, hrx_ids);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_set_rows" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(hidden_size * cache_row_count),
                                             ggml_fp32_to_fp16(-2.0f));
        const std::vector<float>   rows = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 42, 0.04f);
        const std::vector<int64_t> ids  = { 6, 2, 4 };
        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_rows, hrx_backend, hrx_rows, rows.data(), rows.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int64_t));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-3f, 1.0e-3f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_input = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * cpu_pos   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * cpu_freq  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * cpu_rope = ggml_rope_ext(cpu_ctx, cpu_input, cpu_pos, cpu_freq, head_size, GGML_ROPE_TYPE_NEOX, 0,
                                               10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * cpu_rows = ggml_reshape_2d(cpu_ctx, cpu_rope, hidden_size, token_count);
        ggml_tensor * cpu_cache = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * cpu_ids   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * cpu_out   = ggml_set_rows(cpu_ctx, cpu_cache, cpu_rows, cpu_ids);
        ggml_tensor * hrx_input = ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * hrx_pos   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * hrx_freq  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * hrx_rope = ggml_rope_ext(hrx_ctx, hrx_input, hrx_pos, hrx_freq, head_size, GGML_ROPE_TYPE_NEOX, 0,
                                               10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * hrx_rows = ggml_reshape_2d(hrx_ctx, hrx_rope, hidden_size, token_count);
        ggml_tensor * hrx_cache = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * hrx_ids   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * hrx_out   = ggml_set_rows(hrx_ctx, hrx_cache, hrx_rows, hrx_ids);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rope_set_rows_f32" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<float>   input = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 43, 0.03f);
        const std::vector<int32_t> pos   = { 0, 1, 7 };
        const std::vector<float>   freq  = { 1.0f, 0.5f, 0.25f, 0.125f };
        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(hidden_size * cache_row_count),
                                             ggml_fp32_to_fp16(-3.0f));
        const std::vector<int64_t>     ids = { 5, 1, 3 };
        set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(),
                              input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_pos, hrx_backend, hrx_pos, pos.data(), pos.size() * sizeof(int32_t));
        set_tensor_pair_bytes(cpu_backend, cpu_freq, hrx_backend, hrx_freq, freq.data(), freq.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int64_t));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-3f, 1.0e-3f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * cpu_ctx  = ggml_init(params);
        ggml_context * hrx_ctx  = ggml_init(params);
        REQUIRE(cpu_ctx != nullptr);
        REQUIRE(hrx_ctx != nullptr);

        ggml_tensor * cpu_input = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * cpu_pos   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * cpu_freq  = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * cpu_rope  = ggml_rope_ext(cpu_ctx, cpu_input, cpu_pos, cpu_freq, head_size, GGML_ROPE_TYPE_NORMAL,
                                                0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * cpu_rows  = ggml_reshape_2d(cpu_ctx, cpu_rope, hidden_size, token_count);
        ggml_tensor * cpu_cache = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * cpu_ids   = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * cpu_out   = ggml_set_rows(cpu_ctx, cpu_cache, cpu_rows, cpu_ids);
        ggml_tensor * hrx_input = ggml_new_tensor_3d(hrx_ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * hrx_pos   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * hrx_freq  = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * hrx_rope  = ggml_rope_ext(hrx_ctx, hrx_input, hrx_pos, hrx_freq, head_size, GGML_ROPE_TYPE_NORMAL,
                                                0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * hrx_rows  = ggml_reshape_2d(hrx_ctx, hrx_rope, hidden_size, token_count);
        ggml_tensor * hrx_cache = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F16, hidden_size, cache_row_count);
        ggml_tensor * hrx_ids   = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * hrx_out   = ggml_set_rows(hrx_ctx, hrx_cache, hrx_rows, hrx_ids);
        REQUIRE(cpu_out != nullptr);
        REQUIRE(hrx_out != nullptr);

        ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
        ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
        REQUIRE(cpu_graph != nullptr);
        REQUIRE(hrx_graph != nullptr);
        ggml_build_forward_expand(cpu_graph, cpu_out);
        ggml_build_forward_expand(hrx_graph, hrx_out);
        require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "loom_libs:ggml_rope_set_rows_f32" });

        ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
        ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
        REQUIRE(cpu_buffer != nullptr);
        REQUIRE(hrx_buffer != nullptr);

        const std::vector<float>   input = make_pattern_f32(static_cast<size_t>(hidden_size * token_count), 46, 0.03f);
        const std::vector<int32_t> pos   = { 0, 1, 7 };
        const std::vector<float>   freq  = { 1.0f, 0.5f, 0.25f, 0.125f };
        const std::vector<ggml_fp16_t> cache(static_cast<size_t>(hidden_size * cache_row_count),
                                             ggml_fp32_to_fp16(-5.0f));
        const std::vector<int64_t>     ids = { 4, 2, 6 };
        set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(),
                              input.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_pos, hrx_backend, hrx_pos, pos.data(), pos.size() * sizeof(int32_t));
        set_tensor_pair_bytes(cpu_backend, cpu_freq, hrx_backend, hrx_freq, freq.data(), freq.size() * sizeof(float));
        set_tensor_pair_bytes(cpu_backend, cpu_cache, hrx_backend, hrx_cache, cache.data(),
                              cache.size() * sizeof(ggml_fp16_t));
        set_tensor_pair_bytes(cpu_backend, cpu_ids, hrx_backend, hrx_ids, ids.data(), ids.size() * sizeof(int64_t));

        REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu_backend);
        ggml_backend_synchronize(hrx_backend);
        require_close(get_f32_tensor(hrx_backend, hrx_out), get_f32_tensor(cpu_backend, cpu_out), 1.0e-3f, 1.0e-3f);

        ggml_backend_buffer_free(cpu_buffer);
        ggml_backend_buffer_free(hrx_buffer);
        ggml_free(cpu_ctx);
        ggml_free(hrx_ctx);
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_attention_postprocess_cpu_reference_case() {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 32 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    constexpr int64_t         token_count          = 2;
    constexpr int64_t         query_head_count     = 4;
    constexpr int64_t         key_value_head_count = 2;
    constexpr int64_t         cache_row_count      = 8;
    const int64_t             query_size           = query_head_count * kQwenFlashHeadSize;
    const int64_t             key_value_size       = key_value_head_count * kQwenFlashHeadSize;
    AttentionPostprocessGraph cpu = build_attention_postprocess_graph(cpu_ctx, token_count, query_head_count,
                                                                      key_value_head_count, cache_row_count);
    AttentionPostprocessGraph hrx = build_attention_postprocess_graph(hrx_ctx, token_count, query_head_count,
                                                                      key_value_head_count, cache_row_count);

    auto hrx_query_buffer = allocate_test_weight(hrx_backend, hrx.query_weight);
    auto hrx_key_buffer   = allocate_test_weight(hrx_backend, hrx.key_weight);
    auto hrx_value_buffer = allocate_test_weight(hrx_backend, hrx.value_weight);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu.query_output);
    ggml_build_forward_expand(cpu_graph, cpu.key_output);
    ggml_build_forward_expand(cpu_graph, cpu.value_output);
    ggml_build_forward_expand(hrx_graph, hrx.query_output);
    ggml_build_forward_expand(hrx_graph, hrx.key_output);
    ggml_build_forward_expand(hrx_graph, hrx.value_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph),
                               { "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float>   input     = make_pattern_f32(kQwenHiddenSize * token_count, 10, 0.01f);
    const std::vector<uint8_t> query_w   = make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, query_size, 11);
    const std::vector<uint8_t> key_w     = make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, key_value_size, 12);
    const std::vector<uint8_t> value_w   = make_quantized_rows(GGML_TYPE_Q6_K, kQwenHiddenSize, key_value_size, 13);
    const std::vector<float>   query_nw  = make_weight(kQwenFlashHeadSize);
    const std::vector<float>   key_nw    = make_pattern_f32(kQwenFlashHeadSize, 14, 0.02f);
    const std::vector<int32_t> positions = make_i32_mod_data(token_count, 1024);
    std::vector<float>         inv_freq(static_cast<size_t>(kQwenFlashHeadSize / 2));
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(kQwenFlashHeadSize));
    }
    const std::vector<ggml_fp16_t> key_cache(static_cast<size_t>(key_value_size * cache_row_count),
                                             ggml_fp32_to_fp16(0.0f));
    const std::vector<ggml_fp16_t> value_cache(static_cast<size_t>(key_value_size * cache_row_count),
                                               ggml_fp32_to_fp16(0.0f));
    const std::vector<int64_t>     cache_indices = make_i64_mod_data(token_count, cache_row_count);

    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.query_weight, hrx_backend, hrx.query_weight, query_w.data(), query_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.key_weight, hrx_backend, hrx.key_weight, key_w.data(), key_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.value_weight, hrx_backend, hrx.value_weight, value_w.data(), value_w.size());
    set_tensor_pair_bytes(cpu_backend, cpu.query_norm_weight, hrx_backend, hrx.query_norm_weight, query_nw.data(),
                          query_nw.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.key_norm_weight, hrx_backend, hrx.key_norm_weight, key_nw.data(),
                          key_nw.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.positions, hrx_backend, hrx.positions, positions.data(),
                          positions.size() * sizeof(int32_t));
    set_tensor_pair_bytes(cpu_backend, cpu.inverse_frequencies, hrx_backend, hrx.inverse_frequencies, inv_freq.data(),
                          inv_freq.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.key_cache, hrx_backend, hrx.key_cache, key_cache.data(),
                          key_cache.size() * sizeof(ggml_fp16_t));
    set_tensor_pair_bytes(cpu_backend, cpu.value_cache, hrx_backend, hrx.value_cache, value_cache.data(),
                          value_cache.size() * sizeof(ggml_fp16_t));
    set_tensor_pair_bytes(cpu_backend, cpu.key_cache_indices, hrx_backend, hrx.key_cache_indices, cache_indices.data(),
                          cache_indices.size() * sizeof(int64_t));
    set_tensor_pair_bytes(cpu_backend, cpu.value_cache_indices, hrx_backend, hrx.value_cache_indices,
                          cache_indices.data(), cache_indices.size() * sizeof(int64_t));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx.query_output), get_f32_tensor(cpu_backend, cpu.query_output), 2.0f,
                  5.0e-2f);
    require_close(get_f32_tensor(hrx_backend, hrx.key_output), get_f32_tensor(cpu_backend, cpu.key_output), 2.0f,
                  5.0e-2f);
    require_close(get_f32_tensor(hrx_backend, hrx.value_output), get_f32_tensor(cpu_backend, cpu.value_output), 2.0f,
                  5.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_routed_moe_cpu_reference_case(ggml_type down_weight_type, bool include_next_rmsnorm) {
    ggml_backend_t cpu_backend = init_cpu_backend();
    ggml_backend_t hrx_backend = ggml_backend_hrx_init(0);
    REQUIRE(hrx_backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 64 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * cpu_ctx  = ggml_init(params);
    ggml_context * hrx_ctx  = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    REQUIRE(hrx_ctx != nullptr);

    RoutedMoeGraph cpu = build_routed_moe_graph(cpu_ctx, down_weight_type, include_next_rmsnorm);
    RoutedMoeGraph hrx = build_routed_moe_graph(hrx_ctx, down_weight_type, include_next_rmsnorm);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu.output);
    ggml_build_forward_expand(hrx_graph, hrx.output);

    std::vector<std::string> expected = {
        "qwen3_moe:qwen3_moe_router_top8_f32",
        "loom_libs:ggml_moe_build_expert_table",
        "loom_libs:ggml_moe_build_expert_partition_table",
        "loom_libs:ggml_mul_mat_id_swiglu_f16_f16_wmma",
        "loom_libs:ggml_mul_mat_id_f16_f16_wmma",
        include_next_rmsnorm ? "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32" :
                               "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32",
    };
    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), expected);

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<float> logits = make_router_logits(1);
    const std::vector<float> input  = make_pattern_f32(kQwenHiddenSize, 15, 0.01f);
    const std::vector<float> hidden = make_pattern_f32(kQwenHiddenSize, 16, 0.02f);
    set_tensor_pair_bytes(cpu_backend, cpu.logits, hrx_backend, hrx.logits, logits.data(),
                          logits.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.input, hrx_backend, hrx.input, input.data(), input.size() * sizeof(float));
    set_tensor_pair_bytes(cpu_backend, cpu.hidden_state, hrx_backend, hrx.hidden_state, hidden.data(),
                          hidden.size() * sizeof(float));
    if (include_next_rmsnorm) {
        const std::vector<float> norm = make_weight(kQwenHiddenSize);
        set_tensor_pair_bytes(cpu_backend, cpu.norm_weight, hrx_backend, hrx.norm_weight, norm.data(),
                              norm.size() * sizeof(float));
    }

    {
        const std::vector<uint8_t> gate =
            make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate * kQwenRouterExpertCount, 17);
        set_tensor_pair_bytes(cpu_backend, cpu.gate_weight, hrx_backend, hrx.gate_weight, gate.data(), gate.size());
    }
    {
        const std::vector<uint8_t> up =
            make_quantized_rows(GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate * kQwenRouterExpertCount, 18);
        set_tensor_pair_bytes(cpu_backend, cpu.up_weight, hrx_backend, hrx.up_weight, up.data(), up.size());
    }
    {
        const std::vector<uint8_t> down =
            make_quantized_rows(down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize * kQwenRouterExpertCount, 19);
        set_tensor_pair_bytes(cpu_backend, cpu.down_weight, hrx_backend, hrx.down_weight, down.data(), down.size());
    }

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx.output), get_f32_tensor(cpu_backend, cpu.output), 2.5f, 1.0e-1f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(hrx_backend);
}

static void run_decode_routed_moe_scheduling_case(ggml_type down_weight_type, bool alias_gate_input = false) {
    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * hidden_state     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, 1);
    ggml_tensor * attention_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(hidden_state != nullptr);
    REQUIRE(attention_weight != nullptr);
    ggml_tensor * attention_rms = ggml_rms_norm(ctx, hidden_state, kQwenRmsNormEps);
    REQUIRE(attention_rms != nullptr);
    ggml_tensor * attention_prepared = ggml_mul(ctx, attention_rms, attention_weight);
    REQUIRE(attention_prepared != nullptr);
    ggml_tensor * moe_input = attention_prepared;
    if (alias_gate_input) {
        moe_input = ggml_reshape_2d(ctx, attention_prepared, kQwenHiddenSize, 1);
        REQUIRE(moe_input != nullptr);
    }

    ggml_tensor * router_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenHiddenSize, kQwenRouterExpertCount);
    REQUIRE(router_weight != nullptr);
    ggml_tensor * logits = ggml_mul_mat(ctx, router_weight, attention_prepared);
    REQUIRE(logits != nullptr);
    ggml_tensor * route_ids     = nullptr;
    ggml_tensor * route_weights = build_qwen_router_top8_graph(ctx, logits, &route_ids);
    REQUIRE(route_ids != nullptr);
    REQUIRE(route_weights != nullptr);

    ggml_tensor * gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    ggml_tensor * up_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenMoeIntermediate, kQwenRouterExpertCount);
    ggml_tensor * down_weight =
        ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediate, kQwenHiddenSize, kQwenRouterExpertCount);
    REQUIRE(gate_weight != nullptr);
    REQUIRE(up_weight != nullptr);
    REQUIRE(down_weight != nullptr);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_weight, moe_input, route_ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, up_weight, moe_input, route_ids);
    REQUIRE(gate != nullptr);
    REQUIRE(up != nullptr);
    ggml_tensor * glu = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
    REQUIRE(glu != nullptr);
    ggml_tensor * down = ggml_mul_mat_id(ctx, down_weight, glu, route_ids);
    REQUIRE(down != nullptr);
    ggml_tensor * weighted = ggml_mul(ctx, down, route_weights);
    REQUIRE(weighted != nullptr);

    std::vector<ggml_tensor *> route_views;
    route_views.reserve(kQwenRouterRouteCount);
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view = ggml_view_2d(ctx, weighted, kQwenHiddenSize, 1, weighted->nb[2],
                                          static_cast<size_t>(route) * weighted->nb[1]);
        REQUIRE(view != nullptr);
        route_views.push_back(view);
    }

    ggml_tensor * reduced = route_views.front();
    for (size_t i = 1; i < route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, route_views[i]);
        REQUIRE(reduced != nullptr);
    }
    ggml_tensor * residual = ggml_add(ctx, hidden_state, reduced);
    REQUIRE(residual != nullptr);
    ggml_tensor * next_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenHiddenSize);
    REQUIRE(next_norm_weight != nullptr);
    ggml_tensor * next_rms = ggml_rms_norm(ctx, residual, kQwenRmsNormEps);
    REQUIRE(next_rms != nullptr);
    ggml_tensor * output = ggml_mul(ctx, next_rms, next_norm_weight);
    REQUIRE(output != nullptr);
    ggml_tensor * next_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenHiddenSize, kQwenHiddenSize);
    ggml_tensor * next_output = ggml_mul_mat(ctx, next_weight, output);
    REQUIRE(next_weight != nullptr);
    REQUIRE(next_output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, next_output);

    std::vector<std::string> expected = {
        "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4",
        "qwen3_moe:qwen3_moe_router_projection_top8_fused_decode_f32",
        down_weight_type == GGML_TYPE_Q4_K ? "loom_libs:ggml_mul_mat_id_vector_pair_binary_publish_q8" :
                                             "loom_libs:ggml_mul_mat_id_vector_pair_binary_publish_f32",
        down_weight_type == GGML_TYPE_Q4_K ? "loom_libs:ggml_mul_mat_id_vector_weighted_wave32_publish_q8" :
                                             "loom_libs:ggml_mul_mat_id_vector_weighted_wave64_publish_q8",
    };
    require_kernel_subsequence(scheduled_kernel_sequence(graph), expected);
    ggml_free(ctx);
}

static void run_decode_attention_qkv_scheduling_case() {
    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    AttentionPostprocessGraph graph  = build_decode_attention_qkv_graph(ctx);
    ggml_cgraph *             cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, graph.query_output);
    ggml_build_forward_expand(cgraph, graph.key_output);
    ggml_build_forward_expand(cgraph, graph.value_output);

    require_kernel_subsequence(scheduled_kernel_sequence(cgraph),
                               { "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4",
                                 "qwen3_moe:qwen3_moe_attention_qkv_postprocess_fused_decode" });
    ggml_free(ctx);
}

struct SsmConvPrefillGraph {
    ggml_context * ctx = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * pool = nullptr;
    ggml_tensor * initial = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * filter = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * cache = nullptr;
    ggml_cgraph * graph = nullptr;
};

static SsmConvPrefillGraph make_ssm_conv_prefill_graph(ggml_type type,
                                                       int64_t   k,
                                                       int64_t   n,
                                                       bool      expose,
                                                       bool      gathered,
                                                       bool      clear_state) {
    ggml_init_params params = {};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    SsmConvPrefillGraph c;
    c.ctx = ggml_init(params);
    REQUIRE(c.ctx != nullptr);
    c.input = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, k, 512);
    c.weight = ggml_new_tensor_2d(c.ctx, type, k, n);
    c.pool = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 3 * n, 2);
    c.initial = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 3, n, 1);
    c.ids = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 1);
    c.filter = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 4, n);
    c.graph = ggml_new_graph(c.ctx);
    auto state = c.initial;
    if (gathered) {
        if (clear_state) {
            auto row = ggml_view_2d(c.ctx, c.pool, 3 * n, 1, c.pool->nb[1], 0);
            ggml_build_forward_expand(c.graph, ggml_scale_inplace(c.ctx, row, 0.0f));
        }
        state = ggml_get_rows(c.ctx, c.pool, c.ids);
        ggml_build_forward_expand(c.graph, state);
        state = ggml_reshape_3d(c.ctx, state, 3, n, 1);
    }
    auto raw = ggml_mul_mat(c.ctx, c.weight, c.input);
    auto x = ggml_reshape_3d(c.ctx, raw, n, 512, 1);
    auto window = ggml_concat(c.ctx, state, ggml_transpose(c.ctx, x), 0);
    auto tail = ggml_view_3d(c.ctx, window, 3, n, 1, window->nb[1], window->nb[2], 512 * sizeof(float));
    c.cache = ggml_view_2d(c.ctx, c.pool, 3 * n, 1, c.pool->nb[1], 0);
    ggml_build_forward_expand(c.graph, ggml_cpy(c.ctx, tail, c.cache));
    c.output = ggml_silu(c.ctx, ggml_ssm_conv(c.ctx, window, c.filter));
    ggml_set_output(c.output);
    ggml_build_forward_expand(c.graph, c.output);
    if (expose) {
        ggml_set_output(raw);
        ggml_build_forward_expand(c.graph, ggml_scale(c.ctx, raw, 0.5f));
    }
    const auto sequence = scheduled_kernel_sequence(c.graph);
    const char * fused = gathered ? "loom_libs:ggml_mul_mat_quantized_f16_wmma_prefill_conv4_interior" :
                                    "loom_libs:ggml_mul_mat_quantized_f16_wmma_prefill_conv4";
    REQUIRE((std::find(sequence.begin(), sequence.end(), fused) != sequence.end()) == !expose);
    if (gathered && !expose) {
        REQUIRE(std::find(sequence.begin(), sequence.end(), "loom_libs:llm_ssm_conv_dconv4_silu_prefill_finish_f32") !=
                sequence.end());
    }
    return c;
}

static void run_ssm_conv_prefill_recurrent_case(ggml_type type, bool gathered, bool clear_state) {
    const int64_t k = 256;
    const int64_t n = 8192;
    auto backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    auto a = make_ssm_conv_prefill_graph(type, k, n, false, gathered, clear_state);
    auto b = make_ssm_conv_prefill_graph(type, k, n, true, gathered, clear_state);
    auto aw = allocate_test_weight(backend, a.weight);
    auto bw = allocate_test_weight(backend, b.weight);
    auto ab = ggml_backend_alloc_ctx_tensors(a.ctx, backend);
    auto bb = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    REQUIRE(ab != nullptr && bb != nullptr);
    const auto weights = make_matmul_weight_bytes(type, k, n, 6);
    const auto filters = make_pattern_f32(4 * n, 11, 0.0625f);
    const auto states = make_pattern_f32(6 * n, 17, 0.03125f);
    set_tensor_pair_bytes(backend, a.weight, backend, b.weight, weights.data(), weights.size());
    set_tensor_pair_bytes(backend, a.filter, backend, b.filter, filters.data(), filters.size() * sizeof(float));
    set_tensor_pair_bytes(backend, a.pool, backend, b.pool, states.data(), states.size() * sizeof(float));
    set_tensor_pair_bytes(backend, a.initial, backend, b.initial, states.data(), 3 * n * sizeof(float));
    for (int round = 0; round < 5; ++round) {
        const int32_t id = round % 2;
        set_tensor_pair_bytes(backend, a.ids, backend, b.ids, &id, sizeof(id));
        const auto input = make_pattern_f32(k * 512, 7 + round, 0.0625f);
        set_tensor_pair_bytes(backend, a.input, backend, b.input, input.data(), input.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, a.graph) == GGML_STATUS_SUCCESS);
        REQUIRE(ggml_backend_graph_compute(backend, b.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        const auto actual = get_f32_tensor(backend, a.output);
        const auto expected = get_f32_tensor(backend, b.output);
        require_close(get_f32_tensor(backend, a.cache), get_f32_tensor(backend, b.cache), 0.0f, 0.0f);
        require_close(actual, expected, 0.0f, 0.0f);
    }
    ggml_backend_buffer_free(ab);
    ggml_backend_buffer_free(bb);
    aw.reset();
    bw.reset();
    ggml_free(a.ctx);
    ggml_free(b.ctx);
    ggml_backend_free(backend);
}

static void run_vector_output_capacity_experiment() {
    constexpr int64_t kSmallVectorInputSize = 256;
    constexpr int64_t kAboveOldOutputSize   = 262208;
    constexpr int64_t kLiftedOutputSize     = 524288;

    run_dense_matmul_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, kAboveOldOutputSize,
                                        kSmallVectorInputSize);
    run_dense_matmul_unary_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_vector_f32_f32", 1,
                                              kAboveOldOutputSize, kSmallVectorInputSize);
    run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16, "loom_libs:ggml_mul_mat_vector_bias_residual_f32_f32", 1,
                                                kAboveOldOutputSize, true, false, false, kSmallVectorInputSize);
    run_dense_matmul_cpu_reference_case(GGML_TYPE_F16, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, kLiftedOutputSize,
                                        kSmallVectorInputSize);
}

static void run_vector_postops_cpu_reference_case() {
    constexpr int64_t kVectorInputSize  = 256;
    constexpr int64_t kVectorOutputSize = 256;

    run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16, "loom_libs:ggml_mul_mat_vector_bias_residual_f32_f32", 1,
                                                kVectorOutputSize, true, false, false, kVectorInputSize);
    run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16, "loom_libs:ggml_mul_mat_vector_bias_residual_f32_f32", 1,
                                                kVectorOutputSize, true, true, false, kVectorInputSize);
}

static const char * glu_op_name(ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_REGLU:
            return "reglu";
        case GGML_GLU_OP_GEGLU:
            return "geglu";
        case GGML_GLU_OP_SWIGLU:
            return "swiglu";
        case GGML_GLU_OP_SWIGLU_OAI:
            return "swiglu_oai";
        case GGML_GLU_OP_GEGLU_ERF:
            return "geglu_erf";
        case GGML_GLU_OP_GEGLU_QUICK:
            return "geglu_quick";
        case GGML_GLU_OP_COUNT:
            break;
    }
    return "unknown_glu";
}

static std::string type_name(ggml_type type) {
    return ggml_type_name(type);
}

using test_runner::Suite;

static void register_basic_ops_cases(Suite & suite) {
    suite.host_case("support.rmsnorm", [] { run_rmsnorm_support_checks(); });
    suite.host_case("support.alternate_value_alias_lookup", [] { run_alternate_value_alias_lookup_checks(); });
    suite.host_case("support.activation_publication_contract", [] { run_activation_publication_contract_checks(); });
    suite.device_case("add_f32.cpu_reference", [] { run_add_f32_cpu_reference_case(); });
    suite.device_case("dynamic_generic_ops.cpu_reference", [] { run_dynamic_generic_ops_cpu_reference_case(); });
    for (const ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        suite.device_case("ssm_conv_prefill_recurrent." + type_name(type) + ".plain",
                          [type] { run_ssm_conv_prefill_recurrent_case(type, false, false); });
        suite.device_case("ssm_conv_prefill_recurrent." + type_name(type) + ".gathered",
                          [type] { run_ssm_conv_prefill_recurrent_case(type, true, false); });
        suite.device_case("ssm_conv_prefill_recurrent." + type_name(type) + ".gathered_clear_state",
                          [type] { run_ssm_conv_prefill_recurrent_case(type, true, true); });
    }
    suite.device_case("view_mul_f32.cpu_reference", [] { run_view_mul_f32_cpu_reference_case(); });
    for (const int64_t token_count : { 2, 23 }) {
        suite.device_case("strided_cont.tokens" + std::to_string(token_count),
                          [token_count] { run_strided_cont_cpu_reference_case(token_count); });
    }
    for (const ggml_glu_op op :
         { GGML_GLU_OP_REGLU, GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_GEGLU_ERF, GGML_GLU_OP_GEGLU_QUICK }) {
        suite.device_case(std::string("glu_split_f32.") + glu_op_name(op),
                          [op] { run_glu_split_f32_cpu_reference_case(op); });
        suite.device_case(std::string("glu_packed_f32.") + glu_op_name(op) + ".normal",
                          [op] { run_glu_packed_f32_cpu_reference_case(op, false); });
    }
    suite.device_case("glu_packed_f32.swiglu.gated",
                      [] { run_glu_packed_f32_cpu_reference_case(GGML_GLU_OP_SWIGLU, true); });
    suite.device_case("glu_packed_f32.swiglu.outputs8192.tokens2",
                      [] { run_glu_packed_f32_cpu_reference_case(GGML_GLU_OP_SWIGLU, false, 8192, 2); });
    suite.device_case("glu_packed_f32.swiglu.outputs8192.tokens14",
                      [] { run_glu_packed_f32_cpu_reference_case(GGML_GLU_OP_SWIGLU, false, 8192, 14); });
    suite.device_case("gather_add_f32.cpu_reference", [] { run_gather_add_f32_cpu_reference_case(); });
    suite.device_case("scale_add_f32.cpu_reference.packed", [] { run_scale_add_f32_cpu_reference_case(false); });
    suite.device_case("scale_add_f32.cpu_reference.strided", [] { run_scale_add_f32_cpu_reference_case(true); });
    for (const ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_F16,
                                  GGML_TYPE_BF16, GGML_TYPE_F32 }) {
        suite.device_case("get_rows_f32." + type_name(type), [type] { run_get_rows_f32_cpu_reference_case(type); });
    }
    suite.device_case("get_rows_f32.q1_0.rows2048", [] { run_get_rows_f32_cpu_reference_case(GGML_TYPE_Q1_0, 2048); });
    suite.device_case("get_rows_f32.q5_1.rows640", [] { run_get_rows_f32_cpu_reference_case(GGML_TYPE_Q5_1, 640); });
    suite.device_case("get_rows_q8_1.zero_weight", [] { run_get_rows_q8_1_zero_weight_case(); });
    suite.device_case("get_rows_scale_f32.cpu_reference", [] { run_get_rows_scale_f32_cpu_reference_case(); });
    suite.device_case("get_rows_scale_f32.q5_1.cpu_reference",
                      [] { run_get_rows_scale_f32_cpu_reference_case(GGML_TYPE_Q5_1, 640); });
}

static void register_dense_matmul_cases(Suite & suite) {
    // K=384 exercises a partial 256-weight tile; N=65 exercises an unpaired output row.
    for (const int64_t input_size : { 256, 384, 768 }) {
        suite.device_case("dense_matmul.q1_0.vector_tail.k" + std::to_string(input_size), [input_size] {
            run_dense_matmul_cpu_reference_case(
                GGML_TYPE_Q1_0, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, 65, input_size);
        });
    }

    for (const ggml_type type : { GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M }) {
        const char * type_suffix   = type == GGML_TYPE_IQ1_S ? "iq1_s" : "iq1_m";
        const char * vector_kernel = type == GGML_TYPE_IQ1_S ?
                                         "loom_libs:ggml_mul_mat_vector_iq1_s_f32_f32" :
                                         "loom_libs:ggml_mul_mat_vector_iq1_m_f32_f32";
        const char * tiled_kernel  = type == GGML_TYPE_IQ1_S ?
                                         "loom_libs:ggml_mul_mat_tiled_input_f32_iq1_s_publish_f32" :
                                         "loom_libs:ggml_mul_mat_tiled_input_f32_iq1_m_publish_f32";
        suite.device_case(std::string("dense_matmul.") + type_suffix + ".vector.tokens1.outputs128", [=] {
            run_dense_matmul_cpu_reference_case(type, vector_kernel, 1, 128, 2048);
        });
        suite.device_case(std::string("dense_matmul.") + type_suffix + ".tiled.tokens64.outputs128", [=] {
            run_dense_matmul_cpu_reference_case(type, tiled_kernel, 64, 128, 2048);
        });
    }
    for (const int64_t tokens : { 1, 3, 5 }) {
        for (const int64_t outputs : { 1, 4, 47, 48, 63 }) {
            suite.device_case(
                "dense_matmul.q4_k.tokens" + std::to_string(tokens) + ".outputs" + std::to_string(outputs),
                [tokens, outputs] {
                    run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K,
                                                        tokens == 1 ?
                                                            "loom_libs:ggml_mul_mat_vector_f32_f32" :
                                      "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                        tokens, outputs);
                });
        }
    }
    for (const ggml_type type :
         { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32 }) {
        suite.device_case("dense_matmul." + type_name(type) + ".tokens3.outputs47", [type] {
            run_dense_matmul_cpu_reference_case(type, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 3, 47);
        });
    }
    for (const ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        suite.device_case("dense_matmul." + type_name(type) + ".tiled.tokens33.outputs65.input512", [type] {
            run_dense_matmul_cpu_reference_case(type, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 33, 65,
                                                512);
        });
    }
    suite.device_case("dense_matmul_unary.f32.tokens3.outputs47", [] {
        run_dense_matmul_unary_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                                                  3, 47);
    });
    suite.device_case("get_rows_f32.f32.rows262144.tokens4.outputs1",
                      [] { run_get_rows_f32_cpu_reference_case(GGML_TYPE_F32, 262144, 4, 1); });
    suite.device_case("dense_matmul.q4_k.skinny.tokens2.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 2,
                                            128);
    });
    suite.device_case("dense_matmul.q4_k.vector.tokens1.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, 128);
    });
    for (const ggml_type type : { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS }) {
        suite.device_case("dense_matmul." + type_name(type) + ".codebook.vector.tokens1.outputs128", [type] {
            run_dense_matmul_cpu_reference_case(type, "loom_libs:ggml_mul_mat_vector_iq2_s_f32_f32", 1, 128,
                                                2048);
        });
        suite.device_case("dense_matmul." + type_name(type) + ".codebook.tiled.tokens64.outputs128", [type] {
            run_dense_matmul_cpu_reference_case(
                type, "loom_libs:ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32", 64, 128, 2048);
        });
    }
    suite.device_case("dense_matmul.iq3_xxs.codebook.vector.tokens1.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ3_XXS,
                                            "loom_libs:ggml_mul_mat_vector_iq3_xxs_f32_f32", 1, 128, 2048);
    });
    suite.device_case("dense_matmul.iq3_xxs.codebook.tiled.tokens64.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ3_XXS,
                                            "loom_libs:ggml_mul_mat_tiled_input_f32_iq3_xxs_publish_f32", 64,
                                            128, 2048);
    });
    for (const ggml_type type : { GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S }) {
        const char * vector_kernel = type == GGML_TYPE_IQ2_S ?
                                         "loom_libs:ggml_mul_mat_vector_iq2_s_f32_f32" :
                                         "loom_libs:ggml_mul_mat_vector_iq3_s_f32_f32";
        const char * tiled_kernel = type == GGML_TYPE_IQ2_S ?
                                        "loom_libs:ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32" :
                                        "loom_libs:ggml_mul_mat_tiled_input_f32_iq3_s_publish_f32";
        suite.device_case("dense_matmul." + type_name(type) + ".codebook_control.vector.tokens1.outputs128",
                          [type, vector_kernel] {
                              run_dense_matmul_cpu_reference_case(type, vector_kernel, 1, 128, 2048);
                          });
        suite.device_case("dense_matmul." + type_name(type) + ".codebook_control.tiled.tokens64.outputs128",
                          [type, tiled_kernel] {
                              run_dense_matmul_cpu_reference_case(type, tiled_kernel, 64, 128, 2048);
                          });
    }
    suite.device_case("dense_matmul.iq4_xs.vector.tokens1.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ4_XS, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, 128);
    });
    suite.device_case("dense_matmul.iq4_xs.tiled.tokens64.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ4_XS,
                                            "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned", 64, 128);
    });
    suite.device_case("dense_matmul.iq4_xs.prefill.tokens256.outputs128", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ4_XS,
                                            "loom_libs:ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256", 256, 128,
                                            kQwenHiddenSize, true);
    });
    // Exercise the production K dimension and a partial output tile.
    suite.device_case("dense_matmul.iq4_xs.prefill.tokens512.outputs64", [] {
        run_dense_matmul_cpu_reference_case(
            GGML_TYPE_IQ4_XS, "loom_libs:ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256", 512, 64, 3072, true);
    });
    for (const ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        for (const ggml_type cache_type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            suite.device_case(
                "attention_v_matmul." + type_name(type) + ".cache_" + type_name(cache_type), [type, cache_type] {
                    run_dense_matmul_cpu_reference_case(type,
                                                        "loom_libs:llm_attention_v_matmul_set_rows_vector_f32_f32", 1,
                                                        1024, 2048, false, false, cache_type);
                });
        }
    }
    suite.device_case("attention_v_matmul.q6_k.prefill_f16.repeated_indices",
                      [] { run_prefill_value_cache_repeated_indices_case(); });
    for (const ggml_type type :
         { GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ2_S,
           GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        const char *  vector_kernel = type == GGML_TYPE_Q6_K ? "loom_libs:ggml_mul_mat_vector_q6_f32_f32" :
                                                               "loom_libs:ggml_mul_mat_vector_f32_f32";
        const int64_t outputs       = type == GGML_TYPE_IQ2_S ? 640 : (type == GGML_TYPE_IQ4_NL ? 1024 : 128);
        const int64_t input         = type == GGML_TYPE_IQ4_NL ? 640 : kQwenHiddenSize;
        const char *  route         = type == GGML_TYPE_IQ4_NL ? "tiled" : "skinny";
        const char *  kernel        = type == GGML_TYPE_IQ4_NL ? "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32" :
                                                                 "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32";
        suite.device_case(
            "dense_matmul." + type_name(type) + "." + route + ".tokens2.outputs" + std::to_string(outputs),
            [type, outputs, input, kernel] { run_dense_matmul_cpu_reference_case(type, kernel, 2, outputs, input); });
        suite.device_case("dense_matmul." + type_name(type) + ".vector.tokens1.outputs" + std::to_string(outputs),
                          [type, vector_kernel, outputs, input] {
                              run_dense_matmul_cpu_reference_case(type, vector_kernel, 1, outputs, input);
                          });
    }
    suite.device_case("dense_matmul.iq4_nl.tiled.tokens64.outputs1024", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_IQ4_NL,
                                            "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned", 64, 1024,
                                            640);
    });
    suite.device_case("dense_matmul.q5_0.tiled.tokens2.outputs256.input640", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q5_0, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 2,
                                            256, 640);
    });
    suite.device_case("dense_matmul.q5_0.tiled.tokens2.outputs128.input544", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q5_0, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 2,
                                            128, 544);
    });
    suite.device_case("dense_matmul.q5_0.vector.tokens1.outputs256.input640", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q5_0, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, 256, 640);
    });
    suite.device_case("dense_matmul.q8_1.zero_weight.skinny", [] {
        run_dense_matmul_zero_weight_case(GGML_TYPE_Q8_1, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32");
    });
    suite.device_case("dense_matmul.q8_1.zero_weight.vector", [] {
        run_dense_matmul_zero_weight_case(GGML_TYPE_Q8_1, "loom_libs:ggml_mul_mat_vector_f32_f32", 1);
    });
    suite.device_case("dense_matmul.f32.skinny.tokens2.outputs256", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 2,
                                            256);
    });
    suite.device_case("dense_matmul.f32.vector.tokens1.outputs256", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_vector_f32_f32", 1, 256);
    });
    suite.device_case("dense_matmul_unary.f32.tokens2.outputs128", [] {
        run_dense_matmul_unary_cpu_reference_case(GGML_TYPE_F32, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                                                  2, 128);
    });
    suite.device_case("dense_matmul_swiglu.q4_k.f16.swiglu.tokens2.outputs128", [] {
        run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_F16,
                                                   "loom_libs:ggml_mul_mat_swiglu_f32_f32_lowtoken_dot", 2, 128);
    });
    suite.device_case("dense_matmul_swiglu.q4_k.f16.geglu.tokens2.outputs128", [] {
        run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_F16,
                                                   "loom_libs:ggml_mul_mat_swiglu_f32_f32_lowtoken_dot", 2, 128,
                                                   GGML_GLU_OP_GEGLU);
    });
    suite.device_case("dense_matmul_swiglu.iq4_nl.iq4_nl.geglu.tokens2.outputs2048.input640", [] {
        run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL,
                                                   "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32", 2,
                                                   2048, GGML_GLU_OP_GEGLU, 640);
    });
    suite.device_case("dense_matmul_swiglu.iq4_nl.iq4_nl.geglu.tokens64.outputs128.input640", [] {
        run_dense_matmul_swiglu_cpu_reference_case(
            GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL,
            "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32_aligned",
            64, 128, GGML_GLU_OP_GEGLU, 640);
    });
    for (const ggml_glu_op op : { GGML_GLU_OP_REGLU, GGML_GLU_OP_GEGLU_ERF, GGML_GLU_OP_GEGLU_QUICK }) {
        suite.device_case(std::string("dense_matmul_swiglu.q4_k.f16.") + glu_op_name(op) + ".tokens2.outputs128", [op] {
            run_dense_matmul_swiglu_cpu_reference_case(
                GGML_TYPE_Q4_K, GGML_TYPE_F16, "loom_libs:ggml_mul_mat_swiglu_f32_f32_lowtoken_dot", 2, 128, op);
        });
    }
    suite.device_case("dense_matmul_packed_glu.q4_k.tokens2.outputs128", [] {
        run_dense_matmul_packed_glu_cpu_reference_case(
            GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32", 2, 128);
    });
    suite.device_case("dense_matmul_packed_glu.q4_k.swiglu.tokens2.outputs128.gated", [] {
        run_dense_matmul_packed_glu_cpu_reference_case(GGML_TYPE_Q4_K,
                                                       "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32",
                                                       2, 128, GGML_GLU_OP_SWIGLU, true);
    });
    suite.device_case("dense_matmul_binary.q4_k.f16.tokens2.outputs128", [] {
        run_dense_matmul_binary_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_F16,
                                                   "loom_libs:ggml_mul_mat_swiglu_f32_f32_lowtoken_dot", 2, 128);
    });
    suite.device_case("vector_q8_publish.cpu_reference", [] { run_vector_q8_publish_cpu_reference_case(); });
    for (const ggml_glu_op op :
         { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_REGLU, GGML_GLU_OP_GEGLU_ERF, GGML_GLU_OP_GEGLU_QUICK }) {
        for (const int64_t outputs : { 128, 4096 }) {
            suite.device_case(std::string("dense_matmul_swiglu.q4_k.q4_k.lowtoken.") + glu_op_name(op) +
                                  ".tokens4.outputs" + std::to_string(outputs),
                              [op, outputs] {
                                  run_dense_matmul_swiglu_cpu_reference_case(
                                      GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                      "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot", 4, outputs, op);
                              });
        }
        for (const int64_t tokens : { 256, 512 }) {
            if (tokens == 512) {
                suite.device_case(
                    std::string("dense_matmul_swiglu.q4_k.q4_k.prefill.") + glu_op_name(op) + ".tokens512.outputs4096",
                    [op] {
                        run_dense_matmul_swiglu_cpu_reference_case(
                            GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                            "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32", 512, 4096, op);
                    });
            } else {
                for (const int64_t outputs : { 128, 4096 }) {
                    suite.device_case(std::string("dense_matmul_swiglu.q4_k.q4_k.prefill.") + glu_op_name(op) +
                                          ".tokens256.outputs" + std::to_string(outputs),
                                      [op, outputs] {
                                          run_dense_matmul_swiglu_cpu_reference_case(
                                              GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                              "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32", 256,
                                              outputs, op);
                                      });
                }
            }
        }
    }
    for (const int64_t tokens : { 1, 2, 3, 5 }) {
        suite.device_case(
            "dense_matmul_swiglu.q4_k.q4_k.lowtoken.tokens" + std::to_string(tokens) + ".outputs4160", [tokens] {
                run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                                           "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot",
                                                           tokens, 4160);
            });
    }
    for (const int64_t tokens : { 1, 2, 5 }) {
        suite.device_case(
            "dense_matmul_swiglu.q4_k.q4_k.swiglu.tokens" + std::to_string(tokens) + ".outputs16384.input4096",
            [tokens] {
                run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                                           "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot",
                                                           tokens, 16384, GGML_GLU_OP_SWIGLU, 4096);
            });
    }
    suite.device_case("dense_matmul_swiglu.q4_k.q4_k.reglu.tokens3.outputs16384.input4352", [] {
        run_dense_matmul_swiglu_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                                   "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot", 3, 16384,
                                                   GGML_GLU_OP_REGLU, 4352);
    });
    suite.device_case("dense_matmul_binary.q4_k.q4_k.lowtoken.tokens4.outputs128", [] {
        run_dense_matmul_binary_cpu_reference_case(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                                                   "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot", 4, 128);
    });
    suite.device_case("dense_matmul_binary.q4_k.q4_k.prefill.tokens256.outputs128", [] {
        run_dense_matmul_binary_cpu_reference_case(
            GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32", 256, 128);
    });
    suite.device_case("dense_matmul_binary.q4_k.q4_k.prefill.tokens256.outputs4096", [] {
        run_dense_matmul_binary_cpu_reference_case(
            GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32", 256, 4096);
    });
    suite.device_case("dense_matmul_binary.q4_k.q4_k.prefill.tokens1024.outputs2048", [] {
        run_dense_matmul_binary_cpu_reference_case(
            GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32", 1024, 2048);
    });
    suite.device_case("dense_matmul_postops.f16.bias.tokens33.outputs256", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16,
                                                    "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
                                                    33, 256, true, false, false);
    });
    suite.device_case("dense_matmul_postops.f16.residual.tokens33.outputs256", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16,
                                                    "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
                                                    33, 256, false, true, false);
    });
    suite.device_case("dense_matmul_postops.f16.bias_residual.tokens33.outputs256", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16,
                                                    "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
                                                    33, 256, true, true, false);
    });
    suite.device_case("dense_matmul_postops.f16.residual.alias.tokens33.outputs256", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16,
                                                    "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
                                                    33, 256, false, true, true);
    });
    suite.device_case("dense_matmul_postops.f16.bias_residual.alias.tokens33.outputs256", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_F16,
                                                    "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32",
                                                    33, 256, true, true, true);
    });
    for (const ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        const int64_t input_size = type == GGML_TYPE_Q4_K ? 20480 : 8192;
        suite.device_case(
            "dense_matmul." + type_name(type) + ".skinny.tokens5.outputs4096.input" + std::to_string(input_size),
            [type, input_size] {
                run_dense_matmul_cpu_reference_case(type, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 5,
                                                    4096, input_size);
            });
        suite.device_case("dense_matmul_postops." + type_name(type) + ".add.tokens5.outputs4160", [type, input_size] {
            run_dense_matmul_postops_cpu_reference_case(
                type, "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32", 5, 4160, false, true, false,
                input_size + 256);
        });
        suite.device_case("dense_matmul_postops." + type_name(type) + ".add.alias.tokens5.outputs4224",
                          [type, input_size] {
                run_dense_matmul_postops_cpu_reference_case(
                                  type, "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32", 5, 4224,
                                  false, true, true, input_size + 256);
            });
    }
    suite.device_case("dense_matmul.q4_k.skinny.tokens5.outputs4096.input8192", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 5,
                                            4096, 8192);
    });
    for (const int64_t input_size : { 6144, 6400, 6656 }) {
        suite.device_case(
            "dense_matmul.q4_k.skinny.tokens5.outputs5120.input" + std::to_string(input_size), [input_size] {
                              run_dense_matmul_cpu_reference_case(
                    GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 5, 5120, input_size);
                          });
        suite.device_case("dense_matmul_postops.q4_k.add.alias.tokens5.outputs5120.input" + std::to_string(input_size),
                          [input_size] {
                run_dense_matmul_postops_cpu_reference_case(
                                  GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
                                  5, 5120, false, true, true, input_size);
            });
    }
    suite.device_case("dense_matmul.q4_k.skinny.tokens4.outputs4096.input4096", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4,
                                            4096, 4096);
    });
    suite.device_case("dense_matmul_postops.q4_k.add.tokens4.outputs4160.input6400", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_Q4_K,
                                                    "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
                                                    4, 4160, false, true, false, 6400);
    });
    suite.device_case("dense_matmul_postops.q4_k.add.alias.tokens4.outputs4224.input6656", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_Q4_K,
                                                    "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
                                                    4, 4224, false, true, true, 6656);
    });
    suite.device_case("endpoint_rmsnorm_q6k_q8.cpu_reference",
                      [] { run_endpoint_rmsnorm_q6k_q8_cpu_reference_case(); });
    suite.device_case("dense_matmul.q6_k.skinny.tokens5.outputs4096.input16384", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 5,
                                            4096, 16384);
    });
    suite.device_case("dense_matmul_postops.q6_k.add.tokens5.outputs4160.input16640", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_Q6_K,
                                                    "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
                                                    5, 4160, false, true, false, 16640);
    });
    suite.device_case("dense_matmul_postops.q6_k.add.alias.tokens5.outputs4224.input16640", [] {
        run_dense_matmul_postops_cpu_reference_case(GGML_TYPE_Q6_K,
                                                    "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32",
                                                    5, 4224, false, true, true, 16640);
    });
    for (const int64_t tokens : { 1, 3, 5 }) {
        suite.device_case(
            "dense_matmul.q4_k.contiguous.tokens" + std::to_string(tokens) + ".outputs256.input5120", [tokens] {
                run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K,
                    tokens == 1 ? "loom_libs:ggml_mul_mat_vector_f32_f32" :
                                  "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                    tokens, 256, 5120, true);
            });
        suite.device_case(
            "dense_matmul.q6_k.contiguous.tokens" + std::to_string(tokens) + ".outputs256.input5120", [tokens] {
                run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K,
                    tokens == 1 ? "loom_libs:ggml_mul_mat_vector_f32_f32" :
                                  "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                    tokens, 256, 5120, true, true);
            });
    }
    suite.device_case("dense_matmul.q6_k.vector_q6.tokens1.outputs256.input5120", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K, "loom_libs:ggml_mul_mat_vector_q6_f32_f32", 1, 256, 5120,
                                            true);
    });
    suite.device_case("dense_matmul.q6_k.vector_q6.tokens1.outputs320.input512", [] {
        run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K, "loom_libs:ggml_mul_mat_vector_q6_f32_f32", 1, 320, 512,
                                            true);
    });
}

static void register_rmsnorm_and_scheduling_cases(Suite & suite) {
    suite.host_case("support.lfm_head_rope.availability", [] { run_lfm_head_rope_availability_checks(); });
    suite.host_case("gemma_post_proj.matcher_negatives", [] { run_gemma_post_proj_matcher_negatives(); });
    for (const int64_t token_count : { 1, 3, 64 }) {
        suite.device_case("gemma_post_proj.tokens" + std::to_string(token_count),
                          [token_count] { run_gemma_post_proj_cpu_reference_case(token_count); });
    }
    suite.device_case("rmsnorm.default", [] { run_rmsnorm_cpu_reference_case(); });
    suite.device_case("rmsnorm.hidden3840.tokens18", [] { run_rmsnorm_cpu_reference_case(3840, 18, 1.0e-6f); });
    for (const int64_t token_count : { 2, 23 }) {
        suite.device_case("strided_rmsnorm.tokens" + std::to_string(token_count),
                          [token_count] { run_strided_rmsnorm_cpu_reference_case(token_count); });
    }
    suite.device_case("strided_rmsnorm_binary.minicpm_pp512_witness",
                      [] { run_strided_rmsnorm_binary_cpu_reference_case(); });
    suite.device_case("rmsnorm_binary_add.cpu_reference", [] { run_rmsnorm_binary_add_cpu_reference_case(); });
    suite.device_case("rmsnorm_gate.silu.hidden128.tokens1.stride7",
                      [] { run_rmsnorm_gate_cpu_reference_case(128, 1, 7, GGML_UNARY_OP_SILU); });
    suite.device_case("rmsnorm_gate.gelu.hidden256.tokens3.stride3",
                      [] { run_rmsnorm_gate_cpu_reference_case(256, 3, 3, GGML_UNARY_OP_GELU); });
    suite.device_case("rmsnorm_gate.relu.hidden512.tokens3.stride5",
                      [] { run_rmsnorm_gate_cpu_reference_case(512, 3, 5, GGML_UNARY_OP_RELU); });
    suite.device_case("rmsnorm_gate.gelu_erf.hidden1024.tokens1.stride4",
                      [] { run_rmsnorm_gate_cpu_reference_case(1024, 1, 4, GGML_UNARY_OP_GELU_ERF); });
    suite.device_case("rmsnorm_gate.silu.hidden128.tokens48.stride512",
                      [] { run_rmsnorm_gate_cpu_reference_case(128, 48, 512, GGML_UNARY_OP_SILU); });
    suite.device_case("rmsnorm_mul.hidden3840.tokens18", [] { run_rmsnorm_mul_cpu_reference_case(3840, 18, 1.0e-6f); });
    for (const int64_t hidden_size : { 1024, 2048, 2560, 3072, 4096, 5120 }) {
        suite.device_case("rmsnorm_mul.lowtoken.random.hidden" + std::to_string(hidden_size) + ".tokens1",
                          [hidden_size] { run_rmsnorm_mul_cpu_reference_case(hidden_size, 1, 1.0e-6f); });
        suite.device_case("rmsnorm_mul.lowtoken.adversarial.hidden" + std::to_string(hidden_size) + ".tokens1",
                          [hidden_size] { run_rmsnorm_mul_cpu_reference_case(hidden_size, 1, 1.0e-5f, true); });
    }
    for (const int64_t hidden_size : { 2048, 5120 }) {
        for (const int64_t token_count : { 2, 4 }) {
            suite.device_case("rmsnorm_mul.lowtoken.random.hidden" + std::to_string(hidden_size) + ".tokens" +
                                  std::to_string(token_count),
                              [hidden_size, token_count] {
                                  run_rmsnorm_mul_cpu_reference_case(hidden_size, token_count, 1.0e-6f);
                              });
        }
    }
    suite.device_case("gemma_scaled_rmsnorm.cpu_reference", [] { run_gemma_scaled_rmsnorm_cpu_reference_case(); });
    suite.device_case("gemma_scaled_rmsnorm_mul.cpu_reference",
                      [] { run_gemma_scaled_rmsnorm_mul_cpu_reference_case(); });
    suite.device_case("scheduled.scale_rmsnorm", [] { run_scheduled_hrx_scale_rmsnorm_case(); });
    suite.device_case("scheduled.rmsnorm_scale", [] { run_scheduled_hrx_rmsnorm_scale_case(); });
    suite.device_case("scheduled.rmsnorm_view_scale", [] { run_scheduled_hrx_rmsnorm_view_scale_case(); });
    suite.device_case("external_view_input_rmsnorm", [] { run_external_view_input_rmsnorm_case(); });
    suite.device_case("split_local_view_alias_import", [] { run_split_local_view_alias_import_case(); });
    suite.device_case("rope_set_rows.cpu_reference", [] { run_rope_set_rows_cpu_reference_case(); });
    for (const int64_t head_count : { 8, 32 }) {
        for (const int64_t token_count : { 1, 64 }) {
            suite.device_case(
                "lfm_head_rope.query.heads" + std::to_string(head_count) + ".tokens" + std::to_string(token_count),
                [head_count, token_count] { run_lfm_head_rope_cpu_reference_case(head_count, token_count, false); });
        }
    }
    suite.device_case("lfm_head_rope.query.heads7.tokens3", [] { run_lfm_head_rope_cpu_reference_case(7, 3, false); });
    for (const int64_t token_count : { 1, 64 }) {
        suite.device_case("lfm_head_rope.cache.heads8.tokens" + std::to_string(token_count),
                          [token_count] { run_lfm_head_rope_cpu_reference_case(8, token_count, true); });
    }
    suite.device_case("attention_postprocess.cpu_reference", [] { run_attention_postprocess_cpu_reference_case(); });
    suite.device_case("routed_moe.q4_k.next_rmsnorm", [] { run_routed_moe_cpu_reference_case(GGML_TYPE_Q4_K, true); });
    suite.device_case("flash_attention.asymmetric", [] { run_asymmetric_flash_attention_cpu_reference_case(); });
    suite.device_case("routed_moe.q6_k", [] { run_routed_moe_cpu_reference_case(GGML_TYPE_Q6_K, false); });
    suite.device_case("decode_attention_qkv.scheduling", [] { run_decode_attention_qkv_scheduling_case(); });
    suite.device_case("decode_routed_moe.q4_k", [] { run_decode_routed_moe_scheduling_case(GGML_TYPE_Q4_K); });
    suite.device_case("decode_routed_moe.q6_k", [] { run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K); });
    suite.device_case("decode_routed_moe.q6_k.alias_gate_input",
                      [] { run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K, true); });
    suite.device_case("rmsnorm_mul.hidden256.tokens1", [] { run_rmsnorm_mul_case(256, 1); });
    suite.device_case("rmsnorm_mul.hidden256.tokens4", [] { run_rmsnorm_mul_case(256, 4); });
    suite.device_case("rmsnorm_mul.hidden2048.tokens1", [] { run_rmsnorm_mul_case(2048, 1); });
    suite.device_case("router_projection.tokens4", [] { run_router_projection_case(4); });
    suite.device_case("router_top8.tokens4", [] { run_router_top8_case(4); });
    suite.device_case("qwen_flash_attention", [] { run_qwen_flash_attention_case(); });
}

static void register_flash_attention_cases(Suite & suite) {
    suite.device_case("common_flash_attention.prefill.head128.query256.kv256.exact", [] {
        run_common_flash_attention_cpu_reference_case(128, 256, 256, 1.0f / std::sqrt(128.0f),
                                                      "loom_libs:ggml_flash_attention_f32_f16_wmma");
    });
    for (int64_t head_size = 64; head_size <= 512; head_size += 64) {
        suite.device_case("common_flash_attention.decode.head" + std::to_string(head_size) + ".query2.kv65",
                          [head_size] {
                              run_common_flash_attention_cpu_reference_case(
                                  head_size, 2, 65, 1.0f / std::sqrt(static_cast<float>(head_size)),
                                  "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
                          });
        suite.device_case(
            "common_flash_attention.prefill.head" + std::to_string(head_size) + ".query16.kv64", [head_size] {
                run_common_flash_attention_cpu_reference_case(head_size, 16, 64,
                                                              1.0f / std::sqrt(static_cast<float>(head_size)),
                                                              "loom_libs:ggml_flash_attention_f32_f16_wmma");
            });
        suite.device_case(
            "common_flash_attention.prefill.head" + std::to_string(head_size) + ".query17.kv65", [head_size] {
                run_common_flash_attention_cpu_reference_case(head_size, 17, 65,
                                                              1.0f / std::sqrt(static_cast<float>(head_size)),
                                                              "loom_libs:ggml_flash_attention_f32_f16_wmma");
            });
    }
    suite.device_case("common_flash_attention.decode.head96.query2.kv4.capacity320", [] {
        run_common_flash_attention_cpu_reference_case(
            96, 2, 4, 1.0f / std::sqrt(96.0f), "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8", 320);
    });
    suite.device_case("common_flash_attention.decode.head320.query2.kv64.capacity64", [] {
        run_common_flash_attention_cpu_reference_case(
            320, 2, 64, 1.0f / std::sqrt(320.0f), "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8",
            64);
    });
    suite.device_case("qwen_decode_split_flash_attention.query1.kv512",
                      [] { run_qwen_decode_split_flash_attention_scheduling_case(1, 512); });
    suite.device_case("qwen_decode_split_flash_attention.query4.kv513",
                      [] { run_qwen_decode_split_flash_attention_scheduling_case(4, 513); });
    suite.device_case("qwen_decode_split_flash_attention.query1.kv512.masked",
                      [] { run_qwen_decode_split_flash_attention_scheduling_case(1, 512, true); });
    suite.device_case("qwen_decode_split_flash_attention.query1.kv512.masked.head96.value64", [] {
        run_qwen_decode_split_flash_attention_scheduling_case(1, 512, true, 96, 64, 1.0f / std::sqrt(96.0f));
    });
    suite.device_case("qwen_decode_split_flash_attention.query4.kv513.masked.head96.value64", [] {
        run_qwen_decode_split_flash_attention_scheduling_case(4, 513, true, 96, 64, 1.0f / std::sqrt(96.0f));
    });
    suite.device_case("qwen_decode_attention_output_next_q8.no_selectors",
                      [] { run_qwen_decode_attention_output_next_q8_scheduling_case(false); });
    suite.device_case("qwen_decode_attention_output_next_q8.selectors",
                      [] { run_qwen_decode_attention_output_next_q8_scheduling_case(true); });
    suite.device_case("qwen_full_cache_prefill_flash_attention.tokens512",
                      [] { run_qwen_full_cache_prefill_flash_attention_scheduling_case(512); });
}

static void register_hrx_ops_cases(Suite & suite) {
    register_basic_ops_cases(suite);
    register_dense_matmul_cases(suite);
    register_rmsnorm_and_scheduling_cases(suite);
    register_flash_attention_cases(suite);
}

int main(int argc, char ** argv) {
    Suite suite(test_runner::Config::with_prefix("HRX test", "hrx", "GGML_HRX_TEST"));
    register_hrx_ops_cases(suite);

    test_runner::Options options;
    if (!suite.parse_options(argc, argv, options)) {
        suite.print_usage(argc > 0 ? argv[0] : "test-hrx-ops");
        return 2;
    }
    if (options.help) {
        return suite.run(options, false);
    }

    const bool run_vector_cap_experiment = std::getenv("GGML_HRX_RUN_VECTOR_CAP_EXPERIMENT") != nullptr;
    const bool run_vector_publish_test   = std::getenv("GGML_HRX_RUN_VECTOR_PUBLISH_TEST") != nullptr;
    const bool run_vector_postops_test   = std::getenv("GGML_HRX_RUN_VECTOR_POSTOPS_TEST") != nullptr;

    const bool has_device = ggml_backend_hrx_get_device_count() != 0;
    if (!options.runner_control && run_vector_cap_experiment) {
        run_vector_output_capacity_experiment();
        return 0;
    }
    if (!options.runner_control && run_vector_publish_test) {
        run_vector_q8_publish_cpu_reference_case();
        return 0;
    }
    if (!options.runner_control && run_vector_postops_test) {
        run_vector_postops_cpu_reference_case();
        return 0;
    }

    return suite.run(options, has_device);
}
