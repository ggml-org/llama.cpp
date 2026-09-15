#include "backend-context.h"
#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hrx.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/graph-executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static constexpr float   kQwenRmsNormEps        = 0.000001f;
static constexpr int64_t kQwenFlashHeadSize     = 128;
static constexpr int64_t kQwenRouterExpertCount = 128;
static constexpr int64_t kQwenRouterRouteCount  = 8;
static constexpr int64_t kQwenHiddenSize        = 2048;
static constexpr int64_t kQwenMoeIntermediate   = 768;
static constexpr int64_t kQwenVocabularyCount   = 151936;

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

static std::vector<float> make_flash_query(int64_t token_count) {
    std::vector<float> data(kQwenFlashHeadSize * token_count);
    for (int64_t i = 0; i < static_cast<int64_t>(data.size()); ++i) {
        data[i] = static_cast<float>((i % 37) - 18) * 0.01f;
    }
    return data;
}

static std::vector<ggml_fp16_t> make_flash_key_value(int64_t token_count, int offset) {
    std::vector<ggml_fp16_t> data(kQwenFlashHeadSize * token_count);
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
                                                    int64_t                          key_value_token_count) {
    std::vector<float> output(query_token_count * kQwenFlashHeadSize);
    const float        scale = 1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize));
    for (int64_t query_token = 0; query_token < query_token_count; ++query_token) {
        std::vector<float> scores(key_value_token_count);
        float              max_score = -std::numeric_limits<float>::infinity();
        for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
            float dot = 0.0f;
            for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
                dot += query[query_token * kQwenFlashHeadSize + channel] *
                       ggml_fp16_to_fp32(key[key_token * kQwenFlashHeadSize + channel]);
            }
            const float score = dot * scale + ggml_fp16_to_fp32(mask[query_token * key_value_token_count + key_token]);
            scores[key_token] = score;
            max_score         = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (int64_t channel = 0; channel < kQwenFlashHeadSize; ++channel) {
            float weighted_sum = 0.0f;
            for (int64_t key_token = 0; key_token < key_value_token_count; ++key_token) {
                const float probability = scores[key_token] / sum;
                weighted_sum += probability * ggml_fp16_to_fp32(value[key_token * kQwenFlashHeadSize + channel]);
            }
            output[query_token * kQwenFlashHeadSize + channel] = weighted_sum;
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
                                                      ggml_tensor *  mask) {
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask,
                                               1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize)), 0.0f, 0.0f);
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
        graph, &graph.nodes()[root_index], root_index, covered_nodes, plan, next_plan_value(graph, plan),
    };
    ggml::hrx::DispatchMatchDiagnostics diagnostics;
    if (!registry.match(context, match, &diagnostics)) {
        std::fprintf(stderr, "manual matcher failed for node %zu %s\n", root_index,
                     ggml_op_name(graph.nodes()[root_index].op));
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
    constexpr int64_t element_count = 2048;
    const size_t      full_bytes    = ggml_row_size(GGML_TYPE_F32, element_count);
    const size_t      q8_bytes      = ggml_row_size(GGML_TYPE_Q8_1, element_count);

    ggml::hrx::Graph       graph;
    ggml::hrx::Status      status;
    ggml::hrx::CommandPlan plan;

    const ggml::hrx::ValueId        root(0);
    const ggml::hrx::ValueId        full_alias(1);
    const ggml::hrx::ValueId        partial_alias(2);
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

    REQUIRE(plan.metadata.append_alternate_value({ root, q8_alternate, GGML_TYPE_Q8_1, q8_bytes, "q8" }, status));

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
}

static std::vector<float> make_pattern_f32(size_t element_count, int seed, float scale = 0.01f) {
    std::vector<float> data(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        const int value = static_cast<int>((i * 17 + static_cast<size_t>(seed) * 29) % 97) - 48;
        data[i]         = static_cast<float>(value) * scale;
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
    REQUIRE(traits->from_float_ref != nullptr);
    const size_t         row_size = ggml_row_size(type, row_length);
    std::vector<uint8_t> data(static_cast<size_t>(row_count) * row_size);
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
        if (diff > allowed) {
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

static QwenFlashAttentionLayoutGraph build_qwen_flash_attention_layout_graph(ggml_context * ctx,
                                                                             int64_t        query_token_count,
                                                                             int64_t        key_value_token_count) {
    QwenFlashAttentionLayoutGraph graph;
    constexpr int64_t             query_head_count     = 32;
    constexpr int64_t             key_value_head_count = 4;

    ggml_tensor * query_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize, query_head_count, query_token_count);
    ggml_tensor * key_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, key_value_head_count, key_value_token_count);
    ggml_tensor * value_storage =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kQwenFlashHeadSize, key_value_head_count, key_value_token_count);
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

    graph.output = build_qwen_flash_attention_graph(ctx, graph.query, graph.key, graph.value, graph.mask);
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
    REQUIRE(!wrong_eps_support.supported);

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

    require_kernel_subsequence(scheduled_kernel_sequence(graph), { "qwen3_moe:qwen3_moe_rmsnorm_f32" });

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
                               { "qwen3_moe:qwen3_moe_flash_attention_f32_f16_wmma" });

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
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_flash_attention_f32_f16_wmma") {
            flash_dispatch = &dispatch;
        }
    }
    REQUIRE(flash_dispatch != nullptr);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("query_token_count") == token_count);
    REQUIRE(flash_dispatch->kernel.integer_parameters.at("key_value_token_count") == token_count);
    REQUIRE(flash_dispatch->bindings.size() == 5);
    REQUIRE(flash_dispatch->bindings[3].value == compact_mask->alternate_value);
    REQUIRE(flash_dispatch->bindings[3].length == active_mask_byte_count);

    ggml_free(ctx);
}

static void run_qwen_decode_split_flash_attention_scheduling_case(int64_t query_token_count,
                                                                  int64_t key_value_token_count) {
    constexpr int64_t query_head_count     = 32;
    constexpr int64_t key_value_head_count = 4;
    constexpr int64_t hidden_size          = query_head_count * kQwenFlashHeadSize;
    const size_t      q8_row_bytes         = ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
    const size_t      q8_output_bytes      = static_cast<size_t>(query_token_count) * q8_row_bytes;
    const int64_t     key_value_capacity   = (key_value_token_count + 63) / 64 * 64;
    const int64_t     key_value_blocks     = key_value_capacity / 64;
    const size_t      partial_scalar_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16) * sizeof(float);
    const size_t partial_output_bytes =
        static_cast<size_t>(key_value_head_count * key_value_blocks * 16 * kQwenFlashHeadSize) * sizeof(ggml_fp16_t);

    ggml_init_params params = {};
    params.mem_size         = 128 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    QwenFlashAttentionLayoutGraph attention =
        build_qwen_flash_attention_layout_graph(ctx, query_token_count, key_value_token_count);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, attention.output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == static_cast<size_t>(query_token_count));
    REQUIRE(plan.transients.size() == 4);
    REQUIRE(plan.completion_counter_requests.size() == 1);

    REQUIRE(plan.transients[0].size == partial_scalar_bytes);
    REQUIRE(plan.transients[1].size == partial_scalar_bytes);
    REQUIRE(plan.transients[2].size == partial_output_bytes);
    REQUIRE(plan.transients[3].size == q8_output_bytes);
    REQUIRE(plan.completion_counter_requests[0].count == key_value_head_count);

    const ggml::hrx::Value * query_value  = imported.graph.values().find_tensor(attention.query);
    const ggml::hrx::Value * mask_value   = imported.graph.values().find_tensor(attention.mask);
    const ggml::hrx::Value * output_value = imported.graph.values().find_tensor(attention.output);
    REQUIRE(query_value != nullptr);
    REQUIRE(mask_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * q8_alternate =
        plan.metadata.find_alternate_value(output_value->id, GGML_TYPE_Q8_1, q8_output_bytes);
    REQUIRE(q8_alternate != nullptr);
    REQUIRE(q8_alternate->alternate_value == plan.transients[3].value);

    const size_t query_row_bytes  = static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = query_row_bytes;
    for (int64_t row = 0; row < query_token_count; ++row) {
        const ggml::hrx::Dispatch & dispatch = plan.dispatches[static_cast<size_t>(row)];
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8");
        REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
        REQUIRE(dispatch.kernel.compile_parameters.at("qwen3_moe.attention.key_value_token_capacity") ==
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
    REQUIRE(attention_output != nullptr);
    REQUIRE(output_weight != nullptr);
    REQUIRE(projection != nullptr);
    REQUIRE(residual_input != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(norm_weight != nullptr);
    REQUIRE(rms != nullptr);
    REQUIRE(normalized != nullptr);

    ggml_cgraph * cgraph = ggml_new_graph(ctx);
    REQUIRE(cgraph != nullptr);
    ggml_build_forward_expand(cgraph, normalized);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 2);
    REQUIRE(plan.completion_counter_requests.size() == 2);

    const ggml::hrx::Dispatch & projection_dispatch = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(projection_dispatch.kernel.kernel_id) ==
            "qwen3_moe:qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8");
    REQUIRE(projection_dispatch.kernel.integer_parameters.at("token_count") == query_token_count);
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.input_size") ==
            std::to_string(attention_hidden_size));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_size") ==
            std::to_string(kQwenHiddenSize));
    REQUIRE(projection_dispatch.kernel.compile_parameters.at("qwen3_moe.dense_quantized.output_accumulation") == "1");
    REQUIRE(projection_dispatch.bindings.size() == 7);

    const ggml::hrx::Value * flash_output_value   = imported.graph.values().find_tensor(attention.output);
    const ggml::hrx::Value * residual_input_value = imported.graph.values().find_tensor(residual_input);
    const ggml::hrx::Value * residual_value       = imported.graph.values().find_tensor(residual);
    const ggml::hrx::Value * normalized_value     = imported.graph.values().find_tensor(normalized);
    REQUIRE(flash_output_value != nullptr);
    REQUIRE(residual_input_value != nullptr);
    REQUIRE(residual_value != nullptr);
    REQUIRE(normalized_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * attention_q8 =
        ggml::hrx::find_alternate_value(plan, flash_output_value->id, GGML_TYPE_Q8_1, attention_q8_bytes);
    const ggml::hrx::CommandPlanAlternateValue * next_q8 =
        ggml::hrx::find_alternate_value(plan, normalized_value->id, GGML_TYPE_Q8_1, next_q8_bytes);
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

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:ggml_add_f32" });

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

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:ggml_gather_add_f32" });

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

static void run_token_embedding_q4k_cpu_reference_case() {
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

    constexpr int64_t vocabulary_count = 64;
    constexpr int64_t hidden_size      = kQwenHiddenSize;
    constexpr int64_t token_count      = 7;
    ggml_tensor *     cpu_weight       = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_Q4_K, hidden_size, vocabulary_count);
    ggml_tensor *     cpu_ids          = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor *     cpu_output       = ggml_get_rows(cpu_ctx, cpu_weight, cpu_ids);
    ggml_tensor *     hrx_weight       = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_Q4_K, hidden_size, vocabulary_count);
    ggml_tensor *     hrx_ids          = ggml_new_tensor_1d(hrx_ctx, GGML_TYPE_I32, token_count);
    ggml_tensor *     hrx_output       = ggml_get_rows(hrx_ctx, hrx_weight, hrx_ids);
    REQUIRE(cpu_output != nullptr);
    REQUIRE(hrx_output != nullptr);

    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_cgraph * hrx_graph = ggml_new_graph(hrx_ctx);
    REQUIRE(cpu_graph != nullptr);
    REQUIRE(hrx_graph != nullptr);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    ggml_build_forward_expand(hrx_graph, hrx_output);

    require_kernel_subsequence(scheduled_kernel_sequence(hrx_graph), { "qwen3_moe:qwen_token_embedding_q4k" });

    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    ggml_backend_buffer_t hrx_buffer = ggml_backend_alloc_ctx_tensors(hrx_ctx, hrx_backend);
    REQUIRE(cpu_buffer != nullptr);
    REQUIRE(hrx_buffer != nullptr);

    const std::vector<uint8_t> weight = make_quantized_rows(GGML_TYPE_Q4_K, hidden_size, vocabulary_count, 5);
    const std::vector<int32_t> ids    = { 3, 17, 29, 41, 53, 7, 19 };
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

static void run_dense_matmul_cpu_reference_case(ggml_type    weight_type,
                                                const char * expected_kernel,
                                                int64_t      token_count,
                                                int64_t      output_size) {
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

    constexpr int64_t input_size = kQwenHiddenSize;
    ggml_tensor *     cpu_weight = ggml_new_tensor_2d(cpu_ctx, weight_type, input_size, output_size);
    ggml_tensor *     cpu_input  = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     cpu_output = ggml_mul_mat(cpu_ctx, cpu_weight, cpu_input);
    ggml_tensor *     hrx_weight = ggml_new_tensor_2d(hrx_ctx, weight_type, input_size, output_size);
    ggml_tensor *     hrx_input  = ggml_new_tensor_2d(hrx_ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor *     hrx_output = ggml_mul_mat(hrx_ctx, hrx_weight, hrx_input);
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

    const std::vector<uint8_t> weight = make_quantized_rows(weight_type, input_size, output_size, 6);
    const std::vector<float>   input  = make_pattern_f32(input_size * token_count, 7, 0.01f);
    set_tensor_pair_bytes(cpu_backend, cpu_weight, hrx_backend, hrx_weight, weight.data(), weight.size());
    set_tensor_pair_bytes(cpu_backend, cpu_input, hrx_backend, hrx_input, input.data(), input.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(cpu_backend, cpu_graph) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_graph_compute(hrx_backend, hrx_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu_backend);
    ggml_backend_synchronize(hrx_backend);
    require_close(get_f32_tensor(hrx_backend, hrx_output), get_f32_tensor(cpu_backend, cpu_output), 3.0e-1f, 2.0e-2f);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(hrx_buffer);
    ggml_free(cpu_ctx);
    ggml_free(hrx_ctx);
    ggml_backend_free(cpu_backend);
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
        "qwen3_moe:qwen3_moe_build_expert_table",
        "qwen3_moe:qwen3_moe_build_expert_partition_table",
        "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_down_q4k_f16_wmma_grouped" :
                                             "qwen3_moe:qwen3_moe_routed_down_q6k_f16_wmma_grouped",
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

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    std::vector<std::string> expected = {
        "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4",
        "qwen3_moe:qwen3_moe_router_projection_top8_fused_decode_f32",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8" :
                                             "qwen3_moe:qwen3_moe_routed_gate_up_swiglu_q4k_q8",
        down_weight_type == GGML_TYPE_Q4_K ? "qwen3_moe:qwen3_moe_routed_down_q4k_q8_1_x4_next_q8" :
                                             "qwen3_moe:qwen3_moe_routed_down_q6k_f32_wave64_next_q8",
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

int main() {
    run_rmsnorm_support_checks();
    run_alternate_value_alias_lookup_checks();

    if (ggml_backend_hrx_get_device_count() == 0) {
        std::fprintf(stderr, "test skipped: no HRX devices available\n");
        return 0;
    }

    run_add_f32_cpu_reference_case();
    run_gather_add_f32_cpu_reference_case();
    run_token_embedding_q4k_cpu_reference_case();
    run_dense_matmul_cpu_reference_case(GGML_TYPE_Q4_K, "qwen3_moe:qwen3_moe_dense_linear_q4k_f16_wmma", 2, 128);
    run_dense_matmul_cpu_reference_case(GGML_TYPE_Q6_K, "qwen3_moe:qwen3_moe_dense_linear_q6k_f16_wmma", 2, 128);
    run_endpoint_rmsnorm_q6k_q8_cpu_reference_case();
    run_attention_postprocess_cpu_reference_case();
    run_routed_moe_cpu_reference_case(GGML_TYPE_Q4_K, true);
    run_routed_moe_cpu_reference_case(GGML_TYPE_Q6_K, false);
    run_decode_attention_qkv_scheduling_case();
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q4_K);
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K);
    run_decode_routed_moe_scheduling_case(GGML_TYPE_Q6_K, true);
    run_rmsnorm_mul_case(256, 1);
    run_rmsnorm_mul_case(256, 4);
    run_rmsnorm_mul_case(2048, 1);
    run_router_projection_case(4);
    run_router_top8_case(4);
    run_qwen_flash_attention_case();
    run_qwen_decode_split_flash_attention_scheduling_case(1, 512);
    run_qwen_decode_split_flash_attention_scheduling_case(4, 513);
    run_qwen_decode_attention_output_next_q8_scheduling_case(false);
    run_qwen_decode_attention_output_next_q8_scheduling_case(true);
    run_qwen_full_cache_prefill_flash_attention_scheduling_case(16);
    run_qwen_full_cache_prefill_flash_attention_scheduling_case(512);
    return 0;
}
