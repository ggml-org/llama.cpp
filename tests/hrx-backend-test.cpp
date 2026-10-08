#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/common/dispatch-gather-add.h"
#include "dispatch_registration/common/dispatch-symmetric-i4.h"
#include "dispatch_registration/dispatch-registry.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/graph-diagnostics.h"
#include "graph/graph-matcher.h"
#include "graph/graph-traversal.h"
#include "graph/graph.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/command-program-executor.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"
#include "runtime/graph-replay.h"
#include "runtime/loom-kernel-jit.h"
#include "testing_suite.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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

static hrx_buffer_t dummy_hrx_buffer(uintptr_t value) {
    return reinterpret_cast<hrx_buffer_t>(value);
}

static hrx_stream_t dummy_hrx_stream(uintptr_t value) {
    return reinterpret_cast<hrx_stream_t>(value);
}

static hrx_graph_exec_t dummy_hrx_graph_exec(uintptr_t value) {
    return reinterpret_cast<hrx_graph_exec_t>(value);
}

static bool contains_value_id(const std::vector<ggml::hrx::ValueId> & ids, ggml::hrx::ValueId id) {
    for (const ggml::hrx::ValueId candidate : ids) {
        if (candidate == id) {
            return true;
        }
    }
    return false;
}

static bool command_program_verifies(const ggml::hrx::CommandProgram & program) {
    const ggml::hrx::VerificationResult result =
        ggml::hrx::verify_command_program(program, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    if (!result.valid()) {
        for (const std::string & error : result.status.errors()) {
            std::fprintf(stderr, "%s\n", error.c_str());
        }
    }
    return result.valid();
}

static bool async_jit_expected_from_environment() {
    const char * value = std::getenv("GGML_HRX_ASYNC_JIT");
    return value == nullptr ||
           (std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
            std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0);
}

static ggml::hrx::CommandProgram copy_command_program_shape(const ggml::hrx::CommandProgram & program) {
    ggml::hrx::CommandProgram copy;
    copy.initialization_commands  = program.initialization_commands;
    copy.commands                 = program.commands;
    copy.transients               = program.transients;
    copy.completion_counters      = program.completion_counters;
    copy.constant_initializations = program.constant_initializations;
    return copy;
}

static bool status_contains(const ggml::hrx::Status & status, const char * text) {
    for (const std::string & message : status.errors()) {
        if (message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool string_contains(const std::string & value, const char * text) {
    return value.find(text) != std::string::npos;
}

static std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

static std::vector<std::filesystem::path> list_directories(const std::filesystem::path & path) {
    std::vector<std::filesystem::path> directories;
    if (!std::filesystem::exists(path)) {
        return directories;
    }
    for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            directories.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end());
    return directories;
}

struct EnvironmentVariableGuard {
    const char * name;
    bool         had_value = false;
    std::string  value;

    explicit EnvironmentVariableGuard(const char * name) : name(name) {
        const char * current = std::getenv(name);
        if (current != nullptr) {
            had_value = true;
            value     = current;
        }
    }

    ~EnvironmentVariableGuard() {
        if (had_value) {
            setenv(name, value.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    void set(const std::filesystem::path & path) const { setenv(name, path.string().c_str(), 1); }

    void unset() const { unsetenv(name); }
};

static std::filesystem::path fresh_test_directory(const char * name) {
    static uint64_t       sequence = 0;
    std::filesystem::path path     = std::filesystem::temp_directory_path() /
                                 ("hrx-backend-test-" + std::string(name) + "-" + std::to_string(sequence++));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml::hrx::DispatchTarget test_dispatch_target() {
    return { "gfx1151" };
}

static const ggml::hrx::DispatchRegistry & test_dispatch_registry() {
    const ggml::hrx::DispatchRegistry * registry = ggml::hrx::find_dispatch_registry(test_dispatch_target());
    REQUIRE(registry != nullptr);
    return *registry;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

template <typename Check> static void require_scheduled_command_program(ggml_cgraph * graph, Check && check) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));
    check(imported.graph, scheduler.plan(), commands);
}

static std::vector<std::string> scheduled_kernel_names(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());

    std::vector<std::string> result;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        result.push_back(kernel_name_for_id(dispatch.kernel.kernel_id));
    }
    return result;
}

static void require_compile_parameter(const ggml::hrx::Dispatch & dispatch,
                                      const char *                name,
                                      const std::string &         value) {
    const auto found = dispatch.kernel.compile_parameters.find(name);
    REQUIRE(found != dispatch.kernel.compile_parameters.end());
    if (found->second != value) {
        std::fprintf(stderr, "compile parameter %s expected '%s' got '%s'\n", name, value.c_str(),
                     found->second.c_str());
        std::abort();
    }
}

static std::string expected_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static constexpr int64_t kQwenFlashHeadSize       = 128;
static constexpr int64_t kQwenRouterExpertCount   = 128;
static constexpr int64_t kQwenRouterRouteCount    = 8;
static constexpr int64_t kQwenMoeHiddenSize       = 2048;
static constexpr int64_t kQwenMoeIntermediateSize = 768;

static size_t qwen_expert_table_size(int64_t token_count, int64_t expert_count = kQwenRouterExpertCount) {
    return static_cast<size_t>(expert_count + expert_count * token_count) * sizeof(int32_t);
}

static size_t qwen_partition_table_size(int64_t token_count,
                                        int64_t route_count  = kQwenRouterRouteCount,
                                        int64_t expert_count = kQwenRouterExpertCount) {
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + expert_count) * sizeof(int32_t);
}

static size_t qwen_q8_1_x4_size(int64_t token_count, int64_t hidden_size) {
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static int64_t matmul_weight_format_config(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return 10;
        case GGML_TYPE_Q3_K:
            return 11;
        case GGML_TYPE_Q4_K:
            return 4;
        case GGML_TYPE_Q5_K:
            return 5;
        case GGML_TYPE_Q6_K:
            return 6;
        case GGML_TYPE_Q4_0:
            return 40;
        case GGML_TYPE_Q4_1:
            return 41;
        case GGML_TYPE_Q5_0:
            return 50;
        case GGML_TYPE_Q5_1:
            return 51;
        case GGML_TYPE_IQ1_S:
            return 19;
        case GGML_TYPE_IQ1_M:
            return 29;
        case GGML_TYPE_IQ2_S:
            return 22;
        case GGML_TYPE_IQ3_S:
            return 21;
        case GGML_TYPE_IQ4_NL:
            return 20;
        case GGML_TYPE_IQ4_XS:
            return 23;
        case GGML_TYPE_Q8_0:
            return 80;
        case GGML_TYPE_Q8_1:
            return 81;
        case GGML_TYPE_F16:
            return 16;
        case GGML_TYPE_BF16:
            return 30;
        case GGML_TYPE_F32:
            return 32;
        default:
            REQUIRE(false);
            return 0;
    }
}

static std::vector<int32_t> make_qwen_route_ids_iota(int64_t token_count,
                                                     int64_t route_count,
                                                     int64_t route_stride,
                                                     int64_t expert_count) {
    std::vector<int32_t> route_ids(static_cast<size_t>(token_count * route_stride), -1);
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            route_ids[static_cast<size_t>(token * route_stride + route)] =
                static_cast<int32_t>((token * route_count + route) % expert_count);
        }
    }
    return route_ids;
}

static std::vector<int32_t> make_qwen_expert_table_reference(const std::vector<int32_t> & route_ids,
                                                             int64_t                      token_count,
                                                             int64_t                      route_count,
                                                             int64_t                      route_stride,
                                                             int64_t                      expert_count) {
    std::vector<int32_t> expert_table(qwen_expert_table_size(token_count, expert_count) / sizeof(int32_t), -1);
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        expert_table[static_cast<size_t>(expert)] = 0;
    }
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t route = 0; route < route_count; ++route) {
            const int32_t expert = route_ids[static_cast<size_t>(token * route_stride + route)];
            REQUIRE(expert >= 0);
            REQUIRE(expert < expert_count);
            int32_t &    count = expert_table[static_cast<size_t>(expert)];
            const size_t assignment_offset =
                static_cast<size_t>(expert_count + static_cast<int64_t>(expert) * token_count + count);
            expert_table[assignment_offset] = static_cast<int32_t>(token * route_count + route);
            ++count;
        }
    }
    return expert_table;
}

static std::vector<int32_t> make_qwen_partition_table_reference(const std::vector<int32_t> & expert_table,
                                                                int64_t                      token_count,
                                                                int64_t                      route_count,
                                                                int64_t                      expert_count) {
    std::vector<int32_t> partition_table(
        qwen_partition_table_size(token_count, route_count, expert_count) / sizeof(int32_t), -1);
    int32_t partition_count = 0;
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        const int32_t expert_assignment_count = expert_table[static_cast<size_t>(expert)];
        REQUIRE(expert_assignment_count >= 0);
        const int32_t expert_partition_count = (expert_assignment_count + 31) / 32;
        for (int32_t partition = 0; partition < expert_partition_count; ++partition) {
            int32_t row_count = expert_assignment_count - partition * 32;
            if (row_count > 32) {
                row_count = 32;
            }
            const int32_t descriptor = static_cast<int32_t>(expert) | (partition << 7) | ((row_count - 1) << 13);
            partition_table[static_cast<size_t>(1 + partition_count)] = descriptor;
            ++partition_count;
        }
    }
    partition_table[0] = partition_count;
    return partition_table;
}

static void require_qwen_expert_table_matches(const std::vector<int32_t> & actual,
                                              const std::vector<int32_t> & expected,
                                              int64_t                      token_count,
                                              int64_t                      expert_count) {
    REQUIRE(actual.size() == expected.size());
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        const size_t count_index = static_cast<size_t>(expert);
        REQUIRE(actual[count_index] == expected[count_index]);
        for (int32_t ordinal = 0; ordinal < expected[count_index]; ++ordinal) {
            const size_t assignment_index =
                static_cast<size_t>(expert_count + expert * token_count + static_cast<int64_t>(ordinal));
            REQUIRE(actual[assignment_index] == expected[assignment_index]);
        }
    }
}

static void require_qwen_partition_table_matches(const std::vector<int32_t> & actual,
                                                 const std::vector<int32_t> & expected) {
    REQUIRE(actual.size() == expected.size());
    REQUIRE(actual[0] == expected[0]);
    for (int32_t i = 0; i < actual[0]; ++i) {
        REQUIRE(actual[static_cast<size_t>(1 + i)] == expected[static_cast<size_t>(1 + i)]);
    }
}

static size_t qwen_routed_gate_up_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeIntermediateSize) * sizeof(ggml_fp16_t);
}

static size_t qwen_routed_down_f16_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kQwenRouterRouteCount * kQwenMoeHiddenSize) * sizeof(ggml_fp16_t);
}

static void set_qwen_flash_query_layout(ggml_tensor * tensor, int64_t head_count, int64_t head_size) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(float);
    tensor->nb[1] = static_cast<size_t>(head_count * head_size) * sizeof(float);
    tensor->nb[2] = static_cast<size_t>(head_size) * sizeof(float);
}

static void set_qwen_flash_key_value_layout(ggml_tensor * tensor, int64_t head_count, int64_t head_size) {
    REQUIRE(tensor != nullptr);
    tensor->nb[0] = sizeof(ggml_fp16_t);
    tensor->nb[1] = static_cast<size_t>(head_count * head_size) * sizeof(ggml_fp16_t);
    tensor->nb[2] = static_cast<size_t>(head_size) * sizeof(ggml_fp16_t);
}

static ggml_tensor * build_qwen_flash_attention_graph(ggml_context * ctx,
                                                      int64_t        query_token_count,
                                                      int64_t        key_value_token_count,
                                                      int64_t        query_head_count,
                                                      int64_t        key_value_head_count,
                                                      ggml_type      query_type      = GGML_TYPE_F32,
                                                      ggml_type      key_value_type  = GGML_TYPE_F16,
                                                      bool           include_mask    = true,
                                                      bool           include_sinks   = false,
                                                      int64_t        head_size       = kQwenFlashHeadSize,
                                                      float          scale           = 1.0f / std::sqrt(128.0f),
                                                      int64_t        value_head_size = -1,
                                                      float          max_bias        = 0.0f,
                                                      float          logit_softcap   = 0.0f) {
    const int64_t actual_value_head_size = value_head_size > 0 ? value_head_size : head_size;
    ggml_tensor * query = ggml_new_tensor_3d(ctx, query_type, head_size, query_token_count, query_head_count);
    ggml_tensor * key = ggml_new_tensor_3d(ctx, key_value_type, head_size, key_value_token_count, key_value_head_count);
    ggml_tensor * value =
        ggml_new_tensor_3d(ctx, key_value_type, actual_value_head_size, key_value_token_count, key_value_head_count);
    REQUIRE(query != nullptr);
    REQUIRE(key != nullptr);
    REQUIRE(value != nullptr);
    if (query_type == GGML_TYPE_F32) {
        set_qwen_flash_query_layout(query, query_head_count, head_size);
    }
    if (key_value_type == GGML_TYPE_F16) {
        set_qwen_flash_key_value_layout(key, key_value_head_count, head_size);
        set_qwen_flash_key_value_layout(value, key_value_head_count, actual_value_head_size);
    }
    ggml_tensor * mask = nullptr;
    if (include_mask) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_token_count, query_token_count);
        REQUIRE(mask != nullptr);
    }
    ggml_tensor * output = ggml_flash_attn_ext(ctx, query, key, value, mask, scale, max_bias, logit_softcap);
    REQUIRE(output != nullptr);
    if (include_sinks) {
        ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, query_head_count);
        REQUIRE(sinks != nullptr);
        ggml_flash_attn_ext_add_sinks(output, sinks);
    }
    return output;
}

static ggml_tensor * build_qwen_router_top8_graph(ggml_context *  ctx,
                                                  ggml_tensor *   logits,
                                                  ggml_tensor **  route_ids   = nullptr,
                                                  ggml_sort_order order       = GGML_SORT_ORDER_DESC,
                                                  int64_t         route_count = kQwenRouterRouteCount,
                                                  float           clamp_min   = 1.0e-7f) {
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    REQUIRE(probs != nullptr);
    ggml_tensor * probs_reshaped = ggml_reshape_3d(ctx, probs, 1, logits->ne[0], logits->ne[1]);
    REQUIRE(probs_reshaped != nullptr);
    ggml_tensor * argsort = ggml_argsort(ctx, probs, order);
    REQUIRE(argsort != nullptr);
    ggml_tensor * topk = ggml_view_2d(ctx, argsort, route_count, logits->ne[1], argsort->nb[1], 0);
    REQUIRE(topk != nullptr);
    if (route_ids != nullptr) {
        *route_ids = topk;
    }
    ggml_tensor * selected = ggml_get_rows(ctx, probs_reshaped, topk);
    REQUIRE(selected != nullptr);
    ggml_tensor * selected_reshaped = ggml_reshape_2d(ctx, selected, route_count, logits->ne[1]);
    REQUIRE(selected_reshaped != nullptr);
    ggml_tensor * sum = ggml_sum_rows(ctx, selected_reshaped);
    REQUIRE(sum != nullptr);
    ggml_tensor * clamped_sum = ggml_clamp(ctx, sum, clamp_min, std::numeric_limits<float>::infinity());
    REQUIRE(clamped_sum != nullptr);
    ggml_tensor * normalized = ggml_div(ctx, selected_reshaped, clamped_sum);
    REQUIRE(normalized != nullptr);
    ggml_tensor * output = ggml_reshape_3d(ctx, normalized, 1, route_count, logits->ne[1]);
    REQUIRE(output != nullptr);
    return output;
}

static std::vector<size_t> traversal_indices(const ggml::hrx::Graph & graph) {
    const ggml::hrx::GraphTraversalOrder order = ggml::hrx::GraphTraversalOrder::build(graph);
    std::vector<size_t>                  indices;
    indices.reserve(order.nodes().size());
    for (const ggml::hrx::GraphNode * node : order.nodes()) {
        size_t index = 0;
        REQUIRE(node != nullptr);
        REQUIRE(graph.index().node_index(node, index));
        indices.push_back(index);
    }
    return indices;
}

static size_t find_position(const std::vector<size_t> & indices, size_t node_index) {
    const std::vector<size_t>::const_iterator it = std::find(indices.begin(), indices.end(), node_index);
    REQUIRE(it != indices.end());
    return static_cast<size_t>(it - indices.begin());
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

static bool match_dispatch_at_index(const ggml::hrx::Graph &       graph,
                                    const ggml::hrx::CommandPlan & plan,
                                    const std::vector<bool> &      covered_nodes,
                                    size_t                         node_index,
                                    ggml::hrx::DispatchMatch &     match);

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes,
                                 ggml::hrx::Graph *         graph = nullptr);

static void run_status_checks() {
    ggml::hrx::Status status;
    REQUIRE(status.success());
    REQUIRE(status.errors().empty());

    status.log("first");
    REQUIRE(!status.success());
    REQUIRE(!status.errors().empty());
    REQUIRE(status.errors().size() == 1);
    REQUIRE(status.errors()[0] == "first");

    status.log("value %d", 7);
    REQUIRE(status.errors().size() == 2);
    REQUIRE(status.errors()[1] == "value 7");

    ggml::hrx::Status other;
    other.log("third");
    status.append(other);
    REQUIRE(status.errors().size() == 3);
    REQUIRE(status.errors()[2] == "third");
}

static void run_command_plan_metadata_checks() {
    const ggml::hrx::MoeRoutingResourceMetadata routing = {
        4,
        8,
        128,
        128,
    };
    const ggml::hrx::CommandPlanResourceMetadata metadata = ggml::hrx::make_command_plan_resource_metadata(routing);

    REQUIRE(metadata.kind == ggml::hrx::CommandPlanResourceMetadataKind::MoeRoutingResource);
    ggml::hrx::MoeRoutingResourceMetadata decoded;
    REQUIRE(metadata.read(decoded));
    REQUIRE(decoded.token_count == routing.token_count);
    REQUIRE(decoded.route_count == routing.route_count);
    REQUIRE(decoded.route_stride == routing.route_stride);
    REQUIRE(decoded.expert_count == routing.expert_count);

    const ggml::hrx::CommandPlanResourceMetadata empty;
    REQUIRE(!empty.read(decoded));

    ggml::hrx::CommandPlanMetadata metadata_plan;
    ggml::hrx::Status              status;
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(2), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(metadata_plan.alternate_values().size() == 1);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 16) != nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F32, 16) == nullptr);
    REQUIRE(metadata_plan.find_alternate_value(ggml::hrx::ValueId(1), GGML_TYPE_F16, 32) == nullptr);
    REQUIRE(!metadata_plan.append_alternate_value(
        { ggml::hrx::ValueId(1), ggml::hrx::ValueId(3), GGML_TYPE_F16, 16, "alternate" }, status));
    REQUIRE(!status.success());

    ggml::hrx::CommandPlanMetadata               bundle_plan;
    ggml::hrx::Status                            bundle_status;
    const ggml::hrx::CommandPlanMoeRoutingBundle bundle = {
        ggml::hrx::ValueId(10),
        ggml::hrx::ValueId(11),
        ggml::hrx::ValueId(12),
        ggml::hrx::ValueId(13),
        128,
        64,
        4,
        8,
        128,
        128,
    };
    REQUIRE(bundle_plan.append_moe_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.append_moe_routing_bundle(bundle, bundle_status));
    REQUIRE(bundle_plan.moe_routing_bundles().size() == 1);
    const ggml::hrx::CommandPlanMoeRoutingBundle * found_bundle =
        bundle_plan.find_moe_routing_bundle(ggml::hrx::ValueId(10));
    REQUIRE(found_bundle != nullptr);
    REQUIRE(found_bundle->route_weights == ggml::hrx::ValueId(11));
    REQUIRE(found_bundle->expert_table == ggml::hrx::ValueId(12));
    REQUIRE(found_bundle->partition_table == ggml::hrx::ValueId(13));
    ggml::hrx::CommandPlanMoeRoutingBundle conflicting_bundle = bundle;
    conflicting_bundle.route_weights                          = ggml::hrx::ValueId(14);
    REQUIRE(!bundle_plan.append_moe_routing_bundle(conflicting_bundle, bundle_status));
    REQUIRE(!bundle_status.success());
}

static bool has_dispatch_registration(const std::vector<ggml::hrx::DispatchRegistration> & registrations,
                                      const char *                                         name) {
    for (const ggml::hrx::DispatchRegistration & registration : registrations) {
        if (std::string(registration.name) == name) {
            return true;
        }
    }
    return false;
}

static bool has_dispatch_registration_kind(const std::vector<ggml::hrx::DispatchRegistration> & registrations,
                                           const char *                                         name,
                                           ggml::hrx::DispatchMatchKind                         kind) {
    for (const ggml::hrx::DispatchRegistration & registration : registrations) {
        if (std::string(registration.name) == name && registration.kind == kind) {
            return true;
        }
    }
    return false;
}

static void restore_environment_value(const char * name, bool had_value, const std::string & value) {
    if (had_value) {
        REQUIRE(setenv(name, value.c_str(), 1) == 0);
    } else {
        REQUIRE(unsetenv(name) == 0);
    }
}

static void add_binary_f32_exact_dispatch_params(ggml::hrx::Dispatch & dispatch, int64_t element_count) {
    dispatch.kernel.integer_parameters.emplace("element_count", element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne0", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne1", "1");
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne2", "1");
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride1", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride2", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride3", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride1", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride2", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride3", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_span", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_span", std::to_string(element_count));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.op", "0");
}

static bool match_test_single_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                       ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 1);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_fused_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                      ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 2);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_test_wrong_root_dispatch(const ggml::hrx::DispatchMatchContext & context,
                                           ggml::hrx::DispatchMatch &              match) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.integer_parameters.emplace("route", 3);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void run_dispatch_registry_checks() {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);

    const ggml::hrx::DispatchRegistry & registry = test_dispatch_registry();
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1100" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1151" }) != nullptr);
    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx0000" }) == nullptr);

    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_ADD), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SUB), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_DIV), "common.binary_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.tiled_pair_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.tiled_pair_postops_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.packed_mul_mat_glu.tiled_pair_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "llm.attention_qkv_matmul_postprocess.tiled_vector_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.tiled_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.skinny_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.skinny_f32_f32_decode",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_add.skinny_f32_f32_decode",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_unary.tiled_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_unary.skinny_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.tiled_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.skinny_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT), "qwen.matmul.q6k_q8_1_x4"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT),
                                      "llm.moe_router.projection_f32_four_row_wave32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "qwen.rmsnorm_f32_quantize_q8_1_x4"));
    REQUIRE(
        has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_binary_q8_1_x4"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_binary_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_f32_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_decode_split_next_q8"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_RESHAPE),
                                      "qwen.attention_postprocess_f32_f16"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32_next"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows_scale.f32"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_GET_ROWS), "common.gather_add_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_ROPE), "common.rope_set_rows.f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_ROPE), "common.rope_concat.f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_ROPE), "common.rope.f32",
                                           ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SET_ROWS), "common.set_rows"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_SOFT_MAX), "llm.moe_router.top8_f32"));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_swiglu.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_postops.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id.f32_f32_wmma", ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.gate_up_swiglu_f16_wmma"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.down_q4k_f16_wmma_grouped"));
    REQUIRE(has_dispatch_registration(registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                      "llm.routed_ffn.down_q6k_f16_wmma_grouped"));
    REQUIRE(registry.single_op_registrations().size() >= 5);

    REQUIRE(setenv(kDisableQwenDispatchEnv, "0", 1) == 0);
    const ggml::hrx::DispatchRegistry & false_env_registry = test_dispatch_registry();
    REQUIRE(
        has_dispatch_registration(false_env_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));

    REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
    const ggml::hrx::DispatchRegistry & generic_registry = test_dispatch_registry();
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_ADD), "common.binary_f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "common.rmsnorm_binary_f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                      "common.rmsnorm_binary_q8_1_x4"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM), "common.rmsnorm_f32"));
    REQUIRE(
        has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.gather_add_f32"));
    REQUIRE(
        has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS), "common.get_rows.f32"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS),
                                      "common.get_rows.f32_next"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_ROPE),
                                           "common.rope_set_rows.f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_ROPE), "common.rope.f32",
                                           ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_SET_ROWS), "common.set_rows"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RMS_NORM),
                                       "qwen.rmsnorm_f32_quantize_q8_1_x4"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.tiled_pair_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_swiglu.tiled_pair_postops_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.packed_mul_mat_glu.tiled_pair_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "llm.attention_qkv_matmul_postprocess.tiled_vector_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.tiled_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.skinny_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat.skinny_f32_f32_decode",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_add.skinny_f32_f32_decode",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_unary.tiled_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_unary.skinny_f32_f32", ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.tiled_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                           "common.mul_mat_postops.skinny_f32_f32",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT),
                                       "qwen.matmul.q6k_q8_1_x4"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_f32_f16_wmma"));
    REQUIRE(has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_FLASH_ATTN_EXT),
                                      "common.flash_attention_decode_split_next_q8"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_RESHAPE),
                                       "qwen.attention_postprocess_f32_f16"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_GET_ROWS),
                                       "qwen.preamble.token_embedding_q4k"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_SOFT_MAX),
                                       "llm.moe_router.top8_f32"));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_swiglu.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id_postops.f32_f32_wmma",
                                           ggml::hrx::DispatchMatchKind::Fused));
    REQUIRE(has_dispatch_registration_kind(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                           "common.mul_mat_id.f32_f32_wmma", ggml::hrx::DispatchMatchKind::SingleOp));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.gate_up_swiglu_f16_wmma"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.down_q4k_f16_wmma_grouped"));
    REQUIRE(!has_dispatch_registration(generic_registry.registrations_for_root(GGML_OP_MUL_MAT_ID),
                                       "llm.routed_ffn.down_q6k_f16_wmma_grouped"));
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);

    ggml::hrx::DispatchRegistryBuilder builder;
    builder.add({
        "test.single_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::SingleOp,
        1000,
        ggml::hrx::DispatchSource::Common,
        match_test_single_dispatch,
    });
    builder.add({
        "test.fused_add",
        GGML_OP_ADD,
        ggml::hrx::DispatchMatchKind::Fused,
        0,
        ggml::hrx::DispatchSource::Common,
        match_test_fused_dispatch,
    });
    builder.add({
        "test.wrong_root",
        GGML_OP_MUL_MAT,
        ggml::hrx::DispatchMatchKind::Fused,
        2000,
        ggml::hrx::DispatchSource::Common,
        match_test_wrong_root_dispatch,
    });
    const ggml::hrx::DispatchRegistry ordering_registry = builder.build();

    ggml::hrx::Graph graph;
    graph.add_node(GGML_OP_ADD, ggml::hrx::ValueId(0), {});
    REQUIRE(graph.build_index().success());

    const std::vector<bool>               covered_nodes(graph.nodes().size(), false);
    const ggml::hrx::CommandPlan          plan;
    const ggml::hrx::DispatchMatchContext context = {
        graph, &graph.nodes().front(),
        0,     covered_nodes,
        plan,  ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };
    ggml::hrx::DispatchMatch match;
    REQUIRE(ordering_registry.match(context, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(match.dispatches.front().kernel.integer_parameters.at("route") == 2);
}

static void run_graph_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    const ggml::hrx::GraphNode * node = &imported.graph.nodes().front();
    REQUIRE(node->op == GGML_OP_ADD);
    REQUIRE(node->inputs.size() == 2);
    REQUIRE(node->inputs[0] != node->inputs[1]);
    REQUIRE(ggml::hrx::DispatchScheduler::supports_node(imported.graph, node, test_dispatch_target()));

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * b_value   = imported.graph.values().find_tensor(b);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(b_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(b_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(!a_value->buffer.has_value());
    REQUIRE(!b_value->buffer.has_value());
    REQUIRE(!out_value->buffer.has_value());

    const std::vector<ggml::hrx::ValueId> external_ids = imported.graph.values().external_value_ids();
    REQUIRE(external_ids.size() == 3);
    REQUIRE(contains_value_id(external_ids, a_value->id));
    REQUIRE(contains_value_id(external_ids, b_value->id));
    REQUIRE(contains_value_id(external_ids, out_value->id));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    REQUIRE(scheduler.plan().dispatches.front().bindings.size() == 3);
    REQUIRE(scheduler.plan().dispatches.front().bindings[0].length == a_value->byte_count);

    ggml::hrx::CommandProgramBindings missing_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!missing_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count }));
    ggml::hrx::CommandProgramBindings partial_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!partial_bindings.valid());

    REQUIRE(imported.graph.values().bind_buffer(b_value->id, { dummy_hrx_buffer(0x1800), 0, b_value->byte_count }));
    ggml::hrx::CommandProgramBindings missing_output_binding =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!missing_output_binding.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, 0 }));
    ggml::hrx::CommandProgramBindings empty_runtime_binding =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(!empty_runtime_binding.valid());

    REQUIRE(imported.graph.values().bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    ggml::hrx::CommandProgramBindings runtime_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(runtime_bindings.valid());
    REQUIRE(runtime_bindings.bindings().size() == 3);
    const ggml::hrx::CommandProgramBinding * a_binding = runtime_bindings.find(a_value->id);
    REQUIRE(a_binding != nullptr);
    REQUIRE(a_binding->buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(a_binding->offset == 0);
    REQUIRE(a_binding->length == a_value->byte_count);
    const ggml::hrx::CommandProgramBinding * b_binding = runtime_bindings.find(b_value->id);
    REQUIRE(b_binding != nullptr);
    REQUIRE(b_binding->buffer == dummy_hrx_buffer(0x1800));
    REQUIRE(b_binding->offset == 0);
    REQUIRE(b_binding->length == b_value->byte_count);
    REQUIRE(runtime_bindings.find(ggml::hrx::ValueId(123456)) == nullptr);

    const ggml::hrx::CommandProgramBindingsFingerprint runtime_fingerprint =
        ggml::hrx::command_program_bindings_fingerprint(runtime_bindings);
    REQUIRE(!runtime_fingerprint.value.empty());

    ggml::hrx::ValueMap changed_identity = imported.graph.values();
    REQUIRE(changed_identity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 1, 0, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_identity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_identity);
    REQUIRE(changed_identity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_identity_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_generation = imported.graph.values();
    REQUIRE(changed_generation.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 1, a_value->byte_count }));
    const ggml::hrx::CommandProgramBindings changed_generation_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_generation);
    REQUIRE(changed_generation_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_generation_bindings).value !=
            runtime_fingerprint.value);

    ggml::hrx::ValueMap changed_capacity = imported.graph.values();
    REQUIRE(changed_capacity.bind_buffer(
        a_value->id, { dummy_hrx_buffer(0x1000), 0, a_value->byte_count, 0, 0, a_value->byte_count + 256 }));
    const ggml::hrx::CommandProgramBindings changed_capacity_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(changed_capacity);
    REQUIRE(changed_capacity_bindings.valid());
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(changed_capacity_bindings).value !=
            runtime_fingerprint.value);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    const ggml::hrx::Command & command = commands.commands.front();
    REQUIRE(command.ordinal == 0);
    REQUIRE(command.kind == ggml::hrx::CommandKind::Kernel);
    REQUIRE(command.kernel.kernel_id != ggml::hrx::kUncatalogedKernelId);
    REQUIRE(command.bindings.size() == 3);
    REQUIRE(command.bindings[0].name == "lhs");
    REQUIRE(command.bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[0].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[1].name == "rhs");
    REQUIRE(command.bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[1].access == ggml::hrx::ResourceAccess::Read);
    REQUIRE(command.bindings[2].name == "output");
    REQUIRE(command.bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(command.bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
    REQUIRE(command_program_verifies(commands));

    const ggml::hrx::PreparedCommand default_prepared_command;
    REQUIRE(default_prepared_command.kind == ggml::hrx::CommandKind::Invalid);
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Invalid) == "Invalid");
    REQUIRE(ggml::hrx::command_kind_name(ggml::hrx::CommandKind::Kernel) == "Kernel");
    REQUIRE(ggml::hrx::command_kind_name(static_cast<ggml::hrx::CommandKind>(255)) == "Unknown(255)");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::GraphValue) == "GraphValue");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::Transient) == "Transient");
    REQUIRE(ggml::hrx::command_binding_origin_name(ggml::hrx::CommandBindingOrigin::ProgramConstant) ==
            "ProgramConstant");
    REQUIRE(ggml::hrx::command_binding_origin_name(static_cast<ggml::hrx::CommandBindingOrigin>(255)) ==
            "Unknown(255)");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Read) == "Read");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::Write) == "Write");
    REQUIRE(ggml::hrx::resource_access_name(ggml::hrx::ResourceAccess::ReadWrite) == "ReadWrite");
    REQUIRE(ggml::hrx::resource_access_name(static_cast<ggml::hrx::ResourceAccess>(255)) == "Unknown(255)");

    const std::string binding_text = ggml::hrx::format_command_binding(command.bindings[0]);
    REQUIRE(string_contains(binding_text, "binding lhs"));
    REQUIRE(string_contains(binding_text, "value="));
    REQUIRE(string_contains(binding_text, "origin=GraphValue"));
    REQUIRE(string_contains(binding_text, "access=Read"));
    REQUIRE(std::string(ggml::hrx::hrx_graph_replay_event_name(ggml::hrx::HrxGraphReplayEvent::Disabled)) ==
            "disabled");
    REQUIRE(std::string(ggml::hrx::hrx_graph_replay_event_name(ggml::hrx::HrxGraphReplayEvent::Hit)) == "hit");
    REQUIRE(string_contains(binding_text, "range=[0, "));
    REQUIRE(string_contains(binding_text, std::to_string(a_value->byte_count).c_str()));

    const std::string command_text = ggml::hrx::format_command(command);
    REQUIRE(string_contains(command_text, "command 0"));
    REQUIRE(string_contains(command_text, "kind=Kernel"));
    REQUIRE(string_contains(command_text, "kernel_id="));
    REQUIRE(string_contains(command_text, "bindings=3"));
    REQUIRE(string_contains(command_text, "deps=0"));

    const std::string program_text = ggml::hrx::format_command_program(commands);
    REQUIRE(string_contains(program_text, "command_program commands=1"));
    REQUIRE(string_contains(program_text, "command 0"));
    REQUIRE(string_contains(program_text, "binding lhs"));
    REQUIRE(string_contains(program_text, "binding rhs"));
    REQUIRE(string_contains(program_text, "binding output"));

    ggml::hrx::ResolvedCommandProgram resolved =
        ggml::hrx::resolve_command_program_bindings(commands, runtime_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 1);
    REQUIRE(resolved.commands.front().ordinal == command.ordinal);
    REQUIRE(resolved.commands.front().kind == command.kind);
    REQUIRE(resolved.commands.front().kernel.kernel_id == command.kernel.kernel_id);
    REQUIRE(resolved.commands.front().bindings.size() == 3);
    REQUIRE(resolved.commands.front().bindings[0].binding.name == "lhs");
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == a_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[1].binding.name == "rhs");
    REQUIRE(resolved.commands.front().bindings[1].ref.buffer == dummy_hrx_buffer(0x1800));
    REQUIRE(resolved.commands.front().bindings[1].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[1].ref.length == b_value->byte_count);
    REQUIRE(resolved.commands.front().bindings[2].binding.name == "output");
    REQUIRE(resolved.commands.front().bindings[2].ref.buffer == dummy_hrx_buffer(0x2000));
    REQUIRE(resolved.commands.front().bindings[2].ref.offset == 0);
    REQUIRE(resolved.commands.front().bindings[2].ref.length == out_value->byte_count);

    ggml::hrx::CommandProgram offset_command           = copy_command_program_shape(commands);
    offset_command.commands.front().bindings[0].offset = 4;
    offset_command.commands.front().bindings[0].length = 8;
    ggml::hrx::ValueMap offset_values                  = imported.graph.values();
    REQUIRE(offset_values.bind_buffer(a_value->id, { dummy_hrx_buffer(0x3000), 16, a_value->byte_count }));
    REQUIRE(offset_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x4000), 32, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings offset_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(offset_values);
    REQUIRE(offset_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(offset_command, offset_bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.front().bindings[0].ref.buffer == dummy_hrx_buffer(0x3000));
    REQUIRE(resolved.commands.front().bindings[0].ref.offset == 20);
    REQUIRE(resolved.commands.front().bindings[0].ref.length == 8);

    std::vector<uint8_t>          host_storage(a_value->byte_count + 64);
    ggml::hrx::ValueBufferBinding host_value_binding;
    host_value_binding.host_data    = host_storage.data();
    host_value_binding.offset       = 16;
    host_value_binding.length       = a_value->byte_count;
    host_value_binding.identity     = 42;
    host_value_binding.generation   = 1;
    host_value_binding.capacity     = host_storage.size();
    host_value_binding.weight       = true;
    ggml::hrx::ValueMap host_values = imported.graph.values();
    REQUIRE(host_values.bind_buffer(a_value->id, host_value_binding));
    REQUIRE(host_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x5000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings host_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(host_values);
    REQUIRE(host_bindings.valid());
    const ggml::hrx::CommandProgramBinding * host_binding = host_bindings.find(a_value->id);
    REQUIRE(host_binding != nullptr);
    REQUIRE(host_binding->buffer == nullptr);
    REQUIRE(host_binding->host_data == host_storage.data());
    REQUIRE(host_binding->offset == 16);
    REQUIRE(host_binding->length == a_value->byte_count);
    REQUIRE(host_binding->weight);
    REQUIRE(ggml::hrx::command_program_bindings_fingerprint(host_bindings).value != runtime_fingerprint.value);

    resolved = ggml::hrx::resolve_command_program_bindings(commands, missing_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));
    REQUIRE(status_contains(resolved.status, "binding output"));
    REQUIRE(status_contains(resolved.status, "value="));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, partial_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "is not bound"));

    resolved = ggml::hrx::resolve_command_program_bindings(commands, empty_runtime_binding);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty binding"));

    ggml::hrx::ValueMap null_values = imported.graph.values();
    REQUIRE(null_values.bind_buffer(a_value->id, { nullptr, 0, a_value->byte_count }));
    REQUIRE(null_values.bind_buffer(out_value->id, { dummy_hrx_buffer(0x2000), 0, out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings null_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(null_values);
    REQUIRE(!null_bindings.valid());
    resolved = ggml::hrx::resolve_command_program_bindings(commands, null_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "null buffer"));
    REQUIRE(status_contains(resolved.status, "binding lhs"));

    ggml::hrx::CommandProgram empty_resolve_binding           = copy_command_program_shape(commands);
    empty_resolve_binding.commands.front().bindings[0].length = 0;
    resolved = ggml::hrx::resolve_command_program_bindings(empty_resolve_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "empty range"));
    REQUIRE(status_contains(resolved.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram out_of_range_binding           = copy_command_program_shape(commands);
    out_of_range_binding.commands.front().bindings[0].offset = a_value->byte_count;
    out_of_range_binding.commands.front().bindings[0].length = 4;
    resolved = ggml::hrx::resolve_command_program_bindings(out_of_range_binding, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "outside runtime binding length"));
    REQUIRE(status_contains(resolved.status, "binding lhs"));

    ggml::hrx::CommandProgram unsupported_origin           = copy_command_program_shape(commands);
    unsupported_origin.commands.front().bindings[0].origin = static_cast<ggml::hrx::CommandBindingOrigin>(255);
    resolved = ggml::hrx::resolve_command_program_bindings(unsupported_origin, runtime_bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "unsupported binding origin"));
    REQUIRE(status_contains(resolved.status, "origin=Unknown(255)"));
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(unsupported_origin, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "unsupported binding origin"));
    REQUIRE(status_contains(verification.status, "origin=Unknown(255)"));

    ggml::hrx::CommandProgram invalid_kernel         = copy_command_program_shape(commands);
    invalid_kernel.commands.front().kernel.kernel_id = ggml::hrx::kUncatalogedKernelId;
    verification = ggml::hrx::verify_command_program(invalid_kernel, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "command 0"));
    REQUIRE(status_contains(verification.status, "kernel_id="));

    ggml::hrx::CommandProgram empty_bindings = copy_command_program_shape(commands);
    empty_bindings.commands.front().bindings.clear();
    verification = ggml::hrx::verify_command_program(empty_bindings, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "bindings=0"));

    ggml::hrx::CommandProgram empty_binding           = copy_command_program_shape(commands);
    empty_binding.commands.front().bindings[0].length = 0;
    verification = ggml::hrx::verify_command_program(empty_binding, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding lhs"));
    REQUIRE(status_contains(verification.status, "range=[0, 0)"));

    ggml::hrx::CommandProgram invalid_value          = copy_command_program_shape(commands);
    invalid_value.commands.front().bindings[0].value = ggml::hrx::ValueId(-1);
    verification = ggml::hrx::verify_command_program(invalid_value, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "value=-1"));

    ggml::hrx::CommandProgram forward_dependency = copy_command_program_shape(commands);
    forward_dependency.commands.front().dependencies.push_back(0);
    verification =
        ggml::hrx::verify_command_program(forward_dependency, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "forward dependency 0"));

    ggml::hrx::CommandProgram wrong_binding_name         = copy_command_program_shape(commands);
    wrong_binding_name.commands.front().bindings[0].name = "wrong";
    verification =
        ggml::hrx::verify_command_program(wrong_binding_name, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "binding wrong"));

    ggml::hrx::CommandProgram wrong_binding_access           = copy_command_program_shape(commands);
    wrong_binding_access.commands.front().bindings[0].access = ggml::hrx::ResourceAccess::Write;
    verification =
        ggml::hrx::verify_command_program(wrong_binding_access, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "access=Write"));

    ggml::hrx::CommandProgram transient_flow = copy_command_program_shape(commands);
    transient_flow.commands.resize(2);
    transient_flow.commands[0].ordinal = 0;
    transient_flow.commands[0].dependencies.clear();
    transient_flow.commands[1] = transient_flow.commands[0];
    transient_flow.commands[1].ordinal = 1;
    transient_flow.commands[1].dependencies.push_back(0);
    const ggml::hrx::ValueId transient_value(100000);
    transient_flow.transients.allocations.push_back({
        transient_value,
        command.bindings[2].length,
        256,
        0,
    });
    transient_flow.transients.arena_size = command.bindings[2].length;
    transient_flow.commands[0].bindings[2].origin = ggml::hrx::CommandBindingOrigin::Transient;
    transient_flow.commands[0].bindings[2].value  = transient_value;
    transient_flow.commands[1].bindings[0].origin = ggml::hrx::CommandBindingOrigin::Transient;
    transient_flow.commands[1].bindings[0].value  = transient_value;
    transient_flow.commands[1].bindings[0].length = command.bindings[2].length;
    verification = ggml::hrx::verify_command_program(transient_flow, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(verification.valid());

    ggml::hrx::CommandProgram read_before_write = copy_command_program_shape(transient_flow);
    std::swap(read_before_write.commands[0], read_before_write.commands[1]);
    read_before_write.commands[0].ordinal = 0;
    read_before_write.commands[0].dependencies.clear();
    read_before_write.commands[1].ordinal = 1;
    read_before_write.commands[1].dependencies.clear();
    read_before_write.commands[1].dependencies.push_back(0);
    verification = ggml::hrx::verify_command_program(read_before_write, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "reads transient value 100000 before write by command 1"));

    const ggml::hrx::KernelCorpus &                 corpus          = ggml::hrx::get_qwen_kernel_corpus();
    const ggml::hrx::CommandProgramExecutionContext prepare_context = {
        nullptr, nullptr, "gfx1151", &corpus, nullptr, nullptr, nullptr, nullptr,
    };

    ggml::hrx::PreparedCommandProgram prepared =
        ggml::hrx::prepare_command_program(prepare_context, invalid_kernel, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "kernel_id="));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, missing_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "is not bound"));

    prepared = ggml::hrx::prepare_command_program(prepare_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "missing HRX device"));

    const ggml::hrx::CommandProgramExecutionContext missing_kernel_cache_context = {
        reinterpret_cast<hrx_device_t>(uintptr_t(1)), nullptr, "gfx1151", &corpus, nullptr, nullptr, nullptr, nullptr,
    };
    prepared = ggml::hrx::prepare_command_program(missing_kernel_cache_context, commands, runtime_bindings);
    REQUIRE(!prepared.valid());
    REQUIRE(status_contains(prepared.status, "missing HRX kernel executable cache"));

    ggml_free(ctx);
}

static void run_graph_import_mixed_backend_boundary_checks() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * cpu_sin  = ggml_sin(ctx, input);
    ggml_tensor * hrx_norm = ggml_rms_norm(ctx, cpu_sin, 1.0e-6f);
    REQUIRE(input != nullptr);
    REQUIRE(cpu_sin != nullptr);
    REQUIRE(hrx_norm != nullptr);

    ggml_cgraph * cpu_to_hrx = ggml_new_graph(ctx);
    REQUIRE(cpu_to_hrx != nullptr);
    ggml_graph_add_node(cpu_to_hrx, hrx_norm);

    ggml::hrx::GraphImportResult imported_cpu_to_hrx = ggml::hrx::import_ggml_graph(*cpu_to_hrx);
    REQUIRE(imported_cpu_to_hrx.valid());
    REQUIRE(imported_cpu_to_hrx.graph.nodes().size() == 1);
    REQUIRE(imported_cpu_to_hrx.graph.nodes()[0].op == GGML_OP_RMS_NORM);

    const ggml::hrx::Value * cpu_sin_value  = imported_cpu_to_hrx.graph.values().find_tensor(cpu_sin);
    const ggml::hrx::Value * hrx_norm_value = imported_cpu_to_hrx.graph.values().find_tensor(hrx_norm);
    REQUIRE(cpu_sin_value != nullptr);
    REQUIRE(hrx_norm_value != nullptr);
    REQUIRE(cpu_sin_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(hrx_norm_value->kind == ggml::hrx::ValueKind::External);

    ggml_tensor * hrx_norm_for_cpu = ggml_rms_norm(ctx, input, 1.0e-6f);
    ggml_tensor * cpu_sin_after    = ggml_sin(ctx, hrx_norm_for_cpu);
    REQUIRE(hrx_norm_for_cpu != nullptr);
    REQUIRE(cpu_sin_after != nullptr);

    ggml_cgraph * hrx_to_cpu = ggml_new_graph(ctx);
    REQUIRE(hrx_to_cpu != nullptr);
    ggml_graph_add_node(hrx_to_cpu, hrx_norm_for_cpu);

    ggml::hrx::GraphImportResult imported_hrx_to_cpu = ggml::hrx::import_ggml_graph(*hrx_to_cpu);
    REQUIRE(imported_hrx_to_cpu.valid());
    REQUIRE(imported_hrx_to_cpu.graph.nodes().size() == 1);
    REQUIRE(imported_hrx_to_cpu.graph.nodes()[0].op == GGML_OP_RMS_NORM);

    const ggml::hrx::Value * input_value  = imported_hrx_to_cpu.graph.values().find_tensor(input);
    const ggml::hrx::Value * output_value = imported_hrx_to_cpu.graph.values().find_tensor(hrx_norm_for_cpu);
    REQUIRE(input_value != nullptr);
    REQUIRE(output_value != nullptr);
    REQUIRE(input_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(output_value->kind == ggml::hrx::ValueKind::External);

    ggml_tensor * cpu_sin_for_view   = ggml_sin(ctx, input);
    ggml_tensor * cpu_sin_view       = ggml_reshape_2d(ctx, cpu_sin_for_view, 8, 4);
    ggml_tensor * hrx_norm_from_view = ggml_rms_norm(ctx, cpu_sin_view, 1.0e-6f);
    REQUIRE(cpu_sin_for_view != nullptr);
    REQUIRE(cpu_sin_view != nullptr);
    REQUIRE(hrx_norm_from_view != nullptr);

    ggml_cgraph * cpu_view_to_hrx = ggml_new_graph(ctx);
    REQUIRE(cpu_view_to_hrx != nullptr);
    ggml_graph_add_node(cpu_view_to_hrx, cpu_sin_view);
    ggml_graph_add_node(cpu_view_to_hrx, hrx_norm_from_view);

    ggml::hrx::GraphImportResult imported_cpu_view_to_hrx = ggml::hrx::import_ggml_graph(*cpu_view_to_hrx);
    REQUIRE(imported_cpu_view_to_hrx.valid());
    REQUIRE(imported_cpu_view_to_hrx.graph.nodes().size() == 2);

    const ggml::hrx::Value * cpu_sin_for_view_value =
        imported_cpu_view_to_hrx.graph.values().find_tensor(cpu_sin_for_view);
    const ggml::hrx::Value * cpu_sin_view_value = imported_cpu_view_to_hrx.graph.values().find_tensor(cpu_sin_view);
    REQUIRE(cpu_sin_for_view_value != nullptr);
    REQUIRE(cpu_sin_view_value != nullptr);
    REQUIRE(cpu_sin_for_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(cpu_sin_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported_cpu_view_to_hrx.graph.values().same_storage(cpu_sin_for_view_value->id, cpu_sin_view_value->id));

    ggml_tensor * hrx_norm_for_view = ggml_rms_norm(ctx, input, 1.0e-6f);
    ggml_tensor * hrx_norm_view     = ggml_reshape_2d(ctx, hrx_norm_for_view, 8, 4);
    ggml_tensor * cpu_sin_view_out  = ggml_sin(ctx, hrx_norm_view);
    REQUIRE(hrx_norm_for_view != nullptr);
    REQUIRE(hrx_norm_view != nullptr);
    REQUIRE(cpu_sin_view_out != nullptr);

    ggml_cgraph * hrx_view_to_cpu = ggml_new_graph(ctx);
    REQUIRE(hrx_view_to_cpu != nullptr);
    ggml_build_forward_expand(hrx_view_to_cpu, hrx_norm_view);

    ggml::hrx::GraphImportResult imported_hrx_view_to_cpu = ggml::hrx::import_ggml_graph(*hrx_view_to_cpu);
    REQUIRE(imported_hrx_view_to_cpu.valid());

    const ggml::hrx::Value * hrx_norm_for_view_value =
        imported_hrx_view_to_cpu.graph.values().find_tensor(hrx_norm_for_view);
    const ggml::hrx::Value * hrx_norm_view_value = imported_hrx_view_to_cpu.graph.values().find_tensor(hrx_norm_view);
    REQUIRE(hrx_norm_for_view_value != nullptr);
    REQUIRE(hrx_norm_view_value != nullptr);
    REQUIRE(hrx_norm_for_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(hrx_norm_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported_hrx_view_to_cpu.graph.values().same_storage(hrx_norm_for_view_value->id, hrx_norm_view_value->id));

    ggml_free(ctx);
}

static void run_graph_view_external_use_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * shared = ggml_add(ctx, a, b);
    ggml_tensor * left   = ggml_mul(ctx, shared, c);
    ggml_tensor * right  = ggml_sqr(ctx, shared);
    ggml_tensor * output = ggml_add(ctx, left, right);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(shared != nullptr);
    REQUIRE(left != nullptr);
    REQUIRE(right != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    REQUIRE(graph->n_nodes == 4);
    REQUIRE(graph->nodes[0] == shared);
    REQUIRE(graph->nodes[1] == left);

    ggml_cgraph                  graph_view = ggml_graph_view(graph, 0, 2);
    ggml::hrx::GraphImportResult imported   = ggml::hrx::import_ggml_graph(graph_view);
    REQUIRE(imported.valid());

    const ggml::hrx::Value * shared_value = imported.graph.values().find_tensor(shared);
    const ggml::hrx::Value * left_value   = imported.graph.values().find_tensor(left);
    REQUIRE(shared_value != nullptr);
    REQUIRE(left_value != nullptr);
    REQUIRE(shared_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(left_value->kind == ggml::hrx::ValueKind::External);

    ggml_free(ctx);
}

static ggml::hrx::Dispatch schedule_single_dispatch_for_tensor(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    return scheduler.plan().dispatches.front();
}

static bool can_schedule_tensor(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static void require_compile_param(const ggml::hrx::Dispatch & dispatch, const char * name, const char * value) {
    REQUIRE(dispatch.kernel.compile_parameters.at(name) == value);
}

static void run_scale_f32_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 18);
    ggml_tensor * output = ggml_scale(ctx, input, 1.75f);
    REQUIRE(input != nullptr);
    REQUIRE(output != nullptr);

    const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_scale_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 69120);
    require_compile_parameter(dispatch, "ggml.scale_f32.scale", expected_config_value(1.75f));
    require_compile_parameter(dispatch, "ggml.scale_f32.bias", expected_config_value(0.0f));

    ggml_tensor * biased = ggml_scale_bias(ctx, input, -0.5f, 0.25f);
    REQUIRE(biased != nullptr);

    const ggml::hrx::Dispatch bias_dispatch = schedule_single_dispatch_for_tensor(ctx, biased);
    REQUIRE(kernel_name_for_id(bias_dispatch.kernel.kernel_id) == "loom_libs:ggml_scale_f32");
    REQUIRE(bias_dispatch.kernel.integer_parameters.at("element_count") == 69120);
    require_compile_parameter(bias_dispatch, "ggml.scale_f32.scale", expected_config_value(-0.5f));
    require_compile_parameter(bias_dispatch, "ggml.scale_f32.bias", expected_config_value(0.25f));

    ggml_tensor * inplace = ggml_scale_bias_inplace(ctx, input, 2.0f, -1.0f);
    REQUIRE(inplace != nullptr);

    const ggml::hrx::Dispatch inplace_dispatch = schedule_single_dispatch_for_tensor(ctx, inplace);
    REQUIRE(kernel_name_for_id(inplace_dispatch.kernel.kernel_id) == "loom_libs:ggml_scale_bias_f32");
    REQUIRE(inplace_dispatch.kernel.integer_parameters.at("element_count") == 69120);
    require_compile_parameter(inplace_dispatch, "ggml.scale.scale", expected_config_value(2.0f));
    require_compile_parameter(inplace_dispatch, "ggml.scale.bias", expected_config_value(-1.0f));

    ggml_free(ctx);
}

static void require_scale_add_dispatch(ggml_context * ctx,
                                       ggml_tensor *  input,
                                       ggml_tensor *  residual,
                                       int64_t        expected_element_count,
                                       int64_t        expected_input_stride1,
                                       int64_t               expected_residual_stride1,
                                       ggml::hrx::BinaryKind binary_kind = ggml::hrx::BinaryKind::Add,
                                       bool                  scaled_lhs  = true) {
    ggml_tensor * scaled = ggml_scale_bias(ctx, input, 0.177800179f, 0.25f);
    ggml_tensor * output = nullptr;
    switch (binary_kind) {
        case ggml::hrx::BinaryKind::Add:
            output = ggml_add(ctx, scaled, residual);
            break;
        case ggml::hrx::BinaryKind::Sub:
            output = scaled_lhs ? ggml_sub(ctx, scaled, residual) : ggml_sub(ctx, residual, scaled);
            break;
        case ggml::hrx::BinaryKind::Mul:
            output = ggml_mul(ctx, scaled, residual);
            break;
        case ggml::hrx::BinaryKind::Div:
            output = scaled_lhs ? ggml_div(ctx, scaled, residual) : ggml_div(ctx, residual, scaled);
            break;
        default:
            REQUIRE(false);
    }
    REQUIRE(scaled != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    require_scheduled_command_program(graph, [&](const ggml::hrx::Graph &, const ggml::hrx::CommandPlan & plan,
            const ggml::hrx::CommandProgram & commands) {
            REQUIRE(plan.dispatches.size() == 1);
            const ggml::hrx::Dispatch & dispatch = plan.dispatches.front();
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_scale_add_f32");
            REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == expected_element_count);
            require_compile_parameter(dispatch, "ggml.scale_add_f32.scale", expected_config_value(0.177800179f));
            require_compile_parameter(dispatch, "ggml.scale_add_f32.bias", expected_config_value(0.25f));
        require_compile_parameter(dispatch, "ggml.scale_add_f32.binary_op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(binary_kind)));
        require_compile_parameter(dispatch, "ggml.scale_add_f32.scaled_lhs", scaled_lhs ? "1" : "0");
        require_compile_parameter(dispatch, "ggml.scale_add_f32.input_stride1", std::to_string(expected_input_stride1));
            require_compile_parameter(dispatch, "ggml.scale_add_f32.residual_stride1",
                                      std::to_string(expected_residual_stride1));
            REQUIRE(dispatch.bindings.size() == 3);
            REQUIRE(commands.commands.size() == 1);
            REQUIRE(commands.commands.front().bindings[0].name == "input");
            REQUIRE(commands.commands.front().bindings[1].name == "residual");
            REQUIRE(commands.commands.front().bindings[2].name == "output");
        });
}

static void run_scale_add_f32_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * packed_input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 64);
    ggml_tensor * packed_residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 64);
    REQUIRE(packed_input != nullptr);
    REQUIRE(packed_residual != nullptr);
    require_scale_add_dispatch(ctx, packed_input, packed_residual, 163840, 2560, 2560);
    require_scale_add_dispatch(ctx, packed_input, packed_residual, 163840, 2560, 2560, ggml::hrx::BinaryKind::Sub,
                               false);
    require_scale_add_dispatch(ctx, packed_input, packed_residual, 163840, 2560, 2560, ggml::hrx::BinaryKind::Mul,
                               true);
    require_scale_add_dispatch(ctx, packed_input, packed_residual, 163840, 2560, 2560, ggml::hrx::BinaryKind::Div,
                               true);

    ggml_tensor * input_storage    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 320, 3);
    ggml_tensor * residual_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 3);
    ggml_tensor * strided_input    = ggml_view_2d(ctx, input_storage, 257, 3, input_storage->nb[1], 0);
    ggml_tensor * strided_residual = ggml_view_2d(ctx, residual_storage, 257, 3, residual_storage->nb[1], 0);
    REQUIRE(strided_input != nullptr);
    REQUIRE(strided_residual != nullptr);
    require_scale_add_dispatch(ctx, strided_input, strided_residual, 771, 320, 384);

    ggml_tensor * scaled       = ggml_scale(ctx, packed_input, 0.5f);
    ggml_tensor * alias_output = ggml_add_inplace(ctx, scaled, packed_residual);
    REQUIRE(scaled != nullptr);
    REQUIRE(alias_output != nullptr);
    REQUIRE(!can_schedule_tensor(ctx, alias_output));

    ggml_tensor * alias_scaled      = ggml_scale_inplace(ctx, packed_input, 0.5f);
    ggml_tensor * output_from_alias = ggml_add(ctx, alias_scaled, packed_residual);
    REQUIRE(alias_scaled != nullptr);
    REQUIRE(output_from_alias != nullptr);
    ggml_cgraph * alias_graph = ggml_new_graph(ctx);
    REQUIRE(alias_graph != nullptr);
    ggml_build_forward_expand(alias_graph, output_from_alias);
    ggml::hrx::GraphImportResult alias_imported = ggml::hrx::import_ggml_graph(*alias_graph);
    REQUIRE(alias_imported.valid());
    ggml::hrx::DispatchScheduler alias_scheduler;
    REQUIRE(alias_scheduler.schedule_graph(alias_imported.graph, test_dispatch_target()));
    REQUIRE(alias_scheduler.plan().dispatches.size() == 2);
    REQUIRE(kernel_name_for_id(alias_scheduler.plan().dispatches[0].kernel.kernel_id) ==
            "loom_libs:ggml_scale_bias_f32");
    REQUIRE(kernel_name_for_id(alias_scheduler.plan().dispatches[1].kernel.kernel_id) == "loom_libs:ggml_binary_f32");

    ggml_free(ctx);
}

static ggml_tensor * make_test_binary(ggml_context *        ctx,
                                      ggml_tensor *         lhs,
                                      ggml_tensor *         rhs,
                                      ggml::hrx::BinaryKind kind) {
    switch (kind) {
        case ggml::hrx::BinaryKind::Add:
            return ggml_add(ctx, lhs, rhs);
        case ggml::hrx::BinaryKind::Sub:
            return ggml_sub(ctx, lhs, rhs);
        case ggml::hrx::BinaryKind::Mul:
            return ggml_mul(ctx, lhs, rhs);
        case ggml::hrx::BinaryKind::Div:
            return ggml_div(ctx, lhs, rhs);
        default:
            REQUIRE(false);
            return nullptr;
    }
}

static const char * expected_plain_rmsnorm_binary_kernel(int64_t hidden_size, int64_t token_count) {
    return token_count < 6 && hidden_size >= 2048 && hidden_size <= 5120 ?
               "loom_libs:ggml_rmsnorm_binary_f32_lowtoken_striped" :
               "loom_libs:ggml_rmsnorm_binary_f32";
}

static void run_rmsnorm_lowtoken_striped_dispatch_checks() {
    const struct {
        int64_t              hidden_size;
        int64_t              token_count;
        ggml::hrx::BinaryKind op;
        const char *         expected_kernel;
    } cases[] = {
        { 1024, 1, ggml::hrx::BinaryKind::Mul, "loom_libs:ggml_rmsnorm_binary_f32"                  },
        { 2048, 1, ggml::hrx::BinaryKind::Mul, "loom_libs:ggml_rmsnorm_binary_f32_lowtoken_striped" },
        { 3072, 2, ggml::hrx::BinaryKind::Mul, "loom_libs:ggml_rmsnorm_binary_f32_lowtoken_striped" },
        { 5120, 4, ggml::hrx::BinaryKind::Mul, "loom_libs:ggml_rmsnorm_binary_f32_lowtoken_striped" },
        { 5120, 6, ggml::hrx::BinaryKind::Mul, "loom_libs:ggml_rmsnorm_binary_f32"                  },
        { 2048, 1, ggml::hrx::BinaryKind::Add, "loom_libs:ggml_rmsnorm_binary_f32"                  },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size = static_cast<size_t>(test.hidden_size * test.token_count * sizeof(float) * 8 + 1024 * 1024);
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden_size, test.token_count);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden_size);
        ggml_tensor * rms = ggml_rms_norm(ctx, input, 1.0e-6f);
        ggml_tensor * output = make_test_binary(ctx, rms, weight, test.op);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == test.expected_kernel);
        REQUIRE(dispatch.bindings.size() == 3);
        require_compile_parameter(dispatch, "ggml.rmsnorm_binary_f32.hidden_size",
                                  std::to_string(test.hidden_size));
        require_compile_parameter(dispatch, "ggml.rmsnorm_binary_f32.op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(test.op)));
        ggml_free(ctx);
    }
}

static void run_rmsnorm_two_binary_dispatch_checks() {
    const struct {
        ggml::hrx::BinaryKind normalized_op;
        ggml::hrx::BinaryKind output_op;
        bool                  binary_lhs;
    } cases[] = {
        { ggml::hrx::BinaryKind::Add, ggml::hrx::BinaryKind::Mul, true  },
        { ggml::hrx::BinaryKind::Sub, ggml::hrx::BinaryKind::Sub, true  },
        { ggml::hrx::BinaryKind::Mul, ggml::hrx::BinaryKind::Div, true  },
        { ggml::hrx::BinaryKind::Div, ggml::hrx::BinaryKind::Sub, false },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 512 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        ggml_tensor * weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        ggml_tensor * rms      = ggml_rms_norm(ctx, input, 1.0e-6f);
        ggml_tensor * scaled   = make_test_binary(ctx, rms, weight, test.normalized_op);
        ggml_tensor * output   = test.binary_lhs ? make_test_binary(ctx, scaled, residual, test.output_op) :
                                                   make_test_binary(ctx, residual, scaled, test.output_op);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_mul_add_f32");
        REQUIRE(dispatch.bindings.size() == 4);
        require_compile_parameter(dispatch, "ggml.rmsnorm_mul_add_f32.normalized_op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(test.normalized_op)));
        require_compile_parameter(dispatch, "ggml.rmsnorm_mul_add_f32.output_op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(test.output_op)));
        require_compile_parameter(dispatch, "ggml.rmsnorm_mul_add_f32.binary_lhs", test.binary_lhs ? "1" : "0");
        ggml_free(ctx);
    }
}

static void run_cont_f32_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t width        = 64;
    constexpr int64_t row_count    = 40;
    constexpr int64_t token_count  = 23;
    constexpr int64_t token_stride = 4096;
    constexpr size_t  view_offset  = 8 * sizeof(float);
    const size_t      storage_elements =
        view_offset / sizeof(float) + static_cast<size_t>((token_count - 1) * token_stride + width * row_count);

    ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, storage_elements);
    ggml_tensor * view    = ggml_view_3d(ctx, storage, width, row_count, token_count, width * sizeof(float),
                                         token_stride * sizeof(float), view_offset);
    ggml_tensor * output  = ggml_cont(ctx, view);
    REQUIRE(storage != nullptr);
    REQUIRE(view != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_VIEW);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_CONT);

    const ggml::hrx::Value * view_value = imported.graph.values().find(imported.graph.nodes()[1].inputs[0]);
    REQUIRE(view_value != nullptr);
    const ggml::hrx::Value * storage_value = imported.graph.values().find(view_value->storage_root);
    REQUIRE(storage_value != nullptr);
    REQUIRE(view_value->storage_offset == view_offset);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_strided_source_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == width * row_count * token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("source_span") ==
            static_cast<int64_t>((token_count - 1) * token_stride + width * row_count));
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.ne0", std::to_string(width));
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.ne1", std::to_string(row_count));
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.ne2", std::to_string(token_count));
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.stride0", "1");
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.stride1", std::to_string(width));
    require_compile_parameter(dispatch, "ggml.copy_strided_source_f32.stride2", std::to_string(token_stride));
    REQUIRE(dispatch.bindings.size() == 2);
    REQUIRE(dispatch.bindings[0].value == storage_value->id);
    REQUIRE(dispatch.bindings[0].offset == view_offset);
    REQUIRE(dispatch.bindings[0].length ==
            static_cast<size_t>((token_count - 1) * token_stride + width * row_count) * sizeof(float));

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_binary_f32_broadcast_dispatch_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * lhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * rhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * output = ggml_add(ctx, lhs, rhs);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 48);
        require_compile_param(dispatch, "ggml.binary_f32.ne0", "16");
        require_compile_param(dispatch, "ggml.binary_f32.src0_stride1", "16");
        require_compile_param(dispatch, "ggml.binary_f32.src1_stride1", "16");
        REQUIRE(dispatch.kernel.integer_parameters.count("src0_element_count") == 0);
        require_compile_param(dispatch, "ggml.binary_f32.op", "0");
        REQUIRE(dispatch.kernel.compile_parameters.count("ggml.binary_bc_f32.src0_broadcast_dim0") == 0);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * source = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 3, 2);
        REQUIRE(source != nullptr);
        ggml_tensor * lhs = ggml_view_2d(ctx, source, 2048, 2, source->nb[2], 0);
        ggml_tensor * rhs = ggml_view_2d(ctx, source, 2048, 2, source->nb[2], 2 * source->nb[1]);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        ggml_tensor * output = ggml_mul(ctx, lhs, rhs);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 4096);
        require_compile_param(dispatch, "ggml.binary_f32.src0_stride1", "6144");
        require_compile_param(dispatch, "ggml.binary_f32.src1_stride1", "6144");
        require_compile_param(dispatch, "ggml.binary_f32.src0_span", "8192");
        require_compile_param(dispatch, "ggml.binary_f32.src1_span", "8192");
        require_compile_param(dispatch, "ggml.binary_f32.op", "2");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 3);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 48);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 16);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 3);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 48);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 16);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "2");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim1", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "1");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * numerator   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * denominator = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
        ggml_tensor * output      = ggml_div(ctx, numerator, denominator);
        REQUIRE(numerator != nullptr);
        REQUIRE(denominator != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 32);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 8);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 4);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 32);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 4);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "3");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src0_broadcast_dim0", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "1");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "0");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 6, 7);
        ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 6, 7);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);

        const ggml::hrx::Dispatch dispatch = schedule_single_dispatch_for_tensor(ctx, output);
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_binary_bc_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == 210);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne0") == 5);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne1") == 6);
        REQUIRE(dispatch.kernel.integer_parameters.at("ne2") == 7);
        REQUIRE(dispatch.kernel.integer_parameters.at("src0_element_count") == 210);
        REQUIRE(dispatch.kernel.integer_parameters.at("src1_element_count") == 42);
        require_compile_param(dispatch, "ggml.binary_bc_f32.op", "2");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim0", "1");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim1", "0");
        require_compile_param(dispatch, "ggml.binary_bc_f32.src1_broadcast_dim2", "0");

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
        ggml_tensor * output = ggml_mul(ctx, input, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(output != nullptr);
        REQUIRE(!can_schedule_tensor(ctx, output));

        ggml_free(ctx);
    }
}

static void run_binary_q8_publication_checks() {
    const struct {
        int64_t      hidden;
        int64_t      tokens;
        bool         broadcast;
        bool         multiply;
        bool         side_use;
        int          projections;
        const char * expected_kernel;
    } cases[] = {
        { 2048, 2, false, true,  false, 1, "loom_libs:ggml_binary_f32_publish_q8_1_x4"    },
        { 2048, 4, false, false, true,  2, "loom_libs:ggml_binary_f32_publish_q8_1_x4"    },
        { 256,  2, true,  true,  false, 1, "loom_libs:ggml_binary_bc_f32_publish_q8_1_x4" },
        { 2048, 6, false, true,  false, 1, "loom_libs:ggml_binary_f32"                    },
        { 2048, 2, false, true,  false, 0, "loom_libs:ggml_binary_f32"                    },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * lhs    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens);
        ggml_tensor * rhs    = test.broadcast ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden) :
                                                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens);
        ggml_tensor * binary = test.multiply ? ggml_mul(ctx, lhs, rhs) : ggml_add(ctx, lhs, rhs);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        REQUIRE(binary != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        for (int i = 0; i < test.projections; ++i) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, test.hidden, 512);
            ggml_build_forward_expand(graph, ggml_mul_mat(ctx, weight, binary));
        }
        if (test.projections == 0 || test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, binary, 0.5f));
        }

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const ggml::hrx::Value * binary_value = imported.graph.values().find_tensor(binary);
        REQUIRE(binary_value != nullptr);
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return dispatch.bindings.size() >= 3 && dispatch.bindings[2].value == binary_value->id;
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(kernel_name_for_id(producer->kernel.kernel_id) == test.expected_kernel);

        const bool publishes = std::string_view(test.expected_kernel).find("publish_q8_1_x4") != std::string_view::npos;
        const size_t q8_bytes = qwen_q8_1_x4_size(test.tokens, test.hidden);
        const auto * q8       = plan.metadata.find_alternate_value(binary_value->id, GGML_TYPE_Q8_1, q8_bytes);
        REQUIRE((q8 != nullptr) == publishes);
        if (publishes) {
            REQUIRE(producer->bindings.size() == 4);
            REQUIRE(producer->bindings.back().value == q8->alternate_value);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                        return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32";
                    }) == 0);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                        return &dispatch != &*producer && !dispatch.bindings.empty() &&
                               dispatch.bindings.front().value == q8->alternate_value;
                    }) == test.projections);
        }

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_graph_snapshot_diagnostics_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    const std::string snapshot_json = ggml::hrx::serialize_graph_snapshot_json(imported.graph, "gfx1151", 42);
    REQUIRE(string_contains(snapshot_json, "ggml-hrx-graph-snapshot-v1"));
    REQUIRE(string_contains(snapshot_json, "ADD"));

    ggml::hrx::GraphSnapshotLoadResult loaded = ggml::hrx::load_graph_snapshot_json(snapshot_json);
    REQUIRE(loaded.valid());
    REQUIRE(loaded.uid == 42);
    REQUIRE(loaded.target == "gfx1151");
    REQUIRE(loaded.graph.nodes().size() == imported.graph.nodes().size());
    REQUIRE(loaded.graph.values().size() == imported.graph.values().size());

    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(scheduler.schedule_graph(loaded.graph, { loaded.target }, &diagnostics));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        loaded.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), loaded.target);
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_unmatched_graph_diagnostics_checks() {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 512);
    ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    REQUIRE(cache != nullptr);
    REQUIRE(rows != nullptr);
    REQUIRE(indices != nullptr);
    ggml_tensor * out = ggml_set_rows(ctx, cache, rows, indices);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes().front().op == GGML_OP_SET_ROWS);

    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target(), &diagnostics));
    REQUIRE(diagnostics.unsupported_node != nullptr);
    REQUIRE(diagnostics.unsupported_node_index == 0);
    REQUIRE(diagnostics.unsupported_node->op == GGML_OP_SET_ROWS);
    REQUIRE(!diagnostics.match.attempts.empty());
    REQUIRE(status_contains(scheduler.plan().status, "unsupported HRX node 0: SET_ROWS"));

    const std::string diagnostics_text =
        ggml::hrx::format_schedule_diagnostics_text(imported.graph, scheduler.plan(), diagnostics);
    REQUIRE(string_contains(diagnostics_text, "unsupported_node=0:SET_ROWS"));
    REQUIRE(string_contains(diagnostics_text, "matcher_attempts="));
    const std::string diagnostics_json =
        ggml::hrx::serialize_schedule_diagnostics_json(imported.graph, scheduler.plan(), diagnostics);
    REQUIRE(string_contains(diagnostics_json, "SET_ROWS"));
    REQUIRE(string_contains(diagnostics_json, "matcher_attempts"));

    ggml_free(ctx);
}

static void run_completion_counter_plan_checks() {
    ggml::hrx::Graph         graph;
    ggml::hrx::CommandPlan   plan;
    const ggml::hrx::ValueId first_counter(1000);
    const ggml::hrx::ValueId second_counter(1001);
    plan.completion_counter_requests.push_back({ first_counter, "first_completion_counter", 1 });
    plan.completion_counter_requests.push_back({ second_counter, "second_completion_counters", 2 });

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.empty());
    REQUIRE(commands.constant_initializations.empty());
    REQUIRE(commands.completion_counters.count == 3);
    REQUIRE(commands.completion_counters.arena_offset == 0);
    REQUIRE(commands.completion_counters.byte_count == 24);
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 256);
    REQUIRE(command_program_verifies(commands));

    const ggml::hrx::TransientAllocation * first_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, first_counter);
    const ggml::hrx::TransientAllocation * second_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, second_counter);
    REQUIRE(first_allocation != nullptr);
    REQUIRE(second_allocation != nullptr);
    REQUIRE(first_allocation->size == sizeof(int32_t));
    REQUIRE(first_allocation->alignment == 16);
    REQUIRE(first_allocation->arena_offset == commands.completion_counters.arena_offset);
    REQUIRE(second_allocation->size == 2 * sizeof(int32_t));
    REQUIRE(second_allocation->alignment == 16);
    REQUIRE(second_allocation->arena_offset == 16);

    ggml::hrx::CommandPlan bound_plan;
    bound_plan.completion_counter_requests.push_back({ first_counter, "first_bound_completion_counter", 1 });
    bound_plan.completion_counter_requests.push_back({ second_counter, "second_bound_completion_counter", 1 });
    ggml::hrx::Dispatch first_dispatch;
    first_dispatch.kernel =
        ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref("loom_libs", "ggml_binary_f32"));
    add_binary_f32_exact_dispatch_params(first_dispatch, 1);
    first_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    first_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    first_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    bound_plan.dispatches.push_back(std::move(first_dispatch));
    ggml::hrx::Dispatch second_dispatch;
    second_dispatch.kernel =
        ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref("loom_libs", "ggml_binary_f32"));
    add_binary_f32_exact_dispatch_params(second_dispatch, 1);
    second_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    second_dispatch.bindings.push_back({ first_counter, 0, sizeof(int32_t) });
    second_dispatch.bindings.push_back({ second_counter, 0, sizeof(int32_t) });
    bound_plan.dispatches.push_back(std::move(second_dispatch));

    const ggml::hrx::CommandProgram bound_commands =
        ggml::hrx::build_command_program(graph, bound_plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(bound_commands.valid());
    REQUIRE(bound_commands.commands.size() == 2);
    REQUIRE(bound_commands.completion_counters.count == 2);
    REQUIRE(bound_commands.completion_counters.arena_offset == 0);
    REQUIRE(bound_commands.completion_counters.byte_count == 20);
    REQUIRE(command_program_verifies(bound_commands));

    const ggml::hrx::TransientArenaAllocationRef transient_arena = {
        dummy_hrx_buffer(0x8000),
        bound_commands.transients.arena_size,
        7,
    };
    ggml::hrx::PreparedCommandProgram prepared_shape;
    for (const ggml::hrx::Command & prepared_source : bound_commands.commands) {
        ggml::hrx::PreparedCommand prepared_command;
        prepared_command.ordinal               = prepared_source.ordinal;
        prepared_command.kind                  = prepared_source.kind;
        prepared_command.kernel.specialization = prepared_source.kernel;
        for (const ggml::hrx::CommandBinding & binding : prepared_source.bindings) {
            prepared_command.kernel.bindings.push_back({
                binding, { dummy_hrx_buffer(0x4000), 123, binding.length }
            });
        }
        prepared_shape.commands.push_back(std::move(prepared_command));
    }
    prepared_shape.bound_transient_arena_allocation_id = 1;

    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(bound_commands, transient_arena, prepared_shape));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[0].ref.offset == 0);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[1].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[1].ref.offset == 16);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 16);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[1].ref.offset == 0);
}

static void run_graph_index_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    REQUIRE(input != nullptr);
    REQUIRE(weight != nullptr);
    ggml_tensor * rms = ggml_rms_norm(ctx, input, 0.000001f);
    REQUIRE(rms != nullptr);
    ggml_tensor * out = ggml_mul(ctx, rms, weight);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.has_index());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::GraphNode * rms_node = &imported.graph.nodes()[0];
    const ggml::hrx::GraphNode * mul_node = &imported.graph.nodes()[1];
    REQUIRE(rms_node->op == GGML_OP_RMS_NORM);
    REQUIRE(mul_node->op == GGML_OP_MUL);
    const ggml::hrx::RmsNormParams * rms_params = ggml::hrx::op_params_as<ggml::hrx::RmsNormParams>(rms_node->params);
    REQUIRE(rms_params != nullptr);
    REQUIRE(rms_params->eps == 0.000001f);
    REQUIRE(imported.graph.index().producer(rms_node->output) == rms_node);
    REQUIRE(imported.graph.index().producer(mul_node->output) == mul_node);
    REQUIRE(imported.graph.index().has_single_consumer(rms_node->output));
    const std::vector<const ggml::hrx::GraphNode *> & consumers = imported.graph.index().consumers(rms_node->output);
    REQUIRE(consumers.size() == 1);
    REQUIRE(consumers.front() == mul_node);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    ggml_free(ctx);

    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ctx             = ggml_init(params);
    REQUIRE(ctx != nullptr);

    input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
    REQUIRE(input != nullptr);
    rms = ggml_rms_norm(ctx, input, 0.00001f);
    REQUIRE(rms != nullptr);

    graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, rms);

    imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);

    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);
    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 4);
    require_compile_parameter(dispatch, "ggml.rmsnorm_f32.hidden_size", "256");
    require_compile_parameter(dispatch, "ggml.rmsnorm_f32.input_stride", "256");
    REQUIRE(dispatch.kernel.compile_parameters.count("ggml.rmsnorm_f32.rms_epsilon") == 1);
    REQUIRE(dispatch.bindings.size() == 2);

    {
        constexpr int64_t view_hidden_size = 256;
        constexpr int64_t view_tokens      = 23;
        constexpr int64_t view_stride      = 512;
        constexpr size_t  view_offset      = 4 * sizeof(float);
        ggml_tensor *     storage          = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, view_offset / sizeof(float) + (view_tokens - 1) * view_stride + view_hidden_size);
        ggml_tensor * view =
            ggml_view_2d(ctx, storage, view_hidden_size, view_tokens, view_stride * sizeof(float), view_offset);
        ggml_tensor * norm = ggml_rms_norm(ctx, view, 0.00001f);
        REQUIRE(storage != nullptr);
        REQUIRE(view != nullptr);
        REQUIRE(norm != nullptr);

        ggml_cgraph * view_graph = ggml_new_graph(ctx);
        REQUIRE(view_graph != nullptr);
        ggml_build_forward_expand(view_graph, norm);

        ggml::hrx::GraphImportResult view_imported = ggml::hrx::import_ggml_graph(*view_graph);
        REQUIRE(view_imported.valid());
        REQUIRE(view_imported.graph.nodes().size() == 2);
        REQUIRE(view_imported.graph.nodes()[0].op == GGML_OP_VIEW);
        REQUIRE(view_imported.graph.nodes()[1].op == GGML_OP_RMS_NORM);

        const ggml::hrx::Value * view_value =
            view_imported.graph.values().find(view_imported.graph.nodes()[1].inputs[0]);
        REQUIRE(view_value != nullptr);
        const ggml::hrx::Value * storage_value = view_imported.graph.values().find(view_value->storage_root);
        REQUIRE(storage_value != nullptr);
        REQUIRE(view_value->storage_offset == view_offset);

        REQUIRE(scheduler.schedule_graph(view_imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & view_dispatch = scheduler.plan().dispatches[0];
        REQUIRE(kernel_name_for_id(view_dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_f32");
        REQUIRE(view_dispatch.kernel.integer_parameters.at("token_count") == view_tokens);
        require_compile_parameter(view_dispatch, "ggml.rmsnorm_f32.hidden_size", std::to_string(view_hidden_size));
        require_compile_parameter(view_dispatch, "ggml.rmsnorm_f32.input_stride", std::to_string(view_stride));
        REQUIRE(view_dispatch.bindings.size() == 2);
        REQUIRE(view_dispatch.bindings[0].value == storage_value->id);
        REQUIRE(view_dispatch.bindings[0].offset == view_offset);
        REQUIRE(view_dispatch.bindings[0].length == static_cast<size_t>(view_tokens - 1) * view->nb[1] +
                                                        static_cast<size_t>(view_hidden_size) * sizeof(float));

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            view_imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t hidden_size  = 256;
        constexpr int64_t token_count  = 64;
        constexpr int64_t input_stride = 288;
        const size_t storage_elements = static_cast<size_t>(token_count - 1) * input_stride + hidden_size;
        ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, storage_elements);
        ggml_tensor *     view = ggml_view_2d(ctx, storage, hidden_size, token_count, input_stride * sizeof(float), 0);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor * norm = ggml_rms_norm(ctx, view, 0.00001f);
        ggml_tensor * scaled = ggml_mul(ctx, norm, weight);
        REQUIRE(storage != nullptr);
        REQUIRE(view != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(norm != nullptr);
        REQUIRE(scaled != nullptr);

        ggml_cgraph * strided_graph = ggml_new_graph(ctx);
        REQUIRE(strided_graph != nullptr);
        ggml_build_forward_expand(strided_graph, scaled);
        ggml::hrx::GraphImportResult strided_imported = ggml::hrx::import_ggml_graph(*strided_graph);
        REQUIRE(strided_imported.valid());
        REQUIRE(scheduler.schedule_graph(strided_imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & strided_dispatch = scheduler.plan().dispatches[0];
        REQUIRE(kernel_name_for_id(strided_dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_strided_f32");
        require_compile_parameter(strided_dispatch, "ggml.rmsnorm_binary_f32.hidden_size", "256");
        require_compile_parameter(strided_dispatch, "ggml.rmsnorm_binary_f32.input_stride", "288");
        require_compile_parameter(strided_dispatch, "ggml.rmsnorm_binary_f32.op", "2");
        REQUIRE(strided_dispatch.bindings.size() == 3);
        REQUIRE(strided_dispatch.bindings[0].length ==
                static_cast<size_t>(token_count - 1) * view->nb[1] + hidden_size * sizeof(float));

        ggml_tensor * full_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
        ggml_tensor * rejected_norm = ggml_rms_norm(ctx, view, 0.00001f);
        ggml_tensor * rejected_scaled = ggml_mul(ctx, rejected_norm, full_weight);
        REQUIRE(full_weight != nullptr);
        REQUIRE(rejected_scaled != nullptr);
        ggml_cgraph * rejected_graph = ggml_new_graph(ctx);
        REQUIRE(rejected_graph != nullptr);
        ggml_build_forward_expand(rejected_graph, rejected_scaled);
        ggml::hrx::GraphImportResult rejected_imported = ggml::hrx::import_ggml_graph(*rejected_graph);
        REQUIRE(rejected_imported.valid());
        REQUIRE(scheduler.schedule_graph(rejected_imported.graph, test_dispatch_target()));
        for (const ggml::hrx::Dispatch & rejected_dispatch : scheduler.plan().dispatches) {
            REQUIRE(kernel_name_for_id(rejected_dispatch.kernel.kernel_id) !=
                    "loom_libs:ggml_rmsnorm_binary_strided_f32");
        }

        ggml_tensor * second_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor * shared_norm = ggml_rms_norm(ctx, view, 0.00001f);
        ggml_tensor * scaled_a = ggml_mul(ctx, shared_norm, weight);
        ggml_tensor * scaled_b = ggml_add(ctx, shared_norm, second_weight);
        ggml_tensor * joined = ggml_add(ctx, scaled_a, scaled_b);
        REQUIRE(second_weight != nullptr);
        REQUIRE(joined != nullptr);
        ggml_cgraph * fanout_graph = ggml_new_graph(ctx);
        REQUIRE(fanout_graph != nullptr);
        ggml_build_forward_expand(fanout_graph, joined);
        ggml::hrx::GraphImportResult fanout_imported = ggml::hrx::import_ggml_graph(*fanout_graph);
        REQUIRE(fanout_imported.valid());
        REQUIRE(scheduler.schedule_graph(fanout_imported.graph, test_dispatch_target()));
        for (const ggml::hrx::Dispatch & fanout_dispatch : scheduler.plan().dispatches) {
            REQUIRE(kernel_name_for_id(fanout_dispatch.kernel.kernel_id) !=
                    "loom_libs:ggml_rmsnorm_binary_strided_f32");
        }
    }

    ggml_free(ctx);
}

static void run_rope_set_rows_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t head_size   = 8;
    constexpr int64_t head_count  = 2;
    constexpr int64_t token_count = 3;
    constexpr int64_t cache_rows  = 8;
    constexpr int64_t hidden_size = head_size * head_count;

    const auto check_rope_concat = [&](int64_t concat_head_count, int64_t concat_token_count, int64_t n_dims) {
        constexpr int64_t prefix_size = 64;
        constexpr int64_t rope_size   = 32;
        constexpr int64_t source_size = prefix_size + rope_size;
        const size_t      source_elements = static_cast<size_t>(source_size) * concat_head_count * concat_token_count;
        ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, source_elements);
        ggml_tensor *     prefix =
            ggml_view_3d(ctx, storage, prefix_size, concat_head_count, concat_token_count, source_size * sizeof(float),
                                            source_size * concat_head_count * sizeof(float), 0);
        ggml_tensor * rope_input =
            ggml_view_3d(ctx, storage, rope_size, concat_head_count, concat_token_count, source_size * sizeof(float),
                         source_size * concat_head_count * sizeof(float), prefix_size * sizeof(float));
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, concat_token_count);
        ggml_tensor * freqs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
        ggml_tensor * rope = ggml_rope_ext(ctx, rope_input, positions, freqs, n_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f,
                                           1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * output = ggml_concat(ctx, prefix, rope, 0);
        REQUIRE(storage != nullptr);
        REQUIRE(prefix != nullptr);
        REQUIRE(rope_input != nullptr);
        REQUIRE(positions != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_concat_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == concat_token_count);
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.prefix_size", std::to_string(prefix_size));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.rope_size", std::to_string(rope_size));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.n_dims", std::to_string(n_dims));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.head_count", std::to_string(concat_head_count));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.token_capacity", std::to_string(concat_token_count));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.input_stride1", std::to_string(source_size));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.input_stride2",
                                  std::to_string(source_size * concat_head_count));
        require_compile_parameter(dispatch, "ggml.rope_concat_f32.mode", "2");
        REQUIRE(dispatch.bindings.size() == 5);
        REQUIRE(dispatch.bindings[1].length == source_elements * sizeof(float));

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    };

    check_rope_concat(40, 64, 32);
    check_rope_concat(3, 2, 16);

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                                             0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.n_dims", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", "2");
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", "3");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");
        REQUIRE(dispatch.bindings.size() == 5);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f,
                                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.n_dims", "8");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", "2");
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", "3");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "0");
        REQUIRE(dispatch.bindings.size() == 5);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * output = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NEOX);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        REQUIRE(scheduler.plan().transients.size() == 2);
        REQUIRE(scheduler.plan().constant_initializations.size() == 2);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");
        require_compile_parameter(dispatch, "ggml.rope_f32.n_dims", "8");
    }

    {
        constexpr int64_t partial_head_size  = 128;
        constexpr int64_t partial_n_dims     = 96;
        constexpr int64_t partial_head_count = 24;
        constexpr int64_t partial_tokens     = 2;
        ggml_tensor *     input =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, partial_head_size, partial_head_count, partial_tokens);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, partial_tokens);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, partial_n_dims / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, partial_n_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f,
                                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", std::to_string(partial_head_size));
        require_compile_parameter(dispatch, "ggml.rope_f32.n_dims", std::to_string(partial_n_dims));
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", std::to_string(partial_head_count));
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", std::to_string(partial_tokens));
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t partial_head_size    = 128;
        constexpr int64_t partial_n_dims       = 96;
        constexpr int64_t partial_head_count   = 24;
        constexpr int64_t partial_tokens       = 2;
        constexpr int64_t partial_token_stride = 5120;
        constexpr size_t  input_offset         = 64;
        constexpr size_t  input_elements       = input_offset / sizeof(float) +
                                          static_cast<size_t>(partial_tokens - 1) * partial_token_stride +
                                          partial_head_size * partial_head_count;

        ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_elements);
        ggml_tensor * input =
            ggml_view_3d(ctx, storage, partial_head_size, partial_head_count, partial_tokens,
                         partial_head_size * sizeof(float), partial_token_stride * sizeof(float), input_offset);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, partial_tokens);
        ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, partial_n_dims / 2);
        ggml_tensor * output = ggml_rope_ext(ctx, input, pos, freqs, partial_n_dims, GGML_ROPE_TYPE_NEOX, 0, 10000.0f,
                                             1.0f, 0.0f, 1.19024f, 32.0f, 1.0f);
        REQUIRE(storage != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_f32");
        require_compile_parameter(dispatch, "ggml.rope_f32.head_size", std::to_string(partial_head_size));
        require_compile_parameter(dispatch, "ggml.rope_f32.n_dims", std::to_string(partial_n_dims));
        require_compile_parameter(dispatch, "ggml.rope_f32.head_count", std::to_string(partial_head_count));
        require_compile_parameter(dispatch, "ggml.rope_f32.token_capacity", std::to_string(partial_tokens));
        require_compile_parameter(dispatch, "ggml.rope_f32.input_stride1", std::to_string(partial_head_size));
        require_compile_parameter(dispatch, "ggml.rope_f32.input_stride2", std::to_string(partial_token_stride));
        require_compile_parameter(dispatch, "ggml.rope_f32.mscale", "1.19024003");
        require_compile_parameter(dispatch, "ggml.rope_f32.mode", "2");

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * raw_scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, token_count);
        ggml_tensor * scales     = ggml_scale(ctx, raw_scales, 0.5f);
        ggml_tensor * rope       = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NORMAL);
        ggml_tensor * output     = ggml_mul(ctx, rope, scales);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(raw_scales != nullptr);
        REQUIRE(scales != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        const size_t             rope_index  = producer_index_for_tensor(imported.graph, rope);
        const size_t             scale_index = producer_index_for_tensor(imported.graph, scales);
        std::vector<bool>        covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan   plan;
        ggml::hrx::DispatchMatch match;

        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, rope_index, match));
        REQUIRE(match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) == "loom_libs:ggml_rope_f32");

        covered_nodes[scale_index] = true;
        match                      = {};
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, rope_index, match));
        REQUIRE(match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) == "loom_libs:ggml_rope_token_scale_f32");
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * rope   = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NEOX);
        ggml_tensor * output = ggml_scale(ctx, rope, 0.125f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_scale_f32");
        require_compile_parameter(dispatch, "ggml.rope_scale_f32.scale", "0.125");
        REQUIRE(dispatch.bindings.size() == 5);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, token_count);
        ggml_tensor * rope   = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NORMAL);
        ggml_tensor * output = ggml_mul(ctx, rope, scales);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(scales != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_token_scale_f32");
        REQUIRE(dispatch.bindings.size() == 6);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * rope   = ggml_rope(ctx, input, pos, head_size, GGML_ROPE_TYPE_NEOX);
        ggml_tensor * output = ggml_scale(ctx, rope, 0.125f);
        ggml_tensor * side   = ggml_scale(ctx, rope, 0.25f);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(output != nullptr);
        REQUIRE(side != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml_build_forward_expand(graph, side);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 3);
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches.front().kernel.kernel_id) == "loom_libs:ggml_rope_f32");
    }

    {
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, token_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(cache != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_set_rows");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == hidden_size);
        require_compile_parameter(dispatch, "ggml.set_rows.input_format", "32");
        require_compile_parameter(dispatch, "ggml.set_rows.output_format", "16");
        REQUIRE(dispatch.bindings.size() == 3);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t row_stride = hidden_size * 5;
        constexpr size_t  row_offset = 4 * sizeof(float);
        ggml_tensor *     cache      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor *     storage    = ggml_new_tensor_1d(
            ctx, GGML_TYPE_F32, row_offset / sizeof(float) + (token_count - 1) * row_stride + hidden_size);
        ggml_tensor * rows =
            ggml_view_2d(ctx, storage, hidden_size, token_count, row_stride * sizeof(float), row_offset);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(cache != nullptr);
        REQUIRE(storage != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_VIEW);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_SET_ROWS);

        const ggml::hrx::Value * rows_value = imported.graph.values().find(imported.graph.nodes()[1].inputs[0]);
        REQUIRE(rows_value != nullptr);
        const ggml::hrx::Value * storage_value = imported.graph.values().find(rows_value->storage_root);
        REQUIRE(storage_value != nullptr);
        REQUIRE(rows_value->storage_offset == row_offset);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);

        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_set_rows");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == hidden_size);
        require_compile_parameter(dispatch, "ggml.set_rows.input_format", "32");
        require_compile_parameter(dispatch, "ggml.set_rows.output_format", "16");
        require_compile_parameter(dispatch, "ggml.set_rows.input_stride", std::to_string(row_stride));
        REQUIRE(dispatch.bindings.size() == 3);
        REQUIRE(dispatch.bindings[0].value == storage_value->id);
        REQUIRE(dispatch.bindings[0].offset == row_offset);
        REQUIRE(dispatch.bindings[0].length ==
                static_cast<size_t>(token_count - 1) * rows->nb[1] + static_cast<size_t>(hidden_size) * sizeof(float));

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * rows    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, token_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(cache != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_set_rows");
        require_compile_parameter(dispatch, "ggml.set_rows.input_format", "16");
        require_compile_parameter(dispatch, "ggml.set_rows.output_format", "16");
    }

    {
        ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * rope    = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f,
                                              0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rows    = ggml_reshape_2d(ctx, rope, hidden_size, token_count);
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);
        REQUIRE(imported.graph.nodes()[2].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_set_rows_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.output_format", "16");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.n_dims", "8");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.mode", "2");
        REQUIRE(dispatch.bindings.size() == 6);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
        ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
        ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
        ggml_tensor * rope  = ggml_rope_ext(ctx, input, pos, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f,
                                            0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rows  = ggml_reshape_2d(ctx, rope, hidden_size, token_count);
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden_size, cache_rows);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        ggml_tensor * output  = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(input != nullptr);
        REQUIRE(pos != nullptr);
        REQUIRE(freqs != nullptr);
        REQUIRE(rope != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ROPE);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);
        REQUIRE(imported.graph.nodes()[2].op == GGML_OP_SET_ROWS);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rope_set_rows_f32");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_rows);
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.output_format", "16");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.n_dims", "8");
        require_compile_parameter(dispatch, "ggml.rope_set_rows_f32.mode", "0");
        REQUIRE(dispatch.bindings.size() == 6);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    ggml_free(ctx);
}

static void run_graph_traversal_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * out0 = ggml_add(ctx, a, b);
        ggml_tensor * out1 = ggml_add(ctx, c, d);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(d != nullptr);
        REQUIRE(out0 != nullptr);
        REQUIRE(out1 != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out0);
        ggml_build_forward_expand(graph, out1);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].output == imported.graph.values().find_tensor(out0)->id);
        REQUIRE(imported.graph.nodes()[1].output == imported.graph.values().find_tensor(out1)->id);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * add_out = ggml_add(ctx, a, b);
        ggml_tensor * weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor * matmul  = ggml_mul_mat(ctx, weight, input);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(add_out != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(matmul != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, add_out);
        ggml_build_forward_expand(graph, matmul);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 1);
        REQUIRE(order[1] == 0);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * rms    = ggml_rms_norm(ctx, input, 0.000001f);
        ggml_tensor * out    = ggml_mul(ctx, rms, weight);
        REQUIRE(input != nullptr);
        REQUIRE(weight != nullptr);
        REQUIRE(rms != nullptr);
        REQUIRE(out != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, out);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_RMS_NORM);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 2);
        REQUIRE(order[0] == 0);
        REQUIRE(order[1] == 1);

        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 256 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * c     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * left  = ggml_add(ctx, a, b);
        ggml_tensor * right = ggml_mul(ctx, a, c);
        ggml_tensor * join  = ggml_add(ctx, left, right);
        REQUIRE(a != nullptr);
        REQUIRE(b != nullptr);
        REQUIRE(c != nullptr);
        REQUIRE(left != nullptr);
        REQUIRE(right != nullptr);
        REQUIRE(join != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, join);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 3);

        const size_t left_index  = producer_index_for_tensor(imported.graph, left);
        const size_t right_index = producer_index_for_tensor(imported.graph, right);
        const size_t join_index  = producer_index_for_tensor(imported.graph, join);

        const std::vector<size_t> order = traversal_indices(imported.graph);
        REQUIRE(order.size() == 3);
        REQUIRE(find_position(order, join_index) > find_position(order, left_index));
        REQUIRE(find_position(order, join_index) > find_position(order, right_index));

        ggml_free(ctx);
    }
}

static bool matmul_graph_is_supported(ggml_context * ctx, ggml_tensor * output);

static void schedule_single_matmul_command(ggml_context * ctx,
                                           ggml_tensor *  output,
                                           const char *   expected_kernel_name,
                                           int64_t        expected_token_count,
                                           int64_t        expected_input_size,
                                           int64_t        expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    const ggml_type weight_type = output->src[0]->type;
    REQUIRE(!scheduler.plan().dispatches.empty());
    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    const bool vector_kernel    = string_contains(kernel_name, "ggml_mul_mat_vector");
    const bool q6_vector_kernel = kernel_name == "loom_libs:ggml_mul_mat_vector_q6_f32_f32";
    const bool aligned_dynamic_kernel = kernel_name == "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned";
    const bool codebook_kernel = string_contains(kernel_name, "_iq1_") || string_contains(kernel_name, "_iq2_s_") ||
                                 string_contains(kernel_name, "_iq3_s_") ||
                                 string_contains(kernel_name, "_iq3_xxs_");
    const std::string expected_kernel(expected_kernel_name);
    const bool q8_capable_kernel = expected_kernel == "loom_libs:ggml_mul_mat_f32_f32_wmma" ||
                                   expected_kernel == "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32";
    const bool q8_input = !vector_kernel && expected_token_count <= 5 && expected_output_size % 64 == 0 &&
                          (weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q6_K) && q8_capable_kernel;
    const size_t command_count = q8_input ? 2 : 1;
    REQUIRE(scheduler.plan().dispatches.size() == command_count);
    if (q8_input) {
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches.front().kernel.kernel_id) ==
                "qwen3_moe:ggml_quantize_q8_1_x4_f32");
    }

    const bool        accepts_vector_token1 =
        expected_token_count == 1 && (expected_kernel == "loom_libs:ggml_mul_mat_f32_f32_wmma" ||
                                      expected_kernel == "loom_libs:ggml_mul_mat_f32_f32_decode_wave64" ||
                                      expected_kernel == "loom_libs:ggml_mul_mat_q6_k_packed_token1_f16_wmma" ||
                                      expected_kernel == "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32");
    if (vector_kernel && accepts_vector_token1) {
        REQUIRE(string_contains(kernel_name, "ggml_mul_mat_vector"));
    } else {
        REQUIRE(kernel_name == expected_kernel_name);
    }
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    if (vector_kernel) {
        REQUIRE(dispatch.bindings.size() == (codebook_kernel ? 5 : q6_vector_kernel ? 3 : 4));
        if (!q6_vector_kernel) {
            require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
        }
        require_compile_parameter(dispatch, "ggml.matmul.vector.input_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "ggml.matmul.vector.output_size", std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "ggml.matmul.vector.output_accumulation", "0");
        if (q6_vector_kernel) {
            require_compile_parameter(dispatch, "ggml.matmul.vector.q6_storage", "0");
        }
    } else if (string_contains(kernel_name, "ggml_mul_mat_q6_k_packed_token1_f16_wmma")) {
        REQUIRE(dispatch.bindings.size() == 3);
        require_compile_parameter(dispatch, "ggml.mul_mat_q6_k_packed.input_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "ggml.mul_mat_q6_k_packed.output_size",
                                  std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "ggml.mul_mat_q6_k_packed.output_accumulation", "0");
    } else if (string_contains(kernel_name, "_decode")) {
        REQUIRE(dispatch.bindings.size() == 3);
        REQUIRE(dispatch.kernel.integer_parameters.at("input_size") == expected_input_size);
        REQUIRE(dispatch.kernel.integer_parameters.at("output_size") == expected_output_size);
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.token_capacity",
                                  std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.output_capacity",
                                  std::to_string(expected_output_size));
        const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
        REQUIRE(weight != nullptr);
        require_compile_parameter(dispatch, "ggml.mul_mat_f32_f32_decode.weight_format",
                                  std::to_string(matmul_weight_format_config(weight->type)));
    } else if (string_contains(kernel_name, "ggml_mul_mat")) {
        REQUIRE(dispatch.bindings.size() == (codebook_kernel ? 4 : 3));
        if (aligned_dynamic_kernel) {
            REQUIRE(dispatch.kernel.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
            REQUIRE(dispatch.kernel.compile_parameters.count("ggml.workload.token_capacity") == 0);
        } else {
            require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
        }
        require_compile_parameter(dispatch, "ggml.mul_mat.input_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "ggml.mul_mat.output_size", std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "ggml.mul_mat.output_accumulation", "0");
        require_compile_parameter(dispatch, "ggml.mul_mat.output_unary_op",
                                  std::to_string(ggml::hrx::unary_kind_config_value(ggml::hrx::UnaryKind::Identity)));
        const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
        REQUIRE(weight != nullptr);
        require_compile_parameter(dispatch, "ggml.mul_mat.weight_format",
                                  std::to_string(q8_input ? (weight_type == GGML_TYPE_Q4_K ? 44 : 46) :
                                                           matmul_weight_format_config(weight->type)));
    } else if (string_contains(kernel_name, "dense_linear")) {
        REQUIRE(dispatch.bindings.size() == 3);
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.input_size",
                                  std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_size",
                                  std::to_string(expected_output_size));
        require_compile_parameter(dispatch, "qwen3_moe.dense_quantized.output_accumulation", "0");
    } else {
        REQUIRE(dispatch.bindings.size() == 3);
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(expected_token_count));
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", std::to_string(expected_input_size));
        require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expected_output_size));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == command_count);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() ==
            (codebook_kernel ? (vector_kernel ? 5 : 4) : vector_kernel && !q6_vector_kernel ? 4 : 3));
    REQUIRE(commands.commands.back().bindings[0].name == "input");
    REQUIRE(commands.commands.back().bindings[1].name == "weight");
    const size_t output_binding = codebook_kernel ? 3 : 2;
    if (codebook_kernel) {
        REQUIRE(string_contains(commands.commands.back().bindings[2].name, "grid"));
    }
    REQUIRE(commands.commands.back().bindings[output_binding].name == "output");
    if (vector_kernel && !q6_vector_kernel) {
        REQUIRE(commands.commands.back().bindings[output_binding + 1].name == "next_output");
    }
}

static void run_iq1_codebook_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    auto check_single = [&](ggml_type type, int64_t token_count, const char * expected_kernel) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, token_count);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches[0].kernel.kernel_id) == expected_kernel);
        REQUIRE(scheduler.plan().transients.size() == 1);
        REQUIRE(scheduler.plan().transients[0].name == "common.iq1.grid.v1");
        REQUIRE(scheduler.plan().transients[0].size == 16384);
        REQUIRE(scheduler.plan().constant_initializations.size() == 1);
        REQUIRE(scheduler.plan().constant_initializations[0].name == "common.iq1.grid.v1");
        REQUIRE(scheduler.plan().constant_initializations[0].data.size() == 16384);
        const std::array<uint8_t, 8> expected_grid_1542 = { 0, 1, 0, 0xff, 1, 0, 0xff, 1 };
        REQUIRE(std::equal(expected_grid_1542.begin(), expected_grid_1542.end(),
                           scheduler.plan().constant_initializations[0].data.begin() + 1542 * 8));

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 1);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[0].bindings.size() == (token_count == 1 ? 5 : 4));
        REQUIRE(commands.commands[0].bindings[0].name == "input");
        REQUIRE(commands.commands[0].bindings[1].name == "weight");
        REQUIRE(commands.commands[0].bindings[2].name == "iq1_grid");
        REQUIRE(commands.commands[0].bindings[3].name == "output");
    };

    check_single(GGML_TYPE_IQ1_S, 1, "loom_libs:ggml_mul_mat_vector_iq1_s_f32_f32");
    check_single(GGML_TYPE_IQ1_S, 64, "loom_libs:ggml_mul_mat_tiled_input_f32_iq1_s_publish_f32");
    check_single(GGML_TYPE_IQ1_M, 1, "loom_libs:ggml_mul_mat_vector_iq1_m_f32_f32");
    check_single(GGML_TYPE_IQ1_M, 64, "loom_libs:ggml_mul_mat_tiled_input_f32_iq1_m_publish_f32");

    for (const auto [type, disable_env] :
         { std::pair{ GGML_TYPE_IQ1_S, "GGML_HRX_DISABLE_IQ1_S_CODEBOOK_MATMUL" },
           std::pair{ GGML_TYPE_IQ1_M, "GGML_HRX_DISABLE_IQ1_M_CODEBOOK_MATMUL" } }) {
        EnvironmentVariableGuard guard(disable_env);
        REQUIRE(setenv(disable_env, "1", 1) == 0);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    ggml_tensor * iq1_s_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ1_S, 2048, 128);
    ggml_tensor * iq1_m_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ1_M, 2048, 128);
    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * iq1_s_output = ggml_mul_mat(ctx, iq1_s_weight, input);
    ggml_tensor * iq1_m_output = ggml_mul_mat(ctx, iq1_m_weight, input);
    REQUIRE(iq1_s_output != nullptr);
    REQUIRE(iq1_m_output != nullptr);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, iq1_s_output);
    ggml_build_forward_expand(graph, iq1_m_output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);
    REQUIRE(scheduler.plan().transients.size() == 1);
    REQUIRE(scheduler.plan().transients[0].name == "common.iq1.grid.v1");
    REQUIRE(scheduler.plan().constant_initializations.size() == 1);
    REQUIRE(scheduler.plan().dispatches[0].bindings[2].value == scheduler.plan().transients[0].value);
    REQUIRE(scheduler.plan().dispatches[1].bindings[2].value == scheduler.plan().transients[0].value);

    ggml_free(ctx);
}

static void schedule_fused_matmul_unary_command(ggml_context *       ctx,
                                                ggml_tensor *        output,
                                                const char *         expected_kernel_name,
                                                ggml::hrx::UnaryKind expected_unary_op,
                                                int64_t              expected_token_count,
                                                int64_t              expected_input_size,
                                                int64_t              expected_output_size) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    const ggml_type weight_type = output->src[0]->src[0]->type;
    const bool q8_input = expected_token_count <= 5 && expected_output_size % 64 == 0 &&
                          (weight_type == GGML_TYPE_Q4_K || weight_type == GGML_TYPE_Q6_K);
    const size_t command_count = q8_input ? 2 : 1;
    REQUIRE(scheduler.plan().dispatches.size() == command_count);
    if (q8_input) {
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches.front().kernel.kernel_id) ==
                "qwen3_moe:ggml_quantize_q8_1_x4_f32");
    }

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 3);
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat.output_accumulation", "0");
    require_compile_parameter(dispatch, "ggml.mul_mat.output_unary_op",
                              std::to_string(ggml::hrx::unary_kind_config_value(expected_unary_op)));
    const ggml::hrx::Value * weight = imported.graph.values().find(dispatch.bindings[1].value);
    REQUIRE(weight != nullptr);
    require_compile_parameter(dispatch, "ggml.mul_mat.weight_format",
                              std::to_string(q8_input ? (weight_type == GGML_TYPE_Q4_K ? 44 : 46) :
                                                       matmul_weight_format_config(weight->type)));
    REQUIRE(dispatch.bindings[2].value == imported.graph.nodes()[1].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == command_count);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() == 3);
    REQUIRE(commands.commands.back().bindings[0].name == "input");
    REQUIRE(commands.commands.back().bindings[1].name == "weight");
    REQUIRE(commands.commands.back().bindings[2].name == "output");
}

static void schedule_fused_matmul_swiglu_command(
    ggml_context *        ctx,
    ggml_tensor *         output,
    ggml_type             expected_gate_type,
    ggml_type             expected_up_type,
    int64_t               expected_token_count,
    int64_t               expected_input_size,
    int64_t               expected_output_size,
    ggml::hrx::BinaryKind expected_binary_op = ggml::hrx::BinaryKind::SwiGLU,
    ggml_op               expected_output_op = GGML_OP_GLU) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);
    REQUIRE(imported.graph.nodes()[2].op == expected_output_op);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    const bool low_token = expected_token_count <= 5 && expected_input_size % 256 == 0;
    const bool q8_input  = low_token && expected_gate_type == GGML_TYPE_Q4_K && expected_up_type == GGML_TYPE_Q4_K &&
                          expected_output_size % 64 == 0;
    const size_t command_count = q8_input ? 2 : 1;
    REQUIRE(scheduler.plan().dispatches.size() == command_count);
    if (q8_input) {
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches.front().kernel.kernel_id) ==
                "qwen3_moe:ggml_quantize_q8_1_x4_f32");
    }

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == (q8_input ? "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot" :
                           low_token ? "loom_libs:ggml_mul_mat_swiglu_f32_f32_lowtoken_dot" :
                                       "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32"));
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 4);
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.gate_weight_format",
                              std::to_string(q8_input ? 44 : matmul_weight_format_config(expected_gate_type)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.up_weight_format",
                              std::to_string(q8_input ? 44 : matmul_weight_format_config(expected_up_type)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.op",
                              std::to_string(ggml::hrx::binary_kind_config_value(expected_binary_op)));
    REQUIRE(dispatch.bindings[3].value == imported.graph.nodes()[2].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == command_count);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() == 4);
    REQUIRE(commands.commands.back().bindings[0].name == "input");
    REQUIRE(commands.commands.back().bindings[1].name == "gate_weight");
    REQUIRE(commands.commands.back().bindings[2].name == "up_weight");
    REQUIRE(commands.commands.back().bindings[3].name == "output");
}

static void schedule_fused_matmul_swiglu_postops_command(ggml_context *                  ctx,
                                                         ggml_tensor *                   output,
                                                         const char *                    expected_kernel_name,
                                                         size_t                          expected_node_count,
                                                         const std::vector<ggml_op> &    expected_ops,
                                                         const std::vector<const char *> expected_binding_names) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == expected_node_count);
    REQUIRE(expected_ops.size() == expected_node_count);
    for (size_t i = 0; i < expected_ops.size(); ++i) {
        REQUIRE(imported.graph.nodes()[i].op == expected_ops[i]);
    }

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == 8);
    REQUIRE(dispatch.bindings.size() == expected_binding_names.size());
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", "8");
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.input_size", "640");
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.output_size", "256");
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.gate_weight_format",
                              std::to_string(matmul_weight_format_config(GGML_TYPE_IQ4_NL)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.up_weight_format",
                              std::to_string(matmul_weight_format_config(GGML_TYPE_IQ4_NL)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.op",
                              std::to_string(ggml::hrx::binary_kind_config_value(ggml::hrx::BinaryKind::SwiGLU)));
    REQUIRE(dispatch.bindings.back().value == imported.graph.nodes().back().output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == expected_binding_names.size());
    for (size_t i = 0; i < expected_binding_names.size(); ++i) {
        REQUIRE(commands.commands.front().bindings[i].name == expected_binding_names[i]);
    }
}

static void schedule_packed_glu_command(ggml_context *        ctx,
                                        ggml_tensor *         output,
                                        ggml::hrx::BinaryKind expected_binary_op,
                                        int64_t               expected_hidden_size,
                                        int64_t               expected_token_count,
                                        bool                  expected_swapped) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GLU);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "loom_libs:ggml_binary_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("element_count") == expected_hidden_size * expected_token_count);
    require_compile_parameter(dispatch, "ggml.binary_f32.op",
                              std::to_string(ggml::hrx::binary_kind_config_value(expected_binary_op)));
    require_compile_parameter(dispatch, "ggml.binary_f32.ne0", std::to_string(expected_hidden_size));
    require_compile_parameter(dispatch, "ggml.binary_f32.ne1", std::to_string(expected_token_count));
    REQUIRE(dispatch.bindings.size() == 3);

    const size_t half_offset = static_cast<size_t>(expected_hidden_size) * sizeof(float);
    REQUIRE(dispatch.bindings[0].offset == (expected_swapped ? half_offset : 0));
    REQUIRE(dispatch.bindings[1].offset == (expected_swapped ? 0 : half_offset));
    REQUIRE(dispatch.bindings[2].value == imported.graph.nodes()[0].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "lhs");
    REQUIRE(commands.commands.front().bindings[1].name == "rhs");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static void schedule_packed_matmul_glu_command(ggml_context *        ctx,
                                               ggml_tensor *         output,
                                               ggml_type             expected_weight_type,
                                               int64_t               expected_token_count,
                                               int64_t               expected_input_size,
                                               int64_t               expected_output_size,
                                               ggml::hrx::BinaryKind expected_binary_op,
                                               bool                  expected_swapped) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_GLU);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == 4);
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.gate_weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.up_weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    require_compile_parameter(dispatch, "ggml.mul_mat_swiglu.op",
                              std::to_string(ggml::hrx::binary_kind_config_value(expected_binary_op)));

    const size_t half_offset =
        ggml_row_size(expected_weight_type, expected_input_size) * static_cast<size_t>(expected_output_size);
    REQUIRE(dispatch.bindings[1].offset == (expected_swapped ? half_offset : 0));
    REQUIRE(dispatch.bindings[2].offset == (expected_swapped ? 0 : half_offset));
    REQUIRE(dispatch.bindings[3].value == imported.graph.nodes()[1].output);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 4);
    REQUIRE(commands.commands.front().bindings[0].name == "input");
    REQUIRE(commands.commands.front().bindings[1].name == "gate_weight");
    REQUIRE(commands.commands.front().bindings[2].name == "up_weight");
    REQUIRE(commands.commands.front().bindings[3].name == "output");
}

static void schedule_fused_matmul_postops_command(ggml_context *                  ctx,
                                                  ggml_tensor *                   output,
                                                  const char *                    expected_kernel_name,
                                                  ggml_type                       expected_weight_type,
                                                  int64_t                         expected_token_count,
                                                  int64_t                         expected_input_size,
                                                  int64_t                         expected_output_size,
                                                  size_t                          expected_node_count,
                                                  const std::vector<ggml_op> &    expected_ops,
                                                  const std::vector<const char *> expected_binding_names,
                                                  bool                            expected_rmsnorm) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == expected_node_count);
    REQUIRE(expected_ops.size() == expected_node_count);
    for (size_t i = 0; i < expected_ops.size(); ++i) {
        REQUIRE(imported.graph.nodes()[i].op == expected_ops[i]);
    }

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == (expected_rmsnorm ? 2 : 1));
    REQUIRE(scheduler.plan().completion_counter_requests.empty());

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == expected_binding_names.size());
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "ggml.mul_mat_postops.weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    REQUIRE(dispatch.kernel.compile_parameters.count("ggml.mul_mat_postops.rms_epsilon") == 0);
    if (expected_rmsnorm) {
        const ggml::hrx::Dispatch & rmsnorm_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(rmsnorm_dispatch.kernel.kernel_id) ==
                expected_plain_rmsnorm_binary_kernel(expected_output_size, expected_token_count));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == (expected_rmsnorm ? 2 : 1));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == expected_binding_names.size());
    for (size_t i = 0; i < expected_binding_names.size(); ++i) {
        REQUIRE(commands.commands.front().bindings[i].name == expected_binding_names[i]);
    }
}

static void schedule_fused_llama_attention_matmul_command(ggml_context *                  ctx,
                                                          ggml_tensor *                   output,
                                                          const char *                    expected_kernel_name,
                                                          ggml_type                       expected_weight_type,
                                                          int64_t                         expected_token_count,
                                                          int64_t                         expected_input_size,
                                                          int64_t                         expected_output_size,
                                                          int64_t                         expected_head_size,
                                                          int64_t                         expected_head_count,
                                                          int64_t                         expected_cache_row_count,
                                                          int64_t                         expected_cache_format,
                                                          size_t                          expected_node_count,
                                                          const std::vector<ggml_op> &    expected_ops,
                                                          const std::vector<const char *> expected_binding_names) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == expected_node_count);
    REQUIRE(expected_ops.size() == expected_node_count);
    for (size_t i = 0; i < expected_ops.size(); ++i) {
        REQUIRE(imported.graph.nodes()[i].op == expected_ops[i]);
    }

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.bindings.size() == expected_binding_names.size());
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(expected_token_count));
    require_compile_parameter(dispatch, "llm.attention_qkv.input_size", std::to_string(expected_input_size));
    require_compile_parameter(dispatch, "llm.attention_qkv.output_size", std::to_string(expected_output_size));
    require_compile_parameter(dispatch, "llm.attention_qkv.weight_format",
                              std::to_string(matmul_weight_format_config(expected_weight_type)));
    if (expected_head_size > 0) {
        require_compile_parameter(dispatch, "llm.attention_qkv.head_size", std::to_string(expected_head_size));
        require_compile_parameter(dispatch, "llm.attention_qkv.head_count", std::to_string(expected_head_count));
    }
    if (expected_cache_row_count > 0) {
        require_compile_parameter(dispatch, "llm.attention_qkv.cache_row_count",
                                  std::to_string(expected_cache_row_count));
        require_compile_parameter(dispatch, "llm.attention_qkv.cache_output_format",
                                  std::to_string(expected_cache_format));
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == expected_binding_names.size());
    for (size_t i = 0; i < expected_binding_names.size(); ++i) {
        REQUIRE(commands.commands.front().bindings[i].name == expected_binding_names[i]);
    }
}

static void require_matmul_swiglu_falls_back(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() > 1);
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) != "loom_libs:ggml_mul_mat_swiglu_f32_f32_wmma");
    }
}

static void require_matmul_swiglu_falls_back_to_compilable_plan(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() > 1);
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) != "loom_libs:ggml_mul_mat_swiglu_f32_f32_wmma");
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));
}

static void require_matmul_root_matches_identity_single(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(!imported.graph.nodes().empty());
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_MUL_MAT);

    const std::vector<bool>               covered_nodes(imported.graph.nodes().size(), false);
    const ggml::hrx::CommandPlan          plan;
    const ggml::hrx::DispatchMatchContext context = {
        imported.graph,
        &imported.graph.nodes()[0],
        0,
        covered_nodes,
        plan,
        ggml::hrx::ValueId(static_cast<int32_t>(imported.graph.values().size())),
    };
    ggml::hrx::DispatchMatch match;
    REQUIRE(test_dispatch_registry().match(context, match));
    REQUIRE(match.covered_nodes.size() == 1);
    REQUIRE(match.covered_nodes.front() == 0);
    const ggml::hrx::Value * weight = imported.graph.values().find(imported.graph.nodes()[0].inputs[0]);
    const ggml::hrx::Value * input = imported.graph.values().find(imported.graph.nodes()[0].inputs[1]);
    REQUIRE(weight != nullptr);
    REQUIRE(input != nullptr);
    const bool q8_input = input->ne[1] <= 5 && weight->ne[1] % 64 == 0 &&
                          (weight->type == GGML_TYPE_Q4_K || weight->type == GGML_TYPE_Q6_K);
    REQUIRE(match.dispatches.size() == (q8_input ? 2 : 1));
    require_compile_parameter(match.dispatches.back(), "ggml.mul_mat.output_unary_op",
                              std::to_string(ggml::hrx::unary_kind_config_value(ggml::hrx::UnaryKind::Identity)));
}

static bool matmul_graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

static void schedule_qwen_terminal_q6k_q8_command(int64_t token_count) {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, token_count);
    ggml_tensor * norm_weight  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    ggml_tensor * vocab_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
    REQUIRE(input != nullptr);
    REQUIRE(norm_weight != nullptr);
    REQUIRE(vocab_weight != nullptr);
    ggml_tensor * rms        = ggml_rms_norm(ctx, input, 0.000001f);
    ggml_tensor * normalized = ggml_mul(ctx, rms, norm_weight);
    ggml_tensor * logits     = ggml_mul_mat(ctx, vocab_weight, normalized);
    REQUIRE(rms != nullptr);
    REQUIRE(normalized != nullptr);
    REQUIRE(logits != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, logits);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);
    REQUIRE(scheduler.plan().transients.size() == 1);

    const ggml::hrx::Dispatch & rms_dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(rms_dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4");
    REQUIRE(rms_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    require_compile_parameter(rms_dispatch, "qwen3_moe.model.hidden_size", "2048");
    require_compile_parameter(rms_dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));
    REQUIRE(rms_dispatch.bindings.size() == 4);

    const ggml::hrx::CommandPlanTransient & q8_transient = scheduler.plan().transients.front();
    REQUIRE(q8_transient.size == qwen_q8_1_x4_size(token_count, 2048));
    REQUIRE(rms_dispatch.bindings[3].value == q8_transient.value);
    REQUIRE(rms_dispatch.bindings[3].length == q8_transient.size);

    const ggml::hrx::Dispatch & vocab_dispatch = scheduler.plan().dispatches[1];
    REQUIRE(kernel_name_for_id(vocab_dispatch.kernel.kernel_id) == "qwen3_moe:ggml_linear_q6k_q8_1_x4");
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("input_size") == 2048);
    REQUIRE(vocab_dispatch.kernel.integer_parameters.at("output_size") == 151936);
    require_compile_parameter(vocab_dispatch, "ggml.linear_q6k_q8_1_x4.token_capacity", std::to_string(token_count));
    require_compile_parameter(vocab_dispatch, "ggml.linear_q6k_q8_1_x4.output_capacity", "151936");
    REQUIRE(vocab_dispatch.bindings.size() == 3);
    REQUIRE(vocab_dispatch.bindings[0].value == q8_transient.value);
    REQUIRE(vocab_dispatch.bindings[0].length == q8_transient.size);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 4);
    REQUIRE(commands.commands[0].bindings[3].name == "q8_output");
    REQUIRE(commands.commands[0].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings[0].name == "q8_input");
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == q8_transient.value);

    ggml_free(ctx);
}

static void run_qwen_decode_rmsnorm_publication_checks() {
    const struct {
        bool projection;
        bool side_use;
        bool q8;
    } cases[] = {
        { true,  false, true  },
        { true,  true,  true  },
        { false, false, false },
    };
    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 8 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        ggml_tensor * normalized  = ggml_mul(ctx, ggml_rms_norm(ctx, input, 0.000001f), norm_weight);
        ggml_tensor * projection  = nullptr;
        if (test.projection) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 512);
            projection           = ggml_mul_mat(ctx, weight, normalized);
        }
        ggml_tensor * side = test.side_use ? ggml_scale(ctx, normalized, 0.5f) : nullptr;

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, projection != nullptr ? projection : normalized);
        if (side != nullptr) {
            ggml_build_forward_expand(graph, side);
        }

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());

        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4";
        });
        REQUIRE((producer != plan.dispatches.end()) == test.q8);
        const ggml::hrx::Value * output = imported.graph.values().find_tensor(normalized);
        REQUIRE(output != nullptr);
        const auto * q8 = plan.metadata.find_alternate_value(
            output->id, GGML_TYPE_Q8_1, qwen_q8_1_x4_size(1, 2048));
        REQUIRE((q8 != nullptr) == test.q8);
        if (test.q8) {
            REQUIRE(producer->bindings.size() == 4);
            REQUIRE(producer->bindings.back().value == q8->alternate_value);
            REQUIRE(std::any_of(std::next(producer), plan.dispatches.end(), [&](const auto & dispatch) {
                return !dispatch.bindings.empty() && dispatch.bindings.front().value == q8->alternate_value;
            }));
        }
        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_qwen_decode_publication_consumer_qualification_checks() {
    enum class ConsumerCase {
        Dense,
        UnsupportedWeight,
        WrongOperand,
        IncompatibleGeometry,
        Routed,
        RoutedQ6,
        RoutedShape,
        RoutedRoute,
        AliasedDense,
    };
    const struct {
        ConsumerCase consumer;
        bool         q8;
    } cases[] = {
        { ConsumerCase::Dense,                true  },
        { ConsumerCase::UnsupportedWeight,    false },
        { ConsumerCase::WrongOperand,         false },
        { ConsumerCase::IncompatibleGeometry, false },
        { ConsumerCase::Routed,               true  },
        { ConsumerCase::RoutedQ6,             true  },
        { ConsumerCase::RoutedShape,          false },
        { ConsumerCase::RoutedRoute,          false },
        { ConsumerCase::AliasedDense,         true  },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 32 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        ggml_tensor * rms         = ggml_rms_norm(ctx, input, 0.000001f);
        ggml_tensor * normalized  = ggml_mul(ctx, rms, norm_weight);
        REQUIRE(rms != nullptr);
        REQUIRE(normalized != nullptr);

        ggml_tensor * consumer_input = normalized;
        if (test.consumer == ConsumerCase::AliasedDense) {
            consumer_input = ggml_reshape_2d(ctx, normalized, 2048, 1);
            REQUIRE(consumer_input != nullptr);
        }

        ggml_tensor * terminal = nullptr;
        if (test.consumer == ConsumerCase::Dense || test.consumer == ConsumerCase::AliasedDense) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 512);
            terminal             = ggml_mul_mat(ctx, weight, consumer_input);
        } else if (test.consumer == ConsumerCase::UnsupportedWeight) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2048, 512);
            terminal             = ggml_mul_mat(ctx, weight, normalized);
        } else if (test.consumer == ConsumerCase::WrongOperand) {
            ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
            terminal                 = ggml_mul_mat(ctx, normalized, activation);
        } else if (test.consumer == ConsumerCase::IncompatibleGeometry) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 63);
            terminal             = ggml_mul_mat(ctx, weight, normalized);
        } else {
            const int64_t route_count = test.consumer == ConsumerCase::RoutedShape ? 4 : 8;
            ggml_tensor * route_ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, 1);
            ggml_tensor * gate_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 2048, 768, 128);
            ggml_tensor * up_weight   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 2048, 768, 128);
            const ggml_type down_weight_type = test.consumer == ConsumerCase::RoutedQ6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
            ggml_tensor * down_weight = ggml_new_tensor_3d(ctx, down_weight_type, 768, 2048, 128);
            ggml_tensor * gate        = ggml_mul_mat_id(ctx, gate_weight, normalized, route_ids);
            ggml_tensor * up          = ggml_mul_mat_id(ctx, up_weight, normalized, route_ids);
            ggml_tensor * glu         = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            ggml_tensor * down_route_ids = test.consumer == ConsumerCase::RoutedRoute ?
                                               ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, 1) :
                                               route_ids;
            ggml_tensor * down = ggml_mul_mat_id(ctx, down_weight, glu, down_route_ids);
            ggml_tensor * route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, route_count, 1);
            terminal                    = ggml_mul(ctx, down, route_weights);
        }
        REQUIRE(terminal != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, terminal);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        const ggml::hrx::Value * rms_value        = imported.graph.values().find_tensor(rms);
        const ggml::hrx::Value * normalized_value = imported.graph.values().find_tensor(normalized);
        REQUIRE(rms_value != nullptr);
        REQUIRE(normalized_value != nullptr);
        const ggml::hrx::GraphNode * rms_node = imported.graph.index().producer(rms_value->id);
        size_t rms_index = 0;
        REQUIRE(rms_node != nullptr);
        REQUIRE(imported.graph.index().node_index(rms_node, rms_index));

        const ggml::hrx::CommandPlan empty_plan;
        std::vector<bool> covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::DispatchMatch match;
        REQUIRE(match_dispatch_at_index(imported.graph, empty_plan, covered_nodes, rms_index, match));
        const auto * q8 = match.metadata.find_alternate_value(
            normalized_value->id, GGML_TYPE_Q8_1, qwen_q8_1_x4_size(1, 2048));
        REQUIRE((q8 != nullptr) == test.q8);
        if (test.q8) {
            REQUIRE(kernel_name_for_id(match.dispatches.back().kernel.kernel_id) ==
                    "qwen3_moe:qwen3_moe_rmsnorm_f32_quantize_q8_1_x4");
        }
        ggml_free(ctx);
    }
}

static void schedule_qwen_terminal_q6k_vector_command() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * vocab_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
    REQUIRE(input != nullptr);
    REQUIRE(vocab_weight != nullptr);
    ggml_tensor * logits = ggml_mul_mat(ctx, vocab_weight, input);
    REQUIRE(logits != nullptr);

    schedule_single_matmul_command(ctx, logits, "loom_libs:ggml_mul_mat_vector_q6_f32_f32", 1, 2048, 151936);
    ggml_free(ctx);
}

static void schedule_get_rows_q8_1_alternate_command(ggml_type embedding_weight_type) {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * embedding_weight = ggml_new_tensor_2d(ctx, embedding_weight_type, 2048, 151936);
    ggml_tensor * token_ids        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * vocab_weight     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
    REQUIRE(embedding_weight != nullptr);
    REQUIRE(token_ids != nullptr);
    REQUIRE(vocab_weight != nullptr);

    ggml_tensor * embedding = ggml_get_rows(ctx, embedding_weight, token_ids);
    ggml_tensor * logits    = ggml_mul_mat(ctx, vocab_weight, embedding);
    REQUIRE(embedding != nullptr);
    REQUIRE(logits != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, logits);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_MUL_MAT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);
    REQUIRE(scheduler.plan().transients.size() == 1);

    const size_t q8_byte_count = qwen_q8_1_x4_size(1, 2048);

    const ggml::hrx::Dispatch & get_rows_dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(get_rows_dispatch.kernel.kernel_id) == "loom_libs:ggml_get_rows_f32_next");
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("token_count") == 1);
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("row_count") == 151936);
    REQUIRE(get_rows_dispatch.kernel.integer_parameters.at("hidden_size") == 2048);
    require_compile_parameter(get_rows_dispatch, "ggml.get_rows_f32.weight_format",
                              std::to_string(matmul_weight_format_config(embedding_weight_type)));
    require_compile_parameter(get_rows_dispatch, "ggml.get_rows_f32.next_format", "81");
    REQUIRE(get_rows_dispatch.bindings.size() == 4);

    const ggml::hrx::CommandPlanTransient & q8_transient = scheduler.plan().transients.front();
    REQUIRE(q8_transient.size == q8_byte_count);
    REQUIRE(get_rows_dispatch.bindings[3].value == q8_transient.value);
    REQUIRE(get_rows_dispatch.bindings[3].length == q8_transient.size);

    const ggml::hrx::CommandPlanAlternateValue * alternate =
        scheduler.plan().metadata.find_alternate_value(imported.graph.nodes()[0].output, GGML_TYPE_Q8_1, q8_byte_count);
    REQUIRE(alternate != nullptr);
    REQUIRE(alternate->alternate_value == q8_transient.value);

    const ggml::hrx::Dispatch & vocab_dispatch = scheduler.plan().dispatches[1];
    const std::string vocab_kernel_name = kernel_name_for_id(vocab_dispatch.kernel.kernel_id);
    REQUIRE(vocab_dispatch.bindings.size() == 3);
    if (vocab_kernel_name == "qwen3_moe:ggml_linear_q6k_q8_1_x4") {
        REQUIRE(vocab_dispatch.bindings[0].value == alternate->alternate_value);
        REQUIRE(vocab_dispatch.bindings[0].length == alternate->byte_count);
    } else {
        REQUIRE(vocab_kernel_name == "loom_libs:ggml_mul_mat_vector_q6_f32_f32");
        REQUIRE(vocab_dispatch.bindings[0].value == imported.graph.nodes()[0].output);
        require_compile_parameter(vocab_dispatch, "ggml.matmul.vector.input_size", "2048");
        require_compile_parameter(vocab_dispatch, "ggml.matmul.vector.output_size", "151936");
        require_compile_parameter(vocab_dispatch, "ggml.matmul.vector.q6_storage", "0");
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 4);
    REQUIRE(commands.commands[0].bindings[3].name == "next_output");
    REQUIRE(commands.commands[0].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings[0].name ==
            (vocab_kernel_name == "qwen3_moe:ggml_linear_q6k_q8_1_x4" ? "q8_input" : "input"));

    ggml_free(ctx);
}

static bool graph_is_supported(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(imported.graph, test_dispatch_target());
}

struct ManualQwenRouterTop8Graph {
    ggml::hrx::Graph   graph;
    ggml::hrx::ValueId route_ids;
};

static ggml::hrx::ValueId add_manual_graph_tensor(ggml::hrx::Graph &   graph,
                                                  ggml_tensor *        tensor,
                                                  ggml::hrx::ValueKind kind) {
    REQUIRE(tensor != nullptr);
    return graph.values().get_or_add_tensor_value(tensor, kind);
}

static ManualQwenRouterTop8Graph build_manual_qwen_router_top8_graph(ggml_context * ctx,
                                                                     int64_t        expert_count,
                                                                     int64_t        route_count,
                                                                     int64_t        token_count) {
    ManualQwenRouterTop8Graph manual;
    ggml::hrx::Graph &        graph = manual.graph;

    const ggml::hrx::ValueId logits = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, token_count), ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId probs = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId probs_reshaped = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId argsort = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, expert_count, token_count), ggml::hrx::ValueKind::Transient);
    manual.route_ids = add_manual_graph_tensor(graph, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count),
                                               ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId selected = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId weights_flat = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId sum = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId clamped = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId normalized = add_manual_graph_tensor(
        graph, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, route_count, token_count), ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId route_weights = add_manual_graph_tensor(
        graph, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, route_count, token_count), ggml::hrx::ValueKind::External);

    ggml::hrx::GraphNode & softmax = graph.add_node(GGML_OP_SOFT_MAX, probs, { logits });
    softmax.params                 = ggml::hrx::SoftMaxParams{ 1.0f, 0.0f };
    graph.add_node(GGML_OP_RESHAPE, probs_reshaped, { probs });
    ggml::hrx::GraphNode & argsort_node = graph.add_node(GGML_OP_ARGSORT, argsort, { probs });
    argsort_node.params                 = ggml::hrx::ArgsortParams{ GGML_SORT_ORDER_DESC };
    graph.add_node(GGML_OP_VIEW, manual.route_ids, { argsort });
    graph.add_node(GGML_OP_GET_ROWS, selected, { probs_reshaped, manual.route_ids });
    graph.add_node(GGML_OP_RESHAPE, weights_flat, { selected });
    graph.add_node(GGML_OP_SUM_ROWS, sum, { weights_flat });
    ggml::hrx::GraphNode & clamp = graph.add_node(GGML_OP_CLAMP, clamped, { sum });
    clamp.params                 = ggml::hrx::ClampParams{ 0.00006103515625f, std::numeric_limits<float>::infinity() };
    graph.add_node(GGML_OP_DIV, normalized, { weights_flat, clamped });
    graph.add_node(GGML_OP_RESHAPE, route_weights, { normalized });
    REQUIRE(graph.build_index().success());
    return manual;
}

static ggml::hrx::Graph build_manual_token_embedding_graph(ggml_tensor * weight,
                                                           ggml_tensor * token_ids,
                                                           ggml_tensor * output) {
    ggml::hrx::Graph   graph;
    ggml::hrx::ValueId weight_value = graph.values().get_or_add_tensor_value(weight, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueId token_ids_value =
        graph.values().get_or_add_tensor_value(token_ids, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueId output_value = graph.values().get_or_add_tensor_value(output, ggml::hrx::ValueKind::External);
    graph.add_node(GGML_OP_GET_ROWS, output_value, { weight_value, token_ids_value });
    REQUIRE(graph.build_index().success());
    return graph;
}

static bool manual_token_embedding_graph_is_supported(ggml_context * ctx,
                                                      ggml_type      weight_type,
                                                      ggml_type      token_ids_type,
                                                      ggml_type      output_type,
                                                      int64_t        hidden_size,
                                                      int64_t        vocabulary_count,
                                                      int64_t        token_count,
                                                      int64_t        output_hidden_size = -1,
                                                      int64_t        output_token_count = -1) {
    if (output_hidden_size < 0) {
        output_hidden_size = hidden_size;
    }
    if (output_token_count < 0) {
        output_token_count = token_count;
    }
    ggml_tensor * weight    = ggml_new_tensor_2d(ctx, weight_type, hidden_size, vocabulary_count);
    ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, token_ids_type, token_count);
    ggml_tensor * output    = ggml_new_tensor_2d(ctx, output_type, output_hidden_size, output_token_count);
    REQUIRE(weight != nullptr);
    REQUIRE(token_ids != nullptr);
    REQUIRE(output != nullptr);

    ggml::hrx::Graph graph = build_manual_token_embedding_graph(weight, token_ids, output);
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static void schedule_token_embedding_command(ggml_context * ctx,
                                             ggml_tensor *  output,
                                             int64_t        expected_token_count,
                                             int64_t        expected_vocabulary_count,
                                             int64_t        expected_hidden_size,
                                             const char *   expected_weight_format) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.front();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "loom_libs:ggml_get_rows_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == expected_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("row_count") == expected_vocabulary_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == expected_hidden_size);
    require_compile_parameter(dispatch, "ggml.get_rows_f32.weight_format", expected_weight_format);
    REQUIRE(dispatch.bindings.size() == 3);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 3);
    REQUIRE(commands.commands.front().bindings[0].name == "token_ids");
    REQUIRE(commands.commands.front().bindings[1].name == "weight");
    REQUIRE(commands.commands.front().bindings[2].name == "output");
}

static void run_qwen_token_embedding_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 1, 151936, 2048, "4");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 13);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 13, 151936, 2048, "4");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 5, 151936, 2048, "30");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, 2048, 151936);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 1, 151936, 2048, "10");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_1, 640, 262144);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 1, 262144, 640, "51");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3840, 262208);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 64, 262208, 3840, "80");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 3840, 262208);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 64, 262208, 3840, "6");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 3840, 262208);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 64, 262208, 3840, "30");
    }
    {
        ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 262144, 4);
        ggml_tensor * token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(token_ids != nullptr);
        ggml_tensor * output = ggml_get_rows(ctx, weight, token_ids);
        REQUIRE(output != nullptr);
        schedule_token_embedding_command(ctx, output, 1, 4, 262144, "32");
    }

    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q6_K, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q1_0, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q5_1, GGML_TYPE_I32, GGML_TYPE_F32, 640, 262144, 1));
    REQUIRE(manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_IQ4_XS, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936,
                                                      1));
    REQUIRE(!manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_IQ3_S, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936,
                                                       1));
    REQUIRE(!manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_IQ4_NL, GGML_TYPE_I32, GGML_TYPE_F32, 2048,
                                                       151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q8_0, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q8_1, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_BF16, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_BF16, GGML_TYPE_I32, GGML_TYPE_F32, 3840, 262208, 64));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I64, GGML_TYPE_F32, 2048, 151936, 1));
    REQUIRE(
        !manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I32, GGML_TYPE_F16, 2048, 151936, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 128, 151936, 1));
    REQUIRE(manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 262144, 4, 1));
    REQUIRE(
        manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F32, 33024, 151936, 1));
    REQUIRE(!manual_token_embedding_graph_is_supported(ctx, GGML_TYPE_Q4_K, GGML_TYPE_I32, GGML_TYPE_F32, 2048, 151936,
                                                       1, 2048, 2));

    ggml_free(ctx);
}

static void run_get_rows_rmsnorm_binary_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128256);
    ggml_tensor * token_ids        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
    ggml_tensor * norm_weight      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    REQUIRE(embedding_weight != nullptr);
    REQUIRE(token_ids != nullptr);
    REQUIRE(norm_weight != nullptr);
    ggml_tensor * embedding = ggml_get_rows(ctx, embedding_weight, token_ids);
    ggml_tensor * rms       = ggml_rms_norm(ctx, embedding, 0.00001f);
    ggml_tensor * output    = ggml_mul(ctx, rms, norm_weight);
    REQUIRE(embedding != nullptr);
    REQUIRE(rms != nullptr);
    REQUIRE(output != nullptr);
    ggml_set_output(embedding);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 1);
    REQUIRE(plan.transients.size() == 2);
    const ggml::hrx::Dispatch & dispatch = plan.dispatches.front();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_get_rows_rmsnorm_binary_q8_1_x4_f16");
    REQUIRE(dispatch.bindings.size() == 7);
    require_compile_parameter(dispatch, "ggml.get_rows_f32.weight_format", "6");
    require_compile_parameter(dispatch, "ggml.get_rows_rmsnorm.rms_epsilon", "9.99999975e-06");

    const ggml::hrx::Value * output_value = imported.graph.values().find_tensor(output);
    REQUIRE(output_value != nullptr);
    const size_t q8_bytes  = qwen_q8_1_x4_size(512, 2048);
    const size_t f16_bytes = output_value->byte_count / 2;
    const auto * q8        = plan.metadata.find_alternate_value(output_value->id, GGML_TYPE_Q8_1, q8_bytes);
    const auto * f16       = plan.metadata.find_alternate_value(output_value->id, GGML_TYPE_F16, f16_bytes);
    REQUIRE(q8 == nullptr);
    REQUIRE(f16 == nullptr);
    REQUIRE(dispatch.bindings[5].value == plan.transients[0].value);
    REQUIRE(dispatch.bindings[6].value == plan.transients[1].value);
    REQUIRE(plan.transients[0].size == q8_bytes);
    REQUIRE(plan.transients[1].size == f16_bytes);

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_get_rows_scale_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 64);
    ggml_tensor * ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
    REQUIRE(weight != nullptr);
    REQUIRE(ids != nullptr);
    ggml_tensor * rows   = ggml_get_rows(ctx, weight, ids);
    ggml_tensor * output = ggml_scale(ctx, rows, 0.177800179f);
    REQUIRE(rows != nullptr);
    REQUIRE(output != nullptr);
    ggml_set_output(rows);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    require_scheduled_command_program(
        graph, [&](const ggml::hrx::Graph & imported_graph, const ggml::hrx::CommandPlan & plan,
            const ggml::hrx::CommandProgram & commands) {
            REQUIRE(imported_graph.nodes().size() == 2);
            REQUIRE(plan.dispatches.size() == 1);
            const ggml::hrx::Dispatch & dispatch = plan.dispatches.front();
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_get_rows_scale_f32");
            REQUIRE(dispatch.bindings.size() == 4);
            require_compile_parameter(dispatch, "ggml.get_rows_f32.weight_format", "32");
            require_compile_parameter(dispatch, "ggml.get_rows_scale_f32.scale", expected_config_value(0.177800179f));
            require_compile_parameter(dispatch, "ggml.get_rows_scale_f32.bias", expected_config_value(0.0f));
            REQUIRE(commands.commands.size() == 1);
            REQUIRE(commands.commands.front().bindings[0].name == "token_ids");
            REQUIRE(commands.commands.front().bindings[1].name == "weight");
            REQUIRE(commands.commands.front().bindings[2].name == "raw_output");
            REQUIRE(commands.commands.front().bindings[3].name == "output");
        });

    ggml_tensor * q5_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_1, 640, 262144);
    ggml_tensor * q5_ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);
    REQUIRE(q5_weight != nullptr);
    REQUIRE(q5_ids != nullptr);
    ggml_tensor * q5_rows   = ggml_get_rows(ctx, q5_weight, q5_ids);
    ggml_tensor * q5_output = ggml_scale(ctx, q5_rows, 25.2982216f);
    REQUIRE(q5_rows != nullptr);
    REQUIRE(q5_output != nullptr);
    ggml_set_output(q5_rows);

    ggml_cgraph * q5_graph = ggml_new_graph(ctx);
    REQUIRE(q5_graph != nullptr);
    ggml_build_forward_expand(q5_graph, q5_output);
    require_scheduled_command_program(
        q5_graph, [&](const ggml::hrx::Graph & imported_graph, const ggml::hrx::CommandPlan & plan,
            const ggml::hrx::CommandProgram & commands) {
            REQUIRE(imported_graph.nodes().size() == 2);
            REQUIRE(plan.dispatches.size() == 1);
            const ggml::hrx::Dispatch & dispatch = plan.dispatches.front();
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_get_rows_scale_f32");
            REQUIRE(dispatch.bindings.size() == 4);
            require_compile_parameter(dispatch, "ggml.get_rows_f32.weight_format", "51");
            require_compile_parameter(dispatch, "ggml.get_rows_scale_f32.scale", expected_config_value(25.2982216f));
            require_compile_parameter(dispatch, "ggml.get_rows_scale_f32.bias", expected_config_value(0.0f));
            REQUIRE(commands.commands.size() == 1);
        });

    ggml_tensor * q5_decode_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    REQUIRE(q5_decode_ids != nullptr);
    ggml_tensor * q5_decode_rows   = ggml_get_rows(ctx, q5_weight, q5_decode_ids);
    ggml_tensor * q5_decode_output = ggml_scale(ctx, q5_decode_rows, 25.2982216f);
    REQUIRE(q5_decode_rows != nullptr);
    REQUIRE(q5_decode_output != nullptr);
    ggml_cgraph * q5_decode_graph = ggml_new_graph(ctx);
    REQUIRE(q5_decode_graph != nullptr);
    ggml_build_forward_expand(q5_decode_graph, q5_decode_output);
    const std::vector<std::string> q5_decode_kernels = scheduled_kernel_names(q5_decode_graph);
    REQUIRE(std::find(q5_decode_kernels.begin(), q5_decode_kernels.end(), "loom_libs:ggml_get_rows_scale_f32") ==
            q5_decode_kernels.end());
    REQUIRE(std::find(q5_decode_kernels.begin(), q5_decode_kernels.end(), "loom_libs:ggml_get_rows_f32") !=
            q5_decode_kernels.end());
    REQUIRE(std::find(q5_decode_kernels.begin(), q5_decode_kernels.end(), "loom_libs:ggml_scale_f32") !=
            q5_decode_kernels.end());

    ggml_tensor * q4_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 640, 64);
    ggml_tensor * q4_ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
    REQUIRE(q4_weight != nullptr);
    REQUIRE(q4_ids != nullptr);
    ggml_tensor * q4_rows   = ggml_get_rows(ctx, q4_weight, q4_ids);
    ggml_tensor * q4_output = ggml_scale(ctx, q4_rows, 0.5f);
    REQUIRE(q4_rows != nullptr);
    REQUIRE(q4_output != nullptr);
    ggml_cgraph * q4_graph = ggml_new_graph(ctx);
    REQUIRE(q4_graph != nullptr);
    ggml_build_forward_expand(q4_graph, q4_output);
    const std::vector<std::string> q4_kernels = scheduled_kernel_names(q4_graph);
    REQUIRE(std::find(q4_kernels.begin(), q4_kernels.end(), "loom_libs:ggml_get_rows_scale_f32") == q4_kernels.end());
    REQUIRE(std::find(q4_kernels.begin(), q4_kernels.end(), "loom_libs:ggml_get_rows_f32") != q4_kernels.end());

    ggml_tensor * side_use = ggml_add(ctx, rows, rows);
    REQUIRE(side_use != nullptr);
    ggml_cgraph * fallback_graph = ggml_new_graph(ctx);
    REQUIRE(fallback_graph != nullptr);
    ggml_build_forward_expand(fallback_graph, output);
    ggml_build_forward_expand(fallback_graph, side_use);
    const std::vector<std::string> fallback_kernels = scheduled_kernel_names(fallback_graph);
    REQUIRE(std::find(fallback_kernels.begin(), fallback_kernels.end(), "loom_libs:ggml_get_rows_scale_f32") ==
            fallback_kernels.end());
    REQUIRE(std::find(fallback_kernels.begin(), fallback_kernels.end(), "loom_libs:ggml_get_rows_f32") !=
            fallback_kernels.end());

    ggml_tensor * inplace_rows   = ggml_get_rows(ctx, weight, ids);
    ggml_tensor * inplace_output = ggml_scale_inplace(ctx, inplace_rows, 0.5f);
    REQUIRE(inplace_rows != nullptr);
    REQUIRE(inplace_output != nullptr);
    ggml_cgraph * inplace_graph = ggml_new_graph(ctx);
    REQUIRE(inplace_graph != nullptr);
    ggml_build_forward_expand(inplace_graph, inplace_output);
    const std::vector<std::string> inplace_kernels = scheduled_kernel_names(inplace_graph);
    REQUIRE(std::find(inplace_kernels.begin(), inplace_kernels.end(), "loom_libs:ggml_get_rows_scale_f32") ==
            inplace_kernels.end());
    REQUIRE(std::find(inplace_kernels.begin(), inplace_kernels.end(), "loom_libs:ggml_get_rows_f32") !=
            inplace_kernels.end());

    ggml_free(ctx);
}

static ggml::hrx::Graph build_manual_gather_add_graph(ggml_context * ctx,
                                                      int64_t        hidden_size,
                                                      int64_t        source_token_count,
                                                      int64_t        output_token_count,
                                                      bool           shared_row_ids,
                                                      int64_t        second_source_hidden_size = -1) {
    if (second_source_hidden_size < 0) {
        second_source_hidden_size = hidden_size;
    }

    ggml::hrx::Graph graph;

    ggml_tensor * attention = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * residual  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, second_source_hidden_size, source_token_count);
    ggml_tensor * row_ids0  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * row_ids1  = shared_row_ids ? row_ids0 : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * selected0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    ggml_tensor * selected1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    ggml_tensor * output    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, output_token_count);
    REQUIRE(attention != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(row_ids0 != nullptr);
    REQUIRE(row_ids1 != nullptr);
    REQUIRE(selected0 != nullptr);
    REQUIRE(selected1 != nullptr);
    REQUIRE(output != nullptr);

    const ggml::hrx::ValueId attention_value =
        graph.values().get_or_add_tensor_value(attention, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId residual_value =
        graph.values().get_or_add_tensor_value(residual, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids0_value =
        graph.values().get_or_add_tensor_value(row_ids0, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids1_value =
        graph.values().get_or_add_tensor_value(row_ids1, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId selected0_value =
        graph.values().get_or_add_tensor_value(selected0, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId selected1_value =
        graph.values().get_or_add_tensor_value(selected1, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId output_value =
        graph.values().get_or_add_tensor_value(output, ggml::hrx::ValueKind::External);

    graph.add_node(GGML_OP_GET_ROWS, selected0_value, { attention_value, row_ids0_value });
    graph.add_node(GGML_OP_GET_ROWS, selected1_value, { residual_value, row_ids1_value });
    graph.add_node(GGML_OP_ADD, output_value, { selected0_value, selected1_value });
    REQUIRE(graph.build_index().success());
    return graph;
}

struct GatherAddRmsNormGraph {
    ggml_tensor * attention         = nullptr;
    ggml_tensor * residual          = nullptr;
    ggml_tensor * row_ids           = nullptr;
    ggml_tensor * weight            = nullptr;
    ggml_tensor * selected          = nullptr;
    ggml_tensor * raw_output        = nullptr;
    ggml_tensor * normalized_output = nullptr;
    ggml_cgraph * graph             = nullptr;
};

static GatherAddRmsNormGraph build_gather_add_rmsnorm_graph(
    ggml_context *        ctx,
                                                            int64_t        hidden_size,
                                                            int64_t        source_token_count,
                                                            int64_t        output_token_count,
                                                            bool           produced_row_ids = false,
    bool                  produced_weight  = false,
    ggml::hrx::BinaryKind gather_op        = ggml::hrx::BinaryKind::Add,
    bool                  first_is_lhs     = true) {
    GatherAddRmsNormGraph result;
    result.attention                 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    result.residual                  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, source_token_count);
    ggml_tensor * raw_row_ids        = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output_token_count);
    ggml_tensor * raw_weight         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    result.row_ids                   = produced_row_ids ? ggml_dup(ctx, raw_row_ids) : raw_row_ids;
    result.weight                    = produced_weight ? ggml_scale(ctx, raw_weight, 0.5f) : raw_weight;
    ggml_tensor * selected_attention = ggml_get_rows(ctx, result.attention, result.row_ids);
    ggml_tensor * selected_residual  = ggml_get_rows(ctx, result.residual, result.row_ids);
    result.selected                  = selected_attention;
    switch (gather_op) {
        case ggml::hrx::BinaryKind::Add:
    result.raw_output                = ggml_add(ctx, selected_attention, selected_residual);
            break;
        case ggml::hrx::BinaryKind::Sub:
            result.raw_output = first_is_lhs ? ggml_sub(ctx, selected_attention, selected_residual) :
                                               ggml_sub(ctx, selected_residual, selected_attention);
            break;
        case ggml::hrx::BinaryKind::Mul:
            result.raw_output = ggml_mul(ctx, selected_attention, selected_residual);
            break;
        case ggml::hrx::BinaryKind::Div:
            result.raw_output = first_is_lhs ? ggml_div(ctx, selected_attention, selected_residual) :
                                               ggml_div(ctx, selected_residual, selected_attention);
            break;
        default:
            REQUIRE(false);
    }
    ggml_tensor * rms                = ggml_rms_norm(ctx, result.raw_output, 0.000001f);
    result.normalized_output         = ggml_mul(ctx, rms, result.weight);
    REQUIRE(result.attention != nullptr);
    REQUIRE(result.residual != nullptr);
    REQUIRE(raw_row_ids != nullptr);
    REQUIRE(raw_weight != nullptr);
    REQUIRE(result.row_ids != nullptr);
    REQUIRE(result.weight != nullptr);
    REQUIRE(selected_attention != nullptr);
    REQUIRE(selected_residual != nullptr);
    REQUIRE(result.raw_output != nullptr);
    REQUIRE(rms != nullptr);
    REQUIRE(result.normalized_output != nullptr);

    ggml_set_output(result.raw_output);
    result.graph = ggml_new_graph(ctx);
    REQUIRE(result.graph != nullptr);
    ggml_build_forward_expand(result.graph, result.normalized_output);
    return result;
}

static ggml::hrx::DispatchRegistry gather_add_test_registry() {
    ggml::hrx::DispatchRegistryBuilder builder;
    ggml::hrx::register_gather_add_dispatch(builder);
    return builder.build();
}

static bool match_gather_add_at(const ggml::hrx::DispatchRegistry & registry,
                                const ggml::hrx::Graph &            graph,
                                size_t                              root_index,
                                const std::vector<bool> &           covered_nodes,
                                ggml::hrx::DispatchMatch &          match) {
    ggml::hrx::CommandPlan                plan;
    const ggml::hrx::DispatchMatchContext context = {
        graph,      &graph.nodes()[root_index],
        root_index, covered_nodes,
        plan,       ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };
    return registry.match(context, match);
}

static void require_gather_add_rmsnorm_route(ggml_context *        ctx,
                                             ggml::hrx::BinaryKind gather_op    = ggml::hrx::BinaryKind::Add,
                                             bool                  first_is_lhs = true) {
    GatherAddRmsNormGraph tensors =
        build_gather_add_rmsnorm_graph(ctx, 128, 4, 5, false, false, gather_op, first_is_lhs);
    require_scheduled_command_program(tensors.graph, [&](const ggml::hrx::Graph &, const ggml::hrx::CommandPlan & plan,
            const ggml::hrx::CommandProgram & commands) {
            REQUIRE(plan.dispatches.size() == 1);
            const ggml::hrx::Dispatch & dispatch = plan.dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_gather_add_rmsnorm_binary_f32");
            REQUIRE(dispatch.bindings.size() == 6);
            require_compile_parameter(dispatch, "ggml.gather_add_rmsnorm_binary_f32.hidden_size", "128");
            require_compile_parameter(dispatch, "ggml.gather_add_rmsnorm_binary_f32.rms_epsilon", "9.99999997e-07");
        require_compile_parameter(dispatch, "ggml.gather_add_rmsnorm_binary_f32.gather_op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(gather_op)));
            REQUIRE(commands.commands.size() == 1);
            REQUIRE(commands.commands.front().bindings.size() == 6);
            REQUIRE(commands.commands.front().bindings[0].name == "attention");
            REQUIRE(commands.commands.front().bindings[1].name == "residual");
            REQUIRE(commands.commands.front().bindings[2].name == "output_ids");
            REQUIRE(commands.commands.front().bindings[3].name == "raw_output");
            REQUIRE(commands.commands.front().bindings[4].name == "weight");
            REQUIRE(commands.commands.front().bindings[5].name == "normalized_output");
        });
}

static void require_gather_add_rmsnorm_availability(ggml_context * ctx) {
    GatherAddRmsNormGraph        tensors  = build_gather_add_rmsnorm_graph(ctx, 128, 4, 5, true, true);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*tensors.graph);
    REQUIRE(imported.valid());
    const ggml::hrx::DispatchRegistry registry     = gather_add_test_registry();
    const size_t                      root_index   = producer_index_for_tensor(imported.graph, tensors.selected);
    const size_t                      row_index    = producer_index_for_tensor(imported.graph, tensors.row_ids);
    const size_t                      weight_index = producer_index_for_tensor(imported.graph, tensors.weight);
    std::vector<bool>                 covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::DispatchMatch          match;

    REQUIRE(!match_gather_add_at(registry, imported.graph, root_index, covered_nodes, match));

    covered_nodes[row_index] = true;
    match                    = {};
    REQUIRE(match_gather_add_at(registry, imported.graph, root_index, covered_nodes, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) == "hrx:ggml_gather_add_f32");

    covered_nodes[weight_index] = true;
    match                       = {};
    REQUIRE(match_gather_add_at(registry, imported.graph, root_index, covered_nodes, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) ==
            "loom_libs:ggml_gather_add_rmsnorm_binary_f32");
}

static void require_gather_add_rmsnorm_row_ids_noalias(ggml_context * ctx) {
    GatherAddRmsNormGraph        tensors  = build_gather_add_rmsnorm_graph(ctx, 128, 4, 5);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*tensors.graph);
    REQUIRE(imported.valid());
    const ggml::hrx::Value * row_ids   = imported.graph.values().find_tensor(tensors.row_ids);
    const ggml::hrx::Value * attention = imported.graph.values().find_tensor(tensors.attention);
    REQUIRE(row_ids != nullptr);
    REQUIRE(attention != nullptr);

    ggml::hrx::Graph aliased_graph;
    for (const ggml::hrx::ValueStorage & storage : imported.graph.values().storages()) {
        REQUIRE(aliased_graph.values().add_snapshot_storage(storage).success());
    }
    for (ggml::hrx::Value value : imported.graph.values().values()) {
        if (value.id == row_ids->id) {
            value.storage            = attention->storage;
            value.storage_root       = attention->storage_root;
            value.alias_source       = attention->id;
            value.storage_offset     = attention->storage_offset;
            value.storage_byte_count = attention->storage_byte_count;
        }
        REQUIRE(aliased_graph.values().add_snapshot_value(std::move(value)).success());
    }
    for (const ggml::hrx::GraphNode & node : imported.graph.nodes()) {
        ggml::hrx::GraphNode & copied = aliased_graph.add_node(node.op, node.output, node.inputs);
        copied.params                 = node.params;
    }
    REQUIRE(aliased_graph.build_index().success());

    const ggml::hrx::DispatchRegistry registry   = gather_add_test_registry();
    size_t                            root_index = aliased_graph.nodes().size();
    for (size_t i = 0; i < aliased_graph.nodes().size(); ++i) {
        if (aliased_graph.nodes()[i].op == GGML_OP_GET_ROWS) {
            root_index = i;
            break;
        }
    }
    REQUIRE(root_index < aliased_graph.nodes().size());
    std::vector<bool>        covered_nodes(aliased_graph.nodes().size(), false);
    ggml::hrx::DispatchMatch match;
    REQUIRE(match_gather_add_at(registry, aliased_graph, root_index, covered_nodes, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) == "hrx:ggml_gather_add_f32");
}

static bool manual_gather_add_graph_is_supported(ggml_context * ctx,
                                                 int64_t        hidden_size,
                                                 int64_t        source_token_count,
                                                 int64_t        output_token_count,
                                                 bool           shared_row_ids,
                                                 int64_t        second_source_hidden_size = -1) {
    ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, hidden_size, source_token_count, output_token_count,
                                                           shared_row_ids, second_source_hidden_size);
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static bool partial_gather_add_graph_is_supported(ggml_context * ctx) {
    ggml::hrx::Graph graph;

    ggml_tensor * attention  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 13);
    ggml_tensor * row_ids    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * selected   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * add_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * add_output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    REQUIRE(attention != nullptr);
    REQUIRE(row_ids != nullptr);
    REQUIRE(selected != nullptr);
    REQUIRE(add_input != nullptr);
    REQUIRE(add_output != nullptr);

    const ggml::hrx::ValueId attention_value =
        graph.values().get_or_add_tensor_value(attention, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids_value =
        graph.values().get_or_add_tensor_value(row_ids, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId selected_value =
        graph.values().get_or_add_tensor_value(selected, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId add_input_value =
        graph.values().get_or_add_tensor_value(add_input, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId add_output_value =
        graph.values().get_or_add_tensor_value(add_output, ggml::hrx::ValueKind::External);

    graph.add_node(GGML_OP_GET_ROWS, selected_value, { attention_value, row_ids_value });
    graph.add_node(GGML_OP_ADD, add_output_value, { selected_value, add_input_value });
    REQUIRE(graph.build_index().success());
    return ggml::hrx::DispatchScheduler::can_schedule_graph(graph, test_dispatch_target());
}

static void require_gather_add_rejects_unavailable_sources(ggml_context * ctx) {
    ggml::hrx::Graph graph;

    ggml_tensor * raw_attention = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 13);
    ggml_tensor * attention     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 13);
    ggml_tensor * residual      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 13);
    ggml_tensor * row_ids       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * selected0     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * selected1     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    ggml_tensor * output        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    REQUIRE(raw_attention != nullptr);
    REQUIRE(attention != nullptr);
    REQUIRE(residual != nullptr);
    REQUIRE(row_ids != nullptr);
    REQUIRE(selected0 != nullptr);
    REQUIRE(selected1 != nullptr);
    REQUIRE(output != nullptr);

    const ggml::hrx::ValueId raw_attention_value =
        graph.values().get_or_add_tensor_value(raw_attention, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId attention_value =
        graph.values().get_or_add_tensor_value(attention, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId residual_value =
        graph.values().get_or_add_tensor_value(residual, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId row_ids_value =
        graph.values().get_or_add_tensor_value(row_ids, ggml::hrx::ValueKind::External);
    const ggml::hrx::ValueId selected0_value =
        graph.values().get_or_add_tensor_value(selected0, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId selected1_value =
        graph.values().get_or_add_tensor_value(selected1, ggml::hrx::ValueKind::Transient);
    const ggml::hrx::ValueId output_value =
        graph.values().get_or_add_tensor_value(output, ggml::hrx::ValueKind::External);

    graph.add_node(GGML_OP_SCALE, attention_value, { raw_attention_value });
    graph.add_node(GGML_OP_GET_ROWS, selected0_value, { attention_value, row_ids_value });
    graph.add_node(GGML_OP_GET_ROWS, selected1_value, { residual_value, row_ids_value });
    graph.add_node(GGML_OP_ADD, output_value, { selected0_value, selected1_value });
    REQUIRE(graph.build_index().success());

    ggml::hrx::DispatchRegistryBuilder builder;
    ggml::hrx::register_gather_add_dispatch(builder);
    const ggml::hrx::DispatchRegistry gather_add_registry = builder.build();

    const size_t producer_index = 0;
    const size_t get_rows_index = 1;
    ggml::hrx::CommandPlan plan;
    ggml::hrx::DispatchMatch match;
    std::vector<bool> covered_nodes(graph.nodes().size(), false);
    ggml::hrx::DispatchMatchContext context = {
        graph,          &graph.nodes()[get_rows_index],
        get_rows_index, covered_nodes,
        plan,           ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };

    REQUIRE(!gather_add_registry.match(context, match));

    covered_nodes[producer_index] = true;
    const ggml::hrx::DispatchMatchContext available_context = {
        graph,          &graph.nodes()[get_rows_index],
        get_rows_index, covered_nodes,
        plan,           ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size())),
    };
    match = {};
    REQUIRE(gather_add_registry.match(available_context, match));
    REQUIRE(match.dispatches.size() == 1);
    REQUIRE(kernel_name_for_id(match.dispatches.front().kernel.kernel_id) == "hrx:ggml_gather_add_f32");
}

static void schedule_gather_add_command(ggml::hrx::Graph & graph,
                                        int64_t            expected_hidden_size,
                                        int64_t            expected_source_token_count,
                                        int64_t            expected_output_token_count) {
    ggml::hrx::DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(graph, test_dispatch_target())) {
        for (const std::string & error : scheduler.plan().status.errors()) {
            std::fprintf(stderr, "%s\n", error.c_str());
        }
        REQUIRE(false);
    }
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "hrx:ggml_gather_add_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("hidden_size") == expected_hidden_size);
    REQUIRE(dispatch.kernel.integer_parameters.at("source_token_count") == expected_source_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("output_token_count") == expected_output_token_count);
    require_compile_parameter(dispatch, "ggml.gather_add_f32.binary_op", "0");
    REQUIRE(dispatch.bindings.size() == 4);

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 4);
    REQUIRE(commands.commands.front().bindings[0].name == "attention");
    REQUIRE(commands.commands.front().bindings[1].name == "residual");
    REQUIRE(commands.commands.front().bindings[2].name == "output_ids");
    REQUIRE(commands.commands.front().bindings[3].name == "output");
}

static void run_gather_add_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, 2048, 13, 1, true);
        schedule_gather_add_command(graph, 2048, 13, 1);
    }
    {
        ggml::hrx::Graph graph = build_manual_gather_add_graph(ctx, 2048, 128, 8, true);
        schedule_gather_add_command(graph, 2048, 128, 8);
    }

    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 2048, 13, 1, false));
    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 2048, 13, 1, true, 1024));
    REQUIRE(!manual_gather_add_graph_is_supported(ctx, 96, 13, 1, true));
    REQUIRE(!partial_gather_add_graph_is_supported(ctx));
    require_gather_add_rejects_unavailable_sources(ctx);
    require_gather_add_rmsnorm_route(ctx);
    require_gather_add_rmsnorm_route(ctx, ggml::hrx::BinaryKind::Sub, false);
    require_gather_add_rmsnorm_route(ctx, ggml::hrx::BinaryKind::Mul);
    require_gather_add_rmsnorm_route(ctx, ggml::hrx::BinaryKind::Div);
    require_gather_add_rmsnorm_availability(ctx);
    require_gather_add_rmsnorm_row_ids_noalias(ctx);

    ggml_free(ctx);
}

static void schedule_flash_attention_command(ggml_context * ctx,
                                             ggml_tensor *  output,
                                             const char *   expected_kernel_name,
                                             const char *   config_prefix,
                                             int64_t        query_token_count,
                                             int64_t        key_value_token_count,
                                             int64_t        query_head_count,
                                             int64_t        key_value_head_count,
                                             int64_t        qk_head_size    = kQwenFlashHeadSize,
                                             int64_t        value_head_size = kQwenFlashHeadSize,
                                             float          scale           = 1.0f / std::sqrt(128.0f),
                                             bool           transposed_value = false) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_FLASH_ATTN_EXT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == (transposed_value ? 2 : 1));

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("query_token_count") == query_token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
    REQUIRE(dispatch.bindings.size() == 6);
    require_compile_parameter(dispatch, (std::string(config_prefix) + "query_head_count").c_str(),
                              std::to_string(query_head_count));
    require_compile_parameter(dispatch, (std::string(config_prefix) + "key_value_head_count").c_str(),
                              std::to_string(key_value_head_count));
    if (std::strcmp(config_prefix, "ggml.flash_attention.") == 0) {
        require_compile_parameter(dispatch, "ggml.flash_attention.qk_head_size", std::to_string(qk_head_size));
        require_compile_parameter(dispatch, "ggml.flash_attention.value_head_size", std::to_string(value_head_size));
        require_compile_parameter(dispatch, "ggml.flash_attention.attention_scale", expected_config_value(scale));
        require_compile_parameter(dispatch, "ggml.flash_attention.apply_gate", "0");
        require_compile_parameter(dispatch, "ggml.flash_attention.gate_stride_head", "1");
        require_compile_parameter(dispatch, "ggml.flash_attention.gate_stride_token", "1");
    } else {
        require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(query_token_count));
    }

    if (transposed_value) {
        REQUIRE(scheduler.plan().transients.size() == 1);
        const auto & copy = scheduler.plan().dispatches.front();
        REQUIRE(kernel_name_for_id(copy.kernel.kernel_id) == "loom_libs:ggml_copy_transpose_f16");
        REQUIRE(copy.kernel.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
        REQUIRE(copy.kernel.integer_parameters.at("row_count") == key_value_token_count);
        REQUIRE(copy.kernel.integer_parameters.at("column_count") == key_value_head_count * value_head_size);
        REQUIRE(copy.bindings.size() == 2);
        REQUIRE(copy.bindings[1].value == dispatch.bindings[2].value);
        REQUIRE(copy.bindings[0].value != copy.bindings[1].value);
        REQUIRE(copy.bindings[1].length ==
                static_cast<size_t>(key_value_token_count * key_value_head_count * value_head_size) *
                    sizeof(ggml_fp16_t));
        require_compile_parameter(dispatch, "ggml.flash_attention.value_layout", "1");
    } else {
        REQUIRE(dispatch.kernel.compile_parameters.count("ggml.flash_attention.value_layout") == 0);
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == (transposed_value ? 2 : 1));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() == 6);
    REQUIRE(commands.commands.back().bindings[0].name == "query");
    REQUIRE(commands.commands.back().bindings[1].name == "key");
    REQUIRE(commands.commands.back().bindings[2].name == "value");
    REQUIRE(commands.commands.back().bindings[3].name == "mask");
    REQUIRE(commands.commands.back().bindings[4].name == "gate");
    REQUIRE(commands.commands.back().bindings[5].name == "output");
}

static void schedule_qwen_flash_attention_command(ggml_context * ctx, ggml_tensor * output) {
    schedule_flash_attention_command(ctx, output, "loom_libs:ggml_flash_attention_f32_f16_wmma",
                                     "ggml.flash_attention.", 16, 16, 4, 2);
}

static void schedule_common_flash_attention_command(ggml_context * ctx,
                                                    ggml_tensor *  output,
                                                    int64_t        query_token_count,
                                                    int64_t        key_value_token_count,
                                                    int64_t        query_head_count,
                                                    int64_t        key_value_head_count,
                                                    int64_t        qk_head_size    = kQwenFlashHeadSize,
                                                    int64_t        value_head_size = kQwenFlashHeadSize,
                                                    float          scale           = 1.0f / std::sqrt(128.0f)) {
    schedule_flash_attention_command(ctx, output, "loom_libs:ggml_flash_attention_f32_f16_wmma",
                                     "ggml.flash_attention.", query_token_count, key_value_token_count,
                                     query_head_count, key_value_head_count, qk_head_size, value_head_size, scale);
}

static void schedule_common_flash_attention_decode_fallback_command(ggml_context * ctx,
                                                                    ggml_tensor *  output,
                                                                    int64_t        query_token_count,
                                                                    int64_t        key_value_token_count,
                                                                    int64_t        query_head_count,
                                                                    int64_t        key_value_head_count,
                                                                    int64_t        qk_head_size    = kQwenFlashHeadSize,
                                                                    int64_t        value_head_size = kQwenFlashHeadSize,
                                                                    float scale = 1.0f / std::sqrt(128.0f)) {
    ggml_cgraph * graph              = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_FLASH_ATTN_EXT);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == static_cast<size_t>(query_token_count));
    REQUIRE(scheduler.plan().transients.size() == 4);
    REQUIRE(scheduler.plan().completion_counter_requests.size() == 1);
    REQUIRE(scheduler.plan().completion_counter_requests.front().count == key_value_head_count);
    REQUIRE(scheduler.plan().metadata.alternate_values().empty());

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches.front();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
            "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
    REQUIRE(dispatch.kernel.integer_parameters.at("key_value_token_count") == key_value_token_count);
    REQUIRE(dispatch.bindings.size() == 10);
    require_compile_parameter(dispatch, "ggml.flash_attention.query_head_count", std::to_string(query_head_count));
    require_compile_parameter(dispatch, "ggml.flash_attention.key_value_head_count",
                              std::to_string(key_value_head_count));
    require_compile_parameter(dispatch, "ggml.flash_attention.qk_head_size", std::to_string(qk_head_size));
    require_compile_parameter(dispatch, "ggml.flash_attention.value_head_size", std::to_string(value_head_size));
    require_compile_parameter(dispatch, "ggml.flash_attention.attention_scale", expected_config_value(scale));

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == static_cast<size_t>(query_token_count));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 10);
    REQUIRE(commands.commands.front().bindings[0].name == "query");
    REQUIRE(commands.commands.front().bindings[1].name == "key");
    REQUIRE(commands.commands.front().bindings[2].name == "value");
    REQUIRE(commands.commands.front().bindings[3].name == "mask");
    REQUIRE(commands.commands.front().bindings[8].name == "output");
    REQUIRE(commands.commands.front().bindings[9].name == "next_q8_output");
}

static void schedule_common_flash_attention_decode_q8_command(ggml_context * ctx, ggml_type weight_type) {
    constexpr int64_t query_token_count     = 1;
    constexpr int64_t key_value_token_count = 512;
    constexpr int64_t query_head_count      = 4;
    constexpr int64_t key_value_head_count  = 2;
    constexpr int64_t hidden_size           = query_head_count * kQwenFlashHeadSize;
    constexpr int64_t output_size           = 256;
    const size_t      q8_bytes              = ggml_row_size(GGML_TYPE_Q8_1, hidden_size);

    ggml_tensor * output = build_qwen_flash_attention_graph(
        ctx, query_token_count, key_value_token_count, query_head_count, key_value_head_count);
    ggml_tensor * reshaped  = ggml_reshape_2d(ctx, output, hidden_size, query_token_count);
    ggml_tensor * weight    = ggml_new_tensor_2d(ctx, weight_type, hidden_size, output_size);
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, reshaped);
    REQUIRE(reshaped != nullptr);
    REQUIRE(weight != nullptr);
    REQUIRE(projected != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, projected);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.dispatches.size() == 2);
    REQUIRE(plan.transients.size() == 4);
    REQUIRE(plan.completion_counter_requests.size() == 1);

    const ggml::hrx::Value * native_value  = imported.graph.values().find_tensor(output);
    const ggml::hrx::Value * reshape_value = imported.graph.values().find_tensor(reshaped);
    REQUIRE(native_value != nullptr);
    REQUIRE(reshape_value != nullptr);
    const ggml::hrx::CommandPlanAlternateValue * q8 =
        ggml::hrx::find_alternate_value(plan, reshape_value->id, GGML_TYPE_Q8_1, q8_bytes);
    REQUIRE(q8 != nullptr);
    REQUIRE(ggml::hrx::find_alternate_value(plan, native_value->id, GGML_TYPE_Q8_1, q8_bytes) == nullptr);

    const ggml::hrx::Dispatch & producer = plan.dispatches.front();
    const ggml::hrx::Dispatch & consumer = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(producer.kernel.kernel_id) ==
            "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
    REQUIRE(producer.bindings.size() == 10);
    REQUIRE(producer.bindings[9].value == q8->alternate_value);
    REQUIRE(consumer.bindings[0].value == q8->alternate_value);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.front().bindings.size() == 10);
    REQUIRE(commands.commands.front().bindings[8].name == "output");
    REQUIRE(commands.commands.front().bindings[9].name == "next_q8_output");
}

static void schedule_common_flash_attention_decode_unqualified_command(ggml_context * ctx) {
    constexpr int64_t hidden_size = 4 * kQwenFlashHeadSize;
    ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 1, 512, 4, 2);
    ggml_tensor * reshaped  = ggml_reshape_2d(ctx, output, hidden_size, 1);
    ggml_tensor * weight    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 256);
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, reshaped);
    REQUIRE(reshaped != nullptr);
    REQUIRE(weight != nullptr);
    REQUIRE(projected != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, projected);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());
    REQUIRE(plan.metadata.alternate_values().empty());
    REQUIRE(plan.dispatches.size() == 2);
    REQUIRE(plan.transients.size() == 4);
    REQUIRE(plan.completion_counter_requests.size() == 1);
    REQUIRE(kernel_name_for_id(plan.dispatches.front().kernel.kernel_id) ==
            "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
    REQUIRE(plan.dispatches.front().bindings.size() == 10);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands.front().bindings.size() == 10);
}

static void run_qwen_flash_attention_dispatch_checks() {
    static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
    const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
    const bool                    had_original_env        = original_env != nullptr;
    const std::string             original_env_value      = had_original_env ? original_env : "";
    REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);

    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    for (const auto & shape : {
             std::array<int64_t, 4>{ 256, 512,  64,  1 },
                                std::array<int64_t, 4>{ 512, 512, 256, 1 },
                                std::array<int64_t, 4>{ 512, 2048, 320, 1 },
                                std::array<int64_t, 4>{ 255, 512, 256, 0 },
             std::array<int64_t, 4>{ 512, 513,  256, 0 }
    }) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(shape[2]));
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, shape[0], shape[1], 4, 2, GGML_TYPE_F32,
                                                                GGML_TYPE_F16, true, false, shape[2], scale);
        schedule_flash_attention_command(ctx, output, "loom_libs:ggml_flash_attention_f32_f16_wmma",
                                          "ggml.flash_attention.", shape[0], shape[1], 4, 2, shape[2], shape[2], scale,
                                          shape[3] != 0);
    }

    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2);
        schedule_common_flash_attention_decode_fallback_command(ctx, output, 4, 8, 4, 2);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 16, 16, 4, 2);
        schedule_qwen_flash_attention_command(ctx, output);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 16, 16, 4, 2);
        schedule_common_flash_attention_command(ctx, output, 16, 16, 4, 2);
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        constexpr int64_t head_size = 256;
        ggml_tensor *     output =
            build_qwen_flash_attention_graph(ctx, 16, 16, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, false, head_size,
                                             1.0f / std::sqrt(static_cast<float>(head_size)));
        schedule_common_flash_attention_command(ctx, output, 16, 16, 4, 2, head_size, head_size,
                                                1.0f / std::sqrt(static_cast<float>(head_size)));
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        constexpr int64_t qk_head_size    = 96;
        constexpr int64_t value_head_size = 64;
        ggml_tensor *     output          = build_qwen_flash_attention_graph(
            ctx, 23, 256, 40, 40, GGML_TYPE_F32, GGML_TYPE_F16, true, false, qk_head_size,
            1.0f / std::sqrt(static_cast<float>(qk_head_size)), value_head_size);
        schedule_common_flash_attention_command(ctx, output, 23, 256, 40, 40, qk_head_size, value_head_size,
                                                1.0f / std::sqrt(static_cast<float>(qk_head_size)));
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 1, 8, 4, 2);
        schedule_common_flash_attention_decode_fallback_command(ctx, output, 1, 8, 4, 2);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        constexpr int64_t qk_head_size    = 96;
        constexpr int64_t value_head_size = 64;
        ggml_tensor *     output          = build_qwen_flash_attention_graph(
            ctx, 1, 512, 32, 4, GGML_TYPE_F32, GGML_TYPE_F16, true, false, qk_head_size,
            1.0f / std::sqrt(static_cast<float>(qk_head_size)), value_head_size);
        schedule_common_flash_attention_decode_fallback_command(ctx, output, 1, 512, 32, 4, qk_head_size,
                                                                value_head_size,
                                                                1.0f / std::sqrt(static_cast<float>(qk_head_size)));
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        schedule_common_flash_attention_decode_q8_command(ctx, GGML_TYPE_Q4_K);
        schedule_common_flash_attention_decode_q8_command(ctx, GGML_TYPE_Q6_K);
        schedule_common_flash_attention_decode_unqualified_command(ctx);
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);
        constexpr int64_t tokens = 512;
        constexpr int64_t hidden = 512;
        ggml_tensor * output      = build_qwen_flash_attention_graph(ctx, tokens, tokens, 4, 2);
        ggml_tensor * reshaped    = ggml_reshape_2d(ctx, output, hidden, tokens);
        ggml_tensor * weight      = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, hidden, hidden);
        ggml_tensor * projected   = ggml_mul_mat(ctx, weight, reshaped);
        ggml_cgraph * graph       = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, projected);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_flash_attention_f32_f16_wmma_publish_f16";
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings.size() == 7);
        const ggml::hrx::Value * output_value = imported.graph.values().find_tensor(output);
        REQUIRE(output_value != nullptr);
        const auto * alternate =
            plan.metadata.find_alternate_value(output_value->id, GGML_TYPE_F16, output_value->byte_count / 2);
        REQUIRE(alternate != nullptr);
        REQUIRE(producer->bindings.back().value == alternate->alternate_value);
        const auto consumer = std::next(producer);
        REQUIRE(consumer != plan.dispatches.end());
        REQUIRE(consumer->bindings.front().value == alternate->alternate_value);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        REQUIRE(unsetenv(kDisableQwenDispatchEnv) == 0);
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F32);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, false);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, true);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * output = build_qwen_flash_attention_graph(ctx, 16, 16, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true,
                                                                false, kQwenFlashHeadSize, 1.0f);
        schedule_common_flash_attention_command(ctx, output, 16, 16, 4, 2, kQwenFlashHeadSize, kQwenFlashHeadSize,
                                                1.0f);
    }
    {
        ggml_tensor * output =
            build_qwen_flash_attention_graph(ctx, 4, 8, 4, 2, GGML_TYPE_F32, GGML_TYPE_F16, true, false,
                                             kQwenFlashHeadSize, 1.0f / std::sqrt(128.0f), -1, 1.0f);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
    restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
}

struct QwenAttentionPostprocessTensors {
    ggml_tensor * query_raw           = nullptr;
    ggml_tensor * key_raw             = nullptr;
    ggml_tensor * value_raw           = nullptr;
    ggml_tensor * query_reshape       = nullptr;
    ggml_tensor * key_reshape         = nullptr;
    ggml_tensor * value_reshape       = nullptr;
    ggml_tensor * query_output        = nullptr;
    ggml_tensor * key_cache           = nullptr;
    ggml_tensor * value_cache         = nullptr;
    ggml_tensor * key_output          = nullptr;
    ggml_tensor * value_output        = nullptr;
    ggml_tensor * positions           = nullptr;
    ggml_tensor * key_cache_indices   = nullptr;
    ggml_tensor * value_cache_indices = nullptr;
    ggml_tensor * mask                = nullptr;
    ggml_tensor * flash_output        = nullptr;
};

static QwenAttentionPostprocessTensors build_qwen_attention_postprocess_graph(ggml_context * ctx,
                                                                              int64_t        token_count,
                                                                              int64_t        query_head_count,
                                                                              int64_t        key_value_head_count,
                                                                              int64_t        cache_row_count,
                                                                              float          rms_epsilon = 0.000001f,
                                                                              bool  include_inverse_frequencies = true,
                                                                              int   rope_n_ctx_orig             = 0,
                                                                              float rope_freq_base  = 10000.0f,
                                                                              float rope_freq_scale = 1.0f,
                                                                              float rope_ext_factor = 0.0f,
                                                                              float rope_beta_fast  = 0.0f,
                                                                              float rope_beta_slow  = 0.0f) {
    QwenAttentionPostprocessTensors tensors;
    const int64_t                   query_size     = query_head_count * kQwenFlashHeadSize;
    const int64_t                   key_value_size = key_value_head_count * kQwenFlashHeadSize;

    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, token_count);
    ggml_tensor * query_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, query_size);
    ggml_tensor * key_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, key_value_size);
    ggml_tensor * value_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, kQwenMoeHiddenSize, key_value_size);
    REQUIRE(input != nullptr);
    REQUIRE(query_weight != nullptr);
    REQUIRE(key_weight != nullptr);
    REQUIRE(value_weight != nullptr);

    tensors.query_raw = ggml_mul_mat(ctx, query_weight, input);
    tensors.key_raw   = ggml_mul_mat(ctx, key_weight, input);
    tensors.value_raw = ggml_mul_mat(ctx, value_weight, input);
    REQUIRE(tensors.query_raw != nullptr);
    REQUIRE(tensors.key_raw != nullptr);
    REQUIRE(tensors.value_raw != nullptr);

    tensors.query_reshape = ggml_reshape_3d(ctx, tensors.query_raw, kQwenFlashHeadSize, query_head_count, token_count);
    tensors.key_reshape = ggml_reshape_3d(ctx, tensors.key_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    tensors.value_reshape =
        ggml_reshape_3d(ctx, tensors.value_raw, kQwenFlashHeadSize, key_value_head_count, token_count);
    REQUIRE(tensors.query_reshape != nullptr);
    REQUIRE(tensors.key_reshape != nullptr);
    REQUIRE(tensors.value_reshape != nullptr);

    ggml_tensor * query_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    ggml_tensor * key_norm_weight   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize);
    tensors.positions               = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * inverse_frequencies =
        include_inverse_frequencies ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenFlashHeadSize / 2) : nullptr;
    REQUIRE(query_norm_weight != nullptr);
    REQUIRE(key_norm_weight != nullptr);
    REQUIRE(tensors.positions != nullptr);
    REQUIRE(include_inverse_frequencies == (inverse_frequencies != nullptr));

    ggml_tensor * query_norm = ggml_rms_norm(ctx, tensors.query_reshape, rms_epsilon);
    ggml_tensor * query_mul  = ggml_mul(ctx, query_norm, query_norm_weight);
    REQUIRE(query_norm != nullptr);
    REQUIRE(query_mul != nullptr);
    tensors.query_output = ggml_rope_ext(ctx, query_mul, tensors.positions, inverse_frequencies, kQwenFlashHeadSize,
                                         GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig, rope_freq_base, rope_freq_scale,
                                         rope_ext_factor, 1.0f, rope_beta_fast, rope_beta_slow);
    REQUIRE(tensors.query_output != nullptr);

    ggml_tensor * key_norm = ggml_rms_norm(ctx, tensors.key_reshape, rms_epsilon);
    ggml_tensor * key_mul  = ggml_mul(ctx, key_norm, key_norm_weight);
    REQUIRE(key_norm != nullptr);
    REQUIRE(key_mul != nullptr);
    ggml_tensor * key_rope = ggml_rope_ext(ctx, key_mul, tensors.positions, inverse_frequencies, kQwenFlashHeadSize,
                                           GGML_ROPE_TYPE_NEOX, rope_n_ctx_orig, rope_freq_base, rope_freq_scale,
                                           rope_ext_factor, 1.0f, rope_beta_fast, rope_beta_slow);
    REQUIRE(key_rope != nullptr);

    ggml_tensor * key_cache_rows =
        ggml_reshape_2d(ctx, key_rope, kQwenFlashHeadSize * key_value_head_count, token_count);
    ggml_tensor * value_cache_rows =
        ggml_reshape_2d(ctx, tensors.value_reshape, kQwenFlashHeadSize * key_value_head_count, token_count);
    REQUIRE(key_cache_rows != nullptr);
    REQUIRE(value_cache_rows != nullptr);

    tensors.key_cache           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    tensors.value_cache         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, key_value_size, cache_row_count);
    tensors.key_cache_indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    tensors.value_cache_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
    REQUIRE(tensors.key_cache != nullptr);
    REQUIRE(tensors.value_cache != nullptr);
    REQUIRE(tensors.key_cache_indices != nullptr);
    REQUIRE(tensors.value_cache_indices != nullptr);

    tensors.key_output   = ggml_set_rows(ctx, tensors.key_cache, key_cache_rows, tensors.key_cache_indices);
    tensors.value_output = ggml_set_rows(ctx, tensors.value_cache, value_cache_rows, tensors.value_cache_indices);
    REQUIRE(tensors.key_output != nullptr);
    REQUIRE(tensors.value_output != nullptr);
    return tensors;
}

static ggml::hrx::GraphImportResult import_qwen_attention_postprocess_graph(
    ggml_context *                          ctx,
    const QwenAttentionPostprocessTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.query_output);
    ggml_build_forward_expand(graph, tensors.key_output);
    ggml_build_forward_expand(graph, tensors.value_output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static ggml_tensor * append_qwen_flash_attention_consumer(ggml_context *                    ctx,
                                                          QwenAttentionPostprocessTensors & tensors,
                                                          int64_t                           token_count,
                                                          int64_t                           query_head_count,
                                                          int64_t                           key_value_head_count,
                                                          int64_t                           cache_row_count) {
    ggml_tensor * query_layout =
        ggml_reshape_3d(ctx, tensors.query_output, kQwenFlashHeadSize, query_head_count, token_count);
    ggml_tensor * query_permute = ggml_permute(ctx, query_layout, 0, 2, 1, 3);
    ggml_tensor * key_cache_layout =
        ggml_reshape_3d(ctx, tensors.key_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * key_permute = ggml_permute(ctx, key_cache_layout, 0, 2, 1, 3);
    ggml_tensor * value_cache_layout =
        ggml_reshape_3d(ctx, tensors.value_cache, kQwenFlashHeadSize, key_value_head_count, cache_row_count);
    ggml_tensor * value_permute = ggml_permute(ctx, value_cache_layout, 0, 2, 1, 3);
    tensors.mask                = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cache_row_count, token_count);
    REQUIRE(query_layout != nullptr);
    REQUIRE(query_permute != nullptr);
    REQUIRE(key_cache_layout != nullptr);
    REQUIRE(key_permute != nullptr);
    REQUIRE(value_cache_layout != nullptr);
    REQUIRE(value_permute != nullptr);
    REQUIRE(tensors.mask != nullptr);

    tensors.flash_output = ggml_flash_attn_ext(ctx, query_permute, key_permute, value_permute, tensors.mask,
                                               1.0f / std::sqrt(static_cast<float>(kQwenFlashHeadSize)), 0.0f, 0.0f);
    REQUIRE(tensors.flash_output != nullptr);
    return tensors.flash_output;
}

static void schedule_qwen_attention_postprocess_command(ggml_context *                          ctx,
                                                        const QwenAttentionPostprocessTensors & tensors,
                                                        int64_t                                 token_count,
                                                        int64_t                                 query_head_count,
                                                        int64_t                                 key_value_head_count,
                                                        int64_t                                 cache_row_count,
                                                        bool expect_synthetic_inverse_frequencies = false,
                                                        const std::string & expected_rope_mscale  = "1") {
    ggml::hrx::GraphImportResult imported = import_qwen_attention_postprocess_graph(ctx, tensors);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().initialization_dispatches.empty());
    const bool q8_input = token_count <= 5;
    REQUIRE(scheduler.plan().dispatches.size() == (q8_input ? 5 : 4));
    if (q8_input) {
        REQUIRE(kernel_name_for_id(scheduler.plan().dispatches.front().kernel.kernel_id) ==
                "qwen3_moe:ggml_quantize_q8_1_x4_f32");
    }

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches.back();
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("cache_row_count") == cache_row_count);
    REQUIRE(dispatch.bindings.size() == 12);
    require_compile_parameter(dispatch, "qwen3_moe.model.rms_epsilon", "0.000001");
    require_compile_parameter(dispatch, "qwen3_moe.attention.head_size", std::to_string(kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.attention.rope_mscale", expected_rope_mscale);
    require_compile_parameter(dispatch, "qwen3_moe.attention.query_size",
                              std::to_string(query_head_count * kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.attention.key_value_size",
                              std::to_string(key_value_head_count * kQwenFlashHeadSize));
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    const ggml::hrx::Value * positions_value     = imported.graph.values().find_tensor(tensors.positions);
    const ggml::hrx::Value * key_indices_value   = imported.graph.values().find_tensor(tensors.key_cache_indices);
    const ggml::hrx::Value * value_indices_value = imported.graph.values().find_tensor(tensors.value_cache_indices);
    const ggml::hrx::Value * query_raw_value     = imported.graph.values().find_tensor(tensors.query_raw);
    const ggml::hrx::Value * key_raw_value       = imported.graph.values().find_tensor(tensors.key_raw);
    const ggml::hrx::Value * value_raw_value     = imported.graph.values().find_tensor(tensors.value_raw);
    const ggml::hrx::Value * query_output_value  = imported.graph.values().find_tensor(tensors.query_output);
    const ggml::hrx::Value * key_cache_value     = imported.graph.values().find_tensor(tensors.key_cache);
    const ggml::hrx::Value * value_cache_value   = imported.graph.values().find_tensor(tensors.value_cache);
    REQUIRE(positions_value != nullptr);
    REQUIRE(key_indices_value != nullptr);
    REQUIRE(value_indices_value != nullptr);
    REQUIRE(query_raw_value != nullptr);
    REQUIRE(key_raw_value != nullptr);
    REQUIRE(value_raw_value != nullptr);
    REQUIRE(query_output_value != nullptr);
    REQUIRE(key_cache_value != nullptr);
    REQUIRE(value_cache_value != nullptr);
    REQUIRE(dispatch.bindings[0].value == positions_value->id);
    REQUIRE(dispatch.bindings[1].value == key_indices_value->id);
    REQUIRE(dispatch.bindings[2].value == value_indices_value->id);
    REQUIRE(dispatch.bindings[3].value == query_raw_value->id);
    REQUIRE(dispatch.bindings[4].value == key_raw_value->id);
    REQUIRE(dispatch.bindings[5].value == value_raw_value->id);
    REQUIRE(dispatch.bindings[9].value == query_output_value->id);
    REQUIRE(dispatch.bindings[10].value == key_cache_value->id);
    REQUIRE(dispatch.bindings[11].value == value_cache_value->id);
    if (expect_synthetic_inverse_frequencies) {
        REQUIRE(scheduler.plan().transients.size() == (q8_input ? 2 : 1));
        REQUIRE(scheduler.plan().constant_initializations.size() == 1);
        REQUIRE(dispatch.bindings[8].value == scheduler.plan().transients.back().value);
        REQUIRE(scheduler.plan().constant_initializations[0].value == dispatch.bindings[8].value);
        REQUIRE(scheduler.plan().constant_initializations[0].data.size() ==
                static_cast<size_t>(kQwenFlashHeadSize / 2) * sizeof(float));
    } else {
        REQUIRE(scheduler.plan().transients.size() == (q8_input ? 1 : 0));
        REQUIRE(scheduler.plan().constant_initializations.empty());
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.initialization_commands.empty());
    REQUIRE(commands.commands.size() == (q8_input ? 5 : 4));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands.back().bindings.size() == 12);
    REQUIRE(commands.commands.back().bindings[0].name == "positions");
    REQUIRE(commands.commands.back().bindings[1].name == "key_cache_indices");
    REQUIRE(commands.commands.back().bindings[2].name == "value_cache_indices");
    REQUIRE(commands.commands.back().bindings[3].name == "query_input");
    REQUIRE(commands.commands.back().bindings[4].name == "key_input");
    REQUIRE(commands.commands.back().bindings[5].name == "value_input");
    REQUIRE(commands.commands.back().bindings[6].name == "query_norm_weight");
    REQUIRE(commands.commands.back().bindings[7].name == "key_norm_weight");
    REQUIRE(commands.commands.back().bindings[8].name == "inverse_frequencies");
    REQUIRE(commands.commands.back().bindings[9].name == "query_output");
    REQUIRE(commands.commands.back().bindings[10].name == "key_cache");
    REQUIRE(commands.commands.back().bindings[11].name == "value_cache");
    REQUIRE(commands.constant_initializations.size() == (expect_synthetic_inverse_frequencies ? 1 : 0));
}

static void run_qwen_attention_postprocess_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 8 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        constexpr int64_t                     token_count          = 4;
        constexpr int64_t                     query_head_count     = 4;
        constexpr int64_t                     key_value_head_count = 2;
        constexpr int64_t                     cache_row_count      = 16;
        const QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count);
        schedule_qwen_attention_postprocess_command(ctx, tensors, token_count, query_head_count, key_value_head_count,
                                                    cache_row_count);
    }

    {
        const QwenAttentionPostprocessTensors tensors =
            build_qwen_attention_postprocess_graph(ctx, 4, 4, 2, 16, 0.00001f);
        ggml::hrx::GraphImportResult imported = import_qwen_attention_postprocess_graph(ctx, tensors);
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) !=
                    "qwen3_moe:qwen3_moe_attention_postprocess_f32_f16");
        }
    }

    {
        const QwenAttentionPostprocessTensors tensors =
            build_qwen_attention_postprocess_graph(ctx, 4, 4, 2, 16, 0.000001f, false);
        schedule_qwen_attention_postprocess_command(ctx, tensors, 4, 4, 2, 16, true);
    }

    {
        const QwenAttentionPostprocessTensors tensors = build_qwen_attention_postprocess_graph(
            ctx, 4, 4, 2, 16, 0.000001f, false, 8192, 1000000.0f, 0.25f, 1.0f, 32.0f, 1.0f);
        schedule_qwen_attention_postprocess_command(ctx, tensors, 4, 4, 2, 16, true, "1.13862944");
    }

    {
        constexpr int64_t                     token_count          = 13;
        constexpr int64_t                     query_head_count     = 32;
        constexpr int64_t                     key_value_head_count = 4;
        constexpr int64_t                     cache_row_count      = 512;
        const QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);
        schedule_qwen_attention_postprocess_command(ctx, tensors, token_count, query_head_count, key_value_head_count,
                                                    cache_row_count, true);
    }

    {
        constexpr int64_t token_count          = 13;
        constexpr int64_t query_head_count     = 32;
        constexpr int64_t key_value_head_count = 4;
        constexpr int64_t cache_row_count      = 512;

        const QwenAttentionPostprocessTensors first = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);
        const QwenAttentionPostprocessTensors second = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count, 0.000001f, false);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, first.query_output);
        ggml_build_forward_expand(graph, first.key_output);
        ggml_build_forward_expand(graph, first.value_output);
        ggml_build_forward_expand(graph, second.query_output);
        ggml_build_forward_expand(graph, second.key_output);
        ggml_build_forward_expand(graph, second.value_output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 8);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 8);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t               token_count          = 4;
        constexpr int64_t               query_head_count     = 4;
        constexpr int64_t               key_value_head_count = 2;
        constexpr int64_t               cache_row_count      = 16;
        QwenAttentionPostprocessTensors tensors              = build_qwen_attention_postprocess_graph(
            ctx, token_count, query_head_count, key_value_head_count, cache_row_count);
        ggml_tensor * flash_output = append_qwen_flash_attention_consumer(ctx, tensors, token_count, query_head_count,
                                                                          key_value_head_count, cache_row_count);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, tensors.key_output);
        ggml_build_forward_expand(graph, tensors.value_output);
        ggml_build_forward_expand(graph, flash_output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        std::vector<bool>        covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan   plan;
        ggml::hrx::DispatchMatch match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes,
                                        producer_index_for_tensor(imported.graph, tensors.query_reshape), match));
        append_match_to_plan(plan, match, covered_nodes);
        REQUIRE(plan.valid());
        REQUIRE(plan.initialization_dispatches.size() == 2);

        const ggml::hrx::Dispatch & context_capture = plan.initialization_dispatches[0];
        const ggml::hrx::Dispatch & metadata        = plan.initialization_dispatches[1];
        REQUIRE(kernel_name_for_id(context_capture.kernel.kernel_id) == "qwen:qwen_attention_context_base_capture");
        REQUIRE(kernel_name_for_id(metadata.kernel.kernel_id) == "qwen3_moe:qwen_attention_metadata");
        REQUIRE(context_capture.bindings.size() == 2);
        REQUIRE(metadata.bindings.size() == 5);
        REQUIRE(context_capture.bindings[1].value == metadata.bindings[0].value);
        REQUIRE(metadata.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(metadata.kernel.integer_parameters.at("context_capacity") == cache_row_count);

        const ggml::hrx::Value * positions_value     = imported.graph.values().find_tensor(tensors.positions);
        const ggml::hrx::Value * key_indices_value   = imported.graph.values().find_tensor(tensors.key_cache_indices);
        const ggml::hrx::Value * value_indices_value = imported.graph.values().find_tensor(tensors.value_cache_indices);
        const ggml::hrx::Value * mask_value          = imported.graph.values().find_tensor(tensors.mask);
        REQUIRE(positions_value != nullptr);
        REQUIRE(key_indices_value != nullptr);
        REQUIRE(value_indices_value != nullptr);
        REQUIRE(mask_value != nullptr);
        REQUIRE(context_capture.bindings[0].value == positions_value->id);
        REQUIRE(metadata.bindings[1].value == positions_value->id);
        REQUIRE(metadata.bindings[2].value == key_indices_value->id);
        REQUIRE(metadata.bindings[3].value == value_indices_value->id);
        REQUIRE(metadata.bindings[4].value == mask_value->id);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.initialization_commands.size() == 2);
        REQUIRE(commands.initialization_commands[0].bindings[0].name == "positions");
        REQUIRE(commands.initialization_commands[0].bindings[1].name == "control");
        REQUIRE(commands.initialization_commands[1].bindings[0].name == "control");
        REQUIRE(commands.initialization_commands[1].bindings[4].name == "attention_mask");
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, context_capture.bindings[1].value) !=
                nullptr);

        ggml::hrx::CommandProgram initialization_commands = copy_command_program_shape(commands);
        initialization_commands.commands.clear();
        REQUIRE(command_program_verifies(initialization_commands));
    }

    ggml_free(ctx);
}

static void run_packed_f16_producer_consumer_checks() {
    ggml_init_params params = {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type type;
        int64_t tokens;
        int64_t hidden;
        int64_t outputs;
        bool side_use;
        bool packed;
        bool alternate;
        bool packed_copy = false;
    } cases[] = {
        { GGML_TYPE_Q6_K,  512, 4096, 2048, false, true,  false },
        { GGML_TYPE_Q6_K, 1024, 4096, 2048, false, true,  false },
        { GGML_TYPE_Q6_K,  256, 4096, 2048, false, false, true  },
        { GGML_TYPE_Q6_K,  512, 4096, 1024, false, false, true  },
        { GGML_TYPE_Q4_K,  512, 4096, 4096, false, true,  false },
        { GGML_TYPE_Q4_K,  512, 8192, 2048, false, false, true  },
        { GGML_TYPE_Q4_K,  512,12288, 4096, false, true,  false },
        { GGML_TYPE_Q4_K,  512,12288, 4096, true,  false, true  },
        { GGML_TYPE_Q4_K,  512,32768, 4096, false, false, false },
        { GGML_TYPE_Q4_K, 1024, 4096, 4096, false, false, true  },
        { GGML_TYPE_Q6_K,  512, 4096, 2048, true,  false, true, true },
    };

    for (const auto & test : cases) {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, test.tokens);
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, test.hidden);
        ggml_tensor * up_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, test.hidden);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up = ggml_mul_mat(ctx, up_weight, input);
        ggml_tensor * glu = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, test.type, test.hidden, test.outputs);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, glu);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        if (test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, glu, 0.5f));
        }
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto * value = imported.graph.values().find_tensor(glu);
        REQUIRE(value != nullptr);
        const auto * generated =
            plan.metadata.find_generated_resource(value->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
        const auto * alternate = plan.metadata.find_alternate_value(value->id, GGML_TYPE_F16, value->byte_count / 2);
        REQUIRE((generated != nullptr) == (test.packed || test.packed_copy));
        REQUIRE((alternate != nullptr) == test.alternate);
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32";
        });
        REQUIRE(producer != plan.dispatches.end());
        require_compile_parameter(*producer, "ggml.mul_mat_swiglu.f16_output_layout", test.packed ? "1" : "0");
        if (test.packed) {
            REQUIRE(generated->byte_count == value->byte_count / 2);
            REQUIRE(producer->bindings.back().value == generated->generated_value);
            const auto consumer = std::next(producer);
            REQUIRE(consumer != plan.dispatches.end());
            REQUIRE(consumer->bindings.front().value == generated->generated_value);
            if (test.type == GGML_TYPE_Q4_K) {
                require_compile_parameter(*consumer, "ggml.mul_mat.f16_input_layout", "1");
            }
        } else if (test.packed_copy) {
            const auto copy = std::next(producer);
            REQUIRE(copy != plan.dispatches.end());
            REQUIRE(kernel_name_for_id(copy->kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major");
            REQUIRE(copy->bindings.front().value == alternate->alternate_value);
            REQUIRE(copy->bindings.back().value == generated->generated_value);
            REQUIRE(std::next(copy) != plan.dispatches.end());
            REQUIRE(std::next(copy)->bindings.front().value == generated->generated_value);
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    ggml_free(ctx);
}

static void run_generic_swiglu_k16_publish_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t input_size  = 2048;
    constexpr int64_t hidden_size = 8192;
    constexpr int64_t output_size = 2048;
    constexpr int64_t token_count = 512;
    ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
    ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
    ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
    ggml_tensor * gate        = ggml_mul_mat(ctx, gate_weight, input);
    ggml_tensor * up          = ggml_mul_mat(ctx, up_weight, input);
    ggml_tensor * glu         = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * projection_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, hidden_size, output_size);
    ggml_tensor * output            = ggml_mul_mat(ctx, projection_weight, glu);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    REQUIRE(plan.valid());

    const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
        return kernel_name_for_id(dispatch.kernel.kernel_id) ==
               "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32_k16";
    });
    REQUIRE(producer != plan.dispatches.end());
    REQUIRE(producer->bindings.size() == 5);
    const ggml::hrx::Value * glu_value = imported.graph.values().find_tensor(glu);
    REQUIRE(glu_value != nullptr);
    const auto * packed =
        plan.metadata.find_generated_resource(glu_value->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
    REQUIRE(packed != nullptr);
    REQUIRE(packed->byte_count == glu_value->byte_count / 2);
    REQUIRE(producer->bindings.back().value == packed->generated_value);
    const auto consumer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
        return &dispatch != &*producer && !dispatch.bindings.empty() &&
               dispatch.bindings.front().value == packed->generated_value;
    });
    REQUIRE(consumer != plan.dispatches.end());
    REQUIRE(std::none_of(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
        return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major";
    }));

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(command_program_verifies(commands));
    ggml_free(ctx);
}

static void run_generic_swiglu_k16_publication_edge_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t input_size  = 2048;
        constexpr int64_t hidden_size = 8192;
        constexpr int64_t token_count = 512;
        ggml_tensor *     input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor *     gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
        ggml_tensor *     up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
        ggml_tensor *     gate        = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor *     up          = ggml_mul_mat(ctx, up_weight, input);
        ggml_tensor *     glu         = ggml_swiglu_split(ctx, gate, up);
        REQUIRE(glu != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, glu);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());

        const ggml::hrx::Value * glu_value = imported.graph.values().find_tensor(glu);
        REQUIRE(glu_value != nullptr);
        REQUIRE(plan.metadata.find_generated_resource(glu_value->id, ggml::hrx::GeneratedResourceRole::F16K16Major) ==
                nullptr);
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32";
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings.size() == 4);
        REQUIRE(producer->bindings.back().value == glu_value->id);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 8 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t input_size        = 2048;
        constexpr int64_t hidden_size       = 8192;
        constexpr int64_t token_count       = 512;
        ggml_tensor *     input             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor *     gate_weight       = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
        ggml_tensor *     up_weight         = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, input_size, hidden_size);
        ggml_tensor *     gate              = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor *     up                = ggml_mul_mat(ctx, up_weight, input);
        ggml_tensor *     glu               = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor *     reshaped          = ggml_reshape_2d(ctx, glu, 4096, 1024);
        ggml_tensor *     projection_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 4096, 2048);
        ggml_tensor *     output            = ggml_mul_mat(ctx, projection_weight, reshaped);
        REQUIRE(output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());

        const ggml::hrx::Value * glu_value      = imported.graph.values().find_tensor(glu);
        const ggml::hrx::Value * reshaped_value = imported.graph.values().find_tensor(reshaped);
        REQUIRE(glu_value != nullptr);
        REQUIRE(reshaped_value != nullptr);
        REQUIRE(plan.metadata.find_generated_resource(glu_value->id, ggml::hrx::GeneratedResourceRole::F16K16Major) ==
                nullptr);
        const auto * packed =
            plan.metadata.find_generated_resource(reshaped_value->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
        REQUIRE(packed != nullptr);

        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32";
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings.size() == 4);
        const auto copy = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major" &&
                   dispatch.bindings.front().value == reshaped_value->id;
        });
        REQUIRE(copy != plan.dispatches.end());
        REQUIRE(copy->bindings.back().value == packed->generated_value);
        require_compile_parameter(*copy, "ggml.copy_f16_k16_major.input_is_f16", "0");
        const auto consumer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return &dispatch != &*copy && !dispatch.bindings.empty() &&
                   dispatch.bindings.front().value == packed->generated_value;
        });
        REQUIRE(consumer != plan.dispatches.end());

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_tiled_matmul_alternate_publish_checks() {
    struct Case {
        ggml_type consumer_type;
        const char * producer_kernel;
        const char * consumer_kernel;
        ggml_type alternate_type;
    };

    const Case cases[] = {
        { GGML_TYPE_Q6_K, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_f16_alternate",
         "loom_libs:ggml_mul_mat_q6_k_f16_wmma_prefill_wave32",      GGML_TYPE_F16  },
        { GGML_TYPE_Q5_K, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32",
         "loom_libs:ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256", GGML_TYPE_Q8_1 },
    };

    for (const Case & test : cases) {
        ggml_init_params params = {};
        params.mem_size = 8 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * producer_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 2048);
        ggml_tensor * projection = ggml_mul_mat(ctx, producer_weight, input);
        ggml_tensor * consumer_weight = ggml_new_tensor_2d(ctx, test.consumer_type, 2048, 1024);
        ggml_tensor * output = ggml_mul_mat(ctx, consumer_weight, projection);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());

        const auto * projection_value = imported.graph.values().find_tensor(projection);
        REQUIRE(projection_value != nullptr);
        const size_t alternate_bytes = test.alternate_type == GGML_TYPE_F16 ?
            projection_value->byte_count / 2 :
            static_cast<size_t>(256) * ggml_row_size(GGML_TYPE_Q8_1, 2048);
        const auto * alternate =
            plan.metadata.find_alternate_value(projection_value->id, test.alternate_type, alternate_bytes);
        REQUIRE(alternate != nullptr);

        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == test.producer_kernel;
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings[2].value == projection_value->id);
        if (test.alternate_type == GGML_TYPE_F16) {
            REQUIRE(producer->bindings.size() == 4);
            REQUIRE(producer->bindings[3].value == alternate->alternate_value);
        } else {
            const auto quantize =
                std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32" &&
                       dispatch.bindings.front().value == projection_value->id;
            });
            REQUIRE(quantize != plan.dispatches.end());
            REQUIRE(quantize->bindings.back().value == alternate->alternate_value);
        }

        const auto consumer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == test.consumer_kernel;
        });
        REQUIRE(consumer != plan.dispatches.end());
        REQUIRE(consumer->bindings.front().value == alternate->alternate_value);

        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size = 12 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * producer_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 2048);
        ggml_tensor * projection = ggml_mul_mat(ctx, producer_weight, input);
        ggml_tensor * q5_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, 2048, 1024);
        ggml_tensor * q6_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 1024);
        ggml_tensor * q5_output = ggml_mul_mat(ctx, q5_weight, projection);
        ggml_tensor * q6_output = ggml_mul_mat(ctx, q6_weight, projection);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, q5_output);
        ggml_build_forward_expand(graph, q6_output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());

        const auto * projection_value = imported.graph.values().find_tensor(projection);
        REQUIRE(projection_value != nullptr);
        const size_t q8_bytes = static_cast<size_t>(256) * ggml_row_size(GGML_TYPE_Q8_1, 2048);
        const auto * q8 = plan.metadata.find_alternate_value(projection_value->id, GGML_TYPE_Q8_1, q8_bytes);
        REQUIRE(q8 != nullptr);
        const auto publication_count = std::count_if(
            plan.metadata.activation_publication_diagnostics().begin(),
            plan.metadata.activation_publication_diagnostics().end(), [&](const auto & diagnostic) {
                return diagnostic.source_value == projection_value->id;
            });
        REQUIRE(publication_count == 1);
        const auto publication = std::find_if(
            plan.metadata.activation_publication_diagnostics().begin(),
            plan.metadata.activation_publication_diagnostics().end(), [&](const auto & diagnostic) {
                return diagnostic.source_value == projection_value->id;
            });
        REQUIRE(publication != plan.metadata.activation_publication_diagnostics().end());
        REQUIRE(publication->requested_format == "q8-1-x4");

        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32" &&
                   dispatch.bindings[2].value == projection_value->id;
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings.size() == 3);

        const auto q5_consumer =
            std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                           "loom_libs:ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256" &&
                       dispatch.bindings.front().value == q8->alternate_value;
            });
        REQUIRE(q5_consumer != plan.dispatches.end());
        const auto q6_consumer =
            std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                       "loom_libs:ggml_mul_mat_q6_k_f16_wmma_prefill_wave32";
            });
        REQUIRE(q6_consumer != plan.dispatches.end());
        const auto f16_copy =
            std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f32_f16" &&
                       dispatch.bindings.front().value == projection_value->id;
            });
        REQUIRE(f16_copy != plan.dispatches.end());
        REQUIRE(q6_consumer->bindings.front().value == f16_copy->bindings.back().value);

        const size_t q8_quantizer_count = std::count_if(
            plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32" &&
                       dispatch.bindings.front().value == projection_value->id;
            });
        REQUIRE(q8_quantizer_count == 1);

        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_tiled_pair_matmul_postops_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    auto build_glu = [&]() {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 256);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 8);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * glu = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(glu != nullptr);
        return glu;
    };

    {
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(bias != nullptr);
        ggml_tensor * output = ggml_add(ctx, build_glu(), bias);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_bias_residual_publish_f32", 4,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_ADD },
            { "input", "gate_weight", "up_weight", "bias", "residual_input", "residual_output" });
    }

    {
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 8);
        REQUIRE(residual != nullptr);
        ggml_tensor * output = ggml_add(ctx, build_glu(), residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_bias_residual_publish_f32", 4,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_ADD },
            { "input", "gate_weight", "up_weight", "bias", "residual_input", "residual_output" });
    }

    {
        ggml_tensor * bias     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 8);
        REQUIRE(bias != nullptr);
        REQUIRE(residual != nullptr);
        ggml_tensor * biased = ggml_add(ctx, build_glu(), bias);
        REQUIRE(biased != nullptr);
        ggml_tensor * output = ggml_add(ctx, biased, residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_pair_input_f32_binary_bias_residual_publish_f32", 5,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_ADD, GGML_OP_ADD },
            { "input", "gate_weight", "up_weight", "bias", "residual_input", "residual_output" });
    }

    ggml_free(ctx);
}

static void run_swiglu_q8_output_checks() {
    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t outputs;
        ggml_type down_type;
        int projections;
        bool terminal;
        bool side_use;
        bool observed;
        bool geglu;
        bool weight_alias;
        bool packed;
    } cases[] = {
        { 2, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 3, 5120, 17408, GGML_TYPE_Q6_K, 1, false, false, false, false, false, true  },
        { 4, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 5, 5120, 17408, GGML_TYPE_Q6_K, 2, false, true,  true,  false, false, true  },
        { 5, 4096, 16384, GGML_TYPE_Q4_K, 1, true,  false, false, false, false, true  },
        { 5, 5120, 32768, GGML_TYPE_Q6_K, 1, false, false, false, false, false, true  },
        { 1, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 6, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, false },
        { 5, 2048, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 5, 6144, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 5, 5120,  8192, GGML_TYPE_Q4_K, 1, false, false, false, false, false, true  },
        { 5, 5120, 33024, GGML_TYPE_Q4_K, 1, false, false, false, false, false, false },
        { 5, 5120, 17408, GGML_TYPE_F32,  1, false, true,  false, false, false, false },
        { 5, 5120, 17408, GGML_TYPE_Q6_K, 1, true,  false, false, false, false, true  },
        { 5, 5120, 17408, GGML_TYPE_Q4_K, 0, false, true,  true,  false, false, false },
        { 5, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, true,  false, false },
        { 5, 5120, 17408, GGML_TYPE_Q4_K, 1, false, false, false, false, true,  false },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size = 4 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.inputs, test.tokens);
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, test.inputs, test.outputs);
        ggml_tensor * up_weight   = test.weight_alias ?
                                        ggml_view_tensor(ctx, gate_weight) :
            ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, test.inputs, test.outputs);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up = ggml_mul_mat(ctx, up_weight, input);
        ggml_tensor * glu = ggml_glu_split(ctx, gate, up, test.geglu ? GGML_GLU_OP_GEGLU : GGML_GLU_OP_SWIGLU);
        if (test.observed) {
            ggml_set_output(glu);
        }
        ggml_cgraph * graph = ggml_new_graph(ctx);
        for (int i = 0; i < test.projections; ++i) {
            ggml_tensor * down_weight = ggml_new_tensor_2d(ctx, test.down_type, test.outputs, 5120);
            ggml_tensor * down = ggml_mul_mat(ctx, down_weight, glu);
            ggml_build_forward_expand(graph, test.terminal ? down : ggml_add(ctx, down, ggml_dup_tensor(ctx, down)));
        }
        ggml_tensor * side = test.side_use ? ggml_scale(ctx, glu, 0.5f) : nullptr;
        if (side != nullptr) {
            ggml_build_forward_expand(graph, side);
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        const bool scheduled = scheduler.schedule_graph(imported.graph, test_dispatch_target());
        const auto & plan = scheduler.plan();
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot_q8_output";
        });
        REQUIRE((producer != plan.dispatches.end()) == test.packed);
        if (test.outputs > 32768) {
            REQUIRE(!scheduled);
            REQUIRE(status_contains(plan.status, "unsupported HRX node"));
            ggml_free(ctx);
            continue;
        }
        REQUIRE(scheduled);
        REQUIRE(plan.valid());
        const auto * output = imported.graph.values().find_tensor(glu);
        REQUIRE(output != nullptr);
        if (test.packed) {
            const size_t bytes = static_cast<size_t>(test.tokens) * ggml_row_size(GGML_TYPE_Q8_1, test.outputs);
            const auto * alternate = plan.metadata.find_alternate_value(output->id, GGML_TYPE_Q8_1, bytes);
            REQUIRE(alternate != nullptr);
            REQUIRE(producer->bindings.size() == 5);
            REQUIRE(producer->bindings[3].value == output->id);
            REQUIRE(producer->bindings[4].value == alternate->alternate_value);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32" &&
                       dispatch.bindings.front().value == output->id;
            }) == 0);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return &dispatch != &*producer && !dispatch.bindings.empty() &&
                       dispatch.bindings.front().value == alternate->alternate_value;
            }) == test.projections);
        }
        if (side != nullptr) {
            const auto * side_value = imported.graph.values().find_tensor(side);
            REQUIRE(side_value != nullptr);
            const auto consumer =
                std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return !dispatch.bindings.empty() && dispatch.bindings.back().value == side_value->id;
            });
            REQUIRE(consumer != plan.dispatches.end());
            REQUIRE(consumer->bindings.front().value == output->id);
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_k16_major_preparation_reuse_checks() {
    ggml_init_params params = {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type first_type;
        ggml_type second_type;
        bool shared;
        bool reshape;
    } cases[] = {
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, true,  false },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, true,  false },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, true,  false },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, true,  false },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, false, false },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, false, false },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, false, true  },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, false, true  },
    };

    for (const auto & test : cases) {
        ggml_tensor * first = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 512);
        ggml_tensor * second = test.reshape ? ggml_reshape_2d(ctx, first, 2048, 1024) :
                                 test.shared  ? first :
                                                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 512);
        ggml_tensor * inputs[] = { first, second };
        ggml_tensor * outputs[2];
        ggml_cgraph * graph = ggml_new_graph(ctx);
        for (int i = 0; i < 2; ++i) {
            const ggml_type type = i == 0 ? test.first_type : test.second_type;
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, inputs[i]->ne[0], 4096);
            outputs[i] = ggml_mul_mat(ctx, weight, inputs[i]);
            ggml_build_forward_expand(graph, outputs[i]);
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto copy_count =
            std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major";
        });
        REQUIRE(copy_count == (test.shared ? 1 : 2));
        ggml::hrx::ValueId packed[2];
        for (int i = 0; i < 2; ++i) {
            const auto * input = imported.graph.values().find_tensor(inputs[i]);
            const auto * output = imported.graph.values().find_tensor(outputs[i]);
            REQUIRE(input != nullptr);
            REQUIRE(output != nullptr);
            const auto * generated =
                plan.metadata.find_generated_resource(input->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
            REQUIRE(generated != nullptr);
            REQUIRE(generated->byte_count == input->byte_count / 2);
            packed[i] = generated->generated_value;
            const auto consumer =
                std::find_if(plan.dispatches.begin(), plan.dispatches.end(),
                             [&](const auto & dispatch) { return dispatch.bindings.back().value == output->id; });
            REQUIRE(consumer != plan.dispatches.end());
            REQUIRE(consumer->bindings.front().value == packed[i]);
        }
        REQUIRE((packed[0] == packed[1]) == test.shared);
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_rmsnorm_binary_k16_output_checks() {
    const struct {
        ggml_type type;
        int64_t hidden;
        int64_t tokens;
        int projections;
        bool side_use;
        bool observed;
        bool inplace;
        bool multiply;
        bool packed;
    } cases[] = {
        { GGML_TYPE_Q4_K, 2048,  512, 1, false, false, false, true,  true  },
        { GGML_TYPE_Q4_K, 3072,  512, 1, false, false, false, true,  true  },
        { GGML_TYPE_Q4_K, 4096,  512, 1, false, false, false, true,  true  },
        { GGML_TYPE_Q4_K, 5120,  512, 2, false, false, false, true,  true  },
        { GGML_TYPE_Q6_K, 8192,  512, 2, true,  false, false, true,  true  },
        { GGML_TYPE_Q6_K, 5120, 1024, 1, false, false, false, true,  true  },
        { GGML_TYPE_Q4_K, 5120,  512, 2, true,  true,  false, true,  true  },
        { GGML_TYPE_Q4_K, 6144,  512, 1, false, false, false, true,  true  },
        { GGML_TYPE_Q4_K, 7168,  512, 1, false, false, false, true,  false },
        { GGML_TYPE_Q4_K, 4096,  256, 1, false, false, false, true,  false },
        { GGML_TYPE_Q4_K, 5120,    5, 1, false, false, false, true,  false },
        { GGML_TYPE_Q4_K, 4096, 1024, 1, false, false, false, true,  false },
        { GGML_TYPE_Q4_K, 5120,  512, 0, true,  true,  false, true,  false },
        { GGML_TYPE_F32,  5120,  512, 1, true,  false, false, true,  false },
        { GGML_TYPE_Q4_K, 5120,  512, 1, false, false, true,  true,  false },
        { GGML_TYPE_Q4_K, 5120,  512, 1, false, false, false, false, false },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size = 4 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden);
        ggml_tensor * rms =
            test.inplace ? ggml_rms_norm_inplace(ctx, input, 1.0e-6f) : ggml_rms_norm(ctx, input, 1.0e-6f);
        ggml_tensor * norm = test.inplace ? ggml_mul_inplace(ctx, rms, weight) :
                             test.multiply ? ggml_mul(ctx, rms, weight) :
                                             ggml_add(ctx, rms, weight);
        if (test.observed) {
            ggml_set_output(norm);
        }
        ggml_cgraph * graph = ggml_new_graph(ctx);
        for (int i = 0; i < test.projections; ++i) {
            ggml_tensor * projection = ggml_new_tensor_2d(ctx, test.type, test.hidden, 4096);
            ggml_build_forward_expand(graph, ggml_mul_mat(ctx, projection, norm));
        }
        ggml_tensor * side = test.side_use ? ggml_scale(ctx, norm, 0.5f) : nullptr;
        if (side != nullptr) {
            ggml_build_forward_expand(graph, side);
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_f32_k16";
        });
        REQUIRE((producer != plan.dispatches.end()) == test.packed);
        const auto * output = imported.graph.values().find_tensor(norm);
        REQUIRE(output != nullptr);
        if (test.packed) {
            const auto * generated =
                plan.metadata.find_generated_resource(output->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
            REQUIRE(generated != nullptr);
            REQUIRE(generated->byte_count == output->byte_count / 2);
            REQUIRE(producer->bindings.size() == 4);
            REQUIRE(producer->bindings[2].value == output->id);
            REQUIRE(producer->bindings[3].value == generated->generated_value);
            REQUIRE(plan.metadata.find_alternate_value(output->id, GGML_TYPE_F16, output->byte_count / 2) == nullptr);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major";
            }) == 0);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return &dispatch != &*producer && !dispatch.bindings.empty() &&
                       dispatch.bindings.front().value == generated->generated_value;
            }) == test.projections);
        } else if (test.hidden == 7168 && test.tokens == 512) {
            const auto * f16 =
                plan.metadata.find_alternate_value(output->id, GGML_TYPE_F16, output->byte_count / 2);
            REQUIRE(f16 != nullptr);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major";
            }) == 1);
        } else if (test.projections == 0) {
            REQUIRE(plan.metadata.find_generated_resource(
                        output->id, ggml::hrx::GeneratedResourceRole::F16K16Major) == nullptr);
            REQUIRE(plan.metadata.find_alternate_value(output->id) == nullptr);
        }
        if (side != nullptr) {
            const auto * side_value = imported.graph.values().find_tensor(side);
            REQUIRE(side_value != nullptr);
            const auto consumer =
                std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
                return !dispatch.bindings.empty() && dispatch.bindings.back().value == side_value->id;
            });
            REQUIRE(consumer != plan.dispatches.end());
            REQUIRE(consumer->bindings.front().value == output->id);
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_rmsnorm_binary_f16_output_checks() {
    {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 7168, 512);
        ggml_tensor * norm_weight       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7168);
        ggml_tensor * rms               = ggml_rms_norm(ctx, input, 1.0e-6f);
        ggml_tensor * norm              = ggml_mul(ctx, rms, norm_weight);
        ggml_tensor * projection_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 7168, 2048);
        ggml_tensor * output            = ggml_mul_mat(ctx, projection_weight, norm);
        ggml_cgraph * graph             = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_f32_f16";
        });
        REQUIRE(producer != plan.dispatches.end());
        REQUIRE(producer->bindings.size() == 4);
        const ggml::hrx::Value * norm_value = imported.graph.values().find_tensor(norm);
        REQUIRE(norm_value != nullptr);
        const auto * f16 =
            plan.metadata.find_alternate_value(norm_value->id, GGML_TYPE_F16, norm_value->byte_count / 2);
        REQUIRE(f16 != nullptr);
        REQUIRE(producer->bindings.back().value == f16->alternate_value);
        const auto consumer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return &dispatch != &*producer && !dispatch.bindings.empty() &&
                   dispatch.bindings.front().value == f16->alternate_value;
        });
        REQUIRE(consumer != plan.dispatches.end());
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 512);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        ggml_tensor * rms         = ggml_rms_norm(ctx, input, 1.0e-6f);
        ggml_tensor * norm        = ggml_mul(ctx, rms, norm_weight);
        ggml_tensor * q8_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, 2048, 1024);
        ggml_tensor * f16_weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 2048);
        ggml_tensor * q8_output   = ggml_mul_mat(ctx, q8_weight, norm);
        ggml_tensor * f16_output  = ggml_mul_mat(ctx, f16_weight, norm);
        ggml_cgraph * graph       = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, q8_output);
        ggml_build_forward_expand(graph, f16_output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_q8_1_x4_publish";
        });
        REQUIRE(producer != plan.dispatches.end());
        require_compile_parameter(*producer, "ggml.rmsnorm_binary_q8_1_x4.publish_f16", "1");
        REQUIRE(producer->bindings.size() == 5);
        const ggml::hrx::Value * norm_value = imported.graph.values().find_tensor(norm);
        REQUIRE(norm_value != nullptr);
        const auto * q8 =
            plan.metadata.find_alternate_value(norm_value->id, GGML_TYPE_Q8_1, qwen_q8_1_x4_size(512, 2048));
        const auto * f16 =
            plan.metadata.find_alternate_value(norm_value->id, GGML_TYPE_F16, norm_value->byte_count / 2);
        REQUIRE(q8 != nullptr);
        REQUIRE(f16 != nullptr);
        REQUIRE(producer->bindings[3].value == q8->alternate_value);
        REQUIRE(producer->bindings[4].value == f16->alternate_value);
        REQUIRE(std::any_of(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return &dispatch != &*producer && !dispatch.bindings.empty() &&
                   dispatch.bindings.front().value == q8->alternate_value;
        }));
        REQUIRE(std::any_of(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return &dispatch != &*producer && !dispatch.bindings.empty() &&
                   dispatch.bindings.front().value == f16->alternate_value;
        }));
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }

    {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t hidden = 2048;
        constexpr int64_t tokens = 2;
        ggml_tensor * input       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, tokens);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        ggml_tensor * norm        = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1.0e-5f), norm_weight);
        ggml_tensor * reshaped    = ggml_reshape_2d(ctx, norm, hidden, tokens);
        ggml_tensor * projection  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, hidden, 6144);
        ggml_tensor * output      = ggml_silu(ctx, ggml_mul_mat(ctx, projection, reshaped));
        ggml_cgraph * graph       = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_q8_1_x4_publish";
        });
        REQUIRE(producer != plan.dispatches.end());
        require_compile_parameter(*producer, "ggml.rmsnorm_binary_q8_1_x4.publish_f16", "1");
        REQUIRE(producer->bindings.size() == 5);
        const ggml::hrx::Value * norm_value = imported.graph.values().find_tensor(norm);
        REQUIRE(norm_value != nullptr);
        const size_t q8_bytes = qwen_q8_1_x4_size(tokens, hidden);
        const auto * q8 = plan.metadata.find_alternate_value(norm_value->id, GGML_TYPE_Q8_1, q8_bytes);
        REQUIRE(q8 != nullptr);
        REQUIRE(producer->bindings[3].value == q8->alternate_value);
        REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32";
        }) == 0);
        REQUIRE(std::any_of(std::next(producer), plan.dispatches.end(), [&](const auto & dispatch) {
            return !dispatch.bindings.empty() && dispatch.bindings.front().value == q8->alternate_value;
        }));
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_rmsnorm_gate_packed_output_checks() {
    ggml_init_params params = {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type type;
        int64_t hidden;
        int64_t channels;
        int64_t tokens;
        bool side_use;
        bool packed;
        int copies;
    } cases[] = {
        { GGML_TYPE_Q4_K,  128, 6144,  512, false, true,  0 },
        { GGML_TYPE_Q6_K,  256, 4096, 1024, false, true,  0 },
        { GGML_TYPE_Q6_K, 1024, 4096,  512, false, true,  0 },
        { GGML_TYPE_Q4_K,  128, 6144,  512, true,  false, 1 },
        { GGML_TYPE_Q4_K,  128, 4096,  256, false, false, 0 },
        { GGML_TYPE_Q4_K,  256,  256,  512, false, true,  0 },
        { GGML_TYPE_Q5_K,  128, 6144,   19, false, false, 0 },
        { GGML_TYPE_IQ4_XS, 128, 6144,  19, false, false, 0 },
    };

    for (const auto & test : cases) {
        const int64_t heads = test.channels / test.hidden;
        ggml_tensor * input = heads == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens) :
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test.hidden, heads, test.tokens);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden);
        ggml_tensor * gate = ggml_dup_tensor(ctx, input);
        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1.0e-6f), weight);
        ggml_tensor * gated = ggml_mul(ctx, norm, ggml_silu(ctx, gate));
        ggml_tensor * reshaped = heads == 1 ? gated : ggml_reshape_2d(ctx, gated, test.channels, test.tokens);
        ggml_tensor * projection = ggml_new_tensor_2d(ctx, test.type, test.channels, 4096);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_mul_mat(ctx, projection, reshaped));
        if (test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, gated, 0.5f));
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_gate_f32_publish";
        });
        REQUIRE(producer != plan.dispatches.end());
        require_compile_parameter(*producer, "ggml.rmsnorm_gate_f32.publish_q8", "0");
        require_compile_parameter(*producer, "ggml.rmsnorm_gate_f32.f16_output_row_width",
                                  std::to_string(test.packed ? test.channels : 0));
        const auto * output = imported.graph.values().find_tensor(gated);
        const auto * matmul_input = imported.graph.values().find_tensor(reshaped);
        REQUIRE(output != nullptr && matmul_input != nullptr);
        const auto * alternate = plan.metadata.find_alternate_value(output->id, GGML_TYPE_F16, output->byte_count / 2);
        REQUIRE((alternate == nullptr) == test.packed);
        if (test.packed) {
            const auto * generated =
                plan.metadata.find_generated_resource(matmul_input->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
            REQUIRE(generated != nullptr);
            REQUIRE(producer->bindings.back().value == generated->generated_value);
            REQUIRE(std::next(producer) != plan.dispatches.end());
            REQUIRE(std::next(producer)->bindings.front().value == generated->generated_value);
        }
        REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_copy_f16_k16_major";
        }) == test.copies);
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        ggml_tensor * gate   = ggml_dup_tensor(ctx, input);
        ggml_tensor * norm   = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1.0e-6f), weight);
        ggml_tensor * gated  = ggml_mul(ctx, norm, ggml_silu(ctx, gate));
        ggml_cgraph * graph  = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_scale(ctx, gated, 0.5f));

        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_gate_f32_publish";
        });
        REQUIRE(producer != plan.dispatches.end());
        require_compile_parameter(*producer, "ggml.rmsnorm_gate_f32.publish_q8", "0");
        REQUIRE(producer->bindings.size() == 5);
        const auto * output = imported.graph.values().find_tensor(gated);
        REQUIRE(output != nullptr);
        REQUIRE(plan.metadata.find_alternate_value(output->id) == nullptr);
        REQUIRE(plan.metadata.find_generated_resource(
                    output->id, ggml::hrx::GeneratedResourceRole::F16K16Major) == nullptr);
        REQUIRE(std::any_of(plan.transients.begin(), plan.transients.end(), [&](const auto & transient) {
            return transient.value == producer->bindings.back().value &&
                   transient.size == output->byte_count / 2;
        }));
        const auto commands = ggml::hrx::build_command_program(
            imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_rmsnorm_gate_q8_output_checks() {
    ggml_init_params params = {};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type type;
        int64_t hidden;
        int64_t channels;
        int64_t tokens;
        int64_t outputs;
        bool side_use;
        bool terminal;
        bool q8;
    } cases[] = {
        { GGML_TYPE_Q4_K,  128, 6144,   1, 5120, false, false, true  },
        { GGML_TYPE_Q4_K,  128, 6144,   2, 5120, false, false, true  },
        { GGML_TYPE_Q4_K,  128, 6144,   3, 5120, false, false, true  },
        { GGML_TYPE_Q4_K,  128, 6144,   4, 5120, false, false, true  },
        { GGML_TYPE_Q4_K,  128, 6144,   5, 5120, false, false, true  },
        { GGML_TYPE_Q6_K,  256, 4096,   3,  256, false, false, true  },
        { GGML_TYPE_Q4_K,  512, 1536,   5,  256, false, false, true  },
        { GGML_TYPE_Q4_K, 1024, 1024,   4,  256, false, false, true  },
        { GGML_TYPE_Q4_K,  256,  256,   1,   64, false, false, true  },
        { GGML_TYPE_Q4_K,  128, 6144,   6, 5120, false, false, false },
        { GGML_TYPE_Q4_K,  128, 6144, 256, 5120, false, false, false },
        { GGML_TYPE_Q4_K,  128, 6144, 512, 5120, false, false, false },
        { GGML_TYPE_Q4_K,  128, 6144,   5, 5120, true,  false, false },
        { GGML_TYPE_Q4_K,  128, 6144,   5,   48, false, false, false },
        { GGML_TYPE_Q6_K,  128, 6144,   5,  256, false, true,  false },
    };

    for (const auto & test : cases) {
        const int64_t heads = test.channels / test.hidden;
        ggml_tensor * input = heads == 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens) :
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test.hidden, heads, test.tokens);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden);
        ggml_tensor * gate = ggml_dup_tensor(ctx, input);
        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1.0e-6f), weight);
        ggml_tensor * gated = ggml_mul(ctx, norm, ggml_silu(ctx, gate));
        ggml_tensor * reshaped = heads == 1 ? gated : ggml_reshape_2d(ctx, gated, test.channels, test.tokens);
        ggml_tensor * projection = ggml_new_tensor_2d(ctx, test.type, test.channels, test.outputs);
        ggml_tensor * projected = ggml_mul_mat(ctx, projection, reshaped);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, test.terminal ? projected : ggml_scale(ctx, projected, 0.5f));
        if (test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, gated, 0.25f));
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            const auto publish_q8 = dispatch.kernel.compile_parameters.find("ggml.rmsnorm_gate_f32.publish_q8");
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_gate_f32_publish" &&
                   publish_q8 != dispatch.kernel.compile_parameters.end() && publish_q8->second == "1";
        });
        REQUIRE((producer != plan.dispatches.end()) == test.q8);
        if (test.q8) {
            require_compile_parameter(*producer, "ggml.rmsnorm_gate_f32.publish_q8", "1");
            const auto * output = imported.graph.values().find_tensor(gated);
            REQUIRE(output != nullptr);
            const size_t bytes = static_cast<size_t>(test.tokens) * ggml_row_size(GGML_TYPE_Q8_1, test.channels);
            const auto * alternate = plan.metadata.find_alternate_value(output->id, GGML_TYPE_Q8_1, bytes);
            REQUIRE(alternate != nullptr);
            REQUIRE(producer->bindings.back().value == alternate->alternate_value);
            REQUIRE(producer->bindings.back().length == bytes);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32";
            }) == 0);
            REQUIRE(std::any_of(std::next(producer), plan.dispatches.end(), [&](const auto & dispatch) {
                return std::any_of(dispatch.bindings.begin(), dispatch.bindings.end(),
                                   [&](const auto & binding) { return binding.value == alternate->alternate_value; });
            }));
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_symmetric_i4_consumer_qualification_checks() {
    const struct {
        bool direct_consumer;
        bool reshape_consumer;
        bool strided_consumer;
        bool q8_publication;
    } cases[] = {
        { true,  false, false, true  },
        { false, true,  false, true  },
        { false, false, true,  false },
        { false, false, false, false },
    };
    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t hidden = 2048;
        constexpr int64_t tokens = 2;
        const int64_t producer_hidden = test.strided_consumer ? hidden + 128 : hidden;
        ggml_tensor * input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, producer_hidden, tokens);
        ggml_tensor * weight     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, producer_hidden);
        ggml_tensor * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1.0e-6f), weight);
        ggml_cgraph * graph      = ggml_new_graph(ctx);
        ggml_tensor * consumer_input = normalized;
        if (test.reshape_consumer) {
            consumer_input = ggml_reshape_2d(ctx, normalized, hidden, tokens);
        } else if (test.strided_consumer) {
            consumer_input = ggml_view_2d(
                ctx, normalized, hidden, tokens, static_cast<size_t>(producer_hidden) * sizeof(float), 0);
        }
        if (test.direct_consumer || test.reshape_consumer || test.strided_consumer) {
            ggml_tensor * i4_projection = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, hidden, hidden);
            ggml_tensor * i4_projected  = ggml_mul_mat(ctx, i4_projection, consumer_input);
            ggml_build_forward_expand(graph, ggml_scale(ctx, i4_projected, 0.5f));
            if (!test.strided_consumer) {
                ggml_tensor * q8_projection = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, hidden, 512);
                ggml_tensor * q8_projected  = ggml_mul_mat(ctx, q8_projection, consumer_input);
                ggml_build_forward_expand(graph, ggml_scale(ctx, q8_projected, 0.5f));
            }
        } else {
            ggml_build_forward_expand(graph, ggml_scale(ctx, normalized, 0.5f));
        }

        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        const ggml::hrx::Value * normalized_value = imported.graph.values().find_tensor(normalized);
        const ggml::hrx::Value * qualified_input  = imported.graph.values().find_tensor(consumer_input);
        REQUIRE(normalized_value != nullptr);
        REQUIRE(qualified_input != nullptr);
        for (const ggml::hrx::GraphNode * consumer : imported.graph.index().consumers(qualified_input->id)) {
            REQUIRE(consumer == nullptr ||
                    !ggml::hrx::common_symmetric_i4_lowrow_mul_mat_eligible(
                        imported.graph, *consumer, *qualified_input));
        }
        if (test.strided_consumer) {
            ggml_free(ctx);
            continue;
        }
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto symmetric_producer =
            std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:ggml_rmsnorm_binary_symmetric_i4_k32";
        });
        REQUIRE(symmetric_producer == plan.dispatches.end());
        const auto q8_producer = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_binary_q8_1_x4_publish";
        });
        REQUIRE((q8_producer != plan.dispatches.end()) == test.q8_publication);
        if (test.q8_publication) {
            require_compile_parameter(*q8_producer, "ggml.rmsnorm_binary_q8_1_x4.publish_f16", "1");
            const auto * q8 = plan.metadata.find_alternate_value(
                normalized_value->id, GGML_TYPE_Q8_1, qwen_q8_1_x4_size(tokens, hidden));
            REQUIRE(q8 != nullptr);
            REQUIRE(q8_producer->bindings.size() == 5);
            REQUIRE(q8_producer->bindings[3].value == q8->alternate_value);
            REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:ggml_quantize_q8_1_x4_f32";
            }) == 0);
            REQUIRE(std::any_of(std::next(q8_producer), plan.dispatches.end(), [&](const auto & dispatch) {
                return !dispatch.bindings.empty() && dispatch.bindings.front().value == q8->alternate_value;
            }));
        } else if (!test.strided_consumer) {
            REQUIRE(plan.metadata.alternate_values().empty());
        }
        const auto commands = ggml::hrx::build_command_program(
            imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_quantized_conv4_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type type;
        int64_t inputs;
        int64_t channels;
        int64_t tokens;
        bool side_use;
        bool exposed;
        bool fused;
        bool gathered_state = false;
    } cases[] = {
        { GGML_TYPE_Q4_K, 5120, 10240, 512, false, false, true },
        { GGML_TYPE_Q6_K, 5120, 10240, 512, false, false, true },
        { GGML_TYPE_Q4_K, 4096, 8192, 512, false, false, true },
        { GGML_TYPE_Q6_K, 4096, 8192, 512, false, false, true },
        { GGML_TYPE_Q4_K, 5120, 10240, 512, true, false, false },
        { GGML_TYPE_Q4_K, 5120, 10240, 512, false, true, false },
        { GGML_TYPE_Q4_K, 5120, 10240, 256, false, false, false },
        { GGML_TYPE_Q4_K, 5120, 10240, 512, false, false, true, true },
        { GGML_TYPE_Q6_K, 5120, 10240, 512, false, false, true, true },
        { GGML_TYPE_Q4_K, 4096, 8192, 512, false, false, true, true },
        { GGML_TYPE_Q6_K, 4096, 8192, 512, false, false, true, true },
        { GGML_TYPE_Q4_K, 5120, 10240, 512, true, false, false, true },
        { GGML_TYPE_Q4_K, 5120, 10240, 512, false, true, false, true },
    };

    for (const auto & test : cases) {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.inputs, test.tokens);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, test.type, test.inputs, test.channels);
        ggml_tensor * raw = ggml_mul_mat(ctx, weight, input);
        if (test.exposed) {
            ggml_set_output(raw);
        }
        ggml_tensor * x = ggml_reshape_3d(ctx, raw, test.channels, test.tokens, 1);
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, test.channels, 1);
        if (test.gathered_state) {
            ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            state = ggml_get_rows(ctx, ggml_reshape_2d(ctx, state, 3 * test.channels, 1), indices);
            state = ggml_reshape_3d(ctx, state, 3, test.channels, 1);
        }
        ggml_tensor * filter = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, test.channels);
        ggml_tensor * window = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);
        ggml_tensor * tail =
            ggml_view_3d(ctx, window, 3, test.channels, 1, window->nb[1], window->nb[2], test.tokens * sizeof(float));
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * test.channels, 1);
        ggml_tensor * target = ggml_view_2d(ctx, cache, 3 * test.channels, 1, cache->nb[1], 0);
        ggml_tensor * output = ggml_silu(ctx, ggml_ssm_conv(ctx, window, filter));
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, tail, target));
        ggml_build_forward_expand(graph, output);
        if (test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, raw, 0.5f));
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [&](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   (test.gathered_state ? "loom_libs:ggml_mul_mat_quantized_f16_wmma_prefill_conv4_interior" :
                                          "loom_libs:ggml_mul_mat_quantized_f16_wmma_prefill_conv4");
        });
        REQUIRE((fused != plan.dispatches.end()) == test.fused);
        if (test.fused) {
            REQUIRE(fused->bindings.size() == (test.gathered_state ? 5 : 6));
            require_compile_parameter(*fused, "ggml.mul_mat.weight_format", test.type == GGML_TYPE_Q4_K ? "4" : "6");
            if (test.gathered_state) {
                const auto finish =
                    std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                    return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                           "loom_libs:llm_ssm_conv_dconv4_silu_prefill_finish_f32";
                });
                const auto gather =
                    std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                    return kernel_name_for_id(dispatch.kernel.kernel_id).find("get_rows") != std::string::npos;
                });
                REQUIRE(finish != plan.dispatches.end());
                REQUIRE(gather != plan.dispatches.end());
                REQUIRE(fused < gather && gather < finish);
                REQUIRE(finish->bindings[2].value == fused->bindings[4].value);
                REQUIRE(finish->bindings[3].value == fused->bindings[3].value);
                REQUIRE(finish->bindings[2].length == static_cast<size_t>(6 * test.channels) * sizeof(float));
            }
            REQUIRE(std::none_of(plan.transients.begin(), plan.transients.end(), [](const auto & transient) {
                return transient.name == "llm.ssm_conv.window_snapshot";
            }));
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_generic_ssm_conv_binary_dispatch_checks() {
    const struct {
        ggml::hrx::BinaryKind kind;
        bool                  conv_is_lhs;
    } cases[] = {
        { ggml::hrx::BinaryKind::Add, true  },
        { ggml::hrx::BinaryKind::Sub, true  },
        { ggml::hrx::BinaryKind::Sub, false },
        { ggml::hrx::BinaryKind::Mul, true  },
        { ggml::hrx::BinaryKind::Div, true  },
        { ggml::hrx::BinaryKind::Div, false },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t d_conv  = 4;
        constexpr int64_t d_inner = 64;
        constexpr int64_t n_t     = 8;
        ggml_tensor *     window  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv + n_t - 1, d_inner);
        ggml_tensor *     filter  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
        ggml_tensor *     operand = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, n_t);
        ggml_tensor *     conv    = ggml_ssm_conv(ctx, window, filter);
        ggml_tensor *     output  = nullptr;
        switch (test.kind) {
            case ggml::hrx::BinaryKind::Add:
                output = ggml_add(ctx, conv, operand);
                break;
            case ggml::hrx::BinaryKind::Sub:
                output = test.conv_is_lhs ? ggml_sub(ctx, conv, operand) : ggml_sub(ctx, operand, conv);
                break;
            case ggml::hrx::BinaryKind::Mul:
                output = ggml_mul(ctx, conv, operand);
                break;
            case ggml::hrx::BinaryKind::Div:
                output = test.conv_is_lhs ? ggml_div(ctx, conv, operand) : ggml_div(ctx, operand, conv);
                break;
            default:
                REQUIRE(false);
        }

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        REQUIRE(plan.dispatches.size() == 1);
        const auto & dispatch = plan.dispatches.front();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:llm_ssm_conv_binary_f32");
        REQUIRE(dispatch.kernel.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
        REQUIRE(dispatch.kernel.integer_parameters.at("n_t") == n_t);
        REQUIRE(dispatch.kernel.integer_parameters.at("n_s") == 1);
        REQUIRE(dispatch.kernel.compile_parameters.count("llm.ssm_conv.generic.n_t") == 0);
        REQUIRE(dispatch.kernel.compile_parameters.count("llm.ssm_conv.generic.n_s") == 0);
        REQUIRE(dispatch.bindings.size() == 4);
        require_compile_parameter(dispatch, "llm.ssm_conv.generic.binary_op",
                                  std::to_string(ggml::hrx::binary_kind_config_value(test.kind)));
        require_compile_parameter(dispatch, "llm.ssm_conv.generic.binary_lhs", test.conv_is_lhs ? "1" : "0");
        REQUIRE(dispatch.bindings[2].value == imported.graph.values().find_tensor(operand)->id);
        REQUIRE(dispatch.bindings[3].value == imported.graph.values().find_tensor(output)->id);

        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        ggml_free(ctx);
    }
}

static void run_lfm_dconv3_dispatch_checks() {
    const struct {
        int64_t tokens;
        int64_t hidden;
        int64_t state_rows;
        int64_t filter_rows;
        bool disabled;
        bool fused;
    } cases[] = {
        {1, 2048, 2, 3, false, true},
        {64, 2048, 2, 3, false, true},
        {2, 2048, 2, 3, false, false},
        {64, 2016, 2, 3, false, false},
        {64, 2048, 3, 3, false, false},
        {64, 2048, 2, 4, false, false},
        {64, 2048, 2, 3, true, false},
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size = 4 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test.hidden, test.tokens, 1);
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, test.state_rows, test.hidden, 1);
        ggml_tensor * filter = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.filter_rows, test.hidden);
        ggml_tensor * window = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);
        ggml_tensor * tail   = ggml_view_3d(ctx, window, test.state_rows, test.hidden, 1, window->nb[1], window->nb[2],
                                            test.tokens * sizeof(float));
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.state_rows * test.hidden, 1);
        ggml_tensor * target = ggml_view_2d(ctx, cache, test.state_rows * test.hidden, 1, cache->nb[1], 0);
        ggml_tensor * output = ggml_ssm_conv(ctx, window, filter);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, tail, target));
        ggml_build_forward_expand(graph, output);
        if (test.disabled) {
            REQUIRE(setenv("GGML_HRX_DISABLE_LFM_DCONV3_FUSION", "1", 1) == 0);
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:llm_ssm_conv_lfm_dconv3_state_f32";
        });
        REQUIRE((fused != plan.dispatches.end()) == test.fused);
        if (test.fused) {
            REQUIRE(fused->bindings.size() == 5);
            require_compile_parameter(*fused, "llm.ssm_conv.lfm_dconv3.n_t", std::to_string(test.tokens));
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        if (test.disabled) {
            REQUIRE(unsetenv("GGML_HRX_DISABLE_LFM_DCONV3_FUSION") == 0);
        }
        ggml_free(ctx);
    }
}

static void run_gdn_native_projection_pair_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t heads;
        bool different_input;
        bool mutate_input;
        bool aliased_weight;
        ggml_type weight_type;
        bool fused;
    } cases[] = {
        {1, 5120, 48, false, false, false, GGML_TYPE_Q4_K, true},
        {1, 4096, 24, false, false, false, GGML_TYPE_Q4_K, true},
        {1, 8192, 48, false, false, false, GGML_TYPE_Q4_K, true},
        {1, 32768, 48, false, false, false, GGML_TYPE_Q4_K, true},
        {2, 5120, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {5, 5120, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {16, 5120, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {17, 5120, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {1, 3072, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {1, 4352, 48, false, false, false, GGML_TYPE_Q4_K, false},
        {1, 5120, 16, false, false, false, GGML_TYPE_Q4_K, false},
        {1, 5120, 48, true, false, false, GGML_TYPE_Q4_K, false},
        {1, 5120, 48, false, true, false, GGML_TYPE_Q4_K, false},
        {1, 5120, 48, false, false, true, GGML_TYPE_Q4_K, false},
        {1, 5120, 48, false, false, false, GGML_TYPE_Q6_K, false},
    };

    for (const auto & test : cases) {
        const int64_t width = 128;
        const int64_t qheads = test.heads / 4;
        const int64_t hidden = width * (2 * qheads + test.heads);
        const int64_t state_elements = width * width * test.heads;
        const size_t state_bytes = state_elements * sizeof(float);
        ggml_tensor * input = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.inputs, test.tokens), 0.5f);
        ggml_tensor * first_weight = ggml_new_tensor_2d(ctx, test.weight_type, test.inputs, test.heads);
        ggml_tensor * second_weight = ggml_dup_tensor(ctx, first_weight);
        if (test.aliased_weight) {
            second_weight = ggml_view_2d(ctx, second_weight, test.inputs, test.heads, second_weight->nb[1], 0);
        }
        ggml_tensor * alpha = ggml_mul_mat(ctx, first_weight, input);
        ggml_tensor * second_input = test.different_input ? ggml_dup_tensor(ctx, input) : input;
        ggml_tensor * beta_raw = ggml_mul_mat(ctx, second_weight, second_input);
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.heads);
        ggml_tensor * scale = ggml_dup_tensor(ctx, bias);
        ggml_tensor * gate         = ggml_mul(
            ctx, ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_3d(ctx, alpha, test.heads, test.tokens, 1), bias)),
            scale);
        gate = ggml_reshape_4d(ctx, gate, 1, test.heads, test.tokens, 1);
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta_raw, 1, test.heads, test.tokens, 1));
        ggml_tensor * raw = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, test.tokens), 0.5f);
        const auto view = [&](int64_t heads, size_t offset) {
            return ggml_view_4d(ctx, raw, width, heads, test.tokens, 1, width * sizeof(float), hidden * sizeof(float),
                                   hidden * test.tokens * sizeof(float), offset);
        };
        ggml_tensor * q = ggml_l2_norm(ctx, view(qheads, 0), 1.e-6f);
        ggml_tensor * k = ggml_l2_norm(ctx, view(qheads, width * qheads * sizeof(float)), 1.e-6f);
        ggml_tensor * v = view(test.heads, 2 * width * qheads * sizeof(float));
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 20);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * gathered = ggml_get_rows(ctx, cache, ids);
        ggml_tensor * state = ggml_reshape_4d(ctx, gathered, width, width, test.heads, 1);
        ggml_tensor * gdn = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, 5);
        const int64_t count = std::min<int64_t>(test.tokens, 5);
        const size_t attention_bytes = width * test.heads * test.tokens * sizeof(float);
        ggml_tensor * attention       = ggml_view_4d(ctx, gdn, width, test.heads, test.tokens, 1, width * sizeof(float),
                                                     width * test.heads * sizeof(float), attention_bytes, 0);
        ggml_tensor * snapshots =
            ggml_view_3d(ctx, gdn, state_elements, 1, count, state_bytes, state_bytes, attention_bytes);
        ggml_tensor * target = ggml_view_3d(ctx, cache, state_elements, 1, count, state_bytes, 4 * state_bytes, 0);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, raw);
        ggml_build_forward_expand(graph, alpha);
        if (test.mutate_input) {
            ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_dup_tensor(ctx, input), input));
        }
        ggml_build_forward_expand(graph, beta_raw);
        ggml_build_forward_expand(graph, gathered);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, snapshots, target));
        ggml_build_forward_expand(graph, ggml_scale(ctx, attention, 0.5f));
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_dual_q4_f32_decode";
        });
        REQUIRE((fused != plan.dispatches.end()) == test.fused);
        if (test.fused) {
            REQUIRE(fused->bindings.size() == 5);
            require_compile_parameter(*fused, "ggml.mul_mat_dual_q4_f32_c1.input_size", std::to_string(test.inputs));
            REQUIRE(fused->bindings[0].value == imported.graph.values().find_tensor(input)->id);
            REQUIRE(fused->bindings[1].value == imported.graph.values().find_tensor(first_weight)->id);
            REQUIRE(fused->bindings[2].value == imported.graph.values().find_tensor(second_weight)->id);
            REQUIRE(fused->bindings[3].value == imported.graph.values().find_tensor(alpha)->id);
            REQUIRE(fused->bindings[4].value == imported.graph.values().find_tensor(beta_raw)->id);
        }
        const auto epilogue = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:llm_gated_delta_net_projection_epilogue_f32";
        });
        REQUIRE((epilogue != plan.dispatches.end()) == (test.tokens > 16));
        if (epilogue != plan.dispatches.end()) {
            REQUIRE(epilogue->kernel.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
            REQUIRE(epilogue->kernel.integer_parameters.at("element_count") == test.heads * test.tokens);
            REQUIRE(epilogue->kernel.compile_parameters.count("llm.gated_delta_net.epilogue_element_count") == 0);
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_gdn_selected_snapshot_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        int64_t tokens;
        int64_t sequences;
        bool ready;
        bool side_use;
        bool cache_read;
        bool observed;
        bool aligned;
        bool fused;
    } cases[] = {
        {2,1,true,false,false,false,true,true},
        {3,1,true,false,false,false,true,true},
        {4,1,true,false,false,false,true,true},
        {5,1,true,false,false,false,true,true},
        {1,1,true,false,false,false,true,false},
        {5,2,true,false,false,false,true,false},
        {5,1,false,false,false,false,true,false},
        {5,1,true,true,false,false,true,false},
        {5,1,true,false,true,false,true,false},
        {5,1,true,false,false,true,true,false},
        {5,1,true,false,false,false,false,false},
    };
    for (const auto & test : cases) {
        const int64_t width=128, heads=6, qheads=2, hidden=1280;
        const int64_t state_elements=width*width*heads;
        const size_t state_bytes=state_elements*sizeof(float);
        ggml_tensor * raw=ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, test.tokens*test.sequences), 0.5f);
        ggml_tensor * alpha=ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, heads, test.tokens*test.sequences), 0.5f);
        ggml_tensor * beta_raw=ggml_scale(ctx, ggml_dup_tensor(ctx, alpha), 0.5f);
        ggml_tensor * bias=ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
        ggml_tensor * scale=ggml_dup_tensor(ctx, bias);
        ggml_tensor * gate=ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_3d(ctx, alpha, heads, test.tokens, test.sequences), bias)), scale);
        gate=ggml_reshape_4d(ctx, gate, 1, heads, test.tokens, test.sequences);
        ggml_tensor * beta=ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta_raw, 1, heads, test.tokens, test.sequences));
        ggml_tensor * cache=ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 20);
        ggml_tensor * ids=ggml_new_tensor_1d(ctx, GGML_TYPE_I32, test.sequences);
        ggml_tensor * gathered=ggml_get_rows(ctx, cache, ids);
        if (test.observed) {
            ggml_set_output(gathered);
        }
        ggml_tensor * state=ggml_reshape_4d(ctx, gathered, width, width, heads, test.sequences);
        const auto view=[&](int64_t nheads, size_t offset) {
            return ggml_view_4d(ctx, raw, width, nheads, test.tokens, test.sequences,
                               width*sizeof(float), hidden*sizeof(float), hidden*test.tokens*sizeof(float), offset);
        };
        ggml_tensor * q=ggml_l2_norm(ctx, view(qheads,0), 1.e-6f);
        ggml_tensor * k=ggml_l2_norm(ctx, view(qheads,width*qheads*sizeof(float)), 1.e-6f);
        ggml_tensor * v=view(heads,2*width*qheads*sizeof(float));
        ggml_tensor * gdn=ggml_gated_delta_net(ctx,q,k,v,gate,beta,state,5);
        const size_t attention_bytes=width*heads*test.tokens*test.sequences*sizeof(float);
        ggml_tensor * attention=ggml_view_4d(ctx,gdn,width,heads,test.tokens,test.sequences,
                                            width*sizeof(float),width*heads*sizeof(float),
                                            width*heads*test.tokens*sizeof(float),0);
        ggml_tensor * snapshots=ggml_view_3d(ctx,gdn,state_elements,test.sequences,test.tokens,
                                            state_bytes,state_bytes*test.sequences,attention_bytes);
        ggml_tensor * target=ggml_view_3d(ctx,cache,state_elements,test.sequences,test.tokens,
                                         state_bytes,4*state_bytes,test.aligned ? state_bytes : sizeof(float));
        ggml_cgraph * graph=ggml_new_graph(ctx);
        if (test.ready) {
            ggml_build_forward_expand(graph,raw);
        }
        ggml_build_forward_expand(graph,alpha);
        ggml_build_forward_expand(graph,beta_raw);
        ggml_build_forward_expand(graph,gathered);
        if (test.cache_read) {
            ggml_build_forward_expand(graph,ggml_scale(ctx,cache,0.25f));
        }
        ggml_build_forward_expand(graph,ggml_cpy(ctx,snapshots,target));
        ggml_build_forward_expand(graph,ggml_scale(ctx,attention,0.5f));
        if (test.side_use) {
            ggml_build_forward_expand(graph,ggml_scale(ctx,gathered,0.5f));
        }
        auto imported=ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph,test_dispatch_target()));
        const auto & plan=scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused=std::find_if(plan.dispatches.begin(),plan.dispatches.end(),[](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id)==
                   "loom_libs:llm_gated_delta_net_f32_wmma_head128_selected_snapshot_projection_epilogue";
        });
        REQUIRE((fused!=plan.dispatches.end())==test.fused);
        if (test.fused) {
            REQUIRE(fused->bindings.size()==11);
            require_compile_parameter(*fused,"llm.gated_delta_net.state_row_count","20");
            REQUIRE(fused->bindings[7].value==imported.graph.values().find_tensor(cache)->id);
            REQUIRE(fused->bindings[10].value==imported.graph.values().find_tensor(ids)->id);
            REQUIRE(std::none_of(plan.dispatches.begin(),plan.dispatches.end(),[](const auto & dispatch) {
                return kernel_name_for_id(dispatch.kernel.kernel_id)=="loom_libs:ggml_get_rows_f32";
            }));
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_gdn_rmsnorm_gate_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const struct {
        ggml_type type;
        int64_t tokens;
        bool side_use;
        bool attention_output;
        bool gdn_output;
        bool fused;
    } cases[] = {
        { GGML_TYPE_Q4_K, 512, false, false, false, true },
        { GGML_TYPE_Q6_K, 512, false, false, false, true },
        { GGML_TYPE_Q4_K, 512, true, false, false, false },
        { GGML_TYPE_Q4_K, 256, false, false, false, false },
        { GGML_TYPE_Q4_K, 512, false, true, false, false },
        { GGML_TYPE_Q4_K, 512, false, false, true, false },
    };

    for (const auto & test : cases) {
        const int64_t width = 128;
        const int64_t heads = 2;
        const int64_t channels = width * heads;
        const int64_t state_elements = width * channels;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, 1, test.tokens, 1);
        ggml_tensor * k = ggml_dup_tensor(ctx, q);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, heads, test.tokens, 1);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, test.tokens, 1);
        ggml_tensor * beta = ggml_dup_tensor(ctx, g);
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, width, heads, 1);
        ggml_tensor * gdn = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
        ggml_tensor * attention      = ggml_view_4d(ctx, gdn, width, heads, test.tokens, 1, width * sizeof(float),
                                                    channels * sizeof(float), channels * test.tokens * sizeof(float), 0);
        if (test.attention_output) {
            ggml_set_output(attention);
        }
        if (test.gdn_output) {
            ggml_set_output(gdn);
        }
        ggml_tensor * new_state  = ggml_view_2d(ctx, gdn, state_elements, 1, state_elements * sizeof(float),
                                              channels * test.tokens * sizeof(float));
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 1);
        ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_tensor * gate = ggml_dup_tensor(ctx, attention);
        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, attention, 2.0e-6f), weight);
        ggml_tensor * gated = ggml_mul(ctx, norm, ggml_silu(ctx, gate));
        ggml_tensor * input = ggml_reshape_2d(ctx, gated, channels, test.tokens);
        ggml_tensor * projection = ggml_new_tensor_2d(ctx, test.type, channels, 4096);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, new_state, cache));
        ggml_build_forward_expand(graph, ggml_mul_mat(ctx, projection, input));
        if (test.side_use) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, attention, 0.5f));
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                   "loom_libs:llm_gated_delta_net_f32_wmma_head128_rmsnorm_gate";
        });
        REQUIRE((fused != plan.dispatches.end()) == test.fused);
        REQUIRE(std::count_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_rmsnorm_gate_f32_publish";
        }) == (test.fused ? 0 : 1));
        if (test.fused) {
            REQUIRE(fused->bindings.size() == 11);
            const auto * projected = imported.graph.values().find_tensor(input);
            REQUIRE(projected != nullptr);
            const auto * packed =
                plan.metadata.find_generated_resource(projected->id, ggml::hrx::GeneratedResourceRole::F16K16Major);
            REQUIRE(packed != nullptr && fused->bindings.back().value == packed->generated_value);
            require_compile_parameter(*fused, "ggml.rmsnorm_gate_f32.rms_epsilon", "1.99999999e-06");
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }
    ggml_free(ctx);
}

static void run_qwen_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 12 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 2, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 3);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 3, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 5);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 5, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 6);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 6, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 640, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 2, 640, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_q6_k_packed_token1_f16_wmma", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       256);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048, 256);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * output = ggml_add(ctx, projection, bias);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_F16, 33, 2048,
            256, 2, { GGML_OP_MUL_MAT, GGML_OP_ADD },
            { "input", "weight", "bias", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2048, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * output = ggml_add(ctx, projection, bias);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_BF16, 33, 2048,
            256, 2, { GGML_OP_MUL_MAT, GGML_OP_ADD },
            { "input", "weight", "bias", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(residual != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * output = ggml_add(ctx, projection, residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_F16, 33, 2048,
            256, 2, { GGML_OP_MUL_MAT, GGML_OP_ADD },
            { "input", "weight", "bias", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        REQUIRE(residual != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * biased = ggml_add(ctx, projection, bias);
        REQUIRE(biased != nullptr);
        ggml_tensor * output = ggml_add(ctx, biased, residual);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_F16, 33, 2048,
            256, 3, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD },
            { "input", "weight", "bias", "residual_input", "residual_output" }, false);
    }

    {
        ggml_tensor * weight      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * residual    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(residual != nullptr);
        REQUIRE(norm_weight != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * added = ggml_add(ctx, projection, residual);
        REQUIRE(added != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, added, 0.000001f);
        REQUIRE(rms != nullptr);
        ggml_tensor * output = ggml_mul(ctx, rms, norm_weight);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_F16, 33, 2048,
            256, 4, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL },
            { "input", "weight", "bias", "residual_input", "residual_output" }, true);
    }

    {
        ggml_tensor * weight      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 33);
        ggml_tensor * bias        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * residual    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 33);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(bias != nullptr);
        REQUIRE(residual != nullptr);
        REQUIRE(norm_weight != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        REQUIRE(projection != nullptr);
        ggml_tensor * biased = ggml_add(ctx, projection, bias);
        REQUIRE(biased != nullptr);
        ggml_tensor * added = ggml_add(ctx, biased, residual);
        REQUIRE(added != nullptr);
        ggml_tensor * rms = ggml_rms_norm(ctx, added, 0.000001f);
        REQUIRE(rms != nullptr);
        ggml_tensor * output = ggml_mul(ctx, rms, norm_weight);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_postops_command(
            ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", GGML_TYPE_F16, 33, 2048,
            256, 5, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL },
            { "input", "weight", "bias", "residual_input", "residual_output" }, true);
    }

    {
        struct FormatCase {
            ggml_type type;
            int64_t   output_size;
        };

        const FormatCase cases[] = {
            { GGML_TYPE_Q3_K,   128 },
            { GGML_TYPE_Q4_K,   128 },
            { GGML_TYPE_Q6_K,   128 },
            { GGML_TYPE_IQ3_S,  128 },
            { GGML_TYPE_IQ4_NL, 128 },
            { GGML_TYPE_IQ4_XS, 128 },
            { GGML_TYPE_Q8_0,   128 },
            { GGML_TYPE_Q8_1,   128 },
            { GGML_TYPE_F16,    128 },
            { GGML_TYPE_BF16,   128 },
            { GGML_TYPE_F32,    256 },
        };

        for (const FormatCase & c : cases) {
            ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, c.type, 2048, c.output_size);
            ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, c.type, 2048, c.output_size);
            ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
            REQUIRE(gate_weight != nullptr);
            REQUIRE(up_weight != nullptr);
            REQUIRE(input != nullptr);
            ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
            ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
            REQUIRE(gate != nullptr);
            REQUIRE(up != nullptr);
            ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            REQUIRE(output != nullptr);
            schedule_fused_matmul_swiglu_command(ctx, output, c.type, c.type, 4, 2048, c.output_size);
        }
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 2048);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 2048);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_GEGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, 2, 640, 2048,
                                             ggml::hrx::BinaryKind::GeGLU);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 2048);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 2048);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 1);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        require_matmul_swiglu_falls_back_to_compilable_plan(ctx, output);
    }

    {
        ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        REQUIRE(packed != nullptr);
        ggml_tensor * output = ggml_glu(ctx, packed, GGML_GLU_OP_SWIGLU, false);
        REQUIRE(output != nullptr);
        schedule_packed_glu_command(ctx, output, ggml::hrx::BinaryKind::SwiGLU, 128, 3, false);
    }

    {
        ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        REQUIRE(packed != nullptr);
        ggml_tensor * output = ggml_glu(ctx, packed, GGML_GLU_OP_GEGLU, true);
        REQUIRE(output != nullptr);
        schedule_packed_glu_command(ctx, output, ggml::hrx::BinaryKind::GeGLU, 128, 3, true);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * packed = ggml_mul_mat(ctx, weight, input);
        REQUIRE(packed != nullptr);
        ggml_tensor * output = ggml_glu(ctx, packed, GGML_GLU_OP_SWIGLU, false);
        REQUIRE(output != nullptr);
        schedule_packed_matmul_glu_command(ctx, output, GGML_TYPE_Q4_K, 2, 256, 128, ggml::hrx::BinaryKind::SwiGLU,
                                           false);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 256);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * packed = ggml_mul_mat(ctx, weight, input);
        REQUIRE(packed != nullptr);
        ggml_tensor * output = ggml_glu(ctx, packed, GGML_GLU_OP_SWIGLU, true);
        REQUIRE(output != nullptr);
        schedule_packed_matmul_glu_command(ctx, output, GGML_TYPE_Q4_K, 2, 256, 128, ggml::hrx::BinaryKind::SwiGLU,
                                           true);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 256);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 640, 256);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, 2, 640, 256);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_Q4_K, GGML_TYPE_F16, 4, 2048, 128);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_F32, GGML_TYPE_Q8_1, 4, 2048, 128);
    }

    {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * gate_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        ggml_tensor * up_input    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(gate_input != nullptr);
        REQUIRE(up_input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, gate_input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, up_input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
        REQUIRE(output != nullptr);
        require_matmul_swiglu_falls_back(ctx, output);
    }

    for (const auto & variant : {
             std::pair<ggml_glu_op, ggml::hrx::BinaryKind>{ GGML_GLU_OP_GEGLU,       ggml::hrx::BinaryKind::GeGLU    },
             std::pair<ggml_glu_op, ggml::hrx::BinaryKind>{ GGML_GLU_OP_REGLU,       ggml::hrx::BinaryKind::RegLU    },
             std::pair<ggml_glu_op, ggml::hrx::BinaryKind>{ GGML_GLU_OP_GEGLU_ERF,   ggml::hrx::BinaryKind::GeGLUErf },
             std::pair<ggml_glu_op, ggml::hrx::BinaryKind>{ GGML_GLU_OP_GEGLU_QUICK,
                                                           ggml::hrx::BinaryKind::GeGLUQuick                         },
    }) {
        ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(gate_weight != nullptr);
        REQUIRE(up_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * gate = ggml_mul_mat(ctx, gate_weight, input);
        ggml_tensor * up   = ggml_mul_mat(ctx, up_weight, input);
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        ggml_tensor * output = ggml_glu_split(ctx, gate, up, variant.first);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 4, 2048, 128, variant.second,
                                             GGML_OP_GLU);
    }

    {
        ggml_tensor * lhs_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * rhs_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(lhs_weight != nullptr);
        REQUIRE(rhs_weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * lhs = ggml_mul_mat(ctx, lhs_weight, input);
        ggml_tensor * rhs = ggml_mul_mat(ctx, rhs_weight, input);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        ggml_tensor * output = ggml_sub(ctx, lhs, rhs);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_swiglu_command(ctx, output, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 4, 2048, 128,
                                             ggml::hrx::BinaryKind::Sub, GGML_OP_SUB);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_silu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                                            ggml::hrx::UnaryKind::Silu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_relu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                                            ggml::hrx::UnaryKind::Relu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_gelu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32",
                                            ggml::hrx::UnaryKind::Gelu, 4, 2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 640, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_gelu(ctx, matmul);
        REQUIRE(output != nullptr);
        schedule_fused_matmul_unary_command(ctx, output, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32",
                                            ggml::hrx::UnaryKind::Gelu, 2, 640, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * relu = ggml_relu(ctx, matmul);
        ggml_tensor * sqr  = ggml_sqr(ctx, matmul);
        REQUIRE(relu != nullptr);
        REQUIRE(sqr != nullptr);
        ggml_tensor * output = ggml_add(ctx, relu, sqr);
        REQUIRE(output != nullptr);
        require_matmul_root_matches_identity_single(ctx, output);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * matmul = ggml_mul_mat(ctx, weight, input);
        REQUIRE(matmul != nullptr);
        ggml_tensor * output = ggml_softplus(ctx, matmul);
        REQUIRE(output != nullptr);
        require_matmul_root_matches_identity_single(ctx, output);
    }

    {
        static constexpr const char * kDisableQwenDispatchEnv = "GGML_HRX_DISABLE_QWEN_DISPATCH";
        const char *                  original_env            = std::getenv(kDisableQwenDispatchEnv);
        const bool                    had_original_env        = original_env != nullptr;
        const std::string             original_env_value      = had_original_env ? original_env : "";
        REQUIRE(setenv(kDisableQwenDispatchEnv, "1", 1) == 0);

        ggml_tensor * q4_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * q4_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q4_weight != nullptr);
        REQUIRE(q4_input != nullptr);
        ggml_tensor * q4_output = ggml_mul_mat(ctx, q4_weight, q4_input);
        REQUIRE(q4_output != nullptr);
        schedule_single_matmul_command(ctx, q4_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * q4_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * q4_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q4_decode_weight != nullptr);
        REQUIRE(q4_decode_input != nullptr);
        ggml_tensor * q4_decode_output = ggml_mul_mat(ctx, q4_decode_weight, q4_decode_input);
        REQUIRE(q4_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q4_decode_output, "loom_libs:ggml_mul_mat_f32_f32_wmma", 1, 2048, 128);

        for (ggml_type legacy_type :
             { GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1 }) {
            ggml_tensor * legacy_weight = ggml_new_tensor_2d(ctx, legacy_type, 2048, 128);
            ggml_tensor * legacy_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
            REQUIRE(legacy_weight != nullptr);
            REQUIRE(legacy_input != nullptr);
            ggml_tensor * legacy_output = ggml_mul_mat(ctx, legacy_weight, legacy_input);
            REQUIRE(legacy_output != nullptr);
            schedule_single_matmul_command(ctx, legacy_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4,
                                           2048, 128);

            ggml_tensor * legacy_decode_weight = ggml_new_tensor_2d(ctx, legacy_type, 2048, 128);
            ggml_tensor * legacy_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
            REQUIRE(legacy_decode_weight != nullptr);
            REQUIRE(legacy_decode_input != nullptr);
            ggml_tensor * legacy_decode_output = ggml_mul_mat(ctx, legacy_decode_weight, legacy_decode_input);
            REQUIRE(legacy_decode_output != nullptr);
            schedule_single_matmul_command(ctx, legacy_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                           2048, 128);
        }

        ggml_tensor * q3_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, 2048, 128);
        ggml_tensor * q3_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q3_weight != nullptr);
        REQUIRE(q3_input != nullptr);
        ggml_tensor * q3_output = ggml_mul_mat(ctx, q3_weight, q3_input);
        REQUIRE(q3_output != nullptr);
        schedule_single_matmul_command(ctx, q3_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * q3_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, 2048, 128);
        ggml_tensor * q3_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q3_decode_weight != nullptr);
        REQUIRE(q3_decode_input != nullptr);
        ggml_tensor * q3_decode_output = ggml_mul_mat(ctx, q3_decode_weight, q3_decode_input);
        REQUIRE(q3_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q3_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * iq2_s_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ2_S, 2048, 640);
        ggml_tensor * iq2_s_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(iq2_s_weight != nullptr);
        REQUIRE(iq2_s_input != nullptr);
        ggml_tensor * iq2_s_output = ggml_mul_mat(ctx, iq2_s_weight, iq2_s_input);
        REQUIRE(iq2_s_output != nullptr);
        schedule_single_matmul_command(ctx, iq2_s_output,
                                       "loom_libs:ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32", 2, 2048, 640);

        ggml_tensor * iq2_s_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ2_S, 2048, 640);
        ggml_tensor * iq2_s_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(iq2_s_decode_weight != nullptr);
        REQUIRE(iq2_s_decode_input != nullptr);
        ggml_tensor * iq2_s_decode_output = ggml_mul_mat(ctx, iq2_s_decode_weight, iq2_s_decode_input);
        REQUIRE(iq2_s_decode_output != nullptr);
        schedule_single_matmul_command(ctx, iq2_s_decode_output, "loom_libs:ggml_mul_mat_vector_iq2_s_f32_f32", 1,
                                       2048, 640);

        ggml_tensor * iq3_s_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_S, 2048, 128);
        ggml_tensor * iq3_s_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(iq3_s_weight != nullptr);
        REQUIRE(iq3_s_input != nullptr);
        ggml_tensor * iq3_s_output = ggml_mul_mat(ctx, iq3_s_weight, iq3_s_input);
        REQUIRE(iq3_s_output != nullptr);
        schedule_single_matmul_command(ctx, iq3_s_output,
                                       "loom_libs:ggml_mul_mat_tiled_input_f32_iq3_s_publish_f32", 4, 2048, 128);

        ggml_tensor * iq3_s_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_S, 2048, 128);
        ggml_tensor * iq3_s_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(iq3_s_decode_weight != nullptr);
        REQUIRE(iq3_s_decode_input != nullptr);
        ggml_tensor * iq3_s_decode_output = ggml_mul_mat(ctx, iq3_s_decode_weight, iq3_s_decode_input);
        REQUIRE(iq3_s_decode_output != nullptr);
        schedule_single_matmul_command(ctx, iq3_s_decode_output, "loom_libs:ggml_mul_mat_vector_iq3_s_f32_f32", 1,
                                       2048, 128);

        ggml_tensor * iq4_nl_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 2048, 128);
        ggml_tensor * iq4_nl_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(iq4_nl_weight != nullptr);
        REQUIRE(iq4_nl_input != nullptr);
        ggml_tensor * iq4_nl_output = ggml_mul_mat(ctx, iq4_nl_weight, iq4_nl_input);
        REQUIRE(iq4_nl_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_nl_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4,
                                       2048, 128);

        ggml_tensor * iq4_nl_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 2048, 128);
        ggml_tensor * iq4_nl_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(iq4_nl_decode_weight != nullptr);
        REQUIRE(iq4_nl_decode_input != nullptr);
        ggml_tensor * iq4_nl_decode_output = ggml_mul_mat(ctx, iq4_nl_decode_weight, iq4_nl_decode_input);
        REQUIRE(iq4_nl_decode_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_nl_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                       2048, 128);

        ggml_tensor * iq4_nl_small_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 1024);
        ggml_tensor * iq4_nl_small_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(iq4_nl_small_weight != nullptr);
        REQUIRE(iq4_nl_small_input != nullptr);
        ggml_tensor * iq4_nl_small_output = ggml_mul_mat(ctx, iq4_nl_small_weight, iq4_nl_small_input);
        REQUIRE(iq4_nl_small_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_nl_small_output, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32",
                                       2, 640, 1024);

        ggml_tensor * iq4_nl_small_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 640, 1024);
        ggml_tensor * iq4_nl_small_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 1);
        REQUIRE(iq4_nl_small_decode_weight != nullptr);
        REQUIRE(iq4_nl_small_decode_input != nullptr);
        ggml_tensor * iq4_nl_small_decode_output =
            ggml_mul_mat(ctx, iq4_nl_small_decode_weight, iq4_nl_small_decode_input);
        REQUIRE(iq4_nl_small_decode_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_nl_small_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64",
                                       1, 640, 1024);

        ggml_tensor * q5_0_small_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 640, 256);
        ggml_tensor * q5_0_small_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 2);
        REQUIRE(q5_0_small_weight != nullptr);
        REQUIRE(q5_0_small_input != nullptr);
        ggml_tensor * q5_0_small_output = ggml_mul_mat(ctx, q5_0_small_weight, q5_0_small_input);
        REQUIRE(q5_0_small_output != nullptr);
        schedule_single_matmul_command(ctx, q5_0_small_output, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32", 2,
                                       640, 256);

        ggml_tensor * q5_0_small_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 640, 256);
        ggml_tensor * q5_0_small_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 640, 1);
        REQUIRE(q5_0_small_decode_weight != nullptr);
        REQUIRE(q5_0_small_decode_input != nullptr);
        ggml_tensor * q5_0_small_decode_output = ggml_mul_mat(ctx, q5_0_small_decode_weight, q5_0_small_decode_input);
        REQUIRE(q5_0_small_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q5_0_small_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                       640, 256);

        ggml_tensor * q6_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * q6_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 2);
        REQUIRE(q6_weight != nullptr);
        REQUIRE(q6_input != nullptr);
        ggml_tensor * q6_output = ggml_mul_mat(ctx, q6_weight, q6_input);
        REQUIRE(q6_output != nullptr);
        schedule_single_matmul_command(ctx, q6_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 2, 2048,
                                       128);

        ggml_tensor * q6_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 128);
        ggml_tensor * q6_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q6_decode_weight != nullptr);
        REQUIRE(q6_decode_input != nullptr);
        ggml_tensor * q6_decode_output = ggml_mul_mat(ctx, q6_decode_weight, q6_decode_input);
        REQUIRE(q6_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q6_decode_output, "loom_libs:ggml_mul_mat_q6_k_packed_token1_f16_wmma", 1,
                                       2048, 128);

        ggml_tensor * q6_gemma_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 3840, 262208);
        ggml_tensor * q6_gemma_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 1);
        REQUIRE(q6_gemma_decode_weight != nullptr);
        REQUIRE(q6_gemma_decode_input != nullptr);
        ggml_tensor * q6_gemma_decode_output = ggml_mul_mat(ctx, q6_gemma_decode_weight, q6_gemma_decode_input);
        REQUIRE(q6_gemma_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q6_gemma_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                       3840, 262208);

        ggml_tensor * iq4_xs_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, 2048, 128);
        ggml_tensor * iq4_xs_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(iq4_xs_weight != nullptr);
        REQUIRE(iq4_xs_input != nullptr);
        ggml_tensor * iq4_xs_output = ggml_mul_mat(ctx, iq4_xs_weight, iq4_xs_input);
        REQUIRE(iq4_xs_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_xs_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4,
                                       2048, 128);

        ggml_tensor * iq4_xs_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, 2048, 128);
        ggml_tensor * iq4_xs_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(iq4_xs_decode_weight != nullptr);
        REQUIRE(iq4_xs_decode_input != nullptr);
        ggml_tensor * iq4_xs_decode_output = ggml_mul_mat(ctx, iq4_xs_decode_weight, iq4_xs_decode_input);
        REQUIRE(iq4_xs_decode_output != nullptr);
        schedule_single_matmul_command(ctx, iq4_xs_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                       2048, 128);

        ggml_tensor * q8_0_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * q8_0_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q8_0_weight != nullptr);
        REQUIRE(q8_0_input != nullptr);
        ggml_tensor * q8_0_output = ggml_mul_mat(ctx, q8_0_weight, q8_0_input);
        REQUIRE(q8_0_output != nullptr);
        schedule_single_matmul_command(ctx, q8_0_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * q8_0_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2048, 128);
        ggml_tensor * q8_0_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q8_0_decode_weight != nullptr);
        REQUIRE(q8_0_decode_input != nullptr);
        ggml_tensor * q8_0_decode_output = ggml_mul_mat(ctx, q8_0_decode_weight, q8_0_decode_input);
        REQUIRE(q8_0_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q8_0_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * q8_0_gemma_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 3840, 262208);
        ggml_tensor * q8_0_gemma_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 1);
        REQUIRE(q8_0_gemma_decode_weight != nullptr);
        REQUIRE(q8_0_gemma_decode_input != nullptr);
        ggml_tensor * q8_0_gemma_decode_output = ggml_mul_mat(ctx, q8_0_gemma_decode_weight, q8_0_gemma_decode_input);
        REQUIRE(q8_0_gemma_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q8_0_gemma_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1,
                                       3840, 262208);

        ggml_tensor * q8_0_unsafe_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32768, 262144);
        ggml_tensor * q8_0_unsafe_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32768, 1);
        REQUIRE(q8_0_unsafe_decode_weight != nullptr);
        REQUIRE(q8_0_unsafe_decode_input != nullptr);
        ggml_tensor * q8_0_unsafe_decode_output =
            ggml_mul_mat(ctx, q8_0_unsafe_decode_weight, q8_0_unsafe_decode_input);
        REQUIRE(q8_0_unsafe_decode_output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, q8_0_unsafe_decode_output));

        ggml_tensor * q8_1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * q8_1_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(q8_1_weight != nullptr);
        REQUIRE(q8_1_input != nullptr);
        ggml_tensor * q8_1_output = ggml_mul_mat(ctx, q8_1_weight, q8_1_input);
        REQUIRE(q8_1_output != nullptr);
        schedule_single_matmul_command(ctx, q8_1_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * q8_1_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 2048, 128);
        ggml_tensor * q8_1_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(q8_1_decode_weight != nullptr);
        REQUIRE(q8_1_decode_input != nullptr);
        ggml_tensor * q8_1_decode_output = ggml_mul_mat(ctx, q8_1_decode_weight, q8_1_decode_input);
        REQUIRE(q8_1_decode_output != nullptr);
        schedule_single_matmul_command(ctx, q8_1_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * f16_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * f16_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(f16_weight != nullptr);
        REQUIRE(f16_input != nullptr);
        ggml_tensor * f16_output = ggml_mul_mat(ctx, f16_weight, f16_input);
        REQUIRE(f16_output != nullptr);
        schedule_single_matmul_command(ctx, f16_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * f16_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * f16_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(f16_decode_weight != nullptr);
        REQUIRE(f16_decode_input != nullptr);
        ggml_tensor * f16_decode_output = ggml_mul_mat(ctx, f16_decode_weight, f16_decode_input);
        REQUIRE(f16_decode_output != nullptr);
        schedule_single_matmul_command(ctx, f16_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        ggml_tensor * f32_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * f32_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(f32_weight != nullptr);
        REQUIRE(f32_input != nullptr);
        ggml_tensor * f32_output = ggml_mul_mat(ctx, f32_weight, f32_input);
        REQUIRE(f32_output != nullptr);
        schedule_single_matmul_command(ctx, f32_output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);

        ggml_tensor * f32_decode_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * f32_decode_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(f32_decode_weight != nullptr);
        REQUIRE(f32_decode_input != nullptr);
        ggml_tensor * f32_decode_output = ggml_mul_mat(ctx, f32_decode_weight, f32_decode_input);
        REQUIRE(f32_decode_output != nullptr);
        schedule_single_matmul_command(ctx, f32_decode_output, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64", 1, 2048,
                                       128);

        restore_environment_value(kDisableQwenDispatchEnv, had_original_env, original_env_value);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2048, 151936);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_q6_k_packed_token1_f16_wmma", 1, 2048,
                                       151936);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32", 4,
                                       2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "qwen3_moe:qwen3_moe_router_projection_f32_four_row_wave32", 1,
                                       2048, 128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, "loom_libs:ggml_mul_mat_skinny_input_f32_publish_f32", 4, 2048,
                                       128);
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2048, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 128, 2);
        ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 4, 2);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        REQUIRE(!matmul_graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void run_iq3_xxs_codebook_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    auto check_plan = [&](int64_t token_count, const char * expected_kernel, size_t expected_binding_count,
                          const char * weight_format_parameter) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_XXS, 2048, 128);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, token_count);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandPlan & plan = scheduler.plan();
        REQUIRE(plan.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(plan.dispatches[0].kernel.kernel_id) == expected_kernel);
        REQUIRE(plan.dispatches[0].bindings.size() == expected_binding_count);
        REQUIRE(plan.dispatches[0].bindings[2].length == 1024);
        REQUIRE(plan.transients.size() == 1);
        REQUIRE(plan.constant_initializations.size() == 1);
        REQUIRE(plan.constant_initializations[0].value == plan.dispatches[0].bindings[2].value);
        REQUIRE(plan.constant_initializations[0].data.size() == 1024);
        require_compile_parameter(plan.dispatches[0], weight_format_parameter, "18");
    };

    check_plan(1, "loom_libs:ggml_mul_mat_vector_iq3_xxs_f32_f32", 5, "ggml.matmul.vector.weight_format");
    check_plan(64, "loom_libs:ggml_mul_mat_tiled_input_f32_iq3_xxs_publish_f32", 4,
               "ggml.mul_mat.weight_format");

    const char * disable_name = "GGML_HRX_DISABLE_IQ3_XXS_CODEBOOK_MATMUL";
    REQUIRE(setenv(disable_name, "1", 1) == 0);
    ggml_tensor * disabled_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_XXS, 2048, 128);
    ggml_tensor * disabled_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    REQUIRE(disabled_weight != nullptr);
    REQUIRE(disabled_input != nullptr);
    ggml_tensor * disabled_output = ggml_mul_mat(ctx, disabled_weight, disabled_input);
    REQUIRE(disabled_output != nullptr);
    ggml_cgraph * disabled_graph = ggml_new_graph(ctx);
    REQUIRE(disabled_graph != nullptr);
    ggml_build_forward_expand(disabled_graph, disabled_output);
    ggml::hrx::GraphImportResult disabled_imported = ggml::hrx::import_ggml_graph(*disabled_graph);
    REQUIRE(disabled_imported.valid());
    ggml::hrx::DispatchScheduler disabled_scheduler;
    REQUIRE(!disabled_scheduler.schedule_graph(disabled_imported.graph, test_dispatch_target()));
    REQUIRE(unsetenv(disable_name) == 0);

    ggml_free(ctx);
}

static void run_iq2_codebook_resource_reuse_checks() {
    for (const ggml_type type : { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS }) {
        ggml_init_params params = {};
        params.mem_size         = 8 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * vector_weight = ggml_new_tensor_2d(ctx, type, 2048, 128);
        ggml_tensor * vector_input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
        ggml_tensor * tiled_weight  = ggml_new_tensor_2d(ctx, type, 2048, 128);
        ggml_tensor * tiled_input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 64);
        REQUIRE(vector_weight != nullptr);
        REQUIRE(vector_input != nullptr);
        REQUIRE(tiled_weight != nullptr);
        REQUIRE(tiled_input != nullptr);
        ggml_tensor * vector_output = ggml_mul_mat(ctx, vector_weight, vector_input);
        ggml_tensor * tiled_output  = ggml_mul_mat(ctx, tiled_weight, tiled_input);
        REQUIRE(vector_output != nullptr);
        REQUIRE(tiled_output != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, vector_output);
        ggml_build_forward_expand(graph, tiled_output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.size() == 2);
        REQUIRE(scheduler.plan().constant_initializations.size() == 1);
        REQUIRE(scheduler.plan().transients.size() == 1);

        const char * expected_key =
            type == GGML_TYPE_IQ2_XXS ? "common.iq2_xxs.grid.v1" : "common.iq2_xs.grid.v1";
        const ggml::hrx::CommandPlanConstantInitialization & initialization =
            scheduler.plan().constant_initializations.front();
        REQUIRE(initialization.name == expected_key);
        REQUIRE(scheduler.plan().transients.front().name == expected_key);
        REQUIRE(scheduler.plan().transients.front().value == initialization.value);
        REQUIRE(std::count_if(scheduler.plan().dispatches.begin(), scheduler.plan().dispatches.end(),
                              [&](const ggml::hrx::Dispatch & dispatch) {
                                  return std::any_of(dispatch.bindings.begin(), dispatch.bindings.end(),
                                                     [&](const ggml::hrx::DispatchBinding & binding) {
                                                         return binding.value == initialization.value;
                                                     });
                              }) == 2);

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.constant_initializations.size() == 1);
        ggml_free(ctx);
    }
}

static void run_iq4_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const auto check = [ctx](ggml_type weight_type, int64_t input_size, int64_t output_size,
                             int64_t token_count, const char * expected_kernel) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
        ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
        REQUIRE(output != nullptr);
        schedule_single_matmul_command(ctx, output, expected_kernel, token_count, input_size, output_size);
    };

    check(GGML_TYPE_IQ4_NL, 640, 1024, 1, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64");
    check(GGML_TYPE_IQ4_NL, 640, 1024, 64, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned");
    check(GGML_TYPE_IQ4_XS, 2048, 128, 1, "loom_libs:ggml_mul_mat_f32_f32_decode_wave64");
    check(GGML_TYPE_IQ4_XS, 2048, 128, 64, "loom_libs:ggml_mul_mat_tiled_input_f32_publish_f32_aligned");

    ggml_free(ctx);
}

static void run_quantized_value_projection_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    for (ggml_type weight_type : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        for (ggml_type cache_type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, 768, 256);
            ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 1);
            ggml_tensor * cache = ggml_new_tensor_2d(ctx, cache_type, 256, 17);
            ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ggml_tensor * output = ggml_set_rows(ctx, cache, ggml_mul_mat(ctx, weight, input), indices);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            auto imported = ggml::hrx::import_ggml_graph(*graph);
            REQUIRE(imported.valid());
            ggml::hrx::DispatchScheduler scheduler;
            REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
            const auto & plan = scheduler.plan();
            REQUIRE(plan.valid());
            REQUIRE(plan.dispatches.size() == 1 || plan.dispatches.size() == 2);
            const auto & dispatch = plan.dispatches.back();
            const std::string kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
            if (plan.dispatches.size() == 1) {
                REQUIRE(kernel_name == "loom_libs:llm_attention_v_matmul_set_rows_vector_f32_f32");
                require_compile_parameter(dispatch, "llm.attention_qkv.weight_format",
                                          std::to_string(matmul_weight_format_config(weight_type)));
            } else {
                REQUIRE(kernel_name_for_id(plan.dispatches.front().kernel.kernel_id) ==
                        "qwen3_moe:ggml_quantize_q8_1_x4_f32");
                REQUIRE(kernel_name == "loom_libs:llm_attention_v_matmul_set_rows_legacy_f32_f32");
                require_compile_parameter(dispatch, "llm.attention_qkv.weight_format",
                                          weight_type == GGML_TYPE_Q4_K ? "44" : "46");
                require_compile_parameter(dispatch, "ggml.mul_mat.activation_format", std::to_string(GGML_TYPE_Q8_1));
                REQUIRE(!dispatch.bindings[1].layout.empty());
            }
            require_compile_parameter(dispatch, "llm.attention_qkv.cache_output_format",
                                      cache_type == GGML_TYPE_F16 ? "16" : "32");
            const auto commands =
                ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
            REQUIRE(commands.valid());
            REQUIRE(command_program_verifies(commands));
        }
    }
    ggml_free(ctx);
}

static void run_prefill_value_projection_cache_dispatch_checks() {
    static constexpr const char * kDisableFusionEnv = "GGML_HRX_DISABLE_PREFILL_V_CACHE_FUSION";
    EnvironmentVariableGuard     disable_fusion(kDisableFusionEnv);
    disable_fusion.unset();

    auto check_case = [&](ggml_type weight_type, int64_t input_size, int64_t output_size, int64_t token_count,
                          ggml_type cache_type, int64_t cache_row_count, bool publish_f16_alternate,
                          bool set_rows_available, bool disable, bool expect_fusion, bool observe_projection = false,
                          bool alias_weight = false) {
        ggml_init_params params = {};
        params.mem_size         = 4 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        ggml_tensor * weight_storage = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
        REQUIRE(weight_storage != nullptr);
        ggml_tensor * weight =
            alias_weight ? ggml_view_2d(ctx, weight_storage, input_size, output_size, weight_storage->nb[1], 0) :
            weight_storage;
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
        ggml_tensor * cache   = ggml_new_tensor_2d(ctx, cache_type, output_size, cache_row_count);
        ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
        REQUIRE(weight != nullptr);
        REQUIRE(input != nullptr);
        REQUIRE(cache != nullptr);
        REQUIRE(indices != nullptr);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * value      = ggml_reshape_3d(ctx, projection, 128, output_size / 128, token_count);
        ggml_tensor * rows       = ggml_reshape_2d(ctx, value, output_size, token_count);
        ggml_tensor * output     = ggml_set_rows(ctx, cache, rows, indices);
        REQUIRE(projection != nullptr);
        REQUIRE(value != nullptr);
        REQUIRE(rows != nullptr);
        REQUIRE(output != nullptr);
        if (observe_projection) {
            ggml_set_output(projection);
        }

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, output);
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        ggml::hrx::CommandPlan plan;
        const ggml::hrx::Value * input_value = imported.graph.values().find_tensor(input);
        REQUIRE(input_value != nullptr);
        if (publish_f16_alternate) {
            const size_t bytes = static_cast<size_t>(input_size * token_count) * sizeof(ggml_fp16_t);
            const ggml::hrx::ValueId alternate(static_cast<int32_t>(imported.graph.values().size()));
            plan.transients.push_back({ alternate, "test.prefill_v_cache.f16", bytes, 256 });
            plan.constant_initializations.push_back(
                { alternate, "test.prefill_v_cache.f16", 0, std::vector<uint8_t>(bytes) });
            ggml::hrx::Status status;
            REQUIRE(plan.metadata.append_alternate_value(
                { input_value->id, alternate, GGML_TYPE_F16, bytes, "test.prefill_v_cache.f16" }, status));
        }

        std::vector<bool> covered_nodes(imported.graph.nodes().size(), false);
        if (!set_rows_available) {
            covered_nodes[producer_index_for_tensor(imported.graph, output)] = true;
        }
        if (disable) {
            disable_fusion.set("1");
        } else {
            disable_fusion.unset();
        }

        ggml::hrx::DispatchMatch match;
        const size_t projection_index = producer_index_for_tensor(imported.graph, projection);
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, projection_index, match));
        REQUIRE(!match.dispatches.empty());
        const ggml::hrx::Dispatch & dispatch = match.dispatches.back();
        const bool                  fused =
            kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:llm_attention_v_matmul_set_rows_tiled_f32_f32";
        REQUIRE(fused == expect_fusion);
        if (expect_fusion) {
            REQUIRE(match.dispatches.size() == 1);
            REQUIRE(match.covered_nodes.size() == 4);
            REQUIRE(dispatch.bindings.size() == 4);
            REQUIRE(dispatch.bindings[0].value == plan.transients.front().value);
            REQUIRE(dispatch.bindings[0].length == plan.transients.front().size);
            require_compile_parameter(dispatch, "ggml.mul_mat.activation_format", std::to_string(GGML_TYPE_F16));
            require_compile_parameter(dispatch, "llm.attention_qkv.weight_format", "6");
            require_compile_parameter(dispatch, "llm.attention_qkv.input_size", "1536");
            require_compile_parameter(dispatch, "llm.attention_qkv.output_size", "256");
            require_compile_parameter(dispatch, "llm.attention_qkv.cache_row_count", "512");
            require_compile_parameter(dispatch, "llm.attention_qkv.cache_output_format", "16");

            append_match_to_plan(plan, match, covered_nodes, &imported.graph);
            const ggml::hrx::CommandProgram commands =
                ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
            REQUIRE(commands.valid());
            REQUIRE(command_program_verifies(commands));
            REQUIRE(commands.commands.size() == 1);
            REQUIRE(commands.commands.front().bindings.size() == 4);
            REQUIRE(commands.commands.front().bindings[0].name == "input");
            REQUIRE(commands.commands.front().bindings[1].name == "weight");
            REQUIRE(commands.commands.front().bindings[2].name == "indices");
            REQUIRE(commands.commands.front().bindings[3].name == "cache");
        }
        ggml_free(ctx);
    };

    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, true, false, true);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, true, true, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, false, true, false, false);
    check_case(GGML_TYPE_Q4_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 2048, 256, 64, GGML_TYPE_F16, 512, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 512, 64, GGML_TYPE_F16, 512, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 32, GGML_TYPE_F16, 512, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F32, 512, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 1024, true, true, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, false, false, false);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, true, false, false, true);
    check_case(GGML_TYPE_Q6_K, 1536, 256, 64, GGML_TYPE_F16, 512, true, true, false, false, false, true);
}

static void run_llama_attention_matmul_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 16 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t input_size      = 2048;
    constexpr int64_t head_size       = 128;
    constexpr int64_t head_count      = 2;
    constexpr int64_t output_size     = head_size * head_count;
    constexpr int64_t token_count     = 7;
    constexpr int64_t cache_row_count = 16;

    const ggml_type weight_types[] = { GGML_TYPE_F16, GGML_TYPE_BF16 };
    for (ggml_type weight_type : weight_types) {
        {
            ggml_tensor * weight    = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
            ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
            ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
            ggml_tensor * freqs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
            REQUIRE(weight != nullptr);
            REQUIRE(input != nullptr);
            REQUIRE(positions != nullptr);
            REQUIRE(freqs != nullptr);
            ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
            ggml_tensor * query      = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
            ggml_tensor * output     = ggml_rope_ext(ctx, query, positions, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0,
                                                     10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            REQUIRE(projection != nullptr);
            REQUIRE(query != nullptr);
            REQUIRE(output != nullptr);
            schedule_fused_llama_attention_matmul_command(
                ctx, output, "loom_libs:llm_attention_q_matmul_rope_tiled_f32_f32", weight_type, token_count,
                input_size, output_size, head_size, head_count, 0, 0, 3,
                { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ROPE },
                { "input", "weight", "positions", "theta", "freq_factors", "output" });
        }

        {
            ggml_tensor * weight    = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
            ggml_tensor * input     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
            ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
            ggml_tensor * freqs     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_size / 2);
            ggml_tensor * cache     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, output_size, cache_row_count);
            ggml_tensor * indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
            REQUIRE(weight != nullptr);
            REQUIRE(input != nullptr);
            REQUIRE(positions != nullptr);
            REQUIRE(freqs != nullptr);
            REQUIRE(cache != nullptr);
            REQUIRE(indices != nullptr);
            ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
            ggml_tensor * key        = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
            ggml_tensor * rope       = ggml_rope_ext(ctx, key, positions, freqs, head_size, GGML_ROPE_TYPE_NORMAL, 0,
                                                     10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * rows       = ggml_reshape_2d(ctx, rope, output_size, token_count);
            ggml_tensor * output     = ggml_set_rows(ctx, cache, rows, indices);
            REQUIRE(projection != nullptr);
            REQUIRE(key != nullptr);
            REQUIRE(rope != nullptr);
            REQUIRE(rows != nullptr);
            REQUIRE(output != nullptr);
            schedule_fused_llama_attention_matmul_command(
                ctx, output, "loom_libs:llm_attention_k_matmul_rope_set_rows_tiled_f32_f32", weight_type, token_count,
                input_size, output_size, head_size, head_count, cache_row_count, 16, 5,
                { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_SET_ROWS },
                { "input", "weight", "positions", "indices", "theta", "freq_factors", "cache" });
        }

        {
            ggml_tensor * weight  = ggml_new_tensor_2d(ctx, weight_type, input_size, output_size);
            ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size, token_count);
            ggml_tensor * cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, output_size, cache_row_count);
            ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, token_count);
            REQUIRE(weight != nullptr);
            REQUIRE(input != nullptr);
            REQUIRE(cache != nullptr);
            REQUIRE(indices != nullptr);
            ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
            ggml_tensor * value      = ggml_reshape_3d(ctx, projection, head_size, head_count, token_count);
            ggml_tensor * rows       = ggml_reshape_2d(ctx, value, output_size, token_count);
            ggml_tensor * output     = ggml_set_rows(ctx, cache, rows, indices);
            REQUIRE(projection != nullptr);
            REQUIRE(value != nullptr);
            REQUIRE(rows != nullptr);
            REQUIRE(output != nullptr);
            schedule_fused_llama_attention_matmul_command(
                ctx, output, "loom_libs:llm_attention_v_matmul_set_rows_tiled_f32_f32", weight_type, token_count,
                input_size, output_size, 0, 0, cache_row_count, 16, 4,
                { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_SET_ROWS },
                { "input", "weight", "indices", "cache" });
        }
    }

    ggml_free(ctx);
}

static void schedule_qwen_router_top8_command(ggml_context * ctx,
                                              ggml_tensor *  output,
                                              ggml_tensor *  route_ids,
                                              int64_t        expected_expert_count = kQwenRouterExpertCount,
                                              int64_t        expected_route_count  = kQwenRouterRouteCount) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 10);

    const ggml::hrx::Value * route_ids_value = imported.graph.values().find_tensor(route_ids);
    const ggml::hrx::Value * output_value    = imported.graph.values().find_tensor(output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(output_value != nullptr);
    REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(output_value->kind == ggml::hrx::ValueKind::External);

    const ggml::hrx::GraphNode * softmax_node  = nullptr;
    const ggml::hrx::GraphNode * get_rows_node = nullptr;
    for (const ggml::hrx::GraphNode & node : imported.graph.nodes()) {
        if (node.op == GGML_OP_SOFT_MAX) {
            softmax_node = &node;
        } else if (node.op == GGML_OP_GET_ROWS) {
            get_rows_node = &node;
        }
    }
    REQUIRE(softmax_node != nullptr);
    REQUIRE(get_rows_node != nullptr);
    const ggml::hrx::GraphNode * probs_reshape =
        ggml::hrx::find_single_layout_alias_consumer_with_op(imported.graph, softmax_node->output, GGML_OP_RESHAPE);
    REQUIRE(probs_reshape != nullptr);
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, *probs_reshape));
    REQUIRE(ggml::hrx::find_single_consumer_with_op_through_layout_aliases(imported.graph, softmax_node->output,
                                                                           GGML_OP_GET_ROWS) == get_rows_node);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());

    const int64_t token_count        = output->ne[2];
    const size_t  route_id_length    = static_cast<size_t>(token_count) * route_ids->nb[1];
    const size_t  expert_table_bytes = qwen_expert_table_size(token_count, expected_expert_count);
    const size_t  partition_table_bytes =
        qwen_partition_table_size(token_count, expected_route_count, expected_expert_count);
    const bool uses_fused_prefill_expert_table_partition =
        token_count == 512 && expected_route_count == 8 &&
        route_ids->nb[1] / sizeof(int32_t) == static_cast<size_t>(expected_route_count) && expected_expert_count == 128;
    REQUIRE(scheduler.plan().dispatches.size() == (uses_fused_prefill_expert_table_partition ? 2 : 3));
    REQUIRE(scheduler.plan().transients.size() == 2);
    REQUIRE(scheduler.plan().constant_initializations.empty());
    REQUIRE(scheduler.plan().completion_counter_requests.size() == (uses_fused_prefill_expert_table_partition ? 1 : 0));

    const ggml::hrx::CommandPlanTransient & expert_table_transient    = scheduler.plan().transients[0];
    const ggml::hrx::CommandPlanTransient & partition_table_transient = scheduler.plan().transients[1];
    REQUIRE(expert_table_transient.value.value == static_cast<int32_t>(imported.graph.values().size()));
    REQUIRE(expert_table_transient.name == "qwen.router.expert_table");
    REQUIRE(expert_table_transient.size == expert_table_bytes);
    REQUIRE(partition_table_transient.value.value == expert_table_transient.value.value + 1);
    REQUIRE(partition_table_transient.name == "qwen.router.partition_table");
    REQUIRE(partition_table_transient.size == partition_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        REQUIRE(completion_counter_request.value.value == partition_table_transient.value.value + 1);
        REQUIRE(completion_counter_request.name == "qwen.router.prefill_expert_table_partition_completion_counter");
        REQUIRE(completion_counter_request.count == 1);
    }

    const ggml::hrx::Dispatch & dispatch    = scheduler.plan().dispatches[0];
    const std::string           kernel_name = kernel_name_for_id(dispatch.kernel.kernel_id);
    REQUIRE(kernel_name == "qwen3_moe:qwen3_moe_router_top8_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == route_ids->nb[1] / sizeof(int32_t));
    REQUIRE(dispatch.bindings.size() == 3);
    REQUIRE(dispatch.bindings[1].value == route_ids_value->id);
    REQUIRE(dispatch.bindings[1].length == route_id_length);
    REQUIRE(dispatch.bindings[2].value == output_value->id);
    require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expected_expert_count));
    require_compile_parameter(dispatch, "qwen3_moe.router.route_count", std::to_string(expected_route_count));
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(token_count));

    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        const ggml::hrx::Dispatch & expert_table_partition_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(expert_table_partition_dispatch.kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_build_expert_table_partition_prefill_512");
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("route_stride") ==
                route_ids->nb[1] / sizeof(int32_t));
        REQUIRE(expert_table_partition_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(expert_table_partition_dispatch.bindings.size() == 4);
        REQUIRE(expert_table_partition_dispatch.bindings[0].value == route_ids_value->id);
        REQUIRE(expert_table_partition_dispatch.bindings[0].length == route_id_length);
        REQUIRE(expert_table_partition_dispatch.bindings[1].value == expert_table_transient.value);
        REQUIRE(expert_table_partition_dispatch.bindings[1].length == expert_table_bytes);
        REQUIRE(expert_table_partition_dispatch.bindings[2].value == partition_table_transient.value);
        REQUIRE(expert_table_partition_dispatch.bindings[2].length == partition_table_bytes);
        REQUIRE(expert_table_partition_dispatch.bindings[3].value == completion_counter_request.value);
        REQUIRE(expert_table_partition_dispatch.bindings[3].length == sizeof(int32_t));
    } else {
        const ggml::hrx::Dispatch & expert_table_dispatch = scheduler.plan().dispatches[1];
        REQUIRE(kernel_name_for_id(expert_table_dispatch.kernel.kernel_id) == "loom_libs:ggml_moe_build_expert_table");
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("route_stride") ==
                route_ids->nb[1] / sizeof(int32_t));
        REQUIRE(expert_table_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(expert_table_dispatch.bindings.size() == 2);
        REQUIRE(expert_table_dispatch.bindings[0].value == route_ids_value->id);
        REQUIRE(expert_table_dispatch.bindings[0].length == route_id_length);
        REQUIRE(expert_table_dispatch.bindings[1].value == expert_table_transient.value);
        REQUIRE(expert_table_dispatch.bindings[1].length == expert_table_bytes);
        require_compile_parameter(expert_table_dispatch, "ggml.moe_routing.expert_count",
                                  std::to_string(expected_expert_count));
        require_compile_parameter(expert_table_dispatch, "ggml.moe_routing.route_count",
                                  std::to_string(expected_route_count));
        require_compile_parameter(expert_table_dispatch, "ggml.workload.token_capacity", std::to_string(token_count));

        const ggml::hrx::Dispatch & partition_table_dispatch = scheduler.plan().dispatches[2];
        REQUIRE(kernel_name_for_id(partition_table_dispatch.kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_partition_table");
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("route_count") == expected_route_count);
        REQUIRE(partition_table_dispatch.kernel.integer_parameters.at("expert_count") == expected_expert_count);
        REQUIRE(partition_table_dispatch.bindings.size() == 2);
        REQUIRE(partition_table_dispatch.bindings[0].value == expert_table_transient.value);
        REQUIRE(partition_table_dispatch.bindings[0].length == expert_table_bytes);
        REQUIRE(partition_table_dispatch.bindings[1].value == partition_table_transient.value);
        REQUIRE(partition_table_dispatch.bindings[1].length == partition_table_bytes);
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.expert_count",
                                  std::to_string(expected_expert_count));
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.route_count",
                                  std::to_string(expected_route_count));
        require_compile_parameter(partition_table_dispatch, "ggml.workload.token_capacity",
                                  std::to_string(token_count));
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_expert_mask", "127");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_partition_shift", "7");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.descriptor_row_count_shift", "13");
        require_compile_parameter(partition_table_dispatch, "ggml.moe_routing.partition_workgroup_size", "128");
    }

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == (uses_fused_prefill_expert_table_partition ? 2 : 3));
    REQUIRE(commands.constant_initializations.empty());
    REQUIRE(commands.completion_counters.count == (uses_fused_prefill_expert_table_partition ? 1 : 0));
    REQUIRE(commands.completion_counters.byte_count ==
            (uses_fused_prefill_expert_table_partition ? sizeof(int32_t) : 0));
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].name == "logits");
    REQUIRE(commands.commands[0].bindings[1].name == "route_ids");
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[0].bindings[1].value == route_ids_value->storage_root);
    REQUIRE(commands.commands[0].bindings[1].offset == route_ids_value->storage_offset);
    REQUIRE(commands.commands[0].bindings[1].length == dispatch.bindings[1].length);
    REQUIRE(commands.commands[0].bindings[2].name == "route_weights");
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[1].bindings.size() == (uses_fused_prefill_expert_table_partition ? 4 : 2));
    REQUIRE(commands.commands[1].bindings[0].name == "route_ids");
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == route_ids_value->storage_root);
    REQUIRE(commands.commands[1].bindings[0].offset == route_ids_value->storage_offset);
    REQUIRE(commands.commands[1].bindings[1].name == "expert_table");
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].length == expert_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        REQUIRE(commands.commands[1].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[1].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[1].bindings[2].length == partition_table_bytes);
        REQUIRE(commands.commands[1].bindings[3].name == "completion_counter");
        REQUIRE(commands.commands[1].bindings[3].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[1].bindings[3].length == sizeof(int32_t));
    } else {
        REQUIRE(commands.commands[2].dependencies.size() == 1);
        REQUIRE(commands.commands[2].dependencies[0] == 1);
        REQUIRE(commands.commands[2].bindings.size() == 2);
        REQUIRE(commands.commands[2].bindings[0].name == "expert_table");
        REQUIRE(commands.commands[2].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[2].bindings[1].name == "partition_table");
        REQUIRE(commands.commands[2].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[2].bindings[1].length == partition_table_bytes);
    }
    const ggml::hrx::TransientAllocation * route_ids_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, route_ids_value->storage_root);
    REQUIRE(route_ids_allocation != nullptr);
    REQUIRE(route_ids_allocation->size == dispatch.bindings[1].length);
    const ggml::hrx::TransientAllocation * expert_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, expert_table_transient.value);
    REQUIRE(expert_table_allocation != nullptr);
    REQUIRE(expert_table_allocation->size == expert_table_bytes);
    const ggml::hrx::TransientAllocation * partition_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, partition_table_transient.value);
    REQUIRE(partition_table_allocation != nullptr);
    REQUIRE(partition_table_allocation->size == partition_table_bytes);
    if (uses_fused_prefill_expert_table_partition) {
        const ggml::hrx::CommandPlanCompletionCounterRequest & completion_counter_request =
            scheduler.plan().completion_counter_requests[0];
        const ggml::hrx::TransientAllocation * completion_counter_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, completion_counter_request.value);
        REQUIRE(completion_counter_allocation != nullptr);
        REQUIRE(completion_counter_allocation->size == sizeof(int32_t));
        REQUIRE(completion_counter_allocation->arena_offset == commands.completion_counters.arena_offset);
    }
}

static void schedule_manual_qwen_router_top8_command(ggml::hrx::Graph & graph,
                                                     ggml::hrx::ValueId route_ids,
                                                     int64_t            token_count,
                                                     int64_t            expert_count,
                                                     int64_t            route_count) {
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);
    REQUIRE(scheduler.plan().transients.size() == 2);

    const size_t route_id_length       = static_cast<size_t>(token_count * route_count) * sizeof(int32_t);
    const size_t expert_table_bytes    = qwen_expert_table_size(token_count, expert_count);
    const size_t partition_table_bytes = qwen_partition_table_size(token_count, route_count, expert_count);

    const ggml::hrx::Dispatch & dispatch = scheduler.plan().dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "qwen3_moe:qwen3_moe_router_top8_f32");
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
    REQUIRE(dispatch.kernel.integer_parameters.at("route_id_stride") == route_count);
    REQUIRE(dispatch.bindings.size() == 3);
    REQUIRE(dispatch.bindings[1].value == route_ids);
    REQUIRE(dispatch.bindings[1].length == route_id_length);
    require_compile_parameter(dispatch, "qwen3_moe.router.expert_count", std::to_string(expert_count));
    require_compile_parameter(dispatch, "qwen3_moe.router.route_count", std::to_string(route_count));

    const ggml::hrx::CommandPlanTransient & expert_table_transient    = scheduler.plan().transients[0];
    const ggml::hrx::CommandPlanTransient & partition_table_transient = scheduler.plan().transients[1];
    REQUIRE(expert_table_transient.size == expert_table_bytes);
    REQUIRE(partition_table_transient.size == partition_table_bytes);

    const ggml::hrx::CommandPlanMoeRoutingBundle * bundle =
        scheduler.plan().metadata.find_moe_routing_bundle(route_ids);
    REQUIRE(bundle != nullptr);
    REQUIRE(bundle->token_count == token_count);
    REQUIRE(bundle->route_count == route_count);
    REQUIRE(bundle->route_stride == route_count);
    REQUIRE(bundle->expert_count == expert_count);
    REQUIRE(bundle->expert_table_byte_count == expert_table_bytes);
    REQUIRE(bundle->partition_table_byte_count == partition_table_bytes);

    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[0].bindings[1].value == route_ids);
    REQUIRE(commands.commands[0].bindings[1].length == route_id_length);
}

struct QwenRoutedGateUpTensors {
    ggml_tensor *              route_ids     = nullptr;
    ggml_tensor *              route_weights = nullptr;
    ggml_tensor *              input         = nullptr;
    ggml_tensor *              gate_weight   = nullptr;
    ggml_tensor *              up_weight     = nullptr;
    ggml_tensor *              down_weight   = nullptr;
    ggml_tensor *              gate          = nullptr;
    ggml_tensor *              up            = nullptr;
    ggml_tensor *              glu           = nullptr;
    ggml_tensor *              output        = nullptr;
    ggml_tensor *              weighted      = nullptr;
    ggml_tensor *              hidden_state  = nullptr;
    ggml_tensor *              residual      = nullptr;
    ggml_tensor *              next_rms      = nullptr;
    ggml_tensor *              next_output   = nullptr;
    ggml_tensor *              hidden_use    = nullptr;
    std::vector<ggml_tensor *> route_views;
};

static QwenRoutedGateUpTensors build_qwen_routed_gate_up_graph(ggml_context * ctx,
                                                               int64_t        token_count,
                                                               ggml_glu_op    glu_op           = GGML_GLU_OP_SWIGLU,
                                                               ggml_type      up_weight_type   = GGML_TYPE_Q4_K,
                                                               bool           include_down     = true,
                                                               ggml_type      down_weight_type = GGML_TYPE_Q6_K) {
    QwenRoutedGateUpTensors tensors;
    ggml_tensor *           logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
    REQUIRE(logits != nullptr);
    tensors.route_weights = build_qwen_router_top8_graph(ctx, logits, &tensors.route_ids);
    REQUIRE(tensors.route_weights != nullptr);
    REQUIRE(tensors.route_ids != nullptr);

    tensors.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, 1, token_count);
    tensors.gate_weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    tensors.up_weight =
        ggml_new_tensor_3d(ctx, up_weight_type, kQwenMoeHiddenSize, kQwenMoeIntermediateSize, kQwenRouterExpertCount);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.gate_weight != nullptr);
    REQUIRE(tensors.up_weight != nullptr);

    tensors.gate = ggml_mul_mat_id(ctx, tensors.gate_weight, tensors.input, tensors.route_ids);
    tensors.up   = ggml_mul_mat_id(ctx, tensors.up_weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.gate != nullptr);
    REQUIRE(tensors.up != nullptr);
    tensors.glu = ggml_glu_split(ctx, tensors.gate, tensors.up, glu_op);
    REQUIRE(tensors.glu != nullptr);

    if (include_down) {
        tensors.down_weight = ggml_new_tensor_3d(ctx, down_weight_type, kQwenMoeIntermediateSize, kQwenMoeHiddenSize,
                                                 kQwenRouterExpertCount);
        REQUIRE(tensors.down_weight != nullptr);
        tensors.output = ggml_mul_mat_id(ctx, tensors.down_weight, tensors.glu, tensors.route_ids);
        REQUIRE(tensors.output != nullptr);
    } else {
        tensors.output = tensors.glu;
    }
    return tensors;
}

static void append_qwen_weighted_reduce_tail(ggml_context *            ctx,
                                             QwenRoutedGateUpTensors & tensors,
                                             bool                      include_next_rmsnorm = false,
                                             ggml_tensor *             routed_input         = nullptr) {
    REQUIRE(tensors.output != nullptr);
    REQUIRE(tensors.route_weights != nullptr);
    tensors.weighted = ggml_mul(ctx, routed_input != nullptr ? routed_input : tensors.output, tensors.route_weights);
    REQUIRE(tensors.weighted != nullptr);
    tensors.route_views.clear();
    for (int64_t route = 0; route < kQwenRouterRouteCount; ++route) {
        ggml_tensor * view =
            ggml_view_2d(ctx, tensors.weighted, kQwenMoeHiddenSize, tensors.weighted->ne[2], tensors.weighted->nb[2],
                         static_cast<size_t>(route) * tensors.weighted->nb[1]);
        REQUIRE(view != nullptr);
        tensors.route_views.push_back(view);
    }

    ggml_tensor * reduced = tensors.route_views.front();
    for (size_t i = 1; i < tensors.route_views.size(); ++i) {
        reduced = ggml_add(ctx, reduced, tensors.route_views[i]);
        REQUIRE(reduced != nullptr);
    }

    tensors.hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, tensors.output->ne[2]);
    REQUIRE(tensors.hidden_state != nullptr);
    tensors.residual = ggml_add(ctx, tensors.hidden_state, reduced);
    REQUIRE(tensors.residual != nullptr);

    if (include_next_rmsnorm) {
        ggml_tensor * next_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize);
        REQUIRE(next_norm_weight != nullptr);
        tensors.next_rms = ggml_rms_norm(ctx, tensors.residual, 0.000001f);
        REQUIRE(tensors.next_rms != nullptr);
        tensors.next_output = ggml_mul(ctx, tensors.next_rms, next_norm_weight);
        REQUIRE(tensors.next_output != nullptr);
    }
}

static ggml::hrx::GraphImportResult import_qwen_routed_gate_up_graph(ggml_context *                  ctx,
                                                                     const QwenRoutedGateUpTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.route_weights);
    ggml_build_forward_expand(graph, tensors.output);
    if (tensors.residual != nullptr) {
        ggml_build_forward_expand(graph, tensors.residual);
    }
    if (tensors.next_output != nullptr) {
        ggml_build_forward_expand(graph, tensors.next_output);
    }
    if (tensors.hidden_use != nullptr) {
        ggml_build_forward_expand(graph, tensors.hidden_use);
    }

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static bool match_dispatch_at_index(const ggml::hrx::Graph &       graph,
                                    const ggml::hrx::CommandPlan & plan,
                                    const std::vector<bool> &      covered_nodes,
                                    size_t                         node_index,
                                    ggml::hrx::DispatchMatch &     match) {
    REQUIRE(node_index < graph.nodes().size());
    const ggml::hrx::DispatchMatchContext context = {
        graph,
        &graph.nodes()[node_index],
        node_index,
        covered_nodes,
        plan,
        ggml::hrx::ValueId(static_cast<int32_t>(graph.values().size() + plan.transients.size() +
                                                plan.completion_counter_requests.size())),
        &test_dispatch_registry(),
    };
    return test_dispatch_registry().match(context, match);
}

static void append_match_to_plan(ggml::hrx::CommandPlan &   plan,
                                 ggml::hrx::DispatchMatch & match,
                                 std::vector<bool> &        covered_nodes,
                                 ggml::hrx::Graph *         graph) {
    if (graph != nullptr) {
        for (const ggml::hrx::DispatchValueAliasRequest & alias : match.value_aliases) {
            ggml::hrx::Status status = graph->values().alias_storage(alias.target_value, alias.source_value);
            REQUIRE(status.success());
        }
    }
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
    for (const size_t covered_node : match.covered_nodes) {
        REQUIRE(covered_node < covered_nodes.size());
        REQUIRE(!covered_nodes[covered_node]);
        covered_nodes[covered_node] = true;
    }
}

static ggml::hrx::CommandPlan build_qwen_router_plan_for_graph(ggml::hrx::Graph &  graph,
                                                               std::vector<bool> & covered_nodes) {
    ggml::hrx::CommandPlan plan;
    size_t                 softmax_index = graph.nodes().size();
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].op == GGML_OP_SOFT_MAX) {
            softmax_index = i;
            break;
        }
    }
    REQUIRE(softmax_index < graph.nodes().size());
    ggml::hrx::DispatchMatch router_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, softmax_index, router_match));
    append_match_to_plan(plan, router_match, covered_nodes, &graph);
    return plan;
}

static void append_qwen_routed_gate_up_for_graph(ggml::hrx::Graph &              graph,
                                                 const QwenRoutedGateUpTensors & tensors,
                                                 std::vector<bool> &             covered_nodes,
                                                 ggml::hrx::CommandPlan &        plan) {
    const size_t             gate_index = producer_index_for_tensor(graph, tensors.gate);
    ggml::hrx::DispatchMatch gate_up_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, gate_index, gate_up_match));
    append_match_to_plan(plan, gate_up_match, covered_nodes, &graph);
}

static void append_qwen_routed_down_for_graph(ggml::hrx::Graph &              graph,
                                              const QwenRoutedGateUpTensors & tensors,
                                              std::vector<bool> &             covered_nodes,
                                              ggml::hrx::CommandPlan &        plan,
                                              const char *                    expected_kernel_name) {
    const size_t             down_index = producer_index_for_tensor(graph, tensors.output);
    ggml::hrx::DispatchMatch down_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, down_index, down_match));
    append_match_to_plan(plan, down_match, covered_nodes, &graph);

    REQUIRE(plan.dispatches.size() >= 1);
    REQUIRE(plan.transients.size() >= 1);
    const ggml::hrx::Value * route_ids_value = graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * glu_value       = graph.values().find_tensor(tensors.glu);
    const ggml::hrx::Value * output_value    = graph.values().find_tensor(tensors.output);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(glu_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        plan.metadata.find_moe_routing_bundle(route_ids_value->id);
    const ggml::hrx::CommandPlanAlternateValue * gate_up_alternate = plan.metadata.find_alternate_value(
        glu_value->id, GGML_TYPE_F16, qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    const ggml::hrx::CommandPlanAlternateValue * routed_down_alternate = plan.metadata.find_alternate_value(
        output_value->id, GGML_TYPE_F16, qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(routing_bundle != nullptr);
    REQUIRE(gate_up_alternate != nullptr);
    REQUIRE(routed_down_alternate != nullptr);
    const ggml::hrx::Dispatch &             dispatch              = plan.dispatches.back();
    const ggml::hrx::CommandPlanTransient & routed_down_transient = plan.transients.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 4);
    REQUIRE(routed_down_transient.name == "qwen.moe.routed_down_f16");
    REQUIRE(routed_down_transient.size == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[0].value == gate_up_alternate->alternate_value);
    REQUIRE(dispatch.bindings[0].length == qwen_routed_gate_up_f16_output_size(tensors.output->ne[2]));
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[1].length == routing_bundle->expert_table_byte_count);
    REQUIRE(routed_down_alternate->alternate_value == routed_down_transient.value);
    REQUIRE(routed_down_alternate->byte_count == routed_down_transient.size);
    REQUIRE(dispatch.bindings[3].value == routed_down_transient.value);
    REQUIRE(dispatch.bindings[3].length == routed_down_transient.size);
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.input_size", "768");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.route_count", "8");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.expert_count", "128");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.output_size", "2048");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_f16_f16.weight_format",
                              tensors.down_weight->type == GGML_TYPE_Q4_K ? "4" : "6");
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(tensors.output->ne[2]));
}

static std::string common_mul_mat_weight_format(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K:
            return "11";
        case GGML_TYPE_Q4_K:
            return "4";
        case GGML_TYPE_Q5_K:
            return "5";
        case GGML_TYPE_Q6_K:
            return "6";
        case GGML_TYPE_IQ3_S:
            return "21";
        case GGML_TYPE_IQ4_NL:
            return "20";
        case GGML_TYPE_IQ4_XS:
            return "23";
        case GGML_TYPE_Q8_0:
            return "80";
        case GGML_TYPE_Q8_1:
            return "81";
        case GGML_TYPE_F16:
            return "16";
        case GGML_TYPE_BF16:
            return "30";
        case GGML_TYPE_F32:
            return "32";
        default:
            return "0";
    }
}

static const char * common_mul_mat_id_f32_kernel_name(int64_t token_count) {
    return token_count <= 5 ? "loom_libs:ggml_mul_mat_id_skinny_input_f32_publish_f32" :
                              "loom_libs:ggml_mul_mat_id_tiled_input_f32_publish_f32";
}

static const char * common_mul_mat_id_postops_f32_kernel_name(int64_t token_count, bool has_rmsnorm) {
    if (has_rmsnorm) {
        return token_count <= 5 ? "loom_libs:ggml_mul_mat_id_skinny_input_f32_postops_next_rmsnorm_publish_f32" :
                                  "loom_libs:ggml_mul_mat_id_tiled_input_f32_postops_next_rmsnorm_publish_f32";
    }
    return token_count <= 5 ? "loom_libs:ggml_mul_mat_id_skinny_input_f32_postops_publish_f32" :
                              "loom_libs:ggml_mul_mat_id_tiled_input_f32_postops_publish_f32";
}

static const char * common_mul_mat_id_swiglu_f32_kernel_name(int64_t token_count) {
    return token_count <= 5 ? "loom_libs:ggml_mul_mat_id_skinny_pair_input_f32_swiglu_publish_f32" :
                              "loom_libs:ggml_mul_mat_id_tiled_pair_input_f32_swiglu_publish_f32";
}

static ggml::hrx::DispatchMatch require_common_mul_mat_id_for_graph(ggml::hrx::Graph &       graph,
                                                                    ggml::hrx::CommandPlan & plan,
                                                                    std::vector<bool> &      covered_nodes,
                                                                    ggml_tensor *            output,
                                                                    ggml_tensor *            input,
                                                                    ggml_tensor *            weight,
                                                                    ggml_tensor *            route_ids) {
    const ggml::hrx::Value * output_value    = graph.values().find_tensor(output);
    const ggml::hrx::Value * input_value     = graph.values().find_tensor(input);
    const ggml::hrx::Value * weight_value    = graph.values().find_tensor(weight);
    const ggml::hrx::Value * route_ids_value = graph.values().find_tensor(route_ids);
    REQUIRE(output_value != nullptr);
    REQUIRE(input_value != nullptr);
    REQUIRE(weight_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        plan.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    const size_t             output_index = producer_index_for_tensor(graph, output);
    ggml::hrx::DispatchMatch match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, output_index, match));
    REQUIRE(match.dispatches.size() == 1);
    const ggml::hrx::Dispatch & dispatch = match.dispatches[0];
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == common_mul_mat_id_f32_kernel_name(output->ne[2]));
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 5);
    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[0].length == input_value->byte_count);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[1].length == qwen_expert_table_size(output->ne[2], weight->ne[2]));
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[2].length == qwen_partition_table_size(output->ne[2], route_ids->ne[0], weight->ne[2]));
    REQUIRE(dispatch.bindings[3].value == weight_value->id);
    REQUIRE(dispatch.bindings[3].length == weight_value->byte_count);
    REQUIRE(dispatch.bindings[4].value == output_value->id);
    REQUIRE(dispatch.bindings[4].length == output_value->byte_count);
    require_compile_parameter(dispatch, "ggml.mul_mat_id.input_size", std::to_string(weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.output_size", std::to_string(weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.expert_count", std::to_string(weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.route_count", std::to_string(route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.input_route_count", std::to_string(input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id.weight_format", common_mul_mat_weight_format(weight->type));
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(output->ne[2]));
    return match;
}

struct CommonMulMatIdSwiGLUTensors {
    ggml_tensor * input       = nullptr;
    ggml_tensor * route_ids   = nullptr;
    ggml_tensor * gate_weight = nullptr;
    ggml_tensor * up_weight   = nullptr;
    ggml_tensor * gate        = nullptr;
    ggml_tensor * up          = nullptr;
    ggml_tensor * output      = nullptr;
};

static CommonMulMatIdSwiGLUTensors build_common_mul_mat_id_swiglu_graph(ggml_context * ctx,
                                                                        ggml_type      gate_weight_type,
                                                                        ggml_type      up_weight_type,
                                                                        ggml_glu_op    glu_op      = GGML_GLU_OP_SWIGLU,
                                                                        int64_t        input_size  = 256,
                                                                        int64_t        output_size = 128,
                                                                        int64_t        token_count = 4) {
    CommonMulMatIdSwiGLUTensors tensors;
    constexpr int64_t           route_count       = 8;
    constexpr int64_t           input_route_count = 1;
    constexpr int64_t           expert_count      = 16;
    tensors.route_ids                             = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count);
    tensors.input       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_size, input_route_count, token_count);
    tensors.gate_weight = ggml_new_tensor_3d(ctx, gate_weight_type, input_size, output_size, expert_count);
    tensors.up_weight   = ggml_new_tensor_3d(ctx, up_weight_type, input_size, output_size, expert_count);
    REQUIRE(tensors.route_ids != nullptr);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.gate_weight != nullptr);
    REQUIRE(tensors.up_weight != nullptr);

    tensors.gate = ggml_mul_mat_id(ctx, tensors.gate_weight, tensors.input, tensors.route_ids);
    tensors.up   = ggml_mul_mat_id(ctx, tensors.up_weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.gate != nullptr);
    REQUIRE(tensors.up != nullptr);
    tensors.output = ggml_glu_split(ctx, tensors.gate, tensors.up, glu_op);
    REQUIRE(tensors.output != nullptr);
    return tensors;
}

static ggml::hrx::GraphImportResult import_common_mul_mat_id_swiglu_graph(ggml_context *                      ctx,
                                                                          const CommonMulMatIdSwiGLUTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static void require_common_mul_mat_id_swiglu_match(ggml_context *                      ctx,
                                                   const CommonMulMatIdSwiGLUTensors & tensors,
                                                   ggml_tensor *                       root_projection) {
    ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_swiglu_graph(ctx, tensors);
    std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan       plan;
    const size_t                 root_index = producer_index_for_tensor(imported.graph, root_projection);
    ggml::hrx::DispatchMatch     match;
    REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, root_index, match));
    REQUIRE(match.dispatches.size() == 3);
    REQUIRE(match.transients.size() == 2);
    REQUIRE(match.metadata.generated_resources().size() == 2);
    REQUIRE(match.metadata.moe_routing_bundles().size() == 1);

    const ggml::hrx::Dispatch & dispatch = match.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) ==
            common_mul_mat_id_swiglu_f32_kernel_name(tensors.output->ne[2]));
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == 6);
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.input_size",
                              std::to_string(tensors.gate_weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.output_size",
                              std::to_string(tensors.gate_weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.expert_count",
                              std::to_string(tensors.gate_weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.route_count", std::to_string(tensors.route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.input_route_count",
                              std::to_string(tensors.input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.gate_weight_format",
                              common_mul_mat_weight_format(tensors.gate_weight->type));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu.up_weight_format",
                              common_mul_mat_weight_format(tensors.up_weight->type));
    require_compile_parameter(dispatch, "ggml.workload.token_capacity", std::to_string(tensors.output->ne[2]));

    const ggml::hrx::Value * input_value       = imported.graph.values().find_tensor(tensors.input);
    const ggml::hrx::Value * route_ids_value   = imported.graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * gate_weight_value = imported.graph.values().find_tensor(tensors.gate_weight);
    const ggml::hrx::Value * up_weight_value   = imported.graph.values().find_tensor(tensors.up_weight);
    const ggml::hrx::Value * output_value      = imported.graph.values().find_tensor(tensors.output);
    REQUIRE(input_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(gate_weight_value != nullptr);
    REQUIRE(up_weight_value != nullptr);
    REQUIRE(output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        match.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[3].value == gate_weight_value->id);
    REQUIRE(dispatch.bindings[4].value == up_weight_value->id);
    REQUIRE(dispatch.bindings[5].value == output_value->id);

    append_match_to_plan(plan, match, covered_nodes, &imported.graph);
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
    REQUIRE(command_program_verifies(commands));
    REQUIRE(commands.commands[2].bindings.size() == 6);
    REQUIRE(commands.commands[2].bindings[0].name == "input");
    REQUIRE(commands.commands[2].bindings[1].name == "expert_table");
    REQUIRE(commands.commands[2].bindings[2].name == "partition_table");
    REQUIRE(commands.commands[2].bindings[3].name == "gate_weight");
    REQUIRE(commands.commands[2].bindings[4].name == "up_weight");
    REQUIRE(commands.commands[2].bindings[5].name == "output");
}

static void run_common_mul_mat_id_swiglu_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_Q4_K, GGML_TYPE_F16);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_F32, GGML_TYPE_Q8_1);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.up);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_BF16, GGML_TYPE_BF16);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, GGML_GLU_OP_SWIGLU, 640, 128);
        require_common_mul_mat_id_swiglu_match(ctx, tensors, tensors.gate);
    }
    {
        const CommonMulMatIdSwiGLUTensors tensors =
            build_common_mul_mat_id_swiglu_graph(ctx, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_swiglu_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan;
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, match));
        REQUIRE(kernel_name_for_id(match.dispatches.back().kernel.kernel_id) ==
                common_mul_mat_id_f32_kernel_name(tensors.output->ne[2]));
    }

    ggml_free(ctx);
}

struct CommonMulMatIdPostOpsTensors {
    ggml_tensor * input             = nullptr;
    ggml_tensor * weight            = nullptr;
    ggml_tensor * route_ids         = nullptr;
    ggml_tensor * projection        = nullptr;
    ggml_tensor * bias              = nullptr;
    ggml_tensor * residual_input    = nullptr;
    ggml_tensor * residual_output   = nullptr;
    ggml_tensor * rms_output        = nullptr;
    ggml_tensor * norm_weight       = nullptr;
    ggml_tensor * normalized_output = nullptr;
};

static int64_t common_mul_mat_id_partition_descriptor_capacity(const CommonMulMatIdPostOpsTensors & tensors) {
    const int64_t token_count                = tensors.projection->ne[2];
    const int64_t route_count                = tensors.route_ids->ne[0];
    const int64_t expert_count               = tensors.weight->ne[2];
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return assignment_partition_count + expert_count;
}

static CommonMulMatIdPostOpsTensors build_common_mul_mat_id_postops_graph(ggml_context * ctx,
                                                                          bool           include_bias,
                                                                          bool           include_residual,
                                                                          bool           include_rmsnorm,
                                                                          ggml_type      weight_type  = GGML_TYPE_Q4_K,
                                                                          int64_t        token_count  = 4) {
    CommonMulMatIdPostOpsTensors tensors;
    constexpr int64_t            route_count  = 8;
    constexpr int64_t            input_size   = 256;
    constexpr int64_t            output_size  = 128;
    constexpr int64_t            expert_count = 16;
    tensors.route_ids                         = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_count, token_count);
    tensors.input                             = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_size, 1, token_count);
    tensors.weight = ggml_new_tensor_3d(ctx, weight_type, input_size, output_size, expert_count);
    REQUIRE(tensors.route_ids != nullptr);
    REQUIRE(tensors.input != nullptr);
    REQUIRE(tensors.weight != nullptr);

    tensors.projection = ggml_mul_mat_id(ctx, tensors.weight, tensors.input, tensors.route_ids);
    REQUIRE(tensors.projection != nullptr);

    ggml_tensor * current = tensors.projection;
    if (include_bias) {
        tensors.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_size);
        REQUIRE(tensors.bias != nullptr);
        current = ggml_add(ctx, current, tensors.bias);
        REQUIRE(current != nullptr);
    }
    if (include_residual) {
        tensors.residual_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, output_size, route_count, token_count);
        REQUIRE(tensors.residual_input != nullptr);
        current = ggml_add(ctx, current, tensors.residual_input);
        REQUIRE(current != nullptr);
    }
    tensors.residual_output = current;

    if (include_rmsnorm) {
        REQUIRE(include_residual);
        tensors.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_size);
        REQUIRE(tensors.norm_weight != nullptr);
        tensors.rms_output = ggml_rms_norm(ctx, tensors.residual_output, 0.000001f);
        REQUIRE(tensors.rms_output != nullptr);
        tensors.normalized_output = ggml_mul(ctx, tensors.rms_output, tensors.norm_weight);
        REQUIRE(tensors.normalized_output != nullptr);
    } else {
        tensors.normalized_output = tensors.residual_output;
    }
    return tensors;
}

static ggml::hrx::GraphImportResult import_common_mul_mat_id_postops_graph(
    ggml_context *                       ctx,
    const CommonMulMatIdPostOpsTensors & tensors) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, tensors.normalized_output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    return imported;
}

static void require_common_mul_mat_id_postops_match(ggml_context *                       ctx,
                                                    const CommonMulMatIdPostOpsTensors & tensors,
                                                    const char *                         expected_kernel_name,
                                                    bool                                 expect_bias,
                                                    bool                                 expect_residual,
                                                    bool                                 expect_rmsnorm) {
    ggml::hrx::GraphImportResult imported = import_common_mul_mat_id_postops_graph(ctx, tensors);
    std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
    ggml::hrx::CommandPlan       plan;
    const size_t                 projection_index = producer_index_for_tensor(imported.graph, tensors.projection);
    ggml::hrx::DispatchMatch     match;
    REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, projection_index, match));
    REQUIRE(match.dispatches.size() == 3);
    REQUIRE(match.transients.size() == 2);

    const ggml::hrx::Dispatch & dispatch = match.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.projection->ne[2]);
    REQUIRE(dispatch.bindings.size() == (expect_rmsnorm ? 10 : 7));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.input_size", std::to_string(tensors.weight->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.output_size", std::to_string(tensors.weight->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.expert_count", std::to_string(tensors.weight->ne[2]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.route_count",
                              std::to_string(tensors.route_ids->ne[0]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.input_route_count",
                              std::to_string(tensors.input->ne[1]));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.weight_format",
                              common_mul_mat_weight_format(tensors.weight->type));
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.has_bias", expect_bias ? "1" : "0");
    require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.has_residual", expect_residual ? "1" : "0");

    const ggml::hrx::Value * input_value             = imported.graph.values().find_tensor(tensors.input);
    const ggml::hrx::Value * weight_value            = imported.graph.values().find_tensor(tensors.weight);
    const ggml::hrx::Value * route_ids_value         = imported.graph.values().find_tensor(tensors.route_ids);
    const ggml::hrx::Value * residual_output_value   = imported.graph.values().find_tensor(tensors.residual_output);
    const ggml::hrx::Value * normalized_output_value = imported.graph.values().find_tensor(tensors.normalized_output);
    REQUIRE(input_value != nullptr);
    REQUIRE(weight_value != nullptr);
    REQUIRE(route_ids_value != nullptr);
    REQUIRE(residual_output_value != nullptr);
    REQUIRE(normalized_output_value != nullptr);
    const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
        match.metadata.find_moe_routing_bundle(route_ids_value->id);
    REQUIRE(routing_bundle != nullptr);

    REQUIRE(dispatch.bindings[0].value == input_value->id);
    REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
    REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
    REQUIRE(dispatch.bindings[3].value == weight_value->id);
    if (expect_bias) {
        const ggml::hrx::Value * bias_value = imported.graph.values().find_tensor(tensors.bias);
        REQUIRE(bias_value != nullptr);
        REQUIRE(dispatch.bindings[4].value == bias_value->id);
    }
    if (expect_residual) {
        const ggml::hrx::Value * residual_input_value = imported.graph.values().find_tensor(tensors.residual_input);
        REQUIRE(residual_input_value != nullptr);
        REQUIRE(dispatch.bindings[5].value == residual_input_value->id);
    }
    REQUIRE(dispatch.bindings[6].value == residual_output_value->id);

    if (expect_rmsnorm) {
        const ggml::hrx::Value * norm_weight_value = imported.graph.values().find_tensor(tensors.norm_weight);
        REQUIRE(norm_weight_value != nullptr);
        REQUIRE(dispatch.bindings[7].value == norm_weight_value->id);
        REQUIRE(dispatch.bindings[8].value == normalized_output_value->id);
        REQUIRE(match.completion_counter_requests.size() == 1);
        REQUIRE(match.completion_counter_requests.front().count ==
                common_mul_mat_id_partition_descriptor_capacity(tensors));
        REQUIRE(dispatch.bindings[9].value == match.completion_counter_requests.front().value);
        require_compile_parameter(dispatch, "ggml.mul_mat_id_postops.rms_epsilon", "9.99999997e-07");
    } else {
        REQUIRE(match.completion_counter_requests.empty());
    }

    append_match_to_plan(plan, match, covered_nodes, &imported.graph);
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 3);
}

static void run_common_mul_mat_id_postops_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, false, false);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], false), true, false,
            false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, false, true, false);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], false), false, true,
            false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, true, false);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], false), true, true,
            false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors =
            build_common_mul_mat_id_postops_graph(ctx, true, true, false, GGML_TYPE_BF16);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], false), true, true,
            false);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, false, true, true);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], true), false, true,
            true);
    }
    {
        const CommonMulMatIdPostOpsTensors tensors = build_common_mul_mat_id_postops_graph(ctx, true, true, true);
        require_common_mul_mat_id_postops_match(
            ctx, tensors, common_mul_mat_id_postops_f32_kernel_name(tensors.projection->ne[2], true), true, true, true);
    }

    ggml_free(ctx);
}

static void append_qwen_weighted_reduce_for_graph(ggml::hrx::Graph &              graph,
                                                  const QwenRoutedGateUpTensors & tensors,
                                                  std::vector<bool> &             covered_nodes,
                                                  ggml::hrx::CommandPlan &        plan,
                                                  const char *                    expected_kernel_name) {
    REQUIRE(tensors.weighted != nullptr);
    REQUIRE(tensors.residual != nullptr);
    const size_t             weighted_index = producer_index_for_tensor(graph, tensors.weighted);
    ggml::hrx::DispatchMatch weighted_match;
    REQUIRE(match_dispatch_at_index(graph, plan, covered_nodes, weighted_index, weighted_match));
    append_match_to_plan(plan, weighted_match, covered_nodes, &graph);

    const ggml::hrx::Value * route_weights_value = graph.values().find_tensor(tensors.route_weights);
    const ggml::hrx::Value * routed_output_value = graph.values().find_tensor(tensors.output);
    const ggml::hrx::Value * hidden_state_value  = graph.values().find_tensor(tensors.hidden_state);
    const ggml::hrx::Value * residual_value      = graph.values().find_tensor(tensors.residual);
    REQUIRE(route_weights_value != nullptr);
    REQUIRE(routed_output_value != nullptr);
    REQUIRE(hidden_state_value != nullptr);
    REQUIRE(residual_value != nullptr);

    const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
    REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == expected_kernel_name);
    REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == tensors.output->ne[2]);
    REQUIRE(dispatch.bindings.size() == (tensors.next_output != nullptr ? 5 : 4));
    REQUIRE(dispatch.bindings[0].value == route_weights_value->id);
    REQUIRE(dispatch.bindings[0].length == route_weights_value->byte_count);
    REQUIRE(dispatch.bindings[1].value == plan.metadata.alternate_values().back().alternate_value);
    REQUIRE(dispatch.bindings[1].length == qwen_routed_down_f16_output_size(tensors.output->ne[2]));
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.route_count", "8");
    require_compile_parameter(dispatch, "qwen3_moe.routed_down.output_size", "2048");
    require_compile_parameter(dispatch, "qwen3_moe.workload.token_capacity", std::to_string(tensors.output->ne[2]));

    if (tensors.next_output != nullptr) {
        REQUIRE(dispatch.bindings[2].value == residual_value->id);
        REQUIRE(dispatch.bindings[2].length == residual_value->byte_count);
        const ggml::hrx::Value * next_output_value = graph.values().find_tensor(tensors.next_output);
        REQUIRE(next_output_value != nullptr);
        REQUIRE(dispatch.bindings[4].value == next_output_value->id);
        REQUIRE(dispatch.bindings[4].length == next_output_value->byte_count);
        require_compile_parameter(dispatch, "qwen3_moe.model.hidden_size", "2048");
        require_compile_parameter(dispatch, "qwen3_moe.model.rms_epsilon", "0.000001");
    } else {
        REQUIRE(dispatch.bindings[2].value == hidden_state_value->id);
        REQUIRE(dispatch.bindings[2].length == hidden_state_value->byte_count);
        REQUIRE(dispatch.bindings[3].value == residual_value->id);
        REQUIRE(dispatch.bindings[3].length == residual_value->byte_count);
    }
}

static void run_qwen_routed_gate_up_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 4 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        constexpr int64_t            token_count = 4;
        QwenRoutedGateUpTensors      tensors     = build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        ggml::hrx::DispatchMatch     gate_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.gate, tensors.input, tensors.gate_weight, tensors.route_ids);
        append_match_to_plan(plan, gate_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 2);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings.size() == 5);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[1].name == "expert_table");
        REQUIRE(commands.commands[3].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[3].name == "weight");
        REQUIRE(commands.commands[3].bindings[4].name == "output");
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        ggml::hrx::DispatchMatch      down_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.output, tensors.glu, tensors.down_weight, tensors.route_ids);
        append_match_to_plan(plan, down_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 2);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[3].name == "weight");
        REQUIRE(commands.commands[3].bindings[4].name == "output");
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_BF16);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        ggml::hrx::DispatchMatch     down_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.output, tensors.glu, tensors.down_weight, tensors.route_ids);
        append_match_to_plan(plan, down_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
    }

    {
        constexpr int64_t             token_count = 4;
        QwenRoutedGateUpTensors tensors = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);

        const ggml::hrx::Value * route_ids_value     = imported.graph.values().find_tensor(tensors.route_ids);
        const ggml::hrx::Value * route_weights_value = imported.graph.values().find_tensor(tensors.route_weights);
        const ggml::hrx::Value * glu_value           = imported.graph.values().find_tensor(tensors.glu);
        REQUIRE(route_ids_value != nullptr);
        REQUIRE(route_weights_value != nullptr);
        REQUIRE(glu_value != nullptr);
        REQUIRE(route_ids_value->kind == ggml::hrx::ValueKind::Transient);
        REQUIRE(route_weights_value->kind == ggml::hrx::ValueKind::Transient);
        REQUIRE(glu_value->kind == ggml::hrx::ValueKind::Transient);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const ggml::hrx::CommandPlanGeneratedResource * expert_table_resource = plan.metadata.find_generated_resource(
            route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * partition_table_resource =
            plan.metadata.find_generated_resource(route_ids_value->id,
                                                  ggml::hrx::GeneratedResourceRole::MoePartitionTable);
        const ggml::hrx::CommandPlanMoeRoutingBundle * routing_bundle =
            plan.metadata.find_moe_routing_bundle(route_ids_value->id);
        REQUIRE(expert_table_resource != nullptr);
        REQUIRE(partition_table_resource != nullptr);
        REQUIRE(routing_bundle != nullptr);
        REQUIRE(routing_bundle->route_ids == route_ids_value->id);
        REQUIRE(routing_bundle->route_weights == route_weights_value->id);
        REQUIRE(routing_bundle->expert_table == expert_table_resource->generated_value);
        REQUIRE(routing_bundle->partition_table == partition_table_resource->generated_value);
        REQUIRE(routing_bundle->expert_table_byte_count == qwen_expert_table_size(token_count));
        REQUIRE(routing_bundle->partition_table_byte_count == qwen_partition_table_size(token_count));
        REQUIRE(routing_bundle->route_count == kQwenRouterRouteCount);
        REQUIRE(routing_bundle->expert_count == kQwenRouterExpertCount);
        ggml::hrx::MoeRoutingResourceMetadata expert_metadata;
        ggml::hrx::MoeRoutingResourceMetadata partition_metadata;
        REQUIRE(expert_table_resource->metadata.read(expert_metadata));
        REQUIRE(partition_table_resource->metadata.read(partition_metadata));
        REQUIRE(expert_metadata.token_count == token_count);
        REQUIRE(expert_metadata.route_count == kQwenRouterRouteCount);
        REQUIRE(expert_metadata.expert_count == kQwenRouterExpertCount);
        REQUIRE(partition_metadata.route_stride == expert_metadata.route_stride);
        REQUIRE(expert_table_resource->byte_count == qwen_expert_table_size(token_count));
        REQUIRE(partition_table_resource->byte_count == qwen_partition_table_size(token_count));

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 3);
        REQUIRE(plan.metadata.alternate_values().size() == 1);
        const ggml::hrx::CommandPlanTransient & f16_output_transient = plan.transients.back();
        REQUIRE(f16_output_transient.name == "qwen.moe.gate_up_swiglu_f16");
        REQUIRE(f16_output_transient.size == qwen_routed_gate_up_f16_output_size(token_count));
        REQUIRE(plan.metadata.alternate_values().front().graph_value == glu_value->id);
        REQUIRE(plan.metadata.alternate_values().front().alternate_value == f16_output_transient.value);
        REQUIRE(plan.metadata.alternate_values().front().type == GGML_TYPE_F16);
        REQUIRE(plan.metadata.alternate_values().front().byte_count == f16_output_transient.size);

        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) == "loom_libs:ggml_mul_mat_id_swiglu_f16_f16_wmma");
        REQUIRE(dispatch.kernel.integer_parameters.at("token_count") == token_count);
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].length == qwen_expert_table_size(token_count));
        REQUIRE(dispatch.bindings[2].value == routing_bundle->partition_table);
        REQUIRE(dispatch.bindings[2].length == qwen_partition_table_size(token_count));
        REQUIRE(dispatch.bindings[5].value == f16_output_transient.value);
        REQUIRE(dispatch.bindings[5].length == f16_output_transient.size);
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.input_size", "2048");
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.expert_count", "128");
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.route_count", "8");
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.output_size", "768");
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.gate_weight_format", "4");
        require_compile_parameter(dispatch, "ggml.mul_mat_id_swiglu_f16_f16.up_weight_format", "4");

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 4);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands[3].bindings.size() == 6);
        REQUIRE(commands.commands[3].bindings[0].name == "input");
        REQUIRE(commands.commands[3].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[1].name == "expert_table");
        REQUIRE(commands.commands[3].bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[2].name == "partition_table");
        REQUIRE(commands.commands[3].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands[3].bindings[3].name == "gate_weight");
        REQUIRE(commands.commands[3].bindings[3].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[4].name == "up_weight");
        REQUIRE(commands.commands[3].bindings[4].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
        REQUIRE(commands.commands[3].bindings[5].name == "output");
        REQUIRE(commands.commands[3].bindings[5].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, f16_output_transient.value) != nullptr);
    }

    {
        constexpr int64_t             token_count = 1;
        QwenRoutedGateUpTensors tensors = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 4);
        REQUIRE(plan.transients.size() == 3);
        REQUIRE(kernel_name_for_id(plan.dispatches.back().kernel.kernel_id) ==
                "loom_libs:ggml_mul_mat_id_swiglu_f16_f16_wmma");
        REQUIRE(plan.dispatches.back().kernel.integer_parameters.at("token_count") == 1);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t token_count     = 4;
        ggml_tensor *     first_logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, token_count);
        ggml_tensor *     first_route_ids = nullptr;
        REQUIRE(first_logits != nullptr);
        ggml_tensor * first_route_weights = build_qwen_router_top8_graph(ctx, first_logits, &first_route_ids);
        REQUIRE(first_route_ids != nullptr);
        REQUIRE(first_route_weights != nullptr);

        QwenRoutedGateUpTensors tensors = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml_cgraph *                 graph   = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, first_route_weights);
        ggml_build_forward_expand(graph, tensors.route_weights);
        ggml_build_forward_expand(graph, tensors.residual);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());

        const ggml::hrx::Value * first_route_ids_value  = imported.graph.values().find_tensor(first_route_ids);
        const ggml::hrx::Value * second_route_ids_value = imported.graph.values().find_tensor(tensors.route_ids);
        REQUIRE(first_route_ids_value != nullptr);
        REQUIRE(second_route_ids_value != nullptr);
        REQUIRE(first_route_ids_value->id != second_route_ids_value->id);

        std::vector<bool>      covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan plan;
        size_t                 router_matches = 0;
        for (size_t i = 0; i < imported.graph.nodes().size(); ++i) {
            if (imported.graph.nodes()[i].op != GGML_OP_SOFT_MAX) {
                continue;
            }
            ggml::hrx::DispatchMatch router_match;
            REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, i, router_match));
            append_match_to_plan(plan, router_match, covered_nodes);
            ++router_matches;
        }
        REQUIRE(router_matches == 2);
        REQUIRE(plan.metadata.generated_resources().size() == 4);
        REQUIRE(plan.metadata.moe_routing_bundles().size() == 2);

        const ggml::hrx::CommandPlanGeneratedResource * first_expert_table = plan.metadata.find_generated_resource(
            first_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_expert_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoeExpertTable);
        const ggml::hrx::CommandPlanGeneratedResource * second_partition_table = plan.metadata.find_generated_resource(
            second_route_ids_value->id, ggml::hrx::GeneratedResourceRole::MoePartitionTable);
        const ggml::hrx::CommandPlanMoeRoutingBundle * second_routing_bundle =
            plan.metadata.find_moe_routing_bundle(second_route_ids_value->id);
        REQUIRE(first_expert_table != nullptr);
        REQUIRE(second_expert_table != nullptr);
        REQUIRE(second_partition_table != nullptr);
        REQUIRE(second_routing_bundle != nullptr);
        REQUIRE(second_routing_bundle->expert_table == second_expert_table->generated_value);
        REQUIRE(second_routing_bundle->partition_table == second_partition_table->generated_value);
        REQUIRE(first_expert_table->generated_value != second_expert_table->generated_value);

        const size_t             gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        append_match_to_plan(plan, gate_up_match, covered_nodes);

        REQUIRE(plan.dispatches.size() == 7);
        REQUIRE(plan.transients.size() == 5);
        const ggml::hrx::Dispatch & dispatch = plan.dispatches.back();
        REQUIRE(dispatch.bindings.size() == 6);
        REQUIRE(dispatch.bindings[1].value == second_routing_bundle->expert_table);
        REQUIRE(dispatch.bindings[1].value != first_expert_table->generated_value);
        REQUIRE(dispatch.bindings[2].value == second_routing_bundle->partition_table);

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 7);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t             token_count = 4;
        const QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml::hrx::GraphImportResult  imported    = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        REQUIRE(kernel_name_for_id(plan.dispatches.back().kernel.kernel_id) ==
                common_mul_mat_id_swiglu_f32_kernel_name(token_count));
        ggml::hrx::DispatchMatch down_match = require_common_mul_mat_id_for_graph(
            imported.graph, plan, covered_nodes, tensors.output, tensors.glu, tensors.down_weight, tensors.route_ids);
        append_match_to_plan(plan, down_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 2);
        REQUIRE(plan.metadata.alternate_values().empty());
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 4);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "residual_input");
        REQUIRE(commands.commands.back().bindings[2].access == ggml::hrx::ResourceAccess::Read);
        REQUIRE(commands.commands.back().bindings[3].name == "output");
        REQUIRE(commands.commands.back().bindings[3].access == ggml::hrx::ResourceAccess::Write);
    }

    {
        constexpr int64_t       token_count = 1;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.dispatches.back().kernel.integer_parameters.at("token_count") == 1);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        ggml_tensor * glu_reshape = ggml_reshape_4d(ctx, tensors.glu, tensors.glu->ne[0], tensors.glu->ne[1],
                                                    tensors.glu->ne[2], tensors.glu->ne[3]);
        REQUIRE(glu_reshape != nullptr);
        tensors.output = ggml_mul_mat_id(ctx, tensors.down_weight, glu_reshape, tensors.route_ids);
        REQUIRE(tensors.output != nullptr);
        ggml_tensor * routed_output_view =
            ggml_view_4d(ctx, tensors.output, tensors.output->ne[0], tensors.output->ne[1], tensors.output->ne[2],
                         tensors.output->ne[3], tensors.output->nb[1], tensors.output->nb[2], tensors.output->nb[3], 0);
        REQUIRE(routed_output_view != nullptr);
        append_qwen_weighted_reduce_tail(ctx, tensors, false, routed_output_view);

        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        const ggml::hrx::Value *     glu_reshape_value = imported.graph.values().find_tensor(glu_reshape);
        const ggml::hrx::Value *     routed_output_view_value = imported.graph.values().find_tensor(routed_output_view);
        REQUIRE(glu_reshape_value != nullptr);
        REQUIRE(routed_output_view_value != nullptr);
        const ggml::hrx::GraphNode * glu_reshape_node = imported.graph.index().producer(glu_reshape_value->id);
        const ggml::hrx::GraphNode * routed_output_view_node =
            imported.graph.index().producer(routed_output_view_value->id);
        REQUIRE(glu_reshape_node != nullptr);
        REQUIRE(routed_output_view_node != nullptr);
        REQUIRE(glu_reshape_node->op == GGML_OP_RESHAPE);
        REQUIRE(routed_output_view_node->op == GGML_OP_VIEW);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        const ggml::hrx::Value * glu_value = imported.graph.values().find_tensor(tensors.glu);
        REQUIRE(glu_value != nullptr);
        const ggml::hrx::CommandPlanAlternateValue * gate_up_alternate = ggml::hrx::find_alternate_value(
            imported.graph, plan, glu_value->id, GGML_TYPE_F16, qwen_routed_gate_up_f16_output_size(token_count));
        REQUIRE(gate_up_alternate != nullptr);
        const ggml::hrx::ValueId gate_up_alternate_value = gate_up_alternate->alternate_value;

        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        REQUIRE(plan.dispatches.back().bindings[0].value == gate_up_alternate_value);
        const ggml::hrx::Value * routed_output_value = imported.graph.values().find_tensor(tensors.output);
        REQUIRE(routed_output_value != nullptr);
        const ggml::hrx::CommandPlanAlternateValue * routed_down_alternate = ggml::hrx::find_alternate_value(
            imported.graph, plan, routed_output_value->id, GGML_TYPE_F16,
            qwen_routed_down_f16_output_size(token_count));
        REQUIRE(routed_down_alternate != nullptr);
        const ggml::hrx::ValueId routed_down_alternate_value = routed_down_alternate->alternate_value;

        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");
        REQUIRE(plan.dispatches.back().bindings[1].value == routed_down_alternate_value);
        for (const ggml::hrx::Dispatch & dispatch : plan.dispatches) {
            REQUIRE(kernel_name_for_id(dispatch.kernel.kernel_id) != "loom_libs:ggml_copy_f32_f16");
        }

        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(command_program_verifies(commands));
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");
        append_qwen_weighted_reduce_for_graph(imported.graph, tensors, covered_nodes, plan,
                                              "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::Value * hidden_state_value = imported.graph.values().find_tensor(tensors.hidden_state);
        const ggml::hrx::Value * residual_value     = imported.graph.values().find_tensor(tensors.residual);
        REQUIRE(hidden_state_value != nullptr);
        REQUIRE(residual_value != nullptr);
        REQUIRE(residual_value->alias_source == hidden_state_value->id);
        REQUIRE(imported.graph.values().same_storage(hidden_state_value->id, residual_value->id));
        REQUIRE(residual_value->storage_root == hidden_state_value->storage_root);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 6);
        REQUIRE(command_program_verifies(commands));
        REQUIRE(commands.commands.back().bindings.size() == 5);
        REQUIRE(commands.commands.back().bindings[0].name == "route_weights");
        REQUIRE(commands.commands.back().bindings[1].name == "routed_output");
        REQUIRE(commands.commands.back().bindings[1].origin == ggml::hrx::CommandBindingOrigin::Transient);
        REQUIRE(commands.commands.back().bindings[2].name == "hidden_state");
        REQUIRE(commands.commands.back().bindings[2].value == hidden_state_value->storage_root);
        REQUIRE(commands.commands.back().bindings[2].access == ggml::hrx::ResourceAccess::ReadWrite);
        REQUIRE(commands.commands.back().bindings[3].name == "next_norm_weight");
        REQUIRE(commands.commands.back().bindings[4].name == "next_projection_input");
        REQUIRE(commands.commands.back().bindings[4].access == ggml::hrx::ResourceAccess::Write);
    }

    {
        constexpr int64_t       token_count = 4;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml_tensor * hidden_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, token_count);
        REQUIRE(hidden_bias != nullptr);
        tensors.hidden_use = ggml_add(ctx, tensors.hidden_state, hidden_bias);
        REQUIRE(tensors.hidden_use != nullptr);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        const size_t             weighted_index = producer_index_for_tensor(imported.graph, tensors.weighted);
        ggml::hrx::DispatchMatch weighted_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, weighted_index, weighted_match));
        append_match_to_plan(plan, weighted_match, covered_nodes, &imported.graph);

        REQUIRE(plan.dispatches.size() == 6);
        REQUIRE(kernel_name_for_id(plan.dispatches.back().kernel.kernel_id) ==
                "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_f16_f32");
        REQUIRE(weighted_match.value_aliases.empty());
        REQUIRE(plan.dispatches.back().bindings.size() == 4);
        const ggml::hrx::Value * hidden_state_value = imported.graph.values().find_tensor(tensors.hidden_state);
        const ggml::hrx::Value * residual_value     = imported.graph.values().find_tensor(tensors.residual);
        REQUIRE(hidden_state_value != nullptr);
        REQUIRE(residual_value != nullptr);
        REQUIRE(residual_value->alias_source != hidden_state_value->id);
        REQUIRE(!imported.graph.values().same_storage(hidden_state_value->id, residual_value->id));
    }

    {
        constexpr int64_t             token_count = 4;
        QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, token_count, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q4_K);
        append_qwen_weighted_reduce_tail(ctx, tensors);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        REQUIRE(plan.dispatches.size() == 5);
        REQUIRE(plan.transients.size() == 4);
        const ggml::hrx::CommandProgram commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.size() == 5);
        REQUIRE(command_program_verifies(commands));
    }

    {
        const QwenRoutedGateUpTensors tensors    = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported   = import_qwen_routed_gate_up_graph(ctx, tensors);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        const ggml::hrx::CommandPlan  empty_plan;
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, empty_plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 3);
        REQUIRE(gate_up_match.transients.size() == 2);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_table");
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[1].kernel.kernel_id) ==
                "loom_libs:ggml_moe_build_expert_partition_table");
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[2].kernel.kernel_id) ==
                common_mul_mat_id_swiglu_f32_kernel_name(tensors.output->ne[2]));
        REQUIRE(gate_up_match.dispatches[2].bindings.size() == 6);
        REQUIRE(gate_up_match.dispatches[2].bindings[1].value == gate_up_match.transients[0].value);
        REQUIRE(gate_up_match.dispatches[2].bindings[2].value == gate_up_match.transients[1].value);
        REQUIRE(gate_up_match.metadata.generated_resources().size() == 2);
        REQUIRE(gate_up_match.metadata.moe_routing_bundles().size() == 1);
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_GEGLU);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch      gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                common_mul_mat_id_f32_kernel_name(tensors.gate->ne[2]));
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q6_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                common_mul_mat_id_swiglu_f32_kernel_name(tensors.output->ne[2]));
        REQUIRE(gate_up_match.dispatches[0].bindings.size() == 6);
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, false);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                 gate_index = producer_index_for_tensor(imported.graph, tensors.gate);
        ggml::hrx::DispatchMatch     gate_up_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, gate_index, gate_up_match));
        REQUIRE(gate_up_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(gate_up_match.dispatches[0].kernel.kernel_id) ==
                common_mul_mat_id_swiglu_f32_kernel_name(tensors.output->ne[2]));
        REQUIRE(gate_up_match.dispatches[0].bindings.size() == 6);
    }

    {
        const QwenRoutedGateUpTensors tensors  = build_qwen_routed_gate_up_graph(ctx, 4);
        ggml::hrx::GraphImportResult  imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>             covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan        plan       = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        const size_t                  down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch      down_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
        REQUIRE(down_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(down_match.dispatches[0].kernel.kernel_id) ==
                common_mul_mat_id_f32_kernel_name(tensors.output->ne[2]));
    }

    {
        const QwenRoutedGateUpTensors tensors =
            build_qwen_routed_gate_up_graph(ctx, 4, GGML_GLU_OP_SWIGLU, GGML_TYPE_Q4_K, true, GGML_TYPE_Q5_K);
        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        const size_t             down_index = producer_index_for_tensor(imported.graph, tensors.output);
        ggml::hrx::DispatchMatch down_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, down_index, down_match));
        REQUIRE(down_match.dispatches.size() == 1);
        REQUIRE(kernel_name_for_id(down_match.dispatches[0].kernel.kernel_id) ==
                common_mul_mat_id_f32_kernel_name(tensors.output->ne[2]));
    }

    ggml_free(ctx);
}

static void run_qwen_weighted_reduce_next_rmsnorm_q8_publication_checks() {
    enum class ConsumerCase {
        Direct,
        Alias,
        UnsupportedWeight,
        WrongOperand,
        IncompatibleGeometry,
    };
    const struct {
        ConsumerCase consumer;
        bool         publishes_q8;
    } cases[] = {
        { ConsumerCase::Direct,               true  },
        { ConsumerCase::Alias,                true  },
        { ConsumerCase::UnsupportedWeight,    false },
        { ConsumerCase::WrongOperand,         false },
        { ConsumerCase::IncompatibleGeometry, false },
    };

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 128 * 1024 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        constexpr int64_t       token_count = 2;
        QwenRoutedGateUpTensors tensors     = build_qwen_routed_gate_up_graph(ctx, token_count);
        append_qwen_weighted_reduce_tail(ctx, tensors, true);
        ggml_tensor * consumer_input = tensors.next_output;
        if (test.consumer == ConsumerCase::Alias) {
            consumer_input = ggml_reshape_2d(ctx, tensors.next_output, kQwenMoeHiddenSize, token_count);
            REQUIRE(consumer_input != nullptr);
        }

        if (test.consumer == ConsumerCase::WrongOperand) {
            ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenMoeHiddenSize, 64);
            REQUIRE(activation != nullptr);
            tensors.hidden_use = ggml_mul_mat(ctx, tensors.next_output, activation);
        } else {
            const ggml_type weight_type =
                test.consumer == ConsumerCase::UnsupportedWeight ? GGML_TYPE_BF16 : GGML_TYPE_Q4_K;
            const int64_t output_size = test.consumer == ConsumerCase::IncompatibleGeometry ? 63 : 512;
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, kQwenMoeHiddenSize, output_size);
            REQUIRE(weight != nullptr);
            tensors.hidden_use = ggml_mul_mat(ctx, weight, consumer_input);
        }
        REQUIRE(tensors.hidden_use != nullptr);

        ggml::hrx::GraphImportResult imported = import_qwen_routed_gate_up_graph(ctx, tensors);
        std::vector<bool>            covered_nodes(imported.graph.nodes().size(), false);
        ggml::hrx::CommandPlan       plan = build_qwen_router_plan_for_graph(imported.graph, covered_nodes);
        append_qwen_routed_gate_up_for_graph(imported.graph, tensors, covered_nodes, plan);
        append_qwen_routed_down_for_graph(imported.graph, tensors, covered_nodes, plan,
                                          "loom_libs:ggml_mul_mat_id_f16_f16_wmma");

        const size_t             weighted_index = producer_index_for_tensor(imported.graph, tensors.weighted);
        ggml::hrx::DispatchMatch weighted_match;
        REQUIRE(match_dispatch_at_index(imported.graph, plan, covered_nodes, weighted_index, weighted_match));
        const char * expected_kernel =
            test.publishes_q8 ?
                "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32_publish_q8" :
                "qwen3_moe:qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32";
        REQUIRE(kernel_name_for_id(weighted_match.dispatches.back().kernel.kernel_id) == expected_kernel);
        REQUIRE(weighted_match.dispatches.back().bindings.size() == (test.publishes_q8 ? 6 : 5));

        const ggml::hrx::Value * next_output_value = imported.graph.values().find_tensor(tensors.next_output);
        REQUIRE(next_output_value != nullptr);
        const ggml::hrx::CommandPlanAlternateValue * q8 = weighted_match.metadata.find_alternate_value(
            next_output_value->id, GGML_TYPE_Q8_1, qwen_q8_1_x4_size(token_count, kQwenMoeHiddenSize));
        REQUIRE((q8 != nullptr) == test.publishes_q8);
        if (test.publishes_q8) {
            REQUIRE(weighted_match.metadata.activation_publication_diagnostics().size() == 1);
            const auto & diagnostic = weighted_match.metadata.activation_publication_diagnostics().front();
            REQUIRE(diagnostic.source_value == next_output_value->id);
            REQUIRE(diagnostic.alternate_value == q8->alternate_value);
            REQUIRE(diagnostic.publication_stage == "published");
            REQUIRE(diagnostic.requested_format == "q8-1-x4");
        }
        ggml_free(ctx);
    }
}

static void run_qwen_router_top8_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 2 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 1);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 13);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids, GGML_SORT_ORDER_DESC,
                                                            kQwenRouterRouteCount, 0.00006103515625f);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 512);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids);
        schedule_qwen_router_top8_command(ctx, output, route_ids, 64, kQwenRouterRouteCount);
    }
    {
        ggml_tensor * logits    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        ggml_tensor * route_ids = nullptr;
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, &route_ids, GGML_SORT_ORDER_DESC, 4);
        schedule_qwen_router_top8_command(ctx, output, route_ids, kQwenRouterExpertCount, 4);
    }
    {
        ManualQwenRouterTop8Graph manual = build_manual_qwen_router_top8_graph(ctx, 64, 4, 5);
        schedule_manual_qwen_router_top8_command(manual.graph, manual.route_ids, 5, 64, 4);
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_ASC);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits, nullptr, GGML_SORT_ORDER_DESC, 33);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }
    {
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kQwenRouterExpertCount, 4);
        REQUIRE(logits != nullptr);
        ggml_tensor * output = build_qwen_router_top8_graph(ctx, logits);
        REQUIRE(!graph_is_supported(ctx, output));
    }

    ggml_free(ctx);
}

static void bind_external_values(ggml::hrx::ValueMap & values) {
    uintptr_t buffer = 0x1000;
    for (const ggml::hrx::ValueId id : values.external_value_ids()) {
        const ggml::hrx::Value * value = values.find(id);
        REQUIRE(value != nullptr);
        REQUIRE(values.bind_buffer(id, { dummy_hrx_buffer(buffer), 0, value->byte_count }));
        buffer += 0x1000;
    }
}

static void run_alias_value_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * view   = ggml_view_1d(ctx, source, 4, 2 * sizeof(float));
    REQUIRE(source != nullptr);
    REQUIRE(view != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, view);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 1);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_VIEW);

    const ggml::hrx::Value * source_value = imported.graph.values().find_tensor(source);
    const ggml::hrx::Value * view_value   = imported.graph.values().find_tensor(view);
    REQUIRE(source_value != nullptr);
    REQUIRE(view_value != nullptr);
    REQUIRE(source_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported.graph.values().same_storage(source_value->id, view_value->id));
    REQUIRE(view_value->storage_root == source_value->id);
    REQUIRE(view_value->alias_source == source_value->id);
    REQUIRE(view_value->storage_offset == 2 * sizeof(float));
    REQUIRE(view_value->byte_count == 4 * sizeof(float));
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, imported.graph.nodes()[0]));

    REQUIRE(imported.graph.values().bind_buffer(source_value->id,
                                                { dummy_hrx_buffer(0x4000), 128, source_value->byte_count }));
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    const ggml::hrx::CommandProgramBinding * view_binding = bindings.find(view_value->id);
    REQUIRE(view_binding != nullptr);
    REQUIRE(view_binding->buffer == dummy_hrx_buffer(0x4000));
    REQUIRE(view_binding->offset == 128 + 2 * sizeof(float));
    REQUIRE(view_binding->length == view_value->byte_count);

    ggml_tensor * internal_source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * internal_bias   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    ggml_tensor * internal_view =
        ggml_view_2d(ctx, internal_source, 8, 1, internal_source->nb[1], internal_source->nb[1]);
    ggml_tensor * internal_out = ggml_add(ctx, internal_view, internal_bias);
    REQUIRE(internal_source != nullptr);
    REQUIRE(internal_bias != nullptr);
    REQUIRE(internal_view != nullptr);
    REQUIRE(internal_out != nullptr);

    ggml_cgraph * internal_graph = ggml_new_graph(ctx);
    REQUIRE(internal_graph != nullptr);
    ggml_build_forward_expand(internal_graph, internal_out);

    ggml::hrx::GraphImportResult internal_imported = ggml::hrx::import_ggml_graph(*internal_graph);
    REQUIRE(internal_imported.valid());
    REQUIRE(internal_imported.graph.nodes().size() == 2);
    REQUIRE(internal_imported.graph.nodes()[0].op == GGML_OP_VIEW);
    REQUIRE(internal_imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * internal_source_value = internal_imported.graph.values().find_tensor(internal_source);
    const ggml::hrx::Value * internal_view_value   = internal_imported.graph.values().find_tensor(internal_view);
    const ggml::hrx::Value * internal_bias_value   = internal_imported.graph.values().find_tensor(internal_bias);
    const ggml::hrx::Value * internal_out_value    = internal_imported.graph.values().find_tensor(internal_out);
    REQUIRE(internal_source_value != nullptr);
    REQUIRE(internal_view_value != nullptr);
    REQUIRE(internal_bias_value != nullptr);
    REQUIRE(internal_out_value != nullptr);
    REQUIRE(internal_source_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(internal_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(internal_view_value->storage_root == internal_source_value->id);
    REQUIRE(internal_view_value->storage_offset == internal_source->nb[1]);

    ggml::hrx::DispatchScheduler internal_scheduler;
    REQUIRE(internal_scheduler.schedule_graph(internal_imported.graph, test_dispatch_target()));

    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_source_value->id, { dummy_hrx_buffer(0x5000), 256, internal_source_value->byte_count }));
    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_bias_value->id, { dummy_hrx_buffer(0x6000), 0, internal_bias_value->byte_count }));
    REQUIRE(internal_imported.graph.values().bind_buffer(
        internal_out_value->id, { dummy_hrx_buffer(0x7000), 0, internal_out_value->byte_count }));
    const ggml::hrx::CommandProgramBindings internal_bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(internal_imported.graph.values());
    REQUIRE(internal_bindings.valid());
    const ggml::hrx::CommandProgramBinding * internal_view_binding = internal_bindings.find(internal_view_value->id);
    REQUIRE(internal_view_binding != nullptr);
    REQUIRE(internal_view_binding->buffer == dummy_hrx_buffer(0x5000));
    REQUIRE(internal_view_binding->offset == 256 + internal_source->nb[1]);
    REQUIRE(internal_view_binding->length == internal_view_value->byte_count);

    ggml_tensor * cache       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * rows        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * row_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * updated     = ggml_set_rows(ctx, cache, rows, row_indices);
    REQUIRE(cache != nullptr);
    REQUIRE(rows != nullptr);
    REQUIRE(row_indices != nullptr);
    REQUIRE(updated != nullptr);
    ggml_tensor * updated_view = ggml_view_2d(ctx, updated, 8, 2, updated->nb[1], 0);
    ggml_tensor * updated_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * updated_out  = ggml_add(ctx, updated_view, updated_bias);
    REQUIRE(updated_view != nullptr);
    REQUIRE(updated_bias != nullptr);
    REQUIRE(updated_out != nullptr);

    ggml_cgraph * set_rows_graph = ggml_new_graph(ctx);
    REQUIRE(set_rows_graph != nullptr);
    ggml_build_forward_expand(set_rows_graph, updated_out);

    ggml::hrx::GraphImportResult set_rows_imported = ggml::hrx::import_ggml_graph(*set_rows_graph);
    REQUIRE(set_rows_imported.valid());
    REQUIRE(set_rows_imported.graph.nodes().size() == 3);
    REQUIRE(set_rows_imported.graph.nodes()[0].op == GGML_OP_SET_ROWS);
    REQUIRE(set_rows_imported.graph.nodes()[1].op == GGML_OP_VIEW);
    REQUIRE(set_rows_imported.graph.nodes()[2].op == GGML_OP_ADD);

    const ggml::hrx::Value * cache_value        = set_rows_imported.graph.values().find_tensor(cache);
    const ggml::hrx::Value * updated_value      = set_rows_imported.graph.values().find_tensor(updated);
    const ggml::hrx::Value * updated_view_value = set_rows_imported.graph.values().find_tensor(updated_view);
    REQUIRE(cache_value != nullptr);
    REQUIRE(updated_value != nullptr);
    REQUIRE(updated_view_value != nullptr);
    REQUIRE(cache_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_view_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(updated_value->storage_root == cache_value->id);
    REQUIRE(updated_view_value->storage_root == cache_value->id);
    REQUIRE(updated_view_value->alias_source == cache_value->id);

    ggml::hrx::ValueMap                      value_map;
    const std::array<int64_t, GGML_MAX_DIMS> ne = { 8, 1, 1, 1 };
    const std::array<size_t, GGML_MAX_DIMS>  nb = { sizeof(float), 8 * sizeof(float), 8 * sizeof(float),
                                                    8 * sizeof(float) };
    REQUIRE(value_map.add_snapshot_storage({ ggml::hrx::ValueStorageId(0), ggml::hrx::ValueId(0), 8 * sizeof(float) })
                .success());
    REQUIRE(
        value_map
            .add_snapshot_value({ ggml::hrx::ValueId(0), ggml::hrx::ValueKind::External, ggml::hrx::ValueStorageId(0),
                                  ggml::hrx::ValueId(0), ggml::hrx::ValueId(), 0, 8 * sizeof(float), GGML_TYPE_F32, ne,
                                  nb, 8, 8 * sizeof(float), true, nullptr, std::nullopt })
            .success());
    REQUIRE(value_map.add_snapshot_storage({ ggml::hrx::ValueStorageId(1), ggml::hrx::ValueId(1), 8 * sizeof(float) })
                .success());
    REQUIRE(
        value_map
            .add_snapshot_value({ ggml::hrx::ValueId(1), ggml::hrx::ValueKind::Transient, ggml::hrx::ValueStorageId(1),
                                  ggml::hrx::ValueId(1), ggml::hrx::ValueId(), 0, 8 * sizeof(float), GGML_TYPE_F32, ne,
                                  nb, 8, 8 * sizeof(float), true, nullptr, std::nullopt })
            .success());
    REQUIRE(value_map.alias_storage(ggml::hrx::ValueId(1), ggml::hrx::ValueId(0)).success());
    const ggml::hrx::Value * aliased_value = value_map.find(ggml::hrx::ValueId(1));
    REQUIRE(aliased_value != nullptr);
    REQUIRE(aliased_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(aliased_value->alias_source == ggml::hrx::ValueId(0));
    REQUIRE(aliased_value->storage_root == ggml::hrx::ValueId(0));
    REQUIRE(value_map.same_storage(ggml::hrx::ValueId(0), ggml::hrx::ValueId(1)));
    REQUIRE(value_map.bind_buffer(ggml::hrx::ValueId(0), { dummy_hrx_buffer(0x8000), 64, 8 * sizeof(float) }));
    const std::optional<ggml::hrx::ValueBufferBinding> aliased_binding =
        value_map.resolve_buffer_binding(ggml::hrx::ValueId(1));
    REQUIRE(aliased_binding.has_value());
    REQUIRE(aliased_binding->buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(aliased_binding->offset == 64);
    REQUIRE(aliased_binding->length == 8 * sizeof(float));

    ggml_free(ctx);
}

static void run_multi_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1001;

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    const ggml::hrx::Value * out0_value = imported.graph.values().find_tensor(out0);
    const ggml::hrx::Value * out1_value = imported.graph.values().find_tensor(out1);
    REQUIRE(out0_value != nullptr);
    REQUIRE(out1_value != nullptr);
    REQUIRE(imported.graph.nodes()[0].output == out0_value->id);
    REQUIRE(imported.graph.nodes()[1].output == out1_value->id);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[0].ordinal == 0);
    REQUIRE(commands.commands[0].dependencies.empty());
    REQUIRE(commands.commands[1].ordinal == 1);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(command_program_verifies(commands));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(resolved.valid());
    REQUIRE(resolved.commands.size() == 2);

    ggml_free(ctx);
}

static void run_layout_alias_scheduler_elision_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum      = ggml_add(ctx, a, b);
    ggml_tensor * reshaped = ggml_reshape_2d(ctx, sum, 4, 2);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(reshaped != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, reshaped);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RESHAPE);

    const ggml::hrx::Value * sum_value      = imported.graph.values().find_tensor(sum);
    const ggml::hrx::Value * reshaped_value = imported.graph.values().find_tensor(reshaped);
    REQUIRE(sum_value != nullptr);
    REQUIRE(reshaped_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(reshaped_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(imported.graph.values().same_storage(sum_value->id, reshaped_value->id));
    REQUIRE(reshaped_value->storage_root == sum_value->id);
    REQUIRE(reshaped_value->alias_source == sum_value->id);
    REQUIRE(ggml::hrx::is_layout_alias_node(imported.graph, imported.graph.nodes()[1]));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 1);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.transients.allocations.empty());
    REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, sum_value->id) == nullptr);
    REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, reshaped_value->id) == nullptr);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_zero_output_scheduler_elision_checks() {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 1);
        ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0);
        REQUIRE(input != nullptr);
        REQUIRE(ids != nullptr);
        ggml_tensor * rows = ggml_get_rows(ctx, input, ids);
        REQUIRE(rows != nullptr);
        REQUIRE(rows->ne[0] == 3840);
        REQUIRE(rows->ne[1] == 0);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, rows);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);
        const ggml::hrx::Value * rows_value = imported.graph.values().find_tensor(rows);
        REQUIRE(rows_value != nullptr);
        REQUIRE(rows_value->element_count == 0);
        REQUIRE(rows_value->byte_count == 0);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.empty());

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        REQUIRE(commands.commands.empty());
        REQUIRE(command_program_verifies(commands));

        ggml::hrx::GraphProgramCache  cache;
        ggml::hrx::GraphProgramLookup lookup =
            cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(lookup.valid());
        REQUIRE(lookup.match.external_bindings.size() == 1);
        REQUIRE(lookup.match.external_bindings[0].tensor == input);

        const auto * ids_value = lookup.program->graph().values().find_tensor(ids);
        REQUIRE(ids_value != nullptr);
        ggml::hrx::Command referenced;
        referenced.bindings.push_back({ "ids", ids_value->id });
        for (auto * list :
             { &lookup.program->commands().initialization_commands, &lookup.program->commands().commands }) {
            list->push_back(referenced);
            const auto match = lookup.program->match_current_graph(*graph);
            REQUIRE(match.valid());
            REQUIRE(match.external_bindings.size() == 2);
            REQUIRE(std::any_of(match.external_bindings.begin(), match.external_bindings.end(),
                                [&](const auto & binding) { return binding.tensor == ids; }));
            list->clear();
            const auto unreferenced = lookup.program->match_current_graph(*graph);
            REQUIRE(unreferenced.valid());
            REQUIRE(unreferenced.external_bindings.size() == 1);
            for (const auto origin :
                 { ggml::hrx::CommandBindingOrigin::Transient, ggml::hrx::CommandBindingOrigin::ProgramConstant }) {
                ggml::hrx::Command unrelated;
                unrelated.bindings.push_back({ "negative", ggml::hrx::ValueId(-1) });
                unrelated.bindings.push_back({ "large", ggml::hrx::ValueId(INT32_MAX) });
                unrelated.bindings.push_back({ "ids", ids_value->id, origin });
                list->push_back(unrelated);
                const auto ignored = lookup.program->match_current_graph(*graph);
                REQUIRE(ignored.valid());
                REQUIRE(ignored.external_bindings.size() == 1);
                REQUIRE(ignored.external_bindings[0].tensor == input);
                list->clear();
            }
        }
    }

    {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 1);
        ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0);
        REQUIRE(input != nullptr);
        REQUIRE(ids != nullptr);
        ggml_tensor * rows = ggml_get_rows(ctx, input, ids);
        ggml_tensor * norm = ggml_rms_norm(ctx, rows, 0.000001f);
        REQUIRE(rows != nullptr);
        REQUIRE(norm != nullptr);
        REQUIRE(norm->ne[0] == 3840);
        REQUIRE(norm->ne[1] == 0);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, norm);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 2);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_GET_ROWS);
        REQUIRE(imported.graph.nodes()[1].op == GGML_OP_RMS_NORM);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.empty());

        ggml::hrx::GraphProgramCache  cache;
        ggml::hrx::GraphProgramLookup lookup =
            cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(lookup.valid());
        REQUIRE(lookup.match.external_bindings.size() == 1);
        REQUIRE(lookup.match.external_bindings[0].tensor == input);
    }

    {
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 0);
        ggml_tensor * rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 0);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        ggml_tensor * sum = ggml_add(ctx, lhs, rhs);
        REQUIRE(sum != nullptr);
        REQUIRE(sum->ne[0] == 3840);
        REQUIRE(sum->ne[1] == 0);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, sum);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.empty());

        ggml::hrx::GraphProgramCache  cache;
        ggml::hrx::GraphProgramLookup lookup =
            cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(lookup.valid());
        REQUIRE(lookup.match.external_bindings.empty());
    }

    {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 0);
        REQUIRE(input != nullptr);
        ggml_tensor * scaled = ggml_scale(ctx, input, 2.0f);
        REQUIRE(scaled != nullptr);
        REQUIRE(scaled->ne[0] == 3840);
        REQUIRE(scaled->ne[1] == 0);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, scaled);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        REQUIRE(imported.graph.nodes().size() == 1);
        REQUIRE(imported.graph.nodes()[0].op == GGML_OP_SCALE);

        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        REQUIRE(scheduler.plan().valid());
        REQUIRE(scheduler.plan().dispatches.empty());

        ggml::hrx::GraphProgramCache  cache;
        ggml::hrx::GraphProgramLookup lookup =
            cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(lookup.valid());
        REQUIRE(lookup.match.external_bindings.empty());
    }

    {
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 1);
        REQUIRE(input != nullptr);
        ggml_tensor * sin = ggml_sin(ctx, input);
        REQUIRE(sin != nullptr);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        REQUIRE(graph != nullptr);
        ggml_build_forward_expand(graph, sin);

        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    }

    ggml_free(ctx);
}

static void run_zero_output_device_support_checks() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 0);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3840, 0);
    REQUIRE(lhs != nullptr);
    REQUIRE(rhs != nullptr);

    ggml_tensor * sum    = ggml_add(ctx, lhs, rhs);
    ggml_tensor * prod   = ggml_mul(ctx, lhs, rhs);
    ggml_tensor * scaled = ggml_scale(ctx, lhs, 2.0f);
    REQUIRE(sum != nullptr);
    REQUIRE(prod != nullptr);
    REQUIRE(scaled != nullptr);
    REQUIRE(ggml_backend_supports_op(backend, sum));
    REQUIRE(ggml_backend_supports_op(backend, prod));
    REQUIRE(ggml_backend_supports_op(backend, scaled));

    ggml_tensor * acc = ggml_acc(ctx, lhs, rhs, lhs->nb[1], lhs->nb[2], lhs->nb[3], 0);
    ggml_tensor * set = ggml_set(ctx, lhs, rhs, lhs->nb[1], lhs->nb[2], lhs->nb[3], 0);
    ggml_tensor * cpy = ggml_cpy(ctx, lhs, rhs);
    REQUIRE(acc != nullptr);
    REQUIRE(set != nullptr);
    REQUIRE(cpy != nullptr);
    REQUIRE(!ggml_backend_supports_op(backend, acc));
    REQUIRE(!ggml_backend_supports_op(backend, set));
    REQUIRE(ggml_backend_supports_op(backend, cpy));

    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_scale_f32_device_support_checks() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * input     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_tensor * out       = ggml_scale_bias(ctx, input, 1.5f, 0.5f);
    ggml_tensor * inplace   = ggml_scale_bias_inplace(ctx, input, -2.0f, 1.0f);
    ggml_tensor * view      = ggml_view_1d(ctx, input, 16, 8 * sizeof(float));
    ggml_tensor * view_out  = ggml_scale_bias(ctx, view, 0.25f, -0.5f);
    ggml_tensor * view_inpl = ggml_scale_bias_inplace(ctx, view, 0.5f, 0.25f);
    REQUIRE(input != nullptr);
    REQUIRE(out != nullptr);
    REQUIRE(inplace != nullptr);
    REQUIRE(view != nullptr);
    REQUIRE(view_out != nullptr);
    REQUIRE(view_inpl != nullptr);

    REQUIRE(ggml_backend_supports_op(backend, out));
    REQUIRE(ggml_backend_supports_op(backend, inplace));
    REQUIRE(ggml_backend_supports_op(backend, view_out));
    REQUIRE(ggml_backend_supports_op(backend, view_inpl));

    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_transient_import_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_sin(ctx, sum);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_SIN);

    const ggml::hrx::Value * a_value   = imported.graph.values().find_tensor(a);
    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    const ggml::hrx::Value * out_value = imported.graph.values().find_tensor(out);
    REQUIRE(a_value != nullptr);
    REQUIRE(sum_value != nullptr);
    REQUIRE(out_value != nullptr);
    REQUIRE(a_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(out_value->kind == ggml::hrx::ValueKind::External);
    REQUIRE(
        !imported.graph.values().bind_buffer(sum_value->id, { dummy_hrx_buffer(0x3000), 0, sum_value->byte_count }));

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(!scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.empty());
    REQUIRE(status_contains(scheduler.plan().status, "unsupported HRX node 1"));

    ggml_free(ctx);
}

static void run_graph_view_preserves_shared_output_storage_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum  = ggml_add(ctx, a, b);
    ggml_tensor * out0 = ggml_add(ctx, sum, c);
    ggml_tensor * out1 = ggml_add(ctx, sum, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    REQUIRE(graph->n_nodes == 3);
    REQUIRE(graph->nodes[0] == sum);
    REQUIRE(graph->nodes[1] == out0);
    REQUIRE(graph->nodes[2] == out1);
    REQUIRE(ggml_node_get_use_count(graph, 0) == 2);

    ggml_cgraph                  graph_view = ggml_graph_view(graph, 0, 2);
    ggml::hrx::GraphImportResult imported   = ggml::hrx::import_ggml_graph(graph_view);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    REQUIRE(sum_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::External);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[1].bindings[0].value == sum_value->id);
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(ggml::hrx::find_transient_allocation(commands.transients, sum_value->id) == nullptr);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_chained_dispatch_requires_transients() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 2);
    REQUIRE(imported.graph.nodes()[0].op == GGML_OP_ADD);
    REQUIRE(imported.graph.nodes()[1].op == GGML_OP_ADD);

    const ggml::hrx::Value * sum_value = imported.graph.values().find_tensor(sum);
    REQUIRE(sum_value != nullptr);
    REQUIRE(sum_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 2);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 2);
    REQUIRE(commands.commands[1].dependencies.size() == 1);
    REQUIRE(commands.commands[1].dependencies[0] == 0);
    REQUIRE(commands.commands[0].bindings.size() == 3);
    REQUIRE(commands.commands[1].bindings.size() == 3);
    REQUIRE(commands.commands[0].bindings[0].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[0].bindings[2].value == sum_value->id);
    REQUIRE(commands.commands[0].bindings[2].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[0].value == sum_value->id);
    REQUIRE(commands.commands[1].bindings[0].origin == ggml::hrx::CommandBindingOrigin::Transient);
    REQUIRE(commands.commands[1].bindings[1].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.commands[1].bindings[2].origin == ggml::hrx::CommandBindingOrigin::GraphValue);
    REQUIRE(commands.transients.allocations.size() == 1);
    const ggml::hrx::TransientAllocation * sum_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum_value->id);
    REQUIRE(sum_allocation != nullptr);
    REQUIRE(sum_allocation->value == sum_value->id);
    REQUIRE(sum_allocation->size == sum_value->byte_count);
    REQUIRE(sum_allocation->alignment == 256);
    REQUIRE(sum_allocation->arena_offset == 0);
    REQUIRE(commands.transients.arena_size == 256);
    REQUIRE(command_program_verifies(commands));

    const std::string transient_binding_text = ggml::hrx::format_command_binding(commands.commands[1].bindings[0]);
    REQUIRE(string_contains(transient_binding_text, "origin=Transient"));

    bind_external_values(imported.graph.values());
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(imported.graph.values());
    REQUIRE(bindings.valid());
    REQUIRE(bindings.find(sum_value->id) == nullptr);

    const ggml::hrx::ResolvedCommandProgram resolved = ggml::hrx::resolve_command_program_bindings(commands, bindings);
    REQUIRE(!resolved.valid());
    REQUIRE(status_contains(resolved.status, "no transient arena"));
    REQUIRE(status_contains(resolved.status, "origin=Transient"));
    REQUIRE(status_contains(resolved.status, "value="));

    const ggml::hrx::TransientArenaAllocationRef transient_arena = {
        dummy_hrx_buffer(0x8000),
        commands.transients.arena_size,
        7,
    };
    const ggml::hrx::ResolvedCommandProgram resolved_with_transients =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &transient_arena);
    REQUIRE(resolved_with_transients.valid());
    REQUIRE(resolved_with_transients.commands.size() == 2);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[0].bindings[2].ref.length == sum_value->byte_count);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.offset == 0);
    REQUIRE(resolved_with_transients.commands[1].bindings[0].ref.length == sum_value->byte_count);

    ggml::hrx::PreparedCommandProgram prepared_shape;
    for (const ggml::hrx::Command & prepared_source : commands.commands) {
        ggml::hrx::PreparedCommand prepared_command;
        prepared_command.ordinal               = prepared_source.ordinal;
        prepared_command.kind                  = prepared_source.kind;
        prepared_command.kernel.specialization = prepared_source.kernel;
        for (const ggml::hrx::CommandBinding & binding : prepared_source.bindings) {
            prepared_command.kernel.bindings.push_back({
                binding, { dummy_hrx_buffer(0x4000), 123, binding.length }
            });
        }
        prepared_shape.commands.push_back(prepared_command);
    }
    prepared_shape.bound_transient_arena_allocation_id = 1;

    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.offset == 0);
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x8000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 0);

    const ggml::hrx::TransientArenaAllocationRef grown_transient_arena = {
        dummy_hrx_buffer(0x9000),
        commands.transients.arena_size + 256,
        8,
    };
    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, grown_transient_arena, prepared_shape));
    REQUIRE(prepared_shape.bound_transient_arena_allocation_id == grown_transient_arena.allocation_id);
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0x9000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0x9000));

    prepared_shape.commands[1].kernel.bindings[0].binding.origin = ggml::hrx::CommandBindingOrigin::ProgramConstant;
    prepared_shape.commands[1].kernel.bindings[0].ref = { dummy_hrx_buffer(0xb000), 32, sum_value->byte_count };
    const ggml::hrx::TransientArenaAllocationRef rebinding_transient_arena = {
        dummy_hrx_buffer(0xc000),
        commands.transients.arena_size + 512,
        9,
    };
    REQUIRE(ggml::hrx::bind_prepared_command_program_transients(commands, rebinding_transient_arena, prepared_shape));
    REQUIRE(prepared_shape.commands[0].kernel.bindings[2].ref.buffer == dummy_hrx_buffer(0xc000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.buffer == dummy_hrx_buffer(0xb000));
    REQUIRE(prepared_shape.commands[1].kernel.bindings[0].ref.offset == 32);

    const ggml::hrx::TransientArenaAllocationRef invalid_transient_arena = {
        dummy_hrx_buffer(0xa000),
        commands.transients.arena_size,
        ggml::hrx::kInvalidTransientArenaAllocationId,
    };
    const ggml::hrx::ResolvedCommandProgram invalid_transient_resolved =
        ggml::hrx::resolve_command_program_bindings(commands, bindings, &invalid_transient_arena);
    REQUIRE(!invalid_transient_resolved.valid());
    REQUIRE(status_contains(invalid_transient_resolved.status, "no transient arena allocation id"));

    ggml::hrx::CommandProgram missing_allocation = copy_command_program_shape(commands);
    missing_allocation.transients.allocations.clear();
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(missing_allocation, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "no transient allocation"));

    ggml::hrx::CommandProgram out_of_range      = copy_command_program_shape(commands);
    out_of_range.commands[0].bindings[2].length = sum_value->byte_count + 1;
    verification = ggml::hrx::verify_command_program(out_of_range, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "outside transient allocation length"));

    ggml_free(ctx);
}

static void run_graph_replay_host_staging_is_not_ineligible() {
    ggml::hrx::CommandProgram commands;

    std::vector<uint8_t> source0(64, 1);
    std::vector<uint8_t> source1(64, 2);

    ggml::hrx::PreparedCommandProgram prepared;
    ggml::hrx::HostStagingBuffer      staging;
    staging.buffer    = dummy_hrx_buffer(0x1000);
    staging.host_data = source0.data();
    staging.value     = 7;
    staging.length    = source0.size();
    staging.upload    = true;
    prepared.host_staging.push_back(std::move(staging));

    ggml::hrx::CommandProgramBinding live_binding;
    live_binding.value     = ggml::hrx::ValueId(7);
    live_binding.length    = source1.size();
    live_binding.capacity  = source1.size();
    live_binding.host_data = source1.data();
    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_bindings({ live_binding });
    REQUIRE(bindings.valid());

    ggml::hrx::RecordedCommandGraph recorded;
    recorded.exec                                = dummy_hrx_graph_exec(0x2000);
    recorded.bound_transient_arena_allocation_id = ggml::hrx::kInvalidTransientArenaAllocationId;

    ggml::hrx::CommandProgramExecutionContext context;
    context.stream = dummy_hrx_stream(0x3000);

    const ggml::hrx::RecordedCommandGraphExecutionResult result =
        ggml::hrx::bind_and_launch_recorded_command_graph(context, commands, bindings, prepared, recorded);
    recorded.exec = nullptr;

    REQUIRE(!result.success);
    REQUIRE(result.event == ggml::hrx::HrxGraphReplayEvent::LaunchFailed);
    REQUIRE(result.ineligible_reason.empty());
    REQUIRE(status_contains(result.status, "missing HRX host transfer manager"));
    REQUIRE(prepared.host_staging.size() == 1);
    REQUIRE(prepared.host_staging[0].buffer == dummy_hrx_buffer(0x1000));
    REQUIRE(prepared.host_staging[0].host_data == source1.data());
    prepared.host_staging[0].buffer = nullptr;
}

static void run_multiple_transient_plan_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum0 = ggml_add(ctx, a, b);
    ggml_tensor * sum1 = ggml_add(ctx, c, d);
    ggml_tensor * out  = ggml_add(ctx, sum0, sum1);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(sum0 != nullptr);
    REQUIRE(sum1 != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 3);

    const ggml::hrx::Value * sum0_value = imported.graph.values().find_tensor(sum0);
    const ggml::hrx::Value * sum1_value = imported.graph.values().find_tensor(sum1);
    REQUIRE(sum0_value != nullptr);
    REQUIRE(sum1_value != nullptr);
    REQUIRE(sum0_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(sum1_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 3);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 512);
    const ggml::hrx::TransientAllocation * sum0_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum0_value->id);
    const ggml::hrx::TransientAllocation * sum1_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum1_value->id);
    REQUIRE(sum0_allocation != nullptr);
    REQUIRE(sum1_allocation != nullptr);
    REQUIRE(sum0_allocation->arena_offset != sum1_allocation->arena_offset);
    REQUIRE(sum0_allocation->arena_offset % 256 == 0);
    REQUIRE(sum1_allocation->arena_offset % 256 == 0);

    ggml::hrx::CommandProgram        overlapping_live_transients = copy_command_program_shape(commands);
    ggml::hrx::TransientAllocation * overlapping_sum0            = nullptr;
    ggml::hrx::TransientAllocation * overlapping_sum1            = nullptr;
    for (ggml::hrx::TransientAllocation & allocation : overlapping_live_transients.transients.allocations) {
        if (allocation.value == sum0_value->id) {
            overlapping_sum0 = &allocation;
        } else if (allocation.value == sum1_value->id) {
            overlapping_sum1 = &allocation;
        }
    }
    REQUIRE(overlapping_sum0 != nullptr);
    REQUIRE(overlapping_sum1 != nullptr);
    overlapping_sum1->arena_offset = overlapping_sum0->arena_offset;
    ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(overlapping_live_transients, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(!verification.valid());
    REQUIRE(status_contains(verification.status, "transient allocations overlap"));

    ggml_free(ctx);
}

static void run_disjoint_transient_plan_packing_checks() {
    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * e    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * f    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum0 = ggml_add(ctx, a, b);
    ggml_tensor * out0 = ggml_add(ctx, sum0, c);
    ggml_tensor * sum1 = ggml_add(ctx, d, e);
    ggml_tensor * out1 = ggml_add(ctx, sum1, f);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(e != nullptr);
    REQUIRE(f != nullptr);
    REQUIRE(sum0 != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(sum1 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    REQUIRE(imported.graph.nodes().size() == 4);

    const ggml::hrx::Value * sum0_value = imported.graph.values().find_tensor(sum0);
    const ggml::hrx::Value * sum1_value = imported.graph.values().find_tensor(sum1);
    REQUIRE(sum0_value != nullptr);
    REQUIRE(sum1_value != nullptr);
    REQUIRE(sum0_value->kind == ggml::hrx::ValueKind::Transient);
    REQUIRE(sum1_value->kind == ggml::hrx::ValueKind::Transient);

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    REQUIRE(scheduler.plan().valid());
    REQUIRE(scheduler.plan().dispatches.size() == 4);

    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(commands.valid());
    REQUIRE(commands.transients.allocations.size() == 2);
    REQUIRE(commands.transients.arena_size == 256);
    const ggml::hrx::TransientAllocation * sum0_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum0_value->id);
    const ggml::hrx::TransientAllocation * sum1_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, sum1_value->id);
    REQUIRE(sum0_allocation != nullptr);
    REQUIRE(sum1_allocation != nullptr);
    REQUIRE(sum0_allocation->arena_offset == sum1_allocation->arena_offset);
    REQUIRE(sum0_allocation->arena_offset % 256 == 0);
    REQUIRE(command_program_verifies(commands));

    ggml_free(ctx);
}

static void run_command_shape_hash_checks() {
    REQUIRE(ggml::hrx::command_program_shape_hash("") == UINT64_C(1469598103934665603));
    std::string bytes;
    for (int i = 0; i < 256; ++i) {
        bytes.push_back(static_cast<char>(i));
    }
    bytes.append("\0key\xff", 5);
    REQUIRE(ggml::hrx::command_program_shape_hash(bytes) == UINT64_C(6643458358697029213));
    ggml::hrx::GraphProgram program(1, "gfx1151", nullptr, nullptr, bytes);
    bytes.clear();
    REQUIRE(program.command_shape_hash() == UINT64_C(6643458358697029213));
    REQUIRE(program.command_shape_hash() == ggml::hrx::command_program_shape_hash(program.command_shape()));
}

static void run_graph_program_cache_uid_mismatch_checks() {
    ggml::hrx::GraphProgramCache    cache;
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    ggml_tensor * out1 = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 3001;

    ggml::hrx::GraphProgramLookup lookup = cache.get_or_build(*graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    const uint64_t first_shape_hash = lookup.program->command_shape_hash();
    REQUIRE(first_shape_hash == ggml::hrx::command_program_shape_hash(lookup.program->command_shape()));
    ggml::hrx::GraphProgramCacheStats stats = cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 0);

    lookup = cache.get_or_build(*graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 1);
    REQUIRE(lookup.program->command_shape_hash() == first_shape_hash);

    ggml_cgraph * graph1 = ggml_new_graph(ctx);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out0);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = 3001;

    lookup = cache.get_or_build(*graph1, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 1);
    REQUIRE(lookup.program->command_shape_hash() != first_shape_hash);
    REQUIRE(lookup.program->command_shape_hash() ==
            ggml::hrx::command_program_shape_hash(lookup.program->command_shape()));

    ggml_tensor * unsupported = ggml_sin(ctx, a);
    REQUIRE(unsupported != nullptr);
    ggml_cgraph * graph2 = ggml_new_graph(ctx);
    REQUIRE(graph2 != nullptr);
    ggml_build_forward_expand(graph2, unsupported);
    graph2->uid = 3001;

    lookup = cache.get_or_build(*graph2, corpus, "gfx1151");
    REQUIRE(!lookup.valid());
    stats = cache.stats();
    REQUIRE(stats.builds == 2);
    REQUIRE(stats.hits == 1);

    ggml::hrx::GraphProgramCache alias_cache;
    ggml_tensor *                alias_sum = ggml_add(ctx, a, b);
    ggml_tensor *                view0     = ggml_view_1d(ctx, alias_sum, 4, 0);
    ggml_tensor *                view1     = ggml_view_1d(ctx, alias_sum, 4, sizeof(float));
    REQUIRE(alias_sum != nullptr);
    REQUIRE(view0 != nullptr);
    REQUIRE(view1 != nullptr);

    ggml_cgraph * alias_graph0 = ggml_new_graph(ctx);
    REQUIRE(alias_graph0 != nullptr);
    ggml_build_forward_expand(alias_graph0, view0);
    alias_graph0->uid = 3002;
    lookup            = alias_cache.get_or_build(*alias_graph0, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = alias_cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 0);

    ggml_cgraph * alias_graph1 = ggml_new_graph(ctx);
    REQUIRE(alias_graph1 != nullptr);
    ggml_build_forward_expand(alias_graph1, view1);
    alias_graph1->uid = 3002;
    lookup            = alias_cache.get_or_build(*alias_graph1, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    stats = alias_cache.stats();
    REQUIRE(stats.builds == 1);
    REQUIRE(stats.hits == 1);

    ggml_free(ctx);
}

static void run_graph_match_hash_collision_checks() {
    ggml_context * ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);
    const size_t buckets = ggml_hash_size(10);
    std::vector<ggml_tensor *> seen(buckets, nullptr);
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    for (size_t i = 0; i <= buckets && b == nullptr; ++i) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        const size_t bucket = ggml_hash(tensor) % buckets;
        if (seen[bucket] != nullptr) {
            a = seen[bucket];
            b = tensor;
        } else {
            seen[bucket] = tensor;
        }
    }
    REQUIRE(a != nullptr && b != nullptr && a != b);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, sum, c);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);
    ggml::hrx::GraphProgramCache cache;
    auto lookup = cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(lookup.valid());
    REQUIRE(lookup.program->graph().values().size() == 5);
    const auto match = lookup.program->match_current_graph(*graph);
    REQUIRE(match.valid());
    for (const ggml_tensor * tensor : { a, b, c, out }) {
        REQUIRE(std::any_of(match.external_bindings.begin(), match.external_bindings.end(),
                            [&](const auto & binding) { return binding.tensor == tensor; }));
    }
    for (int i = 0; i < 128; ++i) {
        out = ggml_add(ctx, out, i % 2 == 0 ? a : b);
    }
    ggml_build_forward_expand(graph, out);
    lookup = cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(lookup.valid());
    REQUIRE(lookup.program->graph().values().size() == 133);
    REQUIRE(lookup.program->match_current_graph(*graph).valid());
    ggml_free(ctx);
}

static void run_graph_match_bijection_checks() {
    ggml_context * ctx = ggml_init({ 512 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * out = ggml_add(ctx, sum, b);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, out);
    ggml::hrx::GraphProgramCache cache;
    auto lookup = cache.get_or_build(*graph, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(lookup.valid());
    REQUIRE(lookup.program->match_current_graph(*graph).valid());

    sum->src[1] = a;
    auto duplicate_tensor = lookup.program->match_current_graph(*graph);
    REQUIRE(!duplicate_tensor.valid());
    REQUIRE(status_contains(duplicate_tensor.status, "tensor maps to cached values"));
    sum->src[1] = b;
    out->src[1] = c;
    auto duplicate_value = lookup.program->match_current_graph(*graph);
    REQUIRE(!duplicate_value.valid());
    REQUIRE(status_contains(duplicate_value.status, "maps to multiple current tensors"));
    out->src[1] = b;
    REQUIRE(lookup.program->match_current_graph(*graph).valid());
    ggml_free(ctx);
}

static void run_validated_graph_uid_match_checks() {
    ggml::hrx::GraphProgramCache cache;
    const auto & corpus = ggml::hrx::get_qwen_kernel_corpus();
    ggml_context * ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);
    auto make_graph = [&](uint64_t uid, int64_t elements, bool two_nodes) {
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        ggml_tensor * out = ggml_add(ctx, a, b);
        if (two_nodes) {
            out = ggml_add(ctx, out, b);
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, out);
        graph->uid = uid;
        return std::make_pair(graph, a);
    };
    auto first = make_graph(4001, 8, false);
    auto alias = make_graph(4002, 8, false);
    auto original = cache.get_or_build(*first.first, corpus, "gfx1151");
    REQUIRE(original.valid());
    for (int i = 0; i < 2; ++i) {
        auto match = cache.get_or_build(*alias.first, corpus, "gfx1151");
        REQUIRE(match.valid());
        REQUIRE(match.program == original.program);
        REQUIRE(std::any_of(match.match.external_bindings.begin(), match.match.external_bindings.end(),
                            [&](const auto & binding) { return binding.tensor == alias.second; }));
    }
    REQUIRE(cache.stats().builds == 1);

    auto new_nodes = make_graph(4002, 16, false);
    auto changed = cache.get_or_build(*new_nodes.first, corpus, "gfx1151");
    REQUIRE(changed.valid());
    REQUIRE(changed.program->uid() == 4002);
    REQUIRE(cache.stats().builds == 2);

    auto second_alias = make_graph(4003, 8, false);
    REQUIRE(cache.get_or_build(*second_alias.first, corpus, "gfx1151").program == original.program);
    auto replacement = make_graph(4001, 8, true);
    REQUIRE(cache.get_or_build(*replacement.first, corpus, "gfx1151").valid());
    REQUIRE(cache.stats().builds == 3);
    auto after_replacement = cache.get_or_build(*second_alias.first, corpus, "gfx1151");
    REQUIRE(after_replacement.valid());
    REQUIRE(after_replacement.program->uid() == 4003);
    REQUIRE(cache.stats().builds == 4);

    auto checked_alias = make_graph(4004, 8, false);
    REQUIRE(cache.get_or_build(*checked_alias.first, corpus, "gfx1151").valid());
    const char * validate_name = "GGML_HRX_VALIDATE_GRAPH_UID_CACHE";
    const char * validate_value = std::getenv(validate_name);
    const bool had_validate = validate_value != nullptr;
    const std::string saved_validate = had_validate ? validate_value : "";
    REQUIRE(setenv(validate_name, "1", 1) == 0);
    checked_alias.second->ne[0] = 12;
    REQUIRE(!cache.get_or_build(*checked_alias.first, corpus, "gfx1151").valid());
    checked_alias.second->ne[0] = 8;
    REQUIRE(cache.get_or_build(*checked_alias.first, corpus, "gfx1151").valid());
    restore_environment_value(validate_name, had_validate, saved_validate);

    for (uint64_t uid = 4100; uid < 4230; ++uid) {
        checked_alias.first->uid = uid;
        auto match = cache.get_or_build(*checked_alias.first, corpus, "gfx1151");
        REQUIRE(match.valid());
        REQUIRE(match.program->uid() == 4003);
    }
    REQUIRE(cache.stats().builds == 4);
    cache.clear();
    auto after_clear = cache.get_or_build(*second_alias.first, corpus, "gfx1151");
    REQUIRE(after_clear.valid());
    REQUIRE(after_clear.program->uid() == 4003);
    REQUIRE(cache.stats().builds == 5);
    ggml_free(ctx);
}

static void run_graph_executor_contract_checks() {
    ggml_backend_hrx_device_context device_context  = {};
    ggml_backend_hrx_context        backend_context = {};
    device_context.architecture                     = "gfx1151";
    backend_context.device                          = &device_context;
    const ggml::hrx::GraphExecutor executor(backend_context);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * add_out = ggml_add(ctx, a, b);
    ggml_tensor * sin_out = ggml_sin(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(add_out != nullptr);
    REQUIRE(sin_out != nullptr);

    ggml_cgraph * add_graph = ggml_new_graph(ctx);
    REQUIRE(add_graph != nullptr);
    ggml_build_forward_expand(add_graph, add_out);
    const ggml::hrx::GraphSupportResult add_support = executor.can_execute(*add_graph);
    REQUIRE(add_support.supported);
    REQUIRE(add_support.status.success());

    const ggml::hrx::GraphExecutionResult missing_binding = executor.execute(*add_graph);
    REQUIRE(!missing_binding.success());
    REQUIRE(missing_binding.code == GGML_STATUS_FAILED);
    REQUIRE(status_contains(missing_binding.status, "external value"));
    REQUIRE(status_contains(missing_binding.status, "not bound"));

    ggml_cgraph * sin_graph = ggml_new_graph(ctx);
    REQUIRE(sin_graph != nullptr);
    ggml_build_forward_expand(sin_graph, sin_out);
    const ggml::hrx::GraphSupportResult sin_support = executor.can_execute(*sin_graph);
    REQUIRE(!sin_support.supported);
    REQUIRE(status_contains(sin_support.status, "unsupported HRX node 0"));
    REQUIRE(status_contains(sin_support.status, "SIN"));

    ggml_free(ctx);
}

static void build_add_graph_program(ggml::hrx::GraphProgramCache & cache, uint64_t uid, bool two_outputs) {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out0 = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out0 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    if (two_outputs) {
        ggml_tensor * c    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * d    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * out1 = ggml_add(ctx, c, d);
        REQUIRE(c != nullptr);
        REQUIRE(d != nullptr);
        REQUIRE(out1 != nullptr);
        ggml_build_forward_expand(graph, out1);
    }
    graph->uid = uid;

    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    ggml::hrx::GraphProgramLookup   lookup = cache.get_or_build(*graph, corpus, "gfx1151");
    REQUIRE(lookup.valid());
    ggml_free(ctx);
}

static void run_command_program_kernel_dump_checks() {
    static constexpr const char * kDumpEnv = "GGML_HRX_DUMP_COMMAND_PROGRAM_DIR";
    EnvironmentVariableGuard      env(kDumpEnv);

    {
        const std::filesystem::path dump_dir = fresh_test_directory("kernel-dump-disabled");
        env.unset();
        ggml::hrx::GraphProgramCache cache;
        build_add_graph_program(cache, 5001, false);
        REQUIRE(list_directories(dump_dir).empty());
        std::filesystem::remove_all(dump_dir);
    }

    const std::filesystem::path dump_dir = fresh_test_directory("kernel-dump-enabled");
    env.set(dump_dir);

    ggml::hrx::GraphProgramCache cache;
    build_add_graph_program(cache, 5002, false);
    std::vector<std::filesystem::path> dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 1);
    REQUIRE(std::filesystem::exists(dumps[0] / "kernels.txt"));
    REQUIRE(std::filesystem::exists(dumps[0] / "kernels.dot"));

    const std::string add_kernels = read_text_file(dumps[0] / "kernels.txt");
    const std::string add_dot     = read_text_file(dumps[0] / "kernels.dot");
    REQUIRE(string_contains(add_kernels, "main 0 loom_libs:ggml_binary_f32"));
    REQUIRE(!string_contains(add_kernels, "binding"));
    REQUIRE(!string_contains(add_kernels, "transient"));
    REQUIRE(string_contains(add_dot, "digraph hrx_kernel_invocations"));
    REQUIRE(string_contains(add_dot, "main 0\\nloom_libs:ggml_binary_f32"));

    build_add_graph_program(cache, 5003, false);
    dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 1);

    const std::filesystem::path second_dump_dir = fresh_test_directory("kernel-dump-second-directory");
    env.set(second_dump_dir);
    build_add_graph_program(cache, 5005, false);
    std::vector<std::filesystem::path> second_dumps = list_directories(second_dump_dir);
    REQUIRE(second_dumps.size() == 1);
    REQUIRE(std::filesystem::exists(second_dumps[0] / "kernels.txt"));
    std::filesystem::remove_all(second_dump_dir);
    env.set(dump_dir);

    build_add_graph_program(cache, 5004, true);
    dumps = list_directories(dump_dir);
    REQUIRE(dumps.size() == 2);

    const std::string two_add_kernels = read_text_file(dumps[1] / "kernels.txt");
    const std::string two_add_dot     = read_text_file(dumps[1] / "kernels.dot");
    REQUIRE(string_contains(two_add_kernels, "main 0 loom_libs:ggml_binary_f32"));
    REQUIRE(string_contains(two_add_kernels, "main 1 loom_libs:ggml_binary_f32 deps=0"));
    REQUIRE(string_contains(two_add_dot, "main0 -> main1"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "commands.txt"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "commands.json"));
    REQUIRE(!std::filesystem::exists(dumps[1] / "manifest.json"));

    std::filesystem::remove_all(dump_dir);
}

static std::vector<int32_t> read_transient_i32(ggml_backend_hrx_context *             context,
                                               ggml::hrx::TransientArenaAllocationRef arena,
                                               const ggml::hrx::TransientAllocation & allocation) {
    REQUIRE(context != nullptr);
    REQUIRE(arena.buffer != nullptr);
    REQUIRE(allocation.size % sizeof(int32_t) == 0);
    std::vector<int32_t> data(allocation.size / sizeof(int32_t));
    require_hrx_status(hrx_synchronous_d2h(context->device->device, arena.buffer, allocation.arena_offset, data.data(),
                                           allocation.size));
    return data;
}

static void write_transient_i32_value(ggml_backend_hrx_context *             context,
                                      ggml::hrx::TransientArenaAllocationRef arena,
                                      const ggml::hrx::TransientAllocation & allocation,
                                      int32_t                                value) {
    REQUIRE(context != nullptr);
    REQUIRE(arena.buffer != nullptr);
    REQUIRE(allocation.size >= sizeof(value));
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, &value, arena.buffer, allocation.arena_offset, sizeof(value)));
}

static std::vector<uint8_t> read_transient_bytes(ggml_backend_hrx_context *             context,
                                                 ggml::hrx::TransientArenaAllocationRef arena,
                                                 const ggml::hrx::TransientAllocation & allocation) {
    REQUIRE(context != nullptr);
    REQUIRE(arena.buffer != nullptr);
    std::vector<uint8_t> data(allocation.size);
    require_hrx_status(hrx_synchronous_d2h(context->device->device, arena.buffer, allocation.arena_offset, data.data(),
                                           allocation.size));
    return data;
}

static void run_binary_q8_publication_numerics() {
    const struct {
        int64_t hidden;
        int64_t tokens;
        bool    broadcast;
        int     op;
    } cases[] = {
        { 2048, 3, false, 0 },
        { 2176, 4, false, 2 },
        { 2048, 4, true,  0 },
        { 2176, 5, true,  2 },
    };

    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    auto * backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(backend_context != nullptr);
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    const char * target = backend_context->device->architecture.c_str();

    for (const auto & test : cases) {
        ggml_init_params params = {};
        params.mem_size         = 512 * 1024;
        params.no_alloc         = true;
        ggml_context * ctx      = ggml_init(params);
        REQUIRE(ctx != nullptr);

        const int64_t element_count = test.hidden * test.tokens;
        const int64_t rhs_count     = test.broadcast ? test.hidden : element_count;
        ggml_tensor * lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens);
        ggml_tensor * rhs = test.broadcast ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test.hidden) :
                                             ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test.hidden, test.tokens);
        REQUIRE(lhs != nullptr && rhs != nullptr);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer != nullptr);

        std::vector<float> lhs_data(element_count);
        std::vector<float> rhs_data(rhs_count);
        std::vector<float> expected(element_count);
        for (int64_t i = 0; i < element_count; ++i) {
            lhs_data[i] = static_cast<float>((i * 17 + 3) % 251 - 125) / 31.0f;
        }
        for (int64_t i = 0; i < rhs_count; ++i) {
            rhs_data[i] = static_cast<float>((i * 29 + 11) % 239 - 119) / 37.0f;
        }
        for (int64_t token = 0; token < test.tokens; ++token) {
            for (int64_t channel = 0; channel < test.hidden; ++channel) {
                const int64_t i   = token * test.hidden + channel;
                const float rhs_v = rhs_data[test.broadcast ? channel : i];
                expected[i]       = test.op == 0 ? lhs_data[i] + rhs_v : lhs_data[i] * rhs_v;
            }
        }
        ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
        ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
        ggml_backend_synchronize(backend);

        ggml::hrx::Graph graph;
        const ggml::hrx::ValueId lhs_value =
            graph.values().get_or_add_tensor_value(lhs, ggml::hrx::ValueKind::External);
        const ggml::hrx::ValueId rhs_value =
            graph.values().get_or_add_tensor_value(rhs, ggml::hrx::ValueKind::External);
        ggml::hrx::ValueBufferBinding lhs_binding;
        ggml::hrx::ValueBufferBinding rhs_binding;
        REQUIRE(ggml_backend_hrx_resolve_value_buffer(lhs, lhs_binding));
        REQUIRE(ggml_backend_hrx_resolve_value_buffer(rhs, rhs_binding));
        REQUIRE(graph.values().bind_buffer(lhs_value, lhs_binding));
        REQUIRE(graph.values().bind_buffer(rhs_value, rhs_binding));

        const ggml::hrx::ValueId reference_f32(static_cast<int32_t>(graph.values().size()));
        const ggml::hrx::ValueId reference_q8(reference_f32.value + 1);
        const ggml::hrx::ValueId published_f32(reference_f32.value + 2);
        const ggml::hrx::ValueId published_q8(reference_f32.value + 3);
        const size_t f32_bytes = static_cast<size_t>(element_count) * sizeof(float);
        const size_t q8_bytes  = qwen_q8_1_x4_size(test.tokens, test.hidden);

        ggml::hrx::CommandPlan plan;
        plan.transients.push_back({ reference_f32, "test.binary.reference_f32", f32_bytes, 256 });
        plan.transients.push_back({ reference_q8, "test.binary.reference_q8", q8_bytes, 256 });
        plan.transients.push_back({ published_f32, "test.binary.published_f32", f32_bytes, 256 });
        plan.transients.push_back({ published_q8, "test.binary.published_q8", q8_bytes, 256 });

        ggml::hrx::Dispatch legacy;
        legacy.kernel = ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref(
            "loom_libs", test.broadcast ? "ggml_binary_bc_f32" : "ggml_binary_f32"));
        legacy.kernel.integer_parameters.emplace("element_count", element_count);
        legacy.kernel.compile_parameters.emplace(
            test.broadcast ? "ggml.binary_bc_f32.op" : "ggml.binary_f32.op", std::to_string(test.op));
        if (test.broadcast) {
            legacy.kernel.integer_parameters.emplace("ne0", test.hidden);
            legacy.kernel.integer_parameters.emplace("ne1", test.tokens);
            legacy.kernel.integer_parameters.emplace("ne2", 1);
            legacy.kernel.integer_parameters.emplace("ne3", 1);
            legacy.kernel.integer_parameters.emplace("src0_element_count", element_count);
            legacy.kernel.integer_parameters.emplace("src1_element_count", rhs_count);
        } else {
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.ne0", std::to_string(test.hidden));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.ne1", std::to_string(test.tokens));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.ne2", "1");
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride1", std::to_string(test.hidden));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride2", std::to_string(element_count));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride3", std::to_string(element_count));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride1", std::to_string(test.hidden));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride2", std::to_string(element_count));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride3", std::to_string(element_count));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src0_span", std::to_string(element_count));
            legacy.kernel.compile_parameters.emplace("ggml.binary_f32.src1_span", std::to_string(element_count));
        }
        if (test.broadcast) {
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                legacy.kernel.compile_parameters.emplace("ggml.binary_bc_f32.src0_broadcast_dim" +
                                                              std::to_string(dim),
                                                          "0");
                legacy.kernel.compile_parameters.emplace("ggml.binary_bc_f32.src1_broadcast_dim" +
                                                              std::to_string(dim),
                                                          dim == 1 ? "1" : "0");
            }
        }
        legacy.bindings.push_back({ lhs_value, 0, ggml_nbytes(lhs) });
        legacy.bindings.push_back({ rhs_value, 0, ggml_nbytes(rhs) });
        legacy.bindings.push_back({ reference_f32, 0, f32_bytes });
        plan.dispatches.push_back(std::move(legacy));

        ggml::hrx::Dispatch quantize;
        quantize.kernel = ggml::hrx::make_kernel_specialization(
            ggml::hrx::kernel_catalog_ref("qwen3_moe", "ggml_quantize_q8_1_x4_f32"));
        quantize.kernel.integer_parameters.emplace("token_count", test.tokens);
        quantize.kernel.integer_parameters.emplace("input_size", test.hidden);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   std::to_string(element_count / 128));
        quantize.bindings.push_back({ reference_f32, 0, f32_bytes });
        quantize.bindings.push_back({ reference_q8, 0, q8_bytes });
        plan.dispatches.push_back(std::move(quantize));

        ggml::hrx::Dispatch publish;
        publish.kernel = ggml::hrx::make_kernel_specialization(ggml::hrx::kernel_catalog_ref(
            "loom_libs", test.broadcast ? "ggml_binary_bc_f32_publish_q8_1_x4" :
                                           "ggml_binary_f32_publish_q8_1_x4"));
        publish.kernel.integer_parameters.emplace("token_count", test.tokens);
        if (test.broadcast) {
            publish.kernel.integer_parameters.emplace("hidden_size", test.hidden);
            publish.kernel.integer_parameters.emplace("src0_element_count", element_count);
            publish.kernel.integer_parameters.emplace("src1_element_count", rhs_count);
            publish.kernel.compile_parameters = plan.dispatches.front().kernel.compile_parameters;
        } else {
            publish.kernel.compile_parameters = plan.dispatches.front().kernel.compile_parameters;
        }
        publish.bindings.push_back({ lhs_value, 0, ggml_nbytes(lhs) });
        publish.bindings.push_back({ rhs_value, 0, ggml_nbytes(rhs) });
        publish.bindings.push_back({ published_f32, 0, f32_bytes });
        publish.bindings.push_back({ published_q8, 0, q8_bytes });
        plan.dispatches.push_back(std::move(publish));

        const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(graph, plan, corpus, target);
        REQUIRE(commands.valid());
        REQUIRE(ggml::hrx::verify_command_program(commands, corpus, target).valid());
        const ggml::hrx::CommandProgramBindings bindings =
            ggml::hrx::CommandProgramBindings::from_value_map(graph.values());
        REQUIRE(bindings.valid());
        const ggml::hrx::CommandProgramExecutionContext execution_context = {
            backend_context->device->device,
            backend_context->stream,
            target,
            &corpus,
            &backend_context->kernel_executables,
            &backend_context->transient_arena,
            &backend_context->host_transfers,
            &backend_context->host_weights,
        };
        REQUIRE(ggml::hrx::execute_command_program(execution_context, commands, bindings));
        ggml_backend_synchronize(backend);

        const auto arena = backend_context->transient_arena.current_allocation();
        const auto * reference_f32_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, reference_f32);
        const auto * reference_q8_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, reference_q8);
        const auto * published_f32_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, published_f32);
        const auto * published_q8_allocation =
            ggml::hrx::find_transient_allocation(commands.transients, published_q8);
        REQUIRE(reference_f32_allocation != nullptr && reference_q8_allocation != nullptr);
        REQUIRE(published_f32_allocation != nullptr && published_q8_allocation != nullptr);
        const auto reference_f32_bytes = read_transient_bytes(backend_context, arena, *reference_f32_allocation);
        const auto reference_q8_bytes  = read_transient_bytes(backend_context, arena, *reference_q8_allocation);
        const auto published_f32_bytes = read_transient_bytes(backend_context, arena, *published_f32_allocation);
        const auto published_q8_bytes  = read_transient_bytes(backend_context, arena, *published_q8_allocation);
        REQUIRE(reference_f32_bytes == published_f32_bytes);
        REQUIRE(reference_q8_bytes == published_q8_bytes);
        REQUIRE(reference_f32_bytes.size() == expected.size() * sizeof(float));
        REQUIRE(std::memcmp(reference_f32_bytes.data(), expected.data(), reference_f32_bytes.size()) == 0);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}

static void run_qwen_expert_table_partition_prefill_512_execution() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    auto * backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(backend_context != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t token_count  = 512;
    constexpr int64_t route_count  = 8;
    constexpr int64_t route_stride = 8;
    constexpr int64_t expert_count = 128;

    ggml_tensor * route_ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, route_stride, token_count);
    REQUIRE(route_ids_tensor != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    const std::vector<int32_t> route_ids =
        make_qwen_route_ids_iota(token_count, route_count, route_stride, expert_count);
    const std::vector<int32_t> expected_expert_table =
        make_qwen_expert_table_reference(route_ids, token_count, route_count, route_stride, expert_count);
    const std::vector<int32_t> expected_partition_table =
        make_qwen_partition_table_reference(expected_expert_table, token_count, route_count, expert_count);
    ggml_backend_tensor_set(route_ids_tensor, route_ids.data(), 0, route_ids.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend);

    ggml::hrx::Graph         graph;
    const ggml::hrx::ValueId route_ids_value =
        graph.values().get_or_add_tensor_value(route_ids_tensor, ggml::hrx::ValueKind::External);
    ggml::hrx::ValueBufferBinding route_ids_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(route_ids_tensor, route_ids_binding));
    REQUIRE(graph.values().bind_buffer(route_ids_value, route_ids_binding));

    const ggml::hrx::ValueId expert_table_value(static_cast<int32_t>(graph.values().size()));
    const ggml::hrx::ValueId partition_table_value(expert_table_value.value + 1);
    const ggml::hrx::ValueId completion_counter_value(expert_table_value.value + 2);
    ggml::hrx::CommandPlan   plan;
    plan.transients.push_back(
        { expert_table_value, "qwen.test.expert_table", qwen_expert_table_size(token_count, expert_count), 256 });
    plan.transients.push_back({ partition_table_value, "qwen.test.partition_table",
                                qwen_partition_table_size(token_count, route_count, expert_count), 256 });
    plan.completion_counter_requests.push_back({ completion_counter_value, "qwen.test.completion_counter", 1 });
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel = ggml::hrx::make_kernel_specialization(
        ggml::hrx::kernel_catalog_ref("qwen3_moe", "qwen3_moe_build_expert_table_partition_prefill_512"));
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.kernel.integer_parameters.emplace("route_count", route_count);
    dispatch.kernel.integer_parameters.emplace("route_stride", route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", expert_count);
    dispatch.bindings.push_back({ route_ids_value, 0, route_ids.size() * sizeof(int32_t) });
    dispatch.bindings.push_back({ expert_table_value, 0, qwen_expert_table_size(token_count, expert_count) });
    dispatch.bindings.push_back(
        { partition_table_value, 0, qwen_partition_table_size(token_count, route_count, expert_count) });
    dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
    plan.dispatches.push_back(std::move(dispatch));

    const ggml::hrx::KernelCorpus & corpus   = ggml::hrx::get_qwen_kernel_corpus();
    const char *                    target   = backend_context->device->architecture.c_str();
    const ggml::hrx::CommandProgram commands = ggml::hrx::build_command_program(graph, plan, corpus, target);
    REQUIRE(commands.valid());
    REQUIRE(commands.commands.size() == 1);
    REQUIRE(commands.completion_counters.count == 1);
    REQUIRE(commands.completion_counters.byte_count == sizeof(int32_t));
    REQUIRE(commands.transients.allocations.size() == 3);
    REQUIRE(ggml::hrx::verify_command_program(commands, corpus, target).valid());

    const ggml::hrx::TransientAllocation * expert_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, expert_table_value);
    const ggml::hrx::TransientAllocation * partition_table_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, partition_table_value);
    const ggml::hrx::TransientAllocation * completion_counter_allocation =
        ggml::hrx::find_transient_allocation(commands.transients, completion_counter_value);
    REQUIRE(expert_table_allocation != nullptr);
    REQUIRE(partition_table_allocation != nullptr);
    REQUIRE(completion_counter_allocation != nullptr);
    REQUIRE(completion_counter_allocation->arena_offset == commands.completion_counters.arena_offset);

    const ggml::hrx::CommandProgramBindings bindings =
        ggml::hrx::CommandProgramBindings::from_value_map(graph.values());
    REQUIRE(bindings.valid());
    const ggml::hrx::CommandProgramExecutionContext execution_context = {
        backend_context->device->device,
        backend_context->stream,
        target,
        &corpus,
        &backend_context->kernel_executables,
        &backend_context->transient_arena,
        &backend_context->host_transfers,
        &backend_context->host_weights,
    };

    REQUIRE(ggml::hrx::execute_command_program(execution_context, commands, bindings));
    ggml_backend_synchronize(backend);
    ggml::hrx::TransientArenaAllocationRef arena = backend_context->transient_arena.current_allocation();
    require_qwen_expert_table_matches(read_transient_i32(backend_context, arena, *expert_table_allocation),
                                      expected_expert_table, token_count, expert_count);
    require_qwen_partition_table_matches(read_transient_i32(backend_context, arena, *partition_table_allocation),
                                         expected_partition_table);
    REQUIRE(read_transient_i32(backend_context, arena, *completion_counter_allocation)[0] == 0);

    write_transient_i32_value(backend_context, arena, *completion_counter_allocation, 17);
    REQUIRE(ggml::hrx::execute_command_program(execution_context, commands, bindings));
    ggml_backend_synchronize(backend);
    arena = backend_context->transient_arena.current_allocation();
    require_qwen_expert_table_matches(read_transient_i32(backend_context, arena, *expert_table_allocation),
                                      expected_expert_table, token_count, expert_count);
    require_qwen_partition_table_matches(read_transient_i32(backend_context, arena, *partition_table_allocation),
                                         expected_partition_table);
    REQUIRE(read_transient_i32(backend_context, arena, *completion_counter_allocation)[0] == 0);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out           = ggml_add(ctx, a, b);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static std::vector<float> run_rope_scale_numerical_case(ggml_backend_t backend, bool per_token, int mode) {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t head_size   = 128;
    constexpr int64_t n_dims      = 96;
    constexpr int64_t head_count  = 2;
    constexpr int64_t token_count = 3;
    constexpr int64_t elements    = head_size * head_count * token_count;

    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, head_count, token_count);
    ggml_tensor * pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    ggml_tensor * freqs  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
    ggml_tensor * rope =
        ggml_rope_ext(ctx, input, pos, freqs, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.19024f, 32.0f, 1.0f);
    ggml_tensor * scales = per_token ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, token_count) : nullptr;
    ggml_tensor * output = per_token ? ggml_mul(ctx, rope, scales) : ggml_scale(ctx, rope, 0.0883883461f);
    REQUIRE(input != nullptr);
    REQUIRE(pos != nullptr);
    REQUIRE(freqs != nullptr);
    REQUIRE(rope != nullptr);
    REQUIRE(output != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    std::vector<float> input_data(elements);
    std::vector<float> freq_data(n_dims / 2);
    const std::vector<int32_t> positions = { 3, 7, 11 };
    const std::vector<float> scale_data  = { 0.125f, -0.25f, 0.5f };
    for (int64_t i = 0; i < elements; ++i) {
        input_data[i] = static_cast<float>((i * 17) % 113) * 0.03125f - 1.5f;
    }
    for (int64_t i = 0; i < n_dims / 2; ++i) {
        freq_data[i] = 0.75f + static_cast<float>(i % 7) * 0.125f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(freqs, freq_data.data(), 0, freq_data.size() * sizeof(float));
    if (per_token) {
        ggml_backend_tensor_set(scales, scale_data.data(), 0, scale_data.size() * sizeof(float));
    }

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> actual(elements);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return actual;
}

static void run_rope_scale_f32_numerics() {
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr);
    REQUIRE(cpu != nullptr);

    for (const bool per_token : { false, true }) {
        for (const int mode : { static_cast<int>(GGML_ROPE_TYPE_NORMAL), static_cast<int>(GGML_ROPE_TYPE_NEOX) }) {
            const std::vector<float> expected = run_rope_scale_numerical_case(cpu, per_token, mode);
            const std::vector<float> actual   = run_rope_scale_numerical_case(hrx, per_token, mode);
            REQUIRE(actual.size() == expected.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                const float tolerance = 2.0e-4f + 2.0e-4f * std::fabs(expected[i]);
                REQUIRE(std::fabs(actual[i] - expected[i]) <= tolerance);
            }
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
}

struct GatherAddRmsNormOutputs {
    std::vector<float> raw;
    std::vector<float> normalized;
};

static GatherAddRmsNormOutputs run_gather_add_rmsnorm_numerical_case(ggml_backend_t backend) {
    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t     hidden_size        = 128;
    constexpr int64_t     source_token_count = 4;
    constexpr int64_t     output_token_count = 5;
    GatherAddRmsNormGraph tensors =
        build_gather_add_rmsnorm_graph(ctx, hidden_size, source_token_count, output_token_count);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float>         attention(hidden_size * source_token_count);
    std::vector<float>         residual(hidden_size * source_token_count);
    std::vector<float>         weight(hidden_size);
    const std::vector<int32_t> row_ids = { 3, 1, 3, 0, 2 };
    for (size_t i = 0; i < attention.size(); ++i) {
        attention[i] = static_cast<float>((i * 13) % 47) * 0.03125f - 0.5f;
        residual[i]  = static_cast<float>((i * 7) % 31) * -0.015625f + 0.25f;
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = 0.5f + static_cast<float>(i % 11) * 0.0625f;
    }
    ggml_backend_tensor_set(tensors.attention, attention.data(), 0, attention.size() * sizeof(float));
    ggml_backend_tensor_set(tensors.residual, residual.data(), 0, residual.size() * sizeof(float));
    ggml_backend_tensor_set(tensors.row_ids, row_ids.data(), 0, row_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(tensors.weight, weight.data(), 0, weight.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, tensors.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    GatherAddRmsNormOutputs outputs;
    outputs.raw.resize(hidden_size * output_token_count);
    outputs.normalized.resize(hidden_size * output_token_count);
    ggml_backend_tensor_get(tensors.raw_output, outputs.raw.data(), 0, outputs.raw.size() * sizeof(float));
    ggml_backend_tensor_get(tensors.normalized_output, outputs.normalized.data(), 0,
                            outputs.normalized.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return outputs;
}

static void run_gather_add_rmsnorm_f32_numerics() {
    ggml_backend_t hrx = ggml_backend_hrx_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr);
    REQUIRE(cpu != nullptr);

    const GatherAddRmsNormOutputs expected = run_gather_add_rmsnorm_numerical_case(cpu);
    const GatherAddRmsNormOutputs actual   = run_gather_add_rmsnorm_numerical_case(hrx);
    REQUIRE(actual.raw.size() == expected.raw.size());
    REQUIRE(actual.normalized.size() == expected.normalized.size());
    for (size_t i = 0; i < actual.raw.size(); ++i) {
        REQUIRE(actual.raw[i] == expected.raw[i]);
        const float tolerance = 3.0e-4f + 3.0e-4f * std::fabs(expected.normalized[i]);
        REQUIRE(std::fabs(actual.normalized[i] - expected.normalized[i]) <= tolerance);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
}

static void run_scale_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     input         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out           = ggml_scale_bias(ctx, input, -1.5f, 0.25f);
    REQUIRE(input != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> input_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        input_data[i] = static_cast<float>(i % 23) * 0.125f - 1.0f;
        expected[i]   = input_data[i] * -1.5f + 0.25f;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(std::fabs(actual[i] - expected[i]) <= 1.0e-6f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_scale_f32_inplace() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 512 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     input         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out           = ggml_scale_bias_inplace(ctx, input, -1.5f, 0.25f);
    REQUIRE(input != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> input_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        input_data[i] = static_cast<float>(i % 23) * 0.125f - 1.0f;
        expected[i]   = input_data[i] * -1.5f + 0.25f;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(std::fabs(actual[i] - expected[i]) <= 1.0e-6f);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_two_independent_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     d             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     out0          = ggml_add(ctx, a, b);
    ggml_tensor *     out1          = ggml_add(ctx, c, d);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    REQUIRE(out0 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out0);
    ggml_build_forward_expand(graph, out1);
    graph->uid = 1002;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> d_data(element_count);
    std::vector<float> expected0(element_count);
    std::vector<float> expected1(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]    = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]    = static_cast<float>(i % 19) * 0.125f + 1.0f;
        d_data[i]    = static_cast<float>(i % 11) * 0.75f - 4.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 0);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 0);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual0(element_count);
    std::vector<float> actual1(element_count);
    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]    = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]    = static_cast<float>(i % 7) * 0.5f - 1.0f;
        c_data[i]    = static_cast<float>(i % 5) * -0.125f + 2.0f;
        d_data[i]    = static_cast<float>(i % 29) * 0.75f - 6.0f;
        expected0[i] = a_data[i] + b_data[i];
        expected1[i] = c_data[i] + d_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
    ggml_backend_tensor_set(d, d_data.data(), 0, d_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out0, actual0.data(), 0, actual0.size() * sizeof(float));
    ggml_backend_tensor_get(out1, actual1.data(), 0, actual1.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual0[i] == expected0[i]);
        REQUIRE(actual1[i] == expected1[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_chained_add_f32() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    constexpr int64_t element_count = 1024;
    ggml_tensor *     a             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     c             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, element_count);
    ggml_tensor *     sum           = ggml_add(ctx, a, b);
    ggml_tensor *     out           = ggml_add(ctx, sum, c);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(sum != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);
    graph->uid = 1004;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> c_data(element_count);
    std::vector<float> expected(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        c_data[i]   = static_cast<float>(i % 7) * 0.125f + 1.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }

    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_builds == 1);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 11) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 5) * 0.5f - 1.0f;
        c_data[i]   = static_cast<float>(i % 19) * 0.75f - 6.0f;
        expected[i] = a_data[i] + b_data[i] + c_data[i];
    }
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
    ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_hits == 1);

    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_same_uid_distinct_graph_reuses_graph_program(bool distinct_uid = false) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    constexpr int64_t  element_count = 1024;
    std::vector<float> a_data(element_count);
    std::vector<float> b_data(element_count);
    std::vector<float> expected(element_count);

    ggml_init_params params0 = {};
    params0.mem_size         = 256 * 1024;
    params0.no_alloc         = true;
    ggml_context * ctx0      = ggml_init(params0);
    REQUIRE(ctx0 != nullptr);

    ggml_tensor * a0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * b0   = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, element_count);
    ggml_tensor * out0 = ggml_add(ctx0, a0, b0);
    REQUIRE(a0 != nullptr);
    REQUIRE(b0 != nullptr);
    REQUIRE(out0 != nullptr);

    ggml_cgraph * graph0 = ggml_new_graph(ctx0);
    REQUIRE(graph0 != nullptr);
    ggml_build_forward_expand(graph0, out0);
    graph0->uid = 1003;

    ggml_backend_buffer_t buffer0 = ggml_backend_alloc_ctx_tensors(ctx0, backend);
    REQUIRE(buffer0 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 17) * 0.25f - 2.0f;
        b_data[i]   = static_cast<float>(i % 13) * -0.5f + 3.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a0, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b0, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph0) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    ggml_backend_hrx_cache_stats cache_stats = {};
    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 0);
    REQUIRE(cache_stats.prepared_program_builds == 1);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    ggml_init_params params1 = {};
    params1.mem_size         = 256 * 1024;
    params1.no_alloc         = true;
    ggml_context * ctx1      = ggml_init(params1);
    REQUIRE(ctx1 != nullptr);

    ggml_tensor * a1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * b1   = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, element_count);
    ggml_tensor * out1 = ggml_add(ctx1, a1, b1);
    REQUIRE(a1 != nullptr);
    REQUIRE(b1 != nullptr);
    REQUIRE(out1 != nullptr);

    ggml_cgraph * graph1 = ggml_new_graph(ctx1);
    REQUIRE(graph1 != nullptr);
    ggml_build_forward_expand(graph1, out1);
    graph1->uid = distinct_uid ? 1004 : 1003;

    ggml_backend_buffer_t buffer1 = ggml_backend_alloc_ctx_tensors(ctx1, backend);
    REQUIRE(buffer1 != nullptr);

    for (int64_t i = 0; i < element_count; ++i) {
        a_data[i]   = static_cast<float>(i % 23) * -0.25f + 5.0f;
        b_data[i]   = static_cast<float>(i % 7) * 0.5f - 1.0f;
        expected[i] = a_data[i] + b_data[i];
    }
    ggml_backend_tensor_set(a1, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(b1, b_data.data(), 0, b_data.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);

    REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
    REQUIRE(cache_stats.graph_program_builds == 1);
    REQUIRE(cache_stats.graph_program_hits == 1);
    REQUIRE(cache_stats.prepared_program_builds == 2);
    REQUIRE(cache_stats.prepared_program_hits == 0);

    std::vector<float> actual(element_count);
    ggml_backend_tensor_get(out1, actual.data(), 0, actual.size() * sizeof(float));
    for (int64_t i = 0; i < element_count; ++i) {
        REQUIRE(actual[i] == expected[i]);
    }

    if (distinct_uid) {
        for (int64_t i = 0; i < element_count; ++i) {
            a_data[i] += 1.0f;
            expected[i] = a_data[i] + b_data[i];
        }
        ggml_backend_tensor_set(a1, a_data.data(), 0, a_data.size() * sizeof(float));
        REQUIRE(ggml_backend_graph_compute(backend, graph1) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(out1, actual.data(), 0, actual.size() * sizeof(float));
        REQUIRE(actual == expected);
        REQUIRE(ggml_backend_hrx_get_cache_stats(backend, &cache_stats));
        REQUIRE(cache_stats.graph_program_builds == 1);
        REQUIRE(cache_stats.graph_program_hits == 2);
        REQUIRE(cache_stats.prepared_program_builds == 2);
        REQUIRE(cache_stats.prepared_program_hits == 1);
    }

    ggml_backend_buffer_free(buffer1);
    ggml_free(ctx1);
    ggml_backend_buffer_free(buffer0);
    ggml_free(ctx0);
    ggml_backend_free(backend);
}

static void run_unsupported_op_fails() {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);

    ggml_init_params params = {};
    params.mem_size         = 256 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * out = ggml_sin(ctx, a);
    REQUIRE(a != nullptr);
    REQUIRE(out != nullptr);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);

    std::vector<float> input(8, 2.0f);
    ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void run_lowtoken_residual_dispatch_checks() {
    struct Case {
        ggml_type type;
        int64_t k, n, tokens;
        int layout;
        int side_use;
        bool alias_weight;
        bool second_add;
        bool fused;
    };

    const Case cases[] = {
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 17408, 5120, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q6_K, 17408, 5120, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 4096, 4096, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 32768, 4096, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 1, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 2, 1, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 3, 1, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 4, 1, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 1, 0, false, false, true },
        { GGML_TYPE_Q6_K, 17408, 5120, 5, 1, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 0, 0, false, true, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 1, 0, false, true, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 0, 1, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 1, 1, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 1, 1, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 1, 2, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 1, 2, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 2, 0, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 2, 0, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 3, 0, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 3, 0, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 1, 0, 0, true, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 5, 1, 0, true, false, false },
        { GGML_TYPE_Q4_K, 3840, 4096, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 4032, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 4096, 8192, 1, 0, 0, false, false, true },
        { GGML_TYPE_F16, 6144, 5120, 1, 0, 0, false, false, true },
        { GGML_TYPE_Q4_K, 6144, 5120, 6, 1, 0, false, false, false },
        { GGML_TYPE_Q4_K, 6144, 5120, 256, 1, 0, false, false, false },
    };
    for (const Case & c : cases) {
        ggml_context * ctx = ggml_init({ 16 * 1024 * 1024, nullptr, true });
        REQUIRE(ctx != nullptr);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, c.type, c.k, c.n + (c.alias_weight ? 64 : 0));
        if (c.alias_weight) {
            weight = ggml_view_2d(ctx, weight, c.k, c.n, weight->nb[1], 0);
        }
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.tokens);
        ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
        ggml_tensor * laid_out = projection;
        if (c.layout == 1) {
            laid_out = ggml_reshape_2d(ctx, projection, c.n, c.tokens);
        } else if (c.layout == 2) {
            laid_out = ggml_reshape_2d(ctx, projection, c.n / 2, c.tokens * 2);
        } else if (c.layout == 3) {
            laid_out = ggml_view_2d(ctx, projection, c.n - 64, c.tokens, projection->nb[1], 0);
        }
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, laid_out->ne[0], laid_out->ne[1]);
        ggml_tensor * output = ggml_add(ctx, laid_out, residual);
        if (c.second_add) {
            output = ggml_add(ctx, output, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, laid_out->ne[0], laid_out->ne[1]));
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, output);
        if (c.side_use != 0) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, c.side_use == 1 ? projection : laid_out, 0.5f));
        }
        ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const ggml::hrx::CommandProgram program = ggml::hrx::build_command_program(
            imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(program.valid());
        REQUIRE(command_program_verifies(program));
        size_t fused = 0, binary = 0;
        bool   fused_second_add = false;
        for (const auto & command : program.commands) {
            const std::string name = kernel_name_for_id(command.kernel.kernel_id);
            binary += name == "loom_libs:ggml_binary_f32";
            const bool tiled_postops  = name == "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32";
            const bool skinny_postops = name == "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32";
            const bool vector_postops = name == "loom_libs:ggml_mul_mat_vector_bias_residual_f32_f32";
            if (!tiled_postops && !skinny_postops && !vector_postops) {
                continue;
            }
            ++fused;
            if (vector_postops) {
                fused_second_add =
                    fused_second_add ||
                    (command.kernel.compile_parameters.at("ggml.matmul.vector_postops.apply_bias") == "1" &&
                     command.kernel.compile_parameters.at("ggml.matmul.vector_postops.apply_residual") == "1");
            } else {
                fused_second_add =
                    fused_second_add || command.kernel.compile_parameters.at("ggml.mul_mat_postops.epilogue") == "3";
            }
            REQUIRE(command.bindings.size() >= 4);
            REQUIRE(command.kernel.integer_parameters.at("token_count") == c.tokens);
            if (vector_postops) {
                REQUIRE(command.kernel.compile_parameters.count("ggml.matmul.vector_postops.weight_format") == 1);
            } else {
                REQUIRE(command.kernel.compile_parameters.count("ggml.mul_mat_postops.weight_format") == 1);
            }
        }
        REQUIRE(fused == (c.fused ? 1 : 0));
        const size_t expected_binary = c.fused ? (c.second_add && !fused_second_add ? 1 : 0) : 1;
        REQUIRE(binary == expected_binary);
        ggml_free(ctx);
    }
}

static void run_lowtoken_bias_dispatch_checks() {
    ggml_context * ctx = ggml_init({ 16 * 1024 * 1024, nullptr, true });
    REQUIRE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 3584, 1024);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3584, 2);
    ggml_tensor * projection = ggml_mul_mat(ctx, weight, input);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * output = ggml_add(ctx, projection, bias);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());

    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
    const ggml::hrx::CommandProgram program = ggml::hrx::build_command_program(
        imported.graph, scheduler.plan(), ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    REQUIRE(program.valid());
    REQUIRE(command_program_verifies(program));

    size_t skinny_bias = 0;
    size_t tiled_bias = 0;
    for (const auto & command : program.commands) {
        const std::string name = kernel_name_for_id(command.kernel.kernel_id);
        skinny_bias += name == "loom_libs:ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32";
        tiled_bias += name == "loom_libs:ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32";
    }
    REQUIRE(skinny_bias == 1);
    REQUIRE(tiled_bias == 0);

    ggml_free(ctx);
}

static void run_gdn_selected_rms_q8_dispatch_checks() {
    ggml_init_params params = {};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    enum {
        UnreadyInput      = 1,
        UnreadyGate       = 2,
        GatherSide        = 4,
        AttentionSide     = 8,
        GdnObserved       = 16,
        GatherObserved    = 32,
        Unaligned         = 64,
        CacheRead         = 128,
        OutputSide        = 256,
        CachedGate        = 512,
        TanhGate          = 1024,
        AttentionReshape  = 2048,
        AttentionObserved = 4096,
    };

    const struct {
        int64_t tokens, heads, sequences;
        int flags;
        ggml_type type;
        bool fused;
    } cases[] = {
        {1,48,1,0,GGML_TYPE_Q4_K,true},
        {2,48,1,0,GGML_TYPE_Q4_K,true},
        {3,48,1,0,GGML_TYPE_Q4_K,true},
        {4,48,1,0,GGML_TYPE_Q4_K,true},
        {5,48,1,0,GGML_TYPE_Q4_K,true},
        {1,24,1,0,GGML_TYPE_Q4_K,true},
        {5,24,1,0,GGML_TYPE_Q4_K,true},
        {1,48,1,0,GGML_TYPE_Q6_K,true},
        {5,48,1,0,GGML_TYPE_Q6_K,true},
        {1,48,1,AttentionReshape,GGML_TYPE_Q4_K,true},
        {5,48,1,AttentionReshape,GGML_TYPE_Q4_K,true},
        {1,48,1,UnreadyInput,GGML_TYPE_Q4_K,false},
        {1,48,1,UnreadyGate,GGML_TYPE_Q4_K,false},
        {5,48,1,UnreadyGate,GGML_TYPE_Q4_K,false},
        {1,48,1,GatherSide,GGML_TYPE_Q4_K,false},
        {5,48,1,AttentionSide,GGML_TYPE_Q4_K,false},
        {1,48,1,GdnObserved,GGML_TYPE_Q4_K,false},
        {1,48,1,GatherObserved,GGML_TYPE_Q4_K,false},
        {1,48,1,AttentionObserved,GGML_TYPE_Q4_K,false},
        {1,48,1,Unaligned,GGML_TYPE_Q4_K,false},
        {5,48,1,CacheRead,GGML_TYPE_Q4_K,false},
        {1,48,1,OutputSide,GGML_TYPE_Q4_K,false},
        {1,48,1,CachedGate,GGML_TYPE_Q4_K,false},
        {5,48,1,TanhGate,GGML_TYPE_Q4_K,false},
        {1,48,1,0,GGML_TYPE_F16,false},
        {6,48,1,0,GGML_TYPE_Q4_K,false},
        {1,48,2,0,GGML_TYPE_Q4_K,false},
        {1,6,1,0,GGML_TYPE_Q4_K,false},
    };
    for (const auto & test : cases) {
        const int64_t width = 128, heads = test.heads, qheads = heads / 3;
        const int64_t hidden = width * (heads + 2 * qheads), channels = width * heads;
        const int64_t state_elements = width * channels;
        const int64_t written = std::min(test.tokens, int64_t{5});
        const size_t state_bytes = state_elements * sizeof(float);
        ggml_tensor * raw = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, test.tokens * test.sequences), 0.5f);
        ggml_tensor * alpha = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, heads, test.tokens * test.sequences), 0.5f);
        ggml_tensor * beta_raw = ggml_scale(ctx, ggml_dup_tensor(ctx, alpha), 0.5f);
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
        ggml_tensor * scale = ggml_dup_tensor(ctx, bias);
        ggml_tensor * gate = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_3d(ctx, alpha, heads, test.tokens, test.sequences), bias)), scale);
        gate = ggml_reshape_4d(ctx, gate, 1, heads, test.tokens, test.sequences);
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta_raw, 1, heads, test.tokens, test.sequences));
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 20);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, test.sequences);
        ggml_tensor * gathered = ggml_get_rows(ctx, cache, ids);
        if (test.flags & GatherObserved) {
            ggml_set_output(gathered);
        }
        ggml_tensor * state = ggml_reshape_4d(ctx, gathered, width, width, heads, test.sequences);
        const auto view = [&](int64_t nheads, size_t offset) {
            return ggml_view_4d(ctx, raw, width, nheads, test.tokens, test.sequences, width * sizeof(float),
                                    hidden * sizeof(float), hidden * test.tokens * sizeof(float), offset);
        };
        ggml_tensor * q = ggml_l2_norm(ctx, view(qheads, 0), 1.e-6f);
        ggml_tensor * k = ggml_l2_norm(ctx, view(qheads, width * qheads * sizeof(float)), 1.e-6f);
        ggml_tensor * v = view(heads, 2 * width * qheads * sizeof(float));
        ggml_tensor * gdn = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, 5);
        if (test.flags & GdnObserved) {
            ggml_set_output(gdn);
        }
        const size_t attention_bytes = channels * test.tokens * test.sequences * sizeof(float);
        ggml_tensor * attention =
            ggml_view_4d(ctx, gdn, width, heads, test.tokens, test.sequences, width * sizeof(float),
                         channels * sizeof(float), channels * test.tokens * sizeof(float), 0);
        if (test.flags & AttentionObserved) {
            ggml_set_output(attention);
        }
        ggml_tensor * snapshots  = ggml_view_3d(ctx, gdn, state_elements, test.sequences, written, state_bytes,
                                                state_bytes * test.sequences, attention_bytes);
        ggml_tensor * target     = ggml_view_3d(ctx, cache, state_elements, test.sequences, written, state_bytes,
                                                4 * state_bytes, (test.flags & Unaligned) ? sizeof(float) : state_bytes);
        ggml_tensor * norm_input = (test.flags & AttentionReshape) ?
                                       ggml_reshape_2d(ctx, attention, width, heads * test.tokens * test.sequences) :
                                       attention;
        ggml_tensor * rms_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_tensor * raw_gate =
            (test.flags & CachedGate) ?
            ggml_view_4d(ctx, cache, width, heads, test.tokens, test.sequences, width * sizeof(float),
                          channels * sizeof(float), channels * test.tokens * sizeof(float), 0) :
                ggml_scale(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, heads, test.tokens, test.sequences),
                           0.5f);
        ggml_tensor * shaped_gate = (test.flags & AttentionReshape) ?
                                        ggml_reshape_2d(ctx, raw_gate, width, heads * test.tokens * test.sequences) :
                                        raw_gate;
        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, norm_input, 1.e-6f), rms_weight);
        ggml_tensor * gated =
            ggml_mul(ctx, norm, (test.flags & TanhGate) ? ggml_tanh(ctx, shaped_gate) : ggml_silu(ctx, shaped_gate));
        ggml_tensor * projection_input = ggml_reshape_2d(ctx, gated, channels, test.tokens * test.sequences);
        ggml_tensor * projection_weight = ggml_new_tensor_2d(ctx, test.type, channels, 4096);
        ggml_tensor * projection = ggml_mul_mat(ctx, projection_weight, projection_input);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        if (!(test.flags & UnreadyInput)) {
            ggml_build_forward_expand(graph, raw);
        }
        ggml_build_forward_expand(graph, alpha);
        ggml_build_forward_expand(graph, beta_raw);
        if (!(test.flags & UnreadyGate)) {
            ggml_build_forward_expand(graph, raw_gate);
        }
        ggml_build_forward_expand(graph, gathered);
        if (test.flags & CacheRead) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, cache, 0.25f));
        }
        ggml_build_forward_expand(graph, ggml_cpy(ctx, snapshots, target));
        ggml_build_forward_expand(graph, ggml_scale(ctx, projection, 0.5f));
        if (test.flags & GatherSide) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, gathered, 0.5f));
        }
        if (test.flags & AttentionSide) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, attention, 0.5f));
        }
        if (test.flags & OutputSide) {
            ggml_build_forward_expand(graph, ggml_scale(ctx, gated, 0.5f));
        }
        auto imported = ggml::hrx::import_ggml_graph(*graph);
        REQUIRE(imported.valid());
        ggml::hrx::DispatchScheduler scheduler;
        REQUIRE(scheduler.schedule_graph(imported.graph, test_dispatch_target()));
        const auto & plan = scheduler.plan();
        REQUIRE(plan.valid());
        const auto fused = std::find_if(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
            return kernel_name_for_id(dispatch.kernel.kernel_id) ==
                "loom_libs:llm_gated_delta_net_f32_wmma_head128_selected_snapshot_projection_rms_gate_q8";
        });
        REQUIRE((fused != plan.dispatches.end()) == test.fused);
        if (test.fused) {
            REQUIRE(fused->bindings.size() == 14);
            require_compile_parameter(*fused, "llm.gated_delta_net.state_row_count", "20");
            REQUIRE(fused->bindings[7].value == imported.graph.values().find_tensor(cache)->id);
            REQUIRE(fused->bindings[10].value == imported.graph.values().find_tensor(ids)->id);
            const auto * output = imported.graph.values().find_tensor(gated);
            const size_t bytes = static_cast<size_t>(test.tokens * heads) * ggml_row_size(GGML_TYPE_Q8_1, width);
            const auto * alternate = plan.metadata.find_alternate_value(output->id, GGML_TYPE_Q8_1, bytes);
            REQUIRE(alternate != nullptr && alternate->alternate_value == fused->bindings.back().value);
            REQUIRE(std::none_of(plan.dispatches.begin(), plan.dispatches.end(), [](const auto & dispatch) {
                const auto name = kernel_name_for_id(dispatch.kernel.kernel_id);
                return name == "loom_libs:ggml_get_rows_f32" || name == "loom_libs:ggml_copy_f32" ||
                       name == "loom_libs:ggml_rmsnorm_gate_f32_publish";
            }));
        }
        const auto commands =
            ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
        REQUIRE(commands.valid());
        if (!(test.flags & UnreadyGate)) {
            REQUIRE(command_program_verifies(commands));
        }
    }
    ggml_free(ctx);
}

static void register_hrx_backend_host_cases(test_runner::Suite & suite) {
    suite.host_case("status", [] { run_status_checks(); });
    suite.host_case("command_plan_metadata", [] { run_command_plan_metadata_checks(); });
    suite.host_case("dispatch_registry", [] { run_dispatch_registry_checks(); });
    suite.host_case("graph_import", [] { run_graph_import_checks(); });
    suite.host_case("graph_import_mixed_backend_boundary", [] { run_graph_import_mixed_backend_boundary_checks(); });
    suite.host_case("graph_view_external_use", [] { run_graph_view_external_use_checks(); });
    suite.host_case("scale_f32_dispatch", [] { run_scale_f32_dispatch_checks(); });
    suite.host_case("scale_add_f32_dispatch", [] { run_scale_add_f32_dispatch_checks(); });
    suite.host_case("rmsnorm_two_binary_dispatch", [] { run_rmsnorm_two_binary_dispatch_checks(); });
    suite.host_case("rmsnorm_lowtoken_striped_dispatch", [] { run_rmsnorm_lowtoken_striped_dispatch_checks(); });
    suite.host_case("cont_f32_dispatch", [] { run_cont_f32_dispatch_checks(); });
    suite.host_case("binary_f32_broadcast_dispatch", [] { run_binary_f32_broadcast_dispatch_checks(); });
    suite.host_case("binary_q8_publication", [] { run_binary_q8_publication_checks(); });
    suite.host_case("graph_snapshot_diagnostics", [] { run_graph_snapshot_diagnostics_checks(); });
    suite.host_case("unmatched_graph_diagnostics", [] { run_unmatched_graph_diagnostics_checks(); });
    suite.host_case("completion_counter_plan", [] { run_completion_counter_plan_checks(); });
    suite.host_case("graph_index", [] { run_graph_index_checks(); });
    suite.host_case("rope_set_rows_dispatch", [] { run_rope_set_rows_dispatch_checks(); });
    suite.host_case("graph_traversal", [] { run_graph_traversal_checks(); });
    suite.host_case("qwen_token_embedding_dispatch", [] { run_qwen_token_embedding_dispatch_checks(); });
    suite.host_case("get_rows_rmsnorm_binary_dispatch", [] { run_get_rows_rmsnorm_binary_dispatch_checks(); });
    suite.host_case("get_rows_scale_dispatch", [] { run_get_rows_scale_dispatch_checks(); });
    suite.host_case("gather_add_dispatch", [] { run_gather_add_dispatch_checks(); });
    suite.host_case("qwen_flash_attention_dispatch", [] { run_qwen_flash_attention_dispatch_checks(); });
    suite.host_case("tiled_pair_matmul_postops_dispatch", [] { run_tiled_pair_matmul_postops_dispatch_checks(); });
    suite.host_case("qwen_attention_postprocess_dispatch", [] { run_qwen_attention_postprocess_dispatch_checks(); });
    suite.host_case("qwen_matmul_dispatch", [] { run_qwen_matmul_dispatch_checks(); });
    suite.host_case("iq1_codebook_matmul_dispatch", [] { run_iq1_codebook_matmul_dispatch_checks(); });
    suite.host_case("iq3_xxs_codebook_matmul_dispatch", [] { run_iq3_xxs_codebook_matmul_dispatch_checks(); });
    suite.host_case("iq2_codebook_resource_reuse", [] { run_iq2_codebook_resource_reuse_checks(); });
    suite.host_case("iq4_matmul_dispatch", [] { run_iq4_matmul_dispatch_checks(); });
    suite.host_case("packed_f16_producer_consumer", [] { run_packed_f16_producer_consumer_checks(); });
    suite.host_case("generic_swiglu_k16_publish", [] { run_generic_swiglu_k16_publish_checks(); });
    suite.host_case("generic_swiglu_k16_publication_edges", [] { run_generic_swiglu_k16_publication_edge_checks(); });
    suite.host_case("tiled_matmul_alternate_publish", [] { run_tiled_matmul_alternate_publish_checks(); });
    suite.host_case("k16_major_preparation_reuse", [] { run_k16_major_preparation_reuse_checks(); });
    suite.host_case("rmsnorm_binary_k16_output", [] { run_rmsnorm_binary_k16_output_checks(); });
    suite.host_case("rmsnorm_binary_f16_output", [] { run_rmsnorm_binary_f16_output_checks(); });
    suite.host_case("swiglu_q8_output", [] { run_swiglu_q8_output_checks(); });
    suite.host_case("rmsnorm_gate_packed_output", [] { run_rmsnorm_gate_packed_output_checks(); });
    suite.host_case("rmsnorm_gate_q8_output", [] { run_rmsnorm_gate_q8_output_checks(); });
    suite.host_case("symmetric_i4_consumer_qualification",
                    [] { run_symmetric_i4_consumer_qualification_checks(); });
    suite.host_case("lowtoken_residual_dispatch", [] { run_lowtoken_residual_dispatch_checks(); });
    suite.host_case("lowtoken_bias_dispatch", [] { run_lowtoken_bias_dispatch_checks(); });
    suite.host_case("generic_ssm_conv_binary_dispatch", [] { run_generic_ssm_conv_binary_dispatch_checks(); });
    suite.host_case("quantized_conv4_dispatch", [] { run_quantized_conv4_dispatch_checks(); });
    suite.host_case("lfm_dconv3_dispatch", [] { run_lfm_dconv3_dispatch_checks(); });
    suite.host_case("gdn_rmsnorm_gate_dispatch", [] { run_gdn_rmsnorm_gate_dispatch_checks(); });
    suite.host_case("gdn_native_projection_pair_dispatch", [] { run_gdn_native_projection_pair_dispatch_checks(); });
    suite.host_case("gdn_selected_snapshot_dispatch", [] { run_gdn_selected_snapshot_dispatch_checks(); });
    suite.host_case("gdn_selected_rms_q8_dispatch", [] { run_gdn_selected_rms_q8_dispatch_checks(); });
    suite.host_case("llama_attention_matmul_dispatch", [] { run_llama_attention_matmul_dispatch_checks(); });
    suite.host_case("quantized_value_projection_dispatch", [] { run_quantized_value_projection_dispatch_checks(); });
    suite.host_case("prefill_value_projection_cache_dispatch",
                    [] { run_prefill_value_projection_cache_dispatch_checks(); });
    suite.host_case("qwen_terminal_q6k_q8_command.tokens1", [] { schedule_qwen_terminal_q6k_q8_command(1); });
    suite.host_case("qwen_terminal_q6k_q8_command.tokens18", [] { schedule_qwen_terminal_q6k_q8_command(18); });
    suite.host_case("qwen_terminal_q6k_vector_command", [] { schedule_qwen_terminal_q6k_vector_command(); });
    suite.host_case("qwen_decode_rmsnorm_publication", [] { run_qwen_decode_rmsnorm_publication_checks(); });
    suite.host_case("qwen_decode_publication_consumer_qualification", [] {
        run_qwen_decode_publication_consumer_qualification_checks();
    });
    suite.host_case("get_rows_q8_1_alternate_command.q4_k",
                    [] { schedule_get_rows_q8_1_alternate_command(GGML_TYPE_Q4_K); });
    suite.host_case("get_rows_q8_1_alternate_command.bf16",
                    [] { schedule_get_rows_q8_1_alternate_command(GGML_TYPE_BF16); });
    suite.host_case("qwen_router_top8_dispatch", [] { run_qwen_router_top8_dispatch_checks(); });
    suite.host_case("common_mul_mat_id_swiglu_dispatch", [] { run_common_mul_mat_id_swiglu_dispatch_checks(); });
    suite.host_case("qwen_routed_gate_up_dispatch", [] { run_qwen_routed_gate_up_dispatch_checks(); });
    suite.host_case("qwen_weighted_reduce_next_rmsnorm_q8_publication", [] {
        run_qwen_weighted_reduce_next_rmsnorm_q8_publication_checks();
    });
    suite.host_case("common_mul_mat_id_postops_dispatch", [] { run_common_mul_mat_id_postops_dispatch_checks(); });
    suite.host_case("alias_value_import", [] { run_alias_value_import_checks(); });
    suite.host_case("multi_dispatch", [] { run_multi_dispatch_checks(); });
    suite.host_case("layout_alias_scheduler_elision", [] { run_layout_alias_scheduler_elision_checks(); });
    suite.host_case("zero_output_scheduler_elision", [] { run_zero_output_scheduler_elision_checks(); });
    suite.host_case("transient_import", [] { run_transient_import_checks(); });
    suite.host_case("graph_view_preserves_shared_output_storage",
                    [] { run_graph_view_preserves_shared_output_storage_checks(); });
    suite.host_case("chained_dispatch_requires_transients", [] { run_chained_dispatch_requires_transients(); });
    suite.host_case("graph_replay_host_staging_is_not_ineligible",
                    [] { run_graph_replay_host_staging_is_not_ineligible(); });
    suite.host_case("multiple_transient_plan", [] { run_multiple_transient_plan_checks(); });
    suite.host_case("disjoint_transient_plan_packing", [] { run_disjoint_transient_plan_packing_checks(); });
    suite.host_case("command_program_kernel_dump", [] { run_command_program_kernel_dump_checks(); });
    suite.host_case("command_shape_hash", [] { run_command_shape_hash_checks(); });
    suite.host_case("graph_program_cache_uid_mismatch", [] { run_graph_program_cache_uid_mismatch_checks(); });
    suite.host_case("graph_match_hash_collision", [] { run_graph_match_hash_collision_checks(); });
    suite.host_case("graph_match_bijection", [] { run_graph_match_bijection_checks(); });
    suite.host_case("validated_graph_uid_match", [] { run_validated_graph_uid_match_checks(); });
    suite.host_case("graph_executor_contract", [] { run_graph_executor_contract_checks(); });
    suite.host_case("loom_async_jit_environment", [] {
        REQUIRE(ggml::hrx::loom_async_jit_enabled_from_environment() == async_jit_expected_from_environment());
    });
}

static void register_hrx_backend_device_cases(test_runner::Suite & suite) {
    suite.device_case("zero_output_device_support", [] { run_zero_output_device_support_checks(); });
    suite.device_case("scale_f32_device_support", [] { run_scale_f32_device_support_checks(); });
    suite.device_case("binary_q8_publication_numerics", [] { run_binary_q8_publication_numerics(); });
    suite.device_case("qwen_expert_table_partition_prefill_512_execution",
                      [] { run_qwen_expert_table_partition_prefill_512_execution(); });
    suite.device_case("add_f32", [] { run_add_f32(); });
    suite.device_case("rope_scale_f32_numerics", [] { run_rope_scale_f32_numerics(); });
    suite.device_case("gather_add_rmsnorm_f32_numerics", [] { run_gather_add_rmsnorm_f32_numerics(); });
    suite.device_case("scale_f32", [] { run_scale_f32(); });
    suite.device_case("scale_f32_inplace", [] { run_scale_f32_inplace(); });
    suite.device_case("two_independent_add_f32", [] { run_two_independent_add_f32(); });
    suite.device_case("chained_add_f32", [] { run_chained_add_f32(); });
    suite.device_case("same_uid_distinct_graph_reuses_graph_program.default",
                      [] { run_same_uid_distinct_graph_reuses_graph_program(); });
    suite.device_case("same_uid_distinct_graph_reuses_graph_program.distinct_uid",
                      [] { run_same_uid_distinct_graph_reuses_graph_program(true); });
    suite.device_case("unsupported_op_fails", [] { run_unsupported_op_fails(); });
}

static void register_hrx_backend_cases(test_runner::Suite & suite) {
    register_hrx_backend_host_cases(suite);
    register_hrx_backend_device_cases(suite);
}

int main(int argc, char ** argv) {
    test_runner::Suite suite(
        test_runner::Config::with_prefix("HRX backend test", "hrx-backend", "GGML_HRX_BACKEND_TEST"));
    register_hrx_backend_cases(suite);

    const bool has_device = ggml_backend_hrx_get_device_count() != 0;
    return suite.run(argc, argv, has_device);
}
