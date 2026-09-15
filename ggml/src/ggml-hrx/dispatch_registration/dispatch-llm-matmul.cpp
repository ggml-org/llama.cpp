#include "dispatch-llm-matmul.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kLlmDenseLinearQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_f16_wmma");
static constexpr KernelCatalogRef kLlmDenseLinearQ6KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q6k_f16_wmma");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

enum class LlmDenseMatmulRoute {
    Q4K,
    Q6K,
};

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct LlmDenseMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

static LlmDenseMatmulMatch match_llm_dense_matmul(const Graph &       graph,
                                                  const GraphNode *   node,
                                                  LlmDenseMatmulRoute route) {
    LlmDenseMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_llm_prefill_query_length(kActiveLlmMoeDispatchProfile, token_count) ||
        !is_supported_dense_input_size(input_size) || !is_supported_dense_output_size(output_size)) {
        return {};
    }

    if (route == LlmDenseMatmulRoute::Q4K && weight->type == GGML_TYPE_Q4_K) {
        match.kernel = kLlmDenseLinearQ4KF16WmmaKernel;
    } else if (route == LlmDenseMatmulRoute::Q6K && weight->type == GGML_TYPE_Q6_K) {
        match.kernel = kLlmDenseLinearQ6KF16WmmaKernel;
    } else {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

}  // namespace

static void build_llm_dense_matmul_dispatch(const LlmDenseMatmulMatch & match,
                                            DispatchMatch &             dispatch_match,
                                            size_t                      root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "0");
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_llm_dense_q4k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, LlmDenseMatmulRoute::Q4K);
    if (!match.matched()) {
        return false;
    }
    build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_llm_dense_q6k_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LlmDenseMatmulMatch match =
        match_llm_dense_matmul(context.graph, context.root_node, LlmDenseMatmulRoute::Q6K);
    if (!match.matched()) {
        return false;
    }
    build_llm_dense_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

void register_llm_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.matmul.dense_q4k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q4k_dispatch,
    });
    registry.add({
        "llm.matmul.dense_q6k_f16_wmma",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Llm,
        match_llm_dense_q6k_dispatch,
    });
}

}  // namespace ggml::hrx
