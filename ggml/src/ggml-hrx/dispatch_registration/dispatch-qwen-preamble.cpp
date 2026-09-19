#include "dispatch-qwen-preamble.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenTokenEmbeddingQ4KKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen_token_embedding_q4k");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_1d_or_2d_column(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size == 2048;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_vocabulary_count(int64_t vocabulary_count) {
    return vocabulary_count >= 1 && vocabulary_count <= 262144;
}

struct QwenTokenEmbeddingMatch {
    const Value * token_ids        = nullptr;
    const Value * weight           = nullptr;
    const Value * output           = nullptr;
    int64_t       token_count      = 0;
    int64_t       vocabulary_count = 0;
    int64_t       hidden_size      = 0;

    bool matched() const { return token_ids != nullptr && weight != nullptr && output != nullptr; }
};

static QwenTokenEmbeddingMatch match_qwen_token_embedding(const Graph & graph, const GraphNode * node) {
    QwenTokenEmbeddingMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight    = graph_value(graph, node->inputs[0]);
    const Value * token_ids = graph_value(graph, node->inputs[1]);
    const Value * output    = graph_value(graph, node->output);
    if (weight == nullptr || token_ids == nullptr || output == nullptr) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q4_K || token_ids->type != GGML_TYPE_I32 || output->type != GGML_TYPE_F32) {
        return {};
    }
    if (!weight->contiguous || !token_ids->contiguous || !output->contiguous) {
        return {};
    }
    if (!is_2d(*weight) || !is_1d_or_2d_column(*token_ids) || !is_2d(*output)) {
        return {};
    }

    const int64_t hidden_size      = weight->ne[0];
    const int64_t vocabulary_count = weight->ne[1];
    const int64_t token_count      = token_ids->ne[0];
    if (output->ne[0] != hidden_size || output->ne[1] != token_count) {
        return {};
    }
    if (!is_supported_hidden_size(hidden_size) || !is_supported_vocabulary_count(vocabulary_count) ||
        !is_supported_token_count(token_count)) {
        return {};
    }

    match.token_ids        = token_ids;
    match.weight           = weight;
    match.output           = output;
    match.token_count      = token_count;
    match.vocabulary_count = vocabulary_count;
    match.hidden_size      = hidden_size;
    return match;
}

}  // namespace

static bool match_qwen_token_embedding_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenTokenEmbeddingMatch match = match_qwen_token_embedding(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenTokenEmbeddingQ4KKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("vocabulary_count", match.vocabulary_count);
    dispatch.bindings.push_back({ match.token_ids->id, 0, match.token_ids->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_preamble_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.preamble.token_embedding_q4k",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_token_embedding_dispatch,
    });
}

}  // namespace ggml::hrx
