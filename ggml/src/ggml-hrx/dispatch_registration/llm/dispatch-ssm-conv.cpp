#include "dispatch-ssm-conv.h"

#include "../common/dispatch-binary-common.h"
#include "../common/dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatConv4Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_quantized_f16_wmma_prefill_conv4");
static constexpr KernelCatalogRef kMulMatConv4InteriorKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_quantized_f16_wmma_prefill_conv4_interior");
static constexpr KernelCatalogRef kSsmConvFinishKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_prefill_finish_f32");
static constexpr KernelCatalogRef kSsmConvSnapshotKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_snapshot_window_tail_f32");
static constexpr KernelCatalogRef kSsmConvPrefillKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_prefill_512_wg1024");
static constexpr KernelCatalogRef kSsmConvDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_decode_f32");
static constexpr KernelCatalogRef kSsmConvRollbackKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_rollback_f32");
static constexpr KernelCatalogRef kSsmConvGenericKernel = GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_f32");
static constexpr KernelCatalogRef kSsmConvGenericBinaryKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_binary_f32");
static constexpr KernelCatalogRef kLfmDconv3Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_lfm_dconv3_state_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_f32(const Value * value) {
    return value != nullptr && value->type == GGML_TYPE_F32;
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs.ne[dim] != rhs.ne[dim]) {
            return false;
        }
    }
    return true;
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (value.ne[dim] <= 0 || value.nb[dim] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[dim]);
    }
    return true;
}

static bool supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 32 && hidden_size <= 65536 && hidden_size % 32 == 0;
}

static const Value * root_alias_value(const Graph & graph, const Value * value) {
    for (size_t depth = 0; value != nullptr && value->alias_source.value >= 0 && depth < graph.values().size();
         ++depth) {
        value = graph_value(graph, value->alias_source);
    }
    return value;
}

static bool distinct_storage(const Value & lhs, const Value & rhs) {
    return lhs.storage != rhs.storage;
}

static bool ranges_overlap(const Value & lhs, const Value & rhs) {
    if (lhs.storage != rhs.storage || lhs.byte_count == 0 || rhs.byte_count == 0) {
        return false;
    }
    if (lhs.storage_offset <= rhs.storage_offset) {
        return rhs.storage_offset - lhs.storage_offset < lhs.byte_count;
    }
    return lhs.storage_offset - rhs.storage_offset < rhs.byte_count;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

static void set_compile_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

struct SsmConvCacheUpdate {
    const GraphNode * state_tail_view = nullptr;
    const GraphNode * cache_view      = nullptr;
    const GraphNode * cache_copy      = nullptr;
    const Value *     state_tail      = nullptr;
    const Value *     cache           = nullptr;
    int64_t           source_row      = 0;
};

struct SsmConvPrefillMatch {
    const GraphNode *               concat = nullptr;
    const GraphNode *               ssm    = nullptr;
    const GraphNode *               silu   = nullptr;
    const Value *                   state  = nullptr;
    const Value *                   x      = nullptr;
    const Value *                   filter = nullptr;
    const Value *                   output = nullptr;
    std::vector<SsmConvCacheUpdate> cache_updates;
    int64_t                         hidden_size    = 0;
    int64_t                         token_count    = 0;
    int64_t                         sequence_count = 0;

    bool matched() const {
        return concat != nullptr && ssm != nullptr && silu != nullptr && state != nullptr && x != nullptr &&
               filter != nullptr && output != nullptr && !cache_updates.empty();
    }
};

struct SsmConvCoreMatch {
    const GraphNode * ssm            = nullptr;
    const Value *     window         = nullptr;
    const Value *     filter         = nullptr;
    const Value *     conv_output    = nullptr;
    const GraphNode * unary          = nullptr;
    const GraphNode * binary         = nullptr;
    const Value *     binary_operand = nullptr;
    const Value *     output         = nullptr;
    int64_t           d_conv         = 0;
    int64_t           d_inner        = 0;
    int64_t           token_count    = 0;
    int64_t           sequence_count = 0;
    UnaryKind         unary_op       = UnaryKind::Identity;
    BinaryKind        binary_op      = BinaryKind::Mul;
    bool              binary_lhs     = true;

    bool matched() const {
        return ssm != nullptr && window != nullptr && filter != nullptr && conv_output != nullptr && output != nullptr;
    }

    bool has_unary_fusion() const { return unary != nullptr; }

    bool has_binary_fusion() const { return binary != nullptr; }
};

struct LfmDconv3Match {
    const GraphNode * concat     = nullptr;
    const GraphNode * transpose  = nullptr;
    const GraphNode * ssm        = nullptr;
    const GraphNode * tail_view  = nullptr;
    const GraphNode * cache_view = nullptr;
    const GraphNode * cache_copy = nullptr;
    const Value * state          = nullptr;
    const Value * x              = nullptr;
    const Value * filter         = nullptr;
    const Value * output         = nullptr;
    const Value * cache          = nullptr;
    int64_t token_count          = 0;

    bool matched() const {
        return concat != nullptr && transpose != nullptr && ssm != nullptr && tail_view != nullptr &&
               cache_view != nullptr && cache_copy != nullptr && state != nullptr && x != nullptr &&
               filter != nullptr && output != nullptr && cache != nullptr;
    }
};

static bool lfm_dconv3_disabled() {
    return std::getenv("GGML_HRX_DISABLE_LFM_DCONV3_FUSION") != nullptr;
}

static LfmDconv3Match match_lfm_dconv3(const Graph & graph, const GraphNode * node) {
    LfmDconv3Match match;
    if (lfm_dconv3_disabled() || node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2 ||
        !graph.has_index()) {
        return match;
    }

    const Value * state        = graph_value(graph, node->inputs[0]);
    const Value * x_transposed = graph_value(graph, node->inputs[1]);
    const Value * window       = graph_value(graph, node->output);
    if (!is_f32(state) || !is_f32(x_transposed) || !is_f32(window)) {
        return {};
    }
    const GraphNode * transpose = graph.index().producer(x_transposed->id);
    if (transpose == nullptr || transpose->op != GGML_OP_TRANSPOSE || transpose->inputs.size() != 1) {
        return {};
    }
    const Value * x_layout = graph_value(graph, transpose->inputs[0]);
    const Value * x        = root_alias_value(graph, x_layout);
    if (!is_f32(x_layout) || !x_layout->contiguous || !is_f32(x) || !x->contiguous) {
        return {};
    }

    const int64_t token_count = x_layout->ne[1];
    if ((token_count != 1 && token_count != 64) || !is_shape(*state, 2, 2048, 1, 1) ||
        !is_shape(*x_layout, 2048, token_count, 1, 1) || !is_shape(*x_transposed, token_count, 2048, 1, 1) ||
        !is_shape(*window, token_count + 2, 2048, 1, 1) || state->nb[0] != sizeof(float) ||
        state->nb[1] != 2 * sizeof(float) || x_layout->nb[0] != sizeof(float) ||
        x_layout->nb[1] != 2048 * sizeof(float) || x_transposed->nb[0] != 2048 * sizeof(float) ||
        x_transposed->nb[1] != sizeof(float) || window->nb[0] != sizeof(float) ||
        window->nb[1] != static_cast<size_t>(token_count + 2) * sizeof(float)) {
        return {};
    }

    const GraphNode * ssm       = nullptr;
    const GraphNode * tail_view = nullptr;
    const auto & consumers = graph.index().consumers(window->id);
    if (consumers.size() != 2) {
        return {};
    }
    for (const GraphNode * consumer : consumers) {
        if (consumer != nullptr && consumer->op == GGML_OP_SSM_CONV && ssm == nullptr) {
            ssm = consumer;
        } else if (consumer != nullptr && consumer->op == GGML_OP_VIEW && tail_view == nullptr) {
            tail_view = consumer;
        } else {
            return {};
        }
    }
    if (ssm == nullptr || tail_view == nullptr || ssm->inputs.size() != 2 || ssm->inputs[0] != window->id ||
        tail_view->inputs.size() != 1) {
        return {};
    }
    const Value * filter = graph_value(graph, ssm->inputs[1]);
    const Value * output = graph_value(graph, ssm->output);
    const Value * tail   = graph_value(graph, tail_view->output);
    if (!is_f32(filter) || !is_f32(output) || !is_f32(tail) || !filter->contiguous || !output->contiguous ||
        !is_shape(*filter, 3, 2048, 1, 1) || !is_shape(*output, 2048, token_count, 1, 1) ||
        !is_shape(*tail, 2, 2048, 1, 1) || tail->alias_source != window->id ||
        tail->storage_offset < window->storage_offset ||
        tail->storage_offset - window->storage_offset != static_cast<size_t>(token_count) * sizeof(float)) {
        return {};
    }
    const auto & tail_consumers = graph.index().consumers(tail->id);
    if (tail_consumers.size() != 1 || tail_consumers.front() == nullptr || tail_consumers.front()->op != GGML_OP_CPY ||
        tail_consumers.front()->inputs.size() != 2 || tail_consumers.front()->inputs[0] != tail->id) {
        return {};
    }
    const GraphNode * cache_copy   = tail_consumers.front();
    const Value * cache_target     = graph_value(graph, cache_copy->inputs[1]);
    const Value * cache            = graph_value(graph, cache_copy->output);
    const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
    if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr || cache_view->op != GGML_OP_VIEW ||
        cache_view->inputs.size() != 1 || !same_full_value_range(*cache_target, *cache) ||
        cache->byte_count != static_cast<size_t>(4096) * sizeof(float)) {
        return {};
    }
    if (!distinct_storage(*state, *x) || !distinct_storage(*state, *filter) || !distinct_storage(*state, *output) ||
        !distinct_storage(*state, *cache) || !distinct_storage(*x, *filter) || !distinct_storage(*x, *output) ||
        !distinct_storage(*x, *cache) || !distinct_storage(*filter, *output) || !distinct_storage(*filter, *cache) ||
        !distinct_storage(*output, *cache)) {
        return {};
    }

    match.concat = node;
    match.transpose = transpose;
    match.ssm = ssm;
    match.tail_view = tail_view;
    match.cache_view = cache_view;
    match.cache_copy = cache_copy;
    match.state = state;
    match.x = x;
    match.filter = filter;
    match.output = output;
    match.cache = cache;
    match.token_count = token_count;
    return match;
}

static bool match_lfm_dconv3_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const LfmDconv3Match match = match_lfm_dconv3(context.graph, context.root_node);
    if (!match.matched() || !append_covered_node(context, match.concat, dispatch_match) ||
        !append_covered_node(context, match.ssm, dispatch_match) ||
        !append_covered_node(context, match.tail_view, dispatch_match) ||
        !append_covered_node(context, match.cache_view, dispatch_match) ||
        !append_covered_node(context, match.cache_copy, dispatch_match)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kLfmDconv3Kernel);
    set_compile_parameter(dispatch.kernel, "llm.ssm_conv.lfm_dconv3.n_t", match.token_count);
    dispatch.bindings.push_back({match.state->id, 0, match.state->byte_count});
    dispatch.bindings.push_back({match.x->id, 0, match.x->byte_count});
    dispatch.bindings.push_back({match.filter->id, 0, match.filter->byte_count});
    dispatch.bindings.push_back({match.output->id, 0, match.output->byte_count});
    dispatch.bindings.push_back({match.cache->id, 0, match.cache->byte_count});
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static SsmConvPrefillMatch match_ssm_conv_prefill(const Graph & graph, const GraphNode * node) {
    SsmConvPrefillMatch match;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * state        = graph_value(graph, node->inputs[0]);
    const Value * x_transposed = graph_value(graph, node->inputs[1]);
    const Value * window       = graph_value(graph, node->output);
    if (!is_f32(state) || !is_f32(x_transposed) || !is_f32(window)) {
        return {};
    }

    const GraphNode * transpose = graph.index().producer(x_transposed->id);
    if (transpose == nullptr || transpose->op != GGML_OP_TRANSPOSE || transpose->inputs.size() != 1) {
        return {};
    }
    const Value * x_layout = graph_value(graph, transpose->inputs[0]);
    const Value * x        = root_alias_value(graph, x_layout);
    if (!is_f32(x_layout) || !x_layout->contiguous || !is_f32(x) || !x->contiguous) {
        return {};
    }

    const int64_t hidden_size    = x_layout->ne[0];
    const int64_t token_count    = x_layout->ne[1];
    const int64_t sequence_count = x_layout->ne[2];
    if (!supported_hidden_size(hidden_size) || token_count < 1 || token_count > 512 || sequence_count < 1 ||
        sequence_count > 3 || !is_shape(*x_layout, hidden_size, token_count, sequence_count, 1) ||
        !is_shape(*x_transposed, token_count, hidden_size, sequence_count, 1) ||
        !is_shape(*state, 3, hidden_size, sequence_count, 1) ||
        !is_shape(*window, token_count + 3, hidden_size, sequence_count, 1)) {
        return {};
    }
    if (x_layout->nb[0] != sizeof(float) || x_layout->nb[1] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_layout->nb[2] != static_cast<size_t>(hidden_size * token_count) * sizeof(float) ||
        x_transposed->nb[0] != static_cast<size_t>(hidden_size) * sizeof(float) ||
        x_transposed->nb[1] != sizeof(float) ||
        x_transposed->nb[2] != static_cast<size_t>(hidden_size * token_count) * sizeof(float) ||
        state->nb[0] != sizeof(float) || state->nb[1] != 3 * sizeof(float) ||
        state->nb[2] != static_cast<size_t>(3 * hidden_size) * sizeof(float) || window->nb[0] != sizeof(float) ||
        window->nb[1] != static_cast<size_t>(token_count + 3) * sizeof(float) ||
        window->nb[2] != static_cast<size_t>((token_count + 3) * hidden_size) * sizeof(float)) {
        return {};
    }

    const std::vector<const GraphNode *> & window_consumers = graph.index().consumers(window->id);
    if (window_consumers.size() < 2 || window_consumers.size() > 6) {
        return {};
    }
    std::vector<const GraphNode *> state_tail_views;
    const GraphNode *              ssm = nullptr;
    for (const GraphNode * consumer : window_consumers) {
        if (consumer != nullptr && consumer->op == GGML_OP_VIEW) {
            state_tail_views.push_back(consumer);
        } else if (consumer != nullptr && consumer->op == GGML_OP_SSM_CONV) {
            if (ssm != nullptr) {
                return {};
            }
            ssm = consumer;
        } else {
            return {};
        }
    }
    if (state_tail_views.empty() || state_tail_views.size() > 5 || ssm == nullptr || ssm->inputs.size() != 2 ||
        ssm->inputs[0] != window->id) {
        return {};
    }

    const Value * filter     = graph_value(graph, ssm->inputs[1]);
    const Value * ssm_output = graph_value(graph, ssm->output);
    if (!is_f32(filter) || !is_f32(ssm_output) || !filter->contiguous || !ssm_output->contiguous ||
        !is_shape(*filter, 4, hidden_size, 1, 1) ||
        !is_shape(*ssm_output, hidden_size, token_count, sequence_count, 1) || filter->nb[0] != sizeof(float) ||
        filter->nb[1] != 4 * sizeof(float)) {
        return {};
    }

    std::vector<SsmConvCacheUpdate> cache_updates;
    for (const GraphNode * state_tail_view : state_tail_views) {
        if (state_tail_view == nullptr || state_tail_view->inputs.size() != 1) {
            return {};
        }
        const Value * state_tail = graph_value(graph, state_tail_view->output);
        if (!is_f32(state_tail) || state_tail->alias_source != window->id ||
            state_tail->storage_offset < window->storage_offset ||
            (state_tail->storage_offset - window->storage_offset) % sizeof(float) != 0 ||
            !is_shape(*state_tail, 3, hidden_size, sequence_count, 1)) {
            return {};
        }
        const int64_t source_row =
            static_cast<int64_t>((state_tail->storage_offset - window->storage_offset) / sizeof(float));
        if (source_row < 0 || source_row > token_count) {
            return {};
        }
        const std::vector<const GraphNode *> & tail_consumers = graph.index().consumers(state_tail->id);
        if (tail_consumers.size() != 1 || tail_consumers.front() == nullptr ||
            tail_consumers.front()->op != GGML_OP_CPY || tail_consumers.front()->inputs.size() != 2 ||
            tail_consumers.front()->inputs[0] != state_tail->id) {
            return {};
        }
        const GraphNode * cache_copy   = tail_consumers.front();
        const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
        const Value *     cache        = graph_value(graph, cache_copy->output);
        const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
        if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr || cache_view->op != GGML_OP_VIEW ||
            cache_view->inputs.size() != 1 || !same_full_value_range(*cache_target, *cache) ||
            cache->byte_count != static_cast<size_t>(3 * hidden_size * sequence_count) * sizeof(float)) {
            return {};
        }
        cache_updates.push_back({ state_tail_view, cache_view, cache_copy, state_tail, cache, source_row });
    }

    std::sort(cache_updates.begin(), cache_updates.end(),
              [](const auto & lhs, const auto & rhs) { return lhs.cache->storage_offset < rhs.cache->storage_offset; });
    for (size_t slot = 0; slot < cache_updates.size(); ++slot) {
        const int64_t expected_source_row = std::max<int64_t>(0, token_count - static_cast<int64_t>(slot));
        if (cache_updates[slot].source_row != expected_source_row ||
            cache_updates[slot].cache->storage != cache_updates.front().cache->storage) {
            return {};
        }
        for (size_t prior = 0; prior < slot; ++prior) {
            if (ranges_overlap(*cache_updates[slot].cache, *cache_updates[prior].cache)) {
                return {};
            }
        }
    }

    const std::vector<const GraphNode *> & ssm_consumers = graph.index().consumers(ssm_output->id);
    if (ssm_consumers.size() != 1 || ssm_consumers.front() == nullptr || ssm_consumers.front()->op != GGML_OP_UNARY ||
        ssm_consumers.front()->inputs.size() != 1) {
        return {};
    }
    const GraphNode *   silu   = ssm_consumers.front();
    const UnaryParams * unary  = op_params_as<UnaryParams>(silu->params);
    const Value *       output = graph_value(graph, silu->output);
    if (unary == nullptr || unary->op != UnaryKind::Silu || !is_f32(output) || !output->contiguous ||
        !is_shape(*output, hidden_size, token_count, sequence_count, 1)) {
        return {};
    }

    if (!distinct_storage(*state, *x) || !distinct_storage(*state, *filter) || !distinct_storage(*x, *filter) ||
        (!distinct_storage(*output, *x) && !same_full_value_range(*output, *x)) || !distinct_storage(*output, *state) ||
        !distinct_storage(*output, *filter)) {
        return {};
    }
    for (const SsmConvCacheUpdate & update : cache_updates) {
        if (!distinct_storage(*state, *update.cache) || !distinct_storage(*x, *update.cache) ||
            !distinct_storage(*filter, *update.cache) || !distinct_storage(*output, *update.cache)) {
            return {};
        }
    }

    match.concat         = node;
    match.ssm            = ssm;
    match.silu           = silu;
    match.state          = state;
    match.x              = x;
    match.filter         = filter;
    match.output         = output;
    match.cache_updates  = std::move(cache_updates);
    match.hidden_size    = hidden_size;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    return match;
}

static SsmConvCoreMatch match_ssm_conv_core(const Graph & graph, const GraphNode * node) {
    SsmConvCoreMatch match;
    if (node == nullptr || node->op != GGML_OP_SSM_CONV || node->inputs.size() != 2) {
        return match;
    }

    const Value * window      = graph_value(graph, node->inputs[0]);
    const Value * filter      = graph_value(graph, node->inputs[1]);
    const Value * conv_output = graph_value(graph, node->output);
    if (!is_f32(window) || !is_f32(filter) || !is_f32(conv_output) || !packed_f32_layout(*window) ||
        !packed_f32_layout(*filter) || !packed_f32_layout(*conv_output)) {
        return {};
    }

    const int64_t d_conv         = filter->ne[0];
    const int64_t d_inner        = filter->ne[1];
    const int64_t token_count    = window->ne[0] - d_conv + 1;
    const int64_t sequence_count = window->ne[2];
    if (d_conv < 1 || d_conv > 16 || !supported_hidden_size(d_inner) || token_count < 1 || token_count > 512 ||
        sequence_count < 1 || sequence_count > 4 || window->ne[1] != d_inner || window->ne[3] != 1 ||
        filter->ne[2] != 1 || filter->ne[3] != 1 || !is_shape(*conv_output, d_inner, token_count, sequence_count, 1) ||
        ranges_overlap(*window, *conv_output) || ranges_overlap(*filter, *conv_output)) {
        return {};
    }

    match.ssm            = node;
    match.window         = window;
    match.filter         = filter;
    match.conv_output    = conv_output;
    match.output         = conv_output;
    match.d_conv         = d_conv;
    match.d_inner        = d_inner;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    return match;
}

static bool try_match_unary_fusion(const Graph & graph, SsmConvCoreMatch & match) {
    if (!graph.has_index()) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(match.conv_output->id);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != GGML_OP_UNARY ||
        consumers.front()->inputs.size() != 1) {
        return false;
    }

    const GraphNode *   unary  = consumers.front();
    const UnaryParams * params = op_params_as<UnaryParams>(unary->params);
    const Value *       output = graph_value(graph, unary->output);
    if (params == nullptr || !unary_kind_supported(params->op) || output == nullptr || output->type != GGML_TYPE_F32 ||
        !packed_f32_layout(*output) || !same_shape(*output, *match.conv_output) ||
        ranges_overlap(*match.window, *output) || ranges_overlap(*match.filter, *output)) {
        return false;
    }

    match.unary    = unary;
    match.output   = output;
    match.unary_op = params->op;
    return true;
}

static bool try_match_binary_fusion(const Graph & graph, SsmConvCoreMatch & match) {
    if (!graph.has_index()) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(match.conv_output->id);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->inputs.size() != 2) {
        return false;
    }

    const GraphNode *    binary = consumers.front();
    const CommonBinaryMatch binary_match =
        common_match_binary_consumer(binary, match.conv_output->id, kCommonArithmeticBinaryKinds);
    if (!binary_match.matched()) {
        return false;
    }

    const Value * operand = graph_value(graph, binary_match.operand);
    const Value * output  = graph_value(graph, binary->output);
    if (!is_f32(operand) || output == nullptr || output->type != GGML_TYPE_F32 || !packed_f32_layout(*operand) ||
        !packed_f32_layout(*output) || !same_shape(*operand, *match.conv_output) ||
        !same_shape(*output, *match.conv_output) || ranges_overlap(*match.window, *output) ||
        ranges_overlap(*match.filter, *output) || ranges_overlap(*operand, *output)) {
        return false;
    }

    match.binary         = binary;
    match.binary_operand = operand;
    match.output         = output;
    match.binary_op      = binary_match.kind;
    match.binary_lhs     = binary_match.producer_is_lhs;
    return true;
}

static void set_generic_ssm_conv_parameters(KernelSpecialization & kernel, const SsmConvCoreMatch & match) {
    set_compile_parameter(kernel, "llm.ssm_conv.generic.d_conv", match.d_conv);
    set_compile_parameter(kernel, "llm.ssm_conv.generic.d_inner", match.d_inner);
    kernel.integer_parameters.emplace("n_t", match.token_count);
    kernel.integer_parameters.emplace("n_s", match.sequence_count);
    kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    set_compile_parameter(kernel, "llm.ssm_conv.generic.unary_op", unary_kind_config_value(match.unary_op));
    set_compile_parameter(kernel, "llm.ssm_conv.generic.workgroup_size", 256);
}

static bool append_ssm_conv_covered_nodes(const DispatchMatchContext & context,
                                          const SsmConvPrefillMatch & match,
                                          DispatchMatch & dispatch_match) {
    if (!append_covered_node(context, match.concat, dispatch_match) ||
        !append_covered_node(context, match.ssm, dispatch_match) ||
        !append_covered_node(context, match.silu, dispatch_match)) {
        return false;
    }
    for (const SsmConvCacheUpdate & update : match.cache_updates) {
        if (!append_covered_node(context, update.state_tail_view, dispatch_match) ||
            !append_covered_node(context, update.cache_view, dispatch_match) ||
            !append_covered_node(context, update.cache_copy, dispatch_match)) {
            return false;
        }
    }
    return true;
}

static bool match_ssm_conv_prefill_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const SsmConvPrefillMatch match = match_ssm_conv_prefill(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    if (!append_ssm_conv_covered_nodes(context, match, dispatch_match)) {
        return false;
    }

    const CommandPlanGeneratedResource * edges =
        context.plan.metadata.find_generated_resource(match.x->id, GeneratedResourceRole::Conv4Edges);
    if (edges != nullptr) {
        Dispatch finish;
        finish.kernel = make_kernel_specialization(kSsmConvFinishKernel);
        set_compile_parameter(finish.kernel, "llm.ssm_conv.generic.d_inner", match.hidden_size);
        finish.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        finish.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
        finish.bindings.push_back({ edges->generated_value, 0, edges->byte_count });
        finish.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        const Value * cache = match.cache_updates.front().cache;
        finish.bindings.push_back({ cache->id, 0, cache->byte_count });
        dispatch_match.dispatches.push_back(std::move(finish));
        return true;
    }

    const bool optimized_prefill = match.token_count == 512 && match.sequence_count == 1 &&
                                   match.cache_updates.size() == 1 && match.hidden_size >= 8192 &&
                                   match.hidden_size <= 10240;
    if (!optimized_prefill) {
        Dispatch   ssm_dispatch;
        const bool decode = match.token_count == 1 && match.sequence_count == 1 && match.cache_updates.size() == 1;
        if (decode) {
            ssm_dispatch.kernel = make_kernel_specialization(kSsmConvDecodeKernel);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.decode.d_inner", match.hidden_size);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.decode.workgroup_size", 256);
        } else {
            ssm_dispatch.kernel = make_kernel_specialization(kSsmConvRollbackKernel);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.d_inner", match.hidden_size);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.n_t", match.token_count);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.n_s", match.sequence_count);
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.cache_count", match.cache_updates.size());
            set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.rollback.workgroup_size", 256);
        }
        ssm_dispatch.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        ssm_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
        ssm_dispatch.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
        ssm_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        if (decode) {
            const Value * cache = match.cache_updates.front().cache;
            ssm_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });
        } else {
            for (size_t slot = 0; slot < 5; ++slot) {
                const Value * cache = match.cache_updates[std::min(slot, match.cache_updates.size() - 1)].cache;
                ssm_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });
            }
        }
        dispatch_match.dispatches.push_back(std::move(ssm_dispatch));
        return true;
    }

    const size_t  snapshot_bytes = static_cast<size_t>(64 * match.hidden_size) * sizeof(float);
    const ValueId snapshot       = context.next_plan_value;
    dispatch_match.transients.push_back({ snapshot, "llm.ssm_conv.window_snapshot", snapshot_bytes, 256 });

    Dispatch snapshot_dispatch;
    snapshot_dispatch.kernel = make_kernel_specialization(kSsmConvSnapshotKernel);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.d_conv", 4);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.d_inner", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.n_t", match.token_count);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.n_s", match.sequence_count);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.state_row_stride", 1);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.state_channel_stride", 3);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.x_row_stride", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.dst_row_stride", match.hidden_size);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.cache_row_stride", 1);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.cache_channel_stride", 3);
    set_compile_parameter(snapshot_dispatch.kernel, "llm.ssm_conv.snapshot.workgroup_size", 256);
    snapshot_dispatch.bindings.push_back({ match.state->id, 0, match.state->byte_count });
    snapshot_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    snapshot_dispatch.bindings.push_back({ snapshot, 0, snapshot_bytes });
    const Value * cache = match.cache_updates.front().cache;
    snapshot_dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });

    Dispatch ssm_dispatch;
    ssm_dispatch.kernel = make_kernel_specialization(kSsmConvPrefillKernel);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.d_conv", 4);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.d_inner", match.hidden_size);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.n_t", match.token_count);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.n_s", match.sequence_count);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.state_row_stride", match.hidden_size);
    set_compile_parameter(ssm_dispatch.kernel, "llm.ssm_conv.prefill.x_row_stride", match.hidden_size);
    ssm_dispatch.bindings.push_back({ snapshot, 0, snapshot_bytes });
    ssm_dispatch.bindings.push_back({ match.x->id, 0, match.x->byte_count });
    ssm_dispatch.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
    ssm_dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.dispatches.push_back(std::move(snapshot_dispatch));
    dispatch_match.dispatches.push_back(std::move(ssm_dispatch));
    return true;
}

static bool match_ssm_conv_generic_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    SsmConvCoreMatch match = match_ssm_conv_core(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    try_match_binary_fusion(context.graph, match) || try_match_unary_fusion(context.graph, match);

    if (!append_covered_node(context, match.ssm, dispatch_match)) {
        return false;
    }
    if (match.has_binary_fusion() && !append_covered_node(context, match.binary, dispatch_match)) {
        return false;
    }
    if (match.has_unary_fusion() && !append_covered_node(context, match.unary, dispatch_match)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel =
        make_kernel_specialization(match.has_binary_fusion() ? kSsmConvGenericBinaryKernel : kSsmConvGenericKernel);
    set_generic_ssm_conv_parameters(dispatch.kernel, match);
    dispatch.bindings.push_back({ match.window->id, 0, match.window->byte_count });
    dispatch.bindings.push_back({ match.filter->id, 0, match.filter->byte_count });
    if (match.has_binary_fusion()) {
        common_set_binary_compile_parameters(dispatch.kernel, "llm.ssm_conv.generic.binary_op",
                                             "llm.ssm_conv.generic.binary_lhs", match.binary_op, match.binary_lhs);
        dispatch.bindings.push_back({ match.binary_operand->id, 0, match.binary_operand->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_mul_mat_conv4_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatMatch mat =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatConv4Kernel, false);
    if (!mat.matched() || !context.graph.has_index() || mat.token_count != 512 ||
        (mat.weight->type != GGML_TYPE_Q4_K && mat.weight->type != GGML_TYPE_Q6_K) ||
        mat.weight->alias_source.value >= 0 || mat.input_size % 256 != 0 || mat.output_size % 64 != 0 ||
        mat.output_size / 64 < 32 ||
        !common_mul_mat_uses_k16_major_f16(mat.weight_format, mat.input_size, mat.output_size, mat.token_count)) {
        return false;
    }

    const Value * value = mat.output;
    std::vector<const GraphNode *> layouts;
    const GraphNode * concat = nullptr;
    while (value != nullptr && value->kind == ValueKind::Transient) {
        const auto & consumers = context.graph.index().consumers(value->id);
        if (consumers.size() != 1 || consumers.front() == nullptr) {
            return false;
        }
        const GraphNode * consumer = consumers.front();
        if (consumer->op == GGML_OP_CONCAT) {
            concat = consumer;
            break;
        }
        if (!is_layout_alias_node(context.graph, *consumer)) {
            return false;
        }
        layouts.push_back(consumer);
        value = graph_value(context.graph, consumer->output);
    }
    const SsmConvPrefillMatch conv = match_ssm_conv_prefill(context.graph, concat);
    if (!conv.matched() || conv.x->id != mat.output->id || conv.sequence_count != 1 || conv.token_count != 512 ||
        conv.cache_updates.size() != 1) {
        return false;
    }
    const Value * window = graph_value(context.graph, conv.concat->output);
    const Value * intermediate = graph_value(context.graph, conv.ssm->output);
    if (window->kind != ValueKind::Transient || intermediate->kind != ValueKind::Transient) {
        return false;
    }
    bool deferred_state = false;
    for (const Value * input : {conv.state, conv.filter}) {
        const GraphNode * producer = context.graph.index().producer(input->id);
        size_t index = 0;
        if (producer != nullptr &&
            (!context.graph.index().node_index(producer, index) || !context.covered_nodes[index])) {
            if (input == conv.filter) {
                return false;
            }
            deferred_state = true;
        }
    }
    if (deferred_state && (conv.hidden_size < 8192 || conv.hidden_size > 10240)) {
        return false;
    }
    const Value * cache = conv.cache_updates.front().cache;
    for (const Value * output : {conv.output, cache}) {
        if (!distinct_storage(*output, *mat.input) || !distinct_storage(*output, *mat.weight)) {
            return false;
        }
    }
    if (!append_covered_node(context, context.root_node, dispatch_match) ||
        (!deferred_state && !append_ssm_conv_covered_nodes(context, conv, dispatch_match))) {
        return false;
    }
    for (const GraphNode * layout : layouts) {
        if (!append_covered_node(context, layout, dispatch_match)) {
            return false;
        }
    }

    DispatchBinding activation;
    if (!common_prepare_k16_major_f16_input(context, *mat.input, mat.input_size, mat.token_count, dispatch_match,
                                            activation)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(deferred_state ? kMulMatConv4InteriorKernel : kMulMatConv4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", mat.token_count);
    set_compile_parameter(dispatch.kernel, "ggml.mul_mat.input_size", mat.input_size);
    set_compile_parameter(dispatch.kernel, "ggml.mul_mat.output_size", mat.output_size);
    set_compile_parameter(dispatch.kernel, "ggml.mul_mat.weight_format", mat.weight->type == GGML_TYPE_Q4_K ? 4 : 6);
    dispatch.bindings.push_back(activation);
    if (mat.weight->type == GGML_TYPE_Q4_K) {
        dispatch.bindings.push_back({mat.weight->id, 0, mat.weight->byte_count, kQ4KPackedK256Row64Layout,
                                     mat.weight->type, mat.input_size, mat.output_size, mat.weight->byte_count});
    } else {
        dispatch.bindings.push_back({mat.weight->id, 0, mat.weight->byte_count});
    }
    if (!deferred_state) {
        dispatch.bindings.push_back({conv.state->id, 0, conv.state->byte_count});
    }
    dispatch.bindings.push_back({conv.filter->id, 0, conv.filter->byte_count});
    dispatch.bindings.push_back({conv.output->id, 0, conv.output->byte_count});
    if (deferred_state) {
        const size_t edge_bytes = static_cast<size_t>(6 * conv.hidden_size) * sizeof(float);
        const ValueId edges(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()));
        dispatch_match.transients.push_back({ edges, "llm.ssm_conv.projection_edges", edge_bytes, 256 });
        Status status;
        if (!dispatch_match.metadata.append_generated_resource(
                { mat.output->id, GeneratedResourceRole::Conv4Edges, edges, edge_bytes, {} }, status)) {
            dispatch_match.status.append(status);
            return false;
        }
        dispatch.bindings.push_back({ edges, 0, edge_bytes });
    } else {
        dispatch.bindings.push_back({cache->id, 0, cache->byte_count});
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_llm_ssm_conv_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.ssm_conv.quantized_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        306,
        DispatchSource::Llm,
        match_mul_mat_conv4_dispatch,
    });
    registry.add({
        "llm.ssm_conv.dconv4_silu",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Llm,
        match_ssm_conv_prefill_dispatch,
    });
    registry.add({
        "llm.ssm_conv.lfm_dconv3",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        201,
        DispatchSource::Llm,
        match_lfm_dconv3_dispatch,
    });
    registry.add({
        "llm.ssm_conv.generic_f32",
        GGML_OP_SSM_CONV,
        DispatchMatchKind::Fused,
        50,
        DispatchSource::Llm,
        match_ssm_conv_generic_dispatch,
    });
}

}  // namespace ggml::hrx
