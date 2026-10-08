#include "dispatch-gated-delta-net.h"

#include "../common/dispatch-mul-mat-common.h"
#include "../common/dispatch-rmsnorm.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGatedDeltaNetProjectionEpilogueKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_projection_epilogue_f32");
static constexpr KernelCatalogRef kGatedDeltaNetPrefillKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128");
static constexpr KernelCatalogRef kGatedDeltaNetRmsNormGateKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_rmsnorm_gate");
static constexpr KernelCatalogRef kRmsNormGateKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_gate_f32_publish");
static constexpr KernelCatalogRef kGatedDeltaNetPrefillProjectionEpilogueKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_projection_epilogue");
static constexpr KernelCatalogRef kGatedDeltaNetInplaceKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_inplace");
static constexpr KernelCatalogRef kGatedDeltaNetInplaceProjectionEpilogueKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_inplace_projection_epilogue");
static constexpr KernelCatalogRef kGatedDeltaNetSnapshotKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_snapshot");
static constexpr KernelCatalogRef kGatedDeltaNetSnapshotProjectionEpilogueKernel = GGML_HRX_KERNEL_REF(
    "loom_libs", "llm_gated_delta_net_f32_wmma_head128_snapshot_projection_epilogue");
static constexpr KernelCatalogRef kGatedDeltaNetSelectedSnapshotKernel = GGML_HRX_KERNEL_REF(
    "loom_libs", "llm_gated_delta_net_f32_wmma_head128_selected_snapshot_projection_epilogue");
static constexpr KernelCatalogRef kGatedDeltaNetSelectedRmsQ8Kernel = GGML_HRX_KERNEL_REF(
    "loom_libs", "llm_gated_delta_net_f32_wmma_head128_selected_snapshot_projection_rms_gate_q8");
static constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_adjacent_dual_wmma");
static constexpr KernelCatalogRef kMulMatDualQ4F32DecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_dual_q4_f32_decode");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool is_f32(const Value * value) {
    return value != nullptr && value->type == GGML_TYPE_F32;
}

static bool distinct_storage(const Value & lhs, const Value & rhs) {
    return lhs.storage != rhs.storage;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static bool has_unary_op(const GraphNode * node, UnaryKind op) {
    const UnaryParams * params = node != nullptr ? op_params_as<UnaryParams>(node->params) : nullptr;
    return node != nullptr && node->op == GGML_OP_UNARY && params != nullptr && params->op == op;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

struct GatedDeltaNetMatch {
    std::vector<const GraphNode *> covered;
    const Value *                  alpha_raw      = nullptr;
    const Value *                  beta_raw       = nullptr;
    const Value *                  bias           = nullptr;
    const Value *                  a_scale        = nullptr;
    const Value *                  gate           = nullptr;
    const Value *                  gate_flat      = nullptr;
    const Value *                  beta           = nullptr;
    const Value *                  raw_q          = nullptr;
    const Value *                  raw_k          = nullptr;
    const Value *                  v              = nullptr;
    const Value *                  state          = nullptr;
    const Value *                  gdn_output     = nullptr;
    const Value *                  attention      = nullptr;
    const Value *                  new_state      = nullptr;
    const Value *                  cache          = nullptr;
    int64_t                        width          = 0;
    int64_t                        q_head_count   = 0;
    int64_t                        head_count     = 0;
    int64_t                        token_count    = 0;
    int64_t                        sequence_count = 0;
    int64_t                        snapshot_count = 0;
    float                          l2_epsilon     = 0.0f;

    bool matched() const {
        return !covered.empty() && gate != nullptr && beta != nullptr && raw_q != nullptr && raw_k != nullptr &&
               v != nullptr && state != nullptr && gdn_output != nullptr && new_state != nullptr && cache != nullptr;
    }

    bool has_projection_epilogue() const {
        return alpha_raw != nullptr && beta_raw != nullptr && bias != nullptr && a_scale != nullptr &&
               gate_flat != nullptr;
    }
};

struct GatedDeltaNetProjectionPairMatch {
    const GraphNode * first_node    = nullptr;
    const GraphNode * second_node   = nullptr;
    const Value *     input         = nullptr;
    const Value *     first_weight  = nullptr;
    const Value *     second_weight = nullptr;
    const Value *     first_output  = nullptr;
    const Value *     second_output = nullptr;
    ValueId           activation;
    int64_t           input_size  = 0;
    int64_t           output_size = 0;
    int64_t           token_count = 0;
    bool              native_f32 = false;

    bool matched() const {
        return first_node != nullptr && second_node != nullptr && input != nullptr && first_weight != nullptr &&
               second_weight != nullptr && first_output != nullptr && second_output != nullptr;
    }
};

static GatedDeltaNetMatch match_gated_delta_net(const Graph & graph, const GraphNode * node) {
    GatedDeltaNetMatch match;
    if (node == nullptr || node->op != GGML_OP_L2_NORM || node->inputs.size() != 1 || !graph.has_index()) {
        return match;
    }

    const Value * raw_q  = graph_value(graph, node->inputs[0]);
    const Value * q_norm = graph_value(graph, node->output);
    if (!is_f32(raw_q) || !is_f32(q_norm)) {
        return {};
    }
    const std::vector<const GraphNode *> & q_consumers = graph.index().consumers(q_norm->id);
    if (q_consumers.size() != 1 || q_consumers.front() == nullptr ||
        q_consumers.front()->op != GGML_OP_GATED_DELTA_NET || q_consumers.front()->inputs.size() != 6 ||
        q_consumers.front()->inputs[0] != q_norm->id) {
        return {};
    }
    const GraphNode * gdn = q_consumers.front();

    const Value *     k_norm      = graph_value(graph, gdn->inputs[1]);
    const Value *     v           = graph_value(graph, gdn->inputs[2]);
    const Value *     gate        = graph_value(graph, gdn->inputs[3]);
    const Value *     beta        = graph_value(graph, gdn->inputs[4]);
    const Value *     state       = graph_value(graph, gdn->inputs[5]);
    const Value *     gdn_output  = graph_value(graph, gdn->output);
    const GraphNode * k_norm_node = k_norm != nullptr ? producer_with_op(graph, k_norm->id, GGML_OP_L2_NORM) : nullptr;
    const Value *     raw_k       = k_norm_node != nullptr && k_norm_node->inputs.size() == 1 ?
                                        graph_value(graph, k_norm_node->inputs[0]) :
                                        nullptr;
    if (!is_f32(k_norm) || !is_f32(raw_k) || !is_f32(v) || !is_f32(gate) || !is_f32(beta) || !is_f32(state) ||
        !is_f32(gdn_output)) {
        return {};
    }
    const RmsNormParams * l2_params = op_params_as<RmsNormParams>(node->params);
    if (l2_params == nullptr || !std::isfinite(l2_params->eps) || l2_params->eps <= 0.0f ||
        !op_params_equivalent(GGML_OP_L2_NORM, node->params, k_norm_node->params)) {
        return {};
    }

    const int64_t width          = raw_q->ne[0];
    const int64_t q_head_count   = raw_q->ne[1];
    const int64_t token_count    = raw_q->ne[2];
    const int64_t sequence_count = raw_q->ne[3];
    const int64_t head_count     = v->ne[1];
    if (width != 128 || q_head_count <= 0 || q_head_count > 4096 || head_count <= 0 || head_count > 4096 ||
        token_count < 1 || token_count > 512 || sequence_count < 1 || sequence_count > 3) {
        return {};
    }
    const int64_t hidden_size = width * (2 * q_head_count + head_count);
    if (!is_shape(*raw_k, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*q_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*k_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*v, width, head_count, token_count, sequence_count) ||
        !is_shape(*gate, 1, head_count, token_count, sequence_count) ||
        !is_shape(*beta, 1, head_count, token_count, sequence_count) ||
        !is_shape(*state, width, width, head_count, sequence_count) || gdn_output->ne[0] != width * head_count ||
        gdn_output->ne[2] != 1 || gdn_output->ne[3] != 1) {
        return {};
    }
    const int64_t attention_rows = token_count * sequence_count;
    const int64_t snapshot_rows  = width * sequence_count;
    if (gdn_output->ne[1] <= attention_rows || (gdn_output->ne[1] - attention_rows) % snapshot_rows != 0) {
        return {};
    }
    const int64_t snapshot_count = (gdn_output->ne[1] - attention_rows) / snapshot_rows;
    if (snapshot_count < 1 || snapshot_count > 5) {
        return {};
    }
    if (!q_norm->contiguous || !k_norm->contiguous || !state->contiguous || !gdn_output->contiguous ||
        raw_q->storage != raw_k->storage || raw_q->storage != v->storage || raw_q->storage_offset != 0 ||
        raw_k->storage_offset != static_cast<size_t>(width * q_head_count) * sizeof(float) ||
        v->storage_offset != static_cast<size_t>(2 * width * q_head_count) * sizeof(float) ||
        raw_q->nb[0] != sizeof(float) || raw_q->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        raw_q->nb[2] != static_cast<size_t>(hidden_size) * sizeof(float) || raw_k->nb != raw_q->nb ||
        v->nb[0] != sizeof(float) || v->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        v->nb[2] != static_cast<size_t>(hidden_size) * sizeof(float)) {
        return {};
    }

    const GraphNode * gate_reshape = producer_with_op(graph, gate->id, GGML_OP_RESHAPE);
    const Value *     gate_flat    = gate_reshape != nullptr && gate_reshape->inputs.size() == 1 ?
                                         graph_value(graph, gate_reshape->inputs[0]) :
                                         nullptr;
    const GraphNode * gate_mul = gate_flat != nullptr ? producer_with_op(graph, gate_flat->id, GGML_OP_MUL) : nullptr;
    if (gate_mul == nullptr || gate_mul->inputs.size() != 2) {
        return {};
    }
    const Value *     alpha_softplus = graph_value(graph, gate_mul->inputs[0]);
    const Value *     a_scale        = graph_value(graph, gate_mul->inputs[1]);
    const GraphNode * softplus =
        alpha_softplus != nullptr ? producer_with_op(graph, alpha_softplus->id, GGML_OP_UNARY) : nullptr;
    if (!has_unary_op(softplus, UnaryKind::SoftPlus) || softplus->inputs.size() != 1) {
        return {};
    }
    const Value *     alpha_biased = graph_value(graph, softplus->inputs[0]);
    const GraphNode * add_bias =
        alpha_biased != nullptr ? producer_with_op(graph, alpha_biased->id, GGML_OP_ADD) : nullptr;
    if (add_bias == nullptr || add_bias->inputs.size() != 2) {
        return {};
    }
    const Value *     alpha         = graph_value(graph, add_bias->inputs[0]);
    const Value *     bias          = graph_value(graph, add_bias->inputs[1]);
    const GraphNode * alpha_reshape = alpha != nullptr ? producer_with_op(graph, alpha->id, GGML_OP_RESHAPE) : nullptr;
    const Value *     alpha_raw     = alpha_reshape != nullptr && alpha_reshape->inputs.size() == 1 ?
                                          graph_value(graph, alpha_reshape->inputs[0]) :
                                          nullptr;

    const GraphNode * sigmoid = producer_with_op(graph, beta->id, GGML_OP_UNARY);
    if (!has_unary_op(sigmoid, UnaryKind::Sigmoid) || sigmoid->inputs.size() != 1) {
        return {};
    }
    const Value *     beta_pre = graph_value(graph, sigmoid->inputs[0]);
    const GraphNode * beta_reshape =
        beta_pre != nullptr ? producer_with_op(graph, beta_pre->id, GGML_OP_RESHAPE) : nullptr;
    const Value * beta_raw = beta_reshape != nullptr && beta_reshape->inputs.size() == 1 ?
                                 graph_value(graph, beta_reshape->inputs[0]) :
                                 nullptr;
    if (!is_f32(alpha_raw) || !is_f32(beta_raw) || !is_f32(alpha) || !is_f32(alpha_biased) || !is_f32(alpha_softplus) ||
        !is_f32(a_scale) || !is_f32(bias) || !is_f32(gate_flat) || !is_f32(beta_pre) ||
        !is_shape(*alpha_raw, head_count, token_count * sequence_count, 1, 1) ||
        !is_shape(*beta_raw, head_count, token_count * sequence_count, 1, 1) ||
        !is_shape(*gate_flat, head_count, token_count, sequence_count, 1) || !is_shape(*bias, head_count, 1, 1, 1) ||
        !is_shape(*a_scale, head_count, 1, 1, 1) || !alpha_raw->contiguous || !beta_raw->contiguous ||
        !bias->contiguous || !a_scale->contiguous || !gate_flat->contiguous || !beta->contiguous) {
        return {};
    }

    const std::array<const Value *, 6> epilogue_values = { alpha_raw, beta_raw, bias, a_scale, gate_flat, beta };
    for (size_t lhs = 0; lhs < epilogue_values.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < epilogue_values.size(); ++rhs) {
            if (!distinct_storage(*epilogue_values[lhs], *epilogue_values[rhs])) {
                return {};
            }
        }
    }

    const GraphNode * state_reshape = producer_with_op(graph, state->id, GGML_OP_RESHAPE);
    if (state_reshape == nullptr || state_reshape->inputs.size() != 1) {
        return {};
    }
    const std::vector<const GraphNode *> & gdn_consumers = graph.index().consumers(gdn_output->id);
    if (gdn_consumers.size() != 2) {
        return {};
    }
    const GraphNode * new_state_view = nullptr;
    const GraphNode * attention_view = nullptr;
    const size_t      attention_bytes =
        static_cast<size_t>(width * head_count * token_count * sequence_count) * sizeof(float);
    for (const GraphNode * consumer : gdn_consumers) {
        const Value * output =
            consumer != nullptr && consumer->op == GGML_OP_VIEW ? graph_value(graph, consumer->output) : nullptr;
        if (output != nullptr && output->storage == gdn_output->storage &&
            output->storage_offset == gdn_output->storage_offset + attention_bytes) {
            new_state_view = consumer;
        } else if (output != nullptr && output->storage == gdn_output->storage &&
                   output->storage_offset == gdn_output->storage_offset &&
                   is_shape(*output, width, head_count, token_count, sequence_count)) {
            attention_view = consumer;
        } else {
            return {};
        }
    }
    const Value * new_state = new_state_view != nullptr ? graph_value(graph, new_state_view->output) : nullptr;
    if (new_state == nullptr || attention_view == nullptr) {
        return {};
    }
    const int64_t written_snapshot_count = std::min(token_count, snapshot_count);
    const size_t  state_bytes            = static_cast<size_t>(width * width * head_count) * sizeof(float);
    const size_t  state_plane_bytes      = state_bytes * static_cast<size_t>(sequence_count);
    const bool    single_snapshot_state =
        snapshot_count == 1 && is_shape(*new_state, width, width, head_count, sequence_count);
    const bool rollback_snapshot_state =
        is_shape(*new_state, width * width * head_count, sequence_count, written_snapshot_count, 1);
    if ((!single_snapshot_state && !rollback_snapshot_state) || !new_state->contiguous ||
        new_state->storage != gdn_output->storage ||
        new_state->storage_offset != gdn_output->storage_offset + attention_bytes ||
        new_state->byte_count != state_plane_bytes * static_cast<size_t>(written_snapshot_count)) {
        return {};
    }
    const std::vector<const GraphNode *> & state_consumers = graph.index().consumers(new_state->id);
    if (state_consumers.size() != 1 || state_consumers.front() == nullptr ||
        state_consumers.front()->op != GGML_OP_CPY || state_consumers.front()->inputs.size() != 2 ||
        state_consumers.front()->inputs[0] != new_state->id) {
        return {};
    }
    const GraphNode * cache_copy   = state_consumers.front();
    const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
    const Value *     cache        = graph_value(graph, cache_copy->output);
    const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
    if (!is_f32(cache_target) || !is_f32(cache) || cache_view == nullptr) {
        return {};
    }
    const size_t snapshot_stride_count = static_cast<size_t>(written_snapshot_count - 1);
    if (snapshot_stride_count != 0 &&
        cache->nb[2] > (std::numeric_limits<size_t>::max() - state_bytes) / snapshot_stride_count) {
        return {};
    }
    const size_t required_cache_bytes = state_plane_bytes + snapshot_stride_count * cache->nb[2];
    if (cache_view->op != GGML_OP_VIEW || cache_view->inputs.size() != 1 ||
        !same_full_value_range(*cache_target, *cache) || cache->ne[0] != width * width * head_count ||
        cache->ne[1] != sequence_count || cache->ne[2] != written_snapshot_count || cache->ne[3] != 1 ||
        cache->nb[0] != sizeof(float) || cache->nb[1] != state_bytes ||
        (written_snapshot_count > 1 && cache->nb[2] < state_plane_bytes) || cache->byte_count < required_cache_bytes ||
        !distinct_storage(*new_state, *cache)) {
        return {};
    }

    if (!distinct_storage(*gdn_output, *state) || !distinct_storage(*gdn_output, *gate_flat) ||
        !distinct_storage(*gdn_output, *beta)) {
        return {};
    }

    match.covered        = { alpha_reshape, add_bias,       softplus,   gate_mul,    gate_reshape,
                             beta_reshape,  sigmoid,        node,       k_norm_node, state_reshape,
                             gdn,           new_state_view, cache_view, cache_copy,  attention_view };
    match.alpha_raw      = alpha_raw;
    match.beta_raw       = beta_raw;
    match.bias           = bias;
    match.a_scale        = a_scale;
    match.gate           = gate;
    match.gate_flat      = gate_flat;
    match.beta           = beta;
    match.raw_q          = raw_q;
    match.raw_k          = raw_k;
    match.v              = v;
    match.state          = state;
    match.gdn_output     = gdn_output;
    match.attention      = graph_value(graph, attention_view->output);
    match.new_state      = new_state;
    match.cache          = cache;
    match.width          = width;
    match.q_head_count   = q_head_count;
    match.head_count     = head_count;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    match.snapshot_count = snapshot_count;
    match.l2_epsilon     = l2_params->eps;
    return match;
}

static GatedDeltaNetProjectionPairMatch match_gated_delta_net_projection_pair(const DispatchMatchContext & context) {
    GatedDeltaNetProjectionPairMatch match;
    const GraphNode *                first_node = context.root_node;
    if (first_node == nullptr || first_node->op != GGML_OP_MUL_MAT || first_node->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const GraphNode * reshape = common_find_only_consumer_with_op(context.graph, first_node->output, GGML_OP_RESHAPE);
    const GraphNode * add =
        reshape != nullptr ? common_find_only_consumer_with_op(context.graph, reshape->output, GGML_OP_ADD) : nullptr;
    const GraphNode * softplus =
        add != nullptr ? common_find_only_consumer_with_op(context.graph, add->output, GGML_OP_UNARY) : nullptr;
    const GraphNode * mul = has_unary_op(softplus, UnaryKind::SoftPlus) ?
                                common_find_only_consumer_with_op(context.graph, softplus->output, GGML_OP_MUL) :
                                nullptr;
    const GraphNode * gate_reshape =
        mul != nullptr ? common_find_only_consumer_with_op(context.graph, mul->output, GGML_OP_RESHAPE) : nullptr;
    const GraphNode * gdn =
        gate_reshape != nullptr ?
            common_find_only_consumer_with_op(context.graph, gate_reshape->output, GGML_OP_GATED_DELTA_NET) :
            nullptr;
    if (gdn == nullptr || gdn->inputs.size() != 6) {
        return {};
    }

    const GraphNode *        q_norm    = producer_with_op(context.graph, gdn->inputs[0], GGML_OP_L2_NORM);
    const GatedDeltaNetMatch gdn_match = match_gated_delta_net(context.graph, q_norm);
    if (!gdn_match.matched() || !gdn_match.has_projection_epilogue() || gdn_match.alpha_raw->id != first_node->output) {
        return {};
    }

    const GraphNode * second_node = producer_with_op(context.graph, gdn_match.beta_raw->id, GGML_OP_MUL_MAT);
    if (second_node == nullptr || second_node == first_node || second_node->inputs.size() != 2 ||
        second_node->inputs[1] != first_node->inputs[1]) {
        return {};
    }

    CommonMulMatMatch first = common_match_mul_mat_any_format(
        context.graph, first_node, kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel, false);
    CommonMulMatMatch second = common_match_mul_mat_any_format(
        context.graph, second_node, kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel, false);
    if (!first.matched()) {
        first = common_match_mul_mat_any_format(
            context.graph, first_node, kMulMatDualQ4F32DecodeKernel, true);
    }
    if (!second.matched()) {
        second = common_match_mul_mat_any_format(
            context.graph, second_node, kMulMatDualQ4F32DecodeKernel, true);
    }
    if (!first.matched() || !second.matched() || first.weight->type != GGML_TYPE_Q4_K ||
        second.weight->type != GGML_TYPE_Q4_K || first.weight->alias_source.value >= 0 ||
        second.weight->alias_source.value >= 0 || first.input->id != second.input->id ||
        first.input_size != second.input_size || first.output_size != second.output_size ||
        first.token_count != second.token_count || first.output_size != gdn_match.head_count ||
        first.token_count != gdn_match.token_count * gdn_match.sequence_count ||
        !distinct_storage(*first.weight, *second.weight) || !distinct_storage(*first.output, *second.output)) {
        return {};
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(first.input_size, first.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, first.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    const bool symmetric_input = first.token_count > 1 &&
        alternate != nullptr && alternate->name == kCommonSymmetricI4K32ActivationAlternateName;
    const bool native_f32 = is_f32(first.input) && first.input->contiguous &&
        first.token_count == 1 && first.output_size >= 24 && first.output_size < 64 &&
        first.input_size >= 4096 && first.input_size <= 32768 && first.input_size % 1024 == 0;
    if (!symmetric_input && !native_f32) {
        return {};
    }
    if (!symmetric_input) {
        size_t second_index = 0;
        if (!context.graph.index().node_index(second_node, second_index) || second_index <= context.root_index) {
            return {};
        }
        for (const Value * input : { first.input, second.weight }) {
            const GraphNode * producer = context.graph.index().producer(input->storage_root);
            size_t index = 0;
            if (producer != nullptr && (!context.graph.index().node_index(producer, index) ||
                index >= context.covered_nodes.size() || !context.covered_nodes[index])) {
                return {};
            }
        }
        const auto & nodes = context.graph.nodes();
        for (size_t i = context.root_index + 1; i < second_index; ++i) {
            const GraphNode & node = nodes[i];
            if (node.op == GGML_OP_NONE || node.op == GGML_OP_VIEW || node.op == GGML_OP_RESHAPE ||
                node.op == GGML_OP_PERMUTE || node.op == GGML_OP_TRANSPOSE) {
                continue;
            }
            const Value * output = graph_value(context.graph, node.output);
            if (output != nullptr && (output->storage == first.input->storage ||
                output->storage == second.weight->storage)) {
                return {};
            }
        }
    }

    match.first_node    = first_node;
    match.second_node   = second_node;
    match.input         = first.input;
    match.first_weight  = first.weight;
    match.second_weight = second.weight;
    match.first_output  = first.output;
    match.second_output = second.output;
    match.activation    = symmetric_input ? alternate->alternate_value : ValueId{};
    match.native_f32    = !symmetric_input;
    match.input_size    = first.input_size;
    match.output_size   = first.output_size;
    match.token_count   = first.token_count;
    return match;
}

static GatedDeltaNetMatch match_direct_gated_delta_net(const Graph & graph, const GraphNode * gdn) {
    GatedDeltaNetMatch match;
    if (gdn == nullptr || gdn->op != GGML_OP_GATED_DELTA_NET || gdn->inputs.size() != 6 || !graph.has_index()) {
        return match;
    }

    const Value * q_norm     = graph_value(graph, gdn->inputs[0]);
    const Value * k_norm     = graph_value(graph, gdn->inputs[1]);
    const Value * v          = graph_value(graph, gdn->inputs[2]);
    const Value * gate       = graph_value(graph, gdn->inputs[3]);
    const Value * beta       = graph_value(graph, gdn->inputs[4]);
    const Value * state      = graph_value(graph, gdn->inputs[5]);
    const Value * gdn_output = graph_value(graph, gdn->output);
    if (!is_f32(q_norm) || !is_f32(k_norm) || !is_f32(v) || !is_f32(gate) || !is_f32(beta) || !is_f32(state) ||
        !is_f32(gdn_output)) {
        return {};
    }

    const int64_t width          = q_norm->ne[0];
    const int64_t q_head_count   = q_norm->ne[1];
    const int64_t token_count    = q_norm->ne[2];
    const int64_t sequence_count = q_norm->ne[3];
    const int64_t head_count     = v->ne[1];
    if (width != 128 || q_head_count <= 0 || q_head_count > 4096 || head_count <= 0 || head_count > 4096 ||
        token_count < 1 || token_count > 512 || sequence_count < 1 || sequence_count > 3) {
        return {};
    }

    if (!is_shape(*q_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*k_norm, width, q_head_count, token_count, sequence_count) ||
        !is_shape(*v, width, head_count, token_count, sequence_count) ||
        !is_shape(*gate, 1, head_count, token_count, sequence_count) ||
        !is_shape(*beta, 1, head_count, token_count, sequence_count) ||
        !is_shape(*state, width, width, head_count, sequence_count) || gdn_output->ne[0] != width * head_count ||
        gdn_output->ne[2] != 1 || gdn_output->ne[3] != 1 || !q_norm->contiguous || !k_norm->contiguous ||
        !gate->contiguous || !beta->contiguous || !state->contiguous || !gdn_output->contiguous ||
        q_norm->nb[0] != sizeof(float) || q_norm->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        q_norm->nb[2] != static_cast<size_t>(width * q_head_count) * sizeof(float) || k_norm->nb != q_norm->nb ||
        v->nb[0] != sizeof(float) || v->nb[1] != static_cast<size_t>(width) * sizeof(float) ||
        v->nb[2] < static_cast<size_t>(width * head_count) * sizeof(float) || v->nb[2] % sizeof(float) != 0 ||
        v->nb[3] % sizeof(float) != 0) {
        return {};
    }

    const int64_t attention_rows = token_count * sequence_count;
    const int64_t snapshot_rows  = width * sequence_count;
    if (gdn_output->ne[1] <= attention_rows || (gdn_output->ne[1] - attention_rows) % snapshot_rows != 0) {
        return {};
    }
    const int64_t snapshot_count = (gdn_output->ne[1] - attention_rows) / snapshot_rows;
    if (snapshot_count < 1 || snapshot_count > 5) {
        return {};
    }

    const std::vector<const GraphNode *> & gdn_consumers = graph.index().consumers(gdn_output->id);
    if (gdn_consumers.size() != 2) {
        return {};
    }
    const GraphNode * attention_view = nullptr;
    const GraphNode * new_state_view = nullptr;
    const size_t      attention_bytes =
        static_cast<size_t>(width * head_count * token_count * sequence_count) * sizeof(float);
    for (const GraphNode * consumer : gdn_consumers) {
        const Value * output =
            consumer != nullptr && consumer->op == GGML_OP_VIEW ? graph_value(graph, consumer->output) : nullptr;
        if (output != nullptr && output->storage == gdn_output->storage &&
            output->storage_offset == gdn_output->storage_offset &&
            is_shape(*output, width, head_count, token_count, sequence_count)) {
            attention_view = consumer;
        } else if (output != nullptr && output->storage == gdn_output->storage &&
                   output->storage_offset == gdn_output->storage_offset + attention_bytes) {
            new_state_view = consumer;
        } else {
            return {};
        }
    }

    const Value * new_state = new_state_view != nullptr ? graph_value(graph, new_state_view->output) : nullptr;
    if (attention_view == nullptr || new_state == nullptr) {
        return {};
    }
    const int64_t written_snapshot_count = std::min(token_count, snapshot_count);
    const size_t  state_bytes            = static_cast<size_t>(width * width * head_count) * sizeof(float);
    const size_t  state_plane_bytes      = state_bytes * static_cast<size_t>(sequence_count);
    const bool    single_snapshot_state =
        snapshot_count == 1 && is_shape(*new_state, width, width, head_count, sequence_count);
    const bool rollback_snapshot_state =
        is_shape(*new_state, width * width * head_count, sequence_count, written_snapshot_count, 1);
    if ((!single_snapshot_state && !rollback_snapshot_state) || !new_state->contiguous ||
        new_state->storage != gdn_output->storage ||
        new_state->storage_offset != gdn_output->storage_offset + attention_bytes ||
        new_state->byte_count != state_plane_bytes * static_cast<size_t>(written_snapshot_count)) {
        return {};
    }

    const std::vector<const GraphNode *> & state_consumers = graph.index().consumers(new_state->id);
    if (state_consumers.size() != 1 || state_consumers.front() == nullptr ||
        state_consumers.front()->op != GGML_OP_CPY || state_consumers.front()->inputs.size() != 2 ||
        state_consumers.front()->inputs[0] != new_state->id) {
        return {};
    }
    const GraphNode * cache_copy   = state_consumers.front();
    const Value *     cache_target = graph_value(graph, cache_copy->inputs[1]);
    const Value *     cache        = graph_value(graph, cache_copy->output);
    const GraphNode * cache_view   = cache_target != nullptr ? graph.index().producer(cache_target->id) : nullptr;
    if (!is_f32(cache_target) || !is_f32(cache) ||
        (cache_view != nullptr && (cache_view->op != GGML_OP_VIEW || cache_view->inputs.size() != 1)) ||
        !same_full_value_range(*cache_target, *cache) || cache->ne[0] != width * width * head_count ||
        cache->ne[1] != sequence_count || cache->ne[2] != written_snapshot_count || cache->ne[3] != 1 ||
        cache->nb[0] != sizeof(float) || cache->nb[1] != state_bytes ||
        (written_snapshot_count > 1 && cache->nb[2] < state_plane_bytes) || !distinct_storage(*new_state, *cache) ||
        !distinct_storage(*gdn_output, *state) || !distinct_storage(*gdn_output, *gate) ||
        !distinct_storage(*gdn_output, *beta)) {
        return {};
    }
    const size_t snapshot_stride_count = static_cast<size_t>(written_snapshot_count - 1);
    if (snapshot_stride_count != 0 &&
        cache->nb[2] > (std::numeric_limits<size_t>::max() - state_bytes) / snapshot_stride_count) {
        return {};
    }
    const size_t required_cache_bytes = state_plane_bytes + snapshot_stride_count * cache->nb[2];
    if (cache->byte_count < required_cache_bytes) {
        return {};
    }

    match.covered        = { gdn, attention_view, new_state_view, cache_copy };
    match.gate           = gate;
    match.beta           = beta;
    match.raw_q          = q_norm;
    match.raw_k          = k_norm;
    match.v              = v;
    match.state          = state;
    match.gdn_output     = gdn_output;
    match.attention      = graph_value(graph, attention_view->output);
    match.new_state      = new_state;
    match.cache          = cache;
    match.width          = width;
    match.q_head_count   = q_head_count;
    match.head_count     = head_count;
    match.token_count    = token_count;
    match.sequence_count = sequence_count;
    match.snapshot_count = snapshot_count;
    match.l2_epsilon     = 1.0e-6f;
    return match;
}

static void set_compile_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

static void set_float_compile_parameter(KernelSpecialization & kernel, const char * name, float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    kernel.compile_parameters.emplace(name, stream.str());
}

static void configure_gated_delta_net_kernel(Dispatch &                 dispatch,
                                             const GatedDeltaNetMatch & match,
                                             int64_t                    token_count) {
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.head_width", match.width);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.head_count", match.head_count);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.token_count", token_count);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.sequence_count", match.sequence_count);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.qk_stride1", match.raw_q->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.qk_stride2", match.raw_q->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.qk_stride3", match.raw_q->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.value_stride1", match.v->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.value_stride2", match.v->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.value_stride3", match.v->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.scalar_stride1", match.beta->nb[1] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.scalar_stride2", match.beta->nb[2] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.scalar_stride3", match.beta->nb[3] / sizeof(float));
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.query_head_count", match.q_head_count);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.query_sequence_ratio", 1);
    set_float_compile_parameter(dispatch.kernel, "llm.gated_delta_net.l2_epsilon", match.l2_epsilon);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.workgroup_size", 256);
}

static bool match_gated_delta_net_projection_pair_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              dispatch_match) {
    const GatedDeltaNetProjectionPairMatch match = match_gated_delta_net_projection_pair(context);
    if (!match.matched() || !append_covered_node(context, match.first_node, dispatch_match) ||
        !append_covered_node(context, match.second_node, dispatch_match)) {
        return false;
    }

    if (match.native_f32) {
        Dispatch projections;
        projections.kernel = make_kernel_specialization(kMulMatDualQ4F32DecodeKernel);
        set_compile_parameter(projections.kernel, "ggml.mul_mat_dual_q4_f32_c1.input_size", match.input_size);
        set_compile_parameter(projections.kernel, "ggml.mul_mat_dual_q4_f32_c1.output_size", match.output_size);
        projections.bindings = {
            { match.input->id, 0, match.input->byte_count },
            { match.first_weight->id, 0, match.first_weight->byte_count },
            { match.second_weight->id, 0, match.second_weight->byte_count },
            { match.first_output->id, 0, match.first_output->byte_count },
            { match.second_output->id, 0, match.second_output->byte_count },
        };
        dispatch_match.dispatches.push_back(std::move(projections));
        return true;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);

    Dispatch projections;
    projections.kernel = make_kernel_specialization(kMulMatSymmetricI4LowRowAdjacentDualWmmaKernel);
    projections.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.input_size",
                                                  common_to_config_value(match.input_size));
    projections.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.output_size",
                                                  common_to_config_value(match.output_size));
    projections.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.token_count",
                                                  common_to_config_value(match.token_count));
    projections.kernel.compile_parameters.emplace(
        "ggml.mul_mat.symmetric_i4.lowrow.row_group_size",
        common_to_config_value(static_cast<int64_t>(
            common_symmetric_shared4_row_group_size(match.input_size, match.output_size, 4))));
    projections.bindings.push_back(
        common_symmetric_i4_shared4_weight_binding(*match.first_weight, match.input_size, match.output_size));
    projections.bindings.push_back(
        common_symmetric_i4_shared4_weight_binding(*match.second_weight, match.input_size, match.output_size));
    projections.bindings.push_back({ match.first_output->id, 0, match.first_output->byte_count });
    projections.bindings.push_back({ match.second_output->id, 0, match.second_output->byte_count });
    projections.bindings.push_back({ match.activation, 0, activation_layout.payload_bytes });
    projections.bindings.push_back(
        { match.activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    projections.bindings.push_back(
        { match.activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    dispatch_match.dispatches.push_back(std::move(projections));
    return true;
}

static bool match_gated_delta_net_rmsnorm_gate(const DispatchMatchContext & context,
                                              const GatedDeltaNetMatch & gdn,
                                              DispatchMatch & match) {
    if (gdn.attention->kind != ValueKind::Transient || gdn.gdn_output->kind != ValueKind::Transient) {
        return false;
    }
    const GraphNode * rms = common_find_only_consumer_with_op(context.graph, gdn.attention->id, GGML_OP_RMS_NORM);
    DispatchMatchContext rms_context = context;
    rms_context.root_node = rms;
    if (rms == nullptr || !context.graph.index().node_index(rms, rms_context.root_index) ||
        !common_match_rmsnorm_gate_dispatch(rms_context, match) || match.dispatches.size() != 1) {
        return false;
    }
    const Dispatch & norm = match.dispatches.front();
    if (norm.kernel.kernel_id != kRmsNormGateKernel.id || norm.bindings.size() != 5 ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.hidden_size") != std::to_string(gdn.width) ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.publish_q8") != "0" ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.f16_output_row_width") !=
            std::to_string(gdn.width * gdn.head_count)) {
        return false;
    }
    // These inputs remain live while the fused operation publishes its rows.
    const Value * output = graph_value(context.graph, norm.bindings[3].value);
    for (const Value * input : {gdn.raw_q, gdn.raw_k, gdn.v, gdn.state, gdn.gate, gdn.beta, gdn.cache}) {
        if (!distinct_storage(*output, *input)) {
            return false;
        }
    }
    return true;
}

static bool match_gated_delta_net_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GatedDeltaNetMatch match = context.root_node != nullptr && context.root_node->op == GGML_OP_GATED_DELTA_NET ?
                                         match_direct_gated_delta_net(context.graph, context.root_node) :
                                         match_gated_delta_net(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const bool can_fuse_projection_epilogue = match.has_projection_epilogue() && match.token_count <= 16;
    DispatchMatch rms_match;
    const bool fuse_rmsnorm_gate = match.snapshot_count == 1 && !can_fuse_projection_epilogue &&
                                  match_gated_delta_net_rmsnorm_gate(context, match, rms_match);
    Dispatch rms_gate;
    if (fuse_rmsnorm_gate) {
        rms_gate = std::move(rms_match.dispatches.front());
        rms_match.dispatches.clear();
        dispatch_match = std::move(rms_match);
    }

    for (const GraphNode * node : match.covered) {
        if (!append_covered_node(context, node, dispatch_match)) {
            return false;
        }
    }

    const int64_t written_snapshot_count = std::min(match.token_count, match.snapshot_count);
    const int64_t prefix_token_count     = match.token_count - written_snapshot_count;
    const size_t  cache_stride           = match.cache->nb[2];
    const bool fuse_snapshot_projection_epilogue =
        can_fuse_projection_epilogue && match.snapshot_count != 1 && prefix_token_count == 0 &&
        written_snapshot_count == match.token_count && (match.token_count >= 2 || match.sequence_count > 1) &&
        match.token_count <= 5 && cache_stride % sizeof(float) == 0;

    if (match.has_projection_epilogue() && !can_fuse_projection_epilogue) {
        Dispatch epilogue;
        epilogue.kernel = make_kernel_specialization(kGatedDeltaNetProjectionEpilogueKernel);
        epilogue.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
        epilogue.kernel.integer_parameters.emplace("element_count",
                                                   match.head_count * match.token_count * match.sequence_count);
        set_compile_parameter(epilogue.kernel, "llm.gated_delta_net.epilogue_head_count", match.head_count);
        set_compile_parameter(epilogue.kernel, "llm.gated_delta_net.epilogue_workgroup_size", 256);
        epilogue.bindings.push_back({ match.alpha_raw->id, 0, match.alpha_raw->byte_count });
        epilogue.bindings.push_back({ match.beta_raw->id, 0, match.beta_raw->byte_count });
        epilogue.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
        epilogue.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        epilogue.bindings.push_back({ match.gate_flat->id, 0, match.gate_flat->byte_count });
        epilogue.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
        dispatch_match.dispatches.push_back(std::move(epilogue));
    }

    if (match.snapshot_count == 1) {
        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(fuse_rmsnorm_gate ? kGatedDeltaNetRmsNormGateKernel :
                                                can_fuse_projection_epilogue ?
                                                    kGatedDeltaNetPrefillProjectionEpilogueKernel :
                                                    kGatedDeltaNetPrefillKernel);
        configure_gated_delta_net_kernel(gdn, match, match.token_count);
        gdn.bindings.push_back({ match.raw_q->id, 0, match.raw_q->byte_count });
        gdn.bindings.push_back({ match.raw_k->id, 0, match.raw_k->byte_count });
        gdn.bindings.push_back({ match.v->id, 0, match.v->byte_count });
        if (can_fuse_projection_epilogue) {
            gdn.bindings.push_back({ match.alpha_raw->id, 0, match.alpha_raw->byte_count });
            gdn.bindings.push_back({ match.beta_raw->id, 0, match.beta_raw->byte_count });
            gdn.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
            gdn.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        } else {
            gdn.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
            gdn.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
        }
        gdn.bindings.push_back({ match.state->id, 0, match.state->byte_count });
        gdn.bindings.push_back({ match.gdn_output->id, 0, match.gdn_output->byte_count });
        if (fuse_rmsnorm_gate) {
            for (const char * parameter : {"ggml.rmsnorm_gate_f32.rms_epsilon", "ggml.rmsnorm_gate_f32.gate_op"}) {
                gdn.kernel.compile_parameters.emplace(parameter, rms_gate.kernel.compile_parameters.at(parameter));
            }
            gdn.bindings.insert(gdn.bindings.end(), rms_gate.bindings.begin() + 1, rms_gate.bindings.end());
        }

        Dispatch cache_copy;
        cache_copy.kernel = make_kernel_specialization(kCopyF32Kernel);
        cache_copy.kernel.integer_parameters.emplace("element_count", match.new_state->element_count);
        cache_copy.bindings.push_back({ match.new_state->id, 0, match.new_state->byte_count });
        cache_copy.bindings.push_back({ match.cache->id, 0, match.cache->byte_count });

        dispatch_match.dispatches.push_back(std::move(gdn));
        dispatch_match.dispatches.push_back(std::move(cache_copy));
        return true;
    }

    const size_t  q_token_bytes          = static_cast<size_t>(match.width * match.q_head_count) * sizeof(float);
    const size_t  v_token_bytes          = static_cast<size_t>(match.width * match.head_count) * sizeof(float);
    const size_t  gate_token_bytes       = static_cast<size_t>(match.head_count) * sizeof(float);
    const size_t  attention_token_bytes  = v_token_bytes;
    const size_t  state_bytes       = static_cast<size_t>(match.width * match.width * match.head_count) * sizeof(float);
    const size_t  state_plane_bytes = state_bytes * static_cast<size_t>(match.sequence_count);

    if (prefix_token_count == 0 && written_snapshot_count == match.token_count &&
        (match.token_count >= 2 || match.sequence_count > 1) && match.token_count <= 5 &&
        cache_stride % sizeof(float) == 0) {
        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(fuse_snapshot_projection_epilogue ?
                                                    kGatedDeltaNetSnapshotProjectionEpilogueKernel :
                                                    kGatedDeltaNetSnapshotKernel);
        configure_gated_delta_net_kernel(gdn, match, match.token_count);
        set_compile_parameter(gdn.kernel, "llm.gated_delta_net.snapshot_stride", cache_stride / sizeof(float));
        gdn.bindings.push_back({ match.raw_q->id, 0, match.raw_q->byte_count });
        gdn.bindings.push_back({ match.raw_k->id, 0, match.raw_k->byte_count });
        gdn.bindings.push_back({ match.v->id, 0, match.v->byte_count });
        if (fuse_snapshot_projection_epilogue) {
            gdn.bindings.push_back({ match.alpha_raw->id, 0, match.alpha_raw->byte_count });
            gdn.bindings.push_back({ match.beta_raw->id, 0, match.beta_raw->byte_count });
            gdn.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
            gdn.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        } else {
            gdn.bindings.push_back({ match.gate->id, 0, match.gate->byte_count });
            gdn.bindings.push_back({ match.beta->id, 0, match.beta->byte_count });
        }
        gdn.bindings.push_back({ match.state->id, 0, state_plane_bytes });
        gdn.bindings.push_back(
            { match.cache->id, 0, state_plane_bytes + static_cast<size_t>(written_snapshot_count - 1) * cache_stride });
        gdn.bindings.push_back(
            { match.gdn_output->id, 0,
              static_cast<size_t>(match.token_count * match.sequence_count) * attention_token_bytes });
        dispatch_match.dispatches.push_back(std::move(gdn));
        return true;
    }

    auto append_state_copy = [&](ValueId source, size_t source_offset, ValueId target, size_t target_offset) {
        Dispatch copy;
        copy.kernel = make_kernel_specialization(kCopyF32Kernel);
        copy.kernel.integer_parameters.emplace("element_count", state_bytes / sizeof(float));
        copy.bindings.push_back({ source, source_offset, state_bytes });
        copy.bindings.push_back({ target, target_offset, state_bytes });
        dispatch_match.dispatches.push_back(std::move(copy));
    };

    if (prefix_token_count > 0) {
        const size_t q_span    = static_cast<size_t>(prefix_token_count - 1) * match.raw_q->nb[2] + q_token_bytes;
        const size_t k_span    = static_cast<size_t>(prefix_token_count - 1) * match.raw_k->nb[2] + q_token_bytes;
        const size_t v_span    = static_cast<size_t>(prefix_token_count - 1) * match.v->nb[2] + v_token_bytes;
        const size_t gate_span = static_cast<size_t>(prefix_token_count - 1) * match.gate->nb[2] + gate_token_bytes;
        const size_t beta_span = static_cast<size_t>(prefix_token_count - 1) * match.beta->nb[2] + gate_token_bytes;
        const size_t prefix_attention_bytes = static_cast<size_t>(prefix_token_count) * attention_token_bytes;

        Dispatch prefix;
        prefix.kernel = make_kernel_specialization(can_fuse_projection_epilogue ?
                                                       kGatedDeltaNetPrefillProjectionEpilogueKernel :
                                                       kGatedDeltaNetPrefillKernel);
        configure_gated_delta_net_kernel(prefix, match, prefix_token_count);
        prefix.bindings.push_back({ match.raw_q->id, 0, q_span });
        prefix.bindings.push_back({ match.raw_k->id, 0, k_span });
        prefix.bindings.push_back({ match.v->id, 0, v_span });
        if (can_fuse_projection_epilogue) {
            const size_t alpha_span = static_cast<size_t>(prefix_token_count - 1) * match.alpha_raw->nb[1] +
                                      gate_token_bytes;
            const size_t beta_raw_span = static_cast<size_t>(prefix_token_count - 1) * match.beta_raw->nb[1] +
                                         gate_token_bytes;
            prefix.bindings.push_back({ match.alpha_raw->id, 0, alpha_span });
            prefix.bindings.push_back({ match.beta_raw->id, 0, beta_raw_span });
            prefix.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
            prefix.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        } else {
            prefix.bindings.push_back({ match.gate->id, 0, gate_span });
            prefix.bindings.push_back({ match.beta->id, 0, beta_span });
        }
        prefix.bindings.push_back({ match.state->id, 0, state_bytes });
        prefix.bindings.push_back({ match.gdn_output->id, 0, prefix_attention_bytes + state_bytes });
        dispatch_match.dispatches.push_back(std::move(prefix));
        append_state_copy(match.gdn_output->id, prefix_attention_bytes, match.cache->id,
                          static_cast<size_t>(written_snapshot_count - 1) * cache_stride);
    } else {
        append_state_copy(match.state->id, 0, match.cache->id,
                          static_cast<size_t>(written_snapshot_count - 1) * cache_stride);
    }

    for (int64_t snapshot = 0; snapshot < written_snapshot_count; ++snapshot) {
        const int64_t token = prefix_token_count + snapshot;
        const int64_t slot  = written_snapshot_count - 1 - snapshot;
        if (snapshot > 0) {
            append_state_copy(match.cache->id, static_cast<size_t>(slot + 1) * cache_stride, match.cache->id,
                              static_cast<size_t>(slot) * cache_stride);
        }

        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(can_fuse_projection_epilogue ?
                                                    kGatedDeltaNetInplaceProjectionEpilogueKernel :
                                                    kGatedDeltaNetInplaceKernel);
        configure_gated_delta_net_kernel(gdn, match, 1);
        gdn.bindings.push_back({ match.raw_q->id, static_cast<size_t>(token) * match.raw_q->nb[2], q_token_bytes });
        gdn.bindings.push_back({ match.raw_k->id, static_cast<size_t>(token) * match.raw_k->nb[2], q_token_bytes });
        gdn.bindings.push_back({ match.v->id, static_cast<size_t>(token) * match.v->nb[2], v_token_bytes });
        if (can_fuse_projection_epilogue) {
            gdn.bindings.push_back(
                { match.alpha_raw->id, static_cast<size_t>(token) * match.alpha_raw->nb[1], gate_token_bytes });
            gdn.bindings.push_back(
                { match.beta_raw->id, static_cast<size_t>(token) * match.beta_raw->nb[1], gate_token_bytes });
            gdn.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
            gdn.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        } else {
            gdn.bindings.push_back(
                { match.gate->id, static_cast<size_t>(token) * match.gate->nb[2], gate_token_bytes });
            gdn.bindings.push_back(
                { match.beta->id, static_cast<size_t>(token) * match.beta->nb[2], gate_token_bytes });
        }
        gdn.bindings.push_back({ match.cache->id, static_cast<size_t>(slot) * cache_stride, state_bytes });
        gdn.bindings.push_back(
            { match.gdn_output->id, static_cast<size_t>(token) * attention_token_bytes, attention_token_bytes });
        dispatch_match.dispatches.push_back(std::move(gdn));
    }

    return true;
}
static bool match_gated_delta_net_rmsnorm_q8(const DispatchMatchContext & context,
                                           const GatedDeltaNetMatch & gdn,
                                           const Value & state_cache,
                                           DispatchMatch & result) {
    if (gdn.head_count < 24 || gdn.attention->kind != ValueKind::Transient || !gdn.attention->contiguous) {
        return false;
    }
    const Graph & graph = context.graph;
    const Value * input = gdn.attention;
    const GraphNode * reshape = nullptr;
    const GraphNode * rms = common_find_only_consumer_with_op(graph, input->id, GGML_OP_RMS_NORM);
    if (rms == nullptr) {
        reshape = common_find_only_consumer_with_op(graph, input->id, GGML_OP_RESHAPE);
        const Value * reshaped = reshape != nullptr ? graph_value(graph, reshape->output) : nullptr;
        if (reshaped == nullptr || !is_layout_alias_node(graph, *reshape) || !reshaped->contiguous ||
            !same_full_value_range(*input, *reshaped)) {
            return false;
        }
        input = reshaped;
        rms = common_find_only_consumer_with_op(graph, input->id, GGML_OP_RMS_NORM);
    }
    DispatchMatchContext rms_context = context;
    rms_context.root_node = rms;
    DispatchMatch match;
    if (rms == nullptr || !graph.index().node_index(rms, rms_context.root_index) ||
        !common_match_rmsnorm_gate_dispatch(rms_context, match) || match.dispatches.size() != 1) {
        return false;
    }
    const Dispatch & norm = match.dispatches.front();
    if (norm.kernel.kernel_id != kRmsNormGateKernel.id || norm.bindings.size() != 5 ||
        norm.bindings[0].value != input->id ||
        norm.kernel.integer_parameters.at("token_count") != gdn.head_count * gdn.token_count ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.hidden_size") != "128" ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.gate_op") != "15" ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.publish_q8") != "1" ||
        norm.kernel.compile_parameters.at("ggml.rmsnorm_gate_f32.f16_output_row_width") != "0") {
        return false;
    }
    const Value * output = graph_value(graph, norm.bindings[3].value);
    if (output == nullptr || output->kind != ValueKind::Transient || !distinct_storage(*output, state_cache)) {
        return false;
    }
    for (const Value * value : {gdn.raw_q, gdn.raw_k, gdn.v, gdn.alpha_raw, gdn.beta_raw,
                                gdn.bias, gdn.a_scale, gdn.state, gdn.cache}) {
        if (!distinct_storage(*output, *value)) {
            return false;
        }
    }
    for (size_t binding : {size_t{1}, size_t{2}}) {
        const Value * value = graph_value(graph, norm.bindings[binding].value);
        if (value == nullptr || !distinct_storage(*value, *gdn.cache) || !distinct_storage(*value, *output)) {
            return false;
        }
        const GraphNode * producer = graph.index().producer(value->storage_root);
        size_t index = 0;
        if (producer != nullptr && (!graph.index().node_index(producer, index) ||
            index >= context.covered_nodes.size() || !context.covered_nodes[index])) {
            return false;
        }
    }
    if (reshape != nullptr && !append_covered_node(context, reshape, match)) {
        return false;
    }
    result = std::move(match);
    return true;
}

static bool match_selected_gated_delta_net_dispatch(const DispatchMatchContext & context, DispatchMatch & result) {
    const Graph & graph = context.graph;
    const GraphNode * gather = context.root_node;
    if (gather == nullptr || gather->op != GGML_OP_GET_ROWS || gather->inputs.size() != 2 || !graph.has_index()) {
        return false;
    }
    const Value * cache = graph_value(graph, gather->inputs[0]);
    const Value * ids = graph_value(graph, gather->inputs[1]);
    const Value * gathered = graph_value(graph, gather->output);
    if (!is_f32(cache) || !is_f32(gathered) || ids == nullptr || ids->type != GGML_TYPE_I32 ||
        !cache->contiguous || !ids->contiguous || !gathered->contiguous ||
        cache->ne[0] <= 0 || cache->ne[1] < 1 || cache->ne[1] > 262208 ||
        cache->ne[2] != 1 || cache->ne[3] != 1 || cache->element_count > (int64_t{1} << 30) ||
        !is_shape(*ids, 1, 1, 1, 1) || !is_shape(*gathered, cache->ne[0], 1, 1, 1) ||
        gathered->kind != ValueKind::Transient || !graph.index().has_single_consumer(gathered->id)) {
        return false;
    }
    const GraphNode * reshape = graph.index().consumers(gathered->id).front();
    if (reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->inputs.size() != 1 ||
        !graph.index().has_single_consumer(reshape->output)) {
        return false;
    }
    const GraphNode * gdn = graph.index().consumers(reshape->output).front();
    if (gdn == nullptr || gdn->op != GGML_OP_GATED_DELTA_NET || gdn->inputs.size() != 6 ||
        gdn->inputs[5] != reshape->output) {
        return false;
    }
    const GraphNode * q_norm = producer_with_op(graph, gdn->inputs[0], GGML_OP_L2_NORM);
    const GatedDeltaNetMatch match = match_gated_delta_net(graph, q_norm);
    if (!match.matched() || !match.has_projection_epilogue() || match.sequence_count != 1 ||
        match.token_count < 1 || match.token_count > 5 || match.snapshot_count < match.token_count ||
        (match.token_count == 1 && match.snapshot_count == 1) ||
        cache->ne[0] != match.width * match.width * match.head_count ||
        !same_full_value_range(*gathered, *match.state) || match.gdn_output->kind != ValueKind::Transient) {
        return false;
    }
    const size_t state_bytes = static_cast<size_t>(cache->ne[0]) * sizeof(float);
    if (match.cache->nb[2] % state_bytes != 0) {
        return false;
    }
    if (cache->storage == match.cache->storage &&
        (match.cache->storage_offset < cache->storage_offset ||
         (match.cache->storage_offset - cache->storage_offset) % state_bytes != 0 ||
         match.cache->storage_offset - cache->storage_offset > cache->byte_count ||
         match.cache->byte_count > cache->byte_count - (match.cache->storage_offset - cache->storage_offset))) {
        return false;
    }
    const auto internal = [&](const GraphNode * node) {
        return std::find(match.covered.begin(), match.covered.end(), node) != match.covered.end();
    };
    const std::array<const Value *, 9> inputs = {
        match.raw_q, match.raw_k, match.v, match.alpha_raw, match.beta_raw, match.bias, match.a_scale, cache, ids
    };
    for (const Value * input : inputs) {
        const GraphNode * producer = graph.index().producer(input->storage_root);
        size_t index = 0;
        if (producer != nullptr && (!graph.index().node_index(producer, index) ||
            index >= context.covered_nodes.size() || !context.covered_nodes[index])) {
            return false;
        }
        if (!distinct_storage(*input, *match.gdn_output) ||
            (input != cache && !distinct_storage(*input, *match.cache))) {
            return false;
        }
    }
    size_t copy_index = 0;
    for (const GraphNode * node : match.covered) {
        const Value * output = graph_value(graph, node->output);
        if (node->op == GGML_OP_CPY) {
            if (!graph.index().node_index(node, copy_index)) {
                return false;
            }
        }
        if (output->id == match.cache->id || output->id == match.attention->id) {
            continue;
        }
        if (!is_layout_alias_node(graph, *node) && output->kind != ValueKind::Transient) {
            return false;
        }
        for (const GraphNode * consumer : graph.index().consumers(output->id)) {
            if (!internal(consumer)) {
                return false;
            }
        }
    }
    if (copy_index <= context.root_index) {
        return false;
    }
    // The fused dispatch publishes snapshots at the gather's position.
    for (size_t index = context.root_index + 1; index < copy_index; ++index) {
        const GraphNode & node = graph.nodes()[index];
        const Value * output = graph_value(graph, node.output);
        if (internal(&node) || is_layout_alias_node(graph, node) || output->byte_count == 0) {
            continue;
        }
        const auto touches_cache = [&](const Value * value) {
            return value != nullptr && value->byte_count != 0 && value->storage == match.cache->storage;
        };
        if (touches_cache(output)) {
            return false;
        }
        for (ValueId input : node.inputs) {
            if (touches_cache(graph_value(graph, input))) {
                return false;
            }
        }
    }
    DispatchMatchContext gdn_context = context;
    gdn_context.root_node = q_norm;
    if (!graph.index().node_index(q_norm, gdn_context.root_index)) {
        return false;
    }
    DispatchMatch selected;
    DispatchMatch rms_match;
    const bool fuse_rms_q8 = match_gated_delta_net_rmsnorm_q8(context, match, *cache, rms_match);
    if ((match.token_count == 1 && !fuse_rms_q8) ||
        !match_gated_delta_net_dispatch(gdn_context, selected) ||
        selected.dispatches.size() != (match.token_count == 1 ? 2 : 1) ||
        selected.dispatches.back().bindings.size() != (match.token_count == 1 ? 9 : 10) ||
        !append_covered_node(context, gather, selected)) {
        return false;
    }
    if (fuse_rms_q8) {
        Dispatch dispatch = std::move(selected.dispatches.back());
        const Dispatch & norm = rms_match.dispatches.front();
        const auto snapshot = dispatch.bindings[match.token_count == 1 ? 7 : 8];
        auto kernel = make_kernel_specialization(kGatedDeltaNetSelectedRmsQ8Kernel);
        kernel.compile_parameters = std::move(dispatch.kernel.compile_parameters);
        set_compile_parameter(kernel, "llm.gated_delta_net.state_row_count", cache->ne[1]);
        set_compile_parameter(kernel, "llm.gated_delta_net.snapshot_stride", match.cache->nb[2] / sizeof(float));
        for (const auto & parameter : norm.kernel.compile_parameters) {
            kernel.compile_parameters.emplace(parameter);
        }
        dispatch.kernel = std::move(kernel);
        dispatch.bindings.resize(7);
        dispatch.bindings.push_back({ cache->id, 0, cache->byte_count });
        dispatch.bindings.push_back(snapshot);
        dispatch.bindings.push_back(norm.bindings[3]);
        dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
        dispatch.bindings.push_back(norm.bindings[1]);
        dispatch.bindings.push_back(norm.bindings[2]);
        dispatch.bindings.push_back(norm.bindings[4]);
        selected.dispatches.clear();
        selected.dispatches.push_back(std::move(dispatch));
        selected.covered_nodes.insert(selected.covered_nodes.end(), rms_match.covered_nodes.begin(), rms_match.covered_nodes.end());
        selected.transients = std::move(rms_match.transients);
        if (!selected.metadata.append(std::move(rms_match.metadata), selected.status)) {
            return false;
        }
        result = std::move(selected);
        return true;
    }
    Dispatch & dispatch = selected.dispatches.front();
    auto kernel = make_kernel_specialization(kGatedDeltaNetSelectedSnapshotKernel);
    kernel.compile_parameters = std::move(dispatch.kernel.compile_parameters);
    dispatch.kernel = std::move(kernel);
    set_compile_parameter(dispatch.kernel, "llm.gated_delta_net.state_row_count", cache->ne[1]);
    dispatch.bindings[7] = { cache->id, 0, cache->byte_count };
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    result = std::move(selected);
    return true;
}
}  // namespace

void register_llm_gated_delta_net_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.gated_delta_net.selected_snapshot.f32_wmma_head128",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        250,
        DispatchSource::Llm,
        match_selected_gated_delta_net_dispatch,
    });
    registry.add({
        "llm.gated_delta_net.projection_pair.q4_k",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Llm,
        match_gated_delta_net_projection_pair_dispatch,
    });
    registry.add({
        "llm.gated_delta_net.direct.f32_wmma_head128",
        GGML_OP_GATED_DELTA_NET,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Llm,
        match_gated_delta_net_dispatch,
    });
    registry.add({
        "llm.gated_delta_net.f32_wmma_head128",
        GGML_OP_L2_NORM,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Llm,
        match_gated_delta_net_dispatch,
    });
}

}  // namespace ggml::hrx
