#pragma once

#include "../dispatch-registry.h"
#include "dispatch-binary-common.h"
#include "dispatch-mul-mat-weight-format.h"
#include "dispatch-symmetric-i4.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

struct CommonMulMatMatch {
    const Value *            input            = nullptr;
    const Value *            weight           = nullptr;
    const Value *            output           = nullptr;
    KernelCatalogRef         kernel           = {};
    int64_t                  input_size       = 0;
    int64_t                  output_size      = 0;
    int64_t                  token_count      = 0;
    CommonMulMatWeightFormat weight_format    = CommonMulMatWeightFormat::Q4K;
    UnaryKind                output_unary_op  = UnaryKind::Identity;
    size_t                   unary_node_index = 0;
    bool                     has_fused_unary  = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

struct CommonSymmetricI4ActivationLayout {
    size_t payload_bytes  = 0;
    size_t scales_offset  = 0;
    size_t metadata_bytes = 0;
    size_t sums_offset    = 0;
    size_t total_bytes    = 0;
};

inline constexpr const char kCommonSymmetricI4K32ActivationAlternateName[] =
    "common.mul_mat.symmetric_i4_k32.activation";

enum class CommonQ8ActivationPolicy {
    ExistingAlternateOnly,
    AllowStandaloneQuantize,
};

inline size_t common_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline CommonSymmetricI4ActivationLayout common_symmetric_i4_activation_layout(int64_t input_size,
                                                                               int64_t token_count) {
    const size_t element_count  = static_cast<size_t>(input_size) * static_cast<size_t>(token_count);
    const size_t payload_bytes  = element_count / 2;
    const size_t metadata_bytes = element_count / 8;
    const size_t scales_offset  = common_align_up(payload_bytes, 256);
    const size_t sums_offset    = common_align_up(scales_offset + metadata_bytes, 256);
    return {
        payload_bytes, scales_offset, metadata_bytes, sums_offset, common_align_up(sums_offset + metadata_bytes, 256),
    };
}

inline size_t common_symmetric_shared4_row_group_size(int64_t input_size, int64_t output_size, int64_t quant_bits) {
    const size_t logical_output_size    = static_cast<size_t>(output_size);
    const size_t materialized_row_bytes = 4 + 8 * 32 * static_cast<size_t>(quant_bits) / 8;
    const size_t materialized_bytes =
        logical_output_size * static_cast<size_t>(input_size / 256) * materialized_row_bytes;
    if (quant_bits == 4 && materialized_bytes < size_t{ 16 } * 1024 * 1024) {
        return 32;
    }
    if (quant_bits == 4 && materialized_bytes <= size_t{ 32 } * 1024 * 1024 && output_size > input_size) {
        return 96;
    }
    return ((logical_output_size + 255) / 256) * 32;
}

inline size_t common_symmetric_shared4_weight_byte_count(int64_t input_size, int64_t output_size, int64_t quant_bits) {
    const size_t logical_output_size    = static_cast<size_t>(output_size);
    const size_t row_group_size         = common_symmetric_shared4_row_group_size(input_size, output_size, quant_bits);
    const size_t padded_output_size     = (logical_output_size + row_group_size - 1) / row_group_size * row_group_size;
    const size_t materialized_row_bytes = 4 + 8 * 32 * static_cast<size_t>(quant_bits) / 8;
    return padded_output_size * static_cast<size_t>(input_size / 256) * materialized_row_bytes;
}

inline size_t common_symmetric_i4_shared4_weight_byte_count(int64_t input_size, int64_t output_size) {
    return common_symmetric_shared4_weight_byte_count(input_size, output_size, 4);
}

inline DispatchBinding common_symmetric_i4_shared4_weight_binding(const Value & weight,
                                                                  int64_t       input_size,
                                                                  int64_t       output_size) {
    DispatchBinding binding;
    binding.value         = weight.id;
    binding.length        = common_symmetric_i4_shared4_weight_byte_count(input_size, output_size);
    binding.layout        = kSymmetricI4K32EightGroupsShared4Layout;
    binding.source_type   = weight.type;
    binding.input_size    = input_size;
    binding.output_size   = output_size;
    binding.source_length = weight.byte_count;
    return binding;
}

inline DispatchBinding common_symmetric_i4_shared4_multistart_weight_binding(const Value & weight,
                                                                             int64_t       input_size,
                                                                             int64_t       output_size) {
    DispatchBinding binding = common_symmetric_i4_shared4_weight_binding(weight, input_size, output_size);
    binding.layout          = kSymmetricI4K32EightGroupsShared4MultistartLayout;
    return binding;
}

inline const Value * common_graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

inline bool common_same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

inline bool common_is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

inline bool common_is_supported_dense_input_size(CommonMulMatWeightFormat format, int64_t input_size) {
    return common_mul_mat_supported_dense_input_size(format, input_size);
}

inline bool common_is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

// Select the packed prefill schedule. Long Q4 contractions only benefit when
// their producer supplies the layout without a separate conversion pass.
inline bool common_mul_mat_uses_k16_major_f16(CommonMulMatWeightFormat format,
                                               int64_t input_size,
                                               int64_t output_size,
                                               int64_t token_count,
                                               bool packed_producer = false) {
    if (token_count < 512 || token_count > 2048 || token_count % 512 != 0 || input_size % 256 != 0 ||
        output_size % 64 != 0) {
        return false;
    }
    if (format == CommonMulMatWeightFormat::Q4K) {
        return token_count == 512 && output_size / 64 >= 64 &&
               (input_size <= 2 * output_size || (packed_producer && input_size <= 4 * output_size));
    }
    if (format == CommonMulMatWeightFormat::Q6K) {
        const int64_t tile = input_size > output_size ? 64 : 128;
        return output_size % tile == 0 && (token_count / 512) * (output_size / tile) >= 32;
    }
    return false;
}

inline ggml_type common_mul_mat_format_type(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::Q1_0:
            return GGML_TYPE_Q1_0;
        case CommonMulMatWeightFormat::Q3K:
            return GGML_TYPE_Q3_K;
        case CommonMulMatWeightFormat::Q4K:
        case CommonMulMatWeightFormat::Q4KRow64:
            return GGML_TYPE_Q4_K;
        case CommonMulMatWeightFormat::Q5K:
            return GGML_TYPE_Q5_K;
        case CommonMulMatWeightFormat::Q6K:
        case CommonMulMatWeightFormat::Q6KRow64:
            return GGML_TYPE_Q6_K;
        case CommonMulMatWeightFormat::Q4_0:
            return GGML_TYPE_Q4_0;
        case CommonMulMatWeightFormat::Q4_1:
            return GGML_TYPE_Q4_1;
        case CommonMulMatWeightFormat::Q5_0:
            return GGML_TYPE_Q5_0;
        case CommonMulMatWeightFormat::Q5_1:
            return GGML_TYPE_Q5_1;
        case CommonMulMatWeightFormat::IQ1_S:
            return GGML_TYPE_IQ1_S;
        case CommonMulMatWeightFormat::IQ1_M:
            return GGML_TYPE_IQ1_M;
        case CommonMulMatWeightFormat::IQ2_XXS:
            return GGML_TYPE_IQ2_XXS;
        case CommonMulMatWeightFormat::IQ2_XS:
            return GGML_TYPE_IQ2_XS;
        case CommonMulMatWeightFormat::IQ3_XXS:
            return GGML_TYPE_IQ3_XXS;
        case CommonMulMatWeightFormat::IQ2_S:
            return GGML_TYPE_IQ2_S;
        case CommonMulMatWeightFormat::IQ3_S:
            return GGML_TYPE_IQ3_S;
        case CommonMulMatWeightFormat::IQ4_NL:
            return GGML_TYPE_IQ4_NL;
        case CommonMulMatWeightFormat::IQ4_XS:
            return GGML_TYPE_IQ4_XS;
        case CommonMulMatWeightFormat::Q8_0:
            return GGML_TYPE_Q8_0;
        case CommonMulMatWeightFormat::Q8_1:
            return GGML_TYPE_Q8_1;
        case CommonMulMatWeightFormat::F16:
            return GGML_TYPE_F16;
        case CommonMulMatWeightFormat::BF16:
            return GGML_TYPE_BF16;
        case CommonMulMatWeightFormat::F32:
            return GGML_TYPE_F32;
    }
    return GGML_TYPE_COUNT;
}

inline bool common_is_supported_dense_decode_output_size(CommonMulMatWeightFormat format,
                                                         int64_t                  input_size,
                                                         int64_t                  output_size) {
    static constexpr uint64_t kMaxDenseDecodeOutputSize       = 1048576;
    static constexpr uint64_t kAmdgpuAddressableByteRangeSize = uint64_t{ 1 } << 32;

    if (output_size < 1) {
        return false;
    }

    const ggml_type type = common_mul_mat_format_type(format);
    if (type == GGML_TYPE_COUNT) {
        return false;
    }

    const size_t weight_row_size = ggml_row_size(type, input_size);
    if (weight_row_size == 0) {
        return false;
    }

    const uint64_t addressable_output_size = kAmdgpuAddressableByteRangeSize / static_cast<uint64_t>(weight_row_size);
    const uint64_t max_output_size         = std::min(kMaxDenseDecodeOutputSize, addressable_output_size);
    return static_cast<uint64_t>(output_size) <= max_output_size;
}

inline bool common_is_supported_rmsnorm_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

inline bool common_is_supported_prefill_token_count(int64_t token_count) {
    return token_count > 1 && token_count <= 2048;
}

inline bool common_is_supported_decode_token_count(int64_t token_count) {
    return token_count == 1;
}

inline bool common_is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

inline std::string common_to_config_value(int64_t value) {
    return std::to_string(value);
}

inline std::string common_to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

inline int64_t common_ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline bool common_is_weight_shape(const Value & weight, int64_t hidden_size) {
    if (weight.ne[0] != hidden_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (weight.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

inline const GraphNode * common_find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != op) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
}

inline const GraphNode * common_find_only_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != op) {
        return nullptr;
    }
    return consumers.front();
}

inline bool common_has_symmetric_i4_lowrow_consumer(const Graph & graph, const Value & value) {
    if (!graph.has_index()) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        if (consumer != nullptr && common_symmetric_i4_lowrow_mul_mat_eligible(graph, *consumer, value)) {
            return true;
        }
    }

    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        if (consumer == nullptr || !is_layout_alias_node(graph, *consumer) || consumer->inputs.size() != 1 ||
            consumer->inputs.front() != value.id) {
            continue;
        }

        const Value * reshaped = common_graph_value(graph, consumer->output);
        if (reshaped != nullptr && reshaped->type == GGML_TYPE_F32 && reshaped->contiguous &&
            reshaped->storage_root == value.storage_root && reshaped->element_count == value.element_count &&
            reshaped->byte_count == value.byte_count) {
            for (const GraphNode * reshaped_consumer : graph.index().consumers(reshaped->id)) {
                if (reshaped_consumer != nullptr &&
                    common_symmetric_i4_lowrow_mul_mat_eligible(graph, *reshaped_consumer, *reshaped)) {
                    return true;
                }
            }
        }
    }
    return false;
}

inline bool common_binary_node_is_mul(const GraphNode & node) {
    if (node.op != GGML_OP_MUL || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Mul;
}

inline bool common_binary_node_is_add(const GraphNode & node) {
    if (node.op != GGML_OP_ADD || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Add;
}

inline bool common_is_swiglu_params(const OpParams & params) {
    const BinaryParams * binary_params = op_params_as<BinaryParams>(params);
    if (binary_params != nullptr) {
        return binary_params->op == BinaryKind::SwiGLU;
    }

    const GluParams * glu_params = op_params_as<GluParams>(params);
    return glu_params != nullptr && glu_params->op == GGML_GLU_OP_SWIGLU;
}

inline CommonMulMatMatch common_match_mul_mat_any_format(const Graph &     graph,
                                                         const GraphNode * node,
                                                         KernelCatalogRef  kernel,
                                                         bool              decode) {
    CommonMulMatMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = common_graph_value(graph, node->inputs[0]);
    const Value * input  = common_graph_value(graph, node->inputs[1]);
    const Value * output = common_graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !common_is_2d(*weight) || !common_is_2d(*input) ||
        !common_is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    CommonMulMatWeightFormat format = CommonMulMatWeightFormat::Q4K;
    if (!common_mul_mat_format_for_type(weight->type, format)) {
        if (weight->type == GGML_TYPE_IQ1_S) {
            format = CommonMulMatWeightFormat::IQ1_S;
        } else if (weight->type == GGML_TYPE_IQ1_M) {
            format = CommonMulMatWeightFormat::IQ1_M;
        } else {
            return {};
        }
    }
    if ((format == CommonMulMatWeightFormat::IQ1_S || format == CommonMulMatWeightFormat::IQ1_M) &&
        (std::getenv("GGML_HRX_DISABLE_IQ_CODEBOOK_MATMUL") != nullptr ||
         (format == CommonMulMatWeightFormat::IQ1_S &&
          std::getenv("GGML_HRX_DISABLE_IQ1_S_CODEBOOK_MATMUL") != nullptr) ||
         (format == CommonMulMatWeightFormat::IQ1_M &&
          std::getenv("GGML_HRX_DISABLE_IQ1_M_CODEBOOK_MATMUL") != nullptr))) {
        return {};
    }

    const int64_t input_size            = weight->ne[0];
    const int64_t output_size           = weight->ne[1];
    const int64_t token_count           = input->ne[1];
    const bool    token_count_supported = decode ? common_is_supported_decode_token_count(token_count) :
                                                   common_is_supported_prefill_token_count(token_count);
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !token_count_supported || !common_is_supported_dense_input_size(format, input_size) ||
        !(decode ? common_is_supported_dense_decode_output_size(format, input_size, output_size) :
                   common_is_supported_dense_output_size(output_size))) {
        return {};
    }

    match.input         = input;
    match.weight        = weight;
    match.output        = output;
    match.kernel        = kernel;
    match.input_size    = input_size;
    match.output_size   = output_size;
    match.token_count   = token_count;
    match.weight_format = format;
    return match;
}

struct CommonMulMatActivationConsumerMatch {
    const Value *            input         = nullptr;
    const Value *            weight        = nullptr;
    const Value *            output        = nullptr;
    int64_t                  input_size    = 0;
    int64_t                  output_size   = 0;
    int64_t                  token_count   = 0;
    CommonMulMatWeightFormat weight_format = CommonMulMatWeightFormat::Q4K;

    bool matched() const { return input != nullptr && weight != nullptr && output != nullptr; }
};

inline CommonMulMatActivationConsumerMatch common_match_mul_mat_activation_consumer(const Graph &     graph,
                                                                                     const GraphNode * consumer,
                                                                                     const Value &     input) {
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
        consumer->inputs[1] != input.id) {
        return {};
    }

    const Value * weight = common_graph_value(graph, consumer->inputs[0]);
    const Value * output = common_graph_value(graph, consumer->output);
    if (weight == nullptr || output == nullptr || input.type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input.contiguous || !weight->contiguous || !output->contiguous || !common_is_2d(input) ||
        !common_is_2d(*weight) || !common_is_2d(*output)) {
        return {};
    }

    CommonMulMatWeightFormat format;
    if (!common_mul_mat_format_for_type(weight->type, format)) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input.ne[1];
    if (input.ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !common_is_supported_dense_input_size(input_size) || !common_is_supported_dense_output_size(output_size) ||
        output_size % 64 != 0) {
        return {};
    }

    return { &input, weight, output, input_size, output_size, token_count, format };
}

inline bool common_accepts_q8_1_x4_decode_mul_mat(const Graph &     graph,
                                                   const GraphNode * consumer,
                                                   const Value &     input) {
    const CommonMulMatActivationConsumerMatch match =
        common_match_mul_mat_activation_consumer(graph, consumer, input);
    return match.matched() && match.weight->alias_source.value < 0 && match.token_count >= 1 &&
           match.token_count <= 5 &&
           (match.weight_format == CommonMulMatWeightFormat::Q4K ||
            match.weight_format == CommonMulMatWeightFormat::Q6K);
}

inline bool common_accepts_q8_1_x4_decode_mul_mat(const DispatchMatchContext & context,
                                                   const GraphNode &            consumer,
                                                   const Value &                input) {
    return common_accepts_q8_1_x4_decode_mul_mat(context.graph, &consumer, input);
}

inline bool common_accepts_q8_1_x4_prefill_mul_mat(const Graph &     graph,
                                                    const GraphNode * consumer,
                                                    const Value &     input) {
    const CommonMulMatActivationConsumerMatch match =
        common_match_mul_mat_activation_consumer(graph, consumer, input);
    return match.matched() && match.token_count >= 256 && match.token_count <= 2048 &&
           match.token_count % 256 == 0 &&
           (match.weight_format == CommonMulMatWeightFormat::Q5K ||
            match.weight_format == CommonMulMatWeightFormat::IQ4_XS);
}

inline bool common_accepts_f16_k16_major_mul_mat(const Graph &     graph,
                                                  const GraphNode * consumer,
                                                  const Value &     input,
                                                  bool              packed_producer = false) {
    const CommonMulMatActivationConsumerMatch match =
        common_match_mul_mat_activation_consumer(graph, consumer, input);
    return match.matched() && match.weight->alias_source.value < 0 &&
           common_mul_mat_uses_k16_major_f16(match.weight_format, match.input_size, match.output_size,
                                             match.token_count, packed_producer);
}

inline bool common_accepts_f16_k16_major_mul_mat(const DispatchMatchContext & context,
                                                  const GraphNode &            consumer,
                                                  const Value &                input) {
    return common_accepts_f16_k16_major_mul_mat(context.graph, &consumer, input);
}

inline bool common_accepts_f16_k16_major_packed_producer_mul_mat(const DispatchMatchContext & context,
                                                                  const GraphNode &            consumer,
                                                                  const Value &                input) {
    return common_accepts_f16_k16_major_mul_mat(context.graph, &consumer, input, true);
}

inline bool common_accepts_f16_row_prefill_mul_mat(const DispatchMatchContext & context,
                                                    const GraphNode &            consumer,
                                                    const Value &                input) {
    const CommonMulMatActivationConsumerMatch match =
        common_match_mul_mat_activation_consumer(context.graph, &consumer, input);
    if (!match.matched() || match.weight->alias_source.value >= 0 ||
        !common_is_supported_prefill_token_count(match.token_count)) {
        return false;
    }
    if (match.weight_format == CommonMulMatWeightFormat::Q6K && match.token_count % 128 == 0) {
        return true;
    }
    return match.weight_format == CommonMulMatWeightFormat::Q4K && match.token_count % 256 == 0 &&
           match.output_size >= match.input_size / 4;
}

// Share the conversion used by F16-operand matmuls.
inline bool common_prepare_f16_input(const DispatchMatchContext & context,
                                    const Value &                input,
                                    int64_t                      input_size,
                                    int64_t                      token_count,
                                    DispatchMatch &              match,
                                    DispatchBinding &            binding) {
    const size_t bytes = static_cast<size_t>(token_count * input_size) * sizeof(ggml_fp16_t);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(context.graph, context.plan, input.id, GGML_TYPE_F16, bytes);
    if (alternate != nullptr) {
        binding = { alternate->alternate_value, 0, bytes };
        return true;
    }

    constexpr KernelCatalogRef kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32_f16");
    constexpr const char * name = "common.mul_mat.f16";
    const ValueId activation = context.next_plan_value;
    match.transients.push_back({ activation, name, bytes, 256 });
    Dispatch convert;
    convert.kernel = make_kernel_specialization(kernel);
    convert.kernel.integer_parameters.emplace("element_count", token_count * input_size);
    convert.bindings.push_back({ input.id, 0, input.byte_count });
    convert.bindings.push_back({ activation, 0, bytes });
    match.dispatches.push_back(std::move(convert));
    Status status;
    if (!match.metadata.append_alternate_value({ input.id, activation, GGML_TYPE_F16, bytes, name }, status)) {
        match.status.append(status);
        return false;
    }
    binding = { activation, 0, bytes };
    return true;
}

// Private K16-major copy; it is not published as an ordinary F16 alternate.
inline bool common_prepare_k16_major_f16_input(const DispatchMatchContext & context,
                                               const Value &                input,
                                               int64_t                      input_size,
                                               int64_t                      token_count,
                                               DispatchMatch &              match,
                                               DispatchBinding &            binding) {
    constexpr KernelCatalogRef kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f16_k16_major");
    constexpr const char * name = "common.mul_mat.k16_major_f16";
    const size_t bytes = static_cast<size_t>(token_count * input_size) * sizeof(ggml_fp16_t);
    const CommandPlanGeneratedResource * generated =
        context.plan.metadata.find_generated_resource(input.id, GeneratedResourceRole::F16K16Major);
    if (generated != nullptr && generated->byte_count == bytes) {
        binding = { generated->generated_value, 0, bytes };
        return true;
    }
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(context.graph, context.plan, input.id, GGML_TYPE_F16, bytes);
    const ValueId packed(context.next_plan_value.value + static_cast<int32_t>(match.transients.size()));
    match.transients.push_back({ packed, name, bytes, 256 });
    Dispatch copy;
    copy.kernel = make_kernel_specialization(kernel);
    copy.kernel.integer_parameters.emplace("token_count", token_count);
    copy.kernel.compile_parameters.emplace("ggml.copy_f16_k16_major.input_size", common_to_config_value(input_size));
    copy.kernel.compile_parameters.emplace("ggml.copy_f16_k16_major.token_count", common_to_config_value(token_count));
    copy.kernel.compile_parameters.emplace("ggml.copy_f16_k16_major.input_is_f16", alternate != nullptr ? "1" : "0");
    copy.bindings.push_back(alternate != nullptr ? DispatchBinding{ alternate->alternate_value, 0, bytes } :
                                                  DispatchBinding{ input.id, 0, input.byte_count });
    copy.bindings.push_back({ packed, 0, bytes });
    match.dispatches.push_back(std::move(copy));
    Status status;
    if (!match.metadata.append_generated_resource({ input.id, GeneratedResourceRole::F16K16Major, packed, bytes, {} },
                                                  status)) {
        match.status.append(status);
        return false;
    }
    binding = { packed, 0, bytes };
    return true;
}

// Share one activation producer between ordinary and fused matmuls.
inline bool common_prepare_q8_1_x4_input(const DispatchMatchContext & context,
                                         const Value &                input,
                                         int64_t                      input_size,
                                         int64_t                      token_count,
                                         DispatchMatch &              match,
                                         DispatchBinding &            binding,
                                         CommonQ8ActivationPolicy     policy) {
    const size_t bytes = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(context.graph, context.plan, input.id, GGML_TYPE_Q8_1, bytes);
    if (alternate != nullptr) {
        binding = { alternate->alternate_value, 0, bytes };
        return true;
    }
    if (policy != CommonQ8ActivationPolicy::AllowStandaloneQuantize) {
        return false;
    }

    constexpr KernelCatalogRef kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
    constexpr const char * name = "common.mul_mat.q8_1_x4";
    const ValueId activation = context.next_plan_value;
    match.transients.push_back({ activation, name, bytes, 256 });
    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kernel);
    quantize.kernel.integer_parameters.emplace("token_count", token_count);
    quantize.kernel.integer_parameters.emplace("input_size", input_size);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                common_to_config_value(token_count * input_size / 128));
    quantize.bindings.push_back({ input.id, 0, input.byte_count });
    quantize.bindings.push_back({ activation, 0, bytes });
    match.dispatches.push_back(std::move(quantize));

    Status status;
    if (!match.metadata.append_alternate_value({ input.id, activation, GGML_TYPE_Q8_1, bytes, name }, status)) {
        match.status.append(status);
        return false;
    }
    binding = { activation, 0, bytes };
    return true;
}

}  // namespace ggml::hrx
