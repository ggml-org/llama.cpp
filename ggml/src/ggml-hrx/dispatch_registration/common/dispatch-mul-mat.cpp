#include "dispatch-mul-mat.h"

#include "dispatch-activation-publication.h"
#include "dispatch-mul-mat-common.h"
#include "dispatch_registration/immutable-program-resource.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "iq-codebook-resource.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatF32F32WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_wmma");
static constexpr KernelCatalogRef kMulMatTiledF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledIQ1SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_iq1_s_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledIQ1MF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_iq1_m_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledIQ2SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_iq2_s_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledIQ3SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_iq3_s_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledAlignedF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_publish_f32_aligned");
static constexpr KernelCatalogRef kMulMatTiledToken64AlignedF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_publish_f32_token64_aligned");
static constexpr KernelCatalogRef kMulMatTiledIQ3XXSF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_iq3_xxs_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledF32F16AlternateKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_publish_f32_f16_alternate");
static constexpr KernelCatalogRef kMulMatSkinnyF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_skinny_input_f32_publish_f32");
static constexpr KernelCatalogRef kMulMatSkinnyBiasAddF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32");
static constexpr KernelCatalogRef kMulMatF32F32DecodeWave64Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_decode_wave64");
static constexpr KernelCatalogRef kMulMatAddF32F32DecodeWave64Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_add_f32_f32_decode_wave64");
static constexpr KernelCatalogRef kMulMatVectorF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorQ1_0Q8_1X4F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_q1_0_q8_1_x4_f32");
static constexpr KernelCatalogRef kMulMatVectorQ1_0Q8_1X4Rows2F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_q1_0_q8_1_x4_rows2_f32");
static constexpr KernelCatalogRef kMulMatVectorIQ1SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_iq1_s_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorIQ1MF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_iq1_m_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorIQ2SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_iq2_s_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorIQ3SF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_iq3_s_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorIQ3XXSF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_iq3_xxs_f32_f32");
static constexpr KernelCatalogRef kMulMatVectorBiasAddF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_bias_residual_f32_f32");
static constexpr KernelCatalogRef kQuantizeQ8_1X4F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
static constexpr KernelCatalogRef kMulMatTiledBiasAddF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32");
static constexpr KernelCatalogRef kMulMatTiledAlignedBiasAddF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32_aligned");
static constexpr KernelCatalogRef kQuantizeF32SymmetricI4K64PlaneKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i4_k64_plane");
static constexpr KernelCatalogRef kMulMatSymmetricI4WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_wmma");
static constexpr KernelCatalogRef kQuantizeF32SymmetricI4K32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i4_k32");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowWmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_wmma");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowSplitK2WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_wmma");
static constexpr KernelCatalogRef kMulMatSymmetricI4LowRowSplitK2DirectDotKernels[] = {
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c1"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c2"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c3"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c4"),
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_symmetric_i4_lowrow_split_k2_direct_dot_c5"),
};
static constexpr KernelCatalogRef kQuantizeF32SymmetricI8K256Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_quantize_f32_symmetric_i8_k256");
static constexpr KernelCatalogRef kMulMatQ5KSymmetricI8WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_symmetric_i8_wmma");
static constexpr KernelCatalogRef kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256");
static constexpr KernelCatalogRef kMulMatQ4KF16WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q4_k_f16_wmma_prefill_wave32");
static constexpr KernelCatalogRef kMulMatVectorQ6F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_q6_f32_f32");
static constexpr int64_t          kMulMatVectorDenseMaxOutputSize           = 524288;
static constexpr int64_t          kMulMatVectorPackedMaxOutputSize          = 262144;
static constexpr int64_t          kMulMatVectorSelectedOutputsMaxOutputSize = 262144;
static constexpr KernelCatalogRef kMulMatQ6KF32WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_f32_wmma_prefill_wave32");
static constexpr KernelCatalogRef kMulMatQ6KF16WmmaPrefillWave32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_f16_wmma_prefill_wave32");
static constexpr KernelCatalogRef kSelectSymmetricI4K32GroupsKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_select_symmetric_i4_k32_groups");
static constexpr KernelCatalogRef kMulMatQ6KSymmetricI2ScanToken1Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q6_k_symmetric_i2_scan_token1");
static constexpr KernelCatalogRef kTopK8F32PartitionsRegisterKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k8_f32_partitions_register");
static constexpr KernelCatalogRef kTopK128F32ReduceGatherRegisterKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_top_k128_f32_reduce_gather_register");
static constexpr KernelCatalogRef kFillNegativeF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_fill_negative_f32");
static constexpr KernelCatalogRef kMulMatVectorQ6SelectedOutputsF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_vector_q6_selected_outputs_f32_f32");
static constexpr int64_t kQwenVectorVocabularyCount = 151936;

static bool deprecated_mul_mat_dispatch_disabled() {
    return std::getenv("GGML_HRX_DISABLE_DEPRECATED_MUL_MAT_DISPATCH") != nullptr;
}

static bool iq_codebook_matmul_disabled(CommonMulMatWeightFormat format) {
    if (std::getenv("GGML_HRX_DISABLE_IQ_CODEBOOK_MATMUL") != nullptr) {
        return true;
    }
    if (format == CommonMulMatWeightFormat::IQ2_XXS) {
        return std::getenv("GGML_HRX_DISABLE_IQ2_XXS_CODEBOOK_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ2_XS) {
        return std::getenv("GGML_HRX_DISABLE_IQ2_XS_CODEBOOK_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ2_S) {
        return std::getenv("GGML_HRX_DISABLE_IQ2_S_GRID_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ3_S) {
        return std::getenv("GGML_HRX_DISABLE_IQ3_S_CODEBOOK_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ3_XXS) {
        return std::getenv("GGML_HRX_DISABLE_IQ3_XXS_CODEBOOK_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ1_S) {
        return std::getenv("GGML_HRX_DISABLE_IQ1_S_CODEBOOK_MATMUL") != nullptr;
    }
    if (format == CommonMulMatWeightFormat::IQ1_M) {
        return std::getenv("GGML_HRX_DISABLE_IQ1_M_CODEBOOK_MATMUL") != nullptr;
    }
    return false;
}

static bool iq_codebook_matmul_required(CommonMulMatWeightFormat format) {
    return format == CommonMulMatWeightFormat::IQ1_S || format == CommonMulMatWeightFormat::IQ1_M ||
           format == CommonMulMatWeightFormat::IQ2_XXS || format == CommonMulMatWeightFormat::IQ2_XS ||
           format == CommonMulMatWeightFormat::IQ3_XXS;
}

static const IQCodebookResourceSpec * iq_codebook_for_weight_format(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::IQ1_S:
        case CommonMulMatWeightFormat::IQ1_M:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ1);
        case CommonMulMatWeightFormat::IQ2_XXS:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ2XXS);
        case CommonMulMatWeightFormat::IQ2_XS:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ2XS);
        case CommonMulMatWeightFormat::IQ3_XXS:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ3XXS);
        case CommonMulMatWeightFormat::IQ2_S:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ2S);
        case CommonMulMatWeightFormat::IQ3_S:
            return &iq_codebook_resource_spec(IQCodebookResource::IQ3S);
        default:
            return nullptr;
    }
}

static ValueId require_iq_codebook(const DispatchMatchContext &   context,
                                   DispatchMatch &                dispatch_match,
                                   const IQCodebookResourceSpec & codebook) {
    return require_immutable_program_resource(context, dispatch_match,
                                              { codebook.key, codebook.data, codebook.alignment });
}

static KernelCatalogRef tiled_iq_codebook_kernel(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::IQ1_S:
            return kMulMatTiledIQ1SF32F32Kernel;
        case CommonMulMatWeightFormat::IQ1_M:
            return kMulMatTiledIQ1MF32F32Kernel;
        case CommonMulMatWeightFormat::IQ3_XXS:
            return kMulMatTiledIQ3XXSF32F32Kernel;
        case CommonMulMatWeightFormat::IQ3_S:
            return kMulMatTiledIQ3SF32F32Kernel;
        default:
            return kMulMatTiledIQ2SF32F32Kernel;
    }
}

static KernelCatalogRef vector_iq_codebook_kernel(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::IQ1_S:
            return kMulMatVectorIQ1SF32F32Kernel;
        case CommonMulMatWeightFormat::IQ1_M:
            return kMulMatVectorIQ1MF32F32Kernel;
        case CommonMulMatWeightFormat::IQ3_XXS:
            return kMulMatVectorIQ3XXSF32F32Kernel;
        case CommonMulMatWeightFormat::IQ3_S:
            return kMulMatVectorIQ3SF32F32Kernel;
        default:
            return kMulMatVectorIQ2SF32F32Kernel;
    }
}

struct MulMatPostOpsMatch {
    const Value *                  input             = nullptr;
    const Value *                  weight            = nullptr;
    const Value *                  projection_output = nullptr;
    const Value *                  bias              = nullptr;
    const Value *                  residual_input    = nullptr;
    const Value *                  residual_output   = nullptr;
    const GraphNode *              bias_add_node     = nullptr;
    const GraphNode *              residual_add_node = nullptr;
    const GraphNode *              layout_node       = nullptr;
    std::vector<const GraphNode *> add_nodes;
    KernelCatalogRef               kernel                 = {};
    CommonMulMatWeightFormat       weight_format          = CommonMulMatWeightFormat::Q4K;
    int64_t                        input_size             = 0;
    int64_t                        output_size            = 0;
    int64_t                        token_count            = 0;
    bool                           has_bias               = false;
    bool                           has_residual           = false;
    bool                           requires_q8_activation = false;

    bool matched() const {
        return input != nullptr && weight != nullptr && projection_output != nullptr && residual_output != nullptr &&
               kernel.id != kUncatalogedKernelId && (has_bias || has_residual);
    }
};

struct DecodeMulMatAddMatch {
    CommonMulMatMatch root;
    const Value *     residual_input  = nullptr;
    const Value *     residual_output = nullptr;
    const GraphNode * add_node        = nullptr;

    bool matched() const {
        return root.matched() && residual_input != nullptr && residual_output != nullptr && add_node != nullptr;
    }
};

struct VectorPublishPlan {
    CommonActivationPublicationDemand demand;
    CommonActivationPublication       publication;
};

static bool common_mul_mat_is_low_token_batch(int64_t token_count) {
    return token_count >= 2 && token_count <= 5;
}

static bool common_mul_mat_has_skinny_reduction_shape(int64_t input_size) {
    return input_size % 256 == 0;
}

static bool common_mul_mat_is_skinny_route(const CommonMulMatMatch & match) {
    return match.matched() && common_mul_mat_is_low_token_batch(match.token_count) &&
           common_mul_mat_has_skinny_reduction_shape(match.input_size);
}

// Skinny owns the aligned 2..5 token optimization. Tiled owns normal prefill
// and the low-token tail fallback shapes.
static bool common_mul_mat_is_tiled_route(const CommonMulMatMatch & match) {
    return match.matched() && match.token_count > 1 && !common_mul_mat_is_skinny_route(match);
}

static bool common_mul_mat_postops_is_skinny_route(const MulMatPostOpsMatch & match) {
    return match.matched() && common_mul_mat_is_low_token_batch(match.token_count) &&
           common_mul_mat_has_skinny_reduction_shape(match.input_size);
}

static bool common_mul_mat_postops_is_tiled_route(const MulMatPostOpsMatch & match) {
    return match.matched() && match.layout_node == nullptr && !match.requires_q8_activation && match.token_count > 1 &&
           !common_mul_mat_postops_is_skinny_route(match);
}

static bool common_mul_mat_uses_factored_tiled_kernel(KernelCatalogRef kernel) {
    return kernel.id == kMulMatTiledF32F32Kernel.id || kernel.id == kMulMatTiledF32F16AlternateKernel.id ||
           kernel.id == kMulMatTiledBiasAddF32F32Kernel.id;
}

static bool is_bias_shape(const Value & value, int64_t output_size) {
    if (value.kind != ValueKind::External || value.type != GGML_TYPE_F32 || !value.contiguous ||
        value.ne[0] != output_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool accepts_tiled_q8_publication(const DispatchMatchContext & context,
                                         const GraphNode &            consumer,
                                         const Value &                input) {
    return common_accepts_q8_1_x4_prefill_mul_mat(context.graph, &consumer, input);
}

static bool accepts_preferred_q8_publication(const DispatchMatchContext & context,
                                             const GraphNode &            consumer,
                                             const Value &                input) {
    if (common_accepts_q8_1_x4_prefill_mul_mat(context.graph, &consumer, input)) {
        return true;
    }
    if (!common_accepts_q8_1_x4_decode_mul_mat(context.graph, &consumer, input)) {
        return false;
    }
    const Value * weight = context.graph.values().find(consumer.inputs[0]);
    const Value * output = context.graph.values().find(consumer.output);
    return weight != nullptr && output != nullptr &&
           (weight->type == GGML_TYPE_Q4_K || !context.graph.index().consumers(output->id).empty());
}

static bool is_q1_0_q8_1_x4_vector_consumer(const CommonMulMatActivationConsumerMatch & match) {
    return match.matched() && match.weight->alias_source.value < 0 && match.token_count == 1 &&
           match.input_size % 128 == 0 && match.weight_format == CommonMulMatWeightFormat::Q1_0;
}

static bool accepts_preferred_q1_0_q8_publication(const DispatchMatchContext & context,
                                                   const GraphNode &            consumer,
                                                   const Value &                input) {
    const CommonMulMatActivationConsumerMatch match =
        common_match_mul_mat_activation_consumer(context.graph, &consumer, input);
    if (!is_q1_0_q8_1_x4_vector_consumer(match)) {
        return false;
    }
    int64_t compatible_consumer_count = 0;
    int64_t total_output_size         = 0;
    for (const GraphNode * candidate : context.graph.index().consumers(input.id)) {
        const CommonMulMatActivationConsumerMatch candidate_match =
            common_match_mul_mat_activation_consumer(context.graph, candidate, input);
        if (!is_q1_0_q8_1_x4_vector_consumer(candidate_match)) {
            continue;
        }
        ++compatible_consumer_count;
        total_output_size += candidate_match.output_size;
    }
    return compatible_consumer_count >= 2 && total_output_size >= std::max<int64_t>(match.input_size, 4096);
}

static bool accepts_factored_tiled_f16_input(const DispatchMatchContext & context,
                                             const GraphNode &            consumer,
                                             const Value &                input) {
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, &consumer, kMulMatTiledF32F32Kernel, false);
    return match.matched() && match.input->id == input.id && common_mul_mat_is_tiled_route(match);
}

static bool has_normalized_rope_consumer(const Graph & graph, const Value & value) {
    if (!graph.has_index()) {
        return false;
    }
    for (const GraphNode * layout : graph.index().consumers(value.id)) {
        if (layout == nullptr || (layout->op != GGML_OP_RESHAPE && layout->op != GGML_OP_VIEW)) {
            continue;
        }
        const GraphNode * rms = common_find_single_consumer_with_op(graph, layout->output, GGML_OP_RMS_NORM);
        const GraphNode * mul =
            rms != nullptr ? common_find_single_consumer_with_op(graph, rms->output, GGML_OP_MUL) : nullptr;
        if (mul != nullptr && common_find_single_consumer_with_op(graph, mul->output, GGML_OP_ROPE) != nullptr) {
            return true;
        }
    }
    return false;
}

static bool has_reshaped_unary_mul_consumer(const Graph & graph, const Value & value) {
    const GraphNode * reshape = common_find_single_consumer_with_op(graph, value.id, GGML_OP_RESHAPE);
    const GraphNode * unary =
        reshape != nullptr ? common_find_single_consumer_with_op(graph, reshape->output, GGML_OP_UNARY) : nullptr;
    return unary != nullptr && common_find_single_consumer_with_op(graph, unary->output, GGML_OP_MUL) != nullptr;
}

static bool has_reshaped_transpose_concat_consumer(const Graph & graph, const Value & value) {
    const GraphNode * reshape = common_find_single_consumer_with_op(graph, value.id, GGML_OP_RESHAPE);
    const GraphNode * transpose =
        reshape != nullptr ? common_find_single_consumer_with_op(graph, reshape->output, GGML_OP_TRANSPOSE) : nullptr;
    return transpose != nullptr &&
           common_find_single_consumer_with_op(graph, transpose->output, GGML_OP_CONCAT) != nullptr;
}

static const Value * match_scaled_normalized_branch(const Graph & graph, ValueId branch_id, int64_t hidden_size) {
    const Value *     branch = common_graph_value(graph, branch_id);
    const GraphNode * mul    = graph.index().producer(branch_id);
    if (branch == nullptr || mul == nullptr || !common_binary_node_is_mul(*mul) || branch->type != GGML_TYPE_F32 ||
        !common_is_2d(*branch) || branch->ne[0] != hidden_size || branch->ne[1] != 1) {
        return nullptr;
    }

    const GraphNode * rms   = nullptr;
    const Value *     scale = nullptr;
    for (ValueId input_id : mul->inputs) {
        const GraphNode * producer = graph.index().producer(input_id);
        if (producer != nullptr && producer->op == GGML_OP_RMS_NORM) {
            if (rms != nullptr) {
                return nullptr;
            }
            rms = producer;
        } else {
            scale = common_graph_value(graph, input_id);
        }
    }
    if (rms == nullptr || rms->inputs.size() != 1 || scale == nullptr || scale->type != GGML_TYPE_F32 ||
        !common_is_2d(*scale) || scale->ne[0] != hidden_size || scale->ne[1] != 1) {
        return nullptr;
    }

    const Value * normalized = common_graph_value(graph, rms->output);
    const Value * source     = common_graph_value(graph, rms->inputs[0]);
    if (normalized == nullptr || source == nullptr || normalized->type != GGML_TYPE_F32 ||
        source->type != GGML_TYPE_F32 || !common_is_2d(*normalized) || !common_is_2d(*source) ||
        normalized->ne[0] != hidden_size || normalized->ne[1] != 1 || source->ne[0] != hidden_size ||
        source->ne[1] != 1) {
        return nullptr;
    }
    return source;
}

static bool matches_embedded_external_concat_projection(const Graph &     graph,
                                                        const GraphNode & projection,
                                                        int64_t           hidden_size) {
    if (projection.op != GGML_OP_MUL_MAT || projection.inputs.size() != 2) {
        return false;
    }

    const Value * weight = common_graph_value(graph, projection.inputs[0]);
    const Value * input  = common_graph_value(graph, projection.inputs[1]);
    const Value * output = common_graph_value(graph, projection.output);
    if (weight == nullptr || input == nullptr || output == nullptr || !common_is_2d(*weight) || !common_is_2d(*input) ||
        !common_is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || weight->ne[0] != 2 * hidden_size ||
        weight->ne[1] != hidden_size || input->ne[0] != 2 * hidden_size || input->ne[1] != 1 ||
        output->ne[0] != hidden_size || output->ne[1] != 1) {
        return false;
    }

    const GraphNode * concat = graph.index().producer(input->id);
    if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->inputs.size() != 2) {
        return false;
    }
    const Value * first_source  = match_scaled_normalized_branch(graph, concat->inputs[0], hidden_size);
    const Value * second_source = match_scaled_normalized_branch(graph, concat->inputs[1], hidden_size);
    if (first_source == nullptr || second_source == nullptr) {
        return false;
    }

    const GraphNode * first_producer  = graph.index().producer(first_source->id);
    const GraphNode * second_producer = graph.index().producer(second_source->id);
    const bool        first_is_embedding =
        first_producer != nullptr && first_producer->op == GGML_OP_GET_ROWS && first_producer->inputs.size() == 2;
    const bool second_is_embedding =
        second_producer != nullptr && second_producer->op == GGML_OP_GET_ROWS && second_producer->inputs.size() == 2;
    const bool first_is_external  = first_producer == nullptr;
    const bool second_is_external = second_producer == nullptr;
    return (first_is_embedding && second_is_external) || (second_is_embedding && first_is_external);
}

static bool has_embedded_external_concat_projection_ancestor(const Graph & graph,
                                                             ValueId       endpoint_input,
                                                             int64_t       hidden_size) {
    if (!graph.has_index() || endpoint_input.value < 0 || hidden_size <= 0) {
        return false;
    }

    std::vector<bool>    visited(graph.values().size(), false);
    std::vector<ValueId> pending = { endpoint_input };
    while (!pending.empty()) {
        const ValueId current = pending.back();
        pending.pop_back();
        if (current.value < 0 || static_cast<size_t>(current.value) >= visited.size() || visited[current.value]) {
            continue;
        }
        visited[current.value]     = true;
        const GraphNode * producer = graph.index().producer(current);
        if (producer == nullptr) {
            continue;
        }
        if (matches_embedded_external_concat_projection(graph, *producer, hidden_size)) {
            return true;
        }
        pending.insert(pending.end(), producer->inputs.begin(), producer->inputs.end());
    }
    return false;
}

static bool is_qwen_endpoint_rmsnorm_projection(const Graph & graph, const CommonMulMatMatch & match) {
    if (!graph.has_index() || match.weight == nullptr || match.input == nullptr ||
        match.weight->type != GGML_TYPE_Q6_K || match.token_count != 1 ||
        match.output_size != kQwenVectorVocabularyCount) {
        return false;
    }

    const GraphNode * mul = graph.index().producer(match.input->id);
    if (mul == nullptr || mul->op != GGML_OP_MUL || mul->inputs.size() != 2) {
        return false;
    }
    for (ValueId input : mul->inputs) {
        const GraphNode * producer = graph.index().producer(input);
        if (producer != nullptr && producer->op == GGML_OP_RMS_NORM) {
            return true;
        }
    }
    return false;
}

static bool match_symmetric_i4_low_row_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4LowRowWmmaKernel, false);
    if (!match.matched()) {
        match =
            common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4LowRowWmmaKernel, true);
    }
    if (!match.matched() ||
        !common_symmetric_i4_lowrow_mul_mat_eligible(context.graph, *context.root_node, *match.input)) {
        return false;
    }
    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kCommonSymmetricI4K32ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t next_value = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                                   common_to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                                   common_to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.input->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                  kCommonSymmetricI4K32ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    const size_t materialized_weight_bytes =
        common_symmetric_i4_shared4_weight_byte_count(match.input_size, match.output_size);
    const bool split_k = match.input_size % 128 == 0 && materialized_weight_bytes >= size_t{ 16 } * 1024 * 1024 &&
                         match.output_size <= 2 * match.input_size;

    const ValueId  partial             = split_k ? ValueId(next_value) : ValueId{};
    const ValueId  completion_counters = split_k ? ValueId(next_value + 1) : ValueId{};
    const uint32_t completion_counter_count =
        static_cast<uint32_t>(common_ceil_div(match.output_size, 16) * common_ceil_div(match.token_count, 16));
    if (split_k) {
        dispatch_match.transients.push_back(
            { partial, "common.mul_mat.symmetric_i4_lowrow.split_k_partial", match.output->byte_count, 256 });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "common.mul_mat.symmetric_i4_lowrow.split_k_completion_counters",
            completion_counter_count,
        });
    }

    const bool use_split_direct_dot = split_k && match.token_count <= 5 && match.input_size % 256 == 0;

    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(
        use_split_direct_dot ? kMulMatSymmetricI4LowRowSplitK2DirectDotKernels[match.token_count - 1] :
        split_k              ? kMulMatSymmetricI4LowRowSplitK2WmmaKernel :
                               kMulMatSymmetricI4LowRowWmmaKernel);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.lowrow.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.kernel.compile_parameters.emplace(
        "ggml.mul_mat.symmetric_i4.lowrow.row_group_size",
        common_to_config_value(
            static_cast<int64_t>(common_symmetric_shared4_row_group_size(match.input_size, match.output_size, 4))));
    contraction.bindings.push_back(
        match.weight->type == GGML_TYPE_Q5_K && match.token_count == 1 ?
            common_symmetric_i4_shared4_multistart_weight_binding(*match.weight, match.input_size, match.output_size) :
            common_symmetric_i4_shared4_weight_binding(*match.weight, match.input_size, match.output_size));
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
    if (split_k) {
        contraction.bindings.push_back({ partial, 0, match.output->byte_count });
        contraction.bindings.push_back({ completion_counters, 0, completion_counter_count * sizeof(int32_t) });
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(contraction));
    return dispatch_match.status.success();
}

static bool match_q6_vector_selected_outputs_dispatch(const DispatchMatchContext & context,
                                                      DispatchMatch &              dispatch_match) {
    constexpr int64_t selected_group_count       = 96;
    constexpr int64_t selected_output_count      = 128;
    constexpr int64_t selected_output_chunk_size = 64;
    constexpr size_t  partition_entry_count      = 1024;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KSymmetricI2ScanToken1Kernel, true);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q6_K ||
        match.token_count != 1 || match.weight->alias_source.value >= 0 || match.input_size % 256 != 0 ||
        match.input_size > 8192 || match.output_size < 65536 || match.output_size % 64 != 0 ||
        match.output_size > kMulMatVectorSelectedOutputsMaxOutputSize || match.output_size < 16 * match.input_size ||
        !context.graph.index().consumers(match.output->id).empty() ||
        !has_embedded_external_concat_projection_ancestor(context.graph, match.input->id, match.input_size)) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const CommandPlanAlternateValue * alternate = find_alternate_value(context.graph, context.plan, match.input->id,
                                                                       GGML_TYPE_COUNT, activation_layout.total_bytes);
    if (alternate != nullptr && alternate->name != kCommonSymmetricI4K32ActivationAlternateName) {
        alternate = nullptr;
    }

    const ValueId activation     = alternate != nullptr ? alternate->alternate_value : context.next_plan_value;
    const int32_t transient_base = context.next_plan_value.value + (alternate == nullptr ? 1 : 0);
    const ValueId selected_groups(transient_base);
    const ValueId partial_values(transient_base + 1);
    const ValueId partial_ids(transient_base + 2);
    const ValueId selected_outputs(transient_base + 3);
    const ValueId selected_values(transient_base + 4);

    if (alternate == nullptr) {
        dispatch_match.transients.push_back(
            { activation, kCommonSymmetricI4K32ActivationAlternateName, activation_layout.total_bytes, 256 });

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K32Kernel);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.input_size",
                                                   common_to_config_value(match.input_size));
        quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k32.token_count",
                                                   common_to_config_value(match.token_count));
        quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
        quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
        quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });
        dispatch_match.dispatches.push_back(std::move(quantize));

        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.input->id, activation, GGML_TYPE_COUNT, activation_layout.total_bytes,
                  kCommonSymmetricI4K32ActivationAlternateName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    dispatch_match.transients.push_back({ selected_groups, "common.mul_mat.vector_q6_selected_outputs.groups",
                                          static_cast<size_t>(selected_group_count) * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back({ partial_values, "common.mul_mat.vector_q6_selected_outputs.partial_values",
          partition_entry_count * sizeof(float), 256 });
    dispatch_match.transients.push_back({ partial_ids, "common.mul_mat.vector_q6_selected_outputs.partial_ids",
          partition_entry_count * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back({ selected_outputs, "common.mul_mat.vector_q6_selected_outputs.indices",
          selected_output_count * sizeof(int32_t), 256 });
    dispatch_match.transients.push_back({ selected_values, "common.mul_mat.vector_q6_selected_outputs.values",
          selected_output_count * sizeof(float), 256 });

    Dispatch select_groups;
    select_groups.kernel = make_kernel_specialization(kSelectSymmetricI4K32GroupsKernel);
    select_groups.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.input_size",
                                                    common_to_config_value(match.input_size));
    select_groups.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.selected_group_count",
                                                    common_to_config_value(selected_group_count));
    select_groups.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    select_groups.bindings.push_back(
        { selected_groups, 0, static_cast<size_t>(selected_group_count) * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(select_groups));

    const size_t symmetric_i2_bytes =
        common_symmetric_shared4_weight_byte_count(match.input_size, match.output_size, 2);
    if (symmetric_i2_bytes > std::numeric_limits<size_t>::max() - match.weight->byte_count) {
        return false;
    }
    const size_t materialized_weight_bytes = symmetric_i2_bytes + match.weight->byte_count;
    Dispatch     scan;
    scan.kernel = make_kernel_specialization(kMulMatQ6KSymmetricI2ScanToken1Kernel);
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.input_size",
                                           common_to_config_value(match.input_size));
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.output_size",
                                           common_to_config_value(match.output_size));
    scan.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_shortlist.selected_group_count",
                                           common_to_config_value(selected_group_count));
    scan.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                              kQ6KSymmetricI2PackedK256Row64ScaleRowLayout, match.weight->type, match.input_size,
                              match.output_size, match.weight->byte_count });
    scan.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    scan.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    scan.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    scan.bindings.push_back({ selected_groups, 0, static_cast<size_t>(selected_group_count) * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(scan));

    Dispatch partition_top_k;
    partition_top_k.kernel = make_kernel_specialization(kTopK8F32PartitionsRegisterKernel);
    partition_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    partition_top_k.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    partition_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    partition_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    dispatch_match.dispatches.push_back(std::move(partition_top_k));

    Dispatch reduce_top_k;
    reduce_top_k.kernel = make_kernel_specialization(kTopK128F32ReduceGatherRegisterKernel);
    reduce_top_k.kernel.integer_parameters.emplace("element_count", match.output_size);
    reduce_top_k.bindings.push_back({ partial_values, 0, partition_entry_count * sizeof(float) });
    reduce_top_k.bindings.push_back({ partial_ids, 0, partition_entry_count * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ selected_outputs, 0, selected_output_count * sizeof(int32_t) });
    reduce_top_k.bindings.push_back({ selected_values, 0, selected_output_count * sizeof(float) });
    dispatch_match.dispatches.push_back(std::move(reduce_top_k));

    Dispatch fill;
    fill.kernel = make_kernel_specialization(kFillNegativeF32Kernel);
    fill.kernel.integer_parameters.emplace("element_count", match.output_size);
    fill.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch_match.dispatches.push_back(std::move(fill));

    for (int64_t selected_output_offset = 0; selected_output_offset < selected_output_count;
         selected_output_offset += selected_output_chunk_size) {
        Dispatch selected_output_pass;
        selected_output_pass.kernel = make_kernel_specialization(kMulMatVectorQ6SelectedOutputsF32Kernel);
        selected_output_pass.kernel.integer_parameters.emplace("token_count", match.token_count);
        selected_output_pass.kernel.integer_parameters.emplace("selected_count", selected_output_chunk_size);
        selected_output_pass.kernel.compile_parameters.emplace("ggml.matmul.vector.input_size",
                                                               common_to_config_value(match.input_size));
        selected_output_pass.kernel.compile_parameters.emplace("ggml.matmul.vector.output_size",
                                                               common_to_config_value(match.output_size));
        selected_output_pass.kernel.compile_parameters.emplace("ggml.matmul.vector.output_accumulation", "0");
        selected_output_pass.kernel.compile_parameters.emplace(
            "ggml.matmul.vector.weight_offset", common_to_config_value(static_cast<int64_t>(symmetric_i2_bytes)));
        selected_output_pass.bindings.push_back({ match.input->id, 0, match.input->byte_count });
        selected_output_pass.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes,
                                                  kQ6KSymmetricI2PackedK256Row64ScaleRowLayout, match.weight->type,
                                                  match.input_size, match.output_size, match.weight->byte_count });
        selected_output_pass.bindings.push_back({ selected_outputs,
                                                  static_cast<size_t>(selected_output_offset) * sizeof(int32_t),
              static_cast<size_t>(selected_output_chunk_size) * sizeof(int32_t) });
        selected_output_pass.bindings.push_back({ selected_values,
                                                  static_cast<size_t>(selected_output_offset) * sizeof(float),
              static_cast<size_t>(selected_output_chunk_size) * sizeof(float) });
        selected_output_pass.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch_match.dispatches.push_back(std::move(selected_output_pass));
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    return dispatch_match.status.success();
}

enum class Q6VectorStorage : int64_t {
    Packed = 0,
    I8     = 1,
};

static bool build_q6_vector_dispatch(const CommonMulMatMatch & match,
                                     size_t                    root_index,
                                     Q6VectorStorage           storage,
                                     size_t                    weight_bytes,
                                     DispatchMatch &           dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatVectorQ6F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.q6_storage",
                                               common_to_config_value(static_cast<int64_t>(storage)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    if (storage == Q6VectorStorage::I8) {
        dispatch.bindings.push_back({ match.weight->id, 0, weight_bytes, kQ6KI8K32Row64Layout, match.weight->type,
                                      match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, weight_bytes, kQ6KPackedK256Row64ScaleRowLayout,
                                      match.weight->type, match.input_size, match.output_size,
                                      match.weight->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool build_q6_vector_dispatch(const CommonMulMatMatch & match,
                                     size_t                    root_index,
                                     DispatchMatch &           dispatch_match) {
    return build_q6_vector_dispatch(match, root_index, Q6VectorStorage::Packed, match.weight->byte_count,
                                    dispatch_match);
}

static bool match_q6_vector_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatVectorQ6F32Kernel, true);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q6_K ||
        match.weight->alias_source.value >= 0 || match.token_count != 1 || match.input_size % 256 != 0 ||
        match.output_size % 64 != 0 || match.output_size > kMulMatVectorPackedMaxOutputSize ||
        !context.graph.index().consumers(match.output->id).empty()) {
        return false;
    }
    if (is_qwen_endpoint_rmsnorm_projection(context.graph, match)) {
        return false;
    }
    if (find_alternate_value(
            context.graph, context.plan, match.input->id, GGML_TYPE_Q8_1,
            static_cast<size_t>(match.token_count) * ggml_row_size(GGML_TYPE_Q8_1, match.input_size)) != nullptr) {
        return false;
    }

    return build_q6_vector_dispatch(match, context.root_index, dispatch_match);
}

static bool build_q6_k_prefill_wave32_dispatch(const CommonMulMatMatch & match,
                                               ValueId                   input,
                                               size_t                    input_bytes,
                                               KernelCatalogRef          kernel,
                                               size_t                    root_index,
                                               DispatchMatch &           dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q6_k_packed.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.bindings.push_back({ input, 0, input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q6_k_prefill_wave32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ6KF32WmmaPrefillWave32Kernel, false);
    if (!match.matched() || match.weight->type != GGML_TYPE_Q6_K || match.weight->alias_source.value >= 0 ||
        match.token_count < 128 || match.token_count > 2048 || match.token_count % 128 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0) {
        return false;
    }

    const bool packed_input =
        common_mul_mat_uses_k16_major_f16(match.weight_format, match.input_size, match.output_size, match.token_count);
    DispatchBinding activation;
    const bool prepared = packed_input ?
                                   common_prepare_k16_major_f16_input(context, *match.input, match.input_size,
                                                                      match.token_count, dispatch_match, activation) :
        common_prepare_f16_input(context, *match.input, match.input_size, match.token_count,
                                  dispatch_match, activation);
    if (!prepared) {
        return false;
    }
    return build_q6_k_prefill_wave32_dispatch(match, activation.value, activation.length,
                                              kMulMatQ6KF16WmmaPrefillWave32Kernel, context.root_index, dispatch_match);
}

static bool match_symmetric_i4_prefill_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    // Disabled: this symmetric matmul route causes a substantial numeric performance regression.
    (void) context;
    (void) dispatch_match;
    return false;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSymmetricI4WmmaKernel, false);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q5_K ||
        match.weight->alias_source.value >= 0 || match.token_count < 128 || match.token_count % 128 != 0 ||
        match.input_size % 64 != 0 || match.output_size < match.input_size || match.output_size % 128 != 0) {
        return false;
    }

    const GraphNode * input_producer = context.graph.index().producer(match.input->id);
    if (input_producer != nullptr && input_producer->op == GGML_OP_GLU) {
        return false;
    }
    if (!has_normalized_rope_consumer(context.graph, *match.output) &&
        !has_reshaped_unary_mul_consumer(context.graph, *match.output) &&
        !has_reshaped_transpose_concat_consumer(context.graph, *match.output)) {
        return false;
    }

    const CommonSymmetricI4ActivationLayout activation_layout =
        common_symmetric_i4_activation_layout(match.input_size, match.token_count);
    const ValueId activation = context.next_plan_value;
    dispatch_match.transients.push_back(
        { activation, "common.mul_mat.symmetric_i4_k64.activation", activation_layout.total_bytes, 256 });

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI4K64PlaneKernel);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k64.input_size",
                                               common_to_config_value(match.input_size));
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i4_k64.token_count",
                                               common_to_config_value(match.token_count));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    quantize.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    quantize.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    const size_t materialized_weight_bytes =
        static_cast<size_t>(match.output_size) * static_cast<size_t>(match.input_size / 256) * size_t{ 144 };
    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kMulMatSymmetricI4WmmaKernel);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i4.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kSymmetricI4K64Row64Layout,
                                     match.weight->type, match.input_size, match.output_size,
                                     match.weight->byte_count });
    contraction.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    contraction.bindings.push_back({ activation, 0, activation_layout.payload_bytes });
    contraction.bindings.push_back({ activation, activation_layout.scales_offset, activation_layout.metadata_bytes });
    contraction.bindings.push_back({ activation, activation_layout.sums_offset, activation_layout.metadata_bytes });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(contraction));
    return true;
}

static bool match_q5_k_symmetric_i8_prefill_dispatch(const DispatchMatchContext & context,
                                                     DispatchMatch &              dispatch_match) {
    // Disabled: this symmetric matmul route causes a substantial numeric performance regression.
    (void) context;
    (void) dispatch_match;
    return false;

    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatQ5KSymmetricI8WmmaKernel, false);
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q5_K ||
        match.weight->alias_source.value >= 0 || match.token_count < 256 || match.token_count % 256 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0 || match.output_size > match.input_size) {
        return false;
    }

    const GraphNode * input_producer = context.graph.index().producer(match.input->id);
    if (input_producer != nullptr && input_producer->op == GGML_OP_GLU) {
        return false;
    }

    const size_t element_count = static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size);
    const size_t metadata_bytes =
        static_cast<size_t>(match.token_count) * static_cast<size_t>(match.input_size / 256) * sizeof(int32_t);
    const size_t  quantized_bytes = element_count + metadata_bytes;
    const ValueId quantized       = context.next_plan_value;
    dispatch_match.transients.push_back(
        { quantized, "common.mul_mat.symmetric_i8_k256.activation", quantized_bytes, 256 });

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeF32SymmetricI8K256Kernel);
    quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
    quantize.kernel.integer_parameters.emplace("input_size_arg", match.input_size);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i8_k256.token_count",
                                               common_to_config_value(match.token_count));
    quantize.kernel.compile_parameters.emplace("ggml.quantize_symmetric_i8_k256.input_size",
                                               common_to_config_value(match.input_size));
    quantize.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    quantize.bindings.push_back({ quantized, 0, quantized_bytes });

    const size_t materialized_weight_bytes =
        static_cast<size_t>(match.output_size) * static_cast<size_t>(match.input_size / 256) * size_t{ 258 };
    Dispatch contraction;
    contraction.kernel = make_kernel_specialization(kMulMatQ5KSymmetricI8WmmaKernel);
    contraction.kernel.integer_parameters.emplace("token_count", match.token_count);
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.input_size",
                                                  common_to_config_value(match.input_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.output_size",
                                                  common_to_config_value(match.output_size));
    contraction.kernel.compile_parameters.emplace("ggml.mul_mat.symmetric_i8.token_count",
                                                  common_to_config_value(match.token_count));
    contraction.bindings.push_back({ quantized, 0, quantized_bytes });
    contraction.bindings.push_back({ match.weight->id, 0, materialized_weight_bytes, kQ5KSymmetricI8K256Row64Layout,
                                     match.weight->type, match.input_size, match.output_size,
                                     match.weight->byte_count });
    contraction.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(quantize));
    dispatch_match.dispatches.push_back(std::move(contraction));
    return true;
}

static bool match_packed_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                                  DispatchMatch &              dispatch_match) {
    const CommonMulMatMatch match = common_match_mul_mat_any_format(context.graph, context.root_node,
                                                                    kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel, false);
    if (!match.matched() || !context.graph.has_index() ||
        (match.weight->type != GGML_TYPE_Q4_K && match.weight->type != GGML_TYPE_Q5_K &&
         match.weight->type != GGML_TYPE_IQ4_XS) ||
        match.token_count < 256 || match.token_count > 2048 || match.token_count % 256 != 0 ||
        match.input_size % 256 != 0 || match.output_size % 64 != 0) {
        return false;
    }

    for (const GraphNode * consumer : context.graph.index().consumers(match.output->id)) {
        if (match.weight->type != GGML_TYPE_Q4_K && consumer != nullptr && consumer->op == GGML_OP_GLU) {
            return false;
        }
    }

    const bool use_f16 = match.weight->type == GGML_TYPE_Q4_K && match.weight->alias_source.value < 0 &&
                         match.output_size >= match.input_size / 4;
    DispatchBinding activation;
    const bool      packed_input =
        use_f16 &&
        common_mul_mat_uses_k16_major_f16(match.weight_format, match.input_size, match.output_size, match.token_count,
                                          context.plan.metadata.find_generated_resource(
                                              match.input->id, GeneratedResourceRole::F16K16Major) != nullptr);
    if (use_f16) {
        const bool prepared = packed_input ?
                                  common_prepare_k16_major_f16_input(context, *match.input, match.input_size,
                                                                     match.token_count, dispatch_match, activation) :
            common_prepare_f16_input(context, *match.input, match.input_size, match.token_count,
                                      dispatch_match, activation);
        if (!prepared) {
            return false;
        }
    } else if (!common_prepare_q8_1_x4_input(
                   context, *match.input, match.input_size, match.token_count, dispatch_match, activation,
                   match.weight->type == GGML_TYPE_Q4_K ? CommonQ8ActivationPolicy::AllowStandaloneQuantize :
                                                 CommonQ8ActivationPolicy::ExistingAlternateOnly)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(use_f16 ? kMulMatQ4KF16WmmaPrefillWave32Kernel :
                                                            kMulMatQ5KIQ4XSQ8_1X4WmmaToken256Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace(use_f16 ? "ggml.mul_mat.input_size" : "ggml.mul_mat_q8_1_x4.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace(
        use_f16 ? "ggml.mul_mat.output_size" : "ggml.mul_mat_q8_1_x4.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        use_f16 ? "ggml.workload.token_capacity" : "ggml.mul_mat_q8_1_x4.token_capacity",
                                               common_to_config_value(match.token_count));
    const bool pack_q4 = match.weight_format == CommonMulMatWeightFormat::Q4K && match.weight->alias_source.value < 0;
    if (use_f16) {
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.f16_input_layout", packed_input ? "1" : "0");
    } else {
        dispatch.kernel.compile_parameters.emplace(
            "ggml.mul_mat_q8_1_x4.weight_format",
            common_to_config_value(common_mul_mat_format_config_value(pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                                                               match.weight_format)));
    }
    dispatch.bindings.push_back(activation);
    if (pack_q4) {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, kQ4KPackedK256Row64Layout,
                                      match.weight->type, match.input_size, match.output_size,
                                      match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_q4_k_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    if (context.root_node == nullptr || context.root_node->inputs.empty()) {
        return false;
    }
    const Value * weight = common_graph_value(context.graph, context.root_node->inputs[0]);
    return weight != nullptr && weight->type == GGML_TYPE_Q4_K &&
           match_packed_q8_1_x4_prefill_dispatch(context, dispatch_match);
}

static MulMatPostOpsMatch match_mul_mat_postops(const DispatchMatchContext & context,
                                                KernelCatalogRef             root_kernel,
                                                KernelCatalogRef             bias_kernel,
                                                KernelCatalogRef             add_kernel,
                                                KernelCatalogRef             bias_add_kernel,
                                                bool                         allow_batch1_bias_residual = false) {
    MulMatPostOpsMatch      match;
    CommonMulMatMatch root = common_match_mul_mat_any_format(context.graph, context.root_node, root_kernel, false);
    if (!root.matched()) {
        root = common_match_mul_mat_any_format(context.graph, context.root_node, root_kernel, true);
    }
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const bool packed_row_contraction =
        root.token_count <= 5 && root.weight->alias_source.value < 0 &&
        (root.weight_format == CommonMulMatWeightFormat::Q4K || root.weight_format == CommonMulMatWeightFormat::Q6K) &&
        root.input_size >= 4096 && root.output_size >= 4096 && root.output_size <= root.input_size &&
        root.output_size % 64 == 0;
    const bool batch1_requires_q8_activation = root.token_count == 1 && !packed_row_contraction;

    const Value * current = root.output;
    if (packed_row_contraction) {
        const GraphNode * reshape = common_find_only_consumer_with_op(context.graph, current->id, GGML_OP_RESHAPE);
        const Value * reshaped = reshape != nullptr ? common_graph_value(context.graph, reshape->output) : nullptr;
        if (reshaped != nullptr && is_layout_alias_node(context.graph, *reshape) && reshaped->contiguous &&
            same_full_value_range(*current, *reshaped) && common_same_shape(*current, *reshaped)) {
            match.layout_node = reshape;
            current = reshaped;
        }
    }
    const int add_limit =
        match.layout_node != nullptr || (root.token_count == 1 && !allow_batch1_bias_residual) ? 1 : 2;
    for (int add_index = 0; add_index < add_limit; ++add_index) {
        const GraphNode * add_node = common_find_only_consumer_with_op(context.graph, current->id, GGML_OP_ADD);
        if (add_node == nullptr || !common_binary_node_is_add(*add_node)) {
            break;
        }

        const bool current_is_lhs = add_node->inputs[0] == current->id;
        const bool current_is_rhs = add_node->inputs[1] == current->id;
        if (!current_is_lhs && !current_is_rhs) {
            return {};
        }

        const ValueId other_id = current_is_lhs ? add_node->inputs[1] : add_node->inputs[0];
        const Value * other    = common_graph_value(context.graph, other_id);
        const Value * output   = common_graph_value(context.graph, add_node->output);
        if (other == nullptr || output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
            !common_same_shape(*root.output, *output)) {
            return {};
        }

        if (!match.has_bias && is_bias_shape(*other, root.output_size)) {
            match.bias          = other;
            match.bias_add_node = add_node;
            match.add_nodes.push_back(add_node);
            match.has_bias = true;
            current        = output;
            continue;
        }

        if (!match.has_residual && other->type == GGML_TYPE_F32 && other->contiguous &&
            common_same_shape(*root.output, *other)) {
            match.residual_input    = other;
            match.residual_output   = output;
            match.residual_add_node = add_node;
            match.add_nodes.push_back(add_node);
            match.has_residual = true;
            current            = output;
            continue;
        }

        break;
    }

    if (!match.has_bias && !match.has_residual) {
        return {};
    }

    match.residual_output = current;

    if (match.has_bias && match.has_residual) {
        match.kernel = bias_add_kernel;
    } else if (match.has_residual) {
        match.kernel = add_kernel;
    } else if (match.has_bias) {
        match.kernel = bias_kernel;
    } else {
        return {};
    }

    match.input             = root.input;
    match.weight            = root.weight;
    match.projection_output = root.output;
    match.weight_format     = root.weight_format;
    match.input_size        = root.input_size;
    match.output_size       = root.output_size;
    match.token_count       = root.token_count;
    match.requires_q8_activation = batch1_requires_q8_activation;
    return match;
}

static MulMatPostOpsMatch match_mul_mat_postops(const DispatchMatchContext & context, bool allow_batch1_bias_residual) {
    return match_mul_mat_postops(context, kMulMatF32F32WmmaKernel, kMulMatVectorBiasAddF32F32Kernel,
                                 kMulMatVectorBiasAddF32F32Kernel, kMulMatVectorBiasAddF32F32Kernel,
                                 allow_batch1_bias_residual);
}

static bool try_match_fused_unary(const DispatchMatchContext & context, CommonMulMatMatch & match) {
    if (!match.matched()) {
        return false;
    }

    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(context.root_node->output);
    if (consumers.size() != 1 || consumers.front() == nullptr) {
        return false;
    }

    const GraphNode * unary       = consumers.front();
    size_t            unary_index = 0;
    if (!context.graph.index().node_index(unary, unary_index) || unary_index >= context.covered_nodes.size() ||
        context.covered_nodes[unary_index] || unary->inputs.size() != 1) {
        return false;
    }

    const UnaryParams * params = op_params_as<UnaryParams>(unary->params);
    if (params == nullptr || !unary_kind_supported(params->op)) {
        return false;
    }

    const Value * unary_input  = common_graph_value(context.graph, unary->inputs[0]);
    const Value * unary_output = common_graph_value(context.graph, unary->output);
    if (unary_input == nullptr || unary_output == nullptr || unary_input->id != match.output->id ||
        unary_output->type != GGML_TYPE_F32 || !unary_output->contiguous ||
        !common_same_shape(*match.output, *unary_output)) {
        return false;
    }

    match.output           = unary_output;
    match.output_unary_op  = params->op;
    match.unary_node_index = unary_index;
    match.has_fused_unary  = true;
    return true;
}

}  // namespace

static bool build_mul_mat_dispatch(
    const DispatchMatchContext & context,
                                   const CommonMulMatMatch &  match,
                                   DispatchMatch &            dispatch_match,
                                   CommonQ8ActivationPolicy   q8_policy = CommonQ8ActivationPolicy::ExistingAlternateOnly) {
    const CommonActivationPublicationCapabilities capabilities = {
        match.kernel.id == kMulMatTiledF32F32Kernel.id ?
            DispatchActivationInputF16Row | DispatchActivationInputQ8_1X4 :
            DispatchActivationInputNone,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, *match.output, capabilities);
    const CommonActivationPublicationDemand demand =
        publication_plan.matched() ? publication_plan.preferred() : CommonActivationPublicationDemand{};
    const CommonActivationPublicationDemand f16_demand =
        demand.format == CommonActivationPublicationFormat::F16Row ? demand : CommonActivationPublicationDemand{};
    const CommonActivationPublicationDemand q8_demand =
        demand.format == CommonActivationPublicationFormat::Q8_1X4 ? demand : CommonActivationPublicationDemand{};
    const bool publish_f16_alternate = f16_demand.matched();
    const bool publish_q8_alternate  = q8_demand.matched();
    const IQCodebookResourceSpec * codebook = iq_codebook_for_weight_format(match.weight_format);
    if (codebook != nullptr && iq_codebook_matmul_required(match.weight_format) &&
        iq_codebook_matmul_disabled(match.weight_format)) {
        return false;
    }
    const bool use_iq_codebook =
        codebook != nullptr && !publish_f16_alternate && !iq_codebook_matmul_disabled(match.weight_format);
    if ((match.weight_format == CommonMulMatWeightFormat::IQ1_S ||
         match.weight_format == CommonMulMatWeightFormat::IQ1_M ||
         match.weight_format == CommonMulMatWeightFormat::IQ3_XXS) &&
        (!use_iq_codebook || common_mul_mat_is_skinny_route(match))) {
        return false;
    }
    const bool use_aligned_dynamic = match.token_count % 32 == 0 &&
                                     match.kernel.id == kMulMatTiledF32F32Kernel.id && !publish_f16_alternate &&
                                     !use_iq_codebook;
    const bool use_q1_token64 = use_aligned_dynamic && match.token_count == 64 &&
                                match.weight_format == CommonMulMatWeightFormat::Q1_0 &&
                                std::getenv("GGML_HRX_DISABLE_Q1_TOKEN64_MATMUL") == nullptr;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(
        use_iq_codebook ? tiled_iq_codebook_kernel(match.weight_format) :
        use_q1_token64 ? kMulMatTiledToken64AlignedF32F32Kernel :
        use_aligned_dynamic ? kMulMatTiledAlignedF32F32Kernel :
        publish_f16_alternate ? kMulMatTiledF32F16AlternateKernel : match.kernel);
    if (use_aligned_dynamic) {
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    }
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    if (!use_aligned_dynamic) {
        dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                                   common_to_config_value(match.token_count));
    }
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.input_size", common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_size", common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_unary_op",
                                               std::to_string(unary_kind_config_value(match.output_unary_op)));
    const bool pack_rows = match.token_count <= 5 && common_mul_mat_has_skinny_reduction_shape(match.input_size) &&
                           match.output_size % 64 == 0 && match.weight->alias_source.value < 0;
    const bool can_pack_q4 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q4K;
    const bool can_pack_q6 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q6K;
    bool       pack_q4     = false;
    bool       pack_q6     = false;
    DispatchBinding activation = { match.input->id, 0, match.input->byte_count };
    const size_t f16_activation_bytes = static_cast<size_t>(match.token_count * match.input_size) * sizeof(ggml_fp16_t);
    const CommandPlanAlternateValue * f16_activation =
        common_mul_mat_uses_factored_tiled_kernel(match.kernel) ?
            find_alternate_value(context.graph, context.plan, match.input->id, GGML_TYPE_F16, f16_activation_bytes) :
            nullptr;
    if (f16_activation != nullptr) {
        activation = { f16_activation->alternate_value, 0, f16_activation_bytes };
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format", std::to_string(GGML_TYPE_F16));
    } else if (can_pack_q4 || can_pack_q6) {
        if (common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count, dispatch_match,
                                         activation, q8_policy)) {
            pack_q4 = can_pack_q4;
            pack_q6 = can_pack_q6;
            dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format",
                                                       std::to_string(GGML_TYPE_Q8_1));
        } else if (q8_policy == CommonQ8ActivationPolicy::AllowStandaloneQuantize) {
            return false;
        }
    }
    const CommonMulMatWeightFormat format = pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                            pack_q6 ? CommonMulMatWeightFormat::Q6KRow64 :
                                                      match.weight_format;
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(format)));
    dispatch.bindings.push_back(activation);
    if (pack_q4 || pack_q6) {
        const char * layout = pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, layout, match.weight->type,
                                      match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    if (use_iq_codebook) {
        const ValueId grid = require_iq_codebook(context, dispatch_match, *codebook);
        if (grid.value < 0) {
            return false;
        }
        dispatch.bindings.push_back({ grid, 0, codebook->data.size() });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    if (publish_f16_alternate) {
        constexpr const char * name = "common.mul_mat.tiled.f16";
        CommonActivationPublication publication;
        if (!common_reserve_activation_publication(context, dispatch_match, *match.output, f16_demand, name,
                                                   publication) ||
            !common_append_activation_publication(dispatch_match, publication, name)) {
            return false;
        }
        dispatch.bindings.push_back(publication.binding());
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.has_fused_unary) {
        dispatch_match.covered_nodes.push_back(match.unary_node_index);
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    if (publish_q8_alternate) {
        constexpr const char * name = "common.mul_mat.tiled.q8_1_x4";
        CommonActivationPublication publication;
        if (!common_reserve_activation_publication(context, dispatch_match, *match.output, q8_demand, name,
                                                   publication) ||
            !common_append_activation_publication(dispatch_match, publication, name)) {
            return false;
        }

        Dispatch quantize;
        quantize.kernel = make_kernel_specialization(kQuantizeQ8_1X4F32Kernel);
        quantize.kernel.integer_parameters.emplace("token_count", match.token_count);
        quantize.kernel.integer_parameters.emplace("input_size", match.output_size);
        quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                   common_to_config_value(match.token_count * match.output_size / 128));
        quantize.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        quantize.bindings.push_back(publication.binding());
        dispatch_match.dispatches.push_back(std::move(quantize));
    }
    return true;
}

static bool prepare_vector_activation_binding(const DispatchMatchContext &     context,
                                              const Value &                    input,
                                              const Value &                    weight,
                                              int64_t                          input_size,
                                              int64_t                          output_size,
                                              int64_t                          token_count,
                                              CommonMulMatWeightFormat         weight_format,
                                              DispatchMatch &                  dispatch_match,
                                              DispatchBinding &                activation,
                                              CommonMulMatWeightFormat &       kernel_weight_format,
                                              const char *&                    weight_layout,
                                              bool &                           uses_q8_activation) {
    activation           = { input.id, 0, input.byte_count };
    kernel_weight_format = weight_format;
    weight_layout        = nullptr;
    uses_q8_activation   = false;

    if (token_count != 1 || input_size % 256 != 0 || weight.alias_source.value >= 0 || output_size % 64 != 0) {
        return true;
    }

    const bool can_pack_q4 = weight_format == CommonMulMatWeightFormat::Q4K;
    const bool can_pack_q6 = weight_format == CommonMulMatWeightFormat::Q6K;
    if (!can_pack_q4 && !can_pack_q6) {
        return true;
    }

    DispatchBinding q8_activation = activation;
    if (!common_prepare_q8_1_x4_input(context, input, input_size, token_count, dispatch_match, q8_activation,
                                      CommonQ8ActivationPolicy::ExistingAlternateOnly)) {
        return true;
    }

    activation           = q8_activation;
    kernel_weight_format = can_pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 : CommonMulMatWeightFormat::Q6KRow64;
    weight_layout        = can_pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
    uses_q8_activation   = true;
    return true;
}

static bool is_k256_packed_row_layout(CommonMulMatWeightFormat format) {
    return format == CommonMulMatWeightFormat::Q4KRow64 || format == CommonMulMatWeightFormat::Q6KRow64;
}

static int64_t vector_max_output_size(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::Q4K:
        case CommonMulMatWeightFormat::Q4KRow64:
        case CommonMulMatWeightFormat::Q6K:
        case CommonMulMatWeightFormat::Q6KRow64:
            return kMulMatVectorPackedMaxOutputSize;
        default:
            return kMulMatVectorDenseMaxOutputSize;
    }
}

static bool vector_mul_mat_route_supports(const CommonMulMatMatch & match) {
    return match.matched() && match.token_count == 1 && match.input_size >= 256 && match.input_size <= 32768 &&
           match.input_size % 32 == 0 && match.output_size <= vector_max_output_size(match.weight_format);
}

static bool vector_postops_route_supports(const MulMatPostOpsMatch & match) {
    return match.matched() && match.token_count == 1 && match.input_size >= 256 && match.input_size <= 32768 &&
           match.input_size % 32 == 0 && match.output_size <= vector_max_output_size(match.weight_format);
}

static bool prepare_vector_publish_plan(const DispatchMatchContext & context,
                                        const Value &                output,
                                        DispatchMatch &              dispatch_match,
                                        VectorPublishPlan &          plan) {
    const CommonActivationPublicationCapabilities capabilities = {
        DispatchActivationInputQ8_1X4 | DispatchActivationInputF16Row,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, output, capabilities);
    plan.demand = publication_plan.matched() ? publication_plan.preferred() : CommonActivationPublicationDemand{};
    if (!plan.demand.matched()) {
        return true;
    }

    const char * name = plan.demand.format == CommonActivationPublicationFormat::Q8_1X4 ?
                            "common.mul_mat.vector.q8_1_x4" :
                            "common.mul_mat.vector.f16";
    return common_reserve_activation_publication(context, dispatch_match, output, plan.demand, name,
                                                 plan.publication) &&
           common_append_activation_publication(dispatch_match, plan.publication, name);
}

static int64_t vector_publish_format_config_value(const VectorPublishPlan & plan) {
    return plan.publication.format == CommonActivationPublicationFormat::F16Row ? GGML_TYPE_F16 : 0;
}

static bool append_vector_q8_publish(const Value &      output,
                                     int64_t            token_count,
                                     int64_t            output_size,
                                     VectorPublishPlan & publish,
                                     DispatchMatch &    dispatch_match) {
    if (publish.publication.format != CommonActivationPublicationFormat::Q8_1X4) {
        return true;
    }

    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kQuantizeQ8_1X4F32Kernel);
    quantize.kernel.integer_parameters.emplace("token_count", token_count);
    quantize.kernel.integer_parameters.emplace("input_size", output_size);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                               common_to_config_value(token_count * output_size / 128));
    quantize.bindings.push_back({ output.id, 0, output.byte_count });
    quantize.bindings.push_back(publish.publication.binding());
    dispatch_match.dispatches.push_back(std::move(quantize));
    return true;
}

static DispatchBinding vector_next_output_binding(const Value & output, const VectorPublishPlan & publish) {
    if (publish.publication.format == CommonActivationPublicationFormat::F16Row) {
        return publish.publication.binding();
    }
    return { output.id, 0, output.byte_count };
}

static bool build_vector_mul_mat_dispatch(const DispatchMatchContext & context,
                                          const CommonMulMatMatch &  match,
                                          DispatchMatch &            dispatch_match) {
    if (match.token_count != 1 || match.input_size % 32 != 0 ||
        match.output_size > vector_max_output_size(match.weight_format)) {
        return false;
    }

    if (match.input_size % 256 != 0 && is_k256_packed_row_layout(match.weight_format)) {
        return false;
    }

    Dispatch dispatch;
    const IQCodebookResourceSpec * codebook = iq_codebook_for_weight_format(match.weight_format);
    if (codebook != nullptr && iq_codebook_matmul_required(match.weight_format) &&
        iq_codebook_matmul_disabled(match.weight_format)) {
        return false;
    }
    VectorPublishPlan publish;
    if (!prepare_vector_publish_plan(context, *match.output, dispatch_match, publish)) {
        return false;
    }

    if (match.weight_format == CommonMulMatWeightFormat::Q1_0 && match.weight->alias_source.value < 0 &&
        match.input_size % 128 == 0 && !match.has_fused_unary && !publish.demand.matched()) {
        DispatchBinding q8_activation;
        if (common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count, dispatch_match,
                                         q8_activation, CommonQ8ActivationPolicy::ExistingAlternateOnly)) {
            dispatch.kernel = make_kernel_specialization(
                match.output_size <= 1024 ? kMulMatVectorQ1_0Q8_1X4F32Kernel :
                                            kMulMatVectorQ1_0Q8_1X4Rows2F32Kernel);
            dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
            dispatch.kernel.compile_parameters.emplace("ggml.matmul.q1_0_q8_1_x4.input_size",
                                                       common_to_config_value(match.input_size));
            dispatch.kernel.compile_parameters.emplace("ggml.matmul.q1_0_q8_1_x4.output_size",
                                                       common_to_config_value(match.output_size));
            dispatch.bindings.push_back(q8_activation);
            dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
            dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
            dispatch_match.covered_nodes.push_back(context.root_index);
            dispatch_match.dispatches.push_back(std::move(dispatch));
            return true;
        }
    }

    DispatchBinding          activation;
    CommonMulMatWeightFormat weight_format = match.weight_format;
    const char *             weight_layout = nullptr;
    bool                     uses_q8_activation = false;
    if (!prepare_vector_activation_binding(context, *match.input, *match.weight, match.input_size, match.output_size,
                                           match.token_count, match.weight_format, dispatch_match, activation,
                                           weight_format, weight_layout, uses_q8_activation)) {
        return false;
    }

    const bool use_iq_codebook =
        codebook != nullptr && !iq_codebook_matmul_disabled(match.weight_format) && !uses_q8_activation &&
        publish.publication.format != CommonActivationPublicationFormat::F16Row;
    if ((match.weight_format == CommonMulMatWeightFormat::IQ1_S ||
         match.weight_format == CommonMulMatWeightFormat::IQ1_M ||
         match.weight_format == CommonMulMatWeightFormat::IQ3_XXS) &&
        !use_iq_codebook) {
        return false;
    }
    dispatch.kernel = make_kernel_specialization(
        use_iq_codebook ? vector_iq_codebook_kernel(match.weight_format) : kMulMatVectorF32F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.output_unary_op",
                                               std::to_string(unary_kind_config_value(match.output_unary_op)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.matmul.vector.publish_format",
        common_to_config_value(vector_publish_format_config_value(publish)));

    dispatch.kernel.compile_parameters.emplace(
        "ggml.matmul.vector.weight_format", common_to_config_value(common_mul_mat_format_config_value(weight_format)));
    if (uses_q8_activation) {
        dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.activation_format",
                                                   std::to_string(GGML_TYPE_Q8_1));
    }

    dispatch.bindings.push_back(activation);
    if (weight_layout != nullptr) {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, weight_layout, match.weight->type,
                                      match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    if (use_iq_codebook) {
        const ValueId grid = require_iq_codebook(context, dispatch_match, *codebook);
        if (grid.value < 0) {
            return false;
        }
        dispatch.bindings.push_back({ grid, 0, codebook->data.size() });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch.bindings.push_back(vector_next_output_binding(*match.output, publish));

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.has_fused_unary) {
        dispatch_match.covered_nodes.push_back(match.unary_node_index);
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return append_vector_q8_publish(*match.output, match.token_count, match.output_size, publish, dispatch_match);
}

static bool build_vector_mul_mat_postops_dispatch(const DispatchMatchContext & context,
                                                  const MulMatPostOpsMatch & match,
                                                  DispatchMatch &           dispatch_match) {
    if (iq_codebook_matmul_required(match.weight_format)) {
        return false;
    }
    if (match.token_count != 1 || match.input_size % 32 != 0 ||
        match.output_size > vector_max_output_size(match.weight_format)) {
        return false;
    }

    const bool has_tail = match.input_size % 256 != 0;
    if (has_tail && is_k256_packed_row_layout(match.weight_format)) {
        return false;
    }

    if (!match.has_bias && !match.has_residual) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatVectorBiasAddF32F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector_postops.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector_postops.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector_postops.apply_bias",
                                               common_to_config_value(match.has_bias ? int64_t{ 1 } : int64_t{ 0 }));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.matmul.vector_postops.apply_residual",
                                               common_to_config_value(match.has_residual ? int64_t{ 1 } : int64_t{ 0 }));
    VectorPublishPlan publish;
    if (!prepare_vector_publish_plan(context, *match.residual_output, dispatch_match, publish)) {
        return false;
    }
    dispatch.kernel.compile_parameters.emplace(
        "ggml.matmul.vector.publish_format",
        common_to_config_value(vector_publish_format_config_value(publish)));

    DispatchBinding          activation;
    CommonMulMatWeightFormat weight_format = match.weight_format;
    const char *             weight_layout = nullptr;
    bool                     uses_q8_activation = false;
    if (!prepare_vector_activation_binding(context, *match.input, *match.weight, match.input_size, match.output_size,
                                           match.token_count, match.weight_format, dispatch_match, activation,
                                           weight_format, weight_layout, uses_q8_activation)) {
        return false;
    }

    dispatch.kernel.compile_parameters.emplace(
        "ggml.matmul.vector_postops.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(weight_format)));
    if (uses_q8_activation) {
        dispatch.kernel.compile_parameters.emplace("ggml.matmul.vector.activation_format",
                                                   std::to_string(GGML_TYPE_Q8_1));
    }

    dispatch.bindings.push_back(activation);
    if (weight_layout != nullptr) {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, weight_layout, match.weight->type,
                                      match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    const Value * placeholder = match.has_bias ? match.bias : match.residual_input;
    dispatch.bindings.push_back(match.has_bias ? DispatchBinding{ match.bias->id, 0, match.bias->byte_count } :
                                                 DispatchBinding{ placeholder->id, 0, placeholder->byte_count });
    dispatch.bindings.push_back(match.has_residual ?
                                    DispatchBinding{ match.residual_input->id, 0, match.residual_input->byte_count } :
                                    DispatchBinding{ placeholder->id, 0, placeholder->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    dispatch.bindings.push_back(vector_next_output_binding(*match.residual_output, publish));

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    if (match.layout_node != nullptr &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.layout_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    for (const GraphNode * add_node : match.add_nodes) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, add_node,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return append_vector_q8_publish(*match.residual_output, match.token_count, match.output_size, publish,
                                    dispatch_match);
}

static bool match_vector_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatVectorF32F32Kernel, true);
    if (!match.matched() || match.token_count != 1) {
        return false;
    }
    return build_vector_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_vector_mul_mat_unary_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatVectorF32F32Kernel, true);
    if (!match.matched() || match.token_count != 1 || !try_match_fused_unary(context, match)) {
        return false;
    }
    return build_vector_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_vector_mul_mat_postops_dispatch(const DispatchMatchContext & context,
                                                  DispatchMatch &              dispatch_match) {
    const MulMatPostOpsMatch match = match_mul_mat_postops(context, true);
    if (!match.matched() || match.token_count != 1) {
        return false;
    }
    return build_vector_mul_mat_postops_dispatch(context, match, dispatch_match);
}

static bool match_q6_vector_final_projection_q8_dispatch(const DispatchMatchContext & context,
                                                         DispatchMatch &              dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, true);
    if (!match.matched()) {
        match = common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32WmmaKernel, false);
    }
    if (!match.matched() || !context.graph.has_index() || match.weight->type != GGML_TYPE_Q6_K ||
        match.token_count != 1 || match.weight->alias_source.value >= 0 || match.input_size % 256 != 0 ||
        match.output_size % 64 != 0 || match.output_size > kMulMatVectorPackedMaxOutputSize ||
        !context.graph.index().consumers(match.output->id).empty()) {
        return false;
    }

    if (is_qwen_endpoint_rmsnorm_projection(context.graph, match)) {
        return false;
    }

    if (match.output_size < 16 * match.input_size) {
        return false;
    }

    return build_q6_vector_dispatch(match, context.root_index, dispatch_match);
}

static bool build_mul_mat_postops_dispatch(const DispatchMatchContext & context,
                                           const MulMatPostOpsMatch &   match,
                                           DispatchMatch &              dispatch_match) {
    if (!match.matched()) {
        return false;
    }
    if (iq_codebook_matmul_required(match.weight_format)) {
        return false;
    }
    if (vector_postops_route_supports(match)) {
        return false;
    }

    const bool use_aligned_dynamic = match.token_count % 32 == 0 &&
                                     match.kernel.id == kMulMatTiledBiasAddF32F32Kernel.id;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(
        use_aligned_dynamic ? kMulMatTiledAlignedBiasAddF32F32Kernel : match.kernel);
    if (use_aligned_dynamic) {
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    }
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    if (!use_aligned_dynamic) {
        dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                                   common_to_config_value(match.token_count));
    }
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_postops.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_postops.output_size",
                                               common_to_config_value(match.output_size));
    const bool pack_rows = match.token_count <= 5 && common_mul_mat_has_skinny_reduction_shape(match.input_size) &&
                           match.output_size % 64 == 0 && match.weight->alias_source.value < 0;
    const bool can_pack_q4 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q4K;
    const bool can_pack_q6 = pack_rows && match.weight_format == CommonMulMatWeightFormat::Q6K;
    bool       pack_q4     = false;
    bool       pack_q6     = false;
    DispatchBinding activation = { match.input->id, 0, match.input->byte_count };
    if (match.requires_q8_activation && !can_pack_q4 && !can_pack_q6) {
        return false;
    }
    if (can_pack_q4 || can_pack_q6) {
        if (common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count, dispatch_match,
                                         activation, CommonQ8ActivationPolicy::ExistingAlternateOnly)) {
            pack_q4 = can_pack_q4;
            pack_q6 = can_pack_q6;
            dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format",
                                                       std::to_string(GGML_TYPE_Q8_1));
        }
    }
    if (match.requires_q8_activation && !pack_q4 && !pack_q6) {
        return false;
    }
    const CommonMulMatWeightFormat format = pack_q4 ? CommonMulMatWeightFormat::Q4KRow64 :
                                            pack_q6 ? CommonMulMatWeightFormat::Q6KRow64 :
                                                      match.weight_format;
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_postops.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(format)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_postops.epilogue", common_to_config_value((match.has_bias ? int64_t{ 1 } : int64_t{ 0 }) |
                                                                (match.has_residual ? int64_t{ 2 } : int64_t{ 0 })));
    dispatch.bindings.push_back(activation);
    if (pack_q4 || pack_q6) {
        const char * layout = pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count, layout, match.weight->type,
                                      match.input_size, match.output_size, match.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    }
    const Value * placeholder = match.has_bias ? match.bias : match.residual_input;
    dispatch.bindings.push_back(match.has_bias ? DispatchBinding{ match.bias->id, 0, match.bias->byte_count } :
                                                 DispatchBinding{ placeholder->id, 0, placeholder->byte_count });
    dispatch.bindings.push_back(match.has_residual ?
                                    DispatchBinding{ match.residual_input->id, 0, match.residual_input->byte_count } :
                                    DispatchBinding{ placeholder->id, 0, placeholder->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    if (match.layout_node != nullptr &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.layout_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    for (const GraphNode * add_node : match.add_nodes) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, add_node,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_skinny_mul_mat_postops_dispatch(const DispatchMatchContext & context,
                                                  DispatchMatch &              dispatch_match) {
    const MulMatPostOpsMatch match =
        match_mul_mat_postops(context, kMulMatSkinnyF32F32Kernel, kMulMatSkinnyBiasAddF32F32Kernel,
                              kMulMatSkinnyBiasAddF32F32Kernel, kMulMatSkinnyBiasAddF32F32Kernel);
    if (!common_mul_mat_postops_is_skinny_route(match)) {
        return false;
    }
    return build_mul_mat_postops_dispatch(context, match, dispatch_match);
}

static bool match_tiled_mul_mat_postops_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatPostOpsMatch match =
        match_mul_mat_postops(context, kMulMatTiledF32F32Kernel, kMulMatTiledBiasAddF32F32Kernel,
                              kMulMatTiledBiasAddF32F32Kernel, kMulMatTiledBiasAddF32F32Kernel);
    if (!common_mul_mat_postops_is_tiled_route(match)) {
        return false;
    }
    return build_mul_mat_postops_dispatch(context, match, dispatch_match);
}

static DecodeMulMatAddMatch match_decode_mul_mat_add(const DispatchMatchContext & context) {
    DecodeMulMatAddMatch match;
    CommonMulMatMatch    root =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatAddF32F32DecodeWave64Kernel, true);
    if (!root.matched() || !context.graph.has_index() || root.token_count != 1) {
        return match;
    }

    const GraphNode * add_node = common_find_only_consumer_with_op(context.graph, root.output->id, GGML_OP_ADD);
    if (add_node == nullptr || !common_binary_node_is_add(*add_node)) {
        return match;
    }

    const bool root_is_lhs = add_node->inputs[0] == root.output->id;
    const bool root_is_rhs = add_node->inputs[1] == root.output->id;
    if (!root_is_lhs && !root_is_rhs) {
        return match;
    }

    const ValueId residual_id = root_is_lhs ? add_node->inputs[1] : add_node->inputs[0];
    const Value * residual    = common_graph_value(context.graph, residual_id);
    const Value * output      = common_graph_value(context.graph, add_node->output);
    if (residual == nullptr || output == nullptr || residual->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !residual->contiguous || !output->contiguous || !common_same_shape(*root.output, *residual) ||
        !common_same_shape(*root.output, *output)) {
        return match;
    }

    size_t add_index = 0;
    if (!context.graph.index().node_index(add_node, add_index) || add_index >= context.covered_nodes.size() ||
        context.covered_nodes[add_index]) {
        return match;
    }

    root.output           = output;
    match.root            = root;
    match.residual_input  = residual;
    match.residual_output = output;
    match.add_node        = add_node;
    return match;
}

static bool build_decode_mul_mat_add_dispatch(const DecodeMulMatAddMatch & match,
                                              const DispatchMatchContext & context,
                                              DispatchMatch &              dispatch_match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatAddF32F32DecodeWave64Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.root.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.root.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.root.output_size);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.token_capacity",
                                               common_to_config_value(match.root.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.output_capacity",
                                               common_to_config_value(match.root.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_f32_f32_decode.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.root.weight_format)));
    dispatch.bindings.push_back({ match.root.input->id, 0, match.root.input->byte_count });
    dispatch.bindings.push_back({ match.root.weight->id, 0, match.root.weight->byte_count });
    dispatch.bindings.push_back({ match.residual_input->id, 0, match.residual_input->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.add_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static void build_decode_mul_mat_dispatch(const CommonMulMatMatch & match,
                                          DispatchMatch &           dispatch_match,
                                          size_t                    root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_f32_f32_decode.output_capacity",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_f32_f32_decode.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_skinny_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSkinnyF32F32Kernel, false);
    if (!match.matched()) {
        return false;
    }
    try_match_fused_unary(context, match);
    if (!common_mul_mat_is_skinny_route(match)) {
        return false;
    }
    return build_mul_mat_dispatch(context, match, dispatch_match, CommonQ8ActivationPolicy::AllowStandaloneQuantize);
}

static bool match_tiled_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatTiledF32F32Kernel, false);
    if (!match.matched()) {
        return false;
    }
    try_match_fused_unary(context, match);
    if (!common_mul_mat_is_tiled_route(match)) {
        return false;
    }
    return build_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_skinny_mul_mat_unary_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatSkinnyF32F32Kernel, false);
    if (!match.matched() || !try_match_fused_unary(context, match) || !common_mul_mat_is_skinny_route(match)) {
        return false;
    }
    return build_mul_mat_dispatch(context, match, dispatch_match, CommonQ8ActivationPolicy::AllowStandaloneQuantize);
}

static bool match_tiled_mul_mat_unary_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatTiledF32F32Kernel, false);
    if (!match.matched() || !try_match_fused_unary(context, match) || !common_mul_mat_is_tiled_route(match)) {
        return false;
    }
    return build_mul_mat_dispatch(context, match, dispatch_match);
}

static bool match_decode_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    if (deprecated_mul_mat_dispatch_disabled()) {
        return false;
    }
    CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kMulMatF32F32DecodeWave64Kernel, true);
    if (!match.matched()) {
        return false;
    }
    if (iq_codebook_matmul_required(match.weight_format)) {
        return false;
    }
    if (vector_mul_mat_route_supports(match)) {
        return false;
    }
    if ((match.weight_format == CommonMulMatWeightFormat::Q4K ||
         match.weight_format == CommonMulMatWeightFormat::Q6K) &&
        match.output_size % 64 == 0 && common_is_supported_dense_output_size(match.output_size) &&
        match.weight->alias_source.value < 0) {
        match.kernel = kMulMatF32F32WmmaKernel;
        return build_mul_mat_dispatch(context, match, dispatch_match,
                                      CommonQ8ActivationPolicy::AllowStandaloneQuantize);
    } else {
        build_decode_mul_mat_dispatch(match, dispatch_match, context.root_index);
    }
    return true;
}

static bool match_decode_mul_mat_add_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    if (deprecated_mul_mat_dispatch_disabled()) {
        return false;
    }
    const DecodeMulMatAddMatch match = match_decode_mul_mat_add(context);
    if (!match.matched()) {
        return false;
    }
    if (iq_codebook_matmul_required(match.root.weight_format)) {
        return false;
    }
    MulMatPostOpsMatch postops;
    postops.input             = match.root.input;
    postops.weight            = match.root.weight;
    postops.projection_output = match.root.output;
    postops.residual_input    = match.residual_input;
    postops.residual_output   = match.residual_output;
    postops.kernel            = kMulMatVectorBiasAddF32F32Kernel;
    postops.weight_format     = match.root.weight_format;
    postops.input_size        = match.root.input_size;
    postops.output_size       = match.root.output_size;
    postops.token_count       = match.root.token_count;
    postops.has_residual      = true;
    if (vector_postops_route_supports(postops)) {
        return false;
    }
    return build_decode_mul_mat_add_dispatch(match, context, dispatch_match);
}

void register_mul_mat_dispatches(DispatchRegistryBuilder & registry) {
    registry.add_activation_consumer({
        "common.mul_mat.input.f16_row.factored_tiled",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16Row,
        DispatchActivationConsumerUse::Tiled,
        DispatchActivationProducerRequirementFactoredTiledF16,
        DispatchSource::Common,
        accepts_factored_tiled_f16_input,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.q8_1_x4.q1_0_preferred_publication",
        GGML_OP_MUL_MAT,
        DispatchActivationInputQ8_1X4,
        DispatchActivationConsumerUse::BandwidthLimited,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        accepts_preferred_q1_0_q8_publication,
        DispatchActivationConsumerTraitQ8PublicationPreferred,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.q8_1_x4.preferred_publication",
        GGML_OP_MUL_MAT,
        DispatchActivationInputQ8_1X4,
        DispatchActivationConsumerUse::BandwidthLimited,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        accepts_preferred_q8_publication,
        DispatchActivationConsumerTraitQ8PublicationPreferred,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.q8_1_x4.decode",
        GGML_OP_MUL_MAT,
        DispatchActivationInputQ8_1X4,
        DispatchActivationConsumerUse::BandwidthLimited,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        common_accepts_q8_1_x4_decode_mul_mat,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.q8_1_x4.prefill",
        GGML_OP_MUL_MAT,
        DispatchActivationInputQ8_1X4,
        DispatchActivationConsumerUse::BandwidthLimited,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        accepts_tiled_q8_publication,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.f16_k16_major",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16K16Major,
        DispatchActivationConsumerUse::Tiled,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        common_accepts_f16_k16_major_mul_mat,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.f16_k16_major.packed_producer",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16K16Major,
        DispatchActivationConsumerUse::Tiled,
        DispatchActivationProducerRequirementPackedK16Economy,
        DispatchSource::Common,
        common_accepts_f16_k16_major_packed_producer_mul_mat,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.f16_row.k16_preparation",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16Row,
        DispatchActivationConsumerUse::Tiled,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        common_accepts_f16_k16_major_mul_mat,
    });
    registry.add_activation_consumer({
        "common.mul_mat.input.f16_row.prefill",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16Row,
        DispatchActivationConsumerUse::Row,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Common,
        common_accepts_f16_row_prefill_mul_mat,
    });
    registry.add({
        "common.mul_mat.vector_q6_selected_outputs",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        310,
        DispatchSource::Common,
        match_q6_vector_selected_outputs_dispatch,
    });
    registry.add({
        "common.mul_mat.q6_k_prefill_wave32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        295,
        DispatchSource::Common,
        match_q6_k_prefill_wave32_dispatch,
    });
    registry.add({
        "common.mul_mat.vector_q6_final_projection_q8",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        305,
        DispatchSource::Common,
        match_q6_vector_final_projection_q8_dispatch,
    });
    registry.add({
        "common.mul_mat.vector_q6",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_q6_vector_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_iq4_xs_symmetric_i4_lowrow",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        285,
        DispatchSource::Common,
        match_symmetric_i4_low_row_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_iq4_xs_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        260,
        DispatchSource::Common,
        match_packed_q8_1_x4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q4_k_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        305,
        DispatchSource::Common,
        match_q4_k_q8_1_x4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_symmetric_i4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        270,
        DispatchSource::Common,
        match_symmetric_i4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q5_k_symmetric_i8_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        280,
        DispatchSource::Common,
        match_q5_k_symmetric_i8_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat_postops.vector",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        292,
        DispatchSource::Common,
        match_vector_mul_mat_postops_dispatch,
    });
    registry.add({
        "common.mul_mat_unary.vector",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        291,
        DispatchSource::Common,
        match_vector_mul_mat_unary_dispatch,
    });
    registry.add({
        "common.mul_mat_unary.tiled_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        291,
        DispatchSource::Common,
        match_tiled_mul_mat_unary_dispatch,
    });
    registry.add({
        "common.mul_mat_unary.skinny_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        292,
        DispatchSource::Common,
        match_skinny_mul_mat_unary_dispatch,
    });
    registry.add({
        "common.mul_mat.vector",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        190,
        DispatchSource::Common,
        match_vector_mul_mat_dispatch,
    });
    registry.add({
        "common.mul_mat_postops.tiled_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        181,
        DispatchSource::Common,
        match_tiled_mul_mat_postops_dispatch,
    });
    registry.add({
        "common.mul_mat_postops.skinny_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        182,
        DispatchSource::Common,
        match_skinny_mul_mat_postops_dispatch,
    });
    registry.add({
        "common.mul_mat.tiled_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        81,
        DispatchSource::Common,
        match_tiled_mul_mat_dispatch,
    });
    registry.add({
        "common.mul_mat.skinny_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        82,
        DispatchSource::Common,
        match_skinny_mul_mat_dispatch,
    });
    registry.add({
        "common.mul_mat_add.skinny_f32_f32_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        70,
        DispatchSource::Common,
        match_decode_mul_mat_add_dispatch,
    });
    registry.add({
        "common.mul_mat.skinny_f32_f32_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        60,
        DispatchSource::Common,
        match_decode_mul_mat_dispatch,
    });
}

}  // namespace ggml::hrx
