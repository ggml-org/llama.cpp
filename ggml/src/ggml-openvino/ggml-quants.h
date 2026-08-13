#pragma once
#include "ggml-openvino-extra.h"  // For ExtraQuantType
#include "ggml.h"

#include <cstdint>
#include <openvino/op/constant.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/runtime/tensor.hpp>

void unpack_32_4(const uint8_t * data, uint8_t * dst);

void extract_q4_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

void extract_q4_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q5_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q8_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

void unpack_256_4(const uint8_t * data, uint8_t * dst);

void extract_q4_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q5_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q6_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

void extract_mxfp4_data(const ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr);

static constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

// If for_gather_matmul is true, the weight tensor may be N-D (e.g. 3D MoE expert weights
// [n_expert, rows, cols]). The dequantization chain (Convert->[Subtract]->Multiply) is built as
// usual but left in f16 (no final Convert to f32) -- ov::pass::MarkDequantization (registered in
// translate_session.cpp) marks the chain so it survives model-build-time ConstantFolding -- see
// make_int8_weights.cpp/make_int4_weights.cpp. mul_mat_id.cpp constructs ov::op::internal::GatherMatmul
// directly from the resulting f16 dequant chain.
//
// When use_bias is true (explicitly, or implicitly because for_gather_matmul is true), the zp
// tensor is expected to hold an exact f16 bias value (rather than a rounded integer zero point);
// it is converted in place into an exact zero_point = -bias/scale and consumed via Subtract, not
// Add, so the chain still matches OpenVINO's Convert->Subtract->Multiply decompression pattern.
ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       bool use_bias = false,
                                       bool for_gather_matmul = false);

ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       bool use_bias = false,
                                       bool for_gather_matmul = false);

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales);

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight);

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(
    const ggml_tensor * tensor,
    const void * data,  // Source data pointer (may differ from tensor->data)
    ov::Tensor & weights,
    ov::Tensor & scales,
    ov::Tensor & zp,
    bool use_bias = false);  // Use an exact f16 zero point (vs. a rounded integer one); always
                             // used for for_gather_matmul (3D MoE expert) weights regardless of
                             // this flag, and also settable explicitly for test-backend-ops.

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_to_buffers(const ggml_tensor * tensor,
                                                const void * data,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp);

inline const char * extra_quant_type_name(ExtraQuantType t) {
    switch (t) {
    case ExtraQuantType::F16:
        return "F16";
    case ExtraQuantType::Q4_0_C:
        return "Q4_0_C";
    case ExtraQuantType::Q4_0_128:
        return "Q4_0_128";
    case ExtraQuantType::Q8_0_C:
        return "Q8_0_C";
    case ExtraQuantType::Q8_0_32:
        return "Q8_0_32";
    case ExtraQuantType::Q8_1_C:
        return "Q8_1_C";
    default:
        return "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
struct OvWeight {
    std::shared_ptr<ov::Node> weight_node;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() const { return layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    const ggml_tensor * tensor,
    const void * data,                 // Source data pointer (may differ from tensor->data)
    void * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.

void quantize_q4_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk);
void quantize_q8_1(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0);
void quantize_q8_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset = 0);

namespace ov {
namespace op {
namespace util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
bool get_single_value(const std::shared_ptr<ov::op::v0::Constant> & const_node,
                      float & value,
                      bool check_value_range = true);
}  // namespace util
}  // namespace op
}  // namespace ov
