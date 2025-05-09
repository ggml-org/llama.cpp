#include "op_table.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>

#include "utils.hpp"

using namespace ov::op;
namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext& node)

GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_mul);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_unary_silu);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_unary);
GGML_OP_CONVERTER(translate_view);

}  // namespace op

const std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    return {{"GGML_OP_ADD", op::translate_1to1_match_2_inputs<v1::Add>},
            {"GGML_OP_ADD1", op::translate_1to1_match_2_inputs<v1::Add>},
            {"GGML_OP_CONT", op::translate_cont},
            {"GGML_OP_CPY", op::translate_cpy},
            {"GGML_OP_DIV", op::translate_1to1_match_2_inputs<v1::Divide>},
            {"GGML_OP_GET_ROWS", op::translate_get_rows},
            // {"GGML_OP_MUL", op::translate_1to1_match_2_inputs<v1::Multiply>},
            {"GGML_OP_MUL", op::translate_mul},
            {"GGML_OP_MUL_MAT", op::translate_mulmat},
            {"GGML_OP_PERMUTE", op::translate_permute},
            {"GGML_OP_RESHAPE", op::translate_reshape},
            {"GGML_OP_RMS_NORM", op::translate_rms_norm},
            {"GGML_OP_ROPE", op::translate_rope},
            {"GGML_OP_SCALE", op::translate_scale},
            {"GGML_OP_SOFT_MAX", op::translate_soft_max},
            {"GGML_OP_SUB", op::translate_1to1_match_2_inputs<v1::Subtract>},
            {"GGML_OP_TRANSPOSE", op::translate_transpose},
            {"GGML_UNARY_OP_SILU", op::translate_unary_silu},
            {"GGML_OP_VIEW", op::translate_view}};
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
