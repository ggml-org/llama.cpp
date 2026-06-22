#include "op_table.h"

#include "utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tanh.hpp>

namespace ov {
namespace frontend {
namespace ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using namespace ov::op;
    return {
        {"GGML_OP_ADD",             op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD1",            op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD_ID",          op::translate_add_id                           },
        {"GGML_OP_CONCAT",          op::translate_concat                           },
        {"GGML_OP_CONT",            op::translate_cont                             },
        {"GGML_OP_DIV",             op::translate_div                              },
        {"GGML_OP_GET_ROWS",        op::translate_get_rows                         },
        {"GGML_OP_IM2COL",          op::translate_im2col                           },
        {"GGML_OP_MUL",             op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",         op::translate_mulmat                           },
        {"GGML_OP_MUL_MAT_ID",      op::translate_mul_mat_id                       },
        {"GGML_OP_PERMUTE",         op::translate_permute                          },
        {"GGML_OP_RESHAPE",         op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",        op::translate_rms_norm                         },
        {"GGML_OP_NORM",            op::translate_norm                             },
        {"GGML_OP_L2_NORM",         op::translate_l2_norm                          },
        {"GGML_OP_SUM_ROWS",        op::translate_sum_rows                         },
        {"GGML_OP_ROPE",            op::translate_rope                             },
        {"GGML_OP_SCALE",           op::translate_scale                            },
        {"GGML_OP_SOFT_MAX",        op::translate_soft_max                         },
        {"GGML_OP_ARGSORT",         op::translate_argsort                          },
        {"GGML_OP_SUB",             op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",       op::translate_transpose                        },
        {"GGML_UNARY_OP_GELU",      op::translate_1to1_match_1_input<v7::Gelu>     },
        {"GGML_UNARY_OP_SILU",      op::translate_unary_silu                       },
        {"GGML_UNARY_OP_SOFTPLUS",  op::translate_unary_softplus                   },
        {"GGML_UNARY_OP_TANH",      op::translate_1to1_match_1_input<v0::Tanh>     },
        {"GGML_OP_VIEW",            op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU",      op::translate_glu_swiglu                       },
        {"GGML_GLU_OP_GEGLU",       op::translate_glu_geglu                        },
        {"GGML_OP_SET_ROWS",        op::translate_set_rows                         },
        {"GGML_OP_CPY",             op::translate_cpy                              },
        {"GGML_OP_FLASH_ATTN_EXT",  op::translate_flash_attn_ext                   },
        {"GGML_OP_CLAMP",           op::translate_clamp                            },
        {"GGML_OP_PAD",             op::translate_pad                              },
        {"GGML_OP_SSM_CONV",        op::translate_ssm_conv                         },
        {"GGML_OP_GATED_DELTA_NET", op::translate_gated_delta_net                  },
        {"GGML_OP_REPEAT",          op::translate_repeat                           },
    };
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
