
#include "op-config-impl.hpp"

namespace {

using op_constructor_t    = std::shared_ptr<qnn::ggml_qnn_op_config> (*)(const ggml_tensor *, const std::string &,
                                                                      std::shared_ptr<qnn::qnn_instance>);
using op_dims_calc_func_t = void (*)(const std::vector<qnn::ggml_dimension_array_t> & input_dims,
                                     qnn::ggml_dimension_array_t &                    output_dims);

void element_wise_op_dims(const std::vector<qnn::ggml_dimension_array_t> & input_dims,
                          qnn::ggml_dimension_array_t &                    output_dims) {
    for (size_t i = 1; i < std::size(output_dims); i++) {
        output_dims[i] = input_dims.front()[i];
    }
}

void mat_mul_op_dims(const std::vector<qnn::ggml_dimension_array_t> & input_dims,
                     qnn::ggml_dimension_array_t &                    output_dims) {
    GGML_ASSERT(input_dims.size() == 2);
    output_dims[0] = input_dims.front()[1];
    output_dims[1] = input_dims.back()[1];
}

struct qnn_op_caps_t {
    const char *        qnn_op_name       = nullptr;
    const size_t        input_param_count = 0;
    op_dims_calc_func_t calc_dims_func    = nullptr;
    const char *        qnn_param_name    = nullptr;
};

constexpr const qnn_op_caps_t kOpCaps[] = {
    {}, // GGML_OP_NONE
    {}, // GGML_OP_DUP
    {
     // GGML_OP_ADD
        QNN_OP_ELEMENT_WISE_ADD,  // qnn_op_name
        2,                        // input_param_count
        element_wise_op_dims,     // calc_dims_func
    },
    {}, // GGML_OP_ADD1
    {}, // GGML_OP_ACC
    {
     // GGML_OP_SUB
        QNN_OP_ELEMENT_WISE_SUBTRACT,  // qnn_op_name
        2,                             // input_param_count
        element_wise_op_dims,          // calc_dims_func
    },
    {
     // GGML_OP_MUL
        QNN_OP_ELEMENT_WISE_MULTIPLY,  // qnn_op_name
        2,                             // input_param_count
        element_wise_op_dims,          // calc_dims_func
    },
    {
     // GGML_OP_DIV
        QNN_OP_ELEMENT_WISE_DIVIDE,  // qnn_op_name
        2,                           // input_param_count
        element_wise_op_dims,        // calc_dims_func
    },
    {}, // GGML_OP_SQR
    {
     // GGML_OP_SQRT
        QNN_OP_ELEMENT_WISE_SQUARE_ROOT,  // qnn_op_name
        1,                                // input_param_count
        element_wise_op_dims,             // calc_dims_func
    },
    {
     // GGML_OP_LOG
        QNN_OP_ELEMENT_WISE_LOG,  // qnn_op_name
        1,                        // input_param_count
        element_wise_op_dims,     // calc_dims_func
    },
    {}, // GGML_OP_SIN
    {}, // GGML_OP_COS
    {}, // GGML_OP_SUM
    {}, // GGML_OP_SUM_ROWS
    {}, // GGML_OP_MEAN
    {}, // GGML_OP_ARGMAX
    {}, // GGML_OP_COUNT_EQUAL
    {}, // GGML_OP_REPEAT
    {}, // GGML_OP_REPEAT_BACK
    {}, // GGML_OP_CONCAT
    {}, // GGML_OP_SILU_BACK
    {}, // GGML_OP_NORM
    {
     // GGML_OP_RMS_NORM
        QNN_OP_RMS_NORM,                // qnn_op_name
        1,                              // input_param_count
        nullptr,                        // TODO: calc_dims_func
        QNN_OP_RMS_NORM_PARAM_EPSILON,  // qnn_param_name
    },
    {}, // GGML_OP_RMS_NORM_BACK
    {}, // GGML_OP_GROUP_NORM
    {
     // GGML_OP_MUL_MAT
        QNN_OP_MAT_MUL,   // qnn_op_name
        2,                // input_param_count
        mat_mul_op_dims,  // calc_dims_func
    },
    {}, // GGML_OP_MUL_MAT_ID
    {}, // GGML_OP_OUT_PROD
    {}, // GGML_OP_SCALE
    {}, // GGML_OP_SET
    {}, // GGML_OP_CPY
    {}, // GGML_OP_CONT
    {
     // GGML_OP_RESHAPE
        QNN_OP_RESHAPE,  // qnn_op_name
        1,               // input_param_count
        nullptr,         // TODO: calc_dims_func
    },
    {}, // GGML_OP_VIEW
    {}, // GGML_OP_PERMUTE
    {}, // GGML_OP_TRANSPOSE
    {}, // GGML_OP_GET_ROWS
    {}, // GGML_OP_GET_ROWS_BACK
    {}, // GGML_OP_DIAG
    {}, // GGML_OP_DIAG_MASK_INF
    {}, // GGML_OP_DIAG_MASK_ZERO
    {}, // GGML_OP_SOFT_MAX
    {}, // GGML_OP_SOFT_MAX_BACK
    {}, // GGML_OP_ROPE
    {}, // GGML_OP_ROPE_BACK
    {}, // GGML_OP_CLAMP
    {}, // GGML_OP_CONV_TRANSPOSE_1D
    {}, // GGML_OP_IM2COL
    {}, // GGML_OP_IM2COL_BACK
    {}, // GGML_OP_CONV_TRANSPOSE_2D
    {}, // GGML_OP_POOL_1D
    {}, // GGML_OP_POOL_2D
    {}, // GGML_OP_POOL_2D_BACK
    {}, // GGML_OP_UPSCALE
    {}, // GGML_OP_PAD
    {}, // GGML_OP_PAD_REFLECT_1D
    {}, // GGML_OP_ARANGE

    {}, // GGML_OP_TIMESTEP_EMBEDDING
    {}, // GGML_OP_ARGSORT
    {}, // GGML_OP_LEAKY_RELU

    {}, // GGML_OP_FLASH_ATTN_EXT
    {}, // GGML_OP_FLASH_ATTN_BACK
    {}, // GGML_OP_SSM_CONV
    {}, // GGML_OP_SSM_SCAN
    {}, // GGML_OP_WIN_PART
    {}, // GGML_OP_WIN_UNPART
    {}, // GGML_OP_GET_REL_POS
    {}, // GGML_OP_ADD_REL_POS
    {}, // GGML_OP_RWKV_WKV6
    {}, // GGML_OP_GATED_LINEAR_ATTN

    {}, // GGML_OP_UNARY

    {}, // GGML_OP_MAP_UNARY
    {}, // GGML_OP_MAP_BINARY

    {}, // GGML_OP_MAP_CUSTOM1_F32
    {}, // GGML_OP_MAP_CUSTOM2_F32
    {}, // GGML_OP_MAP_CUSTOM3_F32

    {}, // GGML_OP_MAP_CUSTOM1
    {}, // GGML_OP_MAP_CUSTOM2
    {}, // GGML_OP_MAP_CUSTOM3

    {}, // GGML_OP_CROSS_ENTROPY_LOSS
    {}, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    {}, // GGML_OP_OPT_STEP_ADAMW

    // ggml_unary_op
    {}, // GGML_UNARY_OP_ABS
    {}, // GGML_UNARY_OP_SGN
    {}, // GGML_UNARY_OP_NEG
    {}, // GGML_UNARY_OP_STEP
    {}, // GGML_UNARY_OP_TANH
    {}, // GGML_UNARY_OP_ELU
    {}, // GGML_UNARY_OP_RELU
    {}, // GGML_UNARY_OP_SIGMOID
    {
     // GGML_UNARY_OP_GELU
        QNN_OP_GELU,  // qnn_op_name
        1,            // input_param_count
        nullptr,      // TODO: calc_dims_func
    },
    {}, // GGML_UNARY_OP_GELU_QUICK
    {}, // GGML_UNARY_OP_SILU
    {}, // GGML_UNARY_OP_HARDSWISH
    {}, // GGML_UNARY_OP_HARDSIGMOID
    {}, // GGML_UNARY_OP_EXP
};

static_assert(kOpCaps[GGML_OP_NONE].calc_dims_func == nullptr, "GGML_OP_NONE should not have calc_dims_func function");
static_assert(kOpCaps[GGML_OP_ADD].calc_dims_func == element_wise_op_dims,
              "GGML_OP_ADD does not have element_wise_op_dims function");
static_assert(kOpCaps[GGML_OP_MUL_MAT].calc_dims_func == mat_mul_op_dims,
              "GGML_OP_ADD does not have element_wise_op_dims function");
static_assert(kOpCaps[GGML_OP_LOG].calc_dims_func == element_wise_op_dims,
              "GGML_OP_LOG does not have element_wise_op_dims function");
static_assert(kOpCaps[GGML_OP_COUNT + GGML_UNARY_OP_GELU].input_param_count == 1,
              "GGML_UNARY_OP_GELU does not have 1 input parameter");
static_assert(std::size(kOpCaps) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kOpCaps table");

std::shared_ptr<qnn::ggml_qnn_op_config> mat_mul_op_constructor(const ggml_tensor *                op,
                                                                const std::string &                instance_name,
                                                                std::shared_ptr<qnn::qnn_instance> qnn_instance) {
    GGML_UNUSED(op);
    QNN_LOG_DEBUG("create QNN_OP_MAT_MUL, name %s\n", instance_name.c_str());
    return std::make_shared<qnn::ggml_qnn_matmul_op_config>(instance_name, qnn_instance);
}

template <size_t _op>
std::shared_ptr<qnn::ggml_qnn_op_config> generic_op_constructor(const ggml_tensor *                op,
                                                                const std::string &                instance_name,
                                                                std::shared_ptr<qnn::qnn_instance> qnn_instance) {
    GGML_UNUSED(op);
    static_assert(_op < std::size(kOpCaps));
    static_assert(kOpCaps[_op].qnn_op_name != nullptr);
    return std::make_shared<qnn::ggml_qnn_single_op_config>(instance_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            kOpCaps[_op].qnn_op_name, qnn_instance);
}

void add_type_parameters(std::shared_ptr<qnn::ggml_qnn_op_config_base> op, const char * name, float value) {
    Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
    scalar.dataType     = QNN_DATATYPE_FLOAT_32;
    scalar.floatValue   = value;
    op->add_scalar_param(name, scalar);
}

template <size_t _op, typename _ggml_op_param_type, typename _qnn_op_type_name>
std::shared_ptr<qnn::ggml_qnn_op_config> op_constructor_with_type_param(
    const ggml_tensor * op, const std::string & instance_name, std::shared_ptr<qnn::qnn_instance> qnn_instance) {
    static_assert(std::is_base_of<qnn::ggml_qnn_op_config_base, _qnn_op_type_name>::value);
    static_assert(_op < std::size(kOpCaps));

    constexpr auto & op_caps = kOpCaps[_op];
    static_assert(op_caps.qnn_op_name != nullptr);

    _ggml_op_param_type op_param;
    memcpy(&op_param, op->op_params, sizeof(op_param));
    auto qnn_op = std::make_shared<_qnn_op_type_name>(instance_name, QNN_OP_PACKAGE_NAME_QTI_AISW, op_caps.qnn_op_name,
                                                      qnn_instance);
    if (op_caps.qnn_param_name) {
        add_type_parameters(qnn_op, op_caps.qnn_param_name, op_param);
    }
    return qnn_op;
}

constexpr const op_constructor_t kOpConstructors[] = {
    nullptr,                                                                                   // GGML_OP_NONE
    nullptr,                                                                                   // GGML_OP_DUP
    generic_op_constructor<GGML_OP_ADD>,                                                       // GGML_OP_ADD
    nullptr,                                                                                   // GGML_OP_ADD1
    nullptr,                                                                                   // GGML_OP_ACC
    generic_op_constructor<GGML_OP_SUB>,                                                       // GGML_OP_SUB
    generic_op_constructor<GGML_OP_MUL>,                                                       // GGML_OP_MUL
    generic_op_constructor<GGML_OP_DIV>,                                                       // GGML_OP_DIV
    nullptr,                                                                                   // GGML_OP_SQR
    generic_op_constructor<GGML_OP_SQRT>,                                                      // GGML_OP_SQRT
    generic_op_constructor<GGML_OP_LOG>,                                                       // GGML_OP_LOG
    nullptr,                                                                                   // GGML_OP_SIN
    nullptr,                                                                                   // GGML_OP_COS
    nullptr,                                                                                   // GGML_OP_SUM
    nullptr,                                                                                   // GGML_OP_SUM_ROWS
    nullptr,                                                                                   // GGML_OP_MEAN
    nullptr,                                                                                   // GGML_OP_ARGMAX
    nullptr,                                                                                   // GGML_OP_COUNT_EQUAL
    nullptr,                                                                                   // GGML_OP_REPEAT
    nullptr,                                                                                   // GGML_OP_REPEAT_BACK
    nullptr,                                                                                   // GGML_OP_CONCAT
    nullptr,                                                                                   // GGML_OP_SILU_BACK
    nullptr,                                                                                   // GGML_OP_NORM
    op_constructor_with_type_param<GGML_OP_RMS_NORM, float, qnn::ggml_qnn_rmsnorm_op_config>,  // GGML_OP_RMS_NORM
    nullptr,                                                                                   // GGML_OP_RMS_NORM_BACK
    nullptr,                                                                                   // GGML_OP_GROUP_NORM

    mat_mul_op_constructor,                                                                    // GGML_OP_MUL_MAT
    nullptr,                                                                                   // GGML_OP_MUL_MAT_ID
    nullptr,                                                                                   // GGML_OP_OUT_PROD

    nullptr,                                                                                   // GGML_OP_SCALE
    nullptr,                                                                                   // GGML_OP_SET
    nullptr,                                                                                   // GGML_OP_CPY
    nullptr,                                                                                   // GGML_OP_CONT
    generic_op_constructor<GGML_OP_RESHAPE>,                                                   // GGML_OP_RESHAPE
    nullptr,                                                                                   // GGML_OP_VIEW
    nullptr,                                                                                   // GGML_OP_PERMUTE
    nullptr,                                                                                   // GGML_OP_TRANSPOSE
    nullptr,                                                                                   // GGML_OP_GET_ROWS
    nullptr,                                                                                   // GGML_OP_GET_ROWS_BACK
    nullptr,                                                                                   // GGML_OP_DIAG
    nullptr,                                                                                   // GGML_OP_DIAG_MASK_INF
    nullptr,                                                                                   // GGML_OP_DIAG_MASK_ZERO
    nullptr,                                                                                   // GGML_OP_SOFT_MAX
    nullptr,                                                                                   // GGML_OP_SOFT_MAX_BACK
    nullptr,                                                                                   // GGML_OP_ROPE
    nullptr,                                                                                   // GGML_OP_ROPE_BACK
    nullptr,                                                                                   // GGML_OP_CLAMP
    nullptr,  // GGML_OP_CONV_TRANSPOSE_1D
    nullptr,  // GGML_OP_IM2COL
    nullptr,  // GGML_OP_IM2COL_BACK
    nullptr,  // GGML_OP_CONV_TRANSPOSE_2D
    nullptr,  // GGML_OP_POOL_1D
    nullptr,  // GGML_OP_POOL_2D
    nullptr,  // GGML_OP_POOL_2D_BACK
    nullptr,  // GGML_OP_UPSCALE
    nullptr,  // GGML_OP_PAD
    nullptr,  // GGML_OP_PAD_REFLECT_1D
    nullptr,  // GGML_OP_ARANGE
    nullptr,  // GGML_OP_TIMESTEP_EMBEDDING
    nullptr,  // GGML_OP_ARGSORT
    nullptr,  // GGML_OP_LEAKY_RELU

    nullptr,  // GGML_OP_FLASH_ATTN_EXT
    nullptr,  // GGML_OP_FLASH_ATTN_BACK
    nullptr,  // GGML_OP_SSM_CONV
    nullptr,  // GGML_OP_SSM_SCAN
    nullptr,  // GGML_OP_WIN_PART
    nullptr,  // GGML_OP_WIN_UNPART
    nullptr,  // GGML_OP_GET_REL_POS
    nullptr,  // GGML_OP_ADD_REL_POS
    nullptr,  // GGML_OP_RWKV_WKV6
    nullptr,  // GGML_OP_GATED_LINEAR_ATTN

    nullptr,  // GGML_OP_UNARY

    nullptr,  // GGML_OP_MAP_UNARY
    nullptr,  // GGML_OP_MAP_BINARY

    nullptr,  // GGML_OP_MAP_CUSTOM1_F32
    nullptr,  // GGML_OP_MAP_CUSTOM2_F32
    nullptr,  // GGML_OP_MAP_CUSTOM3_F32

    nullptr,  // GGML_OP_MAP_CUSTOM1
    nullptr,  // GGML_OP_MAP_CUSTOM2
    nullptr,  // GGML_OP_MAP_CUSTOM3

    nullptr,  // GGML_OP_CROSS_ENTROPY_LOSS
    nullptr,  // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    nullptr,  // GGML_OP_OPT_STEP_ADAMW

    // ggml_unary_op
    nullptr,  // GGML_UNARY_OP_ABS
    nullptr,  // GGML_UNARY_OP_SGN
    nullptr,  // GGML_UNARY_OP_NEG
    nullptr,  // GGML_UNARY_OP_STEP
    nullptr,  // GGML_UNARY_OP_TANH
    nullptr,  // GGML_UNARY_OP_ELU
    nullptr,  // GGML_UNARY_OP_RELU
    nullptr,  // GGML_UNARY_OP_SIGMOID
    nullptr,  // GGML_UNARY_OP_GELU
    nullptr,  // GGML_UNARY_OP_GELU_QUICK
    nullptr,  // GGML_UNARY_OP_SILU
    nullptr,  // GGML_UNARY_OP_HARDSWISH
    nullptr,  // GGML_UNARY_OP_HARDSIGMOID
    nullptr,  // GGML_UNARY_OP_EXP
};

static_assert(kOpConstructors[GGML_OP_NONE] == nullptr, "GGML_OP_NONE does not match the nullptr function");
static_assert(kOpConstructors[GGML_OP_ADD] == generic_op_constructor<GGML_OP_ADD>,
              "GGML_OP_ADD does not match the generic_op_constructor<GGML_OP_ADD> function");
static_assert(kOpConstructors[GGML_OP_MUL_MAT] == mat_mul_op_constructor,
              "GGML_OP_MUL_MAT does not match the mat_mul_op_constructor function");
static_assert(std::size(kOpConstructors) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kOpConstructors table");

}  // namespace

namespace qnn {

size_t get_qnn_op_index(const ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_UNARY) {
        return kGgmlUnaryOpStart + ggml_get_unary_op(tensor);
    }

    return tensor->op;
}

const char * get_qnn_op_name(const ggml_tensor * op) {
    auto op_index = get_qnn_op_index(op);
    GGML_ASSERT(op_index < std::size(kOpCaps));
    GGML_ASSERT(kOpCaps[op_index].qnn_op_name);
    return kOpCaps[op_index].qnn_op_name;
}

size_t get_qnn_op_input_param_count(const ggml_tensor * op) {
    auto op_index = get_qnn_op_index(op);
    GGML_ASSERT(op_index < std::size(kOpCaps));
    return kOpCaps[op_index].input_param_count;
}

std::shared_ptr<ggml_qnn_op_config> create_op(const ggml_tensor * op, const std::string & name,
                                              std::shared_ptr<qnn_instance> qnn_instance) {
    auto op_index = get_qnn_op_index(op);
    GGML_ASSERT(op_index < std::size(kOpCaps));
    auto op_constructor = kOpConstructors[op_index];
    GGML_ASSERT(op_constructor);
    return op_constructor(op, name, qnn_instance);
}

}  // namespace qnn
