
#include "backend-ops.hpp"

#include "ggml-impl.h"
#include "graph.hpp"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <memory>

namespace {

qnn::qnn_graph * get_qnn_graph_from_cache(qnn::ggml_backend_qnn_device_context * ctx, const ggml_cgraph * cgraph) {
    auto &      graph_cache = ctx->qnn_graph_cache;
    std::string graph_key;
    auto        op_data_type = qnn::qnn_graph::get_graph_key_from_cgraph(cgraph, graph_key);
    if (graph_key.empty()) {
        QNN_LOG_DEBUG("[%s]empty graph key for cgraph: %p, size: %d\n", qnn::get_backend_name(ctx->device),
                      (const void *) cgraph, (int) cgraph->n_nodes);
        return nullptr;
    }

    auto             it        = graph_cache.find(graph_key);
    qnn::qnn_graph * graph_ptr = nullptr;
    if (it != graph_cache.end()) {
        auto it = graph_cache.find(graph_key);
        QNN_LOG_DEBUG("[%s]found graph %s in cache, cache size: %d\n", qnn::get_backend_name(ctx->device),
                      graph_key.c_str(), (int) graph_cache.size());
        graph_ptr = it->second.get();
    } else {
        auto precision = qnn::qnn_graph::kHtpDefault;
        if (op_data_type == GGML_TYPE_F16) {
            QNN_LOG_DEBUG("[%s][%s]set graph precision to FP16\n", qnn::get_backend_name(ctx->device),
                          graph_key.c_str());
            precision = qnn::qnn_graph::kHtpFp16;
        }

        auto graph = std::make_unique<qnn::qnn_graph>(graph_key, ctx->device, ctx->instance, precision,
                                                      ctx->socinfo.vtcm_size_in_mb);
        if (!graph->is_valid()) {
            return nullptr;
        }

        if (!graph->build_graph_from_ggml_graph(cgraph)) {
            QNN_LOG_ERROR("[%s]build_graph_from_op failed\n", qnn::get_backend_name(ctx->device));
            return nullptr;
        }

        graph_ptr              = graph.get();
        graph_cache[graph_key] = std::move(graph);
        QNN_LOG_DEBUG("[%s]add graph %s to cache, cache size: %d\n", qnn::get_backend_name(ctx->device),
                      graph_key.c_str(), (int) graph_cache.size());
    }

    return graph_ptr;
}

// TODO: could be merge into op caps array
constexpr const bool kQnnSupportedOps[] = {
    true,   // GGML_OP_NONE
    false,  // GGML_OP_DUP
    true,   // GGML_OP_ADD
    false,  // GGML_OP_ADD_ID
    false,  // GGML_OP_ADD1
    false,  // GGML_OP_ACC
    true,   // GGML_OP_SUB
    true,   // GGML_OP_MUL
    false,  // GGML_OP_DIV, disabled for now cause failed on test-backend-ops
    false,  // GGML_OP_SQR
    false,  // GGML_OP_SQRT, disabled for now cause failed on test-backend-ops
    true,   // GGML_OP_LOG
    false,  // GGML_OP_SIN
    false,  // GGML_OP_COS
    false,  // GGML_OP_SUM
    false,  // GGML_OP_SUM_ROWS
    false,  // GGML_OP_CUMSUM
    false,  // GGML_OP_MEAN
    false,  // GGML_OP_ARGMAX
    false,  // GGML_OP_COUNT_EQUAL
    false,  // GGML_OP_REPEAT
    false,  // GGML_OP_REPEAT_BACK
    false,  // GGML_OP_CONCAT
    false,  // GGML_OP_SILU_BACK
    false,  // GGML_OP_NORM
    false,  // GGML_OP_RMS_NORM
    false,  // GGML_OP_RMS_NORM_BACK
    false,  // GGML_OP_GROUP_NORM
    false,  // GGML_OP_L2_NORM

    true,   // GGML_OP_MUL_MAT
    false,  // GGML_OP_MUL_MAT_ID
    false,  // GGML_OP_OUT_PROD

    false,  // GGML_OP_SCALE
    false,  // GGML_OP_SET
    false,  // GGML_OP_CPY
    false,  // GGML_OP_CONT
    false,  // GGML_OP_RESHAPE
    false,  // GGML_OP_VIEW
    false,  // GGML_OP_PERMUTE
    false,  // GGML_OP_TRANSPOSE
    false,  // GGML_OP_GET_ROWS
    false,  // GGML_OP_GET_ROWS_BACK
    false,  // GGML_OP_SET_ROWS
    false,  // GGML_OP_DIAG
    false,  // GGML_OP_DIAG_MASK_INF
    false,  // GGML_OP_DIAG_MASK_ZERO
    false,  // GGML_OP_SOFT_MAX
    false,  // GGML_OP_SOFT_MAX_BACK
    false,  // GGML_OP_ROPE
    false,  // GGML_OP_ROPE_BACK
    false,  // GGML_OP_CLAMP
    false,  // GGML_OP_CONV_TRANSPOSE_1D
    false,  // GGML_OP_IM2COL
    false,  // GGML_OP_IM2COL_BACK
    false,  // GGML_OP_IM2COL_3D
    false,  // GGML_OP_CONV_2D
    false,  // GGML_OP_CONV_3D
    false,  // GGML_OP_CONV_2D_DW
    false,  // GGML_OP_CONV_TRANSPOSE_2D
    false,  // GGML_OP_POOL_1D
    false,  // GGML_OP_POOL_2D
    false,  // GGML_OP_POOL_2D_BACK
    false,  // GGML_OP_UPSCALE
    false,  // GGML_OP_PAD
    false,  // GGML_OP_ROLL
    false,  // GGML_OP_PAD_REFLECT_1D
    false,  // GGML_OP_ARANGE
    false,  // GGML_OP_TIMESTEP_EMBEDDING
    false,  // GGML_OP_ARGSORT
    false,  // GGML_OP_TOP_K
    false,  // GGML_OP_LEAKY_RELU
    false,  // GGML_OP_TRI
    false,  // GGML_OP_FILL

    false,  // GGML_OP_FLASH_ATTN_EXT
    false,  // GGML_OP_FLASH_ATTN_BACK
    false,  // GGML_OP_SSM_CONV
    false,  // GGML_OP_SSM_SCAN
    false,  // GGML_OP_WIN_PART
    false,  // GGML_OP_WIN_UNPART
    false,  // GGML_OP_GET_REL_POS
    false,  // GGML_OP_ADD_REL_POS
    false,  // GGML_OP_RWKV_WKV6
    false,  // GGML_OP_GATED_LINEAR_ATTN
    false,  // GGML_OP_RWKV_WKV7
    false,  // GGML_OP_SOLVE_TRI

    false,  // GGML_OP_UNARY

    false,  // GGML_OP_MAP_CUSTOM1
    false,  // GGML_OP_MAP_CUSTOM2
    false,  // GGML_OP_MAP_CUSTOM3

    false,  // GGML_OP_CUSTOM

    false,  // GGML_OP_CROSS_ENTROPY_LOSS
    false,  // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    false,  // GGML_OP_OPT_STEP_ADAMW
    false,  // GGML_OP_OPT_STEP_SGD
    false,  // GGML_OP_GLU

    // ggml_unary_op
    false,  // GGML_UNARY_OP_ABS
    false,  // GGML_UNARY_OP_SGN
    false,  // GGML_UNARY_OP_NEG
    false,  // GGML_UNARY_OP_STEP
    false,  // GGML_UNARY_OP_TANH
    false,  // GGML_UNARY_OP_ELU
    false,  // GGML_UNARY_OP_RELU
    false,  // GGML_UNARY_OP_SIGMOID
    true,   // GGML_UNARY_OP_GELU
    false,  // GGML_UNARY_OP_GELU_QUICK
    false,  // GGML_UNARY_OP_SILU
    false,  // GGML_UNARY_OP_HARDSWISH
    false,  // GGML_UNARY_OP_HARDSIGMOID
    false,  // GGML_UNARY_OP_EXP
    false,  // GGML_UNARY_OP_EXPM1
    false,  // GGML_UNARY_OP_SOFTPLUS
    false,  // GGML_UNARY_OP_GELU_ERF
    false,  // GGML_UNARY_OP_XIELU
    false,  // GGML_UNARY_OP_FLOOR
    false,  // GGML_UNARY_OP_CEIL
    false,  // GGML_UNARY_OP_ROUND
    false,  // GGML_UNARY_OP_TRUNC
};

static_assert(kQnnSupportedOps[GGML_OP_NONE], "GGML_OP_NONE is not true");
static_assert(kQnnSupportedOps[GGML_OP_ADD], "GGML_OP_ADD is not true");
static_assert(kQnnSupportedOps[GGML_OP_MUL], "GGML_OP_MUL is not true");
static_assert(kQnnSupportedOps[GGML_OP_MUL_MAT], "GGML_OP_MUL_MAT is not true");
static_assert(!kQnnSupportedOps[GGML_OP_RESHAPE], "GGML_OP_RESHAPE should not be true");
static_assert(!kQnnSupportedOps[GGML_OP_VIEW], "GGML_OP_VIEW is not false");
static_assert(std::size(kQnnSupportedOps) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kQnnSupportedOps table");

inline bool is_type_bit_enabled(uint64_t bits, ggml_type type) {
    return bits & (uint64_t(1) << type);
}

inline bool is_tensor_size_valid(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * tensor) {
    constexpr const auto get_tensor_size_in_bytes = [](const ggml_tensor * tensor, ggml_type type) -> size_t {
        return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3] * ggml_type_size(type);
    };

    auto type = tensor->type;
    if (ggml_is_quantized(type) && ctx->enable_cpu_dequantize) {
        type = GGML_TYPE_F32;  // TODO: [quantize] fix me if plan to dequantize to other types
    }

    const auto tensor_size = get_tensor_size_in_bytes(tensor, type);
    if (ctx->max_tensor_size_in_bytes && tensor_size >= ctx->max_tensor_size_in_bytes) {
        QNN_LOG_DEBUG("[%s]tensor(%s_%dx%dx%dx%d) size(%lld) exceeds the limit(%lld)\n",
                      qnn::get_backend_name(ctx->device), ggml_get_name(tensor), (int) tensor->ne[0],
                      (int) tensor->ne[1], (int) tensor->ne[2], (int) tensor->ne[3], (long long int) tensor_size,
                      (long long int) ctx->max_tensor_size_in_bytes);
        return false;
    }

    return true;
}

bool is_tensor_type_valid(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * tensor) {
    if (!tensor) {
        QNN_LOG_DEBUG("tensor is nullptr\n");
        return false;
    }

#ifndef NDEBUG
    if (tensor->view_src) {
        auto * src_tensor = tensor->view_src;
        QNN_LOG_DEBUG("[%s]tensor(%s_%dx%dx%dx%d) is a view, src: %s_%dx%dx%dx%d\n", qnn::get_backend_name(ctx->device),
                      ggml_get_name(tensor), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2],
                      (int) tensor->ne[3], ggml_get_name(src_tensor), (int) src_tensor->ne[0], (int) src_tensor->ne[1],
                      (int) src_tensor->ne[2], (int) src_tensor->ne[3]);
    }
#endif

    switch (tensor->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            if (!is_type_bit_enabled(ctx->supported_types, tensor->type)) {
                QNN_LOG_DEBUG("[%s]unsupported data type %s, supported_types: 0x%x\n",
                              qnn::get_backend_name(ctx->device), ggml_type_name(tensor->type),
                              (unsigned int) ctx->supported_types);
                return false;
            }
            break;
        default:
            QNN_LOG_DEBUG("[%s]unsupported data type %s\n", qnn::get_backend_name(ctx->device),
                          ggml_type_name(tensor->type));
            return false;
    }

    return true;
}

bool is_data_reinterpretation_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_PERMUTE;
}

bool ggnl_qnn_supports_op_tensor(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!is_tensor_type_valid(ctx, op) || !is_tensor_size_valid(ctx, op)) {
        return false;
    }

    // TODO: fix for other op
    const bool cpu_dequant = ctx->enable_cpu_dequantize && op->op == GGML_OP_MUL_MAT;
    for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
        auto * src = op->src[i];
        if (!is_tensor_size_valid(ctx, src)) {
            return false;
        }

        // passthrough the quantized tensor for CPU dequantization
        if (!is_tensor_type_valid(ctx, src) && (!cpu_dequant || !ggml_is_quantized(src->type))) {
            return false;
        }
    }

    return true;
}

bool ggml_qnn_have_same_tensor_types(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    auto * src0 = op->src[0];
    auto * src1 = op->src[1];
    if (src1) {
        if (src0->type != op->type || src1->type != op->type) {
            QNN_LOG_DEBUG("[%s][%s]type src0(%s), src1(%s) and op(%s) are not equal\n",
                          qnn::get_backend_name(ctx->device), ggml_op_name(op->op), ggml_type_name(src0->type),
                          ggml_type_name(src1->type), ggml_type_name(op->type));
            return false;
        }
    } else {
        if (src0->type != op->type) {
            QNN_LOG_DEBUG("[%s][%s]type src0(%s) and op(%s) are not equal\n", qnn::get_backend_name(ctx->device),
                          ggml_op_name(op->op), ggml_type_name(src0->type), ggml_type_name(op->type));
            return false;
        }
    }

#ifdef NDEBUG
    GGML_UNUSED(ctx);
#endif

    return true;
}

// TODO: move to caps array?
bool ggml_qnn_supports_matmul_op(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    auto * src0 = op->src[0];
    auto * src1 = op->src[1];
    if (is_data_reinterpretation_op(src0->op) || is_data_reinterpretation_op(src1->op)) {
        // TODO: remove the blocker here when we support permute op
        QNN_LOG_DEBUG("[%s][MUL_MAT]data reorganization op is not supported, (%s, %s)\n",
                      qnn::get_backend_name(ctx->device), ggml_op_name(src0->op), ggml_op_name(src1->op));
        return false;
    }

    switch (ctx->device) {
        case QNN_BACKEND_NPU:
            if (src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3]) {
                /*
                 * TODO: remove the blocker here when NPU backend supports mul_mat like this:
                 *   [ne03, ne02, n, k] * [ne03 * x, ne02 * y, m, k] -> [ne03 * x, ne02 * y, m, n]
                 */
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]src0 and src1 dimensions are not equal\n");
                return false;
            }
            // fall through, from test here, the convert op is super slow on NPU:
            //   https://github.com/usefulsensors/qc_npu_benchmark
        case QNN_BACKEND_GPU:
            if (!ggml_qnn_have_same_tensor_types(ctx, op) && op->type != GGML_TYPE_F32) {
                // for different tensor types and not float32, we don't support it currently, since there's no convert
                QNN_LOG_DEBUG("[%s][MUL_MAT]src0 and src1 and dst types are not equal\n",
                              qnn::get_backend_name(ctx->device));
                return false;
            }
            if (op->type == GGML_TYPE_F32 && ggml_is_quantized(src0->type) &&
                !is_type_bit_enabled(ctx->cpu_preprocess_types, src0->type)) {
                // for such cases that src0 is quantized and op is float32, check if the quant type is enabled
                QNN_LOG_DEBUG("[%s][MUL_MAT]quantized src0 type %s is not enabled\n",
                              qnn::get_backend_name(ctx->device), ggml_type_name(src0->type));
                return false;
            }
            break;
        default:
            break;
    }

    if ((src1->ne[2] % src0->ne[2]) != 0 || (src1->ne[3] % src0->ne[3]) != 0) {
        QNN_LOG_DEBUG("[%s][MUL_MAT]src0 and src1 dimensions are not equal\n", qnn::get_backend_name(ctx->device));
        return false;
    }

    QNN_LOG_DEBUG("[%s][MUL_MAT]supported matmul op\n", qnn::get_backend_name(ctx->device));
    return true;
}

#ifndef NDEBUG

void print_tensor_info(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * op, bool is_supported) {
    const char * supported = is_supported ? "supported" : "unsupported";
    std::string  op_key;
    qnn::get_qnn_op_desc(op, true, GGML_TYPE_COUNT, op_key);

    QNN_LOG_DEBUG("[%s][%s]op was %s, support/unsupported: %d/%d\n", qnn::get_backend_name(ctx->device), op_key.c_str(),
                  supported, ctx->supported_op_count.load(), ctx->unsupported_op_count.load());
}

#endif

}  // namespace

namespace qnn {

bool device_supports_op(qnn::ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    // Note that this function could be called before the device context is initialized
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!kQnnSupportedOps[qnn::get_qnn_op_index(op)]) {
#ifndef NDEBUG
        ctx->unsupported_op_count++;
        print_tensor_info(ctx, op, false);
#endif
        return false;
    }

    if (!ggnl_qnn_supports_op_tensor(ctx, op)) {
#ifndef NDEBUG
        ctx->unsupported_op_count++;
        print_tensor_info(ctx, op, false);
#endif
        return false;
    }

    bool is_op_supported = true;
    if (op->op == GGML_OP_UNARY) {
        const auto unary_op = ggml_get_unary_op(op);
        if (unary_op == GGML_UNARY_OP_GELU) {
            // TODO: fix this
            QNN_LOG_DEBUG("[GELU]unsupported unary op GGML_UNARY_OP_GELU for NPU\n");
            is_op_supported = false;
        }
    } else {
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];
        switch (op->op) {
            case GGML_OP_MUL:
                // TODO: fix this when we have the support for mul with rms_norm
                if (ctx->enable_cpu_dequantize && (src0->op == GGML_OP_RMS_NORM || src1->op == GGML_OP_RMS_NORM)) {
                    QNN_LOG_DEBUG("[%s][%s]skip unsupported mul with rms norm, (%s, %s)\n",
                                  qnn::get_backend_name(ctx->device), ggml_op_desc(op), ggml_op_desc(src0),
                                  ggml_op_desc(src1));
                    is_op_supported = false;
                    break;
                }
                // fall through, just skip the mul with rms_norm, in llama, its at start of decoder block
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
                // TODO: move to op caps array?
                if (!ggml_are_same_shape(src0, src1)) {
                    QNN_LOG_DEBUG("[%s][%s] src0 and src1 dimensions are not equal\n",
                                  qnn::get_backend_name(ctx->device), ggml_op_desc(op));
                    is_op_supported = false;
                }
                break;
            case GGML_OP_MUL_MAT:
                is_op_supported = ggml_qnn_supports_matmul_op(ctx, op);
                break;

            default:
                is_op_supported = ggml_qnn_have_same_tensor_types(ctx, op);
                break;
        }
    }

#ifndef NDEBUG
    if (is_op_supported) {
        ctx->supported_op_count++;
    } else {
        ctx->unsupported_op_count++;
    }

    print_tensor_info(ctx, op, is_op_supported);
#endif

    return is_op_supported;
}

bool device_compute_graph(qnn::ggml_backend_qnn_device_context * ctx, ggml_cgraph * cgraph) {
    QNN_LOG_DEBUG("[%s]compute graph start, nodes count: %d\n", qnn::get_backend_name(ctx->device),
                  (int) cgraph->n_nodes);

    auto qnn_graph = get_qnn_graph_from_cache(ctx, cgraph);
    bool success   = qnn_graph && qnn_graph->execute(cgraph, ctx->convert_context);

    QNN_LOG_DEBUG("[%s]compute graph, success: %d\n", qnn::get_backend_name(ctx->device), (int) success);
    return success;
}

}  // namespace qnn
