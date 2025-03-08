
#include "backend-ops.hpp"

#include <memory>

#include "ggml-impl.h"
#include "graph.hpp"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace {

bool qnn_is_op_valid(ggml_backend_qnn_device_context * ctx, const ggml_tensor * dst) {
    if (!ctx || !dst) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    auto instance = ctx->instance;
    if (!instance) {
        QNN_LOG_WARN("invalid instance\n");
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(dst);
    switch (param_count) {
        case 1:
            return dst->src[0];
        case 2:
            return dst->src[0] && dst->src[1];
        default:
            QNN_LOG_WARN("invalid op param count %d\n", (int) param_count);
            break;
    }

    return false;
}

#ifndef NDEBUG
void print_ggml_tensor(const ggml_tensor * tensor) {
    QNN_LOG_DEBUG("%s: type:%s ne: %ldx%ldx%ldx%ld, nb: %ldx%ldx%ldx%ld\n", tensor->name, ggml_type_name(tensor->type),
                  (long) tensor->ne[0], (long) tensor->ne[1], (long) tensor->ne[2], (long) tensor->ne[3],
                  (long) tensor->nb[0], (long) tensor->nb[1], (long) tensor->nb[2], (long) tensor->nb[3]);
}
#endif

}  // namespace

namespace {

typedef bool (*ggml_qnn_op_t)(ggml_backend_qnn_device_context * ctx, ggml_tensor * dst);

void append_tensor_dimensions(const ggml_tensor * tensor, std::string & output) {
    char         buffer[256] = {};
    const auto * type_name   = qnn::get_ggml_type_name(tensor->type);
    int          len         = 0;
    switch (ggml_n_dims(tensor)) {
        case 1:
            len = snprintf(buffer, sizeof(buffer), "%ld%s", (long) tensor->ne[0], type_name);
            break;
        case 2:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ld%s", (long) tensor->ne[0], (long) tensor->ne[1], type_name);
            break;
        case 3:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ld%s", (long) tensor->ne[0], (long) tensor->ne[1],
                           (long) tensor->ne[2], type_name);
            break;
        case 4:
        default:
            len = snprintf(buffer, sizeof(buffer), "%ldx%ldx%ldx%ld%s", (long) tensor->ne[0], (long) tensor->ne[1],
                           (long) tensor->ne[2], (long) tensor->ne[3], type_name);
            break;
    }
    GGML_ASSERT(len > 0 && len < (int) sizeof(buffer));
    output.append(buffer, len);
}

void get_graph_key_from_op(const ggml_tensor * op, std::string & output) {
    GGML_ASSERT(op->op != GGML_OP_NONE);
    output += ggml_op_desc(op);
    output += qnn::get_ggml_type_name(op->type);
    const auto param_count = qnn::get_qnn_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        auto * input = op->src[i];
        if (!input) {
            break;
        }

        output += '_';
        append_tensor_dimensions(input, output);
    }
}

void get_op_key_with_src_op_desc(const ggml_tensor * op, std::string & output) {
    output += ggml_op_desc(op);
    output += '(';
    if (op->src[0]) {
        output += ggml_op_desc(op->src[0]);
    }
    for (size_t i = 1; i < GGML_MAX_DIMS && op->src[i]; ++i) {
        output += ',';
        output += ggml_op_desc(op->src[i]);
    }
    output += ')';
}

/**
 * @brief Generates a unique key for a given computation graph (cgraph).
 *
 * This key is used to cache the graph, enabling efficient reuse of previously
 * compiled graphs. The key is constructed by concatenating the descriptions
 * of the operations and their associated tensor dimensions within the graph.
 *
 * Example key format: "MUL_MATf32_256x16x10f32_256x1x10f32#LOG#ADD#ADDf32_16x1x10f32"
 *
 * @param cgraph The computation graph for which the key is generated.
 * @param output The string where the generated key will be stored.
 *
 * TODO: Improve the key generation logic to handle more complex graph structures and edge cases.
 */
void get_graph_key_from_cgraph(const ggml_cgraph * cgraph, std::string & output) {
    if (cgraph->n_nodes == 0) {
        QNN_LOG_DEBUG("empty cgraph\n");
        return;
    }

    {
        bool is_start = true;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            auto * op = cgraph->nodes[i];
            if (ggml_is_empty(op)) {
                QNN_LOG_DEBUG("empty op in graph, skipping\n");
                continue;
            }

            if (op->op == GGML_OP_NONE) {
                QNN_LOG_DEBUG("GGML_OP_NONE in graph, skipping\n");
                continue;
            }

            if (is_start) {
                get_graph_key_from_op(cgraph->nodes[0], output);
                is_start = false;
            } else {
                output += '#';
                get_op_key_with_src_op_desc(op, output);
            }
        }
    }

    if (cgraph->n_nodes > 1) {
        auto * last_op = cgraph->nodes[cgraph->n_nodes - 1];
        output += qnn::get_ggml_type_name(last_op->type);
        output += '_';
        append_tensor_dimensions(last_op, output);
    }
}

qnn::qnn_graph * get_qnn_graph_from_cache(ggml_backend_qnn_device_context * ctx, const ggml_cgraph * cgraph) {
    auto &      graph_cache = ctx->qnn_graph_cache;
    std::string graph_key;
    get_graph_key_from_cgraph(cgraph, graph_key);
    if (graph_key.empty()) {
        QNN_LOG_DEBUG("[%s]empty graph key for cgraph: %p, size: %d\n", qnn::get_backend_name(ctx->device),
                      (const void *) cgraph, (int) cgraph->n_nodes);
        return nullptr;
    }

    auto             it        = graph_cache.find(graph_key);
    qnn::qnn_graph * graph_ptr = nullptr;
    if (it != graph_cache.end()) {
        QNN_LOG_DEBUG("[%s]found graph %s in cache\n", qnn::get_backend_name(ctx->device), graph_key.c_str());
        graph_ptr = it->second.get();
    } else {
        auto graph =
            std::make_unique<qnn::qnn_graph>(graph_key, ctx->device, ctx->instance, ctx->socinfo.vtcm_size_in_mb);
        if (!graph->is_valid()) {
            return nullptr;
        }

        if (!graph->build_graph_from_ggml_graph(cgraph)) {
            QNN_LOG_ERROR("[%s]build_graph_from_op failed\n", qnn::get_backend_name(ctx->device));
            return nullptr;
        }

        graph_ptr              = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
}

// TODO: could be merge into op caps array
constexpr const bool kQnnSupportedOps[] = {
    true,   // GGML_OP_NONE
    false,  // GGML_OP_DUP
    true,   // GGML_OP_ADD
    false,  // GGML_OP_ADD1
    false,  // GGML_OP_ACC
    true,   // GGML_OP_SUB
    true,   // GGML_OP_MUL
    true,   // GGML_OP_DIV
    false,  // GGML_OP_SQR
    true,   // GGML_OP_SQRT
    true,   // GGML_OP_LOG
    false,  // GGML_OP_SIN
    false,  // GGML_OP_COS
    false,  // GGML_OP_SUM
    false,  // GGML_OP_SUM_ROWS
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

    true,   // GGML_OP_MUL_MAT
    false,  // GGML_OP_MUL_MAT_ID
    false,  // GGML_OP_OUT_PROD

    false,  // GGML_OP_SCALE
    false,  // GGML_OP_SET
    false,  // GGML_OP_CPY
    false,  // GGML_OP_CONT
    true,   // GGML_OP_RESHAPE
    false,  // GGML_OP_VIEW
    false,  // GGML_OP_PERMUTE
    false,  // GGML_OP_TRANSPOSE
    false,  // GGML_OP_GET_ROWS
    false,  // GGML_OP_GET_ROWS_BACK
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
    false,  // GGML_OP_CONV_TRANSPOSE_2D
    false,  // GGML_OP_POOL_1D
    false,  // GGML_OP_POOL_2D
    false,  // GGML_OP_POOL_2D_BACK
    false,  // GGML_OP_UPSCALE
    false,  // GGML_OP_PAD
    false,  // GGML_OP_PAD_REFLECT_1D
    false,  // GGML_OP_ARANGE
    false,  // GGML_OP_TIMESTEP_EMBEDDING
    false,  // GGML_OP_ARGSORT
    false,  // GGML_OP_LEAKY_RELU

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

    false,  // GGML_OP_UNARY

    false,  // GGML_OP_MAP_UNARY
    false,  // GGML_OP_MAP_BINARY

    false,  // GGML_OP_MAP_CUSTOM1_F32
    false,  // GGML_OP_MAP_CUSTOM2_F32
    false,  // GGML_OP_MAP_CUSTOM3_F32

    false,  // GGML_OP_MAP_CUSTOM1
    false,  // GGML_OP_MAP_CUSTOM2
    false,  // GGML_OP_MAP_CUSTOM3

    false,  // GGML_OP_CROSS_ENTROPY_LOSS
    false,  // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    false,  // GGML_OP_OPT_STEP_ADAMW

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
};

static_assert(kQnnSupportedOps[GGML_OP_NONE], "GGML_OP_NONE is not true");
static_assert(kQnnSupportedOps[GGML_OP_ADD], "GGML_OP_ADD is not true");
static_assert(kQnnSupportedOps[GGML_OP_MUL], "GGML_OP_MUL is not true");
static_assert(kQnnSupportedOps[GGML_OP_MUL_MAT],
              "GGML_OP_MUL_MAT is not true, please check the kQnnSupportedOps table in the backend-ops.cpp file");
static_assert(kQnnSupportedOps[GGML_OP_RESHAPE], "GGML_OP_RESHAPE is not true");
static_assert(!kQnnSupportedOps[GGML_OP_VIEW], "GGML_OP_VIEW is not false");
static_assert(std::size(kQnnSupportedOps) == (GGML_OP_COUNT + GGML_UNARY_OP_COUNT),
              "GGML_OP_COUNT does not match the size of the kQnnSupportedOps table");

bool ggml_qnn_supports_tensor(ggml_backend_qnn_device_context * ctx, const ggml_tensor * tensor) {
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
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            if (!(ctx->supported_types & (uint64_t(1) << tensor->type))) {
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

bool ggnl_qnn_supports_op_tensor(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!ggml_qnn_supports_tensor(ctx, op)) {
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(op);
    for (size_t i = 0; i < param_count; ++i) {
        if (!ggml_qnn_supports_tensor(ctx, op->src[i])) {
            return false;
        }
    }

    return true;
}

bool ggml_qnn_have_same_tensor_types(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
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

bool ggml_qnn_supports_matmul_op(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    constexpr const size_t kMaxNpuTensorSize = 8192L * 2048 + 8192 * 512 + 2048 * 512;
    constexpr const auto   get_tensor_size   = [](const ggml_tensor * tensor) -> size_t {
        return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    };

    auto * src0 = op->src[0];
    auto * src1 = op->src[1];
    switch (ctx->device) {
        case QNN_BACKEND_NPU:
            if (src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3]) {
                /*
                 * TODO: remove the blocker here when NPU backend supports mul_mat like this:
                 *   [ne03, ne02, n, k] * [ne03 * x, ne02 * y, m, k] -> [ne03 * x, ne02 * y, m, n]
                 */
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]src0 and src1 dimensions are not equal\n");
                return false;
            } else if (get_tensor_size(src0) + get_tensor_size(src1) + get_tensor_size(op) >= kMaxNpuTensorSize) {
                QNN_LOG_DEBUG("[qnn-npu][MUL_MAT]tensor size is too large\n");
                return false;
            }
            // fall through, from test here, the convert op is super slow on NPU:
            //   https://github.com/usefulsensors/qc_npu_benchmark
        case QNN_BACKEND_GPU:
            if (ggml_qnn_have_same_tensor_types(ctx, op)) {
                // there's no convert op for GPU.
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

}  // namespace

namespace qnn {

bool device_supports_op(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op) {
    // Note that this function could be called before the device context is initialized
    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (!kQnnSupportedOps[qnn::get_qnn_op_index(op)]) {
#ifndef NDEBUG
        std::string op_key;
        get_graph_key_from_op(op, op_key);
        ctx->unsupported_op_count++;
        QNN_LOG_DEBUG("[%s][%s]op was unsupported, support/unsupported: %d/%d\n", qnn::get_backend_name(ctx->device),
                      op_key.c_str(), ctx->supported_op_count.load(), ctx->unsupported_op_count.load());
#endif
        return false;
    }

    if (!ggnl_qnn_supports_op_tensor(ctx, op)) {
#ifndef NDEBUG
        std::string tensor_dims;
        append_tensor_dimensions(op, tensor_dims);
        QNN_LOG_DEBUG("[%s][%s]unsupported tensor(%s), support/unsupported: %d/%d\n",
                      qnn::get_backend_name(ctx->device), ggml_op_name(op->op), tensor_dims.c_str(),
                      ctx->supported_op_count.load(), ctx->unsupported_op_count.load());
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
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
                if (!ggml_are_same_shape(src0, src1)) {
                    QNN_LOG_DEBUG("[%s][%s] src0 and src1 dimensions are not equal\n",
                                  qnn::get_backend_name(ctx->device), ggml_op_name(op->op));
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
        QNN_LOG_DEBUG("[%s][%s]op was supported, support/unsupported: %d/%d\n", qnn::get_backend_name(ctx->device),
                      ggml_op_name(op->op), ctx->supported_op_count.load(), ctx->unsupported_op_count.load());
    } else {
        ctx->unsupported_op_count++;
        QNN_LOG_DEBUG("[%s][%s]op was unsupported, support/unsupported: %d/%d\n", qnn::get_backend_name(ctx->device),
                      ggml_op_name(op->op), ctx->supported_op_count.load(), ctx->unsupported_op_count.load());
    }
#endif

    return is_op_supported;
}

bool device_compute_graph(ggml_backend_qnn_device_context * ctx, ggml_cgraph * cgraph) {
    QNN_LOG_DEBUG("[%s]compute graph start, nodes count: %d\n", qnn::get_backend_name(ctx->device),
                  (int) cgraph->n_nodes);

    auto qnn_graph = get_qnn_graph_from_cache(ctx, cgraph);
    bool success   = qnn_graph && qnn_graph->execute(cgraph);

    QNN_LOG_DEBUG("[%s]compute graph, success: %d\n", qnn::get_backend_name(ctx->device), (int) success);
    return success;
}

}  // namespace qnn
