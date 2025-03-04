/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "ggml-impl.h"
#include "ggml-common.h"
#include "ggml-qnn-ops.h"

static inline uint32_t ggmlqnn_get_tensor_data_size(const ggml_tensor * tensor) {
    /*
    size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t n_dims = ggml_get_tensor_rank(tensor);
    for (int i = 1; i < n_dims; i++) {
        data_size *= tensor->ne[i];
    }

    return data_size;
    */
    return ggml_nbytes(tensor);
}

static inline bool ggmlqnn_is_valid_params(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                             const ggml_tensor * src1, ggml_tensor * dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    qnn_instance * instance = ctx->instance;
    if (nullptr == instance) {
        GGMLQNN_LOG_WARN("invalid params\n");
        return false;
    }

    return true;
}

#define GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst)                          \
    do {                                                                    \
        if (!ggmlqnn_is_valid_params((ctx), (src0), (src1), (dst))) {       \
            return;                                                         \
        }                                                                   \
    } while (0)

/*
 * provide a general skeleton to offload ggml op to QNN backend: a single node contains 2 input
 * tensor and 1 output tensor
*/
void ggml_qnn_general_node(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor * dst                           = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    size_t qnn_op_index                         = ggmlqnn_get_op_index(op);
    GGML_ASSERT(qnn_op_index < ggmlqnn_get_opcaps_size());
    const char * qnn_op_name                    = ggmlqnn_k_op_caps[qnn_op_index].qnn_op_name;
    std::string ggml_op_name_string             = std::string("ggml_") + ggml_op_name(op->op);
    const char * ggml_op_name                   = ggml_op_name_string.c_str();

    qnn_perf op_perf                            = qnn_perf(ggml_op_name);
    op_perf.start();

    //ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);
    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == QNN_BACKEND_NPU;

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensor  = std::get<1>(graph_item);
        p_tensor0               = tensor[0];
        p_tensor1               = tensor[1];
        p_tensor2               = tensor[2];
    } else {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        GGML_ASSERT(instance->get_device_id() == ctx->device);
        //create QNN graph
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        //create computational tensor
        p_tensor0 = ggmlqnn_create_compute_tensor(instance, graph_handle, src0, QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor1 = ggmlqnn_create_compute_tensor(instance, graph_handle, src1, QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor2 = ggmlqnn_create_compute_tensor(instance, graph_handle, dst,  QNN_TENSOR_TYPE_APP_READ);

        //compose QNN graph
        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        Qnn_OpConfig_t op_config = {
                QNN_OPCONFIG_VERSION_1, .v1 = {
                        ggml_op_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        nullptr,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_tensors_t ggml_op_add_tensors;
        ggml_op_add_tensors.reserve(3);
        ggml_op_add_tensors.push_back(p_tensor0);
        ggml_op_add_tensors.push_back(p_tensor1);
        ggml_op_add_tensors.push_back(p_tensor2);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_add_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    }

    if (enable_npu_rpc) {
        uint8_t * qnn_buffer_0 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor0)->memHandle));
        GGMLQNN_LOG_INFO("qnn_rpcbuffer_0 = %p\n", qnn_buffer_0);
        if (nullptr != qnn_buffer_0) {
            memcpy(qnn_buffer_0, src0->data, ggml_nbytes(src0));
        }

        uint8_t * qnn_buffer_1 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor1)->memHandle));
        GGMLQNN_LOG_INFO("qnn_rpcbuffer_1 = %p\n", qnn_buffer_1);
        if (nullptr != qnn_buffer_1) {
            memcpy(qnn_buffer_1, src1->data, ggml_nbytes(src1));
        }
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};
    }

    Qnn_Tensor_t tensor_inputs[] = {
            *p_tensor0,
            *p_tensor1
    };
    Qnn_Tensor_t tensor_outputs[] = {
            *p_tensor2
    };
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                        tensor_inputs, 2,
                                                        tensor_outputs, 1,
                                                        nullptr, nullptr));
    if (enable_npu_rpc) {
        //TODO:NPU RPC feature will failed with test-backend-ops
        uint8_t * qnn_buffer_2 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
        if (nullptr != qnn_buffer_2) {
            memcpy(dst->data, qnn_buffer_2, ggml_nbytes(dst));
        }
    }

#if GGMLQNN_PRINT_OP_ADD_LOG
    op_perf.info();
#endif
}

/*
 * this function is AI-assisted code from Grok 3 for purpose of offload 4d matrix mulmat to QNN backend
 * UT in ggml-qnn-ut.cpp passed:
 * ./scripts/build-run-android.sh run_ut_mulmat 0
 * ./scripts/build-run-android.sh run_ut_mulmat 1
 * ./scripts/build-run-android.sh run_ut_mulmat 2
 *
 * the logic of ggml_qnn_mul_mat_4d is similar to ggml_qnn_mul_mat but much more complicated
 * than ggml_qnn_mul_mat, so it's a standalone function.
 * it will be combined with ggml_qnn_mul_mat in the future
 */
static void ggml_qnn_mul_mat_4d(ggml_backend_qnn_context *ctx, ggml_tensor *op) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    bool graph_initialized = false;
    qnn_perf op_perf = qnn_perf("ggml_qnn_mul_mat_4d");
    qnn_instance *instance = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->raw_interface;

    const ggml_tensor *src0 = op->src[0];
    const ggml_tensor *src1 = op->src[1];
    ggml_tensor *dst = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    GGML_ASSERT(ggml_n_dims(src0) == 4 && ggml_n_dims(src1) == 4);
    op_perf.start();

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    GGMLQNN_LOG_DEBUG("graph name %s\n", graph_name.c_str());

    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    Qnn_GraphHandle_t graph_handle = nullptr;
    Qnn_Tensor_t *p_tensor0 = nullptr;
    Qnn_Tensor_t *p_reshape0_out = nullptr;
    Qnn_Tensor_t *p_tile0_out = nullptr;
    Qnn_Tensor_t *p_tensor1 = nullptr;
    Qnn_Tensor_t *p_permute1_out = nullptr;
    Qnn_Tensor_t *p_reshape1_out = nullptr;
    Qnn_Tensor_t *p_matmul_out = nullptr;
    Qnn_Tensor_t *p_reshape2_out = nullptr;

    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t &graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t &tensors = std::get<1>(graph_item);
        p_tensor0 = tensors[0];
        p_reshape0_out = tensors[1];
        p_tile0_out = tensors[2];
        p_tensor1 = tensors[3];
        p_permute1_out = tensors[4];
        p_reshape1_out = tensors[5];
        p_matmul_out = tensors[6];
        p_reshape2_out = tensors[7];
    } else {
        CHECK_QNN_API(error, qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                                           graph_name.c_str(), NULL, &graph_handle));

        // Define dimensions
        uint32_t K = src0->ne[0];               // Inner dimension
        uint32_t M = src0->ne[1];               // Rows of src0
        uint32_t N = src1->ne[1];               // Columns of src1
        uint32_t B0 = src0->ne[2] * src0->ne[3]; // src0 batch
        uint32_t B1 = src1->ne[2] * src1->ne[3]; // src1 batch (drives output)

        // Validate K only
        GGML_ASSERT(src0->ne[0] == src1->ne[0]); // K must match

        // src0: [K, M, H0, B0] -> QNN: [B0, H0, M, K]
        uint32_t src0_dims[] = {static_cast<uint32_t>(src0->ne[3]), static_cast<uint32_t>(src0->ne[2]), static_cast<uint32_t>(src0->ne[1]), static_cast<uint32_t>(src0->ne[0])};
        p_tensor0 = GQCGT(src0, "input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src0_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));

        // Reshape src0 to [B0, M, K]
        uint32_t reshape0_out_dims[] = {B0, M, K};
        p_reshape0_out = GQCGT(nullptr, "reshape0_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                               reshape0_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape0_out));
        Qnn_Tensor_t reshape0_inputs[] = {*p_tensor0};
        Qnn_Tensor_t reshape0_outputs[] = {*p_reshape0_out};
        Qnn_OpConfig_t reshape0_op = ggmlqnn_create_op_config("reshape0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape0_inputs, 1, reshape0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape0_op));

        // Tile src0 to match B1: [B0, M, K] -> [B1, M, K]
        uint32_t tile0_out_dims[] = {B1, M, K};
        p_tile0_out = GQCGT(nullptr, "tile0_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                            tile0_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tile0_out));
        uint32_t tile_multiples[] = {B1 / B0, 1, 1};
        uint32_t tile_dims[] = {3};
        Qnn_Tensor_t *p_tile_multiples = GQCGT(nullptr, "tile_multiples", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                               tile_dims, tile_multiples, sizeof(tile_multiples));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tile_multiples));
        Qnn_Param_t tile_params[] = {{QNN_PARAMTYPE_TENSOR, "multiples", .tensorParam = *p_tile_multiples}};
        Qnn_Tensor_t tile0_inputs[] = {*p_reshape0_out};
        Qnn_Tensor_t tile0_outputs[] = {*p_tile0_out};
        Qnn_OpConfig_t tile0_op = ggmlqnn_create_op_config("tile0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                           QNN_OP_TILE, tile_params, 1,
                                                           tile0_inputs, 1, tile0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, tile0_op));

        // src1: [N, K, H1, B1] -> QNN: [B1, H1, N, K]
        uint32_t src1_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]), static_cast<uint32_t>(src1->ne[1]), static_cast<uint32_t>(src1->ne[0])};
        p_tensor1 = GQCGT(src1, "input1", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src1_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));

        // Permute src1 to [B1, H1, K, N]
        uint32_t perm_data[] = {0, 1, 3, 2};
        uint32_t perm_dims[] = {4};
        Qnn_Tensor_t *p_perm = GQCGT(nullptr, "perm", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                     perm_dims, perm_data, sizeof(perm_data));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_perm));
        uint32_t permute1_out_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]), static_cast<uint32_t>(src1->ne[0]), static_cast<uint32_t>(src1->ne[1])};
        p_permute1_out = GQCGT(nullptr, "permute1_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 4,
                               permute1_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_permute1_out));
        Qnn_Param_t permute1_params[] = {{QNN_PARAMTYPE_TENSOR, "perm", .tensorParam = *p_perm}};
        Qnn_Tensor_t permute1_inputs[] = {*p_tensor1};
        Qnn_Tensor_t permute1_outputs[] = {*p_permute1_out};
        Qnn_OpConfig_t permute1_op = ggmlqnn_create_op_config("permute1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_TRANSPOSE, permute1_params, 1,
                                                              permute1_inputs, 1, permute1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, permute1_op));

        // Reshape src1 to [B1, K, N]
        uint32_t reshape1_out_dims[] = {B1, K, N};
        p_reshape1_out = GQCGT(nullptr, "reshape1_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                               reshape1_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape1_out));
        Qnn_Tensor_t reshape1_inputs[] = {*p_permute1_out};
        Qnn_Tensor_t reshape1_outputs[] = {*p_reshape1_out};
        Qnn_OpConfig_t reshape1_op = ggmlqnn_create_op_config("reshape1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape1_inputs, 1, reshape1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape1_op));

        // MatMul: [B1, M, K] x [B1, K, N] -> [B1, M, N]
        uint32_t matmul_out_dims[] = {B1, M, N};
        p_matmul_out = GQCGT(nullptr, "matmul_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 3,
                             matmul_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_matmul_out));
        Qnn_Tensor_t matmul_inputs[] = {*p_tile0_out, *p_reshape1_out};
        Qnn_Tensor_t matmul_outputs[] = {*p_matmul_out};
        Qnn_OpConfig_t matmul_op = ggmlqnn_create_op_config("matmul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            QNN_OP_MAT_MUL, nullptr, 0,
                                                            matmul_inputs, 2, matmul_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, matmul_op));

        // Output: [N, M, H1, B1] -> QNN: [B1, H1, M, N]
        uint32_t reshape2_out_dims[] = {static_cast<uint32_t>(dst->ne[3]), static_cast<uint32_t>(dst->ne[2]), static_cast<uint32_t>(dst->ne[1]), static_cast<uint32_t>(dst->ne[0])};
        p_reshape2_out = GQCGT(dst, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, 4,
                               reshape2_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_reshape2_out));
        Qnn_Tensor_t reshape2_inputs[] = {*p_matmul_out};
        Qnn_Tensor_t reshape2_outputs[] = {*p_reshape2_out};
        Qnn_OpConfig_t reshape2_op = ggmlqnn_create_op_config("reshape2", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                              QNN_OP_RESHAPE, nullptr, 0,
                                                              reshape2_inputs, 1, reshape2_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, reshape2_op));

        // Finalize
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, NULL, NULL));

        // Cache
        qnn_tensors_t ggml_op_mulmat_tensors = {p_tensor0, p_reshape0_out, p_tile0_out, p_tensor1, p_permute1_out, p_reshape1_out, p_matmul_out, p_reshape2_out};
        instance->_qnn_graph_map[graph_name] = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
    }

    // Execute
    QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
    QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
    QNN_VER_PTR(*p_reshape2_out)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

    Qnn_Tensor_t input_tensors[] = {*p_tensor0, *p_tensor1};
    Qnn_Tensor_t output_tensors[] = {*p_reshape2_out};
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle, input_tensors, 2,
                                                        output_tensors, 1, NULL, NULL));

#if 0
    // Log dst for debugging
    float *dst_data = (float *)dst->data;
    GGMLQNN_LOG_DEBUG("dst shape: [%d, %d, %d, %d]\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    for (int i = 0; i < dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3]; i++) {
        GGMLQNN_LOG_DEBUG("dst[%d] = %f\n", i, dst_data[i]);
    }
#endif

    op_perf.info();
}

/*
 * @brief performs matrix multiplication with FP32 & quantized weights and floating-point inputs
 *        using the QNN backend. this function performs matrix multiplication of the input tensor
 *        `src1` and the weight tensor `src0`, handling transposing, and quantization as needed,
 *        and stores the result in the destination tensor `dst`.
 *
         there are two key-points in properly handling how to offload mulmat to the QNN backend in ggml-qnn
         1. transpose
            a 3x2 f32 matrix which means 3 rows and 2 columns. in ggml, it could be created from:
            struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            which like this:
            +---+---+
            | 0 | 1 |
            +---+---+
            | 2 | 3 |
            +---+---+
            | 4 | 5 |
            +---+---+
            with
                ne[0] = 2
                ne[1] = 3
            there are different dimension order between ggml tensor and qnn tensor

          2. QNN's MatMul can only support input tensors with rank >= 2

             in the all, there is gap between ggml mulmat and QNN mulmat,we need to perform a transpose
             operation when offloading mulmat to QNN backend. this implementation will handle transpose
             in func ggml_qnn_create_general_tensor()
 *
 *        this function is a good example to illustrated the second technical approach "mapping the
 *        entire ggml computational graph to QNN graph" without complex C++ encapsulation. or another
 *        pipeline of "how to utilize the Hexagon NPU maximally through QNN SDK", details could be found at
 *        https://github.com/ggml-org/llama.cpp/pull/12049#issuecomment-2678308360
 *
 * @param ctx     the context of ggml-qnn backend
 * @param op      the destination tensor where the result of the matrix multiplication will be stored.
 *
 * @note the logic of ggml_qnn_mul_mat is similar to ggml_qnn_general_node but much more complicated
 *       than ggml_qnn_general_node. so it's a standalone function. accordingly, this is another
 *       typical skeleton for offload other ggml ops to QNN backend. MUL_MAT take most of the compute
 *       time (about 95%).so to speed up llama inference, should focus on this func. there are three kinds
 *       of MUL_MAT to compute:
 *       mul_mat_f32:     both src0 and src1 are F32, this will be naturally handled in QNN backend
 *       mul_mat_f16_f32: src0 is F16 and src1 is F32, f16 in src0 -> f32 in src0', then src0' * src1
 *       mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, Q6_K...)
 *                        and src1 is F32, src0 -> f32 in src0', then src0' * src1
*/
void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    qnn_perf op_perf                            = qnn_perf("ggml_qnn_mul_mat");
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Tensor_t * p_param_tensor               = nullptr;
    Qnn_Tensor_t * p_tensor2_transpose          = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor       * dst                     = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    op_perf.start();

    const enum ggml_type src0_type              = src0->type;
    const uint32_t src0_rank                    = ggml_n_dims(src0);
    const uint32_t src1_rank                    = ggml_n_dims(src1);
    GGML_ASSERT(src0_rank == src1_rank);
    GGML_ASSERT(src0_rank >= 2); //QNN SDK's limitation, make QNN SDK happy
    if (4 == src0_rank) {
        return ggml_qnn_mul_mat_4d(ctx, op);
    }
    void * wdata                                = ggmlqnn_type_trait(ctx, op);
    const size_t desired_size                   = ctx->desired_size;

    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        //retrieve computational resource from cached QNN graph
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensors = std::get<1>(graph_item);
        p_tensor0               = tensors[0];
        p_tensor1               = tensors[1];
        p_tensor2               = tensors[2];
        p_param_tensor          = tensors[3];
        p_tensor2_transpose     = tensors[4];
    } else {
        //create QNN graph
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(), graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_WARN("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }

        //create computational tensor
        p_tensor0 = GQCGT(src0, nullptr, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor1 = GQCGT(src1, nullptr, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor2 = GQCGT(dst, nullptr, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        //create param tensor for offload 2d/3d/4d matrix multiplication
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1] = {src0_rank};
        p_param_tensor = GQCGT(nullptr, "param", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1, param_tensor_dims, (void *)(param_tensor_data[src0_rank - 1]), src0_rank * sizeof(uint32_t));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_param_tensor));

        //create transpose tensor
        p_tensor2_transpose = GQCGT(dst, "transpose", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0, true);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2_transpose));

        //compose QNN graph: add mulmat node
        Qnn_Param_t out_0_params[]   = {{QNN_PARAMTYPE_SCALAR, QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, .scalarParam = {QNN_DATATYPE_BOOL_8, .bool8Value = 1}}};
        Qnn_Tensor_t out_0_inputs[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
        Qnn_OpConfig_t out_0         = ggmlqnn_create_op_config("mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL, out_0_params, 1, out_0_inputs, 2, out_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_0));

        //compose QNN graph: add transpose node
        Qnn_Param_t out_trans1_0_params[]   = { {QNN_PARAMTYPE_TENSOR, "perm", .tensorParam = *p_param_tensor}};
        Qnn_Tensor_t out_trans1_0_inputs[]  = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
        Qnn_OpConfig_t out_trans1_0         = ggmlqnn_create_op_config("mulmat_transpose_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE, out_trans1_0_params, 1, out_trans1_0_inputs, 1, out_trans1_0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_trans1_0));

        //finalize QNN graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));

        //cache QNN graph
        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    }

    if (src0_type != GGML_TYPE_F32) {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {wdata, static_cast<uint32_t>(desired_size)};
    } else {
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
    }
    QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
    QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};

    Qnn_Tensor_t tensor_inputs[] = {
            *p_tensor0,
            *p_tensor1
    };
    Qnn_Tensor_t tensor_outputs[] = {
            *p_tensor2
    };
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                        tensor_inputs, 2,
                                                        tensor_outputs, 1,
                                                        nullptr, nullptr));
    op_perf.info();
}

void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(value);
}

void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    ggml_qnn_dup(ctx, dst);
}

void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}
