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
    enum ggml_status result                     = GGML_STATUS_SUCCESS;
    bool graph_initialized                      = false;
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Param_t qnn_params[]                    = {};
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

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t & graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t & tensor = std::get<1>(graph_item);
        p_tensor0     = tensor[0];
        p_tensor1     = tensor[1];
        p_tensor2     = tensor[2];
    } else {
        p_tensor0 = ggmlqnn_create_compute_tensor(src0);
        p_tensor1 = ggmlqnn_create_compute_tensor(src1);
        p_tensor2 = ggmlqnn_create_compute_tensor(dst);
    }
    //ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == QNN_BACKEND_NPU;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        if (enable_npu_rpc) {
            QNN_VER_PTR(*p_tensor0)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor0)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor1)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor1)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor2)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor2)->clientBuf = {.data=nullptr, .dataSize=0};
        }

        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer_0 = ggmlqnn_create_rpc_buffer(instance, src0, p_tensor0, true);
            uint8_t * qnn_rpcbuffer_1 = ggmlqnn_create_rpc_buffer(instance, src1, p_tensor1, true);
            uint8_t * qnn_rpcbuffer_2 = ggmlqnn_create_rpc_buffer(instance, dst, p_tensor2, false);
            if (nullptr == qnn_rpcbuffer_0 || nullptr == qnn_rpcbuffer_1 || nullptr == qnn_rpcbuffer_2) {
                GGMLQNN_LOG_INFO("create rpc buffer failure\n");
                //TODO: potential memory leak although it shouldn't happen
                return;
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
        Qnn_OpConfig_t op_config = {
                QNN_OPCONFIG_VERSION_1, .v1 = {
                        ggml_op_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer = %p\n", qnn_rpcbuffer);
            if (nullptr != qnn_rpcbuffer) {
                memcpy(dst->data, qnn_rpcbuffer, ggml_nbytes(dst));
            }
        }

        qnn_tensors_t ggml_op_add_tensors;
        ggml_op_add_tensors.reserve(3);
        ggml_op_add_tensors.push_back(p_tensor0);
        ggml_op_add_tensors.push_back(p_tensor1);
        ggml_op_add_tensors.push_back(p_tensor2);

        auto  graph_item = std::make_tuple(graph_handle, ggml_op_add_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    } else {
        Qnn_DataType_t src0_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t src1_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t dst_qnn_type     = QNN_DATATYPE_FLOAT_32;

        src0_qnn_type                   = ggmlqnn_datatype_from_ggml_datatype(src0->type);
        src1_qnn_type                   = ggmlqnn_datatype_from_ggml_datatype(src1->type);
        dst_qnn_type                    = ggmlqnn_datatype_from_ggml_datatype(dst->type);

        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

        QNN_VER_PTR(*p_tensor0)->dimensions  = dimensions_input_0;
        QNN_VER_PTR(*p_tensor0)->rank        = ggml_n_dims(src0);
        QNN_VER_PTR(*p_tensor0)->dataType    = src0_qnn_type;

        QNN_VER_PTR(*p_tensor1)->dimensions  = dimensions_input_1;
        QNN_VER_PTR(*p_tensor1)->rank        = ggml_n_dims(src1);
        QNN_VER_PTR(*p_tensor1)->dataType    = src1_qnn_type;

        QNN_VER_PTR(*p_tensor2)->dimensions  = dimensions_output;
        QNN_VER_PTR(*p_tensor2)->rank        = ggml_n_dims(dst);
        QNN_VER_PTR(*p_tensor2)->dataType    = dst_qnn_type;

        if (enable_npu_rpc) {
            //TODO: NPU RPC feature will failed with test-backend-ops
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
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;

#if GGMLQNN_PRINT_OP_ADD_LOG
    op_perf.info();
#endif
}

//TODO:there is issue in this function
/*
 * this function is AI-assisted code from Grok 3 for purpose of 4d mulmat UT in ggml-qnn-ut.cpp
 * ./scripts/build-run-android.sh run_ut_mulmat 0
 * ./scripts/build-run-android.sh run_ut_mulmat 1
 * ./scripts/build-run-android.sh run_ut_mulmat 2
 *
 * the logic of ggml_qnn_mul_mat_4d is similar to ggml_qnn_mul_mat but much more complicated
 * than ggml_qnn_mul_mat, so it's a standalone function.
 * it will be combined with ggml_qnn_mul_mat after bugfix
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

    Qnn_GraphHandle_t graph_handle = nullptr;
    Qnn_Tensor_t *p_tensor0 = nullptr;
    Qnn_Tensor_t *p_gather0_out = nullptr;
    Qnn_Tensor_t *p_gather0_indices = nullptr;
    Qnn_Tensor_t *p_tensor1 = nullptr;
    Qnn_Tensor_t *p_gather1_out = nullptr;
    Qnn_Tensor_t *p_gather1_indices = nullptr;
    Qnn_Tensor_t *p_matmul_out = nullptr;
    Qnn_Tensor_t *p_transpose_perm = nullptr;
    Qnn_Tensor_t *p_tensor2 = nullptr;

    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst); // Keep debug line

    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t &graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t &tensors = std::get<1>(graph_item);
        p_tensor0 = tensors[0];
        p_gather0_out = tensors[1];
        p_gather0_indices = tensors[2];
        p_tensor1 = tensors[3];
        p_gather1_out = tensors[4];
        p_gather1_indices = tensors[5];
        p_matmul_out = tensors[6];
        p_transpose_perm = tensors[7];
        p_tensor2 = tensors[8];
    } else {
        CHECK_QNN_API(error, qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                                           graph_name.c_str(), NULL, &graph_handle));

        // Step 1: Define dimensions (ne = [K2, K1, M, B] for src0, [N, K, N1, B] for src1)
        uint32_t B = src0->ne[3];
        uint32_t M = src0->ne[2];
        uint32_t K0 = src0->ne[0] * src0->ne[1];
        uint32_t N1 = src1->ne[2];
        uint32_t K1 = src1->ne[1] * src1->ne[0];
        uint32_t N = src1->ne[0];

        GGML_ASSERT(src0->ne[3] == src1->ne[3]); // Matching batch
        GGML_ASSERT(dst->ne[2] == M);            // M matches dst
        GGML_ASSERT(K0 == K1);                   // K must match

        // src0: [K2, K1, M, B] -> QNN sees [B, M, K1, K2] after GQCGT reversal
        uint32_t src0_dims[] = {static_cast<uint32_t>(src0->ne[3]), static_cast<uint32_t>(src0->ne[2]), static_cast<uint32_t>(src0->ne[1]), static_cast<uint32_t>(src0->ne[0])};
        p_tensor0 = GQCGT(src0, "input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src0_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));

        // Gather on src0: [B, M, K1, K2] -> [M, B, K1, K2]
        uint32_t gather0_indices_data[] = {1, 0, 2, 3}; // [B, M, K1, K2] -> [M, B, K1, K2]
        uint32_t gather0_indices_dims[] = {4};
        p_gather0_indices = GQCGT(nullptr, "gather0_indices", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                  gather0_indices_dims, gather0_indices_data, sizeof(gather0_indices_data));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_gather0_indices));

        uint32_t gather0_out_dims[] = {M, B, static_cast<uint32_t>(src0->ne[1]), static_cast<uint32_t>(src0->ne[0])};
        p_gather0_out = GQCGT(nullptr, "gather0_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 4,
                              gather0_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_gather0_out));

        Qnn_Param_t gather0_params[] = {
                {QNN_PARAMTYPE_SCALAR, "axis", .scalarParam = {QNN_DATATYPE_INT_32, .int32Value = 0}}
        };
        Qnn_Tensor_t gather0_inputs[] = {*p_tensor0, *p_gather0_indices};
        Qnn_Tensor_t gather0_outputs[] = {*p_gather0_out};
        Qnn_OpConfig_t gather0_op = ggmlqnn_create_op_config("gather0", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                             QNN_OP_GATHER, gather0_params, 1,
                                                             gather0_inputs, 2, gather0_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, gather0_op));

        // src1: [N, K, N1, B] -> QNN sees [B, N1, K, N]
        uint32_t src1_dims[] = {static_cast<uint32_t>(src1->ne[3]), static_cast<uint32_t>(src1->ne[2]), static_cast<uint32_t>(src1->ne[1]), static_cast<uint32_t>(src1->ne[0])};
        p_tensor1 = GQCGT(src1, "input1", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, 4,
                          src1_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));

        // Gather on src1: [B, N1, K, N] -> [N1, B, K, N]
        uint32_t gather1_indices_data[] = {1, 0, 2, 3}; // [B, N1, K, N] -> [N1, B, K, N]
        uint32_t gather1_indices_dims[] = {4};
        p_gather1_indices = GQCGT(nullptr, "gather1_indices", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                  gather1_indices_dims, gather1_indices_data, sizeof(gather1_indices_data));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_gather1_indices));

        uint32_t gather1_out_dims[] = {N1, B, static_cast<uint32_t>(src1->ne[1]), static_cast<uint32_t>(src1->ne[0])};
        p_gather1_out = GQCGT(nullptr, "gather1_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 4,
                              gather1_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_gather1_out));

        Qnn_Param_t gather1_params[] = {
                {QNN_PARAMTYPE_SCALAR, "axis", .scalarParam = {QNN_DATATYPE_INT_32, .int32Value = 0}}
        };
        Qnn_Tensor_t gather1_inputs[] = {*p_tensor1, *p_gather1_indices};
        Qnn_Tensor_t gather1_outputs[] = {*p_gather1_out};
        Qnn_OpConfig_t gather1_op = ggmlqnn_create_op_config("gather1", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                             QNN_OP_GATHER, gather1_params, 1,
                                                             gather1_inputs, 2, gather1_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, gather1_op));

        // MatMul: [M, B * K0] x [B * K1, N]
        uint32_t matmul_in0_dims[] = {M, B * K0};
        Qnn_Tensor_t matmul_in0 = *p_gather0_out;
        QNN_VER_PTR(matmul_in0)->dimensions = matmul_in0_dims;
        QNN_VER_PTR(matmul_in0)->rank = 2;

        uint32_t matmul_in1_dims[] = {B * K1, N};
        Qnn_Tensor_t matmul_in1 = *p_gather1_out;
        QNN_VER_PTR(matmul_in1)->dimensions = matmul_in1_dims;
        QNN_VER_PTR(matmul_in1)->rank = 2;

        uint32_t matmul_out_dims[] = {M, N};
        p_matmul_out = GQCGT(nullptr, "matmul_out", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, 2,
                             matmul_out_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_matmul_out));

        Qnn_Tensor_t matmul_inputs[] = {matmul_in0, matmul_in1};
        Qnn_Tensor_t matmul_outputs[] = {*p_matmul_out};
        Qnn_OpConfig_t matmul_op = ggmlqnn_create_op_config("matmul", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                            QNN_OP_MAT_MUL, nullptr, 0,
                                                            matmul_inputs, 2, matmul_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, matmul_op));

        // Transpose: [M, N] -> Match dst->ne ([N, K, N1, M] reversed)
        uint32_t perm_data[] = {1, 0}; // [M, N] -> [N, M] for 2D
        uint32_t perm_dims[] = {2};
        p_transpose_perm = GQCGT(nullptr, "transpose_perm", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1,
                                 perm_dims, perm_data, sizeof(perm_data));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_transpose_perm));

        uint32_t dst_dims[] = {static_cast<uint32_t>(dst->ne[0]), static_cast<uint32_t>(dst->ne[1]), static_cast<uint32_t>(dst->ne[2]), static_cast<uint32_t>(dst->ne[3])};
        p_tensor2 = GQCGT(dst, "output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, 4,
                          dst_dims, nullptr, 0);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        Qnn_Param_t transpose_params[] = {
                {QNN_PARAMTYPE_TENSOR, "perm", .tensorParam = *p_transpose_perm}
        };
        Qnn_Tensor_t transpose_inputs[] = {*p_matmul_out};
        Qnn_Tensor_t transpose_outputs[] = {*p_tensor2};
        Qnn_OpConfig_t transpose_op = ggmlqnn_create_op_config("transpose", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                               QNN_OP_TRANSPOSE, transpose_params, 1,
                                                               transpose_inputs, 1, transpose_outputs, 1);
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, transpose_op));

        // Finalize
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, NULL, NULL));

        // Cache
        qnn_tensors_t ggml_op_mulmat_tensors = {p_tensor0, p_gather0_out, p_gather0_indices, p_tensor1,
                                                p_gather1_out, p_gather1_indices, p_matmul_out,
                                                p_transpose_perm, p_tensor2};
        instance->_qnn_graph_map[graph_name] = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
    }

    // Save dimensions
    uint32_t *tensor_0_dims = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t *gather0_out_dims = QNN_VER_PTR(*p_gather0_out)->dimensions;
    uint32_t *gather0_indices_dims = QNN_VER_PTR(*p_gather0_indices)->dimensions;
    uint32_t *tensor_1_dims = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t *gather1_out_dims = QNN_VER_PTR(*p_gather1_out)->dimensions;
    uint32_t *gather1_indices_dims = QNN_VER_PTR(*p_gather1_indices)->dimensions;
    uint32_t *matmul_out_dims = QNN_VER_PTR(*p_matmul_out)->dimensions;
    uint32_t *transpose_perm_dims = QNN_VER_PTR(*p_transpose_perm)->dimensions;
    uint32_t *tensor_2_dims = QNN_VER_PTR(*p_tensor2)->dimensions;

    // Execute
    QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
    QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
    QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

    Qnn_Tensor_t input_tensors[] = {*p_tensor0, *p_tensor1};
    Qnn_Tensor_t output_tensors[] = {*p_tensor2};
    CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle, input_tensors, 2,
                                                        output_tensors, 1, NULL, NULL));

    // Restore dimensions
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dims;
    QNN_VER_PTR(*p_gather0_out)->dimensions = gather0_out_dims;
    QNN_VER_PTR(*p_gather0_indices)->dimensions = gather0_indices_dims;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dims;
    QNN_VER_PTR(*p_gather1_out)->dimensions = gather1_out_dims;
    QNN_VER_PTR(*p_gather1_indices)->dimensions = gather1_indices_dims;
    QNN_VER_PTR(*p_matmul_out)->dimensions = matmul_out_dims;
    QNN_VER_PTR(*p_transpose_perm)->dimensions = transpose_perm_dims;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dims;

    op_perf.info();
}

/*
 * @brief performs matrix multiplication with FP32 & quantized weights and floating-point inputs
 *        using the QNN backend. this function performs matrix multiplication of the input tensor
 *        `src1` and the weight tensor `src0`, handling transposing, and quantization as needed,
 *        and stores the result in the destination tensor `dst`.
 *
 * @param backend the context which got through (ggml_backend_qnn_context *)backend->context for the
 *                QNN backend operations.
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
    bool graph_initialized                      = false;
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
    //GGML_ASSERT(src0_rank != 4); //TODO: 4D matrix mulmat
    if (4 == src0_rank) {
        return ggml_qnn_mul_mat_4d(ctx, op);
    }
    void * wdata                                = ggmlqnn_type_trait(ctx, op);
    const size_t desired_size                   = ctx->desired_size;

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized       = true;
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensors = std::get<1>(graph_item);
        p_tensor0               = tensors[0];
        p_tensor1               = tensors[1];
        p_tensor2               = tensors[2];
        p_param_tensor          = tensors[3];
        p_tensor2_transpose     = tensors[4];
    } else {
        p_tensor0 = GQCGT(src0, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor1 = GQCGT(src1, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor2 = GQCGT(dst, nullptr, QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
    }
    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        /*
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
             operation when offloading mulmat to QNN backend. this concise implementation will handle
             transpose in func ggml_qnn_create_general_tensor()
        */
        //step-1: create qnn graph
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                              graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        //step-2: create param tensor for mulmat of 2d/3d/4d matrix
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1]   = {src0_rank};
        p_param_tensor = GQCGT(nullptr, "param", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1, param_tensor_dims, (void *)(param_tensor_data[src0_rank - 1]), src0_rank * sizeof(uint32_t));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_param_tensor));

        //step-3: create compute tensor from ggml tensor
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));
        if (src0_type != GGML_TYPE_F32) {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {wdata, static_cast<uint32_t>(desired_size)};
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, ggmlqnn_get_tensor_data_size(src0)};
        }
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, ggmlqnn_get_tensor_data_size(src1)};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, ggmlqnn_get_tensor_data_size(dst)};

        //step-4: create a transpose tensor
        p_tensor2_transpose = GQCGT(dst, "transpose", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0, true);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2_transpose));

        //step-5: compose qnn graph: add mat_mul node
        Qnn_Param_t out_0_params[] = {
                {QNN_PARAMTYPE_SCALAR,
                 QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                        .scalarParam = {QNN_DATATYPE_BOOL_8, .bool8Value = 1}
                }
        };

        Qnn_Tensor_t out_0_inputs[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
#if 0 //leave here for easily understand code, can be removed in the future
        Qnn_OpConfig_t out_0 = {
                QNN_OPCONFIG_VERSION_1, .v1 =
                        {"ggmlqnn_mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                         1,
                         out_0_params,
                         2,
                         out_0_inputs,
                         1,
                         out_0_outputs}
        };
#else
        Qnn_OpConfig_t out_0 = ggmlqnn_create_op_config("ggmlqnn_mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                out_0_params, 1, out_0_inputs, 2, out_0_outputs, 1);
#endif
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_0));

        //step-5: compose qnn graph: add transpose node
        Qnn_Param_t out_trans1_0_params[] = {
                {(Qnn_ParamType_t) 1,
                 "perm", .tensorParam = *p_param_tensor
                }
        };
        Qnn_Tensor_t out_trans1_0_inputs[]  = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
#if 0 //leave here for easily understand code, can be removed in the future
        Qnn_OpConfig_t out_trans1_0 = {
                QNN_OPCONFIG_VERSION_1,
                .v1 =  {"ggmlqnn_mulmat_transpose_opconfig",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_TRANSPOSE, 1,
                        out_trans1_0_params,
                        1,
                        out_trans1_0_inputs,
                        1,
                        out_trans1_0_outputs}
        };
#else
        Qnn_OpConfig_t out_trans1_0 = ggmlqnn_create_op_config("ggmlqnn_mulmat_transpose_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                                       out_trans1_0_params, 1, out_trans1_0_inputs, 1, out_trans1_0_outputs, 1);
#endif
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_trans1_0));

        //step-6: finalize qnn graph and execute qnn graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));
        Qnn_Tensor_t input_tensors_0[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t output_tensors_0[] = {*p_tensor2};
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            input_tensors_0, 2,
                                                            output_tensors_0, 1,
                                                            nullptr, nullptr));

        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    } else {
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
        // this is the second technical approach or another pipeline of "how to utilize the Hexagon
        // NPU maximally" through QNN SDK, details could be found at
        // https://github.com/ggml-org/llama.cpp/pull/12049#issuecomment-2678308360
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;
    op_perf.info();
}

void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}


void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value) {
}

void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    ggml_qnn_dup(ctx, dst);
}

void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}
