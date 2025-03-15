
#include "graph.hpp"

#include <algorithm>
#include <unordered_map>

#include "ggml-impl.h"
#include "logger.hpp"
#include "op-config.hpp"
#include "tensor.hpp"

namespace {
using qnn_tensor_cache_t = std::unordered_map<ggml_tensor *, qnn::qnn_tensor_ptr_t>;

int get_op_max_rank(const ggml_tensor * op) {
    int       max_rank = ggml_n_dims(op);
    const int count    = (int) qnn::get_qnn_op_input_param_count(op);
    for (int i = 0; i < count; ++i) {
        max_rank = std::max(max_rank, ggml_n_dims(op->src[i]));
    }

    return max_rank;
}

qnn::qnn_tensor_ptr_t create_tensor_with_cache(ggml_tensor * tensor, qnn::ggml_qnn_tensor::tensor_type_t type, int rank,
                                               QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                               std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                               qnn_tensor_cache_t &               tensor_cache) {
    GGML_ASSERT(tensor);
    if (tensor_cache.count(tensor)) {
        return tensor_cache[tensor];
    }

    auto qnn_tensor = std::make_shared<qnn::ggml_qnn_tensor>(type, tensor->name, tensor->ne, tensor->type, rank, device,
                                                             graph_handle, qnn_instance);
    tensor_cache[tensor] = qnn_tensor;
    return qnn_tensor;
}

qnn::qnn_tensor_array_t create_tensors_with_cache(const qnn::ggml_tensor_array_t &    ggml_tensors,
                                                  qnn::ggml_qnn_tensor::tensor_type_t type, int rank, QNNBackend device,
                                                  Qnn_GraphHandle_t                  graph_handle,
                                                  std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                  qnn_tensor_cache_t &               tensor_cache) {
    qnn::qnn_tensor_array_t tensors;
    for (auto * tensor : ggml_tensors) {
        tensors.push_back(
            create_tensor_with_cache(tensor, type, rank, device, graph_handle, qnn_instance, tensor_cache));
    }

    return tensors;
}

qnn::qnn_op_config_ptr_t create_operation_from_op_tensor(ggml_tensor * dst, const std::string & name, int rank,
                                                         QNNBackend device, Qnn_GraphHandle_t graph_handle,
                                                         std::shared_ptr<qnn::qnn_instance> qnn_instance,
                                                         bool is_intermediate, qnn_tensor_cache_t & tensor_cache) {
    auto operation = qnn::create_op(dst, name, qnn_instance);

    // input tensors
    qnn::qnn_tensor_array_t input_qnn_tensors;
    auto tensor_type = is_intermediate ? qnn::ggml_qnn_tensor::INTERMEDIATE : qnn::ggml_qnn_tensor::INPUT;
    for (size_t i = 0; i < qnn::get_qnn_op_input_param_count(dst); ++i) {
        auto input_qnn_tensor =
            create_tensor_with_cache(dst->src[i], tensor_type, rank, device, graph_handle, qnn_instance, tensor_cache);
        input_qnn_tensors.push_back(input_qnn_tensor);
    }
    operation->set_input_tensors(input_qnn_tensors);

    // output tensor
    tensor_type = is_intermediate ? qnn::ggml_qnn_tensor::INTERMEDIATE : qnn::ggml_qnn_tensor::OUTPUT;
    qnn::qnn_tensor_array_t output_qnn_tensors =
        create_tensors_with_cache({ dst }, tensor_type, rank, device, graph_handle, qnn_instance, tensor_cache);
    operation->set_output_tensors(output_qnn_tensors);

    // initialize operation
    if (!operation->initialize_op_nodes(device, graph_handle)) {
        QNN_LOG_ERROR("[%s][%s]initialize_op_nodes failed\n", qnn::get_backend_name(device), name.c_str());
        return nullptr;
    }

    return operation;
}

bool bind_src_tensors(ggml_tensor * op, qnn::qnn_tensor_array_t & tensor_wrappers,
                      std::vector<Qnn_Tensor_t> & qnn_tensors) {
    if (op->op == GGML_OP_NONE) {
        QNN_LOG_DEBUG("op %s is not a valid op\n", ggml_get_name(op));
        return false;
    }

    const auto param_count = qnn::get_qnn_op_input_param_count(op);
    GGML_ASSERT(tensor_wrappers.size() == param_count);
    qnn_tensors.resize(param_count);
    for (size_t i = 0; i < param_count; ++i) {
        auto * ggml_tensor = op->src[i];
        if (!tensor_wrappers[i]->bind_ggml_tensor(ggml_tensor)) {
            QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
            return false;
        }

        qnn_tensors[i] = tensor_wrappers[i]->get_qnn_tensor();
    }

    return true;
}

/**
 * @brief Extracts input and output tensors from a computational graph.
 *
 * This function identifies the input and output tensors of a computational graph by analyzing the connectivity between
 * tensor nodes. It does this by iterating over each node in the graph, using a connectivity map that associates every
 * tensor with its number of incoming connections (in_degree), outgoing connections (out_degree), and an insertion index
 * that preserves order. The insertion index is used later to sort the tensors in their original discovery order.
 *
 * TODO: this algorithm is not perfect and may not work for all cases. It assumes that the tensors are
 *   connected in a way that allows for unambiguous categorization.
 */
int get_io_tensors_from_graph(const ggml_cgraph * cgraph, qnn::ggml_tensor_array_t & inputs,
                              qnn::ggml_tensor_array_t & outputs) {
    struct _tensor_connectivity_info {
        size_t in_degree    = 0;
        size_t out_degree   = 0;
        size_t insert_index = 0;
    };

    using ggml_tensor_connectivity_map_t = std::unordered_map<ggml_tensor *, _tensor_connectivity_info>;

    ggml_tensor_connectivity_map_t connectivity_map;
    int                            rank = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * dst = cgraph->nodes[i];
        if (ggml_is_empty(dst)) {
            continue;
        }

        if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW) {
            // TODO: remove GGML_OP_VIEW after view op is supported
            continue;
        }

        rank = std::max(rank, ggml_n_dims(dst));
        if (connectivity_map.count(dst) == 0) {
            connectivity_map[dst] = {
                1,  // in-degree, at least 1
                0,
                connectivity_map.size(),
            };
        } else {
            ++(connectivity_map[dst].in_degree);
        }

        for (size_t i = 0; i < GGML_MAX_DIMS && dst->src[i]; ++i) {
            auto * src = dst->src[i];
            rank       = std::max(rank, ggml_n_dims(src));

            if (connectivity_map.count(src) == 0) {
                connectivity_map[src] = {
                    0,
                    1,  // out-degree, at least 1
                    connectivity_map.size(),
                };
            } else {
                ++(connectivity_map[src].out_degree);
            }
        }
    }

    for (const auto & kv : connectivity_map) {
        if (kv.second.in_degree == 0) {
            inputs.push_back(kv.first);
        }

        if (kv.second.out_degree == 0) {
            outputs.push_back(kv.first);
        }
    }

    std::sort(inputs.begin(), inputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    std::sort(outputs.begin(), outputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    return rank;
}

}  // namespace

namespace qnn {

qnn_graph::qnn_graph(const std::string & graph_name, QNNBackend device, std::shared_ptr<qnn_instance> qnn_instance,
                     size_t vtcm_size_in_mb) :
    _graph_name(graph_name),
    _device(device),
    _qnn_instance(qnn_instance) {
    QNN_LOG_DEBUG("[%s][%s]created\n", get_backend_name(device), graph_name.c_str());

    auto              qnn_interface = qnn_instance->get_qnn_interface();
    auto              qnn_context   = qnn_instance->get_qnn_context_handle();
    Qnn_ErrorHandle_t error         = QNN_SUCCESS;
    Qnn_GraphHandle_t graph_handle  = nullptr;
    if (device == QNN_BACKEND_NPU) {
        // TODO: fix graph config here for NPU
        QnnHtpGraph_CustomConfig_t hvx_config;
        hvx_config.option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
        hvx_config.numHvxThreads = 8;
        QnnGraph_Config_t graph_hvx_config;
        graph_hvx_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_hvx_config.customConfig = &hvx_config;

        QnnHtpGraph_CustomConfig_t dlbc_config;
        dlbc_config.option                        = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        dlbc_config.optimizationOption.type       = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
        dlbc_config.optimizationOption.floatValue = 1.0;  // set to 0.0 to turn off DLBC
        QnnGraph_Config_t graph_dlbc_config;
        graph_dlbc_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_dlbc_config.customConfig = &dlbc_config;

        QnnHtpGraph_CustomConfig_t opt_config;
        opt_config.optimizationOption.type       = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        opt_config.optimizationOption.floatValue = 1;  // 1 / 3
        QnnGraph_Config_t graph_opt_config;
        graph_opt_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_opt_config.customConfig = &opt_config;

        QnnHtpGraph_CustomConfig_t vtcm_config;
        vtcm_config.option       = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcm_config.vtcmSizeInMB = (uint32_t) vtcm_size_in_mb;
        QnnGraph_Config_t graph_vtcm_config;
        graph_vtcm_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        graph_vtcm_config.customConfig = &vtcm_config;

        const QnnGraph_Config_t * graph_configs[] = { &graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                      &graph_opt_config, nullptr };
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), graph_configs, &graph_handle);
    } else {
        error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
    }

    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]failed to create qnn graph, error: %s\n", get_backend_name(device), graph_name.c_str(),
                      get_qnn_error_string(error));
        return;
    }

    QNN_LOG_DEBUG("[%s][%s]create succeed\n", get_backend_name(device), graph_name.c_str());
    _graph_handle  = graph_handle;
    _qnn_interface = qnn_interface;
}

qnn_graph::~qnn_graph() {
    QNN_LOG_DEBUG("[%s][%s]destroy\n", get_backend_name(_device), _graph_name.c_str());
}

bool qnn_graph::build_graph_from_ggml_graph(const ggml_cgraph * cgraph) {
    QNN_LOG_DEBUG("[%s][%s]build start\n", get_backend_name(_device), _graph_name.c_str());

    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
    int                 rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d\n", get_backend_name(_device), rank, int(inputs.size()),
                  int(outputs.size()));

    {
        qnn_tensor_cache_t tensor_cache;
        auto input_tensors  = create_tensors_with_cache(inputs, ggml_qnn_tensor::INPUT, rank, _device, _graph_handle,
                                                        _qnn_instance, tensor_cache);
        auto output_tensors = create_tensors_with_cache(outputs, ggml_qnn_tensor::OUTPUT, rank, _device, _graph_handle,
                                                        _qnn_instance, tensor_cache);
        qnn_op_config_array_t operations;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * dst = cgraph->nodes[i];
            if (ggml_is_empty(dst)) {
                continue;
            }

            if (dst->op == GGML_OP_NONE || dst->op == GGML_OP_VIEW) {
                // TODO: remove GGML_OP_VIEW after view op is supported
                continue;
            }

            QNN_LOG_DEBUG("[%s]create op: %s\n", get_backend_name(_device), get_qnn_op_name(dst));
            auto operation = create_operation_from_op_tensor(dst, dst->name, rank, _device, _graph_handle,
                                                             _qnn_instance, true, tensor_cache);  // TODO: fix op name
            operations.push_back(operation);
        }

        _tensor_inputs  = std::move(input_tensors);
        _tensor_outputs = std::move(output_tensors);
        _operations     = std::move(operations);
        if (!finalize()) {
            return false;
        }
    }

    QNN_LOG_DEBUG("[%s][%s]build succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

bool qnn_graph::execute(const ggml_cgraph * cgraph) {
    ggml_tensor_array_t inputs;
    ggml_tensor_array_t outputs;
#ifdef NDEBUG
    get_io_tensors_from_graph(cgraph, inputs, outputs);
#else
    int rank = get_io_tensors_from_graph(cgraph, inputs, outputs);
    QNN_LOG_DEBUG("[%s]rank: %d, input_set: %d, output_set: %d\n", get_backend_name(_device), rank, int(inputs.size()),
                  int(outputs.size()));
#endif

    {
        if (!qnn::bind_tensors(inputs, _tensor_inputs, _qnn_tensor_inputs)) {
            QNN_LOG_ERROR("[%s][%s]bind input tensors failed\n", get_backend_name(_device), _graph_name.c_str());
            return false;
        }

        if (!qnn::bind_tensors(outputs, _tensor_outputs, _qnn_tensor_outputs)) {
            QNN_LOG_ERROR("[%s][%s]bind output tensors failed\n", get_backend_name(_device), _graph_name.c_str());
            return false;
        }

        auto & qnn_tensor_inputs  = _qnn_tensor_inputs;
        auto & qnn_tensor_outputs = _qnn_tensor_outputs;
        auto   error =
            _qnn_interface->qnn_graph_execute(_graph_handle, qnn_tensor_inputs.data(), qnn_tensor_inputs.size(),
                                              qnn_tensor_outputs.data(), qnn_tensor_outputs.size(), nullptr, nullptr);
        unbind_tensors(_tensor_inputs);
        unbind_tensors(_tensor_outputs);

        if (error != QNN_SUCCESS) {
            if (_device == QNN_BACKEND_NPU && error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
                QNN_LOG_WARN("[%s][%s]NPU crashed. SSR detected. Caused QNN graph execute error.\n",
                             get_backend_name(_device), _graph_name.c_str());
            } else {
                QNN_LOG_ERROR("[%s][%s]error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                              get_qnn_error_string(error));
            }
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]execute succeed\n", get_backend_name(_device), _graph_name.c_str());
        return true;
    }
}

bool qnn_graph::finalize() {
    if (!qnn::add_op_to_graph(_graph_handle, _operations)) {
        QNN_LOG_ERROR("[%s]add nodes failed\n", _graph_name.c_str());
        return false;
    }

    auto error = _qnn_interface->qnn_graph_finalize(_graph_handle, nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        QNN_LOG_ERROR("[%s][%s]qnn_graph_finalize.error: %s\n", get_backend_name(_device), _graph_name.c_str(),
                      get_qnn_error_string(error));
        return false;
    }

    QNN_LOG_DEBUG("[%s][%s]finalize succeed\n", get_backend_name(_device), _graph_name.c_str());
    return true;
}

}  // namespace qnn
