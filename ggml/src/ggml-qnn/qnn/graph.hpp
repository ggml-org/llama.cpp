
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "convert.hpp"
#include "event_tracer.hpp"
#include "ggml-qnn.h"
#include "op-config.hpp"
#include "qnn-lib.hpp"

namespace qnn {

/**
 * @class qnn_graph
 * @brief Manages a QNN graph, converting a GGML graph to QNN format and handling its execution.
 *
 * This class is responsible for building a QNN graph from a given GGML graph,
 * determining its input/output tensors, finalizing the configuration, and
 * executing the graph on the specified backend device.
 */
class qnn_graph {
  public:
    enum htp_precision {
        kHtpDefault = 0,
        kHtpFp16,
    };

    /**
     * @brief Generates a unique key for a given computation graph (cgraph).
     *
     * This key is used to cache the graph, enabling efficient reuse of previously
     * compiled graphs. The key is constructed by concatenating the descriptions
     * of the operations and their associated tensor dimensions within the graph.
     *
     * Example key format: "MUL_MATf32_2048x8192q4_K_2048x2f32#MUL(SILU,MUL_MAT)#MUL_MAT(NONE,MUL)#ADD(MUL_MAT,ADD)f32_2048x2f32"
     *
     * @param cgraph The computation graph for which the key is generated.
     * @param output The string where the generated key will be stored.
     * @return The max ggml_type of all tensors in the graph.
     *
     * TODO: Improve the key generation logic to handle more complex graph structures and edge cases.
     */
    static ggml_type get_graph_key_from_cgraph(const ggml_cgraph * cgraph, std::string & output);

    explicit qnn_graph(const std::string & graph_name, backend_index_type device, qnn_instance_ptr qnn_instance,
                       htp_precision precision, size_t vtcm_size_in_mb);

    ~qnn_graph();

    bool build_graph_from_ggml_graph(const ggml_cgraph * cgraph);

    bool execute(const ggml_cgraph * cgraph, std::shared_ptr<qnn_convert_context_t> convert_context);

    bool is_valid() const { return _graph_handle != nullptr; }

    Qnn_GraphHandle_t get_graph_handler() const { return _graph_handle; }

    qnn_instance_ptr get_qnn_instance() { return _qnn_instance; }

    const std::string & get_name() const { return _graph_name; }

    backend_index_type get_device() const { return _device; }

  private:
    bool finalize();

    const std::string        _graph_name;
    const backend_index_type _device;
    Qnn_GraphHandle_t        _graph_handle = nullptr;
    qnn_instance_ptr         _qnn_instance;
    qnn_interface_ptr        _qnn_interface;
    qnn_op_config_array_t    _operations;

    qnn_tensor_array_t        _tensor_inputs;
    qnn_tensor_array_t        _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;

#ifdef GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
    // profiler
    qnn_event_tracer_ptr _event_tracer;
#endif

    DISABLE_COPY(qnn_graph);
    DISABLE_MOVE(qnn_graph);
};

using qnn_graph_ptr_t = std::shared_ptr<qnn_graph>;

}  // namespace qnn
