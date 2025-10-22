#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "op-config.hpp"
#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

class ggml_qnn_op_config_base : public ggml_qnn_op_config {
  public:
    explicit ggml_qnn_op_config_base(const std::string & name, const std::string & package_name,
                                     const std::string & op_type, qnn_instance_ptr qnn_instance) :
        _name(name),
        _package_name(package_name),
        _op_type(op_type),
        _qnn_instance(qnn_instance) {}

    void add_scalar_param(const std::string & name, const Qnn_Scalar_t scalar);
    bool add_tensor_param(const std::string & name, const qnn_dimension_array_t & dimensions, int rank,
                          const uint8_t * data, const Qnn_DataType_t data_type, backend_index_type device,
                          Qnn_GraphHandle_t graph_handle);

    void set_input_tensors(qnn::qnn_tensor_array_t & tensor_inputs) override;
    void set_input_tensors(qnn::qnn_tensor_array_t && tensor_inputs) override;
    void set_output_tensors(qnn::qnn_tensor_array_t & tensor_inputs) override;
    void set_output_tensors(qnn::qnn_tensor_array_t && tensor_inputs) override;
    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) override;
    bool bind_input_tensors(const ggml_tensor_array_t & tensor_inputs) override;
    bool bind_output_tensors(const ggml_tensor_array_t & tensor_outputs) override;
    void unbind_input_tensors() override;
    void unbind_output_tensors() override;

    qnn_tensor_array_t & get_input_tensors() override { return _tensor_inputs; }

    qnn_tensor_array_t & get_output_tensors() override { return _tensor_outputs; }

  protected:
    Qnn_OpConfig_t get_op_config();

    std::string               _name;
    std::string               _package_name;
    std::string               _op_type;
    qnn_instance_ptr          _qnn_instance;
    qnn_tensor_array_t        _tensor_inputs;
    qnn_tensor_array_t        _tensor_outputs;
    qnn_tensor_array_t        _tensor_parameters;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t>  _qnn_parameters;
    std::vector<std::string>  _param_names;

    DISABLE_COPY(ggml_qnn_op_config_base);
    DISABLE_MOVE(ggml_qnn_op_config_base);
};

class ggml_qnn_single_op_config : public ggml_qnn_op_config_base {
  public:
    explicit ggml_qnn_single_op_config(const std::string & name, const std::string & package_name,
                                       const std::string & op_type, qnn_instance_ptr qnn_instance) :
        ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance) {}

    bool initialize_op_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle) override;

  private:
    DISABLE_COPY(ggml_qnn_single_op_config);
    DISABLE_MOVE(ggml_qnn_single_op_config);
};

class ggml_qnn_rmsnorm_op_config : public ggml_qnn_op_config_base {
  public:
    explicit ggml_qnn_rmsnorm_op_config(const std::string & name, const std::string & package_name,
                                        const std::string & op_type, qnn_instance_ptr qnn_instance) :
        ggml_qnn_op_config_base(name, package_name, op_type, qnn_instance) {}

    bool initialize_op_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle) override;

  private:
    DISABLE_COPY(ggml_qnn_rmsnorm_op_config);
    DISABLE_MOVE(ggml_qnn_rmsnorm_op_config);
};

class ggml_qnn_aggregate_op_config : public ggml_qnn_op_config {
  public:
    explicit ggml_qnn_aggregate_op_config(const std::string & name, qnn_instance_ptr qnn_instance) :
        _name(name),
        _qnn_instance(qnn_instance) {}

    ~ggml_qnn_aggregate_op_config() {
        _tensor_inputs.clear();
        _tensor_outputs.clear();
        _operations.clear();
    }

    void set_input_tensors(qnn::qnn_tensor_array_t & tensor_inputs) override;
    void set_input_tensors(qnn::qnn_tensor_array_t && tensor_inputs) override;
    void set_output_tensors(qnn::qnn_tensor_array_t & tensor_inputs) override;
    void set_output_tensors(qnn::qnn_tensor_array_t && tensor_inputs) override;

    bool add_op_to_graph(Qnn_GraphHandle_t graph_handle) override {
        return qnn::add_op_to_graph(graph_handle, _operations);
    }

    bool bind_input_tensors(const ggml_tensor_array_t & tensor_inputs) override;
    bool bind_output_tensors(const ggml_tensor_array_t & tensor_outputs) override;

    void unbind_input_tensors() override {
        for (auto & tensor : _tensor_inputs) {
            tensor->unbind();
        }
    }

    void unbind_output_tensors() override {
        for (auto & tensor : _tensor_outputs) {
            tensor->unbind();
        }
    }

    qnn_tensor_array_t & get_input_tensors() override { return _tensor_inputs; }

    qnn_tensor_array_t & get_output_tensors() override { return _tensor_outputs; }

  protected:
    std::string      _name;
    qnn_instance_ptr _qnn_instance;

    std::vector<qnn_op_config_ptr_t> _operations;
    qnn_tensor_array_t               _tensor_inputs;
    qnn_tensor_array_t               _tensor_outputs;

  private:
    DISABLE_COPY(ggml_qnn_aggregate_op_config);
    DISABLE_MOVE(ggml_qnn_aggregate_op_config);
};

class ggml_qnn_matmul_op_config : public ggml_qnn_aggregate_op_config {
  public:
    ggml_qnn_matmul_op_config(const std::string & name, qnn_instance_ptr qnn_instance) :
        ggml_qnn_aggregate_op_config(name, qnn_instance) {}

    bool initialize_op_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle) override;

  private:
    qnn_tensor_ptr_t create_gather_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle, const int rank,
                                         qnn_tensor_ptr_t tensor_input, qnn_dimension_array_t output_dimensions);
    Qnn_DataType_t create_input_convert_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle, const int rank,
                                              qnn_tensor_array_t & tensor_inputs);
    qnn_op_config_ptr_t create_output_convert_nodes(backend_index_type device, Qnn_GraphHandle_t graph_handle,
                                                    const int rank, Qnn_DataType_t tensor_type,
                                                    qnn_tensor_array_t & tensor_outputs);
    bool                create_mat_mul_nodes(qnn_tensor_array_t & tensor_inputs, qnn_tensor_array_t & tensor_outputs);

    DISABLE_COPY(ggml_qnn_matmul_op_config);
    DISABLE_MOVE(ggml_qnn_matmul_op_config);
};

}  // namespace qnn
