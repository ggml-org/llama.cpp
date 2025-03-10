#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "op-config-base.hpp"
#include "qnn-lib.hpp"
#include "qnn-types.hpp"
#include "tensor.hpp"

namespace qnn {

constexpr const size_t kGgmlUnaryOpStart = GGML_OP_COUNT;

size_t                              get_qnn_op_index(const ggml_tensor * tensor);
const char *                        get_qnn_op_name(const ggml_tensor * op);
size_t                              get_qnn_op_input_param_count(const ggml_tensor * op);
std::shared_ptr<ggml_qnn_op_config> create_op(const ggml_tensor * op, const std::string & name,
                                              std::shared_ptr<qnn_instance> qnn_instance);

inline bool add_op_to_graph(Qnn_GraphHandle_t graph_handle, std::vector<qnn_op_config_ptr_t> & operations) {
    for (auto & op : operations) {
        if (!op->add_op_to_graph(graph_handle)) {
            return false;
        }
    }

    return true;
}

}  // namespace qnn
