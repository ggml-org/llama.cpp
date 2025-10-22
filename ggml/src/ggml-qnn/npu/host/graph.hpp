#pragma once

#include <vector>

#include "common.hpp"
#include "ggml-backend-impl.h"
#include "hexagon_npu.h"

namespace hexagon {

class host_graph {
  public:
    host_graph(ggml_cgraph * cgraph, remote_handle64 device_handle);

    ~host_graph();

    bool is_valid() const { return _graph_handle != 0; }

    bool update(ggml_cgraph * cgraph);

    bool compute();

  private:
    remote_handle64                              _device_handle = 0;
    npu_device_graph_handle_t                    _graph_handle  = npu_device_INVALID_DEVICE_GRAPH_HANDLE;
    std::vector<npu_device_tensor_handle_t>      _tensor_handles;
    std::vector<npu_device_tensor_update_config> _tensor_update_configs;

    DISABLE_COPY(host_graph);
    DISABLE_MOVE(host_graph);
};

}  // namespace hexagon
