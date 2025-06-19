#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cinttypes>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "../ggml-remotingbackend/shared/venus_cs_ggml-rpc.h"

#include "ggml-remoting.h"

rpc_tensor
serialize_tensor(const ggml_tensor * tensor) {
  rpc_tensor result;
  result.id = reinterpret_cast<uint64_t>(tensor);
  result.type = tensor->type;
  if (tensor->buffer) {
    ggml_backend_buffer_t buffer = tensor->buffer;

    result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
  } else {
    result.buffer = 0;
  }
  for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
    result.ne[i] = tensor->ne[i];
    result.nb[i] = tensor->nb[i];
  }
  result.op = tensor->op;
  for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
    result.op_params[i] = tensor->op_params[i];
  }
  result.flags = tensor->flags;
  for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
    result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
  }
  result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
  result.view_offs = tensor->view_offs;
  result.data = reinterpret_cast<uint64_t>(tensor->data);
  if (tensor->data) {
    if (!tensor->buffer) {
      FATAL("tensor has data but not buffer :/");
    }
    // tensor->data is serialized as an offset to the buffer base address
    result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
  }
  snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
  return result;
}

void
add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
  if (tensor == nullptr) {
    return;
  }
  if (visited.find(tensor) != visited.end()) {
    return;
  }
  visited.insert(tensor);
  for (int i = 0; i < GGML_MAX_SRC; i++) {
    add_tensor(tensor->src[i], tensors, visited);
  }
  add_tensor(tensor->view_src, tensors, visited);
  tensors.push_back(serialize_tensor(tensor));
}

void
serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
  uint32_t n_nodes = cgraph->n_nodes;
  std::vector<rpc_tensor> tensors;
  std::unordered_set<ggml_tensor*> visited;
  for (uint32_t i = 0; i < n_nodes; i++) {
    add_tensor(cgraph->nodes[i], tensors, visited);
  }
  // serialization format:
  // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
  uint32_t n_tensors = tensors.size();
  int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
  output.resize(output_size, 0);
  memcpy(output.data(), &n_nodes, sizeof(n_nodes));
  for (uint32_t i = 0; i < n_nodes; i++) {
    memcpy(output.data() + sizeof(n_nodes) + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
  }
  uint32_t * out_ntensors = (uint32_t *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
  *out_ntensors = n_tensors;
  rpc_tensor * out_tensors = (rpc_tensor *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
  memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}
