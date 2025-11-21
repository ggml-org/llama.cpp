#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cinttypes>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "venus_cs_ggml-rpc.h"

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

void
track_backend_buffer(ggml_backend_buffer_t buffer) {
  backend_buffers.insert(buffer);
}

rpc_tensor
serialize_tensor(const ggml_tensor * tensor) {
  rpc_tensor result;
  result.id = reinterpret_cast<uint64_t>(tensor);
  result.type = tensor->type;
  if (tensor->buffer) {
    ggml_backend_buffer_t buffer = tensor->buffer;

    result.buffer = BUFFER_TO_HANDLE(buffer);
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
  snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
  return result;
}

ggml_tensor *
deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
  ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
                                            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
  for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = tensor->nb[i];
  }
  result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
  if (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
    printf("WARNING: BUFFER NOT FOUND | %p\n", (void *)result->buffer);
    result->buffer = nullptr;
  }

  if (result->buffer) {
    // require that the tensor data does not go beyond the buffer end
    uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
    uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
    uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
    GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
    GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
  }

  result->op = (ggml_op) tensor->op;
  for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
    result->op_params[i] = tensor->op_params[i];
  }
  result->flags = tensor->flags;
  result->data = reinterpret_cast<void *>(tensor->data);
  ggml_set_name(result, tensor->name);
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

ggml_tensor *
create_node(uint64_t id,
            struct ggml_context * ctx,
            const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
            std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
  if (id == 0) {
    return nullptr;
  }
  if (tensor_map.find(id) != tensor_map.end()) {
    return tensor_map[id];
  }
  const rpc_tensor * tensor = tensor_ptrs.at(id);
  struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
  if (result == nullptr) {
    return nullptr;
  }
  tensor_map[id] = result;
  for (int i = 0; i < GGML_MAX_SRC; i++) {
    result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
  }
  result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
  result->view_offs = tensor->view_offs;
  return result;
}

ggml_cgraph *
deserialize_graph(uint32_t n_nodes, uint32_t n_tensors, const rpc_tensor * tensors, const uint64_t * nodes) {
  size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
  struct ggml_init_params params = {
    /*.mem_size   =*/ buf_size,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
  };
  struct ggml_context * ctx = ggml_init(params);
  struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
  graph->n_nodes = n_nodes;
  std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
  for (uint32_t i = 0; i < n_tensors; i++) {
    tensor_ptrs[tensors[i].id] = &tensors[i];
  }
  std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
  for (uint32_t i = 0; i < n_nodes; i++) {
    int64_t id;
    memcpy(&id, &nodes[i], sizeof(id));
    graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
  }

  return graph;
}
