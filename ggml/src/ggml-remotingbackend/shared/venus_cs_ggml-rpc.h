#include <vector>
#include <unordered_set>
#include <unordered_map>

// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
  uint64_t id;
  uint32_t type;
  uint64_t buffer;
  uint32_t ne[GGML_MAX_DIMS];
  uint32_t nb[GGML_MAX_DIMS];
  uint32_t op;
  int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
  int32_t  flags;
  uint64_t src[GGML_MAX_SRC];
  uint64_t view_src;
  uint64_t view_offs;
  uint64_t data;
  char name[GGML_MAX_NAME];

  char padding[4];
};

/* frontend */

rpc_tensor serialize_tensor(const ggml_tensor * tensor);

void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output);

/* backend */

void track_backend_buffer(ggml_backend_buffer_t buffer);
bool untrack_backend_buffer(ggml_backend_buffer_t buffer);
std::unordered_set<ggml_backend_buffer_t> get_track_backend_buffers();

void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited);

ggml_tensor *deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);

ggml_tensor *create_node(uint64_t id,
			 struct ggml_context * ctx,
			 const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
			 std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);

ggml_cgraph *deserialize_graph(uint32_t n_nodes, uint32_t n_tensors, const rpc_tensor * tensors, const uint64_t * nodes);
