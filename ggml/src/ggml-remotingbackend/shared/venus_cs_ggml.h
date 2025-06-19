// needs the ggml-backend-impl.h definition
// needs venus_cs.h definition

#include "venus_cs_ggml-rpc.h"

// needs
// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

static inline void
vn_encode_ggml_buffer_host_handle(struct vn_cs_encoder *enc, const apir_buffer_host_handle_t *handle);

static inline ggml_backend_buffer_t
vn_decode_ggml_buffer(struct vn_cs_decoder *dec);

/* rpc_tensor */

static inline void
vn_encode_rcp_tensor(struct vn_cs_encoder *enc, const rpc_tensor *rpc_tensor) {
  size_t rpc_tensor_size = sizeof(*rpc_tensor);
  vn_encode(enc, rpc_tensor_size, rpc_tensor, rpc_tensor_size);
}

static inline rpc_tensor *
vn_decode_rpc_tensor_inplace(struct vn_cs_decoder *dec) {
  size_t rpc_tensor_size = sizeof(rpc_tensor);

  return (rpc_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, rpc_tensor_size);
}

static inline rpc_tensor *
vn_decode_rpc_tensor_array_inplace(struct vn_cs_decoder *dec, uint32_t n_tensors) {
  size_t rpc_tensor_size = sizeof(rpc_tensor) * n_tensors;

  return (rpc_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, rpc_tensor_size);
}

/* ggml_tensor */

static inline void
vn_encode_ggml_tensor(struct vn_cs_encoder *enc, const ggml_tensor *tensor) {
  rpc_tensor serialized = serialize_tensor(tensor);

  vn_encode_rcp_tensor(enc, &serialized);
}

static inline const ggml_tensor *
vn_decode_ggml_tensor(struct vn_cs_decoder *dec) {
  const rpc_tensor *rpc_tensor = vn_decode_rpc_tensor_inplace(dec);
  struct ggml_init_params params {
    /*.mem_size   =*/ ggml_tensor_overhead(),
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
  };
  struct ggml_context * ctx = ggml_init(params);

  const ggml_tensor *tensor = deserialize_tensor(ctx, rpc_tensor);

  return tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.


static inline void
vn_encode_ggml_buffer_type(struct vn_cs_encoder *enc, ggml_backend_buffer_type_t buft) {
  apir_buffer_type_host_handle_t handle = ggml_buffer_type_to_apir_handle(buft);
  vn_cs_encoder_write(enc, sizeof(handle), &handle, sizeof(handle));
}

static inline ggml_backend_buffer_type_t
vn_decode_ggml_buffer_type(struct vn_cs_decoder *dec) {
  apir_buffer_type_host_handle_t handle;

  vn_cs_decoder_read(dec, sizeof(handle), &handle, sizeof(handle));

  return (ggml_backend_buffer_type_t) handle;
}

static inline apir_buffer_type_host_handle_t
vn_decode_apir_buffer_type_host_handle(struct vn_cs_decoder *dec) {
  apir_buffer_type_host_handle_t handle;

  vn_cs_decoder_read(dec, sizeof(handle), &handle, sizeof(handle));

  return handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

static inline void
vn_encode_ggml_buffer(struct vn_cs_encoder *enc, const ggml_backend_buffer_t buffer) {
  apir_buffer_host_handle_t handle = BUFFER_TO_HOST_HANDLE(buffer);
  vn_cs_encoder_write(enc, sizeof(handle), &handle, sizeof(handle));
}

static inline ggml_backend_buffer_t
vn_decode_ggml_buffer(struct vn_cs_decoder *dec) {
  ggml_backend_buffer_t buffer;
  size_t buffer_ptr_size = sizeof(buffer);

  vn_cs_decoder_read(dec, buffer_ptr_size, &buffer, buffer_ptr_size);

  return buffer;
}

/* enum ggml_status */

static inline void
vn_encode_ggml_status(struct vn_cs_encoder *enc, const enum ggml_status *status) {
  vn_cs_encoder_write(enc, sizeof(*status), status, sizeof(*status));
}

static inline void
vn_decode_ggml_status(struct vn_cs_decoder *dec, enum ggml_status *status) {
  vn_cs_decoder_read(dec, sizeof(*status), status, sizeof(*status));
}

/* vn_renderer_shmem */

static inline void
vn_encode_virtgpu_shmem_res_id(struct vn_cs_encoder *enc, uint32_t shmem_res_id) {
  vn_encode_uint32_t(enc, &shmem_res_id);
}

static inline void
vn_decode_virtgpu_shmem_res_id(struct vn_cs_decoder *dec, uint32_t *shmem_res_id) {
  vn_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

static inline size_t
vn_serialize_ggml_cgraph(ggml_cgraph *cgraph, std::vector<uint8_t> & cgraph_data) {
  serialize_graph(cgraph, cgraph_data);

  return cgraph_data.size();
}

static inline void
vn_encode_cgraph_data(struct vn_cs_encoder *enc, std::vector<uint8_t> & cgraph_data) {
  size_t cgraph_size = cgraph_data.size();

  vn_encode(enc, cgraph_size, cgraph_data.data(), cgraph_size);
}

static inline ggml_cgraph *
vn_decode_ggml_cgraph(struct vn_cs_decoder *dec, size_t cgraph_size) {
  UNUSED(cgraph_size);

  uint32_t n_nodes;
  vn_decode_uint32_t(dec, &n_nodes);
  const uint64_t * nodes = vn_decode_uint64_t_array_inplace(dec, n_nodes);

  uint32_t n_tensors;
  vn_decode_uint32_t(dec, &n_tensors);
  const rpc_tensor *tensors = vn_decode_rpc_tensor_array_inplace(dec, n_tensors);

  return deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

static inline void
vn_encode_ggml_buffer_handle(struct vn_cs_encoder *enc, const apir_buffer_host_handle_t *handle) {
  vn_cs_encoder_write(enc, sizeof(*handle), &handle, sizeof(*handle));
}

static inline void
vn_encode_ggml_tensor_inline(struct vn_cs_encoder *enc, const ggml_tensor *tensor) {
  size_t tensor_size = sizeof(*tensor);

  if (tensor->extra) {
    FATAL("Cannot pass tensors with extra");
  }

  if (tensor->src[0] && tensor->buffer) {
    static int first = 1;
    if (first) {
      // not sure if the buffer needs to be updated inside the src tensors or not
      WARNING("Cannot pass tensors with src and buffer");
      first = 0;
    }
  }

  vn_cs_encoder_write(enc, tensor_size, tensor, tensor_size);

  // tensor->data is a pointer inside the device buffer. No need to touch it
  // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
  // (could also make a copy of the tensor, and update locally.)

  if (tensor->buffer) {
    apir_buffer_host_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
    vn_encode_ggml_buffer_handle(enc, &buffer_handle);
  }

  if (tensor->view_src) {
    vn_cs_encoder_write(enc, tensor_size, tensor->view_src, tensor_size);
  }

  for (int i = 0; tensor->src[i]; i++) {
    const ggml_tensor *tensor_src = tensor->src[i];
    vn_cs_encoder_write(enc, tensor_size, tensor_src, tensor_size);
  }
}

static inline const ggml_tensor *
vn_decode_ggml_tensor_inplace(struct vn_cs_decoder *dec) {

  // it safe to remove the `const` qualifier here, we *do* want to
  // modify the shared memory data to fix the `src` pointers.
  ggml_tensor *tensor = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));

  // tensor->data is a pointer inside the device buffer. No need to touch it
  // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
  if (tensor->buffer) {
    tensor->buffer = vn_decode_ggml_buffer(dec);
  }

  if (tensor->view_src) {
    ggml_tensor *tensor_view_src = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));
    tensor->view_src = tensor_view_src;
  }

  for (int i = 0; tensor->src[i]; i++) {
    ggml_tensor *tensor_src = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));
    tensor->src[i] = tensor_src; // overwrite op->src[i] pointer with the actual location of the src tensor
  }

  return tensor;
}
