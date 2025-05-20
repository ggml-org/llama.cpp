// needs the ggml-backend-impl.h definition
// needs venus_cs.h definition

#include "venus_cs_ggml-rpc.h"

// needs
// ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer);

static inline void
vn_encode_ggml_buffer_handle(struct vn_cs_encoder *enc, const apir_buffer_handle_t *handle);

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
vn_encode_apir_buffer_type_handle_t(struct vn_cs_encoder *enc, apir_buffer_type_handle_t *handle) {
  vn_cs_encoder_write(enc, sizeof(*handle), handle, sizeof(*handle));
}

static inline ggml_backend_buffer_type_t
vn_decode_ggml_buft(struct vn_cs_decoder *dec) {
  apir_buffer_type_handle_t handle;

  vn_cs_decoder_read(dec, sizeof(handle), &handle, sizeof(handle));

  return (ggml_backend_buffer_type_t) handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

static inline void
vn_encode_ggml_buffer_handle(struct vn_cs_encoder *enc, const apir_buffer_handle_t *handle) {
  vn_cs_encoder_write(enc, sizeof(*handle), &handle, sizeof(*handle));
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
vn_encode_sizeof_ggml_cgraph_data(ggml_cgraph *cgraph) {
  /* must match the encoding of vn_encode_ggml_cgraph and vn_encode_ggml_tensor */
  size_t size = 0;

  // don't include the `ggml_cgraph`, only it's data

  // include the array of tensors
  size += sizeof(ggml_tensor*) * cgraph->n_nodes;

  // include the size of all the tensors
  for (int i = 0; i < cgraph->n_nodes; i++) {
    size += vn_encode_sizeof_ggml_tensor(cgraph->nodes[i], TENSOR_MAX_DEPTH_CGRAPH_DATA);
  }
  INFO("SIZEOF(cgraph) --> %lu", size);
  return size;
}

static inline void
vn_encode_ggml_cgraph(struct vn_cs_encoder *enc, ggml_cgraph *cgraph, struct vn_cs_encoder *secondary_enc) {
  UNUSED(enc);
  UNUSED(cgraph);
  UNUSED(secondary_enc);
}

static inline ggml_cgraph *
vn_decode_ggml_cgraph(struct vn_cs_decoder *dec, struct vn_cs_decoder *secondary_dec) {
  // it safe to remove the `const` qualifier here, we *do* want to
  // modify the shared memory data to fix the `src` pointers.

  UNUSED(dec);
  UNUSED(secondary_dec);

  return NULL;
}
