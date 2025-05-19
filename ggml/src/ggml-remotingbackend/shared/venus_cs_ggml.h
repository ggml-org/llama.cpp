// needs the ggml-backend-impl.h definition
// needs venus_cs.h definition

// needs
// ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer);

static inline void
vn_encode_ggml_buffer_handle(struct vn_cs_encoder *enc, const apir_buffer_handle_t *handle);

static inline ggml_backend_buffer_t
vn_decode_ggml_buffer(struct vn_cs_decoder *dec);

static inline void
vn_encode_ggml_tensor(struct vn_cs_encoder *enc, const ggml_tensor *tensor) {
  size_t tensor_size = sizeof(*tensor);

  if (tensor->view_src) {
    FATAL("Cannot pass tensors with view_src");
  }
  if (tensor->extra) {
    FATAL("Cannot pass tensors with extra");
  }

  if (tensor->src[0] && tensor->buffer) {
    // not sure if the buffer needs to be updated inside the src tensors or not
    FATAL("Cannot pass tensors with src and buffer");
  }

  vn_cs_encoder_write(enc, tensor_size, tensor, tensor_size);

  // tensor->data is a pointer inside the device buffer. No need to touch it
  // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
  // (could also make a copy of the tensor, and update locally.)

  if (tensor->buffer) {
    apir_buffer_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
    vn_encode_ggml_buffer_handle(enc, &buffer_handle);
  }

  for (int i = 0; tensor->src[i]; i++) {
    const ggml_tensor *src_tensor = tensor->src[i];
    vn_cs_encoder_write(enc, tensor_size, src_tensor, tensor_size);
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

  for (int i = 0; tensor->src[i]; i++) {
    ggml_tensor *src_tensor = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));
    tensor->src[i] = src_tensor; // overwrite op->src[i] pointer with the actual location of the src tensor
  }

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
  vn_cs_encoder_write(enc, sizeof(*status), &status, sizeof(*status));
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
