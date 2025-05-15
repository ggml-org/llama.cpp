// needs the ggml-backend-impl.h definition
// needs venus_cs.h definition

static inline void
vn_encode_ggml_tensor(struct vn_cs_encoder *enc, const ggml_tensor *op) {
  size_t tensor_size = sizeof(*op);

  if (op->buffer || op->data || op->view_src || op->extra) {
    FATAL("Cannot pass tensors with data");
  }

  vn_cs_encoder_write(enc, tensor_size, op, tensor_size);

  for (int i = 0; op->src[i]; i++) {
    const ggml_tensor *src_op = op->src[i];
    vn_cs_encoder_write(enc, tensor_size, src_op, tensor_size);
  }
}

static inline const ggml_tensor *
vn_decode_ggml_tensor_inplace(struct vn_cs_decoder *dec) {

  // it safe to remove the `const` qualifier here, we *do* want to
  // modify the shared memory data to fix the `src` pointers.
  ggml_tensor *op = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));


  for (int i = 0; op->src[i]; i++) {
    ggml_tensor *src_op = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));
    op->src[i] = src_op; // overwrite op->src[i] pointer with the actual location of the src tensor
  }

  return op;
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
