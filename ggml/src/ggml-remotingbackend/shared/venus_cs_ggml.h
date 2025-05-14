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

static inline void
vn_encode_ggml_buft(struct vn_cs_encoder *enc, ggml_backend_buffer_type_t buft) {
  size_t buft_ctx_size = sizeof(buft->context);

  vn_cs_encoder_write(enc, buft_ctx_size, &buft->context, buft_ctx_size);
}

static inline ggml_backend_buffer_type_t
vn_decode_ggml_buft(struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  size_t buft_size = sizeof(buft);

  vn_cs_decoder_read(dec, buft_size, &buft, buft_size);

  return buft;
}
