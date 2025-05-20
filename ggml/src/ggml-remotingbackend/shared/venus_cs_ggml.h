// needs the ggml-backend-impl.h definition
// needs venus_cs.h definition

// needs
// ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer);

static inline void
vn_encode_ggml_buffer_handle(struct vn_cs_encoder *enc, const apir_buffer_handle_t *handle);

static inline ggml_backend_buffer_t
vn_decode_ggml_buffer(struct vn_cs_decoder *dec);

/* ggml_tensor */

static inline size_t
vn_encode_sizeof_ggml_tensor(const ggml_tensor *tensor, int depth_to_go) {
  /* must match the encoding vn_encode_ggml_tensor */
  size_t size = 0;
  size_t tensor_size = sizeof(ggml_tensor);

  size += tensor_size; // the main tensor

  if (depth_to_go != 0) {
    if (tensor->view_src) {
      size += vn_encode_sizeof_ggml_tensor(tensor->view_src, depth_to_go-1);
    }

    for (int i = 0; tensor->src[i]; i++) {
      size += vn_encode_sizeof_ggml_tensor(tensor->src[i], depth_to_go-1);
    }
  }
  return size;
}

static inline void
vn_encode_ggml_tensor(struct vn_cs_encoder *enc, const ggml_tensor *tensor, int depth_to_go) {
  size_t tensor_size = sizeof(*tensor);

  if (tensor->extra) {
    FATAL("Cannot pass tensors with extra");
  }

  ggml_tensor *cs_tensor = (ggml_tensor *) vn_cs_encoder_write(enc, tensor_size, tensor, tensor_size);

  // tensor->data is a pointer inside the device buffer. No need to touch it
  // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
  // (could also make a copy of the tensor, and update locally.)

  if (tensor->buffer) {
    apir_buffer_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
    cs_tensor->buffer = (ggml_backend_buffer *) buffer_handle;
  }

  if (depth_to_go != 0) {
    if (tensor->view_src) {
      vn_encode_ggml_tensor(enc, tensor->view_src, depth_to_go-1);
    }

    for (int i = 0; tensor->src[i]; i++) {
      vn_encode_ggml_tensor(enc, tensor->src[i], depth_to_go-1);
    }
  }
}

static inline ggml_tensor *
vn_decode_ggml_tensor_inplace(struct vn_cs_decoder *dec, int depth_to_go) {

  // it safe to remove the `const` qualifier here, we *do* want to
  // modify the shared memory data to fix the `src` pointers.
  ggml_tensor *tensor = (ggml_tensor *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_tensor));

  // tensor->data is a pointer inside the device buffer. No need to touch it
  // tensor->buffer has already been updated to the correct pointer

  if (depth_to_go != 0) {
    if (tensor->view_src) {
      ggml_tensor *tensor_view_src = vn_decode_ggml_tensor_inplace(dec, depth_to_go-1);
      tensor->view_src = tensor_view_src;
    }

    for (int i = 0; tensor->src[i]; i++) {
      ggml_tensor *tensor_src_i = vn_decode_ggml_tensor_inplace(dec, depth_to_go-1);
      tensor->src[i] = tensor_src_i;
    }
  }

  return tensor;
}

/* tensor array */

static inline void
vn_encode_ggml_tensor_array(struct vn_cs_encoder *enc, ggml_tensor **addr, size_t count)
{
  size_t buffer_size = sizeof(*addr) * count;
  vn_encode(enc, buffer_size, addr, buffer_size);
}

static inline ggml_tensor **
vn_decode_ggml_tensor_array_inplace(struct vn_cs_decoder *dec, size_t count)
{
  size_t buffer_size = sizeof(ggml_tensor*) * count;

  return (ggml_tensor **)(uintptr_t) vn_cs_decoder_use_inplace(dec, buffer_size);
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

  return size;
}

static inline void
vn_encode_ggml_cgraph(struct vn_cs_encoder *enc, ggml_cgraph *cgraph, struct vn_cs_encoder *secondary_enc) {
  UNUSED(enc);
  UNUSED(cgraph);

  if (cgraph->n_leafs) {
    FATAL("Cannot pass cgraphs with leaves");
  }
  if (cgraph->size) {
    FATAL("Cannot pass cgraphs with size");
  }
  if (cgraph->grads) {
    FATAL("Cannot pass cgraphs with grads");
  }
  if (cgraph->grad_accs) {
    FATAL("Cannot pass cgraphs with grad_accs");
  }
  if (cgraph->visited_hash_set.size || cgraph->visited_hash_set.used || cgraph->visited_hash_set.keys) {
    FATAL("Cannot pass cgraphs with visited_hash_set");
  }

  size_t cgraph_size = sizeof(*cgraph);
  vn_cs_encoder_write(enc, cgraph_size, cgraph, cgraph_size);

  vn_encode_ggml_tensor_array(secondary_enc, cgraph->nodes, cgraph->n_nodes);

  for (int i = 0; i < cgraph->n_nodes; i++) {
    ggml_tensor *tensor = cgraph->nodes[i];
    vn_encode_ggml_tensor(secondary_enc, tensor, TENSOR_MAX_DEPTH_CGRAPH_DATA);
  }
}

static inline ggml_cgraph *
vn_decode_ggml_cgraph(struct vn_cs_decoder *dec, struct vn_cs_decoder *secondary_dec) {
  // it safe to remove the `const` qualifier here, we *do* want to
  // modify the shared memory data to fix the `src` pointers.
  ggml_cgraph *cgraph = (ggml_cgraph *)(uintptr_t) vn_cs_decoder_use_inplace(dec, sizeof(ggml_cgraph));

  cgraph->nodes = vn_decode_ggml_tensor_array_inplace(secondary_dec, cgraph->n_nodes);

  for (int i = 0; i < cgraph->n_nodes; i++) {
    cgraph->nodes[i] = (ggml_tensor *)(uintptr_t) vn_decode_ggml_tensor_inplace(secondary_dec, TENSOR_MAX_DEPTH_CGRAPH_DATA);
  }

  return cgraph;
}
