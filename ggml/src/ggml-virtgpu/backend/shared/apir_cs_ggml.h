#include "ggml-impl.h"
#include "apir_cs.h"
#include "apir_cs_rpc.h"

// ggml_buffer_to_apir_host_handle(ggml_backend_buffer_t buffer);

static inline void apir_encode_ggml_buffer_host_handle(apir_encoder *                    enc,
                                                       const apir_buffer_host_handle_t * handle);

static inline ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec);

/* apir_rpc_tensor */

static inline void apir_encode_rcp_tensor(apir_encoder * enc, const apir_rpc_tensor * apir_rpc_tensor) {
    size_t apir_rpc_tensor_size = sizeof(*apir_rpc_tensor);
    apir_encode(enc, apir_rpc_tensor_size, apir_rpc_tensor, apir_rpc_tensor_size);
}

static inline apir_rpc_tensor * apir_decode_apir_rpc_tensor_inplace(apir_decoder * dec) {
    size_t apir_rpc_tensor_size = sizeof(apir_rpc_tensor);

    return (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

static inline apir_rpc_tensor * apir_decode_apir_rpc_tensor_array_inplace(apir_decoder * dec,
                                                                          uint32_t       n_tensors) {
    size_t apir_rpc_tensor_size = sizeof(apir_rpc_tensor) * n_tensors;

    return (apir_rpc_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, apir_rpc_tensor_size);
}

/* ggml_tensor */

static inline void apir_encode_ggml_tensor(apir_encoder * enc, const ggml_tensor * tensor) {
    apir_rpc_tensor serialized = apir_serialize_tensor(tensor);

    apir_encode_rcp_tensor(enc, &serialized);
}

static inline const ggml_tensor * apir_decode_ggml_tensor(apir_decoder * dec) {
    const apir_rpc_tensor * apir_rpc_tensor = apir_decode_apir_rpc_tensor_inplace(dec);
    ggml_init_params params{
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    const ggml_tensor * tensor = apir_deserialize_tensor(ctx, apir_rpc_tensor);

    return tensor;
}

/* *** ggml_backend_buffer_type_t *** */

// ggml_backend_buffer_type_t is a POINTER (to a struct).
// Only the host pointer is shared between the host and guest.
// The guest stores it in `buft->context`.
// The host simply writes the pointer address in the buffer variable.

static inline void apir_encode_ggml_buffer_type(apir_encoder * enc, ggml_backend_buffer_type_t buft) {
    apir_buffer_type_host_handle_t handle = ggml_buffer_type_to_apir_handle(buft);
    apir_encoder_write(enc, sizeof(handle), &handle, sizeof(handle));
}

static inline ggml_backend_buffer_type_t apir_decode_ggml_buffer_type(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, sizeof(handle), &handle, sizeof(handle));

    return (ggml_backend_buffer_type_t) handle;
}

static inline apir_buffer_type_host_handle_t apir_decode_apir_buffer_type_host_handle(apir_decoder * dec) {
    apir_buffer_type_host_handle_t handle;

    apir_decoder_read(dec, sizeof(handle), &handle, sizeof(handle));

    return handle;
}

/* *** ggml_backend_type_t *** */

// ggml_backend_buffer_t is a POINTER.
// same logic as for ggml_backend_buffer_type_t

static inline void apir_encode_ggml_buffer(apir_encoder * enc, const ggml_backend_buffer_t buffer) {
    apir_buffer_host_handle_t handle = BUFFER_TO_HOST_HANDLE(buffer);
    apir_encoder_write(enc, sizeof(handle), &handle, sizeof(handle));
}

static inline ggml_backend_buffer_t apir_decode_ggml_buffer(apir_decoder * dec) {
    ggml_backend_buffer_t buffer;
    size_t                buffer_ptr_size = sizeof(buffer);

    apir_decoder_read(dec, buffer_ptr_size, &buffer, buffer_ptr_size);

    return buffer;
}

/* enum ggml_status */

static inline void apir_encode_ggml_status(apir_encoder * enc, const ggml_status * status) {
    apir_encoder_write(enc, sizeof(*status), status, sizeof(*status));
}

static inline void apir_decode_ggml_status(apir_decoder * dec, ggml_status * status) {
    apir_decoder_read(dec, sizeof(*status), status, sizeof(*status));
}

/* virtgpu_shmem */

static inline void apir_encode_virtgpu_shmem_res_id(apir_encoder * enc, uint32_t shmem_res_id) {
    apir_encode_uint32_t(enc, &shmem_res_id);
}

static inline void apir_decode_virtgpu_shmem_res_id(apir_decoder * dec, uint32_t * shmem_res_id) {
    apir_decode_uint32_t(dec, shmem_res_id);
}

/* ggml_cgraph */

static inline size_t apir_serialize_ggml_cgraph(ggml_cgraph * cgraph, std::vector<uint8_t> & cgraph_data) {
    apir_serialize_graph(cgraph, cgraph_data);

    return cgraph_data.size();
}

static inline void apir_encode_cgraph_data(apir_encoder * enc, std::vector<uint8_t> & cgraph_data) {
    size_t cgraph_size = cgraph_data.size();

    apir_encode(enc, cgraph_size, cgraph_data.data(), cgraph_size);
}

static inline ggml_cgraph * apir_decode_ggml_cgraph(apir_decoder * dec, size_t cgraph_size) {
    GGML_UNUSED(cgraph_size);

    uint32_t n_nodes;
    apir_decode_uint32_t(dec, &n_nodes);
    const uint64_t * nodes = apir_decode_uint64_t_array_inplace(dec, n_nodes);

    uint32_t n_tensors;
    apir_decode_uint32_t(dec, &n_tensors);
    const apir_rpc_tensor * tensors = apir_decode_apir_rpc_tensor_array_inplace(dec, n_tensors);

    return apir_deserialize_graph(n_nodes, n_tensors, tensors, nodes);
}

static inline void apir_encode_ggml_buffer_handle(apir_encoder * enc, const apir_buffer_host_handle_t * handle) {
    apir_encoder_write(enc, sizeof(*handle), &handle, sizeof(*handle));
}

static inline void apir_encode_ggml_tensor_inline(apir_encoder * enc, const ggml_tensor * tensor) {
    size_t tensor_size = sizeof(*tensor);

    if (tensor->extra) {
        GGML_ABORT("Cannot pass tensors with extra");
    }

    if (tensor->src[0] && tensor->buffer) {
        static int first = 1;
        if (first) {
            GGML_LOG_WARN("Cannot pass tensors with src and buffer\n");
            first = 0;
        }
    }

    apir_encoder_write(enc, tensor_size, tensor, tensor_size);

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Encoding the buffer handle in sequence.
    // (could also make a copy of the tensor, and update locally.)

    if (tensor->buffer) {
        apir_buffer_host_handle_t buffer_handle = ggml_buffer_to_apir_handle(tensor->buffer);
        apir_encode_ggml_buffer_handle(enc, &buffer_handle);
    }

    if (tensor->view_src) {
        apir_encoder_write(enc, tensor_size, tensor->view_src, tensor_size);
    }

    for (int i = 0; tensor->src[i]; i++) {
        const ggml_tensor * tensor_src = tensor->src[i];
        apir_encoder_write(enc, tensor_size, tensor_src, tensor_size);
    }
}

static inline const ggml_tensor * apir_decode_ggml_tensor_inplace(apir_decoder * dec) {
    // it safe to remove the `const` qualifier here, we *do* want to
    // modify the shared memory data to fix the `src` pointers.
    ggml_tensor * tensor = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, sizeof(ggml_tensor));

    // tensor->data is a pointer inside the device buffer. No need to touch it
    // tensor->buffer is a pointer to a buffer. Decode the buffer handle encoded in sequence.
    if (tensor->buffer) {
        tensor->buffer = apir_decode_ggml_buffer(dec);
    }

    if (tensor->view_src) {
        ggml_tensor * tensor_view_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, sizeof(ggml_tensor));
        tensor->view_src              = tensor_view_src;
    }

    for (int i = 0; tensor->src[i]; i++) {
        ggml_tensor * tensor_src = (ggml_tensor *) (uintptr_t) apir_decoder_use_inplace(dec, sizeof(ggml_tensor));
        tensor->src[i] = tensor_src;  // overwrite op->src[i] pointer with the actual location of the src tensor
    }

    return tensor;
}
