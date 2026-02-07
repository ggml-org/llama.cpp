#include "virtgpu-forward-impl.h"

char * apir_buffer_type_get_name(virtgpu * gpu, apir_buffer_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME);

    apir_encode_apir_buffer_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    const size_t string_size = apir_decode_array_size_unchecked(decoder);
    char *       string      = (char *) apir_decoder_alloc_array(sizeof(char), string_size);
    if (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_set_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    return string;
}

size_t apir_buffer_type_get_alignment(virtgpu * gpu, apir_buffer_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT);

    apir_encode_apir_buffer_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_size_t(decoder, &alignment);

    remote_call_finish(gpu, encoder, decoder);

    return alignment;
}

size_t apir_buffer_type_get_max_size(virtgpu * gpu, apir_buffer_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE);

    apir_encode_apir_buffer_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_size_t(decoder, &max_size);

    remote_call_finish(gpu, encoder, decoder);

    return max_size;
}

apir_buffer_context_t apir_buffer_type_alloc_buffer(virtgpu * gpu, apir_buffer_type_host_handle_t host_handle, size_t size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER);

    apir_encode_apir_buffer_type_host_handle(encoder, host_handle);

    apir_encode_size_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);

    remote_call_finish(gpu, encoder, decoder);

    return buffer_context;
}

size_t apir_buffer_type_get_alloc_size(virtgpu * gpu, apir_buffer_type_host_handle_t host_handle, const ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE);

    apir_encode_apir_buffer_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_size;
    apir_decode_size_t(decoder, &alloc_size);

    remote_call_finish(gpu, encoder, decoder);

    return alloc_size;
}
