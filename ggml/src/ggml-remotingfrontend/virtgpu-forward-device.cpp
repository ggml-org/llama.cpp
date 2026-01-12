#include "virtgpu-forward-impl.h"
#include "virtgpu-shm.h"

int apir_device_get_count(struct virtgpu * gpu) {
    static int32_t dev_count = -1;
    if (dev_count != -1) {
        return dev_count;
    }

    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_COUNT);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_int32_t(decoder, &dev_count);

    remote_call_finish(gpu, encoder, decoder);

    return dev_count;
}

const char * apir_device_get_name(struct virtgpu * gpu) {
    static char * string = nullptr;
    if (string) {
        return string;
    }
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    const size_t string_size = apir_decode_array_size_unchecked(decoder);
    string                   = (char *) apir_decoder_alloc_array(sizeof(char), string_size);
    if (!string) {
        ERROR("%s: Could not allocate the device name buffer", __func__);
        apir_decoder_set_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    return string;
}

const char * apir_device_get_description(struct virtgpu * gpu) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    const size_t string_size = apir_decode_array_size_unchecked(decoder);
    char *       string      = (char *) apir_decoder_alloc_array(sizeof(char), string_size);
    if (!string) {
        ERROR("%s: Could not allocate the device description buffer", __func__);
        apir_decoder_set_fatal(decoder);

        return NULL;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    return string;
}

uint32_t apir_device_get_type(struct virtgpu * gpu) {
    static uint32_t dev_type = 255;
    if (dev_type != 255) {
        return dev_type;
    }

    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    return dev_type;
}

void apir_device_get_memory(struct virtgpu * gpu, size_t * free, size_t * total) {
    static size_t         dev_free  = 0;
    static size_t         dev_total = 0;
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_size_t(decoder, &dev_free);
    apir_decode_size_t(decoder, &dev_total);

    *free  = dev_free;
    *total = dev_total;

    remote_call_finish(gpu, encoder, decoder);

    return;
}

bool apir_device_supports_op(struct virtgpu * gpu, const ggml_tensor * op) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    bool supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    return supports_op;
}

apir_buffer_type_host_handle_t apir_device_get_buffer_type(struct virtgpu * gpu) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_buffer_type_host_handle_t buft_handle;
    apir_decode_apir_buffer_type_host_handle_t(decoder, &buft_handle);

    remote_call_finish(gpu, encoder, decoder);

    return buft_handle;
}

void apir_device_get_props(struct virtgpu * gpu,
                           bool *           async,
                           bool *           host_buffer,
                           bool *           buffer_from_host_ptr,
                           bool *           events) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_buffer);
    apir_decode_bool_t(decoder, buffer_from_host_ptr);
    apir_decode_bool_t(decoder, events);

    remote_call_finish(gpu, encoder, decoder);

    return;
}

apir_buffer_context_t apir_device_buffer_from_ptr(struct virtgpu * gpu, size_t size, size_t max_tensor_size) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

    if (virtgpu_shmem_create(gpu, size, &buffer_context.shmem)) {
        FATAL("Couldn't allocate the guest-host shared buffer :/");
    }

    apir_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem.res_id);

    apir_encode_size_t(encoder, &size);
    apir_encode_size_t(encoder, &max_tensor_size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
    buffer_context.buft_host_handle = apir_decode_apir_buffer_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    return buffer_context;
}
