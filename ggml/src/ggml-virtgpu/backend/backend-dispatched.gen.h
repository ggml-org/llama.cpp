#pragma once

/* device */
uint32_t backend_device_get_device_count(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_count(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_description(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_memory(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_supports_op(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_buffer_type(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_get_props(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_device_buffer_from_ptr(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer-type */
uint32_t backend_buffer_type_get_name(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_type_get_alignment(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_type_get_max_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
/* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST is deprecated. Keeping the handler for backward compatibility. */
uint32_t backend_buffer_type_is_host(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_type_alloc_buffer(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_type_get_alloc_size(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* buffer */
uint32_t backend_buffer_get_base(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_set_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);
uint32_t backend_buffer_free_buffer(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

/* backend */
uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);

static inline const char * backend_dispatch_command_name(ApirBackendCommandType type) {
    switch (type) {
        /* device */
        case APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT:
            return "backend_device_get_device_count";
        case APIR_COMMAND_TYPE_DEVICE_GET_COUNT:
            return "backend_device_get_count";
        case APIR_COMMAND_TYPE_DEVICE_GET_NAME:
            return "backend_device_get_name";
        case APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION:
            return "backend_device_get_description";
        case APIR_COMMAND_TYPE_DEVICE_GET_TYPE:
            return "backend_device_get_type";
        case APIR_COMMAND_TYPE_DEVICE_GET_MEMORY:
            return "backend_device_get_memory";
        case APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP:
            return "backend_device_supports_op";
        case APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE:
            return "backend_device_get_buffer_type";
        case APIR_COMMAND_TYPE_DEVICE_GET_PROPS:
            return "backend_device_get_props";
        case APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR:
            return "backend_device_buffer_from_ptr";
        /* buffer-type */
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME:
            return "backend_buffer_type_get_name";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT:
            return "backend_buffer_type_get_alignment";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE:
            return "backend_buffer_type_get_max_size";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST:
            return "backend_buffer_type_is_host (DEPRECATED)";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER:
            return "backend_buffer_type_alloc_buffer";
        case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE:
            return "backend_buffer_type_get_alloc_size";
        /* buffer */
        case APIR_COMMAND_TYPE_BUFFER_GET_BASE:
            return "backend_buffer_get_base";
        case APIR_COMMAND_TYPE_BUFFER_SET_TENSOR:
            return "backend_buffer_set_tensor";
        case APIR_COMMAND_TYPE_BUFFER_GET_TENSOR:
            return "backend_buffer_get_tensor";
        case APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR:
            return "backend_buffer_cpy_tensor";
        case APIR_COMMAND_TYPE_BUFFER_CLEAR:
            return "backend_buffer_clear";
        case APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER:
            return "backend_buffer_free_buffer";
        /* backend */
        case APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE:
            return "backend_backend_graph_compute";

        default:
            return "unknown";
    }
}

extern "C" {
static const backend_dispatch_t apir_backend_dispatch_table[APIR_BACKEND_DISPATCH_TABLE_COUNT] = {

    /* device */

    /* APIR_COMMAND_TYPE_DEVICE_GET_DEVICE_COUNT  = */ backend_device_get_device_count,
    /* APIR_COMMAND_TYPE_DEVICE_GET_COUNT  = */ backend_device_get_count,
    /* APIR_COMMAND_TYPE_DEVICE_GET_NAME  = */ backend_device_get_name,
    /* APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION  = */ backend_device_get_description,
    /* APIR_COMMAND_TYPE_DEVICE_GET_TYPE  = */ backend_device_get_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_MEMORY  = */ backend_device_get_memory,
    /* APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP  = */ backend_device_supports_op,
    /* APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE  = */ backend_device_get_buffer_type,
    /* APIR_COMMAND_TYPE_DEVICE_GET_PROPS  = */ backend_device_get_props,
    /* APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR  = */ backend_device_buffer_from_ptr,

    /* buffer-type */

    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME  = */ backend_buffer_type_get_name,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT  = */ backend_buffer_type_get_alignment,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE  = */ backend_buffer_type_get_max_size,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST  = */ backend_buffer_type_is_host /* DEPRECATED */,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER  = */ backend_buffer_type_alloc_buffer,
    /* APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALLOC_SIZE  = */ backend_buffer_type_get_alloc_size,

    /* buffer */

    /* APIR_COMMAND_TYPE_BUFFER_GET_BASE  = */ backend_buffer_get_base,
    /* APIR_COMMAND_TYPE_BUFFER_SET_TENSOR  = */ backend_buffer_set_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_GET_TENSOR  = */ backend_buffer_get_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR  = */ backend_buffer_cpy_tensor,
    /* APIR_COMMAND_TYPE_BUFFER_CLEAR  = */ backend_buffer_clear,
    /* APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER  = */ backend_buffer_free_buffer,

    /* backend */

    /* APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE  = */ backend_backend_graph_compute,
};
}
