#pragma once

#include <cstdint>
#include <cstddef>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "backend-convert.h"
#include "shared/apir_backend.h"
#include "shared/venus_cs.h"
#include "shared/venus_cs_ggml.h"

uint32_t backend_dispatch_initialize(void *ggml_backend_reg_fct_p, void *ggml_backend_init_fct_p);

typedef uint32_t (*backend_dispatch_t)(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* *** */

uint32_t backend_reg_get_device_count(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* device */
uint32_t backend_device_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_get_description(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_get_type(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_get_memory(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_supports_op(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_get_buffer_type(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_get_props(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_device_buffer_from_ptr(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* buffer-type */
uint32_t backend_buffer_type_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_type_get_alignment(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_type_get_max_size(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_type_is_host(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_type_alloc_buffer(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* buffer */
uint32_t backend_buffer_get_base(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_set_tensor(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_get_tensor(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_clear(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);
uint32_t backend_buffer_free_buffer(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* backend */
uint32_t backend_graph_compute(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

/* metal */
uint32_t backend_metal_get_device_context(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx);

static inline const char *backend_dispatch_command_name(ApirBackendCommandType type)
{
  switch (type) {
  /* device */
  case APIR_COMMAND_TYPE_DEVICE_GET_COUNT: return "backend_get_device_count";
  case APIR_COMMAND_TYPE_DEVICE_GET_NAME: return "backend_get_device_name";
  case APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION: return "backend_get_device_description";
  case APIR_COMMAND_TYPE_DEVICE_GET_TYPE: return "backend_device_get_type";
  case APIR_COMMAND_TYPE_DEVICE_GET_MEMORY: return "backend_get_device_memory";
  case APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP: return "backend_device_supports_op";
  case APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE: return "backend_get_buffer_type";
  case APIR_COMMAND_TYPE_DEVICE_GET_PROPS: return "backend_get_props";
  case APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR: return "backend_buffer_from_ptr";

  /* buffer-type */
  case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME: return "backend_buffer_type_get_name";
  case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT: return "backend_buffer_type_get_alignment";
  case APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE: return "backend_buffer_type_get_max_size";
  case APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST: return "backend_buffer_type_is_host";
  case APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER: return "backend_buffer_type_alloc_buffer";

  /* buffer */
  case APIR_COMMAND_TYPE_BUFFER_GET_BASE: return "backend_buffer_get_base";
  case APIR_COMMAND_TYPE_BUFFER_SET_TENSOR: return "backend_buffer_set_tensor";
  case APIR_COMMAND_TYPE_BUFFER_GET_TENSOR: return "backend_buffer_get_tensor";
  case APIR_COMMAND_TYPE_BUFFER_CLEAR: return "backend_buffer_clear";
  case APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER: return "backend_buffer_free_buffer";

  /* backend */
  case APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE: return "backend_graph_compute";

  /* metal */
  case APIR_COMMAND_TYPE_METAL_GET_DEVICE_CONTEXT: return "metal_get_device_context";

  default: return "unknown";
  }
}

static const backend_dispatch_t apir_backend_dispatch_table[APIR_BACKEND_DISPATCH_TABLE_COUNT] = {
  /* device */
  [APIR_COMMAND_TYPE_DEVICE_GET_COUNT] = backend_reg_get_device_count,
  [APIR_COMMAND_TYPE_DEVICE_GET_NAME] = backend_device_get_name,
  [APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION] = backend_device_get_description,
  [APIR_COMMAND_TYPE_DEVICE_GET_TYPE] = backend_device_get_type,
  [APIR_COMMAND_TYPE_DEVICE_GET_MEMORY] = backend_device_get_memory,
  [APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP] = backend_device_supports_op,
  [APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE] = backend_device_get_buffer_type,
  [APIR_COMMAND_TYPE_DEVICE_GET_PROPS] = backend_device_get_props,
  [APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR] = backend_device_buffer_from_ptr,

  /* buffer-type */
  [APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME] = backend_buffer_type_get_name,
  [APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT] = backend_buffer_type_get_alignment,
  [APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE] = backend_buffer_type_get_max_size,
  [APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST] = backend_buffer_type_is_host,
  [APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER] = backend_buffer_type_alloc_buffer,

  /* buffer */
  [APIR_COMMAND_TYPE_BUFFER_GET_BASE] = backend_buffer_get_base,
  [APIR_COMMAND_TYPE_BUFFER_SET_TENSOR] = backend_buffer_set_tensor,
  [APIR_COMMAND_TYPE_BUFFER_GET_TENSOR] = backend_buffer_get_tensor,
  [APIR_COMMAND_TYPE_BUFFER_CLEAR] = backend_buffer_clear,
  [APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER] = backend_buffer_free_buffer,

  /* backend */
  [APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE] = backend_graph_compute,

  /* metal */
  [APIR_COMMAND_TYPE_METAL_GET_DEVICE_CONTEXT] = backend_metal_get_device_context,
};
