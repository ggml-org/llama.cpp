#pragma once

#define APIR_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-remotingbackend.dylib"
#define APIR_INITIALIZE_FCT_NAME "apir_backend_initialize"
#define APIR_DEINIT_FCT_NAME "apir_backend_deinit"
#define APIR_DISPATCH_FCT_NAME "apir_backend_dispatcher"

#define APIR_BACKEND_INITIALIZE_SUCCESSS 0
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY 2
#define APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS 3
#define APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS 4
#define APIR_BACKEND_INITIALIZE_BACKEND_FAILED 5

#define APIR_BACKEND_FORWARD_INDEX_INVALID 6

typedef uintptr_t apir_buffer_type_handle_t;
typedef uintptr_t apir_buffer_handle_t;

typedef uint32_t (*apir_backend_initialize_t)(void);
typedef void (*apir_backend_deinit_t)(void);

struct vn_dispatch_context;
struct virgl_apir_context;

typedef uint32_t (*apir_backend_dispatch_t)(uint32_t cmd_type, struct virgl_apir_context *ctx,
                                            char *dec_cur, const char *dec_end,
                                            char *enc_cur, const char *enc_end,
                                            char **enc_cur_after
  );

typedef enum ApirBackendCommandType {
  /* device */
  APIR_COMMAND_TYPE_DEVICE_GET_COUNT = 0,
  APIR_COMMAND_TYPE_DEVICE_GET_NAME = 1,
  APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION = 2,
  APIR_COMMAND_TYPE_DEVICE_GET_TYPE = 3,
  APIR_COMMAND_TYPE_DEVICE_GET_MEMORY = 4,
  APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP = 5,
  APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE = 6,
  APIR_COMMAND_TYPE_DEVICE_GET_PROPS = 7,

  /* buffer-type */
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME = 8,
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT = 9,
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE = 10,
  APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST = 11,
  APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER = 12,

  /* buffer */
  APIR_COMMAND_TYPE_BUFFER_GET_BASE = 13,
  APIR_COMMAND_TYPE_BUFFER_SET_TENSOR = 14,
  APIR_COMMAND_TYPE_BUFFER_GET_TENSOR = 15,
  APIR_COMMAND_TYPE_BUFFER_CLEAR = 16,

  /* backend */
  APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 17,

  // last command_type index + 1
  APIR_BACKEND_DISPATCH_TABLE_COUNT = 18,
} ApirBackendCommandType;


struct virgl_apir_callbacks {
  void *(*get_shmem_ptr)(struct vn_dispatch_context *ctx, uint32_t res_id);
} ;

struct virgl_apir_context {
  struct vn_dispatch_context *virgl_ctx;

  struct virgl_apir_callbacks iface;
};

#define TENSOR_MAX_DEPTH_DEVICE_SUPPORTS_OP 2
#define TENSOR_MAX_DEPTH_BUFFER_GET_TENSOR 2
#define TENSOR_MAX_DEPTH_BUFFER_SET_TENSOR 2
#define TENSOR_MAX_DEPTH_CGRAPH_DATA 10
