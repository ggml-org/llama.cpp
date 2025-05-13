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

typedef uint32_t (*apir_backend_initialize_t)(void);
typedef void (*apir_backend_deinit_t)(void);

typedef uint32_t (*apir_backend_dispatch_t)(uint32_t cmd_type,
					    char *dec_cur, const char *dec_end,
					    char *enc_cur, const char *enc_end,
					    char **enc_cur_after
  );

typedef enum ApirBackendCommandType {
    APIR_COMMAND_TYPE_DEVICE_GET_COUNT = 0,
    APIR_COMMAND_TYPE_DEVICE_GET_NAME = 1,
    APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION = 2,
    APIR_COMMAND_TYPE_DEVICE_GET_TYPE = 3,
    APIR_COMMAND_TYPE_DEVICE_GET_MEMORY = 4,

    APIR_BACKEND_DISPATCH_TABLE_COUNT = 5, // last command_type index + 1
} ApirBackendCommandType;
