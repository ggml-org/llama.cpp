#pragma once

#define APIR_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend-prod/bin/libggml-remotingbackend.dylib"
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

#define APIR_ALLOC_FROM_HOST_PTR 0

typedef uintptr_t apir_buffer_type_host_handle_t;
typedef uintptr_t apir_buffer_host_handle_t;

typedef struct {
  apir_buffer_host_handle_t host_handle;
#if APIR_ALLOC_FROM_HOST_PTR
  struct vn_renderer_shmem *shmem;
  apir_buffer_type_host_handle_t buft_host_handle;
#endif
} apir_buffer_context_t;

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
  APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER = 17,

  /* backend */
  APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 18,

  // last command_type index + 1
  APIR_BACKEND_DISPATCH_TABLE_COUNT = 19,
} ApirBackendCommandType;


struct virgl_apir_callbacks {
  void *(*get_shmem_ptr)(struct vn_dispatch_context *ctx, uint32_t res_id);
} ;

struct virgl_apir_context {
  struct vn_dispatch_context *virgl_ctx;

  struct virgl_apir_callbacks iface;
};

extern long long timer_start;
extern long long timer_total;
extern long long timer_count;

static inline void start_timer(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
  timer_start = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void stop_timer(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
  long long timer_end = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;

  timer_total += (timer_end - timer_start);
  timer_count += 1;
}

static inline void show_timer(void) {
  long long ms = timer_total/1000000;
  long long itl = ms/timer_count;
  float speed = 1/((float)itl) * 1000;

  INFO("compute_graph: [%9ld] ms for %ld invokations | ITL %lldms | throughput = %.2f t/s\n", timer_total/1000000, timer_count, itl, speed);
  INFO("compute_graph: [%9ld] s", (ms)/1000);
}
