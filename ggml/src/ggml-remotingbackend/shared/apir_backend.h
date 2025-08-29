#pragma once

#define APIR_BACKEND_INITIALIZE_SUCCESS 0
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY 2
#define APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS 3
#define APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS 4

#define APIR_BACKEND_INITIALIZE_BACKEND_FAILED 5
// new entries here need to be added to the apir_backend_initialize_error function below

#define APIR_BACKEND_FORWARD_INDEX_INVALID 6

// 0 is fast, 1 avoids the backend to crash if an unsupported tensor is received
#define APIR_BACKEND_CHECK_SUPPORTS_OP 0

typedef uintptr_t apir_buffer_type_host_handle_t;
typedef uintptr_t apir_buffer_host_handle_t;

typedef struct {
  apir_buffer_host_handle_t host_handle;

  struct vn_renderer_shmem *shmem;
  apir_buffer_type_host_handle_t buft_host_handle;
} apir_buffer_context_t;

struct vn_dispatch_context;
struct virgl_apir_context;

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
  APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR = 8,

  /* buffer-type */
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME = 9,
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT = 10,
  APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE = 11,
  APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST = 12,
  APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER = 13,

  /* buffer */
  APIR_COMMAND_TYPE_BUFFER_GET_BASE = 14,
  APIR_COMMAND_TYPE_BUFFER_SET_TENSOR = 15,
  APIR_COMMAND_TYPE_BUFFER_GET_TENSOR = 16,
  APIR_COMMAND_TYPE_BUFFER_CLEAR = 17,
  APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER = 18,

  /* backend */
  APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE = 19,

  /* metal */
  APIR_COMMAND_TYPE_METAL_GET_DEVICE_CONTEXT = 20,

  // last command_type index + 1
  APIR_BACKEND_DISPATCH_TABLE_COUNT = 21,
} ApirBackendCommandType;


struct virgl_apir_callbacks {
  void *(*get_shmem_ptr)(struct vn_dispatch_context *ctx, uint32_t res_id);
};

struct virgl_apir_context {
  struct vn_dispatch_context *virgl_ctx;

  struct virgl_apir_callbacks iface;
};

struct timer_data {
  long long start;
  long long total;
  long long count;
  const char *name;
};

extern struct timer_data graph_compute_timer;
extern struct timer_data get_tensor_timer;
extern struct timer_data set_tensor_timer;
extern struct timer_data wait_host_reply_timer;
extern struct timer_data get_tensor_from_ptr_timer;
extern struct timer_data set_tensor_from_ptr_timer;

static inline void start_timer(struct timer_data *timer) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  timer->start = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
static inline long long stop_timer(struct timer_data *timer) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  long long timer_end = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;

  long long duration = (timer_end - timer->start);
  timer->total += duration;
  timer->count += 1;

  return duration;
}

static inline void show_timer(struct timer_data *timer) {
  double ms = timer->total/1000000;
  double itl = ms/timer->count;
  double speed = 1/itl * 1000;

  if (!timer->total) {
    return;
  }

  INFO("%15s [%9.0f] ms for %4ld invocations | ITL %2.2f ms | throughput = %4.2f t/s (%4.2f ms/call)",
       timer->name, ms, timer->count, itl, speed, ms/timer->count);
}

static const char *apir_backend_initialize_error(int code) {
#define APIR_BACKEND_INITIALIZE_ERROR(code_name) \
  do {						 \
    if (code == code_name) return #code_name;	 \
  } while (0)					 \

  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_SUCCESS);
  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY);
  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY);
  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS);
  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS);
  APIR_BACKEND_INITIALIZE_ERROR(APIR_BACKEND_INITIALIZE_BACKEND_FAILED);

  return "Unknown APIR_BACKEND_INITIALIZE error:/";

#undef APIR_BACKEND_INITIALIZE_ERROR
}
