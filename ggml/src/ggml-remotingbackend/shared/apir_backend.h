#pragma once

#define APIR_BACKEND_INITIALIZE_SUCCESSS 0
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_BACKEND_LIBRARY 1
#define APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY 2
#define APIR_BACKEND_INITIALIZE_MISSING_BACKEND_SYMBOLS 3
#define APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS 4
#define APIR_BACKEND_INITIALIZE_BACKEND_FAILED 5

#define APIR_BACKEND_FORWARD_INDEX_INVALID 6

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

  // last command_type index + 1
  APIR_BACKEND_DISPATCH_TABLE_COUNT = 20,
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

static inline void start_timer(struct timer_data *timer) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
  timer->start = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void stop_timer(struct timer_data *timer) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
  long long timer_end = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;

  timer->total += (timer_end - timer->start);
  timer->count += 1;
}

static inline void show_timer(struct timer_data *timer) {
  double ms = timer->total/1000000;
  double itl = ms/timer->count;
  double speed = 1/itl * 1000;

  INFO("%14s [%9.0f] ms for %4ld invocations | ITL %2.2f ms | throughput = %4.2f t/s",
       timer->name, ms, timer->count, itl, speed);
}
