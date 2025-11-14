#pragma once

#include <string>
#include <memory>

#include "ggml-remoting-frontend.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-metal-remoting.h"
#include "virtgpu.h"


// 1 is fast, 0 avoid micro-benchmark crashes
#define USE_ALWAYS_TRUE_SUPPORTS_OP 0
#define USE_METAL_GUEST_SUPPORTS_OP 1

#define DEV_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->context)->gpu

#define BUFFER_TO_GGML_CONTEXT(name) \
  ((struct ggml_backend_remoting_buffer_context *) (name)->context)

#define BUFFER_TO_APIR_CONTEXT(name) \
  &((struct ggml_backend_remoting_buffer_context *) (name)->context)->apir_context

#define BUFFER_TO_HOST_HANDLE(name) \
  ((struct ggml_backend_remoting_buffer_context *) (name)->context)->apir_context.host_handle

#define GET_DEVICE_CONTEXT() \
  (struct ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

static inline apir_buffer_type_host_handle_t
ggml_buffer_type_to_apir_handle(ggml_backend_buffer_type_t buft) {
  // in the backend, the buffer handle is the buffer pointer
  return (apir_buffer_type_host_handle_t) buft->context;
}

#define NOT_IMPLEMENTED							\
  do {									\
    static bool first = true;						\
    if (first) {							\
      printf("\nWARN: ###\nWARN: ### reached unimplemented function %s\nWARN: ###\n\n", __func__); \
      first = false;							\
    }									\
  } while(0)

#define BEING_IMPLEMENTED							\
  do {									\
      printf("\nINFO: ###\nINFO: ### function being implemented: %s\nINFO: ###\n\n", __func__); \
  } while(0)

#define NEXT

#define STOP_HERE \
  thks_bye()

#define BREAKPOINT \
  breakpoint()

#ifndef NDEBUG
#define IMPLEMENTED							\
  printf("INFO: ### reached implemented function %s\n", __func__)
#else
#define IMPLEMENTED							\
  do {} while(0)
#endif

#ifndef NDEBUG
#define IMPLEMENTED_ONCE						\
  do {									\
    static bool first = true;						\
    if (first) {							\
      printf("INFO: ### reached implemented function %s\n", __func__);  \
      first = false;							\
    }									\
  } while(0)
#else
#define IMPLEMENTED_ONCE			\
  do {} while(0)
#endif

#define RMT_LOG_DEBUG(msg) std::cerr << msg << std::endl

struct ggml_backend_remoting_device_context {
  size_t device;
  std::string name;
  std::string description;

  std::vector<std::tuple<void*, size_t, struct vn_renderer_shmem *>> shared_memory;

  struct virtgpu *gpu;

  const struct ggml_backend_metal_device_context *metal_dev_ctx;
};

struct ggml_backend_remoting_buffer_context {
  apir_buffer_context_t apir_context;

  struct virtgpu *gpu;

  void *base;

  bool is_host_buffer;
  bool is_from_ptr;
};

extern const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface;
extern const struct ggml_backend_device_i ggml_backend_remoting_device_interface;
extern const ggml_backend_buffer_type_i ggml_backend_remoting_host_buffer_type_interface;
extern const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface;
extern const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface;
extern const ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device);
ggml_backend_buffer_type_t ggml_backend_remoting_host_buffer_type();
ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params);
ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev);

struct remoting_buffer_struct;
typedef std::shared_ptr<remoting_buffer_struct> remoting_buffer;
typedef std::weak_ptr<remoting_buffer_struct> remoting_buffer_ref;

void ggml_remoting_destroy_buffer(remoting_buffer& buf);

struct remoting_device_struct;
typedef std::shared_ptr<remoting_device_struct> remoting_device;
typedef std::weak_ptr<remoting_device_struct> remoting_device_ref;

struct remoting_context_struct {
  int i;
};
typedef std::shared_ptr<remoting_context_struct> remoting_context;
typedef std::weak_ptr<remoting_context_struct> remoting_context_ref;

static inline apir_buffer_host_handle_t ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
  return BUFFER_TO_HOST_HANDLE(buffer);
}
