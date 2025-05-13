#pragma once

#include <string>
#include <memory>

#include "ggml-remoting-frontend.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "virtgpu.h"

#define NOT_IMPLEMENTED							\
  do {									\
    static bool first = true;						\
    if (first) {							\
      printf("\nWARN: ###\nWARN: ### reached unimplemented function %s\nWARN: ###\n\n", __func__); \
      first = false;							\
    }									\
  } while(0)

#define IMPLEMENTED
//  printf("INFO: ### reached implemented function %s\n", __func__)

#define RMT_LOG_DEBUG(msg) std::cerr << msg << std::endl

struct ggml_backend_remoting_device_context {
  size_t device;
  std::string name;
  std::string description;

  struct virtgpu *gpu;
};

extern const struct ggml_backend_device_i ggml_backend_remoting_device_i;

ggml_backend_buffer_type_t ggml_backend_remoting_host_buffer_type();
ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params);
ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev);
ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params);

struct remoting_buffer_struct;
typedef std::shared_ptr<remoting_buffer_struct> remoting_buffer;
typedef std::weak_ptr<remoting_buffer_struct> remoting_buffer_ref;

void ggml_remoting_destroy_buffer(remoting_buffer& buf);

struct remoting_device_struct;
typedef std::shared_ptr<remoting_device_struct> remoting_device;
typedef std::weak_ptr<remoting_device_struct> remoting_device_ref;

struct ggml_backend_remoting_buffer_context {
  remoting_device_ref device;
  remoting_buffer dev_buffer;
  std::string name;

  ggml_backend_remoting_buffer_context(remoting_device_ref device, remoting_buffer&& dev_buffer, std::string& name) :
    name(name) {
    UNUSED(device);
    UNUSED(dev_buffer);
  }

  ~ggml_backend_remoting_buffer_context() {
    ggml_remoting_destroy_buffer(dev_buffer);
  }
};


struct remoting_context_struct {
  int i;
};
typedef std::shared_ptr<remoting_context_struct> remoting_context;
typedef std::weak_ptr<remoting_context_struct> remoting_context_ref;
