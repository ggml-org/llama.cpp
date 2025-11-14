#include "ggml-remoting.h"

#define BUFT_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->device->context)->gpu

extern const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface;

static void
ggml_backend_remoting_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  BEING_IMPLEMENTED;

  void *ptr = buffer->context;

  if (ptr == nullptr) {
        return;
  }
  struct ggml_backend_remoting_device_context *device_ctx = GET_DEVICE_CONTEXT();

  struct vn_renderer_shmem *shmem = nullptr;
  size_t index;

  for (size_t i = 0; i < device_ctx->shared_memory.size(); i++) {
    const uint8_t* addr = (const uint8_t*) std::get<0>(device_ctx->shared_memory[i]) /* ptr */;
    const uint8_t* endr = addr + std::get<1>(device_ctx->shared_memory[i]) /* size */;
    if (ptr >= addr && ptr < endr) {
      shmem = std::get<2>(device_ctx->shared_memory[i]) /* shmem */;
      index = i;
      break;
    }
  }

  if (shmem == nullptr) {
    WARNING("failed to free host shared memory: memory not in map\n");
    return;
  }

  virtgpu_shmem_destroy(device_ctx->gpu, shmem->shmem);

  device_ctx->shared_memory.erase(device_ctx->shared_memory.begin() + index);
}

static ggml_backend_buffer_t
ggml_backend_remoting_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  IMPLEMENTED;

  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  struct ggml_backend_remoting_buffer_context *context = (struct ggml_backend_remoting_buffer_context *) malloc(sizeof(*context));
  if (!context) {
    FATAL("Couldn't allocate the buffer context ...");
  }

  context->gpu = gpu;
  context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
  context->base = context->apir_context.shmem->mmap_ptr;
  context->is_host_buffer = true;

  ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (void *) context, size);
  INFO("##");
  INFO("## %s(%llx) --> %p <======================", __func__, size, buffer);
  INFO("##\n");

  return buffer;
}

static const char *
ggml_backend_remoting_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  IMPLEMENTED_ONCE;

  return "GUEST host buffer";
}

static size_t
ggml_backend_remoting_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  IMPLEMENTED_ONCE;

  return 64; // not 100% sure ...
}

static bool
ggml_backend_remoting_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  IMPLEMENTED_ONCE;

  return true;
}

static size_t
ggml_backend_remoting_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  IMPLEMENTED;
  STOP_HERE;

  return SIZE_MAX;
}

const ggml_backend_buffer_type_i ggml_backend_remoting_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_host_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_host_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
    /* .is_host          = */ ggml_backend_remoting_host_buffer_type_is_host,
  };
