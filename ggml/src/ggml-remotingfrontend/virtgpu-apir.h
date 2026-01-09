#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"

#include "virtgpu-shm.h"
#include "virtgpu-utils.h"

#include "../ggml-remotingbackend/shared/apir_backend.h"

typedef struct {
  apir_buffer_host_handle_t host_handle;

  struct virtgpu_shmem shmem;
  apir_buffer_type_host_handle_t buft_host_handle;
} apir_buffer_context_t;

#include "virtgpu-forward.gen.h"
