#include "backend/shared/apir_backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "virtgpu-shm.h"
#include "virtgpu-utils.h"

struct apir_buffer_context_t {
    apir_buffer_host_handle_t host_handle;

    struct virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#include "virtgpu-forward.gen.h"
