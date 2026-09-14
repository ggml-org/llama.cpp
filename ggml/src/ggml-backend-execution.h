#pragma once
#include "ggml-backend-impl.h"

// Private device-buffer dispatch. A backend must explicitly support this storage contract.
struct ggml_backend_execution_ops {
    bool (*supports)(void * context, const ggml_tensor * op);
    ggml_status (*compute)(void * context, ggml_backend_t backend, ggml_tensor * op);
    bool (*can_write)(void * context);
    void (*modified)(void * context);
    void (*destroy)(void * context);
};

// Retain stateless host backing and adopt context only on success; native reset state is not bypassed.
// Destruction callbacks must not throw. Tensor byte layout remains the backing's layout.
GGML_API ggml_backend_buffer_t ggml_backend_execution_buffer_new(ggml_backend_dev_t device,
        ggml_backend_buffer_t backing, const ggml_backend_execution_ops & ops, void * context);
GGML_API bool ggml_backend_buft_is_execution(ggml_backend_buffer_type_t type);
GGML_API bool ggml_backend_execution_buffers_present();
// Multiple unrelated owners are rejected. A null owner means ordinary storage.
GGML_API bool ggml_backend_execution_owner(const ggml_tensor * op, ggml_backend_buffer_t & owner);
GGML_API bool ggml_backend_execution_supports(ggml_backend_buffer_t owner, ggml_backend_dev_t device, const ggml_tensor * op);
GGML_API ggml_status ggml_backend_execution_compute(ggml_backend_buffer_t owner, ggml_backend_t backend, ggml_tensor * op);
