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

// Managed attention may omit native output extras only when its graph uses external scratch.
GGML_API void ggml_backend_execution_set_external_workspace(ggml_tensor * op, bool external);
GGML_API bool ggml_backend_execution_has_external_workspace(const ggml_tensor * op);
// Match native conversion of the original tensor storage rather than a physical span layout.
GGML_API void ggml_backend_execution_set_stock_conversion(ggml_tensor * op, bool enabled);
GGML_API bool ggml_backend_execution_has_stock_conversion(const ggml_tensor * op);
// Wider workloads omit native extras only after the owner quotes their replacement storage.
GGML_API void ggml_backend_execution_set_verified_workspace(ggml_tensor * op);
GGML_API bool ggml_backend_execution_has_verified_workspace(const ggml_tensor * op);

// One exact [begin,end) visibility interval per query. Tokens includes stock KV padding.
struct ggml_backend_execution_mask_interval {
    int32_t begin, end, tokens, version;
};
static_assert(sizeof(ggml_backend_execution_mask_interval) == 4*sizeof(int32_t), "mask descriptor layout");
// Reject holes, biases, NaNs and signed zero rather than approximating a dense mask.
GGML_API bool ggml_backend_execution_mask_interval_encode(const ggml_fp16_t * mask, size_t tokens,
        ggml_backend_execution_mask_interval & interval);
GGML_API ggml_fp16_t ggml_backend_execution_mask_interval_value(const ggml_backend_execution_mask_interval & interval,
        int32_t token);
// Private source 5 is an I32 [4,queries] input, never a dense F16 mask alias.
GGML_API bool ggml_backend_execution_set_mask_bounds(ggml_tensor * op, ggml_tensor * bounds);
// Produce planning-only dense metadata with no storage. Never use it for physical range validation or byte I/O.
GGML_API bool ggml_backend_execution_mask_metadata(const ggml_tensor * op, ggml_tensor & logical, ggml_tensor & mask);
