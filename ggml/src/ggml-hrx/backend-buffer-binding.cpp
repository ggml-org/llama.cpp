#include "backend-buffer-binding.h"

#include "ggml-backend-impl.h"
#include "ggml.h"

ggml_backend_hrx_buffer_context * ggml_backend_hrx_buffer_context_from_buffer(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

void * ggml_backend_hrx_buffer_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_buffer_context_from_buffer(buffer)->base;
}

bool ggml_backend_hrx_tensor_binding(const ggml_tensor *                tensor,
                                     ggml_backend_hrx_buffer_context ** out_context,
                                     size_t *                           out_offset) {
    if (tensor == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || buffer->iface.get_base != ggml_backend_hrx_buffer_base) {
        return false;
    }
    auto *       context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    const size_t offset  = ggml_backend_hrx_tensor_offset(context, tensor);
    if (context->buffer == nullptr || offset > buffer->size || ggml_nbytes(tensor) > buffer->size - offset) {
        return false;
    }
    *out_context = context;
    *out_offset  = offset;
    return true;
}

bool ggml_backend_hrx_resolve_value_buffer(const ggml_tensor * tensor, ggml::hrx::ValueBufferBinding & binding) {
    ggml_backend_hrx_buffer_context * context = nullptr;
    size_t                            offset  = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &offset)) {
        if (tensor == nullptr) {
            return false;
        }
        const ggml_tensor *   root   = tensor->view_src != nullptr ? tensor->view_src : tensor;
        ggml_backend_buffer_t buffer = root->buffer;
        if (buffer == nullptr || !ggml_backend_buffer_is_host(buffer)) {
            return false;
        }
        void *       base     = ggml_backend_buffer_get_base(buffer);
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (base == nullptr || tensor->data == nullptr ||
            static_cast<const uint8_t *>(tensor->data) < static_cast<const uint8_t *>(base)) {
            return false;
        }
        const size_t host_offset =
            static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - static_cast<const uint8_t *>(base));
        if (host_offset > capacity || ggml_nbytes(tensor) > capacity - host_offset) {
            return false;
        }
        const uint64_t buffer_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer));
        const uint64_t base_address   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
        binding.host_data             = base;
        binding.offset                = host_offset;
        binding.length                = ggml_nbytes(tensor);
        binding.identity =
            buffer_address ^ (base_address + 0x9e3779b97f4a7c15ull + (buffer_address << 6) + (buffer_address >> 2));
        binding.generation = 1;
        binding.capacity   = capacity;
        binding.weight     = ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        return true;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    const bool directly_bindable = !ggml_backend_buffer_is_host(buffer) || context->direct_host_binding;
    // Coherent HRX host allocations are directly device-addressable. Represent them with an HRX buffer handle so
    // command-program preparation bypasses host materialization. Noncoherent host allocations remain host data.
    binding.buffer               = directly_bindable ? context->buffer : nullptr;
    binding.host_data            = directly_bindable ? nullptr : context->base;
    binding.offset               = offset;
    binding.length               = ggml_nbytes(tensor);
    binding.identity             = context->identity;
    binding.generation           = context->generation;
    binding.capacity             = buffer != nullptr ? buffer->size : 0;
    binding.weight = buffer != nullptr && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    return true;
}
