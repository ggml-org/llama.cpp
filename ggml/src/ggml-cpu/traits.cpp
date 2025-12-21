#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <new>

namespace ggml::cpu {

buffer::buffer(std::size_t size) : m_size(size) {
    m_data = new (std::align_val_t(32)) uint8_t[m_size];
    GGML_ASSERT(m_data);
}

buffer::~buffer() {
    delete [] m_data;
}

void* buffer::get_base() {
    return m_data;
}

void buffer::memset_tensor(ggml_tensor & tensor, uint8_t value, std::size_t offset, std::size_t size) {
    GGML_ASSERT(value == 0);
    memset((uint8_t *) tensor.data + offset, value, size);
}

void buffer::get_tensor(const ggml_tensor &, void *, std::size_t, std::size_t size) {
    GGML_ASSERT(size == 0);
}

void buffer::clear(uint8_t value) {
    memset(m_data, value, m_size);
}

tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}

namespace {
    const char *buffer_type_get_name (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((extra_buffer_type*) (buft->context));
        return ctx.get_name().c_str();
    }
    std::size_t buffer_type_get_alignment (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((extra_buffer_type*) (buft->context));
        return ctx.get_alignment();
    }
    std::size_t buffer_type_get_max_size  (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((extra_buffer_type*) (buft->context));
        return ctx.get_max_size();
    }
    std::size_t buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
        auto& ctx = *((extra_buffer_type*) (buft->context));
        return ctx.get_alloc_size(*tensor);
    }
    bool buffer_type_is_host(ggml_backend_buffer_type_t /*buft*/) {
        return false;
    }
    ggml_backend_buffer_t buffer_type_alloc_buffer (ggml_backend_buffer_type_t buft, std::size_t size) {
        auto& ctx = *((extra_buffer_type*) (buft->context));
        return c_wrapper(buft, ctx.alloc_buffer(size), size);
    }
}

ggml_backend_buffer_type_t c_wrapper(extra_buffer_type* ctx) {
        if (!ctx) { return nullptr; }
        return new ggml_backend_buffer_type {
            /* .iface    = */ {
                /* .get_name         = */ buffer_type_get_name,
                /* .alloc_buffer     = */ buffer_type_alloc_buffer,
                /* .get_alignment    = */ buffer_type_get_alignment,
                /* .get_max_size     = */ buffer_type_get_max_size,
                /* .get_alloc_size   = */ buffer_type_get_alloc_size,
                /* .is_host          = */ buffer_type_is_host,
            },
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
            /* .context = */ ctx,
        };
}


}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
