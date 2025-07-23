
#include <algorithm>
#include <string>

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-traits.h"
#include "lut_mul_mat.h"
#include "tmac.h"

#if defined(GGML_USE_TMAC)
namespace ggml::cpu::tmac {

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}


class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // auto is_contiguous = [](const struct ggml_tensor * t) {
        //     return ggml_is_contiguous(t);
        // };

        if (// ggml_is_contiguous(src0) &&         // src0 must be contiguous
            // ggml_is_contiguous(src1) &&         // src1 must be contiguous
            // op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_tmac_buffer_type() &&
            ggml_tmac_can_mul_mat(op)) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {    // src1 must be host buffer
                return false;
            }
            return true;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_tmac_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::tmac

void ggml_tmac_init() {
    tmac_init();
}

static void ggml_backend_tmac_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_aligned_free(buffer->context, buffer->size);
}

static void * ggml_backend_tmac_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static enum ggml_status ggml_backend_tmac_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::tmac::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_tmac_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_tmac_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    if (is_type_supported(tensor->type) && !is_embedding_weight(tensor)) {
        GGML_LOG_DEBUG("%s: tmac repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_tmac_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_tmac_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}


static ggml_backend_buffer_i ggml_backend_tmac_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tmac_buffer_free_buffer,      // same as ggml_backend_cpu_buffer_free_buffer
    /* .get_base        = */ ggml_backend_tmac_buffer_get_base,         // same as ggml_backend_cpu_buffer_get_base
    /* .init_tensor     = */ ggml_backend_tmac_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_tmac_buffer_memset_tensor,    // same as ggml_backend_cpu_buffer_memset_tensor
    /* .set_tensor      = */ ggml_backend_tmac_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_tmac_buffer_clear,            // same as ggml_backend_cpu_buffer_clear
    /* .reset           = */ nullptr,
};


// T-MAC backend buffer type
static const char * ggml_backend_tmac_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "TMAC";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_tmac_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_tmac_buffer_interface, data, size);
}

static size_t ggml_backend_tmac_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_tmac_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // T-MAC version of ggml_nbytes
    return ggml_tmac_get_nbytes(tensor);  

    GGML_UNUSED(buft);
}

static bool ggml_backend_tmac_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_tmac_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_tmac = {
        /* .iface = */ {
                        /* .get_name         = */ ggml_backend_tmac_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_tmac_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ ggml_backend_tmac_buffer_type_get_alignment,      // same as ggml_backend_cpu_*
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ ggml_backend_tmac_buffer_type_get_alloc_size,
                        /* .is_host          = */ ggml_backend_tmac_buffer_type_is_host,            // same as ggml_backend_cpu_*
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::tmac::extra_buffer_type(),
    };

    return &ggml_backend_buffer_type_tmac;
}

#endif // GGML_USE_TMAC