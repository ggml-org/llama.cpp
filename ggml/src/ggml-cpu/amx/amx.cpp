#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx

namespace ggml::cpu::amx {

// AMX buffer
class buffer : public ggml::cpu::buffer {
public:
    buffer(std::size_t size) : ggml::cpu::buffer(size) { }

    virtual ~buffer() { }

    ggml_status init_tensor(ggml_tensor& tensor) override {
        tensor.extra = (void *) ggml::cpu::amx::get_tensor_traits(&tensor);
        return GGML_STATUS_SUCCESS;
    }

    void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
        if (qtype_has_amx_kernels(tensor.type)) {
            GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor.name, ggml_type_name(tensor.type));
            ggml_backend_amx_convert_weight(&tensor, data, offset, size);
        } else {
            memcpy((char *) tensor.data + offset, data, size);
        }
    }

};

class extra_buffer_type : public ggml::cpu::extra_buffer_type {

    const std::string& get_name() override {
        static const std::string name {"AMX"};
        return name;
    }

    ggml::cpp::backend::buffer* alloc_buffer(std::size_t size) override {
        return new buffer(size);
    }

    std::size_t get_alloc_size(const ggml_tensor& tensor) override {
        return ggml_backend_amx_get_alloc_size(&tensor);
    }

    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            return false;
        }
        if (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            return false;
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;
        }
        if (op->ne[0] % (TILE_N * 2)) {
            return false;
        }
        int alignment;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
                alignment = TILE_K;
                break;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                break;
            case GGML_TYPE_F16:
                alignment = 16;
                break;
            default:
                return false;
        }
        if (src0->ne[0] % alignment) {
            return false;
        }
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        return true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};
}  // namespace ggml::cpu::amx

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#else
    return false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static auto* buffer_type = ggml::cpu::c_wrapper(new ggml::cpu::amx::extra_buffer_type());
    if (!ggml_amx_init()) {
        return nullptr;
    }
    return buffer_type;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
