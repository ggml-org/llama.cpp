#pragma once

#include "common.hpp"
#include "ggml-impl.h"
#include "hexagon_npu.h"
#include "util.hpp"

namespace hexagon {

// TODO: merge this with device tensor?
class host_tensor {
  public:
    static host_tensor * from_ggml_tensor(ggml_tensor * tensor) {
        if (!tensor || !tensor->extra) {
            return nullptr;
        }
        return static_cast<host_tensor *>(tensor->extra);
    }

    explicit host_tensor(ggml_tensor * tensor, int buffer_fd, uint64_t offset, remote_handle64 device_handle) :
        _device_handle(device_handle) {
        _info.buffer_fd = buffer_fd;
        _info.offset    = offset;
        _info.type      = type_to_npu_type(tensor->type);
        _info.op        = op_to_npu_op(tensor->op);
        _info.size      = ggml_nbytes(tensor);

        static_assert(DEVICE_TENSOR_MAX_DIMS == GGML_MAX_DIMS, "tensor dimensions mismatch");
        static_assert(sizeof(_info.ne) == sizeof(tensor->ne), "tensor ne size mismatch");
        static_assert(sizeof(_info.nb) == sizeof(tensor->nb), "tensor nb size mismatch");
        memcpy(_info.ne, tensor->ne, sizeof(_info.ne));
        memcpy(_info.nb, tensor->nb, sizeof(_info.nb));

        auto status = npu_device_tensor_init(_device_handle, &_info, &_device_tensor_handle);
        if (status != AEE_SUCCESS) {
            LOG_ERROR("Failed to init tensor: %d", (int) status);
            _device_tensor_handle = 0;
            return;
        }

        tensor->extra = this;
        _ggml_tensor  = tensor;
        LOG_DEBUG("host_tensor(%p), ggml_tensor(%p[%ldx%ldx%ldx%ld], nb[%ld][%ld][%ld][%ld], %s), handle(%p)\n",
                  (void *) this, (void *) tensor, (long) tensor->ne[0], (long) tensor->ne[1], (long) tensor->ne[2],
                  (long) tensor->ne[3], (long) tensor->nb[0], (long) tensor->nb[1], (long) tensor->nb[2],
                  (long) tensor->nb[3], ggml_type_name(tensor->type), (void *) _device_tensor_handle);
    }

    ~host_tensor() {
        LOG_DEBUG("host_tensor(%p) destroy, device_tensor_handle: %p\n", (void *) this, (void *) _device_tensor_handle);
        if (_device_tensor_handle) {
            npu_device_tensor_free(_device_handle, _device_tensor_handle);
            // TODO: figure out why the _ggml_tensor is invalid here
        }
    }

    npu_device_tensor_handle_t get_device_tensor_handle() const { return _device_tensor_handle; }

    void set_src(size_t index, host_tensor * src) {
        if (index >= DEVICE_TENSOR_MAX_SRC) {
            LOG_ERROR("host_tensor(%p) set_src[%zu] out of range\n", (void *) this, index);
            return;
        }

        LOG_DEBUG("host_tensor(%p) set_src[%zu]: %p\n", (void *) this, index, (void *) src);
        npu_device_tensor_set_src(_device_handle, _device_tensor_handle, index, src->get_device_tensor_handle());
    }

    void set_op(ggml_op op) {
        _info.op = op_to_npu_op(op);
        npu_device_tensor_set_op(_device_handle, _device_tensor_handle, _info.op);
    }

    bool is_valid() const { return _device_tensor_handle != 0; }

  private:
    remote_handle64            _device_handle        = 0;
    npu_device_tensor_handle_t _device_tensor_handle = 0;
    npu_device_tensor_config   _info                 = {};
    ggml_tensor *              _ggml_tensor          = nullptr;

    DISABLE_COPY(host_tensor);
    DISABLE_MOVE(host_tensor);
};

}  // namespace hexagon
