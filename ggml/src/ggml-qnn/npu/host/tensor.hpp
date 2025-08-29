#pragma once

#include "common.hpp"
#include "ggml-impl.h"
#include "hexagon_npu.h"
#include "util.hpp"

#include <list>
#include <type_traits>
#include <vector>

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
        // TODO: figure out why the npu_device_tensor_config can't be larger than 100 bytes
        static_assert(sizeof(npu_device_tensor_config) < kMaxNpuRpcStructSize,
                      "npu_device_tensor_config size too large");

        _info.buffer_fd   = buffer_fd;
        _info.offset      = offset;
        _info.type        = type_to_npu_type(tensor->type);
        _info.size        = ggml_nbytes(tensor);
        _info.is_constant = false;  // TODO: support constant tensors in the future
        // _info.op will be updated in update_params()
        _info_update.op   = NPU_OP_COUNT;

        static_assert(DEVICE_TENSOR_MAX_DIMS == GGML_MAX_DIMS, "tensor dimensions mismatch");
        static_assert(sizeof(_info.ne) == sizeof(tensor->ne), "tensor ne size mismatch");
        static_assert(sizeof(_info.nb) == sizeof(tensor->nb), "tensor nb size mismatch");
        memcpy(_info.ne, tensor->ne, sizeof(_info.ne));
        memcpy(_info.nb, tensor->nb, sizeof(_info.nb));

        auto status = npu_device_tensor_init(_device_handle, &_info, &_device_tensor_handle);
        if (status != AEE_SUCCESS) {
            LOG_ERROR("Failed to init tensor: %d", (int) status);
            _device_tensor_handle = npu_device_INVALID_DEVICE_TENSOR_HANDLE;
            return;
        }

        tensor->extra = this;
        _ggml_tensor  = tensor;

#ifndef NDEBUG
        {
            char desc[1024];
            get_desc(desc, sizeof(desc));
            LOG_DEBUG("host_tensor(%s)\n", desc);
        }
#endif
    }

    ~host_tensor() {
        LOG_DEBUG("host_tensor(%p) destroy, device_tensor_handle: %p\n", (void *) this, (void *) _device_tensor_handle);
        if (_device_tensor_handle != npu_device_INVALID_DEVICE_TENSOR_HANDLE) {
            npu_device_tensor_free(_device_handle, _device_tensor_handle);
            // TODO: figure out why the _ggml_tensor is invalid here
        }
    }

    static void destroy_tensors(std::list<std::shared_ptr<host_tensor>> & tensors) {
        std::vector<npu_device_tensor_handle_t> handles;

        handles.reserve(tensors.size());
        remote_handle64 device_handle = 0;

        for (auto tensor : tensors) {
            if (tensor && tensor->_device_tensor_handle != npu_device_INVALID_DEVICE_TENSOR_HANDLE) {
                handles.push_back(tensor->_device_tensor_handle);
                tensor->_device_tensor_handle = npu_device_INVALID_DEVICE_TENSOR_HANDLE;  // prevent double free
                device_handle                 = tensor->_device_handle;
            }
        }

        if (!handles.empty()) {
            npu_device_tensors_free(device_handle, handles.data(), handles.size());
        }

        tensors.clear();
    }

    npu_device_tensor_handle_t get_device_tensor_handle() const { return _device_tensor_handle; }

    void update_params(ggml_tensor * ggml_tensor) {
        static_assert(sizeof(_info_update.params) <= sizeof(_ggml_tensor->op_params),
                      "device tensor params size mismatch");
        static_assert(DEVICE_TENSOR_MAX_SRC <= GGML_MAX_SRC, "device tensor src size mismatch");

        GGML_ASSERT(ggml_tensor == _ggml_tensor);
        if (!_ggml_tensor) {
            LOG_DEBUG("host_tensor(%p) _ggml_tensor is null\n", (void *) this);
            return;
        }

        auto new_op         = op_to_npu_op(_ggml_tensor->op);
        bool params_changed = new_op != _info_update.op;
        if (params_changed) {
            LOG_DEBUG("host_tensor(%p) op changed: %s\n", (void *) this, get_npu_op_desc(new_op));
        }

        _info_update.op = new_op;

        if (memcmp(_info_update.params, _ggml_tensor->op_params, sizeof(_info_update.params)) != 0) {
            params_changed = true;
            memcpy(_info_update.params, _ggml_tensor->op_params, sizeof(_info_update.params));
            LOG_DEBUG("host_tensor(%p) op_params changed: [%x, %x, %x, %x]\n",
                      (void *) this,
                      (int) _info_update.params[0],
                      (int) _info_update.params[1],
                      (int) _info_update.params[2],
                      (int) _info_update.params[3]);
        }

        npu_device_tensor_handle_t src_tensor_handles[DEVICE_TENSOR_MAX_SRC] = {};
        static_assert(std::is_same<decltype(_info_update.src_handles), decltype(src_tensor_handles)>::value,
                      "src tensor handles type mismatch");

        for (size_t j = 0; j < DEVICE_TENSOR_MAX_SRC && _ggml_tensor->src[j]; ++j) {
            auto * ggml_src       = _ggml_tensor->src[j];
            auto * src            = host_tensor::from_ggml_tensor(ggml_src);
            src_tensor_handles[j] = src->get_device_tensor_handle();
#ifndef NDEBUG
            char desc[1024];
            src->get_desc(desc, sizeof(desc));
            LOG_DEBUG("host_tensor(%p) set_src[%zu]: (%s)\n", (void *) this, j, desc);
#endif
        }

        if (memcmp(_info_update.src_handles, src_tensor_handles, sizeof(_info_update.src_handles)) != 0) {
            params_changed = true;
            memcpy(_info_update.src_handles, src_tensor_handles, sizeof(_info_update.src_handles));
            LOG_DEBUG("host_tensor(%p) src changed, handles: [%p, %p]\n",
                      (void *) this,
                      (void *) _info_update.src_handles[0],
                      (void *) _info_update.src_handles[1]);
        }

        if (params_changed) {
            npu_device_tensor_update_params(_device_handle, _device_tensor_handle, &_info_update);
            LOG_DEBUG("host_tensor(%p) update_params, op: %s, params: [%x, %x, %x, %x]\n",
                      (void *) this,
                      ggml_op_desc(_ggml_tensor),
                      (int) _info_update.params[0],
                      (int) _info_update.params[1],
                      (int) _info_update.params[2],
                      (int) _info_update.params[3]);
        } else {
            LOG_DEBUG("host_tensor(%p) update_params, no changes, op: %s, params: [%x, %x, %x, %x]\n",
                      (void *) this,
                      ggml_op_desc(_ggml_tensor),
                      (int) _info_update.params[0],
                      (int) _info_update.params[1],
                      (int) _info_update.params[2],
                      (int) _info_update.params[3]);
        }
    }

    const npu_device_tensor_update_config & update_hosts_params_only(ggml_tensor * ggml_tensor) {
        static_assert(sizeof(_info_update.params) <= sizeof(ggml_tensor->op_params),
                      "device tensor params size mismatch");
        static_assert(DEVICE_TENSOR_MAX_SRC <= GGML_MAX_SRC, "device tensor src size mismatch");

        GGML_ASSERT(ggml_tensor == _ggml_tensor);

        auto new_op     = op_to_npu_op(_ggml_tensor->op);
        _info_update.op = new_op;
        memcpy(_info_update.params, _ggml_tensor->op_params, sizeof(_info_update.params));

        for (size_t j = 0; j < DEVICE_TENSOR_MAX_SRC && _ggml_tensor->src[j]; ++j) {
            auto * ggml_src             = _ggml_tensor->src[j];
            auto * src                  = host_tensor::from_ggml_tensor(ggml_src);
            _info_update.src_handles[j] = src->get_device_tensor_handle();
#ifndef NDEBUG
            char desc[1024];
            src->get_desc(desc, sizeof(desc));
            LOG_DEBUG("host_tensor(%p) set_src[%zu]: (%s)\n", (void *) this, j, desc);
#endif
        }

        LOG_DEBUG("host_tensor(%p) update_params, op: %s, params: [%x, %x, %x, %x]\n",
                  (void *) this,
                  ggml_op_desc(_ggml_tensor),
                  (int) _info_update.params[0],
                  (int) _info_update.params[1],
                  (int) _info_update.params[2],
                  (int) _info_update.params[3]);
        return _info_update;
    }

    bool is_valid() const { return _device_tensor_handle != npu_device_INVALID_DEVICE_TENSOR_HANDLE; }

    int64_t get_ne(size_t index) const {
        if (index >= DEVICE_TENSOR_MAX_DIMS) {
            LOG_ERROR("host_tensor(%p) get_ne: index out of bounds: %zu\n", (void *) this, index);
            return 0;
        }

        return _info.ne[index];
    }

    int get_desc(char * buffer, size_t size) const {
        return snprintf(buffer,
                        size,
                        "%s[%ldx%ldx%ldx%ld], nb[%ld,%ld,%ld,%ld], %s, addr: %p, ggml: %p, handle:%p",
                        _ggml_tensor->name,
                        (long) _ggml_tensor->ne[0],
                        (long) _ggml_tensor->ne[1],
                        (long) _ggml_tensor->ne[2],
                        (long) _ggml_tensor->ne[3],
                        (long) _ggml_tensor->nb[0],
                        (long) _ggml_tensor->nb[1],
                        (long) _ggml_tensor->nb[2],
                        (long) _ggml_tensor->nb[3],
                        ggml_type_name(_ggml_tensor->type),
                        (void *) this,
                        (void *) _ggml_tensor,
                        (void *) _device_tensor_handle);
    }

  private:
    remote_handle64                 _device_handle        = 0;
    npu_device_tensor_handle_t      _device_tensor_handle = 0;
    npu_device_tensor_config        _info                 = {};
    npu_device_tensor_update_config _info_update          = {};
    ggml_tensor *                   _ggml_tensor          = nullptr;

    DISABLE_COPY(host_tensor);
    DISABLE_MOVE(host_tensor);
};

}  // namespace hexagon
