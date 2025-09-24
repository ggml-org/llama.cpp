#pragma once

#include "hexagon_npu.h"
#include "util.hpp"

#include <HAP_mem.h>
#include <qurt.h>

#include <atomic>

namespace hexagon {

constexpr const size_t kMaxTensorSrc   = DEVICE_TENSOR_MAX_SRC;
constexpr const size_t kMaxParamsCount = DEVICE_TENSOR_MAX_OP_PARAMS;

class tensor {
  public:
    explicit tensor(const npu_device_tensor_config & info) noexcept : _info(info) {
        uint64 phy_address  = 0;
        void * mmap_address = nullptr;
        auto   ret          = HAP_mmap_get(_info.buffer_fd, &mmap_address, &phy_address);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to mmap tensor buffer: %d\n", (int) ret);
            return;
        }

        _data = static_cast<uint8_t *>(mmap_address);
        DEVICE_LOG_DEBUG("tensor(%p[%ldx%ldx%ldx%ld]), fd: %d, offset: %zu, mmap_addr: %p, phy_addr: 0x%lx\n",
                         (void *) this,
                         (long) _info.ne[0],
                         (long) _info.ne[1],
                         (long) _info.ne[2],
                         (long) _info.ne[3],
                         (int) _info.buffer_fd,
                         (size_t) _info.offset,
                         (void *) mmap_address,
                         (long) phy_address);
    }

    ~tensor() noexcept {
        auto ret = HAP_mmap_put(_info.buffer_fd);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to unmap tensor buffer: %d\n", (int) ret);
        }

        DEVICE_LOG_DEBUG("~tensor(%p) fd: %d\n", (void *) this, _info.buffer_fd);
    }

    void flush() const {
        if (_data) {
            qurt_mem_cache_clean(
                (qurt_addr_t) (_data + _info.offset), (qurt_size_t) _info.size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
    }

    void invalidate() const {
        if (_data) {
            qurt_mem_cache_clean((qurt_addr_t) (_data + _info.offset),
                                 (qurt_size_t) _info.size,
                                 QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL,
                                 QURT_MEM_DCACHE);
        }
    }

    void update_config(const npu_device_tensor_update_config & config) {
        static_assert(sizeof(_op_params) == sizeof(config.params), "op params size mismatch");

        _op_type = config.op;
        memcpy(_op_params, config.params, sizeof(_op_params));
        for (size_t i = 0; i < DEVICE_TENSOR_MAX_SRC; ++i) {
            auto src_handle = config.src_handles[i];
            _src[i] = (src_handle != npu_device_INVALID_DEVICE_TENSOR_HANDLE ? reinterpret_cast<tensor *>(src_handle) :
                                                                               nullptr);
        }
    }

    tensor * get_src(size_t index) const {
        if (index >= kMaxTensorSrc) {
            return nullptr;
        }

        return _src[index];
    }

    const npu_device_tensor_config & get_info() const { return _info; }

    const int64_t get_ne(size_t index) const { return _info.ne[index]; }

    const size_t get_nb(size_t index) const { return _info.nb[index]; }

    const bool is_permuted() const {
        // Check if the tensor is permuted by comparing the nb values
        return is_transposed_or_permuted(_info.nb);
    }

    npu_device_tensor_op get_op() const { return _op_type; }

    template <typename _TyParam> const _TyParam get_op_param(size_t index) const {
        static_assert(sizeof(_TyParam) <= sizeof(_op_params), "_op_param type size exceeds op params size");

        if (sizeof(_TyParam) * (index + 1) >= sizeof(_op_params)) {
            return 0;
        }

        return reinterpret_cast<const _TyParam *>(_op_params)[index];
    }

    const int32_t * get_op_params() const { return _op_params; }

    const size_t get_op_param_count() const { return kMaxParamsCount; }

    npu_device_tensor_data_type get_type() const { return _info.type; }

    const uint8_t * get_read_buffer(const bool force_invalidate = false) const {
        if (force_invalidate || (!_info.is_constant && _has_modified)) {
            invalidate();
            const_cast<tensor *>(this)->_has_modified = false;  // TODO: avoid const_cast
        }

        return _data + _info.offset;
    }

    template <typename _Ty> const _Ty * get_read_buffer_as() const {
        const auto * buffer = get_read_buffer();
        if (!buffer) {
            return nullptr;
        }

        return reinterpret_cast<const _Ty *>(buffer);
    }

    uint8_t * get_write_buffer() const {
        if (_info.is_constant) {
            DEVICE_LOG_ERROR("Attempt to write to a constant tensor: %p\n", (void *) this);
            return nullptr;  // Do not allow writing to constant tensors
        }

        return _data + _info.offset;
    }

    void release_write_buffer() { _has_modified = true; }

    bool is_valid() const { return _data != nullptr; }

  private:
    npu_device_tensor_config _info                       = {};
    npu_device_tensor_op     _op_type                    = NPU_OP_COUNT;
    int32_t                  _op_params[kMaxParamsCount] = {};
    tensor *                 _src[kMaxTensorSrc]         = {};
    uint8_t *                _data                       = nullptr;
    std::atomic_bool         _has_modified               = false;

    DISABLE_COPY_AND_MOVE(tensor);
};

}  // namespace hexagon
