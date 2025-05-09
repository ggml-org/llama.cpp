#pragma once

#include <HAP_mem.h>
#include <qurt.h>

#include "hexagon_npu.h"
#include "util.hpp"

namespace hexagon {

constexpr const size_t kMaxTensorSrc = DEVICE_TENSOR_MAX_SRC;

class tensor {
  public:
    explicit tensor(const npu_device_tensor_config & info) noexcept : _info(info) {
        uint64 phy_address  = 0;
        void * mmap_address = nullptr;
        auto   ret          = HAP_mmap_get(_info.buffer_fd, &mmap_address, &phy_address);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to mmap tensor buffer: %d", (int) ret);
            return;
        }

        _data = static_cast<uint8_t *>(mmap_address);
        DEVICE_LOG_INFO("tensor(%p[%ldx%ldx%ldx%ld]), fd: %d, offset: %zu, mmap_address: %p, phy_address: 0x%lx\n",
                        (void *) this, (long) _info.ne[0], (long) _info.ne[1], (long) _info.ne[2], (long) _info.ne[3],
                        _info.buffer_fd, _info.offset, (void *) mmap_address, phy_address);
    }

    ~tensor() noexcept {
        auto ret = HAP_mmap_put(_info.buffer_fd);
        if (ret != AEE_SUCCESS) {
            DEVICE_LOG_ERROR("Failed to unmap tensor buffer: %d", (int) ret);
        }

        DEVICE_LOG_INFO("~tensor(%p) fd: %d", (void *) this, _info.buffer_fd);
    }

    void flush() {
        if (_data) {
            qurt_mem_cache_clean((qurt_addr_t) (_data + _info.offset), (qurt_size_t) _info.size,
                                 QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        }
    }

    bool set_src(size_t index, tensor * src) {
        if (index >= kMaxTensorSrc) {
            return false;
        }

        _src[index] = src;
        return true;
    }

    void set_op(npu_device_tensor_op op) { _info.op = op; }

    tensor * get_src(size_t index) const {
        if (index >= kMaxTensorSrc) {
            return nullptr;
        }

        return _src[index];
    }

    const npu_device_tensor_config & get_info() const { return _info; }

    const int64_t get_ne(size_t index) const { return _info.ne[index]; }

    const size_t get_nb(size_t index) const { return _info.nb[index]; }

    npu_device_tensor_op get_op() const { return _info.op; }

    npu_device_tensor_data_type get_type() const { return _info.type; }

    uint8_t * get_data() const { return _data + _info.offset; }

    bool is_valid() const { return _data != nullptr; }

  private:
    npu_device_tensor_config _info;
    tensor *                 _src[kMaxTensorSrc] = {};
    uint8_t *                _data               = nullptr;

    DISABLE_COPY_AND_MOVE(tensor);
};

}  // namespace hexagon
