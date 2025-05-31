
#pragma once

#include <limits>
#include <memory>

#include "common.hpp"
#include "dyn-lib-loader.hpp"
#include "rpc-interface.hpp"

namespace common {

class rpc_mem {
  public:
    rpc_mem() {
        auto interface = std::make_shared<rpc_interface>();
        if (!interface->is_valid()) {
            LOG_ERROR("failed to load rpcmem lib\n");
            return;
        }

        interface->rpcmem_init();
        _rpc_interface = interface;
        LOG_DEBUG("load rpcmem lib successfully\n");
    }

    explicit rpc_mem(rpc_interface_ptr interface) {
        if (!interface->is_valid()) {
            LOG_ERROR("failed to load rpcmem lib\n");
            return;
        }

        interface->rpcmem_init();
        _rpc_interface = interface;
        LOG_DEBUG("load rpcmem lib successfully\n");
    }

    ~rpc_mem() {
        if (!is_valid()) {
            LOG_DEBUG("rpc memory not initialized\n");
            return;
        }

        if (_rpc_interface) {
            _rpc_interface->rpcmem_deinit();
            _rpc_interface.reset();
        }

        LOG_DEBUG("unload rpcmem lib successfully\n");
    }

    bool is_valid() const { return (bool) _rpc_interface; }

    void * alloc(int heapid, uint32_t flags, size_t size) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return nullptr;
        }

        if (size > get_max_alloc_size()) {
            LOG_ERROR("rpc memory size %zu exceeds max alloc size %zu\n", size, get_max_alloc_size());
            return nullptr;
        }

        void * buf = nullptr;
        if (_rpc_interface->is_alloc2_available()) {
            LOG_DEBUG("rpcmem_alloc2 available, using it\n");
            buf = _rpc_interface->rpcmem_alloc2(heapid, flags, size);
        } else {
            LOG_DEBUG("rpcmem_alloc2 not available, using rpcmem_alloc\n");
            buf = _rpc_interface->rpcmem_alloc(heapid, flags, size);
        }

        if (!buf) {
            LOG_ERROR("failed to allocate rpc memory, size: %d MB\n", (int) (size / (1 << 20)));
            return nullptr;
        }

        LOG_DEBUG("rpc buffer allocated, heapid: %d, flags: 0x%x, size: %zu\n", heapid, flags, size);
        return buf;
    }

    void free(void * buf) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            _rpc_interface->rpcmem_free(buf);
        }
    }

    int to_fd(void * buf) {
        int mem_fd = -1;
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
        } else {
            mem_fd = _rpc_interface->rpcmem_to_fd(buf);
        }

        return mem_fd;
    }

    size_t get_max_alloc_size() {
        return _rpc_interface->is_alloc2_available() ? std::numeric_limits<size_t>::max() :
                                                       std::numeric_limits<int>::max();
    }

    int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return -1;
        }

        return _rpc_interface->fastrpc_mmap(domain, fd, addr, offset, length, flags);
    }

    int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
        if (!is_valid()) {
            LOG_ERROR("rpc memory not initialized\n");
            return -1;
        }

        return _rpc_interface->fastrpc_munmap(domain, fd, addr, length);
    }

  private:
    rpc_interface_ptr _rpc_interface;
};

using rpc_mem_ptr = std::shared_ptr<rpc_mem>;

}  // namespace common
