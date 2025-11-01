#pragma once

#include <memory>

#include "common.hpp"
#include "dyn-lib-loader.hpp"
#ifdef GGML_QNN_ENABLE_HEXAGON_BACKEND
#    include <remote.h>
#else
// TODO: remove this when not needed

/**
 * @enum fastrpc_map_flags for fastrpc_mmap and fastrpc_munmap
 * @brief Types of maps with cache maintenance
 */
enum fastrpc_map_flags {
    /**
     * Map memory pages with RW- permission and CACHE WRITEBACK.
     * Driver will clean cache when buffer passed in a FastRPC call.
     * Same remote virtual address will be assigned for subsequent
     * FastRPC calls.
     */
    FASTRPC_MAP_STATIC,

    /** Reserved for compatibility with deprecated flag */
    FASTRPC_MAP_RESERVED,

    /**
     * Map memory pages with RW- permission and CACHE WRITEBACK.
     * Mapping tagged with a file descriptor. User is responsible for
     * maintenance of CPU and DSP caches for the buffer. Get virtual address
     * of buffer on DSP using HAP_mmap_get() and HAP_mmap_put() functions.
     */
    FASTRPC_MAP_FD,

    /**
     * Mapping delayed until user calls HAP_mmap() and HAP_munmap()
     * functions on DSP. User is responsible for maintenance of CPU and DSP
     * caches for the buffer. Delayed mapping is useful for users to map
     * buffer on DSP with other than default permissions and cache modes
     * using HAP_mmap() and HAP_munmap() functions.
     */
    FASTRPC_MAP_FD_DELAYED,

    /** Reserved for compatibility **/
    FASTRPC_MAP_RESERVED_4,
    FASTRPC_MAP_RESERVED_5,
    FASTRPC_MAP_RESERVED_6,
    FASTRPC_MAP_RESERVED_7,
    FASTRPC_MAP_RESERVED_8,
    FASTRPC_MAP_RESERVED_9,
    FASTRPC_MAP_RESERVED_10,
    FASTRPC_MAP_RESERVED_11,
    FASTRPC_MAP_RESERVED_12,
    FASTRPC_MAP_RESERVED_13,
    FASTRPC_MAP_RESERVED_14,
    FASTRPC_MAP_RESERVED_15,

    /**
     * This flag is used to skip CPU mapping,
     * otherwise behaves similar to FASTRPC_MAP_FD_DELAYED flag.
     */
    FASTRPC_MAP_FD_NOMAP,

    /** Update FASTRPC_MAP_MAX when adding new value to this enum **/
};

#endif

namespace common {

#ifdef _WIN32
constexpr const char * kQnnRpcLibName = "libcdsprpc.dll";
#else
constexpr const char * kQnnRpcLibName = "libcdsprpc.so";
#endif

class rpc_interface {
    using rpc_mem_init_t           = void (*)();
    using rpc_mem_deinit_t         = void (*)();
    using rpc_mem_alloc_t          = void * (*) (int heapid, uint32_t flags, int size);
    using rpc_mem_alloc2_t         = void * (*) (int heapid, uint32_t flags, size_t size);
    using rpc_mem_free_t           = void (*)(void * po);
    using rpc_mem_to_fd_t          = int (*)(void * po);
    using rpc_mem_fastrpc_mmap_t   = int (*)(int domain, int fd, void * addr, int offset, size_t length,
                                           enum fastrpc_map_flags flags);
    using rpc_mem_fastrpc_munmap_t = int (*)(int domain, int fd, void * addr, size_t length);
    using remote_handle_control_t  = int (*)(uint32_t req, void * data, uint32_t datalen);
    using remote_session_control_t = int (*)(uint32_t req, void * data, uint32_t datalen);

  public:
    rpc_interface(const std::string & rpc_lib_path = kQnnRpcLibName) {
        _rpc_lib_handle = dl_load(rpc_lib_path);
        if (!_rpc_lib_handle) {
            LOG_ERROR("failed to load %s, error: %s\n", rpc_lib_path.c_str(), dl_error());
            return;
        }

        _rpc_mem_init           = reinterpret_cast<rpc_mem_init_t>(dl_sym(_rpc_lib_handle, "rpcmem_init"));
        _rpc_mem_deinit         = reinterpret_cast<rpc_mem_deinit_t>(dl_sym(_rpc_lib_handle, "rpcmem_deinit"));
        _rpc_mem_alloc          = reinterpret_cast<rpc_mem_alloc_t>(dl_sym(_rpc_lib_handle, "rpcmem_alloc"));
        _rpc_mem_alloc2         = reinterpret_cast<rpc_mem_alloc2_t>(dl_sym(_rpc_lib_handle, "rpcmem_alloc2"));
        _rpc_mem_free           = reinterpret_cast<rpc_mem_free_t>(dl_sym(_rpc_lib_handle, "rpcmem_free"));
        _rpc_mem_to_fd          = reinterpret_cast<rpc_mem_to_fd_t>(dl_sym(_rpc_lib_handle, "rpcmem_to_fd"));
        _rpc_mem_fastrpc_mmap   = reinterpret_cast<rpc_mem_fastrpc_mmap_t>(dl_sym(_rpc_lib_handle, "fastrpc_mmap"));
        _rpc_mem_fastrpc_munmap = reinterpret_cast<rpc_mem_fastrpc_munmap_t>(dl_sym(_rpc_lib_handle, "fastrpc_munmap"));
        _remote_handle_control =
            reinterpret_cast<remote_handle_control_t>(dl_sym(_rpc_lib_handle, "remote_handle_control"));
        _remote_session_control =
            reinterpret_cast<remote_session_control_t>(dl_sym(_rpc_lib_handle, "remote_session_control"));
    }

    bool is_valid() const { return _rpc_lib_handle != nullptr; }

    bool is_alloc2_available() const { return _rpc_mem_alloc2 != nullptr; }

    void rpcmem_init() {
        if (_rpc_mem_init) {
            _rpc_mem_init();
        }
    }

    void rpcmem_deinit() {
        if (_rpc_mem_deinit) {
            _rpc_mem_deinit();
        }
    }

    void * rpcmem_alloc(int heapid, uint32_t flags, int size) {
        if (!is_valid()) {
            return nullptr;
        }

        return _rpc_mem_alloc(heapid, flags, size);
    }

    void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size) {
        if (!is_valid()) {
            return nullptr;
        }

        return _rpc_mem_alloc2(heapid, flags, size);
    }

    void rpcmem_free(void * buf) {
        if (is_valid()) {
            _rpc_mem_free(buf);
        }
    }

    int rpcmem_to_fd(void * buf) {
        int mem_fd = -1;
        if (is_valid()) {
            mem_fd = _rpc_mem_to_fd(buf);
        }

        return mem_fd;
    }

    int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
        if (!is_valid()) {
            return -1;
        }

        return _rpc_mem_fastrpc_mmap(domain, fd, addr, offset, length, flags);
    }

    int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
        if (!is_valid()) {
            return -1;
        }

        return _rpc_mem_fastrpc_munmap(domain, fd, addr, length);
    }

    int remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
        if (!is_valid()) {
            return -1;
        }

        return _remote_handle_control(req, data, datalen);
    }

    int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
        if (!is_valid()) {
            return -1;
        }

        return _remote_session_control(req, data, datalen);
    }

    ~rpc_interface() {
        if (_rpc_lib_handle) {
            if (_rpc_mem_deinit) {
                _rpc_mem_deinit();
            }

            dl_unload(_rpc_lib_handle);
        }
    }

  private:
    dl_handler_t             _rpc_lib_handle         = nullptr;
    rpc_mem_init_t           _rpc_mem_init           = nullptr;
    rpc_mem_deinit_t         _rpc_mem_deinit         = nullptr;
    rpc_mem_alloc_t          _rpc_mem_alloc          = nullptr;
    rpc_mem_alloc2_t         _rpc_mem_alloc2         = nullptr;
    rpc_mem_free_t           _rpc_mem_free           = nullptr;
    rpc_mem_to_fd_t          _rpc_mem_to_fd          = nullptr;
    rpc_mem_fastrpc_mmap_t   _rpc_mem_fastrpc_mmap   = nullptr;
    rpc_mem_fastrpc_munmap_t _rpc_mem_fastrpc_munmap = nullptr;
    remote_handle_control_t  _remote_handle_control  = nullptr;
    remote_session_control_t _remote_session_control = nullptr;

    rpc_interface(const rpc_interface &)             = delete;
    rpc_interface & operator=(const rpc_interface &) = delete;
    rpc_interface(rpc_interface &&)                  = delete;
    rpc_interface & operator=(rpc_interface &&)      = delete;
};

using rpc_interface_ptr = std::shared_ptr<rpc_interface>;

}  // namespace common
