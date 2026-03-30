// sample drv interface

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"

#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#ifdef __ANDROID__
#    include <android/dlext.h>
#    include <elf.h>
#    include <link.h>
// android_get_exported_namespace is in libdl_android.so but not in public NDK headers
typedef struct android_namespace_t * (*android_get_exported_namespace_pfn_t)(const char *);

// Resolve a symbol from an already-loaded library by parsing its ELF dynamic
// symbol table via dl_iterate_phdr.  This bypasses dlsym namespace restrictions.
struct elf_sym_lookup {
    const char * library;
    const char * symbol;
    void *       result;
};

static int elf_sym_lookup_cb(struct dl_phdr_info * info, size_t /*size*/, void * data) {
    auto * ctx = static_cast<elf_sym_lookup *>(data);
    if (!info->dlpi_name || !strstr(info->dlpi_name, ctx->library)) {
        return 0;  // not the library we want
    }

    // Find PT_DYNAMIC segment
    const ElfW(Dyn) * dyn_start = nullptr;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            dyn_start = reinterpret_cast<const ElfW(Dyn) *>(
                info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            break;
        }
    }
    if (!dyn_start) return 0;

    // Extract symbol table, string table, and hash table from dynamic section.
    // d_un.d_ptr values may be unrelocated VAs — add load bias if they look
    // like small offsets (i.e. less than the library's base address).
    const ElfW(Sym) * symtab = nullptr;
    const char *       strtab = nullptr;
    const ElfW(Word) * hashtab = nullptr;
    const uint32_t *   gnu_hash = nullptr;
    auto dyn_ptr = [&](ElfW(Addr) val) -> uintptr_t {
        return (val < static_cast<ElfW(Addr)>(info->dlpi_addr))
             ? (info->dlpi_addr + val) : val;
    };

    for (const ElfW(Dyn) * d = dyn_start; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_SYMTAB:   symtab   = reinterpret_cast<const ElfW(Sym) *>(dyn_ptr(d->d_un.d_ptr)); break;
            case DT_STRTAB:   strtab   = reinterpret_cast<const char *>(dyn_ptr(d->d_un.d_ptr));      break;
            case DT_HASH:     hashtab  = reinterpret_cast<const ElfW(Word) *>(dyn_ptr(d->d_un.d_ptr)); break;
            case DT_GNU_HASH: gnu_hash = reinterpret_cast<const uint32_t *>(dyn_ptr(d->d_un.d_ptr));   break;
        }
    }
    if (!symtab || !strtab) return 0;

    // Determine number of symbols from DT_HASH (nchain == nsyms)
    size_t nsyms = 0;
    if (hashtab) {
        nsyms = hashtab[1];  // hashtab[0]=nbuckets, hashtab[1]=nchain
    } else if (gnu_hash) {
        // Approximate: scan GNU hash to find max symndx
        uint32_t nbuckets  = gnu_hash[0];
        uint32_t symoffset = gnu_hash[1];
        uint32_t bloom_sz  = gnu_hash[2];
        const uint32_t * buckets =
            reinterpret_cast<const uint32_t *>(&gnu_hash[4] + bloom_sz * (sizeof(ElfW(Addr))/4));
        uint32_t max_idx = symoffset;
        for (uint32_t i = 0; i < nbuckets; i++) {
            if (buckets[i] > max_idx) max_idx = buckets[i];
        }
        // Walk chain from max_idx until we find the end marker (LSB set)
        const uint32_t * chain = buckets + nbuckets - symoffset;
        while (!(chain[max_idx] & 1)) max_idx++;
        nsyms = max_idx + 1;
    } else {
        nsyms = 256;  // small fallback
    }

    for (size_t i = 0; i < nsyms; i++) {
        if (symtab[i].st_shndx == SHN_UNDEF) continue;
        if (strcmp(strtab + symtab[i].st_name, ctx->symbol) == 0) {
            ctx->result = reinterpret_cast<void *>(
                info->dlpi_addr + symtab[i].st_value);
            return 1;  // found, stop iteration
        }
    }
    return 0;
}

static void * elf_find_symbol(const char * lib_name, const char * sym_name) {
    elf_sym_lookup ctx { lib_name, sym_name, nullptr };
    dl_iterate_phdr(elf_sym_lookup_cb, &ctx);
    return ctx.result;
}
#endif
#include "ggml-impl.h"
#include "htp-drv.h"
#include "libdl.h"

#include <domain.h>

//
// Driver API types
//

typedef void * (*rpcmem_alloc_pfn_t)(int heapid, uint32_t flags, int size);
typedef void * (*rpcmem_alloc2_pfn_t)(int heapid, uint32_t flags, size_t size);
typedef void   (*rpcmem_free_pfn_t)(void * po);
typedef int    (*rpcmem_to_fd_pfn_t)(void * po);

typedef AEEResult (*dspqueue_create_pfn_t)(int                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           void *              callback_context,
                                           dspqueue_t *        queue);
typedef AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
typedef AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
typedef AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t num_buffers,
                                          struct dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          const uint8_t *message,
                                          uint32_t timeout_us);
typedef AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *num_buffers,
                                         struct dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);

typedef int (*fastrpc_mmap_pfn_t)(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags);
typedef int (*fastrpc_munmap_pfn_t)(int domain, int fd, void *addr, size_t length);

typedef int (*remote_handle64_open_pfn_t)(const char* name, remote_handle64 *ph);
typedef int (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
typedef int (*remote_handle64_close_pfn_t)(remote_handle h);
typedef int (*remote_handle_control_pfn_t)(uint32_t req, void* data, uint32_t datalen);
typedef int (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
typedef int (*remote_session_control_pfn_t)(uint32_t req, void *data, uint32_t datalen);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t dspqueue_create_pfn = nullptr;
dspqueue_close_pfn_t  dspqueue_close_pfn  = nullptr;
dspqueue_export_pfn_t dspqueue_export_pfn = nullptr;
dspqueue_write_pfn_t  dspqueue_write_pfn  = nullptr;
dspqueue_read_pfn_t   dspqueue_read_pfn   = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;

//
// Driver API
//

void * rpcmem_alloc(int heapid, uint32_t flags, int size) {
    return rpcmem_alloc_pfn(heapid, flags, size);
}

void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size) {
    if (rpcmem_alloc2_pfn) {
        return rpcmem_alloc2_pfn(heapid, flags, size);
    } else {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        return rpcmem_alloc_pfn(heapid, flags, size);
    }
}

void rpcmem_free(void * po) {
    return rpcmem_free_pfn(po);
}

int rpcmem_to_fd(void * po) {
    return rpcmem_to_fd_pfn(po);
}

HTPDRV_API int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    return fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
    return fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(int                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue) {
    return dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    return dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    return dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 num_buffers,
                         struct dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         const uint8_t *          message,
                         uint32_t                 timeout_us) {
    return dspqueue_write_pfn(queue, flags, num_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               num_buffers,
                        struct dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
    return dspqueue_read_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API int remote_handle64_open(const char * name, remote_handle64 * ph) {
    return remote_handle64_open_pfn(name, ph);
}

HTPDRV_API int remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    return remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API int remote_handle64_close(remote_handle64 h) {
    return remote_handle64_close_pfn(h);
}

HTPDRV_API int remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    return remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    return remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    return remote_session_control_pfn(req, data, datalen);
}

#ifdef _WIN32

static std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    if (wstr.empty()) {
        return result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (int) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    if (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        throw std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    int bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (int) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    if (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        throw std::runtime_error("Wstring conversion failed");
    }
    return result;
}

static std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
    if (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        return result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    if (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        return result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    if (!QueryServiceConfigW(schService, NULL, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        return result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    if (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        return result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lpBinaryPathName);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    const std::wstring systemRootPlaceholder = L"\\SystemRoot";
    if (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        return result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    const std::wstring systemRootEnv = L"windir";

    // Query the number of wide characters this variable requires
    DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
    if (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        return result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> systemRoot(numWords + 1);
    numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
    if (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        return result;
    }
    driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

    return wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

int htpdrv_init() {
    static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    static bool initialized = false;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#else
    const char * vendor_lib = getenv("GGML_HEXAGON_VENDOR_LIB");
    std::string drv_path = (vendor_lib && vendor_lib[0] != '\0')
                         ? vendor_lib
                         : "libcdsprpc.so";
#endif
    if (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        return AEE_SUCCESS;
    }

    // When loading a vendor lib from a custom path, try android_dlopen_ext with
    // the "sphal" namespace which has vendor/lib64 in its search paths and links
    // to vndk for transitive deps (libhidlbase, libcutils, libhardware, etc.).
    dl_handle_ptr handle { nullptr };
#ifdef __ANDROID__
    if (vendor_lib && vendor_lib[0] != '\0') {
        // Load directly from /vendor/lib64 via sphal namespace which has
        // vendor search paths and vndk linkage for transitive deps.
        drv_path = "libcdsprpc.so";
        GGML_LOG_INFO("ggml-hex: Trying sphal namespace for vendor FastRPC lib\n");

        // Resolve android_get_exported_namespace by parsing ELF symtab of the
        // already-loaded libdl_android.so — this bypasses dlsym namespace
        // restrictions that prevent apps from calling private linker APIs.
        auto get_ns = (android_get_exported_namespace_pfn_t)
            elf_find_symbol("libdl_android.so", "android_get_exported_namespace");

        if (get_ns) {
            GGML_LOG_INFO("ggml-hex: resolved android_get_exported_namespace via ELF parsing\n");
            struct android_namespace_t * sphal_ns = get_ns("sphal");
            if (sphal_ns) {
                android_dlextinfo extinfo = {};
                extinfo.flags = ANDROID_DLEXT_USE_NAMESPACE;
                extinfo.library_namespace = sphal_ns;
                void * h = android_dlopen_ext(drv_path.c_str(),
                                              RTLD_NOW | RTLD_LOCAL, &extinfo);
                if (h) {
                    handle.reset(static_cast<dl_handle *>(h));
                    GGML_LOG_INFO("ggml-hex: Loaded vendor lib via sphal namespace\n");
                } else {
                    GGML_LOG_WARN("ggml-hex: sphal dlopen failed: %s\n", dlerror());
                }
            } else {
                GGML_LOG_WARN("ggml-hex: sphal namespace not found\n");
            }
        } else {
            GGML_LOG_WARN("ggml-hex: could not resolve android_get_exported_namespace\n");
        }
    }
#endif

    if (!handle) {
        GGML_LOG_INFO("ggml-hex: Loading driver %s (standard dlopen)\n", drv_path.c_str());
        fs::path path{ drv_path.c_str() };
        handle.reset(dl_load_library(path));
    }
    if (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", drv_path.c_str(), dl_error());
        return AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, symbol, ignore)                               \
    do {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        if (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            return AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } while (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, false);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, true);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, false);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, false);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, false);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, false);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, false);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, false);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, false);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, false);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, false);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, false);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, false);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, false);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, false);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, false);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, false);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = true;

    return AEE_SUCCESS;
}

domain * get_domain(int domain_id) {
    int i    = 0;
    int size = sizeof(supported_domains) / sizeof(domain);

    for (i = 0; i < size; i++) {
        if (supported_domains[i].id == domain_id) {
            return &supported_domains[i];
        }
    }

    return NULL;
}

int get_hex_arch_ver(int domain, int * arch) {
    if (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        return AEE_EUNSUPPORTEDAPI;
    }

    struct remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    int err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, sizeof(arch_ver));
    if ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        return AEE_EUNSUPPORTEDAPI;
    }

    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        return err;
    }

    switch (arch_ver.capability & 0xff) {
        case 0x68:
            *arch = 68;
            return 0;
        case 0x69:
            *arch = 69;
            return 0;
        case 0x73:
            *arch = 73;
            return 0;
        case 0x75:
            *arch = 75;
            return 0;
        case 0x79:
            *arch = 79;
            return 0;
        case 0x81:
            *arch = 81;
            return 0;
    }
    return -1;
}
