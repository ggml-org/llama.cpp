#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#include "ggml-impl.h"
#include "htp-drv.h"
#include "libdl.h"

#include <domain.h>

extern int opt_verbose;

#ifndef HEX_VERBOSE
#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)
#endif

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
typedef AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *num_buffers,
                                                 struct dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
typedef int (*fastrpc_mmap_pfn_t)(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags);
typedef int (*fastrpc_munmap_pfn_t)(int domain, int fd, void *addr, size_t length);

typedef int (*remote_handle64_open_pfn_t)(const char* name, remote_handle64 *ph);
typedef int (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
typedef int (*remote_handle64_close_pfn_t)(remote_handle64 h);
typedef int (*remote_handle_control_pfn_t)(uint32_t req, void* data, uint32_t datalen);
typedef int (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
typedef int (*remote_session_control_pfn_t)(uint32_t req, void *data, uint32_t datalen);
typedef int (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

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
#ifdef _WIN32
    if (timeout_us == 0) {
        return dspqueue_read_noblock_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
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

HTPDRV_API int remote_system_request(system_req_payload * req) {
    if (!remote_system_request_pfn) {
        return AEE_EUNSUPPORTEDAPI;
    }
    return remote_system_request_pfn(req);
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
    std::string drv_path = "libcdsprpc.so";
#endif
    if (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        return AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    if (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
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
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, false);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, false);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, false);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, false);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, false);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, false);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, false);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, true);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = true;

    return AEE_SUCCESS;
}

domain * htpdrv_get_domain(int domain_id) {
    int i    = 0;
    int size = sizeof(supported_domains) / sizeof(domain);

    for (i = 0; i < size; i++) {
        if (supported_domains[i].id == domain_id) {
            return &supported_domains[i];
        }
    }

    return NULL;
}

int htpdrv_get_arch(int domain, int * arch) {
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

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (int) ((val >> 4) * 10 + (val & 0x0f));
    return 0;
}

// Directory of the HTP library as a path with a trailing slash, or empty if it cannot be resolved.
// Checks GGML_HEXAGON_LIBHTP_DIR first, then the module directory and ../lib.
// On Windows, non-ASCII directories or paths with spaces use the 8.3 short name.
std::string htp_lib_dir() {
    namespace fs = std::filesystem;

    std::vector<fs::path> candidates;

    const char * env_dir = getenv("GGML_HEXAGON_LIBHTP_DIR");
    if (env_dir && *env_dir) {
        std::error_code ec;
        fs::path p = fs::absolute(env_dir, ec);
        if (!ec) {
            candidates.push_back(p);
        }
    }

#ifdef _WIN32
    HMODULE hm = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR) &htp_lib_dir, &hm)) {
        std::wstring buf(32768, L'\0');
        DWORD n = GetModuleFileNameW(hm, buf.data(), (DWORD) buf.size());
        if (n > 0 && n < buf.size()) {
            buf.resize(n);
            std::error_code ec;
            fs::path mod_path = fs::canonical(buf, ec);
            if (ec) {
                mod_path = fs::absolute(buf, ec);
            }
            const fs::path mod_dir = mod_path.parent_path();
            candidates.push_back(mod_dir);
            candidates.push_back(mod_dir / "lib");
            candidates.push_back(mod_dir / ".." / "lib");
        }
    }
#elif defined(__linux__) || defined(__ANDROID__)
    Dl_info info;
    if (dladdr((void *) &htp_lib_dir, &info) && info.dli_fname) {
        std::error_code ec;
        fs::path mod_path = fs::canonical(info.dli_fname, ec);
        if (ec) {
            mod_path = fs::absolute(info.dli_fname, ec);
        }
        const fs::path mod_dir = mod_path.parent_path();
        candidates.push_back(mod_dir);
        candidates.push_back(mod_dir / "lib");
        candidates.push_back(mod_dir / ".." / "lib");
    }
#endif

    for (size_t i = 0; i < candidates.size(); ++i) {
        const fs::path & d = candidates[i];
        std::error_code ec;
        if (!fs::is_directory(d, ec)) {
            continue;
        }

        // For auto-discovered directories, verify it contains an HTP library
        const bool is_env = (env_dir && i == 0);
        if (!is_env) {
            bool found = false;
            for (const auto & entry : fs::directory_iterator(d, ec)) {
                if (ec) {
                    break;
                }
                const std::string fn = entry.path().filename().string();
                if (fn.rfind("libggml-htp-", 0) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                continue;
            }
        }

        std::error_code ec_canon;
        fs::path abs_d = fs::canonical(d, ec_canon);
        if (ec_canon) {
            abs_d = fs::absolute(d, ec_canon);
        }

#ifdef _WIN32
        std::wstring dir = abs_d.wstring();
        auto is_ascii = [](const std::wstring & s) {
            return std::all_of(s.begin(), s.end(), [](wchar_t c) { return c < 0x80; });
        };
        auto has_spaces = [](const std::wstring & s) {
            return std::any_of(s.begin(), s.end(), [](wchar_t c) { return c == L' '; });
        };

        if (!is_ascii(dir) || has_spaces(dir)) {
            DWORD m = GetShortPathNameW(dir.c_str(), nullptr, 0);
            if (m > 0) {
                std::wstring sp(m, L'\0');
                DWORD res = GetShortPathNameW(dir.c_str(), sp.data(), (DWORD) sp.size());
                if (res > 0 && res < sp.size()) {
                    sp.resize(res);
                    if (is_ascii(sp)) {
                        dir = sp;
                    }
                }
            }
            if (!is_ascii(dir)) {
                continue;
            }
        }
        if (dir.back() != L'\\' && dir.back() != L'/') {
            dir += L'\\';
        }
        return std::string(dir.begin(), dir.end());
#else
        std::string dir = abs_d.string();
        if (dir.back() != '/') {
            dir += '/';
        }
        return dir;
#endif
    }

    return {};
}

int htpdrv_set_dsp_search_path(int domain_id, const std::string & path, const char * dev_name) {
    if (path.empty()) {
        return AEE_SUCCESS;
    }

    std::string clean_path = path;
    while (clean_path.size() > 1 && (clean_path.back() == '/' || clean_path.back() == '\\')) {
        clean_path.pop_back();
    }

    fastrpc_manage_dsp_lib_path p {};
    p.effec_domain_id = domain_id;
    p.type            = FASTRPC_SET_PATH;
    p.path            = const_cast<char *>(clean_path.c_str());
    p.pathlen         = clean_path.size() + 1;

    int err = remote_session_control(FASTRPC_MANAGE_DSP_LIBRARY_PATH, (void *) &p, sizeof(p));
    if (err != AEE_SUCCESS && domain_id != -1) {
        p.effec_domain_id = -1;
        err = remote_session_control(FASTRPC_MANAGE_DSP_LIBRARY_PATH, (void *) &p, sizeof(p));
    }

    const char * name = dev_name ? dev_name : "driver";
    if (err == AEE_SUCCESS) {
        GGML_LOG_INFO("ggml-hex: %s set DSP search path (domain %d) to %s\n", name, p.effec_domain_id, clean_path.c_str());
        return AEE_SUCCESS;
    }

    HEX_VERBOSE("ggml-hex: %s FASTRPC_MANAGE_DSP_LIBRARY_PATH failed (0x%x)\n", name, err);
    return err;
}

void htpdrv_set_dsp_env_path(const std::string & path, const char * dev_name) {
    if (path.empty()) {
        return;
    }

    std::string clean_path = path;
    while (clean_path.size() > 1 && (clean_path.back() == '/' || clean_path.back() == '\\')) {
        clean_path.pop_back();
    }

    const char * name = dev_name ? dev_name : "driver";
    HEX_VERBOSE("ggml-hex: %s setting DSP search path via process env to %s\n", name, clean_path.c_str());

    static const char * const env_names[] = {
        "ADSP_LIBRARY_PATH",
        "CDSP_LIBRARY_PATH",
        "CDSP1_LIBRARY_PATH",
        "CDSP2_LIBRARY_PATH",
        "CDSP3_LIBRARY_PATH",
    };
    for (const char * env_name : env_names) {
#ifdef _WIN32
        SetEnvironmentVariableA(env_name, clean_path.c_str());
#else
        setenv(env_name, clean_path.c_str(), 1);
#endif
    }
}
