// sample drv interface

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
#include <codecvt>

#include "ggml-impl.h"

#include "htp-drv.h"

namespace fs = std::filesystem;

static std::string path_str(const fs::path & path) {
    std::string u8path;
    try {
#if defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        std::u8string u8str = path.u8string();
        u8path = std::string(reinterpret_cast<const char*>(u8str.c_str()));
#else
        // C++17: u8string() returns std::string
        u8path = path.u8string();
#endif
    } catch (...) {
    }
    return u8path;
}

#ifdef _WIN32

static std::string get_service_binary_path(std::wstring const& serviceName) {
    // Get a handle to the SCM database
    SC_HANDLE handleSCManager = OpenSCManagerW(NULL,                   // local computer
                                               NULL,                   // ServicesActive database
                                               STANDARD_RIGHTS_READ);  // standard read access
    if (nullptr == handleSCManager) {
        printf(
            "Failed to open SCManager which is required to access service configuration. "
            "Error: %lu",
            GetLastError());
        return std::string();
    }

    // Get a handle to the service
    SC_HANDLE handleService = OpenServiceW(handleSCManager,        // SCM database
                                           serviceName.c_str(),    // name of service
                                           SERVICE_QUERY_CONFIG);  // need query config access

    if (nullptr == handleService) {
        printf("Failed to open service %s which is required to query service information. Error: %lu",
                std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(serviceName).c_str(),
                GetLastError());
        CloseServiceHandle(handleSCManager);
        return std::string();
    }

    // Query the buffer size required by service configuration
    // When first calling it with null pointer and zero buffer size,
    // this function acts as a query function to return how many bytes it requires
    // and set error to ERROR_INSUFFICIENT_BUFFER.

    DWORD bufferSize;  // Store the size of buffer used as an output
    if (!QueryServiceConfigW(handleService, NULL, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        printf("Failed to query service configuration to get size of config object. Error: %lu",
                GetLastError());
        CloseServiceHandle(handleService);
        CloseServiceHandle(handleSCManager);
        return std::string();
    }
    // Get the configuration of the specified service
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    if (!QueryServiceConfigW(handleService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "Failed to query service configuration. Error: %lu", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(handleService);
        CloseServiceHandle(handleSCManager);
        return std::string();
    }

    // Read the driver file path
    std::wstring driverPath = std::wstring(serviceConfig->lpBinaryPathName);
    // Get the parent directory of the driver file
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(handleService);
    CloseServiceHandle(handleSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\windows)
    const std::wstring systemRootPlaceholder = L"\\SystemRoot";
    if (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
        printf(
            "The string pattern does not match. We expect that we can find [%s] "
            "in the beginning of the queried path [%s].",
            std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(systemRootPlaceholder).c_str(),
            std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(driverPath).c_str());
        return std::string();
    }

    // Replace \SystemRoot with an absolute path which is got from system ENV windir
    // ENV name used to get the root path of the system
    const std::wstring systemRootEnv = L"windir";

    // Query the number of wide charactors this variable requires
    DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
    if (numWords == 0) {
        printf("Failed to query the buffer size when calling GetEnvironmentVariableW().");
        return std::string();
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> systemRoot(numWords + 1);
    numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
    if (numWords == 0) {
        printf("Failed to read value from environment variables.");
        return std::string();
    }
    driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

    // driverPath is wide char string, we need to convert it to std::string
    // Assume to use UTF-8 wide string for conversion
    return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(driverPath);
}

static std::string get_dsp_driver_path() {
    return get_service_binary_path(L"qcnspmcdm");
}

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

static dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

static const char * dl_error() {
    return "";
}

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

static void * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);

    return handle;
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

static const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif

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

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t dspqueue_create_pfn = nullptr;
dspqueue_close_pfn_t dspqueue_close_pfn = nullptr;
dspqueue_export_pfn_t dspqueue_export_pfn = nullptr;
dspqueue_write_pfn_t dspqueue_write_pfn = nullptr;
dspqueue_read_pfn_t dspqueue_read_pfn = nullptr;

remote_handle64_open_pfn_t remote_handle64_open_pfn = nullptr;
remote_handle64_invoke_pfn_t remote_handle64_invoke_pfn = nullptr;
remote_handle64_close_pfn_t remote_handle64_close_pfn = nullptr;
remote_handle_control_pfn_t remote_handle_control_pfn = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t remote_session_control_pfn = nullptr;

void * rpcmem_alloc(int heapid, uint32_t flags, int size)
{
    return rpcmem_alloc_pfn(heapid, flags, size);
}

void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size)
{
    return rpcmem_alloc2_pfn(heapid, flags, size);
}

void rpcmem_free(void *po)
{
    return rpcmem_free_pfn(po);
}

int rpcmem_to_fd(void *po)
{
    return rpcmem_to_fd_pfn(po);
}

int fastrpc_mmap(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags)
{
    return fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

int fastrpc_munmap(int domain, int fd, void *addr, size_t length)
{
    return fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(int                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue)
{
    return dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue)
{
    return dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t *queue_id)
{
    return dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags,
                         uint32_t num_buffers,
                         struct dspqueue_buffer *buffers,
                         uint32_t message_length,
                         const uint8_t *message,
                         uint32_t timeout_us)
{
    return dspqueue_write_pfn(queue, flags, num_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t queue, uint32_t *flags,
                        uint32_t max_buffers, uint32_t *num_buffers,
                        struct dspqueue_buffer *buffers,
                        uint32_t max_message_length,
                        uint32_t *message_length, uint8_t *message,
                        uint32_t timeout_us)
{
    return dspqueue_read_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

int remote_handle64_open(const char* name, remote_handle64 *ph)
{
    return remote_handle64_open_pfn(name, ph);
}

int remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg *pra)
{
    return remote_handle64_invoke_pfn(h, dwScalars, pra);
}

int remote_handle64_close(remote_handle64 h)
{
    return remote_handle64_close_pfn(h);
}

int remote_handle_control(uint32_t req, void* data, uint32_t datalen)
{
    return remote_handle_control_pfn(req, data, datalen);
}

int remote_handle64_control(remote_handle64 h, uint32_t req, void* data, uint32_t datalen)
{
    return remote_handle64_control_pfn(h, req, data, datalen);
}

int remote_session_control(uint32_t req, void *data, uint32_t datalen)
{
    return remote_session_control_pfn(req, data, datalen);
}

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;

int htpdrv_init() {
    static bool initialized = false;
    int nErr = AEE_SUCCESS;
#ifdef _WIN32
    std::string drv_path = get_dsp_driver_path() + "\\" + "libcdsprpc.dll";
#else
    std::string drv_path = "libcdsprpc.so";
#endif
    if (initialized) {
        GGML_LOG_INFO("%s: HTP driver already loaded\n", __func__);
        goto bail;
    }
    GGML_LOG_INFO("%s: Loading driver %s\n", __func__, drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    if (!handle) {
        nErr = AEE_EUNABLETOLOAD;
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
        goto bail;
    }

#define DLSYM(DRV, TYPE, PTR, SYM)                                      \
    do {                                                                \
        PTR = (TYPE) dl_get_sym(DRV, #SYM);                             \
        if (nullptr == PTR) {                                           \
            nErr = AEE_EUNABLETOLOAD;                                   \
            GGML_LOG_ERROR("%s: failed to dlsym %s\n", __func__, #SYM); \
            goto bail;                                                  \
        }                                                               \
    } while (0)

    DLSYM(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc);
    DLSYM(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2);
    DLSYM(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free);
    DLSYM(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd);
    DLSYM(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap);
    DLSYM(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap);
    DLSYM(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create);
    DLSYM(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close);
    DLSYM(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export);
    DLSYM(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write);
    DLSYM(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read);
    DLSYM(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open);
    DLSYM(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke);
    DLSYM(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control);
    DLSYM(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control);
    DLSYM(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control);
    DLSYM(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized = true;

bail:
    return nErr;
}
