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

RpcMemAllocFn_t htpdrv_rpcmem_alloc = nullptr;
RpcMemAllocFn_t htpdrv_rpcmem_alloc2 = nullptr;
RpcMemFreeFn_t htpdrv_rpcmem_free = nullptr;
RpcMemToFdFn_t htpdrv_rpcmem_to_fd = nullptr;
FastRpcMmapFn_t htpdrv_fastrpc_mmap = nullptr;
FastRpcMunmapFn_t htpdrv_fastrpc_munmap = nullptr;

DspQueueCreateFn_t htpdrv_dspqueue_create = nullptr;
DspQueueCloseFn_t htpdrv_dspqueue_close = nullptr;
DspQueueExportFn_t htpdrv_dspqueue_export = nullptr;
DspQueueWriteFn_t htpdrv_dspqueue_write = nullptr;
DspQueueReadFn_t htpdrv_dspqueue_read = nullptr;

RemoteHandle64OpenFn_t htpdrv_remote_handle64_open = nullptr;
RemoteHandle64InvokeFn_t htpdrv_remote_handle64_invoke = nullptr;
RemoteHandleControlFn_t htpdrv_remote_handle_control = nullptr;
RemoteHandle64ControlFn_t htpdrv_remote_handle64_control = nullptr;
RemoteHandle64CloseFn_t htpdrv_remote_handle64_close = nullptr;

RemoteSessionControlFn_t htpdrv_remote_session_control = nullptr;

RemoteSystemRequestFn_t htpdrv_remote_system_request = nullptr;

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

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;

#define DLSYM(DRV, TYPE, PTR, SYM)                                      \
    do {                                                                \
        PTR = (TYPE) dl_get_sym(DRV, #SYM);                             \
        if (nullptr == PTR) {                                           \
            GGML_LOG_ERROR("%s: failed to dlsym %s\n", __func__, #SYM); \
        }                                                               \
    } while (0)

int htpdrv_initialize() {
#ifdef _WIN32
    std::string drv_path = get_dsp_driver_path() + "\\" + "libcdsprpc.dll";
#else
    std::string drv_path = "libcdsprpc.so";
#endif
    GGML_LOG_INFO("%s: Loading driver %s\n", __func__, drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    if (!handle) {
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
        return -1;
    }

    DLSYM(handle.get(), RpcMemAllocFn_t, htpdrv_rpcmem_alloc, rpcmem_alloc);
    DLSYM(handle.get(), RpcMemAllocFn2_t, htpdrv_rpcmem_alloc2, rpcmem_alloc2);
    DLSYM(handle.get(), RpcMemFreeFn_t, htpdrv_rpcmem_free, rpcmem_free);
    DLSYM(handle.get(), RpcMemToFdFn_t, htpdrv_rpcmem_to_fd, rpcmem_to_fd);
    DLSYM(handle.get(), FastRpcMmapFn_t, htpdrv_fastrpc_mmap, fastrpc_mmap);
    DLSYM(handle.get(), FastRpcMunmapFn_t, htpdrv_fastrpc_munmap, fastrpc_munmap);
    DLSYM(handle.get(), DspQueueCreateFn_t, htpdrv_dspqueue_create, dspqueue_create);
    DLSYM(handle.get(), DspQueueCloseFn_t, htpdrv_dspqueue_close, dspqueue_close);
    DLSYM(handle.get(), DspQueueExportFn_t, htpdrv_dspqueue_export, dspqueue_export);
    DLSYM(handle.get(), DspQueueWriteFn_t, htpdrv_dspqueue_write, dspqueue_write);
    DLSYM(handle.get(), DspQueueReadFn_t, htpdrv_dspqueue_read, dspqueue_read);
    DLSYM(handle.get(), RemoteHandle64OpenFn_t, htpdrv_remote_handle64_open, remote_handle64_open);
    DLSYM(handle.get(), RemoteHandle64InvokeFn_t, htpdrv_remote_handle64_invoke, remote_handle64_invoke);
    DLSYM(handle.get(), RemoteHandleControlFn_t, htpdrv_remote_handle_control, remote_handle_control);
    DLSYM(handle.get(), RemoteHandle64ControlFn_t, htpdrv_remote_handle64_control, remote_handle64_control);
    DLSYM(handle.get(), RemoteSessionControlFn_t, htpdrv_remote_session_control, remote_session_control);
    DLSYM(handle.get(), RemoteHandle64CloseFn_t, htpdrv_remote_handle64_close, remote_handle64_close);
    DLSYM(handle.get(), RemoteSystemRequestFn_t, htpdrv_remote_system_request, remote_system_request);

    lib_cdsp_rpc_handle = std::move(handle);

    return 0;
}
