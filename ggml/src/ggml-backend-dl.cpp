#include "ggml-backend-dl.h"

#include <memory>
#include <string>

#ifdef _WIN32

static thread_local DWORD dl_error_code = ERROR_SUCCESS;

dl_handle * dl_load_library(const fs::path & path) {
    const std::wstring path_w = path.wstring();

    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path_w.c_str());
    dl_error_code = handle ? ERROR_SUCCESS : GetLastError();

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);
    dl_error_code = p ? ERROR_SUCCESS : GetLastError();

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    static thread_local std::string text;
    const DWORD code = dl_error_code;
    dl_error_code = ERROR_SUCCESS;
    text.clear();

    if (code == ERROR_SUCCESS) {
        return text.c_str();
    }

    text = "Windows error " + std::to_string(code);
    wchar_t * message = nullptr;
    DWORD length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr, code, 0, reinterpret_cast<LPWSTR>(&message), 0, nullptr);
    std::unique_ptr<wchar_t, decltype(&LocalFree)> message_ptr(message, LocalFree);
    while (length > 0 && (message[length - 1] == L'\r' || message[length - 1] == L'\n')) {
        --length;
    }
    if (length > 0) {
        const int size = WideCharToMultiByte(CP_UTF8, 0, message, length, nullptr, 0, nullptr, nullptr);
        if (size > 0) {
            std::string utf8(size, '\0');
            if (WideCharToMultiByte(CP_UTF8, 0, message, length, &utf8[0], size, nullptr, nullptr) == size) {
                text += ": " + utf8;
            }
        }
    }

    return text.c_str();
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
