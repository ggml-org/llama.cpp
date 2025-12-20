#include "htp-dl.h"

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

//struct dl_handle_deleter {
//    void operator()(HMODULE handle) {
//        FreeLibrary(handle);
//    }
//};

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

void * dl_load_library(const fs::path & path) {
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