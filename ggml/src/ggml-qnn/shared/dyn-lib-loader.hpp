#pragma once

#ifdef __linux__
#    include <dlfcn.h>
#    include <fcntl.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <string>

namespace common {

#ifdef __linux__
typedef void * dl_handler_t;

inline dl_handler_t dl_load(const std::string & lib_path) {
    return dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
}

inline void * dl_sym(dl_handler_t handle, const std::string & symbol) {
    return dlsym(handle, symbol.c_str());
}

inline bool dl_unload(dl_handler_t handle) {
    return dlclose(handle) == 0;
}

inline const char * dl_error() {
    return dlerror();
}
#elif defined(_WIN32)
using dl_handler_t = HMODULE;

inline dl_handler_t dl_load(const std::string & lib_path) {
    // suppress error dialogs for missing DLLs
    auto old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    auto handle = LoadLibraryA(lib_path.c_str());  // TODO: use wstring version for unicode paths

    SetErrorMode(old_mode);
    return handle;
}

inline void * dl_sym(dl_handler_t handle, const std::string & symbol) {
    auto old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, symbol.c_str());

    SetErrorMode(old_mode);
    return p;
}

inline bool dl_unload(dl_handler_t handle) {
    FreeLibrary(handle);
    return true;
}

inline const char * dl_error() {
    // TODO: implement dl_error for Windows
    return nullptr;
}

#endif

template <typename Fn> Fn dl_sym_typed(dl_handler_t handle, const std::string & function_name) {
    return reinterpret_cast<Fn>(dl_sym(handle, function_name));
}

}  // namespace common
