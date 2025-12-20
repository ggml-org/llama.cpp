// DRY - This is a temp workaround to reuse of GGML's DL functionality which is
//  static. Temporarely adding here until GGML core is refactored to allow
// reuse of dl_load_library and friends in backeds as well.

#pragma once

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
#include <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

dl_handle * dl_load_library(const fs::path & path);

void * dl_get_sym(dl_handle * handle, const char * name);

const char * dl_error() ;

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

void * dl_load_library(const fs::path & path);

void * dl_get_sym(dl_handle * handle, const char * name);

const char * dl_error();

#endif
