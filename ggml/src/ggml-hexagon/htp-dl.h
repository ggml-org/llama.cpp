// DRY - This is a temp workaround to reuse GGML's DL functionality which is
//  static. Temporarely adding here 

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

void * dl_load_library(const fs::path & path);

void * dl_get_sym(dl_handle * handle, const char * name);

const char * dl_error();

#endif
