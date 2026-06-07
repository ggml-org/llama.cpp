#pragma once

#include <stddef.h>

struct llama_ui_asset {
    const char *          name;
    const unsigned char * data;
    size_t                size;
    const char *          etag;
};

#if defined(LLAMA_BUILD_UI)
#define LLAMA_UI_HAS_ASSETS 1
#endif

const llama_ui_asset * llama_ui_find_asset(const char * name);
