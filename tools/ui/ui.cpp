#include "ui.h"

#include <string.h>

#if defined(LLAMA_BUILD_UI)
// auto generated files (see README.md for details)
#include "index.html.hpp"
#include "bundle.js.hpp"
#include "bundle.css.hpp"
#include "loading.html.hpp"

static const char g_bundle_css_etag[]   = "\"0x0\"";
static const char g_bundle_js_etag[]    = "\"0x0\"";
static const char g_index_html_etag[]   = "\"0x0\"";
static const char g_loading_html_etag[] = "\"0x0\"";

static const llama_ui_asset g_assets[] = {
    { "bundle.css", bundle_css,   bundle_css_len,   g_bundle_css_etag   },
    { "bundle.js",  bundle_js,    bundle_js_len,    g_bundle_js_etag    },
    { "index.html", index_html,   index_html_len,   g_index_html_etag   },
    { "loading.html", loading_html, loading_html_len, g_loading_html_etag },
};
#endif

const llama_ui_asset * llama_ui_find_asset(const char * name) {
#if defined(LLAMA_BUILD_UI)
    for (const auto & asset : g_assets) {
        if (strcmp(asset.name, name) == 0) {
            return &asset;
        }
    }
#else
    (void) name;
#endif
    return nullptr;
}
