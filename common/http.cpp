#include "http.h"

// the factory state lives behind functions compiled into the library:
// WINDOWS_EXPORT_ALL_SYMBOLS exports functions but not data, so an inline
// variable would exist once per module and a substitution made by the
// executable would be invisible to the DLL

static common_http_client_ptr (*g_http_client_factory)(const std::string & url) = nullptr;

void common_http_client_set_factory(common_http_client_ptr (*factory)(const std::string & url)) {
    g_http_client_factory = factory;
}

common_http_client_ptr common_http_client_create(const std::string & url) {
    if (g_http_client_factory) {
        return g_http_client_factory(url);
    }
    return std::make_unique<common_http_client>(url);
}
