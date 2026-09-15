#include "http.h"

#undef NDEBUG
#include <cassert>

int main() {
    const auto parts = common_http_parse_url("https://example.com?download=1");

    assert(parts.scheme == "https");
    assert(parts.host == "example.com");
    assert(parts.port == 443);
    assert(parts.path == "?download=1");

    return 0;
}
