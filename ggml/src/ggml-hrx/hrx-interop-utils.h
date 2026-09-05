#pragma once

#include "hrx_runtime.h"

#include <optional>
#include <string>

namespace ggml::hrx {

// Success has no payload; failure carries the diagnostic produced by HRX or
// by the caller. Keeping this distinct from an empty string makes status tests
// explicit at API boundaries.
using ErrorResult = std::optional<std::string>;

inline ErrorResult take_status(hrx_status_t status) {
    if (hrx_status_is_ok(status)) {
        return std::nullopt;
    }
    char *       message       = nullptr;
    size_t       length        = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) {
        hrx_status_ignore(format_status);
    }
    std::string result = message != nullptr ? std::string(message, length) : std::string("unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return result;
}

}  // namespace ggml::hrx
