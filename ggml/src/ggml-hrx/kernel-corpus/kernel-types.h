#pragma once

#include "status.h"

#include <cstdint>
#include <string>

namespace ggml::hrx {

enum class ResourceAccess : uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct VerificationResult {
    Status status;

    bool valid() const { return status.success(); }
};

}  // namespace ggml::hrx
