#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct clipboard_image {
    std::vector<uint8_t> bytes;
    std::string mime_type;  // "image/png", "image/jpeg", "image/webp", "image/gif"
};

// Read image data from the system clipboard.
// Returns std::nullopt when no image is available or the platform is unsupported.
// Blocks for ~100-300ms while the subprocess runs.
std::optional<clipboard_image> clipboard_read_image();
