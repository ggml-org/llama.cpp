#pragma once

#include "base64.hpp"
#include "console.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// stb_image for JPEG/GIF/BMP/WebP → RGBA decoding (already compiled in mtmd)
#include "stb/stb_image.h"

enum class image_protocol { NONE, KITTY, ITERM2 };

// Detect which terminal image protocol is supported (cached)
inline image_protocol detect_image_protocol() {
    static image_protocol cached = []() -> image_protocol {
#if defined(_WIN32)
        return image_protocol::NONE;
#else
        if (!isatty(STDERR_FILENO)) {
            return image_protocol::NONE;
        }

        auto env_set = [](const char * name) -> bool {
            const char * val = std::getenv(name);
            return val && val[0] != '\0';
        };
        auto env_eq = [](const char * name, const char * expected) -> bool {
            const char * val = std::getenv(name);
            if (!val) return false;
            std::string v(val);
            // case-insensitive compare
            std::string e(expected);
            for (auto & c : v) c = (char) std::tolower((unsigned char) c);
            for (auto & c : e) c = (char) std::tolower((unsigned char) c);
            return v == e;
        };

        // Kitty graphics protocol
        if (env_set("KITTY_WINDOW_ID") || env_eq("TERM_PROGRAM", "kitty")) {
            return image_protocol::KITTY;
        }
        if (env_eq("TERM_PROGRAM", "ghostty") || env_set("GHOSTTY_RESOURCES_DIR")) {
            return image_protocol::KITTY;
        }
        if (env_set("WEZTERM_PANE") || env_eq("TERM_PROGRAM", "wezterm")) {
            return image_protocol::KITTY;
        }

        // iTerm2 inline image protocol
        if (env_set("ITERM_SESSION_ID") || env_eq("TERM_PROGRAM", "iterm.app")) {
            return image_protocol::ITERM2;
        }

        return image_protocol::NONE;
#endif
    }();
    return cached;
}

// Encode image as Kitty graphics protocol escape sequence
// fmt: 100 = PNG file, 32 = raw RGBA pixels (requires s= and v= for dimensions)
inline std::string encode_kitty(const std::string & b64, int cols, int fmt = 100,
                                int img_w = 0, int img_h = 0) {
    static constexpr size_t CHUNK_SIZE = 4096;
    std::string result;
    result.reserve(b64.size() + 256);

    // Build the header params
    std::string header = "a=T,f=" + std::to_string(fmt) + ",q=2,c=" + std::to_string(cols);
    if (fmt == 32 && img_w > 0 && img_h > 0) {
        header += ",s=" + std::to_string(img_w) + ",v=" + std::to_string(img_h);
    }

    if (b64.size() <= CHUNK_SIZE) {
        // Single chunk
        result += "\x1b_G";
        result += header;
        result += ";";
        result += b64;
        result += "\x1b\\";
    } else {
        // Multi-chunk
        size_t offset = 0;
        bool first = true;
        while (offset < b64.size()) {
            size_t remaining = b64.size() - offset;
            size_t chunk_len = std::min(remaining, CHUNK_SIZE);
            bool last = (offset + chunk_len >= b64.size());

            if (first) {
                result += "\x1b_G";
                result += header;
                result += ",m=1;";
                first = false;
            } else if (last) {
                result += "\x1b_Gm=0;";
            } else {
                result += "\x1b_Gm=1;";
            }

            result.append(b64, offset, chunk_len);
            result += "\x1b\\";
            offset += chunk_len;
        }
    }
    return result;
}

// Encode image as iTerm2 inline image protocol escape sequence
inline std::string encode_iterm2(const std::string & b64, int cols) {
    std::string result;
    result.reserve(b64.size() + 64);
    result += "\x1b]1337;File=inline=1;width=";
    result += std::to_string(cols);
    result += ":";
    result += b64;
    result += "\x07";
    return result;
}

// Display an image inline in the terminal (no-op if unsupported)
inline void render_image_to_terminal(const uint8_t * data, size_t len,
                                     const std::string & mime) {
    auto proto = detect_image_protocol();
    if (proto == image_protocol::NONE) return;

    static constexpr int IMAGE_PREVIEW_COLS = 60;

    std::string seq;

    if (proto == image_protocol::KITTY) {
        // Decode to RGBA via stb_image, downscale if needed, send as f=32
        int w = 0, h = 0, channels = 0;
        // Check dimensions before decoding to avoid multi-GB allocations on decompression bombs
        if (!stbi_info_from_memory(data, (int) len, &w, &h, &channels)) return;
        if ((size_t) w * h > 16 * 1024 * 1024) return;  // skip images > 16 megapixels

        stbi_uc * pixels = stbi_load_from_memory(data, (int) len, &w, &h, &channels, 4);
        if (!pixels) return;

        // Downscale large images to max 500px longest side (nearest-neighbor)
        static constexpr int MAX_PX = 500;
        std::vector<uint8_t> resized;
        if (w > MAX_PX || h > MAX_PX) {
            float s = (float) MAX_PX / (float) std::max(w, h);
            int nw = std::max(1, (int)(w * s));
            int nh = std::max(1, (int)(h * s));
            resized.resize((size_t) nw * nh * 4);
            for (int y = 0; y < nh; y++) {
                for (int x = 0; x < nw; x++) {
                    std::memcpy(&resized[((size_t) y * nw + x) * 4],
                                &pixels[((size_t)(y * h / nh) * w + x * w / nw) * 4], 4);
                }
            }
            stbi_image_free(pixels);
            pixels = nullptr;
            w = nw;
            h = nh;
        }

        const uint8_t * px = pixels ? pixels : resized.data();
        std::string b64 = base64::encode(reinterpret_cast<const char *>(px), (size_t) w * h * 4);
        if (pixels) stbi_image_free(pixels);

        seq = encode_kitty(b64, IMAGE_PREVIEW_COLS, 32, w, h);
    } else {
        // iTerm2: supports all image formats natively
        std::string b64 = base64::encode(reinterpret_cast<const char *>(data), len);
        seq = encode_iterm2(b64, IMAGE_PREVIEW_COLS);
    }

    console::output_guard guard;
    guard.write_raw(seq.data(), seq.size());
    guard.write("\n");
    guard.flush();
}
