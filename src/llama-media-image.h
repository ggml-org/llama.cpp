#pragma once

// image resampling for media preprocessing
// moved from tools/mtmd/mtmd-image.cpp (img_tool), kept bit-identical with it

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_media_img {

enum resize_algo {
    RESIZE_ALGO_BILINEAR,
    RESIZE_ALGO_BICUBIC,
    RESIZE_ALGO_LANCZOS,

    RESIZE_ALGO_FLAG_TORCH = 1 << 8,
};

enum pad_style {
    PAD_NONE,
    PAD_CEIL,
    PAD_NEAREST,
};

struct image_size {
    int width  = 0;
    int height = 0;

    bool operator==(const image_size & o) const { return width == o.width && height == o.height; }
};

// RGB, row-major, 3 bytes per pixel; a placeholder has a size but no data
struct image_u8 {
    image_size get_size() const { return { nx, ny }; }

    void set_size(image_size size, bool is_placeholder) {
        nx = size.width;
        ny = size.height;
        placeholder = is_placeholder;
        if (is_placeholder) {
            buf.clear();
        } else {
            buf.resize((size_t) nx * (size_t) ny * 3);
        }
    }

    void cpy_buf(const std::vector<uint8_t> & new_buf) { buf = new_buf; }

    const std::vector<uint8_t> & get_ro_buf() const { return buf; }

    bool is_placeholder() const { return placeholder; }

    size_t n_elements() const { return (size_t) nx * (size_t) ny * 3; }

    std::array<uint8_t, 3> get_pixel(int x, int y) const {
        const size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        return { buf[idx], buf[idx + 1], buf[idx + 2] };
    }

    void set_pixel(int x, int y, const std::array<uint8_t, 3> & rgb) {
        const size_t idx = ((size_t) y * (size_t) nx + (size_t) x) * 3;
        buf[idx] = rgb[0]; buf[idx + 1] = rgb[1]; buf[idx + 2] = rgb[2];
    }

    int nx = 0;
    int ny = 0;
    bool placeholder = false;
    std::vector<uint8_t> buf;
};

struct calc_size_opt {
    int align_size = 1;
    int min_pixels = 0;   // 0 = disabled
    int max_pixels = 0;   // 0 = disabled
    int longest_edge = 0; // 0 = disabled, applied before min/max_pixels
    bool round_half_even = false; // Python round() instead of std::round
};

void resize(const image_u8 & src, image_u8 & dst, const image_size & target, resize_algo algo,
            pad_style padding = PAD_CEIL, std::array<uint8_t, 3> pad_color = {0, 0, 0});

// size of the resized image, keeping the aspect ratio and aligned to align_size ("smart_resize" in transformers)
image_size calc_size_preserved_ratio(const image_size & inp_size, const calc_size_opt & opts);

// u8 RGB -> f32, (x/255 - mean)/std per channel, interleaved [ny][nx][3]
void to_f32_normalized(const image_u8 & src, const float mean[3], const float std[3], std::vector<float> & dst);

} // namespace llama_media_img
