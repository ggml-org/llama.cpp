#include "mtmd-image.h"

#include <algorithm>
#include <cmath>
#include <vector>

//
// base implementation
//

void mtmd_image_preprocessor::img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

void mtmd_image_preprocessor::img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<float>(src.buf[i]);
    }
}

// set of tools to manipulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
    static void resize(
            const clip_image_u8 & src,
            clip_image_u8 & dst,
            const clip_image_size & target_resolution,
            resize_algo algo,
            bool add_padding = true, // TODO: define the behavior for add_padding = false
            std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        dst.nx = target_resolution.width;
        dst.ny = target_resolution.height;
        dst.buf.resize(3 * dst.nx * dst.ny);

        if (dst.nx == src.nx && dst.ny == src.ny) {
            // no resize needed, simple copy
            dst.buf = src.buf;
            return;
        }

        if (!add_padding) {
            // direct resize
            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, dst, target_resolution.width, target_resolution.height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }
        } else {
            // resize with padding
            clip_image_u8 resized_image;
            float scale_w = static_cast<float>(target_resolution.width) / src.nx;
            float scale_h = static_cast<float>(target_resolution.height) / src.ny;
            float scale = std::min(scale_w, scale_h);
            int new_width  = std::min(static_cast<int>(std::ceil(src.nx * scale)), target_resolution.width);
            int new_height = std::min(static_cast<int>(std::ceil(src.ny * scale)), target_resolution.height);

            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, resized_image, new_width, new_height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }

            // fill dst with pad_color
            fill(dst, pad_color);

            int offset_x = (target_resolution.width  - new_width)  / 2;
            int offset_y = (target_resolution.height - new_height) / 2;

            composite(dst, resized_image, offset_x, offset_y);
        }
    }

    static void crop(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        GGML_ASSERT(x >= 0 && y >= 0 && w > 0 && h > 0);
        GGML_ASSERT(x + w <= image.nx && y + h <= image.ny);
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = 3 * ((y + i)*image.nx + (x + j));
                int dst_idx = 3 * (i*w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than longest_edge, it will be resized to longest_edge
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int longest_edge) {
        GGML_ASSERT(align_size > 0);
        if (inp_size.width <= 0 || inp_size.height <= 0 || longest_edge <= 0) {
            return {0, 0};
        }

        float scale = std::min(static_cast<float>(longest_edge) / inp_size.width,
                               static_cast<float>(longest_edge) / inp_size.height);

        float target_width_f  = static_cast<float>(inp_size.width)  * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        auto ceil_by_factor = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        int aligned_width  = ceil_by_factor(target_width_f);
        int aligned_height = ceil_by_factor(target_height_f);

        return {aligned_width, aligned_height};
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will have min_pixels <= W*H <= max_pixels
    // this is referred as "smart_resize" in transformers code
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int min_pixels, const int max_pixels) {
        GGML_ASSERT(align_size > 0);
        const int width  = inp_size.width;
        const int height = inp_size.height;

        auto round_by_factor = [f = align_size](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        auto floor_by_factor = [f = align_size](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        // always align up first
        int h_bar = std::max(align_size, round_by_factor(height));
        int w_bar = std::max(align_size, round_by_factor(width));

        if (h_bar * w_bar > max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(height * width) / max_pixels);
            h_bar = std::max(align_size, floor_by_factor(height / beta));
            w_bar = std::max(align_size, floor_by_factor(width  / beta));
        } else if (h_bar * w_bar < min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(min_pixels) / (height * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        return {w_bar, h_bar};
    }

    // draw src image into dst image at offset (offset_x, offset_y)
    static void composite(clip_image_u8 & dst, const clip_image_u8 & src, int offset_x, int offset_y) {
        for (int y = 0; y < src.ny; ++y) {
            for (int x = 0; x < src.nx; ++x) {
                int dx = x + offset_x;
                int dy = y + offset_y;
                // skip pixels that would be out of bounds in the destination
                if (dx < 0 || dy < 0 || dx >= dst.nx || dy >= dst.ny) {
                    continue;
                }
                size_t dst_idx = 3 * (static_cast<size_t>(dy) * dst.nx + static_cast<size_t>(dx));
                size_t src_idx = 3 * (static_cast<size_t>(y) * src.nx + static_cast<size_t>(x));
                dst.buf[dst_idx + 0] = src.buf[src_idx + 0];
                dst.buf[dst_idx + 1] = src.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = src.buf[src_idx + 2];
            }
        }
    }

    // fill the image with a solid color
    static void fill(clip_image_u8 & img, const std::array<uint8_t, 3> & color) {
        for (size_t i = 0; i < img.buf.size(); i += 3) {
            img.buf[i]     = color[0];
            img.buf[i + 1] = color[1];
            img.buf[i + 2] = color[2];
        }
    }

private:
    // Bilinear resize function
    static void resize_bilinear(const clip_image_u8 & src, clip_image_u8 & dst, int target_width, int target_height) {
        GGML_ASSERT(src.nx >= 2 && src.ny >= 2);
        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float x_ratio = static_cast<float>(src.nx - 1) / target_width;
        float y_ratio = static_cast<float>(src.ny - 1) / target_height;

        for (int y = 0; y < target_height; y++) {
            for (int x = 0; x < target_width; x++) {
                float px = x_ratio * x;
                float py = y_ratio * y;
                int x_floor = std::min(static_cast<int>(px), src.nx - 2);
                int y_floor = std::min(static_cast<int>(py), src.ny - 2);
                float x_lerp = px - x_floor;
                float y_lerp = py - y_floor;

                for (int c = 0; c < 3; c++) {
                    float top = lerp(
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    float bottom = lerp(
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool resize_bicubic(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5] = {};
        float d0, d2, d3, a0, a1, a2, a3;
        int i, j, k, jj;
        int x, y;
        float dx, dy;
        float tx, ty;

        tx = (float)nx / (float)target_width;
        ty = (float)ny / (float)target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int)(tx * j);
                y = (int)(ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                    }
                }
            }
        }

        return true;
    }

    // Bicubic resize function using Pillow's ImagingResample algorithm
    // Adapted from https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
    //
    // Key Difference with resize_bicubic:
    // 1. Uses separable filtering: horizontal pass followed by vertical pass
    // 2. Pre-computes normalized filter coefficients for each output pixel
    // 3. Applies convolution using fixed-point integer arithmetic for performance
    static bool resize_bicubic_pillow(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
        // This allows encoding fractional weights as integers: weight * 2^22
        const int PRECISION_BITS = 32 - 8 - 2;

        // Bicubic filter function with a = -0.5 (Note that GGML/PyTorch takes a = -0.75)
        // Returns filter weight for distance x from pixel center
        // Support: [-2, 2], meaning the filter influences pixels within 2 units of distance
        auto bicubic_filter = [](double x) -> double {
            constexpr double a = -0.5;
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            }
            if (x < 2.0) {
                return (((x - 5) * x + 8) * x - 4) * a;
            }
            return 0.0;  // Zero outside [-2, 2]
        };

        // Filter support radius: bicubic extends 2 pixels in each direction
        constexpr double filter_support = 2.0;

        // Clipping function for 8-bit values
        auto clip8 = [](int val) -> uint8_t {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return static_cast<uint8_t>(val);
        };

        // Precompute filter coefficients for ONE dimension (horizontal or vertical)
        //
        // Parameters:
        //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
        //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
        //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
        //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
        //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
        //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
        //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
        //
        // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
        auto precompute_weights = [&](int inSize, int outSize,
                                     std::vector<int> & bounds, std::vector<int32_t> & weights) -> int {
            GGML_ASSERT(inSize > 0 && outSize > 0);
            double support, scale, filterscale;
            double center, ww, ss;
            int xx, x, ksize, xmin, xmax, xcnt;

            // Calculate scaling factor: ratio of input range to output size
            filterscale = scale = (double)inSize / outSize;
            // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
            // For downsampling (scale > 1), widen filter to prevent aliasing
            if (filterscale < 1.0) {
                filterscale = 1.0;
            }

            // Determine filter support radius and kernel size
            support = filter_support * filterscale;  // Widen filter when downsampling
            ksize = static_cast<int>(std::ceil(support)) * 2 + 1;  // Total pixels in kernel

            std::vector<double> pre_weights(outSize * ksize);  // Temporary weights
            bounds.resize(outSize * 2);

            // For each output pixel, compute its filter coefficients
            for (xx = 0; xx < outSize; xx++) {
                // Calculate the center position in input space (pixel-center convention: +0.5)
                center = (xx + 0.5) * scale;
                ww = 0.0;  // Sum of weights for normalization
                ss = 1.0 / filterscale;  // Scale factor for filter function

                // Determine the range of input pixels that contribute to this output pixel
                xmin = static_cast<int>(center - support + 0.5);
                if (xmin < 0) {
                    xmin = 0;
                }

                xmax = static_cast<int>(center + support + 0.5);
                if (xmax > inSize) {
                    xmax = inSize;
                }

                xcnt = xmax - xmin;

                // Compute filter weights for each contributing input pixel
                for (x = 0; x < xcnt; x++) {
                    // Distance from input pixel center to output pixel center in input space
                    double w = bicubic_filter((x + xmin - center + 0.5) * ss);
                    pre_weights[xx * ksize + x] = w;
                    ww += w;  // Accumulate for normalization
                }

                // Normalize weights to sum to 1.0 (preserves brightness)
                for (x = 0; x < xcnt; x++) {
                    if (ww != 0.0) {
                        pre_weights[xx * ksize + x] /= ww;
                    }
                }

                // Zero-pad remaining kernel positions
                for (; x < ksize; x++) {
                    pre_weights[xx * ksize + x] = 0;
                }

                // Store input pixel range for this output pixel
                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xcnt;
            }

            // Convert floating-point coefficients to fixed-point integers
            // Formula: int32 = round(float * 2^PRECISION_BITS)
            weights.resize(outSize * ksize);
            for (int i = 0; i < outSize * ksize; i++) {
                if (pre_weights[i] < 0) {
                    weights[i] = static_cast<int32_t>(-0.5 + pre_weights[i] * (1 << PRECISION_BITS));
                } else {
                    weights[i] = static_cast<int32_t>(0.5 + pre_weights[i] * (1 << PRECISION_BITS));
                }
            }

            return ksize;
        };

        // Horizontal resampling pass
        // Resizes width from imIn.nx to imOut.nx, preserving height
        auto resample_horizontal = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                       int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weights) {
            imOut.ny = imIn.ny;
            imOut.buf.resize(3 * imOut.nx * imOut.ny);

            // Process each row independently
            for (int yy = 0; yy < imOut.ny; yy++) {
                // For each output pixel in this row
                for (int xx = 0; xx < imOut.nx; xx++) {
                    // Get the range of input pixels and filter coefficients
                    int xmin = bounds[xx * 2 + 0];  // First input pixel index
                    int xcnt = bounds[xx * 2 + 1];  // Number of input pixels

                    // Initialize accumulators for RGB channels with rounding bias (0.5 in fixed-point)
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels
                    for (int x = 0; x < xcnt; x++) {
                        int src_idx = ((yy * imIn.nx) + (x + xmin)) * 3;
                        ss0 += static_cast<uint8_t>(imIn.buf[src_idx + 0]) * weights[xx * ksize + x];  // R channel
                        ss1 += static_cast<uint8_t>(imIn.buf[src_idx + 1]) * weights[xx * ksize + x];  // G channel
                        ss2 += static_cast<uint8_t>(imIn.buf[src_idx + 2]) * weights[xx * ksize + x];  // B channel
                    }

                    // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
                    int dst_idx = (yy * imOut.nx + xx) * 3;
                    imOut.buf[dst_idx + 0] = clip8(ss0 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 1] = clip8(ss1 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }
        };

        // Vertical resampling pass
        // Resizes height from imIn.ny to imOut.ny, preserving width
        auto resample_vertical = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                     int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weight) {
            imOut.nx = imIn.nx;
            imOut.buf.resize(3 * imOut.nx * imOut.ny);

            // For each output row
            for (int yy = 0; yy < imOut.ny; yy++) {
                // Get the range of input rows and filter coefficients
                int ymin = bounds[yy * 2 + 0];  // First input row index
                int ycnt = bounds[yy * 2 + 1];  // Number of input rows

                // Process each column in this output row
                for (int xx = 0; xx < imOut.nx; xx++) {
                    // Initialize accumulators for RGB channels with rounding bias
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels vertically
                    for (int y = 0; y < ycnt; y++) {
                        int src_idx = ((y + ymin) * imIn.nx + xx) * 3;
                        ss0 += static_cast<uint8_t>(imIn.buf[src_idx + 0]) * weight[yy * ksize + y];  // R channel
                        ss1 += static_cast<uint8_t>(imIn.buf[src_idx + 1]) * weight[yy * ksize + y];  // G channel
                        ss2 += static_cast<uint8_t>(imIn.buf[src_idx + 2]) * weight[yy * ksize + y];  // B channel
                    }

                    // Convert back from fixed-point and clamp to [0,255]
                    int dst_idx = (yy * imOut.nx + xx) * 3;
                    imOut.buf[dst_idx + 0] = clip8(ss0 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 1] = clip8(ss1 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }
        };

        // Main resampling logic using separable two-pass approach
        const int src_width = img.nx;
        const int src_height = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;

        bool need_horizontal = (target_width != src_width);
        bool need_vertical = (target_height != src_height);

        // Precompute filter coefficients for both dimensions
        std::vector<int> bounds_horiz, bounds_vert;
        std::vector<int32_t> weights_horiz, weights_vert;
        int ksize_horiz = 0, ksize_vert = 0;

        if (need_horizontal) {
            ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
        }

        if (need_vertical) {
            ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
        }

        // Perform two-pass resampling
        if (need_horizontal && need_vertical) {
            // Both horizontal and vertical
            clip_image_u8 temp;
            temp.nx = target_width;
            resample_horizontal(img, temp, ksize_horiz, bounds_horiz, weights_horiz);
            resample_vertical(temp, dst, ksize_vert, bounds_vert, weights_vert);
        } else if (need_horizontal) {
            // Only horizontal
            resample_horizontal(img, dst, ksize_horiz, bounds_horiz, weights_horiz);
        } else if (need_vertical) {
            // Only vertical
            resample_vertical(img, dst, ksize_vert, bounds_vert, weights_vert);
        } else {
            // No resizing needed - direct copy
            dst.buf = img.buf;
        }

        return true;
    }

    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};


//
// mtmd_image_preprocessor_llava_uhd
//

bool mtmd_image_preprocessor_llava_uhd::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const clip_image_size original_size{img.nx, img.ny};
    auto const inst = get_slice_instructions(original_size);
    std::vector<clip_image_u8_ptr> imgs = slice_image(img, inst);

    for (size_t i = 0; i < imgs.size(); ++i) {
        // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }

    output.grid_x = inst.grid_size.width;
    output.grid_y = inst.grid_size.height;
    return true;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_llava_uhd::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions res;
    const int patch_size      = hparams.patch_size;
    const int slice_size      = hparams.image_size;
    const int original_width  = original_size.width;
    const int original_height = original_size.height;

    const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;
    const bool has_pinpoints = !hparams.image_res_candidates.empty();

    if (!has_slices) {
        // skip slicing logic
        res.overview_size = clip_image_size{slice_size, slice_size};
        res.refined_size  = clip_image_size{0, 0};
        res.grid_size     = clip_image_size{0, 0};

        return res;
    }

    if (has_pinpoints) {
        // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
        auto refine_size = select_best_resolution(
            original_size,
            hparams.image_res_candidates);
        res.overview_size         = clip_image_size{slice_size, slice_size};
        res.refined_size          = refine_size;
        res.grid_size             = clip_image_size{0, 0};

        LOG_DBG("%s: using pinpoints for slicing\n", __func__);
        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width,  res.refined_size.height);

        for (int y = 0; y < refine_size.height; y += slice_size) {
            for (int x = 0; x < refine_size.width; x += slice_size) {
                slice_coordinates slice;
                slice.x = x;
                slice.y = y;
                slice.size.width  = std::min(slice_size, refine_size.width  - x);
                slice.size.height = std::min(slice_size, refine_size.height - y);
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }

        res.grid_size.height = refine_size.height / slice_size;
        res.grid_size.width  = refine_size.width  / slice_size;
        LOG_DBG("%s: grid size: %d x %d\n", __func__, res.grid_size.width, res.grid_size.height);

        return res;
    }

    // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

    auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
    res.overview_size = best_size;

    {
        const int max_slice_nums = 9; // TODO: this is only used by minicpmv, maybe remove it
        const float log_ratio = log((float)original_width / original_height);
        const float ratio = (float)original_width * original_height / (slice_size * slice_size);
        const int multiple = fmin(ceil(ratio), max_slice_nums);

        auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
        auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
        res.grid_size    = best_grid;
        res.refined_size = refine_size;

        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width, res.refined_size.height,
                res.grid_size.width, res.grid_size.height);

        int width  = refine_size.width;
        int height = refine_size.height;
        int grid_x = int(width  / best_grid.width);
        int grid_y = int(height / best_grid.height);
        for (int patches_y = 0,                    ic = 0;
                patches_y < refine_size.height && ic < best_grid.height;
                patches_y += grid_y,              ic += 1) {
            for (int patches_x = 0,                   jc = 0;
                    patches_x < refine_size.width && jc < best_grid.width;
                    patches_x += grid_x,             jc += 1) {
                slice_coordinates slice;
                slice.x = patches_x;
                slice.y = patches_y;
                slice.size.width  = grid_x;
                slice.size.height = grid_y;
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }
    }

    return res;
}

std::vector<clip_image_u8_ptr> mtmd_image_preprocessor_llava_uhd::slice_image(const clip_image_u8 & img, const mtmd_image_preprocessor_llava_uhd::slice_instructions & inst, bool overview_first) {
    std::vector<clip_image_u8_ptr> output;

    // resize to overview size
    clip_image_u8_ptr resized_img(clip_image_u8_init());
    img_tool::resize(img, *resized_img, inst.overview_size, hparams.image_resize_algo_ov,
                        hparams.image_pad_ov, hparams.image_pad_color_ov);
    if (overview_first) {
        output.push_back(std::move(resized_img));
    }

    if (inst.slices.empty()) {
        // no slices, just return the resized image
        if (!overview_first) {
            output.push_back(std::move(resized_img));
        }
        return output;
    }

    // resize to refined size
    clip_image_u8_ptr refined_img(clip_image_u8_init());
    img_tool::resize(img, *refined_img, inst.refined_size, hparams.image_resize_algo_rf,
                        hparams.image_pad_rf, hparams.image_pad_color_rf);

    // create slices
    for (const auto & slice : inst.slices) {
        int x = slice.x;
        int y = slice.y;
        int w = slice.size.width;
        int h = slice.size.height;

        clip_image_u8_ptr img_slice(clip_image_u8_init());
        img_tool::crop(*refined_img, *img_slice, x, y, w, h);
        output.push_back(std::move(img_slice));
    }

    if (!overview_first) {
        output.push_back(std::move(resized_img));
    }

    return output;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height  = static_cast<int>(scale_resolution / std::sqrt(r));
        width   = static_cast<int>(height * r);
    }
    clip_image_size res;
    res.width  = ensure_divide(width,  patch_size);
    res.height = ensure_divide(height, patch_size);
    return res;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max) {
    float scale_width  = static_cast<float>(target_max.width)  / orig.width;
    float scale_height = static_cast<float>(target_max.height) / orig.height;
    float scale = std::min(scale_width, scale_height);
    return clip_image_size{
        static_cast<int>(orig.width  * scale),
        static_cast<int>(orig.height * scale),
    };
}

clip_image_size mtmd_image_preprocessor_llava_uhd::select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions) {
    clip_image_size best_fit;
    int min_wasted_area = std::numeric_limits<int>::max();
    int max_effective_resolution = 0;

    for (const clip_image_size & candidate : possible_resolutions) {
        auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
        int effective_resolution = std::min(
            target_size.width * target_size.height,
            original_size.width * original_size.height);
        int wasted_area = (candidate.width * candidate.height) - effective_resolution;

        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
            max_effective_resolution = effective_resolution;
            min_wasted_area = wasted_area;
            best_fit = candidate;
        }

        LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
    }

    return best_fit;
}

int mtmd_image_preprocessor_llava_uhd::ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    int grid_x = grid.width;
    int grid_y = grid.height;

    int refine_width  = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    clip_image_size grid_size;
    grid_size.width  = refine_width  / grid_x;
    grid_size.height = refine_height / grid_y;

    auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
    int best_grid_width  = best_grid_size.width;
    int best_grid_height = best_grid_size.height;

    clip_image_size refine_size;
    refine_size.width  = best_grid_width  * grid_x;
    refine_size.height = best_grid_height * grid_y;
    return refine_size;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<clip_image_size> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.push_back(clip_image_size{m, split_grids_nums / m});
            }
            ++m;
        }
    }

    clip_image_size best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto& grid : candidate_grids) {
        float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
        if (error < min_error) {
            best_grid = grid;
            min_error = error;
        }
    }
    return best_grid;
}

//
// mtmd_image_preprocessor_fixed_size
//

bool mtmd_image_preprocessor_fixed_size::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    clip_image_u8 resized_image;
    int sz = hparams.image_size;
    img_tool::resize(img, resized_image, {sz, sz},
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_dyn_size
//

bool mtmd_image_preprocessor_dyn_size::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size{img.nx, img.ny};
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        hparams.patch_size * cur_merge,
        hparams.image_min_pixels,
        hparams.image_max_pixels);
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_longest_edge
//

bool mtmd_image_preprocessor_longest_edge::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(hparams.image_longest_edge > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size{img.nx, img.ny};
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        hparams.patch_size * cur_merge,
        hparams.image_longest_edge);
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_lfm2
//

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_lfm2::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions inst;
    const int align_size = hparams.patch_size * hparams.n_merge;
    inst.overview_size = img_tool::calc_size_preserved_ratio(
                            original_size, align_size,
                            hparams.image_min_pixels, hparams.image_max_pixels);
    // tile if either dimension exceeds tile_size with tolerance
    const bool needs_tiling = original_size.width > tile_size * max_pixels_tolerance || original_size.height > tile_size * max_pixels_tolerance;

    if (!needs_tiling) {
        inst.refined_size = clip_image_size{0, 0};
        inst.grid_size    = clip_image_size{0, 0};
        return inst;
    }

    const clip_image_size grid = get_grid_layout(original_size.height, original_size.width);

    inst.grid_size    = grid;
    inst.refined_size = clip_image_size{tile_size * grid.width, tile_size * grid.height};

    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
            __func__,
            original_size.width, original_size.height,
            inst.overview_size.width, inst.overview_size.height,
            inst.refined_size.width, inst.refined_size.height,
            grid.width, grid.height);

    for (int row = 0; row < grid.height; row++) {
        for (int col = 0; col < grid.width; col++) {
            mtmd_image_preprocessor_llava_uhd::slice_coordinates slice;
            slice.x    = col * tile_size;
            slice.y    = row * tile_size;
            slice.size = clip_image_size{tile_size, tile_size};
            inst.slices.push_back(slice);
            LOG_DBG("%s: slice %d: x=%d, y=%d, size=%d x %d\n",
                    __func__, (int)inst.slices.size() - 1,
                    slice.x, slice.y, slice.size.width, slice.size.height);
        }
    }

    return inst;
}

clip_image_size mtmd_image_preprocessor_lfm2::find_closest_aspect_ratio(
        float aspect_ratio,
        const std::vector<clip_image_size> & target_ratios,
        int width, int height) {
    float best_ratio_diff = std::numeric_limits<float>::max();
    clip_image_size best_ratio = {1, 1};
    const float area = static_cast<float>(width * height);

    for (const auto & ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.width) / ratio.height;
        const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const float target_area = static_cast<float>(tile_size * tile_size * ratio.width * ratio.height);
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

std::vector<clip_image_size> mtmd_image_preprocessor_lfm2::get_target_ratios() {
    std::vector<clip_image_size> ratios;
    for (int n = min_tiles; n <= max_tiles; n++) {
        for (int w = 1; w <= n; w++) {
            for (int h = 1; h <= n; h++) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    bool found = false;
                    for (const auto & r : ratios) {
                        if (r.width == w && r.height == h) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        ratios.push_back({w, h});
                    }
                }
            }
        }
    }
    std::sort(ratios.begin(), ratios.end(), [](const clip_image_size & a, const clip_image_size & b) {
        return a.width * a.height < b.width * b.height;
    });
    return ratios;
}

clip_image_size mtmd_image_preprocessor_lfm2::get_grid_layout(int height, int width) {
    const float aspect_ratio = static_cast<float>(width) / height;
    const auto ratios = get_target_ratios();
    return find_closest_aspect_ratio(aspect_ratio, ratios, width, height);
}

//
// mtmd_image_preprocessor_idefics3
//

bool mtmd_image_preprocessor_idefics3::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    // The refined size has two steps:
    // 1. Resize w/ aspect-ratio preserving such that the longer side is
    //      the preprocessor longest size
    // 2. Resize w/out preserving aspect ratio such that both sides are
    //      multiples of image_size (always rounding up)
    //
    // CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
    const clip_image_size original_size{img.nx, img.ny};
    const clip_image_size refined_size = img_tool::calc_size_preserved_ratio(
        original_size, hparams.image_size, hparams.image_longest_edge);
    // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
    //         __func__, original_size.width, original_size.height,
    //         refined_size.width, refined_size.height);

    mtmd_image_preprocessor_llava_uhd::slice_instructions instructions;
    instructions.overview_size = clip_image_size{hparams.image_size, hparams.image_size};
    instructions.refined_size = refined_size;
    instructions.grid_size = clip_image_size{
        static_cast<int>(std::ceil(static_cast<float>(refined_size.width) / hparams.image_size)),
        static_cast<int>(std::ceil(static_cast<float>(refined_size.height) / hparams.image_size)),
    };
    for (int y = 0; y < refined_size.height; y += hparams.image_size) {
        for (int x = 0; x < refined_size.width; x += hparams.image_size) {
            // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, x, y);
            instructions.slices.push_back(mtmd_image_preprocessor_llava_uhd::slice_coordinates{
                /* x    */x,
                /* y    */y,
                /* size */clip_image_size{
                    std::min(hparams.image_size, refined_size.width - x),
                    std::min(hparams.image_size, refined_size.height - y)
                }
            });
        }
    }
    auto imgs = slice_image(img, instructions);

    // cast and normalize to f32
    for (size_t i = 0; i < imgs.size(); ++i) {
        // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }

    output.grid_x = instructions.grid_size.width;
    output.grid_y = instructions.grid_size.height;
    return true;
}

//
// mtmd_image_preprocessor_internvl
//

bool mtmd_image_preprocessor_internvl::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(!hparams.image_res_candidates.empty());
    const clip_image_size original_size{img.nx, img.ny};
    auto const inst = get_slice_instructions(original_size);
    std::vector<clip_image_u8_ptr> imgs = slice_image(img, inst, false);

    for (size_t i = 0; i < imgs.size(); ++i) {
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }
    return true;
}

//
// mtmd_image_preprocessor_deepseekocr
//

bool mtmd_image_preprocessor_deepseekocr::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const std::vector native_resolutions = {
        /*512 tiny , 640 small, */ 1024 /* base */, 1280 /* large */
    };
    // original image size
    const clip_image_size original_size{img.nx, img.ny};
    const int orig_w = original_size.width;
    const int orig_h = original_size.height;
    const int orig_area = orig_h * orig_w;

    size_t mode_i = 0;
    int min_diff = orig_area;

    for (size_t i = 0; i < native_resolutions.size(); i++) {
        int r = native_resolutions[i];
        if (std::abs(orig_area - r * r) < min_diff) {
            mode_i = i;
            min_diff = std::abs(orig_area - r * r);
        }
    }

    /* Native Resolution (Base/Large) */
    const int image_size = native_resolutions[mode_i];

    // scaled and padded image
    clip_image_u8_ptr scaled_img(clip_image_u8_init());
    img_tool::resize(img, *scaled_img, clip_image_size{image_size, image_size}, hparams.image_resize_algo);

    clip_image_f32_ptr res(clip_image_f32_init());
    img_u8_to_f32(*scaled_img, *res, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(res));

    output.grid_x = 1;
    output.grid_y = 1;
    return true;
}

//
// mtmd_image_preprocessor_youtuvl
//

bool mtmd_image_preprocessor_youtuvl::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const int patch_size = hparams.patch_size;   // typically 16
    const int merge_size = hparams.n_merge;      // typically 2
    const int align_size = patch_size * merge_size;  // 32

    const int max_num_patches = hparams.image_max_pixels > 0 ?
        hparams.image_max_pixels / (patch_size * patch_size) : 256;

    // Linear search for optimal scale to fit within max_num_patches
    float scale = 1.0f;
    int target_height = img.ny;
    int target_width  = img.nx;

    auto get_scaled_image_size = [align_size](float scale, int size) -> int {
        float scaled_size = size * scale;
        // Round up to nearest multiple of align_size
        int aligned = static_cast<int>(std::ceil(scaled_size / align_size)) * align_size;
        // Ensure at least one patch
        return std::max(align_size, aligned);
    };

    // Linear search with 0.02 step size
    while (scale > 0.0f) {
        target_height = get_scaled_image_size(scale, img.ny);
        target_width  = get_scaled_image_size(scale, img.nx);

        int num_patches_h = target_height / patch_size;
        int num_patches_w = target_width / patch_size;
        int num_patches = num_patches_h * num_patches_w;

        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }

    clip_image_size new_size = {target_width, target_height};

    // Resize the image
    clip_image_u8 resized;
    img_tool::resize(img, resized, new_size, hparams.image_resize_algo, hparams.image_resize_pad);

    // Normalize to float32
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized, *img_f32, hparams.image_mean, hparams.image_std);
    // Add to results
    output.entries.push_back(std::move(img_f32));
    return true;
}
