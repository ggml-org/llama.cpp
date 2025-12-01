#include "phi3_vision_defs.h"

Phi3Grid get_phi3_grid(int img_w, int img_h, int max_crops) {

    float ratio = (float)img_w / (float)img_h;

    // Normalize ratio to be >= 1 for easier math (handle tall images by transposing)
    bool transposed = false;
    if (ratio < 1.0) {
        ratio = 1.0f / ratio;
        transposed = true;
    }

    int scale = 1;

    // Iterate to find the largest 'scale' (width units) that fits the budget
    while (true) {
        int next_scale = scale + 1;

        // Calculate required height units (must be an integer)
        int h_units = (int)std::ceil(next_scale / ratio);

        // Check if the next grid size exceeds the max_crops budget
        if (next_scale * h_units > max_crops) {
            break;
        }
        scale = next_scale;
    }

    // Final grid units (based on the largest scale that fit)
    int grid_w_units = scale;
    int grid_h_units = (int)std::ceil(scale / ratio);

    // Swap back if the image was originally tall
    if (transposed) {
        std::swap(grid_w_units, grid_h_units);
    }

    return {grid_w_units, grid_h_units};
}

image_u8 resize_image(const image_u8& src, int w, int h) {
    if (src.w == w && src.h == h) return src; // No change

    image_u8 dst = {w, h, src.n_comp, std::vector<uint8_t>(w * h * src.n_comp)};

    float scale_x = (float)src.w / w;
    float scale_y = (float)src.h / h;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {

            // Calculate coordinates in the source image
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            float src_y = (y + 0.5f) * scale_y - 0.5f;

            // Get integer and fractional parts
            int x_int = static_cast<int>(std::floor(src_x));
            int y_int = static_cast<int>(std::floor(src_y));
            float x_frac = src_x - x_int;
            float y_frac = src_y - y_int;

            // Apply clamping
            x_int = std::max(0, std::min(x_int, src.w - 2));
            y_int = std::max(0, std::min(y_int, src.h - 2));

            for (int c = 0; c < src.n_comp; ++c) {
                // Get the 4 neighboring pixel values (P11, P12, P21, P22)
                int p11 = src.data[(y_int * src.w + x_int) * src.n_comp + c];
                int p12 = src.data[(y_int * src.w + (x_int + 1)) * src.n_comp + c];
                int p21 = src.data[((y_int + 1) * src.w + x_int) * src.n_comp + c];
                int p22 = src.data[((y_int + 1) * src.w + (x_int + 1)) * src.n_comp + c];

                // Bilinear interpolation formula:
                float val = (p11 * (1 - x_frac) + p12 * x_frac) * (1 - y_frac) +
                            (p21 * (1 - x_frac) + p22 * x_frac) * y_frac;

                dst.data[(y * w + x) * src.n_comp + c] = static_cast<uint8_t>(std::round(val));
            }
        }
    }
    return dst;
}

// 2. Pads source image onto a larger canvas (w x h) using WHITE_PADDING.
image_u8 pad_image(const image_u8& src, int w, int h) {

    // Allocate the w*h canvas and fill it with WHITE_PADDING (255)
    image_u8 canvas = {w, h, src.n_comp, std::vector<uint8_t>(w * h * src.n_comp, WHITE_PADDING)};

    // --- Determine aspect-ratio-preserving resize ---
    // The original image must be resized to fit the canvas while preserving aspect ratio.
    float src_ratio = (float)src.w / src.h;
    float canvas_ratio = (float)w / h;

    int resized_w, resized_h;

    if (src_ratio > canvas_ratio) {
        // Image is wider than canvas ratio, constrained by width (w)
        resized_w = w;
        resized_h = static_cast<int>(w / src_ratio);
    } else {
        // Image is taller than canvas ratio, constrained by height (h)
        resized_h = h;
        resized_w = static_cast<int>(h * src_ratio);
    }

    // 1. Resize the source image
    image_u8 resized_src = resize_image(src, resized_w, resized_h);

    // 2. Calculate top-left pasting coordinates (centered or top-left)
    // Phi-3 typically pastes top-left for simplicity in HDTransform.
    int start_x = 0;
    int start_y = 0;

    // 3. Copy the resized pixels onto the white canvas
    for (int y = 0; y < resized_h; ++y) {
        for (int x = 0; x < resized_w; ++x) {
            for (int c = 0; c < src.n_comp; ++c) {
                // Pixel index in source
                size_t src_idx = (y * resized_w + x) * src.n_comp + c;
                // Pixel index in canvas
                size_t dst_idx = ((start_y + y) * w + (start_x + x)) * src.n_comp + c;

                if (src_idx < resized_src.data.size() && dst_idx < canvas.data.size()) {
                    canvas.data[dst_idx] = resized_src.data[src_idx];
                }
            }
        }
    }

    return canvas;
}

// 3. Crops a 336x336 region from the source image.
image_u8 crop_image(const image_u8& src, int x_start, int y_start) {

    image_u8 tile = {PHI3V_PATCH_SIZE, PHI3V_PATCH_SIZE, src.n_comp,
                     std::vector<uint8_t>(PHI3V_PATCH_SIZE * PHI3V_PATCH_SIZE * src.n_comp)};

    int tile_w = PHI3V_PATCH_SIZE;
    int tile_h = PHI3V_PATCH_SIZE;
    int tile_c = src.n_comp;

    for (int y = 0; y < tile_h; ++y) {
        for (int x = 0; x < tile_w; ++x) {

            // Calculate coordinates in the source (canvas) image
            int src_x = x_start + x;
            int src_y = y_start + y;

            // Calculate linear index in the source (canvas) and destination (tile)
            size_t src_idx_base = (src_y * src.w + src_x) * tile_c;
            size_t dst_idx_base = (y * tile_w + x) * tile_c;

            // Copy all color channels (RGB)
            for (int c = 0; c < tile_c; ++c) {
                tile.data[dst_idx_base + c] = src.data[src_idx_base + c];
            }
        }
    }

    return tile;
}

Phi3ImageBatch hd_transform_preprocess(const image_u8& raw_image, int max_crops) {

    // 1. Calculate the Best Grid
    Phi3Grid grid = get_phi3_grid(raw_image.w, raw_image.h, max_crops);

    // 2. Define the Canvas Size
    int canvas_w = grid.w * PHI3V_PATCH_SIZE;
    int canvas_h = grid.h * PHI3V_PATCH_SIZE;

    // 3. Initialize the Output Batch
    Phi3ImageBatch output;
    output.grid = grid;

    // --- 3a. Global View (Batch[0]) ---
    // Standard low-res view of the entire image, always 336x336.
    output.batch.push_back(resize_image(raw_image, PHI3V_PATCH_SIZE, PHI3V_PATCH_SIZE));

    // --- 3b. Local Crops ---

    // Prepare the Canvas (Resize and Pad with white)
    // The canvas is a perfect multiple of 336x336 patches.
    image_u8 canvas = pad_image(raw_image, canvas_w, canvas_h);

    // Slice the Canvas into Local Crops (Row-Major Order)
    for (int y = 0; y < grid.h; ++y) {
        for (int x = 0; x < grid.w; ++x) {
            output.batch.push_back(
                crop_image(canvas,
                           x * PHI3V_PATCH_SIZE,
                           y * PHI3V_PATCH_SIZE)
            );
        }
    }

    return output;
}