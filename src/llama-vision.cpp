#include "llama.h"
#include "llama-vision.h"
#include "llama-impl.h"
#include "llama-context.h"

#include <string.h> // memcpy
#include <limits>
#include <cmath>

#ifndef NDEBUG
// for debugging
#include <fstream>
#include <cstdint>
#include <iostream>

// export llama_image_u8 to bmp file for debugging
// https://codereview.stackexchange.com/questions/195121/writing-a-bitmap-image-from-c
static int bmp_export(const struct llama_image_u8 &img, const std::string &location);
#endif

struct img_size {
    int width;
    int height;
    img_size(int w, int h) : width(w), height(h) {}
};

// RGB uint8 image
// Memory layout: RGBRGBRGB...
struct llama_image_u8 {
    int nx;
    int ny;
    std::vector<uint8_t> buf;
    llama_image_u8() {}
    llama_image_u8(const llama_vision_bitmap & bmp) {
        nx = bmp.nx;
        ny = bmp.ny;
        buf.resize(nx*ny*3);
        memcpy(buf.data(), bmp.data, buf.size());
    }
};

uint32_t llama_vision_n_mmproj_embd(const llama_vision_model & vmodel) {
    auto & proj_type = vmodel.hparams.proj_type;
    if (proj_type == VISION_PROJECTOR_TYPE_MLP) {
        return vmodel.mm_2_b
            ? vmodel.mm_2_b->ne[0]
            : vmodel.projection->ne[1]; // idefics3
    } else if (proj_type == VISION_PROJECTOR_TYPE_LDPV2) {
        return vmodel.mm_model_peg_0_b->ne[0];
    } else if (proj_type == VISION_PROJECTOR_TYPE_MINICPMV_2_5) {
        return 4096; // resampler
    } else if (proj_type == VISION_PROJECTOR_TYPE_MINICPMV_2_6) {
        return 3584; // resampler
    } else {
        GGML_ASSERT(false && "invalid proj type");
    }
}


//
// internal utils
//

static int get_n_patches_x(const llama_vision_context & ctx) {
    auto & hparams = ctx.model->hparams;
    return hparams.image_size / hparams.patch_size;
}

static int get_n_patches_y(const llama_vision_context & ctx) {
    return get_n_patches_x(ctx);
}

static int get_n_patches(const llama_vision_context & ctx) {
    return get_n_patches_x(ctx) * get_n_patches_y(ctx);
}

//
// bitmap utils
//

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static img_size select_best_resolution(const img_size & original_size, const std::vector<img_size>& possible_resolutions) {
    int original_width  = original_size.width;
    int original_height = original_size.height;

    img_size best_fit(0, 0);
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width   = resolution.width;
        int height  = resolution.height;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_DBG("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static bool bicubic_resize(const llama_image_u8 & img, llama_image_u8 & dst, int target_width, int target_height) {
    auto clip = [](int x, int lower, int upper) -> int {
        return std::max(lower, std::min(x, upper));
    };

    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
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

static std::vector<llama_image_u8> divide_to_patches_u8(const llama_image_u8 & image, int patch_size) {
    std::vector<llama_image_u8> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            llama_image_u8 patch;
            patch.nx = std::min(patch_size, width - j);
            patch.ny = std::min(patch_size, height - i);
            patch.buf.resize(3 * patch.nx * patch.ny);
            for (int y = 0; y < patch.ny; ++y) {
                for (int x = 0; x < patch.nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch.buf[3 * (y * patch.nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(patch);
        }
    }
    return patches;
}

// llava-1.6 type of resize_and_pad (black)
static llama_image_u8 resize_and_pad_image(const llama_image_u8 & image, const img_size & target_resolution) {
    int target_width  = target_resolution.width;
    int target_height = target_resolution.height;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    llama_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    llama_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    return padded_image;
}

static void normalize_image_u8_to_f32(const llama_image_u8 & src, std::vector<float> & dst, const std::array<float, 3> & mean, const std::array<float, 3> & std) {
    dst.resize(src.buf.size());

    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}


//
// processor
//

struct llama_vision_processor {
    const llama_vision_context & ctx;
    llama_vision_processor(const llama_vision_context & ctx) : ctx(ctx) {}
    virtual llama_vision_tokens tokenize(const llama_image_u8 & img) = 0;
    virtual ~llama_vision_processor() = default;
};

// inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
struct llama_vision_processor_llava : llama_vision_processor {
    llama_vision_processor_llava(const llama_vision_context & ctx) : llama_vision_processor(ctx) {}

    virtual llama_vision_tokens tokenize(const llama_image_u8 & img) override {
        bool pad_to_square = true;
        auto & params = ctx.model->hparams;
        // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
        if (params.mm_patch_merge_type == MM_PATCH_MERGE_SPATIAL_UNPAD) {
            pad_to_square = false;
        }

        llama_vision_tokens output_slices;
        output_slices.n_px = get_n_patches_x(ctx);
        output_slices.n_py = get_n_patches_y(ctx);
        output_slices.px = params.patch_size;
        output_slices.py = params.patch_size;

        // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
        // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

        llama_image_u8 temp;
        if (pad_to_square && img.nx != img.ny) {
            // if the image is not square, pad it to a square
            int longer_side = std::max(img.nx, img.ny);
            temp.nx = longer_side;
            temp.ny = longer_side;
            temp.buf.resize(3 * longer_side * longer_side);
            const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA (this is the mean rgb color * 255)

            // fill with background color
            for (size_t i = 0; i < temp.buf.size(); i++) {
                temp.buf[i] = bc[i % 3];
            }

            // copy from the input image
            for (int y = 0; y < img.ny; y++) {
                for (int x = 0; x < img.nx; x++) {
                    const int i = 3 * (y * img.nx + x);
                    const int j = 3 * (y * temp.nx + x);
                    temp.buf[j]   = img.buf[i];
                    temp.buf[j+1] = img.buf[i+1];
                    temp.buf[j+2] = img.buf[i+2];
                }
            }
        } else if (params.image_grid_pinpoints[0] != 0) {
            // "spatial_unpad" with "anyres" processing for llava-1.6
            std::vector<img_size> possible_resolutions;
            for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i += 2) {
                img_size s(0, 0);
                s.width  = params.image_grid_pinpoints[i];
                s.height = params.image_grid_pinpoints[i+1];
                possible_resolutions.push_back(s);
            }
            img_size best_resolution = select_best_resolution(img_size(img.nx, img.ny), possible_resolutions);
            // debug_image_save_to_bmp(*img, "input.bmp");
            temp = resize_and_pad_image(img, best_resolution);  // we do not pad with mean-bg color anymore in llava-1.6
            // debug_image_save_to_bmp(*temp, "resized.bmp");

            std::vector<llama_image_u8> patches = divide_to_patches_u8(temp, params.image_size); // prepare spatial sorted main patches of image_size each (336 in llava-1.6)

            llama_image_u8 image_original_resize;
            // bilinear_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            bicubic_resize(img, image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            patches.insert(patches.begin(), image_original_resize);
            output_slices.buf.resize(patches.size());
            int num = 0;
            for (auto & patch : patches) {
                normalize_image_u8_to_f32(patch, output_slices.buf[num], params.image_mean, params.image_std);
                num++;
            }
            return output_slices;
        } else {
            temp.nx = img.nx;
            temp.ny = img.ny;
            temp.buf.resize(img.buf.size());
            memcpy(temp.buf.data(), img.buf.data(), temp.buf.size());
        }

        const int nx = temp.nx;
        const int ny = temp.ny;
        // bmp_export(temp, "resized_vanilla.bmp");

        const int nx2 = params.image_size;
        const int ny2 = params.image_size;
        std::vector<float> res;
        res.resize(3 * nx2 * ny2);

        const float scale = std::max(nx, ny) / (float)params.image_size;

        const int nx3 = int(nx / scale + 0.5f);
        const int ny3 = int(ny / scale + 0.5f);

        const auto & m3 = params.image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
        const auto & s3 = params.image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

        for (int y = 0; y < ny3; y++) {
            for (int x = 0; x < nx3; x++) {
                for (int c = 0; c < 3; c++) {
                    // linear interpolation
                    const float sx = (x + 0.5f) * scale - 0.5f;
                    const float sy = (y + 0.5f) * scale - 0.5f;

                    const int x0 = std::max(0, (int)std::floor(sx));
                    const int y0 = std::max(0, (int)std::floor(sy));

                    const int x1 = std::min(x0 + 1, nx - 1);
                    const int y1 = std::min(y0 + 1, ny - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = 3 * (y0 * nx + x0) + c;
                    const int j01 = 3 * (y0 * nx + x1) + c;
                    const int j10 = 3 * (y1 * nx + x0) + c;
                    const int j11 = 3 * (y1 * nx + x1) + c;

                    const float v00 = temp.buf[j00];
                    const float v01 = temp.buf[j01];
                    const float v10 = temp.buf[j10];
                    const float v11 = temp.buf[j11];

                    const float v0 = v00 * (1.0f - dx) + v01 * dx;
                    const float v1 = v10 * (1.0f - dx) + v11 * dx;

                    const float v = v0 * (1.0f - dy) + v1 * dy;

                    const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                    const int i = 3 * (y * nx3 + x) + c;

                    res[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
                }
            }
        }

        output_slices.buf.resize(1);
        output_slices.buf[0] = std::move(res);

        return output_slices;
    }
};

struct llama_vision_processor_uhd : llama_vision_processor {
    llama_vision_processor_uhd(const llama_vision_context & ctx) : llama_vision_processor(ctx) {}

    int ensure_divide(int length, int patch_size) {
        return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
    }

    img_size find_best_resize(const img_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width = original_size.width;
        int height = original_size.height;
        if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
            float r = static_cast<float>(width) / height;
            height = static_cast<int>(scale_resolution / std::sqrt(r));
            width = static_cast<int>(height * r);
        }
        int best_width = ensure_divide(width, patch_size);
        int best_height = ensure_divide(height, patch_size);
        return img_size(best_width, best_height);
    }

    img_size get_refine_size(const img_size & original_size, const img_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width = original_size.width;
        int height = original_size.height;
        int grid_x = grid.width;
        int grid_y = grid.height;

        int refine_width = ensure_divide(width, grid_x);
        int refine_height = ensure_divide(height, grid_y);

        int grid_width = refine_width / grid_x;
        int grid_height = refine_height / grid_y;

        // auto best_grid_size = find_best_resize(std::make_tuple(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); (old line)
        auto best_grid = find_best_resize({grid_width, grid_height}, scale_resolution, patch_size, allow_upscale); // (new line) => fixes conversion for make_tuple to make_pair

        // img_size refine_size = std::make_tuple(best_grid_width * grid_x, best_grid_height * grid_y); (old line)
        img_size refine_size = img_size(best_grid.width * grid_x, best_grid.height * grid_y); // (new line)
        return refine_size;
    }

    img_size find_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        std::vector<img_size> candidate_grids;
        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.emplace_back(m, split_grids_nums / m);
                }
                ++m;
            }
        }

        img_size best_grid = img_size(1, 1);
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

    std::vector<std::vector<llama_image_u8>> slice_image(
            const llama_image_u8 & img,
            const int max_slice_nums = 9,
            const int scale_resolution = 448,
            const int patch_size = 14) {
        const img_size original_size = img_size(img.nx, img.ny);
        const int original_width = img.nx;
        const int original_height = img.ny;
        const float log_ratio = log(1.0*original_width/original_height);
        const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
        const int multiple = fmin(ceil(ratio), max_slice_nums);

        std::vector<std::vector<llama_image_u8>> images;
        LLAMA_LOG_DEBUG("%s: multiple %d\n", __func__, multiple);
        images.push_back(std::vector<llama_image_u8>());

        if (multiple <= 1) {
            auto best_size = find_best_resize(original_size, scale_resolution, patch_size, true);
            llama_image_u8 source_image;
            bicubic_resize(img, source_image, best_size.width, best_size.height);
            // source_image = image.resize(best_size, Image.Resampling.BICUBIC)
            images.back().push_back(source_image);
        } else if (multiple > 1) {
            auto best_size = find_best_resize(original_size, scale_resolution, patch_size);
            llama_image_u8 source_image;
            bicubic_resize(img, source_image, best_size.width, best_size.height);
            // source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
            LLAMA_LOG_DEBUG("%s: image_size: %d %d; source_image size: %d %d\n", __func__, img.nx, img.ny, best_size.width, best_size.height);
            images.back().push_back(source_image);

            img_size best_grid = find_best_grid(max_slice_nums, multiple, log_ratio);
            LLAMA_LOG_DEBUG("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img.nx, img.ny, best_grid.width, best_grid.height);

            auto refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
            llama_image_u8 refine_image;
            // TODO: so far, we spend most of the time in bicubic_resize, we should optimize it
            bicubic_resize(img, refine_image, refine_size.width, refine_size.height);

            LLAMA_LOG_DEBUG("%s: refine_image_size: %d %d; refine_size: %d %d\n", __func__, refine_image.nx, refine_image.ny, refine_size.width, refine_size.height);

            // split_to_patches
            int width = refine_image.nx;
            int height = refine_image.ny;
            int grid_x = int(width / best_grid.width);
            int grid_y = int(height / best_grid.height);
            for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.height; patches_i += grid_y, ic += 1){
                std::vector<llama_image_u8> patches_out;
                images.push_back(std::vector<llama_image_u8>());
                for (int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.width; patches_j += grid_x, jc += 1) {
                    llama_image_u8 patch;
                    patch.nx = grid_x;
                    patch.ny = grid_y;
                    patch.buf.resize(3 * patch.nx * patch.ny);
                    for (int y = patches_i; y < patches_i + grid_y; ++y) {
                        for (int x = patches_j; x < patches_j + grid_x; ++x) {
                            const int i = 3 * (y * refine_image.nx + x);
                            const int j = 3 * ((y-patches_i) * patch.nx + (x-patches_j));
                            patch.buf[j]   = refine_image.buf[i];
                            patch.buf[j+1] = refine_image.buf[i+1];
                            patch.buf[j+2] = refine_image.buf[i+2];
                        }
                    }
                    patches_out.push_back(std::move(patch));
                }
                images.push_back(std::move(patches_out));
            }
        }
        return images;
    }

    virtual llama_vision_tokens tokenize(const llama_image_u8 & img) override {
        auto & params = ctx.model->hparams;

        std::vector<std::vector<llama_image_u8>> imgs = slice_image(img);

        llama_vision_tokens output;
        output.n_px = get_n_patches_x(ctx);
        output.n_py = get_n_patches_y(ctx);
        output.px = params.patch_size;
        output.py = params.patch_size;

        for (size_t i = 0; i < imgs.size(); ++i) {
            for (size_t j = 0; j < imgs[i].size(); ++j) {
                std::vector<float> res;
                normalize_image_u8_to_f32(imgs[i][j], res, params.image_mean, params.image_std);
                output.buf.push_back(res);
            }
        }

        return output;
    }
};

//
// cgraph builder
//

// TODO: move this to llm_build_context in llama.cpp
struct llama_vision_graph_builder {
    llama_vision_context & ctx;
    const llama_vision_model & model;
    struct ggml_context * ctx0;
    int batch_size;
    int hidden_size;
    int n_head;
    int d_head;
    int patch_size;
    float eps;
    int num_patches;
    int num_positions;
    int img_w;
    int img_h;
    bool use_gelu;
    int n_layers;
    int rs_n_embd;
    vision_projector_type proj_type;

    llama_vision_graph_builder(llama_vision_context & ctx, const llama_vision_tokens & inp) : ctx(ctx), model(*ctx.model) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx.buf_compute_meta.size(),
            /*.mem_buffer =*/ ctx.buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };
        ctx0 = ggml_init(params);

        auto & hparams = ctx.model->hparams;

        batch_size    = inp.buf.size();
        hidden_size   = hparams.hidden_size;
        n_head        = hparams.n_head;
        d_head        = hidden_size / n_head;
        patch_size    = hparams.patch_size;
        eps           = hparams.eps;
        num_patches   = inp.n_px * inp.n_py;
        num_positions = num_patches + (model.class_embedding ? 1 : 0);
        img_w         = inp.px * inp.n_px;
        img_h         = inp.py * inp.n_py;
        use_gelu      = hparams.use_gelu;
        n_layers      = (int)hparams.n_layer + hparams.select_layer;
        proj_type     = hparams.proj_type;
    }

    ~llama_vision_graph_builder() {
        ggml_free(ctx0);
    }

    struct ggml_tensor * build_inp() {
        struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_w, img_h, 3, batch_size);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);

        struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

        inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        if (model.patch_bias) {
            inp = ggml_add(ctx0, inp, model.patch_bias);
        }
        // auto * ne = inp->ne; printf("%d %d %d %d\n", ne[0], ne[1], ne[2], ne[3]);

        struct ggml_tensor * embd = inp;
        if (model.class_embedding) {
            embd = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
            ggml_set_name(embd, "inp_embd");
            ggml_set_input(embd);

            embd = ggml_acc(ctx0, embd, model.class_embedding,
                embd->nb[1], embd->nb[2], embd->nb[3], 0);
            embd = ggml_acc(ctx0, embd, inp,
                embd->nb[1], embd->nb[2], embd->nb[3], model.class_embedding->nb[1]);
        }

        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
        ggml_set_name(positions, "inp_pos");
        ggml_set_input(positions);

        embd = ggml_add(ctx0,
            embd,
            ggml_get_rows(ctx0, model.position_embeddings, positions));

        return embd;
    }

    struct ggml_tensor * build_pre_norm(struct ggml_tensor * cur) {
        if (model.pre_norm_w) {
            cur = ggml_norm(ctx0, cur, eps);
            ggml_set_name(cur, "pre_ln");

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.pre_norm_w), model.pre_norm_b);
        }
        return cur;
    }

    struct ggml_tensor * build_post_norm(struct ggml_tensor * cur) {
        if (model.post_norm_w) {
            cur = ggml_norm(ctx0, cur, eps);
            ggml_set_name(cur, "post_ln");

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.post_norm_w), model.post_norm_b);
        }
        return cur;
    }

    struct ggml_tensor * build_layer(struct ggml_tensor * inpL, int il) {
        struct ggml_tensor * cur = inpL;

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.layers[il].norm_in_w),
                model.layers[il].norm_in_b);
        }

        // self-attention
        {
            struct ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].q_w, cur),
                model.layers[il].q_b);

            Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].k_w, cur),
                model.layers[il].k_b);

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].v_w, cur),
                model.layers[il].v_b);

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].output_w, cur), model.layers[il].output_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.layers[il].norm_out_w),
                model.layers[il].norm_out_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ffn_up_b);

        if (use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ffn_down_b);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);

        return cur;
    }

    struct ggml_tensor * build_vit() {
        struct ggml_tensor * cur = build_inp();
        cur = build_pre_norm(cur);
        for (int il = 0; il < n_layers; il++) {
            cur = build_layer(cur, il);
        }
        cur = build_post_norm(cur);
        return cur;
    }

    // graph for each vision arch

    struct ggml_cgraph * build_llava() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VISION_GRAPH_MAX_NODE, false);
        struct ggml_tensor * cur = build_vit();

        // llava projector
        {
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1]);

            struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
            ggml_set_name(patches, "inp_patches");
            ggml_set_input(patches);

            // shape [1, 576, 1024]
            // ne is whcn, ne = [1024, 576, 1, 1]
            cur = ggml_get_rows(ctx0, cur, patches);

            if (proj_type == VISION_PROJECTOR_TYPE_MLP) {
                cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
                cur = ggml_add(ctx0, cur, model.mm_1_b);

                cur = ggml_gelu(ctx0, cur);
                cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
                cur = ggml_add(ctx0, cur, model.mm_2_b);

            } else if (proj_type == VISION_PROJECTOR_TYPE_LDPV2) {
                int n_patch = 24;
                struct ggml_tensor * mlp_0 = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, cur);
                mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
                mlp_0 = ggml_gelu(ctx0, mlp_0);
                struct ggml_tensor * mlp_2 = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, mlp_0);
                mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
                // mlp_2 ne = [2048, 576, 1, 1]
                // // AVG Pool Layer 2*2, strides = 2
                mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 0, 2, 3));
                // mlp_2 ne = [576, 2048, 1, 1]
                mlp_2 = ggml_reshape_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
                // mlp_2 ne [24, 24, 2048, 1]
                mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
                // weight ne = [3, 3, 2048, 1]
                struct ggml_tensor * peg_0 = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
                peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
                peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
                mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
                peg_0 = ggml_add(ctx0, peg_0, mlp_2);
                peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
                cur = ggml_cont(ctx0, peg_0);

            } else {
                GGML_ASSERT(false && "unsupported proj type");
            }
        }

        ggml_set_name(cur, "output");
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_minicpmv() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VISION_GRAPH_MAX_NODE, false);
        struct ggml_tensor * cur = build_vit();

        // minicpmv resampler projector
        {
            int hidden_size = llama_vision_n_mmproj_embd(*ctx.model);
            struct ggml_tensor * q = model.mm_model_query;
            // layernorm
            {
                q = ggml_norm(ctx0, q, eps);
                q = ggml_add(ctx0, ggml_mul(ctx0, q, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            }

            struct ggml_tensor * v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, cur);
            // layernorm
            {
                v = ggml_norm(ctx0, v, eps);
                v = ggml_add(ctx0, ggml_mul(ctx0, v, model.mm_model_ln_kv_w), model.mm_model_ln_kv_b);
            }

            // position
            struct ggml_tensor * k = ggml_add(ctx0, v, model.mm_model_pos_embed_k);

            // attention
            {
                const int d_head = 128;
                int n_head = hidden_size/d_head;
                int num_query = -1;
                if (model.hparams.proj_type == VISION_PROJECTOR_TYPE_MINICPMV_2_5) {
                    num_query = 96;
                } else if (model.hparams.proj_type == VISION_PROJECTOR_TYPE_MINICPMV_2_6) {
                    num_query = 64;
                }

                struct ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q), model.mm_model_attn_q_b);
                Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
                struct ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k), model.mm_model_attn_k_b);
                struct ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v), model.mm_model_attn_v_b);
                // permute
                Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_query, batch_size);
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // TODO: do this when converting the model
                Q = ggml_reshape_3d(ctx0, Q, d_head, num_query, n_head * batch_size);
                K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
                K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3)); // TODO: do this when converting the model
                K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);
                V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
                V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // TODO: do this when converting the model
                V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                KQ = ggml_soft_max_inplace(ctx0, KQ);
                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
                KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_query, n_head, batch_size);
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3); // TODO: do this when converting the model
                KQV = ggml_cont_3d(ctx0, KQV, hidden_size, num_query, batch_size);

                cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_o_w, KQV), model.mm_model_attn_o_b);
            }
            // layernorm
            {
                cur = ggml_norm(ctx0, cur, eps);
                cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.mm_model_ln_post_w), model.mm_model_ln_post_b);
            }
            cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);
        }

        // add <image> and </image> token embeddings
        cur = ggml_concat(ctx0, model.mm_tok_embd_image, cur, 1);
        cur = ggml_concat(ctx0, cur, model.mm_tok_embd_end_image, 1);

        ggml_set_name(cur, "output");
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_idefics3() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VISION_GRAPH_MAX_NODE, false);
        struct ggml_tensor * cur = build_vit();

        // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
        {
            const int scale_factor = model.hparams.scale_factor;
            const int n_embd = cur->ne[0];
            const int seq    = cur->ne[1];
            const int bsz    = 1; // batch size, always 1 for now since we don't support batching
            const int height = std::sqrt(seq);
            const int width  = std::sqrt(seq);
            cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, cur),
                n_embd * scale_factor * scale_factor,
                height / scale_factor,
                width / scale_factor,
                bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, cur),
                n_embd * scale_factor * scale_factor,
                seq / (scale_factor * scale_factor),
                bsz);

            cur = ggml_mul_mat(ctx0, model.projection, cur);
        }

        ggml_set_name(cur, "output");
        ggml_build_forward_expand(gf, cur);

        return gf;
    }
};

static int32_t llama_vision_encode_impl(llama_vision_context & ctx, const llama_vision_tokens & inp) {
    int batch_size = inp.buf.size();
    auto & model = *ctx.model;
    auto & hparams = ctx.model->hparams;

    if (hparams.arch == LLM_ARCH_VISION_LLAVA) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }

    img_size image_size     = img_size((int)hparams.image_size, (int)hparams.image_size);
    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size.width / patch_size) * (image_size.height / patch_size));
    const int num_positions = num_patches + (model.class_embedding ? 1 : 0);

    LLAMA_LOG_DEBUG("%s: image_size = %d\n", __func__, hparams.image_size);
    LLAMA_LOG_DEBUG("%s: num_positions = %d\n", __func__, num_positions);

    // build the inference graph
    llama_vision_graph_builder builder(ctx, inp);
    ggml_cgraph * gf;
    switch(hparams.arch) {
        case LLM_ARCH_VISION_LLAVA:
        case LLM_ARCH_VISION_MOBILEVLM:
            gf = builder.build_llava();
            break;
        case LLM_ARCH_VISION_MINICPMV:
            gf = builder.build_minicpmv();
            break;
        case LLM_ARCH_VISION_IDEFICS3:
            gf = builder.build_idefics3();
            break;
        default:
            GGML_ASSERT(false && "unsupported vision arch");
    }

    // alloc memory for graph
    bool ok = ggml_backend_sched_alloc_graph(ctx.sched.get(), gf);
    if (!ok) {
        LLAMA_LOG_ERROR("failed to alloc memory for graph\n");
        return -1;
    }

    // set raw input
    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        std::vector<float> inp_buf(ggml_nelements(inp_raw));

        for (int i = 0; i < batch_size; i++) {
            const int nx = inp.px * inp.n_px;
            const int ny = inp.py * inp.n_py;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            inp_buf[(b * 3 * n) + k * n + y * nx + x] = inp.buf[b][3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, inp_buf.data(), 0, ggml_nbytes(inp_raw));
    }

    if (model.class_embedding) {
        struct ggml_tensor * inp_embd = ggml_graph_get_tensor(gf, "inp_embd");
        ggml_set_zero(inp_embd);
    }

    if (hparams.arch == LLM_ARCH_VISION_MINICPMV) {
        // inspired from siglip:
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
        struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "inp_pos");
        std::vector<int> buf(ggml_nelements(positions));
        GGML_ASSERT(num_positions == (int)buf.size());

        int bucket_coords_h[70];
        int bucket_coords_w[70];
        size_t h = inp.py;
        size_t w = inp.py;
        for (size_t i = 0; i < h; i++) {
            bucket_coords_h[i] = std::floor(70.0*i/h);
        }
        for (size_t i = 0; i < w; i++) {
            bucket_coords_w[i] = std::floor(70.0*i/w);
        }
        for (size_t i = 0, id = 0; i < h; i++){
            for (size_t j = 0; j < w; j++){
                buf[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
            }
        }
        ggml_backend_tensor_set(positions, buf.data(), 0, ggml_nbytes(positions));

    } else {
        struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "inp_pos");
        std::vector<int> pos_buf(ggml_nelements(positions));
        GGML_ASSERT(num_positions == (int)pos_buf.size());
        for (int i = 0; i < num_positions; i++) {
            pos_buf[i] = i;
        }
        ggml_backend_tensor_set(positions, pos_buf.data(), 0, ggml_nbytes(positions));
    }

    struct ggml_tensor * patches = ggml_graph_get_tensor(gf, "inp_patches");
    if (patches) {
        std::vector<int> patches_buf(ggml_nelements(patches));
        GGML_ASSERT(num_patches == (int)patches_buf.size());
        for (int i = 0; i < num_patches; i++) {
            patches_buf[i] = i + 1;
        }
        ggml_backend_tensor_set(patches, patches_buf.data(), 0, ggml_nbytes(patches));
    }

    // compute
    LLAMA_LOG_DEBUG("%s: compute start\n", __func__);
    int64_t t_start = ggml_time_ms();
    ggml_backend_sched_graph_compute(ctx.sched.get(), gf);

    // the last node is the embedding tensor
    struct ggml_tensor * output_node = ggml_graph_node(gf, -1);
    //LLAMA_LOG_INFO("%s: output tensor shape = %lld %lld %lld %lld\n", __func__, output->ne[0], output->ne[1], output->ne[2], output->ne[3]);
    LLAMA_LOG_DEBUG("%s: compute time = %lld ms\n", __func__, ggml_time_ms() - t_start);

    // copy output node to context
    if (ctx.ctx_ggml) {
        ggml_free(ctx.ctx_ggml);
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx.ctx_ggml = ggml_init(params);
    ctx.output = ggml_dup_tensor(ctx.ctx_ggml, output_node);
    ggml_backend_alloc_ctx_tensors_from_buft(ctx.ctx_ggml, ctx.model->buft);
    ggml_backend_tensor_copy(output_node, ctx.output);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
// public API

struct llama_vision_context_params llama_vision_context_default_params() {
    return {
        /*.n_threads =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
    };
}

struct llama_vision_context * llama_vision_init_from_model(const struct llama_model * model, struct llama_vision_context_params params) {
    if (!model->has_vision) {
        return nullptr;
    }

    llama_vision_context * ctx = new llama_vision_context;
    ctx->model = &model->vit;

    // TODO: this looks ugly, mostly copied from llama.cpp, refactor it in the future

    // init backends
    {
        // add CPU backend
        ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (ctx->backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            llama_vision_free(ctx);
            return nullptr;
        }
        ctx->backends.emplace_back(ctx->backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : ctx->backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                ggml_backend_set_n_threads_fn(backend.get(), params.n_threads);
            }
        }
    }

    // scheduler and compute buffers
    {
        // buffer types used for the compute buffer of each backend
        std::vector<ggml_backend_buffer_type_t> backend_buft;
        std::vector<ggml_backend_t> backend_ptrs;
        for (auto & backend : ctx->backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model->devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model->devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }
            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
        }

        const size_t max_nodes = model->max_nodes();

        // buffer used to store the computation graph and the tensor meta data
        ctx->buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

        // TODO: support pipeline_parallel
        const bool pipeline_parallel = false;

        ctx->sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel));

        if (pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(ctx->sched.get()));
        }
    }

    const size_t max_nodes = VISION_GRAPH_MAX_NODE; // TODO: make it dynamic
    ctx->buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    return ctx;
}

void llama_vision_free(struct llama_vision_context * ctx) {
    if (ctx->ctx_ggml) {
        ggml_free(ctx->ctx_ggml);
    }
    delete ctx;
}

struct llama_vision_bitmap * llama_vision_bitmap_init(uint32_t nx, uint32_t ny) {
    llama_vision_bitmap * bmp = new llama_vision_bitmap;
    bmp->nx = nx;
    bmp->ny = ny;
    bmp->data = (unsigned char *)malloc(3 * nx * ny);
    return bmp;
}

void llama_vision_bitmap_free(llama_vision_bitmap * bmp) {
    free(bmp->data);
    delete bmp;
}

struct llama_vision_tokens * llama_vision_tokenize(
        struct llama_vision_context * ctx,
        struct llama_vision_bitmap * bmp) {
    switch (ctx->model->hparams.arch) {
        case LLM_ARCH_VISION_LLAVA:
        case LLM_ARCH_VISION_MOBILEVLM:
        case LLM_ARCH_VISION_IDEFICS3:
            return new llama_vision_tokens(llama_vision_processor_llava(*ctx).tokenize(*bmp));
        case LLM_ARCH_VISION_MINICPMV:
            return new llama_vision_tokens(llama_vision_processor_llava(*ctx).tokenize(*bmp));
        default:
            GGML_ASSERT(false && "unsupported arch");
    }
}

void llama_vision_tokens_free(llama_vision_tokens * p) {
    delete p;
}

int32_t llama_vision_encode(struct llama_vision_context * ctx, struct llama_vision_tokens * p) {
    if (p->buf.empty()) {
        LLAMA_LOG_ERROR("%s: nothing to encode\n", __func__);
        return -1;
    }

    auto & hparams = ctx->model->hparams;
    switch (hparams.mm_patch_merge_type) {
        case MM_PATCH_MERGE_FLAT:
            {
                // flat / default llava-1.5 type embedding
                int32_t encoded = llama_vision_encode_impl(*ctx, *p);
                if (encoded != 0) {
                    LLAMA_LOG_ERROR("Unable to encode image\n");
                    return encoded;
                }
            } break;
        case MM_PATCH_MERGE_SPATIAL_UNPAD:
            {
                // TODO: support llava-1.6
                (void)0;
            } break;
        default:
            GGML_ASSERT(false && "unsupported mm_patch_merge_type");
    }

    return 0;
}

struct ggml_tensor * llama_vision_get_output_tensor(struct llama_vision_context * ctx) {
    return ctx->output;
}

////////////////////////////////////////////////////////////////////////////////////////
// for debugging
#ifndef NDEBUG

static int bmp_export(const struct llama_image_u8 &img, const std::string &location) {
    const uint32_t width = img.nx;
    const uint32_t height = img.ny;
    // swap red and blue channel
    std::vector<uint8_t> buffer(width*height*3);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t base = x*3 + y*3*width;
            buffer[base+2] = img.buf[base];
            buffer[base+1] = img.buf[base+1];
            buffer[base]   = img.buf[base+2];
        }
    }
    const bool hasAlphaChannel = false;

    std::ofstream fout(location, std::ios::out | std::ios::binary);

    if (fout.fail()) {
        return 0;
    }

    //Padding
    const uint8_t padding = hasAlphaChannel ? 0 : (4 - (width * 3) % 4) % 4;

    //Bitmap file header.
    const char signature[2] = { 'B', 'M' };
    const uint32_t fileSize = buffer.size() * sizeof(uint8_t) + padding * (height - 1) + 14 + 124;
    const uint32_t offset = 14 + 124;

    //Bitmap information header file
    const uint32_t DIBSize = 124;
    const int32_t bitmapWidth = width;
    const int32_t bitmapHeight = height;
    const uint16_t numPlanes = 1;
    const uint16_t bitsPerPixel = (hasAlphaChannel) ? 32 : 24;
    const uint32_t compressionMethod = (hasAlphaChannel) ? 3 : 0; //BI_RGB = 0, BI_BITFIELDS = 3
    const uint32_t bitmapSize = buffer.size() * sizeof(uint8_t);
    const int32_t horizontalResolution = 2834;
    const int32_t verticalResolution = 2834;
    const uint32_t numColors = 0;
    const uint32_t impColorCount = 0;
    const uint32_t redBitmask = (hasAlphaChannel) ? 0x0000FF00 : 0; //ARGB32 pixel format
    const uint32_t greenBitmask = (hasAlphaChannel) ? 0x00FF0000 : 0;
    const uint32_t blueBitmask = (hasAlphaChannel) ? 0xFF000000 : 0;
    const uint32_t alphaBitmask = (hasAlphaChannel) ? 0x000000FF : 0;

    //Writing the file header and information header to the file
    std::vector<uint8_t> header(offset, 0);
    header[0] = signature[0];
    header[1] = signature[1];

#define BMP_HEADERS(i, variableName)    header[i] = variableName; header[i+1] = variableName >> 8; header[i+2] = variableName >> 16; header[i+3] = variableName >> 24;

    BMP_HEADERS(2, fileSize);
    BMP_HEADERS(6, 0);
    BMP_HEADERS(10, offset);
    BMP_HEADERS(14, DIBSize);
    BMP_HEADERS(18, bitmapWidth);
    BMP_HEADERS(22, bitmapHeight);

    header[26] = (uint8_t)numPlanes;
    header[27] = (uint8_t)(numPlanes >> 8);
    header[28] = (uint8_t)bitsPerPixel;
    header[29] = (uint8_t)(bitsPerPixel >> 8);

    BMP_HEADERS(30, compressionMethod);
    BMP_HEADERS(34, (unsigned char)bitmapSize);
    BMP_HEADERS(38, (unsigned char)horizontalResolution);
    BMP_HEADERS(42, (unsigned char)verticalResolution);
    BMP_HEADERS(46, (unsigned char)numColors);
    BMP_HEADERS(50, (unsigned char)impColorCount);
    BMP_HEADERS(54, (unsigned char)redBitmask);
    BMP_HEADERS(58, (unsigned char)greenBitmask);
    BMP_HEADERS(62, (unsigned char)blueBitmask);
    BMP_HEADERS(66, alphaBitmask);

#undef BMP_HEADERS

    fout.write((char *)header.data(), sizeof(uint8_t) * header.size());

    //Writing the pixel array
    const uint32_t bWidth = bitsPerPixel / 8 * width;

    for (int i = height - 1; i >= 0; i--) {
        std::vector<uint8_t> row(buffer.begin() + i * bWidth, buffer.begin() + i * bWidth + bWidth);
        fout.write((char *)row.data(), row.size() * sizeof(uint8_t));
        fout.seekp(padding * sizeof(uint8_t), std::ios::cur);
    }

    fout.close();
    return 1;
}

#endif

