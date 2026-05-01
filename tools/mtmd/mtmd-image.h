#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

// base class, models must inherit from this class
struct mtmd_image_preprocessor {
    const clip_hparams & hparams;

    mtmd_image_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_image_preprocessor() = default;
    virtual bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) = 0;

    void img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]);
    void img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst);
};

/**
 * implementation of LLaVA-UHD:
 *  - https://arxiv.org/pdf/2403.11703
 *  - https://github.com/thunlp/LLaVA-UHD
 *  - https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
 *
 * overview:
 *   - an image always have a single overview (downscaled image)
 *   - an image can have 0 or multiple slices, depending on the image size
 *   - each slice can then be considered as a separate image
 *
 * note: the term "slice" and "tile" are used interchangeably
 *
 * for example:
 *
 * [overview] --> [slice 1] --> [slice 2]
 *           |                |
 *           +--> [slice 3] --> [slice 4]
 */
struct mtmd_image_preprocessor_llava_uhd : mtmd_image_preprocessor {
    mtmd_image_preprocessor_llava_uhd(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;

    struct slice_coordinates {
        int x;
        int y;
        clip_image_size size;
    };

    struct slice_instructions {
        clip_image_size overview_size; // size of downscaled image
        clip_image_size refined_size;  // size of image right before slicing (must be multiple of slice size)
        clip_image_size grid_size;     // grid_size.width * grid_size.height = number of slices
        std::vector<slice_coordinates> slices;
    };

    // LFM2 override this function to implement its custom slicing logic
    virtual slice_instructions get_slice_instructions(const clip_image_size & original_size);

    std::vector<clip_image_u8_ptr> slice_image(const clip_image_u8 & img, const slice_instructions & inst, bool overview_first = true);

private:
    clip_image_size get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false);

    clip_image_size resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max);

    /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * For example, when given a list of resolutions:
     *  - 100x100
     *  - 200x100
     *  - 100x200
     *  - 200x200
     *
     * And an input image of size 111x200, then 100x200 is the best fit (least wasted resolution).
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
    clip_image_size select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions);
    int ensure_divide(int length, int patch_size);
    clip_image_size get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false);
    clip_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio);
};

// downscale or upscale the input image to fixed size
struct mtmd_image_preprocessor_fixed_size : mtmd_image_preprocessor {
    mtmd_image_preprocessor_fixed_size(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

// resize image to multiple of patch_size*n_merge, while preserving aspect ratio
// if image_resize_pad is true, the resized image will be padded, otherwise it will be either stretched or center-cropped depending on image_resize_pad
// this is used by models with native support for dynamic image size, for example: Qwen-VL, Pixtral, Kimi-VL, etc
struct mtmd_image_preprocessor_dyn_size : mtmd_image_preprocessor {
    mtmd_image_preprocessor_dyn_size(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

// similar to mtmd_image_preprocessor_dyn_size, but resize the image to have longest edge equal to hparams.image_longest_edge, while preserving aspect ratio
struct mtmd_image_preprocessor_longest_edge : mtmd_image_preprocessor {
    mtmd_image_preprocessor_longest_edge(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

// custom llava-uhd slicing logic for LFM2
// ref: https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py
struct mtmd_image_preprocessor_lfm2 : mtmd_image_preprocessor_llava_uhd {
    // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
    static constexpr int   min_tiles            = 2;
    static constexpr int   max_tiles            = 10;
    static constexpr float max_pixels_tolerance = 2.0f;
    static constexpr int   tile_size            = 512;

    using mtmd_image_preprocessor_llava_uhd::mtmd_image_preprocessor_llava_uhd;
    slice_instructions get_slice_instructions(const clip_image_size & original_size) override;

private:
    clip_image_size find_closest_aspect_ratio(
            float aspect_ratio,
            const std::vector<clip_image_size> & target_ratios,
            int width, int height);
    std::vector<clip_image_size> get_target_ratios();
    clip_image_size get_grid_layout(int height, int width);
};

struct mtmd_image_preprocessor_idefics3 : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_idefics3(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_internvl : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_internvl(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_deepseekocr : mtmd_image_preprocessor {
    mtmd_image_preprocessor_deepseekocr(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

// custom image preprocessing for Step3VL
// ref: https://huggingface.co/stepfun-ai/Step3-VL-10B/blob/main/processing_step3.py
struct mtmd_image_preprocessor_step3vl : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_step3vl(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
    static slice_instructions build_slice_instructions(const clip_hparams & params, const clip_image_size & prepared_size);

private:
    static constexpr int   default_image_longest_edge = 3024;
    static constexpr int   default_image_crop_size    = 504;
    static constexpr float small_aspect_ratio_limit   = 1.5f;
    static constexpr float wide_aspect_ratio_limit    = 4.0f;
    static constexpr float crop_rounding_threshold    = 0.2f;

    void img_u8_resize_bilinear_to_f32(
            const clip_image_u8 & src,
            clip_image_f32 & dst,
            int target_width,
            int target_height,
            const float mean[3],
            const float std[3]);
    static int get_image_longest_edge(const clip_hparams & params);
    static int determine_window_size(const clip_hparams & params, int longer, int shorter);
    static int calc_crop_extent(int length, int window_size);
    static std::vector<int> calc_grid(int length, int window_size);
    static clip_image_u8 prepare_image(const clip_image_u8 & img, const clip_hparams & params);
    static clip_image_u8 crop_with_black_padding(const clip_image_u8 & image, int x, int y, int w, int h);
};

struct mtmd_image_preprocessor_youtuvl : mtmd_image_preprocessor {
    mtmd_image_preprocessor_youtuvl(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};
