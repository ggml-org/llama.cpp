#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-audio.h"
#include "mtmd-subcontext.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"
#include "mtmd-audio.h"
#include "mtmd-internal.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_apertus_audio_preprocessor) {
    mtmd_audio_mel output;
    const float samples[] = {0.0f, -2.0f, 1.0f};
    t.assert_equal(false, mtmd_apertus_audio_prepare(nullptr, 1, output));
    t.assert_equal(false, mtmd_apertus_audio_prepare(samples, 0, output));
    t.assert_equal(false, mtmd_apertus_audio_prepare(samples, size_t(INT32_MAX) + 1, output));
    t.assert_equal(true, mtmd_apertus_audio_prepare(samples, 3, output));
    t.assert_equal(int64_t(3), output.n_len);
    t.assert_equal(int64_t(3), output.n_len_org);
    t.assert_equal(int64_t(1), output.n_mel);
    const float peak = std::pow(10.0f, -3.0f / 20.0f);
    t.assert_equal(true, std::abs(output.data[1] + peak) < 1e-6f);
    t.assert_equal(true, std::abs(output.data[2] - peak / 2) < 1e-6f);
    const auto valid = output.data;
    for (float sample : {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        t.assert_equal(false, mtmd_apertus_audio_prepare(&sample, 1, output));
        t.assert_equal(true, output.data == valid);
    }
    const float silence[] = {0.0f, 0.0f};
    t.assert_equal(true, mtmd_apertus_audio_prepare(silence, 2, output));
    t.assert_equal(true, output.data == std::vector<float>({0.0f, 0.0f}));
}

MAKE_TEST(test_apertus_image_preprocessor) {
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> sizes = {
        {{429, 806}, {432, 800}}, {{1920, 1080}, {1856, 1040}},
        {{1400, 1400}, {1392, 1392}}, {{4032, 3024}, {1616, 1200}}, {{1, 1}, {256, 256}},
    };
    for (const auto & entry : sizes) {
        const auto actual = mtmd_apertus_image_calc_size(entry.first, 16, 65536, 1960000);
        t.assert_equal(entry.second.width, actual.width);
        t.assert_equal(entry.second.height, actual.height);
    }
    const auto small = mtmd_apertus_image_calc_size({128, 128}, 16, 16384, 1960000);
    t.assert_equal(128, small.width);
    t.assert_equal(128, small.height);
    const auto narrow = mtmd_apertus_image_calc_size({1600, 8}, 16, 16384, 1960000);
    t.assert_equal(1824, narrow.width);
    t.assert_equal(16, narrow.height);
    for (const auto & args : std::vector<std::tuple<mtmd_image_size, int, int, int>> {
        {{0, 10}, 16, 65536, 1960000}, {{10, -1}, 16, 65536, 1960000},
        {{10, 10}, 0, 65536, 1960000}, {{10, 10}, 16, 0, 1960000},
        {{10, 10}, 16, 65536, 1},
        {{1, std::numeric_limits<int>::max()}, 16, 1, std::numeric_limits<int>::max()},
    }) {
        bool rejected = false;
        try {
            mtmd_apertus_image_calc_size(std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        t.assert_true("invalid dimensions rejected", rejected);
    }
    mtmd_image_u8 image;
    image.set_size({4, 3}, false);
    image.cpy_buf({
          0,  15,  30,  45,  60,  75,  90, 105, 120, 135, 150, 165,
        180, 195, 210, 225, 240,   0,  15,  30,  45,  60,  75,  90,
        105, 120, 135, 150, 165, 180, 195, 210, 225, 240,   0,  15,
    });
    // Torchvision antialiased uint8 bicubic, channels first, 4x3 -> 3x5.
    const std::vector<uint8_t> expected = {
          0,   9,  30,  64,  79, 104, 133, 148, 162,
         81,  96,  81,  84,  98,  60,  86, 110, 130,
        203, 218, 156, 120, 135,  14,  37,  52,  81,
        154, 169, 153, 157, 177, 138, 159,  49,  70,
        107, 122, 143, 177, 201, 226, 246,  54,  68,
    };
    const auto prepared = mtmd_apertus_image_prepare(image, {3, 5});
    const auto & pixels = prepared.get_ro_buf();
    if (t.assert_equal(expected.size(), pixels.size())) {
        for (size_t i = 0; i < pixels.size(); ++i) {
            t.assert_equal((float(expected[i]) - 127.5f) / 127.5f, pixels[i]);
        }
    }
}

MAKE_TEST(test_linear_media_v2) {
    std::vector<char> wire;
    const auto append = [&wire](auto value) {
        const auto * first = reinterpret_cast<const char *>(&value);
        wire.insert(wire.end(), first, first + sizeof(value));
    };
    append(uint64_t(2));
    append(uint32_t(MTMD_INPUT_CHUNK_TYPE_IMAGE));
    append(uint64_t(0));
    append(uint8_t(1));
    append(uint32_t(21)); // Full framed count, NORMAL positions, one image.
    append(uint32_t(1));
    append(uint32_t(0));
    append(uint32_t(0));
    append(uint32_t(1));
    append(uint64_t(0));
    append(uint8_t(0));
    append(uint64_t(1));
    append(uint8_t(0));
    append(uint8_t(0));
    append(int32_t(0)); // lead_pad
    append(int32_t(64));
    append(int32_t(64));
    append(uint8_t(0));
    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(wire.data(), wire.size()));
    if (!t.assert_true("load v2 fixture", bool(chunk))) {
        return;
    }
    t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(chunk.get()));
    t.assert_equal(llama_pos(21), mtmd_input_chunk_get_n_pos(chunk.get()));
    const auto * image = mtmd_input_chunk_get_tokens_image(chunk.get());
    for (size_t i = 0; i < 21; ++i) {
        const auto pos = mtmd_image_tokens_get_decoder_pos(image, 5, i);
        t.assert_true("linear position", pos.t == 5 + uint32_t(i) && pos.x == pos.t && pos.y == pos.t && pos.z == pos.t);
    }
    size_t size = 0;
    t.assert_equal(0, mtmd_input_chunk_save(chunk.get(), nullptr, 0, &size));
    std::vector<char> saved(size);
    t.assert_equal(0, mtmd_input_chunk_save(chunk.get(), saved.data(), saved.size(), nullptr));
    auto expected = wire;
    const uint64_t current_version = 4;
    std::memcpy(expected.data(), &current_version, sizeof(current_version));
    expected.insert(expected.begin() + 64, 5 * sizeof(int32_t), 0);
    t.assert_true("v2 migrated to current format", saved == expected);
    mtmd::input_chunk_ptr restored(mtmd_input_chunk_load(saved.data(), saved.size()));
    if (!t.assert_true("load current format", bool(restored))) {
        return;
    }
    t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(restored.get()));
    t.assert_equal(llama_pos(21), mtmd_input_chunk_get_n_pos(restored.get()));
    const auto * restored_image = mtmd_input_chunk_get_tokens_image(restored.get());
    for (size_t i = 0; i < 21; ++i) {
        const auto pos = mtmd_image_tokens_get_decoder_pos(restored_image, 5, i);
        t.assert_true("restored linear position", pos.t == 5 + uint32_t(i) && pos.x == pos.t && pos.y == pos.t && pos.z == pos.t);
    }
    auto v3 = expected;
    const uint64_t version3 = 3;
    std::memcpy(v3.data(), &version3, sizeof(version3));
    mtmd::input_chunk_ptr loaded_v3(mtmd_input_chunk_load(v3.data(), v3.size()));
    if (t.assert_true("load v3 fixture", bool(loaded_v3))) {
        t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(loaded_v3.get()));
        t.assert_equal(llama_pos(21), mtmd_input_chunk_get_n_pos(loaded_v3.get()));
    }
    const auto reject_field = [&](size_t offset, auto value) {
        auto invalid = wire;
        std::memcpy(invalid.data() + offset, &value, sizeof(value));
        mtmd::input_chunk_ptr loaded(mtmd_input_chunk_load(invalid.data(), invalid.size()));
        t.assert_true("invalid v2 field rejected", !loaded);
    };
    reject_field(0, uint64_t(1));
    reject_field(0, uint64_t(5));
    reject_field(21, uint32_t(0));
    reject_field(21, std::numeric_limits<uint32_t>::max());
    reject_field(37, uint32_t(0));
    reject_field(41, std::numeric_limits<uint64_t>::max());
    reject_field(50, std::numeric_limits<uint64_t>::max());
    reject_field(60, int32_t(-1));
    reject_field(64, int32_t(0));
    for (size_t length : {size_t(0), size_t(8), wire.size() - 1}) {
        mtmd::input_chunk_ptr loaded(mtmd_input_chunk_load(wire.data(), length));
        t.assert_true("truncated v2 chunk rejected", !loaded);
    }
    mtmd::input_chunk_ptr placeholder(mtmd_input_chunk_get_placeholder(chunk.get()));
    t.assert_equal(size_t(21), mtmd_input_chunk_get_n_tokens(placeholder.get()));

    std::vector<llama_token> tokens(21, 0);
    llama_pos next = -1;
    for (int batch : {0, -1}) {
        t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size(),
            0, 0, batch, &next, nullptr, nullptr));
    }
    t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size(),
        std::numeric_limits<llama_pos>::max() - 20, 0, 8, &next, nullptr, nullptr));
    t.assert_equal(-1, mtmd_helper_decode_tokens_chunk(nullptr, nullptr, chunk.get(), tokens.data(), tokens.size() - 1,
        0, 0, 8, &next, nullptr, nullptr));
    t.assert_equal(llama_pos(-1), next);
}

MAKE_TEST(test_image_preprocessor_longest_edge) {
    mtmd_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const mtmd_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_dyn_size) {
    // { patch_size, image size, expected size }
    // a side that is an even number plus .5 merged patches rounds down, like Python round() in smart_resize
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        { 16, {  768, 400 }, {  768, 384 } },
        { 16, {  800, 336 }, {  800, 320 } },
        { 16, {  400, 400 }, {  384, 384 } },
        { 16, {  640, 240 }, {  640, 256 } },
        { 14, {  966, 560 }, {  952, 560 } },
    };

    for (const auto & [patch_size, input, expected] : cases) {
        mtmd_hparams hparams;
        hparams.patch_size = patch_size;
        hparams.n_merge = 2;
        hparams.image_round_half_even = true;
        hparams.set_limit_image_tokens(8, 4096);

        const mtmd_image_size actual = mtmd_image_preprocessor_dyn_size::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_qwen3vl_video) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { n_frames, frame size, expected size from Qwen3VLVideoProcessor smart_resize }
    const std::vector<std::tuple<int, mtmd_image_size, mtmd_image_size>> cases = {
        {  20, {  720,  358 }, {  704,  352 } }, // 720 / 32 = 22.5 rounds half to even
        { 768, { 1920, 1080 }, {  224,  128 } }, // over the longest_edge budget
        {   4, {   16,   40 }, {   32,   64 } }, // edge below the factor is upscaled first
        {   2, {   32,   32 }, {   64,   64 } }, // under the shortest_edge budget
        {   5, { 2760, 1552 }, { 2752, 1536 } }, // 5 / 2 = 2.5 rounds half to even
        {   4, {  250, 6000 }, {  256, 6016 } },
    };

    for (const auto & [n_frames, input, expected] : cases) {
        mtmd_image_size actual;
        const std::string name = std::to_string(n_frames) + " frames of " + std::to_string(input.width) + "x" + std::to_string(input.height);
        if (!t.assert_true(name + " has a frame size", preproc.get_video_frame_size(n_frames, input.width, input.height, actual))) {
            continue;
        }
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

static std::string image_size_to_str(const mtmd_image_size & size) {
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

static std::string resize_algo_to_str(resize_algo algo) {
    std::string str;
    switch (algo & ~RESIZE_ALGO_FLAG_TORCH) {
        case RESIZE_ALGO_BILINEAR: str = "bilinear"; break;
        case RESIZE_ALGO_BICUBIC:  str = "bicubic";  break;
        case RESIZE_ALGO_LANCZOS:  str = "lanczos";  break;
        default:                   str = "unknown";  break;
    }
    if (algo & RESIZE_ALGO_FLAG_TORCH) {
        str += "_torch";
    }
    return str;
}

// pseudo-random pixels: s = s * 1103515245 + 12345 from s = 1, one byte (s >> 16) & 255 per channel
static mtmd_image_u8 make_test_image(const mtmd_image_size & size) {
    mtmd_image_u8 img;
    img.set_size(size, false);
    uint32_t s = 1;
    auto next = [&s]() {
        s = s * 1103515245u + 12345u;
        return static_cast<uint8_t>(s >> 16);
    };
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            img.set_pixel(x, y, { next(), next(), next() });
        }
    }
    return img;
}

// FNV-1a of the pixels
static uint32_t hash_pixels(const mtmd_image_u8 & img, uint32_t hash = 2166136261u) {
    for (uint8_t v : img.get_ro_buf()) {
        hash = (hash ^ v) * 16777619u;
    }
    return hash;
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;

    // { image size, expected tiling }
    const std::vector<std::pair<mtmd_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }

    // { image size, expected size from HF smart_resize }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> resize_cases = {
        { {  400, 600 }, { 384, 608 } }, // 400 / 32 = 12.5 rounds half to even
        { {  480, 336 }, { 480, 320 } },
        { {   64, 169 }, { 160, 416 } }, // 169 * beta / 32 is exactly 13 in double
        { { 1024, 768 }, { 576, 416 } },
    };

    for (const auto & [size, expected] : resize_cases) {
        t.assert_equal("smart_resize for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::smart_resize(hparams, size)));
    }

    // { image size, expected grid from HF }
    const std::vector<std::pair<mtmd_image_size, mtmd_image_size>> grid_cases = {
        { {  720, 1600 }, { 2, 4 } }, // 0.45 is an exact tie between 2/4 and 2/5 in double, not in float
        { {  420, 1440 }, { 1, 3 } },
        { {  255, 2160 }, { 1, 8 } },
        { { 1024,  768 }, { 3, 2 } },
    };

    for (const auto & [size, expected] : grid_cases) {
        t.assert_equal("grid for " + image_size_to_str(size), image_size_to_str(expected),
            image_size_to_str(mtmd_image_preprocessor_lfm2::get_grid_layout(hparams, size.height, size.width)));
    }

    // do_image_splitting=false
    hparams.preproc_min_tiles = 1;
    hparams.preproc_max_tiles = 1;
    t.assert_equal("tiling for 1024x977 with max_tiles=1", std::string("single"),
            std::string(mtmd_image_preprocessor_lfm2::should_tile(hparams, { 1024, 977 }) ? "tiled" : "single"));
}

MAKE_TEST(test_image_preprocessor_lfm2_torch_resize) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);
    hparams.preproc_min_tiles            = 2;
    hparams.preproc_max_tiles            = 10;
    hparams.preproc_tile_size            = 512;
    hparams.preproc_max_pixels_tolerance = 2.0f;
    // Lfm2VlImageProcessor stretches the image to the tile grid
    hparams.image_pad_rf = PAD_NONE;

    // { resize algo, image size, expected hash of the tiles and the thumbnail from Lfm2VlImageProcessor (torchvision 0.29, native uint8 resize: arm64, or x86 with AVX2/AVX512) }
    const std::vector<std::tuple<resize_algo, mtmd_image_size, uint32_t>> cases = {
        { RESIZE_ALGO_BILINEAR, {  620,  410 }, 0xf5e9c941 }, // 608x416
        { RESIZE_ALGO_BILINEAR, { 2000, 1500 }, 0x3331f56c }, // 3x2 tiles, 576x416 thumbnail
        { RESIZE_ALGO_BICUBIC,  {  620,  410 }, 0x2b34eac9 },
        { RESIZE_ALGO_BICUBIC,  { 2000, 1500 }, 0xb035d942 },
    };

    for (const auto & [base_algo, size, expected] : cases) {
        const resize_algo algo = static_cast<resize_algo>(base_algo | RESIZE_ALGO_FLAG_TORCH);
        hparams.image_resize_algo_rf = algo;
        hparams.image_resize_algo_ov = algo;
        const mtmd_image_preprocessor_lfm2 preproc(hparams);
        const auto sliced = preproc.slice_image(make_test_image(size), preproc.get_slice_instructions(size));

        uint32_t hash = 2166136261u;
        for (const auto & slice : sliced.slices) {
            hash = hash_pixels(slice, hash);
        }
        hash = hash_pixels(sliced.overview, hash);
        t.assert_equal("pixels of " + std::to_string(size.width) + "x" + std::to_string(size.height) + " with resize algo " + resize_algo_to_str(algo), expected, hash);
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// mtmd_helper video
//

MAKE_TEST(test_video_config) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { template, n_frames and fps of the source, expected fps, n_frames and frame size }
    // the 1920x1080 sizes are from Qwen3VLVideoProcessor smart_resize
    const std::vector<std::tuple<mtmd_video_template, int32_t, float, float, int32_t, mtmd_image_size>> cases = {
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,   36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3,      36000, 30.0f, 2.0f, 768, { 224, 128 } },
        { MTMD_VIDEO_TEMPLATE_LFM3_OMNI, 36000, 30.0f, 2.0f, 384, { 320, 192 } },
        { MTMD_VIDEO_TEMPLATE_QWEN3VL,      -1, 30.0f, 2.0f,   0, {   0,   0 } }, // frame count unknown
        { MTMD_VIDEO_TEMPLATE_DEFAULT,   36000, 30.0f, 0.0f,   0, {   0,   0 } },
    };

    for (const auto & [tmpl, n_frames, fps, expected_fps, expected_n_frames, expected_size] : cases) {
        const mtmd_video_config cfg = mtmd_video_config_init(tmpl, &preproc, 1920, 1080, n_frames, fps);
        const std::string name = "template " + std::to_string(tmpl) + " with " + std::to_string(n_frames) + " frames";
        t.assert_equal(name + " template", (int) tmpl, (int) cfg.tmpl);
        t.assert_equal(name + " fps", expected_fps, cfg.fps);
        t.assert_equal(name + " n_frames", expected_n_frames, cfg.n_frames);
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ cfg.frame_width, cfg.frame_height }));
    }
}

// n_frames source frames, each one starts with its index
struct test_video_source : mtmd_helper_video_seq {
    int32_t n_frames;
    int32_t n_read = 0;

    test_video_source(int32_t n_frames) : n_frames(n_frames) {}

    bool read_raw_frame() override {
        if (n_read >= n_frames) {
            return false;
        }
        frame_buf[0] = static_cast<uint8_t>(n_read & 255);
        frame_buf[1] = static_cast<uint8_t>(n_read >> 8);
        n_read++;
        return true;
    }
};

// text as is, a frame as [source index], an audio bitmap as {n_samples}
static std::string video_sequence_to_str(testing & t, mtmd_helper_video_seq & video) {
    std::string out;
    mtmd_bitmap * bitmap;
    char * text;
    int32_t ret;
    while ((ret = video.read_next(&bitmap, &text)) == 0) {
        if (text) {
            out += text;
            free(text);
        } else if (mtmd_bitmap_is_audio(bitmap)) {
            out += "{" + std::to_string(mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float)) + "}";
            mtmd_bitmap_free(bitmap);
        } else {
            const unsigned char * data = mtmd_bitmap_get_data(bitmap);
            out += "[" + std::to_string(data[0] | data[1] << 8) + "]";
            mtmd_bitmap_free(bitmap);
        }
    }
    t.assert_equal("end of stream", -1, ret);
    return out;
}

MAKE_TEST(test_video_sequence_qwen3vl) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_qwen3vl preproc(hparams);

    // { source video, expected frame size and sequence from Qwen3VLProcessor (transformers 4.57.1) }
    const std::vector<std::tuple<mtmd_helper_video_src, mtmd_image_size, std::string>> cases = {
        { {  720, 358, 24.0f, 240 }, { 704, 352 },
            "<0.3 seconds>[0][13]<1.3 seconds>[25][38]<2.4 seconds>[50][63]<3.4 seconds>[75][88]<4.5 seconds>[101][113]"
            "<5.5 seconds>[126][138]<6.6 seconds>[151][164]<7.6 seconds>[176][189]<8.6 seconds>[201][214]<9.7 seconds>[226][239]" },
        // odd frame count: the last group repeats its last frame
        { {  640, 480, 10.0f,  35 }, { 640, 480 },
            "<0.3 seconds>[0][6]<1.4 seconds>[11][17]<2.5 seconds>[23][28]<3.4 seconds>[34]" },
        // fewer frames than the minimum of 4
        { {  320, 240, 30.0f,   3 }, { 320, 256 },
            "<0.0 seconds>[0][1]<0.1 seconds>[2]" },
        { { 1280, 720, 30000.0f / 1001.0f, 300 }, { 1280, 704 },
            "<0.3 seconds>[0][16]<1.3 seconds>[31][47]<2.4 seconds>[63][79]<3.4 seconds>[94][110]<4.5 seconds>[126][142]"
            "<5.5 seconds>[157][173]<6.6 seconds>[189][205]<7.6 seconds>[220][236]<8.7 seconds>[252][268]<9.7 seconds>[283][299]" },
    };

    for (const auto & [src, expected_size, expected] : cases) {
        const std::string name = std::to_string(src.n_frames) + " frames of " + std::to_string(src.width) + "x" + std::to_string(src.height);
        const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_QWEN3VL, &preproc, src.width, src.height, src.n_frames, src.fps);
        test_video_source video(src.n_frames);
        video.configure(cfg, src, 0.0f);
        video.src_alive = true;
        t.assert_equal(name + " frame size", image_size_to_str(expected_size), image_size_to_str({ (int) video.info.width, (int) video.info.height }));
        t.assert_equal(name + " sequence", expected, video_sequence_to_str(t, video));
    }
}

// the soundtrack of each group comes after its frames
MAKE_TEST(test_video_sequence_lfm3_omni) {
    mtmd_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    const mtmd_image_preprocessor_lfm3vl preproc(hparams);

    const mtmd_helper_video_src src = { 64, 64, 2.0f, 9 };
    const mtmd_video_config cfg = mtmd_video_config_init(MTMD_VIDEO_TEMPLATE_LFM3_OMNI, &preproc, src.width, src.height, src.n_frames, src.fps);
    test_video_source video(src.n_frames);
    video.configure(cfg, src, 0.0f);
    video.src_alive  = true;
    video.audio_rate = 100;
    video.soundtrack.assign(450, 0.0f);

    t.assert_equal("sequence",
        std::string("<0.2 seconds>[0][1]{100}<1.2 seconds>[2][3]{100}<2.2 seconds>[4][5]{100}<3.2 seconds>[6][7]{100}<4.0 seconds>[8]{50}"),
        video_sequence_to_str(t, video));
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_conformer) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // 0.4 s of tones, 0.2 s of silence, 0.4 s of a quiet tone
    const double pi = 3.14159265358979323846;
    const int sr = hparams.audio_sample_rate;
    std::vector<float> samples(sr, 0.0f);
    for (int i = 0; i < sr; i++) {
        const double ts = (double) i / sr;
        if (i < 0.4 * sr) {
            samples[i] = (float) (0.3 * std::sin(2 * pi * 300 * ts) + 0.2 * std::sin(2 * pi * 1200 * ts) + 0.1 * std::sin(2 * pi * 3500 * ts));
        } else if (i >= 0.6 * sr) {
            samples[i] = (float) (1e-3 * std::sin(2 * pi * 800 * ts));
        }
    }

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    // the result must not depend on the number of threads
    for (int n_threads : { 1, 4, 7 }) {
        mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
        preproc.mel_config.n_threads = n_threads;
        preproc.initialize();
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        if (!t.assert_true("preprocess", !out.entries.empty() && preproc.compute(out))) {
            return;
        }
        const mtmd_image_f32 & mel = out.entries[0];
        t.assert_equal("n_len", 101, mel.nx());

        for (const auto & [bin, frame, expected] : cases) {
            const float actual = mel.get_ro_buf()[bin * mel.nx() + frame];
            t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
        }
    }
}

MAKE_TEST(test_audio_preprocessor_threads) {
    mtmd_hparams hparams;
    hparams.n_mel_bins        = 128;
    hparams.audio_sample_rate = 16000;
    hparams.audio_n_fft       = 512;
    hparams.audio_window_len  = 400;
    hparams.audio_hop_len     = 160;

    // chirp: every frame has a different content
    std::vector<float> samples(hparams.audio_sample_rate * 5);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 2e-6 * i * i));
    }

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_preprocessor_conformer preproc(hparams, sched.get());
    preproc.initialize();
    // tokenize only gets the shapes, encode must get the same ones
    mtmd_audio_preproc_out ref = preproc.preprocess(samples.data(), samples.size());
    if (!t.assert_true("preprocess", ref.entries.size() == 1 && ref.needs_compute())) {
        return;
    }
    const mtmd_image_size shape = ref.entries[0].get_size();
    if (!t.assert_true("compute", preproc.compute(ref))) {
        return;
    }
    t.assert_true("shape", ref.entries.size() == 1 && ref.entries[0].get_size() == shape);
    t.assert_true("computed", !ref.needs_compute());

    // frames are independent, so the whole output must be bit-exact with the single-threaded one
    for (int n_threads : { 4, 12 }) {
        preproc.mel_config.n_threads = n_threads;
        mtmd_audio_preproc_out out = preproc.preprocess(samples.data(), samples.size());
        const bool ok = preproc.compute(out);
        t.assert_true("n_threads = " + std::to_string(n_threads), ok && out.entries[0].get_ro_buf() == ref.entries[0].get_ro_buf());
    }
}

MAKE_TEST(test_audio_log_mel_precision) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 200;

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched      = sched.get();
    config.n_threads  = 2;
    config.n_mel      = n_mel;
    config.n_fft_bins = 1 + n_fft / 2;
    config.hop_length = hop;
    config.cache      = &cache;

    // a loud tone next to a 74 dB weaker one: rounding the DFT inputs to F16 or TF32 loses the weak one
    const double pi = 3.14159265358979323846;
    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.5 * std::sin(2 * pi * 200 * i / 16000) + 1e-4 * std::sin(2 * pi * 6000 * i / 16000));
    }

    std::vector<float> out((size_t) n_frames * n_mel);
    if (!t.assert_true("log mel", mtmd_audio_log_mel(config, samples.data(), cache.hann_window.data(), n_frames, out.data()))) {
        return;
    }

    const int n_bins = n_fft / 2 + 1;
    double max_diff = 0.0;
    std::vector<double> power(n_bins);
    for (int i = 0; i < n_frames; i++) {
        for (int k = 0; k < n_bins; k++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < n_fft; j++) {
                const double x = (double) cache.hann_window[j] * samples[(size_t) i * hop + j];
                re += x * std::cos(2 * pi * j * k / n_fft);
                im -= x * std::sin(2 * pi * j * k / n_fft);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            for (int k = 0; k < n_bins; k++) {
                sum += power[k] * cache.filters.data[(size_t) m * n_bins + k];
            }
            const double expected = std::log10(std::max(sum, (double) config.mel_floor));
            max_diff = std::max(max_diff, std::fabs(expected - out[(size_t) m * n_frames + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 0.05);
}

MAKE_TEST(test_audio_log_mel_blocks) {
    const int n_mel    = 80;
    const int n_fft    = 400;
    const int hop      = 160;
    const int n_frames = 8192 + 100; // long audio is split into blocks of 8192 frames

    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
    cache.fill_mel_filterbank_matrix(n_mel, n_fft, 16000);

    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 256, false, true));

    mtmd_audio_mel_config config;
    config.sched       = sched.get();
    config.n_threads   = 2;
    config.n_mel       = n_mel;
    config.n_fft_bins  = 1 + n_fft / 2;
    config.hop_length  = hop;
    config.cache       = &cache;

    std::vector<float> samples((size_t) (n_frames - 1) * hop + n_fft);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i] = (float) (0.3 * std::sin(0.05 * i + 1e-6 * i * i));
    }

    // frames are independent: the frames around the seam must not change when computed on their own
    // not bit-exact, matmul kernels may round differently depending on the shape
    const int off   = 8192 - 50;
    const int n_sub = n_frames - off;
    std::vector<float> all((size_t) n_frames * n_mel);
    std::vector<float> sub((size_t) n_sub * n_mel);
    const float * window = cache.hann_window.data();
    if (!t.assert_true("all", mtmd_audio_log_mel(config, samples.data(), window, n_frames, all.data())) ||
        !t.assert_true("sub", mtmd_audio_log_mel(config, samples.data() + (size_t) off * hop, window, n_sub, sub.data()))) {
        return;
    }

    float max_diff = 0.0f;
    for (int j = 0; j < n_mel; j++) {
        for (int i = 0; i < n_sub; i++) {
            max_diff = std::max(max_diff, std::fabs(all[(size_t) j * n_frames + off + i] - sub[(size_t) j * n_sub + i]));
        }
    }
    t.assert_true("max_diff = " + std::to_string(max_diff), max_diff < 1e-5f);
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // the audio preprocessors run on the CPU backend
    ggml_backend_load_all();

    // the video helper logs every frame at debug level
    mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            fputs(text, stderr);
        }
    }, nullptr);

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
