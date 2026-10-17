#include "testing.h"

#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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

MAKE_TEST(test_image_preprocessor_longest_edge) {
    clip_hparams hparams;
    hparams.image_longest_edge = 1540;
    hparams.patch_size = 14;
    hparams.n_merge = 2;

    const std::vector<std::pair<clip_image_size, clip_image_size>> cases = {
        { {  463,  463 }, {  476,  476 } },
        { {  463,  300 }, {  476,  308 } },
        { {  448,  280 }, {  448,  280 } },
        { { 2000, 1000 }, { 1540,  784 } },
    };

    for (const auto & [input, expected] : cases) {
        const clip_image_size actual = mtmd_image_preprocessor_longest_edge::get_target_size(hparams, input);
        const std::string name = std::to_string(input.width) + "x" + std::to_string(input.height);
        t.assert_equal(name + " width", expected.width, actual.width);
        t.assert_equal(name + " height", expected.height, actual.height);
    }
}

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
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
// mtmd decoder positions
//

MAKE_TEST(test_decoder_pos_minicpmv4_7) {
    // reference: transformers 5.19, MiniCPMV4_7Model.get_rope_index(downsample_mode="16x") on
    //   5 text tokens, <image>, the chunk, 3 text tokens
    // the chunk is laid out like the processor does:
    //   [ov] </image> <slice> [S00] </slice> <slice> [S01] </slice> \n <slice> [S10] </slice> <slice> [S11] </slice>
    // <image> is at position 5, the canvas origin, so the chunk starts at pos_0 = 6
    // the expected values are the reference positions for this exact prompt, without any token between the image and
    // the text. note: for an image followed by text, the chat template of the model puts a "\n" there, which the
    // reference counts to the image at [origin, origin, origin]; llama.cpp does not emit it, so on such prompts it stays
    // one token short of the reference. with that token, the reference gives the same positions as below to all the
    // other tokens and to the text after the image
    struct test_case {
        int ov_nx, ov_ny, n_col, n_row, nx, ny;
        llama_pos pos_text; // T = H = W of the first text token after the image
        std::vector<std::array<uint32_t, 3>> pos; // T, H, W of each token of the chunk
    };
    const llama_pos pos_0 = 6;
    const std::vector<test_case> cases = {
        // overview 5x2 tokens (w x h), 2x2 slices of 7x2 tokens, canvas 14x4
        { 5, 2, 2, 2, 7, 2, 20, {
            {5, 5, 5}, {5, 5, 8}, {5, 5, 11}, {5, 5, 15}, {5, 5, 18}, {5, 8, 5}, {5, 8, 8}, {5, 8, 11}, // ov ov ov ov ov ov ov ov
            {5, 8, 15}, {5, 8, 18}, {5, 9, 19}, {5, 5, 5}, {5, 5, 5}, {5, 5, 6}, {5, 5, 7}, {5, 5, 8}, // ov ov </image> <slice> slice slice slice slice
            {5, 5, 9}, {5, 5, 10}, {5, 5, 11}, {5, 6, 5}, {5, 6, 6}, {5, 6, 7}, {5, 6, 8}, {5, 6, 9}, // slice slice slice slice slice slice slice slice
            {5, 6, 10}, {5, 6, 11}, {5, 6, 11}, {5, 5, 12}, {5, 5, 12}, {5, 5, 13}, {5, 5, 14}, {5, 5, 15}, // slice slice </slice> <slice> slice slice slice slice
            {5, 5, 16}, {5, 5, 17}, {5, 5, 18}, {5, 6, 12}, {5, 6, 13}, {5, 6, 14}, {5, 6, 15}, {5, 6, 16}, // slice slice slice slice slice slice slice slice
            {5, 6, 17}, {5, 6, 18}, {5, 6, 18}, {5, 6, 19}, {5, 7, 5}, {5, 7, 5}, {5, 7, 6}, {5, 7, 7}, // slice slice </slice> \n <slice> slice slice slice
            {5, 7, 8}, {5, 7, 9}, {5, 7, 10}, {5, 7, 11}, {5, 8, 5}, {5, 8, 6}, {5, 8, 7}, {5, 8, 8}, // slice slice slice slice slice slice slice slice
            {5, 8, 9}, {5, 8, 10}, {5, 8, 11}, {5, 8, 11}, {5, 7, 12}, {5, 7, 12}, {5, 7, 13}, {5, 7, 14}, // slice slice slice </slice> <slice> slice slice slice
            {5, 7, 15}, {5, 7, 16}, {5, 7, 17}, {5, 7, 18}, {5, 8, 12}, {5, 8, 13}, {5, 8, 14}, {5, 8, 15}, // slice slice slice slice slice slice slice slice
            {5, 8, 16}, {5, 8, 17}, {5, 8, 18}, {5, 8, 18}, // slice slice slice </slice>
        } },
        // overview 4x3 tokens, no slices
        { 4, 3, 0, 0, 0, 0, 10, {
            {5, 5, 5}, {5, 5, 6}, {5, 5, 7}, {5, 5, 8}, {5, 6, 5}, {5, 6, 6}, {5, 6, 7}, {5, 6, 8}, // ov ov ov ov ov ov ov ov
            {5, 7, 5}, {5, 7, 6}, {5, 7, 7}, {5, 7, 8}, {5, 8, 9}, // ov ov ov ov </image>
        } },
    };

    for (const auto & tc : cases) {
        const std::string name = "overview " + std::to_string(tc.ov_nx) + "x" + std::to_string(tc.ov_ny) + ", " +
            std::to_string(tc.n_col) + "x" + std::to_string(tc.n_row) + " slices of " + std::to_string(tc.nx) + "x" + std::to_string(tc.ny);
        mtmd::input_chunk_ptr chunk(mtmd_test_create_canvas_chunk(tc.ov_nx, tc.ov_ny, tc.n_col, tc.n_row, tc.nx, tc.ny));
        const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk.get());

        const size_t n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
        t.assert_equal(name + ": n_tokens", tc.pos.size(), n_tokens);
        std::vector<mtmd_decoder_pos> pos(n_tokens);
        for (size_t i = 0; i < n_tokens; i++) {
            pos[i] = mtmd_image_tokens_get_decoder_pos(image_tokens, pos_0, i);
        }
        for (size_t i = 0; i < std::min(n_tokens, tc.pos.size()); i++) {
            // T is in z, H in y, W in x
            const auto & e = tc.pos[i];
            t.assert_equal(name + ": T, H, W of token " + std::to_string(i),
                std::to_string(e[0]) + ", " + std::to_string(e[1]) + ", " + std::to_string(e[2]),
                std::to_string(pos[i].z) + ", " + std::to_string(pos[i].y) + ", " + std::to_string(pos[i].x));
        }
        const llama_pos n_pos = mtmd_image_tokens_get_n_pos(image_tokens);
        t.assert_equal(name + ": position of the text after the image", tc.pos_text, pos_0 + n_pos);

        // the KV cache position t has to keep the chunk causal: the mask orders the cells by t, then by (y, x)
        // (llama_kv_cell_ext::is_2d_gt), and the text after the image has to see the whole chunk
        for (size_t i = 1; i < n_tokens; i++) {
            const auto & a = pos[i - 1];
            const auto & b = pos[i];
            t.assert_true(name + ": token " + std::to_string(i) + " comes after token " + std::to_string(i - 1) + " in the causal mask",
                std::tie(a.t, a.y, a.x) < std::tie(b.t, b.y, b.x));
        }
        t.assert_true(name + ": the chunk ends before the text", n_tokens > 0 && (llama_pos) pos[n_tokens - 1].t < pos_0 + n_pos);
    }
}

//
// mtmd_audio
//

MAKE_TEST(test_audio_preprocessor_conformer) {
    clip_hparams hparams;
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

    mtmd_audio_preprocessor_conformer preproc(hparams);
    preproc.initialize();
    std::vector<mtmd_audio_mel> mels;
    if (!t.assert_true("preprocess", preproc.preprocess(samples.data(), samples.size(), mels))) {
        return;
    }
    const mtmd_audio_mel & mel = mels[0];
    t.assert_equal("n_len", (int64_t) 101, mel.n_len);

    // reference values from NeMo AudioToMelSpectrogramPreprocessor (liquid-audio 1.3.0)
    // { mel bin, frame, value }
    const std::vector<std::tuple<int, int, float>> cases = {
        {   8,  0,  3.6104f }, // tones
        {  24, 50, -0.6371f }, // silence
        {  24, 60, -0.0476f }, // silence -> quiet tone
        { 100, 20, -0.0913f }, // tones, high bin
    };
    for (const auto & [bin, frame, expected] : cases) {
        const float actual = mel.data[bin * mel.n_len + frame];
        t.assert_true("mel[" + std::to_string(bin) + "][" + std::to_string(frame) + "] = " + std::to_string(actual) + ", expected " + std::to_string(expected), std::fabs(actual - expected) < 1e-3f);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
