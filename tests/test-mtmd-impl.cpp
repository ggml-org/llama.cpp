#include "testing.h"

#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <cmath>
#include <iostream>
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

MAKE_TEST(test_image_preprocessor_gemma4_budget) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 3;
    // Reference get_aspect_ratio_preserving_size, patch16/pooling3.
    const std::vector<std::tuple<int, clip_image_size, clip_image_size>> cases = {
        {   70, { 256, 256 }, {  384,  384 } },
        {  140, { 256, 256 }, {  528,  528 } },
        {  280, { 256, 256 }, {  768,  768 } },
        {  560, { 256, 256 }, { 1104, 1104 } },
        { 1120, { 256, 256 }, { 1584, 1584 } },
        { 1120, { 280, 400 }, { 1344, 1920 } },
        { 1120, { 400, 280 }, { 1920, 1344 } },
        { 1120, {   1, 10000 }, {   48, 53760 } },
        { 1120, { 10000, 1 }, { 53760,   48 } },
        {  280, { 768, 768 }, {  768,  768 } },
    };
    for (const auto & [budget, original, expected] : cases) {
        hparams.custom_image_min_tokens = budget;
        hparams.custom_image_max_tokens = budget;
        hparams.set_limit_image_tokens(70, 1120);
        const auto actual = mtmd_image_preprocessor_gemma4::target_size(hparams, original);
        t.assert_equal("reference width", expected.width, actual.width);
        t.assert_equal("reference height", expected.height, actual.height);
        t.assert_true("aligned under budget", actual.width > 0 && actual.height > 0 && actual.width % 48 == 0 &&
            actual.height % 48 == 0 && int64_t(actual.width) * actual.height <= int64_t(budget) * 48 * 48);
    }
    hparams.custom_image_min_tokens = 1;
    hparams.custom_image_max_tokens = 280;
    hparams.set_limit_image_tokens(70, 1120);
    const auto small = mtmd_image_preprocessor_gemma4::target_size(hparams, { 16, 16 });
    t.assert_equal(768, small.width);
    t.assert_equal(768, small.height);
    for (int budget : {70,140,280,560,1120}) {
        hparams.custom_image_min_tokens = budget;
        hparams.custom_image_max_tokens = budget;
        hparams.set_limit_image_tokens(70,1120);
        for (const auto & size : std::vector<clip_image_size>{ {1,1},{1,2},{2,1},{47,49},{49,47},{384,384},
                {768,1024},{1024,768},{1920,1080},{1080,1920},{32768,1},{1,32768},
                {std::numeric_limits<int>::max(),1},{1,std::numeric_limits<int>::max()} }) {
            const auto result = mtmd_image_preprocessor_gemma4::target_size(hparams,size);
            t.assert_true("bounded extreme/aspect dimensions", result.width > 0 && result.height > 0 && result.width % 48 == 0 &&
                result.height % 48 == 0 && int64_t(result.width) * result.height <= int64_t(budget) * 48 * 48);
        }
    }
    for (int invalid=0;invalid<9;++invalid) {
        auto bad = hparams;
        clip_image_size size{256,256};
        if (invalid==0) size.width=0;
        if (invalid==1) size.height=-1;
        if (invalid==2) bad.patch_size=0;
        if (invalid==3) bad.n_merge=0;
        if (invalid==4) bad.patch_size=std::numeric_limits<int>::max();
        if (invalid==5) bad.image_max_pixels=-1;
        if (invalid==6) bad.image_max_pixels=1;
        if (invalid==7) bad.image_max_pixels=1121*48*48;
        if (invalid==8) bad.n_merge=std::numeric_limits<int>::max();
        bool rejected=false;
        try { mtmd_image_preprocessor_gemma4::target_size(bad,size); }
        catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("invalid metadata rejects before resizing",rejected);
    }
}

MAKE_TEST(test_image_preprocessor_qwen_resize_unchanged) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 4096);
    for (const auto & [input, expected] : std::vector<std::pair<clip_image_size, clip_image_size>>{
            { { 256, 256 }, { 256, 256 } }, { { 512, 384 }, { 512, 384 } },
            { { 8192, 8192 }, { 2048, 2048 } }, { { 16, 16 }, { 256, 256 } } }) {
        const auto actual = mtmd_image_preprocessor_dyn_size::target_size(hparams, input);
        t.assert_equal(expected.width, actual.width);
        t.assert_equal(expected.height, actual.height);
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
