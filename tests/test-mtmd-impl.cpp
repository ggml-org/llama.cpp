#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-tokenize-impl.h"

#include <iostream>
#include <string>
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

static std::string dump_marker_parts(const std::vector<mtmd_marker_part> & parts) {
    std::string s;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            s += "|";
        }
        if (parts[i].bitmap_i >= 0) {
            s += "BM" + std::to_string(parts[i].bitmap_i);
        } else {
            s += parts[i].text;
        }
    }
    return s;
}

MAKE_TEST(test_bind_media_markers_surplus_as_text) {
    const std::string marker = "<__media_abc__>";
    std::vector<mtmd_marker_part> parts;

    // quoted marker with 0 images stays text (the /props leak case)
    t.assert_true(mtmd_bind_media_markers("props said: " + marker + " ok?", marker, 0, parts));
    t.assert_equal(
        "0 bmp quoted marker",
        std::string("props said: |<__media_abc__>| ok?"),
        dump_marker_parts(parts));

    // first marker binds the image; leftover quoted marker stays text
    t.assert_true(mtmd_bind_media_markers(
        "see " + marker + " then quote " + marker, marker, 1, parts));
    t.assert_equal(
        "1 bmp + extra marker",
        std::string("see |BM0| then quote |<__media_abc__>"),
        dump_marker_parts(parts));

    // exact count still binds every marker
    t.assert_true(mtmd_bind_media_markers(
        "a " + marker + " b " + marker + " c", marker, 2, parts));
    t.assert_equal(
        "2 bmp exact",
        std::string("a |BM0| b |BM1| c"),
        dump_marker_parts(parts));

    t.assert_true("too few markers", !mtmd_bind_media_markers("no marker here", marker, 1, parts));

    t.assert_true(mtmd_bind_media_markers("plain text", marker, 0, parts));
    t.assert_equal("plain", std::string("plain text"), dump_marker_parts(parts));
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
