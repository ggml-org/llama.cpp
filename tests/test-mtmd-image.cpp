#include "mtmd-image.h"

#include <cstdio>

struct lfm2_tiling_case {
    clip_image_size size;
    bool expected_tiling;
};

int main() {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    const lfm2_tiling_case cases[] = {
        { {  704, 704 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & test : cases) {
        const bool actual =
            mtmd_image_preprocessor_lfm2::should_tile(hparams, test.size);

        if (actual != test.expected_tiling) {
            std::fprintf(
                stderr,
                "LFM2 tiling mismatch for %dx%d: expected %s, got %s\n",
                test.size.width,
                test.size.height,
                test.expected_tiling ? "tiled" : "single",
                actual               ? "tiled" : "single");

            return 1;
        }
    }

    return 0;
}
