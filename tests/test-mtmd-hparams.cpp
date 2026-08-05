#include "clip-model.h"

#undef NDEBUG
#include <cassert>

static clip_hparams lightonocr_hparams() {
    clip_hparams hparams;
    hparams.patch_size = 14;
    hparams.n_merge = 2;
    hparams.image_longest_edge = 1540;
    return hparams;
}

int main() {
    {
        auto hparams = lightonocr_hparams();
        hparams.set_limit_image_tokens_longest_edge(256);
        assert(hparams.image_longest_edge == 1540);
        assert(hparams.warmup_image_size == 448);
    }

    {
        auto hparams = lightonocr_hparams();
        hparams.custom_image_max_tokens = 2025;
        hparams.set_limit_image_tokens_longest_edge(256);
        assert(hparams.image_longest_edge == 1260);
        assert(hparams.warmup_image_size == 448);
    }

    {
        auto hparams = lightonocr_hparams();
        hparams.custom_image_max_tokens = 2000;
        hparams.set_limit_image_tokens_longest_edge(256);
        assert(hparams.image_longest_edge == 1232);
        assert(hparams.warmup_image_size == 448);
    }

    {
        auto hparams = lightonocr_hparams();
        hparams.custom_image_max_tokens = 200;
        hparams.set_limit_image_tokens_longest_edge(256);
        assert(hparams.image_longest_edge == 392);
        assert(hparams.warmup_image_size == 392);
    }

    {
        auto hparams = lightonocr_hparams();
        hparams.custom_image_max_tokens = 4096;
        hparams.set_limit_image_tokens_longest_edge(256);
        assert(hparams.image_longest_edge == 1792);
        assert(hparams.warmup_image_size == 448);
    }

    return 0;
}
