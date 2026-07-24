#include "ggml-backend-impl.h"

#if defined(__e2k__)

struct e2k_features {
    int iset;
    bool has_fma;

    e2k_features() {
#if defined(__iset__)
        iset = __iset__;
#else
        iset = 0;
#endif
#if defined(__FMA__) || defined(__AVX2__)
        has_fma = true;
#else
        has_fma = false;
#endif
    }
};

static int ggml_backend_cpu_e2k_score() {
    int score = 1;
    e2k_features feat;

    if (feat.iset >= 7) score += 100;
    else if (feat.iset >= 6) score += 50;
    else if (feat.iset >= 5) score += 25;

    if (feat.has_fma) score += 10;

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_e2k_score)

#endif // __e2k__
