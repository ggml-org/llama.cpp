// CPU-only test for the GGML_SYCL_FA_LARGE_GRF parse and launch decision
// (ggml/src/ggml-sycl/fattn-grf.hpp). The device gate and the kernel property are covered by
// the ctest wrapper check-sycl-fa-large-grf.cmake on a GPU.

#include "ggml-sycl/fattn-grf.hpp"

#include <cstdio>
#include <initializer_list>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

static ggml_sycl_fa_grf_parse_result parse(const char * raw, bool variants, ggml_sycl_fa_grf_mode * mode) {
    *mode = GGML_SYCL_FA_GRF_TILE;  // must be overwritten on every path
    return ggml_sycl_fa_large_grf_parse(raw, variants, mode);
}

int main() {
    ggml_sycl_fa_grf_mode m = GGML_SYCL_FA_GRF_OFF;

    // Absent or empty: unset, OFF.
    CHECK(parse(nullptr, true, &m) == GGML_SYCL_FA_GRF_PARSE_UNSET && m == GGML_SYCL_FA_GRF_OFF);
    CHECK(parse("", true, &m) == GGML_SYCL_FA_GRF_PARSE_UNSET && m == GGML_SYCL_FA_GRF_OFF);
    // Exact values.
    CHECK(parse("0", true, &m) == GGML_SYCL_FA_GRF_PARSE_OK && m == GGML_SYCL_FA_GRF_OFF);
    CHECK(parse("1", true, &m) == GGML_SYCL_FA_GRF_PARSE_OK && m == GGML_SYCL_FA_GRF_TILE);
    // Whole-value rule: trailing text, words, fractions, the removed mode 2, negatives.
    for (const char * bad : { "1junk", "on", "true", "1.5", "2", "-1", "abc", "01x" }) {
        CHECK(parse(bad, true, &m) == GGML_SYCL_FA_GRF_PARSE_INVALID && m == GGML_SYCL_FA_GRF_OFF);
    }
    // A valid request on a build without the variants is reported, and OFF.
    CHECK(parse("1", false, &m) == GGML_SYCL_FA_GRF_PARSE_NO_VARIANTS && m == GGML_SYCL_FA_GRF_OFF);
    CHECK(parse("0", false, &m) == GGML_SYCL_FA_GRF_PARSE_OK && m == GGML_SYCL_FA_GRF_OFF);

    // Launch decision: TILE mode, tile route, more than one query row.
    CHECK(ggml_sycl_fa_large_grf_wanted(GGML_SYCL_FA_GRF_TILE, true, 512));
    CHECK(ggml_sycl_fa_large_grf_wanted(GGML_SYCL_FA_GRF_TILE, true, 2));
    CHECK(!ggml_sycl_fa_large_grf_wanted(GGML_SYCL_FA_GRF_TILE, true, 1));   // single-row decode on TILE
    CHECK(!ggml_sycl_fa_large_grf_wanted(GGML_SYCL_FA_GRF_TILE, false, 512)); // vec route never
    CHECK(!ggml_sycl_fa_large_grf_wanted(GGML_SYCL_FA_GRF_OFF, true, 512));

    if (g_failures == 0) {
        printf("test-sycl-fa-large-grf: OK\n");
        return 0;
    }
    printf("test-sycl-fa-large-grf: %d failure(s)\n", g_failures);
    return 1;
}
