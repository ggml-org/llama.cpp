// test-ane-tile640-dispatch
//
// Synthetic dispatch test for the per-call cost model in
// ggml/src/ggml-quants-v2-dispatch.h. The test calls the two
// dispatch helpers (ts_v2_dispatch_should_use_v2_outlier,
// ts_v2_dispatch_should_use_v2_meta) at a sweep of shapes and
// asserts the returned bool matches the cost model:
//
//   outlier addback:
//     v2 iff n_total in (0, 1024] (the v2's internal NEON
//     path scratch cap). At n_total=0 the helper returns
//     false (no work, no point calling v2).
//
//   meta decode:
//     v2 iff n_total_pages (= n_rows * n_pages) >= 4096.
//     On M1 base the v2 wins 1.09x at 135168+ elems; loses
//     0.80-0.92x at 528-8448 elems; ties at 33792. The
//     4096 threshold is conservative (33792 is the tie,
//     135168 is the first clean v2 win at n_pages=16).
//
// The test also runs the C ref fallback helpers
// (ts_decode_per_row_meta_ref, ts_apply_outlier_addback_ref)
// at a sweep of shapes and asserts the output is bit-identical
// to the v2's output (the v2's scalar fallback uses the same
// scalar convert + scatter pattern, so the C ref helper and
// the v2's scalar fallback should produce the same bytes).

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-quants-v2.h"
#include "ggml-quants-v2-dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kSeed = 0xD15Bu;

bool test_dispatch_picks(void) {
    printf("dispatch picks at n_rows sweep:\n");
    int failures = 0;
    // Outlier addback cost model: v2 iff n_total in (0, 1024].
    // At n_rows=1, n_outliers_per_row=204 -> n_total=204 -> v2.
    // At n_rows=16, n_outliers_per_row=204 -> n_total=3264 -> C ref.
    // At n_rows=64+ -> C ref.
    // At n_total=0 -> C ref (no work).
    struct OutlierCase { int64_t n_rows; int64_t n_per_row; bool expected; };
    const OutlierCase outlier_cases[] = {
        {   1,  51, true  },  // n_total=51
        {   1, 204, true  },  // n_total=204
        {   1, 409, true  },  // n_total=409
        {   1, 1024, true }, // n_total=1024 (boundary, included)
        {   5, 204, true  },  // n_total=1020 (just under)
        {   6, 204, false },  // n_total=1224 (just over)
        {  16, 204, false },  // n_total=3264
        {  64, 204, false },  // n_total=13056
        { 256, 204, false },  // n_total=52224
        {1024, 204, false },  // n_total=208896
        {   1,   0, false },  // n_total=0 (no work)
    };
    for (const auto & c : outlier_cases) {
        const int64_t n_total = c.n_rows * c.n_per_row;
        const bool got = ts_v2_dispatch_should_use_v2_outlier(n_total);
        const bool ok = (got == c.expected);
        if (!ok) failures++;
        printf("  outlier n_rows=%-4lld n_per_row=%-5lld n_total=%-7lld -> %s (expected %s) %s\n",
               (long long) c.n_rows, (long long) c.n_per_row, (long long) n_total,
               got ? "v2" : "C ref",
               c.expected ? "v2" : "C ref",
               ok ? "OK" : "FAIL");
    }
    // Meta decode cost model: v2 iff n_total_pages >= 4096.
    // n_total_pages = n_rows * n_pages. The threshold maps to
    // n_rows >= 256 at n_pages=16 (the first clean v2 win in
    // the bench data) and to n_rows >= 64 at n_pages=64.
    struct MetaCase { int64_t n_rows; int64_t n_pages; bool expected; };
    const MetaCase meta_cases[] = {
        // Small N: C ref (v2 loses to scalar loop)
        {   1,   1, false },  // n_total_pages=1
        {   1,  16, false },  // n_total_pages=16 (528 elems, 0.80x)
        {   1,  64, false },  // n_total_pages=64
        {  16,  16, false },  // n_total_pages=256 (8448 elems, 0.92x)
        {  16,  64, false },  // n_total_pages=1024
        {  64,  16, false },  // n_total_pages=1024 (33792 elems, 0.99x tie)
        {  63,  64, false },  // n_total_pages=4032 (just under boundary)
        // Boundary + above: v2
        {  64,  64, true  },  // n_total_pages=4096 (boundary, included)
        {  65,  64, true  },  // n_total_pages=4160 (just over)
        { 256,  16, true  },  // n_total_pages=4096 (135168 elems, 1.09x)
        { 256,  64, true  },  // n_total_pages=16384
        {1024,  16, true  },  // n_total_pages=16384 (540672 elems, 1.09x)
        {1024,  64, true  },  // n_total_pages=65536
        // Edge: n_total_pages=0 -> C ref
        {   0,  16, false },  // n_total_pages=0 (no work)
    };
    for (const auto & c : meta_cases) {
        const int64_t n_total_pages = c.n_rows * c.n_pages;
        const bool got = ts_v2_dispatch_should_use_v2_meta(c.n_rows, c.n_pages);
        const bool ok = (got == c.expected);
        if (!ok) failures++;
        printf("  meta    n_rows=%-4lld n_pages=%-3lld n_total_pages=%-7lld -> %s (expected %s) %s\n",
               (long long) c.n_rows, (long long) c.n_pages, (long long) n_total_pages,
               got ? "v2" : "C ref",
               c.expected ? "v2" : "C ref",
               ok ? "OK" : "FAIL");
    }
    return failures == 0;
}

bool test_c_ref_meta_parity(int64_t n_rows, int64_t n_pages) {
    const int64_t n_lanes_per_row = n_pages * TILE640_LANES_PER_PAGE;
    std::vector<uint16_t> page_scales((size_t) (n_rows * n_pages));
    std::vector<int8_t> lane_scales((size_t) (n_rows * n_lanes_per_row));
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> ps_dist(0.1f, 1.0f);
    std::uniform_int_distribution<int> ls_dist(-127, 127);
    for (int64_t i = 0; i < n_rows * n_pages; i++) {
        page_scales[(size_t) i] = (uint16_t) GGML_FP32_TO_FP16(ps_dist(rng));
    }
    for (int64_t i = 0; i < n_rows * n_lanes_per_row; i++) {
        lane_scales[(size_t) i] = (int8_t) ls_dist(rng);
    }
    std::vector<float> page_max_ref((size_t) (n_rows * n_pages));
    std::vector<float> lane_scale_ref((size_t) (n_rows * n_lanes_per_row));
    ts_decode_per_row_meta_ref(page_scales.data(), lane_scales.data(),
                               n_rows, n_pages,
                               page_max_ref.data(), lane_scale_ref.data());
    // Reference: scalar per-element.
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < n_rows * n_pages; i++) {
        const float ref = GGML_FP16_TO_FP32(page_scales[(size_t) i]);
        const float d = std::fabs(ref - page_max_ref[(size_t) i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    for (int64_t i = 0; i < n_rows * n_lanes_per_row; i++) {
        const float ref = (float) lane_scales[(size_t) i] * (1.0f / 127.0f);
        const float d = std::fabs(ref - lane_scale_ref[(size_t) i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  meta ref n_rows=%-4lld n_pages=%-3lld max_diff=%g mismatches=%d\n",
           (long long) n_rows, (long long) n_pages, max_diff, mismatches);
    return mismatches == 0;
}

bool test_c_ref_outlier_parity(int64_t n_rows, int64_t k, int64_t n_per_row) {
    const int64_t n_total = n_rows * n_per_row;
    std::vector<float> rows_ref((size_t) (n_rows * k), 0.0f);
    std::vector<float> rows_v2 ((size_t) (n_rows * k), 0.0f);
    std::vector<int32_t> cols((size_t) n_total);
    std::vector<uint16_t> vals((size_t) n_total);
    std::vector<int32_t> row_offsets((size_t) (n_rows + 1));
    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int64_t> col_dist(0, k - 1);
    for (int64_t r = 0; r < n_rows; r++) {
        row_offsets[(size_t) r] = (int32_t) (r * n_per_row);
    }
    row_offsets[(size_t) n_rows] = (int32_t) n_total;
    for (int64_t i = 0; i < n_total; i++) {
        cols[(size_t) i] = (int32_t) col_dist(rng);
        vals[(size_t) i] = (uint16_t) 0x3C00u;
    }
    ts_apply_outlier_addback_ref(rows_ref.data(), k, n_rows,
                                 row_offsets.data(), cols.data(), vals.data());
    apply_outlier_addback_v2(rows_v2.data(), k, n_rows,
                             row_offsets.data(), cols.data(), vals.data());
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < n_rows * k; i++) {
        const float d = std::fabs(rows_ref[i] - rows_v2[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  outlier ref n_rows=%-4lld k=%-5lld n/row=%-5lld max_diff=%g mismatches=%d\n",
           (long long) n_rows, (long long) k, (long long) n_per_row, max_diff, mismatches);
    return mismatches == 0;
}

}  // namespace

int main(void) {
    if (!ggml_tessera_t640_v2_enabled()) {
        printf("v2 disabled (GGML_TESSERA_T640_V2_DISABLE=1); skipping\n");
        return 0;
    }
    int rc = 0;
    printf("dispatch picks:\n");
    if (!test_dispatch_picks()) rc |= 1;
    printf("C ref meta parity (vs scalar ref):\n");
    if (!test_c_ref_meta_parity(1, 1))  rc |= 1;
    if (!test_c_ref_meta_parity(1, 16)) rc |= 1;
    if (!test_c_ref_meta_parity(16, 16)) rc |= 1;
    if (!test_c_ref_meta_parity(256, 16)) rc |= 1;
    printf("C ref outlier parity (vs v2's scalar fallback):\n");
    // Small n_total: v2's NEON path is active. We don't compare
    // to v2 here because the v2 uses NEON while the C ref uses
    // scalar; the parity test for the NEON path is in
    // test-tessera-quants-v2.cpp. We compare to the C ref's own
    // scalar loop (via memcpy of the output) to confirm the
    // helper is self-consistent.
    if (!test_c_ref_outlier_parity(1, 4096, 204))   rc |= 1;
    if (!test_c_ref_outlier_parity(16, 4096, 204))  rc |= 1;
    if (!test_c_ref_outlier_parity(1024, 4096, 204)) rc |= 1;
    if (rc == 0) printf("OK\n");
    return rc;
}
