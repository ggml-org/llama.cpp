// test-tessera-quants-v2
//
// Parity test for the 5 TILE640 quant v2 functions
// (dequantize_row_tessera_t640_v2, quantize_row_tessera_t640_v2,
// apply_outlier_addback_v2, decode_per_row_meta_v2,
// apply_act_scale_v2) in ggml/src/ggml-quants-v2.c.
//
// The C references in ggml/src/ggml-quants.c are the
// documented behaviour; the v2 paths are bit-identical
// Accelerate + NEON re-implementations. The parity bar is
// 0 mismatches on the standard fixtures; 1-2 ulp differences
// are accepted on the vDSP-reduction paths (the v2 quantize
// uses vDSP_sve which uses parallel summation).
//
// The test exercises:
//   1. dequantize: bit-identical across the 5 canonical
//      shapes (1024, 1280, 4096; 640 falls back to C ref
//      since k < GGML_TESSERA_T640_V2_MIN_K).
//   2. quantize: round-trip quant->dequant matches the C
//      ref's quant->dequant to within 1e-5 relative err.
//   3. outlier addback: 5% sparse outliers, per-element
//      parity vs the C scalar loop.
//   4. per-row meta decode: page_scales + lane_scales
//      converted in bulk match the scalar decode.
//   5. act_scale: y *= act_scale, bulk vs scalar.
//
// All tests skip when ggml_tessera_t640_v2_enabled() == 0
// (the env var GGML_TESSERA_T640_V2_DISABLE=1 turns v2 off).

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-quants-v2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kSeed = 0xBEEFu;

size_t tessera_t640_row_bytes(int64_t k) {
    const int pages = (int) ((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    return (size_t) pages * TILE640_WORDS_PER_PAGE * sizeof(uint32_t)
         + (size_t) pages * sizeof(uint16_t)
         + (size_t) pages * TILE640_LANES_PER_PAGE * sizeof(int8_t);
}

void make_signal(std::vector<float> & x, int64_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    x.assign((size_t) k, 0.0f);
    for (int64_t i = 0; i < k; i++) {
        const int lane_idx = (int) (i / TILE640_LANE_SIZE);
        const int pos      = (int) (i % TILE640_LANE_SIZE);
        const float amp    = (lane_idx < 16) ? 2.5f : 4.0f;
        if (pos % 3 == 0) {
            x[(size_t) i] = 0.0f;
        } else {
            x[(size_t) i] = (pos % 2 == 0) ? amp : -amp;
        }
    }
    // Add a small uniform noise so the trits are not all
    // deterministic.
    for (int64_t i = 0; i < k; i++) {
        x[(size_t) i] += 0.01f * dist(rng);
    }
}

int test_dequant(int64_t k, uint32_t seed) {
    std::vector<float> x;
    make_signal(x, k, seed);
    std::vector<uint8_t> packed(tessera_t640_row_bytes(k));
    quantize_row_tessera_t640_ref(x.data(), packed.data(), k);
    std::vector<float> y_ref((size_t) k);
    std::vector<float> y_v2((size_t) k);
    dequantize_row_tessera_t640(packed.data(), y_ref.data(), k);
    // v2: pre-decode the per-row meta, then dequant with
    // the pre-decoded arrays (the v2 API takes them as
    // separate inputs; the dispatch calls decode_per_row_meta_v2
    // once for the whole tile).
    const int pages = (int) ((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    const uint8_t * packed_bytes = packed.data();
    const uint32_t * packed_words = (const uint32_t *) packed_bytes;
    const uint16_t * page_scales = (const uint16_t *) (packed_words + pages * TILE640_WORDS_PER_PAGE);
    const int8_t   * lane_scales = (const int8_t   *) (page_scales + pages);
    std::vector<float> page_max((size_t) pages);
    std::vector<float> lane_scale((size_t) (pages * TILE640_LANES_PER_PAGE));
    decode_per_row_meta_v2(page_scales, lane_scales, 1, (int64_t) pages,
                           page_max.data(), lane_scale.data());
    dequantize_row_tessera_t640_v2(packed_words, page_max.data(), lane_scale.data(),
                                   k, y_v2.data());
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        const float d = std::fabs(y_ref[i] - y_v2[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  dequant k=%lld seed=%u max_diff=%g mismatches=%d/%lld\n",
           (long long) k, seed, max_diff, mismatches, (long long) k);
    return mismatches == 0 ? 0 : 1;
}

int test_quant_roundtrip(int64_t k, uint32_t seed) {
    std::vector<float> x;
    make_signal(x, k, seed);
    std::vector<uint8_t> packed_c(tessera_t640_row_bytes(k));
    std::vector<uint8_t> packed_v(tessera_t640_row_bytes(k));
    quantize_row_tessera_t640_ref(x.data(), packed_c.data(), k);
    quantize_row_tessera_t640_v2(x.data(), packed_v.data(), k);
    std::vector<float> y_c((size_t) k);
    std::vector<float> y_v((size_t) k);
    dequantize_row_tessera_t640(packed_c.data(), y_c.data(), k);
    dequantize_row_tessera_t640(packed_v.data(), y_v.data(), k);
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        const float d = std::fabs(y_c[i] - y_v[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    // Round-trip: the dequant of v2's quant should match the
    // input within 1 ulp of the dequant of C's quant. The vDSP
    // reductions can differ from sequential by 1-2 ulp in the
    // threshold, which can flip a trit for elements very close
    // to the threshold; for the uniform random signal we use,
    // the threshold is well-defined and the trits are stable.
    printf("  quant round-trip k=%lld seed=%u max_diff=%g mismatches=%d/%lld\n",
           (long long) k, seed, max_diff, mismatches, (long long) k);
    if (max_diff > 1e-5f) return 1;
    return 0;
}

int test_outlier_addback(int64_t k) {
    // 5% sparse outliers in row 0.
    std::mt19937 rng(kSeed);
    std::vector<float> row((size_t) k, 0.0f);
    const int64_t n_outliers = std::max((int64_t) 1, k / 20);
    std::vector<int32_t> cols;
    std::vector<uint16_t> vals;
    cols.reserve((size_t) n_outliers);
    vals.reserve((size_t) n_outliers);
    std::uniform_int_distribution<int64_t> col_dist(0, k - 1);
    for (int64_t i = 0; i < n_outliers; i++) {
        const int64_t c = col_dist(rng);
        const float v = (rng() & 1) ? 1.0f : -1.0f;
        cols.push_back((int32_t) c);
        vals.push_back((uint16_t) GGML_FP32_TO_FP16(5.0f * v));
    }
    // Batched v2: pass (rows, row_len, n_rows=1, offsets=[0, n_outliers], cols, vals).
    std::vector<float> row_v2 = row;
    std::vector<int32_t> row_offsets = { 0, (int32_t) n_outliers };
    apply_outlier_addback_v2(row_v2.data(), k, 1,
                             row_offsets.data(), cols.data(), vals.data());
    // Reference: scalar per-element.
    std::vector<float> row_ref = row;
    for (int64_t i = 0; i < n_outliers; i++) {
        const int32_t col = cols[(size_t) i];
        if (col >= 0 && col < k) {
            row_ref[(size_t) col] = GGML_FP16_TO_FP32(vals[(size_t) i]);
        }
    }
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        const float d = std::fabs(row_ref[i] - row_v2[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  outlier addback k=%lld n_outliers=%lld max_diff=%g mismatches=%d/%lld\n",
           (long long) k, (long long) n_outliers, max_diff, mismatches, (long long) k);
    return mismatches == 0 ? 0 : 1;
}

int test_meta_decode(int64_t n_rows, int64_t n_pages) {
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
    std::vector<float> page_max_v2((size_t) (n_rows * n_pages));
    std::vector<float> lane_scale_v2((size_t) (n_rows * n_lanes_per_row));
    decode_per_row_meta_v2(page_scales.data(), lane_scales.data(),
                           n_rows, n_pages,
                           page_max_v2.data(), lane_scale_v2.data());
    // Reference: scalar.
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < n_rows * n_pages; i++) {
        const float ref = GGML_FP16_TO_FP32(page_scales[(size_t) i]);
        const float d = std::fabs(ref - page_max_v2[(size_t) i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    for (int64_t i = 0; i < n_rows * n_lanes_per_row; i++) {
        const float ref = (float) lane_scales[(size_t) i] * (1.0f / 127.0f);
        const float d = std::fabs(ref - lane_scale_v2[(size_t) i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  meta decode n_rows=%lld n_pages=%lld max_diff=%g mismatches=%d/%lld\n",
           (long long) n_rows, (long long) n_pages, max_diff,
           mismatches, (long long) (n_rows * (n_pages + n_lanes_per_row)));
    return mismatches == 0 ? 0 : 1;
}

int test_act_scale(int64_t n) {
    std::vector<float> y((size_t) n);
    std::vector<uint16_t> as((size_t) n);
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> as_dist(0.5f, 2.0f);
    for (int64_t i = 0; i < n; i++) {
        y[(size_t) i] = y_dist(rng);
        as[(size_t) i] = (uint16_t) GGML_FP32_TO_FP16(as_dist(rng));
    }
    std::vector<float> y_ref = y;
    for (int64_t i = 0; i < n; i++) {
        y_ref[(size_t) i] *= GGML_FP16_TO_FP32(as[(size_t) i]);
    }
    apply_act_scale_v2(y.data(), as.data(), n);
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        const float d = std::fabs(y_ref[i] - y[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0.0f) mismatches++;
    }
    printf("  act_scale n=%lld max_diff=%g mismatches=%d/%lld\n",
           (long long) n, max_diff, mismatches, (long long) n);
    return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(void) {
    if (!ggml_tessera_t640_v2_enabled()) {
        printf("v2 disabled (GGML_TESSERA_T640_V2_DISABLE=1); skipping\n");
        return 0;
    }
    int rc = 0;
    printf("dequant parity:\n");
    rc |= test_dequant(1024, kSeed);
    rc |= test_dequant(1280, kSeed + 1);
    rc |= test_dequant(4096, kSeed + 2);
    rc |= test_dequant(640,  kSeed + 3);  // below cutoff -> C ref
    printf("quant round-trip:\n");
    rc |= test_quant_roundtrip(1024, kSeed + 4);
    rc |= test_quant_roundtrip(1280, kSeed + 5);
    rc |= test_quant_roundtrip(4096, kSeed + 6);
    printf("outlier addback:\n");
    rc |= test_outlier_addback(1024);
    rc |= test_outlier_addback(4096);
    printf("meta decode:\n");
    rc |= test_meta_decode(1, 1);
    rc |= test_meta_decode(1, 6);    // 4096 / 640
    rc |= test_meta_decode(16, 16);  // batched: 16 rows of 16 pages
    printf("act_scale:\n");
    rc |= test_act_scale(256);
    rc |= test_act_scale(1024);
    rc |= test_act_scale(4096);
    if (rc == 0) printf("OK\n");
    return rc;
}
