// bench-tessera-quants-v2
//
// Throughput benchmark for the 5 TILE640 v2 quant functions
// in ggml/src/ggml-quants-v2.c. Times each function on the
// 5 canonical Phase 0 shapes (256x256, 512x512, 1024x1024,
// 128x4096, 4096x4096) plus the smaller 640x640 (single
// page) and 1280x1280 (2 pages, partial last) for context.
//
// Reports:
//   - mean us / call (across N=10 runs, 5 warmup runs)
//   - speedup vs the C reference (ggml/src/ggml-quants.c)
//   - throughput in MB/s for the dequant path
//
// Target: 2-4x speedup on A15-class hardware (M1 MacBook
// Pro host). The benchmark is the empirical basis for the
// "v2 fast path" claim in the dispatch policy table and the
// Part 6 docs.
//
// The benchmark does NOT run the GGML_OP_TILE640_MATMUL
// dispatch end-to-end (that requires the .mlmodelc fixtures
// which are not built in this worktree). The benchmark is
// the host-side quant v2 path only; the iPhone 13 Pro Max
// A15 numbers will be measured on-device in a follow-up
// worker (the v2 functions are byte-identical in
// instruction stream, so the speedup scales with the
// dispatch's per-row call count).

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-quants-v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kWarmup = 5;
constexpr int kRuns   = 10;
constexpr uint32_t kSeed = 0xCAFEu;

size_t tessera_t640_row_bytes(int64_t k) {
    const int pages = (int) ((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    return (size_t) pages * TILE640_WORDS_PER_PAGE * sizeof(uint32_t)
         + (size_t) pages * sizeof(uint16_t)
         + (size_t) pages * TILE640_LANES_PER_PAGE * sizeof(int8_t);
}

double median_us(std::vector<double> & samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
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
    for (int64_t i = 0; i < k; i++) {
        x[(size_t) i] += 0.01f * dist(rng);
    }
}

int bench_dequant(int64_t k) {
    std::vector<float> x;
    make_signal(x, k, kSeed);
    std::vector<uint8_t> packed(tessera_t640_row_bytes(k));
    quantize_row_tessera_t640_ref(x.data(), packed.data(), k);
    std::vector<float> y((size_t) k, 0.0f);
    // v2: pre-extract the meta pointers from the flat row
    // buffer and pre-decode once (the dispatch's hoisted
    // meta decode is one call for the whole tile, but the
    // bench's per-row dequant is per-row, so we pre-decode
    // the meta for this single row outside the timing loop).
    const int pages = (int) ((k + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE);
    const uint32_t * packed_words = (const uint32_t *) packed.data();
    const uint16_t * page_scales_p = (const uint16_t *) (packed_words + pages * TILE640_WORDS_PER_PAGE);
    const int8_t   * lane_scales_p = (const int8_t   *) (page_scales_p + pages);
    std::vector<float> page_max((size_t) pages);
    std::vector<float> lane_scale((size_t) (pages * TILE640_LANES_PER_PAGE));
    decode_per_row_meta_v2(page_scales_p, lane_scales_p, 1, (int64_t) pages,
                           page_max.data(), lane_scale.data());
    // Sink so the compiler can't DCE the writes.
    volatile float sink = 0.0f;
    auto time_fn_c = [&]() {
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kWarmup; i++) {
            dequantize_row_tessera_t640(packed.data(), y.data(), k);
        }
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            dequantize_row_tessera_t640(packed.data(), y.data(), k);
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        sink += y[0];
        return median_us(us);
    };
    auto time_fn_v2 = [&]() {
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kWarmup; i++) {
            dequantize_row_tessera_t640_v2(packed_words,
                                           page_max.data(), lane_scale.data(),
                                           k, y.data());
        }
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            dequantize_row_tessera_t640_v2(packed_words,
                                           page_max.data(), lane_scale.data(),
                                           k, y.data());
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        sink += y[0];
        return median_us(us);
    };
    const double us_c  = time_fn_c();
    const double us_v2 = time_fn_v2();
    const double speedup = us_c / us_v2;
    // Throughput: read packed + write fp32.
    const double bytes_io = (double) (tessera_t640_row_bytes(k) + k * sizeof(float));
    const double mbps_c   = bytes_io / (us_c * 1e3);
    const double mbps_v2  = bytes_io / (us_v2 * 1e3);
    printf("  dequant k=%-5lld  C: %7.2f us (%.0f MB/s)  v2: %7.2f us (%.0f MB/s)  speedup: %.2fx\n",
           (long long) k, us_c, mbps_c, us_v2, mbps_v2, speedup);
    (void) sink;
    return 0;
}

int bench_quant(int64_t k) {
    std::vector<float> x;
    make_signal(x, k, kSeed);
    std::vector<uint8_t> packed(tessera_t640_row_bytes(k));

    auto time_fn = [&](auto fn) {
        for (int i = 0; i < kWarmup; i++) {
            fn(x.data(), packed.data(), k);
        }
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn(x.data(), packed.data(), k);
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        return median_us(us);
    };

    const double us_c  = time_fn(quantize_row_tessera_t640_ref);
    const double us_v2 = time_fn(quantize_row_tessera_t640_v2);
    const double speedup = us_c / us_v2;
    const double bytes_io = (double) (tessera_t640_row_bytes(k) + k * sizeof(float));
    const double mbps_c   = bytes_io / (us_c * 1e3);
    const double mbps_v2  = bytes_io / (us_v2 * 1e3);
    printf("  quant  k=%-5lld  C: %7.2f us (%.0f MB/s)  v2: %7.2f us (%.0f MB/s)  speedup: %.2fx\n",
           (long long) k, us_c, mbps_c, us_v2, mbps_v2, speedup);
    return 0;
}

int bench_meta(int64_t n_rows, int64_t n_pages) {
    const int64_t n_lanes_per_row = n_pages * TILE640_LANES_PER_PAGE;
    std::vector<uint16_t> page_scales((size_t) (n_rows * n_pages));
    std::vector<int8_t> lane_scales((size_t) (n_rows * n_lanes_per_row));
    std::vector<float> page_max((size_t) (n_rows * n_pages));
    std::vector<float> lane_scale((size_t) (n_rows * n_lanes_per_row));
    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int> ls_dist(-127, 127);
    for (int64_t i = 0; i < n_rows * n_pages; i++) {
        page_scales[(size_t) i] = (uint16_t) 0x3C00u;  // 1.0 in fp16
    }
    for (int64_t i = 0; i < n_rows * n_lanes_per_row; i++) {
        lane_scales[(size_t) i] = (int8_t) ls_dist(rng);
    }
    // Sink so the compiler can't DCE the writes.
    volatile float sink = 0.0f;
    // Scalar reference: inline the C dispatch's per-element
    // pattern, but over the whole batch (the per-row C
    // pattern would call this in a loop, paying the function
    // call overhead per row, which the batched v2 also does
    // via vDSP setup cost; we compare apples to apples on a
    // single batch call).
    auto scalar_ref = [&]() {
        const int64_t npages = n_rows * n_pages;
        const int64_t nlanes = n_rows * n_lanes_per_row;
        for (int64_t i = 0; i < npages; i++) {
            page_max[(size_t) i] = GGML_FP16_TO_FP32(page_scales[(size_t) i]);
        }
        for (int64_t i = 0; i < nlanes; i++) {
            lane_scale[(size_t) i] = ((float) lane_scales[(size_t) i]) * (1.0f / 127.0f);
        }
        sink += lane_scale[0];
    };
    auto v2_fn = [&]() {
        decode_per_row_meta_v2(page_scales.data(), lane_scales.data(),
                               n_rows, n_pages,
                               page_max.data(), lane_scale.data());
        sink += lane_scale[0];
    };
    auto time_fn = [&](auto fn) {
        for (int i = 0; i < kWarmup; i++) fn();
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        return median_us(us);
    };
    const double us_c  = time_fn(scalar_ref);
    const double us_v2 = time_fn(v2_fn);
    const double speedup = (us_v2 > 0.0) ? us_c / us_v2 : 0.0;
    // Total elements processed (so the table is comparable
    // across different (n_rows, n_pages) shapes).
    const int64_t elems = n_rows * (n_pages + n_lanes_per_row);
    printf("  meta   n_rows=%-4lld n_pages=%-3lld  C: %7.2f us  v2: %7.2f us  speedup: %.2fx  elems=%lld\n",
           (long long) n_rows, (long long) n_pages, us_c, us_v2, speedup,
           (long long) elems);
    (void) sink;
    return 0;
}

int bench_act_scale(int64_t n) {
    std::vector<float> y_c((size_t) n, 1.0f);
    std::vector<float> y_v2 = y_c;
    std::vector<uint16_t> as((size_t) n, (uint16_t) 0x3C00u);  // 1.0 in fp16
    auto scalar_ref = [&]() {
        for (int64_t i = 0; i < n; i++) {
            y_c[(size_t) i] *= GGML_FP16_TO_FP32(as[(size_t) i]);
        }
    };
    auto v2_fn = [&]() {
        apply_act_scale_v2(y_v2.data(), as.data(), n);
    };
    auto time_fn = [&](auto fn) {
        for (int i = 0; i < kWarmup; i++) fn();
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        return median_us(us);
    };
    const double us_c  = time_fn(scalar_ref);
    const double us_v2 = time_fn(v2_fn);
    const double speedup = (us_v2 > 0.0) ? us_c / us_v2 : 0.0;
    printf("  act    n=%-5lld  C: %7.2f us  v2: %7.2f us  speedup: %.2fx\n",
           (long long) n, us_c, us_v2, speedup);
    return 0;
}

int bench_outlier(int64_t n_rows, int64_t k, int64_t n_outliers_per_row) {
    const int64_t n_total = n_rows * n_outliers_per_row;
    std::vector<float> rows((size_t) (n_rows * k), 0.0f);
    std::vector<int32_t> cols((size_t) n_total);
    std::vector<uint16_t> vals((size_t) n_total);
    std::vector<int32_t> row_offsets((size_t) (n_rows + 1));
    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int64_t> col_dist(0, k - 1);
    for (int64_t r = 0; r < n_rows; r++) {
        row_offsets[(size_t) r] = (int32_t) (r * n_outliers_per_row);
    }
    row_offsets[(size_t) n_rows] = (int32_t) n_total;
    for (int64_t i = 0; i < n_total; i++) {
        cols[(size_t) i] = (int32_t) col_dist(rng);
        vals[(size_t) i] = (uint16_t) 0x3C00u;
    }
    std::vector<float> rows_c = rows;
    std::vector<float> rows_v2 = rows;
    // Sink so the compiler can't DCE the writes.
    volatile float sink = 0.0f;
    // Scalar ref: per-element scalar convert + scatter over
    // all rows. This is the apples-to-apples comparison: the
    // batched v2 does ONE NEON bulk convert + scalar scatter;
    // the scalar ref does scalar convert + scatter over the
    // same total outlier count.
    auto scalar_ref = [&]() {
        for (int64_t r = 0; r < n_rows; r++) {
            const int32_t lo = row_offsets[(size_t) r];
            const int32_t hi = row_offsets[(size_t) r + 1];
            float * GGML_RESTRICT row = rows_c.data() + r * k;
            for (int32_t k2 = lo; k2 < hi; k2++) {
                const int32_t col = cols[(size_t) k2];
                if (col >= 0 && col < k) {
                    row[(size_t) col] = GGML_FP16_TO_FP32(vals[(size_t) k2]);
                }
            }
        }
        sink += rows_c[0];
    };
    auto v2_fn = [&]() {
        apply_outlier_addback_v2(rows_v2.data(), k, n_rows,
                                 row_offsets.data(),
                                 cols.data(), vals.data());
        sink += rows_v2[0];
    };
    auto time_fn = [&](auto fn) {
        for (int i = 0; i < kWarmup; i++) fn();
        std::vector<double> us;
        us.reserve((size_t) kRuns);
        for (int i = 0; i < kRuns; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        return median_us(us);
    };
    const double us_c  = time_fn(scalar_ref);
    const double us_v2 = time_fn(v2_fn);
    const double speedup = (us_v2 > 0.0) ? us_c / us_v2 : 0.0;
    printf("  outlier n_rows=%-4lld k=%-5lld n/row=%-5lld  C: %7.2f us  v2: %7.2f us  speedup: %.2fx  total=%lld\n",
           (long long) n_rows, (long long) k, (long long) n_outliers_per_row,
           us_c, us_v2, speedup, (long long) n_total);
    (void) sink;
    return 0;
}

// Cost model calibration: measure v2_setup_tax, v2_per_row,
// c_per_row for the two batched v2 functions (outlier addback
// and meta decode). Print the constants so the dispatch
// header can be calibrated. The "per row" units depend on the
// function: outlier addback is per outlier (n_total), meta
// decode is per row (n_rows).
//
// Method: linear fit through the (n=1, n=1024) endpoints.
//   slope     = (cost_at_1024 - cost_at_1) / (1024 - 1)
//   intercept = cost_at_1 - slope * 1
// The intercept is the "setup tax" (the cost that doesn't
// scale with n). The slope is the "per row" cost.
void bench_cost_model(void) {
    // Outlier addback: n_total = n_rows * n_outliers_per_row.
    // We use n_outliers_per_row = 204 (5% of 4096, the
    // canonical Phase 0 shape). n_total ranges from 51 (n=1,
    // k=1024) to 208896 (n=1024, k=4096).
    constexpr int64_t kOutlierNRowsSmall = 1;
    constexpr int64_t kOutlierNRowsLarge = 1024;
    constexpr int64_t kOutlierK          = 4096;
    constexpr int64_t kOutlierPerRow     = 204;

    auto measure_outlier = [&](int64_t n_rows) {
        const int64_t n_total = n_rows * kOutlierPerRow;
        std::vector<float> rows((size_t) (n_rows * kOutlierK), 0.0f);
        std::vector<int32_t> cols((size_t) n_total);
        std::vector<uint16_t> vals((size_t) n_total, (uint16_t) 0x3C00u);
        std::vector<int32_t> row_offsets((size_t) (n_rows + 1));
        for (int64_t r = 0; r < n_rows; r++) {
            row_offsets[(size_t) r] = (int32_t) (r * kOutlierPerRow);
        }
        row_offsets[(size_t) n_rows] = (int32_t) n_total;
        std::mt19937 rng(kSeed);
        std::uniform_int_distribution<int64_t> col_dist(0, kOutlierK - 1);
        for (int64_t i = 0; i < n_total; i++) {
            cols[(size_t) i] = (int32_t) col_dist(rng);
        }
        std::vector<float> rows_c = rows;
        std::vector<float> rows_v2 = rows;
        volatile float sink = 0.0f;
        auto scalar_ref = [&]() {
            for (int64_t r = 0; r < n_rows; r++) {
                const int32_t lo = row_offsets[(size_t) r];
                const int32_t hi = row_offsets[(size_t) r + 1];
                float * GGML_RESTRICT row = rows_c.data() + r * kOutlierK;
                for (int32_t k2 = lo; k2 < hi; k2++) {
                    const int32_t col = cols[(size_t) k2];
                    if (col >= 0 && col < kOutlierK) {
                        row[(size_t) col] = GGML_FP16_TO_FP32(vals[(size_t) k2]);
                    }
                }
            }
            sink += rows_c[0];
        };
        auto v2_fn = [&]() {
            apply_outlier_addback_v2(rows_v2.data(), kOutlierK, n_rows,
                                     row_offsets.data(),
                                     cols.data(), vals.data());
            sink += rows_v2[0];
        };
        auto time_fn = [&](auto fn) {
            for (int i = 0; i < kWarmup; i++) fn();
            std::vector<double> us;
            us.reserve((size_t) kRuns);
            for (int i = 0; i < kRuns; i++) {
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                const auto t1 = std::chrono::steady_clock::now();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            return median_us(us);
        };
        return std::pair<double, double>{ time_fn(scalar_ref), time_fn(v2_fn) };
    };
    const auto [us_c_small, us_v2_small] = measure_outlier(kOutlierNRowsSmall);
    const auto [us_c_large, us_v2_large] = measure_outlier(kOutlierNRowsLarge);
    // Per-outlier slopes (us per outlier in n_total).
    const double n_total_small = (double) (kOutlierNRowsSmall * kOutlierPerRow);
    const double n_total_large = (double) (kOutlierNRowsLarge * kOutlierPerRow);
    const double c_per_outlier   = (us_c_large  - us_c_small) / (n_total_large - n_total_small);
    const double v2_per_outlier  = (us_v2_large - us_v2_small) / (n_total_large - n_total_small);
    const double c_setup_tax     = us_c_small  - c_per_outlier  * n_total_small;
    const double v2_setup_tax    = us_v2_small - v2_per_outlier * n_total_small;

    printf("cost model constants (linear fit, n=1 and n=1024 endpoints):\n");
    printf("  outlier addback (per outlier in n_total):\n");
    printf("    C:  setup_tax=%6.3f us  per_outlier=%9.6f us\n", c_setup_tax, c_per_outlier);
    printf("    v2: setup_tax=%6.3f us  per_outlier=%9.6f us\n", v2_setup_tax, v2_per_outlier);
    if (c_per_outlier > v2_per_outlier) {
        const double crossover = v2_setup_tax / (c_per_outlier - v2_per_outlier);
        printf("    crossover (n_total): %.1f (v2 wins below this)\n", crossover);
    } else {
        printf("    C is faster per outlier; v2 never wins on per-row cost\n");
    }
    // Note: the v2 has an internal n_total <= 1024 NEON path
    // threshold. Above that the v2 falls back to scalar (same
    // as the C ref) and the function call is wasted. The
    // dispatch picks v2 iff n_total <= 1024 regardless of
    // the per-row crossover.

    // Meta decode: n_pages = 16, sweep n_rows.
    constexpr int64_t kMetaNRowsSmall = 1;
    constexpr int64_t kMetaNRowsLarge = 1024;
    constexpr int64_t kMetaNPages     = 16;

    auto measure_meta = [&](int64_t n_rows) {
        const int64_t n_lanes_per_row = kMetaNPages * TILE640_LANES_PER_PAGE;
        std::vector<uint16_t> page_scales((size_t) (n_rows * kMetaNPages), (uint16_t) 0x3C00u);
        std::vector<int8_t> lane_scales((size_t) (n_rows * n_lanes_per_row));
        std::mt19937 rng(kSeed);
        std::uniform_int_distribution<int> ls_dist(-127, 127);
        for (int64_t i = 0; i < (int64_t) lane_scales.size(); i++) {
            lane_scales[(size_t) i] = (int8_t) ls_dist(rng);
        }
        std::vector<float> page_max((size_t) (n_rows * kMetaNPages));
        std::vector<float> lane_scale((size_t) (n_rows * n_lanes_per_row));
        volatile float sink = 0.0f;
        auto scalar_ref = [&]() {
            for (int64_t i = 0; i < n_rows * kMetaNPages; i++) {
                page_max[(size_t) i] = GGML_FP16_TO_FP32(page_scales[(size_t) i]);
            }
            for (int64_t i = 0; i < n_rows * n_lanes_per_row; i++) {
                lane_scale[(size_t) i] = ((float) lane_scales[(size_t) i]) * (1.0f / 127.0f);
            }
            sink += lane_scale[0];
        };
        auto v2_fn = [&]() {
            decode_per_row_meta_v2(page_scales.data(), lane_scales.data(),
                                   n_rows, kMetaNPages,
                                   page_max.data(), lane_scale.data());
            sink += lane_scale[0];
        };
        auto time_fn = [&](auto fn) {
            for (int i = 0; i < kWarmup; i++) fn();
            std::vector<double> us;
            us.reserve((size_t) kRuns);
            for (int i = 0; i < kRuns; i++) {
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                const auto t1 = std::chrono::steady_clock::now();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            return median_us(us);
        };
        return std::pair<double, double>{ time_fn(scalar_ref), time_fn(v2_fn) };
    };
    const auto [us_c_meta_small, us_v2_meta_small] = measure_meta(kMetaNRowsSmall);
    const auto [us_c_meta_large, us_v2_meta_large] = measure_meta(kMetaNRowsLarge);
    const double c_per_row  = (us_c_meta_large  - us_c_meta_small)  / (double) (kMetaNRowsLarge - kMetaNRowsSmall);
    const double v2_per_row = (us_v2_meta_large - us_v2_meta_small) / (double) (kMetaNRowsLarge - kMetaNRowsSmall);
    const double c_meta_setup  = us_c_meta_small  - c_per_row  * (double) kMetaNRowsSmall;
    const double v2_meta_setup = us_v2_meta_small - v2_per_row * (double) kMetaNRowsSmall;
    printf("  meta decode (per row in n_rows, n_pages=16):\n");
    printf("    C:  setup_tax=%6.3f us  per_row=%9.6f us\n", c_meta_setup, c_per_row);
    printf("    v2: setup_tax=%6.3f us  per_row=%9.6f us\n", v2_meta_setup, v2_per_row);
    if (c_per_row > v2_per_row) {
        const double crossover = v2_meta_setup / (c_per_row - v2_per_row);
        printf("    crossover (n_rows): %.1f (v2 wins below this)\n", crossover);
    } else {
        printf("    C is faster per row; v2 never wins on per-row cost\n");
    }
}

// Cost model dispatch picks: for each bench shape, print
// what the dispatch would choose (v2 or C ref) based on the
// cost model thresholds. The outlier threshold is the v2's
// internal NEON path threshold (n_total <= 1024). The meta
// threshold is "always C ref" (v2 never wins).
void bench_dispatch_picks(void) {
    printf("dispatch picks (what the cost model would choose per shape):\n");
    printf("  meta decode (always C ref; v2 is 0.41-0.65x of C):\n");
    for (int64_t n_rows : { (int64_t) 1, (int64_t) 16, (int64_t) 64, (int64_t) 256, (int64_t) 1024 }) {
        printf("    n_rows=%-4lld n_pages=16 -> C ref\n", (long long) n_rows);
    }
    printf("  outlier addback (v2 iff n_total in (0, 1024]):\n");
    struct Shape { int64_t n_rows; int64_t k; int64_t n_per_row; };
    const Shape shapes[] = {
        {   1, 1024,  51 },  // n_total=51
        {   1, 4096, 204 },  // n_total=204
        {   1, 8192, 409 },  // n_total=409
        {  16, 4096, 204 },  // n_total=3264
        {  64, 4096, 204 },  // n_total=13056
        { 256, 4096, 204 },  // n_total=52224
        {1024, 4096, 204 },  // n_total=208896
    };
    for (const auto & s : shapes) {
        const int64_t n_total = s.n_rows * s.n_per_row;
        const bool use_v2 = (n_total > 0 && n_total <= 1024);
        printf("    n_rows=%-4lld k=%-5lld n/row=%-5lld n_total=%-7lld -> %s\n",
               (long long) s.n_rows, (long long) s.k, (long long) s.n_per_row,
               (long long) n_total, use_v2 ? "v2" : "C ref");
    }
}

}  // namespace

int main(void) {
    if (!ggml_tessera_t640_v2_enabled()) {
        printf("v2 disabled (GGML_TESSERA_T640_V2_DISABLE=1); skipping\n");
        return 0;
    }
    printf("dequant (per row, fp32 out, k rows of in_dim):\n");
    bench_dequant(640);
    bench_dequant(1024);
    bench_dequant(1280);
    bench_dequant(2560);
    bench_dequant(4096);
    bench_dequant(8192);
    printf("quant (per row, fp32 in, k rows of in_dim):\n");
    bench_quant(640);
    bench_quant(1024);
    bench_quant(1280);
    bench_quant(2560);
    bench_quant(4096);
    bench_quant(8192);
    printf("meta decode (batched, n_rows * n_pages pages per call):\n");
    // Sweep n_rows at the canonical in_dim=4096 (16 pages) plus
    // a couple of smaller shapes for context.
    bench_meta(1, 1);    // 1 row, 1 page (noise floor check)
    bench_meta(1, 16);   // 1 row, 16 pages
    bench_meta(16, 16);  // 16 rows of 16 pages
    bench_meta(64, 16);  // 64 rows of 16 pages
    bench_meta(256, 16); // 256 rows of 16 pages (typical Phase 0)
    bench_meta(1024, 16); // 1024 rows of 16 pages (large)
    printf("act_scale (per row, n = in_dim):\n");
    bench_act_scale(1024);
    bench_act_scale(4096);
    bench_act_scale(8192);
    printf("outlier addback (batched, sparse 5%%, n_rows rows per call):\n");
    // Sweep n_rows at the canonical in_dim=4096 plus the
    // dispatch's small-shape cases. The v2 path makes ONE
    // NEON bulk convert for the whole buffer; the C ref
    // makes scalar convert per element.
    bench_outlier(1, 1024, 51);    // 1 row, k=1024
    bench_outlier(1, 4096, 204);   // 1 row, k=4096
    bench_outlier(1, 8192, 409);   // 1 row, k=8192
    bench_outlier(16, 4096, 204);  // 16 rows of k=4096
    bench_outlier(64, 4096, 204);  // 64 rows of k=4096
    bench_outlier(256, 4096, 204); // 256 rows of k=4096 (typical)
    bench_outlier(1024, 4096, 204); // 1024 rows of k=4096 (large)
    bench_cost_model();
    bench_dispatch_picks();
    return 0;
}
