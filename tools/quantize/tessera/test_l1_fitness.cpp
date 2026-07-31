//
// test_l1_fitness.cpp
//
// Smoke test for tessera-l1-fitness. Writes synthetic L1 sidecar files
// (original weights + small noise to simulate kernel precision), loads
// them back, and checks the kernel-direct t_l^2, the offline/kernel
// blend, and the no-sidecar fallback.
//

#include "tessera-l1-fitness.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

static void check_close(const char * name, float got, float want, float tol) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-28s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-28s %.7g\n", name, (double)got);
    }
}

static const char * TEST_DIR = "/tmp/test_l1_fitness";

// Write a v3 TDQT sidecar (same on-disk layout as the runtime hook).
static bool write_sidecar(const std::string & name, int64_t rows, int64_t cols,
                          const std::vector<float> & data) {
    const std::string path = std::string(TEST_DIR) + "/" + name + ".dequant.f32";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }

    fwrite("TDQT", 1, 4, f);
    uint32_t version = 3;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    uint32_t dtype = 0;
    fwrite(&dtype, sizeof(dtype), 1, f);
    float outlier_threshold = 6.0f;
    fwrite(&outlier_threshold, sizeof(outlier_threshold), 1, f);
    int64_t outlier_count_total = 0;
    fwrite(&outlier_count_total, sizeof(outlier_count_total), 1, f);

    // row_outlier_counts (int32 per row, zeroed)
    std::vector<int32_t> row_outlier_counts((size_t)rows, 0);
    fwrite(row_outlier_counts.data(), sizeof(int32_t), (size_t)rows, f);

    // row_meta (24 bytes each, zeroed)
    uint8_t row_meta[24];
    memset(row_meta, 0, sizeof(row_meta));
    for (int64_t r = 0; r < rows; r++) {
        fwrite(row_meta, 1, 24, f);
    }

    fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    return true;
}

int main() {
    std::filesystem::create_directories(TEST_DIR);

    const int64_t rows = 8;
    const int64_t cols = 16;
    const int64_t n    = rows * cols;

    // original weights, an offline reconstruction (w_hat), and the kernel's
    // real dequant output (w_original + different small noise).
    std::vector<float> w_orig((size_t)n);
    std::vector<float> w_hat((size_t)n);
    std::vector<float> k_deq((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        w_orig[(size_t)i] = (float)((i % 7) - 3) * 0.1f + 0.05f;
        w_hat[(size_t)i]  = w_orig[(size_t)i] + 0.01f * (float)((i % 5) - 2);
        k_deq[(size_t)i]  = w_orig[(size_t)i] + 0.02f * (float)((i % 3) - 1);
    }

    check("write sidecar", write_sidecar("tensor_a", rows, cols, k_deq));

    // --- load ---
    std::vector<float> loaded;
    int64_t lr = 0;
    int64_t lc = 0;
    int lrc = ts_l1_load_sidecar(TEST_DIR, "tensor_a", &loaded, &lr, &lc);
    check("load rc == 0", lrc == 0);
    check("load rows", lr == rows);
    check("load cols", lc == cols);
    bool data_match = ((int64_t)loaded.size() == n);
    for (int64_t i = 0; i < n && data_match; i++) {
        data_match = (loaded[(size_t)i] == k_deq[(size_t)i]);
    }
    check("load data round-trip", data_match);

    // --- kernel-direct vs offline t_l^2 ---
    // offline proxy: ||w_hat - w_orig||^2 / ||w_orig||^2
    double num_off = 0.0;
    double den     = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)w_hat[(size_t)i] - (double)w_orig[(size_t)i];
        num_off += d * d;
        den     += (double)w_orig[(size_t)i] * (double)w_orig[(size_t)i];
    }
    const float offline_t2 = (float)(num_off / den);
    const float kernel_t2  = ts_l1_kernel_direct_t2(w_hat.data(), w_orig.data(),
                                                    loaded.data(), n);
    check("kernel_t2 > 0", kernel_t2 > 0.0f);
    check("kernel_t2 != offline_t2", std::fabs(kernel_t2 - offline_t2) > 1e-9f);
    // both are small relative errors and of the same order of magnitude
    check("kernel_t2 close to offline", std::fabs(kernel_t2 - offline_t2) < 0.05f);
    std::printf("     offline_t2=%.7g kernel_t2=%.7g\n", (double)offline_t2, (double)kernel_t2);

    // --- blending ---
    check_close("blend=0 -> offline", ts_l1_blended_t2(offline_t2, kernel_t2, 0.0f), offline_t2, 1e-7f);
    check_close("blend=1 -> kernel",  ts_l1_blended_t2(offline_t2, kernel_t2, 1.0f), kernel_t2, 1e-7f);
    check_close("blend=0.5 -> avg",   ts_l1_blended_t2(offline_t2, kernel_t2, 0.5f),
                0.5f * (offline_t2 + kernel_t2), 1e-7f);
    check_close("blend clamps low",   ts_l1_blended_t2(offline_t2, kernel_t2, -1.0f), offline_t2, 1e-7f);
    check_close("blend clamps high",  ts_l1_blended_t2(offline_t2, kernel_t2, 2.0f), kernel_t2, 1e-7f);

    // --- batch + fallback ---
    // tensor_a has a sidecar; tensor_missing does not (falls back to offline).
    std::vector<float> w_hat2((size_t)n), w_orig2((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        w_orig2[(size_t)i] = (float)((i % 5) - 2) * 0.2f + 0.1f;
        w_hat2[(size_t)i]  = w_orig2[(size_t)i] + 0.015f * (float)((i % 4) - 1);
    }
    double num2 = 0.0;
    double den2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)w_hat2[(size_t)i] - (double)w_orig2[(size_t)i];
        num2 += d * d;
        den2 += (double)w_orig2[(size_t)i] * (double)w_orig2[(size_t)i];
    }
    const float offline_t2_2 = (float)(num2 / den2);

    const float * w_hats[2]      = { w_hat.data(),  w_hat2.data() };
    const float * w_originals[2] = { w_orig.data(), w_orig2.data() };
    const char  * names[2]       = { "tensor_a", "tensor_missing" };
    const int64_t sizes[2]       = { n, n };
    float out_t2[2] = { -1.0f, -1.0f };

    int n_with = ts_l1_compute_all_t2(TEST_DIR, w_hats, w_originals, names, 2, sizes, out_t2);
    check("batch sidecar count == 1", n_with == 1);
    check_close("batch tensor_a == kernel_t2", out_t2[0], kernel_t2, 1e-7f);
    check_close("batch missing == offline",    out_t2[1], offline_t2_2, 1e-7f);

    // missing sidecar load fails cleanly
    std::vector<float> none;
    int64_t nr = 0;
    int64_t nc = 0;
    check("missing sidecar rc != 0",
          ts_l1_load_sidecar(TEST_DIR, "tensor_missing", &none, &nr, &nc) != 0);

    std::filesystem::remove_all(TEST_DIR);

    if (g_fail == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_fail);
    return 1;
}
