//
// test_l15.cpp
//
// Smoke test for tessera-l15. Writes a synthetic v3 sidecar, loads it
// back, verifies dimensions/data, and checks relative_frob sanity.
//

#include "tessera-l15.h"
#include "tessera-sidecar-v3.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
        std::printf("FAIL %-24s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-24s %.7g\n", name, (double)got);
    }
}

static const char * k_path = "/tmp/test_l15_ref.act.dequant.f32";

static bool write_synthetic_sidecar() {
    const int64_t rows = 4;
    const int64_t cols = 8;

    FILE * f = fopen(k_path, "wb");
    if (!f) {
        return false;
    }

    // header
    fwrite("TDQT", 1, 4, f);
    uint32_t version = 3;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    uint32_t dtype = 0;
    fwrite(&dtype, sizeof(dtype), 1, f);
    float outlier_threshold = 6.0f;
    fwrite(&outlier_threshold, sizeof(outlier_threshold), 1, f);
    int64_t outlier_count_total = 2;
    fwrite(&outlier_count_total, sizeof(outlier_count_total), 1, f);

    // row_outlier_counts
    int32_t row_outlier_counts[4] = { 1, 0, 1, 0 };
    fwrite(row_outlier_counts, sizeof(int32_t), 4, f);

    // row_meta (24 bytes each, zeroed)
    uint8_t row_meta[24];
    memset(row_meta, 0, sizeof(row_meta));
    for (int i = 0; i < 4; i++) {
        fwrite(row_meta, 1, 24, f);
    }

    // data: 4 x 8, values 0.0 .. 31.0
    float data[32];
    for (int i = 0; i < 32; i++) {
        data[i] = (float)i;
    }
    fwrite(data, sizeof(float), 32, f);

    fclose(f);
    return true;
}

int main() {
    // 1. write synthetic sidecar
    check("write sidecar", write_synthetic_sidecar());

    // 2. load
    ts_l15_reference ref;
    std::string err;
    int rc = ts_l15_load_reference(k_path, &ref, &err);
    check("load rc == 0", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        return 1;
    }

    // 3. verify dimensions and data
    check("rows == 4", ref.rows == 4);
    check("cols == 8", ref.cols == 8);
    check("data.size == 32", (int64_t)ref.data.size() == 32);
    check("outlier_threshold", ref.outlier_threshold == 6.0f);
    check("outlier_count", ref.outlier_count == 2);
    check("tensor_name", ref.tensor_name == "test_l15_ref");

    bool data_ok = true;
    for (int i = 0; i < 32; i++) {
        if (ref.data[i] != (float)i) {
            data_ok = false;
            std::printf("  data[%d] = %f, want %f\n", i, (double)ref.data[i], (double)i);
            break;
        }
    }
    check("data values", data_ok);

    // 4. relative_frob: identical -> 0
    check_close("frob(identical)", ts_l15_relative_frob(ref.data.data(), &ref), 0.0f, 1e-7f);

    // perturbed version: add 0.01 to every element
    std::vector<float> perturbed(ref.data);
    for (auto & v : perturbed) {
        v += 0.01f;
    }
    float frob = ts_l15_relative_frob(perturbed.data(), &ref);
    check("frob(perturbed) > 0", frob > 0.0f);
    check("frob(perturbed) < 0.01", frob < 0.01f);
    std::printf("  frob(perturbed) = %.7g\n", (double)frob);

    // 5. layer_output_mse: identical weights -> 0
    // calib_X: (cols x n_tokens) = (8 x 2), simple values
    const int64_t n_tokens = 2;
    float calib_X[16];
    for (int i = 0; i < 16; i++) {
        calib_X[i] = (float)(i + 1) * 0.1f;
    }
    float mse_identical = ts_l15_layer_output_mse(ref.data.data(), &ref, calib_X, n_tokens);
    check_close("mse(identical)", mse_identical, 0.0f, 1e-6f);

    float mse_perturbed = ts_l15_layer_output_mse(perturbed.data(), &ref, calib_X, n_tokens);
    check("mse(perturbed) > 0", mse_perturbed > 0.0f);
    std::printf("  mse(perturbed) = %.7g\n", (double)mse_perturbed);

    // 6. load_directory on /tmp (should find our file)
    std::vector<ts_l15_reference> refs;
    int n_loaded = ts_l15_load_directory("/tmp", &refs, &err);
    check("load_directory >= 1", n_loaded >= 1);
    if (n_loaded < 0) {
        std::printf("  error: %s\n", err.c_str());
    }

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
