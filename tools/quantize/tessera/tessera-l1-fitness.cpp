//
// tessera-l1-fitness.cpp
//
// L1 kernel-direct fitness loader and metrics. See tessera-l1-fitness.h.
//

#include "tessera-l1-fitness.h"
#include "tessera-sidecar-v3.h"

#include <cstring>
#include <string>

// L1 dequant sidecar suffix written by the runtime hook
// (common/tessera-debug/tessera-debug.h, DEQUANT_FILE_SUFFIX_L1).
static const char * TS_L1_SIDECAR_SUFFIX = ".dequant.f32";

void ts_l1_fitness_default_config(ts_l1_fitness_config * cfg) {
    if (cfg == nullptr) {
        return;
    }
    cfg->sidecar_dir[0]    = '\0';
    cfg->use_kernel_direct = false;
    cfg->blend_factor      = 1.0f;
}

int ts_l1_load_sidecar(const char * sidecar_dir,
                       const char * tensor_name,
                       std::vector<float> * out_data,
                       int64_t * out_rows, int64_t * out_cols) {
    if (sidecar_dir == nullptr || tensor_name == nullptr || out_data == nullptr) {
        return -1;
    }

    std::string path(sidecar_dir);
    path += '/';
    path += tensor_name;
    path += TS_L1_SIDECAR_SUFFIX;

    ts_sidecar_v3 sc;
    if (ts_sidecar_v3_read(path.c_str(), &sc, nullptr) != 0) {
        return -1;
    }
    if (sc.header.rows <= 0 || sc.header.cols <= 0) {
        return -1;
    }
    if ((int64_t)sc.data.size() != sc.header.rows * sc.header.cols) {
        return -1;
    }

    *out_data = std::move(sc.data);
    if (out_rows != nullptr) {
        *out_rows = sc.header.rows;
    }
    if (out_cols != nullptr) {
        *out_cols = sc.header.cols;
    }
    return 0;
}

float ts_l1_kernel_direct_t2(const float * w_hat,
                             const float * w_original,
                             const float * kernel_dequant,
                             int64_t n) {
    if (w_hat == nullptr || w_original == nullptr || kernel_dequant == nullptr || n <= 0) {
        return 0.0f;
    }

    double num = 0.0;
    double den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double d = (double)w_hat[i] - (double)kernel_dequant[i];
        num += d * d;
        den += (double)w_original[i] * (double)w_original[i];
    }
    if (den == 0.0) {
        return 0.0f;
    }
    return (float)(num / den);
}

float ts_l1_blended_t2(float offline_t2, float kernel_direct_t2,
                       float blend_factor) {
    float b = blend_factor;
    if (b < 0.0f) {
        b = 0.0f;
    } else if (b > 1.0f) {
        b = 1.0f;
    }
    return (1.0f - b) * offline_t2 + b * kernel_direct_t2;
}

// Offline proxy t_l^2 = ||W_hat - W||_F^2 / ||W||_F^2 (fallback when no
// sidecar is available).
static float ts_l1_offline_t2(const float * w_hat, const float * w_original,
                              int64_t n) {
    return ts_l1_kernel_direct_t2(w_hat, w_original, w_original, n);
}

int ts_l1_compute_all_t2(const char * sidecar_dir,
                         const float * const * w_hats,
                         const float * const * w_originals,
                         const char * const * tensor_names,
                         int64_t n_tensors,
                         const int64_t * sizes,
                         float * out_t2) {
    if (w_hats == nullptr || w_originals == nullptr || tensor_names == nullptr ||
        sizes == nullptr || out_t2 == nullptr || n_tensors <= 0) {
        return 0;
    }

    int n_with_sidecar = 0;
    for (int64_t l = 0; l < n_tensors; l++) {
        const int64_t n = sizes[l];
        const float   offline = ts_l1_offline_t2(w_hats[l], w_originals[l], n);

        std::vector<float> kdeq;
        int64_t rows = 0;
        int64_t cols = 0;
        if (sidecar_dir != nullptr && sidecar_dir[0] != '\0' &&
            ts_l1_load_sidecar(sidecar_dir, tensor_names[l], &kdeq, &rows, &cols) == 0 &&
            rows * cols == n) {
            out_t2[l] = ts_l1_kernel_direct_t2(w_hats[l], w_originals[l], kdeq.data(), n);
            n_with_sidecar++;
        } else {
            out_t2[l] = offline;
        }
    }
    return n_with_sidecar;
}
