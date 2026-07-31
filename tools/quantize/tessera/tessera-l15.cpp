//
// tessera-l15.cpp
//
// L1.5 FP16 reference reader and kernel-direct fitness metrics.
//

#include "tessera-l15.h"
#include "tessera-vec.h"
#include "tessera-sidecar-v3.h"

#include <dirent.h>
#include <cstring>
#include <cmath>

int ts_l15_load_reference(const char * sidecar_path,
                          ts_l15_reference * out,
                          std::string * err_msg) {
    ts_sidecar_v3 sc;
    if (ts_sidecar_v3_read(sidecar_path, &sc, err_msg) != 0) {
        return -1;
    }

    out->rows = sc.header.rows;
    out->cols = sc.header.cols;
    out->data = std::move(sc.data);
    out->outlier_threshold = sc.header.outlier_threshold;
    out->outlier_count     = sc.header.outlier_count_total;

    // derive tensor name from filename: strip directory and .tdqt suffix
    const char * base = strrchr(sidecar_path, '/');
    base = base ? base + 1 : sidecar_path;
    std::string name(base);
    const size_t dot = name.rfind(".tdqt");
    if (dot != std::string::npos) {
        name.resize(dot);
    }
    out->tensor_name = std::move(name);

    return 0;
}

int ts_l15_load_directory(const char * dir_path,
                          std::vector<ts_l15_reference> * refs,
                          std::string * err_msg) {
    DIR * dir = opendir(dir_path);
    if (!dir) {
        if (err_msg) { *err_msg = "failed to open directory: "; *err_msg += dir_path; }
        return -1;
    }

    int count = 0;
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        const char * name = ent->d_name;
        size_t len = strlen(name);
        if (len < 6 || strcmp(name + len - 5, ".tdqt") != 0) {
            continue;
        }

        std::string path(dir_path);
        path += '/';
        path += name;

        ts_l15_reference ref;
        std::string load_err;
        if (ts_l15_load_reference(path.c_str(), &ref, &load_err) != 0) {
            if (err_msg) { *err_msg = path + ": " + load_err; }
            closedir(dir);
            return -1;
        }

        refs->push_back(std::move(ref));
        count++;
    }

    closedir(dir);
    return count;
}

float ts_l15_relative_frob(const float * w_hat, const ts_l15_reference * ref) {
    const int64_t n = ref->rows * ref->cols;
    if (n <= 0) {
        return 0.0f;
    }

    const float * w_ref = ref->data.data();

    float num = 0.0f;
    float den = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float d = w_hat[i] - w_ref[i];
        num += d * d;
        den += w_ref[i] * w_ref[i];
    }

    if (den == 0.0f) {
        return 0.0f;
    }
    return num / den;
}

float ts_l15_layer_output_mse(const float * w_hat,
                              const ts_l15_reference * ref,
                              const float * calib_X,
                              int64_t n_tokens) {
    const int64_t out_dim = ref->rows;
    const int64_t in_dim  = ref->cols;
    if (out_dim <= 0 || in_dim <= 0 || n_tokens <= 0) {
        return 0.0f;
    }

    // Y_hat = W_hat @ X,  Y_ref = W_ref @ X   (out_dim x n_tokens)
    std::vector<float> y_hat((size_t)out_dim * (size_t)n_tokens);
    std::vector<float> y_ref((size_t)out_dim * (size_t)n_tokens);

    ts_mat_mul(w_hat,          calib_X, y_hat.data(), out_dim, in_dim, n_tokens);
    ts_mat_mul(ref->data.data(), calib_X, y_ref.data(), out_dim, in_dim, n_tokens);

    const int64_t total = out_dim * n_tokens;
    float sse = 0.0f;
    for (int64_t i = 0; i < total; ++i) {
        const float d = y_hat[i] - y_ref[i];
        sse += d * d;
    }

    return sse / (float)total;
}
