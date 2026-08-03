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

namespace {

// Suffix lengths (including the leading dot). The L1.5 reader
// matches either suffix; both decode into F32 via the v3 reader.
constexpr size_t k_suffix_f32_len = 16;  // ".act.dequant.f32"
constexpr size_t k_suffix_f16_len = 16;  // ".act.dequant.f16"

// Strip the longest matching L1.5 suffix from `name`. Returns the
// stripped name and sets `*file_dtype` to the on-disk dtype (0 = F32,
// 1 = F16). When neither suffix is present, the name is returned
// unchanged and `*file_dtype` is set to UINT32_MAX.
std::string strip_l15_suffix(const char * path, uint32_t * file_dtype) {
    const char * base = strrchr(path, '/');
    base = base ? base + 1 : path;
    std::string name(base);
    // Try the longer match first (both suffixes are 16 chars; this
    // branch keeps the order explicit for the future-proof case where
    // one of them changes length).
    const size_t dot_f16 = name.rfind(".act.dequant.f16");
    if (dot_f16 != std::string::npos) {
        name.resize(dot_f16);
        *file_dtype = 1;
        return name;
    }
    const size_t dot_f32 = name.rfind(".act.dequant.f32");
    if (dot_f32 != std::string::npos) {
        name.resize(dot_f32);
        *file_dtype = 0;
        return name;
    }
    *file_dtype = UINT32_MAX;
    return name;
}

} // namespace

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
    out->file_dtype = sc.header.dtype;
    out->outlier_threshold = sc.header.outlier_threshold;
    out->outlier_count     = sc.header.outlier_count_total;

    out->tensor_name = strip_l15_suffix(sidecar_path, &out->file_dtype);
    // The header's dtype is the on-disk truth; the filename strip is
    // only used to derive the tensor name. If they disagree (an
    // inconsistent file), trust the header.
    out->file_dtype = sc.header.dtype;

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

    // First pass: collect all L1.5 sidecar files. The on-disk dtypes
    // (F32 vs F16) are tracked per tensor; when both are present, the
    // F16 (new default) wins and the F32 (legacy) is skipped.
    std::vector<std::string> f16_paths;
    std::vector<std::string> f32_paths;
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        const char * name = ent->d_name;
        size_t len = strlen(name);
        if (len >= k_suffix_f16_len &&
            strcmp(name + len - k_suffix_f16_len, ".act.dequant.f16") == 0) {
            std::string p(dir_path); p += '/'; p += name;
            f16_paths.push_back(std::move(p));
            continue;
        }
        if (len >= k_suffix_f32_len &&
            strcmp(name + len - k_suffix_f32_len, ".act.dequant.f32") == 0) {
            std::string p(dir_path); p += '/'; p += name;
            f32_paths.push_back(std::move(p));
            continue;
        }
    }

    // Tensor-name dedup: when both F16 and F32 sidecars exist for the
    // same tensor, only the F16 is loaded. F16 is the new ground
    // truth; the F32 duplicate is left in place for the legacy reader
    // (l3_sidecar_v3_reader.py etc.) but is not consumed here.
    auto has_f16 = [&](const std::string & f32_path) -> bool {
        uint32_t dt = UINT32_MAX;
        const std::string nm = strip_l15_suffix(f32_path.c_str(), &dt);
        for (const auto & p : f16_paths) {
            uint32_t d2 = UINT32_MAX;
            if (strip_l15_suffix(p.c_str(), &d2) == nm) {
                return true;
            }
        }
        return false;
    };

    int count = 0;
    for (const auto & p : f16_paths) {
        ts_l15_reference ref;
        std::string load_err;
        if (ts_l15_load_reference(p.c_str(), &ref, &load_err) != 0) {
            if (err_msg) { *err_msg = p + ": " + load_err; }
            closedir(dir);
            return -1;
        }
        refs->push_back(std::move(ref));
        count++;
    }
    for (const auto & p : f32_paths) {
        if (has_f16(p)) {
            // F16 is the preferred reference; skip the legacy F32
            // duplicate so the L1.5 metrics see the new ground truth.
            continue;
        }
        ts_l15_reference ref;
        std::string load_err;
        if (ts_l15_load_reference(p.c_str(), &ref, &load_err) != 0) {
            if (err_msg) { *err_msg = p + ": " + load_err; }
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
