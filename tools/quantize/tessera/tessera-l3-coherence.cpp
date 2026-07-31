//
// tessera-l3-coherence.cpp
//
// L3 per-row coherence. See tessera-l3-coherence.h.
//

#include "tessera-l3-coherence.h"
#include "tessera-sidecar-v3.h"

#include <cmath>
#include <string>

// sidecar suffixes written by the runtime hook (common/tessera-debug)
static const char * TS_L3_SUFFIX_L1  = ".dequant.f32";
static const char * TS_L3_SUFFIX_REF = ".act.dequant.f32";

void ts_l3_default_config(ts_l3_config * cfg) {
    if (cfg == nullptr) {
        return;
    }
    cfg->sidecar_dir[0]   = '\0';
    cfg->reference_dir[0] = '\0';
    cfg->threshold        = 0.99f;
}

float ts_l3_row_cosine(const float * a, const float * b, int64_t n) {
    if (a == nullptr || b == nullptr || n <= 0) {
        return 0.0f;
    }

    double dot = 0.0;
    double na  = 0.0;
    double nb  = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }

    if (na == 0.0 || nb == 0.0) {
        return (na == 0.0 && nb == 0.0) ? 1.0f : 0.0f;
    }
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

int ts_l3_tensor_coherence(const float * l1, const float * ref,
                           int64_t rows, int64_t cols,
                           float threshold,
                           ts_l3_tensor_result * out) {
    if (l1 == nullptr || ref == nullptr || out == nullptr ||
        rows <= 0 || cols <= 0) {
        return -1;
    }

    out->rows = rows;
    out->cols = cols;
    out->flagged_rows.clear();
    out->n_flagged = 0;

    double cos_sum = 0.0;
    float  cos_min = 1.0f;
    for (int64_t r = 0; r < rows; r++) {
        const float * a = l1  + r * cols;
        const float * b = ref + r * cols;
        const float c = ts_l3_row_cosine(a, b, cols);
        cos_sum += (double)c;
        if (c < cos_min) {
            cos_min = c;
        }
        if (c < threshold) {
            out->flagged_rows.push_back(r);
            out->n_flagged++;
        }
    }

    out->mean_cosine = (float)(cos_sum / (double)rows);
    out->min_cosine  = cos_min;
    return 0;
}

// Load a v3 sidecar at <dir>/<name><suffix> into out (row-major F32).
static int ts_l3_load(const char * dir, const char * name, const char * suffix,
                      std::vector<float> * out, int64_t * rows, int64_t * cols) {
    std::string path(dir);
    path += '/';
    path += name;
    path += suffix;

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
    *out  = std::move(sc.data);
    *rows = sc.header.rows;
    *cols = sc.header.cols;
    return 0;
}

int ts_l3_run(const ts_l3_config * cfg,
              const char * const * tensor_names,
              int64_t n_tensors,
              ts_l3_report * report) {
    if (cfg == nullptr || tensor_names == nullptr || report == nullptr ||
        n_tensors < 0) {
        return -1;
    }

    const float threshold = cfg->threshold > 0.0f ? cfg->threshold : 0.99f;

    report->tensors.clear();
    report->n_tensors      = 0;
    report->n_flagged_rows = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = tensor_names[i];
        if (name == nullptr) {
            continue;
        }

        std::vector<float> l1;
        std::vector<float> ref;
        int64_t l1_rows = 0, l1_cols = 0;
        int64_t rf_rows = 0, rf_cols = 0;

        if (ts_l3_load(cfg->sidecar_dir,   name, TS_L3_SUFFIX_L1,
                       &l1, &l1_rows, &l1_cols) != 0) {
            continue;   // no L1 sidecar
        }
        if (ts_l3_load(cfg->reference_dir, name, TS_L3_SUFFIX_REF,
                       &ref, &rf_rows, &rf_cols) != 0) {
            continue;   // no reference sidecar
        }
        if (l1_rows != rf_rows || l1_cols != rf_cols) {
            continue;   // shape mismatch
        }

        ts_l3_tensor_result r;
        r.tensor_name = name;
        if (ts_l3_tensor_coherence(l1.data(), ref.data(), l1_rows, l1_cols,
                                   threshold, &r) != 0) {
            continue;
        }

        report->n_flagged_rows += r.n_flagged;
        report->tensors.push_back(std::move(r));
        report->n_tensors++;
    }

    return (int)report->n_tensors;
}
