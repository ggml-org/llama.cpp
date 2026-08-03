//
// tessera-imatrix.cpp
//
// Minimal .npz reader (ZIP + NPY) and regime statistics.
//

#include "tessera-imatrix.h"

#include "imatrix-loader.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// minimal ZIP local-file-header reader (stored entries only, no compression)
// ---------------------------------------------------------------------------

struct zip_entry {
    std::string name;
    const uint8_t * data;
    uint32_t size;
};

static uint16_t read_u16le(const uint8_t * p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t read_u32le(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Parse all local file headers from a ZIP buffer. Returns entries found.
static std::vector<zip_entry> zip_parse(const uint8_t * buf, size_t len) {
    std::vector<zip_entry> entries;
    size_t pos = 0;

    while (pos + 30 <= len) {
        if (read_u32le(buf + pos) != 0x04034b50) {
            break; // no more local file headers
        }

        uint16_t method   = read_u16le(buf + pos + 8);
        uint32_t comp_sz  = read_u32le(buf + pos + 18);
        uint32_t uncomp_sz = read_u32le(buf + pos + 22);
        uint16_t name_len = read_u16le(buf + pos + 26);
        uint16_t extra_len = read_u16le(buf + pos + 28);

        size_t name_off = pos + 30;
        size_t data_off = name_off + name_len + extra_len;

        if (data_off + comp_sz > len) {
            break;
        }

        // only support stored (method 0)
        if (method == 0) {
            zip_entry e;
            e.name.assign((const char *)buf + name_off, name_len);
            e.data = buf + data_off;
            e.size = std::min(uncomp_sz, comp_sz); // bound to the validated extent
            entries.push_back(std::move(e));
        }

        pos = data_off + comp_sz;
    }

    return entries;
}

// ---------------------------------------------------------------------------
// NPY parsing
// ---------------------------------------------------------------------------

// Parse the element count from an NPY header string.
// Looks for 'shape': (N,) or 'shape': (N, M, ...) and returns total elements.
static int64_t npy_parse_shape(const std::string & hdr) {
    size_t pos = hdr.find("'shape'");
    if (pos == std::string::npos) {
        pos = hdr.find("\"shape\"");
    }
    if (pos == std::string::npos) {
        return -1;
    }

    pos = hdr.find('(', pos);
    if (pos == std::string::npos) {
        return -1;
    }

    size_t end = hdr.find(')', pos);
    if (end == std::string::npos) {
        return -1;
    }

    std::string tuple = hdr.substr(pos + 1, end - pos - 1);

    int64_t total = 1;
    size_t cur = 0;
    bool found_any = false;
    while (cur < tuple.size()) {
        while (cur < tuple.size() && (tuple[cur] == ' ' || tuple[cur] == ',')) {
            cur++;
        }
        if (cur >= tuple.size()) {
            break;
        }
        int64_t val = 0;
        bool has_digit = false;
        while (cur < tuple.size() && tuple[cur] >= '0' && tuple[cur] <= '9') {
            val = val * 10 + (tuple[cur] - '0');
            cur++;
            has_digit = true;
        }
        if (has_digit) {
            total *= val;
            found_any = true;
        }
    }

    return found_any ? total : -1;
}

// Parse a .npy buffer into a float vector. Returns false on error.
static bool npy_to_floats(const uint8_t * buf, uint32_t len, std::vector<float> & out) {
    // magic: \x93NUMPY (6 bytes) + version (2 bytes)
    if (len < 10) {
        return false;
    }
    if (buf[0] != 0x93 || std::memcmp(buf + 1, "NUMPY", 5) != 0) {
        return false;
    }

    uint8_t major = buf[6];
    uint32_t hdr_len;
    size_t data_off;

    if (major == 1) {
        hdr_len = read_u16le(buf + 8);
        data_off = (size_t)10 + hdr_len;
    } else {
        if (len < 12) {
            return false;
        }
        hdr_len = read_u32le(buf + 8);
        data_off = (size_t)12 + hdr_len;
    }

    if (data_off > len) {
        return false;
    }

    std::string hdr((const char *)buf + (major == 1 ? 10 : 12), hdr_len);

    int64_t n_elem = npy_parse_shape(hdr);
    if (n_elem <= 0) {
        return false;
    }

    // verify descr is <f4
    if (hdr.find("<f4") == std::string::npos) {
        return false;
    }

    if ((size_t)n_elem > (len - data_off) / 4) {
        return false;
    }
    size_t data_bytes = (size_t)n_elem * 4;

    out.resize(n_elem);
    std::memcpy(out.data(), buf + data_off, data_bytes);
    return true;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

int ts_imatrix_load_npz(const char * path, ts_imatrix * out, std::string * err_msg) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        if (err_msg) *err_msg = std::string("cannot open: ") + path;
        return 1;
    }

    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (sz <= 0) {
        std::fclose(f);
        if (err_msg) *err_msg = "empty file";
        return 1;
    }

    std::vector<uint8_t> buf(sz);
    if (std::fread(buf.data(), 1, sz, f) != (size_t)sz) {
        std::fclose(f);
        if (err_msg) *err_msg = "read error";
        return 1;
    }
    std::fclose(f);

    auto entries = zip_parse(buf.data(), buf.size());
    if (entries.empty()) {
        if (err_msg) *err_msg = "no zip entries found";
        return 1;
    }

    // entries point into buf, so buf must outlive processing.
    // parse all .npy entries now.
    for (auto & e : entries) {
        // strip .npy suffix for the key
        std::string key = e.name;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") {
            key = key.substr(0, key.size() - 4);
        }

        std::vector<float> vals;
        if (!npy_to_floats(e.data, e.size, vals)) {
            continue; // skip non-f32 or malformed entries
        }

        out->data[key] = std::move(vals);
    }

    out->source_path = path;

    if (out->data.empty()) {
        if (err_msg) *err_msg = "no valid .npy entries found";
        return 1;
    }

    return 0;
}

int ts_imatrix_load_gguf(const char * path, ts_imatrix * out, std::string * err_msg) {
    // GGUF imatrix files are emitted by llama-imatrix and the standard
    // quantize path. Each tensor's entry carries per-channel sums, abs_sums,
    // fourth_sums, max_abs and counts. The per-channel activation magnitude
    // used by the tessera AWQ/regime path is sums[i] / counts[i] (the mean
    // squared activation); counts<=0 is treated as 1 to avoid div-by-zero.
    common_imatrix loaded;
    if (!common_imatrix_load(path, loaded)) {
        if (err_msg) {
            *err_msg = std::string("common_imatrix_load failed for: ") + path;
        }
        return 1;
    }

    for (const auto & kv : loaded.entries) {
        const auto & entry = kv.second;
        const size_t n = entry.sums.size();
        if (n == 0) {
            continue;
        }
        std::vector<float> vals(n);
        for (size_t i = 0; i < n; i++) {
            int64_t c = (i < entry.counts.size()) ? entry.counts[i] : 0;
            float denom = (c > 0) ? (float)c : 1.0f;
            vals[i] = entry.sums[i] / denom;
        }
        out->data[kv.first] = std::move(vals);

        // Per-channel max |activation| (.in_maxabs). The producer always
        // collects this; the loader's optional-field path leaves the vector
        // empty when an older GGUF omits it. The outlier-aware experts key
        // off this, so we surface it verbatim - no counts scaling applies
        // (max is non-additive by definition).
        if (!entry.max_abs.empty()) {
            out->max_abs[kv.first] = entry.max_abs;
        }
    }

    out->source_path = path;

    if (out->data.empty()) {
        if (err_msg) *err_msg = "GGUF imatrix contained no tensor entries";
        return 1;
    }

    return 0;
}

// Normalize a tensor name the same way ts_imatrix_lookup does, so lookups
// into data and max_abs cannot drift apart (a typo here would silently drop
// the per-channel max and the experts would fall back to the global budget).
static std::string ts_imatrix_normalize_name(const char * tensor_name) {
    std::string name(tensor_name ? tensor_name : "");
    if (name.size() > 7 && name.substr(name.size() - 7) == ".weight") {
        name = name.substr(0, name.size() - 7);
    }
    return name;
}

const float * ts_imatrix_lookup(const ts_imatrix * imatrix,
                                const char * tensor_name,
                                int64_t * out_dim) {
    if (!imatrix || !tensor_name) {
        return nullptr;
    }

    const std::string name = ts_imatrix_normalize_name(tensor_name);

    auto it = imatrix->data.find(name);
    if (it == imatrix->data.end()) {
        return nullptr;
    }

    if (out_dim) {
        *out_dim = (int64_t)it->second.size();
    }
    return it->second.data();
}

const float * ts_imatrix_lookup_max_abs(const ts_imatrix * imatrix,
                                        const char * tensor_name,
                                        int64_t * out_dim) {
    if (!imatrix || !tensor_name || imatrix->max_abs.empty()) {
        return nullptr;
    }

    const std::string name = ts_imatrix_normalize_name(tensor_name);

    auto it = imatrix->max_abs.find(name);
    if (it == imatrix->max_abs.end()) {
        return nullptr;
    }

    if (out_dim) {
        *out_dim = (int64_t)it->second.size();
    }
    return it->second.data();
}

ts_imatrix_regime_stats ts_imatrix_regime(const float * act_data, int64_t dim) {
    ts_imatrix_regime_stats st = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (!act_data || dim <= 0) {
        return st;
    }

    // mean |x|, E[x^2], E[x^4]
    double sum_abs = 0.0;
    double sum_x2  = 0.0;
    double sum_x4  = 0.0;

    for (int64_t i = 0; i < dim; ++i) {
        double v  = (double)act_data[i];
        double av = std::fabs(v);
        double x2 = v * v;
        sum_abs += av;
        sum_x2  += x2;
        sum_x4  += x2 * x2;
    }

    double n = (double)dim;
    double mean_abs = sum_abs / n;
    double ex2 = sum_x2 / n;
    double ex4 = sum_x4 / n;

    st.mean_magnitude = (float)mean_abs;

    // excess kurtosis: E[x^4] / E[x^2]^2 - 3
    if (ex2 > 1e-30) {
        st.kurtosis = (float)(ex4 / (ex2 * ex2) - 3.0);
    }

    // effective rank proxy: CV = std/mean, mapped to [0,1] via cv/(1+cv)
    double var = ex2 - mean_abs * mean_abs;
    if (var < 0.0) var = 0.0;
    double sd = std::sqrt(var);
    if (mean_abs > 1e-30) {
        double cv = sd / mean_abs;
        st.eff_rank = (float)(cv / (1.0 + cv));
    }

    // p99: sort a copy
    std::vector<float> sorted(act_data, act_data + dim);
    std::sort(sorted.begin(), sorted.end());
    int64_t idx = (int64_t)(0.99 * (dim - 1));
    st.p99 = sorted[idx];

    return st;
}
