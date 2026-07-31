//
// tessera-mm-imatrix.cpp
//
// Multimodal imatrix v3 reader. The v3 npz stores per-modality
// activation arrays named "<tensor>.<modality>.npy" with modality in
// {text, image, audio}. Regime stats are derived per modality.
//

#include "tessera-mm-imatrix.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// minimal ZIP local-file-header reader (stored entries only, no compression)
// copied from tessera-imatrix.cpp to keep this translation unit self-contained
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

static std::vector<zip_entry> zip_parse(const uint8_t * buf, size_t len) {
    std::vector<zip_entry> entries;
    size_t pos = 0;

    while (pos + 30 <= len) {
        if (read_u32le(buf + pos) != 0x04034b50) {
            break; // no more local file headers
        }

        uint16_t method    = read_u16le(buf + pos + 8);
        uint32_t comp_sz   = read_u32le(buf + pos + 18);
        uint32_t uncomp_sz = read_u32le(buf + pos + 22);
        uint16_t name_len  = read_u16le(buf + pos + 26);
        uint16_t extra_len = read_u16le(buf + pos + 28);

        size_t name_off = pos + 30;
        size_t data_off = name_off + name_len + extra_len;

        if (data_off + comp_sz > len) {
            break;
        }

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

static bool npy_to_floats(const uint8_t * buf, uint32_t len, std::vector<float> & out) {
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
// regime stats (mirrors ts_imatrix_regime; kept local so this unit links
// without tessera-imatrix.cpp)
// ---------------------------------------------------------------------------

static void mm_regime(const float * act_data, int64_t dim, ts_imatrix_regime_stats & st) {
    st = { 0.0f, 0.0f, 0.0f, 0.0f };

    if (!act_data || dim <= 0) {
        return;
    }

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

    if (ex2 > 1e-30) {
        st.kurtosis = (float)(ex4 / (ex2 * ex2) - 3.0);
    }

    double var = ex2 - mean_abs * mean_abs;
    if (var < 0.0) var = 0.0;
    double sd = std::sqrt(var);
    if (mean_abs > 1e-30) {
        double cv = sd / mean_abs;
        st.eff_rank = (float)(cv / (1.0 + cv));
    }

    std::vector<float> sorted(act_data, act_data + dim);
    std::sort(sorted.begin(), sorted.end());
    int64_t idx = (int64_t)(0.99 * (dim - 1));
    st.p99 = sorted[idx];
}

static int modality_from_suffix(const std::string & s) {
    if (s == "text")  return TS_MODALITY_TEXT;
    if (s == "image") return TS_MODALITY_IMAGE;
    if (s == "audio") return TS_MODALITY_AUDIO;
    return -1;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

int ts_mm_imatrix_load(const char * path, ts_mm_imatrix * out, std::string * err_msg) {
    if (!out) {
        if (err_msg) *err_msg = "null output";
        return 1;
    }

    out->data.clear();
    out->source_path.clear();
    out->version = 2;

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

    bool found_modality = false;

    for (auto & e : entries) {
        std::string key = e.name;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") {
            key = key.substr(0, key.size() - 4);
        }

        // split "<tensor>.<modality>" on the last dot
        size_t dot = key.rfind('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 >= key.size()) {
            continue;
        }

        int mod = modality_from_suffix(key.substr(dot + 1));
        if (mod < 0) {
            continue; // not a modality array (e.g. v2 rollup)
        }

        std::vector<float> vals;
        if (!npy_to_floats(e.data, e.size, vals)) {
            continue;
        }

        std::string tensor = key.substr(0, dot);

        // map operator[] value-initializes the entry (zeroed aggregate)
        ts_mm_imatrix_entry & en = out->data[tensor];
        en.has_modality[mod] = true;
        en.counts[mod] = (int64_t)vals.size();

        double s2 = 0.0;
        for (float v : vals) {
            s2 += (double)v * (double)v;
        }
        en.in_sum2[mod] = (float)s2;

        mm_regime(vals.data(), (int64_t)vals.size(), en.stats[mod]);

        // retain the raw per-channel array for use as AWQ act_scales
        en.act[mod] = std::move(vals);

        found_modality = true;
    }

    if (!found_modality) {
        out->data.clear();
        if (err_msg) *err_msg = "no modality_breakdown entries found";
        return 1;
    }

    out->source_path = path;
    out->version = 3;

    return 0;
}

const ts_mm_imatrix_entry * ts_mm_imatrix_entry_get(
    const ts_mm_imatrix * mm, const char * tensor_name) {
    if (!mm || !tensor_name) {
        return nullptr;
    }

    std::string name(tensor_name);
    if (name.size() > 7 && name.substr(name.size() - 7) == ".weight") {
        name = name.substr(0, name.size() - 7);
    }

    auto it = mm->data.find(name);
    if (it == mm->data.end()) {
        return nullptr;
    }

    return &it->second;
}

const ts_imatrix_regime_stats * ts_mm_imatrix_modality_stats(
    const ts_mm_imatrix * mm, const char * tensor_name, ts_modality mod) {
    int m = (int)mod;
    if (m < 0 || m >= TS_MODALITY_COUNT) {
        return nullptr;
    }

    const ts_mm_imatrix_entry * en = ts_mm_imatrix_entry_get(mm, tensor_name);
    if (!en || !en->has_modality[m]) {
        return nullptr;
    }

    return &en->stats[m];
}

const float * ts_mm_imatrix_act_scales(
    const ts_mm_imatrix * mm, const char * tensor_name, ts_modality mod, int64_t * dim) {
    int m = (int)mod;
    if (m < 0 || m >= TS_MODALITY_COUNT) {
        return nullptr;
    }

    const ts_mm_imatrix_entry * en = ts_mm_imatrix_entry_get(mm, tensor_name);
    if (!en || !en->has_modality[m] || en->act[m].empty()) {
        return nullptr;
    }

    if (dim) {
        *dim = (int64_t)en->act[m].size();
    }
    return en->act[m].data();
}

float ts_mm_imatrix_joint_kurtosis(const ts_mm_imatrix_entry * entry) {
    if (!entry) {
        return 0.0f;
    }

    double num = 0.0;
    double den = 0.0;
    for (int m = 0; m < TS_MODALITY_COUNT; ++m) {
        if (!entry->has_modality[m]) {
            continue;
        }
        double w = (double)entry->counts[m];
        num += w * (double)entry->stats[m].kurtosis;
        den += w;
    }

    if (den <= 0.0) {
        return 0.0f;
    }

    return (float)(num / den);
}

int ts_mm_imatrix_missing_mask(const ts_mm_imatrix_entry * entry) {
    if (!entry) {
        return (1 << TS_MODALITY_COUNT) - 1;
    }

    int mask = 0;
    for (int m = 0; m < TS_MODALITY_COUNT; ++m) {
        if (!entry->has_modality[m]) {
            mask |= (1 << m);
        }
    }
    return mask;
}
