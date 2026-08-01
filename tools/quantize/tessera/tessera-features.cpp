#include "tessera-features.h"

#include <cstdio>
#include <cstring>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static const char * TS_FEATURES_SCHEMA = "llama.tessera.features.v1";

static const char * ts_features_dtype_str(ts_features_dtype d) {
    switch (d) {
        case TS_FEATURES_F16: return "f16";
        case TS_FEATURES_F32:
        default:              return "f32";
    }
}

static bool ts_features_dtype_from_str(const std::string & s, ts_features_dtype & out) {
    if (s == "f32") { out = TS_FEATURES_F32; return true; }
    if (s == "f16") { out = TS_FEATURES_F16; return true; }
    return false;
}

bool ts_features_writer::open(const std::string & prefix_in,
                              int32_t n_embd_in,
                              const std::vector<int32_t> & layer_order,
                              ts_features_dtype dtype_in) {
    if (fp_bin != nullptr) {
        return false; // already open
    }
    if (n_embd_in <= 0 || layer_order.empty()) {
        return false;
    }
    if (dtype_in != TS_FEATURES_F32) {
        // F16/Q8 are follow-ups; the header format already reserves the field.
        return false;
    }

    header = ts_features_header{};
    header.n_tokens     = 0;
    header.n_embd       = n_embd_in;
    header.n_layers     = (int32_t) layer_order.size();
    header.target_layers = layer_order;
    header.dtype        = dtype_in;

    prefix    = prefix_in;
    n_written = 0;

    const std::string bin_path = prefix + ".bin";
    fp_bin = std::fopen(bin_path.c_str(), "wb");
    if (fp_bin == nullptr) {
        return false;
    }
    return true;
}

bool ts_features_writer::append_token(const float * fused) {
    if (fp_bin == nullptr || fused == nullptr) {
        return false;
    }
    const size_t row = (size_t) header.row_floats();
    if (std::fwrite(fused, sizeof(float), row, fp_bin) != row) {
        return false;
    }
    ++n_written;
    return true;
}

bool ts_features_writer::append_token_layers(const float * const * layers) {
    if (fp_bin == nullptr || layers == nullptr) {
        return false;
    }
    const size_t n_embd = (size_t) header.n_embd;
    for (int32_t i = 0; i < header.n_layers; ++i) {
        if (layers[i] == nullptr) {
            return false;
        }
        if (std::fwrite(layers[i], sizeof(float), n_embd, fp_bin) != n_embd) {
            return false;
        }
    }
    ++n_written;
    return true;
}

bool ts_features_writer::close() {
    if (fp_bin == nullptr) {
        return false;
    }

    const bool blob_ok = (std::fclose(fp_bin) == 0);
    fp_bin = nullptr;
    if (!blob_ok) {
        return false;
    }

    header.n_tokens = (int32_t) n_written;

    // blob filename is stored relative to the prefix so the pair is movable.
    const std::string bin_name    = prefix + ".bin";
    const size_t      slash       = bin_name.find_last_of("/\\");
    const std::string blob_rel    = (slash == std::string::npos)
                                        ? bin_name
                                        : bin_name.substr(slash + 1);

    json j;
    j["schema_version"]  = TS_FEATURES_SCHEMA;
    j["n_tokens"]        = header.n_tokens;
    j["n_embd"]          = header.n_embd;
    j["n_layers"]        = header.n_layers;
    j["target_layers"]   = header.target_layers;
    j["dtype"]           = ts_features_dtype_str(header.dtype);
    j["blob"]            = blob_rel;
    j["bytes_per_float"] = header.bytes_per_float();
    j["row_floats"]      = header.row_floats();
    j["chunk_tokens"]    = header.chunk_tokens;
    j["warmup"]          = header.warmup;
    j["stride"]          = header.stride;

    const std::string json_path = prefix + ".json";
    std::FILE * fp = std::fopen(json_path.c_str(), "w");
    if (fp == nullptr) {
        return false;
    }
    const std::string dumped = j.dump(2);
    const bool ok = (std::fwrite(dumped.data(), 1, dumped.size(), fp) == dumped.size());
    std::fclose(fp);
    return ok;
}

bool ts_features_read_header(const std::string & prefix, ts_features_header & out) {
    const std::string json_path = prefix + ".json";
    std::FILE * fp = std::fopen(json_path.c_str(), "rb");
    if (fp == nullptr) {
        return false;
    }
    std::string content;
    {
        std::fseek(fp, 0, SEEK_END);
        const long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        if (sz < 0) {
            std::fclose(fp);
            return false;
        }
        content.resize((size_t) sz);
        const size_t rd = content.empty() ? 0 : std::fread(&content[0], 1, content.size(), fp);
        std::fclose(fp);
        if (rd != content.size()) {
            return false;
        }
    }

    json j;
    try {
        j = json::parse(content);
    } catch (const std::exception &) {
        return false;
    }

    if (j.value("schema_version", std::string()) != TS_FEATURES_SCHEMA) {
        return false;
    }

    ts_features_header h;
    h.n_tokens = j.value("n_tokens", -1);
    h.n_embd   = j.value("n_embd", -1);
    h.n_layers = j.value("n_layers", -1);

    if (h.n_tokens < 0 || h.n_embd <= 0 || h.n_layers <= 0) {
        return false;
    }

    if (!j.contains("target_layers") || !j["target_layers"].is_array()) {
        return false;
    }
    for (const auto & v : j["target_layers"]) {
        if (!v.is_number_integer()) {
            return false;
        }
        h.target_layers.push_back(v.get<int32_t>());
    }
    if ((int32_t) h.target_layers.size() != h.n_layers) {
        return false;
    }

    ts_features_dtype dtype;
    if (!ts_features_dtype_from_str(j.value("dtype", std::string("f32")), dtype)) {
        return false;
    }
    h.dtype = dtype;

    // cross-check the derived layout fields if present.
    if (j.contains("row_floats") && j["row_floats"].get<int32_t>() != h.row_floats()) {
        return false;
    }
    if (j.contains("bytes_per_float") && j["bytes_per_float"].get<int32_t>() != h.bytes_per_float()) {
        return false;
    }

    // window layout is optional (absent in older files -> contiguous blob).
    h.chunk_tokens = j.value("chunk_tokens", 0);
    h.warmup       = j.value("warmup", 0);
    h.stride       = j.value("stride", 0);   // absent in legacy files -> 0
    if (h.chunk_tokens < 0 || h.warmup < 0 || h.stride < 0) {
        return false;
    }
    if (h.chunk_tokens > 0 && h.warmup >= h.chunk_tokens) {
        return false; // would leave no emitted rows per window
    }
    if (h.chunk_tokens > 0 && h.stride > 0) {
        // stride must tile the emitted rows without gaps or double-emission:
        // rows_per_window <= stride <= chunk_tokens.
        const int32_t per = h.chunk_tokens - h.warmup;
        if (h.stride < per || h.stride > h.chunk_tokens) {
            return false;
        }
    }

    out = h;
    return true;
}

int64_t ts_features_row_to_token(const ts_features_header & h, int64_t row) {
    if (row < 0 || row >= h.n_tokens) {
        return -1;
    }
    if (h.chunk_tokens == 0) {
        // no window layout: identity only when nothing was skipped.
        return h.warmup == 0 ? row : -1;
    }
    const int64_t per = h.rows_per_chunk();
    if (per <= 0) {
        return -1;
    }
    const int64_t stride = h.effective_stride();
    if (stride < per || stride > h.chunk_tokens) {
        return -1;   // would double-emit or skip tokens
    }
    const int64_t chunk  = row / per;
    const int64_t offset = row % per;
    // overlap mode (stride == per) collapses this to warmup + row (contiguous);
    // legacy mode (stride == chunk_tokens) has a warmup gap per window.
    return chunk * stride + h.warmup + offset;
}
