// GGUF binary parser adapted from the huggingface/gguf package.
// Reference: https://github.com/huggingface/huggingface.js

#include "gguf-model-data.h"

#include "common.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "http.h"
#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

// Equivalent of RangeView
struct gguf_buf_reader {
    const char * data;
    size_t       size;
    size_t       pos;

    gguf_buf_reader(const std::vector<char> & buf) : data(buf.data()), size(buf.size()), pos(0) {}

    bool has_n_bytes(size_t n) const {
        return pos + n <= size;
    }

    template <typename T>
    bool read_val(T & out) {
        if (!has_n_bytes(sizeof(T))) {
            return false;
        }
        memcpy(&out, data + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_str(std::string & out) {
        uint64_t len;
        if (!read_val(len)) {
            return false;
        }
        if (!has_n_bytes((size_t)len)) {
            return false;
        }
        out.assign(data + pos, (size_t)len);
        pos += (size_t)len;
        return true;
    }

    bool skip(size_t n) {
        if (!has_n_bytes(n)) {
            return false;
        }
        pos += n;
        return true;
    }
};

static size_t gguf_val_type_size(int32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0; // string/array handled separately
    }
}

// Equivalent of readMetadataValue(), skips unused values rather than storing
static bool gguf_skip_value(gguf_buf_reader & r, int32_t vtype) {
    if (vtype == GGUF_TYPE_STRING) {
        std::string tmp;
        return r.read_str(tmp);
    }
    if (vtype == GGUF_TYPE_ARRAY) {
        int32_t elem_type;
        uint64_t count;
        if (!r.read_val(elem_type)) {
            return false;
        }
        if (!r.read_val(count)) {
            return false;
        }
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < count; i++) {
                std::string tmp;
                if (!r.read_str(tmp)) {
                    return false;
                }
            }
            return true;
        }
        if (elem_type == GGUF_TYPE_ARRAY) {
            // nested arrays - recurse
            for (uint64_t i = 0; i < count; i++) {
                if (!gguf_skip_value(r, GGUF_TYPE_ARRAY)) {
                    return false;
                }
            }
            return true;
        }
        size_t elem_sz = gguf_val_type_size(elem_type);
        if (elem_sz == 0) {
            return false;
        }
        return r.skip((size_t)count * elem_sz);
    }
    size_t sz = gguf_val_type_size(vtype);
    if (sz == 0) {
        return false;
    }
    return r.skip(sz);
}

static bool gguf_read_uint32_val(gguf_buf_reader & r, int32_t vtype, uint32_t & out) {
    if (vtype == GGUF_TYPE_UINT8) {
        uint8_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = v;
        return true;
    }
    if (vtype == GGUF_TYPE_INT8) {
        int8_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = (uint32_t)v;
        return true;
    }
    if (vtype == GGUF_TYPE_UINT16) {
        uint16_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = v;
        return true;
    }
    if (vtype == GGUF_TYPE_INT16) {
        int16_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = (uint32_t)v;
        return true;
    }
    if (vtype == GGUF_TYPE_UINT32) {
        uint32_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = v;
        return true;
    }
    if (vtype == GGUF_TYPE_INT32) {
        int32_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = (uint32_t)v;
        return true;
    }
    if (vtype == GGUF_TYPE_UINT64) {
        uint64_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = (uint32_t)v;
        return true;
    }
    if (vtype == GGUF_TYPE_INT64) {
        int64_t v;
        if (!r.read_val(v)) {
            return false;
        }
        out = (uint32_t)v;
        return true;
    }
    return false;
}

// Follows the same header -> KV -> tensor parsing sequence as gguf() huggingface/gguf
static std::optional<gguf_remote_model> gguf_parse_meta(const std::vector<char> & buf) {
    gguf_buf_reader r(buf);

    // Header: magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes minimum
    uint32_t magic_raw;
    if (!r.read_val(magic_raw)) {
        return std::nullopt;
    }
    if (memcmp(&magic_raw, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf_parse_meta: invalid magic\n");
        return std::nullopt;
    }

    uint32_t version;
    if (!r.read_val(version)) {
        return std::nullopt;
    }
    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf_parse_meta: unsupported version %u\n", version);
        return std::nullopt;
    }

    int64_t tensor_count_raw;
    int64_t kv_count_raw;
    if (!r.read_val(tensor_count_raw)) {
        return std::nullopt;
    }
    if (!r.read_val(kv_count_raw)) {
        return std::nullopt;
    }

    uint64_t tensor_count = (uint64_t)tensor_count_raw;
    uint64_t kv_count     = (uint64_t)kv_count_raw;

    gguf_remote_model model;

    std::string arch_prefix;

    // Parse KV pairs
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key;
        if (!r.read_str(key)) {
            return std::nullopt;
        }

        int32_t vtype;
        if (!r.read_val(vtype)) {
            return std::nullopt;
        }

        if (key == "general.architecture" && vtype == GGUF_TYPE_STRING) {
            if (!r.read_str(model.architecture)) {
                return std::nullopt;
            }
            arch_prefix = model.architecture + ".";
            continue;
        }

        // Extract split.count for proper handling of split files
        if (key == "split.count") {
            uint32_t v;
            if (!gguf_read_uint32_val(r, vtype, v)) {
                return std::nullopt;
            }
            model.n_split = (uint16_t)v;
            continue;
        }

        // Extract split.tensors.count so we can verify we have all tensors
        if (key == "split.tensors.count") {
            uint32_t v;
            if (!gguf_read_uint32_val(r, vtype, v)) {
                return std::nullopt;
            }
            model.n_split_tensors = v;
            continue;
        }

        if (!arch_prefix.empty()) {
            uint32_t * target = nullptr;

            if      (key == arch_prefix + "embedding_length")         { target = &model.n_embd; }
            else if (key == arch_prefix + "feed_forward_length")      { target = &model.n_ff; }
            else if (key == arch_prefix + "block_count")              { target = &model.n_layer; }
            else if (key == arch_prefix + "attention.head_count")     { target = &model.n_head; }
            else if (key == arch_prefix + "attention.head_count_kv")  { target = &model.n_head_kv; }
            else if (key == arch_prefix + "expert_count")             { target = &model.n_expert; }
            else if (key == arch_prefix + "attention.key_length")     { target = &model.n_embd_head_k; }
            else if (key == arch_prefix + "attention.value_length")   { target = &model.n_embd_head_v; }

            if (target) {
                if (!gguf_read_uint32_val(r, vtype, *target)) {
                    return std::nullopt;
                }
                continue;
            }
        }

        if (!gguf_skip_value(r, vtype)) {
            return std::nullopt;
        }
    }

    // Parse tensor info entries
    model.tensors.reserve((size_t)tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        gguf_remote_tensor t;

        if (!r.read_str(t.name)) {
            return std::nullopt;
        }
        if (!r.read_val(t.n_dims)) {
            return std::nullopt;
        }

        if (t.n_dims > 4) {
            fprintf(stderr, "gguf_parse_meta: tensor '%s' has %u dims (max 4)\n", t.name.c_str(), t.n_dims);
            return std::nullopt;
        }

        for (uint32_t d = 0; d < t.n_dims; d++) {
            if (!r.read_val(t.ne[d])) {
                return std::nullopt;
            }
        }

        int32_t type_raw;
        if (!r.read_val(type_raw)) {
            return std::nullopt;
        }
        t.type = (ggml_type)type_raw;

        uint64_t offset;
        if (!r.read_val(offset)) {
            return std::nullopt;
        }

        // Infer n_vocab from token_embd.weight
        if (t.name == "token_embd.weight") {
            model.n_vocab = (uint32_t)t.ne[1];
        }

        model.tensors.push_back(std::move(t));
    }

    return model;
}

// cache handling for local download
static std::string get_default_cache_dir() {
    return fs_get_cache_directory() + "gguf-headers/";
}

static std::string sanitize_for_path(const std::string & s) {
    std::string out = s;
    for (char & c : out) {
        if (c == '/' || c == '\\' || c == ':') {
            c = '_';
        }
    }
    return out;
}

static bool read_file(const std::string & path, std::vector<char> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return false;
    }
    auto sz = f.tellg();
    if (sz <= 0) {
        return false;
    }
    out.resize((size_t)sz);
    f.seekg(0);
    f.read(out.data(), sz);
    return f.good();
}

static bool write_file(const std::string & path, const std::vector<char> & data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.good()) {
        return false;
    }
    f.write(data.data(), (std::streamsize)data.size());
    return f.good();
}

// HuggingFace file auto-detection and HTTP download
static std::pair<long, std::vector<char>> gguf_http_get(
        const std::string & url,
        const httplib::Headers & headers = {},
        int timeout_sec = 60) {
    try {
        auto [cli, parts] = common_http_client(url);

        if (timeout_sec > 0) {
            cli.set_read_timeout(timeout_sec, 0);
            cli.set_write_timeout(timeout_sec, 0);
        }
        cli.set_connection_timeout(30, 0);

        std::vector<char> body;
        auto res = cli.Get(parts.path, headers,
            [&](const char * data, size_t len) {
                body.insert(body.end(), data, data + len);
                return true;
            }, nullptr);

        if (!res) {
            fprintf(stderr, "gguf_fetch: HTTP request failed for %s (error %d)\n",
                    url.c_str(), (int)res.error());
            return {-1, {}};
        }
        return {res->status, std::move(body)};
    } catch (const std::exception & e) {
        fprintf(stderr, "gguf_fetch: HTTP error: %s\n", e.what());
        return {-1, {}};
    }
}

// Find the filename for given repo/quant.
// For split models, returns the first shard (the one containing "00001-of-")
// split_prefix is set to the portion before "-00001-of-XXXXX.gguf" when a split file is found
static std::string detect_gguf_filename(const std::string & repo, const std::string & quant,
                                        std::string & split_prefix) {
    split_prefix.clear();
    std::string api_url = "https://huggingface.co/api/models/" + repo;

    auto [code, body] = gguf_http_get(api_url, {}, 30);
    if (code != 200 || body.empty()) {
        fprintf(stderr, "gguf_fetch: failed to query HF API for %s (HTTP %ld)\n", repo.c_str(), code);
        return "";
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body.begin(), body.end());
    } catch (...) {
        fprintf(stderr, "gguf_fetch: failed to parse HF API response\n");
        return "";
    }

    if (!j.contains("siblings") || !j["siblings"].is_array()) {
        fprintf(stderr, "gguf_fetch: unexpected HF API response format\n");
        return "";
    }

    std::vector<std::string> matches;
    std::string quant_upper = quant;
    for (char & c : quant_upper) { c = (char)toupper(c); }

    for (const auto & sibling : j["siblings"]) {
        if (!sibling.contains("rfilename")) { continue; }
        std::string fname = sibling["rfilename"].get<std::string>();
        if (fname.size() < 5 || fname.substr(fname.size() - 5) != ".gguf") {
            continue;
        }

        std::string fname_upper = fname;
        for (char & c : fname_upper) { c = (char)toupper(c); }
        if (fname_upper.find(quant_upper) != std::string::npos) {
            matches.push_back(fname);
        }
    }

    if (matches.empty()) {
        fprintf(stderr, "gguf_fetch: no .gguf files matching '%s' in %s\n", quant.c_str(), repo.c_str());
        return "";
    }

    std::sort(matches.begin(), matches.end());

    // Prefer non-split, non-supplementary file
    for (const auto & m : matches) {
        if (m.find("-of-") == std::string::npos && m.find("mmproj") == std::string::npos) {
            return m;
        }
    }

    // Return the first shard (00001-of-) and extract the prefix
    for (const auto & m : matches) {
        auto pos = m.find("-00001-of-");
        if (pos != std::string::npos) {
            split_prefix = m.substr(0, pos);
            return m;
        }
    }

    return matches[0];
}

static std::optional<gguf_remote_model> fetch_and_parse(
        const std::string & repo,
        const std::string & filename,
        const std::string & cache_path) {
    std::string url = "https://huggingface.co/" + repo + "/resolve/main/" + filename;

    // Progressive download inspired by RangeView.fetchChunk()
    // Start at 2MB, double each time, cap at 64MB
    size_t chunk_size = 2 * 1024 * 1024;
    const size_t max_chunk = 64 * 1024 * 1024;

    while (chunk_size <= max_chunk) {
        fprintf(stderr, "gguf_fetch: downloading %zu bytes from %s\n", chunk_size, filename.c_str());

        char range_buf[64];
        snprintf(range_buf, sizeof(range_buf), "bytes=0-%zu", chunk_size - 1);
        httplib::Headers headers = {{"Range", range_buf}};

        auto [code, body] = gguf_http_get(url, headers, 120);
        if (code != 200 && code != 206) {
            fprintf(stderr, "gguf_fetch: HTTP %ld fetching %s\n", code, url.c_str());
            return std::nullopt;
        }

        if (body.empty()) {
            fprintf(stderr, "gguf_fetch: empty response\n");
            return std::nullopt;
        }

        auto result = gguf_parse_meta(body);
        if (result.has_value()) {
            write_file(cache_path, body);
            return result;
        }

        if (code == 200) {
            fprintf(stderr, "gguf_fetch: server returned full response but metadata parse failed\n");
            return std::nullopt;
        }

        // Parse failed, try larger chunk
        chunk_size *= 2;
    }

    fprintf(stderr, "gguf_fetch: metadata exceeds 64MB, giving up\n");
    return std::nullopt;
}

// Try cache first, then fetch and parse a single GGUF shard.
static std::optional<gguf_remote_model> fetch_or_cached(
        const std::string & repo,
        const std::string & filename,
        const std::string & cdir,
        const std::string & repo_part) {
    std::string fname_part = sanitize_for_path(filename);
    std::string cache_path = cdir + "/" + repo_part + "--" + fname_part + ".partial";

    {
        std::vector<char> cached;
        if (std::filesystem::exists(cache_path) && read_file(cache_path, cached)) {
            auto result = gguf_parse_meta(cached);
            if (result.has_value()) {
                fprintf(stderr, "gguf_fetch: loaded from cache: %s\n", cache_path.c_str());
                return result;
            }
        }
    }

    fs_create_directory_with_parents(cdir);
    return fetch_and_parse(repo, filename, cache_path);
}

std::optional<gguf_remote_model> gguf_fetch_model_meta(
        const std::string & repo,
        const std::string & quant,
        const std::string & cache_dir) {
    std::string cdir = cache_dir.empty() ? get_default_cache_dir() : cache_dir;
    std::string repo_part = sanitize_for_path(repo);

    std::string split_prefix;
    std::string filename = detect_gguf_filename(repo, quant, split_prefix);
    if (filename.empty()) {
        return std::nullopt;
    }

    auto model_opt = fetch_or_cached(repo, filename, cdir, repo_part);
    if (!model_opt.has_value()) {
        fprintf(stderr, "gguf_fetch: failed to fetch %s\n", filename.c_str());
        return std::nullopt;
    }

    auto & model = model_opt.value();

    // If the model is split across multiple files we need to fetch the remaining shards metadata
    if (model.n_split > 1) {
        if (split_prefix.empty()) {
            fprintf(stderr, "gguf_fetch: model reports %u splits but filename has no split pattern\n", model.n_split);
            return std::nullopt;
        }

        fprintf(stderr, "gguf_fetch: split model with %u shards, fetching remaining %u...\n",
                model.n_split, model.n_split - 1);

        for (int i = 2; i <= model.n_split; i++) {
            char num_buf[6], total_buf[6];
            snprintf(num_buf,   sizeof(num_buf),   "%05d", i);
            snprintf(total_buf, sizeof(total_buf), "%05d", (int)model.n_split);
            std::string shard_name = split_prefix + "-" + num_buf + "-of-" + total_buf + ".gguf";

            auto shard = fetch_or_cached(repo, shard_name, cdir, repo_part);
            if (!shard.has_value()) {
                fprintf(stderr, "gguf_fetch: failed to fetch shard %d: %s\n", i, shard_name.c_str());
                return std::nullopt;
            }

            model.tensors.insert(model.tensors.end(),
                std::make_move_iterator(shard->tensors.begin()),
                std::make_move_iterator(shard->tensors.end()));
        }

        if (model.n_split_tensors > 0 && model.tensors.size() != model.n_split_tensors) {
            fprintf(stderr, "gguf_fetch: WARNING: expected %u tensors from split.tensors.count, got %zu\n",
                    model.n_split_tensors, model.tensors.size());
        }
    }

    return model_opt;
}
