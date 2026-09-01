// SPDX-License-Identifier: MIT
#include "spec_sidecar_assets.h"

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "hash/hash.h"
#include "../src/spec_sidecar/qwen35_draft_vocab_bitmap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using nlohmann::ordered_json;

static constexpr const char * AUTO_ASSET_SCHEMA = "spec-sidecar-native-assets-v1";
static constexpr const char * BUILTIN_IDS_SHA256 =
        "8490b42e79b3c339b2460e57ac9193ab51b8814812c6e058555c2f06ef55dfd4";
static constexpr int32_t QWEN35_VOCAB = 248320;
static constexpr int32_t QWEN35_DRAFT_VOCAB = 40960;

struct tensor_spec {
    std::string output_name;
    std::string source_name;
    ggml_type output_type = GGML_TYPE_COUNT;
    std::vector<int64_t> output_shape;
    bool sliced_head = false;
};

struct artifact_entry {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

static std::string normalized_absolute(const fs::path & path) {
    std::error_code ec;
    fs::path absolute = fs::absolute(path, ec);
    if (ec) {
        return {};
    }
    fs::path normalized = fs::weakly_canonical(absolute, ec);
    return (ec ? absolute.lexically_normal() : normalized).string();
}

static bool is_regular_file(const fs::path & path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) && !ec;
}

static bool path_is_directory(const fs::path & path) {
    std::error_code ec;
    return fs::is_directory(path, ec) && !ec;
}

static bool read_file(const fs::path & path, std::vector<uint8_t> & data, std::string & error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "cannot open " + path.string();
        return false;
    }
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max()) {
        error = "invalid file size for " + path.string();
        return false;
    }
    data.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!data.empty() && !input.read(reinterpret_cast<char *>(data.data()), size)) {
        error = "short read from " + path.string();
        return false;
    }
    return true;
}

static int hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static std::vector<uint8_t> ids_to_le_bytes(const std::vector<int32_t> & ids) {
    std::vector<uint8_t> bytes(ids.size() * sizeof(int32_t));
    for (size_t i = 0; i < ids.size(); ++i) {
        const uint32_t value = static_cast<uint32_t>(ids[i]);
        bytes[4 * i + 0] = static_cast<uint8_t>(value >> 0);
        bytes[4 * i + 1] = static_cast<uint8_t>(value >> 8);
        bytes[4 * i + 2] = static_cast<uint8_t>(value >> 16);
        bytes[4 * i + 3] = static_cast<uint8_t>(value >> 24);
    }
    return bytes;
}

static bool validate_ids(std::vector<int32_t> & ids, std::string & error) {
    std::sort(ids.begin(), ids.end());
    const auto last = std::unique(ids.begin(), ids.end());
    ids.erase(last, ids.end());
    if (ids.size() != QWEN35_DRAFT_VOCAB) {
        error = "sidecar draft vocabulary must contain exactly 40,960 unique IDs";
        return false;
    }
    if (ids.front() < 0 || ids.back() >= QWEN35_VOCAB) {
        error = "sidecar draft vocabulary contains an out-of-range ID";
        return false;
    }
    return true;
}

class gguf_file {
public:
    explicit gguf_file(const std::string & path, std::string & error) : path_(path), input_(path, std::ios::binary) {
        if (!input_) {
            error = "cannot open GGUF: " + path;
            return;
        }
        const gguf_init_params params = {
            /* .no_alloc = */ true,
            /* .ctx      = */ &meta_,
        };
        context_ = gguf_init_from_file(path.c_str(), params);
        if (context_ == nullptr || meta_ == nullptr) {
            error = "cannot parse GGUF: " + path;
            if (context_ != nullptr) {
                gguf_free(context_);
                context_ = nullptr;
            }
            if (meta_ != nullptr) {
                ggml_free(meta_);
                meta_ = nullptr;
            }
        }
    }

    ~gguf_file() {
        if (context_ != nullptr) gguf_free(context_);
        if (meta_ != nullptr) ggml_free(meta_);
    }

    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;

    bool valid() const { return context_ != nullptr && meta_ != nullptr; }
    const std::string & path() const { return path_; }
    int64_t tensor_count() const { return valid() ? gguf_get_n_tensors(context_) : 0; }

    const ggml_tensor * tensor(const std::string & name) const {
        return valid() ? ggml_get_tensor(meta_, name.c_str()) : nullptr;
    }

    bool tensor_data_offset(const std::string & name, uint64_t & offset, std::string & error) const {
        const int64_t id = valid() ? gguf_find_tensor(context_, name.c_str()) : -1;
        if (id < 0) {
            error = "missing required tensor: " + name;
            return false;
        }
        const uint64_t base = static_cast<uint64_t>(gguf_get_data_offset(context_));
        const uint64_t relative = static_cast<uint64_t>(gguf_get_tensor_offset(context_, id));
        if (relative > std::numeric_limits<uint64_t>::max() - base) {
            error = "tensor offset overflow: " + name;
            return false;
        }
        offset = base + relative;
        return true;
    }

    bool read_at(uint64_t offset, void * destination, size_t size, std::string & error) {
        if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            error = "GGUF offset is too large: " + path_;
            return false;
        }
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!input_ || (size > 0 && !input_.read(static_cast<char *>(destination), static_cast<std::streamsize>(size)))) {
            error = "short tensor read from " + path_;
            return false;
        }
        return true;
    }

    bool metadata_ids(std::vector<int32_t> & ids) const {
        static const char * keys[] = {
            "qwen35.nextn.draft_vocab_ids",
            "qwen35moe.nextn.draft_vocab_ids",
            "qwen3.nextn.draft_vocab_ids",
        };
        for (const char * key : keys) {
            const int64_t id = valid() ? gguf_find_key(context_, key) : -1;
            if (id < 0 || gguf_get_kv_type(context_, id) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(context_, id) != GGUF_TYPE_INT32 ||
                    gguf_get_arr_n(context_, id) != QWEN35_DRAFT_VOCAB) {
                continue;
            }
            const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(context_, id));
            ids.assign(values, values + QWEN35_DRAFT_VOCAB);
            return true;
        }
        return false;
    }

private:
    std::string path_;
    std::ifstream input_;
    ggml_context * meta_ = nullptr;
    gguf_context * context_ = nullptr;
};

static bool load_external_ids(const fs::path & path, std::vector<int32_t> & ids, std::string & error) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, bytes, error)) {
        return false;
    }
    if (bytes.size() != QWEN35_DRAFT_VOCAB * sizeof(int32_t)) {
        error = "sidecar ID file must contain exactly 163,840 bytes: " + path.string();
        return false;
    }
    ids.resize(QWEN35_DRAFT_VOCAB);
    for (size_t i = 0; i < ids.size(); ++i) {
        const uint32_t value =
                static_cast<uint32_t>(bytes[4 * i + 0]) << 0 |
                static_cast<uint32_t>(bytes[4 * i + 1]) << 8 |
                static_cast<uint32_t>(bytes[4 * i + 2]) << 16 |
                static_cast<uint32_t>(bytes[4 * i + 3]) << 24;
        ids[i] = static_cast<int32_t>(value);
    }
    return validate_ids(ids, error);
}

static bool resolve_ids(const common_spec_sidecar_profile & profile, gguf_file & target,
        std::vector<int32_t> & ids, std::string & source, std::string & error) {
    const char * external = profile.ids_env != nullptr ? std::getenv(profile.ids_env) : nullptr;
    if (external != nullptr && external[0] != '\0') {
        const std::string path = normalized_absolute(external);
        if (path.empty() || !load_external_ids(path, ids, error)) {
            return false;
        }
        source = path;
        return true;
    }
    if (target.metadata_ids(ids)) {
        if (!validate_ids(ids, error)) {
            return false;
        }
        source = target.path() + "#draft_vocab_ids";
        return true;
    }
    if (!common_spec_sidecar_builtin_draft_vocab_ids(ids, error)) {
        return false;
    }
    source = "builtin:syv-ai/qwen38-27b-rtx3090@c9547241";
    return true;
}

static std::vector<int64_t> tensor_shape(const ggml_tensor * tensor) {
    std::vector<int64_t> result;
    if (tensor == nullptr) return result;
    const int n_dims = ggml_n_dims(tensor);
    result.reserve(n_dims);
    for (int i = 0; i < n_dims; ++i) result.push_back(tensor->ne[i]);
    return result;
}

static bool check_shape(const ggml_tensor * tensor, const std::vector<int64_t> & expected,
        const std::string & name, std::string & error) {
    if (tensor == nullptr) {
        error = "missing required tensor: " + name;
        return false;
    }
    if (tensor_shape(tensor) != expected) {
        std::ostringstream message;
        message << "unexpected shape for " << name;
        error = message.str();
        return false;
    }
    return true;
}

static tensor_spec make_spec(const char * name, ggml_type type, std::initializer_list<int64_t> shape) {
    return {name, name, type, std::vector<int64_t>(shape), false};
}

static tensor_spec make_head_spec(const char * output_name, int64_t width) {
    return {output_name, "output.weight", GGML_TYPE_Q4_0,
            {width, QWEN35_DRAFT_VOCAB}, true};
}

static std::vector<tensor_spec> qwen35_mtp_specs() {
    return {
        make_spec("token_embd.weight",                            GGML_TYPE_Q4_0, {5120, 248320}),
        make_spec("blk.64.nextn.eh_proj.weight",                  GGML_TYPE_Q4_0, {10240, 5120}),
        make_spec("blk.64.attn_norm.weight",                      GGML_TYPE_F32,  {5120}),
        make_spec("blk.64.ffn_down.weight",                       GGML_TYPE_Q4_0, {17408, 5120}),
        make_spec("blk.64.ffn_gate.weight",                       GGML_TYPE_Q4_0, {5120, 17408}),
        make_spec("blk.64.ffn_up.weight",                         GGML_TYPE_Q4_0, {5120, 17408}),
        make_spec("blk.64.post_attention_norm.weight",            GGML_TYPE_F32,  {5120}),
        make_spec("blk.64.attn_k_norm.weight",                    GGML_TYPE_F32,  {256}),
        make_spec("blk.64.attn_k.weight",                         GGML_TYPE_Q4_0, {5120, 1024}),
        make_spec("blk.64.attn_output.weight",                    GGML_TYPE_Q4_0, {6144, 5120}),
        make_spec("blk.64.attn_q_norm.weight",                    GGML_TYPE_F32,  {256}),
        make_spec("blk.64.attn_q.weight",                         GGML_TYPE_Q4_0, {5120, 12288}),
        make_spec("blk.64.attn_v.weight",                         GGML_TYPE_Q4_0, {5120, 1024}),
        make_spec("blk.64.nextn.shared_head_norm.weight",         GGML_TYPE_F32,  {5120}),
        make_spec("blk.64.nextn.enorm.weight",                    GGML_TYPE_F32,  {5120}),
        make_spec("blk.64.nextn.hnorm.weight",                    GGML_TYPE_F32,  {5120}),
        make_head_spec("blk.64.nextn.shared_head_head.weight", 5120),
    };
}

static std::vector<tensor_spec> qwen35moe_mtp_specs() {
    return {
        make_head_spec("output.weight", 2048),
        make_spec("token_embd.weight",                         GGML_TYPE_Q4_0, {2048, 248320}),
        make_spec("blk.40.attn_k.weight",                      GGML_TYPE_Q4_0, {2048, 512}),
        make_spec("blk.40.attn_k_norm.weight",                 GGML_TYPE_F32,  {256}),
        make_spec("blk.40.attn_norm.weight",                   GGML_TYPE_F32,  {2048}),
        make_spec("blk.40.attn_output.weight",                 GGML_TYPE_Q4_0, {4096, 2048}),
        make_spec("blk.40.attn_q.weight",                      GGML_TYPE_Q4_0, {2048, 8192}),
        make_spec("blk.40.attn_q_norm.weight",                 GGML_TYPE_F32,  {256}),
        make_spec("blk.40.attn_v.weight",                      GGML_TYPE_Q4_0, {2048, 512}),
        make_spec("blk.40.ffn_down_exps.weight",               GGML_TYPE_Q4_0, {512, 2048, 256}),
        make_spec("blk.40.ffn_down_shexp.weight",              GGML_TYPE_Q4_0, {512, 2048}),
        make_spec("blk.40.ffn_gate_exps.weight",               GGML_TYPE_Q4_0, {2048, 512, 256}),
        make_spec("blk.40.ffn_gate_inp.weight",                GGML_TYPE_F32,  {2048, 256}),
        make_spec("blk.40.ffn_gate_inp_shexp.weight",          GGML_TYPE_F32,  {2048}),
        make_spec("blk.40.ffn_gate_shexp.weight",              GGML_TYPE_Q4_0, {2048, 512}),
        make_spec("blk.40.ffn_up_exps.weight",                 GGML_TYPE_Q4_0, {2048, 512, 256}),
        make_spec("blk.40.ffn_up_shexp.weight",                GGML_TYPE_Q4_0, {2048, 512}),
        make_spec("blk.40.nextn.eh_proj.weight",               GGML_TYPE_Q4_0, {4096, 2048}),
        make_spec("blk.40.nextn.enorm.weight",                 GGML_TYPE_F32,  {2048}),
        make_spec("blk.40.nextn.hnorm.weight",                 GGML_TYPE_F32,  {2048}),
        make_spec("blk.40.nextn.shared_head_norm.weight",      GGML_TYPE_F32,  {2048}),
        make_spec("blk.40.post_attention_norm.weight",         GGML_TYPE_F32,  {2048}),
    };
}

static std::vector<tensor_spec> dflash_specs() {
    std::vector<tensor_spec> specs = {
        make_spec("enc.output_norm.weight",       GGML_TYPE_F32,  {5120}),
        make_spec("fc.weight",                    GGML_TYPE_Q4_K, {25600, 5120}),
        make_spec("output_norm.weight",            GGML_TYPE_F32,  {5120}),
        make_spec("selector_hidden.weight",        GGML_TYPE_Q4_K, {5120, 256}),
        make_spec("selector_predecessor.weight",   GGML_TYPE_Q4_K, {256, 248320}),
        make_spec("selector_successor.weight",     GGML_TYPE_Q4_K, {256, 248320}),
    };
    for (int layer = 0; layer < 5; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const ggml_type wide_type = layer == 2 || layer == 4 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
        const auto add = [&](const char * suffix, ggml_type type, std::initializer_list<int64_t> shape) {
            tensor_spec spec;
            spec.output_name = prefix + suffix;
            spec.source_name = spec.output_name;
            spec.output_type = type;
            spec.output_shape.assign(shape.begin(), shape.end());
            specs.push_back(std::move(spec));
        };
        add("attn_conv_base",        GGML_TYPE_F32,  {5120, 2, 2});
        add("attn_conv_proj.weight", GGML_TYPE_Q4_K, {5120, 1280});
        add("attn_k.weight",         GGML_TYPE_Q4_K, {5120, 1024});
        add("attn_k_norm.weight",    GGML_TYPE_F32,  {128});
        add("attn_norm.weight",      GGML_TYPE_F32,  {5120});
        add("attn_output.weight",    GGML_TYPE_Q4_K, {4096, 5120});
        add("attn_q.weight",         GGML_TYPE_Q4_K, {5120, 4096});
        add("attn_q_norm.weight",    GGML_TYPE_F32,  {128});
        add("attn_v.weight",         wide_type,      {5120, 1024});
        add("ffn_conv_base",         GGML_TYPE_F32,  {5120, 2, 2});
        add("ffn_conv_proj.weight",  GGML_TYPE_Q4_K, {5120, 1280});
        add("ffn_down.weight",       wide_type,      {17408, 5120});
        add("ffn_gate.weight",       GGML_TYPE_Q4_K, {5120, 17408});
        add("ffn_norm.weight",       GGML_TYPE_F32,  {5120});
        add("ffn_up.weight",         GGML_TYPE_Q4_K, {5120, 17408});
    }
    return specs;
}

static bool validate_specs(gguf_file & source, const std::vector<tensor_spec> & specs,
        bool exact_types, bool exact_count, std::string & error) {
    if (exact_count && source.tensor_count() != static_cast<int64_t>(specs.size())) {
        error = "sidecar source has an unexpected tensor count";
        return false;
    }
    for (const auto & spec : specs) {
        const ggml_tensor * tensor = source.tensor(spec.source_name);
        std::vector<int64_t> source_shape = spec.output_shape;
        if (spec.sliced_head) {
            source_shape[1] = QWEN35_VOCAB;
        }
        if (!check_shape(tensor, source_shape, spec.source_name, error)) {
            return false;
        }
        if (exact_types && tensor->type != spec.output_type) {
            error = "unexpected tensor type for " + spec.source_name + ": expected " +
                    ggml_type_name(spec.output_type) + ", found " + ggml_type_name(tensor->type);
            return false;
        }
        if (!spec.sliced_head && spec.output_type != GGML_TYPE_F32 &&
                ggml_row_size(spec.output_type, tensor->ne[0]) == 0) {
            error = "unsupported output tensor layout: " + spec.output_name;
            return false;
        }
    }
    return true;
}

static bool dequantize_rows(ggml_type type, const void * source, float * output,
        int64_t elements, std::string & error) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(output, source, static_cast<size_t>(elements) * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(source), output, elements);
        return true;
    }
    if (type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(source), output, elements);
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->to_float == nullptr) {
        error = "cannot dequantize tensor type " + std::string(ggml_type_name(type));
        return false;
    }
    traits->to_float(source, output, elements);
    return true;
}

static bool encode_rows(ggml_type type, const float * source, int64_t rows, int64_t width,
        std::vector<uint8_t> & output, std::string & error) {
    if (type == GGML_TYPE_F32) {
        const size_t bytes = static_cast<size_t>(rows * width) * sizeof(float);
        output.resize(bytes);
        std::memcpy(output.data(), source, bytes);
        return true;
    }
    if (ggml_quantize_requires_imatrix(type)) {
        error = "automatic sidecar conversion cannot create " + std::string(ggml_type_name(type)) +
                " without an importance matrix";
        return false;
    }
    const size_t row_size = ggml_row_size(type, width);
    if (row_size == 0 || static_cast<uint64_t>(row_size) * rows > std::numeric_limits<size_t>::max()) {
        error = "invalid quantized row layout";
        return false;
    }
    output.resize(row_size * static_cast<size_t>(rows));
    const size_t written = ggml_quantize_chunk(type, source, output.data(), 0, rows, width, nullptr);
    if (written != output.size() || !ggml_validate_row_data(type, output.data(), output.size())) {
        error = "failed to quantize sidecar tensor as " + std::string(ggml_type_name(type));
        return false;
    }
    return true;
}

static bool write_direct_range(gguf_file & source, uint64_t offset, uint64_t size,
        std::ofstream & output, std::string & error) {
    static constexpr size_t CHUNK = 8 * 1024 * 1024;
    std::vector<uint8_t> buffer(static_cast<size_t>(std::min<uint64_t>(size, CHUNK)));
    uint64_t cursor = 0;
    while (cursor < size) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(buffer.size(), size - cursor));
        if (!source.read_at(offset + cursor, buffer.data(), count, error)) {
            return false;
        }
        output.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(count));
        if (!output) {
            error = "failed writing sidecar weights";
            return false;
        }
        cursor += count;
    }
    return true;
}

static bool write_converted_tensor(gguf_file & source, const ggml_tensor * tensor,
        uint64_t source_offset, ggml_type output_type, std::ofstream & output,
        uint64_t & bytes_written, std::string & error) {
    const int64_t width = tensor->ne[0];
    const int64_t elements = ggml_nelements(tensor);
    if (width <= 0 || elements <= 0 || elements % width != 0) {
        error = "invalid tensor geometry for conversion";
        return false;
    }
    const int64_t rows = elements / width;
    const size_t source_row_size = ggml_row_size(tensor->type, width);
    const size_t rows_per_chunk = static_cast<size_t>(std::max<int64_t>(1,
            std::min<int64_t>(256, (8 * 1024 * 1024) / (width * static_cast<int64_t>(sizeof(float))))));
    std::vector<uint8_t> input;
    std::vector<float> f32;
    std::vector<uint8_t> encoded;
    bytes_written = 0;
    for (int64_t row = 0; row < rows; row += static_cast<int64_t>(rows_per_chunk)) {
        const int64_t count = std::min<int64_t>(rows - row, rows_per_chunk);
        const size_t source_bytes = source_row_size * static_cast<size_t>(count);
        input.resize(source_bytes);
        if (!source.read_at(source_offset + static_cast<uint64_t>(row) * source_row_size,
                input.data(), source_bytes, error)) {
            return false;
        }
        f32.resize(static_cast<size_t>(count * width));
        if (!dequantize_rows(tensor->type, input.data(), f32.data(), count * width, error) ||
                !encode_rows(output_type, f32.data(), count, width, encoded, error)) {
            return false;
        }
        output.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
        if (!output) {
            error = "failed writing converted sidecar tensor";
            return false;
        }
        bytes_written += encoded.size();
    }
    return true;
}

static bool write_sliced_head(gguf_file & source, const ggml_tensor * tensor,
        uint64_t source_offset, ggml_type output_type, const std::vector<int32_t> & ids,
        std::ofstream & output, uint64_t & bytes_written, std::string & error) {
    const int64_t width = tensor->ne[0];
    const size_t source_row_size = ggml_row_size(tensor->type, width);
    const size_t output_row_size = ggml_row_size(output_type, width);
    if (source_row_size == 0 || output_row_size == 0) {
        error = "invalid output-head row layout";
        return false;
    }
    bytes_written = 0;
    if (tensor->type == output_type) {
        std::vector<uint8_t> row(source_row_size);
        for (const int32_t id : ids) {
            if (!source.read_at(source_offset + static_cast<uint64_t>(id) * source_row_size,
                    row.data(), row.size(), error)) {
                return false;
            }
            output.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
            if (!output) {
                error = "failed writing sliced sidecar head";
                return false;
            }
            bytes_written += row.size();
        }
        return true;
    }

    static constexpr size_t ROWS_PER_CHUNK = 256;
    std::vector<uint8_t> input;
    std::vector<float> f32;
    std::vector<uint8_t> encoded;
    for (size_t first = 0; first < ids.size(); first += ROWS_PER_CHUNK) {
        const size_t count = std::min(ROWS_PER_CHUNK, ids.size() - first);
        input.resize(source_row_size * count);
        for (size_t row = 0; row < count; ++row) {
            if (!source.read_at(source_offset + static_cast<uint64_t>(ids[first + row]) * source_row_size,
                    input.data() + row * source_row_size, source_row_size, error)) {
                return false;
            }
        }
        f32.resize(count * static_cast<size_t>(width));
        if (!dequantize_rows(tensor->type, input.data(), f32.data(),
                static_cast<int64_t>(count) * width, error) ||
                !encode_rows(output_type, f32.data(), static_cast<int64_t>(count), width, encoded, error)) {
            return false;
        }
        output.write(reinterpret_cast<const char *>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
        if (!output) {
            error = "failed writing converted sidecar head";
            return false;
        }
        bytes_written += encoded.size();
    }
    return true;
}

static bool write_blob(gguf_file & source, const std::vector<tensor_spec> & specs,
        const std::vector<int32_t> & ids, const fs::path & blob_path,
        const fs::path & manifest_path, const std::string & cache_key,
        const common_spec_sidecar_profile & profile, std::string & error) {
    std::ofstream output(blob_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create " + blob_path.string();
        return false;
    }
    std::vector<artifact_entry> entries;
    entries.reserve(specs.size());
    uint64_t cursor = 0;
    for (const auto & spec : specs) {
        const ggml_tensor * tensor = source.tensor(spec.source_name);
        uint64_t source_offset = 0;
        if (tensor == nullptr || !source.tensor_data_offset(spec.source_name, source_offset, error)) {
            return false;
        }
        uint64_t written = 0;
        if (spec.sliced_head) {
            if (!write_sliced_head(source, tensor, source_offset, spec.output_type, ids, output, written, error)) {
                return false;
            }
        } else if (tensor->type == spec.output_type) {
            written = ggml_nbytes(tensor);
            if (!write_direct_range(source, source_offset, written, output, error)) {
                return false;
            }
        } else if (!write_converted_tensor(source, tensor, source_offset,
                spec.output_type, output, written, error)) {
            return false;
        }
        entries.push_back({spec.output_name, spec.output_type, spec.output_shape, cursor, written});
        cursor += written;
    }
    output.flush();
    if (!output) {
        error = "failed flushing " + blob_path.string();
        return false;
    }
    output.close();

    ordered_json manifest;
    manifest["schema"] = 1;
    manifest["generator"] = AUTO_ASSET_SCHEMA;
    manifest["provider"] = profile.name != nullptr ? profile.name : "unknown";
    manifest["source_file"] = fs::path(source.path()).filename().string();
    manifest["cache_key"] = cache_key;
    const std::vector<uint8_t> manifest_ids = ids_to_le_bytes(ids);
    manifest["draft_ids_sha256"] = hash_sha256_hex(manifest_ids.data(), manifest_ids.size());
    manifest["tensors"] = ordered_json::array();
    for (const auto & entry : entries) {
        manifest["tensors"].push_back({
            {"name", entry.name},
            {"dtype", std::to_string(static_cast<int>(entry.type))},
            {"shape", entry.shape},
            {"offset", entry.offset},
            {"nbytes", entry.nbytes},
        });
    }
    std::ofstream manifest_file(manifest_path, std::ios::binary | std::ios::trunc);
    if (!manifest_file) {
        error = "cannot create " + manifest_path.string();
        return false;
    }
    manifest_file << manifest.dump(2) << '\n';
    if (!manifest_file) {
        error = "failed writing " + manifest_path.string();
        return false;
    }
    return true;
}

static bool write_ids(const fs::path & path, const std::vector<int32_t> & ids, std::string & error) {
    const std::vector<uint8_t> bytes = ids_to_le_bytes(ids);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()))) {
        error = "failed writing " + path.string();
        return false;
    }
    return true;
}

static bool write_full_head(gguf_file & target, const fs::path & path, std::string & error) {
    const ggml_tensor * head = target.tensor("output.weight");
    if (!check_shape(head, {5120, QWEN35_VOCAB}, "output.weight", error) ||
            head->type != GGML_TYPE_Q6_K) {
        if (error.empty()) error = "DFlash full head requires Q6_K output.weight";
        return false;
    }
    uint64_t offset = 0;
    if (!target.tensor_data_offset("output.weight", offset, error)) return false;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create " + path.string();
        return false;
    }
    return write_direct_range(target, offset, ggml_nbytes(head), output, error);
}

static bool build_mtp_bundle(const common_spec_sidecar_profile & profile, gguf_file & target,
        const std::vector<int32_t> & ids, const fs::path & directory,
        const std::string & cache_key, std::string & error) {
    const bool moe = profile.name != nullptr && std::strcmp(profile.name, "qwen35moe-mtp") == 0;
    const auto specs = moe ? qwen35moe_mtp_specs() : qwen35_mtp_specs();
    if (!validate_specs(target, specs, false, false, error)) {
        return false;
    }
    return write_blob(target, specs, ids,
            directory / "drafter_weights.bin", directory / "drafter_manifest.json",
            cache_key, profile, error) &&
           write_ids(directory / "draft_head_ids.bin", ids, error);
}

static bool build_dflash_bundle(const common_spec_sidecar_profile & profile, gguf_file & target,
        gguf_file & draft, const std::vector<int32_t> & ids, bool full_head,
        const fs::path & directory, const std::string & cache_key, std::string & error) {
    const auto draft_specs = dflash_specs();
    if (!validate_specs(draft, draft_specs, true, true, error)) {
        return false;
    }
    const std::vector<tensor_spec> embedding = {
        make_spec("token_embd.weight", GGML_TYPE_Q4_0, {5120, QWEN35_VOCAB}),
    };
    if (!validate_specs(target, embedding, true, false, error)) {
        return false;
    }
    const ggml_tensor * head = target.tensor("output.weight");
    if (!check_shape(head, {5120, QWEN35_VOCAB}, "output.weight", error) ||
            head->type != GGML_TYPE_Q6_K) {
        if (error.empty()) error = "DFlash sidecar requires Q6_K output.weight";
        return false;
    }

    if (!write_blob(draft, draft_specs, ids,
            directory / "dflash_weights.bin", directory / "dflash_manifest.json",
            cache_key, profile, error) ||
        !write_blob(target, embedding, ids,
            directory / "drafter_weights.bin", directory / "drafter_manifest.json",
            cache_key, profile, error)) {
        return false;
    }

    uint64_t source_offset = 0;
    if (!target.tensor_data_offset("output.weight", source_offset, error)) return false;
    std::ofstream sliced_output(directory / "target_head_sliced.bin", std::ios::binary | std::ios::trunc);
    uint64_t written = 0;
    if (!sliced_output || !write_sliced_head(target, head, source_offset,
            GGML_TYPE_Q6_K, ids, sliced_output, written, error) ||
            written != static_cast<uint64_t>(QWEN35_DRAFT_VOCAB) * ggml_row_size(GGML_TYPE_Q6_K, 5120)) {
        if (error.empty()) error = "failed writing DFlash sliced target head";
        return false;
    }
    sliced_output.close();
    if (!write_ids(directory / "draft_head_ids.bin", ids, error)) {
        return false;
    }
    return !full_head || write_full_head(target, directory / "target_head.bin", error);
}

static bool file_identity(const std::string & input, std::string & identity, std::string & error) {
    const std::string path = normalized_absolute(input);
    if (path.empty() || !is_regular_file(path)) {
        error = "sidecar source is not a readable regular file: " + input;
        return false;
    }
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) {
        error = "cannot stat sidecar source: " + path;
        return false;
    }
    const auto modified = fs::last_write_time(path, ec);
    if (ec) {
        error = "cannot read sidecar source timestamp: " + path;
        return false;
    }
    std::ostringstream value;
    value << path << '\n' << size << '\n' << modified.time_since_epoch().count();
    identity = value.str();
    return true;
}

static bool marker_matches(const fs::path & directory, const std::string & cache_key) {
    try {
        std::ifstream input(directory / "bundle.ready", std::ios::binary);
        if (!input) return false;
        ordered_json marker;
        input >> marker;
        return marker.value("schema", 0) == 1 && marker.value("cache_key", std::string()) == cache_key;
    } catch (...) {
        return false;
    }
}

static bool write_marker(const fs::path & directory, const common_spec_sidecar_profile & profile,
        const std::string & cache_key, const std::string & target_identity,
        const std::string & draft_identity, const std::string & ids_source, std::string & error) {
    ordered_json marker = {
        {"schema", 1},
        {"generator", AUTO_ASSET_SCHEMA},
        {"cache_key", cache_key},
        {"profile", profile.name != nullptr ? profile.name : "unknown"},
        {"target_identity", target_identity},
        {"draft_identity", draft_identity},
        {"ids_source", ids_source},
    };
    std::ofstream output(directory / "bundle.ready", std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create sidecar cache marker";
        return false;
    }
    output << marker.dump(2) << '\n';
    if (!output) {
        error = "cannot write sidecar cache marker";
        return false;
    }
    return true;
}

#ifndef _WIN32
class file_lock {
public:
    bool acquire(const fs::path & path, std::string & error) {
        fd_ = open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0) {
            error = "cannot open sidecar cache lock: " + path.string();
            return false;
        }
        if (flock(fd_, LOCK_EX) != 0) {
            error = "cannot acquire sidecar cache lock: " + path.string();
            close(fd_);
            fd_ = -1;
            return false;
        }
        return true;
    }

    ~file_lock() {
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
    }

private:
    int fd_ = -1;
};
#endif

class temporary_directory {
public:
    explicit temporary_directory(fs::path path) : path_(std::move(path)) {}
    ~temporary_directory() {
        if (!released_) {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
    }
    void release() { released_ = true; }
private:
    fs::path path_;
    bool released_ = false;
};

static bool remove_stale_temporary_directories(
        const fs::path & parent, const std::string & destination_name, std::string & error) {
    const std::string prefix = "." + destination_name + ".tmp-";
    std::error_code ec;
    for (fs::directory_iterator it(parent, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.rfind(prefix, 0) != 0) continue;
        fs::remove_all(it->path(), ec);
        if (ec) break;
    }
    if (ec) {
        error = "cannot clear interrupted sidecar cache build: " + ec.message();
        return false;
    }
    return true;
}

static bool directory_empty_or_missing(const fs::path & path, std::string & error) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return !ec;
    if (ec || !fs::is_directory(path, ec) || ec) {
        error = "sidecar artifact path exists but is not a directory: " + path.string();
        return false;
    }
    const bool empty = fs::is_empty(path, ec);
    if (ec) {
        error = "cannot inspect sidecar artifact directory: " + path.string();
        return false;
    }
    if (!empty) {
        error = "refusing to replace non-empty explicit sidecar artifact directory: " + path.string();
        return false;
    }
    return true;
}

static common_spec_sidecar_paths paths_for_directory(const std::string & library,
        const fs::path & directory, bool full_head) {
    common_spec_sidecar_paths paths;
    paths.library = library;
    paths.artifact_dir = normalized_absolute(directory);
    paths.ids = normalized_absolute(directory / "draft_head_ids.bin");
    paths.dflash_full_head = full_head;
    return paths;
}

} // namespace

bool common_spec_sidecar_builtin_draft_vocab_ids(
        std::vector<int32_t> & ids, std::string & error) {
    ids.clear();
    error.clear();
    const char * hex = SPEC_SIDECAR_QWEN35_DRAFT_VOCAB_BITMAP_HEX;
    const size_t length = std::strlen(hex);
    if (length != static_cast<size_t>(QWEN35_VOCAB / 4)) {
        error = "built-in sidecar draft vocabulary bitmap has the wrong size";
        return false;
    }
    ids.reserve(QWEN35_DRAFT_VOCAB);
    for (size_t byte_index = 0; byte_index < length / 2; ++byte_index) {
        const int high = hex_nibble(hex[2 * byte_index]);
        const int low  = hex_nibble(hex[2 * byte_index + 1]);
        if (high < 0 || low < 0) {
            error = "built-in sidecar draft vocabulary bitmap is malformed";
            return false;
        }
        const uint8_t bits = static_cast<uint8_t>((high << 4) | low);
        for (int bit = 0; bit < 8; ++bit) {
            if ((bits & (1u << bit)) != 0) {
                ids.push_back(static_cast<int32_t>(byte_index * 8 + bit));
            }
        }
    }
    if (!validate_ids(ids, error)) {
        return false;
    }
    const std::vector<uint8_t> bytes = ids_to_le_bytes(ids);
    if (hash_sha256_hex(bytes.data(), bytes.size()) != BUILTIN_IDS_SHA256) {
        error = "built-in sidecar draft vocabulary failed its SHA-256 integrity check";
        ids.clear();
        return false;
    }
    return true;
}

bool common_spec_sidecar_prepare_artifacts(
        const common_spec_sidecar_profile & profile,
        const std::string & target_path,
        const std::string & draft_path,
        const std::string & cache_root,
        common_spec_sidecar_paths & paths,
        bool & cache_hit,
        std::string & error) {
    paths = {};
    cache_hit = false;
    error.clear();
#ifdef _WIN32
    (void) profile; (void) target_path; (void) draft_path; (void) cache_root;
    error = "automatic speculative sidecar asset preparation is currently supported on Linux";
    return false;
#else
    if (!common_spec_sidecar_get_library(profile, paths.library, error)) {
        return false;
    }

    const std::string target_normalized = normalized_absolute(target_path);
    gguf_file target(target_normalized, error);
    if (!target.valid()) return false;

    std::unique_ptr<gguf_file> draft;
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_DFLASH) {
        if (draft_path.empty()) {
            error = "DFlash sidecar cache is missing; provide the DFlash GGUF with --spec-draft-model";
            return false;
        }
        const std::string draft_normalized = normalized_absolute(draft_path);
        draft.reset(new gguf_file(draft_normalized, error));
        if (!draft->valid()) return false;
    }

    std::vector<int32_t> ids;
    std::string ids_source;
    if (!resolve_ids(profile, target, ids, ids_source, error)) {
        return false;
    }
    const std::vector<uint8_t> ids_bytes = ids_to_le_bytes(ids);
    const std::string ids_hash = hash_sha256_hex(ids_bytes.data(), ids_bytes.size());

    std::string target_identity;
    std::string draft_identity;
    if (!file_identity(target.path(), target_identity, error) ||
            (draft != nullptr && !file_identity(draft->path(), draft_identity, error))) {
        return false;
    }
    const bool full_head = profile.kind == COMMON_SPEC_SIDECAR_KIND_DFLASH &&
            profile.full_head_env != nullptr && std::getenv(profile.full_head_env) != nullptr;
    const std::string key_material = std::string(AUTO_ASSET_SCHEMA) + "\n" +
            (profile.name != nullptr ? profile.name : "unknown") + "\n" +
            target_identity + "\n" + draft_identity + "\n" + ids_hash + "\n" +
            (full_head ? "full-head" : "sliced-head");
    const std::string cache_key = hash_sha256_hex(key_material.data(), key_material.size()).substr(0, 24);

    const char * explicit_artifact = profile.artifact_env != nullptr ? std::getenv(profile.artifact_env) : nullptr;
    const bool explicit_destination = explicit_artifact != nullptr && explicit_artifact[0] != '\0';
    fs::path root;
    fs::path destination;
    if (explicit_destination) {
        const std::string normalized = normalized_absolute(explicit_artifact);
        if (normalized.empty()) {
            error = "invalid explicit sidecar artifact directory";
            return false;
        }
        destination = normalized;
        root = destination.parent_path();
    } else {
        const std::string default_root = cache_root.empty()
                ? (fs::path(fs_get_cache_directory()) / "spec-sidecar").string()
                : cache_root;
        const std::string normalized = normalized_absolute(default_root);
        if (normalized.empty()) {
            error = "invalid sidecar cache directory: " + default_root;
            return false;
        }
        root = normalized;
        destination = root / (profile.name != nullptr ? profile.name : "unknown") / cache_key;
    }

    std::error_code ec;
    fs::create_directories(explicit_destination ? root : destination.parent_path(), ec);
    if (ec) {
        error = "cannot create sidecar cache root: " + ec.message();
        return false;
    }

    file_lock lock;
    const fs::path lock_path = (explicit_destination ? root : destination.parent_path()) /
            ("." + std::string(profile.name != nullptr ? profile.name : "sidecar") + "-" + cache_key + ".lock");
    if (!lock.acquire(lock_path, error) ||
            !remove_stale_temporary_directories(
                    destination.parent_path(), destination.filename().string(), error)) {
        return false;
    }

    common_spec_sidecar_paths candidate = paths_for_directory(paths.library, destination, full_head);
    std::string validation_error;
    if (path_is_directory(destination) &&
            (explicit_destination || marker_matches(destination, cache_key)) &&
            common_spec_sidecar_validate_artifacts(profile, candidate, validation_error)) {
        paths = std::move(candidate);
        cache_hit = true;
        LOG_INF("spec sidecar: using cached %s assets at %s\n",
                profile.name, paths.artifact_dir.c_str());
        return true;
    }

    if (explicit_destination) {
        if (!directory_empty_or_missing(destination, error)) {
            return false;
        }
        fs::remove_all(destination, ec);
        ec.clear();
    } else if (fs::exists(destination, ec)) {
        fs::remove_all(destination, ec);
        if (ec) {
            error = "cannot clear incomplete sidecar cache entry: " + ec.message();
            return false;
        }
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temporary = destination.parent_path() /
            ("." + destination.filename().string() + ".tmp-" + std::to_string(getpid()) + "-" + std::to_string(nonce));
    fs::create_directories(temporary, ec);
    if (ec) {
        error = "cannot create temporary sidecar cache directory: " + ec.message();
        return false;
    }
    temporary_directory cleanup(temporary);

    const auto started = std::chrono::steady_clock::now();
    LOG_INF("spec sidecar: preparing %s assets natively at %s (first start only)\n",
            profile.name, destination.string().c_str());
    bool built = false;
    if (profile.kind == COMMON_SPEC_SIDECAR_KIND_MTP) {
        built = build_mtp_bundle(profile, target, ids, temporary, cache_key, error);
    } else if (draft != nullptr) {
        built = build_dflash_bundle(profile, target, *draft, ids, full_head, temporary, cache_key, error);
    }
    if (!built) {
        return false;
    }

    common_spec_sidecar_paths temporary_paths = paths_for_directory(paths.library, temporary, full_head);
    if (!common_spec_sidecar_validate_artifacts(profile, temporary_paths, error) ||
            !write_marker(temporary, profile, cache_key, target_identity,
                    draft_identity, ids_source, error)) {
        return false;
    }
    fs::rename(temporary, destination, ec);
    if (ec) {
        error = "cannot commit sidecar cache entry: " + ec.message();
        return false;
    }
    cleanup.release();
    paths = paths_for_directory(paths.library, destination, full_head);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    LOG_INF("spec sidecar: prepared and cached %s assets in %.1f seconds\n", profile.name, seconds);
    return true;
#endif
}
