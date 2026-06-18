#pragma once

#include "ggml.h"
#include "gguf.h"

#include "higgs-sampler.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace higgs_audio {

struct companion_tensor_info {
    int64_t index       = -1;
    size_t  data_offset = 0;
    size_t  size        = 0;
    ggml_type type      = GGML_TYPE_COUNT;
};

struct companion_metadata {
    std::string format;
    std::string backbone_arch;
    codebook_layout layout;
    int boc_id                 = BOC_ID;
    int eoc_id                 = EOC_ID;
    int sample_rate            = 24000;
    int frame_rate             = 25;
    bool use_delay_pattern     = true;
    bool tie_codebook_embeddings = true;
    int64_t n_tensors          = 0;
    int codec_tensor_count     = 0;
    std::vector<std::string> codec_tensor_names;
    std::vector<std::string> codec_original_tensor_names;
    companion_tensor_info codebook_embedding;
    companion_tensor_info codebook_head;
};

struct companion_weights {
    codebook_layout layout;
    std::vector<float> codebook_embedding;
    std::vector<float> codebook_head;
};

class companion_file {
public:
    explicit companion_file(const std::string & path) : path(path) {
        gguf_init_params params {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ nullptr,
        };

        ctx = gguf_init_from_file(path.c_str(), params);
        if (!ctx) {
            throw std::runtime_error("failed to open Higgs audio GGUF: " + path);
        }

        meta = read_metadata();
        validate();
    }

    ~companion_file() {
        if (ctx) {
            gguf_free(ctx);
        }
    }

    companion_file(const companion_file &) = delete;
    companion_file & operator=(const companion_file &) = delete;

    companion_file(companion_file && other) noexcept
            : path(std::move(other.path)), ctx(other.ctx), meta(std::move(other.meta)) {
        other.ctx = nullptr;
    }

    companion_file & operator=(companion_file && other) noexcept {
        if (this != &other) {
            if (ctx) {
                gguf_free(ctx);
            }
            path = std::move(other.path);
            ctx  = other.ctx;
            meta = std::move(other.meta);
            other.ctx = nullptr;
        }
        return *this;
    }

    const std::string & file_path() const {
        return path;
    }

    const companion_metadata & metadata() const {
        return meta;
    }

    const gguf_context * gguf() const {
        return ctx;
    }

    companion_tensor_info tensor_info(const std::string & name) const {
        return require_tensor(name.c_str());
    }

    companion_tensor_info codec_tensor_info(const std::string & original_name) const {
        for (size_t i = 0; i < meta.codec_original_tensor_names.size(); ++i) {
            if (meta.codec_original_tensor_names[i] == original_name) {
                return tensor_info(meta.codec_tensor_names[i]);
            }
        }
        throw std::runtime_error("missing Higgs codec tensor: " + original_name);
    }

    std::vector<float> read_tensor_f32(const companion_tensor_info & info) const {
        return read_f16_tensor_to_f32(info);
    }

    companion_weights load_codebook_weights() const {
        companion_weights out;
        out.layout = meta.layout;
        out.codebook_embedding = read_f16_tensor_to_f32(meta.codebook_embedding);
        out.codebook_head      = read_f16_tensor_to_f32(meta.codebook_head);
        return out;
    }

private:
    static int require_u32(const gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            throw std::runtime_error(std::string("missing GGUF key: ") + key);
        }
        if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_UINT32) {
            throw std::runtime_error(std::string("GGUF key has wrong type: ") + key);
        }
        return (int) gguf_get_val_u32(ctx, kid);
    }

    static bool require_bool(const gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            throw std::runtime_error(std::string("missing GGUF key: ") + key);
        }
        if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_BOOL) {
            throw std::runtime_error(std::string("GGUF key has wrong type: ") + key);
        }
        return gguf_get_val_bool(ctx, kid);
    }

    static std::string require_string(const gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            throw std::runtime_error(std::string("missing GGUF key: ") + key);
        }
        if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("GGUF key has wrong type: ") + key);
        }
        return gguf_get_val_str(ctx, kid);
    }

    static std::vector<std::string> require_string_array(const gguf_context * ctx, const char * key) {
        const int64_t kid = gguf_find_key(ctx, key);
        if (kid < 0) {
            throw std::runtime_error(std::string("missing GGUF key: ") + key);
        }
        if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("GGUF key has wrong type: ") + key);
        }

        std::vector<std::string> values;
        const size_t n = gguf_get_arr_n(ctx, kid);
        values.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            values.emplace_back(gguf_get_arr_str(ctx, kid, i));
        }
        return values;
    }

    companion_tensor_info require_tensor(const char * name) const {
        const int64_t tid = gguf_find_tensor(ctx, name);
        if (tid < 0) {
            throw std::runtime_error(std::string("missing GGUF tensor: ") + name);
        }

        companion_tensor_info info;
        info.index       = tid;
        info.data_offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tid);
        info.size        = gguf_get_tensor_size(ctx, tid);
        info.type        = gguf_get_tensor_type(ctx, tid);
        return info;
    }

    std::vector<float> read_f16_tensor_to_f32(const companion_tensor_info & info) const {
        if (info.type != GGML_TYPE_F16) {
            throw std::runtime_error("only F16 Higgs tensors can be read through this path");
        }
        if (info.size % sizeof(ggml_fp16_t) != 0) {
            throw std::runtime_error("F16 Higgs tensor has invalid byte size");
        }

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to reopen Higgs audio GGUF: " + path);
        }

        file.seekg((std::streamoff) info.data_offset, std::ios::beg);
        if (!file) {
            throw std::runtime_error("failed to seek Higgs tensor data");
        }

        std::vector<ggml_fp16_t> f16(info.size / sizeof(ggml_fp16_t));
        file.read(reinterpret_cast<char *>(f16.data()), (std::streamsize) info.size);
        if ((size_t) file.gcount() != info.size) {
            throw std::runtime_error("failed to read complete Higgs tensor data");
        }

        std::vector<float> f32(f16.size());
        ggml_fp16_to_fp32_row(f16.data(), f32.data(), (int64_t) f16.size());
        return f32;
    }

    companion_metadata read_metadata() const {
        companion_metadata out;
        out.format        = require_string(ctx, "higgs_audio.format");
        out.backbone_arch = require_string(ctx, "higgs_audio.backbone_arch");

        out.layout.num_codebooks = require_u32(ctx, "higgs_audio.num_codebooks");
        out.layout.codebook_size = require_u32(ctx, "higgs_audio.codebook_size");
        out.layout.n_embd        = require_u32(ctx, "higgs_audio.hidden_size");

        out.boc_id                 = require_u32(ctx, "higgs_audio.boc_id");
        out.eoc_id                 = require_u32(ctx, "higgs_audio.eoc_id");
        out.sample_rate            = require_u32(ctx, "higgs_audio.sample_rate");
        out.frame_rate             = require_u32(ctx, "higgs_audio.frame_rate");
        out.use_delay_pattern      = require_bool(ctx, "higgs_audio.use_delay_pattern");
        out.tie_codebook_embeddings = require_bool(ctx, "higgs_audio.tie_codebook_embeddings");
        out.n_tensors              = gguf_get_n_tensors(ctx);
        out.codec_tensor_count     = require_u32(ctx, "higgs_audio.codec_tensor_count");
        out.codec_tensor_names     = require_string_array(ctx, "higgs_audio.codec_tensor_names");
        out.codec_original_tensor_names = require_string_array(ctx, "higgs_audio.codec_original_tensor_names");
        out.codebook_embedding     = require_tensor("higgs.codebook_embd.weight");
        out.codebook_head          = require_tensor("higgs.codebook_head.weight");
        return out;
    }

    void validate() const {
        if (meta.format != "higgs-audio-v3-tts") {
            throw std::runtime_error("unsupported Higgs companion format: " + meta.format);
        }
        if (meta.backbone_arch != "qwen3") {
            throw std::runtime_error("unsupported Higgs backbone arch: " + meta.backbone_arch);
        }
        if (meta.layout.num_codebooks <= 0 || meta.layout.codebook_size <= 0 || meta.layout.n_embd <= 0) {
            throw std::runtime_error("invalid Higgs codebook layout in companion GGUF");
        }
        if (meta.boc_id != BOC_ID || meta.eoc_id != EOC_ID) {
            throw std::runtime_error("unexpected Higgs BOC/EOC ids in companion GGUF");
        }
        if (meta.sample_rate <= 0 || meta.frame_rate <= 0) {
            throw std::runtime_error("invalid Higgs audio timing metadata");
        }
        if (!meta.use_delay_pattern) {
            throw std::runtime_error("Higgs companion GGUF does not use delay pattern");
        }
        if (meta.codebook_embedding.type != GGML_TYPE_F16 || meta.codebook_head.type != GGML_TYPE_F16) {
            throw std::runtime_error("Higgs codebook embedding/head must be F16 in this native path");
        }
        if (meta.codec_tensor_count < 0) {
            throw std::runtime_error("invalid Higgs codec tensor count");
        }
        if ((int) meta.codec_tensor_names.size() != meta.codec_tensor_count ||
                (int) meta.codec_original_tensor_names.size() != meta.codec_tensor_count) {
            throw std::runtime_error("Higgs codec tensor metadata arrays do not match codec tensor count");
        }

        const size_t expected_f16_bytes = meta.layout.n_weights() * ggml_type_size(GGML_TYPE_F16);
        if (meta.codebook_embedding.size != expected_f16_bytes) {
            throw std::runtime_error("Higgs codebook embedding tensor has unexpected byte size");
        }
        if (meta.codebook_head.size != expected_f16_bytes) {
            throw std::runtime_error("Higgs codebook head tensor has unexpected byte size");
        }
    }

    std::string path;
    gguf_context * ctx = nullptr;
    companion_metadata meta;
};

} // namespace higgs_audio
