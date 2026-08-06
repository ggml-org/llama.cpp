#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static uint64_t fnv1a(const void * data, size_t size, uint64_t hash = 14695981039346656037ULL) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

static bool parse_positive_int(const char * text, int & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_uint(const char * text, uint32_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

static bool decode_many(llama_context * ctx, const std::vector<llama_token> & tokens, int n_batch) {
    int processed = 0;
    while (processed < static_cast<int>(tokens.size())) {
        const int count = std::min(n_batch, static_cast<int>(tokens.size()) - processed);
        const int rc = llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + processed, count));
        if (rc != 0) {
            std::fprintf(stderr, "decode_many: llama_decode failed at token %d with rc=%d\n", processed, rc);
            return false;
        }
        processed += count;
    }
    llama_synchronize(ctx);
    return true;
}

static bool capture_depth(
        llama_model * model,
        const common_params & params,
        int depth,
        int n_gen,
        uint32_t seed,
        std::ofstream & logits,
        uint64_t & byte_offset,
        json & output) {
    llama_context_params cparams = llama_context_default_params();
    const uint64_t n_ctx = static_cast<uint64_t>(depth) + static_cast<uint64_t>(n_gen);
    if (n_ctx > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "depth+n_gen exceeds uint32 context limit\n");
        return false;
    }
    cparams.n_ctx           = static_cast<uint32_t>(n_ctx);
    cparams.n_batch         = static_cast<uint32_t>(params.n_batch);
    cparams.n_ubatch        = static_cast<uint32_t>(params.n_ubatch);
    cparams.type_k          = params.cache_type_k;
    cparams.type_v          = params.cache_type_v;
    cparams.offload_kqv     = !params.no_kv_offload;
    cparams.flash_attn_type = params.flash_attn_type;
    cparams.embeddings      = false;
    cparams.op_offload      = !params.no_op_offload;
    cparams.swa_full        = false;

    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        std::fprintf(stderr, "depth %d: failed to create context\n", depth);
        return false;
    }
    llama_set_n_threads(ctx.get(), params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        std::fprintf(stderr, "invalid vocabulary size\n");
        return false;
    }

    std::mt19937 rng(seed ^ static_cast<uint32_t>(depth));
    std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
    std::vector<llama_token> prefix(depth);
    for (int i = 0; i < depth; ++i) {
        prefix[i] = i == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : dist(rng);
    }
    std::vector<llama_token> inputs(n_gen);
    for (llama_token & token : inputs) {
        token = dist(rng);
    }

    std::fprintf(stderr, "depth %d: deterministic prefix (%d tokens), then %d fixed inputs\n", depth, depth, n_gen);
    if (!decode_many(ctx.get(), prefix, params.n_batch)) {
        return false;
    }

    json depth_json = {
        {"depth", depth},
        {"prefix_fnv1a64", hex64(fnv1a(prefix.data(), prefix.size() * sizeof(llama_token)))},
        {"generation_input_tokens", inputs},
        {"records", json::array()},
    };

    std::vector<float> values(n_vocab);
    for (int step = 0; step < n_gen; ++step) {
        llama_token input = inputs[step];
        const int rc = llama_decode(ctx.get(), llama_batch_get_one(&input, 1));
        if (rc != 0) {
            std::fprintf(stderr, "depth %d step %d: llama_decode failed with rc=%d\n", depth, step, rc);
            return false;
        }
        llama_synchronize(ctx.get());
        const float * current = llama_get_logits_ith(ctx.get(), -1);
        if (current == nullptr) {
            std::fprintf(stderr, "depth %d step %d: no logits returned\n", depth, step);
            return false;
        }
        std::memcpy(values.data(), current, values.size() * sizeof(float));
        const llama_token argmax = static_cast<llama_token>(
            std::max_element(values.begin(), values.end()) - values.begin());
        const uint64_t byte_length = values.size() * sizeof(float);
        logits.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(byte_length));
        if (!logits) {
            std::fprintf(stderr, "failed to write logits at depth %d step %d\n", depth, step);
            return false;
        }
        depth_json["records"].push_back({
            {"step", step},
            {"input_token", input},
            {"argmax_token", argmax},
            {"n_vocab", n_vocab},
            {"byte_offset", byte_offset},
            {"byte_length", byte_length},
            {"logits_fnv1a64", hex64(fnv1a(values.data(), byte_length))},
        });
        byte_offset += byte_length;
    }

    output["depths"].push_back(std::move(depth_json));
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_batch = 512;
    params.n_ubatch = 256;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k = GGML_TYPE_F16;
    params.cache_type_v = GGML_TYPE_F16;
    params.cpuparams.n_threads = 12;
    params.cpuparams_batch.n_threads = 12;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }
    params.fit_params = false;

    if (params.speculative.types.size() != 1 || params.speculative.types[0] != COMMON_SPECULATIVE_TYPE_NONE ||
            params.speculative.has_dft()) {
        std::fprintf(stderr, "speculative/draft configuration is forbidden\n");
        return 2;
    }
    if (params.n_batch <= 0 || params.n_ubatch <= 0 || params.n_ubatch > params.n_batch ||
            params.cpuparams.n_threads <= 0 || params.cpuparams_batch.n_threads != params.cpuparams.n_threads) {
        std::fprintf(stderr, "invalid batch/ubatch/thread configuration\n");
        return 2;
    }

    int depth = 2048;
    int n_gen = 4;
    uint32_t seed = 12345;
    if (const char * text = std::getenv("DSV4_BF16_EQ_DEPTH")) {
        if (!parse_positive_int(text, depth)) {
            std::fprintf(stderr, "invalid DSV4_BF16_EQ_DEPTH=%s\n", text);
            return 2;
        }
    }
    if (const char * text = std::getenv("DSV4_BF16_EQ_N_GEN")) {
        if (!parse_positive_int(text, n_gen)) {
            std::fprintf(stderr, "invalid DSV4_BF16_EQ_N_GEN=%s\n", text);
            return 2;
        }
    }
    if (const char * text = std::getenv("DSV4_BF16_EQ_SEED")) {
        if (!parse_uint(text, seed)) {
            std::fprintf(stderr, "invalid DSV4_BF16_EQ_SEED=%s\n", text);
            return 2;
        }
    }

    const char * output_dir_env = std::getenv("DSV4_BF16_EQ_OUTPUT_DIR");
    if (output_dir_env == nullptr || output_dir_env[0] == '\0') {
        std::fprintf(stderr, "DSV4_BF16_EQ_OUTPUT_DIR is required\n");
        return 2;
    }
    const std::filesystem::path output_dir(output_dir_env);
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::fprintf(stderr, "failed to create output directory: %s\n", ec.message().c_str());
        return 2;
    }
    const std::filesystem::path logits_path = output_dir / "logits.f32";
    std::ofstream logits(logits_path, std::ios::binary | std::ios::trunc);
    if (!logits) {
        std::fprintf(stderr, "failed to open %s\n", logits_path.string().c_str());
        return 2;
    }

    const char * candidate = std::getenv("GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE");
    if (candidate == nullptr || (std::strcmp(candidate, "0") != 0 && std::strcmp(candidate, "1") != 0)) {
        std::fprintf(stderr, "candidate option must be explicitly 0 or 1\n");
        return 2;
    }

    ggml_backend_load_all();
    auto init = common_init_from_params(params, true);
    llama_model * model = init ? init->model() : nullptr;
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    json output = {
        {"schema_version", 1},
        {"complete", false},
        {"target_only", true},
        {"state_restore_used", false},
        {"sampling_used", false},
        {"candidate_value", candidate},
        {"depth", depth},
        {"n_gen", n_gen},
        {"seed", seed},
        {"n_batch", params.n_batch},
        {"n_ubatch", params.n_ubatch},
        {"cache_type_k", ggml_type_name(params.cache_type_k)},
        {"cache_type_v", ggml_type_name(params.cache_type_v)},
        {"flash_attn", llama_flash_attn_type_name(params.flash_attn_type)},
        {"logits_file", "logits.f32"},
        {"float_format", "IEEE-754 binary32 native little-endian"},
        {"depths", json::array()},
    };

    uint64_t byte_offset = 0;
    const bool complete = capture_depth(model, params, depth, n_gen, seed, logits, byte_offset, output);
    logits.flush();
    if (!logits) {
        std::fprintf(stderr, "failed to flush logits output\n");
        return 1;
    }
    output["expected_logits_bytes"] = byte_offset;
    output["complete"] = complete;
    std::cout << output.dump(2) << '\n';
    return complete ? 0 : 1;
}