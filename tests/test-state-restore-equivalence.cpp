#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct step_logits {
    std::vector<float> values;
    llama_token argmax = -1;
};

struct depth_result {
    int depth = 0;
    bool complete = false;
    bool accepted = false;
    size_t state_bytes = 0;
    uint64_t state_hash = 0;
    uint64_t prefix_hash = 0;
    uint64_t fresh_logits_hash = 0;
    uint64_t restored_logits_hash = 0;
    size_t bitwise_mismatches = 0;
    size_t tolerance_violations = 0;
    size_t nonfinite_mismatches = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    std::vector<llama_token> prefix_tokens;
    std::vector<llama_token> generation_inputs;
    std::vector<llama_token> fresh_argmax;
    std::vector<llama_token> restored_argmax;
};

static uint64_t fnv1a(const void * data, size_t size, uint64_t hash = 1469598103934665603ULL) {
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

static bool parse_nonnegative_double(const char * text, double & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) {
        return false;
    }
    value = parsed;
    return true;
}

static bool parse_depths(const char * text, std::vector<int> & depths) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        int value = 0;
        if (!parse_positive_int(item.c_str(), value) || std::find(depths.begin(), depths.end(), value) != depths.end()) {
            return false;
        }
        depths.push_back(value);
    }
    return !depths.empty();
}

static bool decode_many(llama_context * ctx, std::vector<llama_token> & tokens, int n_batch) {
    int processed = 0;
    while (processed < static_cast<int>(tokens.size())) {
        const int count = std::min(n_batch, static_cast<int>(tokens.size()) - processed);
        const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data() + processed, count));
        if (rc != 0) {
            std::fprintf(stderr, "decode_many: llama_decode failed at token %d with rc=%d\n", processed, rc);
            return false;
        }
        processed += count;
    }
    llama_synchronize(ctx);
    return true;
}

static bool decode_step(llama_context * ctx, llama_token token, int n_vocab, step_logits & output) {
    const int rc = llama_decode(ctx, llama_batch_get_one(&token, 1));
    if (rc != 0) {
        std::fprintf(stderr, "decode_step: llama_decode failed with rc=%d\n", rc);
        return false;
    }
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        std::fprintf(stderr, "decode_step: no logits returned\n");
        return false;
    }
    output.values.assign(logits, logits + n_vocab);
    output.argmax = static_cast<llama_token>(std::max_element(output.values.begin(), output.values.end()) - output.values.begin());
    return true;
}

static depth_result check_depth(
        llama_model * model,
        const common_params & params,
        int depth,
        int n_gen,
        uint32_t seed,
        double abs_tol,
        double rel_tol) {
    depth_result result;
    result.depth = depth;

    llama_context_params cparams = llama_context_default_params();
    const uint64_t n_ctx = static_cast<uint64_t>(depth) + static_cast<uint64_t>(n_gen);
    if (n_ctx > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "depth %d: depth+n_gen exceeds uint32 context limit\n", depth);
        return result;
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
        return result;
    }
    llama_set_n_threads(ctx.get(), params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        std::fprintf(stderr, "depth %d: invalid vocabulary size\n", depth);
        return result;
    }

    std::mt19937 rng(seed ^ static_cast<uint32_t>(depth));
    std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
    result.prefix_tokens.resize(depth);
    for (int i = 0; i < depth; ++i) {
        result.prefix_tokens[i] = i == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : dist(rng);
    }
    result.prefix_hash = fnv1a(result.prefix_tokens.data(), result.prefix_tokens.size() * sizeof(llama_token));

    std::fprintf(stderr, "depth %d: fresh prefix (%d tokens)\n", depth, depth);
    if (!decode_many(ctx.get(), result.prefix_tokens, params.n_batch)) {
        return result;
    }

    result.state_bytes = llama_state_seq_get_size(ctx.get(), 0);
    if (result.state_bytes == 0) {
        std::fprintf(stderr, "depth %d: sequence state size is zero\n", depth);
        return result;
    }
    std::vector<uint8_t> state(result.state_bytes);
    const size_t copied = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0);
    if (copied != state.size()) {
        std::fprintf(stderr, "depth %d: saved %zu state bytes, expected %zu\n", depth, copied, state.size());
        return result;
    }
    result.state_hash = fnv1a(state.data(), state.size());

    std::vector<step_logits> fresh;
    fresh.reserve(n_gen);
    llama_token input = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : dist(rng);
    for (int step = 0; step < n_gen; ++step) {
        result.generation_inputs.push_back(input);
        fresh.emplace_back();
        if (!decode_step(ctx.get(), input, n_vocab, fresh.back())) {
            return result;
        }
        result.fresh_argmax.push_back(fresh.back().argmax);
        result.fresh_logits_hash = fnv1a(
            fresh.back().values.data(), fresh.back().values.size() * sizeof(float), result.fresh_logits_hash == 0 ? 1469598103934665603ULL : result.fresh_logits_hash);
        input = fresh.back().argmax;
    }

    llama_memory_clear(llama_get_memory(ctx.get()), false);
    const size_t restored = llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0);
    if (restored != state.size()) {
        std::fprintf(stderr, "depth %d: restored %zu state bytes, expected %zu\n", depth, restored, state.size());
        return result;
    }

    std::fprintf(stderr, "depth %d: restored prefix; replaying %d greedy target steps\n", depth, n_gen);
    std::vector<step_logits> replay;
    replay.reserve(n_gen);
    for (int step = 0; step < n_gen; ++step) {
        replay.emplace_back();
        if (!decode_step(ctx.get(), result.generation_inputs[step], n_vocab, replay.back())) {
            return result;
        }
        result.restored_argmax.push_back(replay.back().argmax);
        result.restored_logits_hash = fnv1a(
            replay.back().values.data(), replay.back().values.size() * sizeof(float), result.restored_logits_hash == 0 ? 1469598103934665603ULL : result.restored_logits_hash);

        for (int token = 0; token < n_vocab; ++token) {
            const float a = fresh[step].values[token];
            const float b = replay[step].values[token];
            if (std::memcmp(&a, &b, sizeof(float)) != 0) {
                ++result.bitwise_mismatches;
            }
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (!(std::isnan(a) && std::isnan(b)) && a != b) {
                    ++result.nonfinite_mismatches;
                }
                continue;
            }
            const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
            const double scale = std::max(std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b)));
            const double rel = scale > 0.0 ? diff / scale : 0.0;
            result.max_abs_diff = std::max(result.max_abs_diff, diff);
            result.max_rel_diff = std::max(result.max_rel_diff, rel);
            if (diff > abs_tol + rel_tol * scale) {
                ++result.tolerance_violations;
            }
        }
    }

    result.complete = true;
    result.accepted = result.fresh_argmax == result.restored_argmax &&
        result.nonfinite_mismatches == 0 && result.tolerance_violations == 0;
    return result;
}

static void print_tokens(const std::vector<llama_token> & values) {
    std::printf("[");
    for (size_t i = 0; i < values.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", values[i]);
    }
    std::printf("]");
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

    if (params.speculative.types.size() != 1 || params.speculative.types[0] != COMMON_SPECULATIVE_TYPE_NONE || params.speculative.has_dft()) {
        std::fprintf(stderr, "speculative/draft configuration is forbidden\n");
        return 2;
    }
    if (params.n_batch <= 0 || params.n_ubatch <= 0 || params.n_ubatch > params.n_batch) {
        std::fprintf(stderr, "invalid batch/ubatch\n");
        return 2;
    }
    if (params.cpuparams.n_threads <= 0 || params.cpuparams_batch.n_threads != params.cpuparams.n_threads) {
        std::fprintf(stderr, "normal and batch thread counts must be equal and positive\n");
        return 2;
    }

    std::vector<int> depths;
    const char * depths_text = std::getenv("DSV4_STATE_DEPTHS");
    if (depths_text == nullptr) {
        depths_text = "2048,3072,16384";
    }
    if (!parse_depths(depths_text, depths)) {
        std::fprintf(stderr, "invalid DSV4_STATE_DEPTHS=%s\n", depths_text);
        return 2;
    }

    int n_gen = 4;
    if (const char * text = std::getenv("DSV4_STATE_N_GEN")) {
        if (!parse_positive_int(text, n_gen)) {
            std::fprintf(stderr, "invalid DSV4_STATE_N_GEN=%s\n", text);
            return 2;
        }
    }
    uint32_t seed = 12345;
    if (const char * text = std::getenv("DSV4_STATE_SEED")) {
        if (!parse_uint(text, seed)) {
            std::fprintf(stderr, "invalid DSV4_STATE_SEED=%s\n", text);
            return 2;
        }
    }
    double abs_tol = 1e-5;
    double rel_tol = 1e-5;
    if (const char * text = std::getenv("DSV4_STATE_ABS_TOL")) {
        if (!parse_nonnegative_double(text, abs_tol)) {
            std::fprintf(stderr, "invalid DSV4_STATE_ABS_TOL=%s\n", text);
            return 2;
        }
    }
    if (const char * text = std::getenv("DSV4_STATE_REL_TOL")) {
        if (!parse_nonnegative_double(text, rel_tol)) {
            std::fprintf(stderr, "invalid DSV4_STATE_REL_TOL=%s\n", text);
            return 2;
        }
    }

    ggml_backend_load_all();
    auto init = common_init_from_params(params, true);
    llama_model * model = init ? init->model() : nullptr;
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    std::vector<depth_result> results;
    bool accepted = true;
    for (int depth : depths) {
        depth_result result = check_depth(model, params, depth, n_gen, seed, abs_tol, rel_tol);
        accepted = accepted && result.complete && result.accepted;
        results.push_back(std::move(result));
    }

    std::printf("{\n");
    std::printf("  \"complete\": %s,\n", std::all_of(results.begin(), results.end(), [](const depth_result & r) { return r.complete; }) ? "true" : "false");
    std::printf("  \"accepted\": %s,\n", accepted ? "true" : "false");
    std::printf("  \"target_only\": true,\n");
    std::printf("  \"state_restore_scope\": \"same_context_same_benchmark_instance\",\n");
    std::printf("  \"continuation_contract\": \"greedy replay; semantic state equivalence, not llama-bench random-token timing input\",\n");
    std::printf("  \"n_gen\": %d,\n", n_gen);
    std::printf("  \"n_batch\": %d,\n", params.n_batch);
    std::printf("  \"n_ubatch\": %d,\n", params.n_ubatch);
    std::printf("  \"cache_type_k\": \"%s\",\n", ggml_type_name(params.cache_type_k));
    std::printf("  \"cache_type_v\": \"%s\",\n", ggml_type_name(params.cache_type_v));
    std::printf("  \"flash_attn\": \"%s\",\n", llama_flash_attn_type_name(params.flash_attn_type));
    std::printf("  \"seed\": %u,\n", seed);
    std::printf("  \"abs_tolerance\": %.17g,\n", abs_tol);
    std::printf("  \"rel_tolerance\": %.17g,\n", rel_tol);
    std::printf("  \"records\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const depth_result & r = results[i];
        std::printf("    {\n");
        std::printf("      \"depth\": %d,\n", r.depth);
        std::printf("      \"complete\": %s,\n", r.complete ? "true" : "false");
        std::printf("      \"accepted\": %s,\n", r.accepted ? "true" : "false");
        std::printf("      \"state_bytes\": %zu,\n", r.state_bytes);
        std::printf("      \"state_fnv1a64\": \"%s\",\n", hex64(r.state_hash).c_str());
        std::printf("      \"prefix_fnv1a64\": \"%s\",\n", hex64(r.prefix_hash).c_str());
        std::printf("      \"fresh_logits_fnv1a64\": \"%s\",\n", hex64(r.fresh_logits_hash).c_str());
        std::printf("      \"restored_logits_fnv1a64\": \"%s\",\n", hex64(r.restored_logits_hash).c_str());
        std::printf("      \"bitwise_mismatches\": %zu,\n", r.bitwise_mismatches);
        std::printf("      \"tolerance_violations\": %zu,\n", r.tolerance_violations);
        std::printf("      \"nonfinite_mismatches\": %zu,\n", r.nonfinite_mismatches);
        std::printf("      \"max_abs_diff\": %.17g,\n", r.max_abs_diff);
        std::printf("      \"max_rel_diff\": %.17g,\n", r.max_rel_diff);
        std::printf("      \"prefix_tokens\": "); print_tokens(r.prefix_tokens); std::printf(",\n");
        std::printf("      \"generation_input_tokens\": "); print_tokens(r.generation_inputs); std::printf(",\n");
        std::printf("      \"fresh_argmax_tokens\": "); print_tokens(r.fresh_argmax); std::printf(",\n");
        std::printf("      \"restored_argmax_tokens\": "); print_tokens(r.restored_argmax); std::printf("\n");
        std::printf("    }%s\n", i + 1 == results.size() ? "" : ",");
    }
    std::printf("  ]\n}\n");
    return accepted ? 0 : 1;
}