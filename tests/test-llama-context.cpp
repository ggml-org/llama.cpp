// Tests for llama_context state handling: KV and recurrent state save / load / copy / rollback.
//
// Merged from the former test-save-load-state.cpp, test-recurrent-state-rollback.cpp and
// test-state-restore-fragmented.cpp. The fragmented-restore case is a regression test for a
// fix that made state restore work on a fragmented KV cache:
// ref: https://github.com/ggml-org/llama.cpp/issues/17527
//
// Run a single model:   test-llama-context -m model.gguf -lv 5
// Run a whole dialect:  test-llama-context --models build/tests/test-models

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <csignal>
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

// printing the tail of the capture from an abort handler takes async-signal-safe file access,
// which the POSIX calls below provide - elsewhere the capture is simply left on the disk
#if defined(_WIN32)
#  define TEST_LOG_ABORT_DUMP 0
#else
#  include <fcntl.h>
#  include <unistd.h>
#  define TEST_LOG_ABORT_DUMP 1
#endif

// how much of a captured log to print when a model fails, and how much of it to print when the
// process aborts
static const size_t LOG_DUMP_MAX_BYTES = 64 * 1024;

//
// test logging
//
// table mode silences the common logger to keep the result table readable, which drops every
// message the cases log with the common macros - a failing case would then print nothing but
// FAIL. the cases log here instead:
//
//   no capture (single model)   to stderr, gated by -lv
//   capture (every --models)    to a per-model file, dumped to stdout for each model that
//                               failed once the table has been printed in full
//
// a model that aborts in the middle of a run dies before the dump, so the capture is flushed on
// every error and between cases and an abort handler prints the tail of the open file - the
// handler only uses open/lseek/read/write, the lock is deliberately not taken
//
// llama, ggml and the backends are captured by replacing the callback installed by common_init
// with llama_log_set (which forwards to ggml_log_set), so common/log is left untouched
//

static const char * test_log_prefix(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return "D";
        case GGML_LOG_LEVEL_INFO:  return "I";
        case GGML_LOG_LEVEL_WARN:  return "W";
        case GGML_LOG_LEVEL_ERROR: return "E";
        default:                   return "O";
    }
}

struct test_logger {
    std::mutex    mtx;
    std::ofstream file;
    std::string   path;      // the file an abort dumps the tail of
    std::string   pending;   // logged before the first model opened a file
    bool          capturing = false;
    int           thold = LOG_LEVEL_INFO;

    void set_threshold(int thold_) {
        std::lock_guard<std::mutex> lock(mtx);

        thold = thold_;
    }

    void open(const std::string & path_) {
        std::lock_guard<std::mutex> lock(mtx);

        capturing = true;
        path      = path_;

        file.close();
        file.open(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            fprintf(stderr, "E failed to open the log file '%s'\n", path.c_str());
            return;
        }

        file << pending;
        pending.clear();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mtx);

        file.close();
        path.clear();
    }

    // between cases, so that an abort keeps a usable file behind
    void flush() {
        std::lock_guard<std::mutex> lock(mtx);

        if (file.is_open()) {
            file.flush();
        }
    }

    // GGML_LOG_LEVEL_NONE writes the text as is, used for streams such as the generated tokens
    void write(enum ggml_log_level level, const char * fmt, va_list args) {
        if (common_log_get_verbosity(level) > thold) {
            return;
        }

        char buf[4096];
        vsnprintf(buf, sizeof(buf), fmt, args);

        // the messages are written with a leading newline for visual separation, in a per-line
        //   capture that only adds noise - unless the message is nothing but newlines
        char * text = buf;
        if (text[0] == '\n') {
            char * p = text;
            while (*p == '\n') { ++p; }
            if (*p != '\0') { text = p; }
        }

        std::string line;
        if (level != GGML_LOG_LEVEL_NONE) {
            line += test_log_prefix(level);
            line += ' ';
        }
        line += text;

        std::lock_guard<std::mutex> lock(mtx);

        if (file.is_open()) {
            file << line;
            if (level == GGML_LOG_LEVEL_ERROR) {
                file.flush();
            }
            return;
        }

        if (capturing) {
            pending += line;   // belongs to the model that is about to be run
            return;
        }

        fputs(line.c_str(), stderr);
        fflush(stderr);
    }

    // print the tail of the capture, used when the process is going away
    void dump_tail(size_t max_bytes) {
#if !TEST_LOG_ABORT_DUMP
        (void) max_bytes;
#else
        if (path.empty()) {
            return;
        }

        // note: the members named open/close/write shadow the POSIX calls
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            return;
        }

        static const char banner[] = "\n=== aborted, tail of the captured log ===\n";
        if (::write(STDERR_FILENO, banner, sizeof(banner) - 1) < 0) {
            // nothing else can be done
        }

        const off_t end   = ::lseek(fd, 0, SEEK_END);
        const off_t start = end > (off_t) max_bytes ? end - (off_t) max_bytes : 0;

        ::lseek(fd, start, SEEK_SET);

        char buf[4096];
        for (size_t left = (size_t) (end - start); left > 0; ) {
            const size_t n = left < sizeof(buf) ? left : sizeof(buf);

            const ssize_t r = ::read(fd, buf, n);
            if (r <= 0) {
                break;
            }
            if (::write(STDERR_FILENO, buf, r) < 0) {
                break;
            }
            left -= (size_t) r;
        }

        ::close(fd);
#endif // TEST_LOG_ABORT_DUMP
    }
};

static test_logger g_test_log;

static void test_log_add(enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    g_test_log.write(level, fmt, args);
    va_end(args);
}

// captures the logs of llama, ggml and the backends
static void test_log_callback(enum ggml_log_level level, const char * text, void * /* user_data */) {
    test_log_add(level, "%s", text);
}

static void test_log_abort(int sig) {
    g_test_log.dump_tail(LOG_DUMP_MAX_BYTES);

    signal(sig, SIG_DFL);
    raise(sig);
}

// with a directory the logs are kept, otherwise they are temporary and removed after the dump
static std::string test_log_path(const std::string & logs_dir, const std::string & model_name) {
    const std::string file = "test-llama-context." + model_name + ".log";

    return logs_dir.empty() ? file : (std::filesystem::path(logs_dir) / file).string();
}

// the cases share a working directory, so the session file is named after the model
static std::string test_state_path(const std::string & model_name) {
    return "dump_state." + model_name + ".bin";
}

// the options of this test come in both --opt VALUE and --opt=VALUE forms and are removed from
// argv before the common parser sees them
// returns 0 when the argument is not this option, -1 when it is but the value is missing
static int test_opt(char ** argv, int & i, int argc, const char * name, std::string & value) {
    const size_t len = std::strlen(name);

    if (std::strncmp(argv[i], name, len) != 0) {
        return 0;
    }

    if (argv[i][len] == '=') {
        value = argv[i] + len + 1;
        return value.empty() ? -1 : 1;
    }

    if (argv[i][len] != '\0') {
        return 0;
    }

    if (i + 1 >= argc) {
        return -1;
    }

    value = argv[++i];
    return value.empty() ? -1 : 1;
}

#define TLOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= g_test_log.thold) { \
            test_log_add((level), __VA_ARGS__); \
        } \
    } while (0)

#define TLOGV(verbosity, ...) TLOG_TMPL(GGML_LOG_LEVEL_NONE,  verbosity,       __VA_ARGS__)
#define TLOG_DBG(...)         TLOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define TLOG_TRC(...)         TLOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_TRACE, __VA_ARGS__)
#define TLOG_INF(...)         TLOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,  __VA_ARGS__)
#define TLOG_WRN(...)         TLOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,  __VA_ARGS__)
#define TLOG_ERR(...)         TLOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)

// print a captured log to stdout, keeping its tail when it is too long
// returns true when the log was truncated and the file should be kept on disk
static bool test_log_dump(const std::string & path, size_t max_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to read the captured log '%s'\n", path.c_str());
        return false;
    }

    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    size_t skip = 0;
    if (text.size() > max_bytes) {
        skip = text.find('\n', text.size() - max_bytes);
        skip = skip == std::string::npos ? 0 : skip + 1;
    }

    if (skip > 0) {
        const size_t n_all   = std::count(text.begin(), text.end(), '\n');
        const size_t n_shown = std::count(text.begin() + skip, text.end(), '\n');

        fprintf(stdout, "... %zu of %zu lines omitted, full log: %s\n", n_all - n_shown, n_all, path.c_str());
    }

    fwrite(text.data() + skip, 1, text.size() - skip, stdout);
    fflush(stdout);

    return skip > 0;
}

//
// test status
//

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

static const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mPASS\033[0m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[0m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[0m";
    }
    return "";
}

// a skipped case does not fail the run, a failed case always does
static test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

//
// comparison helpers
//

// normalized mean squared error = mse(a, b) / mse(a, 0), infinity if any logit is not finite
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }

        const double diff = (double) a[i] - (double) b[i];

        mse_a_b += diff*diff;
        mse_a_0 += (double) a[i] * (double) a[i];
    }

    if (mse_a_0 == 0.0) {
        return mse_a_b == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }

    return mse_a_b / mse_a_0;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    return nmse(a.data(), b.data(), a.size());
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

// logits produced by a run, compared against by the replay tests
struct generation_result {
    llama_tokens tokens;
    std::vector<std::vector<float>> logits;

    bool empty() const { return tokens.empty(); }
};

//
// decode helpers
//

static bool get_current_logits(llama_context * ctx, std::vector<float> & out) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return false;
    }
    out.assign(logits, logits + n_vocab);
    return true;
}

// decode tokens [p_begin, p_end) of one sequence
static bool decode_seq(llama_context * ctx, const llama_tokens & tokens, llama_pos p_begin, llama_pos p_end,
                       llama_seq_id seq_id, bool last_output) {
    common_batch batch(ctx);
    for (llama_pos pos = p_begin; pos < p_end; ++pos) {
        batch.add(tokens[pos], pos, seq_id, last_output && pos + 1 == p_end);
    }

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, llama_seq_id seq_id = 0) {
    return decode_seq(ctx, tokens, 0, (llama_pos) tokens.size(), seq_id, true);
}

// tokens derived from (seq, pos): the recurrent cases do not use the tokenizer, so they do
// not depend on the prompt or on the prompt length
static llama_token gen_token(int32_t n_vocab, llama_seq_id seq_id, llama_pos pos) {
    return (llama_token) ((7*(uint32_t) pos + 31*(uint32_t) seq_id + 1) % (uint32_t) n_vocab);
}

static bool decode_gen(llama_context * ctx, int32_t n_vocab, llama_seq_id seq_id, llama_pos p_begin, llama_pos p_end) {
    common_batch batch(ctx);
    for (llama_pos pos = p_begin; pos < p_end; ++pos) {
        batch.add(gen_token(n_vocab, seq_id, pos), pos, seq_id, false);
    }

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq_id = 0) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq_id, true);

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

// greedy-free replay: sample for reporting, but feed the expected tokens back so that the
// compared logits stay on the reference trajectory
static bool generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                            generation_result & result) {
    common_batch batch(ctx);

    for (int32_t i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            TLOG_ERR("\n%s: failed to get logits\n", __func__);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);

        TLOGV(LOG_LEVEL_INFO, "%d ", next_token);
        result.tokens.push_back(next_token);
        result.logits.push_back(std::move(logits));

        batch.clear();
        batch.add(next_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    return true;
}

static bool generate_tokens_compare(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                                    const generation_result & expected, double nmse_eps) {
    if (expected.tokens.size() != expected.logits.size() || expected.tokens.size() < (size_t) n_predict) {
        TLOG_ERR("\n%s: invalid expected generation\n", __func__);
        return false;
    }

    common_batch batch(ctx);

    for (int32_t i = 0; i < n_predict; i++) {
        std::vector<float> logits;
        if (!get_current_logits(ctx, logits)) {
            TLOG_ERR("\n%s: failed to get logits\n", __func__);
            return false;
        }
        if (logits.size() != expected.logits[i].size()) {
            TLOG_ERR("\n%s: logits size mismatch at step %d: %zu != %zu\n", __func__, i, logits.size(), expected.logits[i].size());
            return false;
        }

        const double nmse_val = nmse(expected.logits[i], logits);
        TLOG_TRC("%s: step %d nmse = %.6e\n", __func__, i, nmse_val);
        if (nmse_val > nmse_eps) {
            TLOG_ERR("\n%s: error: NMSE at step %d is %.6e (threshold %.1e)\n", __func__, i, nmse_val, nmse_eps);
            return false;
        }

        const auto next_token = llama_sampler_sample(smpl, ctx, -1);
        const auto expected_token = expected.tokens[i];

        TLOGV(LOG_LEVEL_INFO, "%d ", next_token);
        if (next_token != expected_token) {
            TLOG_TRC("%s: sampled token %d differs from expected %d, using expected token\n", __func__, next_token, expected_token);
        }

        batch.clear();
        batch.add(expected_token, n_past, seq_id, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            TLOG_ERR("\n%s: failed to evaluate\n", __func__);
            return false;
        }
        n_past++;
    }

    return true;
}

//
// state helpers
//

static bool get_seq_state(llama_context * ctx, llama_seq_id seq_id, uint32_t flags, std::vector<uint8_t> & state) {
    const size_t state_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
    if (state_size == 0) {
        TLOG_ERR("%s: sequence state is empty\n", __func__);
        return false;
    }

    state.resize(state_size);
    const size_t ncopy = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (ncopy != state.size()) {
        TLOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                __func__, ncopy, state.size());
        return false;
    }

    return true;
}

// collects the cache buffers touched by a state write, so that they can be filled with a
// known pattern before a restore - a restore that reads cells it never wrote then produces
// garbage instead of silently passing
struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

//
// context helpers
//

// per-test context overrides, sentinel values mean "take the value from common_params"
struct ctx_opts {
    uint32_t n_ctx     = 0; // 0 = from params
    uint32_t n_batch   = 0; // 0 = from params
    uint32_t n_ubatch  = 0; // 0 = from params
    uint32_t n_seq_max = 0; // 0 = from params
    uint32_t n_rs_seq  = 0; // 0 = from params, n_batch / n_ubatch are raised to fit it
    int      kv_unified = -1; // -1 = from params, 0 = off, 1 = on

    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO; // AUTO = from params

    ggml_type type_k = GGML_TYPE_COUNT; // COUNT = from params
    ggml_type type_v = GGML_TYPE_COUNT; // COUNT = from params

    uint8_t fill = 0; // fill the cache buffers with this byte, 0 = disabled
};

static llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context_ptr ctx{llama_init_from_model(model, cparams)};
    if (!ctx || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx.get());
    if (!decode_tokens(ctx.get(), llama_tokens(n_tokens, 0))) {
        return nullptr;
    }
    llama_synchronize(ctx.get());
    cache_buffer_collector collector;
    llama_get_memory(ctx.get())->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    if (collector.buffers.empty()) {
        TLOG_ERR("%s: no cache buffers found\n", __func__);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context_ptr make_ctx(llama_model * model, const common_params & params, const ctx_opts & opts) {
    auto cparams = common_context_params_to_llama(params);

    if (opts.n_ctx) {
        cparams.n_ctx = opts.n_ctx;
    }
    if (opts.n_seq_max) {
        cparams.n_seq_max = opts.n_seq_max;
    }
    if (opts.kv_unified >= 0) {
        cparams.kv_unified = opts.kv_unified != 0;
    }
    if (opts.n_rs_seq) {
        cparams.n_rs_seq = opts.n_rs_seq;
        cparams.n_batch  = std::max(cparams.n_batch,  (uint32_t) (opts.n_rs_seq + 1));
        cparams.n_ubatch = std::max(cparams.n_ubatch, (uint32_t) (opts.n_rs_seq + 1));
    }
    if (opts.n_batch) {
        cparams.n_batch = opts.n_batch;
    }
    if (opts.n_ubatch) {
        cparams.n_ubatch = opts.n_ubatch;
    }
    if (opts.fa != LLAMA_FLASH_ATTN_TYPE_AUTO) {
        cparams.flash_attn_type = opts.fa;
    }
    if (opts.type_k != GGML_TYPE_COUNT) {
        cparams.type_k = opts.type_k;
    }
    if (opts.type_v != GGML_TYPE_COUNT) {
        cparams.type_v = opts.type_v;
    }

    return init_ctx(model, cparams, opts.fill);
}

static llama_sampler_ptr make_dist_sampler(uint32_t seed) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler_ptr smpl{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(seed));
    return smpl;
}

//
// test cases
//

// these archs reserve the final pp graph with n_seqs = 1, so any multi-seq graph has a
// different layout and re-reserves by design - they must not be run with a multi-seq batch
// when GGML_SCHED_DEBUG_REALLOC is enabled
// see [TAG_RESERVE_DIAG_DECAY] in llama-context.cpp
static bool arch_reserves_single_seq(llama_model * model) {
    char arch_str[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch_str, sizeof(arch_str));

    return strcmp(arch_str, "kimi-linear") == 0 || strcmp(arch_str, "minimax-01") == 0;
}

// state of a single model run, handed to every test case
struct model_run {
    common_params params;      // per-model copy, out_file is scoped to this model
    std::string   name;        // model file name, used for the table and for temp files
    llama_model * model        = nullptr;
    llama_tokens  tokens;
    generation_result baseline;
    bool          have_baseline = false;
    uint8_t       fill          = 0; // set by the runner before each case
};

// logits replayed after a state load must match the baseline bit-for-bit on real models,
// dummy models drift a little, so the replay cases allow a small NMSE
static const double NMSE_THRESHOLD = 1e-5;


// - decode all but the last token, saving the state to disk
// - decode the last token
// - generate n_predict tokens
static test_status test_baseline(model_run & mr) {
    ctx_opts opts;
    opts.n_seq_max = 2;

    auto ctx  = make_ctx(mr.model, mr.params, opts);
    auto smpl = make_dist_sampler(mr.params.sampling.seed);

    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), mr.tokens, (int) mr.tokens.size(), n_past, mr.params.n_batch, mr.params.out_file, true)) {
        TLOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }

    generation_result result;
    if (!generate_tokens(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, result)) {
        return test_status::FAIL;
    }
    if (result.empty()) {
        return test_status::FAIL;
    }

    mr.baseline     = std::move(result);
    mr.have_baseline = true;

    return test_status::PASS;
}

// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static test_status test_seq_rm_isolated(model_run & mr) {
    ctx_opts opts;
    opts.n_ctx      = 256;
    opts.n_seq_max  = 2;
    opts.kv_unified = 1;

    auto ctx = make_ctx(mr.model, mr.params, opts);
    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    const size_t n_tokens = mr.tokens.size() < 128 ? mr.tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        if (!decode_seq(ctx.get(), mr.tokens, 0, (llama_pos) n_tokens, seq_id, true)) {
            TLOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return test_status::FAIL;
        }
    }

    std::vector<uint8_t> state_before;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return test_status::FAIL;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        TLOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return test_status::FAIL;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return test_status::FAIL;
    }

    if (state_before != state_after) {
        TLOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// load the state saved by the baseline case, replay the last prompt token and compare the
// generated logits against the baseline
static test_status test_state_load(model_run & mr) {
    ctx_opts opts;
    opts.n_seq_max = 2;

    auto ctx  = make_ctx(mr.model, mr.params, opts);
    auto smpl = make_dist_sampler(mr.params.sampling.seed);

    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    llama_tokens unused_sts(mr.tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), mr.params.out_file.c_str(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        TLOG_ERR("\n%s: failed to load state\n", __func__);
        return test_status::FAIL;
    }

    TLOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), mr.tokens.back(), n_past)) {
        return test_status::FAIL;
    }
    n_past++;

    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 0, mr.baseline, NMSE_THRESHOLD)) {
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// migrate the KV of sequence 0 to sequence 1 over the given io path and continue on seq 1
static test_status test_seq_cp(model_run & mr, bool on_device) {
    ctx_opts opts;
    opts.n_seq_max = 2;

    auto ctx  = make_ctx(mr.model, mr.params, opts);
    auto smpl = make_dist_sampler(mr.params.sampling.seed);

    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    TLOGV(LOG_LEVEL_INFO, "io path: %s\n", on_device ? "device" : "host");

    llama_tokens unused_sts(mr.tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), mr.params.out_file.c_str(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        TLOG_ERR("\n%s: failed to load state\n", __func__);
        return test_status::FAIL;
    }

    TLOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), mr.tokens.back(), n_past)) {
        return test_status::FAIL;
    }
    n_past++;

    {
        std::vector<uint8_t> seq_store;
        if (!get_seq_state(ctx.get(), 0, on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE, seq_store)) {
            return test_status::FAIL;
        }
        TLOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, seq_store.size());

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        TLOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1,
                                                         on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE);
        if (nset != seq_store.size()) {
            TLOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return test_status::FAIL;
        }
        TLOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    if (!generate_tokens_compare(ctx.get(), smpl.get(), n_past, mr.params.n_predict, 1, mr.baseline, NMSE_THRESHOLD)) {
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static test_status test_seq_cp_scatter(model_run & mr, bool on_device) {
    ctx_opts opts;
    opts.n_ctx      = 256;
    opts.n_seq_max  = 2;
    opts.kv_unified = 1;

    auto ctx = make_ctx(mr.model, mr.params, opts);
    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    TLOGV(LOG_LEVEL_INFO, "io path: %s\n", on_device ? "device" : "host");

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!decode_one(ctx.get(), mr.tokens[0], 0, 0) ||
        !decode_one(ctx.get(), mr.tokens[1], 1, 0) ||
        !decode_one(ctx.get(), mr.tokens[0], 0, 1) ||
        !decode_one(ctx.get(), mr.tokens[1], 1, 1) ||
        !decode_one(ctx.get(), mr.tokens[2], 2, 0) ||
        !decode_one(ctx.get(), mr.tokens[2], 2, 1)) {
        TLOG_ERR("%s: failed to build interleaved state\n", __func__);
        return test_status::FAIL;
    }

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return test_status::FAIL;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(ctx.get(), 1, flags, state_save)) {
        return test_status::FAIL;
    }
    TLOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        TLOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return test_status::FAIL;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (nset != state_save.size()) {
        TLOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state_save.size());
        return test_status::FAIL;
    }
    TLOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return test_status::FAIL;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    if (state_before.size() != state_after.size() || memcmp(state_before.data(), state_after.data(), state_before.size()) != 0) {
        TLOG_ERR("\n%s: error: restored KV state is not byte-identical to the saved state\n", __func__);
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// save, erase and restore a sequence, then save again and compare the two blobs
// compares blobs rather than generated text: a partially restored cell still decodes to plausible tokens
static test_status test_state_roundtrip(model_run & mr) {
    auto ctx = make_ctx(mr.model, mr.params, ctx_opts());
    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    common_batch batch = common_batch_get_one(ctx.get(), mr.tokens);
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        TLOG_ERR("\n%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }

    std::vector<uint8_t> blob_a(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_a = llama_state_seq_get_data(ctx.get(), blob_a.data(), blob_a.size(), 0);
    if (n_a != blob_a.size()) {
        TLOG_ERR("\n%s: saved %zu bytes, expected %zu\n", __func__, n_a, blob_a.size());
        return test_status::FAIL;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        TLOG_ERR("\n%s: failed to erase seq 0\n", __func__);
        return test_status::FAIL;
    }

    if (llama_state_seq_set_data(ctx.get(), blob_a.data(), blob_a.size(), 0) != blob_a.size()) {
        TLOG_ERR("\n%s: failed to restore seq 0\n", __func__);
        return test_status::FAIL;
    }

    std::vector<uint8_t> blob_b(llama_state_seq_get_size(ctx.get(), 0));
    const size_t n_b = llama_state_seq_get_data(ctx.get(), blob_b.data(), blob_b.size(), 0);
    if (n_b != n_a) {
        TLOG_ERR("\n%s: re-saved %zu bytes, expected %zu\n", __func__, n_b, n_a);
        return test_status::FAIL;
    }

    size_t n_diff = 0;
    size_t i_diff = 0;
    for (size_t i = 0; i < n_a; i++) {
        if (blob_a[i] != blob_b[i]) {
            if (n_diff == 0) {
                i_diff = i;
            }
            n_diff++;
        }
    }

    if (n_diff > 0) {
        TLOG_ERR("\n%s: state changed across a restore: %zu of %zu bytes differ, first at offset %zu\n",
                __func__, n_diff, n_a, i_diff);
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// overwrite the tensor data with 0xff bytes (NaN when read as f16/f32), so that the restore fails
static bool corrupt_state(std::vector<uint8_t> & data) {
    if (data.size() < 3*4096) {
        TLOG_ERR("%s: state of %zu bytes is too small to corrupt\n", __func__, data.size());
        return false;
    }

    std::fill(data.begin() + 4096, data.end() - data.size()/4, 0xff);
    return true;
}

// a failed restore must leave the sequence empty and must not change the logits of other sequences
static test_status test_state_restore_failure(model_run & mr) {
    ctx_opts opts;
    opts.n_ctx      = 256;
    opts.n_seq_max  = 4;
    opts.kv_unified = 1;

    // without flash attention, corrupted data left behind by the restore shows up as NaN logits on the other sequences
    opts.fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    auto ctx = make_ctx(mr.model, mr.params, opts);
    if (!ctx) {
        TLOG_ERR("%s: failed to create context\n", __func__);
        return test_status::FAIL;
    }

    llama_memory_t mem = llama_get_memory(ctx.get());
    if (mem == nullptr) {
        TLOGV(LOG_LEVEL_INFO, "no memory to test\n");
        return test_status::PASS;
    }

    const auto decode = [&](const llama_tokens & inp, llama_seq_id seq_id, std::vector<float> * logits_out) {
        if (!decode_tokens(ctx.get(), inp, seq_id)) {
            TLOG_ERR("%s: failed to decode on sequence %d\n", __func__, seq_id);
            return false;
        }

        if (logits_out && !get_current_logits(ctx.get(), *logits_out)) {
            TLOG_ERR("%s: failed to get logits\n", __func__);
            return false;
        }

        return true;
    };

    const llama_tokens tokens_save  (mr.tokens.begin(), mr.tokens.begin() + std::min<size_t>(24, mr.tokens.size()));
    const llama_tokens tokens_verify(mr.tokens.end() - std::min<size_t>(8, mr.tokens.size()), mr.tokens.end());

    // the cases share a working directory, so the state file is named after the model
    const std::string path = "state-restore-failure." + mr.name + ".tmp.bin";

    llama_memory_clear(mem, true);

    std::vector<float> baseline;
    if (!decode(tokens_verify, 1, &baseline)) {
        return test_status::FAIL;
    }

    const std::vector<std::pair<const char *, std::function<bool()>>> cases = {
        { "buffer", [&]() {
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
            GGML_ASSERT(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == state.size());
            llama_memory_seq_rm(mem, 0, -1, -1);

            if (!corrupt_state(state)) {
                return false;
            }

            return llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0;
        }},
        { "file", [&]() {
            GGML_ASSERT(llama_state_seq_save_file(ctx.get(), path.c_str(), 0, tokens_save.data(), tokens_save.size()) > 0);
            llama_memory_seq_rm(mem, 0, -1, -1);

            std::vector<uint8_t> data;
            {
                std::ifstream f(path, std::ios::binary);
                data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }

            if (!corrupt_state(data)) {
                std::remove(path.c_str());
                return false;
            }

            {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) data.data(), data.size());
            }

            llama_tokens tokens_out(tokens_save.size());
            size_t n_token_count = 0;
            const size_t nread = llama_state_seq_load_file(ctx.get(), path.c_str(), 0, tokens_out.data(), tokens_out.size(), &n_token_count);
            std::remove(path.c_str());

            return nread == 0;
        }},
    };

    for (const auto & [name, restore_failed] : cases) {
        llama_memory_clear(mem, true);

        if (!decode(tokens_save, 0, nullptr)) {
            return test_status::FAIL;
        }

        if (!restore_failed()) {
            TLOG_ERR("%s: %s: restoring a corrupted state did not fail\n", __func__, name);
            return test_status::FAIL;
        }

        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            TLOG_ERR("%s: %s: sequence not empty after failed restore\n", __func__, name);
            return test_status::FAIL;
        }

        std::vector<float> logits;
        if (!decode(tokens_verify, 1, &logits)) {
            return test_status::FAIL;
        }

        float  diff_max = 0.0f;
        size_t n_nan    = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (std::isnan(logits[i]) || std::isnan(baseline[i])) {
                n_nan++;
            } else {
                diff_max = std::max(diff_max, std::fabs(logits[i] - baseline[i]));
            }
        }

        if (n_nan > 0 || diff_max > 1e-6f) {
            TLOG_ERR("%s: %s: logits changed after failed restore (max diff = %g, nan = %zu)\n", __func__, name, diff_max, n_nan);
            return test_status::FAIL;
        }

        TLOG_TRC("%s: %s: logits match (max diff = %g)\n", __func__, name, diff_max);
    }

    return test_status::PASS;
}

// a KV state saved with attention rotation enabled must restore only into a context with the
// same setting
// note: rotation is only active for quantized KV caches with a head size that is a multiple of
//       64, for other models the restore into the rotation-disabled context is valid and the
//       test passes vacuously
static test_status test_state_rotation(model_run & mr) {
    const std::string attn_rot_disable = common_get_env("LLAMA_ATTN_ROT_DISABLE");

    const auto make_context = [&](ggml_type type_k, ggml_type type_v, bool disable_rotation) {
        common_set_env("LLAMA_ATTN_ROT_DISABLE", disable_rotation ? "1" : "0");

        ctx_opts opts;
        opts.n_ctx    = 32;
        opts.n_batch  = 1;
        opts.n_ubatch = 1;
        opts.type_k   = type_k;
        opts.type_v   = type_v;
        opts.fa       = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        return make_ctx(mr.model, mr.params, opts);
    };

    std::vector<std::pair<ggml_type, ggml_type>> type_pairs;
    for (const auto & types : { std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0} }) {
        if (make_context(types.first, types.second, false)) {
            type_pairs.push_back(types);
        }
    }
    if (type_pairs.empty()) {
        TLOG_WRN("%s: no supported quantized KV cache type combination - skipping\n", __func__);
        return test_status::SKIP;
    }

    bool success = true;
    for (const auto & types : type_pairs) {
        auto src = make_context(types.first, types.second, false);
        if (!src) {
            TLOG_ERR("%s: failed to create source context\n", __func__);
            success = false;
            break;
        }

        llama_token token = 0;
        if (llama_decode(src.get(), llama_batch_get_one(&token, 1))) {
            TLOG_ERR("%s: failed to decode token\n", __func__);
            success = false;
            break;
        }

        const size_t state_size = llama_state_seq_get_size(src.get(), 0);
        if (state_size == 0) {
            continue; // no KV state to test
        }

        std::vector<uint8_t> state(state_size);
        if (llama_state_seq_get_data(src.get(), state.data(), state.size(), 0) != state.size()) {
            TLOG_ERR("%s: failed to save sequence state\n", __func__);
            success = false;
            break;
        }

        auto matching = make_context(types.first, types.second, false);
        if (!matching || llama_state_seq_set_data(matching.get(), state.data(), state.size(), 0) != state.size()) {
            TLOG_ERR("%s: failed to restore matching rotation\n", __func__);
            success = false;
            break;
        }

        auto mismatched = make_context(types.first, types.second, true);
        if (!mismatched) {
            TLOG_ERR("%s: failed to create mismatched rotation context\n", __func__);
            success = false;
            break;
        }
        if (llama_state_seq_set_data(mismatched.get(), state.data(), state.size(), 0) != 0) {
            TLOG_TRC("%s: state restored into rotation-disabled context, model does not use attention rotation\n", __func__);
        }
    }
    common_set_env("LLAMA_ATTN_ROT_DISABLE", attn_rot_disable);

    if (!success) {
        return test_status::FAIL;
    }

    return test_status::PASS;
}

// restore a sequence into a fragmented cache: the slots freed by the removed sequence are
// scattered between live cells of the other sequences, so the restore cannot claim one
// contiguous block
// the restored state must come back byte for byte, and the sequence must continue to behave
// exactly as the same sequence in a context where it was never removed
// ref: https://github.com/ggml-org/llama.cpp/issues/17527
static test_status test_state_restore_fragmented(model_run & mr) {
    if (arch_reserves_single_seq(mr.model)) {
        TLOG_INF("%s: skipping, the interleaved batch is a multi-seq graph\n", __func__);
        return test_status::SKIP;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

    ctx_opts opts;
    opts.n_ctx      = 256;
    opts.n_seq_max  = 3;
    opts.kv_unified = 1;
    // the interleaved prompt is a single multi-seq batch
    opts.n_batch    = 3*70;
    opts.n_ubatch   = 3*70;

    // ctx removes and restores seq 1, ctx_keep keeps it untouched as the reference
    llama_context_ptr ctx      = make_ctx(mr.model, mr.params, opts);
    llama_context_ptr ctx_keep = make_ctx(mr.model, mr.params, opts);
    if (!ctx || !ctx_keep) {
        TLOG_ERR("%s: failed to create contexts\n", __func__);
        return test_status::FAIL;
    }

    const llama_pos n_tokens = (llama_pos) std::min<size_t>(70, mr.tokens.size());

    // interleave the 3 sequences: 01201230123... - seq 1 owns every third cell
    const auto decode_interleaved = [&](llama_context * cur) {
        common_batch batch(cur);
        for (llama_pos i = 0; i < n_tokens; i++) {
            for (llama_seq_id s = 0; s < 3; ++s) {
                batch.add(mr.tokens[i], i, s, false);
            }
        }
        batch.set_output(batch.size() - 1, true);

        return llama_process(cur, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };

    if (!decode_interleaved(ctx.get()) || !decode_interleaved(ctx_keep.get())) {
        TLOG_ERR("%s: failed to decode interleaved prompt\n", __func__);
        return test_status::FAIL;
    }

    TLOG_INF("%s: processed the prompt on seq 0, 1, 2 (%d tokens each)\n", __func__, n_tokens);

    std::vector<uint8_t> state_before;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return test_status::FAIL;
    }

    // clearing seq 1 leaves holes where its cells were - no contiguous block large enough
    // for the seq 1 state is left in the cache
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_memory_seq_rm(mem, 1, -1, -1);

    const size_t nset = llama_state_seq_set_data(ctx.get(), state_before.data(), state_before.size(), 1);
    if (nset != state_before.size()) {
        TLOG_ERR("%s: failed to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, state_before.size());
        return test_status::FAIL;
    }

    TLOG_INF("%s: restored seq 1 into a fragmented cache, %zu bytes\n", __func__, nset);

    // the restore has to be lossless - the cells the sequence ends up in are irrelevant, its
    //   state has to come back exactly as it was saved
    std::vector<uint8_t> state_after;
    if (!get_seq_state(ctx.get(), 1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return test_status::FAIL;
    }

    if (state_before != state_after) {
        size_t i = 0;
        while (i < state_before.size() && i < state_after.size() && state_before[i] == state_after[i]) {
            ++i;
        }

        TLOG_ERR("%s: seq 1 state changed by the restore, %zu bytes -> %zu bytes, first difference at byte %zu (%02x -> %02x)\n",
                __func__, state_before.size(), state_after.size(), i,
                i < state_before.size() ? state_before[i] : 0,
                i < state_after.size()  ? state_after[i]  : 0);
        return test_status::FAIL;
    }

    // the restored sequence must continue exactly as the untouched one - decode the same token
    //   on seq 1 in both contexts and compare the logits they produce
    const llama_token token = gen_token(n_vocab, 1, n_tokens);

    if (!decode_one(ctx.get(), token, n_tokens, 1) || !decode_one(ctx_keep.get(), token, n_tokens, 1)) {
        TLOG_ERR("%s: failed to decode with restored state\n", __func__);
        return test_status::FAIL;
    }

    const float * logits      = llama_get_logits(ctx.get());
    const float * logits_keep = llama_get_logits(ctx_keep.get());
    if (!logits || !logits_keep) {
        TLOG_ERR("%s: missing logits after the restore\n", __func__);
        return test_status::FAIL;
    }

    const double diff = nmse(logits, logits_keep, n_vocab);
    if (diff > NMSE_THRESHOLD) {
        size_t i = 0;
        float  max_diff = 0.0f;
        for (int t = 0; t < n_vocab; ++t) {
            const float d = logit_diff(logits[t], logits_keep[t]);
            if (d > max_diff) {
                max_diff = d;
                i = t;
            }
        }

        TLOG_ERR("%s: logits of the restored seq 1 differ from the reference: nmse %g > %g, worst %g at token %zu\n",
                __func__, diff, NMSE_THRESHOLD, max_diff, i);
        return test_status::FAIL;
    }

    TLOG_INF("%s: restored seq 1 continued identically to the reference, nmse %g\n", __func__, diff);

    return test_status::PASS;
}

//
// recurrent state cases - skipped for architectures without a recurrent cache
//

// roll back multiple sequences, then replay them in a single batch whose per-seq token count
// exceeds n_ubatch: each seq's replay spans several ubatches while its rollback restore is
// still pending. Compared against a reference context that never advanced past the rollback
// point and decodes the identical replay batch.
static test_status test_multi_seq_split_replay(model_run & mr) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        ctx_opts opts;
        opts.n_ctx      = 256;
        opts.n_batch    = 256;
        opts.n_ubatch   = n_ubatch;
        opts.n_seq_max  = n_seqs;
        opts.n_rs_seq   = 8;
        opts.kv_unified = 0;
        opts.fill       = mr.fill;
        return make_ctx(mr.model, mr.params, opts);
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        TLOG_ERR("%s: failed to init multi-seq contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        TLOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return gen_token(n_vocab, (llama_seq_id) seq, pos);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, (llama_pos) p0);
        ok = ok && decode_gen(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, (llama_pos) p0);

        ok = ok && decode_gen(ctx_roll.get(), n_vocab, (llama_seq_id) s, (llama_pos) p0, (llama_pos) n_prompt);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        TLOG_ERR("%s: multi-seq prefill/rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // all seqs replay in a single batch
    const auto decode_replay = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (uint32_t i = 0; i < n_replay; ++i) {
                const llama_pos pos = p0 + (llama_pos) i;
                batch.add(tok(s, pos), pos, (llama_seq_id) s, true);
            }
        }
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    };
    ok = decode_replay(ctx_roll.get());
    ok = ok && decode_replay(ctx_ref.get());
    if (!ok) {
        TLOG_ERR("%s: multi-seq replay decode failed\n", __func__);
        return test_status::FAIL;
    }

    // both contexts decode identical batches, so the logits should match;
    // random dummy models can still drift up to ~1.7e-5, so the bound is 1e-4
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            TLOG_ERR("%s: missing multi-seq logits at index %u\n", __func__, i);
            return test_status::FAIL;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        TLOG_ERR("%s: multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        return test_status::FAIL;
    }

    TLOG_INF("%s: multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        common_batch batch_tail(ctx_ref.get());
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            batch_tail.add(tok(0, pos + 7), pos, 0, false);
        }
        ok = llama_process(ctx_ref.get(), LLAMA_PROCESS_TYPE_DECODE, batch_tail.get()) == 0;
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        ok = decode_one(ctx_roll.get(), tok(1, pos), pos, 1);
        ok = ok && decode_one(ctx_ref.get(), tok(1, pos), pos, 1);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        TLOG_ERR("%s: seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        return test_status::FAIL;
    }

    TLOG_INF("%s: seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    return test_status::PASS;
}

// save a rolled-back single-seq state, restore it into fresh and dirty contexts, and verify
// exact logit matches on replay
static test_status test_rollback(model_run & mr) {
    const llama_vocab * vocab   = llama_model_get_vocab(mr.model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const auto make_rs_ctx = [&]() {
        ctx_opts opts;
        opts.n_seq_max = 1;
        opts.n_rs_seq  = 8;
        opts.fill      = mr.fill;
        return make_ctx(mr.model, mr.params, opts);
    };

    llama_context_ptr ctx_src = make_rs_ctx();
    llama_context_ptr ctx_dst = make_rs_ctx();
    if (!ctx_src || !ctx_dst) {
        TLOG_ERR("%s: failed to init contexts\n", __func__);
        return test_status::FAIL;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        TLOG_INF("%s: skipping because n_rs_seq is disabled\n", __func__);
        return test_status::SKIP;
    }

    llama_tokens tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        TLOG_INF("%s: skipping because n_rs_seq is too small\n", __func__);
        return test_status::SKIP;
    }
    if (tokens.empty()) {
        TLOG_ERR("%s: not enough prompt tokens\n", __func__);
        return test_status::FAIL;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens)) {
        TLOG_ERR("%s: failed to decode prompt\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1)) {
        TLOG_ERR("%s: rollback failed\n", __func__);
        return test_status::FAIL;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src.get(), tokens[pos], pos) ||
                !decode_one(ctx_dst.get(), tokens[pos], pos)) {
                TLOG_ERR("%s: %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                TLOG_ERR("%s: missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                TLOG_ERR("%s: %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return test_status::FAIL;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst.get()), 0, rollback_pos, -1)) {
    //    TLOG_ERR("%s: partial rollback failed\n", __func__);
    //    return test_status::FAIL;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src.get(), 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst.get(), 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return test_status::FAIL;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty = make_rs_ctx();
    if (!ctx_dirty) {
        TLOG_ERR("%s: failed to init dirty ctx\n", __func__);
        return test_status::FAIL;
    }

    llama_tokens noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise)) {
        TLOG_ERR("%s: dirty prompt decode failed\n", __func__);
        return test_status::FAIL;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, rollback_pos, -1)) {
        TLOG_ERR("%s: dirty rollback failed\n", __func__);
        return test_status::FAIL;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty.get(), tokens[pos], pos)) {
            TLOG_ERR("%s: dirty replay failed at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
        if (logits_dirty == nullptr) {
            TLOG_ERR("%s: missing dirty logits at position %d\n", __func__, pos);
            return test_status::FAIL;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            TLOG_ERR("%s: dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return test_status::FAIL;
        }
    }

    TLOG_INF("%s: recurrent rollback checkpoint restored successfully\n", __func__);
    return test_status::PASS;
}

// decode a prompt into seq 0, share its cells with a second sequence, then keep decoding both
static test_status test_shared_seq_reserve(model_run & mr) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mr.model));

    if (arch_reserves_single_seq(mr.model)) {
        TLOG_INF("%s: skipping, the shared-seq batch is a multi-seq graph\n", __func__);
        return test_status::SKIP;
    }

    constexpr uint32_t n_seqs     = 2;
    constexpr uint32_t n_prompt   = 128;
    constexpr uint32_t n_continue = 32;

    ctx_opts opts;
    opts.n_ctx      = 512;
    opts.n_batch    = 256;
    opts.n_ubatch   = 64;
    opts.n_seq_max  = n_seqs;
    opts.kv_unified = 1; // only a unified cache shares cells on seq_cp
    opts.fill       = mr.fill;

    auto ctx = make_ctx(mr.model, mr.params, opts);
    if (!ctx) {
        TLOG_ERR("%s: failed to init context\n", __func__);
        return test_status::FAIL;
    }

    if (!decode_gen(ctx.get(), n_vocab, 0, 0, (llama_pos) n_prompt)) {
        TLOG_ERR("%s: prompt decode failed\n", __func__);
        return test_status::FAIL;
    }

    // this is what llama-batched-bench does for -pps
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);

    for (uint32_t i = 0; i < n_continue; ++i) {
        const llama_pos pos = (llama_pos) (n_prompt + i);

        common_batch batch(ctx.get());
        for (uint32_t s = 0; s < n_seqs; ++s) {
            batch.add(gen_token(n_vocab, (llama_seq_id) s, pos), pos, (llama_seq_id) s, true);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            TLOG_ERR("%s: shared-seq decode failed at step %u\n", __func__, i);
            return test_status::FAIL;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx.get(), (int) s);
            if (logits == nullptr) {
                TLOG_ERR("%s: missing shared-seq logits at index %u\n", __func__, s);
                return test_status::FAIL;
            }
            for (int t = 0; t < n_vocab; ++t) {
                if (!std::isfinite(logits[t])) {
                    TLOG_ERR("%s: non-finite shared-seq logit at step %u, seq %u, index %d\n", __func__, i, s, t);
                    return test_status::FAIL;
                }
            }
        }
    }

    TLOG_INF("%s: shared-seq decode succeeded (%u tokens after seq_cp)\n", __func__, n_continue*n_seqs);
    return test_status::PASS;
}

//
// test registry
//

enum test_group {
    TEST_GROUP_STATE, // KV state save / load / copy
    TEST_GROUP_RS,    // recurrent state, also run against a dirty cache
};

struct test_case {
    const char * name;         // column header of the --models table
    test_group   group;
    bool         needs_baseline; // requires the state file and the logits of the baseline case
    test_status (*run)(model_run &);
};

static const std::vector<test_case> test_cases = {
    { "baseline",   TEST_GROUP_STATE, false, [](model_run & mr) { return test_baseline(mr); } },
    { "seq_rm",     TEST_GROUP_STATE, false, [](model_run & mr) { return test_seq_rm_isolated(mr); } },
    { "state_load", TEST_GROUP_STATE, true,  [](model_run & mr) { return test_state_load(mr); } },
    { "cp_h",       TEST_GROUP_STATE, true,  [](model_run & mr) { return test_seq_cp(mr, false); } },
    { "cp_d",       TEST_GROUP_STATE, true,  [](model_run & mr) { return test_seq_cp(mr, true); } },
    { "cp_h_s",     TEST_GROUP_STATE, false, [](model_run & mr) { return test_seq_cp_scatter(mr, false); } },
    { "cp_d_s",     TEST_GROUP_STATE, false, [](model_run & mr) { return test_seq_cp_scatter(mr, true); } },
    { "rt",         TEST_GROUP_STATE, false, [](model_run & mr) { return test_state_roundtrip(mr); } },
    { "rf",         TEST_GROUP_STATE, false, [](model_run & mr) { return test_state_restore_failure(mr); } },
    { "rot",        TEST_GROUP_STATE, false, [](model_run & mr) { return test_state_rotation(mr); } },
    { "frag",       TEST_GROUP_STATE, false, [](model_run & mr) { return test_state_restore_fragmented(mr); } },
    { "rollback",   TEST_GROUP_RS,    false, [](model_run & mr) { return test_rollback(mr); } },
    { "replay",     TEST_GROUP_RS,    false, [](model_run & mr) { return test_multi_seq_split_replay(mr); } },
    { "shared",     TEST_GROUP_RS,    false, [](model_run & mr) { return test_shared_seq_reserve(mr); } },
};

// run every registered case for one model
static std::vector<test_status> run_cases(model_run & mr) {
    const bool is_rs = llama_model_is_recurrent(mr.model) || llama_model_is_hybrid(mr.model);

    // the state cases keep the historical default of a unified KV cache when running a
    // single sequence, the recurrent cases use whatever was passed on the command line
    const bool unified_base = mr.params.kv_unified;
    const bool unified_state = mr.params.n_parallel == 1 ? true : unified_base;

    std::vector<test_status> results;
    results.reserve(test_cases.size());

    for (const auto & tc : test_cases) {
        test_status res;

        mr.params.kv_unified = tc.group == TEST_GROUP_STATE ? unified_state : unified_base;

        if (tc.group == TEST_GROUP_RS && !is_rs) {
            TLOGV(LOG_LEVEL_INFO, "\n=== %s === skipped: not a recurrent model\n", tc.name);
            res = test_status::SKIP;
        } else if (tc.needs_baseline && !mr.have_baseline) {
            TLOGV(LOG_LEVEL_INFO, "\n=== %s === skipped: no baseline\n", tc.name);
            res = test_status::SKIP;
        } else if (tc.group == TEST_GROUP_RS) {
            // the recurrent cases are also run against a cache pre-filled with a known
            // pattern, so that a restore reading cells it never wrote cannot pass
            res = test_status::SKIP;
            for (uint8_t fill : { 0, 0x3e }) {
                TLOGV(LOG_LEVEL_INFO, "\n=== %s (fill 0x%02x) ===\n", tc.name, fill);
                mr.fill = fill;
                res = merge_status(res, tc.run(mr));
                if (res == test_status::FAIL) {
                    break;
                }
            }
        } else {
            TLOGV(LOG_LEVEL_INFO, "\n=== %s ===\n", tc.name);
            mr.fill = 0;
            res = tc.run(mr);
        }

        TLOGV(LOG_LEVEL_INFO, "%s\n", test_status_str(res));
        results.push_back(res);

        // keep the captured log usable even if the next case crashes the process
        g_test_log.flush();
    }

    return results;
}

// tokenize the prompt, or generate random tokens for models without a tokenizer
static llama_tokens prepare_tokens(llama_model * model, const common_params & params) {
    llama_tokens tokens;

    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        TLOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto   n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        TLOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    TLOG_INF("%s: the input prompt is %d tokens\n", __func__, (int) tokens.size());

    return tokens;
}

static std::vector<std::string> collect_models(const std::string & models_dir) {
    std::vector<std::string> models;

    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());

    return models;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -lv 5\n", argv[0]);
    LOG("\nspecial flags:\n");
    LOG("\n  --models DIR              run every .gguf in DIR and print a result table\n");
    LOG("\n  --logs DIR                keep the log of each model in DIR instead of discarding it\n");
    LOG("\n  --logs-on-fail always     after the table, print the log of every failed model (default)\n");
    LOG("\n  --logs-on-fail never      never print the logs of the failed models\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own options before handing the rest to the common arg parser
    std::string models_dir;
    std::string logs_dir;
    bool dump_logs_on_fail = true;

    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string value;

        const int res_models = test_opt(argv, i, argc, "--models", value);
        if (res_models != 0) {
            if (res_models < 0) {
                fprintf(stderr, "--models requires a directory argument\n");
                return 1;
            }
            models_dir = value;
            continue;
        }

        const int res_logs = test_opt(argv, i, argc, "--logs", value);
        if (res_logs != 0) {
            if (res_logs < 0) {
                fprintf(stderr, "--logs requires a directory argument\n");
                return 1;
            }
            logs_dir = value;
            continue;
        }

        if (test_opt(argv, i, argc, "--logs-on-fail", value) == 0) {
            filtered_argv.push_back(argv[i]);
            continue;
        }

        if (value == "always") {
            dump_logs_on_fail = true;
        } else if (value == "never") {
            dump_logs_on_fail = false;
        } else {
            fprintf(stderr, "--logs-on-fail: expected 'always' or 'never', got '%s'\n", value.c_str());
            return 1;
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // -lv applies to the test's own logging too
    g_test_log.set_threshold(params.verbosity);

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    // capture the logs of each model to a file, so that they can be dumped when a model fails:
    // the table stays clean, but the details of a failure stay close at hand
    if (!logs_dir.empty() && !std::filesystem::create_directories(logs_dir)) {
        fprintf(stderr, "failed to create the log directory '%s'\n", logs_dir.c_str());
        return 1;
    }

    if (!models_dir.empty() || !logs_dir.empty()) {
        // capture everything but the debug chatter of the graph builder - it is orders of
        //   magnitude more than everything else and buries the messages of the failing case
        // note: -lv raises this, use -lv 5 to capture the graph debug as well
        g_test_log.set_threshold(std::max(params.verbosity, LOG_LEVEL_INFO));
        g_test_log.capturing = true;
        llama_log_set(test_log_callback, nullptr);

        // a model that aborts would take the run, and with it the log of the very model that
        //   aborted, down with it - print what was captured so far before going away
#if TEST_LOG_ABORT_DUMP
        signal(SIGABRT, test_log_abort);
        signal(SIGSEGV, test_log_abort);
        signal(SIGFPE,  test_log_abort);
        signal(SIGILL,  test_log_abort);
#  ifdef SIGBUS
        signal(SIGBUS,  test_log_abort);
#  endif
#endif // TEST_LOG_ABORT_DUMP
    }

    llama_backend_init();

    // run the full suite for a single model
    if (models_dir.empty()) {
        model_run mr;
        mr.params          = params;
        mr.name            = std::filesystem::path(params.model.path).filename().string();
        mr.params.out_file = test_state_path(mr.name);

        if (!logs_dir.empty()) {
            g_test_log.open(test_log_path(logs_dir, mr.name));
        }

        auto llama_init = common_init_from_params(params, true);

        GGML_ASSERT(llama_init->context() == nullptr);
        if (llama_init->model() == nullptr) {
            TLOG_ERR("%s: failed to init model '%s'\n", __func__, params.model.path.c_str());
            g_test_log.close();
            return 1;
        }

        mr.model  = llama_init->model();
        mr.tokens = prepare_tokens(mr.model, mr.params);

        const auto results = run_cases(mr);
        g_test_log.close();

        std::remove(mr.params.out_file.c_str()); // the cases leave the session file behind

        const bool failed = std::find(results.begin(), results.end(), test_status::FAIL) != results.end();
        if (!failed) {
            LOG("\nAll applicable tests passed.\n");
        }
        return failed ? 1 : 0;
    }

    // table mode: run the suite for every model in the directory
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        TLOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
        return 1;
    }

    const auto models = collect_models(models_dir);
    if (models.empty()) {
        TLOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
        return 1;
    }

    auto col_width = [](const char * name) { return (int) std::max(strlen(name), (size_t) 4); };

    size_t name_width = 5; // "Model"
    for (const auto & model_path : models) {
        name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
    }

    // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
    common_log_set_verbosity_thold(0);

    LOG("%-*s", (int) name_width, "Model");
    for (const auto & tc : test_cases) {
        LOG("  %-*s", col_width(tc.name), tc.name);
    }
    LOG("\n");
    common_log_flush(common_log_main());

    std::vector<size_t> n_pass(test_cases.size(), 0);
    std::vector<size_t> n_skip(test_cases.size(), 0);
    std::vector<size_t> n_fail(test_cases.size(), 0);

    size_t n_model_fail = 0;

    for (const auto & model_path : models) {
        const auto name = std::filesystem::path(model_path).filename().string();

        LOG("%-*s", (int) name_width, name.c_str());
        common_log_flush(common_log_main());

        std::vector<test_status> results;
        results.reserve(test_cases.size());

        // capture this model from the moment it is loaded, so that even a load failure has a log
        const std::string log_path = test_log_path(logs_dir, name);
        g_test_log.open(log_path);

        {
            struct common_params model_params = params;
            model_params.model.path = model_path;

            // a model that cannot be loaded is a failure, not a skip
            auto llama_init = common_init_from_params(model_params, true);
            if (llama_init->model() == nullptr) {
                TLOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
                results.assign(test_cases.size(), test_status::FAIL);
            } else {
                GGML_ASSERT(llama_init->context() == nullptr);

                model_run mr;
                mr.params              = std::move(model_params);
                mr.name                = name;
                mr.model               = llama_init->model();
                // the cases share a working directory, so state files are named after the model
                mr.params.out_file     = test_state_path(name);
                mr.tokens              = prepare_tokens(mr.model, mr.params);

                results = run_cases(mr);
            }
        }

        for (size_t i = 0; i < results.size(); i++) {
            LOG("  %s%*s", test_status_str(results[i]), col_width(test_cases[i].name) - 4, "");

            switch (results[i]) {
                case test_status::PASS: n_pass[i]++; break;
                case test_status::FAIL: n_fail[i]++; break;
                case test_status::SKIP: n_skip[i]++; break;
            }
        }
        LOG("\n");
        common_log_flush(common_log_main());

        const size_t n_fail_model = std::count(results.begin(), results.end(), test_status::FAIL);
        n_model_fail += n_fail_model > 0;

        g_test_log.close();
        std::remove(test_state_path(name).c_str());     // the cases leave the session file behind

        // dump the log of a failed model once the table is printed - interleaving it with the
        //   rows would make the table unreadable; keep the file when the dump was truncated
        if (n_fail_model > 0 && dump_logs_on_fail) {
            LOG("\n=== logs: %s (%s) ===\n", name.c_str(), log_path.c_str());
            common_log_flush(common_log_main());

            // keep the file when the dump was truncated, or when a log directory was requested
            if (!test_log_dump(log_path, LOG_DUMP_MAX_BYTES) && logs_dir.empty()) {
                std::remove(log_path.c_str());
            }
        } else if (logs_dir.empty()) {
            std::remove(log_path.c_str());
        }
    }

    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    common_log_flush(common_log_main());

    for (size_t i = 0; i < test_cases.size(); i++) {
        if (n_fail[i] == 0 && n_skip[i] == 0) {
            continue;
        }
        LOG_INF("%s: %-13s %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, test_cases[i].name, n_pass[i], n_skip[i], n_fail[i], models.size());
    }
    LOG_INF("%s: models: %zu passed, %zu failed (of %zu)\n", __func__, models.size() - n_model_fail, n_model_fail, models.size());

    return n_model_fail == 0 ? 0 : 1;
}
