#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <signal.h>
#endif

static const uint32_t n_vocab = 256;
static const uint32_t n_embd  = 256;
static const uint32_t n_ff    = 512;
static const uint32_t n_head  = 2;
static const uint32_t n_layer = 4;
static const float target_bpw = 4.5f;
static bool verbose = false;

static void set_tensor_data(ggml_tensor * tensor, void * /*userdata*/) {
    std::mt19937 gen(std::hash<std::string>{}(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);
    std::vector<float> data(ggml_nelements(tensor));
    for (float & v : data) { v = dis(gen); }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

// test model
static bool save_model(const std::string & fname) {
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture",                   "llama");
    gguf_set_val_u32(meta.get(), "llama.vocab_size",                       n_vocab);
    gguf_set_val_u32(meta.get(), "llama.context_length",                   256);
    gguf_set_val_u32(meta.get(), "llama.embedding_length",                 n_embd);
    gguf_set_val_u32(meta.get(), "llama.feed_forward_length",              n_ff);
    gguf_set_val_u32(meta.get(), "llama.block_count",                      n_layer);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count",             n_head);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv",          n_head);
    gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model",                   "none");

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t devs[] = { nullptr };
    mparams.devices = devs;

    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, nullptr, mparams));
    if (!model) { return false; }
    llama_model_save_to_file(model.get(), fname.c_str());
    return true;
}

// test imatrix
struct imatrix_data {
    std::vector<std::string>        names;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> statistics;
    std::vector<std::string>        stats_schema;
};

// stats schema
static const std::vector<std::string> default_stats_schema = {
    "sum_sq", "mean", "elements", "std_deviation", "skewness", "kurtosis", "gain", "h_norm", "l2_dist", "cossim", "pearson", "covariance"
};

static std::vector<float> make_stats(std::mt19937 & gen, const int64_t n_per_row) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    return {
        100.0f + 100.0f * u(gen), // sum_sq
        0.01f * (u(gen) - 0.5f),  // mean
        (float) n_per_row,        // elements
        1.0f + u(gen),            // std_deviation
        u(gen) - 0.5f,            // skewness
        3.0f * u(gen),            // kurtosis
        1.0f + u(gen),            // gain
        80.0f + 20.0f * u(gen),   // h_norm
        10.0f * u(gen),           // l2_dist
        0.5f + 0.5f * u(gen),     // cossim
        0.5f + 0.5f * u(gen),     // pearson
        u(gen),                   // covariance
    };
}

static imatrix_data make_imatrix(const std::string & fname_model, const uint32_t seed) {
    imatrix_data im;

    ggml_context * ctx_meta = nullptr;
    const gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    gguf_context_ptr ctx(gguf_init_from_file(fname_model.c_str(), params));
    ggml_context_ptr meta(ctx_meta);
    if (!ctx) { return im; }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); i++) {
        const char * name = gguf_get_tensor_name(ctx.get(), i);
        const ggml_tensor * tensor = ggml_get_tensor(meta.get(), name);
        if (ggml_n_dims(tensor) < 2 || strcmp(name, "token_embd.weight") == 0) { continue; }

        const int64_t n_per_row = tensor->ne[0];
        std::vector<float> values(n_per_row);
        std::vector<float> activations(n_per_row);
        for (int64_t j = 0; j < n_per_row; j++) {
            activations[j] = u(gen) - 0.5f;
            values[j] = activations[j] * activations[j] + std::exp(2.0f * u(gen) - 1.0f);
        }

        im.names.push_back(name);
        im.values.push_back(std::move(values));
        im.activations.push_back(std::move(activations));
        im.statistics.push_back(make_stats(gen, n_per_row));
    }

    im.stats_schema = default_stats_schema;

    return im;
}

static std::vector<llama_model_imatrix_data> imatrix_list(const imatrix_data & im, const std::vector<std::vector<float>> & data) {
    std::vector<llama_model_imatrix_data> list;
    for (size_t i = 0; i < im.names.size(); i++) {
        if (data[i].empty()) { continue; }
        list.push_back({ im.names[i].c_str(), data[i].data(), data[i].size() });
    }

    list.push_back({ nullptr, nullptr, 0 });

    return list;
}

struct quant_run {
    uint32_t rc = 1; // 0 when quantization succeeds and output is processed
    std::string log;
    std::map<std::string, ggml_type> types;
    size_t n_bytes = 0; // tensor data
    size_t n_elements = 0;
    size_t file_size = 0;
};

static void log_callback(ggml_log_level /*level*/, const char * text, void * user_data) {
    if (user_data) { static_cast<std::string *>(user_data)->append(text); }
    if (verbose) { fputs(text, stderr); }
}

static quant_run quantize(const std::string & fname_inp, const std::string & fname_out, const imatrix_data & im, llama_model_quantize_params params) {
    const auto values      = imatrix_list(im, im.values);
    const auto activations = imatrix_list(im, im.activations);
    const auto statistics  = imatrix_list(im, im.statistics);
    std::vector<const char *> stats_schema;
    for (const auto & name : im.stats_schema) { stats_schema.push_back(name.c_str()); }
    stats_schema.push_back(nullptr);
    params.imatrix      = values.data();
    params.activations  = activations.data();
    params.statistics   = statistics.data();
    params.stats_schema = im.stats_schema.empty() ? nullptr : stats_schema.data();

    quant_run run;
    // clear previous run's output
    std::filesystem::remove(fname_out);
    llama_log_set(log_callback, &run.log);
    run.rc = llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params);
    llama_log_set(log_callback, nullptr);
    if (run.rc != 0) { return run; }

    ggml_context * ctx_meta = nullptr;
    const gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    gguf_context_ptr ctx(gguf_init_from_file(fname_out.c_str(), gparams));
    ggml_context_ptr meta(ctx_meta);
    if (!ctx) {
        run.rc = 1;
        return run;
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); i++) {
        const char * name = gguf_get_tensor_name(ctx.get(), i);
        run.types[name] = gguf_get_tensor_type(ctx.get(), i);
        run.n_bytes    += gguf_get_tensor_size(ctx.get(), i);
        run.n_elements += ggml_nelements(ggml_get_tensor(meta.get(), name));
    }

    run.file_size = std::filesystem::file_size(fname_out);

    return run;
}

static size_t count_matches(const std::string & log, const std::string & text) {
    size_t n = 0;
    for (size_t pos = log.find(text); pos != std::string::npos; pos = log.find(text, pos + text.size())) { n++; }

    return n;
}

static size_t logged_number(const std::string & log, const std::string & text) {
    const size_t pos = log.find(text);
    return pos == std::string::npos ? SIZE_MAX : (size_t) std::strtoull(log.c_str() + pos + text.size(), nullptr, 10);
}

static size_t count_type(const quant_run & run, const ggml_type type) {
    size_t n = 0;
    for (const auto & it : run.types) { n += it.second == type; }

    return n;
}

static size_t count_type_changes(const quant_run & a, const quant_run & b) {
    size_t n = 0;
    for (const auto & it : a.types) {
        const auto other = b.types.find(it.first);
        n += other == b.types.end() || other->second != it.second;
    }

    return n;
}

// tensor quantization mix rows
static std::vector<std::string> mix_rows(const quant_run & run) {
    std::vector<std::string> rows;
    for (size_t pos = run.log.find("\terror: "); pos != std::string::npos; pos = run.log.find("\terror: ", pos + 1)) {
        const size_t start = run.log.rfind('\n', pos) + 1;
        rows.push_back(run.log.substr(start, run.log.find('\n', pos) - start));
    }

    return rows;
}

static size_t count_row_changes(const quant_run & a, const quant_run & b) {
    const std::vector<std::string> rows_a = mix_rows(a);
    const std::vector<std::string> rows_b = mix_rows(b);
    if (rows_a.empty() || rows_a.size() != rows_b.size()) { return SIZE_MAX; }

    size_t n = 0;
    for (size_t i = 0; i < rows_a.size(); i++) { n += rows_a[i] != rows_b[i]; }

    return n;
}

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static void write_file(const std::string & fname, const std::string & data) {
    std::ofstream file(fname, std::ios::binary);
    file << data;
}

static const size_t state_header_size    = 24; // magic, model id, tensor count
static const size_t state_entry_size     = 28; // candidate count, choice, min and max bpw, elements
static const size_t state_candidate_size = 24; // type, bpw, bytes, error

// offset of the first tensor's candidate count; 0 if the checkpoint holds no candidate
static size_t first_count_offset(const std::string & state) {
    uint32_t len = 0;
    if (state.size() < state_header_size + sizeof(len)) { return 0; }
    memcpy(&len, state.data() + state_header_size, sizeof(len));
    const size_t offset = state_header_size + sizeof(len) + len;

    return state.size() >= offset + state_entry_size + state_candidate_size ? offset : 0;
}

struct fixture {
    std::string  fname_model;
    std::string  fname_out;
    std::string  fname_state;
    imatrix_data im;
    quant_run    base;
};

// first run parameters
static llama_model_quantize_params base_params(const fixture & f) {
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.nthread    = 4;
    params.target_bpw = target_bpw;
    params.state_file = f.fname_state.c_str();

    return params;
}

static std::string sibling_path(const fixture & f, const std::string & name) {
    return (std::filesystem::path(f.fname_state).parent_path() / name).string();
}

static void test_target_bpw(testing & t, const fixture & f) {
    const double bpw = 8.0 * f.base.n_bytes / f.base.n_elements;
    t.log("final bpw " + std::to_string(bpw));
    t.assert_true("bpw does not exceed the target", bpw <= target_bpw);
    t.assert_true("bpw is within 0.05 of the target", bpw > target_bpw - 0.05);
}

static void test_state_checkpoint(testing & t, const fixture & f) {
    const size_t n_scored = count_matches(f.base.log, "processing tensor");
    const std::string n_tensors = std::to_string(n_scored) + " tensors";
    t.assert_true("first run scores tensors", n_scored > 0);
    t.assert_equal("first run saves the checkpoint", (size_t) 1, count_matches(f.base.log, "saved target progress for " + n_tensors + " to " + f.fname_state));

    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, base_params(f));
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("second run loads the checkpoint", (size_t) 1, count_matches(run.log, "using " + f.fname_state + " (data for " + n_tensors + " loaded)"));
    t.assert_equal("second run scores no tensor", (size_t) 0, count_matches(run.log, "processing tensor"));
    t.assert_equal("tensors whose type changed", (size_t) 0, count_type_changes(f.base, run));
}

static void test_target_size(testing & t, const fixture & f) {
    llama_model_quantize_params params = base_params(f);
    params.target_bpw  = -1.0f;
    params.target_size = (int64_t) f.base.file_size * 9 / 10;

    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);

    const int64_t diff = (int64_t) run.file_size - params.target_size;
    t.log("file size " + std::to_string(run.file_size) + ", " + std::to_string(diff) + " bytes from the target");
    t.assert_true("the file is within 8 KiB of the target", std::abs(diff) <= 8 * 1024);
}

static void test_excluded_type(testing & t, const fixture & f) {
    ggml_type excluded = GGML_TYPE_COUNT;
    for (const auto & it : f.base.types) {
        if (it.second != GGML_TYPE_F32 && (excluded == GGML_TYPE_COUNT || count_type(f.base, it.second) > count_type(f.base, excluded))) {
            excluded = it.second;
        }
    }

    t.assert_true("first run selects a type", excluded != GGML_TYPE_COUNT);
    if (excluded == GGML_TYPE_COUNT) { return; }
    t.log(std::string("excluding ") + ggml_type_name(excluded) + ", the type of " + std::to_string(count_type(f.base, excluded)) + " tensors");

    const ggml_type exclude[] = { excluded, GGML_TYPE_COUNT };
    llama_model_quantize_params params = base_params(f);
    params.target_exclude = exclude;

    params.state_file = nullptr;
    const quant_run scored = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, scored.rc);
    t.assert_equal("scored with excluded types, tensors of the excluded type", (size_t) 0, count_type(scored, excluded));

    params.state_file = f.fname_state.c_str();
    const std::string n_tensors = std::to_string(count_matches(f.base.log, "processing tensor")) + " tensors";
    const quant_run loaded = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, loaded.rc);
    t.assert_equal("run loads the checkpoint", (size_t) 1, count_matches(loaded.log, "using " + f.fname_state + " (data for " + n_tensors + " loaded)"));
    t.assert_equal("run scores no tensor", (size_t) 0, count_matches(loaded.log, "processing tensor"));
    t.assert_equal("loaded from checkpoint, tensors of the excluded type", (size_t) 0, count_type(loaded, excluded));
    t.assert_equal("run warns that the checkpoint is not saved", (size_t) 1, count_matches(loaded.log, "state checkpoint cannot be saved while --target-exclude is active"));
    t.assert_equal("checkpoint is not saved while types are excluded", (size_t) 0, count_matches(loaded.log, "saved target progress"));
}

static void test_unknown_type(testing & t, const fixture & f) {
    std::string state = read_file(f.fname_state);
    const size_t offset = first_count_offset(state);
    t.assert_true("first run's checkpoint holds a candidate", offset > 0);
    if (offset == 0) { return; }

    // type 4 was Q4_2; within ggml_type range but invalid
    const int32_t type = 4;
    memcpy(&state[offset + state_entry_size], &type, sizeof(type));
    const std::string fname = sibling_path(f, "unknown.bpw_state");
    write_file(fname, state);

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("unknown type is reported", (size_t) 1, count_matches(run.log, "candidate type is not in the selection pool, ignoring state file"));
    t.assert_equal("unknown checkpoint is not loaded", (size_t) 0, count_matches(run.log, "using " + fname));
    t.assert_equal("unknown checkpoint is scored again", count_matches(f.base.log, "processing tensor"), count_matches(run.log, "processing tensor"));
    t.assert_equal("scored again, tensors whose type changed", (size_t) 0, count_type_changes(f.base, run));
}

static void test_zero_candidates(testing & t, const fixture & f) {
    std::string state = read_file(f.fname_state);
    const size_t offset = first_count_offset(state);
    t.assert_true("the first run's checkpoint holds a candidate", offset > 0);
    if (offset == 0) { return; }

    // drop the first tensor's candidates
    uint64_t n = 0;
    memcpy(&n, &state[offset], sizeof(n));
    state.erase(offset + state_entry_size, n * state_candidate_size);
    n = 0;
    memcpy(&state[offset], &n, sizeof(n));
    const std::string fname = sibling_path(f, "zero.bpw_state");
    write_file(fname, state);

    // with a type excluded, an empty candidate list that is loaded fails the call instead of crashing the test
    const ggml_type exclude[] = { GGML_TYPE_IQ1_S, GGML_TYPE_COUNT };
    llama_model_quantize_params params = base_params(f);
    params.state_file     = fname.c_str();
    params.target_exclude = exclude;
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("empty candidate list is reported", (size_t) 1, count_matches(run.log, "candidate count out of range, ignoring state file"));
    t.assert_equal("checkpoint with an empty candidate list is not loaded", (size_t) 0, count_matches(run.log, "using " + fname));
    t.assert_equal("checkpoint with an empty candidate list is scored again", count_matches(f.base.log, "processing tensor"), count_matches(run.log, "processing tensor"));
}

static void test_truncated(testing & t, const fixture & f) {
    const std::string state = read_file(f.fname_state);
    const size_t offset = first_count_offset(state);
    t.assert_true("first run's checkpoint has a candidate", offset > 0);
    if (offset == 0) { return; }

    // cut before the first candidate count, then before the first candidate type
    const std::string fname = sibling_path(f, "truncated.bpw_state");
    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    write_file(fname, state.substr(0, offset));
    const quant_run at_count = quantize(f.fname_model, f.fname_out, f.im, params);
    write_file(fname, state.substr(0, offset + state_entry_size));
    const quant_run at_type = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, at_count.rc);
    t.assert_equal("checkpoint cut before a candidate count is reported as truncated", (size_t) 1, count_matches(at_count.log, "state file truncated or read error, ignoring"));
    t.assert_equal("quantize succeeds", 0u, at_type.rc);
    t.assert_equal("checkpoint cut before a candidate type is reported as truncated", (size_t) 1, count_matches(at_type.log, "state file truncated or read error, ignoring"));
}

static void test_missing_dir(testing & t, const fixture & f) {
    const std::string dir   = sibling_path(f, "no-such-dir");
    const std::string fname = (std::filesystem::path(dir) / "model.bpw_state").string();

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("path in missing directory fails", 1u, run.rc);
    t.assert_equal("path in missing directory is reported", (size_t) 1, count_matches(run.log, "failed to open " + fname + ": "));
    t.assert_equal("path in missing directory stops execution before scoring", (size_t) 0, count_matches(run.log, "processing tensor"));
    t.assert_true("missing directory is not created", !std::filesystem::exists(dir));
}

static void test_unreadable_file(testing & t, const fixture & f) {
#ifdef _WIN32
    GGML_UNUSED(f);
    t.skip("POSIX permissions only");
#else
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string fname = sibling_path(f, "locked.bpw_state");
    write_file(fname, "forbidden");
    // write-only, so that a probe that opens the file for writing could still damage it
    fs::permissions(fname, fs::perms::owner_write, ec);
    if (std::ifstream(fname).good()) {
        fs::permissions(fname, fs::perms::owner_read | fs::perms::owner_write, ec);
        t.skip("the file can still be read, as by root");
        return;
    }

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    const bool kept_perms = fs::status(fname, ec).permissions() == fs::perms::owner_write;
    fs::permissions(fname, fs::perms::owner_read | fs::perms::owner_write, ec);
    t.assert_equal("an unreadable file fails the call", 1u, run.rc);
    t.assert_equal("an unreadable file is reported", (size_t) 1, count_matches(run.log, "failed to open " + fname + ": "));
    t.assert_equal("an unreadable file stops the run before scoring", (size_t) 0, count_matches(run.log, "processing tensor"));
    t.assert_true("an unreadable file keeps its permissions", kept_perms);
    t.assert_true("an unreadable file keeps its content", read_file(fname) == "forbidden");
#endif
}

static void test_directory_path(testing & t, const fixture & f) {
    const std::string fname = sibling_path(f, "dir.bpw_state");
    std::filesystem::create_directory(fname);

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("a directory instead of a file fails the call", 1u, run.rc);
    t.assert_equal("a directory instead of a file is reported", (size_t) 1, count_matches(run.log, "failed to open " + fname + ": "));
    t.assert_equal("a directory stops the run before scoring", (size_t) 0, count_matches(run.log, "processing tensor"));
    t.assert_true("the directory is left as it was", std::filesystem::is_directory(fname) && std::filesystem::is_empty(fname));
}

static void test_save_error(testing & t, const fixture & f) {
    const std::string fname = sibling_path(f, "unsaved.bpw_state");
    write_file(fname, read_file(f.fname_state));
    std::filesystem::create_directory(fname + ".tmp");

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("a failed save is reported", (size_t) 1, count_matches(run.log, "failed to save progress to " + fname + ".tmp (open error)"));
    t.assert_equal("a failed save logs no saved checkpoint", (size_t) 0, count_matches(run.log, "saved target progress"));
}

static void test_imatrix_swap(testing & t, const fixture & f) {
    const imatrix_data im = make_imatrix(f.fname_model, 4321);
    const std::string fname = sibling_path(f, "swap.bpw_state");
    write_file(fname, read_file(f.fname_state));

    llama_model_quantize_params params = base_params(f);
    params.state_file = nullptr;
    const quant_run fresh = quantize(f.fname_model, f.fname_out, im, params);
    params.state_file = fname.c_str();
    const quant_run reused = quantize(f.fname_model, f.fname_out, im, params);
    t.log(std::to_string(count_type_changes(f.base, fresh)) + " tensors change type with the other imatrix");
    t.assert_equal("quantize succeeds", 0u, fresh.rc);
    t.assert_equal("quantize succeeds", 0u, reused.rc);
    t.assert_equal("a checkpoint from another imatrix is reported", (size_t) 1, count_matches(reused.log, "state file does not match this model, imatrix or pruned layers, ignoring"));
    t.assert_equal("a checkpoint from another imatrix is scored again", count_matches(f.base.log, "processing tensor"), count_matches(reused.log, "processing tensor"));
    t.assert_equal("scored again with another imatrix, tensors whose type differs from a fresh run", (size_t) 0, count_type_changes(fresh, reused));
}

static void test_imatrix_parts(testing & t, const fixture & f) {
    // one float doubled in one list at a time
    auto check = [&](const std::string & part, const imatrix_data & im) {
        const std::string fname = sibling_path(f, part + ".bpw_state");
        write_file(fname, read_file(f.fname_state));

        llama_model_quantize_params params = base_params(f);
        params.state_file = fname.c_str();
        const quant_run run = quantize(f.fname_model, f.fname_out, im, params);
        t.assert_equal("quantize succeeds", 0u, run.rc);
        t.assert_equal("a checkpoint from other " + part + " is reported", (size_t) 1, count_matches(run.log, "state file does not match this model, imatrix or pruned layers, ignoring"));
        t.assert_equal("a checkpoint from other " + part + " is scored again", count_matches(f.base.log, "processing tensor"), count_matches(run.log, "processing tensor"));
    };

    imatrix_data im = f.im;
    im.values[0][0] *= 2.0f;
    check("values", im);

    im = f.im;
    im.activations[0][0] *= 2.0f;
    check("activations", im);

    im = f.im;
    im.statistics[0][0] *= 2.0f;
    check("statistics", im);
}

static void test_imatrix_order(testing & t, const fixture & f) {
    // the same imatrix data, passed in reverse order
    imatrix_data im = f.im;
    std::reverse(im.names.begin(),       im.names.end());
    std::reverse(im.values.begin(),      im.values.end());
    std::reverse(im.activations.begin(), im.activations.end());
    std::reverse(im.statistics.begin(),  im.statistics.end());
    const std::string fname = sibling_path(f, "reversed.bpw_state");
    write_file(fname, read_file(f.fname_state));

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const std::string n_tensors = std::to_string(count_matches(f.base.log, "processing tensor")) + " tensors";
    const quant_run run = quantize(f.fname_model, f.fname_out, im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("the same imatrix in another order loads the checkpoint", (size_t) 1, count_matches(run.log, "using " + fname + " (data for " + n_tensors + " loaded)"));
    t.assert_equal("the same imatrix in another order scores no tensor", (size_t) 0, count_matches(run.log, "processing tensor"));
}

static void test_pruned_layer(testing & t, const fixture & f) {
    // pruning a block renumbers the ones after it
    const int32_t prune[] = { 1, -1 };
    const std::string fname = sibling_path(f, "pruned.bpw_state");
    write_file(fname, read_file(f.fname_state));

    llama_model_quantize_params params = base_params(f);
    params.prune_layers = prune;
    params.state_file   = nullptr;
    const quant_run fresh = quantize(f.fname_model, f.fname_out, f.im, params);
    params.state_file = fname.c_str();
    const quant_run reused = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("quantize succeeds", 0u, fresh.rc);
    t.assert_equal("quantize succeeds", 0u, reused.rc);
    t.assert_equal("a checkpoint without pruning is reported", (size_t) 1, count_matches(reused.log, "state file does not match this model, imatrix or pruned layers, ignoring"));
    t.assert_equal("a checkpoint without pruning is scored again", count_matches(fresh.log, "processing tensor"), count_matches(reused.log, "processing tensor"));
    t.assert_equal("scored again with a pruned block, tensors whose type differs from a fresh run", (size_t) 0, count_type_changes(fresh, reused));

    // another block pruned, reusing the checkpoint saved for block 1
    const int32_t prune_other[] = { 2, -1 };
    params.prune_layers = prune_other;
    const quant_run other = quantize(f.fname_model, f.fname_out, f.im, params);
    t.assert_equal("the pruned run saves its checkpoint", (size_t) 1, count_matches(reused.log, "saved target progress"));
    t.assert_equal("quantize succeeds", 0u, other.rc);
    t.assert_equal("a checkpoint for another pruned block is reported", (size_t) 1, count_matches(other.log, "state file does not match this model, imatrix or pruned layers, ignoring"));
    t.assert_equal("a checkpoint for another pruned block is scored again", count_matches(fresh.log, "processing tensor"), count_matches(other.log, "processing tensor"));
}

static void test_stats_order(testing & t, const fixture & f) {
    // stats in reverse order
    imatrix_data im = f.im;
    std::reverse(im.stats_schema.begin(), im.stats_schema.end());
    for (auto & stats : im.statistics) { std::reverse(stats.begin(), stats.end()); }

    llama_model_quantize_params params = base_params(f);
    params.state_file = nullptr;
    const quant_run run = quantize(f.fname_model, f.fname_out, im, params);
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_equal("statistics in different order", (size_t) 0, count_matches(run.log, "skipping CNIF"));
    t.assert_equal("statistics in different order, rows that changed", (size_t) 0, count_row_changes(f.base, run));
}

static void test_stats_missing(testing & t, const fixture & f) {
    // tensor whose stats vector is cut short or left out
    const size_t k = std::find(f.im.names.begin(), f.im.names.end(), "blk.1.ffn_down.weight") - f.im.names.begin();
    t.assert_true("imatrix has blk.1.ffn_down.weight", k < f.im.names.size());
    if (k == f.im.names.size()) { return; }

    // pearson under another name
    imatrix_data renamed = f.im;
    std::replace(renamed.stats_schema.begin(), renamed.stats_schema.end(), std::string("pearson"), std::string("spearman"));
    const std::string fname = sibling_path(f, "renamed.bpw_state");
    write_file(fname, read_file(f.fname_state));

    llama_model_quantize_params params = base_params(f);
    params.state_file = fname.c_str();
    const quant_run missing = quantize(f.fname_model, f.fname_out, renamed, params);

    // no names
    imatrix_data unnamed = f.im;
    unnamed.stats_schema.clear();
    params.state_file = nullptr;
    const quant_run no_names = quantize(f.fname_model, f.fname_out, unnamed, params);

    // stats vector cut before pearson
    imatrix_data cut = f.im;
    cut.statistics[k].resize(10);
    const quant_run short_stats = quantize(f.fname_model, f.fname_out, cut, params);
    imatrix_data dropped = f.im;
    dropped.statistics[k].clear();
    const quant_run no_stats = quantize(f.fname_model, f.fname_out, dropped, params);

    const size_t n_changed = count_row_changes(f.base, missing);
    t.log(std::to_string(n_changed) + " mix rows change without CNIF");
    t.assert_equal("quantize succeeds", 0u, missing.rc);
    t.assert_true("missing metric changes the mix", n_changed > 0 && n_changed != SIZE_MAX);
    t.assert_equal("missing metric is reported", (size_t) 1, count_matches(missing.log, "statistics have no pearson metric, skipping CNIF"));
    t.assert_equal("checkpoint with different statistics names is reported", (size_t) 1, count_matches(missing.log, "state file does not match this model, imatrix or pruned layers, ignoring"));
    t.assert_equal("checkpoint with different statistics names is scored again", count_matches(f.base.log, "processing tensor"), count_matches(missing.log, "processing tensor"));
    t.assert_equal("quantize succeeds", 0u, no_names.rc);
    t.assert_equal("statistics without names are reported", (size_t) 1, count_matches(no_names.log, "statistics have no sum_sq metric, skipping CNIF"));
    t.assert_equal("statistics without names log one line", (size_t) 1, count_matches(no_names.log, "skipping CNIF"));
    t.assert_equal("statistics without names, rows that differ from a missing metric", (size_t) 0, count_row_changes(missing, no_names));
    t.assert_equal("quantize succeeds", 0u, short_stats.rc);
    t.assert_equal("a short stats vector is not reported as a missing metric", (size_t) 0, count_matches(short_stats.log, "skipping CNIF"));
    t.assert_equal("quantize succeeds", 0u, no_stats.rc);
    t.assert_equal("a short stats vector, rows that differ from a left-out one", (size_t) 0, count_row_changes(no_stats, short_stats));
}

// counts its calls, and asks to stop at call number stop_at only
struct terminate_probe {
    size_t n_calls = 0;
    size_t stop_at = 0;
};

static bool terminate_at(void * data) {
    auto * probe = static_cast<terminate_probe *>(data);
    return ++probe->n_calls == probe->stop_at;
}

static void test_abort_callback(testing & t, const fixture & f) {
    // full scoring no interruptions
    terminate_probe counter;
    llama_model_quantize_params params = base_params(f);
    params.state_file          = nullptr;
    params.abort_callback      = terminate_at;
    params.abort_callback_data = &counter;
    const quant_run counted = quantize(f.fname_model, f.fname_out, f.im, params);

    // interrupt at half the calls
    terminate_probe stopper;
    stopper.stop_at = counter.n_calls / 2;
    const std::string fname = sibling_path(f, "interrupted.bpw_state");
    params.state_file          = fname.c_str();
    params.abort_callback_data = &stopper;
    const quant_run stopped = quantize(f.fname_model, f.fname_out, f.im, params);
    const bool stopped_wrote = std::filesystem::exists(f.fname_out);
    params.abort_callback = nullptr;
    const quant_run resumed = quantize(f.fname_model, f.fname_out, f.im, params);

    const size_t n_total = count_matches(f.base.log, "processing tensor");
    const size_t n_saved = logged_number(stopped.log, "interrupted, saving progress for ");
    t.log(std::to_string(n_saved) + " of " + std::to_string(n_total) + " tensors scored before the stop at call " + std::to_string(stopper.stop_at) + " of " + std::to_string(counter.n_calls));
    t.assert_equal("quantize succeeds", 0u, counted.rc);
    t.assert_equal("a callback that never asks to stop, mix rows that changed", (size_t) 0, count_row_changes(f.base, counted));
    t.assert_equal("the interrupted call fails", 1u, stopped.rc);
    t.assert_equal("the stop is reported", (size_t) 1, count_matches(stopped.log, "failed to quantize: interrupted"));
    t.assert_true("the stop comes part way through the scoring", n_saved > 0 && n_saved < n_total);
    t.assert_equal("the stopped run saves what it scored", (size_t) 1, count_matches(stopped.log, "saved target progress for " + std::to_string(n_saved) + " tensors to " + fname));
    t.assert_true("the stopped run writes no output", !stopped_wrote);
    t.assert_equal("quantize succeeds", 0u, resumed.rc);
    t.assert_equal("the resumed run loads the saved tensors", (size_t) 1, count_matches(resumed.log, "using " + fname + " (data for " + std::to_string(n_saved) + " tensors loaded)"));
    t.assert_equal("the resumed run scores the rest", n_total - n_saved, count_matches(resumed.log, "processing tensor"));
    t.assert_equal("resumed, mix rows that differ from the first run", (size_t) 0, count_row_changes(f.base, resumed));

    // without a target, the main loop calls it once per tensor
    terminate_probe loop_stopper;
    loop_stopper.stop_at = 2;
    llama_model_quantize_params plain = llama_model_quantize_default_params();
    plain.nthread             = 4;
    plain.abort_callback      = terminate_at;
    plain.abort_callback_data = &loop_stopper;
    const quant_run main_loop = quantize(f.fname_model, f.fname_out, f.im, plain);
    t.assert_equal("a stop in the main loop fails the call", 1u, main_loop.rc);
    t.assert_equal("a stop in the main loop is reported", (size_t) 1, count_matches(main_loop.log, "failed to quantize: interrupted"));
    t.assert_equal("the main loop stops at the call that asks", (size_t) 2, loop_stopper.n_calls);
}

static void test_interrupt_first_call(testing & t, const fixture & f) {
    // a copy of the checkpoint, stopped at the first call, before the workers read it back
    const size_t n_total = count_matches(f.base.log, "processing tensor");
    const std::string fname = sibling_path(f, "first-call.bpw_state");
    write_file(fname, read_file(f.fname_state));
    terminate_probe stopper;
    stopper.stop_at = 1;
    llama_model_quantize_params params = base_params(f);
    params.state_file          = fname.c_str();
    params.abort_callback      = terminate_at;
    params.abort_callback_data = &stopper;
    const quant_run stopped = quantize(f.fname_model, f.fname_out, f.im, params);
    params.abort_callback = nullptr;
    const quant_run resumed = quantize(f.fname_model, f.fname_out, f.im, params);

    t.assert_equal("the stopped call fails", 1u, stopped.rc);
    t.assert_equal("the checkpoint keeps its tensors", (size_t) 1, count_matches(resumed.log, "using " + fname + " (data for " + std::to_string(n_total) + " tensors loaded)"));
    t.assert_equal("the next run scores nothing", (size_t) 0, count_matches(resumed.log, "processing tensor"));
}

#ifndef _WIN32
static void app_signal_handler(int, siginfo_t *, void *) {}

// true when both have the same handler, flags and mask
static bool same_action(const struct sigaction & a, const struct sigaction & b) {
    if (a.sa_flags != b.sa_flags) { return false; }
    if ((a.sa_flags & SA_SIGINFO) ? a.sa_sigaction != b.sa_sigaction : a.sa_handler != b.sa_handler) { return false; }
    for (int sig = 1; sig < NSIG; sig++) {
        if (sigismember(&a.sa_mask, sig) != sigismember(&b.sa_mask, sig)) { return false; }
    }

    return true;
}

// read SIGINT and SIGTERM disposition
struct disposition_probe {
    size_t n_calls = 0;
    struct sigaction during[2];
};

static bool probe_dispositions(void * data) {
    auto * probe = static_cast<disposition_probe *>(data);
    if (probe->n_calls++ == 0) {
        sigaction(SIGINT,  nullptr, &probe->during[0]);
        sigaction(SIGTERM, nullptr, &probe->during[1]);
    }

    return false;
}
#endif

static void test_signal_handlers(testing & t, const fixture & f) {
#ifdef _WIN32
    GGML_UNUSED(f);
    t.skip("POSIX signals only");
#else
    struct sigaction app = {};
    app.sa_sigaction = app_signal_handler;
    app.sa_flags = SA_SIGINFO;
    sigemptyset(&app.sa_mask);
    sigaddset(&app.sa_mask, SIGTERM);

    const int signals[] = { SIGINT, SIGTERM };
    struct sigaction prev[2], before[2], after[2];
    for (int i = 0; i < 2; i++) {
        sigaction(signals[i], &app, &prev[i]);
        sigaction(signals[i], nullptr, &before[i]);
    }

    // checkpoint exists, should not score
    const std::string fname = sibling_path(f, "signals.bpw_state");
    write_file(fname, read_file(f.fname_state));
    disposition_probe probe = {};
    llama_model_quantize_params params = base_params(f);
    params.state_file          = fname.c_str();
    params.abort_callback      = probe_dispositions;
    params.abort_callback_data = &probe;
    const quant_run run = quantize(f.fname_model, f.fname_out, f.im, params);

    for (int i = 0; i < 2; i++) {
        sigaction(signals[i], nullptr, &after[i]);
        sigaction(signals[i], &prev[i], nullptr);
    }
    t.assert_equal("quantize succeeds", 0u, run.rc);
    t.assert_true("the callback is called", probe.n_calls > 0);
    t.assert_true("SIGINT keeps its handler, flags and mask during a call", same_action(before[0], probe.during[0]));
    t.assert_true("SIGTERM keeps its handler, flags and mask during a call", same_action(before[1], probe.during[1]));
    t.assert_true("SIGINT keeps its handler, flags and mask after a call", same_action(before[0], after[0]));
    t.assert_true("SIGTERM keeps its handler, flags and mask after a call", same_action(before[1], after[1]));
#endif
}

static int run_tests(testing & t, const std::filesystem::path & dir) {
    fixture f;
    f.fname_model = (dir / "model-f32.gguf").string();
    f.fname_out   = (dir / "model-out.gguf").string();
    f.fname_state = (dir / "model.bpw_state").string();
    if (!save_model(f.fname_model)) {
        fprintf(stderr, "failed to create the model\n");
        return 1;
    }

    f.im = make_imatrix(f.fname_model, 1234);
    if (f.im.names.empty()) {
        fprintf(stderr, "failed to read the model\n");
        return 1;
    }

    f.base = quantize(f.fname_model, f.fname_out, f.im, base_params(f));
    if (f.base.rc != 0) {
        fprintf(stderr, "the first quantization failed:\n%s", f.base.log.c_str());
        return 1;
    }

    t.test("target_bpw",           [&](testing & t) { test_target_bpw(t, f); });
    t.test("target_size",          [&](testing & t) { test_target_size(t, f); });
    t.test("state_checkpoint",     [&](testing & t) { test_state_checkpoint(t, f); });
    t.test("excluded_type",        [&](testing & t) { test_excluded_type(t, f); });
    t.test("unknown_type",         [&](testing & t) { test_unknown_type(t, f); });
    t.test("zero_candidates",      [&](testing & t) { test_zero_candidates(t, f); });
    t.test("truncated",            [&](testing & t) { test_truncated(t, f); });
    t.test("missing_dir",          [&](testing & t) { test_missing_dir(t, f); });
    t.test("unreadable_file",      [&](testing & t) { test_unreadable_file(t, f); });
    t.test("directory_path",       [&](testing & t) { test_directory_path(t, f); });
    t.test("save_error",           [&](testing & t) { test_save_error(t, f); });
    t.test("imatrix_swap",         [&](testing & t) { test_imatrix_swap(t, f); });
    t.test("imatrix_parts",        [&](testing & t) { test_imatrix_parts(t, f); });
    t.test("imatrix_order",        [&](testing & t) { test_imatrix_order(t, f); });
    t.test("pruned_layer",         [&](testing & t) { test_pruned_layer(t, f); });
    t.test("stats_order",          [&](testing & t) { test_stats_order(t, f); });
    t.test("stats_missing",        [&](testing & t) { test_stats_missing(t, f); });
    t.test("abort_callback",       [&](testing & t) { test_abort_callback(t, f); });
    t.test("interrupt_first_call", [&](testing & t) { test_interrupt_first_call(t, f); });
    t.test("signal_handlers",      [&](testing & t) { test_signal_handlers(t, f); });

    return t.summary();
}

int main(int argc, char ** argv) {
    testing t;

    const char * env = getenv("LLAMA_TEST_VERBOSE");
    verbose = env && std::string(env) == "1";
    t.verbose = verbose;
    llama_log_set(log_callback, nullptr);

    if (argc > 1) { t.set_filter(argv[1]); }

    llama_backend_init();
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("test-quantize-target-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const int ret = run_tests(t, dir);
    std::filesystem::remove_all(dir);
    llama_backend_free();

    return ret;
}
