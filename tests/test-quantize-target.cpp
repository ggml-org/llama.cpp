#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

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

struct imatrix_data {
    std::vector<std::string>        names;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> statistics;
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

    return im;
}

static std::vector<llama_model_imatrix_data> imatrix_list(const imatrix_data & im, const std::vector<std::vector<float>> & data) {
    std::vector<llama_model_imatrix_data> list;
    for (size_t i = 0; i < im.names.size(); i++) { list.push_back({ im.names[i].c_str(), data[i].data(), data[i].size() }); }
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
    params.imatrix     = values.data();
    params.activations = activations.data();
    params.statistics  = statistics.data();

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

// the model, its imatrix and the first, scored run that the cases compare against
struct fixture {
    std::string  fname_model;
    std::string  fname_out;
    std::string  fname_state;
    imatrix_data im;
    quant_run    base;
};

// the parameters of the first run
static llama_model_quantize_params base_params(const fixture & f) {
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.nthread    = 4;
    params.target_bpw = target_bpw;
    params.state_file = f.fname_state.c_str();

    return params;
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

    t.test("target_bpw",       [&](testing & t) { test_target_bpw(t, f); });
    t.test("target_size",      [&](testing & t) { test_target_size(t, f); });
    t.test("state_checkpoint", [&](testing & t) { test_state_checkpoint(t, f); });
    t.test("excluded_type",    [&](testing & t) { test_excluded_type(t, f); });

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
