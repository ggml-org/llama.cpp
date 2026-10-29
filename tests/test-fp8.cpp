#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

static const char * FP8_TEST_WEIGHTS[] = {
    "output.weight", "blk.0.attn_q.weight",
    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
};

static int64_t prepare_models(const std::string & path, const std::string & input, const std::string & reference) {
    ggml_context * ctx_raw = nullptr;
    gguf_context_ptr source(gguf_init_from_file(path.c_str(), {false, &ctx_raw}));
    ggml_context_ptr source_ctx(ctx_raw);
    if (!source || !source_ctx) {
        throw std::runtime_error("failed to read FP8 test fixture");
    }
    ggml_tensor * attn_q = ggml_get_tensor(source_ctx.get(), "blk.0.attn_q.weight");
    if (!attn_q || !ggml_get_tensor(source_ctx.get(), "output.weight")) {
        throw std::runtime_error("FP8 test fixture must have attention Q and output weights");
    }
    gguf_context_ptr fp8(gguf_init_empty());
    gguf_context_ptr folded(gguf_init_empty());
    gguf_set_kv(fp8.get(), source.get());
    gguf_set_kv(folded.get(), source.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        ggml_tensor * tensor = ggml_get_tensor(source_ctx.get(), gguf_get_tensor_name(source.get(), i));
        gguf_add_tensor(fp8.get(), tensor);
        gguf_add_tensor(folded.get(), tensor);
    }

    ggml_context_ptr scale_ctx(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
    // GGUF keeps pointers into these buffers until the files are written.
    std::vector<std::vector<uint8_t>> fp8_data;
    std::vector<std::vector<ggml_fp16_t>> folded_data;
    std::vector<std::vector<float>> scale_data;
    for (const char * name : FP8_TEST_WEIGHTS) {
        ggml_tensor * tensor = ggml_get_tensor(source_ctx.get(), name);
        if (!tensor) {
            continue;
        }
        std::vector<float> values(ggml_nelements(tensor));
        if (tensor->type == GGML_TYPE_F32) {
            memcpy(values.data(), tensor->data, ggml_nbytes(tensor));
        } else {
            const auto * traits = ggml_get_type_traits(tensor->type);
            if (!traits->to_float) {
                throw std::runtime_error("unsupported FP8 test fixture tensor type");
            }
            traits->to_float(tensor->data, values.data(), values.size());
        }
        const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
        fp8_data.emplace_back(values.size());
        traits->from_float_ref(values.data(), fp8_data.back().data(), values.size());
        gguf_set_tensor_type(fp8.get(), name, GGML_TYPE_F8_E4M3);
        gguf_set_tensor_data(fp8.get(), name, fp8_data.back().data());

        traits->to_float(fp8_data.back().data(), values.data(), values.size());
        const size_t expert_size = tensor->ne[0] * tensor->ne[1];
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] *= i < expert_size ? 2.0f : 4.0f;
        }
        folded_data.emplace_back(values.size());
        ggml_fp32_to_fp16_row(values.data(), folded_data.back().data(), values.size());
        gguf_set_tensor_type(folded.get(), name, GGML_TYPE_F16);
        gguf_set_tensor_data(folded.get(), name, folded_data.back().data());

        const std::string prefix(name, strlen(name) - strlen("weight"));
        for (const char * suffix : {"scale", "input_scale"}) {
            const std::string scale_name = prefix + suffix;
            ggml_tensor * scale = ggml_get_tensor(source_ctx.get(), scale_name.c_str());
            if (scale) {
                if (scale->type != GGML_TYPE_F32 || ggml_nelements(scale) != tensor->ne[2]) {
                    throw std::runtime_error("unsupported FP8 test fixture scale shape or type");
                }
                scale_data.emplace_back(tensor->ne[2], 1.0f);
                gguf_set_tensor_data(folded.get(), scale_name.c_str(), scale_data.back().data());
            } else {
                scale = ggml_new_tensor_1d(scale_ctx.get(), GGML_TYPE_F32, tensor->ne[2]);
                ggml_set_name(scale, scale_name.c_str());
                gguf_add_tensor(fp8.get(), scale);
            }
            // Scales above one avoid subnormal Q6_K block scales in the folded reference.
            scale_data.emplace_back(tensor->ne[2], 4.0f);
            if (strcmp(suffix, "scale") == 0) {
                scale_data.back()[0] = 2.0f;
            }
            gguf_set_tensor_data(fp8.get(), scale_name.c_str(), scale_data.back().data());
        }
    }
    if (!gguf_write_to_file(fp8.get(), input.c_str(), false) ||
        !gguf_write_to_file(folded.get(), reference.c_str(), false)) {
        throw std::runtime_error("failed to write FP8 test models");
    }
    return attn_q->ne[0];
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static llama_model_ptr load_model(const std::string & path, const llama_model_params & params, FILE * file = nullptr) {
    llama_model_ptr model(file ? llama_model_load_from_file_ptr(file, params) :
                                 llama_model_load_from_file(path.c_str(), params));
    if (!model) {
        throw std::runtime_error("failed to load FP8 test model");
    }
    return model;
}

static std::vector<float> load_logits(const std::string & path, const llama_model_params & model_params, bool split = false) {
    std::unique_ptr<FILE, int (*)(FILE *)> file(split ? nullptr : fopen(path.c_str(), "rb"), fclose);
    if (!split && !file) {
        throw std::runtime_error("failed to open FP8 test model");
    }
    llama_model_ptr model = load_model(path, model_params, file.get());
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_ubatch = 64;
    ctx_params.n_threads = ctx_params.n_threads_batch = 4;
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create FP8 test context");
    }
    const auto tokens = get_tokens(8, llama_vocab_n_tokens(llama_model_get_vocab(model.get())), 1);
    return get_logits(model.get(), lctx.get(), tokens);
}

static void check_model_save(const std::string & path, const llama_model_params & model_params) {
    const std::string saved = path + ".saved.gguf";
    const auto logits = load_logits(path, model_params);
    auto save_params = model_params;
    // Packed CPU buffers do not support reading weights back for saving.
    save_params.use_extra_bufts = false;
    save_params.load_mode = LLAMA_LOAD_MODE_NONE;
    {
        llama_model_ptr model = load_model(path, save_params);
        llama_model_save_to_file(model.get(), saved.c_str());
    }
    const double error = nmse(logits, load_logits(saved, model_params));
    LOG("FP8 model save: %s nmse=%.2e\n", path.c_str(), error);
    if (!(error < 1e-10)) {
        throw std::runtime_error("saved FP8 test model differs from original model");
    }
    std::filesystem::rename(saved, path);
}

static void check_fp8_scales(const std::string & input, const std::string & reference, const llama_model_params & model_params) {
    const auto reference_logits = load_logits(reference, model_params);
    const double source_error = nmse(reference_logits, load_logits(input, model_params));
    LOG("FP8 sidecar scales: nmse=%.2e\n", source_error);
    if (!(source_error < 1e-5)) {
        throw std::runtime_error("FP8 sidecar scales differ from folded weights");
    }
}

static void check_split_model(const std::string & path, const std::filesystem::path & dir, const llama_model_params & model_params) {
    ggml_context * ctx_raw = nullptr;
    gguf_context_ptr metadata(gguf_init_from_file(path.c_str(), {false, &ctx_raw}));
    ggml_context_ptr ctx(ctx_raw);
    if (!metadata || !ctx) {
        throw std::runtime_error("failed to read FP8 test model for splitting");
    }
    const auto split_first = (dir / "split-00001-of-00002.gguf").string();
    const auto split_second = (dir / "split-00002-of-00002.gguf").string();
    for (int split = 0; split < 2; ++split) {
        gguf_context_ptr shard(gguf_init_empty());
        gguf_set_kv(shard.get(), metadata.get());
        gguf_set_val_u16(shard.get(), "split.no", split);
        gguf_set_val_u16(shard.get(), "split.count", 2);
        gguf_set_val_i32(shard.get(), "split.tensors.count", gguf_get_n_tensors(metadata.get()));
        for (int64_t i = 0; i < gguf_get_n_tensors(metadata.get()); ++i) {
            auto * tensor = ggml_get_tensor(ctx.get(), gguf_get_tensor_name(metadata.get(), i));
            // Put output scales in the second shard to test lookup across splits.
            const bool output_scale = strcmp(tensor->name, "output.scale") == 0 || strcmp(tensor->name, "output.input_scale") == 0;
            if (output_scale == (split == 1)) {
                gguf_add_tensor(shard.get(), tensor);
            }
        }
        if (!gguf_write_to_file(shard.get(), (split == 0 ? split_first : split_second).c_str(), false)) {
            throw std::runtime_error("failed to write split FP8 test model");
        }
    }
    const double split_error = nmse(load_logits(path, model_params), load_logits(split_first, model_params, true));
    LOG("FP8 split model: nmse=%.2e\n", split_error);
    if (!(split_error < 1e-10)) {
        throw std::runtime_error("split FP8 model differs from unsplit model");
    }
}

static void check_fp8_requantize(
        const std::string & input, const std::string & reference,
        const std::string & output, const std::string & reference_output,
        int64_t n_embd, const llama_model_params & model_params) {
    for (llama_ftype ftype : {LLAMA_FTYPE_ALL_F32, LLAMA_FTYPE_MOSTLY_F16, LLAMA_FTYPE_MOSTLY_BF16, LLAMA_FTYPE_MOSTLY_Q4_K_M}) {
        for (int nthread : {1, 3}) {
            std::vector<float> importance(n_embd, 1.0f);
            llama_model_imatrix_data imatrix[] = {
                {"blk.0.attn_q.weight", importance.data(), importance.size()},
                {nullptr, nullptr, 0},
            };
            auto params = llama_model_quantize_default_params();
            params.ftype = ftype;
            params.nthread = nthread;
            if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M && nthread == 3) {
                params.imatrix = imatrix;
            }
            if (llama_model_quantize(input.c_str(), output.c_str(), &params) != 0 ||
                llama_model_quantize(reference.c_str(), reference_output.c_str(), &params) != 0) {
                throw std::runtime_error("FP8 model requantization failed");
            }
            const double error = nmse(load_logits(reference_output, model_params), load_logits(output, model_params));
            LOG("FP8 requantization: ftype=%d threads=%d nmse=%.2e\n", ftype, nthread, error);
            if (!(error < 1e-4)) {
                throw std::runtime_error("FP8 requantization differs from folded weights");
            }
        }
    }
}

static void test_fp8(const std::string & model_path) {
    const auto dir = std::filesystem::temp_directory_path() / ("llama-fp8-" + std::to_string(std::random_device{}()));
    if (!std::filesystem::create_directory(dir)) {
        throw std::runtime_error("failed to create FP8 test directory");
    }
    struct test_files {
        std::filesystem::path dir;
        ~test_files() {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    } files{dir};

    const auto input = (files.dir / "fp8.gguf").string();
    const auto reference = (files.dir / "reference.gguf").string();
    const auto output = (files.dir / "output.gguf").string();
    const auto reference_output = (files.dir / "reference-output.gguf").string();

    // Use the same CPU kernels for scaled weights and the folded reference.
    const llama_model_tensor_buft_override overrides[] = {
        {".*", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    ggml_backend_dev_t devices[] = {nullptr};
    auto model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.tensor_buft_overrides = overrides;
    model_params.progress_callback = silent_model_load_progress;

    const int64_t n_embd = prepare_models(model_path, input, reference);
    check_model_save(input, model_params);
    check_model_save(reference, model_params);
    check_fp8_scales(input, reference, model_params);
    check_split_model(reference, files.dir, model_params);
    check_split_model(input, files.dir, model_params);
    check_fp8_requantize(input, reference, output, reference_output, n_embd, model_params);
}

int main(int argc, char ** argv) {
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s -m model.gguf | --models DIR\n", argv[0]);
        return 0;
    }
    if (argc != 3 || (strcmp(argv[1], "-m") != 0 && strcmp(argv[1], "--model") != 0 && strcmp(argv[1], "--models") != 0)) {
        fprintf(stderr, "Usage: %s -m model.gguf | --models DIR\n", argv[0]);
        return 1;
    }
    common_init();
    llama_backend_init();
    int result = 0;
    try {
        std::vector<std::string> models;
        if (strcmp(argv[1], "--models") == 0) {
            if (!std::filesystem::is_directory(argv[2])) {
                throw std::runtime_error("models directory does not exist: " + std::string(argv[2]));
            }
            for (const auto & entry : std::filesystem::directory_iterator(argv[2])) {
                if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                    models.push_back(entry.path().string());
                }
            }
            std::sort(models.begin(), models.end());
            if (models.empty()) {
                throw std::runtime_error("no .gguf models found in: " + std::string(argv[2]));
            }
        } else {
            models.push_back(argv[2]);
        }

        std::vector<bool> passed;
        for (const auto & model : models) {
            try {
                test_fp8(model);
                passed.push_back(true);
            } catch (const std::exception & err) {
                fprintf(stderr, "FP8 test failed for '%s': %s\n", model.c_str(), err.what());
                passed.push_back(false);
                result = 1;
            }
        }
        printf("\n| Model | Result |\n|-------|--------|\n");
        for (size_t i = 0; i < models.size(); ++i) {
            printf("| %s | %s |\n", std::filesystem::path(models[i]).filename().string().c_str(), passed[i] ? "PASS" : "FAIL");
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "FP8 test failed: %s\n", err.what());
        result = 1;
    }
    llama_backend_free();
    return result;
}
