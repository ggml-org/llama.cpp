#include "llama.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "log.h"

#include "../src/llama-model.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <regex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct quantize_stats_params {
    std::string model = "models/7B/ggml-model-f16.gguf";
    bool verbose = false;
    bool per_layer_stats = false;
    bool print_histogram = false;
    bool reference = false;
    std::vector<std::string> include_layers;
    std::vector<std::string> exclude_layers;
    std::vector<enum ggml_type> include_types;
};

constexpr size_t HISTOGRAM_BUCKETS = 150;
constexpr double HISTOGRAM_RANGE = 0.03;

struct error_stats {
    size_t num_samples;
    double total_error;
    double max_error;
    uint64_t error_histogram[HISTOGRAM_BUCKETS];
};

// print the usage text at the given level: INFO for -h, ERROR when it follows a bad argument
static void quantize_stats_print_usage(int /*argc*/, char ** argv, ggml_log_level level) {
    quantize_stats_params params;
    const int verbosity = level == GGML_LOG_LEVEL_ERROR ? LOG_LEVEL_ERROR : LOG_LEVEL_INFO;
    LOG_TMPL(level, verbosity, "usage: %s [options]\n", argv[0]);
    LOG_TMPL(level, verbosity, "\n");
    LOG_TMPL(level, verbosity, "options:\n");
    LOG_TMPL(level, verbosity, "  -h, --help            show this help message and exit\n");
    LOG_TMPL(level, verbosity, "  -m FNAME, --model FNAME\n");
    LOG_TMPL(level, verbosity, "                        model path (default: %s)\n", params.model.c_str());
    LOG_TMPL(level, verbosity, "  -r, --reference\n");
    LOG_TMPL(level, verbosity, "                        use reference implementation (default: false)\n");
    LOG_TMPL(level, verbosity, "  -v, --verbose\n");
    LOG_TMPL(level, verbosity, "                        verbose output (default: false)\n");
    LOG_TMPL(level, verbosity, "  -p, --per-layer-stats\n");
    LOG_TMPL(level, verbosity, "                        print stats per layer (default: false)\n");
    LOG_TMPL(level, verbosity, "  --histogram\n");
    LOG_TMPL(level, verbosity, "                        print error histogram (default: false)\n");
    LOG_TMPL(level, verbosity, "  -l LAYER, --include-layer LAYER\n");
    LOG_TMPL(level, verbosity, "                        only test layers matching pattern\n");
    LOG_TMPL(level, verbosity, "  -L LAYER, --exclude-layer LAYER\n");
    LOG_TMPL(level, verbosity, "                        exclude layers matching pattern\n");
    LOG_TMPL(level, verbosity, "  -t TYPE, --type TYPE\n");
    LOG_TMPL(level, verbosity, "                        only test given type (q4_0, q4_1)\n");
    LOG_TMPL(level, verbosity, "\n");
}

// Check if a layer is included/excluded by command line
static bool layer_included(const quantize_stats_params & params, const std::string & layer) {
    for (const auto& excluded : params.exclude_layers) {
        if (std::regex_search(layer, std::regex(excluded))) {
            return false;
        }
    }
    for (const auto& included : params.include_layers) {
        if (std::regex_search(layer, std::regex(included))) {
            return true;
        }
    }
    return params.include_layers.empty();
}

// Update error statistics given vectors with the before/after result of quantization
static void update_error_stats(int64_t nelements, const float * input, const float * output, error_stats & stats) {
    for (int64_t i = 0; i < nelements; i++) {
        double diff = input[i] - output[i];
        stats.total_error += diff * diff;
        stats.max_error = fmax(fabs(diff), stats.max_error);
        stats.error_histogram[std::max(std::min((size_t) floor(fabs(diff) / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS), HISTOGRAM_BUCKETS-1), (size_t) 0)]++;
    }
    stats.num_samples += nelements;
}

static void combine_error_stats(error_stats & into, const error_stats & from) {
    into.num_samples += from.num_samples;
    into.total_error += from.total_error;
    if (from.max_error > into.max_error) into.max_error = from.max_error;
    for (size_t i=0; i<HISTOGRAM_BUCKETS; ++i) into.error_histogram[i] += from.error_histogram[i];
}

static double find_quantile(const error_stats & stats, double quantile) {
    double sum = std::accumulate(std::begin(stats.error_histogram), std::end(stats.error_histogram), 0.0);

    double accum = 0;
    for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
        accum += stats.error_histogram[i];
        if (accum >= sum*quantile) {
            return (i+1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
        }
    }
    return INFINITY;
}

static void print_error_stats(const std::string & name, const error_stats & stats, bool print_histogram) {
    double rmse = sqrt(stats.total_error / (double) stats.num_samples);
    double median = find_quantile(stats, .5);
    double pct95 = find_quantile(stats, .95);
    LOG_INF("%-50s: rmse %.8f, maxerr %.8f, 95pct<%.4f, median<%.4f\n", name.c_str(), rmse, stats.max_error, pct95, median);
    if (print_histogram) {
        LOG_INF("Error distribution:\n");
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            double lower = i * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            double upper = (i+1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            if (i == HISTOGRAM_BUCKETS -1) upper = INFINITY;
            LOG_INF("[%3.4f, %3.4f): %11" PRIu64 "\n", lower, upper, stats.error_histogram[i]);
        }
    }
}

// copied from ggml.h - verify that we can access this as a flat array
static bool tensor_is_contiguous(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static void test_roundtrip_on_chunk(
    const ggml_tensor * layer, int64_t offset, int64_t chunk_size, const ggml_type_traits & qfns, const ggml_type_traits_cpu & qfns_cpu, bool use_reference,
    float * input_scratch, char * quantized_scratch, float * output_scratch, error_stats & stats
) {
    if (layer->type == GGML_TYPE_F16) {
        for (int i = 0; i < chunk_size; i++) {
            input_scratch[i] = ggml_get_f32_1d(layer, i + offset);
        }
    } else {
        input_scratch = ggml_get_data_f32(layer) + offset;
    }

    if (use_reference) {
        qfns.from_float_ref(input_scratch, quantized_scratch, chunk_size);
    } else {
        qfns_cpu.from_float(input_scratch, quantized_scratch, chunk_size);
    }
    qfns.to_float(quantized_scratch, output_scratch, chunk_size);

    update_error_stats(chunk_size, input_scratch, output_scratch, stats);
}


// Run quantization function for a single layer and update error stats
static void test_roundtrip_on_layer(
    std::string & name, bool print_layer_stats, const ggml_type_traits & qfns, const ggml_type_traits_cpu & qfns_cpu, bool use_reference,
    const ggml_tensor * layer, std::vector<float> & input_scratch, std::vector<char> & quantized_scratch,
    std::vector<float> & output_scratch, error_stats & total_error, int max_thread = 0
) {
    assert(tensor_is_contiguous(layer));
    error_stats layer_error {};
    uint64_t nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (layer->type == GGML_TYPE_F16) {
        if (input_scratch.size() < nelements) input_scratch.resize(nelements);
        input_scratch_ptr = input_scratch.data();
    }
    if (quantized_scratch.size() < 4*nelements) quantized_scratch.resize(4*nelements);
    if (output_scratch.size() < nelements) output_scratch.resize(nelements);

    if (max_thread < 1) max_thread = std::thread::hardware_concurrency();
    int chunk_size = 32*512;
    int num_chunks = (nelements + chunk_size - 1)/chunk_size;

    if (num_chunks < 2 || max_thread < 2) {
        test_roundtrip_on_chunk(layer, 0, nelements, qfns, qfns_cpu, use_reference, input_scratch_ptr, quantized_scratch.data(),
                output_scratch.data(), print_layer_stats ? layer_error : total_error);
    } else {
        auto & stats = print_layer_stats ? layer_error : total_error;
        std::mutex mutex;
        uint64_t counter = 0;
        auto compute = [&mutex, &counter, &stats, &qfns, &qfns_cpu, nelements, layer, use_reference, input_scratch_ptr,
             &quantized_scratch, &output_scratch, chunk_size] () {
            error_stats local_stats {};
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                uint64_t offset = counter; counter += chunk_size;
                if (offset >= nelements) {
                    combine_error_stats(stats, local_stats);
                    break;
                }
                lock.unlock();
                uint64_t chunk = offset + chunk_size < nelements ? chunk_size : nelements - offset;
                test_roundtrip_on_chunk(layer, offset, chunk, qfns, qfns_cpu, use_reference, input_scratch_ptr + offset,
                        quantized_scratch.data() + 4*offset, output_scratch.data() + offset, local_stats);
            }
        };
        int nthread = std::min(num_chunks, max_thread);
        std::vector<std::thread> workers(nthread-1);
        for (auto& w : workers) w = std::thread(compute);
        compute();
        for (auto& w : workers) w.join();
    }

    if (print_layer_stats) {
        print_error_stats(name, layer_error, false);
        combine_error_stats(total_error, layer_error);
    }
}

int main(int argc, char ** argv) {
    common_init();

    ggml_time_init();

    quantize_stats_params params;

    // read command line

    int max_thread = 0;
    bool invalid_param = false;
    std::string arg;
    std::vector<char *> common_argv;
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            quantize_stats_print_usage(argc, argv, GGML_LOG_LEVEL_INFO);
            common_log_flush(common_log_main());
            exit(0);
        } else if (arg == "-r" || arg == "--reference") {
            params.reference = true;
        } else if (arg == "-v") {
            params.verbose = true;
        } else if (arg == "-p" || arg == "--per-layer-stats") {
            params.per_layer_stats = true;
        } else if (arg == "--histogram") {
            params.print_histogram = true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-l" || arg == "--include-layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_layers.emplace_back(argv[i]);
        } else if (arg == "-L" || arg == "--exclude-layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.exclude_layers.emplace_back(argv[i]);
        } else if (arg == "-t" || arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int j;
            for (j = 0; j < GGML_TYPE_COUNT; ++j) {
               const auto * name = ggml_type_name((ggml_type) j);
               if (name && strcmp(argv[i], name) == 0) break;
            }
            if (j < GGML_TYPE_COUNT) {
                params.include_types.push_back((ggml_type) j);
            } else {
                LOG_ERR("error: %s not in list of types\n", argv[i]);
                invalid_param = true;
            }
        } else if (arg == "-n" || arg == "--num-threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            max_thread = atoi(argv[i]);
        } else if (argv[i][0] == '-') {
            common_argv.push_back(argv[i]); // an option: let common_params_parse handle it
        } else {
            LOG_ERR("error: unknown argument: %s\n", arg.c_str());
            quantize_stats_print_usage(argc, argv, GGML_LOG_LEVEL_ERROR);
            common_log_flush(common_log_main());
            return 1;
        }
    }
    if (invalid_param) {
        LOG_ERR("error: invalid parameter for argument: %s\n", arg.c_str());
        quantize_stats_print_usage(argc, argv, GGML_LOG_LEVEL_ERROR);
        common_log_flush(common_log_main());
        return 1;
    }
    common_argv.push_back(nullptr);

    // the test's own "params" is declared above, so scope the parsed one
    {
        common_params params;
        params.model.path = "."; // this test takes no model
        if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
    }

    LOG("%s: running\n", "test-quantize-stats");

    llama_print_build_info(llama_version());

    // load the model
    LOG_INF("Loading model\n");

    const int64_t t_main_start_us = ggml_time_us();
    llama_model * model;
    llama_context * ctx;

    {
        auto mparams = llama_model_default_params();
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;

        model = llama_model_load_from_file(params.model.c_str(), mparams);

        if (model == NULL) {
            LOG_ERR("%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            LOG("%s: %s\n", "test-quantize-stats", "FAILED");
            common_log_flush(common_log_main());
            return 1;
        }

        auto cparams = llama_context_default_params();
        cparams.n_ctx = 256;

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            LOG_ERR("%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
            llama_model_free(model);
            LOG("%s: %s\n", "test-quantize-stats", "FAILED");
            common_log_flush(common_log_main());
            return 1;
        }
    }

    const auto & tensors = llama_internal_get_tensor_map(model);

    // check layer tensors
    int included_layers = 0;
    int64_t max_nelements = 0;
    bool is_f16 = false;
    for (const auto & kv_tensor : tensors) {
        if (!layer_included(params, kv_tensor.first)) {
            continue;
        }
        if (params.verbose) {
            LOG_INF("%s: type %s, size %" PRId64 "\n", kv_tensor.first.c_str(), ggml_type_name(kv_tensor.second->type), ggml_nelements(kv_tensor.second));
        }
        if (kv_tensor.second->type == GGML_TYPE_F16) {
            is_f16 = true;
        } else if (kv_tensor.second->type != GGML_TYPE_F32) {
            LOG_ERR("%s: error: Quantization should be tested with a float model, "
                "this model contains already quantized layers (%s is type %d)\n", __func__, kv_tensor.first.c_str(), kv_tensor.second->type);
            llama_free(ctx);
            llama_model_free(model);
            LOG("%s: %s\n", "test-quantize-stats", "FAILED");
            common_log_flush(common_log_main());
            return 1;
        }
        included_layers++;
        max_nelements = std::max(max_nelements, ggml_nelements(kv_tensor.second));
    }

    if (is_f16) {
        LOG_INF("note: source model is f16\n");
    }
    LOG_INF("testing %d layers with max size %" PRId64 "\n", included_layers, max_nelements);
    // allocate scratch space
    std::vector<float> input_scratch;
    std::vector<char> quantized_scratch;
    std::vector<float> output_scratch;

    // loop throught quantization types
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type type = (ggml_type) i;
        if (!params.include_types.empty() && std::find(params.include_types.begin(), params.include_types.end(), i) == params.include_types.end()) {
            continue;
        }
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (qfns_cpu->from_float && qfns->to_float) {
            if (params.verbose) {
                LOG_INF("testing %s ...\n",  ggml_type_name(type));
            }

            ggml_quantize_init(type);

            error_stats global_stats {};

            for (const auto & kv_tensor : tensors) {
                if (!layer_included(params, kv_tensor.first)) {
                    continue;
                }
                if (params.verbose) {
                    LOG_INF("  %s ...\n",  kv_tensor.first.c_str());
                }
                std::string layer_name { ggml_type_name(type) };
                layer_name += "::" + kv_tensor.first;
                test_roundtrip_on_layer(
                        layer_name,
                        params.per_layer_stats,
                        *qfns, *qfns_cpu,
                        params.reference,
                        kv_tensor.second,
                        input_scratch,
                        quantized_scratch,
                        output_scratch,
                        global_stats,
                        max_thread
                );
            }

            print_error_stats(ggml_type_name(type), global_stats, params.print_histogram);
        }
    }


    llama_free(ctx);
    llama_model_free(model);
    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    LOG("%s: %s\n", "test-quantize-stats", "PASSED");
    common_log_flush(common_log_main());
    return 0;
}
