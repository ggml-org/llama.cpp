#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "ggml.h"
#include "llama.h"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static bool tensor_buft_override_equal(const llama_model_tensor_buft_override& a, const llama_model_tensor_buft_override& b) {
    if (a.pattern != b.pattern) {
        // cString comparison that may be null
        if (a.pattern == nullptr || b.pattern == nullptr) {
            return false;
        }
        if (strcmp(a.pattern, b.pattern) != 0) {
            return false;
        }
    }
    if (a.buft != b.buft) {
        return false;
    }
    return true;
}

static bool vec_tensor_buft_override_equal(const std::vector<llama_model_tensor_buft_override>& a, const std::vector<llama_model_tensor_buft_override>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

static bool vec_vec_tensor_buft_override_equal(const std::vector<std::vector<llama_model_tensor_buft_override>>& a, const std::vector<std::vector<llama_model_tensor_buft_override>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!vec_tensor_buft_override_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template <typename T, typename F> static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(cpu_list, ", ");
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(gpu_list, ", ");
}

// command line params
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN, SQL };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:
            return "none";
        case CSV:
            return "csv";
        case JSON:
            return "json";
        case JSONL:
            return "jsonl";
        case MARKDOWN:
            return "md";
        case SQL:
            return "sql";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "jsonl") {
        format = JSONL;
    } else if (s == "md") {
        format = MARKDOWN;
    } else if (s == "sql") {
        format = SQL;
    } else {
        return false;
    }
    return true;
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:
            return "none";
        case LLAMA_SPLIT_MODE_LAYER:
            return "layer";
        case LLAMA_SPLIT_MODE_ROW:
            return "row";
        default:
            GGML_ABORT("invalid split mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

static std::vector<int> parse_int_range(const std::string & s) {
    // first[-last[(+|*)step]]
    std::regex range_regex(R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;
    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

struct cmd_params {
    std::vector<std::string>         model;
    std::vector<int>                 n_prompt;
    std::vector<int>                 n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<int>                 n_depth;
    std::vector<int>                 n_batch;
    std::vector<int>                 n_ubatch;
    std::vector<ggml_type>           type_k;
    std::vector<ggml_type>           type_v;
    std::vector<float>               defrag_thold;
    std::vector<int>                 n_threads;
    std::vector<std::string>         cpu_mask;
    std::vector<bool>                cpu_strict;
    std::vector<int>                 poll;
    std::vector<int>                 n_gpu_layers;
    std::vector<std::string>         rpc_servers;
    std::vector<llama_split_mode>    split_mode;
    std::vector<int>                 main_gpu;
    std::vector<bool>                no_kv_offload;
    std::vector<bool>                flash_attn;
    std::vector<std::vector<float>>  tensor_split;
    std::vector<std::vector<llama_model_tensor_buft_override>> tensor_buft_overrides;
    std::vector<bool>                use_mmap;
    std::vector<bool>                embeddings;
    std::vector<bool>                no_op_offload;
    ggml_numa_strategy               numa;
    int                              reps;
    ggml_sched_priority              prio;
    int                              delay;
    bool                             verbose;
    bool                             progress;
    bool                             no_warmup;
    output_formats                   output_format;
    output_formats                   output_format_stderr;
};

static const cmd_params cmd_params_defaults = {
    /* model                */ { "models/7B/ggml-model-q4_0.gguf" },
    /* n_prompt             */ { 512 },
    /* n_gen                */ { 128 },
    /* n_pg                 */ {},
    /* n_depth              */ { 0 },
    /* n_batch              */ { 2048 },
    /* n_ubatch             */ { 512 },
    /* type_k               */ { GGML_TYPE_F16 },
    /* type_v               */ { GGML_TYPE_F16 },
    /* defrag_thold         */ { -1.0f },
    /* n_threads            */ { cpu_get_num_math() },
    /* cpu_mask             */ { "0x0" },
    /* cpu_strict           */ { false },
    /* poll                 */ { 50 },
    /* n_gpu_layers         */ { 99 },
    /* rpc_servers          */ { "" },
    /* split_mode           */ { LLAMA_SPLIT_MODE_LAYER },
    /* main_gpu             */ { 0 },
    /* no_kv_offload        */ { false },
    /* flash_attn           */ { false },
    /* tensor_split         */ { std::vector<float>(llama_max_devices(), 0.0f) },
    /* tensor_buft_overrides*/ { std::vector<llama_model_tensor_buft_override>{ { nullptr, nullptr } } },
    /* use_mmap             */ { true },
    /* embeddings           */ { false },
    /* no_op_offload        */ { false },
    /* numa                 */ GGML_NUMA_STRATEGY_DISABLED,
    /* reps                 */ 5,
    /* prio                 */ GGML_SCHED_PRIO_NORMAL,
    /* delay                */ 0,
    /* verbose              */ false,
    /* progress             */ false,
    /* no_warmup            */ false,
    /* output_format        */ MARKDOWN,
    /* output_format_stderr */ NONE,
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  --numa <distribute|isolate|numactl>       numa mode (default: disabled)\n");
    printf("  -r, --repetitions <n>                     number of times to repeat each test (default: %d)\n",
           cmd_params_defaults.reps);
    printf("  --prio <-1|0|1|2|3>                          process/thread priority (default: %d)\n",
           cmd_params_defaults.prio);
    printf("  --delay <0...N> (seconds)                 delay between each test (default: %d)\n",
           cmd_params_defaults.delay);
    printf("  -o, --output <csv|json|jsonl|md|sql>      output format printed to stdout (default: %s)\n",
           output_format_str(cmd_params_defaults.output_format));
    printf("  -oe, --output-err <csv|json|jsonl|md|sql> output format printed to stderr (default: %s)\n",
           output_format_str(cmd_params_defaults.output_format_stderr));
    printf("  -v, --verbose                             verbose output\n");
    printf("  --progress                                print test progress indicators\n");
    printf("  --no-warmup                               skip warmup runs before benchmarking\n");
    printf("\n");
    printf("test parameters:\n");
    printf("  -m, --model <filename>                    (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    printf("  -p, --n-prompt <n>                        (default: %s)\n",
           join(cmd_params_defaults.n_prompt, ",").c_str());
    printf("  -n, --n-gen <n>                           (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    printf("  -pg <pp,tg>                               (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.n_pg, pair_str), ",").c_str());
    printf("  -d, --n-depth <n>                         (default: %s)\n",
           join(cmd_params_defaults.n_depth, ",").c_str());
    printf("  -b, --batch-size <n>                      (default: %s)\n",
           join(cmd_params_defaults.n_batch, ",").c_str());
    printf("  -ub, --ubatch-size <n>                    (default: %s)\n",
           join(cmd_params_defaults.n_ubatch, ",").c_str());
    printf("  -ctk, --cache-type-k <t>                  (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.type_k, ggml_type_name), ",").c_str());
    printf("  -ctv, --cache-type-v <t>                  (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.type_v, ggml_type_name), ",").c_str());
    printf("  -dt, --defrag-thold <f>                   (default: %s)\n",
           join(cmd_params_defaults.defrag_thold, ",").c_str());
    printf("  -t, --threads <n>                         (default: %s)\n",
           join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -C, --cpu-mask <hex,hex>                  (default: %s)\n",
           join(cmd_params_defaults.cpu_mask, ",").c_str());
    printf("  --cpu-strict <0|1>                        (default: %s)\n",
           join(cmd_params_defaults.cpu_strict, ",").c_str());
    printf("  --poll <0...100>                          (default: %s)\n", join(cmd_params_defaults.poll, ",").c_str());
    printf("  -ngl, --n-gpu-layers <n>                  (default: %s)\n",
           join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    if (llama_supports_rpc()) {
        printf("  -rpc, --rpc <rpc_servers>                 (default: %s)\n",
               join(cmd_params_defaults.rpc_servers, ",").c_str());
    }
    printf("  -sm, --split-mode <none|layer|row>        (default: %s)\n",
           join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
    printf("  -mg, --main-gpu <i>                       (default: %s)\n",
           join(cmd_params_defaults.main_gpu, ",").c_str());
    printf("  -nkvo, --no-kv-offload <0|1>              (default: %s)\n",
           join(cmd_params_defaults.no_kv_offload, ",").c_str());
    printf("  -fa, --flash-attn <0|1>                   (default: %s)\n",
           join(cmd_params_defaults.flash_attn, ",").c_str());
    printf("  -mmp, --mmap <0|1>                        (default: %s)\n",
           join(cmd_params_defaults.use_mmap, ",").c_str());
    printf("  -embd, --embeddings <0|1>                 (default: %s)\n",
           join(cmd_params_defaults.embeddings, ",").c_str());
    printf("  -ts, --tensor-split <ts0/ts1/..>          (default: 0)\n");
    printf("  -ot --override-tensors <tensor name pattern>=<buffer type>;...\n");
    printf("                                            (default: disabled)\n");
    printf("  -nopo, --no-op-offload <0|1>              (default: 0)\n");
    printf("\n");
    printf(
        "Multiple values can be given for each parameter by separating them with ','\n"
        "or by specifying the parameter multiple times. Ranges can be given as\n"
        "'first-last' or 'first-last+step' or 'first-last*mult'.\n");
}

static ggml_type ggml_type_from_name(const std::string & s) {
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }

    return GGML_TYPE_COUNT;
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params        params;
    std::string       arg;
    bool              invalid_param = false;
    const std::string arg_prefix    = "--";
    const char        split_delim   = ',';

    params.verbose              = cmd_params_defaults.verbose;
    params.output_format        = cmd_params_defaults.output_format;
    params.output_format_stderr = cmd_params_defaults.output_format_stderr;
    params.reps                 = cmd_params_defaults.reps;
    params.numa                 = cmd_params_defaults.numa;
    params.prio                 = cmd_params_defaults.prio;
    params.delay                = cmd_params_defaults.delay;
    params.progress             = cmd_params_defaults.progress;
    params.no_warmup            = cmd_params_defaults.no_warmup;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        try {
            if (arg == "-h" || arg == "--help") {
                print_usage(argc, argv);
                exit(0);
            } else if (arg == "-m" || arg == "--model") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.model.insert(params.model.end(), p.begin(), p.end());
            } else if (arg == "-p" || arg == "--n-prompt") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
            } else if (arg == "-n" || arg == "--n-gen") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
            } else if (arg == "-pg") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], ',');
                if (p.size() != 2) {
                    invalid_param = true;
                    break;
                }
                params.n_pg.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
            } else if (arg == "-d" || arg == "--n-depth") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_depth.insert(params.n_depth.end(), p.begin(), p.end());
            } else if (arg == "-b" || arg == "--batch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
            } else if (arg == "-ub" || arg == "--ubatch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_ubatch.insert(params.n_ubatch.end(), p.begin(), p.end());
            } else if (arg == "-ctk" || arg == "--cache-type-k") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_k.insert(params.type_k.end(), types.begin(), types.end());
            } else if (arg == "-ctv" || arg == "--cache-type-v") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<ggml_type> types;
                for (const auto & t : p) {
                    ggml_type gt = ggml_type_from_name(t);
                    if (gt == GGML_TYPE_COUNT) {
                        invalid_param = true;
                        break;
                    }
                    types.push_back(gt);
                }
                if (invalid_param) {
                    break;
                }
                params.type_v.insert(params.type_v.end(), types.begin(), types.end());
            } else if (arg == "-dt" || arg == "--defrag-thold") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<float>(argv[i], split_delim);
                params.defrag_thold.insert(params.defrag_thold.end(), p.begin(), p.end());
            } else if (arg == "-t" || arg == "--threads") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
            } else if (arg == "-C" || arg == "--cpu-mask") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);
                params.cpu_mask.insert(params.cpu_mask.end(), p.begin(), p.end());
            } else if (arg == "--cpu-strict") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.cpu_strict.insert(params.cpu_strict.end(), p.begin(), p.end());
            } else if (arg == "--poll") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.poll.insert(params.poll.end(), p.begin(), p.end());
            } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = parse_int_range(argv[i]);
                params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
            } else if (llama_supports_rpc() && (arg == "-rpc" || arg == "--rpc")) {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.rpc_servers.push_back(argv[i]);
            } else if (arg == "-sm" || arg == "--split-mode") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<std::string>(argv[i], split_delim);

                std::vector<llama_split_mode> modes;
                for (const auto & m : p) {
                    llama_split_mode mode;
                    if (m == "none") {
                        mode = LLAMA_SPLIT_MODE_NONE;
                    } else if (m == "layer") {
                        mode = LLAMA_SPLIT_MODE_LAYER;
                    } else if (m == "row") {
                        mode = LLAMA_SPLIT_MODE_ROW;
                    } else {
                        invalid_param = true;
                        break;
                    }
                    modes.push_back(mode);
                }
                if (invalid_param) {
                    break;
                }
                params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
            } else if (arg == "-mg" || arg == "--main-gpu") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.main_gpu = parse_int_range(argv[i]);
            } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_kv_offload.insert(params.no_kv_offload.end(), p.begin(), p.end());
            } else if (arg == "--numa") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                std::string value(argv[i]);
                if (value == "distribute" || value == "") {
                    params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
                } else if (value == "isolate") {
                    params.numa = GGML_NUMA_STRATEGY_ISOLATE;
                } else if (value == "numactl") {
                    params.numa = GGML_NUMA_STRATEGY_NUMACTL;
                } else {
                    invalid_param = true;
                    break;
                }
            } else if (arg == "-fa" || arg == "--flash-attn") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.flash_attn.insert(params.flash_attn.end(), p.begin(), p.end());
            } else if (arg == "-mmp" || arg == "--mmap") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.use_mmap.insert(params.use_mmap.end(), p.begin(), p.end());
            } else if (arg == "-embd" || arg == "--embeddings") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.embeddings.insert(params.embeddings.end(), p.begin(), p.end());
            } else if (arg == "-nopo" || arg == "--no-op-offload") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto p = string_split<bool>(argv[i], split_delim);
                params.no_op_offload.insert(params.no_op_offload.end(), p.begin(), p.end());
            } else if (arg == "-ts" || arg == "--tensor-split") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                for (auto ts : string_split<std::string>(argv[i], split_delim)) {
                    // split string by ; and /
                    const std::regex           regex{ R"([;/]+)" };
                    std::sregex_token_iterator it{ ts.begin(), ts.end(), regex, -1 };
                    std::vector<std::string>   split_arg{ it, {} };
                    GGML_ASSERT(split_arg.size() <= llama_max_devices());

                    std::vector<float> tensor_split(llama_max_devices());
                    for (size_t i = 0; i < llama_max_devices(); ++i) {
                        if (i < split_arg.size()) {
                            tensor_split[i] = std::stof(split_arg[i]);
                        } else {
                            tensor_split[i] = 0.0f;
                        }
                    }
                    params.tensor_split.push_back(tensor_split);
                }
            } else if (arg == "-ot" || arg == "--override-tensor") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                auto * value = argv[i];
                /* static */ std::map<std::string, ggml_backend_buffer_type_t> buft_list;
                if (buft_list.empty()) {
                    // enumerate all the devices and add their buffer types to the list
                    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                        auto * dev = ggml_backend_dev_get(i);
                        auto * buft = ggml_backend_dev_buffer_type(dev);
                        if (buft) {
                            buft_list[ggml_backend_buft_name(buft)] = buft;
                        }
                    }
                }
                auto override_group_span_len = std::strcspn(value, ",");
                bool last_group = false;
                do {
                    if (override_group_span_len == 0) {
                        // Adds an empty override-tensors for an empty span
                        params.tensor_buft_overrides.push_back({{}});
                        if (value[override_group_span_len] == '\0') {
                            value = &value[override_group_span_len];
                            last_group = true;
                        } else {
                            value = &value[override_group_span_len + 1];
                            override_group_span_len = std::strcspn(value, ",");
                        }
                        continue;
                    }
                    // Stamps null terminators into the argv
                    // value for this option to avoid the
                    // memory leak present in the implementation
                    // over in arg.cpp. Acceptable because we
                    // only parse these args once in this program.
                    auto * override_group = value;
                    if (value[override_group_span_len] == '\0') {
                        value = &value[override_group_span_len];
                        last_group = true;
                    } else {
                        value[override_group_span_len] = '\0';
                        value = &value[override_group_span_len + 1];
                    }
                    std::vector<llama_model_tensor_buft_override> group_tensor_buft_overrides{};
                    auto override_span_len = std::strcspn(override_group, ";");
                    while (override_span_len > 0) {
                        auto * override = override_group;
                        if (override_group[override_span_len] != '\0') {
                            override_group[override_span_len] = '\0';
                            override_group = &override_group[override_span_len + 1];
                        } else {
                            override_group = &override_group[override_span_len];
                        }
                        auto tensor_name_span_len = std::strcspn(override, "=");
                        if (tensor_name_span_len >= override_span_len) {
                            invalid_param = true;
                            break;
                        }
                        override[tensor_name_span_len] = '\0';
                        auto * tensor_name = override;
                        auto * buffer_type = &override[tensor_name_span_len + 1];
                        if (buft_list.find(buffer_type) == buft_list.end()) {
                            printf("error: unrecognized buffer type '%s'\n", buffer_type);
                            printf("Available buffer types:\n");
                            for (const auto & it : buft_list) {
                                printf("  %s\n", ggml_backend_buft_name(it.second));
                            }
                            invalid_param = true;
                            break;
                        }
                        group_tensor_buft_overrides.push_back({tensor_name, buft_list.at(buffer_type)});
                        override_span_len = std::strcspn(override_group, ";");
                    }
                    if (invalid_param) {
                        break;
                    }
                    group_tensor_buft_overrides.push_back({nullptr,nullptr});
                    params.tensor_buft_overrides.push_back(group_tensor_buft_overrides);
                    override_group_span_len = std::strcspn(value, ",");
                } while (!last_group);
            } else if (arg == "-r" || arg == "--repetitions") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.reps = std::stoi(argv[i]);
            } else if (arg == "--prio") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.prio = (enum ggml_sched_priority) std::stoi(argv[i]);
            } else if (arg == "--delay") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.delay = std::stoi(argv[i]);
            } else if (arg == "-o" || arg == "--output") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format);
            } else if (arg == "-oe" || arg == "--output-err") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                invalid_param = !output_format_from_str(argv[i], params.output_format_stderr);
            } else if (arg == "-v" || arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--progress") {
                params.progress = true;
            } else if (arg == "--no-warmup") {
                params.no_warmup = true;
            } else {
                invalid_param = true;
                break;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            invalid_param = true;
            break;
        }
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    // set defaults
    if (params.model.empty()) {
        params.model = cmd_params_defaults.model;
    }
    if (params.n_prompt.empty()) {
        params.n_prompt = cmd_params_defaults.n_prompt;
    }
    if (params.n_gen.empty()) {
        params.n_gen = cmd_params_defaults.n_gen;
    }
    if (params.n_pg.empty()) {
        params.n_pg = cmd_params_defaults.n_pg;
    }
    if (params.n_depth.empty()) {
        params.n_depth = cmd_params_defaults.n_depth;
    }
    if (params.n_batch.empty()) {
        params.n_batch = cmd_params_defaults.n_batch;
    }
    if (params.n_ubatch.empty()) {
        params.n_ubatch = cmd_params_defaults.n_ubatch;
    }
    if (params.type_k.empty()) {
        params.type_k = cmd_params_defaults.type_k;
    }
    if (params.type_v.empty()) {
        params.type_v = cmd_params_defaults.type_v;
    }
    if (params.defrag_thold.empty()) {
        params.defrag_thold = cmd_params_defaults.defrag_thold;
    }
    if (params.n_gpu_layers.empty()) {
        params.n_gpu_layers = cmd_params_defaults.n_gpu_layers;
    }
    if (params.rpc_servers.empty()) {
        params.rpc_servers = cmd_params_defaults.rpc_servers;
    }
    if (params.split_mode.empty()) {
        params.split_mode = cmd_params_defaults.split_mode;
    }
    if (params.main_gpu.empty()) {
        params.main_gpu = cmd_params_defaults.main_gpu;
    }
    if (params.no_kv_offload.empty()) {
        params.no_kv_offload = cmd_params_defaults.no_kv_offload;
    }
    if (params.flash_attn.empty()) {
        params.flash_attn = cmd_params_defaults.flash_attn;
    }
    if (params.tensor_split.empty()) {
        params.tensor_split = cmd_params_defaults.tensor_split;
    }
    if (params.tensor_buft_overrides.empty()) {
        params.tensor_buft_overrides = cmd_params_defaults.tensor_buft_overrides;
    }
    if (params.use_mmap.empty()) {
        params.use_mmap = cmd_params_defaults.use_mmap;
    }
    if (params.embeddings.empty()) {
        params.embeddings = cmd_params_defaults.embeddings;
    }
    if (params.no_op_offload.empty()) {
        params.no_op_offload = cmd_params_defaults.no_op_offload;
    }
    if (params.n_threads.empty()) {
        params.n_threads = cmd_params_defaults.n_threads;
    }
    if (params.cpu_mask.empty()) {
        params.cpu_mask = cmd_params_defaults.cpu_mask;
    }
    if (params.cpu_strict.empty()) {
        params.cpu_strict = cmd_params_defaults.cpu_strict;
    }
    if (params.poll.empty()) {
        params.poll = cmd_params_defaults.poll;
    }

    return params;
}

struct cmd_params_instance {
    std::string        model;
    int                n_prompt;
    int                n_gen;
    int                n_depth;
    int                n_batch;
    int                n_ubatch;
    ggml_type          type_k;
    ggml_type          type_v;
    float              defrag_thold;
    int                n_threads;
    std::string        cpu_mask;
    bool               cpu_strict;
    int                poll;
    int                n_gpu_layers;
    std::string        rpc_servers_str;
    llama_split_mode   split_mode;
    int                main_gpu;
    bool               no_kv_offload;
    bool               flash_attn;
    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool               use_mmap;
    bool               embeddings;
    bool               no_op_offload;

    llama_model_params to_llama_mparams() const {
        llama_model_params mparams = llama_model_default_params();

        mparams.n_gpu_layers = n_gpu_layers;
        if (!rpc_servers_str.empty()) {
            auto rpc_servers = string_split<std::string>(rpc_servers_str, ',');

            // add RPC devices
            if (!rpc_servers.empty()) {
                ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
                if (!rpc_reg) {
                    fprintf(stderr, "%s: failed to find RPC backend\n", __func__);
                    exit(1);
                }

                typedef ggml_backend_dev_t (*ggml_backend_rpc_add_device_t)(const char * endpoint);
                ggml_backend_rpc_add_device_t ggml_backend_rpc_add_device_fn = (ggml_backend_rpc_add_device_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_device");
                if (!ggml_backend_rpc_add_device_fn) {
                    fprintf(stderr, "%s: failed to find RPC device add function\n", __func__);
                    exit(1);
                }
                static std::vector<ggml_backend_dev_t> devices;
                devices.clear();
                // RPC devices should always come first for performance reasons
                for (const std::string & server : rpc_servers) {
                    ggml_backend_dev_t dev = ggml_backend_rpc_add_device_fn(server.c_str());
                    if (dev) {
                        devices.push_back(dev);
                    } else {
                        fprintf(stderr, "%s: failed to add RPC device for server '%s'\n", __func__, server.c_str());
                        exit(1);
                    }
                }
                // add local GPU devices if any
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    switch (ggml_backend_dev_type(dev)) {
                        case GGML_BACKEND_DEVICE_TYPE_CPU:
                        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                            // skip CPU backends since they are handled separately
                            break;

                        case GGML_BACKEND_DEVICE_TYPE_GPU:
                            devices.push_back(dev);
                            break;
                    }
                }
                devices.push_back(nullptr);
                mparams.devices = devices.data();
            }
        }
        mparams.split_mode   = split_mode;
        mparams.main_gpu     = main_gpu;
        mparams.tensor_split = tensor_split.data();
        mparams.use_mmap     = use_mmap;

        if (tensor_buft_overrides.empty()) {
            mparams.tensor_buft_overrides = nullptr;
        } else {
            GGML_ASSERT(tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
            mparams.tensor_buft_overrides = tensor_buft_overrides.data();
        }

        return mparams;
    }

    bool equal_mparams(const cmd_params_instance & other) const {
        return model == other.model && n_gpu_layers == other.n_gpu_layers && rpc_servers_str == other.rpc_servers_str &&
               split_mode == other.split_mode && main_gpu == other.main_gpu && use_mmap == other.use_mmap &&
               tensor_split == other.tensor_split && vec_tensor_buft_override_equal(tensor_buft_overrides, other.tensor_buft_overrides);
    }

    llama_context_params to_llama_cparams() const {
        llama_context_params cparams = llama_context_default_params();

        cparams.n_ctx        = n_prompt + n_gen + n_depth;
        cparams.n_batch      = n_batch;
        cparams.n_ubatch     = n_ubatch;
        cparams.type_k       = type_k;
        cparams.type_v       = type_v;
        cparams.defrag_thold = defrag_thold;
        cparams.offload_kqv  = !no_kv_offload;
        cparams.flash_attn   = flash_attn;
        cparams.embeddings   = embeddings;
        cparams.op_offload   = !no_op_offload;
        cparams.swa_full     = false;

        return cparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    // clang-format off
    for (const auto & m : params.model)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & rpc : params.rpc_servers)
    for (const auto & sm : params.split_mode)
    for (const auto & mg : params.main_gpu)
    for (const auto & ts : params.tensor_split)
    for (const auto & ot : params.tensor_buft_overrides)
    for (const auto & mmp : params.use_mmap)
    for (const auto & embd : params.embeddings)
    for (const auto & nopo : params.no_op_offload)
    for (const auto & nb : params.n_batch)
    for (const auto & nub : params.n_ubatch)
    for (const auto & tk : params.type_k)
    for (const auto & tv : params.type_v)
    for (const auto & defrag_thold : params.defrag_thold)
    for (const auto & nkvo : params.no_kv_offload)
    for (const auto & fa : params.flash_attn)
    for (const auto & nt : params.n_threads)
    for (const auto & cm : params.cpu_mask)
    for (const auto & cs : params.cpu_strict)
    for (const auto & nd : params.n_depth)
    for (const auto & pl : params.poll) {
        for (const auto & n_prompt : params.n_prompt) {
            if (n_prompt == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ n_prompt,
                /* .n_gen        = */ 0,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
            };
            instances.push_back(instance);
        }

        for (const auto & n_gen : params.n_gen) {
            if (n_gen == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ 0,
                /* .n_gen        = */ n_gen,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
            };
            instances.push_back(instance);
        }

        for (const auto & n_pg : params.n_pg) {
            if (n_pg.first == 0 && n_pg.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .model        = */ m,
                /* .n_prompt     = */ n_pg.first,
                /* .n_gen        = */ n_pg.second,
                /* .n_depth      = */ nd,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .defrag_thold = */ defrag_thold,
                /* .n_threads    = */ nt,
                /* .cpu_mask     = */ cm,
                /* .cpu_strict   = */ cs,
                /* .poll         = */ pl,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .tensor_split = */ ts,
                /* .tensor_buft_overrides = */ ot,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .no_op_offload= */ nopo,
            };
            instances.push_back(instance);
        }
    }
    // clang-format on

    return instances;
}

struct test {
    static const std::string build_commit;
    static const int         build_number;
    const std::string        cpu_info;
    const std::string        gpu_info;
    std::string              model_filename;
    std::string              model_type;
    uint64_t                 model_size;
    uint64_t                 model_n_params;
    int                      n_batch;
    int                      n_ubatch;
    int                      n_threads;
    std::string              cpu_mask;
    bool                     cpu_strict;
    int                      poll;
    ggml_type                type_k;
    ggml_type                type_v;
    float                    defrag_thold;
    int                      n_gpu_layers;
    llama_split_mode         split_mode;
    int                      main_gpu;
    bool                     no_kv_offload;
    bool                     flash_attn;
    std::vector<float>       tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    bool                     use_mmap;
    bool                     embeddings;
    bool                     no_op_offload;
    int                      n_prompt;
    int                      n_gen;
    int                      n_depth;
    std::string              test_time;
    std::vector<uint64_t>    samples_ns;

    test(const cmd_params_instance & inst, const llama_model * lmodel, const llama_context * ctx) :
        cpu_info(get_cpu_info()),
        gpu_info(get_gpu_info()) {

        model_filename = inst.model;
        char buf[128];
        llama_model_desc(lmodel, buf, sizeof(buf));
        model_type     = buf;
        model_size     = llama_model_size(lmodel);
        model_n_params = llama_model_n_params(lmodel);
        n_batch        = inst.n_batch;
        n_ubatch       = inst.n_ubatch;
        n_threads      = inst.n_threads;
        cpu_mask       = inst.cpu_mask;
        cpu_strict     = inst.cpu_strict;
        poll           = inst.poll;
        type_k         = inst.type_k;
        type_v         = inst.type_v;
        defrag_thold   = inst.defrag_thold;
        n_gpu_layers   = inst.n_gpu_layers;
        split_mode     = inst.split_mode;
        main_gpu       = inst.main_gpu;
        no_kv_offload  = inst.no_kv_offload;
        flash_attn     = inst.flash_attn;
        tensor_split   = inst.tensor_split;
        tensor_buft_overrides = inst.tensor_buft_overrides;
        use_mmap       = inst.use_mmap;
        embeddings     = inst.embeddings;
        no_op_offload  = inst.no_op_offload;
        n_prompt       = inst.n_prompt;
        n_gen          = inst.n_gen;
        n_depth        = inst.n_depth;
        // RFC 3339 date-time format
        time_t t       = time(NULL);
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        (void) ctx;
    }

    uint64_t avg_ns() const { return ::avg(samples_ns); }

    uint64_t stdev_ns() const { return ::stdev(samples_ns); }

    std::vector<double> get_ts() const {
        int                 n_tokens = n_prompt + n_gen;
        std::vector<double> ts;
        std::transform(samples_ns.begin(), samples_ns.end(), std::back_inserter(ts),
                       [n_tokens](uint64_t t) { return 1e9 * n_tokens / t; });
        return ts;
    }

    double avg_ts() const { return ::avg(get_ts()); }

    double stdev_ts() const { return ::stdev(get_ts()); }

    static std::string get_backend() {
        std::vector<std::string> backends;
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            auto *      reg  = ggml_backend_reg_get(i);
            std::string name = ggml_backend_reg_name(reg);
            if (name != "CPU") {
                backends.push_back(ggml_backend_reg_name(reg));
            }
        }
        return backends.empty() ? "CPU" : join(backends, ",");
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "build_commit", "build_number", "cpu_info",       "gpu_info",   "backends",     "model_filename",
            "model_type",   "model_size",   "model_n_params", "n_batch",    "n_ubatch",     "n_threads",
            "cpu_mask",     "cpu_strict",   "poll",           "type_k",     "type_v",       "n_gpu_layers",
            "split_mode",   "main_gpu",     "no_kv_offload",  "flash_attn", "tensor_split", "tensor_buft_overrides",
            "defrag_thold",
            "use_mmap",     "embeddings",   "no_op_offload",   "n_prompt",       "n_gen",      "n_depth",      "test_time",
            "avg_ns",       "stddev_ns",    "avg_ts",         "stddev_ts",
        };
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "build_number" || field == "n_batch" || field == "n_ubatch" || field == "n_threads" ||
            field == "poll" || field == "model_size" || field == "model_n_params" || field == "n_gpu_layers" ||
            field == "main_gpu" || field == "n_prompt" || field == "n_gen" || field == "n_depth" ||
            field == "avg_ns" || field == "stddev_ns" || field == "no_op_offload") {
            return INT;
        }
        if (field == "f16_kv" || field == "no_kv_offload" || field == "cpu_strict" || field == "flash_attn" ||
            field == "use_mmap" || field == "embeddings") {
            return BOOL;
        }
        if (field == "avg_ts" || field == "stddev_ts" || field == "defrag_thold") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        std::string tensor_buft_overrides_str;
        int         max_nonzero = 0;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i <= max_nonzero; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
            tensor_split_str += buf;
            if (i < max_nonzero) {
                tensor_split_str += "/";
            }
        }
        if (tensor_buft_overrides.size() == 1) {
            // Last element of tensor_buft_overrides is always a null pattern
            // so if it is only one element long, it must be a null pattern.
            GGML_ASSERT(tensor_buft_overrides[0].pattern == nullptr);
            tensor_buft_overrides_str += "none";
        } else {
            for (size_t i = 0; i < tensor_buft_overrides.size()-1; i++) {
                // Last element of tensor_buft_overrides is always a null pattern
                if (tensor_buft_overrides[i].pattern == nullptr) {
                    tensor_buft_overrides_str += "none";
                } else {
                    tensor_buft_overrides_str += tensor_buft_overrides[i].pattern;
                    tensor_buft_overrides_str += "=";
                    tensor_buft_overrides_str += ggml_backend_buft_name(tensor_buft_overrides[i].buft);
                }
                if (i + 2 < tensor_buft_overrides.size()) {
                    tensor_buft_overrides_str += ";";
                }
            }
        }
        std::vector<std::string> values = { build_commit,
                                            std::to_string(build_number),
                                            cpu_info,
                                            gpu_info,
                                            get_backend(),
                                            model_filename,
                                            model_type,
                                            std::to_string(model_size),
                                            std::to_string(model_n_params),
                                            std::to_string(n_batch),
                                            std::to_string(n_ubatch),
                                            std::to_string(n_threads),
                                            cpu_mask,
                                            std::to_string(cpu_strict),
                                            std::to_string(poll),
                                            ggml_type_name(type_k),
                                            ggml_type_name(type_v),
                                            std::to_string(n_gpu_layers),
                                            split_mode_str(split_mode),
                                            std::to_string(main_gpu),
                                            std::to_string(no_kv_offload),
                                            std::to_string(flash_attn),
                                            tensor_split_str,
                                            tensor_buft_overrides_str,
                                            std::to_string(defrag_thold),
                                            std::to_string(use_mmap),
                                            std::to_string(embeddings),
                                            std::to_string(no_op_offload),
                                            std::to_string(n_prompt),
                                            std::to_string(n_gen),
                                            std::to_string(n_depth),
                                            test_time,
                                            std::to_string(avg_ns()),
                                            std::to_string(stdev_ns()),
                                            std::to_string(avg_ts()),
                                            std::to_string(stdev_ts()) };
        return values;
    }

    std::map<std::string, std::string> get_map() const {
        std::map<std::string, std::string> map;
        auto                               fields = get_fields();
        auto                               values = get_values();
        std::transform(fields.begin(), fields.end(), values.begin(), std::inserter(map, map.end()),
                       std::make_pair<const std::string &, const std::string &>);
        return map;
    }
};

const std::string test::build_commit = LLAMA_COMMIT;
const int         test::build_number = LLAMA_BUILD_NUMBER;

struct printer {
    virtual ~printer() {}

    FILE * fout;

    virtual void print_header(const cmd_params & params) { (void) params; }

    virtual void print_test(const test & t) = 0;

    virtual void print_footer() {}
};

struct csv_printer : public printer {
    static std::string escape_csv(const std::string & field) {
        std::string escaped = "\"";
        for (auto c : field) {
            if (c == '"') {
                escaped += "\"";
            }
            escaped += c;
        }
        escaped += "\"";
        return escaped;
    }

    void print_header(const cmd_params & params) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "%s\n", join(fields, ",").c_str());
        (void) params;
    }

    void print_test(const test & t) override {
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_csv);
        fprintf(fout, "%s\n", join(values, ",").c_str());
    }
};

static std::string escape_json(const std::string & value) {
    std::string escaped;
    for (auto c : value) {
        if (c == '"') {
            escaped += "\\\"";
        } else if (c == '\\') {
            escaped += "\\\\";
        } else if (c <= 0x1f) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            escaped += buf;
        } else {
            escaped += c;
        }
    }
    return escaped;
}

static std::string format_json_value(const std::string & field, const std::string & value) {
    switch (test::get_field_type(field)) {
        case test::STRING:
            return "\"" + escape_json(value) + "\"";
        case test::BOOL:
            return value == "0" ? "false" : "true";
        default:
            return value;
    }
}

struct json_printer : public printer {
    bool first = true;

    void print_header(const cmd_params & params) override {
        fprintf(fout, "[\n");
        (void) params;
    }

    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "    \"%s\": %s,\n", fields.at(i).c_str(),
                    format_json_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        if (first) {
            first = false;
        } else {
            fprintf(fout, ",\n");
        }
        fprintf(fout, "  {\n");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "    \"samples_ns\": [ %s ],\n", join(t.samples_ns, ", ").c_str());
        fprintf(fout, "    \"samples_ts\": [ %s ]\n", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "  }");
        fflush(fout);
    }

    void print_footer() override { fprintf(fout, "\n]\n"); }
};

struct jsonl_printer : public printer {
    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "\"%s\": %s, ", fields.at(i).c_str(), format_json_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        fprintf(fout, "{");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "\"samples_ns\": [ %s ],", join(t.samples_ns, ", ").c_str());
        fprintf(fout, "\"samples_ts\": [ %s ]", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "}\n");
        fflush(fout);
    }
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 20;
        }
        if (field == "size" || field == "params") {
            return 10;
        }
        if (field == "n_gpu_layers") {
            return 3;
        }
        if (field == "n_threads") {
            return 7;
        }
        if (field == "n_batch") {
            return 7;
        }
        if (field == "n_ubatch") {
            return 8;
        }
        if (field == "type_k" || field == "type_v") {
            return 6;
        }
        if (field == "split_mode") {
            return 5;
        }
        if (field == "flash_attn") {
            return 2;
        }
        if (field == "use_mmap") {
            return 4;
        }
        if (field == "test") {
            return 15;
        }
        if (field == "no_op_offload") {
            return 4;
        }

        int width = std::max((int) field.length(), 10);

        if (test::get_field_type(field) == test::STRING) {
            return -width;
        }
        return width;
    }

    static std::string get_field_display_name(const std::string & field) {
        if (field == "n_gpu_layers") {
            return "ngl";
        }
        if (field == "split_mode") {
            return "sm";
        }
        if (field == "n_threads") {
            return "threads";
        }
        if (field == "no_kv_offload") {
            return "nkvo";
        }
        if (field == "flash_attn") {
            return "fa";
        }
        if (field == "use_mmap") {
            return "mmap";
        }
        if (field == "embeddings") {
            return "embd";
        }
        if (field == "no_op_offload") {
            return "nopo";
        }
        if (field == "tensor_split") {
            return "ts";
        }
        if (field == "tensor_buft_overrides") {
            return "ot";
        }
        return field;
    }

    void print_header(const cmd_params & params) override {
        // select fields to print
        fields.emplace_back("model");
        fields.emplace_back("size");
        fields.emplace_back("params");
        fields.emplace_back("backend");
        bool is_cpu_backend = test::get_backend().find("CPU") != std::string::npos ||
                              test::get_backend().find("BLAS") != std::string::npos;
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        if (params.n_threads.size() > 1 || params.n_threads != cmd_params_defaults.n_threads || is_cpu_backend) {
            fields.emplace_back("n_threads");
        }
        if (params.cpu_mask.size() > 1 || params.cpu_mask != cmd_params_defaults.cpu_mask) {
            fields.emplace_back("cpu_mask");
        }
        if (params.cpu_strict.size() > 1 || params.cpu_strict != cmd_params_defaults.cpu_strict) {
            fields.emplace_back("cpu_strict");
        }
        if (params.poll.size() > 1 || params.poll != cmd_params_defaults.poll) {
            fields.emplace_back("poll");
        }
        if (params.n_batch.size() > 1 || params.n_batch != cmd_params_defaults.n_batch) {
            fields.emplace_back("n_batch");
        }
        if (params.n_ubatch.size() > 1 || params.n_ubatch != cmd_params_defaults.n_ubatch) {
            fields.emplace_back("n_ubatch");
        }
        if (params.type_k.size() > 1 || params.type_k != cmd_params_defaults.type_k) {
            fields.emplace_back("type_k");
        }
        if (params.type_v.size() > 1 || params.type_v != cmd_params_defaults.type_v) {
            fields.emplace_back("type_v");
        }
        if (params.defrag_thold.size() > 1 || params.defrag_thold != cmd_params_defaults.defrag_thold) {
            fields.emplace_back("defrag_thold");
        }
        if (params.main_gpu.size() > 1 || params.main_gpu != cmd_params_defaults.main_gpu) {
            fields.emplace_back("main_gpu");
        }
        if (params.split_mode.size() > 1 || params.split_mode != cmd_params_defaults.split_mode) {
            fields.emplace_back("split_mode");
        }
        if (params.no_kv_offload.size() > 1 || params.no_kv_offload != cmd_params_defaults.no_kv_offload) {
            fields.emplace_back("no_kv_offload");
        }
        if (params.flash_attn.size() > 1 || params.flash_attn != cmd_params_defaults.flash_attn) {
            fields.emplace_back("flash_attn");
        }
        if (params.tensor_split.size() > 1 || params.tensor_split != cmd_params_defaults.tensor_split) {
            fields.emplace_back("tensor_split");
        }
        if (params.tensor_buft_overrides.size() > 1 || !vec_vec_tensor_buft_override_equal(params.tensor_buft_overrides, cmd_params_defaults.tensor_buft_overrides)) {
            fields.emplace_back("tensor_buft_overrides");
        }
        if (params.use_mmap.size() > 1 || params.use_mmap != cmd_params_defaults.use_mmap) {
            fields.emplace_back("use_mmap");
        }
        if (params.embeddings.size() > 1 || params.embeddings != cmd_params_defaults.embeddings) {
            fields.emplace_back("embeddings");
        }
        if (params.no_op_offload.size() > 1 || params.no_op_offload != cmd_params_defaults.no_op_offload) {
            fields.emplace_back("no_op_offload");
        }
        fields.emplace_back("test");
        fields.emplace_back("t/s");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char        buf[128];
            if (field == "model") {
                value = t.model_type;
            } else if (field == "size") {
                if (t.model_size < 1024 * 1024 * 1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.model_size / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            } else if (field == "params") {
                if (t.model_n_params < 1000 * 1000 * 1000) {
                    snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f B", t.model_n_params / 1e9);
                }
                value = buf;
            } else if (field == "backend") {
                value = test::get_backend();
            } else if (field == "test") {
                if (t.n_prompt > 0 && t.n_gen == 0) {
                    snprintf(buf, sizeof(buf), "pp%d", t.n_prompt);
                } else if (t.n_gen > 0 && t.n_prompt == 0) {
                    snprintf(buf, sizeof(buf), "tg%d", t.n_gen);
                } else {
                    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.n_prompt, t.n_gen);
                }
                if (t.n_depth > 0) {
                    int len = strlen(buf);
                    snprintf(buf + len, sizeof(buf) - len, " @ d%d", t.n_depth);
                }
                value = buf;
            } else if (field == "t/s") {
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.avg_ts(), t.stdev_ts());
                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = vmap.at(field);
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            if (field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

    void print_footer() override {
        fprintf(fout, "\nbuild: %s (%d)\n", test::build_commit.c_str(), test::build_number);
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "TEXT";
            case test::BOOL:
            case test::INT:
                return "INTEGER";
            case test::FLOAT:
                return "REAL";
            default:
                assert(false);
                exit(1);
        }
    }

    void print_header(const cmd_params & params) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS llama_bench (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields.at(i).c_str(), get_sql_field_type(fields.at(i)).c_str(),
                    i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n");
        fprintf(fout, "\n");
        (void) params;
    }

    void print_test(const test & t) override {
        fprintf(fout, "INSERT INTO llama_bench (%s) ", join(test::get_fields(), ", ").c_str());
        fprintf(fout, "VALUES (");
        std::vector<std::string> values = t.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values.at(i).c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

static bool test_prompt(llama_context * ctx, int n_prompt, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode prompt batch, res = %d\n", __func__, res);
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    return true;
}

static bool test_gen(llama_context * ctx, int n_gen, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    for (int i = 0; i < n_gen; i++) {
        int res = llama_decode(ctx, llama_batch_get_one(&token, 1));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode generation batch, res = %d\n", __func__, res);
            return false;
        }
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }
    return true;
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case NONE:
            return nullptr;
        case CSV:
            return std::unique_ptr<printer>(new csv_printer());
        case JSON:
            return std::unique_ptr<printer>(new json_printer());
        case JSONL:
            return std::unique_ptr<printer>(new jsonl_printer());
        case MARKDOWN:
            return std::unique_ptr<printer>(new markdown_printer());
        case SQL:
            return std::unique_ptr<printer>(new sql_printer());
    }
    GGML_ABORT("fatal error");
}

int main(int argc, char ** argv) {
    // try to set locale for unicode characters in markdown
    setlocale(LC_CTYPE, ".UTF-8");

#if !defined(NDEBUG)
    fprintf(stderr, "warning: asserts enabled, performance may be affected\n");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    fprintf(stderr, "warning: debug build, performance may be affected\n");
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    fprintf(stderr, "warning: sanitizer enabled, performance may be affected\n");
#endif

    // initialize backends
    ggml_backend_load_all();

    cmd_params params = parse_cmd_params(argc, argv);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        fprintf(stderr, "%s: error: CPU backend is not loaded\n", __func__);
        return 1;
    }
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");

    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    set_process_priority(params.prio);

    // initialize printer
    std::unique_ptr<printer> p     = create_printer(params.output_format);
    std::unique_ptr<printer> p_err = create_printer(params.output_format_stderr);

    if (p) {
        p->fout = stdout;
        p->print_header(params);
    }

    if (p_err) {
        p_err->fout = stderr;
        p_err->print_header(params);
    }

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    llama_model *               lmodel    = nullptr;
    const cmd_params_instance * prev_inst = nullptr;

    int  params_idx   = 0;
    auto params_count = params_instances.size();
    for (const auto & inst : params_instances) {
        params_idx++;
        if (params.progress) {
            fprintf(stderr, "llama-bench: benchmark %d/%zu: starting\n", params_idx, params_count);
        }
        // keep the same model between tests when possible
        if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
            if (lmodel) {
                llama_model_free(lmodel);
            }

            lmodel = llama_model_load_from_file(inst.model.c_str(), inst.to_llama_mparams());
            if (lmodel == NULL) {
                fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                return 1;
            }
            prev_inst = &inst;
        }

        llama_context * ctx = llama_init_from_model(lmodel, inst.to_llama_cparams());
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
            llama_model_free(lmodel);
            return 1;
        }

        test t(inst, lmodel, ctx);

        llama_memory_clear(llama_get_memory(ctx), false);

        // cool off before the test
        if (params.delay) {
            std::this_thread::sleep_for(std::chrono::seconds(params.delay));
        }

        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(t.n_threads);
        if (!parse_cpu_mask(t.cpu_mask, tpp.cpumask)) {
            fprintf(stderr, "%s: failed to parse cpu-mask: %s\n", __func__, t.cpu_mask.c_str());
            exit(1);
        }
        tpp.strict_cpu = t.cpu_strict;
        tpp.poll       = t.poll;
        tpp.prio       = params.prio;

        struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
        if (!threadpool) {
            fprintf(stderr, "%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
            exit(1);
        }

        llama_attach_threadpool(ctx, threadpool, NULL);

        // warmup run
        if (!params.no_warmup) {
            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup prompt run\n", params_idx, params_count);
                }
                //test_prompt(ctx, std::min(t.n_batch, std::min(t.n_prompt, 32)), 0, t.n_batch, t.n_threads);
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt warmup\n", __func__);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup generation run\n", params_idx, params_count);
                }
                bool res = test_gen(ctx, 1, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen warmup\n", __func__);
                    exit(1);
                }
            }
        }

        for (int i = 0; i < params.reps; i++) {
            llama_memory_clear(llama_get_memory(ctx), false);

            if (t.n_depth > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: depth run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_prompt(ctx, t.n_depth, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run depth\n", __func__);
                    exit(1);
                }
            }

            uint64_t t_start = get_time_ns();

            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: prompt run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt\n", __func__);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: generation run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_gen(ctx, t.n_gen, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen\n", __func__);
                    exit(1);
                }
            }

            uint64_t t_ns = get_time_ns() - t_start;
            t.samples_ns.push_back(t_ns);
        }

        if (p) {
            p->print_test(t);
            fflush(p->fout);
        }

        if (p_err) {
            p_err->print_test(t);
            fflush(p_err->fout);
        }

        llama_perf_context_print(ctx);

        llama_free(ctx);

        ggml_threadpool_free_fn(threadpool);
    }

    llama_model_free(lmodel);

    if (p) {
        p->print_footer();
    }

    if (p_err) {
        p_err->print_footer();
    }

    llama_backend_free();

    return 0;
}
