#include "arg.h"

#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "json-schema-to-grammar.h"

using json = nlohmann::ordered_json;

common_arg & common_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = std::move(examples);
    return *this;
}

common_arg & common_arg::set_excludes(std::initializer_list<enum llama_example> excludes) {
    this->excludes = std::move(excludes);
    return *this;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sparam() {
    is_sparam = true;
    return *this;
}

bool common_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool common_arg::is_exclude(enum llama_example ex) {
    return excludes.find(ex) != excludes.end();
}

bool common_arg::get_value_from_env(std::string & output) {
    if (env == nullptr) return false;
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() {
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string common_arg::to_string() {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    for (const auto arg : args) {
        if (arg == args.front()) {
            if (args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

//
// utils
//

static void common_params_handle_model_default(
        std::string & model,
        std::string & model_url,
        std::string & hf_repo,
        std::string & hf_file) {
    if (!hf_repo.empty()) {
        // short-hand to avoid specifying --hf-file -> default it to --model
        if (hf_file.empty()) {
            if (model.empty()) {
                throw std::invalid_argument("error: --hf-repo requires either --hf-file or --model\n");
            }
            hf_file = model;
        } else if (model.empty()) {
            // this is to avoid different repo having same file name, or same file name in different subdirs
            std::string filename = hf_repo + "_" + hf_file;
            // to make sure we don't have any slashes in the filename
            string_replace_all(filename, "/", "_");
            model = fs_get_cache_file(filename);
        }
    } else if (!model_url.empty()) {
        if (model.empty()) {
            auto f = string_split<std::string>(model_url, '#').front();
            f = string_split<std::string>(f, '?').front();
            model = fs_get_cache_file(string_split<std::string>(f, '/').back());
        }
    } else if (model.empty()) {
        model = DEFAULT_MODEL_PATH;
    }
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    for (const auto & type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

//
// CLI argument parsing functions
//

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    std::string arg;
    const std::string arg_prefix = "--";
    common_params & params = ctx_arg.params;

    std::unordered_map<std::string, common_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && (value == "1" || value == "true")) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        auto opt = *arg_to_options[arg];
        if (opt.has_value_from_env()) {
            fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
        }
        try {
            if (opt.handler_void) {
                opt.handler_void(params);
                continue;
            }

            // arg with single value
            check_arg(i);
            std::string val = argv[++i];
            if (opt.handler_int) {
                opt.handler_int(params, std::stoi(val));
                continue;
            }
            if (opt.handler_string) {
                opt.handler_string(params, val);
                continue;
            }

            // arg with 2 values
            check_arg(i);
            std::string val2 = argv[++i];
            if (opt.handler_str_str) {
                opt.handler_str_str(params, val, val2);
                continue;
            }
        } catch (std::exception & e) {
            throw std::invalid_argument(string_format(
                "error while handling argument \"%s\": %s\n\n"
                "usage:\n%s\n\nto show complete usage, run with -h",
                arg.c_str(), e.what(), arg_to_options[arg]->to_string().c_str()));
        }
    }

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    postprocess_cpu_params(params.speculative.cpuparams,       &params.cpuparams);
    postprocess_cpu_params(params.speculative.cpuparams_batch, &params.cpuparams_batch);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    // TODO: refactor model params in a common struct
    common_params_handle_model_default(params.model,         params.model_url,         params.hf_repo,         params.hf_file);
    common_params_handle_model_default(params.vocoder.model, params.vocoder.model_url, params.vocoder.hf_repo, params.vocoder.hf_file);

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    if (params.reranking && params.embedding) {
        throw std::invalid_argument("error: either --embedding or --reranking can be specified, but not both");
    }

    return true;
}

static void common_params_print_usage(common_params_context & ctx_arg) {
    auto print_options = [](std::vector<common_arg *> & options) {
        for (common_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sparam_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common params -----\n\n");
    print_options(common_options);
    printf("\n\n----- sampling params -----\n\n");
    print_options(sparam_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific params -----\n\n");
    print_options(specific_options);
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            exit(0);
        }
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // load dynamic backends
    ggml_backend_load_all();

    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    sampler_type_names.pop_back();


    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](common_arg arg) {
        if ((arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) && !arg.is_exclude(ex)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };


    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ));
    add_opt(common_arg(
        {"--no-display-prompt"},
        string_format("don't print prompt at generation (default: %s)", !params.display_prompt ? "true" : "false"),
        [](common_params & params) {
            params.display_prompt = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-co", "--color"},
        string_format("colorise output to distinguish prompt and user input from generations (default: %s)", params.use_color ? "true" : "false"),
        [](common_params & params) {
            params.use_color = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](common_params & params, const std::string & value) {
            params.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](common_params & params, const std::string & value) {
            params.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format("number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)", params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--no-context-shift"},
        string_format("disables context shift on inifinite text generation (default: %s)", params.ctx_shift ? "disabled" : "enabled"),
        [](common_params & params) {
            params.ctx_shift = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY}).set_env("LLAMA_ARG_NO_CONTEXT_SHIFT"));
    add_opt(common_arg(
        {"--chunks"}, "N",
        string_format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](common_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"-fa", "--flash-attn"},
        string_format("enable Flash Attention (default: %s)", params.flash_attn ? "enabled" : "disabled"),
        [](common_params & params) {
            params.flash_attn = true;
        }
    ).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        ex == LLAMA_EXAMPLE_MAIN
            ? "prompt to start generation with\nif -cnv is set, this will be used as system prompt"
            : "prompt to start generation with",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--no-perf"},
        string_format("disable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](common_params & params) {
            params.no_perf = true;
            params.sampling.no_perf = true;
        }
    ).set_env("LLAMA_ARG_NO_PERF"));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--in-file"}, "FNAME",
        "an input file (repeat to specify multiple files)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.in_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-e", "--escape"},
        string_format("process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params) {
            params.escape = true;
        }
    ));
    add_opt(common_arg(
        {"--no-escape"},
        "do not process escape sequences",
        [](common_params & params) {
            params.escape = false;
        }
    ));
    add_opt(common_arg(
        {"-ptc", "--print-token-count"}, "N",
        string_format("print token count every N tokens (default: %d)", params.n_print),
        [](common_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        string_format(
            "run in conversation mode:\n"
            "- does not print special tokens and suffix/prefix\n"
            "- interactive mode is also enabled\n"
            "(default: %s)",
            params.conversation ? "true" : "false"
        ),
        [](common_params & params) {
            params.conversation = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(common_arg(
        {"--no-warmup"},
        "skip warming up the model with an empty run",
        [](common_params & params) {
            params.warmup = false;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spm-infill"},
        string_format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](common_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_INFILL}));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--sampling-seq", "--sampler-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers = common_sampler_types_from_chars(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--temp"}, "N",
        string_format("temperature (default: %.1f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.1f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.1f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.1f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.1f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--typical"}, "N",
        string_format("locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)params.sampling.typ_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typ_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_repeat = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_present = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_freq = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.1f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.1f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.1f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.1f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--grammar"}, "GRAMMAR",
        string_format("BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", params.sampling.grammar.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(params.sampling.grammar)
            );
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = json_schema_to_grammar(json::parse(value));
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--pooling"}, "{none,mean,cls,last,rank}",
        "pooling type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_POOLING"));
    add_opt(common_arg(
        {"--attention"}, "{causal,non-causal}",
        "attention type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-gan", "--grp-attn-n"}, "N",
        string_format("group-attention factor (default: %d)", params.grp_attn_n),
        [](common_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N").set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        string_format("group-attention width (default: %d)", params.grp_attn_w),
        [](common_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W").set_examples({LLAMA_EXAMPLE_MAIN}));
    add_opt(common_arg(
        {"-dkvc", "--dump-kv-cache"},
        "verbose print of the KV cache",
        [](common_params & params) {
            params.dump_kv_cache = true;
        }
    ));
    add_opt(common_arg(
        {"-nkvo", "--no-kv-offload"},
        "disable KV offload",
        [](common_params & params) {
            params.no_kv_offload = true;
        }
    ).set_env("LLAMA_ARG_NO_KV_OFFLOAD"));
    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--perplexity", "--all-logits"},
        string_format("return logits for all tokens in the batch (default: %s)", params.logits_all ? "true" : "false"),
        [](common_params & params) {
            params.logits_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag-tasks"}, "N",
        string_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](common_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande-tasks"}, "N",
        string_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](common_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice-tasks"}, "N",
        string_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](common_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](common_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](common_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-stride"}, "N",
        string_format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](common_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-output-type"}, "<0|1>",
        string_format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](common_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
        string_format("KV cache defragmentation threshold (default: %.1f, < 0 - disabled)", (double)params.defrag_thold),
        [](common_params & params, const std::string & value) {
            params.defrag_thold = std::stof(value);
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    add_opt(common_arg(
        {"-np", "--parallel"}, "N",
        string_format("number of parallel sequences to decode (default: %d)", params.n_parallel),
        [](common_params & params, int value) {
            params.n_parallel = value;
        }
    ).set_env("LLAMA_ARG_N_PARALLEL"));
    add_opt(common_arg(
        {"-ns", "--sequences"}, "N",
        string_format("number of sequences to decode (default: %d)", params.n_sequences),
        [](common_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-cb", "--cont-batching"},
        string_format("enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](common_params & params) {
            params.cont_batching = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(common_arg(
        {"-nocb", "--no-cont-batching"},
        "disable continuous batching",
        [](common_params & params) {
            params.cont_batching = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_CONT_BATCHING"));
    add_opt(common_arg(
        {"--mmproj"}, "FILE",
        "path to a multimodal projector file for LLaVA. see examples/llava/README.md",
        [](common_params & params, const std::string & value) {
            params.mmproj = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
    add_opt(common_arg(
        {"--image"}, "FILE",
        "path to an image file. use with multimodal models. Specify multiple times for batching",
        [](common_params & params, const std::string & value) {
            params.image.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_LLAVA}));
    if (llama_supports_rpc()) {
        add_opt(common_arg(
            {"--rpc"}, "SERVERS",
            "comma separated list of RPC servers",
            [](common_params & params, const std::string & value) {
                params.rpc_servers = value;
            }
        ).set_env("LLAMA_ARG_RPC"));
    }
    add_opt(common_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](common_params & params) {
            params.use_mlock = true;
        }
    ).set_env("LLAMA_ARG_MLOCK"));
    add_opt(common_arg(
        {"--no-mmap"},
        "do not memory-map model (slower load but may reduce pageouts if not using mlock)",
        [](common_params & params) {
            params.use_mmap = false;
        }
    ).set_env("LLAMA_ARG_NO_MMAP"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggerganov/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"-dev", "--device"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.devices = parse_device_list(value);
        }
    ).set_env("LLAMA_ARG_DEVICE"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            printf("Available devices:\n");
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                }
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        "number of layers to store in VRAM",
        [](common_params & params, int value) {
            params.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(common_arg(
        {"-sm", "--split-mode"}, "{none,layer,row}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs\n"
        "- row: split rows across GPUs",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;
            if (arg_next == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (arg_next == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (arg_next == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %d input configs, but system only has %d devices", (int)split_arg.size(), (int)llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(common_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        string_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](common_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE",
        "advanced option to override model metadata by key. may be specified multiple times.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false",
        [](common_params & params, const std::string & value) {
            if (!string_parse_kv_override(value.c_str(), params.kv_overrides)) {
                throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", value.c_str()));
            }
        }
    ));
    add_opt(common_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (can be repeated to use multiple adapters)",
        [](common_params & params, const std::string & value) {
            params.lora_adapters.push_back({ std::string(value), 1.0, nullptr });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-scaled"}, "FNAME", "SCALE",
        "path to LoRA adapter with user defined scaling (can be repeated to use multiple adapters)",
        [](common_params & params, const std::string & fname, const std::string & scale) {
            params.lora_adapters.push_back({ fname, std::stof(scale), nullptr });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: this argument can be repeated to add multiple control vectors",
        [](common_params & params, const std::string & value) {
            params.control_vectors.push_back({ 1.0f, value, });
        }
    ));
    add_opt(common_arg(
        {"--control-vector-scaled"}, "FNAME", "SCALE",
        "add a control vector with user defined scaling SCALE\n"
        "note: this argument can be repeated to add multiple scaled control vectors",
        [](common_params & params, const std::string & fname, const std::string & scale) {
            params.control_vectors.push_back({ std::stof(scale), fname });
        }
    ));
    add_opt(common_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](common_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(common_arg(
        {"-a", "--alias"}, "STRING",
        "set alias for model name (to be used by REST API)",
        [](common_params & params, const std::string & value) {
            params.model_alias = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? std::string("model path from which to load base model")
            : string_format(
                "model path (default: `models/$filename` with filename from `--hf-file` "
                "or `--model-url` if set, otherwise %s)", DEFAULT_MODEL_PATH
            ),
        [](common_params & params, const std::string & value) {
            params.model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model_url = value;
        }
    ).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(common_arg(
        {"-hfr", "--hf-repo"}, "REPO",
        "Hugging Face model repository (default: unused)",
        [](common_params & params, const std::string & value) {
            params.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(common_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file (default: unused)",
        [](common_params & params, const std::string & value) {
            params.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(common_arg(
        {"-hfrv", "--hf-repo-v"}, "REPO",
        "Hugging Face model repository for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO_V"));
    add_opt(common_arg(
        {"-hffv", "--hf-file-v"}, "FILE",
        "Hugging Face model file for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE_V"));
    add_opt(common_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](common_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_env("HF_TOKEN"));
    add_opt(common_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (repeat to specify multiple files)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.context_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-size"}, "N",
        string_format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](common_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-separator"}, "STRING",
        string_format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](common_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--junk"}, "N",
        string_format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](common_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"--pos"}, "N",
        string_format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](common_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        string_format("output file (default: '%s')",
            ex == LLAMA_EXAMPLE_EXPORT_LORA
                ? params.lora_outfile.c_str()
                : ex == LLAMA_EXAMPLE_CVECTOR_GENERATOR
                    ? params.cvector_outfile.c_str()
                    : params.out_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file = value;
            params.cvector_outfile = value;
            params.lora_outfile = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"-ofreq", "--output-frequency"}, "N",
        string_format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](common_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--save-frequency"}, "N",
        string_format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](common_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--process-output"},
        string_format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](common_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--no-ppl"},
        string_format("do not compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](common_params & params) {
            params.compute_ppl = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--chunk", "--from-chunk"}, "N",
        string_format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](common_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-pps"},
        string_format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](common_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--embd-normalize"}, "N",
        string_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](common_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix",
        [](common_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-separator"}, "STRING",
        "separator of embeddings (default \\n) for example \"<#sep#>\"",
        [](common_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--host"}, "HOST",
        string_format("ip address to listen (default: %s)", params.hostname.c_str()),
        [](common_params & params, const std::string & value) {
            params.hostname = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(common_arg(
        {"--port"}, "PORT",
        string_format("port to listen (default: %d)", params.port),
        [](common_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(common_arg(
        {"--path"}, "PATH",
        string_format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](common_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(common_arg(
        {"--no-webui"},
        string_format("Disable the Web UI (default: %s)", params.webui ? "enabled" : "disabled"),
        [](common_params & params) {
            params.webui = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_WEBUI"));
    add_opt(common_arg(
        {"--embedding", "--embeddings"},
        string_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](common_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(common_arg(
        {"--reranking", "--rerank"},
        string_format("enable reranking endpoint on server (default: %s)", params.reranking ? "enabled" : "disabled"),
        [](common_params & params) {
            params.reranking = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
    add_opt(common_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication (default: none)",
        [](common_params & params, const std::string & value) {
            params.api_keys.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(common_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty()) {
                        params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](common_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](common_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(common_arg(
        {"-to", "--timeout"}, "N",
        string_format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](common_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
    add_opt(common_arg(
        {"--threads-http"}, "N",
        string_format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](common_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(common_arg(
        {"--cache-reuse"}, "N",
        string_format("min chunk size to attempt reusing from the cache via KV shifting (default: %d)", params.n_cache_reuse),
        [](common_params & params, int value) {
            params.n_cache_reuse = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--metrics"},
        string_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(common_arg(
        {"--slots"},
        string_format("enable slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_slots = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--props"},
        string_format("enable changing global properties via POST /props (default: %s)", params.endpoint_props ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_props = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_PROPS"));
    add_opt(common_arg(
        {"--no-slots"},
        "disables slots monitoring endpoint",
        [](common_params & params) {
            params.endpoint_slots = false;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_NO_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.slot_save_path = value;
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            if (!common_chat_verify_template(value)) {
                throw std::runtime_error(string_format(
                    "error: the supplied chat template is not supported: %s\n"
                    "note: llama.cpp does not use jinja parser, we only support commonly used templates\n",
                    value.c_str()
                ));
            }
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        string_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](common_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-init-without-apply"},
        string_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](common_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_MAIN, LLAMA_EXAMPLE_INFILL}));
    add_opt(common_arg(
        {"--positive-file"}, "FNAME",
        string_format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--negative-file"}, "FNAME",
        string_format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-batch"}, "N",
        string_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](common_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-iter"}, "N",
        string_format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](common_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ));
    add_opt(common_arg(
        {"--log-colors"},
        "Enable colored logging",
        [](common_params &) {
            common_log_set_colors(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
            common_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        "Set the verbosity threshold. Messages with a higher verbosity will be ignored.",
        [](common_params & params, int value) {
            params.verbosity = value;
            common_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        "Enable prefx in log messages",
        [](common_params &) {
            common_log_set_prefix(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &) {
            common_log_set_timestamps(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_TIMESTAMPS"));

    // speculative parameters
    add_opt(common_arg(
        {"-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.n_threads = value;
            if (params.speculative.cpuparams.n_threads <= 0) {
                params.speculative.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.n_threads = value;
            if (params.speculative.cpuparams_batch.n_threads <= 0) {
                params.speculative.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--prio-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll])",
        [](common_params & params, int value) {
            params.speculative.cpuparams.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--prio-batch-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--draft-max", "--draft", "--draft-n"}, "N",
        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.n_max),
        [](common_params & params, int value) {
            params.speculative.n_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        string_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.n_min),
        [](common_params & params, int value) {
            params.speculative.n_min = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_DRAFT_MIN"));
    add_opt(common_arg(
        {"--draft-p-split"}, "P",
        string_format("speculative decoding split probability (default: %.1f)", (double)params.speculative.p_split),
        [](common_params & params, const std::string & value) {
            params.speculative.p_split = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}).set_env("LLAMA_ARG_DRAFT_P_SPLIT"));
    add_opt(common_arg(
        {"--draft-p-min"}, "P",
        string_format("minimum speculative decoding probability (greedy) (default: %.1f)", (double)params.speculative.p_min),
        [](common_params & params, const std::string & value) {
            params.speculative.p_min = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_DRAFT_P_MIN"));
    add_opt(common_arg(
        {"-cd", "--ctx-size-draft"}, "N",
        string_format("size of the prompt context for the draft model (default: %d, 0 = loaded from model)", params.speculative.n_ctx),
        [](common_params & params, int value) {
            params.speculative.n_ctx = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CTX_SIZE_DRAFT"));
    add_opt(common_arg(
        {"-devd", "--device-draft"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.speculative.devices = parse_device_list(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        "number of layers to store in VRAM for the draft model",
        [](common_params & params, int value) {
            params.speculative.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    add_opt(common_arg(
        {"-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODEL_DRAFT"));

    add_opt(common_arg(
        {"-mv", "--model-vocoder"}, "FNAME",
        "vocoder model for audio generation (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_SERVER}));

    // model-specific
    add_opt(common_arg(
        {"--tts-oute-default"},
        string_format("use default OuteTTS models (note: can download weights from the internet)"),
        [](common_params & params) {
            params.hf_repo = "OuteAI/OuteTTS-0.2-500M-GGUF";
            params.hf_file = "OuteTTS-0.2-500M-Q8_0.gguf";
            params.vocoder.hf_repo = "ggml-org/WavTokenizer";
            params.vocoder.hf_file = "WavTokenizer-Large-75-F16.gguf";
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    return ctx_arg;
}
