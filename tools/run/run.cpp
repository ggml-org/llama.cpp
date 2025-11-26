#include "chat.h"
#include "common.h"
#include "llama-cpp.h"
#include "log.h"
#include "download.h"

#include "linenoise.cpp/linenoise.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <io.h>
#else
#    include <sys/file.h>
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#include <signal.h>

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
[[noreturn]] static void sigint_handler(int) {
    printf("\n" LOG_COL_DEFAULT);
    exit(0);  // not ideal, but it's the only way to guarantee exit in all cases
}
#endif

GGML_ATTRIBUTE_FORMAT(1, 2)
static int printe(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int ret = vfprintf(stderr, fmt, args);
    va_end(args);

    return ret;
}

static std::string strftime_fmt(const char * fmt, const std::tm & tm) {
    std::ostringstream oss;
    oss << std::put_time(&tm, fmt);

    return oss.str();
}

class Opt {
  public:
    int init(int argc, const char ** argv) {
        ctx_params           = llama_context_default_params();
        model_params         = llama_model_default_params();
        context_size_default = ctx_params.n_batch;
        n_threads_default    = ctx_params.n_threads;
        ngl_default          = model_params.n_gpu_layers;
        common_params_sampling sampling;
        temperature_default = sampling.temp;

        if (argc < 2) {
            printe("Error: No arguments provided.\n");
            print_help();
            return 1;
        }

        // Parse arguments
        if (parse(argc, argv)) {
            printe("Error: Failed to parse arguments.\n");
            print_help();
            return 1;
        }

        // If help is requested, show help and exit
        if (help) {
            print_help();
            return 2;
        }

        ctx_params.n_batch        = context_size >= 0 ? context_size : context_size_default;
        ctx_params.n_ctx          = ctx_params.n_batch;
        ctx_params.n_threads = ctx_params.n_threads_batch = n_threads >= 0 ? n_threads : n_threads_default;
        model_params.n_gpu_layers = ngl >= 0 ? ngl : ngl_default;
        temperature               = temperature >= 0 ? temperature : temperature_default;

        return 0;  // Success
    }

    llama_context_params ctx_params;
    llama_model_params   model_params;
    std::string model_;
    std::string chat_template_file;
    std::string          user;
    bool                 use_jinja   = false;
    int                  context_size = -1, ngl = -1, n_threads = -1;
    float                temperature = -1;
    bool                 verbose     = false;

  private:
    int   context_size_default = -1, ngl_default = -1, n_threads_default = -1;
    float temperature_default = -1;
    bool  help                = false;

    bool parse_flag(const char ** argv, int i, const char * short_opt, const char * long_opt) {
        return strcmp(argv[i], short_opt) == 0 || strcmp(argv[i], long_opt) == 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, int & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = std::atoi(argv[++i]);

        return 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, float & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = std::atof(argv[++i]);

        return 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, std::string & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = argv[++i];

        return 0;
    }

    int parse_options_with_value(int argc, const char ** argv, int & i, bool & options_parsing) {
        if (options_parsing && (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context-size") == 0)) {
            if (handle_option_with_value(argc, argv, i, context_size) == 1) {
                return 1;
            }
        } else if (options_parsing &&
                   (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--ngl") == 0)) {
            if (handle_option_with_value(argc, argv, i, ngl) == 1) {
                return 1;
            }
        } else if (options_parsing && (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0)) {
            if (handle_option_with_value(argc, argv, i, n_threads) == 1) {
                return 1;
            }
        } else if (options_parsing && strcmp(argv[i], "--temp") == 0) {
            if (handle_option_with_value(argc, argv, i, temperature) == 1) {
                return 1;
            }
        } else if (options_parsing && strcmp(argv[i], "--chat-template-file") == 0) {
            if (handle_option_with_value(argc, argv, i, chat_template_file) == 1) {
                return 1;
            }
            use_jinja = true;
        } else {
            return 2;
        }

        return 0;
    }

    int parse_options(const char ** argv, int & i, bool & options_parsing) {
        if (options_parsing && (parse_flag(argv, i, "-v", "--verbose") || parse_flag(argv, i, "-v", "--log-verbose"))) {
            verbose = true;
        } else if (options_parsing && strcmp(argv[i], "--jinja") == 0) {
            use_jinja = true;
        } else if (options_parsing && parse_flag(argv, i, "-h", "--help")) {
            help = true;
            return 0;
        } else if (options_parsing && strcmp(argv[i], "--") == 0) {
            options_parsing = false;
        } else {
            return 2;
        }

        return 0;
    }

    int parse_positional_args(const char ** argv, int & i, int & positional_args_i) {
        if (positional_args_i == 0) {
            if (!argv[i][0] || argv[i][0] == '-') {
                return 1;
            }

            ++positional_args_i;
            model_ = argv[i];
        } else if (positional_args_i == 1) {
            ++positional_args_i;
            user = argv[i];
        } else {
            user += " " + std::string(argv[i]);
        }

        return 0;
    }

    int parse(int argc, const char ** argv) {
        bool options_parsing   = true;
        for (int i = 1, positional_args_i = 0; i < argc; ++i) {
            int ret = parse_options_with_value(argc, argv, i, options_parsing);
            if (ret == 0) {
                continue;
            } else if (ret == 1) {
                return ret;
            }

            ret = parse_options(argv, i, options_parsing);
            if (ret == 0) {
                continue;
            } else if (ret == 1) {
                return ret;
            }

            if (parse_positional_args(argv, i, positional_args_i)) {
                return 1;
            }
        }

        if (model_.empty()) {
            return 1;
        }

        return 0;
    }

    void print_help() const {
        printf(
            "Description:\n"
            "  Runs a llm\n"
            "\n"
            "Usage:\n"
            "  llama-run [options] model [prompt]\n"
            "\n"
            "Options:\n"
            "  -c, --context-size <value>\n"
            "      Context size (default: %d)\n"
            "  --chat-template-file <path>\n"
            "      Path to the file containing the chat template to use with the model.\n"
            "      Only supports jinja templates and implicitly sets the --jinja flag.\n"
            "  --jinja\n"
            "      Use jinja templating for the chat template of the model\n"
            "  -n, -ngl, --ngl <value>\n"
            "      Number of GPU layers (default: %d)\n"
            "  --temp <value>\n"
            "      Temperature (default: %.1f)\n"
            "  -t, --threads <value>\n"
            "      Number of threads to use during generation (default: %d)\n"
            "  -v, --verbose, --log-verbose\n"
            "      Set verbosity level to infinity (i.e. log all messages, useful for debugging)\n"
            "  -h, --help\n"
            "      Show help message\n"
            "\n"
            "Commands:\n"
            "  model\n"
            "      Model is a string with an optional prefix of \n"
            "      huggingface:// (hf://), modelscope:// (ms://), ollama://, https:// or file://.\n"
            "      If no protocol is specified and a file exists in the specified\n"
            "      path, file:// is assumed, otherwise if a file does not exist in\n"
            "      the specified path, ollama:// is assumed. Models that are being\n"
            "      pulled are downloaded with .partial extension while being\n"
            "      downloaded and then renamed as the file without the .partial\n"
            "      extension when complete.\n"
            "\n"
            "Examples:\n"
            "  llama-run llama3\n"
            "  llama-run ollama://granite-code\n"
            "  llama-run ollama://smollm:135m\n"
            "  llama-run hf://QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf\n"
            "  llama-run "
            "huggingface://bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF/SmolLM-1.7B-Instruct-v0.2-IQ3_M.gguf\n"
            "  llama-run ms://QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf\n"
            "  llama-run "
            "modelscope://bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF/SmolLM-1.7B-Instruct-v0.2-IQ3_M.gguf\n"
            "  llama-run https://example.com/some-file1.gguf\n"
            "  llama-run some-file2.gguf\n"
            "  llama-run file://some-file3.gguf\n"
            "  llama-run --ngl 999 some-file4.gguf\n"
            "  llama-run --ngl 999 some-file5.gguf Hello World\n",
            context_size_default, ngl_default, temperature_default, n_threads_default);
    }
};

class File {
  public:
    FILE * file = nullptr;

    FILE * open(const std::string & filename, const char * mode) {
        file = ggml_fopen(filename.c_str(), mode);

        return file;
    }

    int lock() {
        if (file) {
#    ifdef _WIN32
            fd    = _fileno(file);
            hFile = (HANDLE) _get_osfhandle(fd);
            if (hFile == INVALID_HANDLE_VALUE) {
                fd = -1;

                return 1;
            }

            OVERLAPPED overlapped = {};
            if (!LockFileEx(hFile, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD,
                            &overlapped)) {
                fd = -1;

                return 1;
            }
#    else
            fd = fileno(file);
            if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                fd = -1;

                return 1;
            }
#    endif
        }

        return 0;
    }

    std::string to_string() {
        fseek(file, 0, SEEK_END);
        const size_t size = ftell(file);
        fseek(file, 0, SEEK_SET);
        std::string out;
        out.resize(size);
        const size_t read_size = fread(&out[0], 1, size, file);
        if (read_size != size) {
            printe("Error reading file: %s", strerror(errno));
        }

        return out;
    }

    ~File() {
        if (fd >= 0) {
#    ifdef _WIN32
            if (hFile != INVALID_HANDLE_VALUE) {
                OVERLAPPED overlapped = {};
                UnlockFileEx(hFile, 0, MAXDWORD, MAXDWORD, &overlapped);
            }
#    else
            flock(fd, LOCK_UN);
#    endif
        }

        if (file) {
            fclose(file);
        }
    }

  private:
    int fd = -1;
#    ifdef _WIN32
    HANDLE hFile = nullptr;
#    endif
};

class LlamaData {
  public:
    llama_model_ptr                 model;
    llama_sampler_ptr               sampler;
    llama_context_ptr               context;
    std::vector<llama_chat_message> messages; // TODO: switch to common_chat_msg
    std::list<std::string>          msg_strs;
    std::vector<char>               fmtted;

    int init(Opt & opt) {
        model = initialize_model(opt);
        if (!model) {
            return 1;
        }

        context = initialize_context(model, opt);
        if (!context) {
            return 1;
        }

        sampler = initialize_sampler(opt);

        return 0;
    }

  private:

    static bool resolve_endpoint(const std::string   & model_endpoint,
                                 const std::string   & model,
                                 common_params_model & params) {
        // Find the second occurrence of '/' after protocol string
        size_t pos = model.find('/');
        pos = model.find('/', pos + 1);

        common_hf_file_res res;

        try {
            if (pos == std::string::npos) {
                res = common_get_hf_file(model, "", false);
            } else {
                res.repo     = model.substr(0, pos);
                res.ggufFile = model.substr(pos + 1);
            }
        } catch (const std::exception & e) {
            printe("Invalid repository format\n");
            return false;
        }

        params.url = model_endpoint + res.repo + "/resolve/main/" + res.ggufFile;
        return true;
    }

    static bool resolve_github(std::string & model, common_params_model & params) {
        std::string repository = model;
        std::string branch = "main";

        const size_t at_pos = model.find('@');
        if (at_pos != std::string::npos) {
            repository = model.substr(0, at_pos);
            branch     = model.substr(at_pos + 1);
        }

        const std::vector<std::string> repo_parts = string_split(repository, "/");
        if (repo_parts.size() < 3) {
            printe("Invalid GitHub repository format\n");
            return false;
        }

        const std::string & org     = repo_parts[0];
        const std::string & project = repo_parts[1];
        std::string url = "https://raw.githubusercontent.com/" + org + "/" + project + "/" + branch;

        for (size_t i = 2; i < repo_parts.size(); ++i) {
            url += "/" + repo_parts[i];
        }

        params.url = url;
        return true;
    }

    static bool resolve_s3(const std::string & model, common_params_model & params, common_header_list & headers) {
        const size_t slash_pos = model.find('/');
        if (slash_pos == std::string::npos) {
            return false;
        }

        const std::string bucket = model.substr(0, slash_pos);
        const std::string key    = model.substr(slash_pos + 1);

        const char * access_key = std::getenv("AWS_ACCESS_KEY_ID");
        const char * secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");

        if (!access_key || !secret_key) {
            printe("AWS credentials not found in environment\n");
            return false;
        }

        // Generate AWS Signature Version 4 headers
        // (Implementation requires HMAC-SHA256 and date handling)
        // Get current timestamp
        const time_t      now         = time(nullptr);
        const tm          tm          = *gmtime(&now);
        const std::string date        = strftime_fmt("%Y%m%d", tm);
        const std::string datetime    = strftime_fmt("%Y%m%dT%H%M%SZ", tm);
        const std::string auth_header = "AWS4-HMAC-SHA256 Credential=" + std::string(access_key) + "/" + date + "/us-east-1/s3/aws4_request";

        headers.push_back({"Authorization", auth_header});
        headers.push_back({"x-amz-content-sha256", "UNSIGNED-PAYLOAD"});
        headers.push_back({"x-amz-date", datetime});

        params.url = "https://" + bucket + ".s3.amazonaws.com/" + key;
        return true;
    }

    static bool remove_prefix(std::string & url, const std::string & prefix) {
        if (string_starts_with(url, prefix)) {
            url = url.substr(prefix.length());
            return true;
        }
        return false;
    }

    static int resolve_model(std::string & model_) {
        if (std::filesystem::exists(model_)) {
            return 0;
        }

        common_params_model m_params;
        common_header_list headers;
        common_oci_params oci_params;

        bool is_ollama = false;

        if (remove_prefix(model_, "file://")) {
            if (std::filesystem::exists(model_)) {
                return 0;
            }
        } else if (remove_prefix(model_, "hf://") ||
                   remove_prefix(model_, "huggingface://")) {
            if (!resolve_endpoint(get_model_endpoint(), model_, m_params)) {
                return 1;
            }
        } else if (remove_prefix(model_, "ms://") ||
                   remove_prefix(model_, "modelscope://")) {
            if (!resolve_endpoint("https://modelscope.cn/models/", model_, m_params)) {
                return 1;
            }
        } else if (remove_prefix(model_, "s3://")) {
            if (!resolve_s3(model_, m_params, headers)) {
                return 1;
            }
        } else if (remove_prefix(model_, "github://")) {
            if (!resolve_github(model_, m_params)) {
                return 1;
            }
        } else if (remove_prefix(model_, "ollama://") ||
                   remove_prefix(model_, "https://ollama.com/library/")) {
            is_ollama = true;
        } else if (string_starts_with(model_, "http://") ||
                   string_starts_with(model_, "https://")) {
            m_params.url = model_;
        } else {
            if (model_.find(".gguf") != std::string::npos) {
                printe("Error: Local file not found: %s\n", model_.c_str());
                return 1;
            }
            // fallback ollama
            is_ollama = true;
        }
        try {
            if (is_ollama) {
                oci_params.registry_url = "https://registry.ollama.ai";
                oci_params.auth_url     = ""; // no auth for ollama
                oci_params.auth_service = "";
                oci_params.media_type   = "application/vnd.ollama.image.model";

                if (model_.find('/') == std::string::npos) {
                    model_ = "library/" + model_;
                }
                model_ = common_docker_resolve_model(model_, oci_params);
            } else {
                std::string name = std::filesystem::path(m_params.url).filename().string();

                if (name.find('?') != std::string::npos) {
                    name = name.substr(0, name.find('?'));
                }
                m_params.path = fs_get_cache_file(name);

                // token and offline are not supported
                if (!common_download_model(m_params, "", false, headers)) {
                    printe("Failed to download model from %s\n", m_params.url.c_str());
                    return 1;
                }
                model_ = m_params.path;
            }
        } catch (const std::exception & e) {
            printe("Model resolution error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    // Initializes the model and returns a unique pointer to it
    static llama_model_ptr initialize_model(Opt & opt) {
        ggml_backend_load_all();
        if (resolve_model(opt.model_)) {
            return nullptr;
        }
        printe("\r" LOG_CLR_TO_EOL "Loading model");
        llama_model_ptr model(llama_model_load_from_file(opt.model_.c_str(), opt.model_params));
        if (!model) {
            printe("%s: error: unable to load model from file: %s\n", __func__, opt.model_.c_str());
        }

        printe("\r" LOG_CLR_TO_EOL);
        return model;
    }

    // Initializes the context with the specified parameters
    static llama_context_ptr initialize_context(const llama_model_ptr & model, const Opt & opt) {
        llama_context_ptr context(llama_init_from_model(model.get(), opt.ctx_params));
        if (!context) {
            printe("%s: error: failed to create the llama_context\n", __func__);
        }

        return context;
    }

    // Initializes and configures the sampler
    static llama_sampler_ptr initialize_sampler(const Opt & opt) {
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(opt.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        return sampler;
    }
};

// Add a message to `messages` and store its content in `msg_strs`
static void add_message(const char * role, const std::string & text, LlamaData & llama_data) {
    llama_data.msg_strs.push_back(text);
    llama_data.messages.push_back({ role, llama_data.msg_strs.back().c_str() });
}

// Function to apply the chat template and resize `formatted` if needed
static int apply_chat_template(const struct common_chat_templates * tmpls, LlamaData & llama_data, const bool append, bool use_jinja) {
    common_chat_templates_inputs inputs;
    for (const auto & msg : llama_data.messages) {
        common_chat_msg cmsg;
        cmsg.role    = msg.role;
        cmsg.content = msg.content;
        inputs.messages.push_back(cmsg);
    }
    inputs.add_generation_prompt = append;
    inputs.use_jinja = use_jinja;

    auto chat_params = common_chat_templates_apply(tmpls, inputs);
    // TODO: use other params for tool calls.
    auto result = chat_params.prompt;
    llama_data.fmtted.resize(result.size() + 1);
    memcpy(llama_data.fmtted.data(), result.c_str(), result.size() + 1);
    return result.size();
}

// Function to tokenize the prompt
static int tokenize_prompt(const llama_vocab * vocab, const std::string & prompt,
                           std::vector<llama_token> & prompt_tokens, const LlamaData & llama_data) {
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(llama_data.context.get()), 0) == -1;
    int n_tokens = prompt.size() + 2 * is_first;
    prompt_tokens.resize(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                              prompt_tokens.data(), prompt_tokens.size(),
                              is_first, /*parse_special =*/true);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        printe("tokenization failed: input too large\n");
        return -1;
    }
    if (n_tokens < 0) {
        prompt_tokens.resize(-n_tokens);
        int check = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   prompt_tokens.data(), prompt_tokens.size(),
                                   is_first, /*parse_special =*/true);
        if (check != -n_tokens) {
            printe("failed to tokenize the prompt (size mismatch)\n");
            return -1;
        }
        n_tokens = check;
    } else {
        prompt_tokens.resize(n_tokens);
    }
    return n_tokens;
}

// Check if we have enough space in the context to evaluate this batch
static int check_context_size(const llama_context_ptr & ctx, const llama_batch & batch) {
    const int n_ctx      = llama_n_ctx(ctx.get());
    const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        printf(LOG_COL_DEFAULT "\n");
        printe("context size exceeded\n");
        return 1;
    }

    return 0;
}

// convert the token to a string
static int convert_token_to_string(const llama_vocab * vocab, const llama_token token_id, std::string & piece) {
    char buf[256];
    int  n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        printe("failed to convert token to piece\n");
        return 1;
    }

    piece = std::string(buf, n);
    return 0;
}

static void print_word_and_concatenate_to_response(const std::string & piece, std::string & response) {
    printf("%s", piece.c_str());
    fflush(stdout);
    response += piece;
}

// helper function to evaluate a prompt and generate a response
static int generate(LlamaData & llama_data, const std::string & prompt, std::string & response) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_data.model.get());

    std::vector<llama_token> tokens;
    if (tokenize_prompt(vocab, prompt, tokens, llama_data) < 0) {
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token new_token_id;
    while (true) {
        check_context_size(llama_data.context, batch);
        if (llama_decode(llama_data.context.get(), batch)) {
            printe("failed to decode\n");
            return 1;
        }

        // sample the next token, check is it an end of generation?
        new_token_id = llama_sampler_sample(llama_data.sampler.get(), llama_data.context.get(), -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        std::string piece;
        if (convert_token_to_string(vocab, new_token_id, piece)) {
            return 1;
        }

        print_word_and_concatenate_to_response(piece, response);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    printf(LOG_COL_DEFAULT);
    return 0;
}

static int read_user_input(std::string & user_input) {
    static const char * prompt_prefix_env = std::getenv("LLAMA_PROMPT_PREFIX");
    static const char * prompt_prefix     = prompt_prefix_env ? prompt_prefix_env : "> ";
#ifdef WIN32
    printf("\r" LOG_CLR_TO_EOL LOG_COL_DEFAULT "%s", prompt_prefix);

    std::getline(std::cin, user_input);
    if (std::cin.eof()) {
        printf("\n");
        return 1;
    }
#else
    std::unique_ptr<char, decltype(&std::free)> line(const_cast<char *>(linenoise(prompt_prefix)), free);
    if (!line) {
        return 1;
    }

    user_input = line.get();
#endif

    if (user_input == "/bye") {
        return 1;
    }

    if (user_input.empty()) {
        return 2;
    }

#ifndef WIN32
    linenoiseHistoryAdd(line.get());
#endif

    return 0;  // Should have data in happy path
}

// Function to generate a response based on the prompt
static int generate_response(LlamaData & llama_data, const std::string & prompt, std::string & response,
                             const bool stdout_a_terminal) {
    // Set response color
    if (stdout_a_terminal) {
        printf(LOG_COL_YELLOW);
    }

    if (generate(llama_data, prompt, response)) {
        printe("failed to generate response\n");
        return 1;
    }

    // End response with color reset and newline
    printf("\n%s", stdout_a_terminal ? LOG_COL_DEFAULT : "");
    return 0;
}

// Helper function to apply the chat template and handle errors
static int apply_chat_template_with_error_handling(const common_chat_templates * tmpls, LlamaData & llama_data, const bool append, int & output_length, bool use_jinja) {
    const int new_len = apply_chat_template(tmpls, llama_data, append, use_jinja);
    if (new_len < 0) {
        printe("failed to apply the chat template\n");
        return -1;
    }

    output_length = new_len;
    return 0;
}

// Helper function to handle user input
static int handle_user_input(std::string & user_input, const std::string & user) {
    if (!user.empty()) {
        user_input = user;
        return 0;  // No need for interactive input
    }

    return read_user_input(user_input);  // Returns true if input ends the loop
}

static bool is_stdin_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD  mode;
    return GetConsoleMode(hStdin, &mode);
#else
    return isatty(STDIN_FILENO);
#endif
}

static bool is_stdout_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode;
    return GetConsoleMode(hStdout, &mode);
#else
    return isatty(STDOUT_FILENO);
#endif
}

// Function to handle user input
static int get_user_input(std::string & user_input, const std::string & user) {
    while (true) {
        const int ret = handle_user_input(user_input, user);
        if (ret == 1) {
            return 1;
        }

        if (ret == 2) {
            continue;
        }

        break;
    }

    return 0;
}

// Reads a chat template file to be used
static std::string read_chat_template_file(const std::string & chat_template_file) {
    File file;
    if (!file.open(chat_template_file, "r")) {
        printe("Error opening chat template file '%s': %s", chat_template_file.c_str(), strerror(errno));
        return "";
    }

    return file.to_string();
}

static int process_user_message(const Opt & opt, const std::string & user_input, LlamaData & llama_data,
                                const common_chat_templates_ptr & chat_templates, int & prev_len,
                                const bool stdout_a_terminal) {
    add_message("user", opt.user.empty() ? user_input : opt.user, llama_data);
    int new_len;
    if (apply_chat_template_with_error_handling(chat_templates.get(), llama_data, true, new_len, opt.use_jinja) < 0) {
        return 1;
    }

    std::string prompt(llama_data.fmtted.begin() + prev_len, llama_data.fmtted.begin() + new_len);
    std::string response;
    if (generate_response(llama_data, prompt, response, stdout_a_terminal)) {
        return 1;
    }

    if (!opt.user.empty()) {
        return 2;
    }

    add_message("assistant", response, llama_data);
    if (apply_chat_template_with_error_handling(chat_templates.get(), llama_data, false, prev_len, opt.use_jinja) < 0) {
        return 1;
    }

    return 0;
}

// Main chat loop function
static int chat_loop(LlamaData & llama_data, const Opt & opt) {
    int prev_len = 0;
    llama_data.fmtted.resize(llama_n_ctx(llama_data.context.get()));
    std::string chat_template;
    if (!opt.chat_template_file.empty()) {
        chat_template = read_chat_template_file(opt.chat_template_file);
    }

    common_chat_templates_ptr chat_templates    = common_chat_templates_init(llama_data.model.get(), chat_template);
    static const bool stdout_a_terminal = is_stdout_a_terminal();
    while (true) {
        // Get user input
        std::string user_input;
        if (get_user_input(user_input, opt.user) == 1) {
            return 0;
        }

        const int ret = process_user_message(opt, user_input, llama_data, chat_templates, prev_len, stdout_a_terminal);
        if (ret == 1) {
            return 1;
        } else if (ret == 2) {
            break;
        }
    }

    return 0;
}

static void log_callback(const enum ggml_log_level level, const char * text, void * p) {
    const Opt * opt = static_cast<Opt *>(p);
    if (opt->verbose || level == GGML_LOG_LEVEL_ERROR) {
        printe("%s", text);
    }
}

static std::string read_pipe_data() {
    std::ostringstream result;
    result << std::cin.rdbuf();  // Read all data from std::cin
    return result.str();
}

static void ctrl_c_handling() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

int main(int argc, const char ** argv) {
    ctrl_c_handling();
    Opt       opt;
    const int ret = opt.init(argc, argv);
    if (ret == 2) {
        return 0;
    } else if (ret) {
        return 1;
    }

    if (!is_stdin_a_terminal()) {
        if (!opt.user.empty()) {
            opt.user += "\n\n";
        }

        opt.user += read_pipe_data();
    }

    llama_log_set(log_callback, &opt);
    LlamaData llama_data;
    if (llama_data.init(opt)) {
        return 1;
    }

    if (chat_loop(llama_data, opt)) {
        return 1;
    }

    return 0;
}
