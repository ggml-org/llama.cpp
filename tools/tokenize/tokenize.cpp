 #include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

// RAII wrapper for LLAMA resources
class LlamaModelWrapper {
private:
    llama_model* model_;
    llama_context* ctx_;

public:
    LlamaModelWrapper() : model_(nullptr), ctx_(nullptr) {}

    ~LlamaModelWrapper() {
        if (ctx_) {
            LOG_DBG("Freeing LLAMA context\n");
            llama_free(ctx_);
        }
        if (model_) {
            LOG_DBG("Freeing LLAMA model\n");
            llama_model_free(model_);
        }
    }

    // Non-copyable
    LlamaModelWrapper(const LlamaModelWrapper&) = delete;
    LlamaModelWrapper& operator=(const LlamaModelWrapper&) = delete;

    bool load_model(const char* model_path) {
        LOG_INF("Loading model from: %s\n", model_path);

        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;

        model_ = llama_model_load_from_file(model_path, model_params);
        if (!model_) {
            LOG_ERR("Failed to load model from: %s\n", model_path);
            return false;
        }

        LOG_DBG("Model loaded successfully, creating context\n");
        llama_context_params ctx_params = llama_context_default_params();
        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            LOG_ERR("Failed to create context from model\n");
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }

        LOG_INF("Model and context initialized successfully\n");
        return true;
    }

    llama_model* model() const { return model_; }
    llama_context* context() const { return ctx_; }
    const llama_vocab* vocab() const {
        return model_ ? llama_model_get_vocab(model_) : nullptr;
    }
};

struct TokenizerConfig {
    std::string model_path;
    std::string prompt;
    bool print_ids = false;
    bool no_bos = false;
    bool no_escape = false;
    bool no_parse_special = false;
    bool disable_logging = false;
    bool show_token_count = false;

    enum class PromptSource { NONE, FILE, ARGUMENT, STDIN } prompt_source = PromptSource::NONE;
};

static void print_usage_information(const char* argv0) {
    LOG("Usage: %s [options]\n\n", argv0);
    LOG("The tokenize program tokenizes a prompt using a given model,\n");
    LOG("and prints the resulting tokens to standard output.\n\n");
    LOG("Required:\n");
    LOG("  -m, --model MODEL_PATH           Path to the model file\n");
    LOG("  One of: --file, --prompt, --stdin\n\n");
    LOG("Prompt sources (exactly one required):\n");
    LOG("  -f, --file FILENAME              Read prompt from file\n");
    LOG("  -p, --prompt TEXT                Use prompt from command line\n");
    LOG("  --stdin                          Read prompt from standard input\n\n");
    LOG("Output options:\n");
    LOG("  --ids                            Print only token IDs as [1, 2, 3]\n");
    LOG("  --show-count                     Show total token count\n\n");
    LOG("Tokenization options:\n");
    LOG("  --no-bos                         Don't add BOS token\n");
    LOG("  --no-escape                      Don't process escape sequences (\\n, \\t)\n");
    LOG("  --no-parse-special               Don't parse special/control tokens\n\n");
    LOG("Other options:\n");
    LOG("  --log-disable                    Disable model loading logs\n");
    LOG("  -h, --help                       Show this help and exit\n");
    LOG("\nExamples:\n");
    LOG("  %s -m model.gguf -p \"Hello world\"\n", argv0);
    LOG("  %s -m model.gguf --file input.txt --ids\n", argv0);
    LOG("  echo \"Hello\" | %s -m model.gguf --stdin\n", argv0);
}

static std::string read_file_safely(const std::string& filepath) {
    LOG_DBG("Reading prompt from file: %s\n", filepath.c_str());

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        LOG_ERR("Cannot open file '%s': %s\n", filepath.c_str(), std::strerror(errno));
        throw std::runtime_error("Cannot open file '" + filepath + "': " + std::strerror(errno));
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    if (file.fail() && !file.eof()) {
        LOG_ERR("Error reading file '%s': %s\n", filepath.c_str(), std::strerror(errno));
        throw std::runtime_error("Error reading file '" + filepath + "': " + std::strerror(errno));
    }

    std::string content = buffer.str();
    LOG_DBG("Successfully read %zu bytes from file\n", content.size());
    return content;
}

static std::string read_stdin_safely() {
    LOG_DBG("Reading prompt from standard input\n");

    std::ostringstream buffer;
    buffer << std::cin.rdbuf();

    if (std::cin.fail()) {
        LOG_ERR("Error reading from standard input\n");
        throw std::runtime_error("Error reading from standard input");
    }

    std::string content = buffer.str();
    LOG_DBG("Successfully read %zu bytes from stdin\n", content.size());
    return content;
}

static std::vector<std::string> process_arguments(int raw_argc, char** raw_argv) {
    LOG_DBG("Processing %d command line arguments\n", raw_argc);
    std::vector<std::string> argv;

#if defined(_WIN32)
    int argc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) {
        LOG_ERR("Failed to process command line arguments on Windows\n");
        throw std::runtime_error("Failed to process command line arguments on Windows");
    }

    for (int i = 0; i < argc; ++i) {
        int length_needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (length_needed <= 0) {
            LocalFree(wargv);
            LOG_ERR("Failed to convert Windows command line argument to UTF-8\n");
            throw std::runtime_error("Failed to convert Windows command line argument to UTF-8");
        }

        std::vector<char> buffer(length_needed);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, buffer.data(), length_needed, nullptr, nullptr);
        argv.emplace_back(buffer.data());
    }

    LocalFree(wargv);
#else
    argv.reserve(raw_argc);
    for (int i = 0; i < raw_argc; ++i) {
        argv.emplace_back(raw_argv[i]);
    }
#endif

    LOG_DBG("Processed %zu arguments\n", argv.size());
    return argv;
}

static void write_utf8_to_stdout(const std::string& str, bool& invalid_utf8) {
    invalid_utf8 = false;

#if defined(_WIN32)
    HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;

    if (console == INVALID_HANDLE_VALUE || !GetConsoleMode(console, &mode)) {
        printf("%s", str.c_str());
        return;
    }

    if (str.empty()) return;

    int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         str.c_str(), static_cast<int>(str.length()),
                                         nullptr, 0);
    if (wide_length == 0) {
        DWORD error = GetLastError();
        if (error == ERROR_NO_UNICODE_TRANSLATION) {
            invalid_utf8 = true;
            printf("<");
            for (size_t i = 0; i < str.length(); ++i) {
                if (i > 0) printf(" ");
                printf("%02x", static_cast<uint8_t>(str[i]));
            }
            printf(">");
            return;
        }
        LOG_ERR("Unexpected error in UTF-8 to wide char conversion\n");
        throw std::runtime_error("Unexpected error in UTF-8 to wide char conversion");
    }

    std::vector<wchar_t> wide_str(wide_length);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()),
                       wide_str.data(), wide_length);

    DWORD written;
    WriteConsoleW(console, wide_str.data(), wide_length, &written, nullptr);
#else
    printf("%s", str.c_str());
#endif
}

static TokenizerConfig parse_arguments(const std::vector<std::string>& argv) {
    LOG_DBG("Parsing %zu command line arguments\n", argv.size());
    TokenizerConfig config;

    if (argv.size() <= 1) {
        LOG_ERR("No arguments provided\n");
        throw std::invalid_argument("No arguments provided");
    }

    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string& arg = argv[i];
        LOG_DBG("Processing argument: %s\n", arg.c_str());

        if (arg == "-h" || arg == "--help") {
            print_usage_information(argv[0].c_str());
            std::exit(0);
        }
        else if (arg == "--ids") {
            LOG_DBG("Enabling ID-only output\n");
            config.print_ids = true;
        }
        else if (arg == "--no-bos") {
            LOG_DBG("Disabling BOS token\n");
            config.no_bos = true;
        }
        else if (arg == "--no-escape") {
            LOG_DBG("Disabling escape sequence processing\n");
            config.no_escape = true;
        }
        else if (arg == "--no-parse-special") {
            LOG_DBG("Disabling special token parsing\n");
            config.no_parse_special = true;
        }
        else if (arg == "--log-disable") {
            LOG_DBG("Disabling logging\n");
            config.disable_logging = true;
        }
        else if (arg == "--show-count") {
            LOG_DBG("Enabling token count display\n");
            config.show_token_count = true;
        }
        else if (arg == "--stdin") {
            if (config.prompt_source != TokenizerConfig::PromptSource::NONE) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                throw std::invalid_argument("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)");
            }
            LOG_DBG("Using stdin as prompt source\n");
            config.prompt_source = TokenizerConfig::PromptSource::STDIN;
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argv.size()) {
            if (!config.model_path.empty()) {
                LOG_ERR("Model path specified multiple times\n");
                throw std::invalid_argument("Model path specified multiple times");
            }
            config.model_path = argv[++i];
            LOG_DBG("Model path set to: %s\n", config.model_path.c_str());
        }
        else if ((arg == "-f" || arg == "--file") && i + 1 < argv.size()) {
            if (config.prompt_source != TokenizerConfig::PromptSource::NONE) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                throw std::invalid_argument("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)");
            }
            config.prompt_source = TokenizerConfig::PromptSource::FILE;
            config.prompt = argv[++i];  // Store filename temporarily
            LOG_DBG("Using file as prompt source: %s\n", config.prompt.c_str());
        }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argv.size()) {
            if (config.prompt_source != TokenizerConfig::PromptSource::NONE) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                throw std::invalid_argument("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)");
            }
            config.prompt_source = TokenizerConfig::PromptSource::ARGUMENT;
            config.prompt = argv[++i];
            LOG_DBG("Using command line argument as prompt\n");
        }
        else if (arg == "-m" || arg == "--model" || arg == "-f" || arg == "--file" || arg == "-p" || arg == "--prompt") {
            LOG_ERR("Option %s requires an argument\n", arg.c_str());
            throw std::invalid_argument("Option " + arg + " requires an argument");
        }
        else {
            LOG_ERR("Unknown option: %s\n", arg.c_str());
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    // Validate required arguments
    if (config.model_path.empty()) {
        LOG_ERR("Model path is required (use -m or --model)\n");
        throw std::invalid_argument("Model path is required (use -m or --model)");
    }

    if (config.prompt_source == TokenizerConfig::PromptSource::NONE) {
        LOG_ERR("Prompt source is required (use --stdin, --file, or --prompt)\n");
        throw std::invalid_argument("Prompt source is required (use --stdin, --file, or --prompt)");
    }

    LOG_DBG("Command line arguments parsed successfully\n");
    return config;
}

static void load_prompt(TokenizerConfig& config) {
    switch (config.prompt_source) {
        case TokenizerConfig::PromptSource::FILE: {
            std::string filename = config.prompt;  // Was stored temporarily
            config.prompt = read_file_safely(filename);
            break;
        }
        case TokenizerConfig::PromptSource::STDIN:
            config.prompt = read_stdin_safely();
            break;
        case TokenizerConfig::PromptSource::ARGUMENT:
            LOG_DBG("Using prompt from command line argument (%zu chars)\n", config.prompt.size());
            break;
        case TokenizerConfig::PromptSource::NONE:
            LOG_ERR("Invalid prompt source\n");
            throw std::logic_error("Invalid prompt source");
    }
}

static void tokenize_and_print(const TokenizerConfig& config, LlamaModelWrapper& model_wrapper) {
    LOG_INF("Starting tokenization\n");
    std::string prompt = config.prompt;

    if (!config.no_escape) {
        LOG_DBG("Processing escape sequences in prompt\n");
        string_process_escapes(prompt);
    }

    const llama_vocab* vocab = model_wrapper.vocab();
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !config.no_bos;
    const bool parse_special = !config.no_parse_special;

    LOG_DBG("Tokenization settings: add_bos=%s, parse_special=%s\n",
            add_bos ? "true" : "false", parse_special ? "true" : "false");

    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, add_bos, parse_special);

    LOG_INF("Tokenized %zu characters into %zu tokens\n", prompt.size(), tokens.size());

    if (config.print_ids) {
        printf("[");
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) printf(", ");
            printf("%d", tokens[i]);
        }
        printf("]\n");
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            bool invalid_utf8 = false;
            printf("%6d -> '", tokens[i]);

            std::string token_piece = common_token_to_piece(model_wrapper.context(), tokens[i]);
            write_utf8_to_stdout(token_piece, invalid_utf8);

            if (invalid_utf8) {
                printf("' (UTF-8 decode failure)\n");
            } else {
                printf("'\n");
            }
        }
    }

    if (config.show_token_count) {
        printf("Total number of tokens: %zu\n", tokens.size());
    }

    LOG_INF("Tokenization completed successfully\n");
}

static void setup_logging(bool disable_logging) {
    // Setup common logging with reasonable defaults
    common_log_set_colors(common_log_main(), true);
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), false); // Keep timestamps off for cleaner output
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);

    if (disable_logging) {
        LOG_DBG("Disabling LLAMA backend logging\n");
        llama_log_set([](ggml_log_level, const char*, void*){}, nullptr);
        // Also reduce verbosity to only show errors
        common_log_set_verbosity_thold(-1);
    }
}

int main(int raw_argc, char** raw_argv) {
    try {
        // Initialize logging first
        setup_logging(false); // Will be reconfigured later if needed

        LOG_DBG("Starting tokenizer application\n");

        // Process command line arguments
        std::vector<std::string> argv = process_arguments(raw_argc, raw_argv);
        TokenizerConfig config = parse_arguments(argv);

        // Reconfigure logging based on user preferences
        if (config.disable_logging) {
            setup_logging(true);
        }

        // Initialize backend
        LOG_INF("Initializing LLAMA backend\n");
        llama_backend_init();

        // Load model
        LlamaModelWrapper model_wrapper;
        if (!model_wrapper.load_model(config.model_path.c_str())) {
            LOG_ERR("Model loading failed\n");
            throw std::runtime_error("Failed to load model from: " + config.model_path);
        }

        // Load prompt (after model loading for better UX - fail fast if model is bad)
        load_prompt(config);

        // Tokenize and output
        tokenize_and_print(config, model_wrapper);

        LOG_DBG("Application completed successfully\n");
        return 0;

    } catch (const std::exception& e) {
        LOG_ERR("Application error: %s\n", e.what());
        if (raw_argc > 0) {
            LOG_ERR("Use '%s --help' for usage information\n", raw_argv[0]);
        }
        return 1;
    } catch (...) {
        LOG_ERR("An unexpected error occurred\n");
        return 1;
    }
}
