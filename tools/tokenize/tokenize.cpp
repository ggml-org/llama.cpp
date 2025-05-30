#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

struct tokenizer_config {
    const char * model_path;
    std::string  prompt;
    bool         print_ids;
    bool         no_bos;
    bool         no_escape;
    bool         no_parse_special;
    bool         disable_logging;
    bool         show_token_count;
    
    enum prompt_source_type {
        PROMPT_SOURCE_NONE,
        PROMPT_SOURCE_FILE,
        PROMPT_SOURCE_ARGUMENT,
        PROMPT_SOURCE_STDIN
    } prompt_source;
    
    tokenizer_config() : 
        model_path(nullptr),
        prompt(""),
        print_ids(false),
        no_bos(false),
        no_escape(false),
        no_parse_special(false),
        disable_logging(false),
        show_token_count(false),
        prompt_source(PROMPT_SOURCE_NONE) {}
};

static void print_usage_information(const char * argv0) {
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

static bool read_file_to_string(const char * filepath, std::string & result) {
    LOG_DBG("Reading prompt from file: %s\n", filepath);
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        LOG_ERR("Cannot open file '%s': %s\n", filepath, strerror(errno));
        return false;
    }
    
    file.seekg(0, std::ios::end);
    const size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    result.resize(file_size);
    file.read(&result[0], file_size);
    
    if (file.fail()) {
        LOG_ERR("Error reading file '%s': %s\n", filepath, strerror(errno));
        return false;
    }
    
    LOG_DBG("Successfully read %zu bytes from file\n", file_size);
    return true;
}

static bool read_stdin_to_string(std::string & result) {
    LOG_DBG("Reading prompt from standard input\n");
    
    result.clear();
    char buffer[4096];
    
    while (fgets(buffer, sizeof(buffer), stdin)) {
        result += buffer;
    }
    
    if (ferror(stdin)) {
        LOG_ERR("Error reading from standard input\n");
        return false;
    }
    
    // Remove trailing newline if present
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    LOG_DBG("Successfully read %zu bytes from stdin\n", result.size());
    return true;
}

static std::vector<std::string> process_command_line_args(int raw_argc, char ** raw_argv) {
    LOG_DBG("Processing %d command line arguments\n", raw_argc);
    std::vector<std::string> argv;
    
#if defined(_WIN32)
    int argc;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv) {
        LOG_ERR("Failed to process command line arguments on Windows\n");
        return argv;
    }
    
    for (int i = 0; i < argc; ++i) {
        const int length_needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        if (length_needed <= 0) {
            LocalFree(wargv);
            LOG_ERR("Failed to convert Windows command line argument to UTF-8\n");
            return argv;
        }
        
        std::vector<char> buffer(length_needed);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &buffer[0], length_needed, nullptr, nullptr);
        argv.push_back(std::string(&buffer[0]));
    }
    
    LocalFree(wargv);
#else
    for (int i = 0; i < raw_argc; ++i) {
        argv.push_back(std::string(raw_argv[i]));
    }
#endif
    
    LOG_DBG("Processed %zu arguments\n", argv.size());
    return argv;
}

static void write_utf8_to_stdout(const char * str, bool & invalid_utf8) {
    invalid_utf8 = false;
    
#if defined(_WIN32)
    const HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    
    if (console == INVALID_HANDLE_VALUE || !GetConsoleMode(console, &mode)) {
        printf("%s", str);
        return;
    }
    
    if (*str == '\0') {
        return;
    }
    
    const int str_len = strlen(str);
    const int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, str_len, nullptr, 0);
    if (wide_length == 0) {
        const DWORD error = GetLastError();
        if (error == ERROR_NO_UNICODE_TRANSLATION) {
            invalid_utf8 = true;
            printf("<");
            for (int i = 0; i < str_len; ++i) {
                if (i > 0) {
                    printf(" ");
                }
                printf("%02x", (uint8_t) str[i]);
            }
            printf(">");
            return;
        }
        LOG_ERR("Unexpected error in UTF-8 to wide char conversion\n");
        return;
    }
    
    std::vector<wchar_t> wide_str(wide_length);
    MultiByteToWideChar(CP_UTF8, 0, str, str_len, &wide_str[0], wide_length);
    
    DWORD written;
    WriteConsoleW(console, &wide_str[0], wide_length, &written, nullptr);
#else
    printf("%s", str);
#endif
}

static bool parse_command_line_args(const std::vector<std::string> & argv, tokenizer_config & config) {
    LOG_DBG("Parsing %zu command line arguments\n", argv.size());
    
    if (argv.size() <= 1) {
        LOG_ERR("No arguments provided\n");
        return false;
    }
    
    bool model_path_set = false;
    bool prompt_path_set = false;
    bool prompt_set = false;
    bool stdin_set = false;
    
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string & arg = argv[i];
        LOG_DBG("Processing argument: %s\n", arg.c_str());
        
        if (arg == "-h" || arg == "--help") {
            print_usage_information(argv[0].c_str());
            return false;
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
            if (prompt_path_set || prompt_set || stdin_set) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                return false;
            }
            LOG_DBG("Using stdin as prompt source\n");
            stdin_set = true;
            config.prompt_source = tokenizer_config::PROMPT_SOURCE_STDIN;
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argv.size()) {
            if (model_path_set) {
                LOG_ERR("Model path specified multiple times\n");
                return false;
            }
            config.model_path = argv[++i].c_str();
            model_path_set = true;
            LOG_DBG("Model path set to: %s\n", config.model_path);
        }
        else if ((arg == "-f" || arg == "--file") && i + 1 < argv.size()) {
            if (prompt_path_set || prompt_set || stdin_set) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                return false;
            }
            const std::string filename = argv[++i];
            if (!read_file_to_string(filename.c_str(), config.prompt)) {
                return false;
            }
            prompt_path_set = true;
            config.prompt_source = tokenizer_config::PROMPT_SOURCE_FILE;
            LOG_DBG("Using file as prompt source: %s\n", filename.c_str());
        }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argv.size()) {
            if (prompt_path_set || prompt_set || stdin_set) {
                LOG_ERR("Multiple prompt sources specified (--stdin, --file, --prompt are mutually exclusive)\n");
                return false;
            }
            config.prompt = argv[++i];
            prompt_set = true;
            config.prompt_source = tokenizer_config::PROMPT_SOURCE_ARGUMENT;
            LOG_DBG("Using command line argument as prompt\n");
        }
        else if (arg == "-m" || arg == "--model" || arg == "-f" || arg == "--file" || arg == "-p" || arg == "--prompt") {
            LOG_ERR("Option %s requires an argument\n", arg.c_str());
            return false;
        }
        else {
            LOG_ERR("Unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    
    // Validate required arguments
    if (!model_path_set) {
        LOG_ERR("Model path is required (use -m or --model)\n");
        return false;
    }
    
    if (config.prompt_source == tokenizer_config::PROMPT_SOURCE_NONE) {
        LOG_ERR("Prompt source is required (use --stdin, --file, or --prompt)\n");
        return false;
    }
    
    LOG_DBG("Command line arguments parsed successfully\n");
    return true;
}

static bool load_prompt_from_source(tokenizer_config & config) {
    if (config.prompt_source == tokenizer_config::PROMPT_SOURCE_STDIN) {
        return read_stdin_to_string(config.prompt);
    }
    return true; // File and argument sources already loaded during parsing
}

static void setup_logging_system(bool disable_logging) {
    // Setup common logging with reasonable defaults
    common_log_set_colors(common_log_main(), true);
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), false);
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
    
    if (disable_logging) {
        LOG_DBG("Disabling LLAMA backend logging\n");
        llama_log_set([](ggml_log_level, const char *, void *){}, nullptr);
        // Reduce verbosity to only show errors
        common_log_set_verbosity_thold(-1);
    }
}

static bool tokenize_and_print_results(const tokenizer_config & config, llama_model * model, llama_context * ctx) {
    LOG_INF("Starting tokenization\n");
    std::string prompt = config.prompt;
    
    if (!config.no_escape) {
        LOG_DBG("Processing escape sequences in prompt\n");
        string_process_escapes(prompt);
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !config.no_bos;
    const bool parse_special = !config.no_parse_special;
    
    LOG_DBG("Tokenization settings: add_bos=%s, parse_special=%s\n", 
            add_bos ? "true" : "false", parse_special ? "true" : "false");
    
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, add_bos, parse_special);
    
    LOG_INF("Tokenized %zu characters into %zu tokens\n", prompt.size(), tokens.size());
    
    if (config.print_ids) {
        printf("[");
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) {
                printf(", ");
            }
            printf("%d", tokens[i]);
        }
        printf("]\n");
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            bool invalid_utf8 = false;
            printf("%6d -> '", tokens[i]);
            
            const std::string token_piece = common_token_to_piece(ctx, tokens[i]);
            write_utf8_to_stdout(token_piece.c_str(), invalid_utf8);
            
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
    return true;
}

int main(int raw_argc, char ** raw_argv) {
    // Initialize logging first
    setup_logging_system(false);
    
    LOG_DBG("Starting tokenizer application\n");
    
    // Process command line arguments
    const std::vector<std::string> argv = process_command_line_args(raw_argc, raw_argv);
    if (argv.empty()) {
        LOG_ERR("Failed to process command line arguments\n");
        return 1;
    }
    
    tokenizer_config config;
    if (!parse_command_line_args(argv, config)) {
        if (raw_argc > 0) {
            LOG_ERR("Use '%s --help' for usage information\n", raw_argv[0]);
        }
        return 1;
    }
    
    // Reconfigure logging based on user preferences
    if (config.disable_logging) {
        setup_logging_system(true);
    }
    
    // Initialize backend
    LOG_INF("Initializing LLAMA backend\n");
    llama_backend_init();
    
    // Load model
    LOG_INF("Loading model from: %s\n", config.model_path);
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    
    llama_model * model = llama_model_load_from_file(config.model_path, model_params);
    if (!model) {
        LOG_ERR("Failed to load model from: %s\n", config.model_path);
        return 1;
    }
    
    LOG_DBG("Model loaded successfully, creating context\n");
    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context from model\n");
        llama_model_free(model);
        return 1;
    }
    
    LOG_INF("Model and context initialized successfully\n");
    
    // Load prompt (after model loading for better UX - fail fast if model is bad)
    if (!load_prompt_from_source(config)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    // Tokenize and output
    const bool success = tokenize_and_print_results(config, model, ctx);
    
    // Cleanup
    LOG_DBG("Cleaning up resources\n");
    llama_free(ctx);
    llama_model_free(model);
    
    if (success) {
        LOG_DBG("Application completed successfully\n");
        return 0;
    } else {
        LOG_ERR("Application failed\n");
        return 1;
    }
}