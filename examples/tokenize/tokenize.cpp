#include "common.h"
//#include "log.h" // TODO: start using log.h
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream> // TODO: remove me

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>   // For CommandLineToArgvW
#endif

static void print_usage_information(const char * argv0) {
    printf("usage: %s [options]\n\n", argv0);
    printf("The tokenize program tokenizes a prompt using a given model,\n");
    printf("and prints the resulting tokens to standard output.\n\n");
    printf("It needs a model file, a prompt, and optionally other flags\n");
    printf("to control the behavior of the tokenizer.\n\n");
    printf("    The possible options are:\n");
    printf("\n");
    printf("    -h, --help                           print this help and exit\n");
    printf("    -m MODEL_PATH, --model MODEL_PATH    path to model.\n");
    printf("    --ids                                if given, only print numerical token IDs, and not token strings.\n");
    printf("                                         The output format looks like [1, 2, 3], i.e. parseable by Python.\n");
    printf("    -f PROMPT_FNAME, --file PROMPT_FNAME read prompt from a file.\n");
    printf("    -p PROMPT, --prompt PROMPT           read prompt from the argument.\n");
    printf("    --stdin                              read prompt from standard input.\n");
    printf("    --no-bos                             do not ever add a BOS token to the prompt, even if normally the model uses a BOS token.\n");
    printf("    --no-escape                          do not escape input (such as \\n, \\t, etc.).\n");
    printf("    --no-parse-special                   do not parse control tokens.\n");
    printf("    --log-disable                        disable logs. Makes stderr quiet when loading the model.\n");
    printf("    --show-count                         print the total number of tokens.\n");
}

static void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static std::string read_prompt_from_file(const char * filepath, bool & success) {
    success = false;

    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: could not open file '%s' for reading: %s\n", __func__, filepath, strerror(errno));
        return std::string();
    }
    // do not assume the file is seekable (e.g. /dev/stdin)
    std::stringstream buffer;
    buffer << in.rdbuf();
    if (in.fail()) {
        fprintf(stderr, "%s: could not read the entire file '%s': %s\n", __func__, filepath, strerror(errno));
        return std::string();
    }

    success = true;
    return buffer.str();
}

//
// Function: ingest_args(...) -> vector<string>
//
//  Takes argc and argv arguments, and converts them to a vector of UTF-8 encoded
//  strings, as an STL vector<string>.
//
//  In particular, it handles character encoding shenanigans on Windows.
//
// Note: raw_argc and raw_argv are not actually read at all on Windows.
//       On Windows we call GetCommandLineW to get the arguments in wchar_t
//       format, ignoring the regular argc/argv arguments to main().
//
// TODO: potential opportunity to roll common stuff into common/console.cpp
//       in relation to Windows wchar_t shenanigans.
static std::vector<std::string> ingest_args(int raw_argc, char ** raw_argv) {
    std::vector<std::string> argv;

    // Handle Windows, if given non-ASCII arguments.
    // We convert wchar_t arguments into UTF-8 char* on this platform.
    // Lets you invoke 'tokenize' on Windows cmd.exe with non-ASCII characters
    // without throwing tantrums.
#if defined(_WIN32)
    int argc;
    const LPWSTR cmdline_wargv = GetCommandLineW();
    LPWSTR * wargv = CommandLineToArgvW(cmdline_wargv, &argc);

    // silence unused arg warnings
    (void) raw_argc;
    (void) raw_argv;

    for (int i = 0; i < argc; ++i) {
        int length_needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], wcslen(wargv[i]), 0, 0, NULL, NULL);
        char * output_buf = (char *) calloc(length_needed+1, sizeof(char));
        GGML_ASSERT(output_buf);

        WideCharToMultiByte(CP_UTF8, 0, wargv[i], wcslen(wargv[i]), output_buf, length_needed, NULL, NULL);
        output_buf[length_needed] = '\0';

        argv.push_back(output_buf);
        free(output_buf);
    }

    LocalFree((HLOCAL) wargv);
#else
    int argc = raw_argc;
    for (int i = 0; i < argc; ++i) {
        argv.push_back(raw_argv[i]);
    }
#endif

    GGML_ASSERT((unsigned int) argc == argv.size());

    return argv;
}

//
// Function: write_utf8_cstr_to_stdout(const char *) -> <writes to stdout>
//
// writes a string to standard output; taking into account that on Windows
// to display correctly you have to use special handling. Works even if the
// user has not set a unicode code page on a Windows cmd.exe.
//
// In case of invalid UTF-8, invalid_utf8 is set to true on Windows, and something
// a human-readable is written instead.
//
// On non-Windows systems, simply printfs() the string.
static void write_utf8_cstr_to_stdout(const char * str, bool & invalid_utf8) {
        invalid_utf8 = false;

#if defined(_WIN32)
        // Are we in a console?
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;

        // According to Microsoft docs:
        // "WriteConsole fails if it is used with a standard handle that is redirected to a file."
        // Also according to the docs, you can use GetConsoleMode to check for that.
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            printf("%s", str);
            return;
        }

        // MultiByteToWideChar reports an error if str is empty, don't report
        // them as invalid_utf8.
        if (*str == 0) {
            return;
        }
        int length_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, strlen(str), NULL, 0);
        if (length_needed == 0) {
            DWORD err = GetLastError();
            if (err == ERROR_NO_UNICODE_TRANSLATION) {
                invalid_utf8 = true;
                int len = strlen(str);
                printf("<");
                for (int i = 0; i < len; ++i) {
                    if (i > 0) {
                        printf(" ");
                    }
                    printf("%02x", (uint8_t) str[i]);
                }
                printf(">");
                return;
            }
            GGML_ABORT("MultiByteToWideChar() failed in an unexpected way.");
        }

        LPWSTR wstr = (LPWSTR) calloc(length_needed+1, sizeof(*wstr));
        GGML_ASSERT(wstr);

        MultiByteToWideChar(CP_UTF8, 0, str, strlen(str), wstr, length_needed);
        WriteConsoleW(hConsole, wstr, length_needed, NULL, NULL);

        free(wstr);
#else
        // TODO: reporting invalid_utf8 would be useful on non-Windows too.
        // printf will silently just write bad unicode.
        printf("%s", str);
#endif
}

int main(int raw_argc, char ** raw_argv) {
    const std::vector<std::string> argv = ingest_args(raw_argc, raw_argv);
    const int argc = argv.size();

    if (argc <= 1) {
        print_usage_information(argv[0].c_str());
        return 1;
    }

    //////
    // Read out all the command line arguments.
    //////

    // variables where to put any arguments we see.
    bool printing_ids = false;
    bool no_bos = false;
    bool no_escape = false;
    bool no_parse_special = false;
    bool disable_logging = false;
    bool show_token_count = false;
    const char * model_path = NULL;
    const char * prompt_path = NULL;
    const char * prompt_arg = NULL;

    // track which arguments were explicitly given
    // used for sanity checking down the line
    bool model_path_set = false;
    bool prompt_path_set = false;
    bool prompt_set = false;
    bool stdin_set = false;

    int iarg = 1;
    for (; iarg < argc; ++iarg) {
        std::string arg{argv[iarg]};
        if (arg == "-h" || arg == "--help") {
            print_usage_information(argv[0].c_str());
            return 0;
        }
        else if (arg == "--ids") {
            printing_ids = true;
        }
        else if (arg == "-m" || arg == "--model") {
            if (model_path_set) {
                fprintf(stderr, "Error: -m or --model specified multiple times.\n");
                return 1;
            }
            model_path = argv[++iarg].c_str();
            model_path_set = true;
        }
        else if (arg == "--no-bos") {
            no_bos = true;
        }
        else if (arg == "--no-escape") {
            no_escape = true;
        }
        else if (arg == "--no-parse-special") {
            no_parse_special = true;
        }
        else if (arg == "-p" || arg == "--prompt") {
            if (prompt_set) {
                fprintf(stderr, "Error: -p or --prompt specified multiple times.\n");
                return 1;
            }
            prompt_arg = argv[++iarg].c_str();
            prompt_set = true;
        }
        else if (arg == "-f" || arg == "--file") {
            if (prompt_path_set) {
                fprintf(stderr, "Error: -f or --file specified multiple times.\n");
                return 1;
            }
            prompt_path = argv[++iarg].c_str();
            prompt_path_set = true;
        }
        else if (arg == "--stdin") {
            stdin_set = true;
        }
        else if (arg == "--log-disable") {
            disable_logging = true;
        }
        else if (arg == "--show-count") {
            show_token_count = true;
        }
        else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[iarg].c_str());
            return 1;
        }
    }

    //////
    // Sanity check the command line arguments.
    //////

    // Check that we have the required stuff set.
    if (model_path_set && model_path == NULL) {
        fprintf(stderr, "Error: --model requires an argument.\n");
        return 1;
    }
    if (!model_path_set) {
        fprintf(stderr, "Error: must specify --model.\n");
        return 1;
    }
    if (prompt_path_set && prompt_path == NULL) {
        fprintf(stderr, "Error: --file requires an argument.\n");
        return 1;
    }
    if (prompt_set && prompt_arg == NULL) {
        fprintf(stderr, "Error: --prompt requires an argument.\n");
        return 1;
    }
    const int prompts_set = !!(prompt_path_set) + !!(prompt_set) + !!(stdin_set);
    if (prompts_set > 1) {
        fprintf(stderr, "Error: --stdin, --file and --prompt are mutually exclusive.\n");
        return 1;
    }
    // Must have some prompt.
    if (prompts_set == 0) {
        fprintf(stderr, "Error: must specify one of: --stdin, --file or --prompt.\n");
        return 1;
    }

    GGML_ASSERT(model_path);
    GGML_ASSERT(prompt_path || prompt_arg || stdin_set);

    //////
    // Figure out where will the prompt come from.
    //////

    std::string prompt;
    if (prompt_path_set) {
        bool success = false;
        prompt = read_prompt_from_file(prompt_path, success);
        if (!success) {
            return 1;
        }
    } else if (prompt_set) {
        prompt = prompt_arg;
    } else {
        GGML_ASSERT(stdin_set);
        // we read stdin *after* loading model (early exit if model cannot
        // be loaded, which can be a nicer user experience)
    }

    //////
    // Start actually doing the tokenizing stuff.
    //////

    if (disable_logging) {
        llama_log_set(llama_log_callback_null, NULL);
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Error: could not load model from file '%s'.\n", model_path);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: could not create context.\n");
        return 1;
    }

    // read entire prompt from stdin?
    if (stdin_set) {
        GGML_ASSERT(!prompt_path_set && !prompt_set);

        std::stringstream stdin_buffer;
        stdin_buffer << std::cin.rdbuf();
        if (std::cin.fail()) {
            fprintf(stderr, "Error: could not read the entire standard input.\n");
            return 1;
        }

        prompt = stdin_buffer.str();
    }

    const bool model_wants_add_bos = llama_add_bos_token(model);
    const bool add_bos = model_wants_add_bos && !no_bos;
    const bool parse_special = !no_parse_special;
    const bool escape = !no_escape;

    if (escape) {
        string_process_escapes(prompt);
    }

    std::vector<llama_token> tokens;
    tokens = common_tokenize(model, prompt, add_bos, parse_special);

    if (printing_ids) {
        printf("[");
    }

    for (int i = 0; i < (int) tokens.size(); i++) {
        if (printing_ids) {
            if (i > 0) {
                printf(", ");
            }
            printf("%d", tokens[i]);
        } else {
            bool invalid_utf8 = false;
            printf("%6d -> '", tokens[i]);
            write_utf8_cstr_to_stdout(common_token_to_piece(ctx, tokens[i]).c_str(), invalid_utf8);
            if (invalid_utf8) {
                printf("' (utf-8 decode failure)\n");
            } else {
                printf("'\n");
            }
        }
    }

    if (printing_ids) {
        printf("]\n");
    }

    if (show_token_count) {
        printf("Total number of tokens: %zu\n", tokens.size());
    }
    // silence valgrind
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
