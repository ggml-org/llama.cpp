#include "arg.h"
#include "common.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.args) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg);
                        exit(1);
                    }
                }
                // check for env var duplications
                if (opt.env) {
                    if (seen_env_vars.find(opt.env) == seen_env_vars.end()) {
                        seen_env_vars.insert(opt.env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", opt.env);
                        exit(1);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--draft", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.n_max == 123);

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);


    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    if (common_has_curl()) {
        printf("test-arg-parser: test curl-related functions\n\n");
        const char * GOOD_URL = "https://ggml.ai/";
        const char * BAD_URL  = "https://www.google.com/404";
        const char * BIG_FILE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin";

        {
            printf("test-arg-parser: test good URL\n\n");
            auto res = common_remote_get_content(GOOD_URL, {});
            assert(res.first == 200);
            assert(res.second.size() > 0);
            std::string str(res.second.data(), res.second.size());
            assert(str.find("llama.cpp") != std::string::npos);
        }

        {
            printf("test-arg-parser: test bad URL\n\n");
            auto res = common_remote_get_content(BAD_URL, {});
            assert(res.first == 404);
        }

        {
            printf("test-arg-parser: test max size error\n");
            common_remote_params params;
            params.max_size = 1;
            try {
                common_remote_get_content(GOOD_URL, params);
                assert(false && "it should throw an error");
            } catch (std::exception & e) {
                printf("  expected error: %s\n\n", e.what());
            }
        }

        {
            printf("test-arg-parser: test timeout error\n");
            common_remote_params params;
            params.timeout = 1;
            try {
                common_remote_get_content(BIG_FILE, params);
                assert(false && "it should throw an error");
            } catch (std::exception & e) {
                printf("  expected error: %s\n\n", e.what());
            }
        }
    } else {
        printf("test-arg-parser: no curl, skipping curl-related functions\n");
    }

    printf("test-arg-parser: all tests OK\n\n");
}
