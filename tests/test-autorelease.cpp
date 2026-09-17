// ref: https://github.com/ggml-org/llama.cpp/issues/4952#issuecomment-1892864763

#include <thread>
#include <vector>

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    // the model path is this test's only positional argument
    char * model_argv[2] = { argv[0], nullptr };

    {
        common_params params;
        params.model.path = "."; // this test takes no model
        common_init();

        std::vector<char *> common_argv;
        common_argv.push_back(argv[0]);
        for (int i = 1; i < argc; i++) {
            if (argv[i][0] == '-') {
                common_argv.push_back(argv[i]); // an option: let common_params_parse handle it
            } else if (model_argv[1] == nullptr) {
                model_argv[1] = argv[i];
            }
        }
        common_argv.push_back(nullptr);
        if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
    }

    // falls back to LLAMACPP_TEST_MODELFILE, or warns and exits if no model is given
    auto * model_path = common_get_model_or_exit(model_argv[1] == nullptr ? 1 : 2, model_argv);

    // that call exits when no model is given, so the verdict below is only reached with a model
    LOG("%s: running\n", "test-autorelease");

    std::thread([&model_path]() {
        llama_backend_init();
        auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
        auto * ctx = llama_init_from_model(model, llama_context_default_params());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }).join();

    LOG("%s: %s\n", "test-autorelease", "PASSED");
    common_log_flush(common_log_main());
    return 0;
}
