#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cstdlib>
#include <vector>

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
    LOG("%s: running\n", "test-model-load-cancel");

    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        LOG_ERR("no model at '%s' found\n", model_path);
        LOG("%s: %s\n", "test-model-load-cancel", "FAILED");
        common_log_flush(common_log_main());
        return EXIT_FAILURE;
    }

    LOG_INF("using '%s'\n", model_path);
    fclose(file);

    llama_backend_init();
    auto params = llama_model_params{};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.50;
    };
    auto * model = llama_model_load_from_file(model_path, params);
    llama_backend_free();
    LOG("%s: %s\n", "test-model-load-cancel", model == nullptr ? "PASSED" : "FAILED");
    common_log_flush(common_log_main());
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
