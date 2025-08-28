#include <cstdlib>

#include "get-model.h"
#include "llama.h"

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);
    auto * file       = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    llama_backend_init();
    auto params              = llama_model_params{};
    params.use_mmap          = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        fprintf(stderr, "%.2f%% ", progress * 100.0f);
        // true means: Don't cancel the load
        return true;
    };
    auto * model = llama_model_load_from_file(model_path, params);

    // Add newline after progress output
    fprintf(stderr, "\n");

    if (model == nullptr) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Model loaded successfully\n");
    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
