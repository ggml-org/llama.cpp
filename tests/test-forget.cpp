// Test that forgetting to release the model and context doesn't cause a crash later on.
//
// This could happen for example if the user decides to store the model and context in a static.

#include "llama.h"
#include "common.h"

int main(int argc, char ** argv) {
    auto * model_path = common_get_model_or_exit(argc, argv);

    llama_backend_init();

    auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
    auto * ctx = llama_init_from_model(model, llama_context_default_params());

    GGML_UNUSED(ctx);
    // Deliberately "forgotten" here.
    // llama_free(ctx);
    // llama_model_free(model);

    llama_backend_free();

    return 0;
}
