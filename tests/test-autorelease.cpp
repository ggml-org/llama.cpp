// ref: https://github.com/ggml-org/llama.cpp/issues/4952#issuecomment-1892864763

#include <thread>

#include "llama.h"
#include "common.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    auto * model_path = common_get_model_or_exit(argc, argv);

    std::thread([&model_path]() {
        llama_backend_init();
        auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
        auto * ctx = llama_init_from_model(model, llama_context_default_params());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }).join();

    return 0;
}
