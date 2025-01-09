// ref: https://github.com/ggerganov/llama.cpp/issues/4952#issuecomment-1892864763

#include <cstdio>
#include <string>
#include <thread>

#include "llama.h"
#include "get-model.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    std::thread([&model_path]() {
        llama_backend_init();
        auto * model = llama_model_load_from_file(model_path, llama_model_default_params());
        auto * ctx = llama_new_context_with_model(model, llama_context_default_params());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }).join();

    return 0;
}
