#include <cstdint>
#include <cstdlib>
#include <vector>

#include "get-model.h"
#include "llama-cpp.h"
#include "load_into_memory.h"

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    if (is_split_file(model_path)) {
        printf("Skipping split model %s\n", model_path);
        return EXIT_SUCCESS;
    }

    // Manually load into a memory buffer first
    std::vector<std::uint8_t> buffer = load_file_into_buffer(model_path);

    llama_backend_init();
    auto params              = llama_model_params{};
    params.use_mmap          = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        fprintf(stderr, "%.2f%% ", progress * 100.0f);
        // true means: Don't cancel the load
        return true;
    };

    // Test that it can load directly from a buffer
    printf("Loading model from buffer of size %zu bytes\n", buffer.size());
    auto * model = llama_model_load_from_buffer(std::move(buffer), params);

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
