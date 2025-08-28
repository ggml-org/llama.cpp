#include <cstdlib>
#include <thread>
#include <vector>

#include "get-model.h"
#include "llama-cpp.h"
#include "load_into_memory.h"

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    if (!is_split_file(model_path)) {
        printf("Skipping not-split model %s\n", model_path);
        return EXIT_SUCCESS;
    }

    // Manually load into a memory buffer first
    file_entry              tensor_list_file = load_tensor_list_file(model_path);
    std::vector<file_entry> files            = load_files_into_streambuf(model_path);

    llama_backend_init();
    auto params              = llama_model_params{};
    params.use_mmap          = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        fprintf(stderr, "%.2f%% ", progress * 100.0f);
        // true means: Don't cancel the load
        return true;
    };

    printf("Loading model from %zu files\n", files.size());

    std::vector<const char *> file_paths;
    for (size_t i = 0; i < files.size(); i++) {
        printf("Found file %s \n", files[i].path.c_str());
        file_paths.push_back(files[i].path.c_str());
    }

    const char * async_load_context = "test-model-load";
    std::thread  fulfill_thread([&files, &tensor_list_file, &async_load_context]() {
        const bool success = llama_model_load_fulfill_split_future(tensor_list_file.path.c_str(), async_load_context,
                                                                    std::move(tensor_list_file.streambuf));
        printf("Fulfilling tensor list file %s: %s\n", tensor_list_file.path.c_str(), success ? "success" : "failure");
        if (!success) {
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < files.size(); i++) {
            const bool success = llama_model_load_fulfill_split_future(files[i].path.c_str(), async_load_context,
                                                                        std::move(files[i].streambuf));
            printf("Fulfilling file %s: %s\n", files[i].path.c_str(), success ? "success" : "failure");
            if (!success) {
                exit(EXIT_FAILURE);
            }
        }
    });
    fprintf(stderr, "Loading model from splits\n");
    auto * model = llama_model_load_from_split_futures(file_paths.data(), file_paths.size(), async_load_context,
                                                       tensor_list_file.path.c_str(), params);
    fulfill_thread.join();

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
