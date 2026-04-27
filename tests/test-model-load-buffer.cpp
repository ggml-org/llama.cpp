#include "ggml-backend.h"
#include "get-model.h"
#include "llama.h"
#include "gguf.h"

#include "../src/llama-model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<uint8_t> read_file_to_buffer(FILE * file) {
    if (file == nullptr || fseek(file, 0, SEEK_END) != 0) {
        return {};
    }

    const long size = ftell(file);
    if (size < 0) {
        return {};
    }

    rewind(file);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (fread(data.data(), 1, data.size(), file) != data.size()) {
        return {};
    }

    return data;
}

static void set_tensor_data_noop(struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
}

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);
    FILE * file = fopen(model_path, "rb");
    if (file == nullptr) {
        fprintf(stderr, "failed to open model at '%s'\n", model_path);
        return EXIT_FAILURE;
    }

    const std::vector<uint8_t> data = read_file_to_buffer(file);
    fclose(file);
    if (data.empty()) {
        fprintf(stderr, "failed to read model at '%s'\n", model_path);
        return EXIT_FAILURE;
    }

    llama_backend_init();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        llama_backend_free();
        return EXIT_FAILURE;
    }

    ggml_backend_dev_t devices[] = { cpu_dev, nullptr };

    llama_model_params model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.no_alloc = true;
    model_params.use_mmap = false;
    model_params.progress_callback = [](float progress, void * user_data) {
        GGML_UNUSED(progress);
        GGML_UNUSED(user_data);
        return true;
    };

    gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_context * gguf_ctx = gguf_init_from_buffer(data.data(), data.size(), gguf_params);
    if (gguf_ctx == nullptr || gguf_get_n_tensors(gguf_ctx) <= 0) {
        gguf_free(gguf_ctx);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_model * model_from_file = llama_model_load_from_file(model_path, model_params);
    if (model_from_file == nullptr) {
        gguf_free(gguf_ctx);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    llama_model * model_from_buffer = llama_model_init_from_user(gguf_ctx, set_tensor_data_noop, nullptr, model_params);
    if (model_from_buffer == nullptr) {
        llama_model_free(model_from_file);
        gguf_free(gguf_ctx);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const auto mb_from_file   = model_from_file->memory_breakdown();
    const auto mb_from_buffer = model_from_buffer->memory_breakdown();
    const bool ok = !mb_from_file.empty() && mb_from_file == mb_from_buffer;

    llama_model_free(model_from_buffer);
    llama_model_free(model_from_file);
    gguf_free(gguf_ctx);
    llama_backend_free();

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
