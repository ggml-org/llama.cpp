#include "ggml-backend.h"
#include "get-model.h"
#include "llama.h"
#include "gguf.h"

#include "../src/llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
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

static ggml_backend_dev_t get_test_device(int argc, char * argv[]) {
    std::string device = "cpu";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--device expects a value (cpu or gpu)\n");
                return nullptr;
            }
            device = argv[++i];
            continue;
        }

        if (std::strncmp(argv[i], "--device=", 9) == 0) {
            device = argv[i] + 9;
        }
    }

    if (device == "cpu") {
        return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }

    if (device == "gpu") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        return dev;
    }

    fprintf(stderr, "unsupported --device value '%s' (expected cpu or gpu)\n", device.c_str());
    return nullptr;
}

static bool should_print_breakdown(int argc, char * argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--print-breakdown") == 0) {
            return true;
        }
    }

    return false;
}

static std::string format_size(size_t size) {
    const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(size);
    size_t unit = 0;

    while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), unit == 0 ? "%.0f %s" : "%.2f %s", value, units[unit]);
    return buf;
}

static void print_memory_breakdown(
        const char * label,
        const std::map<ggml_backend_buffer_type_t, size_t> & memory_breakdown) {
    std::vector<std::pair<ggml_backend_buffer_type_t, size_t>> entries(memory_breakdown.begin(), memory_breakdown.end());
    std::sort(entries.begin(), entries.end(), [](const auto & lhs, const auto & rhs) {
        return std::strcmp(ggml_backend_buft_name(lhs.first), ggml_backend_buft_name(rhs.first)) < 0;
    });

    size_t total = 0;
    std::printf("%s:\n", label);
    for (const auto & [buft, size] : entries) {
        std::printf("  %s: %s\n", ggml_backend_buft_name(buft), format_size(size).c_str());
        total += size;
    }
    std::printf("  total: %s\n", format_size(total).c_str());
}

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);
    const bool print_breakdown = should_print_breakdown(argc, argv);
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

    ggml_backend_dev_t dev = get_test_device(argc, argv);
    if (dev == nullptr) {
        llama_backend_free();
        return EXIT_FAILURE;
    }

    ggml_backend_dev_t devices[] = { dev, nullptr };

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

    if (print_breakdown) {
        std::printf("model_path: %s\n", model_path);
        print_memory_breakdown("memory_breakdown_from_file", mb_from_file);
        print_memory_breakdown("memory_breakdown_from_buffer", mb_from_buffer);
        std::printf("memory_breakdown_equal: %s\n", ok ? "true" : "false");
    }

    llama_model_free(model_from_buffer);
    llama_model_free(model_from_file);
    gguf_free(gguf_ctx);
    llama_backend_free();

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
