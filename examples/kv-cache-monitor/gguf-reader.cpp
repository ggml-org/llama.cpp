#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static void print_tensor_info(struct gguf_context* ctx, int tensor_idx) {
    const char* name = gguf_get_tensor_name(ctx, tensor_idx);
    const size_t size = gguf_get_tensor_size(ctx, tensor_idx);
    const size_t offset = gguf_get_tensor_offset(ctx, tensor_idx);
    
    printf("Tensor[%d]: name=%s, size=%zu bytes, offset=%zu\n", 
           tensor_idx, name, size, offset);
}

static void print_metadata(struct gguf_context* ctx) {
    printf("\n=== GGUF Metadata ===\n");
    printf("Version: %d\n", gguf_get_version(ctx));
    printf("Alignment: %zu\n", gguf_get_alignment(ctx));
    printf("Data offset: %zu\n", gguf_get_data_offset(ctx));
    
    const int n_kv = gguf_get_n_kv(ctx);
    printf("Key-Value pairs: %d\n", n_kv);
    
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        const enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        printf("  [%d] %s (type: %d) = ", i, key, type);
        
        switch (type) {
            case GGUF_TYPE_STRING:
                printf("\"%s\"", gguf_get_val_str(ctx, i));
                break;
            case GGUF_TYPE_INT32:
                printf("%d", gguf_get_val_i32(ctx, i));
                break;
            case GGUF_TYPE_BOOL:
                printf("%s", gguf_get_val_bool(ctx, i) ? "true" : "false");
                break;
            default:
                printf("(unsupported type)");
                break;
        }
        printf("\n");
    }
    printf("=====================\n\n");
}

static void print_tensor_data_sample(struct ggml_context* ctx_data, const char* tensor_name) {
    struct ggml_tensor* tensor = ggml_get_tensor(ctx_data, tensor_name);
    if (!tensor) {
        printf("Tensor '%s' not found in context\n", tensor_name);
        return;
    }
    
    printf("\nTensor '%s' data sample:\n", tensor_name);
    printf("  Type: %s\n", ggml_type_name(tensor->type));
    printf("  Dimensions: [%ld, %ld, %ld, %ld]\n", 
           tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    printf("  Total elements: %ld\n", ggml_nelements(tensor));
    
    // Print first few elements based on type
    const int max_print = 10;
    const int n_print = std::min(max_print, (int)ggml_nelements(tensor));
    
    printf("  First %d elements: ", n_print);
    
    if (tensor->type == GGML_TYPE_F32) {
        const float* data = (const float*)tensor->data;
        for (int i = 0; i < n_print; ++i) {
            printf("%.6f ", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* data = (const ggml_fp16_t*)tensor->data;
        for (int i = 0; i < n_print; ++i) {
            printf("%.6f ", ggml_fp16_to_fp32(data[i]));
        }
    } else {
        printf("(unsupported type for display)");
    }
    printf("\n");
}

static bool read_gguf_file(const std::string& filename, bool show_data_samples) {
    printf("Reading GGUF file: %s\n", filename.c_str());
    
    struct ggml_context* ctx_data = nullptr;
    
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    
    struct gguf_context* ctx = gguf_init_from_file(filename.c_str(), params);
    if (!ctx) {
        printf("ERROR: Failed to load GGUF file: %s\n", filename.c_str());
        return false;
    }
    
    // Print metadata
    print_metadata(ctx);
    
    // Print tensor information
    const int n_tensors = gguf_get_n_tensors(ctx);
    printf("=== Tensors (%d total) ===\n", n_tensors);
    
    for (int i = 0; i < n_tensors; ++i) {
        print_tensor_info(ctx, i);
    }
    printf("==========================\n");
    
    // Show data samples if requested and context is available
    if (show_data_samples && ctx_data) {
        printf("\n=== Tensor Data Samples ===\n");
        for (int i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(ctx, i);
            print_tensor_data_sample(ctx_data, name);
        }
        printf("===========================\n");
    }
    
    // Cleanup
    if (ctx_data) {
        ggml_free(ctx_data);
    }
    gguf_free(ctx);
    
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <gguf_file> [--show-data]\n", argv[0]);
        printf("  <gguf_file>   Path to GGUF file to read\n");
        printf("  --show-data   Show sample data from tensors\n");
        printf("\nExample:\n");
        printf("  %s traced_tensors.gguf\n", argv[0]);
        printf("  %s traced_tensors.gguf --show-data\n", argv[0]);
        return 1;
    }
    
    std::string filename = argv[1];
    bool show_data_samples = false;
    
    // Parse additional arguments
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--show-data") == 0) {
            show_data_samples = true;
        }
    }
    
    printf("GGUF Reader for KQV Traced Tensors\n");
    printf("===================================\n");
    
    if (!read_gguf_file(filename, show_data_samples)) {
        return 1;
    }
    
    printf("\nReading completed successfully!\n");
    return 0;
} 