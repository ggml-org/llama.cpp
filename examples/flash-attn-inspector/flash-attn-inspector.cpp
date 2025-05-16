#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 */
struct callback_data {
    std::vector<uint8_t> data_src0;
    std::vector<uint8_t> data_src1;
    std::vector<uint8_t> data_src2;
    std::vector<uint8_t> data_src3;
    std::vector<uint8_t> data_out;
};

// Forward declaration if ggml_ne_string is used before definition
// static std::string ggml_ne_string(const ggml_tensor * t);

static std::string ggml_tensor_shape_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS && t->ne[i+1] > 0) { // Print comma only if next dim exists
             if (i < GGML_MAX_DIMS -1 && t->ne[i+1] != 0 ) { // check if there is a next dimension
                bool has_more_dims = false;
                for(int j=i+1; j < GGML_MAX_DIMS; ++j) {
                    if (t->ne[j] != 0 && t->ne[j] != 1) { // only count meaningful dims
                        has_more_dims = true;
                        break;
                    }
                }
                if(has_more_dims || (i<2 && t->ne[i+1] > 1)) str += ", "; // Heuristic for 1D/2D vs higher D
             }
        }
    }
    // Remove trailing comma and space if any for tensors with fewer than MAX_DIMS
    if (str.length() > 2 && str.substr(str.length() - 2) == ", ") {
        str = str.substr(0, str.length() - 2);
    }
    return str;
}


static void ggml_print_tensor_summary(const char* title, const ggml_tensor *t) {
    if (!t) return;
    LOG("%s: %s, Type: %s, Shape: [%s]\n",
        title,
        (t->name[0] != '\0' ? t->name : "(unnamed)"),
        ggml_type_name(t->type),
        ggml_tensor_shape_string(t).c_str());
}

static void ggml_print_tensor_data(const ggml_tensor * t, uint8_t * data_ptr_override, int64_t n_to_print) {
    ggml_print_tensor_summary("Tensor Data Dump", t);

    uint8_t * data_to_print = data_ptr_override;
    if (!data_to_print) {
        LOG(" (Data not available or not on host for direct printing)\n");
        return;
    }
    if (ggml_is_quantized(t->type)) {
        LOG(" (Quantized tensor - data printing not implemented for this example)\n");
        return;
    }

    GGML_ASSERT(n_to_print > 0);
    float sum = 0;
    const int64_t* ne = t->ne;
    const size_t* nb = t->nb;
    ggml_type type = t->type;

    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n_to_print && ne[2] > 2*n_to_print) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n_to_print;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n_to_print && ne[1] > 2*n_to_print) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n_to_print;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n_to_print && ne[0] > 2*n_to_print) {
                        LOG("..., ");
                        i0 = ne[0] - n_to_print;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data_to_print[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data_to_print[i];
                    } else {
                        LOG("Unsupported type for printing: %s\n", ggml_type_name(type));
                        GGML_ABORT("fatal error: unsupported tensor type in ggml_print_tensor_data");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}


static void get_tensor_data_if_needed(struct ggml_tensor * t, std::vector<uint8_t>& buffer, uint8_t** data_ptr) {
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host) {
        *data_ptr = (uint8_t *)t->data;
    } else {
        if (t->data == nullptr && ggml_nbytes(t) > 0) { // Tensor might have data on device but t->data is null if not mapped
            LOG("Tensor %s data is on device and not mapped to host, attempting to fetch.\n", (t->name[0] != '\0' ? t->name : "(unnamed)"));
        } else if (t->data == nullptr && ggml_nbytes(t) == 0) {
             LOG("Tensor %s has no data (0 bytes).\n", (t->name[0] != '\0' ? t->name : "(unnamed)"));
            *data_ptr = nullptr;
            return;
        }
        auto n_bytes = ggml_nbytes(t);
        buffer.resize(n_bytes);
        ggml_backend_tensor_get(t, buffer.data(), 0, n_bytes);
        *data_ptr = buffer.data();
    }
}


/**
 * GGML operations callback during the graph execution.
 * This callback specifically looks for GGML_OP_FLASH_ATTN_EXT operations
 * and prints their input and output tensor shapes.
 */
static bool ggml_flash_attn_ext_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    if (t->op != GGML_OP_FLASH_ATTN_EXT) {
        return true; // Continue for other ops
    }

    if (ask) {
        return true; // We are interested in data for GGML_OP_FLASH_ATTN_EXT
    }

    LOG("\nFound GGML_OP_FLASH_ATTN_EXT operation.\n");
    
    // Print output tensor shape
    LOG("Output Tensor Shape: [%d, %d, %d, %d]\n", 
        t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    
    // Print the first input tensor (src[0]) in detail
    if (t->src[0] != nullptr) {
        struct ggml_tensor * q = t->src[0];
        LOG("First input tensor (Q) details:\n");
        LOG("  Name: %s\n", q->name[0] != '\0' ? q->name : "(unnamed)");
        LOG("  Type: %s\n", ggml_type_name(q->type));
        LOG("  Shape: [%d, %d, %d, %d]\n", q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
        LOG("  Stride: [%d, %d, %d, %d]\n", q->nb[0], q->nb[1], q->nb[2], q->nb[3]);
        
        // Get tensor data
        std::vector<uint8_t> buffer;
        uint8_t* data_ptr = nullptr;
        get_tensor_data_if_needed(q, buffer, &data_ptr);
        
        if (data_ptr != nullptr) {
            LOG("  Data preview:\n");
            ggml_print_tensor_data(q, data_ptr, 3);
        } else {
            LOG("  Data: Not available\n");
        }
    }
    
    // Print input tensor shapes
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = t->src[i];
        if (src == nullptr) {
            continue;
        }
        
        LOG("Input %d Shape: [%d, %d, %d, %d]\n", 
            i, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    }

    LOG("Finished processing GGML_OP_FLASH_ATTN_EXT: %s\n\n", 
        (t->name[0] != '\0' ? t->name : "(unnamed)"));

    return true;
}

static void test_prompt(llama_context * ctx, int n_prompt, int n_batch, int n_threads, bool do_profile=false) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
}

static void test_gen(llama_context * ctx, int n_gen, int n_threads, bool do_profile=false) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    for (int i = 0; i < n_gen; i++) {
        llama_decode(ctx, llama_batch_get_one(&token, 1));
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Use a default prompt if none is provided, as Flash Attention might not be triggered by very short/simple prompts.
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        prompt = "The quick brown fox jumps over the lazy dog.";
    }
    LOG("Using prompt: %s\n", prompt.c_str());

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : failed to tokenize prompt\n", __func__);
        return false;
    }
    LOG("Tokenized prompt to %zu tokens.\n", tokens.size());


    // Ensure the context is large enough if n_len is not set by default from common_params
    // This is a simple heuristic; complex models might need more specific context sizing.
    if (static_cast<size_t>(params.n_ctx) < tokens.size() + 16) { // Add some buffer
        LOG_INF("Prompt size (%zu) is close to or exceeds context size (%d). Consider increasing context size.\n", tokens.size(), params.n_ctx);
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), 1))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }
    LOG(" llama_decode successful.\n");

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;
    common_params params;

    // Initialize with a default model that is likely to use Flash Attention.
    // User can override with -m
    params.model.path = "ggml-model-f16.gguf"; // A common default, adjust if needed or rely on user.
    params.flash_attn = true;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "Failed to parse common_params.\n");
        return 1;
    }
    if (params.n_ctx == 0) {
        params.n_ctx = 512; // Default context size for the example
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_flash_attn_ext_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false; // Disable warmup to see the first run with the callback

    LOG("Initializing LLaMA model and context...\n");
    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init LLaMA model or context. Ensure model path is correct and model is compatible.\n", __func__);
        return 1;
    }
    LOG("LLaMA model and context initialized successfully.\n");

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("System Info: %s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    LOG("Running inference...\n");
    bool OK = run(ctx, params);
    if (!OK) {
        LOG_ERR("Execution failed.\n");
        llama_free(ctx); // Ensure resources are freed on failure
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    LOG("Inference completed.\n");

    LOG("\n");
    // llama_perf_context_print(ctx); // Optional: print performance data

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    LOG("Cleaned up LLaMA resources. Exiting.\n");

    return 0;
} 