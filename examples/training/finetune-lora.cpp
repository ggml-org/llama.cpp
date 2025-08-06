#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


static uint32_t parse_lora_modules(const std::string& modules_str) {
    if (modules_str.empty()) {
        return LLAMA_LORA_TARGET_ATTN_Q | LLAMA_LORA_TARGET_ATTN_K | LLAMA_LORA_TARGET_ATTN_V | LLAMA_LORA_TARGET_ATTN_O;
    }
    
    static const std::map<std::string, uint32_t> module_map = {
        {"attn_q",    LLAMA_LORA_TARGET_ATTN_Q},
        {"attn_k",    LLAMA_LORA_TARGET_ATTN_K},  
        {"attn_v",    LLAMA_LORA_TARGET_ATTN_V},
        {"attn_o",    LLAMA_LORA_TARGET_ATTN_O},
        {"ffn_gate",  LLAMA_LORA_TARGET_FFN_GATE},
        {"ffn_up",    LLAMA_LORA_TARGET_FFN_UP},
        {"ffn_down",  LLAMA_LORA_TARGET_FFN_DOWN},
        {"output",    LLAMA_LORA_TARGET_OUTPUT},
        {"all",       LLAMA_LORA_TARGET_ALL}
    };
    
    uint32_t target_modules = 0;
    std::stringstream ss(modules_str);
    std::string module;
    
    while (std::getline(ss, module, ',')) {
        module.erase(0, module.find_first_not_of(" \t"));
        module.erase(module.find_last_not_of(" \t") + 1);
        
        auto it = module_map.find(module);
        if (it != module_map.end()) {
            target_modules |= it->second;
            LOG_INF("Added target module: %s\n", module.c_str());
        } else {
            LOG_ERR("Unknown LoRA target module: %s\n", module.c_str());
            LOG_ERR("Available modules: attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, output, all\n");
            return 0;
        }
    }
    
    return target_modules;
}

static void print_lora_usage() {
    printf("\nLoRA Fine-tuning Parameters:\n");
    printf("  --lora-rank N              LoRA rank (default: 8, range: 1-512)\n");
    printf("  --lora-alpha N             LoRA alpha scaling factor (default: 16.0, range: 0.1-1000.0)\n");
    printf("  --lora-modules MODULES     Target modules as comma-separated list (default: attn_q,attn_k,attn_v,attn_o)\n");
    printf("                             Available modules: attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, output, all\n");
    printf("                             Examples: \"attn_q,attn_v\" or \"all\" or \"attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down\"\n");
    printf("  --output-adapter PATH      Output path for trained adapter (default: auto-generated)\n");
    printf("\nExamples:\n");
    printf("  # Train with rank=16, alpha=32, all attention modules\n");
    printf("  %s -m model.gguf -f dataset.txt --lora-rank 16 --lora-alpha 32 --lora-modules attn_q,attn_k,attn_v,attn_o\n", "finetune-lora");
    printf("\n  # Fine-tune existing adapter with all modules\n");
    printf("  %s -m model.gguf -f dataset.txt --lora existing.gguf --output-adapter improved.gguf\n", "finetune-lora");
    printf("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    int32_t lora_rank = 8;     
    float lora_alpha = 16.0f;
    std::string lora_modules_str;
    std::string output_adapter_path;

    params.escape = false;

    auto remove_arg_pair = [&](int i) {
        for (int j = i; j < argc - 2; j++) {
            argv[j] = argv[j + 2];
        }
        argc -= 2;
    };

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--lora-rank") == 0) {
            lora_rank = std::atoi(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--lora-alpha") == 0) {
            lora_alpha = std::atof(argv[i + 1]);
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--lora-modules") == 0) {
            lora_modules_str = argv[i + 1];
            remove_arg_pair(i);
            i--;
        } else if (strcmp(argv[i], "--output-adapter") == 0) {
            output_adapter_path = argv[i + 1];
            remove_arg_pair(i);
            i--;
        }
    }

    LOG_INF("Using LoRA parameters: rank=%d, alpha=%.1f\n", lora_rank, lora_alpha);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_lora_usage();
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        print_lora_usage();
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n", __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr   & model = llama_init.model;
    llama_context_ptr & ctx   = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    uint32_t target_modules = parse_lora_modules(lora_modules_str);
    if (target_modules == 0) {
        return 1;
    }

    struct llama_lora_training_params lora_params = {
        /*target_modules =*/ target_modules,
        /*rank           =*/ lora_rank,
        /*alpha          =*/ lora_alpha,
        /*dropout        =*/ 0.0f,
        /*init_std       =*/ 0.02f,
    };

    bool has_existing_lora = !params.lora_adapters.empty();
    struct llama_adapter_lora * trained_adapter = nullptr;
    
    if (has_existing_lora) {
        LOG_INF("Finetuning existing LoRA adapters\n");
        LOG_INF("Found %zu existing LoRA adapters to train\n", params.lora_adapters.size());\
        trained_adapter = params.lora_adapters[0].ptr;
        if (!trained_adapter) {
            LOG_ERR("Existing LoRA adapter is null\n");
            return 1;
        }
    } else {
        LOG_INF("Target modules: Q=%s, K=%s, V=%s, O=%s, GATE=%s, UP=%s, DOWN=%s, OUTPUT=%s\n",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_Q) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_K) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_V) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_ATTN_O) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_GATE) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_UP) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_FFN_DOWN) ? "yes" : "no",
            (lora_params.target_modules & LLAMA_LORA_TARGET_OUTPUT) ? "yes" : "no");
        
        LOG_INF("LoRA configuration: rank=%d, alpha=%.1f (scaling=%.3f)\n", 
            lora_params.rank, lora_params.alpha, lora_params.alpha / lora_params.rank);

        trained_adapter = llama_lora_training_init(ctx.get(), model.get(), &lora_params);
        if (!trained_adapter) {
            LOG_ERR("%s: LoRA training initialization failed\n", __func__);
            return 1;
        }
    }

    constexpr float val_split = 0.05f;

    std::vector<llama_token> tokens = common_tokenize(ctx.get(), params.prompt, true);
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx.get(), tokens, llama_n_ctx(ctx.get())/2);

    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = 1e-5f; // learning rate

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ llama_opt_param_filter_lora,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/ &optimizer_params,
    };
    llama_opt_init(ctx.get(), model.get(), lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (int epoch = 0; epoch < 2; ++epoch) {
        llama_opt_epoch(ctx.get(), dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    std::string adapter_filename;
    if (!output_adapter_path.empty()) {
        adapter_filename = output_adapter_path;
    } else if (has_existing_lora) {
        adapter_filename = "finetuned-lora-adapter.gguf";
        LOG_INF("Finetuned existing lora adapter, saving as: %s\n", adapter_filename.c_str());
    } else {
        adapter_filename = "trained-lora-adapter.gguf";
        LOG_INF("Saving new lora adapter: %s\n", adapter_filename.c_str());
    }

    if (trained_adapter) {
        if (llama_lora_save_adapter(trained_adapter, adapter_filename.c_str(), model.get())) {
            std::ifstream adapter_file(adapter_filename, std::ios::binary | std::ios::ate);
            if (adapter_file.is_open()) {
                std::streamsize adapter_size = adapter_file.tellg();
                LOG_INF("LoRA adapter saved: %s (%.2f MB)\n", 
                        adapter_filename.c_str(), adapter_size / (1024.0 * 1024.0));
                adapter_file.close();
            }
        } else {
            LOG_ERR("Failed to save LoRA adapter\n");
        }
    } else {
        LOG_ERR("No trained adapter available for saving\n");
    }

    llama_backend_free();

    return 0;
}
