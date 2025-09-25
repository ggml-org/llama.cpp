#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctype.h>
#include <filesystem>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-ngl n_gpu_layers] -embd-mode [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int ngl = 0;
    bool embedding_mode = false;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-embd-mode") == 0) {
                if (i + 1 < argc) {
                    try {
                        embedding_mode = true;
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }

        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }

        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // Extract basename from model_path
    const char * basename = strrchr(model_path.c_str(), '/');
    basename = (basename == NULL) ? model_path.c_str() : basename + 1;

    char model_name[256];
    strncpy(model_name, basename, 255);
    model_name[255] = '\0';

    char * dot = strrchr(model_name, '.');
    if (dot != NULL && strcmp(dot, ".gguf") == 0) {
        *dot = '\0';
    }
    printf("Model name: %s\n", model_name);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;
    if (embedding_mode) {
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        ctx_params.n_ubatch = ctx_params.n_batch;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("Input prompt: \"%s\"\n", prompt.c_str());
    printf("Tokenized prompt (%d tokens): ", n_prompt);
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
    printf("\n");

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }

    float * logits;
    int n_logits;
    const char * type;

    if (embedding_mode) {
        logits = llama_get_embeddings(ctx);
        n_logits = llama_model_n_embd(model) * batch.n_tokens;
        type = "-embeddings";

        const int n_embd = llama_model_n_embd(model);
        const int n_embd_count = batch.n_tokens;

        printf("Embedding dimension: %d\n", n_embd);
        printf("\n");

        // Print embeddings in the specified format
        for (int j = 0; j < n_embd_count; j++) {
            printf("embedding %d: ", j);

            // Print first 3 values
            for (int i = 0; i < 3 && i < n_embd; i++) {
                printf("%9.6f ", logits[j * n_embd + i]);
            }

            printf(" ... ");

            // Print last 3 values
            for (int i = n_embd - 3; i < n_embd; i++) {
                if (i >= 0) {
                    printf("%9.6f ", logits[j * n_embd + i]);
                }
            }

            printf("\n");
        }
        printf("\n");

        printf("Embeddings size: %d\n", n_logits);
    } else {
        logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        n_logits = llama_vocab_n_tokens(vocab);
        type = "";
        printf("Vocab size: %d\n", n_logits);
    }

    std::filesystem::create_directory("data");

    // Save logits to binary file
    char bin_filename[512];
    snprintf(bin_filename, sizeof(bin_filename), "data/llamacpp-%s%s.bin", model_name, type);
    printf("Saving logits to %s\n", bin_filename);

    FILE * f = fopen(bin_filename, "wb");
    if (f == NULL) {
        fprintf(stderr, "%s: error: failed to open binary output file\n", __func__);
        return 1;
    }
    fwrite(logits, sizeof(float), n_logits, f);
    fclose(f);

    // Also save as text for debugging
    char txt_filename[512];
    snprintf(txt_filename, sizeof(txt_filename), "data/llamacpp-%s%s.txt", model_name, type);
    f = fopen(txt_filename, "w");
    if (f == NULL) {
        fprintf(stderr, "%s: error: failed to open text output file\n", __func__);
        return 1;
    }
    for (int i = 0; i < n_logits; i++) {
        fprintf(f, "%d: %.6f\n", i, logits[i]);
    }
    fclose(f);

    if (!embedding_mode) {
        printf("First 10 logits: ");
        for (int i = 0; i < 10 && i < n_logits; i++) {
            printf("%.6f ", logits[i]);
        }
        printf("\n");

        printf("Last 10 logits: ");
        for (int i = n_logits - 10; i < n_logits; i++) {
            if (i >= 0) printf("%.6f ", logits[i]);
        }
        printf("\n\n");
    }

    printf("Logits saved to %s\n", bin_filename);
    printf("Logits saved to %s\n", txt_filename);

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
