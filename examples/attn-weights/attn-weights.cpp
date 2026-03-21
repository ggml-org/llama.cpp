// Attention weights extraction example
//
// Decodes a prompt and prints an ASCII heatmap of the attention pattern
// for a selected (layer, head) pair.
//
// Usage:
//   ./llama-attn-weights -m model.gguf [-l layer] [-hd head] [-ngl N] [prompt]
//
// Defaults: last layer, head 0.
// Use -l -1 for the last layer.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string token_str(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "???";
    }
    // sanitize: replace newlines/tabs with visible chars
    std::string s(buf, n);
    for (auto & c : s) {
        if (c == '\n') { c = '|'; }
        if (c == '\t') { c = ' '; }
        if (c == '\r') { c = ' '; }
    }
    return s;
}

// Truncate or pad a string to exactly `w` characters
static std::string fit(const std::string & s, int w) {
    if ((int) s.size() <= w) {
        return s + std::string(w - (int) s.size(), ' ');
    }
    return s.substr(0, w);
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-l layer] [-hd head] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
    printf("  -l   layer index (default: -1 = last layer)\n");
    printf("  -hd  head index  (default: 0)\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The cat sat on the mat";
    int ngl       = 99;
    int layer     = -1;  // -1 means last layer
    int head      = 0;

    // parse args
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) { model_path = argv[++i]; }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-l") == 0) {
                if (i + 1 < argc) {
                    try { layer = std::stoi(argv[++i]); } catch (...) { print_usage(argc, argv); return 1; }
                } else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-hd") == 0) {
                if (i + 1 < argc) {
                    try { head = std::stoi(argv[++i]); } catch (...) { print_usage(argc, argv); return 1; }
                } else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try { ngl = std::stoi(argv[++i]); } catch (...) { print_usage(argc, argv); return 1; }
                } else { print_usage(argc, argv); return 1; }
            } else {
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

    // load backends & model
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // resolve layer index
    const int32_t n_layer = llama_model_n_layer(model);
    if (layer < 0) {
        layer = n_layer + layer;  // -1 -> last layer
    }
    if (layer < 0 || layer >= n_layer) {
        fprintf(stderr, "%s: error: layer %d out of range [0, %d)\n", __func__, layer, n_layer);
        llama_model_free(model);
        return 1;
    }

    // tokenize
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // create context with attention weights enabled
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx        = n_prompt;
    ctx_params.n_batch      = n_prompt;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.attn_weights = true;
    ctx_params.no_perf      = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    // configure which (layer, head) to extract
    {
        int32_t layers[] = { (int32_t) layer };
        int32_t heads[]  = { (int32_t) head };
        llama_set_attn_heads(ctx, layers, heads, 1);
    }

    // prepare batch — request output for all tokens
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;  // request output for every token
    }
    batch.n_tokens = n_prompt;

    // decode
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s: error: decode failed\n", __func__);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // collect token labels
    std::vector<std::string> labels(n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        labels[i] = token_str(vocab, tokens[i]);
    }

    const int32_t n_kv = llama_get_attn_n_kv(ctx);
    const int32_t n_ctx = llama_n_ctx(ctx);

    printf("\nAttention weights — layer %d, head %d\n", layer, head);
    printf("Tokens: %d, KV entries: %d\n\n", n_prompt, n_kv);

    // print as a matrix: rows = query tokens, columns = key tokens
    // only the first n_kv columns are valid

    const int col_w = 7;  // column width for values
    const int lbl_w = 12; // label column width

    // header row: key token labels
    printf("%s", fit("", lbl_w).c_str());
    for (int k = 0; k < n_kv && k < n_prompt; k++) {
        printf("%s", fit(labels[k], col_w).c_str());
    }
    printf("\n");

    // separator
    printf("%s", std::string(lbl_w + col_w * std::min(n_kv, (int32_t) n_prompt), '-').c_str());
    printf("\n");

    // ASCII heatmap chars from low to high attention
    const char * heat = " .:-=+*#@";
    const int heat_len = 9;

    // one row per query token
    for (int q = 0; q < n_prompt; q++) {
        float * attn = llama_get_attn_ith(ctx, q);
        if (!attn) {
            printf("%s (no data)\n", fit(labels[q], lbl_w).c_str());
            continue;
        }

        // find max for this row (for heatmap scaling)
        float row_max = 0.0f;
        for (int k = 0; k < n_kv && k < n_prompt; k++) {
            if (attn[k] > row_max) {
                row_max = attn[k];
            }
        }

        // numeric row
        printf("%s", fit(labels[q], lbl_w).c_str());
        for (int k = 0; k < n_kv && k < n_prompt; k++) {
            printf(" %5.3f ", attn[k]);
        }
        printf("\n");

        // heatmap row
        if (row_max > 0.0f) {
            printf("%s", fit("", lbl_w).c_str());
            for (int k = 0; k < n_kv && k < n_prompt; k++) {
                float norm = attn[k] / row_max;
                int idx = (int)(norm * (heat_len - 1));
                idx = std::max(0, std::min(heat_len - 1, idx));
                // center the char in the column
                int pad_l = (col_w - 1) / 2;
                int pad_r = col_w - 1 - pad_l;
                printf("%*s%c%*s", pad_l, "", heat[idx], pad_r, "");
            }
            printf("\n");
        }
    }

    printf("\nLegend: ' .:-=+*#@' (low -> high attention)\n");

    // print the n_ctx value for reference
    (void) n_ctx;

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
