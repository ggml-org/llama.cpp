// Activation capture tool for llama.cpp
//
// Captures per-layer activation vectors during inference and writes them
// to binary files for sparse autoencoder (SAE) training and feature
// interpretability research.
//
// Usage:
//   llama-activation-capture -m model.gguf -p "prompt text" --layer 20
//   llama-activation-capture -m model.gguf -f prompts.txt --layer 12,20,32
//   llama-activation-capture -m model.gguf -p "prompt text" --top-k 10
//
// Output format (binary):
//   Header (16 bytes): "ACTV" + version:u32 + n_embd:u32 + layer_idx:u32
//   Data: float32[n_embd] per token (no framing)
//
// Read with numpy:
//   data = np.fromfile("out.bin", dtype=np.float32, offset=16).reshape(-1, n_embd)
//
// Author: Magdalene Sullivan <magda.sullivan@gmail.com> (HeraldAI, heraldai.org)

#include "ggml.h"
#include "ggml-backend.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n  Capture layer 20 activations from a prompt:\n");
    printf("    %s -m model.gguf -p \"The meaning of life is\" --layer 20\n", argv[0]);
    printf("\n  Capture multiple layers from a file of prompts:\n");
    printf("    %s -m model.gguf -f prompts.txt --layer 12,20,32\n", argv[0]);
    printf("\n  Show top-K activations (no file output):\n");
    printf("    %s -m model.gguf -p \"Hello world\" --layer 20 --top-k 10\n", argv[0]);
    printf("\n  Capture all layers:\n");
    printf("    %s -m model.gguf -p \"Hello world\" --all-layers\n", argv[0]);
    printf("\n");
}

// ---- activation capture data ----

struct capture_data {
    int n_layers = 0;
    int n_embd   = 0;

    // which layers to capture (-1 = all)
    std::vector<int> target_layers;
    bool capture_all = false;

    // per-layer collected activations: layer_idx -> [token vectors]
    // each vector is n_embd floats (one per token)
    struct layer_data {
        std::vector<float> tokens; // flat: n_tokens * n_embd
        int n_tokens = 0;
    };
    std::vector<layer_data> layers;

    // top-k display mode
    int top_k = 0;

    // per-layer mean activations (for top-k display)
    std::vector<std::vector<float>> layer_means;

    void init(int nl, int ne) {
        n_layers = nl;
        n_embd   = ne;
        layers.resize(nl);
        layer_means.resize(nl);
    }

    bool should_capture(int il) const {
        if (capture_all) return true;
        for (int l : target_layers) {
            if (l == il) return true;
        }
        return false;
    }

    void store(int il, const float * data, int ne, int n_tok) {
        if (il < 0 || il >= n_layers) return;

        // store per-token vectors for binary output
        if (top_k <= 0) {
            auto & ld = layers[il];
            size_t old_size = ld.tokens.size();
            ld.tokens.resize(old_size + (size_t)ne * n_tok);
            memcpy(ld.tokens.data() + old_size, data, sizeof(float) * ne * n_tok);
            ld.n_tokens += n_tok;
        }

        // compute mean for display
        std::vector<float> mean(ne, 0.0f);
        for (int t = 0; t < n_tok; t++) {
            for (int j = 0; j < ne; j++) {
                mean[j] += data[t * ne + j];
            }
        }
        if (n_tok > 0) {
            for (int j = 0; j < ne; j++) {
                mean[j] /= (float)n_tok;
            }
        }
        layer_means[il] = std::move(mean);
    }
};

// ---- eval callback ----

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (capture_data *)user_data;

    const bool is_l_out = strncmp(t->name, "l_out", 5) == 0
                       || strncmp(t->name, "final_output", 12) == 0;

    if (ask) {
        if (!is_l_out) return false;
        int il = -1;
        const char * dash = strchr(t->name, '-');
        if (dash) il = atoi(dash + 1);
        return il >= 0 && cap->should_capture(il);
    }

    if (!is_l_out) return true;

    int il = -1;
    const char * dash = strchr(t->name, '-');
    if (dash) il = atoi(dash + 1);
    if (il < 0) return true;

    if (t->type != GGML_TYPE_F32) return true;

    int ne     = (int)t->ne[0];
    int n_tok  = (int)t->ne[1];
    size_t n_bytes = (size_t)ne * n_tok * sizeof(float);

    std::vector<float> buf(ne * n_tok);
    ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);

    cap->store(il, buf.data(), ne, n_tok);

    return true;
}

// ---- binary output ----

static bool write_activations(const std::string & path, const capture_data::layer_data & ld, int n_embd, int layer_idx) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path.c_str());
        return false;
    }

    // header: magic + version + n_embd + layer_idx
    const char magic[4] = {'A', 'C', 'T', 'V'};
    uint32_t version  = 1;
    uint32_t embd     = (uint32_t)n_embd;
    uint32_t lidx     = (uint32_t)layer_idx;

    fwrite(magic, 1, 4, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&embd,    sizeof(uint32_t), 1, f);
    fwrite(&lidx,    sizeof(uint32_t), 1, f);

    // data: float32[n_embd] per token
    fwrite(ld.tokens.data(), sizeof(float), ld.tokens.size(), f);
    fclose(f);

    return true;
}

// ---- top-k display ----

static void print_top_k(const capture_data & cap, int top_k) {
    for (int il = 0; il < cap.n_layers; il++) {
        if (!cap.should_capture(il)) continue;
        if (il >= (int)cap.layer_means.size() || cap.layer_means[il].empty()) continue;

        const auto & means = cap.layer_means[il];
        int n = (int)means.size();
        int k = std::min(top_k, n);

        // sort by magnitude
        std::vector<std::pair<int, float>> indexed(n);
        for (int j = 0; j < n; j++) {
            indexed[j] = {j, means[j]};
        }
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
            [](const auto & a, const auto & b) {
                return std::abs(a.second) > std::abs(b.second);
            });

        printf("\n  Layer %d:\n", il);
        printf("    %8s  %12s\n", "idx", "value");
        printf("    %8s  %12s\n", "--------", "------------");
        for (int j = 0; j < k; j++) {
            printf("    %8d  %12.6f\n", indexed[j].first, indexed[j].second);
        }
    }
}

// ---- prompt reading ----

static std::vector<std::string> read_prompts_from_file(const std::string & path) {
    std::vector<std::string> prompts;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return prompts;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) {
            prompts.push_back(line);
        }
    }
    return prompts;
}

// ---- layer argument parsing ----

static std::vector<int> parse_layers(const std::string & s) {
    std::vector<int> layers;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        layers.push_back(std::stoi(item));
    }
    return layers;
}

// ---- main ----

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // custom args we'll parse ourselves after common_params_parse
    std::string layer_arg;
    bool all_layers = false;
    int top_k = 0;
    std::string output_dir = ".";

    // scan for our custom flags before common_params_parse
    // (common_params_parse doesn't know about --layer, --all-layers, --top-k, --output-dir)
    std::vector<char *> filtered_argv;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            layer_arg = argv[++i];
        } else if (strcmp(argv[i], "--all-layers") == 0) {
            all_layers = true;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int)filtered_argv.size();
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    if (layer_arg.empty() && !all_layers && top_k <= 0) {
        fprintf(stderr, "\nerror: specify --layer <N>, --all-layers, or --top-k <K>\n");
        print_usage(argc, argv);
        return 1;
    }

    // set up capture data
    capture_data cap;
    cap.top_k = top_k;
    cap.capture_all = all_layers;
    if (!layer_arg.empty()) {
        cap.target_layers = parse_layers(layer_arg);
    }

    // install eval callback
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cap;
    params.warmup = false;

    print_build_info();
    llama_backend_init();
    llama_numa_init(params.numa);

    // load model
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    int n_layers = llama_model_n_layer(model);
    int n_embd   = llama_model_n_embd(model);

    cap.init(n_layers, n_embd);

    LOG_INF("model: %d layers, %d embd\n", n_layers, n_embd);

    // validate layer indices
    for (int l : cap.target_layers) {
        if (l < 0 || l >= n_layers) {
            fprintf(stderr, "error: layer %d out of range (model has %d layers)\n", l, n_layers);
            llama_backend_free();
            return 1;
        }
    }

    // gather prompts
    std::vector<std::string> prompts;
    if (!params.prompt.empty()) {
        prompts.push_back(params.prompt);
    }
    if (!params.prompt_file.empty()) {
        auto file_prompts = read_prompts_from_file(params.prompt_file);
        prompts.insert(prompts.end(), file_prompts.begin(), file_prompts.end());
    }

    if (prompts.empty()) {
        fprintf(stderr, "error: no prompts specified (use -p or -f)\n");
        llama_backend_free();
        return 1;
    }

    LOG_INF("prompts: %d\n", (int)prompts.size());

    if (cap.capture_all) {
        LOG_INF("capturing: all %d layers\n", n_layers);
    } else {
        LOG_INF("capturing: layers");
        for (int l : cap.target_layers) {
            LOG_INF(" %d", l);
        }
        LOG_INF("\n");
    }

    // process each prompt
    int total_tokens = 0;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const auto & prompt = prompts[pi];
        LOG_INF("\n[%d/%d] \"%s\" (%d chars)\n", (int)pi + 1, (int)prompts.size(),
                prompt.substr(0, 60).c_str(), (int)prompt.size());

        // tokenize
        auto tokens = common_tokenize(ctx, prompt, true);
        int n_tokens = (int)tokens.size();

        if (n_tokens <= 0) {
            LOG_WRN("  skipping empty prompt\n");
            continue;
        }

        // check context size
        if (n_tokens > (int)llama_n_ctx(ctx)) {
            LOG_WRN("  prompt too long (%d tokens > %d ctx), truncating\n",
                    n_tokens, llama_n_ctx(ctx));
            tokens.resize(llama_n_ctx(ctx));
            n_tokens = (int)tokens.size();
        }

        // clear memory (KV cache) for fresh context
        llama_memory_clear(llama_get_memory(ctx), true);

        // process in batches
        int n_batch = llama_n_batch(ctx);
        for (int i = 0; i < n_tokens; i += n_batch) {
            int batch_size = std::min(n_batch, n_tokens - i);
            llama_batch batch = llama_batch_get_one(tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "  error: llama_decode failed at token %d\n", i);
                break;
            }
        }

        total_tokens += n_tokens;
        LOG_INF("  processed %d tokens\n", n_tokens);
    }

    LOG_INF("\ntotal tokens processed: %d\n", total_tokens);

    // output results
    if (top_k > 0) {
        // display mode: show top-K activations per captured layer
        LOG_INF("\ntop-%d activations by magnitude (from last prompt):\n", top_k);
        print_top_k(cap, top_k);
    } else {
        // file mode: write binary files for each captured layer
        int files_written = 0;
        for (int il = 0; il < n_layers; il++) {
            if (!cap.should_capture(il)) continue;
            if (cap.layers[il].n_tokens == 0) continue;

            char filename[256];
            snprintf(filename, sizeof(filename), "%s/activations_layer%d.bin",
                     output_dir.c_str(), il);

            if (write_activations(filename, cap.layers[il], n_embd, il)) {
                LOG_INF("  layer %2d: %d tokens -> %s (%.1f MB)\n",
                        il, cap.layers[il].n_tokens, filename,
                        (float)(cap.layers[il].tokens.size() * sizeof(float)) / (1024.0f * 1024.0f));
                files_written++;
            }
        }
        LOG_INF("\n%d file(s) written to %s\n", files_written, output_dir.c_str());
    }

    llama_backend_free();

    return 0;
}
