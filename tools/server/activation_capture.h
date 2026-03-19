#pragma once

// Activation capture for llama-server
// Hooks into ggml's eval callback to capture l_out tensors during inference.
//
// Two modes:
//   1. Live monitoring - stores per-layer mean activations in memory
//   2. Collection - streams per-token activation vectors to a binary file (for SAE training)
//
// Author: Magdalene Sullivan <magda.sullivan@gmail.com> (HeraldAI, heraldai.org)

#include "ggml.h"
#include "ggml-backend.h"

#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cmath>

// Binary file format for collected activations:
//   Header (16 bytes):
//     magic:      "ACTV" (4 bytes)
//     version:    uint32_t = 1
//     n_embd:     uint32_t
//     layer_idx:  uint32_t
//   Records (repeated, no delimiter):
//     float32[n_embd]  --one per token
//
// Read with numpy: np.fromfile(f, dtype=np.float32, offset=16).reshape(-1, n_embd)

struct activation_capture {
    std::mutex mtx;
    bool enabled = false;
    int n_layers = 0;
    int n_embd = 0;

    // --- Live monitoring mode ---
    // Per-layer mean activations from the most recent decode
    // layer_means[il] has size n_embd (mean across all tokens in the batch)
    std::vector<std::vector<float>> layer_means;

    // Timestamp of last capture (milliseconds since epoch)
    int64_t last_capture_ms = 0;

    // Number of tokens in the last captured batch
    int last_n_tokens = 0;

    // --- Collection mode (SAE training) ---
    bool collecting = false;
    int collect_layer = -1;         // which layer to collect from (-1 = none)
    FILE * collect_file = nullptr;  // output file handle
    int64_t collect_n_tokens = 0;   // total tokens written so far

    void init(int layers, int embd) {
        std::lock_guard<std::mutex> lock(mtx);
        n_layers = layers;
        n_embd = embd;
        layer_means.clear();
        layer_means.resize(layers);
    }

    void store_layer_mean(int il, const float * data, int n_embd_actual, int n_tokens) {
        if (il < 0 || il >= n_layers) return;

        std::vector<float> mean(n_embd_actual, 0.0f);

        // Mean across all tokens for this layer
        for (int t = 0; t < n_tokens; t++) {
            for (int j = 0; j < n_embd_actual; j++) {
                mean[j] += data[t * n_embd_actual + j];
            }
        }
        if (n_tokens > 0) {
            for (int j = 0; j < n_embd_actual; j++) {
                mean[j] /= (float)n_tokens;
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        layer_means[il] = std::move(mean);
        last_n_tokens = n_tokens;

        auto now = std::chrono::system_clock::now();
        last_capture_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }

    // Write per-token vectors to the collection file (called from callback)
    void collect_tokens(int il, const float * data, int n_embd_actual, int n_tokens) {
        if (il != collect_layer || !collect_file) return;

        // Write raw float32 vectors --one per token, no framing
        size_t written = fwrite(data, sizeof(float), (size_t)n_embd_actual * n_tokens, collect_file);
        if (written == (size_t)n_embd_actual * n_tokens) {
            collect_n_tokens += n_tokens;
        }
        // Flush periodically (every ~1000 tokens) to avoid data loss
        if (collect_n_tokens % 1024 < n_tokens) {
            fflush(collect_file);
        }
    }

    bool start_collection(int layer, const char * path) {
        std::lock_guard<std::mutex> lock(mtx);
        if (collecting) return false; // already collecting

        if (layer < 0 || layer >= n_layers) return false;

        FILE * f = fopen(path, "wb");
        if (!f) return false;

        // Write header
        const char magic[4] = {'A', 'C', 'T', 'V'};
        uint32_t version = 1;
        uint32_t embd = (uint32_t)n_embd;
        uint32_t lidx = (uint32_t)layer;

        fwrite(magic, 1, 4, f);
        fwrite(&version, sizeof(uint32_t), 1, f);
        fwrite(&embd, sizeof(uint32_t), 1, f);
        fwrite(&lidx, sizeof(uint32_t), 1, f);

        collect_file = f;
        collect_layer = layer;
        collect_n_tokens = 0;
        collecting = true;
        // Also ensure capture is enabled so callback fires
        enabled = true;

        return true;
    }

    bool stop_collection() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!collecting) return false;

        if (collect_file) {
            fflush(collect_file);
            fclose(collect_file);
            collect_file = nullptr;
        }
        collecting = false;
        collect_layer = -1;
        return true;
    }
};

// Global instance
inline activation_capture g_activation_capture;

// The eval callback --called for every tensor node during graph computation
// When ask==true: return true if we want to observe this tensor (causes GPU->CPU copy)
// When ask==false: tensor data is available for reading
static bool activation_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    // We care about layer output tensors --named differently by model architecture:
    //   Standard models (qwen3, qwen2, llama, etc.): "l_out-{layer}"
    //   Newer architectures (qwen3next, qwen3.5):    "final_output-{layer}"
    const bool is_target = strncmp(t->name, "l_out", 5) == 0
                        || strncmp(t->name, "final_output", 12) == 0;

    if (ask) {
        // When disabled, return false for all tensors --zero overhead
        if (!g_activation_capture.enabled) {
            return false;
        }
        return is_target; // only request data for l_out tensors
    }

    // ask == false: tensor data has been copied to CPU for us to read
    if (!is_target) {
        return true;
    }

    // Parse layer index from tensor name: "l_out-0", "l_out-1", etc.
    int il = -1;
    const char * dash = strchr(t->name, '-');
    if (dash) {
        il = atoi(dash + 1);
    }
    if (il < 0) {
        return true;
    }

    if (t->type != GGML_TYPE_F32) {
        return true;
    }

    // t is [n_embd, n_tokens] --copy data from backend (GPU) to CPU buffer
    int n_embd   = (int)t->ne[0];
    int n_tokens = (int)t->ne[1];
    size_t n_bytes = (size_t)n_embd * n_tokens * sizeof(float);

    std::vector<float> cpu_buf(n_embd * n_tokens);
    ggml_backend_tensor_get(t, cpu_buf.data(), 0, n_bytes);

    // Always update live monitoring means
    g_activation_capture.store_layer_mean(il, cpu_buf.data(), n_embd, n_tokens);

    // If collecting, also stream per-token vectors to file
    if (g_activation_capture.collecting) {
        g_activation_capture.collect_tokens(il, cpu_buf.data(), n_embd, n_tokens);
    }

    return true;
}
