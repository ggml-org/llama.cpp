#include "llama-expert-pin.h"

#include "llama-expert-heatmap.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace llama_expert_pin {

static void madvise_range(const void * p, size_t len, bool keep) {
#if !defined(_WIN32)
    static const long page = sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) (page - 1);
    const uintptr_t b = ((uintptr_t) p + len + page - 1) & ~(uintptr_t) (page - 1);
    if (b > a) {
        madvise((void *) a, b - a, keep ? MADV_WILLNEED : MADV_DONTNEED);
    }
#else
    (void) p; (void) len; (void) keep;
#endif
}

// only hint file-backed (mmap) host tensors; DONTNEED on anonymous buffers
// would zero them. no data pointer = no pages to hint.
static bool hintable(const ggml_tensor * w) {
    if (!w || !w->data) {
        return false;
    }
    const ggml_backend_buffer_t buf = w->view_src ? w->view_src->buffer : w->buffer;
    return buf != nullptr && ggml_backend_buffer_is_host(buf);
}

static const config g_config = [] {
    config c;
    if (const char * e = getenv("LLAMA_EXPERT_PIN_PERIOD")) {
        c.period = std::max(1, atoi(e));
    }
    if (const char * e = getenv("LLAMA_EXPERT_PIN_START")) {
        c.start_tokens = std::max(0, atoi(e));
    }
    if (const char * e = getenv("LLAMA_EXPERT_PIN_DONTNEED_GPU")) {
        c.dontneed_gpu = (float) atof(e);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PIN_WILLNEED_GPU")) {
        c.willneed_gpu = (float) atof(e);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PIN_WILLNEED_COLD")) {
        c.willneed_cold = (float) atof(e);
    }
    return c;
}();

const config & get_config() {
    return g_config;
}

static int g_pct = -1;

void set_pct(int pct) {
    g_pct = pct;
}

int get_pct() {
    return g_pct;
}

bool active() {
    return getenv("LLAMA_EXPERT_PIN") != nullptr || g_pct > 0;
}

// exps tensors per layer, e.g. blk.0.ffn_gate_exps.weight
static const std::regex g_re_exps("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

static void hint_expert(const ggml_tensor * w, int expert, bool keep) {
    if (!hintable(w) || expert < 0 || expert >= (int) w->ne[2]) {
        return;
    }
    const size_t plane = ggml_nbytes(w) / (size_t) w->ne[2];
    madvise_range((const char *) w->data + (size_t) expert * plane, plane, keep);
}

static void hint_experts(const llama_model * model,
                         const llama_expert_heatmap & heatmap,
                         bool (*is_gpu_resident)(void * ud, int il, int e),
                         void * ud) {
    const int n_layers  = heatmap.n_layers;
    const int n_experts = heatmap.n_experts;

    // RAM-pressure gate: only evict cold pages when the system is genuinely
    // tight (MemAvailable under 10% of total).
    bool ram_tight = false;
    int64_t mem_total = 0, mem_avail = 0;
    FILE * f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %lld kB", &mem_total) == 1) {
                mem_total *= 1024;
            } else if (sscanf(line, "MemAvailable: %lld kB", &mem_avail) == 1) {
                mem_avail *= 1024;
            }
        }
        fclose(f);
    }
    ram_tight = mem_total > 0 && mem_avail * 10 < mem_total;

    // group the exps tensors by layer
    std::vector<std::vector<const ggml_tensor *>> per_layer(n_layers);
    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        std::smatch m;
        if (std::regex_search(name, m, g_re_exps)) {
            const int il = std::stoi(m[1].str());
            if (il >= 0 && il < n_layers) {
                per_layer[il].push_back(tensor);
            }
        }
    }

    std::vector<int> idx(n_experts);
    std::vector<float> score(n_experts);

    for (int il = 0; il < n_layers; il++) {
        if (per_layer[il].empty()) {
            continue;
        }
        for (int e = 0; e < n_experts; e++) {
            score[e] = heatmap.get_score(il, e);
            idx[e] = e;
        }

        // partition by residency, sort each by score
        std::vector<int> gpu, cold;
        gpu.reserve(n_experts);
        cold.reserve(n_experts);
        for (int e = 0; e < n_experts; e++) {
            if (is_gpu_resident && is_gpu_resident(ud, il, e)) {
                gpu.push_back(e);
            } else {
                cold.push_back(e);
            }
        }
        auto by_score = [&](int a, int b) { return score[a] > score[b]; };

        // GPU residents: drop the top fraction, warm the bottom fraction
        if (!gpu.empty()) {
            std::sort(gpu.begin(), gpu.end(), by_score);
            const int n_drop = std::min((int) gpu.size(),
                (int) (g_config.dontneed_gpu * (float) gpu.size()));
            const int n_warm = std::min((int) gpu.size(),
                (int) (g_config.willneed_gpu * (float) gpu.size()));
            for (int i = 0; i < n_drop; i++) {
                for (const ggml_tensor * w : per_layer[il]) {
                    hint_expert(w, gpu[i], false);
                }
            }
            for (int i = (int) gpu.size() - n_warm; i < (int) gpu.size(); i++) {
                for (const ggml_tensor * w : per_layer[il]) {
                    hint_expert(w, gpu[i], true);
                }
            }
        }

        // cold: warm the top fraction (most likely promoted next)
        if (!cold.empty()) {
            std::sort(cold.begin(), cold.end(), by_score);
            float cold_pct = g_config.willneed_cold;
            if (const char * e = getenv("LLAMA_EXPERT_PIN_WILLNEED_COLD")) {
                cold_pct = (float) atof(e);
            } else if (g_pct > 0) {
                cold_pct = (float) g_pct / 100.0f;
            }
            const int n_warm = std::min((int) cold.size(),
                (int) (cold_pct * (float) cold.size()));
            for (int i = 0; i < n_warm; i++) {
                for (const ggml_tensor * w : per_layer[il]) {
                    hint_expert(w, cold[i], true);
                }
            }
        }

        // RAM pressure: evict the bottom 10% of total heat that is resident
        // in the mmap pool and has not been reused for a full dwell cycle.
        if (ram_tight) {
            std::vector<int> all(n_experts);
            for (int e = 0; e < n_experts; e++) {
                all[e] = e;
            }
            std::sort(all.begin(), all.end(), by_score);
            const int n_evict = std::max(1, (int) (0.10f * (float) n_experts));
            for (int i = 0; i < n_evict; i++) {
                const int e = all[i];
                if (is_gpu_resident && is_gpu_resident(ud, il, e)) {
                    continue; // on GPU: its mmap pages are already dropped
                }
                if (heatmap.tokens_total - heatmap.last_reuse[il * n_experts + e] < 8) {
                    continue; // reused within the dwell cycle: keep it
                }
                for (const ggml_tensor * w : per_layer[il]) {
                    hint_expert(w, e, false);
                }
            }
        }
    }
}

void maybe_run(const llama_model * model,
               const llama_expert_heatmap * heatmap,
               bool (*is_gpu_resident)(void * ud, int il, int e),
               void * ud) {
    if (!model || !heatmap) {
        return;
    }
    static int64_t last_run = -1;
    if (heatmap->tokens_total < g_config.start_tokens) {
        return;
    }
    if (last_run >= 0 && heatmap->tokens_total - last_run < g_config.period) {
        return;
    }
    last_run = heatmap->tokens_total;
    hint_experts(model, *heatmap, is_gpu_resident, ud);
}

}
