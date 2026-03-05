#pragma once

#include "common.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>

// Online auto-tuning for flash attention kernel parameters.
//
// Three-layer strategy:
//   1. Runtime heuristic: compute initial values from GPU properties (SM count, occupancy)
//   2. Online profiling: test alternative values during inference using CUDA event timing
//   3. Persistent profiles: save converged results to disk for instant startup
//
// The profiler adds minimal overhead: one CUDA event pair per kernel launch
// during the tuning phase (typically ~50-100 iterations), then zero overhead.
//
// Environment variables:
//   GGML_CUDA_AUTOTUNE=0       — Disable online profiling (use heuristics only)
//   GGML_CUDA_AUTOTUNE_SAVE=0  — Disable saving profiles to disk
//   GGML_CUDA_AUTOTUNE_RESET=1 — Ignore saved profiles, force re-tuning

struct fattn_autotune_key {
    int device_id;
    int n_head;
    int head_dim;
    int kv_type;     // ggml_type of K cache

    bool operator==(const fattn_autotune_key & other) const {
        return device_id == other.device_id &&
               n_head    == other.n_head    &&
               head_dim  == other.head_dim  &&
               kv_type   == other.kv_type;
    }
};

struct fattn_autotune_key_hash {
    size_t operator()(const fattn_autotune_key & k) const {
        size_t h = 0;
        h ^= std::hash<int>()(k.device_id)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.n_head)     + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.head_dim)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.kv_type)    + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct fattn_autotune_state {
    // Current best configuration
    int best_parallel_blocks = 0;
    float best_time_us       = 1e9f;

    // Candidate being evaluated
    int candidate_parallel_blocks = 0;
    float candidate_time_sum_us   = 0.0f;
    int candidate_samples         = 0;

    // Tuning progress
    int iteration       = 0;
    int candidate_idx   = 0;  // Index into candidates list
    bool converged      = false;
    bool profile_loaded = false;

    // CUDA events for timing
    cudaEvent_t evt_start = nullptr;
    cudaEvent_t evt_stop  = nullptr;
    bool timing_active    = false;

    // Candidates to try (populated on first use)
    static constexpr int MAX_CANDIDATES = 8;
    int candidates[MAX_CANDIDATES] = {};
    int n_candidates = 0;

    static constexpr int WARMUP_ITERS     = 10;  // Skip first N iterations
    static constexpr int SAMPLES_PER_EVAL = 20;  // Iterations per candidate

    void init_events() {
        if (!evt_start) {
            CUDA_CHECK(cudaEventCreate(&evt_start));
            CUDA_CHECK(cudaEventCreate(&evt_stop));
        }
    }

    void destroy_events() {
        if (evt_start) {
            cudaEventDestroy(evt_start);
            cudaEventDestroy(evt_stop);
            evt_start = nullptr;
            evt_stop  = nullptr;
        }
    }

    ~fattn_autotune_state() {
        destroy_events();
    }

    // Generate candidate parallel_blocks values around the heuristic default
    void init_candidates(int heuristic_pb, int max_pb) {
        n_candidates = 0;

        // Always include the heuristic value
        candidates[n_candidates++] = heuristic_pb;

        // Try values that are 2x and 0.5x the heuristic
        int pb_half = std::max(1, heuristic_pb / 2);
        int pb_double = std::min(max_pb, heuristic_pb * 2);

        if (pb_half != heuristic_pb && n_candidates < MAX_CANDIDATES) {
            candidates[n_candidates++] = pb_half;
        }
        if (pb_double != heuristic_pb && n_candidates < MAX_CANDIDATES) {
            candidates[n_candidates++] = pb_double;
        }

        // Also try 1 (no parallelism) and max
        if (heuristic_pb != 1 && pb_half != 1 && n_candidates < MAX_CANDIDATES) {
            candidates[n_candidates++] = 1;
        }
        if (max_pb != heuristic_pb && max_pb != pb_double && n_candidates < MAX_CANDIDATES) {
            candidates[n_candidates++] = max_pb;
        }

        best_parallel_blocks = heuristic_pb;
        candidate_idx = 0;
        candidate_parallel_blocks = candidates[0];
    }
};

// ---- Persistent Profile Storage ----
//
// Profiles are saved as simple key=value text files:
//   ~/.cache/llama.cpp/gpu_profiles/<device_name>.profile
//
// Format:
//   h<heads>_d<dim>_k<kv_type>=<parallel_blocks>,<time_us>

static inline std::string fattn_profile_dir() {
    std::string dir;
    const char * cache = getenv("XDG_CACHE_HOME");
    if (cache && cache[0]) {
        dir = std::string(cache) + "/llama.cpp/gpu_profiles";
    } else {
        const char * home = getenv("HOME");
        if (!home || !home[0]) {
            return "";
        }
        dir = std::string(home) + "/.cache/llama.cpp/gpu_profiles";
    }
    return dir;
}

static inline void fattn_profile_mkdir_p(const std::string & path) {
    std::string partial;
    for (size_t i = 0; i < path.size(); i++) {
        if (path[i] == '/' || i == path.size() - 1) {
            partial = path.substr(0, i + 1);
            mkdir(partial.c_str(), 0755);
        }
    }
}

static inline std::string fattn_profile_path(const char * device_name) {
    std::string dir = fattn_profile_dir();
    if (dir.empty()) {
        return "";
    }
    // Sanitize device name for filesystem
    std::string safe_name;
    for (const char * p = device_name; *p; p++) {
        char c = *p;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
            safe_name += c;
        } else {
            safe_name += '_';
        }
    }
    return dir + "/" + safe_name + ".profile";
}

static inline std::string fattn_profile_entry_key(const fattn_autotune_key & key) {
    char buf[64];
    snprintf(buf, sizeof(buf), "h%d_d%d_k%d", key.n_head, key.head_dim, key.kv_type);
    return std::string(buf);
}

static inline bool fattn_profile_load(const char * device_name, const fattn_autotune_key & key,
                                       fattn_autotune_state & state) {
    if (getenv("GGML_CUDA_AUTOTUNE_RESET")) {
        return false;
    }
    std::string path = fattn_profile_path(device_name);
    if (path.empty()) {
        return false;
    }
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    std::string target = fattn_profile_entry_key(key);
    char line[256];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        char * eq = strchr(line, '=');
        if (!eq) {
            continue;
        }
        *eq = '\0';
        if (std::string(line) == target) {
            int pb = 0;
            float time_us = 0.0f;
            if (sscanf(eq + 1, "%d,%f", &pb, &time_us) == 2 && pb > 0) {
                state.best_parallel_blocks = pb;
                state.best_time_us = time_us;
                state.converged = true;
                state.profile_loaded = true;
                found = true;
            }
            break;
        }
    }
    fclose(f);
    return found;
}

static inline void fattn_profile_save(const char * device_name, const fattn_autotune_key & key,
                                       const fattn_autotune_state & state) {
    std::string path = fattn_profile_path(device_name);
    if (path.empty()) {
        return;
    }
    // Read existing entries
    std::string contents;
    {
        FILE * f = fopen(path.c_str(), "r");
        if (f) {
            char buf[256];
            while (fgets(buf, sizeof(buf), f)) {
                contents += buf;
            }
            fclose(f);
        }
    }
    std::string entry_key = fattn_profile_entry_key(key);
    char entry_val[64];
    snprintf(entry_val, sizeof(entry_val), "%d,%.1f", state.best_parallel_blocks, state.best_time_us);

    // Replace existing or append
    std::string new_contents;
    bool replaced = false;
    size_t pos = 0;
    while (pos < contents.size()) {
        size_t eol = contents.find('\n', pos);
        if (eol == std::string::npos) {
            eol = contents.size();
        }
        std::string line = contents.substr(pos, eol - pos);
        size_t eq = line.find('=');
        if (eq != std::string::npos && line.substr(0, eq) == entry_key) {
            new_contents += entry_key + "=" + entry_val + "\n";
            replaced = true;
        } else if (!line.empty()) {
            new_contents += line + "\n";
        }
        pos = eol + 1;
    }
    if (!replaced) {
        if (new_contents.empty()) {
            new_contents = "# Auto-tuned flash attention profiles\n";
        }
        new_contents += entry_key + "=" + entry_val + "\n";
    }

    fattn_profile_mkdir_p(fattn_profile_dir());
    std::string tmp_path = path + ".tmp";
    FILE * f = fopen(tmp_path.c_str(), "w");
    if (f) {
        fputs(new_contents.c_str(), f);
        fclose(f);
        rename(tmp_path.c_str(), path.c_str());
    }
}

// ---- Auto-tune Registry ----

class fattn_autotune_registry {
public:
    static fattn_autotune_registry & instance() {
        static fattn_autotune_registry reg;
        return reg;
    }

    fattn_autotune_state & get(const fattn_autotune_key & key) {
        std::lock_guard<std::mutex> lock(mtx);
        return states[key];
    }

    bool is_enabled() const {
        return enabled.load(std::memory_order_relaxed);
    }

    bool should_save() const {
        return save_profiles.load(std::memory_order_relaxed);
    }

private:
    fattn_autotune_registry() {
        const char * env = getenv("GGML_CUDA_AUTOTUNE");
        if (env && std::string(env) == "0") {
            enabled.store(false, std::memory_order_relaxed);
        }
        const char * env_save = getenv("GGML_CUDA_AUTOTUNE_SAVE");
        if (env_save && std::string(env_save) == "0") {
            save_profiles.store(false, std::memory_order_relaxed);
        }
    }

    std::mutex mtx;
    std::unordered_map<fattn_autotune_key, fattn_autotune_state, fattn_autotune_key_hash> states;
    std::atomic<bool> enabled{true};
    std::atomic<bool> save_profiles{true};
};

// ---- Auto-tune API ----

// Record start event before kernel launch
static inline void fattn_autotune_record_start(fattn_autotune_state & state, cudaStream_t stream) {
    if (state.converged || !fattn_autotune_registry::instance().is_enabled()) {
        return;
    }
    if (state.iteration < fattn_autotune_state::WARMUP_ITERS) {
        state.iteration++;
        return;
    }
    state.init_events();
    CUDA_CHECK(cudaEventRecord(state.evt_start, stream));
    state.timing_active = true;
}

// Record stop event after kernel launch and update tuning state.
// The key and device_name are needed for profile persistence on convergence.
static inline void fattn_autotune_record_stop(fattn_autotune_state & state, cudaStream_t stream,
                                               const fattn_autotune_key & key = {},
                                               const char * device_name = nullptr) {
    if (!state.timing_active) {
        return;
    }
    state.timing_active = false;

    CUDA_CHECK(cudaEventRecord(state.evt_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(state.evt_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, state.evt_start, state.evt_stop));

    state.candidate_time_sum_us += elapsed_ms * 1000.0f;
    state.candidate_samples++;
    state.iteration++;

    if (state.candidate_samples >= fattn_autotune_state::SAMPLES_PER_EVAL) {
        float avg_time = state.candidate_time_sum_us / state.candidate_samples;

        if (avg_time < state.best_time_us) {
            state.best_time_us = avg_time;
            state.best_parallel_blocks = state.candidate_parallel_blocks;
        }

        // Move to next candidate
        state.candidate_idx++;
        if (state.candidate_idx >= state.n_candidates) {
            // All candidates tested — converged
            state.converged = true;
            state.destroy_events();
            GGML_LOG_INFO("fattn autotune: converged to parallel_blocks=%d (%.1f us)\n",
                          state.best_parallel_blocks, state.best_time_us);

            // Save profile to disk
            if (device_name && fattn_autotune_registry::instance().should_save()) {
                fattn_profile_save(device_name, key, state);
            }
        } else {
            state.candidate_parallel_blocks = state.candidates[state.candidate_idx];
            state.candidate_time_sum_us = 0.0f;
            state.candidate_samples = 0;
        }
    }
}

// Get the parallel_blocks value to use for the current iteration.
// On first call, attempts to load a saved profile. If not found, starts online tuning.
// The device_name is used for profile persistence.
static inline int fattn_autotune_get_parallel_blocks(
        fattn_autotune_state & state, int heuristic_pb, int max_pb,
        const fattn_autotune_key & key = {}, const char * device_name = nullptr) {
    if (!fattn_autotune_registry::instance().is_enabled()) {
        return heuristic_pb;
    }

    // On first call, try to load a saved profile
    if (state.n_candidates == 0 && !state.profile_loaded && device_name) {
        if (fattn_profile_load(device_name, key, state)) {
            GGML_LOG_INFO("fattn autotune: loaded profile for %s — parallel_blocks=%d\n",
                          fattn_profile_entry_key(key).c_str(), state.best_parallel_blocks);
            return state.best_parallel_blocks;
        }
    }

    if (state.n_candidates == 0) {
        state.init_candidates(heuristic_pb, max_pb);
    }

    if (state.converged) {
        return state.best_parallel_blocks;
    }

    if (state.iteration < fattn_autotune_state::WARMUP_ITERS) {
        return heuristic_pb;
    }

    return state.candidate_parallel_blocks;
}
