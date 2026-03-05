# Plan: Auto-Tuning KV Cache Attention for Batch=1

## Three-Layer Auto-Tuning Strategy

### Layer 1: Runtime Heuristic (immediate, zero overhead)
Use GPU properties (CU count, occupancy) at kernel launch time to compute good initial parameters. Works out of the box on first run.

### Layer 2: Online Profiling (convergence over ~50-100 tokens)
During inference, measure actual kernel execution times and dynamically adjust tunable parameters. Uses lightweight CUDA/HIP event timing on the hot-path kernels, comparing current settings against alternatives every N iterations.

### Layer 3: Persistent Profiles (saved to disk)
Once the online profiler converges, save the optimal parameters to a per-GPU profile file (`~/.cache/llama.cpp/gpu_profiles/<device_name>.json`). On subsequent runs, load the profile to skip the convergence period.

---

## Implementation

### Step 1: Runtime heuristic for parallel_blocks
**File:** `ggml/src/ggml-cuda/fattn-common.cuh` (~line 940-976)

After the existing wave-efficiency heuristic for `parallel_blocks`, add a batch=1 auto-tuning step:

```cpp
// After existing parallel_blocks calculation (line ~966):

// Auto-tune for batch=1: ensure enough blocks to fill all SMs.
// The occupancy query tells us how many blocks can run per SM.
// Total desired blocks = nsm * max_blocks_per_sm.
// Total available from heads = ntiles_z_gqa * K->ne[2] * Q->ne[3].
// If heads alone can't fill the GPU, increase parallel_blocks.
if (ntiles_x == 1) { // batch=1
    const int total_z = ntiles_z_gqa * K->ne[2] * Q->ne[3];
    const int target_total = nsm * max_blocks_per_sm;
    if (total_z * parallel_blocks < target_total) {
        const int target_pb = (target_total + total_z - 1) / total_z;
        parallel_blocks = std::min(target_pb, ntiles_KQ);
    }
}
```

### Step 2: Runtime heuristic for GEMV threshold
**File:** `ggml/src/ggml-cuda/mmvf.cu` (~line 826)

Auto-tune the GEMV vs GEMM crossover based on CU count:

```cpp
// In the AMD F16 case:
if (GGML_CUDA_CC_IS_AMD(cc)) {
    if (fp16_mma_hardware_available(cc)) {
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        // Fewer CUs = GEMV stays competitive at higher batch sizes
        const int threshold = nsm <= 20 ? 4 : (nsm <= 40 ? 3 : 2);
        return ne11 <= threshold;
    }
    return ne11 <= 8;
}
```

### Step 3: Online profiling infrastructure
**New file:** `ggml/src/ggml-cuda/fattn-autotune.cuh`

A lightweight auto-tuner that:
1. Maintains a small set of tunable parameters and their current values
2. Periodically tests alternative values using CUDA event timing
3. Selects the best-performing configuration after a warmup period

```cpp
struct fattn_autotune {
    // Tunable parameters
    int parallel_blocks;    // Current best parallel_blocks for vec kernel
    int gemv_threshold;     // Current best GEMV vs GEMM crossover

    // Profiling state
    int iteration;          // Total iterations since start
    int warmup_iters;       // Iterations before starting to tune (default: 10)
    int eval_iters;         // Iterations per candidate evaluation (default: 20)
    int converged;          // Whether tuning has converged

    // Timing
    float best_time_us;     // Best observed kernel time
    int candidate_idx;      // Current candidate being evaluated
    float candidate_time_us; // Accumulated time for current candidate

    // Profile key (for persistence)
    char device_name[128];
    int n_heads;
    int head_dim;
    int kv_type;            // Quantization type of KV cache
};

// Per-device, per-model-config auto-tune state
// Key: (device_id, n_heads, head_dim, kv_quant_type)
static thread_local std::unordered_map<uint64_t, fattn_autotune> autotune_state;
```

**Integration point in `launch_fattn`:**
```cpp
// Before kernel launch:
auto & tune = get_autotune_state(id, Q, K);
if (!tune.converged) {
    // Use candidate parallel_blocks instead of heuristic
    parallel_blocks = tune.get_current_candidate();
}

// After kernel launch + sync:
if (!tune.converged) {
    tune.record_timing(event_elapsed_ms);
}
```

### Step 4: Persistent profiles
**New file:** `ggml/src/ggml-cuda/fattn-profile.cuh`

Save/load converged tuning results:

```cpp
// Profile file: ~/.cache/llama.cpp/gpu_profiles/<device_name>_<hash>.json
// {
//   "device": "gfx1150",
//   "profiles": {
//     "fattn_vec_h128_kv_f16_heads32": {
//       "parallel_blocks": 4,
//       "converged_at_iter": 87,
//       "best_time_us": 142.3
//     },
//     "gemv_f16": {
//       "threshold": 4
//     }
//   }
// }

bool fattn_profile_load(const char * device_name, fattn_autotune & tune);
void fattn_profile_save(const char * device_name, const fattn_autotune & tune);
```

**Load at init:**
```cpp
// In ggml_cuda_init() or first kernel launch:
if (fattn_profile_load(device_name, tune)) {
    tune.converged = true;  // Skip online tuning
    GGML_LOG_INFO("Loaded GPU profile for %s\n", device_name);
}
```

**Save on convergence:**
```cpp
// In autotune update:
if (just_converged) {
    fattn_profile_save(device_name, tune);
    GGML_LOG_INFO("Saved GPU profile: parallel_blocks=%d, time=%.1f us\n",
                  tune.parallel_blocks, tune.best_time_us);
}
```

### Step 5: Environment variable controls

```
GGML_CUDA_AUTOTUNE=0          # Disable all auto-tuning (use heuristics only)
GGML_CUDA_AUTOTUNE=1          # Enable online profiling (default)
GGML_CUDA_AUTOTUNE_SAVE=1     # Save profiles to disk (default: 1)
GGML_CUDA_AUTOTUNE_RESET=1    # Force re-tuning, ignore saved profiles
```

---

## Files Modified/Created
1. `ggml/src/ggml-cuda/fattn-common.cuh` — Heuristic parallel_blocks + autotune integration
2. `ggml/src/ggml-cuda/mmvf.cu` — Heuristic GEMV threshold
3. `ggml/src/ggml-cuda/fattn-autotune.cuh` — **NEW** Online profiling engine
4. `ggml/src/ggml-cuda/fattn-profile.cuh` — **NEW** Persistent profile save/load

## Implementation Order
1. **Step 1-2 first** (runtime heuristics) — immediate value, minimal code
2. **Step 3 next** (online profiling) — validates and improves heuristics
3. **Step 4-5 last** (persistence + controls) — quality of life

## Risk Assessment
- Layer 1 (heuristics): Zero risk — only increases parallel_blocks, never decreases
- Layer 2 (online profiling): Low risk — worst case uses heuristic defaults during warmup
- Layer 3 (persistence): Low risk — file I/O only at init/convergence, not in hot path
- All layers independently useful — can ship Step 1-2 alone
