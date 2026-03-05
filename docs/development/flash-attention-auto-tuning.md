# Flash Attention Auto-Tuning

llama.cpp includes an auto-tuning system for flash attention kernel parameters that optimizes single-token generation (batch=1) performance. It uses a three-layer approach: runtime heuristics, online profiling, and persistent profiles.

## How It Works

### Layer 1: Runtime Heuristics

On every kernel launch, the system automatically computes good initial parameters based on GPU hardware properties:

- **parallel_blocks**: For batch=1, the flash attention vec kernel launches a grid of `(1, parallel_blocks, n_heads)`. If `n_heads` alone can't fill all SMs/CUs, `parallel_blocks` is increased automatically based on the GPU's SM count and occupancy query results.

- **GEMV threshold**: For AMD RDNA3 GPUs, the GEMV vs GEMM crossover point is auto-tuned based on CU count. iGPUs with fewer CUs (≤20) use a higher threshold (4) since GEMM can't saturate the matrix units, while discrete GPUs with more CUs use a lower threshold (3).

These heuristics work on any GPU (NVIDIA, AMD, any architecture) without hardcoded per-GPU constants.

### Layer 2: Online Profiling

During inference, the system tests alternative `parallel_blocks` values and measures actual kernel execution times:

1. **Warmup** (first 10 iterations): Uses heuristic values, no timing
2. **Evaluation** (20 iterations per candidate): Tests each candidate value with CUDA event timing
3. **Convergence**: After all candidates are tested (~100 total iterations), locks in the fastest value

Candidates tested are: the heuristic value, 0.5×, 2×, 1 (minimum), and the maximum allowed.

After convergence, there is **zero overhead** — no events are created or recorded.

### Layer 3: Persistent Profiles

Once the online profiler converges, the optimal parameters are saved to disk:

```
~/.cache/llama.cpp/gpu_profiles/<device_name>.profile
```

On subsequent runs, the saved profile is loaded immediately, skipping the convergence period entirely.

Profile file format (plain text):
```
# Auto-tuned flash attention profiles
h32_d128_k1=4,142.3
h8_d128_k1=16,98.7
```

Each entry encodes: `h<n_heads>_d<head_dim>_k<kv_type>=<parallel_blocks>,<time_us>`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_CUDA_AUTOTUNE` | `1` (enabled) | Set to `0` to disable online profiling entirely. Only runtime heuristics will be used. |
| `GGML_CUDA_AUTOTUNE_SAVE` | `1` (enabled) | Set to `0` to disable saving profiles to disk. Online profiling still runs but results are not persisted. |
| `GGML_CUDA_AUTOTUNE_RESET` | not set | Set to `1` to ignore saved profiles and force re-tuning from scratch. Useful after GPU driver updates or hardware changes. |

## Examples

### Default behavior (recommended)

No configuration needed. The system automatically:
1. Uses heuristics on first launch
2. Profiles during the first ~100 tokens
3. Saves the optimal config for future runs

### Disable auto-tuning (use heuristics only)

```bash
GGML_CUDA_AUTOTUNE=0 ./llama-cli -m model.gguf -p "Hello"
```

### Force re-tuning after a driver update

```bash
GGML_CUDA_AUTOTUNE_RESET=1 ./llama-cli -m model.gguf -p "Hello"
```

### Check what was tuned

Look at the log output during inference:
```
fattn autotune: loaded profile for h32_d128_k1 — parallel_blocks=4
```

Or on first run:
```
fattn autotune: converged to parallel_blocks=4 (142.3 us)
```

### View saved profiles

```bash
cat ~/.cache/llama.cpp/gpu_profiles/*.profile
```

## When Does Auto-Tuning Matter?

Auto-tuning primarily affects **single-token generation** (the decode phase), not prompt processing. It matters most when:

- The model has few KV heads (e.g., 8 heads with GQA) relative to the GPU's SM/CU count
- Using an iGPU or GPU with few compute units
- Running models with unusual head dimensions

For most setups with discrete GPUs and standard models (32+ heads), the runtime heuristic alone provides good performance and online profiling will converge quickly to the same or very similar values.

## Implementation Details

The auto-tuning infrastructure lives in:

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/fattn-autotune.cuh` | Core auto-tuning engine: state management, CUDA event timing, profile persistence |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Integration point: hooks auto-tuning into `launch_fattn()` |
| `ggml/src/ggml-cuda/mmvf.cu` | GEMV threshold auto-tuning based on CU count |

The auto-tuning state is keyed by `(device_id, n_heads, head_dim, kv_cache_type)`, so different models on the same GPU each get their own optimal configuration.
