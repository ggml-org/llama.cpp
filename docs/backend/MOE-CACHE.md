# MoE expert cache

This fork includes TheTom's Vulkan expert-cache provider and the shared CPU,
backend-scheduler, and fitting integration. CPU, BLAS, SYCL, and OpenVINO do not
register cache providers. Their ordinary MoE execution remains available.

The Vulkan provider caches CPU-resident expert weights and dispatches supported
expert matvecs to Vulkan. Unsupported operations and failed cache work fall back
to the CPU `MUL_MAT_ID` implementation. Cache sessions belong to a scheduler;
weights are not persisted across process restarts.

## Usage

Build with `GGML_VULKAN=ON`, then select a Vulkan device reported by
`llama-server --list-devices`:

```sh
~/build-vulkan/bin/llama-server -m /path/to/model.gguf \
    --device Vulkan0 --fit on --moe-cache auto -ngl 99 -c 8192
```

The cache needs canonical CPU-resident expert weights. A fully GPU-resident
model has nothing to cache. Explicit placement options remain authoritative.
Fit includes target and draft model memory before budgeting cache capacity.

| Mode | Behavior |
| --- | --- |
| `auto` | Preserve repacking unless cache-aware fitting selects CPU experts; apply the automatic slab floor |
| `on` | Disable repacking and use an automatic budget |
| `soft` | Try spare VRAM with stock placement before evicting experts |
| Positive integer | Per-device cache budget cap in MiB, without repacking |
| `off` or `0` | Disable the cache |

Look for the provider's cache activation and pool messages, not just acceptance
of `--moe-cache`. Use `-lv 4` for diagnostic detail. Missing providers, unsupported
shapes, inadequate capacity, or fully resident weights can leave caching dormant.

## Vulkan implementation limits

- One selected Vulkan device per session.
- Fills are synchronous and bounded per dispatch; this provider has no background
  fill workers or predictive prefetch.
- Fused SwiGLU cache dispatch is unavailable. The ordinary CPU path handles it.
- Pools are allocated lazily by expert shape and weight type. Automatic mode
  requires the shared 1 GiB slab floor; forced modes allow smaller experiments.
- Direct host-pointer writes bypass invalidation. Use backend tensor/buffer APIs
  when mutating cached weights, and obey their synchronization rules.
- Destroy schedulers and sessions before unloading their backend.
- A profitable hit rate is not guaranteed. Transfers, synchronous fills, spare
  VRAM, model shapes, and CPU bandwidth all affect performance.

## Provider controls

The shared configuration retains its historical `GGML_CUDA_MOE_CACHE_*` names.
Vulkan reads these names too; they do not require a CUDA backend.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | 3072 | VRAM kept outside the cache |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | 512 on Vulkan | Minimum expert size |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | 8 | Maximum eligible token batch |
| `GGML_CUDA_MOE_CACHE_INSERTS` | 8 | Bound on fills per plan |
| `GGML_CUDA_MOE_CACHE_QUEUE_MB` | 512 | Bound on fill bytes |
| `GGML_CUDA_MOE_CACHE_STATS` | 0 | Periodic statistics interval; zero means teardown only |

Explicit `--moe-cache` or `LLAMA_ARG_MOE_CACHE` overrides provider mode/budget
settings. Other shared controls may apply only to providers in TheTom's tree;
this document does not promise asynchronous, fused, or multi-device behavior.

## Validation

```sh
timeout 120 ~/build-vulkan/bin/test-moe-cache
```

The synthetic test checks cache hits, invalidation, dispatch/collection/fill
failure fallback, multi-token operation, and session behavior. It explicitly
skips provider-specific functionality and exits 77 if no provider is available.
`test-moe-cache-fit` separately tests the host budget planner.

No throughput improvement on this fork is claimed here. TheTom's CUDA/Metal
measurements describe different implementations and hardware; see the
[pinned source documentation](https://github.com/TheTom/llama-cpp-turboquant/blob/80007e71526b2566bda62a8fc98f68c4d231139c/docs/backend/MOE-CACHE.md)
for that historical context.
