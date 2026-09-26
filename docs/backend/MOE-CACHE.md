# MoE expert cache

This fork includes TheTom's Vulkan expert-cache provider, a SYCL twin of it,
and the shared CPU, backend-scheduler, and fitting integration. CPU, BLAS, and
OpenVINO do not register cache providers. Their ordinary MoE execution remains
available.

The Vulkan and SYCL providers cache CPU-resident expert weights and dispatch
supported expert matvecs to the GPU. Unsupported operations and failed cache
work fall back to the CPU `MUL_MAT_ID` implementation. Cache sessions belong to
a scheduler; weights are not persisted across process restarts.

The SYCL provider reuses the existing `ggml_sycl_mul_mat_vec_q_id()` kernel
dispatcher (`mmvq.cpp`) for the actual matvec: the cache slab is addressed
exactly like the stacked expert-weight buffer that function already expects,
with pool slot indices standing in for expert ids. It needs no shader/pipeline
compilation step and no manual staging buffer - `sycl::queue::memcpy` moves
data between host and USM device allocations directly regardless of device
topology, which is simpler than the Vulkan provider's UMA-vs-discrete split.

## Usage

Build with `GGML_VULKAN=ON` or `GGML_SYCL=ON`, then select the device reported
by `llama-server --list-devices`:

```sh
~/build-vulkan/bin/llama-server -m /path/to/model.gguf \
    --device Vulkan0 --fit on --moe-cache auto -ngl 99 -c 8192

~/build-sycl/bin/llama-server -m /path/to/model.gguf \
    --device SYCL0 --cpu-moe --moe-cache 4096 -ngl 99 -c 8192
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

`on`/`auto`/`soft` (no explicit MiB) reach the provider with `budget_mib=0`
(`arg.cpp`'s encoding for "resolve free-minus-reserve yourself"). `--fit`'s
own placement planner separately computes its own projected free-VRAM figure
to decide whether spilling routed experts to host RAM is worth it at all -
but that computed figure is never wired back into the value
`ggml_backend_sched_set_moe_cache` actually receives (confirmed by reading
`common/fit.cpp`, `common/common.cpp`, and `src/llama-context.cpp`: the
scheduler call always gets the original, still-zero `--moe-cache` value, not
fit's derived one). Fixing that disconnect is a shared fit/scheduler change
outside this backend's files.

The SYCL provider works around it locally: `session_create()` derives its own
free-minus-reserve figure independently when it receives a zero budget from
an explicit `on`/`auto`/`soft` request, so the session isn't permanently
dormant while fit believes a cache exists. This is an independent
approximation of the same free-VRAM state fit already queried moments
earlier, not a wired-through value - the two are not guaranteed identical,
though in practice VRAM state rarely shifts meaningfully between fit running
and scheduler creation. The Vulkan provider does not have this workaround and
stays dormant in the same scenario, exactly as it does today.

## Vulkan and SYCL implementation limits

Both providers share the same v1 shape:

- One selected device per session (Vulkan or SYCL, not mixed).
- Fills are synchronous and bounded per dispatch; neither provider has a
  background fill worker or predictive prefetch yet (`moe-cache-common.h`'s
  `moe_cache_device` already carries the queue/worker/inflight fields for one,
  unused by either provider - a natural v2).
- Fused SwiGLU cache dispatch is unavailable. The ordinary CPU path handles it.
- Pools are allocated lazily by expert shape and weight type. Automatic mode
  requires the shared 1 GiB slab floor; forced modes allow smaller experiments.
- Direct host-pointer writes bypass invalidation. Use backend tensor/buffer APIs
  when mutating cached weights, and obey their synchronization rules.
- Destroy schedulers and sessions before unloading their backend.
- A profitable hit rate is not guaranteed. Transfers, synchronous fills, spare
  VRAM, model shapes, and CPU bandwidth all affect performance.

SYCL-specific: the provider quantizes activation rows to Q8_1 on the CPU thread
that owns them (a plain scalar loop, matching the Vulkan provider's approach)
before uploading; this is not a hot path since rows per dispatch are bounded
by `max_batch`.

## Provider controls

The shared configuration retains its historical `GGML_CUDA_MOE_CACHE_*` names.
Vulkan and SYCL read these names too; they do not require a CUDA backend.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | 3072 | VRAM kept outside the cache |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | 512 (auto) / 1024 (forced) | Minimum expert size; lower it if `-lv 4`/debug shows experts rejected below the floor |
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
timeout 120 ~/build-sycl/bin/test-moe-cache
```

The synthetic test checks cache hits, invalidation, dispatch/collection/fill
failure fallback, multi-token operation, and session behavior. It explicitly
skips provider-specific functionality and exits 77 if no provider is available.
`test-moe-cache-fit` separately tests the host budget planner.

No throughput improvement on this fork is claimed here. TheTom's CUDA/Metal
measurements describe different implementations and hardware; see the
[pinned source documentation](https://github.com/TheTom/llama-cpp-turboquant/blob/80007e71526b2566bda62a8fc98f68c4d231139c/docs/backend/MOE-CACHE.md)
for that historical context.
