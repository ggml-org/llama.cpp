# Reapply Plan: amd-rdna4-egpu fork changes

This document records every change in this fork relative to upstream `ggml-org/llama.cpp`,
so they can be re-applied to a fresh upstream checkout by hand. "Reapplying" means
recreating the changes, not blindly cherry-picking: each entry states *what* changes,
*where*, and *why*, so you can re-apply it and defend it to a reviewer without this
document.

Branch base: upstream commits up to and including the 2026-08 RDNA4-era master, then the
following local commits (oldest to newest):

| commit | title |
|---|---|
| 89fd3c009 | rebase: apply RDNA4 detection commit |
| 003fb8443 | docs: add AMD RDNA4 tune notes for server and MTP |
| f4cc7e97a | feat: add RDNA4 profiling script and Vulkan memory reproducer |
| ec6a6f3d2 | perf(vulkan): larger mul_mat_id tiles for RDNA4; pipeline gated_delta_net |
| 291b13b74 | vulkan: revert gated_delta_net software pipelining, no TG gain |
| (working tree) | Transposed CONCAT fast path + its test + README updates (uncommitted) |

---

## 1. RDNA4 / gfx1201 detection + pipeline config

**Commit:** 89fd3c009. **Files:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.

- The AMD proprietary Windows driver does not expose `VK_NV_cooperative_matrix2`, so
  upstream mis-detects RDNA4 (gfx1201/gfx1200) as RDNA3.
- Detect by PCI device ID `0x755x` / `0x759x` and apply the correct pipeline config,
  in particular subgroup size 32.
- Verify at runtime with the startup line `matrix cores: KHR_coopmat`.

Re-apply: search `get_device_architecture` / device-id handling in `ggml-vulkan.cpp` and
port the gfx12 branch, keeping the coopmat pipeline selection.

## 2. eGPU / Thunderbolt memory handling

**Commits:** 89fd3c009 (+ workaround documented, runtime note in README).

Two distinct pieces:

- **`GGML_VK_IGNORE_BUFFER_MEMORY_TYPE_BITS=1`** (in `ggml_vk_create_buffer`): the AMD
  proprietary driver mis-reports `memoryTypeBits` as host-memory-only on eGPUs, so
  device-local allocations fail with `ErrorOutOfDeviceMemory` despite free VRAM. The
  fork probes for the broken condition at init and trusts the property flags for
  device-local requests.
- **Host-visible vidmem workaround (the big one):** on TB/USB4 eGPUs the driver prefers
  `DEVICE_LOCAL|HOST_VISIBLE` (BAR) memory for small device buffers; GPU access to that
  heap crosses the link (~5 GB/s). This made dense decode ~6 t/s. **No code is required**
  and none should be added upstream: the fix is the existing runtime switch
  `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` (already in `ggml_vk_create_buffer_device`),
  which forces plain `DEVICE_LOCAL` and takes decode from ~6 to ~30 t/s.

Documented in README "Host tuning" section. This is a deployment/config concern, not a
patch.

## 3. AMD RDNA4 tune notes

**Commit:** 003fb8443. **File:** `README.md`.

- Readme section "AMD AI PRO Optimisation and eGPU Compatibility" with server tuning,
  MTP guidance, build, and roadmap. Re-apply as documentation only (can be regenerated
  from this task's findings; numbers below).

## 4. RDNA4 profiling / memory reproducer tooling

**Commit:** f4cc7e97a. **Files:** `rdna4_profile.sh`, `vk_mem_repro.cpp`.

Standalone developer tooling: a shell profiler and a small Vulkan program that reproduces
the eGPU memory-type issue. Optional; re-apply only if you still need the reproducer.

## 5. RDNA4 matmul (mul_mat_id) tile sizing

**Commit:** ec6a6f3d2. **Files:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
`ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp`, `tests/test-backend-ops.cpp`.

Larger `mul_mat_id` tiles for RDNA4 (matches the cooperative-matrix tile shape) plus a
`gated_delta_net` shader pipeline change. This is the substantive RDNA4-tuned kernel work
in the branch.

## 6. Revert of gated_delta_net software pipelining

**Commit:** 291b13b74. **File:** `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp`.

The software-pipelined prefetch of token t+1 (added in ec6a6f3d2) gave **no TG win** once
the real bottleneck (host-visible vidmem, see #2) was fixed; the simple loop is
equivalent and simpler, so it was reverted. Also restored the file's trailing newline.

This revert only makes sense *together with* the perf environment workaround. If you are
re-applying upstream without the eGPU env workaround, keep or evaluate the pipelined
version separately.

## 7. Transposed CONCAT fast path (uncommitted)

**Files:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
`tests/test-backend-ops.cpp` (working tree).

### Problem
`build_conv_state` (delta-net) concats `conv_states` {3, channels} with
`transpose(qkv_mixed)` {tokens, channels} along dim 0. The generic `concat.comp`
assigns consecutive threads along dim 0 (time), so every read from the transposed src1
jumps `channels*4` bytes - a fully uncoalesced gather (~82 GB/s, ~25 ms per 512-token
ubatch).

### Fix
New helper `ggml_vk_concat_transposed_src1` in `ggml-vulkan.cpp` (defined just above
`ggml_vk_concat`, called first in `ggml_vk_concat` with `op_params[0]` = concat dim).

Gate (must ALL hold, else fall through to generic path):
- concat dim == 0
- `src0->type == src1->type == dst->type`, and type is F32 or F16 only
- dst and src0 are contiguous
- src1 is a transposed matrix: `src1->nb[1] == ts` and `src1->nb[0] > ts`
- `src0->ne[0] < 65536` (dst column offset passes as a 16-bit push constant)
- `get_misalign_bytes(...) == 0` for src0, src1, dst

When taken, it issues **two dispatches** (both pre-declared via
`ggml_pipeline_request_descriptor_sets`, and no barrier between them because they write
disjoint dst regions):
1. `pipeline_cpy_f32_f32` / `pipeline_cpy_f16_f16`: copy src0 into the first `ne00`
   columns. Push constants built from `vk_op_unary_push_constants_init(src0, src0, ne)`
   with dst strides overridden to the real dst layout (`nb* / ts`) and
   `init_pushconst_fastdiv`. Elements split 512/512/ceil as usual.
2. `pipeline_cpy_transpose_32` / `pipeline_cpy_transpose_16`: transpose src1 into the
   remaining columns. Same push-constant pattern, plus `misalign_offsets =
   src0->ne[0]` to offset the destination column. Elements are the tile grid
   `{CEIL_DIV(ne0,32), CEIL_DIV(ne1,32), ne2*ne3}`, clamped to max workgroup counts
   (copied from the existing cpy_transpose dispatch at ~line 12280).

Reuses the existing `copy_transpose.comp` shader unmodified (it already honors dst
strides and the doffset via `misalign_offsets`).

### Measurement
- pp512: 804.5 -> 814.1 t/s; pp2048: 804.5 -> 811.1; tg32: 27.7 -> 29.6 (no TG
  regression). ~+1.2% PP, net-positive. The earlier ~+4% estimate was pessimistic; the
  perf logger over-attributes the concat because it only timestamps at sync points.

### Test
`test_concat_transposed_src1` struct added to `tests/test-backend-ops.cpp` (next to
`test_concat`, before ARGSORT), building src0 {3, channels} + a transposed {tokens,
channels} and concat along dim 0. Registered for f32/f16 with {3, 10240} and token counts
{4, 512, 1024}. Passes on Vulkan against the CPU reference (validates the offset/stride
math in the transpose-into-strided-dst path).

---

## MTP / speculative decoding: when to use it

MTP (`--spec-type draft-mtp`) helps only **dense, bandwidth-bound** models on this eGPU.
Rule of thumb: if adding MTP to a greedy request does not raise decode t/s, turn it off.

### Server commands

Dense model with MTP (one per line, all in a shell with `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`):

```powershell
# dense model: enable MTP, keep draft length at default 3
llama-server.exe -m dense.gguf -ngl 99 -c 32768 --device Vulkan1 --parallel 1 --jinja --spec-type draft-mtp --spec-draft-n-max 3 --port 8080

# MoE model: leave MTP OFF (the draft head is a net loss)
llama-server.exe -m moe.gguf -ngl 99 -c 32768 --device Vulkan1 --parallel 1 --jinja --port 8080

# benchmark-only, no server flags (llama-bench has no -c/--parallel/--spec/--port)
llama-bench.exe -m dense.gguf --device Vulkan1 --spec-type draft-mtp -p 512 -n 64
```

Notes:
- `-c`, `--parallel`, `--jinja`, `--spec-type`, `--port` are **llama-server** options.
  llama-bench is a different binary and rejects them.
- `--device Vulkan1` only works when all GPUs are visible; if you set
  `GGML_VK_VISIBLE_DEVICES=1`, drop `--device` (the R9700 becomes `Vulkan0`). Use one
  mechanism, not both.
- `$env:` vars only last for the current shell; persist with
  `[Environment]::SetEnvironmentVariable("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM","1","User")`.

### Draft length

`--spec-draft-n-max` defaults to 3, which is optimal: the single MTP head drafts
recursively, so acceptance collapses past ~3 tokens and rejected drafts waste whole
decode passes. Sweep on the 27B dense model: n-max 3 / 4 / 6 / 8 -> 47.5 / 46.7 / 41.8 /
31.8 t/s. Watch `draft acceptance` and `eval time` lines in the log to confirm.

### Dense vs MoE measured numbers (R9700 eGPU)

| model | raw tg | with MTP | acceptance | verdict |
|---|---|---|---|---|
| Qwen3.6-27B (dense, Q4_K_S) | ~30 t/s | ~47.5 t/s (warm) | ~75% | use MTP |
| Qwen3.6-35B-A3B (MoE, Q4_K_S) | ~80-85 t/s | ~14-15 t/s | ~43% | do NOT use MTP |

Why: MTP verifies ~3 drafted tokens in one forward pass that reads the weights once. For
a dense model the base decode is bandwidth-bound, so batching several drafts over one
weight read is a direct win. For a MoE model only ~3B params are active per token, so
decode is already cheap; the draft head's extra weight traffic is a large net loss.

---

## Concurrency: parallel slots on the R9700 eGPU

Measured by firing `--parallel N` concurrent 300-token requests at a single
`llama-server` (`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`). The old "multi-slot drops 5x"
behavior is gone - that was caused by the host-visible vidmem bug, not by batching.

### Dense 27B Q4_K_S

| --parallel | MTP | KV | total t/s | per-stream t/s | scaling vs p1 |
|---|---|---|---|---|---|
| 1 | on | f16 | 40.0 | 40.0 | 1.00x |
| 2 | on | f16 | 39.3 | 19.7 | 0.98x |
| 2 | off | f16 | 46.0 | 23.0 | 1.15x |
| 4 | on | f16 | 74.5 | 18.6 | 1.86x |
| 4 | off | f16 | 70.6 | 17.6 | 1.77x |
| 4 | on | q8_0 | 71.3 | 17.8 | 1.78x |

### MoE 35B-A3B-UD Q4_K_S

| --parallel | MTP | KV | total t/s | per-stream t/s | scaling vs p1 |
|---|---|---|---|---|---|
| 1 | off | f16 | 60.1 | 60.1 | 1.00x |
| 2 | off | f16 | 106.4 | 53.2 | 1.77x |
| 4 | off | f16 | 130.6 | 32.6 | 2.17x |
| 4 | on | f16 | 112.1 | 28.0 | 1.86x |

### Takeaways

- Concurrency scales: dense p4 = 74.5 t/s (1.86x), MoE p4 = 130.6 t/s (2.17x). MoE wins
  more because 4 requests hit different experts.
- MTP is near-redundant under load: +5% on dense, negative on MoE. Keep MTP only for a
  single dense stream.
- KV q8_0 gives no speed here; useful only for VRAM headroom (higher `-c` or more slots).

### Recommended configs

```powershell
# dense 27B - single low-latency user
--parallel 1 --spec-type draft-mtp                         # 40 t/s/stream
# dense 27B - many concurrent users (max total throughput)
--parallel 4  # MTP optional (~+5%)                         # 74 t/s total
# MoE 35B - ALWAYS --parallel 2-4, never MTP
--parallel 4                                               # 130.6 t/s total
```

---

## Multi-GPU (Phase 2, not yet measured)

For 3-4 R9700 over separate Thunderbolt/USB4 links, the recommended topology is one
`llama-server` per GPU (each pinned with `GGML_VK_VISIBLE_DEVICES=i` or `--device`),
fronted by a round-robin load balancer, so requests scale linearly with no cross-GPU
traffic. Single-server `-sm row`/`layer` splitting is not advised across TB eGPU links
(per-matmul all-reduce over the link becomes the bottleneck) and is pointless for a
15 GB model that already fits one 32 GB GPU. Each instance should use `--parallel 2-4`
per the values above.

---

## Benchmark values for README / verification

Qwen3.6-27B Q4_K_S, R9700 eGPU, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, `--device Vulkan1`:

- raw tg (llama-bench tg64): ~30 t/s
- MTP `--spec-draft-n-max 3`: ~47.5 t/s warm, ~75% acceptance
- pp512: ~814 t/s; pp2048: ~811 t/s; pp512 @ d32768: ~603 t/s
- long-context tg32: ~25 t/s @ d32768
- ubatch optimum: 512

MTP draft-length sweep (n-max 3/4/6/8): 47.5 / 46.7 / 41.8 / 31.8 t/s -> keep default 3.
