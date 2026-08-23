# Vulkan performance work — progress notes

Working notes for the Vulkan optimisation effort on AMD Strix Halo (Radeon 8060S,
gfx1151, RADV / Mesa 26.0.3). Everything here was measured on one machine. Dead
ends are recorded alongside the wins, because most of the value is in not
repeating them.

Baselines are stated against **upstream `ggml-org/llama.cpp` master `95b8e33e1`**,
built from a clean worktree with the same CMake flags, not against a previous
revision of this fork.

---

## 1. Shipped: LDS stride padding for the coopmat matmul

**+13.7% prefill against mainline, on a stock unsloth K-quant.** One constant.

### The bug

`mul_mm.comp` stages its shared A/B tiles with

```glsl
#define SHMEM_STRIDE (BK / 2 + SHMEM_STRIDE_PAD)
shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];
shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];
```

`FLOAT_TYPEV2` is one dword, so **`SHMEM_STRIDE` maps 1:1 onto the 32 LDS banks**.
B is read back with a *column-major* `coopMatLoad`, which puts lane `i` on bank
`(SHMEM_STRIDE * i) % 32` — making `gcd(SHMEM_STRIDE, 32)` the conflict factor.

Upstream only pushes `constantID=12` for Intel-on-Windows. Every other device
falls through to the shader default `SHMEM_STRIDE_PAD = 4`, giving stride 20 at
BK=32, `gcd(20,32) = 4`: **a four-way bank conflict on every B fragment load.**

### The sweep

`test-backend-ops perf -o MUL_MAT`, `q4_0_rocmfp4_fast`, m=4096 n=512 k=14336.
Settled measurements (first run of any set discarded — see §5):

| PAD | stride | gcd(stride,32) | TFLOPS |
|-----|--------|----------------|--------|
| 0 | 16 | 16 | 10.2 |
| 1 | 17 | 1 | **5.6** |
| 2 | 18 | 2 | 15.2 |
| 3 | 19 | 1 | **5.6** |
| **4** (old default) | 20 | 4 | **13.2** |
| 5 | 21 | 1 | **5.6** |
| **6** (new) | 22 | 2 | **15.4** |
| 7 | 23 | 1 | **5.4** |
| 8 | 24 | 8 | 12.3 |

**The odd rows are the surprise.** They are conflict-free by the bank model and
they are *three times slower*. That is alignment, not banking: a 4-byte element at
an odd dword stride misaligns every other row, and RADV loses the wide
`ds_read_b64/b128` path for `coopMatLoad`. The rule is therefore:

> Keep the stride **even** first — then minimise `gcd(stride, 32)`.

### The fix

`ggml_vk_mul_mm_shmem_pad()` returns the pad from one place, consumed by both
`ggml_vk_mul_mm_spec` (constantID=12) and the shared-memory budget in
`ggml_vk_matmul_shmem_support`. The budget counts elements while the shader's pad
counts dwords, hence `bank_conflict_offset = 2 * pad`; these two were already
required to agree and the comment saying so is now enforced by construction.

Scoped to AMD + coopmat. Other vendors keep the shader default, because this has
only been measured on gfx1151.

### Results vs mainline `95b8e33e1`

Both binaries, same models, interleaved, idle GPU.

| model | metric | mainline | this fork | |
|---|---|---|---|---|
| unsloth Qwen3.8-27B UD-Q4_K_XL (dense) | pp512 | 270.1 | **307.2** | **+13.7%** |
| | pp2048 | 260.2 | **291.7** | **+12.1%** |
| | tg64 | 11.72 | 11.71 | — |
| Ornith-1.5-35B-A3B Q4_K_M (MoE) | pp512 | 1034.9 | 1049.9 | +1.5% |
| | pp2048 | 845.3 | **872.5** | **+3.2%** |
| | tg64 | 65.74 | 65.74 | — |

Per-type at m=4096 n=512 k=14336: q8_0 **1.18x**, q4_0 **1.17x**, q4_K 1.12x,
f16 1.09x, rocmfp4_fast 1.09x. **Not FP4-specific — a general RADV matmul win.**

MoE gains less because A3B is far less matmul-bound; its prefill is already ~1030
t/s, so the LDS path is a smaller share of the total.

Generation is unchanged on both, which is correct: decode runs mat-vec, not this
shader, and is already at the memory roof (§3).

Correctness: `test-backend-ops` MUL_MAT 1203/1203 and the full suite green.

---

## 2. Shipped: Gated DeltaNet output-projection reshape

`src/models/qwen35.cpp` reshaped to **3D** `[D, n_seq_tokens, n_seqs]` immediately
before the output projection and flattened to 2D immediately after. With a batch
axis in `ne2`, the backend dispatches `n_seqs` independent `n=n_seq_tokens`
mat-vecs over the same weight instead of one batch-wide GEMV.

Moving the flatten to *before* the matmul:

- op: `n=1 k=6144 batch=8: 48 x 333.6 us` → `n=8 k=6144: 64 x 136.0 us` (2.45x)
- B=8 decode step: 131 ms → 120 ms

| batch | before | after | |
|---|---|---|---|
| 1 | 13.93 | 13.92 | — |
| 2 | 25.25 | 25.76 | +2.0% |
| 4 | 42.81 | **45.06** | **+5.2%** |
| 8 | 60.85 | **66.05** | **+8.6%** |

B=1 being *exactly* unchanged is the correctness signature: at `n_seqs == 1` the
reshape is a genuine no-op. Same pattern fixed in `qwen35moe.cpp` and
`qwen3next.cpp`; `kimi-linear`, `bailingmoe3` and `kimi-k3` do not have it.

**Scope limit — this does not help speculative decoding.** `llama-batched-bench`'s
B is *parallel sequences* (`n_seqs = B`, `n_seq_tokens = 1`); single-stream spec
decoding is the mirror image (`n_seqs = 1`, `n_seq_tokens = draft width`), so
there is no batch axis to merge. Measured with real spec decoding, MTP and DFlash2
are flat to within noise. The win is for **concurrent multi-slot serving**.

---

## 3. Where the time actually goes

Profiled with `GGML_VK_PERF_LOGGER=1` on Qwen3.8-27B-ROCmFP4_FAST.

**Decode (B=1) is at the DRAM roof and has no kernel headroom.** One 74.4 ms step;
every large mat-vec runs at **205–213 GB/s**, i.e. 80–83% of the 256 GB/s
theoretical — the practical LPDDR5X ceiling. FFN gate/up is 29.6 ms of it, FFN
down 14.8 ms, lm_head 3.2 ms. Independently corroborated: an FP4 codebook and a
Q3_K superblock decoder reach the same bandwidth to three significant figures.

**Prefill is compute-bound and was the opportunity.** Before this work the FFN
matmul ran at ~16.5 TFLOPS against a measured 48.8 TFLOPS f16 WMMA instruction
roof — about a third. Uniform across quant types, so it was never an FP4 problem.

---

## 4. Dead ends, with evidence

Recorded so they are not re-attempted.

**int8 WMMA / cooperative-matrix MMQ.** `ggml-vulkan.cpp` detects
`coopmat_int_support` and reads it nowhere, which looks like an unfinished TODO.
It is not worth finishing. A standalone probe on this GPU measures f16 WMMA 48.8
TFLOPS and **s8 WMMA 46.9 TOPS — 0.96x, not 2x**: on RDNA3/3.5 int8 WMMA runs at
the same rate as f16, and only int4 doubles. The current VALU `dotPacked4x8` path
already achieves ~23 TOPS, and an int8 coopmat kernel paying the mandatory
per-32-K rescale through shared memory measures ~24. (Kairic Edge's published
IU4 harness independently reports 1.93x IU8 and 1.94x FP16, reproducing this
relationship.) Vulkan cannot reach IU4 regardless: `VkComponentTypeKHR` has no
4-bit entry and SPIR-V `OpTypeInt` has no 4-bit width, so it is a spec-level gap,
not a driver one — a RADV patch alone cannot express it.

**Warptile tuning.** Swept BK, BM/BN, WM/WN, WMITER, warp count, each paired
back-to-back with the baseline. **Every wave64 variant was worse** than what
upstream already ships (0.83–0.99x). The existing configuration is a local
optimum; do not re-sweep.

**Bigger register tiles.** 32-accumulator configs spill 94–126 VGPRs and collapse
to 3–7 TFLOPS. Upstream's tile is the largest that does not spill — wedged from
both sides.

**Occupancy.** Shader stats show 192 VGPRs, zero spills. Shrinking the register
tile to raise occupancy made throughput *fall* monotonically (16 acc → 14.6 TF,
8 acc → 13.9, 4 acc → 9.9). The kernel is reuse-bound, not occupancy-bound —
which is what pointed at LDS and, eventually, at §1.

**wave32 subgroups.** Worth +2.8% prefill on its own (RDNA3 WMMA is wave32-native:
48.8 TFLOPS at wave32 vs 38.1 at wave64). **Superseded and dropped** — pad+wave32
measures 294 pp512 against 315 for pad alone. Its gain was partly working around
the LDS inefficiency; once the stride is fixed, its register spilling dominates.

**Hoisting the B coopmat fragments.** The inner loop reloads `cache_b` from shared
memory for every `(cm_row, cm_col)` pair, which looks like 20 loads per 16
MulAdds. Hoisting into a per-column register array measured 16.49 vs 16.48 — the
compiler was already CSE-ing them.

---

## 5. Measurement traps on this machine

Every one of these produced a wrong conclusion at least once.

- **First-run boost clock.** The first measurement of any set runs ~15–20% fast.
  A baseline moved 15.65 → 13.19 TFLOPS across one un-paired sweep. Always pair
  each configuration back-to-back with the baseline and compare ratios, and
  discard the first run.
- **`gpu_busy_percent` reports utilisation, not residency.** It read 0% while
  74 GB of other models sat resident, competing for bandwidth and MALL. Those
  numbers were ~20% low and *reversed the sign* of the wave32 result. Check
  `/sys/class/drm/card1/device/mem_info_vram_used` as well.
- **`llama-cli` needs `-st`.** Without it, it spins forever printing `> ` at stdin
  EOF. This looks exactly like a GPU hang — 99% *system* time, GPU at 0% — and
  once wrote a 5.5 GB log into a 16 GB tmpfs.
- **Throughput benchmarks do not validate graph changes.** `llama-bench` reports
  numbers just as happily for a graph that emits garbage. Validate model-builder
  changes with an actual generation.
- **`llama-batched-bench` is not a spec-decoding proxy.** See §2.
- **`GGML_VK_PERF_LOGGER=1`** aborts when a draft model is loaded, but works fine
  with `llama-batched-bench`. Its own overhead is small: it summed to within 3%
  of the throughput implied by `llama-bench`.
- **`RADV_DEBUG=shaderstats` needs `nocache`**, or the pipeline comes from cache,
  never compiles, and prints nothing.

---

## 6. Open leads

- **Prefill is still at ~40% of the WMMA instruction roof** after §1. Tiles,
  occupancy and register blocking are all ruled out; the remaining suspects are
  the global→shared load path and the dequant into shared memory.
- **`CONCAT` in prefill**: 48 x 1004 us = 48 ms, ~3% of pp512, for a pure copy in
  the GDN layers — roughly 16x off what its byte count should cost.
- **~7 ms/step of launch-bound small ops in decode** (9.4%): `GET_ROWS` 97 x
  11.4 us, `RMS_NORM(5120,1,1,1)` 129 x 9.9 us to read 20 KB. Fusion territory.
- **GDN recurrent state traffic** is *not* a lead. At B=8 the state is 3.146 MB
  per sequence per layer, ~1.2 GB/step of write-back at ~210 GB/s. `CPY`,
  `GET_ROWS` and `GATED_DELTA_NET` are at the DRAM roof, not inefficient. The only
  saving would be eliminating the gather→compute→copy-back hop by having
  `GGML_OP_GATED_DELTA_NET` write directly into a view of `ssm_states_all` —
  maybe 7–10% at B=8, but it changes ggml op semantics for every GDN model.
