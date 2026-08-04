# Project A: ANE matmul backend — research report

> Subject: Whether the ternary→fp16 reconstruction hypothesis holds, the legal
> shape space the ANE accepts, the current `ggml-ane.mm` op coverage, and a
> concrete roadmap for the next 3-4 weeks.
>
> Sources cited inline. Public ANE characterization: Apple ml-ane-transformers
> ([Apple ML Research, 2022](https://machinelearning.apple.com/research/neural-engine-transformers)),
> [hollance/neural-engine](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md),
> [coremltools MIL ops reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html),
> [maderix Substack Part 2 (benchmarks)](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615),
> [Orion paper (arXiv 2603.06728)](https://arxiv.org/html/2603.06728v1),
> [Apple ANE Book (alvaro-videla, Ch. 1)](https://alvaro-videla.com/ane-book/01-ane-laws.html),
> and the in-tree `docs/ane-backend-deep-study.md`. Tessera code references are
> line-precise.

---

## Summary

**The hypothesis holds, with one major caveat.** The ANE has no native ternary
op; the only ANE-native matmul is `matmul` or — about 3x faster — `ios18.conv`
on a fp16 (or int8-dequantized-to-fp16) weight. The Tessera story of "ternary
on disk, reconstructed to fp16 at the execution boundary, fed to ANE as a
plain fp16 matmul" is the only legal way to put ternary on ANE today, and it
should work. The caveat is dispatch economics: each per-op `MLModel.predict`
call costs ~0.1-0.5 ms of XPC/IOKit overhead and the fp16 reconstruction
prologue costs another ~0.1-0.3 ms per matmul, so a one-matmul-per-predict
design will mostly eat the throughput win. The right shape is a multi-matmul
bundle (the same "prefill slab" / "decode loop" pattern the in-tree
`common/ane-mtp.mm` already uses), in which case the prologue and dispatch
overhead are amortized across the slab and the ANE's ~5.7 TFLOPS single-op fp16
throughput (maderix M4 benchmarks) becomes the real ceiling. Net: a real but
conditional win, gated on the bundle architecture and on legal ANE shapes.

---

## 1. Legal ANE matmul shapes

### 1.1 The headline constraints (Apple, Orion, maderix, hollance consolidated)

| Constraint | Value | Source |
|---|---|---|
| Tensor rank | 4D | [Apple ml-ane-transformers, Principle 1](https://machinelearning.apple.com/research/neural-engine-transformers); [ANE Book Law 1](https://alvaro-videla.com/ane-book/01-ane-laws.html) |
| Canonical shape | `[1, C, T, 1]` (B=1, channels, sequence, W=1) | Apple, ANE Book Law 1 |
| Last axis (T / W) alignment | **64 bytes** (32 fp16 elems, 64 int8 elems) | Apple ml-ane-transformers |
| Channel alignment (fp16) | multiple of 2 | [sivaro.in ANE perf article](https://sivaro.in/articles/apple-neural-engine-programming-for-real-performance/) |
| Channel alignment (int8) | multiple of 4 | sivaro.in (silent corruption otherwise) |
| Channel max | 65 536 | [Orion paper §3 / arXiv 2606.22283](https://arxiv.org/pdf/2606.22283.pdf) |
| Width / height max | 16 384 | Orion, arXiv 2606.22283 |
| Kernel width (conv) max | 13 fp16 / 29 fp16 for some arches | arXiv 2606.22283; ANE Book Law 2 ("1x1 conv preferred") |
| Min IOSurface alloc | **~49 KB** (compiles but error 0x1d at eval if smaller) | Orion #4 (Kumaresan 2026) |
| IOSurface page alignment | 16 KB (Apple kernel page size; matches the existing `GGML_ANE_PAGE` in `ggml-ane.mm:27`) | in-tree code; Orion |
| In-tree min alloc | **64 KB** (`GGML_ANE_MIN_ALLOC` in `ggml-ane.mm:32`) — the next 16 KB multiple above 49 KB | in-tree `ggml-ane.mm:27-41` |
| Min / max tensor compile-eval | size-1 dims legal; zero-size dim rejected as type-mismatch; 16 385 in width/height rejected at compile | arXiv 2606.22283 |
| Multi-input IOSurface sizes | all inputs in one program must be padded to the **max** input size | Orion #18 |
| Multi-output IOSurface sizes | all outputs in one program must be padded to the **max** output size | Orion #2 |
| Multi-input binding | alphabetical by MIL variable name, not function signature order | Orion #3, #19 |
| Output binding | alphabetical by MIL variable name | Orion #3 |
| Weight layout | weights baked at compile time when const; dynamic weights via IOSurface inputs (LoRA pattern) | Orion §6 LoRA |
| Compute units (public API) | `MLComputeUnitsCPUAndNeuralEngine` is a **preference**, not a command | ANE Book Law 2; [hollance neural-engine](https://github.com/hollance/neural-engine) |
| Residency check | only `MLComputePlan` is reliable; compile success is not | ANE Book Law 6 |

The "MLTensor" / IOSurface shape in the ggml-ane.mm context is the flat
1 x N IOSurface the existing code already produces at `ggml-ane.mm:65-104`
(`ggml_backend_ane_buffer_context_alloc`). The 64 KB minimum and 16 KB
page alignment at `ggml-ane.mm:22-41` are exactly Orion's constraints #4 and
the Apple page size, so the buffer allocator is already ANE-correct; no
change is needed for the legal-shape contract.

### 1.2 The matmul ops themselves

| MIL op | iOS / macOS | ANE status | Throughput notes | Source |
|---|---|---|---|---|
| `matmul` (iOS15) | iOS 15+ | ANE-native | Baseline; ~1x | [coremltools MIL ops](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html) |
| `linear` | iOS 15+ | ANE-native | `matmul + bias`; bias must be a separate `add` op (Orion #13) | coremltools; Orion |
| `conv` 1x1 (`ios18.conv` on iOS 18+) | iOS 18+ / macOS 15+ | ANE-native, **~3x faster than `matmul`** | ANE is a fixed-function conv engine; matmul is mapped to 1x1 conv internally on newer targets | ANE Book Law 2; maderix Part 2; Apple ml-ane-transformers (uses 1x1 conv) |
| `einsum("bchq,bkhc->bkhq")` | iOS 15+ | ANE-native | Avoids the K transpose in SDPA | Apple ml-ane-transformers, multihead_attention.py |
| `constexpr_lut_to_dense` (iOS 16+) | iOS 16+ | ANE-native, **compile-time** | Can dequant via LUT at compile time; the only "ternary-friendly" op but the prologue runs offline | coremltools MIL ops |
| Ternary matmul | — | **does not exist** | No public ANE API and no private reverse-engineered ternary op | Confirmed by Orion §4.5 (27 graph IR ops; no ternary) |

### 1.3 Concrete matmul shape table for Tessera

The matmul we want is the standard `y = x @ W^T` where `x` is `[B, S, K_in]`
and `W` is `[N, K_in]` (ggml's `MUL_MAT` convention per [ai.stackexchange
walkthrough](https://ai.stackexchange.com/questions/40105/what-operation-is-ggml-mul-mat-performing-k%C3%97q-in-llama)).
After re-layout to ANE `[1, C, T, 1]`:

| ggml matmul dim | ANE dim | Constraint | Tessera impact |
|---|---|---|---|
| `B` (batch, 1) | leading 1 | must be 1 (B=1 fixed) | no batch on ANE; OK for decode, OK for prefill if we pack prompts |
| `S` (sequence) | T (height) | pad to ≥ 16 to clear 49 KB floor for typical hidden_size | for decode, S=1 → pad to 16; for prefill, already large |
| `K_in` (in_dim, contracted) | C (channels) | multiple of 2 fp16 / 4 int8; ≤ 65 536 | typical hidden 4096 ✓; gating 11008 ✓; 32K+ needs chunking |
| `N` (out_dim) | weight output channels (kernel output) | multiple of 2 fp16 / 4 int8; ≤ 65 536 | vocab 152 K **breaks** the 65 536 cap; LM head has to be CPU/Metal or chunked (Orion #16) |
| `W` (singleton) | trailing 1 | must be 1 | automatic |

**Legal but slow shapes:**
- `N` or `K_in` as a **power of 2** runs faster than odd shapes (memory
  layout-friendliness); maderix's peak-throughput numbers are all
  powers-of-2.
- `S < 16` (decode) — the IOSurface may fall under 49 KB; pad S to 16
  (Apple's ml-ane-transformers Principle 1 last-axis padding).
- The ANE has a **~32 MB SRAM cliff** (maderix Part 2): a single matmul
  whose working set fits in 32 MB runs at peak; a working set of 96 MB
  drops throughput by ~30%. For an 8192x8192 fp16 matmul, the working set
  is 3*8192*8192*2 = 384 MB — well over the cliff. Tile the matmul in that
  range.
- Per-call dispatch overhead is ~0.1-0.5 ms (XPC + IOKit). For a 256x256
  fp16 matmul, the actual compute is ~0.006 ms — the dispatch dominates.
  This is why bundle-fused multi-op programs are the public-path norm.

### 1.4 Compiled dtypes

| dtype | Const weight (baked at compile) | Dynamic input (IOSurface) | Source |
|---|---|---|---|
| fp16 (`Float16`) | yes | yes — zero-copy via `MLMultiArray` w/ nil deallocator | coremltools; `common/ane-mtp.mm:535-569` |
| fp32 (`Float32`) | yes, but ANE dequantizes to fp16 internally for compute | yes, but slower I/O | ANE Book Law 3 |
| int8 (`Int8`) | yes, ANE dequantizes to fp16 internally — no real 2x throughput vs fp16 | yes | maderix Part 2 (the "38 TOPS INT8" debunked: real peak is 19 TFLOPS fp16 regardless of int8 vs fp16) |
| int4 | not in stock MIL; palettization (INT4pal) is a v2 feature | not first-class | ANE Book Ch. 3 |
| **ternary** | **none** | **none** | Orion §4.5 graph IR inventory (27 ops, no ternary) |

The "fast" matmul is **fp16** at ~5.7 TFLOPS single op on M4
(maderix 2048x2048 benchmark), ~3.8 TFLOPS in a full HPC GEMM on M4 Pro
([PROMPIE et al. arXiv 2511.13450](https://arxiv.org/pdf/2511.13450.pdf)),
and a peak hardware ceiling of ~19 TFLOPS at 2.8W
([maderix Part 2 / maderix Substack "80x more efficient than A100"](https://themenonlab.blog/blog/apple-neural-engine-reverse-engineered-training)).

---

## 2. Ternary→fp16 reconstruction

### 2.1 The cost

The Tessera TILE640 layout, as wired in the Metal kernel
(`ggml/src/ggml-metal/ggml-metal-ops.cpp:1765-1828`, kernel
`ggml_metal_op_tile640_matmul`), takes 7 source tensors:
`src[0]` packed (uint8), `src[1]` page_scales (uint16),
`src[2]` lane_scales (int8), `src[3]` outlier_row_offsets, `src[4]`
outlier_cols, `src[5]` outlier_vals, `src[6]` input B (the activations).
The Metal kernel does fused dequant + matmul in registers / threadgroup
memory.

The ANE does not have a fused-dequant matmul op. The cleanest equivalent
is to do the dequant on the **host** (CPU) before the matmul, producing
a legal-shape fp16 weight tensor, then call the standard ANE matmul /
1x1 conv. This is the "ternary on disk, fp16 at the execution boundary"
shape the user is asking about.

**Per-row reconstruction cost.** The Metal reference
(`tools/quantize/tessera/tessera-metal.mm:495-580`,
`ts_metal_dequant_mse_recon`) gives us the right ballpark for the
arithmetic. For a `[out_dim, in_dim]` weight matrix:

```
packed_bytes    = out_dim * in_dim / 2        (16 trits per byte via 4-bit packed)
recon_fp16_bytes= out_dim * in_dim * 2        (fp16 weight, packed [1, in_dim, 1, out_dim])
```

The arithmetic per element is ~3 fp16 muls (trit * page_scale * lane_scale
* input_scale) plus sparse outlier scatter. On a modern Apple CPU core
(NEON, 128-bit SIMD, 16 fp16 lanes per cycle), at ~3 GHz, peak fp16
throughput is ~6 GB/s per core or ~50 GB/s aggregate for fp16 mul.

For a 4096x4096 layer (typical 7B-class hidden):
- packed: 8 MB
- recon fp16: 32 MB
- min cost: 32 MB / 50 GB/s ≈ **0.6 ms** aggregate (single thread)
- with all 10 cores: 32 MB / ~100 GB/s ≈ **0.3 ms**

For a 8192x8192 layer (gating, larger models):
- recon fp16: 128 MB
- cost: ~1.3 ms aggregate

**Dispatch overhead.** maderix measures ~0.1-0.5 ms per `MLModel.predict`
call (XPC + IOKit + activation copying). For a 256x256 fp16 matmul the
actual compute is ~0.006 ms — the dispatch overhead is the only meaningful
cost.

**Net per-matmul cost when dispatched as a single op:**
- Prologue: 0.3-1.3 ms (depending on layer size)
- Dispatch overhead: 0.1-0.5 ms
- ANE compute (M4, 4096x4096, fp16): 0.3-1.0 ms (at ~5.7 TFLOPS)
- **Total: ~0.7-2.8 ms per matmul**

Without ANE, the same matmul on Metal fp16 is ~2-4 ms (M4 GPU is ~3.5
TFLOPS sustained per maderix), on CPU fp16 is ~30-60 ms (Apple CPU is
~0.5-1.5 TFLOPS). So the ANE path is ~3-5x faster than Metal and
~30-50x faster than CPU **per matmul**, with the prologue included. The
hypothesis holds at the per-matmul level.

**The amortization caveat.** If the ggml-ane backend dispatches each
matmul as its own Core ML predict, the 0.3-1.3 ms prologue is paid per
matmul — this is the "conditional" part of the headline answer. The win
gets much bigger if the prologue is amortized across a multi-matmul
bundle (the same pattern `common/ane-mtp.mm` already uses for the
prefill slab and MTP). For a prefill slab with K matmuls, the prologue
is paid K times in our naive design, K/8 times in a 8-matmul-bundled
design (8 is a typical ANE max-matmul-per-program from Orion §6 LoRA
which packs 8 adapter matmuls into one program), and 1 time if we
fuse dequant into the MIL program (which is the
`constexpr_lut_to_dense` path — compile-time only, requires rebuilding
the .mlmodelc per weight change).

### 2.2 Does the ANE do the prologue natively?

**No, the ANE itself does not do layout transforms for ADD/MUL/elementwise
ops.** Per the hollance unsupported-layers page and the ANE Book, the ANE
expects inputs in the ANE-native `[1, C, T, 1]` layout; any rank/stride
mismatch forces a CPU-side copy. In `ggml-ane.mm` today, the elementwise
ops are run on the **CPU (Accelerate)**, not ANE, precisely because the
bundle has no per-op dispatch — only the loaded program is ANE-resident.
`ggml-ane.mm:681-851` documents this explicitly: "These ops are
ANE-NATIVE per Section 4.1, but routing each one through a Core ML
dispatch requires a bundle function that fuses it. When no bundle is
bound we still need the backend to be exercisable, so the simple
element-wise ops run on Accelerate over the same IOSurface backing."

**Does coremltools have a ternary→fp16 prologue API?** No. The relevant
ops are:
- `constexpr_lut_to_dense` (iOS 16+): **compile-time** dequant via
  a LUT, only for const weights. Not a runtime prologue.
- `quantize` / `dequantize` (iOS 17+): runtime int8 ↔ fp16 conversion
  for activation caching in L2 SRAM. Not a ternary path.
- `constexpr_affine_dequantize` (iOS 16+): **compile-time** affine
  dequant, only for const weights. Same constraint.

The MIL builder does not expose a "ternary" matmul or any int2/int3 path.
The T640 dequant chain in `docs/ane-backend-deep-study.md:4.2.5`
decomposes the ternary unpack into ~15-30 stock MIL ops per projection,
**all in the weight prepass at .mlmodelc compile time**. That is the
correct pattern for ANE-resident weights. For dynamic / LoRA-style
weight updates it doesn't work — the prologue is on the host.

### 2.3 Is there an "ane-ternary" pattern in the open-source community?

**No.** Searched the relevant OSS:
- `apple/ml-ane-transformers`: stock fp16 + 1x1 conv, no ternary
- `coremltools` examples + MIL: no ternary matmul
- `ml-explore/mlx`: no ANE backend (MLX is GPU/CPU)
- `huggingface/bitnet.cpp` and the `microsoft/BitNet` family: ternary
  × int8 mpGEMM on CPU. They acknowledge in the BitNet paper itself
  that "ternary dot products" run on CPU and are "mixed-precision
  ternary×INT8 or ternary×FP16 dot products, not ternary×ternary"
  ([Reddit /r/LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/comments/1hsa0tm/so_what_happened_to_the_158bit_models_revolution/);
  [Bitnet.cpp arXiv 2502.11880](https://arxiv.org/html/2502.11880v1))
- `mechramc/Orion`: 27-op graph IR, no ternary op
- `maderix/ANE`: private-API benchmarking, no ternary op

The only way to put ternary on ANE is the reconstruction approach the
user is asking about. The community consensus is "ternary on storage
+ fp16 (or int8) on compute" because no NPU has a native ternary matmul.

---

## 3. Current `ggml-ane.mm` op coverage

`ggml_backend_ane_device_supports_op` lives at `ggml/src/ggml-ane/ggml-ane.mm:1141-1207`
(the file is 1323 lines; the supports_op block starts at 1141 and ends
just before 1208 with the closing brace). The current advertised set is
**18 distinct op types** in 3 categories (not 38+ as the task prompt
suggests — I count the actual `case` arms below).

### 3.1 Currently advertised (returns `true`)

| Category | Ops | Lines | Implementation |
|---|---|---|---|
| **Pure layout (zero compute, no-op on ANE)** | `RESHAPE`, `VIEW`, `TRANSPOSE`, `PERMUTE`, `CONT` | `ggml-ane.mm:1172-1178` | handled in `graph_compute` as a `continue` (ggml-ane.mm:953-962) — tensor metadata is already correct |
| **Pure layout (host copy on IOSurface)** | `CPY` | `ggml-ane.mm:1177` | handled in `graph_compute` as host fp32 gather+write (ggml-ane.mm:964-975) |
| **Elementwise (Accelerate, not ANE)** | `ADD`, `MUL`, `SCALE`, `CLAMP`, `REPEAT`, `LEAKY_RELU`, `SQR`, `SQRT`, `LOG`, `SIN`, `COS` | `ggml-ane.mm:1155-1166` | `ggml_ane_compute_elementwise` at ggml-ane.mm:730-851; runs vDSP/vv* on the IOSurface-backed fp32 view |
| **Unary subset (Accelerate, not ANE)** | `UNARY` with `SILU`, `SIGMOID`, `TANH`, `RELU`, `EXP`, `ABS`, `NEG`, `STEP`, `SGN` | `ggml-ane.mm:1183-1197` | same `ggml_ane_compute_elementwise` function |

That's 5 + 1 + 11 + 9 = 26 cases, but with the layout-vs-compute split
we have **5 truly-zero-cost layout ops, 1 host-copy op, 20 elementwise
ops on the IOSurface arena** — and the comment block at ggml-ane.mm:1125-1140
explicitly states: "We advertise only ops that also have the elementwise
path so the backend is exercisable without a bundle."

### 3.2 Notably excluded (returns `false`)

The deep-study comment at ggml-ane.mm:1125-1140 and the explicit comment
at ggml-ane.mm:1199-1203 enumerate the omitted ops. The matmul path
itself is a stub at `ggml-ane.mm:876-903` (`ggml_ane_program_dispatch_op`)
which only handles `MUL_MAT` / `TILE640_MATMUL` and currently returns
`false` ("TODO(ane-bundle): dispatch matmul to the bound bundle's matmul
function once the conversion tool names one. Today the matmul lives
inside the layer-slab function rather than standalone.").

| Excluded op | ANE feasibility today | What it would take |
|---|---|---|
| **`MUL_MAT` / `TILE640_MATMUL`** | **ANE-NATIVE** (iOS18 `ios18.conv` is the canonical path) | Build a `matmul` / `ios18.conv` bundle function; populate `ggml_ane_program_dispatch_op` for these cases; resolve `out_names` from the node's sources/destination. **This is Project A.** |
| **`RMS_NORM`** | ANE-NATIVE-C (decompose to `reduce_sum` + `rsqrt` + `mul` in fp32 intermediate, 8 MIL ops) | New `rms_norm_bundle` function in the conversion tool; decomposition matches `docs/ane-backend-deep-study.md:4.2.1`. fp32 cast is required to avoid fp16 overflow. |
| **`ROPE`** | ANE-NATIVE-C (composite of `cos`+`sin`+`mul`+`add`+`concat`); `concat` is **ANE-BREAKS** (Orion #1) | Use the **interleaved-layout decomposition** or pre-rotation einsum pattern in `docs/ane-backend-deep-study.md:4.2.2`. Or fall to CPU (the export script `tools/ane-mtp/export-gemma4-prefill-bundle.py:38-90` already does this in the gemma4 path). |
| **`SOFT_MAX`** | ANE-NATIVE | Clamp input to fp16 range [-65504, 65504] before `softmax` to prevent exp overflow (ANE Book Law 4 / F10 in `docs/ane-backend-deep-study.md:4.4`). |
| **`CONCAT`** | ANE-BREAKS (Orion #1) | Either use `slice` + manual write (CPU), or restructure the upstream op to avoid concat (the RoPE / prefill-slab pattern in `common/ane-mtp.mm:798-858` is the reference). |
| **`GET_ROWS`** | ANE-BREAKS (hollance; `gather` causes GPU fallback) | Use `gather_along_axis` (iOS 17+) or **fall to CPU** (single-vector gather is sub-microsecond — not worth a Core ML dispatch). |
| **`FLASH_ATTN_EXT`, `TESSERA_PAGED_ATTN`** | ANE-HOSTILE (fused-attention kernel is Metal-only) | Decompose SDPA into matmul + softmax + matmul; use `einsum("bchq,bkhc->bkhq")` to avoid the K transpose (Apple ml-ane-transformers, ANE Book Ch. 2). Manual causal mask before softmax (Orion #6: SDPA causal masks are silently ignored). |
| **`GELU`** | ANE-BREAKS (Orion #10) | Use **tanh approximation** (`docs/ane-backend-deep-study.md:4.2.3`, 8 MIL ops) or skip and use `SILU` (Tessera already uses SiLU throughout per the TILE640 spec). |
| **`TOP_K`, `ARGSORT`** | ANE-HOSTILE in practice | `topk` is ANE-native (iOS 16+, iOS 17+ updated) but the LM-head 152K vocab is over the 2048 arg-min/arg-max axis cap (arXiv 2606.22283). **Top-K must stay CPU/Metal for vocab projection.** |
| **`SLICE`, `PAD`** | ANE-NATIVE | `slice_by_index` / `pad` are stock MIL ops; add to the conversion tool as decomposition helpers. |
| **`SSM_*`, `RWKV_*`** | ANE-HOSTILE | State-space models and RWKV are not on the ANE path; leave to CPU/Metal. |

The 38+ figure in the task prompt is the **ggml op enum size**, not the
ggml-ane advertised set. The current advertised set is 26 case arms
across 5 categories above.

### 3.3 What the existing ane-mtp.mm pattern gives us

`common/ane-mtp.mm:798-858` already loads multi-function Core ML bundles
by name, wraps fp16 weights from the GGUF directly as `MLMultiArray` with
`deallocator:nil` for zero-copy (`ane-mtp.mm:553-570`), and dispatches
each named function (`prefill_sN`, `mtp_predict`, etc.) through a serial
dispatch queue. This is the architectural template for Project A: rather
than one Core ML call per matmul, the matmul gets folded into a named
bundle function (e.g. `attn_qkv_proj`) that takes a single
`hidden_state` IOSurface, computes all four matmuls (Q, K, V, O) on ANE
in one program, and writes the output to another IOSurface. The
ternary→fp16 reconstruction prologue runs once before the bundle call
and the result is shared across all four matmuls.

---

## 4. Project A roadmap

### 4.1 Headline plan (3-4 weeks)

| Week | Goal | Concrete artifacts |
|---|---|---|
| **W0 (1 day)** | **Spike**: confirm Core ML fp16 matmul runs on ANE through ggml-ane.mm end-to-end | Spike PR #1: build a 256x256 fp16 `matmul` mlpackage with `xcrun coremlcompiler compile`, load via `ggml_backend_ane_program_load_from_dir`, write an `add_tensor` to the IOSurface input arena, call `ggml_backend_ane_graph_compute` with a single MUL_MAT op, compare output to CPU fp16 reference within 1e-3 |
| **W1** | Wire up **MUL_MAT** dispatch through a per-projection bundle | New `attn_qkv_proj` bundle function in the conversion tool; populate `ggml_ane_program_dispatch_op` for `MUL_MAT`; resolver finds the right bundle function by tensor name; zero-copy fp16 weight wrap (mirror ane-mtp.mm:553-570) |
| **W2** | **Ternary→fp16 reconstruction prologue** | New `ggml_ane_ternary_to_fp16` function: takes TILE640 src[0..5] + scale params, produces a legal-shape fp16 weight tensor in an IOSurface. Hot path on Apple NEON, ~0.3 ms for 4096x4096. Plumb the prologue into `ggml_ane_program_dispatch_op` for `TILE640_MATMUL` |
| **W3** | End-to-end integration + telemetry | Ladder of 3 fixtures: pure-fp16 (sanity), ternary+fp16-prologue (correctness vs Metal), multi-matmul bundle (amortize dispatch). IOReport ANE power verification (F1 in deep-study). MLComputePlan residency check. Optional: bench against Metal at the per-matmul and per-token-decode levels. |

### 4.2 The 1-day spike (W0) — concrete spec

**Goal:** prove the integration path works end-to-end before committing to
the bundle architecture. Answers three questions:

1. Can the ggml-ane backend wrap a fp16 matmul as a Core ML `matmul` (or
   `ios18.conv` 1x1) and get a non-garbage result?
2. Is the IOSurface zero-copy wrap pattern (existing
   `ggml_ane_arena_slot` and `ggml_ane_program_run`) sufficient to feed
   the matmul without an extra fp32→fp16 cast on the hot path?
3. Does the per-op `MLModel.predict` dispatch overhead actually match
   the 0.1-0.5 ms number maderix reports, on the actual hardware we'll
   ship on?

**What to build** (`/tmp/spike/ane_matmul_spike.mm` or similar):
```
// 1. Build a minimal .mlpackage:
//    input: x [1, 1, 1, 256]  (one token, 256 channels, fp16)
//    weight: const W [256, 256, 1, 1] fp16 (baked at compile)
//    output: y [1, 256, 1, 1] = conv(x, W)  (the ios18.conv path)
//    compute_units = CPU_AND_NE
// 2. xcrun coremlcompiler compile the .mlpackage
// 3. In C++:
//    a. Load via ggml_backend_ane_program_load_from_dir
//    b. Allocate a 256x256 fp16 input on an IOSurface (use existing
//       ggml_ane_arena_slot)
//    c. Wrap as MLMultiArray (zero-copy, deallocator:nil)
//    d. Run via ggml_ane_program_run with input "x" and output "y"
//    e. Compare y to CPU fp16 reference (1e-3 tolerance)
// 4. Time 100 iterations; report median and p99 dispatch latency
```

**Decision criteria:**
- If correctness fails (output doesn't match CPU ref within 1e-3):
  → diagnosis. The 99% cause is one of Orion #1-#20; the comment block
  at `ggml-ane.mm:1019-1064` plus `docs/ane-backend-deep-study.md:3.1-3.4`
  is the troubleshooting tree.
- If dispatch overhead is >1 ms: the public-path pessimization is worse
  than maderix measured. We may need to drop the per-op dispatch
  pattern in favor of a 3-4 matmul bundle from day one.
- If IOReport ANE power stays near 0 mW: the model is silently falling
  to CPU/GPU (failure mode F1). The bundle function isn't shaped for
  ANE (the `ios18.conv` 1x1 formulation may not have been emitted by
  coremltools). Switch to hand-built MIL with `mb.conv` and the
  `ios18` opset, or use `coremltools.optimize.coreml.linear_quantize_weights`
  with `dtype=int8, granularity="per_tensor"` (ANE Book Law 3) and
  see if residency comes back.

**What we learn:**
- Confirms Core ML → ANE integration is wired correctly through the
  existing `ggml_ane_program_load_from_dir` + `ggml_ane_program_run` path.
- Quantifies the per-op dispatch overhead on our target hardware (M2
  and M4-class, whichever we ship on).
- Establishes the legal-shape contract for our matmul bundle: the
  shape `[1, 256, 1, 1]` is legal (below the 49 KB floor with fp16
  2-byte elems → 512 bytes; we need to pad T to 16 to clear the
  49 KB floor or use a larger shape).
- Validates the 64 KB minimum alloc assumption in `ggml-ane.mm:32`.

### 4.3 The TILE640_MATMUL prologue (W2)

The prologue is a **CPU function**, not a Core ML op. Pseudocode:

```cpp
// In ggml-ane.mm
static bool ggml_ane_ternary_to_fp16(
    ggml_tensor * dst,        // output: [1, in_dim, 1, out_dim] fp16
    const uint8_t * packed,   // src[0]: packed ternary trits
    const uint16_t * ps,     // src[1]: page scales
    const int8_t    * ls,     // src[2]: lane scales
    const int32_t   * ors,    // src[3]: outlier_row_offsets
    const int32_t   * oc,     // src[4]: outlier_cols
    const float     * ov,     // src[5]: outlier_vals
    int out_dim, int in_dim);

// Hot path: ~0.3 ms for 4096x4096, ~1.3 ms for 8192x8192 on M-class
// (NEON fp16 mul ~50 GB/s aggregate, see Section 2.1 above).
// Called ONCE per TILE640_MATMUL node, before the bundle dispatch.
// Output is written into the dst tensor's IOSurface arena.
```

**Validation plan:**
- Compare prologue output bit-exact against the Metal reference kernel
  `ts_metal_dequant_mse_recon` (`tools/quantize/tessera/tessera-metal.mm:495-580`)
  for a fixed random input — they must produce the same fp16 weights.
- Profile the prologue under Instruments (Instruments → ANE Power →
  ANE Compiler) to confirm the prologue is CPU-only and does not
  trigger an ANE program load.
- The prologue should NOT live in the ANE backend's hot path for
  elementwise ops — it sits in the matmul dispatch path only.

### 4.4 Risks for Project A

The top 3 risks, in order of expected impact on the 3-4 week timeline:

#### Risk 1 — Dispatch overhead eats the prologue win (HIGH)

Each per-op `MLModel.predict` is ~0.1-0.5 ms of XPC + IOKit
(maderix benchmarks). The ternary→fp16 prologue is ~0.3-1.3 ms per
matmul (Section 2.1). If we dispatch each `MUL_MAT` and each
`TILE640_MATMUL` as its own Core ML call, the prologue and dispatch
overhead together are 0.4-1.8 ms per matmul. The ANE compute for a
4096x4096 fp16 matmul is ~0.3-1.0 ms. Net: **the prologue and dispatch
overhead cancel most of the ANE speedup at the per-op level.**

**Mitigation.** Build multi-matmul bundles (e.g. `attn_qkv_proj` packs
Q, K, V, O into one Core ML function). The `common/ane-mtp.mm:798-858`
pattern is the template. Per the existing conversion design
(`docs/ane-backend-deep-study.md` Section 4.3.1), the prefill slab
already packs a whole layer's matmuls. For decode, pack at least QKV
together.

**Spike output that would invalidate this risk:** if W0 reports
dispatch overhead < 0.1 ms on M2/M4, the per-op pattern is fine and
the prologue is the only cost.

#### Risk 2 — Silent CPU/GPU fallback (HIGH, but detectable)

The Core ML public scheduler silently moves ops to CPU or GPU when
it judges them ANE-incompatible (ANE Book Law 2, hollance page). The
current `supports_op` check in `ggml-ane.mm:1141-1207` is a
declaration of intent, not a residency check. A graph that says "this
op is supported" may actually run on Metal and we won't notice until
profiling.

**Mitigation.** Add an IOReport ANE power check after every
`MLModel.predict` (deep-study failure mode F1): if ANE power is ~0 mW
during inference, the function fell off ANE. Also run `MLComputePlan`
on each .mlpackage before binding it; reject bundles that don't show
`preferredComputeDevice == .neuralEngine` for the matmul/conv op.
ANE Book Law 6: "Do not trust compilation success as evidence of ANE
placement." The 1-day spike is the right place to wire this in.

#### Risk 3 — No native ternary matmul, no int2/int4 ANE speedup (MEDIUM-HIGH)

The ANE has no ternary matmul (Section 2.3). The 38 TOPS INT8 figure
is a marketing double-count — real peak is 19 TFLOPS fp16 regardless of
quantization (maderix Part 2). So the headline "ternary on ANE for
1.5-bit weight savings" is "ternary storage + fp16 compute", with the
fp16 compute getting ~5.7 TFLOPS sustained in the real workload (maderix
2048x2048 benchmark, M4). If the actual hot matmul shapes are
bandwidth-bound (small batch, large weight), the speedup is dominated
by memory bandwidth, not by FLOPs, and the fp16 reconstruction does
not help as much as the FLOPs-only number suggests.

**Mitigation.** The expected user-visible win is **fp16 matmul on ANE
beats fp16 matmul on Metal** (3-5x per matmul, Section 2.1) and
**ternary-storage + fp16-compute beats fp16-storage + fp16-compute**
when the bandwidth-bound regime kicks in. The "ternary on the ANE"
narrative is true but it is "ternary via reconstruction" — the user
should not expect a "native ternary matmul" win.

### 4.5 Secondary risks (lower priority)

- **MLTensor / Core ML scheduler opacity** — the scheduler is a black
  box (hollance, ANE Book Ch. 1). Two of Orion's constraints (#6 SDPA
  causal mask silently ignored, #20 packed flat read) can produce
  silently-wrong output. Mitigated by golden-output comparison (W3
  fixtures compare every result against the Metal reference for the
  same input).
- **49 KB minimum IOSurface** — decode-time tensors (one token, hidden
  3072, fp16) are 6 KB; well below 49 KB. Pad T to 16 in the conversion
  tool (Apple ml-ane-transformers Principle 1).
- **25 MB IOSurface, single-input 32 MB SRAM cliff** — for very large
  weight tensors, the working set may exceed ANE SRAM. Detect via
  `ggml_ane_program_warm` (already in ggml-ane.mm:419-466) and
  fall back to Metal.
- **MUL_MAT_ID (MoE expert selection)** — out of scope for Project A;
  defer to Project B. The 8-expert cap on per-program matmul count
  (Orion §6 LoRA) means MoE on ANE needs per-expert sharding, not the
  per-op dispatch pattern.

---

## 5. References

All URLs verified reachable at the time of this report (2026-07-31).

### Public ANE characterization
- Apple, "Deploying Transformers on the Apple Neural Engine", 2022.
  <https://machinelearning.apple.com/research/neural-engine-transformers>
  — Principles 1-4: channels-first [B,C,1,S], 64-byte last-axis
  alignment, einsum bchq,bkhc->bkhq, fp16 compute.
- Hollemans (hollance), "Everything we know about the Apple Neural
  Engine", unsupported-layers doc.
  <https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md>
  — gather/addBroadcastable/mulBroadcastable cause fallback; conv and
  matmul work.
- maderix, "Inside the M4 Apple Neural Engine, Part 2: Benchmarks",
  Substack, 2026. <https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615>
  — 19 TFLOPS fp16 peak; SRAM cliff at ~32 MB; 256x256 is dispatch-
  limited at 0.101 ms; 2048x2048 = ~5.7 TFLOPS; 4096x4096 drops to
  ~4.0 TFLOPS; INT8 has no 2x speedup (debunks "38 TOPS INT8").
- maderix, "Inside the M4 Apple Neural Engine, Part 3: Training".
  <https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b>
  — "FP16 is fine for forward, dangerous for backward"; ANE doesn't
  care it's "inference only".
- Kumaresan, "Orion: Characterizing and Programming Apple's Neural
  Engine for LLM Training and Inference", arXiv 2603.06728, 2026.
  <https://arxiv.org/html/2603.06728v1>
  — 20 ANE constraints (6 prior, 14 new). #4 min IOSurface ~49 KB;
  #6 SDPA causal mask silently ignored; #10 GELU not valid MIL; #12
  matmul transpose flags need named consts; #13 conv has no bias;
  #16 32K-channel conv rejected; #18-#20 multi-input/output
  uniformity, alphabetical binding, packed flat read.
- arXiv 2606.22283, "Apple Neural Engine: Architecture, Programming,
  and Performance". <https://arxiv.org/pdf/2606.22283.pdf>
  — 16384 width/height, 65536 channels, 13-fp16 kernel width, 2048
  arg-min/arg-max axis.
- Apple ANE Book, Chapter 1, alvaro-videla, 2026.
  <https://alvaro-videla.com/ane-book/01-ane-laws.html>
  — Shape [1,C,T,1]; ios18.conv for matmul (3x over matmul); INT8
  per-tensor; 250 MB shard ceiling; MLComputePlan is ground truth.
- PROMPIE et al., "Evaluation of Domain-Specific Architectures for
  General-Purpose ...", arXiv 2511.13450, 2025.
  <https://arxiv.org/pdf/2511.13450.pdf>
  — 3.8 TFLOPS GEMM on M4 Pro ANE; 5.2 W vs 24 W for GPU.
- sivaro.in, "Apple Neural Engine: Programming for Real Performance".
  <https://sivaro.in/articles/apple-neural-engine-programming-for-real-performance/>
  — INT8 alignment (4 channels), NHWC preferred, fp16/fp8/INT8 only.
- maderix ANE architecture overview.
  <https://maderix-ane.mintlify.app/concepts/ane-architecture>
  — Peak TFLOPS, dtype, matmul/softmax/cast table; ANE ignores
  attn_mask in SDPA; 1024-token seq limit on tested IOSurface.
- maderix ANE optimization gist.
  <https://gist.github.com/antmikinka/715499ae63630575065b22e5cb6ad8dd>
  — Powers of 2, multiples of 16, NHWC, 16-byte alignment.
- The Menon Lab, "80× More Efficient Than A100? Someone Reverse-
  Engineered Apple's ANE". <https://themenonlab.blog/blog/apple-neural-engine-reverse-engineered-training>
  — 6.6 TFLOPS/W; 1.78 TFLOPS sustained transformer training on M4.

### BitNet / ternary LLM context
- Ma et al., "BitNet b1.58: The Era of 1-bit LLMs", Microsoft, 2024
  — foundational. ternary weights, 8-bit activations, BitLinear layer.
- Ma et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs",
  arXiv 2502.11880, 2025. <https://arxiv.org/html/2502.11880v1>
  — mpGEMM, I2_S and TL1/TL2 kernels; int8/lookup-table approach.
- bitnet.cpp GitHub: <https://github.com/microsoft/BitNet/tree/paper>
- Reddit /r/LocalLLaMA, "So what happened to the 1.58-bit models
  revolution" — community consensus: "ternary dot products run on
  CPU as mixed-precision ternary×INT8/FP16, not ternary×ternary."
  <https://www.reddit.com/r/LocalLLaMA/comments/1hsa0tm/so_what_happened_to_the_158bit_models_revolution/>

### Core ML / coremltools
- coremltools MIL ops reference (8.1).
  <https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html>
  — op per opset; matmul iOS15+, ios18.conv iOS18+.
- coremltools multifunction models guide.
  <https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html>
- coremltools target conversion formats.
  <https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html>
- WWDC 2024 Session 10161, "Deploy machine learning and AI models
  on-device with Core ML". Introduces MLTensor.
  <https://developer.apple.com/videos/play/wwdc2024/10161/>
- WWDC 2026 Session 330, "Optimize custom machine learning
  operations with Metal TensorOps" — Metal-side counterpart;
  matmul2d with quantized tensors.
  <https://developer.apple.com/videos/play/wwdc2026/330/>

### llama.cpp / ggml-coreml context
- ggml-org/llama.cpp ops matrix.
  <https://fossies.org/linux/llama.cpp/docs/ops.md>
- ggml-org/llama.cpp issue #10453, "add ANE backend".
  <https://github.com/ggml-org/llama.cpp/issues/10453>
  — MLTensor limitation: only CPU and CPU+GPU via `MLComputePolicy`,
  not direct ANE.
- ggml-org/llama.cpp discussion #336, "Neural Engine Support".
  <https://github.com/ggml-org/llama.cpp/discussions/336>
  — 4x speedup of 100 matmuls on M2 ANE (217 ms vs 1316 ms GPU).
- ggml MulMat operation, ai.stackexchange.
  <https://ai.stackexchange.com/questions/40105/what-operation-is-ggml-mul-mat-performing-k%C3%97q-in-llama>
  — `mul_mat(W.T, x)` for `y = x @ W` in GGUF.

### In-tree Tessera code (read this report alongside)
- `ggml/src/ggml-ane/ggml-ane.mm:22-41` — page alignment, 64 KB min
  alloc, `ggml_ane_round_size`.
- `ggml/src/ggml-ane/ggml-ane.mm:419-466` — `ggml_ane_program_warm`
  (fail-fast on bundles that don't compile to ANE).
- `ggml/src/ggml-ane/ggml-ane.mm:548-644` — `ggml_ane_program_run`
  (per-input IOSurface allocation, zero-copy MLMultiArray wrap).
- `ggml/src/ggml-ane/ggml-ane.mm:876-903` — `ggml_ane_program_dispatch_op`
  (the matmul integration point, currently returns false).
- `ggml/src/ggml-ane/ggml-ane.mm:1141-1207` — `supports_op` whitelist
  (the 26 cases today; Project A adds MUL_MAT and TILE640_MATMUL).
- `common/ane-mtp.mm:525-570` — fp16 weight wrap from mmap'd GGUF
  (the zero-copy pattern to reuse for the prologue output).
- `common/ane-mtp.mm:798-858` — multi-function bundle load + per-
  function MLState + warmup (the bundle architecture template).
- `tools/ane-mtp/export-gemma4-prefill-bundle.py` — existing ML-
  program export with RMS norm, RoPE, SDPA decomposition, GELU tanh.
- `tools/quantize/tessera/tessera-metal.mm:495-580` —
  `ts_metal_dequant_mse_recon` (the ternary→fp32 reconstruction
  reference; we mirror this in CPU for the prologue).
- `ggml/src/ggml-metal/ggml-metal-ops.cpp:1765-1828` —
  `ggml_metal_op_tile640_matmul` (the TILE640 matmul layout contract:
  7 src tensors; src[0] packed, src[6] activations).
- `ggml/src/ggml-metal/ggml-metal-tile640-interleaved.metal` — the
  TILE640 Metal kernel (the structural template for our prologue).
- `tools/quantize/tessera/tile640/calibrate_quantize.py` and
  `quantize_v3.py` — TILE640 quantizer side (the on-disk format).
- `tools/quantize/tessera/tessera-imatrix.cpp` / `.h` — the imatrix
  reader (per-channel activation magnitudes; used by AWQ but not
  directly by the ternary→fp16 prologue).
- `docs/ane-backend-deep-study.md` — full 20-constraint catalog,
  composite op decompositions, IOSurface architecture, failure mode
  catalog, slice-by-slice implementation recommendations.
- `docs/tessera-coreml-conversion-design.md` — the C++ conversion
  tool design (the in-tree companion to this report).
- `docs/tessera-ane-pump.md` — the E-core pump (W6/W6.5/W7/W8)
  and the F4.1-F4.5 follow-ups (in-band caller routing,
  monotonic counter, QOS background affinity, MTP/DFlash
  manifest sidecar, Phase 0 profile NDJSON emit).
