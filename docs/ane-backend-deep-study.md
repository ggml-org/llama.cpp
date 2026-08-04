# D1 ANE Backend Deep Study: Compiler Constraints, Op Catalog, and Architecture Model

Read-only design study. No code changes. No commits.

> This document supersedes the op coverage assumptions in the prior D1 design
> study and the `tessera-coreml-conversion-design.md` C1/C7 decisions with
> empirically grounded ANE constraint data drawn from three independent
> research programs: Apple's own `ml-ane-transformers` reference, the Orion
> direct-ANE framework (Kumaresan 2026), and the maderix/ANE reverse-
> engineering project.

## Table of Contents

1. Part 1: coremltools MIL Op Catalog
2. Part 2: ANE Reverse-Engineering Research (Orion + maderix/ANE)
3. Part 3: Orion/Maderix Constraint Catalog (20 Constraints)
4. Part 4: ANE Backend Model for Tessera
   - 4.1 Op Coverage Matrix v2
   - 4.2 Composite Op Decompositions
   - 4.3 IOSurface + Metal Event Architecture
   - 4.4 Failure Mode Catalog
   - 4.5 Implementation Recommendations for Slices 1-4

---

## Part 1: coremltools MIL Op Catalog

### 1.1 Op Registry Structure

The coremltools MIL op registry lives at:
```
coremltools/converters/mil/mil/ops/defs/
  iOS15/     - baseline ops (iOS 15+, macOS 12+, A12+/M1+)
  iOS16/     - expanded ops (iOS 16+)
  iOS17/     - quantize/dequantize, gather ND, topk updates
  iOS18/     - scaled_dot_product_attention, slice_update, read_state
  coreml_dialect/ - coreml_update_state (any iOS)
```

Source: https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html

**There are NO per-op `compute_unit` annotations in the defs.** The ANE dispatch
decision is made at runtime by the Core ML scheduler, not declared at the op
level. The scheduler is a black box: it accepts or rejects ops for ANE based
on undocumented internal heuristics. This is the fundamental problem.

The conversion tool (Section 4.5 of the conversion design) uses `MLModel`
with `MLComputeUnitsCPUAndNeuralEngine`, which is a *preference*, not a
command. The runtime silently falls ops to GPU/CPU when it judges them ANE-
incompatible. There is no diagnostic API to query which ops ran where, except
the Xcode Core ML Performance Report (macOS only, not available on iPhone).

### 1.2 Complete MIL Op Inventory (Transformer-Relevant)

The following is the authoritative list of MIL ops that map to a standard
transformer forward pass, extracted from the coremltools defs index:

| MIL Op | iOS Version | ANE Status | Notes |
|--------|-------------|------------|-------|
| `matmul` | 15+ | ANE-native | Primary linear algebra op. Transpose flags required. |
| `linear` | 15+ | ANE-native | `matmul + bias` fused. Bias must be separate `add` for direct ANE. |
| `conv` | 15+ | ANE-native (preferred) | 1x1 conv is 3x faster than matmul on ANE. Use `conv` formulation. |
| `add` | 15+ | ANE-native | Works. `addBroadcastable` variant causes GPU fallback (hollance). |
| `mul` | 15+ | ANE-native | Works. `mulBroadcastable` variant causes GPU fallback. |
| `sub` | 15+ | ANE-native | Works. |
| `sigmoid` | 15+ | ANE-native | Works. |
| `tanh` | 15+ | ANE-native | Works. Required for GELU tanh approximation. |
| `silu` (sigmoid * x) | 15+ | ANE-native | Available as a single MIL op. Composite of sigmoid + mul also works. |
| `gelu` | 15+ | ANE-BREAKS on direct path | GELU is NOT a valid ANE MIL op (Orion constraint #10). Must use tanh approximation. |
| `softmax` | 15+ | ANE-native | Works for standard softmax. |
| `layer_norm` | 15+ | ANE-native | Available since iOS 15. |
| `batch_norm` | 15+ | ANE-native | Available. |
| `rms_norm` | NOT in MIL | N/A | No native RMS norm op. Must decompose: `reduce_sum -> rsqrt -> mul`. |
| `concat` | 15+ | ANE-BREAKS on direct path | `concat` causes compilation failure on ANE (Orion #1, hollance ND layer). |
| `reshape` | 15+ | ANE-native (with cost) | Triggers memory copy on ANE due to packed axis. Minimize usage. |
| `transpose` | 15+ | ANE-native (with cost) | Triggers memory copy. Apple recommends minimizing transposes. |
| `gather` / `gather_nd` | 15+/16+ | ANE-SILENT FALLBACK | Listed as unsupported by hollance. Causes GPU fallback. |
| `slice_by_index` / `slice_by_size` | 15+ | ANE-native | Works. |
| `topk` | 15+/16+ | ANE-native | Works (iOS 17+ for updated variant). |
| `cast` | 15+ | ANE-native | fp16 <-> fp32 casting works. |
| `exp` | 15+ | ANE-native | Works. Watch fp16 overflow (max 65504). |
| `log` | 15+ | ANE-native | Works. |
| `rsqrt` | 15+ | ANE-native | Used in RMS norm decomposition. |
| `reduce_sum` / `reduce_mean` | 15+ | ANE-native | Used in norm decomposition. |
| `scaled_dot_product_attention` | 18+ | ANE-UNKNOWN | New iOS 18+ op. May not map to ANE. Causal masks silently ignored (Orion #6). |
| `einsum` | 15+ | ANE-native (recommended) | Apple's ml-ane-transformers uses einsum `bchq,bkhc->bkhq` for fused SDPA. |
| `conv_transpose` | 15+ | ANE-native | Not used in transformers. |
| `relu` / `leaky_relu` / `clamped_relu` | 15+ | ANE-native | Work. |
| `abs`, `clip`, `square` | 15+ | ANE-native | Used in norm decompositions. |
| `constexpr_affine_dequantize` | 16+ | ANE-native (compile-time) | Dequantizes int8 weights to fp16 at compile time. |
| `quantize` / `dequantize` | 17+ | ANE-native | Runtime int8 <-> fp16 for activation caching in L2 SRAM. |
| `constexpr_lut_to_dense` | 16+ | ANE-native (compile-time) | LUT-based dequant at compile time. Tessera T640 dequant candidate. |
| `pad` | 15+ | ANE-native | Works. |
| `split` | 15+ | ANE-native | Works. |
| `tile` | 15+ | ANE-native | Used for repeat_interleave in GQA. |
| `stack` | 15+ | ANE-native | Works. |
| `expand_dims` / `squeeze` | 15+ | ANE-native | Layout ops, minimal cost. |

### 1.3 ANE Data Format Constraints

The ANE requires a **4D channels-first** data format: `[B, C, 1, S]`.

Source: Apple ml-ane-transformers research article (Principle 1).

- The last axis (S = spatial/sequence) must be contiguous and **aligned to 64
  bytes**.
- A singleton last axis (size 1) gets padded to 64 bytes, causing 32x memory
  inflation in fp16 (64x in int8). This destroys L2 cache residency.
- **Workaround**: Ensure the sequence dimension is always >= 16 (or >= 32 for
  int8). For decode (seq=1), pad the sequence axis to at least 16 and zero the
  padding.
- The ANE compiler manages packing for the other axes (B, C, D=1).

The minimum IOSurface allocation for an ANE eval is approximately **49 KB**
(Orion constraint #4). A tensor of shape `[1, 768, 1, 1]` (3072 bytes in fp16)
will compile but fail at evaluation with error 0x1d. The fix is to pad to
`[1, 768, 1, 16]` (24576 bytes).

### 1.4 Conv2d vs Matmul Performance

**Conv2d 1x1 is approximately 3x faster than mathematically equivalent matmul
on the ANE.** This is confirmed by both Apple's ml-ane-transformers (which
replaces all `nn.Linear` with `nn.Conv2d`) and the maderix benchmarks.

Source: maderix Substack "Inside the M4 Apple Neural Engine"; Orion constraint #17.

The reason is architectural: the ANE's convolution engine is more efficiently
pipelined than its general matrix multiply engine. For the Tessera backend, all
weight projections (Q, K, V, O, gate, up, down) should be emitted as `conv`
MIL ops with `kernel_size=(1,1)` instead of `matmul` when targeting the ANE
directly. When going through the public Core ML path (coremltools + xcrun),
the `matmul` MIL op still works and will be lowered to conv internally by the
Core ML compiler on some iOS versions, but this is not guaranteed.

### 1.5 MIL Serialization

The MIL program is serialized into a protobuf defined in `MIL.proto` (inside
`mlmodel/format/` in the coremltools repo):

```
MLProgram {
  version: int64
  program_attributes: { ... }
  functions: map<string, Function> {
    Function {
      block_inputs: [Value]
      block_outputs: [string]
      operations: [Operation]
    }
  }
}
```

Source: https://apple.github.io/coremltools/docs-guides/source/comparing-ml-programs-and-neural-networks.html

Key properties:
- Weights are serialized **outside** the protobuf (in the `weights/` directory
  of the `.mlpackage`). The protobuf contains only the architecture (ops + SSA
  graph).
- The `const` op stores small weights inline as attributes. Large weights are
  stored as file-backed `constexpr` values referencing external weight files.
- The MIL builder API maps 1:1 to the protobuf: the same internal MIL Python
  object is serialized directly to the protobuf format.
- The opset is versioned collectively (iOS 15 = baseline, iOS 17 adds
  quantize/dequantize, iOS 18 adds SDPA and stateful ops).

The Tessera C++ conversion tool (planned in `tessera-coreml-conversion-
design.md` Section 4.3) constructs this protobuf by hand using the Prism
`mil_builder.rs` pattern: it builds an in-memory `mil_spec::Program`
protobuf and writes it as the `model.mlmodel` file inside the `.mlpackage`.

### 1.6 Multi-Function Programs

Core ML 7+ (iOS 18+, macOS 15+) supports **multi-function programs**
(`MLMultiFunctionProgram`). The Tessera export script
(`tools/ane-mtp/export-gemma4-prefill-bundle.py`) already uses this pattern.

Source: https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html

API surface:
```python
desc = ct.utils.MultiFunctionDescriptor()
desc.add_function(src_path, src_function_name, target_function_name)
desc.default_function_name = "prefill_s128"
ct.utils.save_multifunction(desc, output_path)
```

Key constraints:
- **Minimum deployment target**: iOS 18+ / macOS 15+. The component models
  must be converted with `minimum_deployment_target=ct.target.iOS18`.
- **Only mlprogram model type** supports multifunction. Neural network type
  does not.
- **Shared weights**: During merging, coremltools deduplicates shared weights
  by hashing. The Tessera prefill bundle uses this: all `prefill_s{N}`
  functions share one set of layer-0 weights, so the bundle size is roughly
  constant regardless of bucket count.
- **Shared MLState**: Multiple functions can share the same `MLState` object.
  The Tessera MTP program uses this: the execution state and keepalive state
  are created from the same loaded model.
- **Function loading**: `MLModel(path, function_name="prefill_s256")` loads a
  specific function. Without a function name, the default function is loaded.

The Tessera ane-mtp runtime (common/ane-mtp.mm lines 798-858) loads each
function from a multifunction bundle by name, creates separate `MLState`
instances per function for warmup, and tracks per-function warm status.

---

## Part 2: ANE Reverse-Engineering Research

### 2.1 The Three Research Programs

Three independent efforts have characterized the ANE from the outside:

| Project | Approach | Ops Discovered | Constraint Catalog | Public? |
|---------|----------|---------------|-------------------|---------|
| Apple ml-ane-transformers | Public Core ML path (coremltools) | matmul, conv, softmax, einsum, reshape avoidance | 4D channels-first, 64-byte alignment, conv preference | Yes |
| maderix/ANE + Orion | Private APIs (_ANEClient, _ANECompiler) | ~27 graph IR ops, fused kernels | 20 constraints (6 prior, 14 new) | Yes |
| ANEForge | Private APIs, Python | 58 fused + 19 bridge = 77 ops | Subset of Orion catalog | Yes |

Source URLs:
- Apple ml-ane-transformers: https://github.com/apple/ml-ane-transformers
- Apple research article: https://machinelearning.apple.com/research/neural-engine-transformers
- Orion paper: https://arxiv.org/html/2603.06728v1
- Orion source: https://github.com/mechramc/Orion
- maderix/ANE: https://github.com/maderix/ANE
- maderix Substack: https://maderix.substack.com/p/inside-the-m4-apple-neural-engine
- ANEForge: https://arxiv.org/abs/2606.17090
- Orion blog: https://ramchandk.com/blog/orion-programming-apple-neural-engine
- hollance/neural-engine: https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md

### 2.2 The Private API Compilation Pipeline

Orion and maderix/ANE both bypass Core ML entirely and talk to the ANE
through three private frameworks:

1. **`_ANEClient`**: ANE initialization, program loading, execution dispatch.
2. **`_ANECompiler`**: Compiles MIL IR text into ANE microcode (E5 binary).
3. **`_ANEInMemoryModelDescriptor`**: In-memory compilation without writing
   `.mlmodelc` to disk.

The compilation pipeline (Orion):
```
Graph IR (~27 ops) -> 5 optimization passes (DCE, identity elim, cast fusion,
SRAM annotation, output padding) -> ANE validation -> MIL text codegen ->
_ANECompiler -> E5 microcode -> _ANEClient exec()
```

This is NOT what Tessera uses. Tessera targets the **public path**: coremltools
(or hand-built protobuf) -> `.mlpackage` -> `xcrun coremlcompiler compile`
(or `[MLModel compileModelAtURL:]`) -> `.mlmodelc` -> `MLModel` load +
predict. The public path goes through the Core ML scheduler, which adds
overhead but is App Store safe.

However, the constraint catalog discovered by Orion/maderix applies equally
to the public path. The ANE hardware is the same regardless of whether you
reach it through Core ML or through `_ANECompiler`. The difference is:
- Public path: Core ML may silently fall ops to GPU/CPU instead of failing.
- Direct path: ops either compile to E5 microcode or you get an explicit error.

### 2.3 ANE Memory Model

All ANE tensor I/O uses **IOSurface** shared memory.

Source: maderix/ANE source code; MacInternals IOSurface article.

IOSurface properties for ANE:
- Format: `[1, C, 1, S]` channel-first layout. The ANE reads flat buffers as
  packed `[1, C, 1, S]` (Orion constraint #20).
- **Minimum allocation**: ~49 KB. Allocations below this compile but fail
  at evaluation (Orion #4).
- **64-byte alignment**: The last axis (S) must be 64-byte aligned. The ANE
  compiler manages packing for other axes.
- **Multi-input constraint**: All input IOSurfaces to a single program must
  have the **same byte allocation size**, padded to the maximum (Orion #18).
  Violation causes error 0x1d at eval.
- **Multi-output constraint**: All output IOSurfaces must have the **same byte
  allocation size**, padded to the maximum (Orion #2).
- **Alphabetical binding**: Input/output IOSurfaces are bound to MIL
  parameters by **alphabetical order** of their variable names, not the
  function signature order (Orion #3, #19). Misalignment causes silent wrong
  data.

The Tessera ane-mtp runtime already implements IOSurface-backed arenas
(ane-mtp.mm lines 30-100, `common_ane_mtp_arena_buffer`). The arena uses:
```objc
NSDictionary * properties = @{
    (id) kIOSurfaceWidth: @(rounded),
    (id) kIOSurfaceHeight: @1,
    (id) kIOSurfaceBytesPerElement: @1,
    (id) kIOSurfaceBytesPerRow: @(rounded),
    (id) kIOSurfaceAllocSize: @(rounded),
};
IOSurfaceRef replacement = IOSurfaceCreate((CFDictionaryRef) properties);
```

With 16 KB page alignment (`const size_t page = 16 * 1024`). This satisfies
the ANE alignment requirements.

### 2.4 ANE Hardware Architecture

Source: maderix Substack Part 1.

The M4 ANE (codename H16G) has:
- 16 cores
- ~19 TFLOPS fp16 compute (Apple's "38 TOPS INT8" claim is misleading because
  the ANE dequantizes INT8 to fp16 before computing)
- Fixed-function graph execution engine: submits an entire compiled neural
  network graph as one atomic operation
- A small set of compute primitives: conv, matmul, elementwise, softmax,
  reduce -- parameterized by tensor shape descriptors
- Queue depth of 127 in-flight evaluation requests
- Fully independent DVFS, separate from CPU/GPU power domains
- Hard power gating to exactly 0 mW when idle
- Compilation cache at `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/`

### 2.5 Fusion Patterns

Orion achieves **170+ tok/s** on GPT-2 124M by fusing transformer
operations into fewer ANE dispatches. The maderix/ANE project reports a
**2-4x throughput improvement** over the public Core ML path by bypassing the
Core ML scheduler overhead.

Key fusion strategies:

**Fused attention kernel** (maderix/ANE, Stories110M):
```
sdpaFwd: QKV projection + SDPA + output projection  (6 kernels/layer for MHA)
```

The entire QKV projection, attention computation, and output projection are
fused into a single ANE program. This avoids intermediate IOSurface writebacks
and kernel launch overhead.

**Einsum-based SDPA** (Apple ml-ane-transformers):
```
einsum("bchq,bkhc->bkhq")  -- batched matmul with Q and K in native layout
```
This avoids the transpose that standard `Q @ K^T` would require.

**LoRA hot-swapping** (Orion):
```
Y = conv1x1(x, W_base) + alpha * (x @ A) @ B
```
Adapter weights A and B are passed as dynamic IOSurface inputs, enabling
sub-millisecond adapter switching without recompilation.

**Can we replicate these with the public MIL builder?** Partially:
- The public path supports `conv`, `matmul`, `einsum`, and element-wise ops
  that can be composed into the same graph.
- However, the public Core ML compiler makes its own fusion/scheduling
  decisions. We cannot force specific fusion boundaries.
- The `const` op for weights and the SSA graph structure give us the
  equivalent of Orion's kernel architecture, but we cannot control which ops
  are fused into a single ANE dispatch vs. split across dispatches.
- The public path is 2-4x slower than the direct path due to Core ML
  scheduler overhead, but it is App Store safe.

---

## Part 3: Orion/Maderix Constraint Catalog

The following is the complete catalog of 20 ANE constraints, with applicability
assessment for the Tessera public-path approach.

### 3.1 Compile/Eval Failure Constraints

| # | Constraint | Source | Public Path Applies? | Tessera Impact | Workaround |
|---|-----------|--------|---------------------|----------------|------------|
| 1 | `concat` MIL op rejected by ANE compiler | Orion (new) | YES - causes silent GPU fallback via Core ML scheduler | RoPE needs concat; decode needs KV append | Decompose concat into alternative ops or multi-output programs |
| 2 | Multi-output buffers must have uniform byte sizes | Orion (new) | YES - Core ML may mask this with copies | Prefill slab returns hidden_states + K + V with different sizes | Pad all outputs to max size; post-process in C++ |
| 3 | Multi-output surfaces bound alphabetically, not by signature order | Orion (new) | YES - Core ML names variables from MIL SSA names | Output binding mismatch if names not sorted | Name outputs alphabetically in the MIL builder |
| 4 | Minimum ~49 KB IOSurface for eval | Orion (new) | YES - but Core ML may pad automatically | Decode (seq=1) tensors are <49 KB | Pad sequence dim to >= 16; the export script already uses fixed buckets |
| 5 | ~119 compilations per process limit | Prior (maderix) | NO - public path uses pre-compiled .mlmodelc | Not applicable; models are compiled offline | N/A for public path |
| 8 | BLOBFILE offset is uint64(64), not 0 or 128 | Orion (new) | NO - public path uses .mlmodelc weight files | Not applicable | N/A for public path |
| 9 | MIL text must be NSData*, not NSString* | Orion (new) | NO - public path uses protobuf, not raw MIL text | Not applicable | N/A for public path |
| 11 | Weight dict must be @{}, not nil | Orion (new) | NO - public path uses .mlmodelc, not _ANECompiler | Not applicable | N/A for public path |

### 3.2 Silent Wrong Data Constraints

| # | Constraint | Source | Public Path Applies? | Tessera Impact | Workaround |
|---|-----------|--------|---------------------|----------------|------------|
| 6 | SDPA causal masks silently ignored | Prior (maderix) | YES - Core ML SDPA op inherits ANE behavior | Causal attention produces wrong results if relying on native mask | Manual causal masking: apply mask before softmax as a separate op |
| 7 | Weights baked at compile time | Prior (maderix) | YES - .mlmodelc bakes weights at compile | Weight update requires recompile | Not an issue for inference; training requires recompile |
| 12 | matmul transpose flags need named consts | Orion (new) | YES if Core ML passes raw MIL to ANE | matmul transpose_x/transpose_y must be const nodes | Use `const` ops for transpose flags; Prism already does this |
| 13 | conv does not support bias= parameter | Orion (new) | YES | Conv + bias fused in one op rejected | Separate `add` op after conv |
| 14 | Output vars must ref live (post-opt) nodes | Orion (new) | YES - Core ML optimizer may DCE nodes | Output referencing dead node after optimization | Validate SSA liveness after optimization passes |
| 16 | 32K-channel convolutions rejected | Orion (new) | YES | LM head (vocab_size channels) hits this | CPU fallback for LM head; or chunk into smaller convs |
| 18 | Multi-input surfaces must have uniform alloc sizes | Orion (new) | YES | Functions with multiple inputs of different shapes | Pad all inputs to max size |
| 19 | Multi-input surfaces ordered alphabetically | Orion (new) | YES | Input binding mismatch | Name inputs alphabetically |
| 20 | ANE reads flat buffer as packed [1,C,1,S] | Orion (new) | YES | Over-allocated inputs read wrong shape | Write packed data at buffer start |

### 3.3 Performance Constraints

| # | Constraint | Source | Public Path Applies? | Tessera Impact | Workaround |
|---|-----------|--------|---------------------|----------------|------------|
| 10 | gelu is not a valid MIL activation | Orion (new) | YES - Core ML may fall to CPU | Gemma 4 uses GELU with tanh approximation in export script | The export script already uses `F.gelu(x, approximate="tanh")`; keep this |
| 15 | exec() restart overhead ~50 ms | Prior (maderix) | NO - public path uses pre-compiled model | Not applicable | N/A |
| 17 | Conv 1x1 is 3x faster than matmul | Prior (maderix) | YES - Core ML may not auto-convert matmul to conv | Weight projections could be faster as conv | Emit `conv` ops instead of `matmul` in the MIL builder |

### 3.4 Additional Constraints from Community Sources

| Constraint | Source | Public Path Applies? | Workaround |
|-----------|--------|---------------------|------------|
| `addBroadcastable` prevents ANE use | coremltools #513, hollance | YES | Use plain `add` with explicit reshaping |
| `mulBroadcastable` prevents ANE use | hollance | YES | Use plain `mul` with explicit reshaping |
| `ConcatND` / `SplitND` prevents ANE use | hollance | YES | Use plain `concat` / `split` |
| Gather prevents ANE use | hollance | YES | Use `gather_along_axis` (iOS 17+) or decompose |
| Custom layers break ANE pipeline | StackOverflow | YES | No custom ops in v1 (C1 decision) |
| State tensor width must be multiple of 32 | Apple Developer Forums | YES - for MLState programs | Pad KV cache width to multiple of 32 |
| FP16 max representable value is 65504 | Apple Developer Forums | YES | Clamp activations before softmax/norm |
| Pooling kernel > 13 or stride > 2 prevents ANE | hollance | N/A for transformers | N/A |
| LSTM/GRU default to BNNS (CPU) | coremltools #337 | N/A | N/A |

---

## Part 4: ANE Backend Model for Tessera

### 4.1 Op Coverage Matrix v2

The following matrix maps every ggml op (from `ggml/include/ggml.h` lines
484-601) to its ANE eligibility classification, informed by the actual ANE
compiler constraints from Part 3.

Classification levels:
- **ANE-NATIVE**: Compiles and runs on ANE via public Core ML path with no
  special constraints.
- **ANE-NATIVE-C**: Compiles on ANE but requires specific dtype, layout, or
  shape constraints (documented in Notes column).
- **ANE-BREAKS**: Known to cause ANE compiler failure or silent wrong data;
  must fall back to CPU/GPU.
- **CPU-GLUE**: Not ANE-eligible, but cheap enough for CPU/Accelerate to handle
  without impacting throughput.
- **N/A**: Not relevant to transformer inference.

| ggml Op | ANE Class | MIL Equivalent | Notes |
|---------|-----------|---------------|-------|
| `GGML_OP_ADD` | ANE-NATIVE | `add` | Use plain `add`, not `addBroadcastable`. |
| `GGML_OP_MUL` | ANE-NATIVE | `mul` | Use plain `mul`, not `mulBroadcastable`. |
| `GGML_OP_MUL_MAT` | ANE-NATIVE-C | `matmul` or `conv` | Use `conv` formulation for 3x throughput gain. fp16 dtype. |
| `GGML_OP_MUL_MAT_ID` | ANE-NATIVE-C | `matmul` | MoE: select expert weights, then matmul. Same constraints. |
| `GGML_OP_RMS_NORM` | ANE-NATIVE-C | `reduce_sum` + `rsqrt` + `mul` | Composite decomposition (Section 4.2). fp16 overflow risk. |
| `GGML_OP_LAYER_NORM` | ANE-NATIVE | `layer_norm` | Native MIL op since iOS 15. |
| `GGML_OP_SOFT_MAX` | ANE-NATIVE-C | `softmax` | Clamp input to fp16 range before softmax. |
| `GGML_OP_SILU` | ANE-NATIVE | `silu` | Native MIL op. |
| `GGML_OP_SIGMOID` | ANE-NATIVE | `sigmoid` | Native MIL op. |
| `GGML_OP_TANH` | ANE-NATIVE | `tanh` | Native MIL op. |
| `GGML_OP_GELU` | ANE-BREAKS | N/A (direct) | GELU not valid on ANE. Use tanh approximation (Orion #10). |
| `GGML_OP_ROPE` | ANE-NATIVE-C | `cos` + `sin` + `mul` + `add` + `concat` | Composite decomposition (Section 4.2). `concat` risk. |
| `GGGL_OP_CONCAT` | ANE-BREAKS | `concat` | Silent GPU fallback. Decompose via reshape + write or multi-output. |
| `GGML_OP_RESHAPE` | ANE-NATIVE-C | `reshape` | Triggers memory copy. Minimize usage. |
| `GGML_OP_VIEW` | CPU-GLUE | N/A | Zero-copy alias; not a compute op. Handle in CPU. |
| `GGML_OP_PERMUTE` / `GGML_OP_TRANSPOSE` | ANE-NATIVE-C | `transpose` | Triggers memory copy. Minimize to 1 per attention block. |
| `GGML_OP_GET_ROWS` | ANE-BREAKS | `gather` | hollance: gather prevents ANE. Use `gather_along_axis` (iOS 17+) or CPU. |
| `GGML_OP_REPEAT` | ANE-NATIVE | `tile` | Used for GQA head expansion. |
| `GGML_OP_SCALE` | ANE-NATIVE | `mul` (scalar * tensor) | Scalar broadcast via `const` op + `mul`. |
| `GGML_OP_CONT` | CPU-GLUE | N/A | Contiguous copy. CPU. |
| `GGML_OP_CPY` | CPU-GLUE | N/A | Type conversion copy. CPU. |
| `GGML_OP_CAST` | ANE-NATIVE | `cast` | fp16 <-> fp32. |
| `GGML_OP_CLAMP` | ANE-NATIVE | `clip` | Native MIL op. Use before softmax to prevent fp16 overflow. |
| `GGML_OP_TOP_K` | ANE-NATIVE | `topk` | iOS 16+. |
| `GGML_OP_SLICE` (via slice ops) | ANE-NATIVE | `slice_by_index` | Native MIL op. |
| `GGML_OP_PAD` | ANE-NATIVE | `pad` | Native MIL op. |
| `GGML_OP_CONV_2D` | ANE-NATIVE-C | `conv` | Preferred formulation for matmul. 1x1 conv only. |
| `GGML_OP_CONV_2D_DW` | ANE-NATIVE | `conv` | Not used in transformers. |
| `GGML_OP_EXP` | ANE-NATIVE | `exp` | fp16 overflow risk above 11.09. |
| `GGML_OP_LOG` | ANE-NATIVE | `log` | Native. |
| `GGML_OP_SQRT` / `GGML_OP_SQR` | ANE-NATIVE | `sqrt` / `square` | Native. |
| `GGML_OP_ABS` | ANE-NATIVE | `abs` | Native. |
| `GGML_OP_SILU_BACK` | CPU-GLUE | N/A | Training only. Not in inference. |
| `GGML_OP_RMS_NORM_BACK` | CPU-GLUE | N/A | Training only. |
| `GGML_OP_FLASH_ATTN_EXT` | CPU-GLUE | N/A | Metal-only fused attention. Use MIL einsum/matmul decomposition on ANE. |
| `GGML_OP_TESSERA_PAGED_ATTN` | CPU-GLUE | N/A | Metal-only paged attention. Same. |
| `GGML_OP_TILE640_MATMUL` | ANE-NATIVE-C | `matmul` + dequant chain | T640 dequant via stock MIL ops (C1). Weight as fp16 after dequant. |
| `GGML_OP_TILE640_DEQUANT` | ANE-NATIVE-C | `constexpr_lut_to_dense` or stock ops | T640 dequant chain (Section 4.2). |
| `GGML_OP_DIAG_MASK_INF` | ANE-NATIVE-C | `add` (with -inf const) | Manual causal masking after Q@K^T, before softmax. |
| `GGML_OP_IM2COL` | CPU-GLUE | N/A | Layout transform for conv. CPU. |
| `GGML_OP_POOL_2D` | ANE-NATIVE-C | `avg_pool` / `max_pool` | Kernel <= 13, stride <= 2. Not used in transformers. |
| `GGML_OP_UPSCALE` | ANE-NATIVE-C | `upsample_bilinear` / `nearest` | Scale factor <= 2. Not used in transformers. |
| `GGML_OP_LEAKY_RELU` | ANE-NATIVE | `leaky_relu` | Native MIL op. |
| `GGML_OP_ARGSORT` | ANE-NATIVE | `argsort` | Native. |
| `GGML_OP_TRI` | CPU-GLUE | N/A | Causal mask construction. Build on CPU, pass as const. |
| `GGML_OP_FILL` | ANE-NATIVE | `fill` | Native. |
| `GGML_OP_ARANGE` | CPU-GLUE | N/A | Position generation. Build on CPU. |
| `GGML_OP_SSM_*` | CPU-GLUE | N/A | State-space models. Not on ANE. |
| `GGML_OP_RWKV_*` | CPU-GLUE | N/A | RWKV. Not on ANE. |

### 4.2 Composite Op Decompositions for ANE

#### 4.2.1 RMS Norm

No native `rms_norm` MIL op exists. Decompose as:

```
# Input: x, shape [B, S, C] (will be [B, C, 1, S] in ANE layout)
# Weight: gamma, shape [C]

x_fp32 = cast(x, fp32)                    # avoid fp16 overflow in square
x_sq   = mul(x_fp32, x_fp32)              # square
x_mean = reduce_mean(x_sq, axis=-1, keep_dims=True)  # mean of squares
x_eps  = add(x_mean, const(1e-6, fp32))   # epsilon
x_rsqrt = rsqrt(x_eps)                    # reciprocal sqrt
x_norm = mul(x_fp32, x_rsqrt)             # normalize
x_out  = cast(x_norm, fp16)               # back to fp16
result = mul(x_out, gamma)                # scale by weight

Total: 8 MIL ops (cast, mul, reduce_mean, add, rsqrt, mul, cast, mul)
```

The fp32 intermediate is critical: squaring fp16 values can overflow to inf,
which propagates through softmax (0/0). The export script
(`export-gemma4-prefill-bundle.py` line 41-44) already uses fp32 for the
variance computation.

#### 4.2.2 RoPE (Rotary Position Embedding)

RoPE is a composite of element-wise ops:

```
# Input: x, shape [B, S, heads, head_dim]
# Positions: pos, shape [B, S]

half = head_dim // 2
inv_freq = const(theta^(-arange(0, half, fp32) / half), fp16)  # precomputed
angles   = mul(pos[...,:,None], inv_freq)       # [B, S, half]
cos_vals = cos(angles)                           # [B, S, half]
sin_vals = sin(angles)                           # [B, S, half]
x_first  = slice(x, ..., 0, half)               # first half of head_dim
x_second = slice(x, ..., half, head_dim)         # second half

out_first  = sub(mul(x_first, cos_vals), mul(x_second, sin_vals))
out_second = add(mul(x_second, cos_vals), mul(x_first, sin_vals))
result     = concat(out_first, out_second, axis=-1)
```

**concat risk**: The final `concat` is ANE-BREAKS (Orion #1). Workaround:
instead of slicing + rotating + concatenating, express RoPE as a single fused
operation using `conv` with pre-rotated weights, or restructure the layout so
that the two halves are interleaved rather than concatenated. The Tessera
export script (lines 53-60) uses torch.cat, which coremltools may decompose
into ANE-compatible ops during conversion.

Alternative: Use the `einsum` formulation that avoids the concat entirely:

```
# Precompute rotation matrix as a [head_dim, head_dim] const
# out = einsum("bshd,hh->bshd", x, rotation_matrix[pos])
```

This avoids both the slice and the concat, but requires per-position rotation
matrices (or a small set of cached matrices).

#### 4.2.3 GELU (Tanh Approximation)

GELU exact is ANE-BREAKS (Orion #10). Must use tanh approximation:

```
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

sqrt_2_over_pi = const(sqrt(2.0 / pi), fp16)  # 0.7978845608
x_cubed         = mul(x, mul(x, x))
inner           = mul(sqrt_2_over_pi, add(x, mul(0.044715, x_cubed)))
tanh_val        = tanh(inner)
result          = mul(0.5, mul(x, add(1.0, tanh_val)))

Total: ~8 MIL ops (mul, mul, add, mul, tanh, add, mul, mul)
```

The export script already uses `F.gelu(x, approximate="tanh")` (line 117),
which coremltools decomposes into this pattern during conversion.

#### 4.2.4 Scaled Dot-Product Attention (SDPA)

Do NOT use the iOS 18 `scaled_dot_product_attention` MIL op for causal
attention: causal masks are silently ignored (Orion #6). Decompose:

```
# Q: [B, heads, S, head_dim]
# K: [B, kv_heads, S, head_dim] (possibly cached)
# V: [B, kv_heads, S, head_dim] (possibly cached)
# Causal mask: built on CPU, passed as const

Q_t = transpose(Q, [0, 1, 3, 2])               # [B, heads, head_dim, S]
scores = matmul(Q_t, K)                         # [B, heads, S_kv, S]
scores = mul(scores, const(1.0 / sqrt(head_dim)))  # scale
scores = add(scores, causal_mask)                # apply -inf mask (CPU-GLUE)
probs  = softmax(scores, axis=-1)                # [B, heads, S_kv, S]
# Clamp probs input: the mask -inf can cause fp16 issues in exp

# For GQA: expand K/V from kv_heads to heads
if kv_heads < heads:
    K_expanded = repeat(K, heads // kv_heads, axis=1)
    V_expanded = repeat(V, heads // kv_heads, axis=1)

attn_out = matmul(V_expanded_t, probs)          # or einsum
result   = transpose(attn_out, [0, 1, 3, 2])
```

Alternative (Apple ml-ane-transformers pattern):
```
# Use einsum to avoid the transpose on K:
# einsum("bchq,bkhc->bkhq", Q, K) directly produces attention scores
# Then softmax, then einsum("bkhq,bkhc->bhcq", probs, V) for attended output
```

This avoids the explicit K transpose, saving one memory copy.

#### 4.2.5 Tile640 Dequant (v1, Stock MIL Ops)

Per the C1 decision: stock ops for v1, custom op as v2.

```
# Inputs (all const ops in the .mlpackage):
#   packed:     uint8  [out, pages * W]     -- ternary trits packed into bytes
#   page_scales: fp16   [out, pages]         -- per-page scaling
#   lane_scales: int8   [out, pages * L]     -- per-lane scaling within page
#   act_scale:  fp16   [in]                 -- per-channel activation scale

# Step 1: Unpack trits to int8
#   (constexpr_lut_to_dense or manual reshape + cast)

# Step 2: Page scaling
#   Broadcast page_scales from [out, pages] to [out, pages * W]
#   scaled = unpacked * page_scales_broadcasted

# Step 3: Lane scaling
#   scaled = scaled * lane_scales_broadcasted

# Step 4: Act scale (pre-matmul)
#   final = scaled * act_scale_broadcasted

# Step 5: Outlier replacement (scatter + select)
#   outlier_mask = scatter(outlier_cols, 1.0, shape=[out, in])
#   base_out = select(outlier_mask, final, outlier_vals_scattered)

Total: ~15-30 MIL ops per projection (estimate from conversion design Section 3.1)
```

For v2 custom op path (gated on >5% dequant time):
```
op_type: "tessera_t640_dequant"
# Signature per conversion design Section 3.4
# Uses private API (App Store risk)
```

### 4.3 IOSurface + Metal Event Architecture

Based on the existing Tessera ane-mtp code and the ANE research findings:

#### 4.3.1 IOSurface Allocation Parameters

The existing ane-mtp.mm already implements the correct IOSurface pattern
(lines 69-76):
```
Properties:
  kIOSurfaceWidth:       = rounded_size (16 KB page aligned)
  kIOSurfaceHeight:      = 1
  kIOSurfaceBytesPerElement: = 1
  kIOSurfaceBytesPerRow: = rounded_size
  kIOSurfaceAllocSize:   = rounded_size
```

For the ggml-ane backend, the allocation policy should be:

1. **Minimum allocation**: 16 KB page alignment (existing code). BUT enforce
   a floor of 49 KB for any tensor that will be passed to ANE as a sole input
   or output (Orion #4). The simplest approach: always round up to 64 KB
   (next power-of-2 multiple of 16 KB that exceeds 49 KB).
2. **Multi-input uniformity**: When a function takes multiple inputs, ALL
   input IOSurfaces must be allocated at the **maximum size** across all
   inputs (Orion #18). This applies to the prefill slab function which takes
   both token_ids and positions.
3. **Multi-output uniformity**: When a function produces multiple outputs,
   ALL output IOSurfaces must be allocated at the maximum size (Orion #2).
   The prefill slab produces hidden_states, K, and V -- pad all to the
   maximum.
4. **Sequence padding**: For decode (seq=1), pad the sequence dimension to
   at least 16 elements to exceed the 49 KB minimum for typical hidden
   sizes. For hidden_size=3072 (Gemma 2 9B), fp16: `1 * 3072 * 1 * 16 =
   98304 bytes > 49 KB`.
5. **KV cache state width**: Must be a multiple of 32 (Apple Developer Forums
   constraint). Pad the head_dim * kv_heads dimension to the next multiple of
   32 if not already aligned.

#### 4.3.2 Zero-Copy Sharing Between ANE and Metal

The zero-copy pipeline on Apple Silicon:

```
IOSurface (shared DRAM)
  |
  +---> MLMultiArray (initWithDataPointer: ... deallocator: nil)
  |       |
  |       +---> Core ML predict (ANE reads directly from IOSurface)
  |
  +---> MTLBuffer (device.newBufferWithBytesNoCopy: length: options: ...)
          |
          +---> Metal compute shader (GPU reads directly from IOSurface)
```

The key API calls:
```objc
// Create IOSurface
IOSurfaceRef surface = IOSurfaceCreate(properties);
void * base = IOSurfaceLock(surface, 0, nullptr);

// Wrap for Core ML
MLMultiArray * mlArray = [[MLMultiArray alloc]
    initWithDataPointer:base
                  shape:@[@(B), @(C), @(1), @(S)]
               dataType:MLMultiArrayDataTypeFloat16
                strides:contiguous_strides(shape)
            deallocator:nil  // IOSurface owns the memory
                  error:&err];

// Wrap for Metal
MTLBuffer * metalBuf = [device newBufferWithBytesNoCopy:base
                                               length:byteLength
                                              options:MTLResourceStorageModeShared];
```

Source: https://www.macinternals.app/en/blog/iosurface-in-depth;
Chromium issue #333392274; webmachinelearning/webnn #542.

**Synchronization**: On Apple Silicon (unified memory), cache coherency is
automatic for IOSurface-backed memory. The CPU, GPU, and ANE all share the
same physical DRAM with hardware cache coherency. However, explicit
synchronization is needed for **ordering** (not coherency):

```
// ANE writes output -> Metal reads input
// The IOSurface lock/unlock provides ordering:
IOSurfaceLock(surface, 0, nullptr);   // CPU: lock for write
// ... write data ...
IOSurfaceUnlock(surface, 0, nullptr); // CPU: unlock -> signals coprocessors

// For Metal, use MTLSharedEvent for fine-grained sync:
id<MTLSharedEvent> event = [device newSharedEvent];
// Metal shader signals when read is complete
[encoder signalEvent:event value:completion_id];
// Core ML predict (synchronously, on serial dispatch queue)
// blocks until previous Metal work is done
```

The existing ane-mtp.mm uses a serial `dispatch_queue_t` (line 773) to
serialize all ANE predictions, which implicitly provides ordering. For the
ggml-ane backend with Metal interop, the synchronization chain should be:

```
Metal compute (ggml-metal)          ANE compute (ggml-ane)
       |                                     |
       +-- MTLSharedEvent.signal ---------->|
       |                                     |
       +-- MTLSharedEvent.wait  <-----------+
       |                                     |
```

The `ggml_backend_event_t` vtable (ggml-backend-impl.h lines 132-136)
provides `event_record` and `event_wait` hooks that the backend can use to
coordinate this.

#### 4.3.3 Accelerate (vDSP/vForce) for CPU-GLUE Ops

Ops classified as CPU-GLUE should be handled by Accelerate on the CPU:

| Op | Accelerate Function | Notes |
|-----|-------------------|-------|
| `GGML_OP_VIEW` | No-op | Zero-cost alias. |
| `GGML_OP_CONT` | `memcpy` | Or `vDSP_vadd` for type conversion. |
| `GGML_OP_CPY` | `ggml_fp16_to_fp32_row` / `ggml_fp32_to_fp16_row` | Existing functions in ggml. |
| `GGML_OP_TRI` | Loop construct | Causal mask. Build once, cache. |
| `GGML_OP_ARANGE` | `vDSP_vramp` | Position generation. |
| `GGML_OP_FLASH_ATTN_EXT` | N/A | Falls to ggml-metal. |
| `GGML_OP_GET_ROWS` | `vDSP_vindex` | Embedding lookup. CPU is fine for single-token decode. |
| `GGML_OP_IM2COL` | Loop construct | Layout transform. |
| `GGML_OP_ROPE` (if concat breaks ANE) | `vDSP_zvma` + `vDSP_zrvm` | Fall back to CPU if concat is needed and ANE rejects it. |

The vDSP RMS norm (mentioned in maderix/ANE as 10x faster than naive) is:
```
vDSP_vsq(x, 1, x_sq, 1, length);      // x^2
vDSP_meanv(x_sq, 1, &mean, length);    // mean(x^2)
float inv_rsqrt = 1.0 / sqrtf(mean + eps);
vDSP_vsmul(x, 1, &inv_rsqrt, x, 1, length);  // normalize
vDSP_vmul(x, 1, gamma, 1, x, 1, length);      // scale
```

### 4.4 Failure Mode Catalog

Every known way the ANE compilation or execution can silently fail, with
detection and mitigation:

| # | Failure Mode | Symptom | Detection | Mitigation |
|---|-------------|---------|-----------|------------|
| F1 | Silent GPU/CPU fallback | Slower than expected throughput | IOReport ANE power near 0 mW during inference | Check ANE power after warmup. If 0, ops fell off ANE. Log fallback. |
| F2 | `concat` causes GPU fallback | Inference runs but on GPU not ANE | IOReport shows GPU active, ANE inactive during concat-containing functions | Decompose concat. For RoPE: use interleaved layout or einsum. |
| F3 | GELU exact causes GPU fallback | Same as F2 | Same detection | Use tanh approximation. Verify export script uses `approximate="tanh"`. |
| F4 | Gather causes GPU fallback | Same as F2 | Same detection | Use `gather_along_axis` (iOS 17+) or CPU fallback for embedding lookup. |
| F5 | SDPA causal mask silently ignored | Wrong attention output, plausible but incorrect | Parity check: compare ANE output to CPU reference for short sequences | Decompose SDPA with manual causal masking. |
| F6 | Multi-output size mismatch | Error 0x1d at eval (Orion #2) | Core ML prediction returns nil with error | Pad all outputs to max size before MIL construction. |
| F7 | Input below 49 KB | Error 0x1d at eval (Orion #4) | Core ML prediction returns nil with error | Pad sequence dimension to >= 16. |
| F8 | Input name ordering mismatch | Silent wrong data (Orion #19) | Output values are garbage | Name MIL inputs in alphabetical order. |
| F9 | Output name ordering mismatch | Silent wrong data (Orion #3) | Same as F8 | Name MIL outputs in alphabetical order. |
| F10 | fp16 overflow in norm/softmax | NaN or inf in output | Output contains NaN/inf | Clamp activations to [-65504, 65504] before norm. Use fp32 intermediates. |
| F11 | KV cache width not multiple of 32 | ANE eval fails for stateful programs | Core ML prediction returns nil | Pad KV width to multiple of 32. |
| F12 | 32K-channel conv (LM head) | Compilation failure (Orion #16) | `xcrun coremlcompiler` fails | CPU fallback for LM head projection. Or chunk into smaller convs. |
| F13 | State tensor dims not multiple of 32 | Stateful program fails on ANE (Apple Forums) | Prediction error | Pad all state dimensions to multiples of 32. |
| F14 | matmul transpose flags as inline bools | MIL rejection (Orion #12) | Conversion tool build error | Emit `const` ops for transpose flags. Prism already does this. |
| F15 | Dynamic shapes prevent ANE | Core ML falls to CPU | Slow inference | Use fixed-shape sequence buckets (existing pattern). |
| F16 | ANE thermal throttling | Throughput drops, thermal state >= 2 | `NSProcessInfo.thermalState` check | Reduce batch size. Switch to Metal. |
| F17 | 119-compile limit (direct path only) | Silent fail/crash after ~119 compiles | Process crash counter | NOT applicable to public path (models are pre-compiled). |
| F18 | Conv bias parameter | MIL rejection (Orion #13) | Conversion tool build error | Separate `add` after conv. |
| F19 | Over-allocated buffer read as [1,C,1,S] | Silent wrong data (Orion #20) | Output values are garbage | Write packed data at buffer start. |
| F20 | Output var references dead node | Invalid program after optimization | Conversion tool build error | Validate SSA liveness after optimization. |

### 4.5 Implementation Recommendations for Slices 1-4

#### Slice 1: Backend Registration + Buffer Type

**Recommendations**:
1. Register the ANE backend at `ggml/src/ggml-backend-reg.cpp` alongside
   `ggml-metal`. Device name: "ANE". Description: "CoreML (ANE-first, iOS)".
2. Buffer type: IOSurface-backed, with 16 KB page alignment and 64 KB
   minimum allocation (exceeds the 49 KB floor).
3. Buffer alignment: 16 KB (to match IOSurface page alignment).
4. `is_host` returns false (IOSurface memory is not in CPU address space
   without explicit mapping).
5. The backend should only register on Apple Silicon + iOS (not macOS, where
   Metal is faster for prefill; not Intel Mac or Linux).

**Rationale**: The existing ane-mtp.mm IOSurface arena code (lines 30-100)
is the reference implementation. The ggml-ane buffer type wraps this arena.

#### Slice 2: Op Dispatch + Compute Graph

**Recommendations**:
1. `supports_op`: Return true only for ops classified ANE-NATIVE or
   ANE-NATIVE-C in Section 4.1. Return false for ANE-BREAKS ops.
2. For `GGML_OP_GET_ROWS` (embedding lookup): Fall to CPU. The CPU does a
   single vector gather, which is sub-microsecond and not worth a Core ML
   dispatch.
3. For `GGML_OP_CONCAT`: If the concat is part of RoPE, fall to CPU or use
   the interleaved-layout decomposition. Otherwise, fall to CPU.
4. For `GGML_OP_MUL_MAT` / `GGML_OP_TILE640_MATMUL`: Emit as `conv` (1x1)
   in the MIL program for 3x throughput. The weight tensor layout must be
   reshaped from `[out, in]` to `[out, in, 1, 1]` (conv kernel format).
5. For `GGML_OP_FLASH_ATTN_EXT`: Fall to ggml-metal. The Metal fused
   attention kernel is highly optimized and cannot be replicated on the ANE
   through the public Core ML path. The ANE gets the decomposed SDPA
   (Section 4.2.4).
6. For `GGML_OP_SOFT_MAX`: Clamp the input to [-65504, 65504] before
   calling softmax. The `exp()` in softmax overflows fp16 at ~11.09, and
   the fp16 max is 65504.
7. Input/output naming: All MIL inputs and outputs MUST be named in
   alphabetical order to avoid the binding mismatch (Orion #3, #19).

#### Slice 3: Multi-Function Support + KV Cache

**Recommendations**:
1. Use the `MLMultiFunctionProgram` pattern (iOS 18+) for per-sequence-bucket
   prefill functions. The existing export script already produces this.
2. KV cache: Use `MLState` (public API, C6 decision). The state tensors must
   have dimensions that are multiples of 32 (Apple Forums constraint).
3. For the stateful decode path, emit a separate function in the
   multifunction bundle that takes the new token and reads/updates K/V
   state. The existing Prism `coreml_state.rs` pattern is the reference.
4. KV cache width padding: `kv_heads * head_dim` must be padded to the next
   multiple of 32 if not already aligned. For Gemma 4 9B: kv_heads=4,
   head_dim=256, width=1024 (already aligned). For other models, this may
   require padding.
5. Multi-output uniformity: The prefill slab returns hidden_states + K + V.
   All three must be padded to the maximum byte size. The hidden_states
   `[batch, seq, hidden]` is typically the largest.

#### Slice 4: Fallback + Telemetry

**Recommendations**:
1. Fallback: When `MLModel.predictionFromFeatures:` returns nil, fall back
   to ggml-metal with a warning. Log the error description (C7 decision).
2. ANE residency detection: After warmup, sample IOReport ANE power. If 0 mW
   during inference, the model fell off the ANE. Log a warning with the
   function name.
3. The `--device ane` CLI flag selects the ANE backend. The `--device metal`
   flag skips the ANE attempt. The default on iOS is `ane` (with Metal
   fallback); on macOS the default is `metal`.
4. For the 49 KB minimum: The sequence buckets in the export script
   (128, 256, 512, 1024) all produce tensors well above 49 KB. The decode
   path (seq=1) is the risk point. Pad the decode sequence dimension to
   16 elements: for hidden_size=3072, fp16: `1 * 3072 * 1 * 16 = 98304
   bytes > 49 KB`.
5. Thermal throttling: Monitor `NSProcessInfo.thermalState`. If >= 2
   (serious), reduce batch size or switch to Metal for the next few tokens.

---

## Appendix A: Source References

### CoreML / coremltools
- MIL Ops API Reference: https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
- Multifunction Models Guide: https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html
- ML Programs vs Neural Networks: https://apple.github.io/coremltools/docs-guides/source/comparing-ml-programs-and-neural-networks.html
- Typed Execution Guide: https://apple.github.io/coremltools/docs-guides/source/typed-execution.html
- coremltools GitHub: https://github.com/apple/coremltools

### Apple ml-ane-transformers
- GitHub: https://github.com/apple/ml-ane-transformers
- Research Article: https://machinelearning.apple.com/research/neural-engine-transformers

### Orion
- Paper (arXiv): https://arxiv.org/html/2603.06728v1
- PDF: https://arxiv.org/pdf/2603.06728
- Source Code: https://github.com/mechramc/Orion
- Blog Post: https://ramchandk.com/blog/orion-programming-apple-neural-engine

### maderix/ANE
- Source: https://github.com/maderix/ANE
- Substack Article: https://maderix.substack.com/p/inside-the-m4-apple-neural-engine

### ANEForge
- Paper: https://arxiv.org/abs/2606.17090
- Source: https://github.com/sbryngelson/ANEForge

### hollance/neural-engine
- Unsupported Layers: https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md

### IOSurface Zero-Copy
- MacInternals Article: https://www.macinternals.app/en/blog/iosurface-in-depth
- Chromium Issue: https://issues.chromium.org/issues/333392274
- webnn Issue #542: https://github.com/webmachinelearning/webnn/issues/542
- WWDC25 Session 262: https://developer.apple.com/videos/play/wwdc2025/262/

### CoreML ANE Issues
- coremltools #513 (addBroadcastable): https://github.com/apple/coremltools/issues/513
- coremltools #2353 (iOS 18 regression): https://github.com/apple/coremltools/issues/2353
- coremltools #2359 (Mish fp16): https://github.com/apple/coremltools/issues/2359
- coremltools #2687 (softplus fp16): https://github.com/apple/coremltools/issues/2687
- coremltools #337 (LSTM BNNS): https://github.com/apple/coremltools/issues/337
- Apple Forums (state tensor multiple of 32): https://developer.apple.com/forums/tags/core-ml?page=2

### Tessera Codebase (D1 Substrate)
- `common/ane-mtp.h` - ANE MTP program header (233 lines)
- `common/ane-mtp.mm` - ANE MTP runtime (2556 lines)
- `docs/tessera-coreml-conversion-design.md` - Conversion design (1382 lines)
- `tools/ane-mtp/export-gemma4-prefill-bundle.py` - Python export script
- `ggml/src/ggml-metal/ggml-metal.cpp` - Metal backend (vtable template)
- `ggml/src/ggml-backend-impl.h` - Backend vtable definitions
- `ggml/include/ggml.h` - Op enum (lines 484-601)

## Appendix B: ANE Op Classification Decision Tree

For implementation agents evaluating whether a new ggml op can target the ANE:

```
Is the op in the ANE-NATIVE or ANE-NATIVE-C list (Section 4.1)?
  |
  +-- YES --> Does it require specific dtype/layout/shape constraints?
  |            |
  |            +-- YES --> Are constraints satisfiable? (fp16, [B,C,1,S], padded dims)
  |            |             |
  |            |             +-- YES --> ANE-NATIVE-C. Implement with constraints.
  |            |             +-- NO  --> Fall back to CPU/GPU.
  |            |
  |            +-- NO  --> ANE-NATIVE. Implement directly.
  |
  +-- NO --> Is the op a cheap CPU-GLUE op? (view, cpy, cast, tri, arange)
             |
             +-- YES --> CPU-GLUE. Use Accelerate. Negligible latency impact.
             |
             +-- NO  --> ANE-BREAKS. Must fall back. Log the fallback.
                        Consider decomposing into ANE-NATIVE ops.
```

## Appendix C: MIL Builder Call Reference for Key Decompositions

The following are the exact `coremltools.converters.mil.mil.builder` calls for
the composite ops, which the C++ conversion tool must replicate in its
protobuf-based builder:

**RMS Norm** (8 ops):
```python
x_fp32 = mb.cast(x, dtype="fp32")
x_sq   = mb.mul(x_fp32, x_fp32)
x_mean = mb.reduce_mean(x_sq, axes=[-1], keep_dims=True)
x_eps  = mb.add(x_mean, const_val(1e-6, dtype="fp32"))
x_rsqrt = mb.rsqrt(x_eps)
x_norm = mb.mul(x_fp32, x_rsqrt)
x_fp16 = mb.cast(x_norm, dtype="fp16")
result = mb.mul(x_fp16, gamma)
```

**GELU tanh approximation** (8 ops):
```python
c       = const_val(0.7978845608, dtype="fp16")  # sqrt(2/pi)
c044    = const_val(0.044715, dtype="fp16")
half    = const_val(0.5, dtype="fp16")
one     = const_val(1.0, dtype="fp16")
x3      = mb.mul(x, mb.mul(x, x))
inner   = mb.mul(c, mb.add(x, mb.mul(c044, x3)))
tanh_v  = mb.tanh(inner)
result  = mb.mul(half, mb.mul(x, mb.add(one, tanh_v)))
```

**SDPA (decomposed, with manual causal mask)** (~6 ops + einsum):
```python
scores  = mb.einsum([Q, K], "bchq,bkhc->bkhq")  # or mb.matmul(Q_t, K)
scores  = mb.mul(scores, const_val(1.0 / sqrt(head_dim)))
scores  = mb.add(scores, causal_mask_const)        # -inf where causal
probs   = mb.softmax(scores, axis=-1)
attn    = mb.einsum([probs, V], "bkhq,bkhc->bhcq")
result  = mb.transpose(attn, perm=[0, 1, 3, 2])
```

**Conv-based linear projection** (2 ops):
```python
# Replace matmul(x, W) where x is [B, C_in, 1, S] and W is [C_out, C_in]
# Reshape W to conv kernel [C_out, C_in, 1, 1]
W_conv = mb.reshape(W, [out_dim, in_dim, 1, 1])
result = mb.conv(x, W_conv, pad_type="valid", strides=[1, 1], dilations=[1, 1])
# Add bias separately (Orion #13):
result = mb.add(result, bias)  # if needed
```
