# Tessera CoreML Conversion + Runtime — Scoping Design

Design only. No implementation code. Targets gemma 4 12b unified (text +
vision + audio), Tessera-quantized, running both prefill AND decode on
the ANE via CoreML on iPhone.

> Roadmap alignment: the runtime-aware proxy-objective research
> (2026-07-30) validates the IOReport runtime-telemetry + receipts thesis
> and the Metal auto-fallback (C7): a practitioner report of ~23% INT8
> accuracy variance across five Snapdragon chipsets is direct external
> evidence that fidelity lives at the runtime boundary and varies per
> hardware. The `modality_scales` translation (Section 2.5) carries the
> regime axis into the CoreML artifact. No structural change. See
> [`research-alignment-2026-07-30.md`](research-alignment-2026-07-30.md)
> Section 4.5.

## Architectural decisions (locked by the prior conversation, not revisited here)

1. Full CoreML inference as a first-class backend, peer to ggml-metal.
   Both prefill AND decode run on the ANE via CoreML, not prefill
   (ANE) + decode (Metal).
2. iPhone is the primary surface (not Mac). The hero metric is
   sustained battery draw per token on a phone, not first-token
   latency on a Mac.
3. IOReport is the runtime telemetry source (ANE power, GPU power,
   DRAM power, battery current, thermal state).
4. The conversion tool is STATELESS: reads a Tessera-quantized GGUF,
   writes a `.mlmodelc`, exits. No inference, no state, no KV cache.
5. The runtime (`ggml-coreml`) is a separate workstream (G7 of the
   prior C++ port plan); this scoping covers the conversion tool AND
   how the runtime will load the `.mlmodelc` and instrument it with
   IOReport.

### Architect decisions on the 10 open questions (2026-07-30)

The scoping agent surfaced 10 questions in section 9. The architect
(2026-07-30) locked the following answers; the agent's leans in
section 9 are superseded by the items below.

**C1. Tessera dequant: CoreML custom op vs stock ops.** Locked:
stock ops for v1, custom op as v2 gated on >5% dequant time.
Custom op is private API (App Store risk). The 5% threshold is
measured; if the regression is real, we revisit. The 2% tightening
I floated in my prediction is NOT applied — the architect stays
at 5%.

**C2. `.mlmodelc` generation timing.** Locked: at quantize time
(Mac), bundle the `.mlmodelc` in the `.app`. The 30-120s
conversion is offline and reproducible; the iPhone user opens
the app and the model is already there.

**C3. IOReport channel stability across iOS versions.** Locked:
research, surface findings, document fallback. ANE power in
"Energy Model" is stable across iOS 15+; ANE activity in DVFS
is more variable. Use power when activity is unavailable.

**C4. Battery attribution granularity.** Locked: per-session for
v1, per-token as v2. The hero metric is mWh/token over a 30-minute
flight test; per-session is the demo number, per-token is
research-grade (per-token rows land in the sidecar regardless;
per-session aggregation is a query over the sidecar).

**C5. `ggml-coreml` memory layout.** Locked: MMAP for the weight
blobs, RAM for the activations. Standard Apple ML stack pattern.

**C6. KV cache: full CoreML state API vs custom.** Locked: full
state API (`MLState`). Public API as designed; custom is more
code for no benefit. The Prism `coreml_state.rs` proves it works.

**C7. Backend fallback (CoreML fail -> Metal).** Locked: yes,
log the fallback. `--device metal` skips the attempt. The
fallback is silent in production; the test harness enables verbose
logging.

**C8. Multimodal `.mlmodelc` shape: 3 per modality vs 1 with
runtime act_scale.** Locked: **1 `.mlmodelc` with runtime act_scale
for v1, 3 as a v2 packaging optimization if profiling shows the
input switch is hot.** This is a PUSHBACK from the agent's lean
(3 for v1, 1 for v2). Rationale:
- 3 separate `.mlmodelc` for gemma 4 12b at 3-4 GB effective bits
  is 9-12 GB bundled in the `.app` (most of an iPhone).
- 3 cold-start loads when the user switches modalities mid-chat.
- The architect's earlier "BOTH modality ID + per-modality
  components" decision already makes the act_scale a runtime
  choice; baking it in at convert time reverses that.
- The Prism pattern is "one canonical reality." One `.mlmodelc`
  is the honest expression.
- The 3-package v2 is gated on profiling: if the runtime input
  switch is hot, we generate 3 separate `.mlmodelc` files (one
  per modality) and the iPhone app picks based on the user's
  task. This is a packaging optimization, not a v1 requirement.

**C9. Conversion tool: `coremlcompiler` CLI vs public
`+compileModelAtURL:` API.** Locked: public API. App Store safe;
testable; what Apple recommends for app-side compilation.

**C10. Tessera config source: GGUF metadata vs sidecar JSON.**
Locked: GGUF metadata is the primary source; sidecar JSON is
an override for non-Tessera-aware tools. The conversion tool
warns if they disagree.

The conversion tool lives in `tools/quantize/tessera-to-coreml/` (new
binary, alongside `llama-quantize`). The runtime lives in
`ggml/src/ggml-coreml/` (new directory, alongside `ggml-cpu`,
`ggml-metal`).

References throughout this doc:

- Prism source: `/Users/user/Developer/GitHub/prism-engine/crates/prism-ane/`
- Tessera C++ port design: `docs/c++-port-design.md`
- Tessera multi-modal calibration: `docs/multimodal-calibration-design.md`
- Tessera W4A4 calibration: `docs/w4a4-calibration-design.md`
- llama.cpp backend registry: `ggml/src/ggml-backend-reg.cpp`,
  `ggml/src/ggml-metal/ggml-metal.cpp:819-895`

## 1. Prism coremltools-rs inventory

Prism Engine ships a Rust implementation of a CoreML converter
(`prism-ane` crate) and a CoreML bridge (`coreml_bridge.mm`). The
converter is the closest existing analogue to what Tessera needs.
This section inventories it file by file.

Total LoC summary (in scope vs out of scope for the Tessera port):

| File                          | LoC  | Scope              | Reason                                    |
|-------------------------------|-----:|--------------------|-------------------------------------------|
| `mil_builder.rs`              | 2208 | IN scope (port)    | Pure MIL protobuf construction in Rust    |
| `mil_gen_full.rs`             |  751 | IN scope (port)    | Full prefill transformer MIL generator    |
| `mil_layer_programs.rs`       |  278 | IN scope (port)    | Per-layer fused program constructors      |
| `mil_helpers.rs`              |  208 | IN scope (port)    | Small op constructors (pow, rsq, etc.)    |
| `mlpackage.rs`                |  299 | IN scope (port)    | .mlpackage directory + Manifest.json      |
| `coreml_bridge.mm`            |   81 | IN scope (port)    | ObjC++ model load + predict bridge        |
| `lib.rs` (pack/unpack mlmodelc)|  89 | Reference only     | .mlmodelc blob packer; we don't need it   |
| `compile_full_model.rs`       |  546 | OUT of scope       | Orchestrator that calls into MIL+mlpackage; not stateless |
| `coreml_bridge.rs`            |  184 | OUT of scope       | Stateful predict API; belongs to runtime  |
| `coreml_state.rs`             |  266 | OUT of scope       | Stateful MLState API; belongs to runtime  |
| `coreml_audit.rs`             |   40 | IN scope (port)    | Tiny ANE compliance check, but probably unnecessary on iOS |
| `lib.rs` (arena, info)        |   85 | OUT of scope       | IOSurface arena; runtime-only             |
| **IN scope subtotal**         | **~3864** |               |                                           |
| **OUT of scope subtotal**     | **~1076** |               |                                           |
| **Reference only**            |   89 |                    |                                           |

LoC column is `wc -l` from the working tree; the column is the
*in-scope-for-Tessera-port* judgment, not the total source size.

### 1.1 `mil_builder.rs` (2,208 LoC, IN scope)

What it does. Pure-Rust MIL program builder using the
`coreml-proto` + `prost` crates. Constructs `mil_spec::Program`
protobufs without Python/coremltools. Generates SSA value names
automatically (`{hint}_{counter}`) and produces a valid MLProgram
that `coremlcompiler` can ingest.

Key public types / functions (in scope):

- `MilBuilder` (struct) — the central builder. Holds `ops`,
  `value_types`, `inputs`, `block_outputs`, `counter`, `opset`.
- `MilBuilder::new(function_name)` — constructor, default opset
  `CoreML9`.
- `MilBuilder::batch_size(n)` — sets the ANE batch broadcasting
  factor (1, 2, or 4). Prism uses this for multi-row matmul.
- `MilBuilder::input(name, dtype, shape)` — declare an input
  tensor.
- `MilBuilder::operation(op, output_type)` — append a hand-built
  op to the SSA chain. Used by every primitive below.
- `MilBuilder::const_f32 / const_f16 / const_uint8 / const_i32` —
  declare a const op with embedded immediate values.
- `MilBuilder::matmul(a, b)` — matmul op with auto shape inference
  on the `(M, N)` output.
- `MilBuilder::matmul_transpose_y(a, b)` — matmul with
  `transpose_y=true` for the standard LLM `[M,K] @ [N,K]^T = [M,N]`
  pattern.
- `MilBuilder::add / mul / sub` — element-wise binary ops.
- `MilBuilder::silu / softmax / sigmoid` — element-wise activations.
- `MilBuilder::gather(params, indices, axis)` — gather with
  automatic output shape inference (params prefix, indices dims,
  params suffix).
- `MilBuilder::topk(x, k, axis)` — two-output (values, indices)
  topk op; Prism uses this for KV compaction.
- `MilBuilder::make_state / read_state / write_state` — the CoreML
  state API for KV cache. These are the **stateful** ops that
  Prism uses to track K/V across decode steps.
- `MilBuilder::slice_update(input, source, starts)` — writes a
  slice into a state tensor. Used for KV cache append.
- `MilBuilder::scaled_dot_product_attention(query, key, value, mask,
  scale)` — the fused SDPA op. Prism feeds it the K cache, V cache,
  and Q from the current step.
- `MilBuilder::quantize / dequantize` — the 8-bit per-row quantize
  / 16-bit per-row dequantize ops (NOT the Tessera dequant; see
  Section 3).
- `MilBuilder::reshape / transpose / concat / slice / repeat` —
  tensor shape and layout ops.
- `MilBuilder::output(name)` — declare a block output.
- `MilBuilder::build()` — produces a `mil_spec::Program` with
  SSA validation. Returns `MilBuildError` on missing
  inputs/outputs.
- `MilBuilder::to_mil_text()` — debug printer for the generated
  MIL program.

Data flow. The builder accumulates ops in order; each op's
`named_arg` references an SSA value that must be either an `input()`
or a previous op's output. `build()` validates that every
referenced name is defined and that every `output()` name exists.

CoreML APIs called. None directly — this is a pure protobuf
constructor. The output is a `mil_spec::Program` (one of the
CoreML MLProgram protobuf messages) ready to be serialized via
`prost::Message::encode` and wrapped in a `proto::Model` by
`mlpackage.rs`.

Stateful vs stateless split. **All of `mil_builder.rs` is in
scope for the conversion tool** — it constructs the op graph
and that graph is identical for stateless prefill and stateful
decode (the decode path just adds `make_state`/`read_state`/
`write_state` ops). For Tessera v1 the conversion tool emits a
**stateless** program (no state ops) and the runtime
(`ggml-coreml`) manages KV cache via the CoreML state API. The
port preserves both paths so v2 decode-on-ANE can reuse the
stateless op generators.

### 1.2 `mil_gen_full.rs` (751 LoC, IN scope)

What it does. Generates a full prefill MIL program for all
transformer layers of a model. Composes the `MilBuilder`
primitives into a single `build_full_prefill_mil` function that
encodes one RMSNorm, QKV projection, RoPE, SDPA, O projection,
residual, RMSNorm, MLP (gate+up+down) for every layer, then
final RMSNorm + LM head. This is the canonical
"build me a full model" entry point.

Key public types / functions (in scope):

- `build_full_prefill_mil(...)` — the all-in-one entry point.
  Takes hidden_dim, n_heads, n_kv_heads, head_dim, n_layers,
  vocab_size, max_seq_len, norm_eps, and a `LayerMILWeights`
  slice with codebook/indices for every projection.
- `LayerMILWeights` (struct) — codebook + packed indices for
  each of q, k, v, o, gate, up, down for one layer. Tessera's
  equivalent is the 9-component cluster (see Section 2.2).

Internal helpers (private, but in scope to port): `op_gather`,
`op_lut_to_dense`, `op_const_f16`, `op_const_uint8`, `op_matmul`,
`op_add`, `op_mul`, `op_sub`, `op_concat`, `op_slice_last_dim`,
`op_repeat_interleave`, `op_const_scalar`, `op_reshape`,
`op_quantize`, `op_dequantize`, `op_slice_update`,
`op_make_state`, `op_read_state`, `op_write_state`,
`op_scaled_dot_product_attention`, `rms_norm_3d`.

Data flow. `build_full_prefill_mil` calls `MilBuilder` in a
chain: input declarations -> embedding lookup -> for each layer:
RMSNorm -> Q/K/V matmul -> RoPE -> make_state K/V -> SDPA -> O
matmul -> residual -> RMSNorm -> MLP (gate matmul -> SiLU ->
mul by up matmul -> down matmul) -> residual -> final RMSNorm
-> LM head matmul -> output. This is the standard decoder
transformer block; Tessera reuses the same pattern but emits
**no state ops** in v1 and feeds pre-dequantized weights (see
Section 3).

CoreML APIs called. None directly. Builds a
`mil_spec::Program` from primitives. The KV cache state pattern
uses `make_state` / `read_state` / `slice_update` / `write_state`
(Prism lines 227-282 in `mil_gen_full.rs`).

Stateful vs stateless split. The function as written
constructs a stateful program. For the Tessera conversion tool
v1 we strip the state ops (lines 227-301, the `op_make_state` /
`op_read_state` / `op_slice_update` / `op_write_state` /
`op_dequantize` chain for K/V). The runtime manages KV cache
externally (Section 5.3). The port keeps the stateful generators
in the codebase so v2 can emit the stateful variant.

### 1.3 `mil_layer_programs.rs` (278 LoC, IN scope)

What it does. Two high-level "single-invocation" fused program
constructors: `build_full_ane_layer_program` (a single fused
transformer layer with integrated KV compaction via topk) and
`build_batched_matmul_program` (a batch-broadcasting matmul).
These were Prism's contribution; the smaller `mil_builder.rs`
did not have them.

Key public types / functions (in scope):

- `build_full_ane_layer_program(hidden, interm, n_h, head)` ->
  `Result<Vec<u8>, MilBuildError>`. Returns the **serialised**
  MIL program bytes ready for `coremlcompiler` to ingest. This
  is the pattern Tessera follows: the conversion tool returns
  bytes, not an in-memory graph.
- `build_batched_matmul_program(in, out, batch)` ->
  `Result<Vec<u8>, MilBuildError>`. Used by Prism to broadcast a
  weight across the batch dimension for ANE-friendly invocation.

Data flow. Same as `mil_gen_full.rs` but packaged as a
serialised byte vector. The function uses `prost::Message::encode`
internally (line 175-178).

Stateful vs stateless split. Both programs are stateless (no
`make_state`); KV compaction in `build_full_ane_layer_program` is
a topk that returns indices but does not persist state.

### 1.4 `mil_helpers.rs` (208 LoC, IN scope)

What it does. Small MIL protobuf construction helpers for ops
not directly supported by `MilBuilder`. Constructs pow,
reduce_sum, rsqrt, composite SiLU (sigmoid then mul) and a
shared `make_operation` helper.

Key public functions (in scope):

- `named_arg(name)` — wraps a name into an `Argument`.
- `float_attr / bool_attr / int32s_attr / string_attr` — typed
  attribute constructors.
- `tensor_type / value_type_tensor / scalar_value_type` — MIL
  type constructors.
- `make_operation(op_type, out_name, inputs, out_vt, extra_attrs)`
  — the shared op constructor used by every high-level helper.
- `op_pow / op_reduce_sum / op_rsqrt / op_composite_silu` — the
  small ops not in `MilBuilder`.

Data flow. Each helper returns `(MilBuilder, String)` so it can
be chained into the `MilBuilder` style. The helpers register
their output type in `MilBuilder::value_types` so that
subsequent shape inference works.

Stateful vs stateless split. All stateless.

### 1.5 `mlpackage.rs` (299 LoC, IN scope)

What it does. Writes a `.mlpackage` directory from a
`mil_spec::Program` and a `ModelMeta`. The package layout is
Apple-standard:

```
model.mlpackage/
  Manifest.json
  Data/
    com.apple.CoreML/
      model.mlmodel    (protobuf-encoded Model)
      weights/         (weight blobs, optional)
```

The Manifest is JSON with Apple's UUID-based `itemInfoEntries` +
`rootModelIdentifier` format. UUIDs are derived deterministically
from the model name + counter so repeated builds produce
byte-identical package contents (see `deterministic_uuid` at
`mlpackage.rs:237-248`).

Key public types / functions (in scope):

- `write_mlpackage(program, output_dir, description)` -> `Result<PathBuf, String>`.
- `write_mlpackage_with_weights(program, output_dir, description, weights)`
  -> `Result<PathBuf, String>`.
- `ModelMeta` (struct) — model_name, function_name, version,
  author, short_description, inputs, outputs.

Data flow. The function builds a `proto::Model` with
`specification_version: 9`, embeds the `mil_spec::Program` as
`model::Type::MlProgram(...)`, encodes to bytes via
`prost::Message::encode_to_vec`, then writes the directory
structure. The Manifest.json uses a SHA-256-derived UUID for
`rootModelIdentifier`.

CoreML APIs called. None directly. The output is a directory on
disk; the runtime loads it via `[MLModel modelWithContentsOfURL:]`
in `coreml_bridge.mm`.

Stateful vs stateless split. All stateless. This is exactly
the file the Tessera port replicates in C++.

### 1.6 `coreml_bridge.mm` (81 LoC, IN scope)

What it does. The Objective-C++ bridge to CoreML. Loads an
`.mlmodelc` directory into an `MLModel` and exposes a tiny C
ABI for prediction. The Rust side calls into this via
`tribunus_coreml_load_model / tribunus_coreml_free_model /
tribunus_coreml_predict / tribunus_coreml_predict_two`.

Key functions (in scope):

- `tribunus_coreml_load_model(out_model, path, units)` — wraps
  `+[MLModel modelWithContentsOfURL:configuration:error:]`.
  `units` is the `MLComputeUnits` enum.
- `tribunus_coreml_free_model(ptr)` — `CFRelease`.
- `tribunus_coreml_predict(ptr, in_name, in, out_name, out)` —
  wraps `-predictionFromFeatures:error:`. Reads/writes via
  `MLMultiArray`.
- `tribunus_coreml_predict_two(...)` — same but with two inputs.

Data flow. `ArenaInfo` (a C struct with `base_address` +
`byte_size` + shape) is wrapped in an `MLMultiArray` via
`+initWithShape:dataType:`. Output is copied back to the
caller's arena. Status is returned as `int` (0 = OK).

CoreML APIs called. `MLModel`, `MLModelConfiguration`,
`MLComputeUnits`, `MLMultiArray`, `MLMultiArrayDataTypeFloat32`,
`MLDictionaryFeatureProvider`, `MLFeatureValue`.

Stateful vs stateless split. The `tribunus_coreml_predict*`
functions are stateless. The stateful variant
(`tribunus_coreml_predict_stateful` and friends) is in
`coreml_state.rs`/`coreml_bridge.rs` and is OUT of scope for
the conversion tool but in scope for the **runtime** work
(Section 5.3). For the **conversion tool** the only
`coreml_bridge.mm` function we need is
`tribunus_coreml_compile` (new) which calls
`+[MLModel compileModelAtURL:error:]` to turn the
`.mlpackage` into a `.mlmodelc`. Prism's pipeline shells out
to `xcrun coremlcompiler compile` (see `compile_full_model.rs:442-447`)
which is the same thing under the hood; the C++ port uses
the direct API to keep the conversion tool self-contained.

### 1.7 `compile_full_model.rs` (546 LoC, OUT of scope)

This is the orchestrator: safetensors load -> palettise -> MIL
gen -> .mlpackage -> coremlcompiler -> .mlmodelc -> pack into
`.cimage`. The Tessera conversion tool replaces every step
except "MIL gen" and ".mlpackage write" with Tessera-native
versions (libgguf reader instead of safetensors, Tessera
9-component dequant instead of palettise, etc.). The Tessera
version is shorter (~3,000 LoC vs 546 plus the upstream
helpers) because it skips the safetensors + palettise step
entirely.

The specific lines NOT ported: 16-91 (the `ModelCfg` struct
and `extract_config` that read a `ModelGraph`); 95-137
(safetensors loading); 142-228 (weight key search and
shape lookup); 232-273 (palettise); 277-301 (RoPE + causal
mask builders); 305-464 (orchestration of all the above).
What IS ported: the call sequence at lines 425-460 (write
package, run coremlcompiler, verify the `.mlmodelc` exists).

### 1.8 `coreml_bridge.rs` (184 LoC, OUT of scope)

The stateful predict API plus the `CoreMlModel` wrapper
(handles MLModel lifecycle). Belongs to the runtime
workstream. The conversion tool does not need any of this.

### 1.9 `coreml_state.rs` (266 LoC, OUT of scope)

`CoreMlStateHandle`, `CoreMlStatefulRequest`,
`StatefulPrefillContext`. Belongs to the runtime workstream
(Section 5.3). The conversion tool does not need this.

### 1.10 `coreml_audit.rs` (40 LoC, IN scope but optional)

`AneModelAudit::new` is a stub: it claims to check MIL op
compatibility but the body is `TODO: Parse MIL spec...`. We
do not port this; `coremlcompiler` itself does the ANE
compatibility check at compile time and reports errors. If
the conversion tool wants a pre-check, it can run
`xcrun coremlcompiler compile` on a tiny probe model and
inspect stderr.

### 1.11 `lib.rs` (89 LoC, partial scope)

`pack_mlmodelc` and `unpack_mlmodelc` (lines 30-89) pack a
`.mlmodelc` directory into a flat byte buffer. The Tessera
conversion tool does NOT need this; the `.mlmodelc` is
written as a directory and bundled in the iOS app bundle as
a regular resource. Not ported.

The 89 LoC of `arena.rs` + `arena_info.rs` (IOSurface-backed
arenas) are out of scope for the conversion tool; they are
part of the runtime workstream.

### 1.12 Key public APIs Tessera must mirror

The conversion tool's C++ surface mirrors these Prism
signatures:

| Prism (Rust)                              | Tessera C++ port                              | LoC est. |
|-------------------------------------------|-----------------------------------------------|---------:|
| `MilBuilder::new(name)` + builder methods | `class mil_builder_t` with the same methods   |   ~2,200 |
| `MilBuilder::build() -> Program`          | `mil_builder_t::build() -> mil_program_t`     |   (in)   |
| `write_mlpackage(prog, dir, meta)`        | `write_mlpackage(prog, dir, meta)`            |     ~250 |
| `ModelMeta`                               | `struct mlpackage_meta_t`                     |     (in) |
| `tribunus_coreml_load_model(...)`         | `coreml_load_model(...)`                      |      ~50 |
| `tribunus_coreml_free_model(...)`         | `coreml_free_model(...)`                      |      ~10 |
| `tribunus_coreml_predict(...)`            | (runtime only, not in conversion tool)        |        0 |
| `tribunus_coreml_compile(...)` (NEW)      | `coreml_compile_mlpackage(dir, out_dir)`      |      ~30 |
| `build_full_prefill_mil(...)` (Tessera-flavored) | `build_tessera_prefill_mil(...)`         |   ~600 |

Total C++ port estimate: ~3,000-3,200 LoC of conversion logic
+ ~200 LoC of bridge. Split per Section 4.3.

## 2. Prism cimage vs Tessera GGUF schema translation

### 2.1 The Prism cimage

The cimage is Prism Engine's compiled model format. It is a
single file that bundles the model weights, the CoreML
`.mlmodelc` blob, and the runtime metadata. The format is
defined by `compile_full_model.rs:469-546` and
`apple_cimage_manifest.rs:1-445`. Key properties:

- Magic: `b"TRB_CIMG"` (8 bytes), then `u64 LE header_size`,
  then JSON header (compile_full_model.rs:472-485).
- Header JSON: a `tensors` object keyed by name; each entry
  has `tensor_type`, `offset`, `size`, `dim_m`, `dim_n`
  (compile_full_model.rs:520-526).
- Payload: 16 KB page-aligned; multiple blobs (weights, the
  `.mlmodelc`, etc.) appended after the header.
- The `.mlmodelc` is packed with `pack_mlmodelc` (lib.rs:30-58)
  as `[name_len:u32][name_bytes][data_len:u64][data_bytes]+`.
- The tri-lane manifest (compute-core:apple_cimage_manifest.rs)
  declares per-slot IOSurface bindings for ANE/GPU/CPU tri-lane
  execution.

The cimage is a "one file, everything" container: weights +
.mlmodelc + arena manifest. Tessera does NOT want this —
Tessera has its own GGUF container (which already holds
weights and metadata) and the conversion tool outputs a
directory `.mlmodelc` (which is itself a directory
container). The Tessera artifact is a
`{model.gguf, model.mlmodelc}` pair, not a single cimage.
This is a deliberate scoping decision; cimage-style
single-file bundling is a future packaging step that does
not need CoreML conversion to be designed today.

### 2.2 The Tessera GGUF 9-component cluster

Per `docs/multimodal-calibration-design.md:556-583` and
`tools/tile640/quantize_v3.py:3835-3844`, the Tessera T640
weight for a single tensor is encoded as 9 GGUF tensors (in
the multi-modal v1.5 spec) or 7 in the legacy v1 spec:

| Component                   | Type  | Shape (rows, in_dim)         | Modality    | Source GGUF field              |
|-----------------------------|-------|------------------------------|-------------|--------------------------------|
| weight_packed               | i32   | (out, pages_per_row*W)       | all         | always                         |
| weight_page_scales          | f16   | (out, pages_per_row)         | all         | always                         |
| weight_lane_scales          | i8    | (out, pages_per_row*L)       | all         | always                         |
| weight_outlier_row_offsets  | i32   | (out+1,)                     | all         | always                         |
| weight_outlier_cols         | i32   | (total_outliers,)            | all         | always                         |
| weight_outlier_vals         | f16   | (total_outliers,)            | all         | always                         |
| weight_act_scale_text       | f16   | (in,)                        | text        | new (v1.5)                     |
| weight_act_scale_image      | f16   | (in,)                        | image       | new (v1.5)                     |
| weight_act_scale_audio      | f16   | (in,)                        | audio       | new (v1.5)                     |

Pre-v1.5 (legacy) the `weight_act_scale_text` field is named
`weight_act_scale` (single field, text-only by convention).
The conversion tool accepts both: if the v1.5
`weight_act_scale_{text,image,audio}` are present, use them
per-modality; if only the legacy `weight_act_scale` is present,
treat it as `weight_act_scale_text` and zero-fill the other
two. The CLI flag `--modality {text,image,audio}` selects
which of the three act_scales to feed into the converted
model.

Tessera's `tessera.matrix_shape.<tensor>` and
`tessera.shape.<tensor>` metadata keys (see
`docs/tessera.md:32-33`) carry the logical (rows, cols) shape
the matmul expects. The conversion tool reads these so the
CoreML matmul can be wired with the right `[M, K] @ [K, N]`
shape.

### 2.3 The translation table: cimage field -> Tessera field

This is the per-component mapping the conversion tool walks
when it reads a Tessera GGUF and emits a `.mlpackage`.

| Tessera GGUF tensor                 | CoreML MIL representation                                                  | Source lines                                 |
|-------------------------------------|----------------------------------------------------------------------------|----------------------------------------------|
| `weight_packed`                     | `const` op, dtype `uint8`, shape `[out, pages_per_row*W]`                | tile640_quantize_v3.py:3835                 |
| `weight_page_scales`                | `const` op, dtype `fp16`, shape `[out, pages_per_row]`                    | tile640_quantize_v3.py:3836                 |
| `weight_lane_scales`                | `const` op, dtype `int8`, shape `[out, pages_per_row*L]`                  | tile640_quantize_v3.py:3837                 |
| `weight_outlier_row_offsets`        | `const` op, dtype `int32`, shape `[out+1]`                                | tile640_quantize_v3.py:3838                 |
| `weight_outlier_cols`               | `const` op, dtype `int32`, shape `[total_outliers]`                        | tile640_quantize_v3.py:3839                 |
| `weight_outlier_vals`               | `const` op, dtype `fp16`, shape `[total_outliers]`                        | tile640_quantize_v3.py:3840                 |
| `weight_act_scale_text` (or legacy `weight_act_scale`) | runtime input (or `const`), dtype `fp16`, shape `[in]`         | tile640_quantize_v3.py:3844                 |
| `weight_act_scale_image`            | runtime input (or `const`), dtype `fp16`, shape `[in]` (per `--modality`) | multimodal-calibration-design.md:573-578     |
| `weight_act_scale_audio`            | runtime input (or `const`), dtype `fp16`, shape `[in]` (per `--modality`) | multimodal-calibration-design.md:573-578     |
| `tessera.matrix_shape.<tensor>`     | the `[out, in]` shape carried in metadata                                  | docs/tessera.md:33                           |
| `tessera.name / profile / version`  | `ModelMeta` fields; written into the `.mlmodel` Metadata proto            | docs/tessera.md:17-19                        |
| `tessera.modality_breakdown` (optional) | a sidecar in the conversion tool's working dir; not embedded in `.mlmodelc` | multimodal-calibration-design.md:395-414 |

The dequant chain (the heart of the conversion) is described
in Section 3.

### 2.4 Per-layer type handling

The conversion tool walks the GGUF tensor list in the
canonical order (`docs/multimodal-calibration-design.md:556-583`
plus the legacy v1.5 conventions). The mapping from
HuggingFace-style layer module names to Tessera tensor names
is well-defined and matches `compile_full_model.rs:142-175`:

| HuggingFace module       | Tessera tensor name pattern                  | Modality default |
|--------------------------|----------------------------------------------|------------------|
| `model.embed_tokens.weight` | `token_embd.weight`                      | text             |
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight`            | text             |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight`            | text             |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight`            | text             |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight`       | text             |
| `model.layers.{i}.mlp.gate_proj.weight`    | `blk.{i}.ffn_gate.weight`          | text             |
| `model.layers.{i}.mlp.up_proj.weight`      | `blk.{i}.ffn_up.weight`            | text             |
| `model.layers.{i}.mlp.down_proj.weight`    | `blk.{i}.ffn_down.weight`          | text             |
| `model.norm.weight`                | `output_norm.weight`                          | text             |
| `lm_head.weight`                   | `output.weight`                               | text             |
| `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight` | `blk.{i}.expert{e}.ffn_{gate,up,down}.weight` | text |

For MoE layers the loop is over both `i` (layer) and `e`
(expert). The dequant op is the same; the only change is the
weight tensor name and the corresponding (out, in) shape.
The conversion tool emits one `matmul` per (layer, proj) pair
and one `dequant` chain per weight tensor.

For multimodal-aware models (gemma 4 12b unified), the
soft-token embedder weights use `act_scale_image` and the
audio embedder uses `act_scale_audio`; the conversion tool
walks the modality metadata in
`docs/multimodal-calibration-design.md:556-583` to pick the
right act_scale per tensor. The CLI `--modality` flag is a
hard override that bakes the chosen act_scale into the
`.mlmodelc` as a `const`; the per-call act_scale runtime path
is a v2 extension.

### 2.5 The modality_scales schema translation

Per `docs/multimodal-calibration-design.md:298-318`, the
Tessera calibration policy JSON carries a `modality_scales`
field. The conversion tool reads this (if present in the
sidecar JSON, not in the GGUF) and uses it to:

1. Override the per-channel `weight_act_scale_{text,image,audio}`
   components if the policy's `scale` array differs from the
   GGUF (the policy is the source of truth for calibration
   time; the GGUF is the final baked version).
2. Log the source of each act_scale component to stdout for
   the audit trail.

If the sidecar JSON is absent (e.g. the user runs the
conversion tool on a GGUF that was not produced with
`per_tensor_calibrate.py`), the conversion tool uses the
GGUF components directly with a one-line warning.

### 2.6 The outliers translation

Tessera outliers are stored as three tensors
(`outlier_row_offsets`, `outlier_cols`, `outlier_vals`).
The translation to CoreML:

- `outlier_row_offsets` becomes a `const` op with the offsets
  in `[int32, out+1]` shape. Used as the row pointer for
  `outlier_cols` / `outlier_vals` in the per-row dequant
  chain.
- `outlier_cols` becomes a `const` op with the column indices
  in `[int32, total_outliers]` shape. Used as the gather axis
  input.
- `outlier_vals` becomes a `const` op with the F16 values in
  `[fp16, total_outliers]` shape. Used as the gather source.

The dequant op (Section 3) reads these three constants and
replaces the outlier values back into the dequantized weight
after the page+lane scaling pass.

## 3. Tessera dequant as a CoreML custom op

The Tile640 dequant has these steps in the CPU/Metal path
(see `docs/c++-port-design.md:316-340` and the C++
`dequantize_row_tessera_t640` implementation in
`ggml/src/ggml-quants.c`):

1. Read the packed ternary trits from `weight_packed` (the
   per-page i32 vector).
2. For each output row, read the page scales
   (`weight_page_scales`, F16) and the lane scales
   (`weight_lane_scales`, I8) and apply them to the unpacked
   trits to get a dense F16 weight.
3. Apply the per-input-channel `weight_act_scale` (F16) as a
   per-row pre-matmul multiplication. (This is the act_scale
   that LLM.int8-style outliers need; it is **not** applied
   per-token at runtime, it is a static scaling that the
   Tile640 matmul applies on the fly.)
4. For outlier columns (per row, the
   `outlier_row_offsets` / `outlier_cols` / `outlier_vals`
   triple), replace the dequantized value with the F16
   `outlier_vals` entry.

The challenge: CoreML does not have a "Tile640 dequant" op.
We have to express the chain in MIL.

### 3.1 Option A: chain of stock CoreML ops

Every step above can be expressed with stock MIL ops
(`gather`, `mul`, `add`, `concat`, `reshape`). The chain is:

```
packed_const          (uint8 [out, pages*W])
page_scales_const     (fp16   [out, pages])
lane_scales_const     (int8   [out, pages*L])
outlier_offsets_const (int32  [out+1])
outlier_cols_const    (int32  [total_outliers])
outlier_vals_const    (fp16   [total_outliers])
act_scale_input       (fp16   [in])            # runtime input

unpacked = unpack_uint8_to_int(packed_const)        # int8 [out, in]
scaled   = unpacked * page_scales_broadcasted        # fp16 [out, in]
scaled   = scaled * lane_scales_broadcasted          # fp16 [out, in]

# Outlier replacement
outlier_mask  = scatter(outlier_cols_const, ...)      # bool [out, in]
base_out      = select(outlier_mask, scaled, outlier_vals_scattered)
final_weight  = base_out                              # fp16 [out, in]
```

This is ~30-50 ops per layer for the dequant alone. Slow on
the ANE (each gather+mul is a separate MIL op), but it
works without any custom op, and the
`xcrun coremlcompiler compile` path is well-tested.

### 3.2 Option B: CoreML custom op (the `custom_op` field)

CoreML MIL supports a `custom_op` field on `Operation`
where the user provides a name + a binary blob. The
`coremlcompiler` ingests the custom op and emits a
`MILCustomOp` shim. To use this for Tile640 dequant:

- Define a custom op `tessera_t640_dequant` with signature
  `(packed, page_scales, lane_scales, outlier_offsets,
  outlier_cols, outlier_vals, act_scale) -> weight`.
- The op runs in the ANE as a single fused matmul-dequant
  that the ANE compiler recognizes.

Custom ops are not officially supported in the public
CoreML API on iOS (they are a private extension). The
`MLCustomLayer` / `MLCustomModel` shim is the public
way to do "run my code on the ANE", but it routes through
CPU/GPU only, not the ANE. To actually run on the ANE,
the custom op must be a MIL-level custom op, which is
private.

### 3.3 Lean recommendation: stock ops for v1, custom op as v2

The lean is: **v1 = stock ops (Option A), v2 = custom op
(Option B) if benchmarks justify it.**

Rationale:

- Stock ops work on iOS 17+ (where CoreML is publicly
  available) without any private API. App Store safe.
- The Tessera dequant is run **once per layer per call**
  (the output of dequant is fed to a matmul that is the
  bulk of the time). The matmul dominates, not the dequant
  chain. The dequant has to produce the right F16 weight
  in ~1ms; the matmul is ~50-200ms. Even a 3x dequant
  slowdown is in the noise.
- The custom op is a maintenance burden (it needs a
  separate C++ implementation that targets the ANE
  compiler's expectations) and an App Store risk (private
  APIs).
- A v2 fast path is straightforward: add the custom op
  only if profiling on a real iPhone shows the dequant
  chain is >5% of total inference time. Threshold-based
  rollout.

### 3.4 The op signature

For v1 (stock ops) the dequant is expressed as a sequence
of stock MIL ops; there is no single "op". For v2 (custom
op) the signature is:

```
op_type: "tessera_t640_dequant"
inputs:
  packed:           uint8[out, pages*W]
  page_scales:      fp16  [out, pages]
  lane_scales:      int8  [out, pages*L]
  outlier_offsets:  int32 [out+1]
  outlier_cols:     int32 [total_outliers]
  outlier_vals:     fp16  [total_outliers]
  act_scale:        fp16  [in]
outputs:
  weight:           fp16  [out, in]
attributes:
  page_size: int64 = 640
  lane_size: int64 = 20
  lanes_per_page: int64 = 32
```

The runtime side (`ggml-coreml`) selects between v1 and v2
by inspecting the `.mlmodelc` model description at load
time: if a `tessera_t640_dequant` op is present, use the
custom op path; otherwise use the stock-op dequant.

## 4. The conversion tool architecture (`tessera-to-coreml`)

### 4.1 What it is

A C++ binary, ~3,000 LoC of conversion logic + ~200 LoC of
ObjC++ bridge, that reads a Tessera-quantized GGUF and
writes a `.mlmodelc` directory. Stateless: no inference,
no state, no KV cache. Exits after the `.mlmodelc` is
written.

### 4.2 CLI

```
tessera-to-coreml --input model.gguf --output model.mlmodelc
                  [--modality text|image|audio]
                  [--mlpackage-version 9]
                  [--coreml-units cpuAndNeuralEngine|cpuAndGpu|all]
                  [--progress]
                  [--tessera-sidecar /path/to/calibration.json]
                  [--allow-missing-act-scale-image]
                  [--allow-missing-act-scale-audio]
```

Flags:

- `--input`: the Tessera GGUF (required).
- `--output`: the output `.mlmodelc` directory. If the path
  ends in `.mlmodelc` it is treated as the directory; if it
  ends in `.mlpackage` the tool writes the package and
  invokes `coreml_compile_mlpackage` to produce the
  `.mlmodelc` next to it.
- `--modality`: which act_scale to bake in. Default `text`.
  Image/audio bake-ins are for "image-only" / "audio-only"
  use; a multi-modal use case would call the conversion
  tool three times and ship three `.mlmodelc` files in the
  app bundle. v2 will support a single `.mlmodelc` that
  takes the act_scale as a runtime input.
- `--mlpackage-version`: CoreML spec version. Default 9
  (matches Prism's `specification_version: 9`).
- `--coreml-units`: the compute units hint baked into the
  `.mlmodelc`. Default `cpuAndNeuralEngine` (ANE-first).
- `--progress`: print progress every 10 layers (the
  coremlcompiler pass takes 30-120s for 12B; progress
  reporting matters).
- `--tessera-sidecar`: optional calibration policy JSON
  (see Section 2.5).
- `--allow-missing-act-scale-image`,
  `--allow-missing-act-scale-audio`: for v1 GGUFs that
  don't have the new fields; the tool zero-fills the
  missing ones with a warning.

### 4.3 File layout (target)

```
tools/quantize/tessera-to-coreml/
  CMakeLists.txt                       ~40 LoC
  main.cpp                              ~200 LoC (CLI, arg parsing, orchestration)
  mil_builder.h                         ~80 LoC (MilBuilder class)
  mil_builder.cpp                       ~1,500 LoC (MilBuilder ops + helpers)
  mil_gen_tessera.h                     ~50 LoC
  mil_gen_tessera.cpp                   ~600 LoC (full prefill + dequant chain)
  mlpackage_writer.h                    ~50 LoC
  mlpackage_writer.cpp                  ~300 LoC (write_mlpackage, ModelMeta)
  coreml_bridge.h                       ~30 LoC (C ABI)
  coreml_bridge.mm                      ~200 LoC (ObjC++ load + compile)
  tests/
    test_mil_builder.cpp                ~200 LoC (SSA validation tests)
    test_tiny_gguf_roundtrip.cpp        ~300 LoC (TinyLlama Q4 Tessera end-to-end)
    test_mlpackage.cpp                  ~100 LoC (Manifest.json + protobuf checks)
    test_modality_act_scale.cpp         ~100 LoC (text/image/audio baking)
    fixtures/                           (small TinyLlama Q4 Tessera GGUF + checksums)
  README.md                             ~80 LoC
```

Total: ~3,000 LoC of conversion logic + ~200 LoC of
ObjC++ bridge + ~700 LoC of tests = ~3,900 LoC. The
test LoC is conservative; the actual test count is gated
by what the Mac test harness runs.

### 4.4 Dependencies

- `libgguf` (in tree, `ggml/src/ggml.c` +
  `gguf-py/gguf-py.py` family) — the GGUF reader.
- `CoreML.framework` (system) — `MLModel`,
  `MLModelConfiguration`, `MLMultiArray`.
- `Foundation.framework` (system) — JSON, file I/O.
- No new third-party deps. No `coremltools` (we are
  building the MIL protobuf by hand). No
  `coreml_proto`/`prost` (we build the protobuf by hand
  too, or use `libprotobuf-lite` from the Apple SDK).

### 4.5 The conversion pipeline

```
1. Parse CLI flags (main.cpp).
2. Open the Tessera GGUF via libgguf; iterate tensors.
3. For each Tessera T640 tensor cluster (9 components),
   emit MIL ops:
     a. const ops for the static components (packed,
        page_scales, lane_scales, outlier_*).
     b. The dequant chain (Section 3.3 v1 stock ops).
4. Emit the per-layer chain (RMSNorm -> QKV matmul ->
   RoPE -> SDPA -> O matmul -> residual -> RMSNorm ->
   MLP -> residual).
5. Emit the final RMSNorm + LM head.
6. Build the mil_spec::Program.
7. Build the .mlpackage directory (Manifest.json +
   Data/com.apple.CoreML/model.mlmodel).
8. Call coreml_compile_mlpackage to produce the
   .mlmodelc (compiles the package via the Apple
   coremlcompiler, which is what Prism does at
   compile_full_model.rs:442-447).
9. Verify the .mlmodelc directory has the expected
   files (model.mil, model.mil.in0, weights/...).
10. Print the output path and a one-line summary
    (size, layer count, modality, compute units).
```

For each step the tool prints a progress line if
`--progress` is set. Step 8 is the slow one (30-120s for
12B); the others are sub-second per layer.

### 4.6 The build system

The new directory is wired into the top-level
`tools/quantize/CMakeLists.txt` as a new
`add_executable(tessera-to-coreml ...)`. The tool builds
unconditionally on macOS (it links CoreML) and is
skipped on Linux/Windows. The CMake check is:

```
if(APPLE)
    add_executable(tessera-to-coreml main.cpp ...)
    target_link_libraries(tessera-to-coreml PRIVATE
        llama-quantize-impl   # for libgguf
        "-framework CoreML"
        "-framework Foundation")
endif()
```

No header-only libraries, no new build flags, no
subprojects. The tool is a self-contained binary that
links against the existing `llama-quantize-impl` (for
libgguf) and the Apple SDKs.

## 5. The runtime design (ggml-coreml, G7)

This section is the design for the runtime that loads
the `.mlmodelc` produced by the conversion tool. It is a
**separate workstream** from the conversion tool; this
scoping doc covers the interface contract so the conversion
tool knows what to produce, but the implementation is
deferred to G7 per the prior C++ port plan.

### 5.1 Where it fits in the existing architecture

`ggml/src/ggml-coreml/` (new directory) is a peer to
`ggml-cpu/`, `ggml-metal/`, `ggml-cuda/`. It registers
itself with the ggml backend registry at
`ggml/src/ggml-backend-reg.cpp:298` via
`ggml_backend_register`. The pattern is the same as
`ggml-metal.cpp:819-895`:

```cpp
struct ggml_backend_coreml_reg {
    std::vector<ggml_backend_dev_t> devices;
};
typedef ggml_backend_coreml_reg * ggml_backend_coreml_reg_t;

static const char * ggml_backend_coreml_reg_get_name(ggml_backend_reg_t reg) {
    return "CoreML";
}
static size_t ggml_backend_coreml_reg_device_count(ggml_backend_reg_t reg) { ... }
static ggml_backend_dev_t ggml_backend_coreml_reg_device_get(ggml_backend_reg_t reg, size_t index) { ... }
```

The new `ggml_backend_reg_t` is added to
`ggml_backend_reg_entry` in
`ggml-backend-reg.cpp:110-115`. The user's CLI flag
`--device coreml` (see Section 5.6) selects this backend.

### 5.2 The `ggml_backend_t` interface contract

The new backend implements the standard `ggml_backend_t`
vtable (see `ggml/src/ggml-backend.cpp:457-633` for the
contract). Key vtable entries:

- `ggml_backend_dev_name`: returns `"CoreML"`.
- `ggml_backend_dev_description`: returns a one-line
  description (`"CoreML (ANE-first, on iOS)"`).
- `ggml_backend_dev_buffer_type`: returns the CoreML
  buffer type (IOSurface-backed, mmappable).
- `ggml_backend_dev_supports_op(op)`: returns true for
  `GGML_OP_MUL_MAT`, `GGML_OP_RMS_NORM`, `GGML_OP_MUL`,
  `GGML_OP_ADD`, `GGML_OP_ROPE`, `GGML_OP_SCALE`,
  `GGML_OP_SOFT_MAX`, `GGML_OP_RESHAPE`, `GGML_OP_VIEW`,
  `GGML_OP_TRANSPOSE`, `GGML_OP_PERMUTE`,
  `GGML_OP_TILE640_MATMUL`, `GGML_OP_TILE640_DEQUANT`.
  Returns false for ops the ANE cannot run (e.g.
  `GGML_OP_CONV_2D`, custom CPU-only ops).
- `ggml_backend_dev_offload_op`: returns true (CoreML is
  the executor, not a CPU offload).
- `ggml_backend_graph_compute(backend, cgraph)`: the
  main entry. Maps the cgraph onto the `.mlmodelc` and
  calls `MLModel.predictionFromFeatures:`.

Approximate LoC estimate: **~3,000-4,000 LoC** for the
runtime. This is consistent with `ggml-metal.cpp`
(2,100 LoC of registration + ops) and
`ggml-cuda.cu` (3,000+ LoC) for similar scope.

### 5.3 The KV cache as CoreML MLState

For the decode path, the KV cache lives in CoreML's
`MLState`. The runtime:

1. Loads the `.mlmodelc` via
   `+[MLModel modelWithContentsOfURL:configuration:error:]`.
2. Creates an `MLState` for the loaded model via
   `+[MLState stateWithModel:error:]`.
3. For each decode step, calls
   `-predictionFromFeatures:state:error:]` with the new
   token's hidden state; the `MLState` holds the running K
   and V tensors.
4. The `MLState` is the per-session handle; multiple
   sessions get multiple `MLState` instances from the same
   loaded model (this is what `coreml_state.rs:107-152`
   does in Prism).

The Prism `coreml_state.rs` is the closest reference
implementation. The Tessera port reuses the same C
binding surface (`tribunus_coreml_state_create` /
`predict_stateful` / `state_destroy` /
`predict_stateful_async`) but the C++ implementation
calls the public CoreML state API directly (no Rust
shim).

### 5.4 Loading the .mlmodelc

```
mlmodel_t * mlmodel_load(const char * path) {
    NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    MLModelConfiguration * cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;  // ANE-first on iOS
    NSError * err = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!model) return NULL;
    return (mlmodel_t *) CFBridgingRetain(model);
}
```

The `.mlmodelc` is mmap-able on iOS (the Apple
`coremlcompiler` produces ELF-style object files; the
runtime mmaps them at load time, no copy). The weight
blobs (`Data/com.apple.CoreML/weights/`) are mmapped
directly; the activations are allocated from the
backend's IOSurface pool.

### 5.5 Memory layout

- **Weight blobs** (the dequantized constants the
  `.mlmodelc` carries): mmapped from the `.mlmodelc` on
  disk. ~2.5 GB for 12B at fp16. The OS pages them in on
  demand and pages them out under memory pressure.
- **Activations**: allocated as IOSurfaces from the
  backend's arena. ~200 MB peak for 12B at seq=512.
- **KV cache**: held inside the `MLState`. ~500 MB for
  12B at seq=2048.

Total resident set for 12B at seq=2048 on an iPhone 15
Pro (8 GB): ~3.2 GB. Tight but feasible. For the
iPhone 13/14 (6 GB) we need a smaller model or a
shorter context.

### 5.6 Backend selection

The user picks the backend at the CLI:

```
./llama-cli --model model.gguf --device coreml
```

Default policy:

- Apple Silicon + iOS: `coreml` (ANE-first).
- Apple Silicon + macOS: `metal` (existing ggml-metal
  path; CoreML on Mac is ANE-capable but Metal is faster
  for prefill).
- Intel Mac: `metal` (no ANE).
- Linux: `cpu` (no CoreML).
- Windows: `cpu` (no CoreML).

The user can override with `--device {coreml,metal,cpu}`.
If the user picks `coreml` and the `.mlmodelc` is
missing, fall back to `metal` with a warning. If the
CoreML backend fails at runtime (e.g. ANE OOM, model
not ANE-compatible), fall back to `metal` and log the
fallback reason (Section 10, Q7).

## 6. IOReport telemetry design

The runtime emits per-token telemetry via IOReport. The
telemetry is surfaced in the iPhone app in real time and
attributed to inference.

### 6.1 IOReport channels

The IOReport API is a private framework on iOS
(`/usr/lib/libIOReport.dylib`). It exposes channel
groups like "Energy Model", "CPU Stats", "GPU Stats",
"ANE DVFS". The available channels on iOS (per the
publicly documented subset + the `powermetrics` source
strings dump, see Anubis OSS and `vladkens.cc`):

| Channel name              | Group          | Unit | Semantics                                          |
|---------------------------|----------------|------|----------------------------------------------------|
| `ANE0/ANE1` (Energy Model)| Energy Model   | mJ   | ANE power draw, summed over both ANE instances     |
| `GPU Power` (Energy Model)| Energy Model   | mJ   | GPU power draw                                     |
| `CPU Power` (Energy Model)| Energy Model   | mJ   | CPU package power (E + P cores)                    |
| `DRAM Power` (Energy Model)| Energy Model  | mJ   | DRAM power                                         |
| `GPU Active Residency`    | GPU Stats      | %    | GPU active residency percentage                    |
| `CPU Active Residency`    | CPU Stats      | %    | CPU active residency across all cores              |
| `ANE Activity` (DVFS)     | ANE DVFS       | %    | ANE activity; computed from clock trigger counts   |
| `CPU Package` (Energy)    | Energy Model   | mJ   | CPU package energy (separate from CPU Power)       |
| `GPU Frequency` (Stats)   | GPU Stats      | Hz   | Weighted avg from P-state residency                |
| `CPU Frequency` (Stats)   | CPU Stats      | Hz   | Weighted avg from P-state residency                |

Sampling is two-shot: subscribe to a channel group
("Energy Model" is the one that has ANE), then
`IOReportCreateSamplesDelta(channel, prev, curr)` returns
the energy delta over the sampling interval. The
energy is in mJ, the time is in seconds, so
`P(W) = E(J) / t(s)`.

There is no genuine ANE *utilization* API; the
"ANE Activity" channel from the ANE DVFS group is a
power-normalized estimate. We use that as a fallback
when ANE power is the only signal.

### 6.2 The IOReport client

A small C++ client in `ggml-coreml/ioreport-client.cpp`
(~150 LoC):

```cpp
class ioreport_client_t {
public:
    ioreport_client_t();
    ~ioreport_client_t();
    void subscribe(const std::vector<std::string> & channels);
    sample_t sample();  // returns ANE/GPU/CPU/DRAM power in mW
private:
    void * handle_;  // IOReport subscription handle
    void * prev_;    // previous samples buffer
    void * curr_;    // current samples buffer
};
```

The `sample_t` struct:

```cpp
struct sample_t {
    double ane_power_mw;
    double gpu_power_mw;
    double cpu_power_mw;
    double dram_power_mw;
    double ane_activity_pct;  // optional, may be 0 if channel absent
    double thermal_state;     // 0..4, see Section 6.4
    int64_t timestamp_ns;
};
```

Sampling cadence: 100 Hz (every 10ms). Per-token telemetry
is the *last sample before the next token's matmul
completes*. The runtime emits one `sample_t` per token
into a ring buffer; the iPhone app reads the ring
buffer at UI cadence.

### 6.3 Battery current

Battery current is on IOKit's `AppleSmartBattery` service
(per `BatteryInfo.swift` in the public gist). The runtime
opens the service and reads `InstantAmperage` (signed,
negative = discharge) at 1 Hz. The iPhone app subscribes
to `AppleSmartBattery` notifications via
`IOBatteryHealthNotification` for charging-state changes
but the per-token attribution is the 1 Hz sample.

The exact property names (iOS 17+):
- `InstantAmperage` (mA, signed)
- `ExternalConnected` (bool)
- `CurrentCapacity` (mAh)
- `MaxCapacity` (mAh)
- `Temperature` (Celsius * 100)

### 6.4 Thermal state

`NSProcessInfo.thermalState` (public API) returns:
- `NSProcessInfoThermalStateNominal` (0)
- `NSProcessInfoThermalStateFair` (1)
- `NSProcessInfoThermalStateSerious` (2)
- `NSProcessInfoThermalStateCritical` (3)

Sampled at 1 Hz, included in `sample_t` for completeness.
The iPhone app shows it as a colored badge.

### 6.5 The v3 sidecar extension

Per `docs/c++-port-design.md:587-606`, the v3 sidecar
header carries per-row metadata. The runtime emits a
new row kind `tessera_sidecar_row_coreml_telemetry_t`
that captures the per-token IOReport sample. The row
format (24 bytes, fits the existing v3 row envelope):

```
struct tessera_sidecar_row_coreml_telemetry_t {
    uint64_t timestamp_ns;
    uint32_t ane_power_mw;
    uint32_t gpu_power_mw;
    uint32_t cpu_power_mw;
    uint32_t dram_power_mw;
    uint16_t ane_activity_pct_x100;  // 0..10000 = 0.00%..100.00%
    uint8_t  thermal_state;
    uint8_t  battery_state;         // 0=discharging, 1=charging, 2=full
    int32_t  battery_current_ma;
    uint32_t token_id;              // the token that was just produced
};
```

The v3 reader (`tools/tessera/l3_sidecar_v3_reader.py`)
gains a new row kind; the sidecar writer in
`ggml-coreml` emits one row per token. The 30-minute
flight test aggregates these rows per session and
attributes them to the inference.

### 6.6 The iPhone app surface

Real-time display (the headline number):

```
ANE:    1.2 W
GPU:    0.3 W
CPU:    0.8 W
DRAM:   0.4 W
Battery: -45 mA
Thermal: nominal
Throughput: 28.5 tok/s
```

This is the user-facing number. The app updates it 4
times per second from the IOReport ring buffer. The
30-minute flight test stores the time series in
CoreData and shows a post-session chart.

### 6.7 The 30-minute flight test

Continuous chat for 30 minutes, with:

- The IOReport ring buffer recording at 100 Hz.
- The sidecar writer emitting one row per token.
- A CoreData log of (timestamp, ane_power_mw, gpu_power_mw,
  cpu_power_mw, dram_power_mw, thermal_state, token_count,
  cumulative_battery_mah).

End-of-session summary:

- Total tokens produced.
- Average throughput (tok/s).
- Average power per token (mWh/token, broken down by ANE,
  GPU, CPU, DRAM).
- Peak thermal state reached.
- Total battery draw (mAh).
- Wall-clock time vs CoreTime (the time the ANE was
  busy, from the IOReport ANE activity channel).

This is the battery hero metric: mWh/token. The
target for gemma 4 12B on iPhone 15 Pro is **< 50
mWh/token** at seq=512, 28 tok/s decode. The 30-minute
test is the acceptance gate.

## 7. Test path

### 7.1 Mac unit tests

- `test_mil_builder.cpp`: SSA validation. Builds a tiny
  MIL program (2 inputs, 1 matmul, 1 output), checks
  `build()` succeeds. Builds a missing-output variant,
  checks `build()` fails with `UndefinedBlockOutput`.
  Mirrors `mil_builder.rs:1753-1837`.
- `test_mlpackage.cpp`: write a tiny `.mlpackage`,
  re-read the `Manifest.json`, verify the UUID and
  protobuf structure. Mirrors `mlpackage.rs:250-298`.
- `test_modality_act_scale.cpp`: verify that the
  `--modality {text,image,audio}` flag picks the right
  GGUF component.
- `test_tiny_gguf_roundtrip.cpp`: build a tiny
  Tessera-quantized GGUF (one layer, 4-dim hidden),
  run `tessera-to-coreml`, load the `.mlmodelc` in a
  test harness, run inference, verify the output is
  bit-equivalent to the GGUF dequant (run the C++
  Tile640 dequant on the same input and compare
  outputs).

### 7.2 Mac integration test

- Build a small Tessera-quantized model
  (TinyLlama 1.1B Q4 Tessera) from
  `tools/tile640/quantize_v3.py`.
- Run `tessera-to-coreml --input tinyllama.gguf
  --output tinyllama.mlmodelc --modality text`.
- Load the `.mlmodelc` in a Mac test harness, run
  inference, verify the output is bit-equivalent to
  the GGUF dequant at the per-tensor level
  (load the F16 reference, run CoreML, diff).

### 7.3 iPhone test

- Same model, same `.mlmodelc`, transfer to a real
  M-series iPhone (iPhone 15 Pro target).
- Run inference, verify the output, capture the
  IOReport ring buffer for the duration.
- Compare per-token output to the Mac integration
  test (must be bit-equivalent; the Mac and iOS
  CoreML runtimes are the same compiler output but
  may differ in floating-point ordering; we allow
  ~1e-3 relative error).

### 7.4 The 30-minute flight test

- iPhone 15 Pro, gemma 4 12B Tessera Q4, seq=512
  context, continuous chat for 30 minutes.
- Measure: total tokens, average tok/s, mWh/token
  (ANE + GPU + CPU + DRAM), peak thermal, total
  battery draw.
- Acceptance: mWh/token < 50, no thermal critical
  events, tok/s > 25.

## 8. Phased implementation plan

| Phase | Duration | Owner | Deliverable |
|------:|---------:|-------|-------------|
| Phase 1: C++ conversion tool | ~2 weeks | 1 dev | `tessera-to-coreml` binary, Mac unit + integration tests |
| Phase 2: `ggml-coreml` runtime | ~3 weeks | 1 dev | `ggml_backend_t` registration, `.mlmodelc` loading, stateful decode, backend selection |
| Phase 3: IOReport telemetry | ~1 week | 1 dev | `ioreport_client_t`, sidecar extension, iPhone app integration |
| Phase 4: iPhone end-to-end + 30-min test | ~1 week | 1 dev | 30-min flight test on iPhone 15 Pro, mWh/token < 50 |
| **Total** | **~7 weeks** | **1 dev** | |

With 3 parallel agents after Phase 1:

- Agent A: Phase 2 (runtime).
- Agent B: Phase 3 (telemetry).
- Agent C: Phase 4 prep (iPhone test harness,
  IOReport capture infrastructure).

The 3-agent path is ~3 weeks wall clock from Phase 1
ship. The 1-agent path is ~7 weeks. The agent split
is safe because the runtime (Phase 2) and the
telemetry (Phase 3) are independent; the telemetry
reads from a public API the runtime exposes (the
sample ring buffer), and Phase 4 depends on both.

## 9. Open design questions (with lean recommendations)

The architect locked the answers to all 10 questions on 2026-07-30
(see items C1-C10 in "Architect decisions on the 10 open questions"
above). The agent's leans below are historical analysis; the
architect's decisions supersede them. Note that the architect
pushed back on C8 (3 `.mlmodelc` per modality became 1 with runtime
act_scale for v1).

| Q#  | Question                                                              | Lean                                              | Notes                                                                                                                  |
|----:|-----------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Q1  | Tessera dequant as CoreML custom op (faster) vs chain of stock ops (simpler)? | Stock ops for v1                                  | Section 3.3. App Store safe. Benchmark on iPhone before committing to v2 custom op. Threshold: if dequant > 5% of total inference time, do v2. |
| Q2  | `.mlmodelc` generation: at quantize time (Mac) or at first iPhone launch (on-device)? | At quantize time, bundle the `.mlmodelc` in the `.app` | Section 4.2. `coremlcompiler` is 30-120s for 12B; the user would notice. Bundle-time generation is offline and reproducible. |
| Q3  | IOReport channel selection: which channels are stable across iOS versions? | Research, surface findings                        | Section 6.1. ANE power in "Energy Model" is stable across iOS 15+; "ANE Activity" in DVFS is more variable. Document the fallback (use power when activity is unavailable). |
| Q4  | Per-token vs per-session battery attribution?                         | Per-session for v1, per-token as v2               | Section 6.5. Per-token attribution requires writing one sidecar row per token; per-session is the total mAh over the session. Per-token is in the sidecar regardless; the per-session aggregation is a query over the sidecar. |
| Q5  | `ggml-coreml` memory layout: MMAP the `.mlmodelc`, or load to RAM?   | MMAP for the weight blobs, RAM for the activations | Section 5.5. `coremlcompiler` produces ELF-style objects; the OS mmaps them on demand. Activations go through the IOSurface pool. |
| Q6  | KV cache in CoreML state: full CoreML state API, or custom?          | Full state API                                    | Section 5.3. `MLState` is the public API for stateful models. The Prism `coreml_state.rs` proves it works. Custom is more code for no benefit. |
| Q7  | Backend fallback: when CoreML fails (OOM, model not ANE-compatible), fall back to Metal automatically? | Yes, log the fallback                              | Section 5.6. The fallback is logged with the reason. `--device metal` skips the attempt. The fallback is silent in production; the test harness enables verbose logging. |
| Q8  | How to handle multimodal: three `.mlmodelc` per modality, or one with a runtime act_scale input? | Three for v1, one with runtime input as v2        | Section 4.2 `--modality`. v1 ships three `.mlmodelc` files in the `.app`; the iPhone app picks based on the user's task. v2 ships one with a `weight_act_scale` input. The v2 work is a separate milestone. |
| Q9  | Should the conversion tool emit a `coremlcompiler` invocation directly, or use the public `+compileModelAtURL:error:`? | Public API                                         | Section 4.5 step 8. The public API is what Apple recommends for App Store apps; the CLI invocation shells out and is harder to test. |
| Q10 | Where does the conversion tool's Tessera config (matrix shape, modality) come from? | GGUF metadata + optional sidecar JSON              | Section 2.2. The GGUF is the source of truth; the sidecar is an override. The tool warns if they disagree. |

## 10. Risk register

| Risk                                                                                              | Severity | Mitigation                                                                                                                       |
|---------------------------------------------------------------------------------------------------|---------:|----------------------------------------------------------------------------------------------------------------------------------|
| IOReport is a private framework on iOS. App Store might reject apps that link it.                  | High     | Ship via TestFlight / dev only; document the App Store risk. The runtime gates the IOReport client behind a `--enable-telemetry` flag; production builds skip the link. |
| `coremlcompiler` model generation is slow for large models (~30s-2min for 12B).                  | Medium   | Progress reporting (`--progress` flag). Pre-compile a smaller model first to surface schema issues in seconds.                 |
| 12B on iPhone: ANE memory constraints; the tool may need to split the model into ANE-friendly chunks. | High    | Start with gemma 4 9B (closer to 6 GB on iPhone 15 Pro). 12B is a stretch goal. Chunking is a future workstream.             |
| The Tessera dequant as stock CoreML ops may be slower than the ggml-metal equivalent.            | Medium   | Benchmark on iPhone 15 Pro before committing to v1. If the dequant is < 5% of total time (expected), ship v1.                  |
| Schema translation: edge cases in cimage (rare layer types, MoE experts, multimodal fusion layers). | Medium   | Build a fixture library (Tessera-quantized TinyLlama, gemma 4 12B, a MoE model) and run the conversion tool against all of them. |
| Test infrastructure: how to verify the `.mlmodelc` output is bit-equivalent to the GGUF dequant on a per-tensor basis. | Medium   | Use the L1.5 reference sidecar (FP16 source weights). Run the C++ Tile640 dequant and the CoreML path on the same input; diff with `numpy.allclose` at 1e-3 relative tolerance. |
| `MLState` lifecycle is per-session, not per-call. Multiple concurrent sessions need multiple states. | Low     | The runtime creates one state per `llama_context` (existing abstraction). No new code.                                          |
| IOReport sample cadence at 100 Hz may miss short decode steps (< 10 ms).                          | Low      | Aggregate at the sidecar writer; per-token samples are the last sample before the next token. Documented in Section 6.2.        |
| `coremlcompiler` is not always present in the iOS toolchain (xcode-only).                        | Low      | The conversion tool is Mac-only; the runtime does not call `coremlcompiler`. Toolchain concern only at quantize time.         |
| Tessera 9-component GGUF is not yet a stable spec; the v1.5 fields may change.                    | Medium   | Lock the v1.5 spec before Phase 1 starts. The conversion tool reads both v1 (7-component) and v1.5 (9-component) GGUFs.        |
| The conversion tool may fail on a GGUF produced by an old version of `tile640_quantize_v3.py`.    | Medium   | Support the v1 7-component format (zero-fill the missing act_scales with a warning). Document the supported version range.    |
