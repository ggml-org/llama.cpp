# Parametric Metal Kernel Design (S2)

Status: DESIGN INPUT (for implementation, not a survey)
Date: 2026-07-30
Depends: ts_format_spec keystone (tools/quantize/tessera/tessera-format.h)
Kernel: kernel_TILE640_MATMUL (ggml/src/ggml-metal/ggml-metal.metal:11370)
Companion: docs/interleaved-kernel-design.md, docs/research-alignment-2026-07-30.md

## 0. Goal and scope

Today the Tile640 dequant/matmul kernel hardcodes the format in compile-time
constants:

```
#define T640_PAGE 640            // ggml-metal.metal:11307
#define T640_LANE 20             // ggml-metal.metal:11308
#define T640_LANES_PER_PAGE 32   // ggml-metal.metal:11309
#define T640_WORDS_PER_PAGE 32   // ggml-metal.metal:11310
threadgroup float decoded_page[T640_PAGE]   // ggml-metal.metal:11411 (2560 bytes, static)
```

and the dequant arithmetic assumes int8 lane scales normalized by 1/127
(ggml-metal.metal:11426-11428) with a radix-243 4x5 trit packing
(ggml-metal.metal:11316-11349).

The goal is ONE parametric kernel that reads each tensor's format from GGUF
metadata at runtime, instead of shipping N hand-specialized kernels. The format
is the `ts_format_spec` keystone, whose layout genes are bounded to a kernel
envelope and whose scalar genes are unbounded (tessera-format.h:22-31).

Hard invariant (the tripwire, from tessera-format.h:33-35): the DEFAULT spec
(page=640, lane=20, lane_scale_bits=8) MUST reproduce the existing kernel output
BIT-IDENTICALLY. Every parametrization strategy in this doc is gated on that.

Scope note: this doc covers the 2D kernel `kernel_TILE640_MATMUL` and its
per-expert sibling `kernel_TILE640_MATMUL_ID` (ggml-metal.metal:11517). The same
parametrization applies to both; the ID variant already takes runtime params
(args.ne21/ne22) and shares the identical decode math, so it inherits the design
with no extra decisions. GET_ROWS/DEQUANT (ggml-metal.metal:11660, 11686) reuse
`tile640_decode_element` (ggml-metal.metal:11620) and follow the same unpack
generalization.

## 1. Parameter classification

`ts_format_spec` splits into two classes with fundamentally different kernel
relationships.

### 1.1 Layout params - change memory addressing/sizing, bounded to envelope

| Field | What it changes in the kernel |
|---|---|
| `page_size` | size of `decoded_page` threadgroup buffer (page_size x 4 bytes); `pages_per_row` stride (ggml-metal.metal:11395-11398); cooperative-decode loop bound (11424) |
| `lane_size` | trit->word packing radix and words-per-lane (11430-11435); per-lane scale index stride `T640_LANES_PER_PAGE` (11404, 11425) |
| `lane_scale_bits` | `lane_scales` storage width (1 byte vs 1 nibble per lane); dequant reciprocal (1/127 vs 1/15) and nibble unpack (11426-11428) |

These change HOW BYTES ARE ADDRESSED AND SIZED. The kernel cannot compute a
stride or allocate a buffer without them. They are bounded to the envelope
(tessera-format.h:51-56) precisely because each supported value needs matching
kernel machinery.

### 1.2 Scalar params - pure numbers, unbounded, do NOT reach the dequant kernel

| Field | Where it actually lands |
|---|---|
| `threshold_mult` | quant-time only: sets which weights become +/- trits vs 0. Baked into `packed`. |
| `outlier_frac` | quant-time only: sets the fp16 outlier CSR arrays. Baked into `outlier_*` buffers. |
| `awq_alpha` | quant-time only: produces the per-channel `act_scale` buffer (already a bound kernel arg, ggml-metal.metal:11379). |

This is the key structural finding: the three scalars never appear as dequant
kernel arguments. Their effect is fully materialized in the data buffers the
kernel already reads (`packed`, `outlier_*`, `act_scale`). The evolutionary
search ranges over them freely (tessera-format.h:27-30) at zero kernel cost,
because varying a scalar changes buffer CONTENTS, not buffer LAYOUT.

Consequence for parametrization: "scalars as free runtime kernel args" is, for
this kernel, already satisfied trivially - they are runtime params of the
QUANTIZER, resolved host-side into data. If a future adaptive path ever needs a
scalar inside the kernel, it is a `constant float &` arg with zero layout
impact. The real parametrization question is entirely about the three layout
params.

## 2. Threadgroup-memory sizing for runtime-variable page_size

`decoded_page` is the dominant threadgroup-memory consumer and therefore the
occupancy lever. Register pressure is NOT the limiter: the kernel uses ~24-40
VGPR/thread (ggml-metal.metal:11406 comment; interleaved-kernel-design.md
section 6) against a ~512/thread budget at 128 threads. Threadgroup memory is
the constraint that moves.

Envelope page sizes and their static buffer cost:

| page_size | decoded_page bytes | resident TGs/SM (32 KB limit) |
|---|---|---|
| 320  | 1280 | 25 |
| 640  | 2560 | 12 |
| 1280 | 5120 | 6 |

(32 KB is the Apple-GPU threadgroup ceiling used throughout
interleaved-kernel-design.md section 6; the kernel is memory-bound, so resident
TG count is the latency-hiding budget.)

### Option A - max-bound static

`threadgroup float decoded_page[1280]` always (5120 bytes). Compiler-known size,
float4 vectorization with known alignment, simplest source. Cost: at page=320
you allocate 4x the needed memory and drop from 25 to 6 resident TGs/SM. Since
the decode is cooperative across all SIMD groups and the kernel is memory-bound,
this is a real latency-hiding loss on every small-page tensor.

### Option B - dynamic threadgroup memory

`threadgroup float * decoded_page [[threadgroup(0)]]`, sized per dispatch via
`setThreadgroupMemoryLength:` on the encoder (or the pipeline). Allocates exactly
`page_size x 4` bytes. page_size is known on the host at dispatch time (it is
per-tensor metadata, section 7), so wiring is one call and correctness is
unaffected. Cost: the compiler loses the constant-known bound (minor; the decode
loop bound is loaded once per page anyway), and the base alignment of dynamic
threadgroup memory must be confirmed 16-byte for the existing float4 loads
(ggml-metal.metal:11454) - Metal guarantees >= 16-byte alignment for dynamic
threadgroup bases, so the float4 path is safe.

### Option C - per-page specialized static (entangled with strategy b/c)

If page_size is a compile-time specialization (section 6), each specialization
emits `decoded_page[<literal>]` at the exact size. Best of both: exact size AND
compiler-known. This falls out of the recommended strategy for free.

### Recommendation

Use dynamic threadgroup memory (Option B) for any runtime-parametric path, and
let the specialized path (Option C) take over when page is a function constant.
Do NOT ship Option A as the default: it pessimizes the common small-page case on
the one axis (occupancy) that actually gates this memory-bound kernel. The
decision is forced by the facts: the thing that matters (occupancy) is exactly
the thing max-bound static wastes, and the host already knows page_size.

## 3. Packing-radix generalization for variable lane_size

### 3.1 Why the current packing is welded to lane=20

The radix-243 scheme packs five trits as one radix-243 digit (3^5 = 243),
decoded through `T640_TRIT5_LUT[243]` (ggml-metal.metal:11316). Four such groups
fit in a uint32 because 243^4 = 3.486e9 < 2^32 = 4.295e9, while 243^5 = 8.47e11
overflows. So one uint32 word holds exactly 4 x 5 = 20 trits - which is exactly
lane_size=20, giving a clean 1:1 word:lane mapping (the per-lane scale is indexed
by word, ggml-metal.metal:11433). The unpack peels groups with a serial
`rem /= 243` chain (ggml-metal.metal:11577-11579 in the ID variant;
`tile640_trit` at 11344 uses precomputed powers).

Both properties - 20 trits/word AND word-aligned-to-lane - are accidents of
243^4 < 2^32 <= 243^5 coinciding with lane=20. Neither holds for lane=16 or 32.

### 3.2 Generalization: two packing families

The packing is parameterized by (radix, groups_per_word, words_per_lane, LUT),
all derived from lane_size. For the envelope lanes {16, 20, 32} the clean split
is two families:

| lane | family | radix | groups/word | trits/word | words/lane | LUT | serial divs/lane |
|---|---|---|---|---|---|---|---|
| 16 | B (new) | 81 (3^4) | 4 | 16 | 1 | TRIT4[81] | 4 |
| 20 | A (current, P0) | 243 (3^5) | 4 | 20 | 1 | TRIT5[243] | 4 |
| 32 | B (new) | 81 (3^4) | 4 | 16 | 2 | TRIT4[81] | 8 |

Family B uses radix-81 (3^4), whose LUT maps one radix-81 index to four 2-bit
trit fields. 81^4 = 4.30e7 fits a uint32 with room to spare. G=4 divides 16
(4x4) and 32 (8x4 = 2 words) evenly:

- lane=16: 1 word = 4 groups x 4 trits = 16. 1 word/lane.
- lane=32: 2 words x (4 groups x 4 trits) = 32. 2 words/lane.

Family A is the EXISTING radix-243 code, kept verbatim for lane=20 so the
default spec stays bit-identical (the hard invariant). We do NOT re-derive
lane=20 under radix-81 - that would produce a different bit layout and break P0.

Derived strides (replace the #defines):

```
lanes_per_page  = page_size / lane_size
words_per_lane  = (lane_size == 32) ? 2 : 1        // family B only; family A = 1
words_per_page  = lanes_per_page * words_per_lane
```

Sanity against today: page=640, lane=20 -> lanes_per_page=32, words_per_page=32,
matching T640_LANES_PER_PAGE and T640_WORDS_PER_PAGE (ggml-metal.metal:11309-11310).

### 3.3 The cost, and why it argues for specialization

The unpack inner loop is a serial integer-divide chain (`rem /= radix` per
group). Two costs scale with parametrization style:

1. Divide-by-constant vs divide-by-variable. With radix known at compile time,
   `/= 243` and `/= 81` lower to multiply-shift sequences. With radix read at
   runtime, each is a true integer divide - the single most expensive ALU op on
   an Apple GPU, sitting in the hottest loop of a memory-bound kernel.
2. Unrolling. The group loop trip count (4 or 8) and the trit-per-group count
   (4 or 5) are unrollable when compile-time-known; a runtime lane_size forces a
   general loop with loop-control overhead and a resident worst-case LUT.

This is the strongest single argument against fully-runtime parametrization
(strategy a) and for specializing on lane_size (strategy b/c). The LUT itself is
also layout data: TRIT5[243] (486 bytes) and TRIT4[81] (162 bytes) - a runtime
kernel would have to keep both resident and select, while a specialization keeps
only its own.

## 4. lane_scale_bits: 4 vs 8 dequant arithmetic and storage

### 4.1 The 8-bit path (current, P0)

`lane_scales` is `device const uchar *` - one byte per lane, unsigned magnitude
code in [0, 255]. Dequant (ggml-metal.metal:11426-11428):

```
scale = page_max * float(row_ls[idx]) * (1.0f / 127.0f);
```

Sign is carried by the trit (d==2 -> -scale, ggml-metal.metal:11436), so the lane
scale is a magnitude. Bit-identity requires this expression verbatim, including
the literal `1.0f / 127.0f`.

### 4.2 The 4-bit path (new)

Two lane scales packed per byte. Unsigned 4-bit magnitude code in [0, 15],
normalized by 1/15 so max code maps to 1.0 (same code/maxcode convention as the
8-bit path). Storage layout (writer and kernel must agree):

```
byte = lane_scales[idx >> 1];
nib  = (idx & 1) ? (byte >> 4) : (byte & 0xF);   // even lane = low nibble
scale = page_max * float(nib) * (1.0f / 15.0f);
```

Storage halves: `ceil(lanes_per_page / 2)` bytes per page instead of
`lanes_per_page`. The only arithmetic differences from the 8-bit path are the
nibble unpack and the reciprocal literal (1/15 vs 1/127). Both specialize
cleanly at compile time; at runtime they are a parity branch plus a distinct fp
constant. The default (8-bit) specialization keeps `1.0f/127.0f` exactly, so P0
is preserved.

### 4.3 Interaction with lane_size

lane_scale_bits is independent of lane_size in the arithmetic but coupled in
storage: the lane_scales byte stride depends on both (nibbles pack two lanes per
byte regardless of how many trits each lane holds). The host writer computes the
lane_scales buffer length from `(lanes_per_page * pages_per_row * bits) / 8`;
the kernel indexes it with the matching shift/mask. Keep the length formula in
the keystone header (tessera-format.h) so writer and kernel cannot drift.

## 5. The layout envelope

### 5.1 What the kernel supports after this design

All 18 envelope points (tessera-format.h:51-56): page {320,640,1280} x lane
{16,20,32} x bits {4,8}. Every page/lane combination yields integer lanes_per_page
(320/16=20, 320/20=16, 320/32=10, 640/16=40, 640/20=32, 640/32=20, 1280/16=80,
1280/20=64, 1280/32=40), so no partial-page edge cases. The machinery:

- page: dynamic threadgroup (section 2) or per-page static specialization.
- lane: family A (lane=20) + family B (lane=16,32) packings (section 3).
- bits: 8-bit + 4-bit dequant (section 4).

### 5.2 Envelope bounds (what limits widening)

1. Threadgroup memory: decoded_page = page x 4 bytes must fit 32 KB -> page <=
   8192. Current max 1280 has ~6x headroom.
2. Packing radix: 3^(groups_per_word x trits_per_group) <= 2^32, else a group set
   spills a uint32. A new lane needs a (radix, groups/word) pair satisfying this.
3. Cooperative-decode stride: lanes_per_page must divide cleanly across the
   thread count (32-128); all envelope points do.

### 5.3 Incremental widening path

Land in this order, each step independently bit-identity-gated:

1. lane=20 across all pages x both bits. Smallest delta from today; the 640/20/8
   point IS the current kernel. Proves the parametrization scaffolding.
2. lane=16 (family B, radix-81). New LUT + unpack; verify the lane=20 path is
   byte-untouched.
3. lane=32 (family B, 2 words/lane).
4. Widen the envelope (e.g. page=2560, lane=64, bits=2) ONLY by adding a packing
   family and raising the dynamic-threadgroup cap. `ts_format_spec_in_envelope`
   (tessera-format.h:51) is the single widening point and must land atomically
   with the matching kernel support - the header comment already mandates this.

## 6. A/B testing across parametrization strategies

### 6.1 The three strategies

(a) Fully-runtime-parametric. All three layout params read at runtime from a
bound `constant ts_format_spec &` (or a packed uint). One kernel, one pipeline.
Decode loop bounds dynamic, packing radix dynamic (true integer divides),
threadgroup dynamically sized. The format buffer is the only layout input.

(b) Template / function-constant specialization per envelope point. page, lane,
bits are function constants (FC_TILE640 + 4/5/6, extending the existing block at
ggml-metal-impl.h:107 which already uses +0..+3). Up to 18 pipelines, each fully
unrolled with multiply-shift divides and exact static threadgroup sizing. Host
selects the PSO by spec (section 7).

(c) Hybrid. Layout via function constants (as b), scalars remain runtime. For
this kernel (b) and (c) nearly coincide, because section 1.2 showed the scalars
never reach the dequant kernel. The meaningful distinction (c) adds over (b) is a
runtime format DESCRIPTOR still bound for validation/telemetry even though the
layout itself is specialized - which we want anyway for the runtime-aware
receipts thesis (research-alignment-2026-07-30.md section 4.5, IOReport). The
real decision axis is (a) runtime-layout vs (b/c) specialized-layout.

Finding to record honestly: for the dequant kernel, "hybrid" and "full
specialization" are the same kernel with different telemetry wiring. The A/B
therefore primarily measures (a) against (b/c); (b) vs (c) measures only the cost
of binding+validating a runtime descriptor (expected ~zero).

### 6.2 Benchmark harness (mirrors bench_interleaved.sh / analyze_bench.py)

New files, same shape as the interleaved bench:

- `tools/quantize/tessera/bench_parametric.m` - self-contained Metal harness.
  Compiles the kernel source with `newLibraryWithSource`, specializes function
  constants via `MTLFunctionConstantValues` for (b)/(c) (exactly the pattern at
  bench_interleaved.m:71-85), binds a format buffer for (a).
- `tools/quantize/tessera/bench_parametric.sh` - clang build + run, matching
  bench_interleaved.sh:8-18.
- `tools/quantize/tessera/analyze_parametric.py` - parses stdout, matching
  analyze_bench.py's regexes for "P0 bit-identity ... PASS/FAIL" and "N/M
  mismatches", extended to parse the timing/occupancy table and emit a
  comparison matrix + receipt JSON (reuse the ts_ab_receipt_json pattern from
  tessera-ab-harness.h:54).

Workload: the fixed bench_interleaved shape (OUT_DIM 512, IN_DIM 640, N_TOKENS 4,
WARMUP 50, ITERS 500) as the P0 anchor, plus a sweep over the 18 envelope points
so each strategy is timed across the full layout space.

### 6.3 Metrics

Per (strategy x envelope point):

1. Dequant time. Wall time of the decode+matmul via GPU timestamps /
   MTLCounterSet, WARMUP 50 then ITERS 500. Report mean and p99 us/dispatch. This
   is the in-harness number.
2. Occupancy. Threadgroup bytes per TG (decoded_page + any staging) -> theoretical
   resident TGs/SM = floor(32768 / tgmem); cross-check against the PSO's
   `maxTotalThreadsPerThreadgroup` / `threadExecutionWidth`
   (ggml-metal-device.m:93-95, 431-432). Report theoretical and, where the
   profiler provides it, achieved.
3. Register pressure. Not exposed by MTLComputePipelineState reflection; measured
   via the Instruments GPU profiler as a separate pass. analyze_bench.py already
   concedes "requires GPU profiler" for throughput (analyze_bench.py:31); register
   pressure is reported as its own profiler-sourced column, not an in-harness
   number. Do not fake it.

### 6.4 The hard invariant as the gate

Before any timing is trusted, the harness runs the reference dequant (the current
kernel, equivalently the CPU reference `tile640_decode_element`,
ggml-metal.metal:11620) and each strategy on IDENTICAL packed input at the DEFAULT
spec (640/20/8), compares output buffers byte-for-byte, and prints, per strategy:

```
P0 bit-identity (strategy=a): PASS
P0 bit-identity (strategy=b): PASS
P0 bit-identity (strategy=c): PASS
0/262144 mismatches
```

Any FAIL is a hard stop - the strategy is disqualified regardless of speed. This
reuses the existing bit-equivalence machinery (test_bit_equiv.cpp / .py) and the
exact PASS/FAIL + mismatch strings analyze_bench.py already parses. Scope: the
bit-identity gate binds ONLY the 640/20/8 point. The new lane=16/32 packings
(section 3) have no old-kernel anchor; they are validated separately by
quant->dequant round-trip against the CPU reference, not by identity-to-old-kernel.

### 6.5 Expected outcome (to be confirmed, not assumed)

(b/c) should win on dequant time at every envelope point (unrolled loops,
multiply-shift divides, exact threadgroup), with (a)'s only advantage being a
single pipeline. The A/B exists to QUANTIFY that gap and to confirm the gap is
large enough to justify the PSO count - if (a) is within noise, ship (a) for
simplicity. The benchmark decides; this doc only predicts.

## 7. Host-side wiring: GGUF metadata -> kernel arg buffer

### 7.1 Quantize time (writer)

The layout is bounded to a 5-bit code: page_size (2 bits: 320/640/1280),
lane_size (2 bits: 16/20/32), lane_scale_bits (1 bit: 4/8). One byte per tensor.
Add `ts_format_spec_to_code` / `ts_format_spec_from_code` to tessera-format.h
(single source of truth, keeps the code table next to the envelope check).

The existing writer emits file-level `tessera.*` KV pairs via `gguf_set_val_*`
(tessera-gguf-writer.cpp:14-36) and a per-tensor 6-component cluster
(tessera-gguf-writer.h:34-37). Extend `ts_gguf_write_tensor_cluster` to carry the
1-byte layout code per tensor - either as a 7th cluster component or as a
file-level parallel array `tessera.format.layout_codes` (u8, indexed by tensor).
The scalars (threshold_mult, outlier_frac, awq_alpha) are quant-time only
(section 1.2); for provenance they join the existing `tessera.quantize.*` keys but
do NOT travel to the kernel.

### 7.2 Load time (ggml Metal backend dispatch)

`ggml_metal_op_tile640_matmul` (ggml-metal-ops.cpp:1769-1785) already builds a
kargs struct, selects a pipeline, and binds buffers 0..5 (weights), 6 (input),
7 (output). Extend it:

1. Read the tensor's layout code -> `ts_format_spec`. Gate on
   `ts_format_spec_in_envelope`; on failure ERROR (matches the project's
   "observed, never silently imputed" stance, research-alignment M2).
2. Strategy (a): pack the spec into the kargs (or a small struct) and bind via
   `ggml_metal_encoder_set_bytes`; set dynamic threadgroup length = page_size x 4.
3. Strategy (b/c): use the spec to SELECT the pipeline. Extend
   `ggml_metal_library_get_pipeline_tile640_matmul`
   (ggml-metal-device.cpp:671-704) to set FC_TILE640 + 4/5/6 (page/lane/bits) in
   the `ggml_metal_cv_t` and to fold them into the pipeline cache `name` key, so
   one PSO is compiled+cached per envelope point actually used. Set dynamic
   threadgroup length (or rely on the per-page static size). Scalars are never
   bound.

The pipeline cache (ggml-metal-device.cpp:691-700) already dedups by name and
compiles lazily, so the 18-point space costs only the specs present in the loaded
model - typically 1-3 distinct specs.

### 7.3 Validation/telemetry descriptor (strategy c)

Even with specialized layout, bind a read-only `constant ts_format_spec &` for the
runtime to assert against the PSO it selected and to emit into the runtime-aware
receipt (research-alignment section 4.5). This is the only thing (c) adds over
(b), and it is cheap.

## 8. Recommendation and risks

### 8.1 Recommendation: hybrid (c)

Specialize the three layout params (page, lane, bits) as function constants per
envelope point; keep the scalars out of the kernel (they are already baked into
the data buffers, section 1.2); bind a read-only format descriptor for
validation and telemetry.

Rationale:

1. The layout params change STRUCTURALLY - packing radix, LUT, loop trip counts,
   threadgroup size. Specialization buys unrolled decode loops, multiply-shift
   instead of integer divide, and exact static threadgroup sizing. These are the
   dominant effects, all concentrated in the hottest loop of a memory-bound
   kernel (section 3.3).
2. The envelope is tiny (18 points; a model uses ~1-3). The existing lazy PSO
   cache makes the pipeline count a non-cost.
3. Bit-identity is easiest when the 640/20/8 specialization is literally the
   current code path - it is. The radix-243 4x5 + uchar/127 dequant is preserved
   verbatim as one specialization, so the hard invariant holds by construction.
4. Scalars staying out of the kernel matches the ts_format_spec design intent
   exactly: layout bounded, scalars unbounded, evolutionary search free.

Why not (a): the integer-divide-by-runtime-radix and the lost unrolling land in
the decode hot path; "one pipeline" is a weak benefit against the tiny envelope
and existing cache. Keep (a) as the A/B baseline to measure the cost, not the
ship target - unless the benchmark (section 6.5) shows the gap is noise, in which
case ship (a) for simplicity.

Why (c) over pure (b): functionally identical for this kernel, but (c) keeps the
runtime format descriptor needed for the receipts/telemetry thesis. (c) dominates
(b).

### 8.2 Risks

1. PSO compile latency. 18 lazy specializations add first-dispatch cost. Mitigate
   via the existing cache and by compiling only specs present in the model.
2. Function-constant array-bound limit. A static `threadgroup float
   decoded_page[FC_page]` likely cannot be sized by a function constant in MSL
   (array bounds need compile-time literals, and function constants are
   specialization constants, not constant expressions). With per-page
   specialization this is moot - emit the literal size per specialization. For the
   runtime path use dynamic threadgroup memory (section 2). VERIFY on the local
   v17.6 Metal toolchain before committing; that toolchain has known
   attribute-placement quirks.
3. Bit-identity drift. The 640/20/8 point must stay byte-identical; a stray FMA
   reassociation breaks it. Mitigate by keeping the default specialization's
   dequant arithmetic textually identical to ggml-metal.metal:11426-11436 (no
   reordering), and by the A/B P0 gate (section 6.4) as a hard stop.
4. New packings have no P0 anchor. lane=16/32 (family B) are new code; validate by
   quant->dequant round-trip vs the CPU reference, and state this scope explicitly
   so the bit-identity claim is not over-read.
5. Envelope drift. `ts_format_spec_in_envelope` (tessera-format.h:51) and the
   kernel's supported set must never diverge. The envelope gate at load (section
   7.2) plus the A/B sweep matrix (section 6) are the two tripwires.

## 9. Sequencing (single-engineer estimates)

| Step | Work | Gate |
|---|---|---|
| P1 | Parametrize lane=20 across page {320,640,1280} x bits {4,8}; dynamic threadgroup; FC +4/5/6 scaffolding | 640/20/8 bit-identical |
| P2 | bench_parametric.m/.sh + analyze_parametric.py; A/B (a) vs (c) at all lane=20 points | P0 PASS + timing table |
| P3 | Family B packing (radix-81, TRIT4) for lane=16; round-trip validation | lane=20 byte-untouched |
| P4 | lane=32 (2 words/lane); full 18-point A/B sweep | P0 PASS + matrix |
| P5 | GGUF layout-code wiring (writer + load-time dispatch + envelope gate) | round-trip through a real GGUF |
| P6 | Widen envelope only if a format need appears | atomic with ts_format_spec_in_envelope |

P2 decides (a) vs (c). Everything after assumes (c) unless P2 says otherwise.
