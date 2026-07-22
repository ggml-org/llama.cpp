# Intel SYCL/A770 P5 performance campaign - 2026-07-18/19

## Decision

This campaign executed the scheduled P5.1-P5.14 standard Intel-SYCL matrix on
an Intel Arc A770 and retained only changes that survived explicit correctness,
performance, and GPU-safety gates.

The principal results are:

- **Retained default change:** per-kernel SYCL device-code split. It reduced
  conventional baseline-relative cold-start latency by 19.94% on Mistral and
  17.01% on Llama-3.1 without a warm throughput or correctness regression.
- **Retained opt-in experiment:** quants-first q8_0 KV rows. It improved tg128
  by 8.01-20.62% from depth 4096 through 16384 while preserving canonical
  host/session bytes and standard correctness. Canonical q8_0 remains the
  default.
- **Retained diagnostics:** standard VEC forcing and q8_0 GQA TILE routing.
  They are default-off because the measurements identified useful mechanisms
  but no globally safe default.
- **Retained correctness fix:** pre-grow quantized FA scratch before SYCL graph
  recording. This fixes a long-depth graph-capture abort.
- **Retained tooling:** fail-closed paired A770 campaigns, isolated-binary
  comparison, cold-JIT measurement, MMVQ geometry sweeps, focused tests, and
  the script catalog.
- **Killed and reverted:** global large-GRF, non-PVC direct upload, and exact
  executable-graph replay.
- **Killed without production source work:** alternate global
  reorder/DMMV routing, alternate MMVQ geometry, GPU-oneDNN quantized prefill,
  and MoE reorder.

No TurboQuant/Turbo FA, XMX FA, asymmetric KV, sparse-V, TriAttention, QJL, or
SLM-LUT work was reopened by this campaign.

"Complete campaign" below refers only to the enumerated P5.1-P5.14 queue. It
does not mean that every candidate from the preceding research survey was
executed: GQA KV-read amortization and KV prefetch were not scheduled as
standalone measurement tasks and remain unmeasured.

## Scope and pins

- Repository: `Raudbjorn/ggml-llama.cpp`
- Campaign branch: `p5-performance-execution`
- Base: `2b5149a98d30a20ad7ca7a0baea1cb471cac62af`
- Final source before this report: `f8f311aa47181c394303994793f7424940bc722b`
- Host: `vinbonesjr`
- GPU: Intel Arc A770/DG2, Level Zero device 0, i915
- Compiler: Intel oneAPI DPC++ 2026.0.0
- Build mode: clean Release JIT builds on ZFS, `GGML_SYCL_DEVICE_ARCH` omitted
- Principal models:
  - Mistral-7B-Instruct v0.1 Q4_K_M
  - Meta-Llama-3.1-8B-Instruct Q4_K_M
  - Qwen3-Coder-30B-A3B-Instruct Q3_K_XL
- Standard attention scope: f16/f16 and q8_0/q8_0 KV with FA enabled

JIT builds were used deliberately. Setting `GGML_SYCL_DEVICE_ARCH=acm-g10`
selects offline AOT and is not the campaign's runtime-comparison mode.

## Measurement contract

Performance conclusions use paired campaigns rather than isolated benchmark
rows:

1. Require sole tenancy of `/dev/dri/renderD128` before each timing run.
2. Alternate baseline and candidate arms.
3. Run six launches per arm and discard sample zero.
4. Pair repetitions by index and report paired median, mean, and 95% interval.
5. Preserve argv, environment, stdout/stderr, selected JSON rows, elapsed time,
   exact source/build identity, and raw samples.
6. Fail the entire cell on a nonzero process, missing row, missing requested
   environment evidence, incomplete pairing, inaccessible dmesg, or a new
   case-insensitive i915/xe reset, hang, timeout, GPU-HANG, or device-lost line.
7. Require the relevant correctness gate before promotion.

Unless a task specified a stricter gate, a performance candidate needed at
least +3% median improvement, a paired 95% lower bound above zero, no material
regression in protected cells, passing correctness, and clean dmesg.

Terminology:

- `pp512`: prompt processing for 512 tokens, tokens/s.
- `tg128`: generation of 128 tokens, tokens/s.
- `dN`: KV depth N.
- "promoted" means retained in the branch. It does not necessarily mean
  enabled by default.
- "killed" means the measured gate rejected the hypothesis.

## Complete experiment matrix

### P5.1 - effective oneDNN truth reporting

**Change tested:** replace a preprocessor presence check with an effective-value
check for `GGML_SYCL_DNNL`.

**Result:** **retained** in `7dfee75cd`.

The system oneDNN package configures the backend with
`GGML_SYCL_DNNL=0`. Before the fix, runtime output contradicted the build by
printing `GGML_SYCL_DNNL: yes` while also reporting that DNN was compile-time
disabled. The fixed Release-JIT build prints literal `GGML_SYCL_DNNL: no`.
Mistral and Llama q8_0/q8_0 one-token smokes succeeded and dmesg gained no GPU
fault. This was a reporting fix; oneDNN was not enabled here.

### P5.2 - paired campaign hardening

**Change tested:** extend `scripts/bench-a770-fork-unique.py` with separate
repeatable baseline/candidate environments, exact environment assertions,
requested-KV bandwidth accounting, stronger row selection, complete raw
artifacts, and i915/xe fault invalidation.

**Result:** **retained** in `7ce032b5a`.

Twelve focused tests passed. A synthetic end-to-end product campaign preserved
six launches per arm, sample-zero discard, alternating order, paired raw
samples, exact env maps, q8_0 requested-KV bandwidth, and an empty GPU-fault
delta. Invalid cells fail closed; sole-tenancy failure remains exit 70.

### P5.3 - spill inventory and global large-GRF

**Changes tested:**

- forced-cold IGC inventory for q8_0 VEC D=128;
- Q4_K reorder-MMVQ inventory;
- the live Mistral f16 GQA route;
- global `SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file`.

**Result:** **global large-GRF killed; no source change retained**.

Observed kernels:

| Path | SIMD | GRF | Spill/scratch result |
|---|---:|---:|---|
| q8_0 VEC D=128 | 16 | 128 | zero spill; zero private bytes |
| Q4_K reorder-MMVQ | 16 | 128 | zero spill |
| selected f16 GQA TILE D=128 | 32 | 128 | 7,744-12,480 spill bytes |

The f16-VEC assumption was false: Mistral 4:1 GQA selected TILE. Because a hot
TILE family spilled, the global 256-GRF option was tested across the full
five-depth, two-KV, six-repetition matrix. All 60 candidate model launches
failed before producing rows: the option reduced the per-kernel work-group
ceiling to 512 while RMS_NORM requested more. The focused correctness harness
alone passed, demonstrating why the full-model gate was binding. No optional
large-GRF configuration was retained.

### P5.4 - same-path f16/q8_0 VEC bandwidth

**Change tested:** default-off `GGML_SYCL_FA_FORCE_VEC_STANDARD=1`, forcing
compatible standard f16/f16 and q8_0/q8_0 D=128 decode through VEC.

**Result:** **diagnostic retained** in `8cf0f14c5`; **forced VEC not promoted**.

At depth 16384 under the same forced-VEC route:

| KV | tg128 | Effective requested-KV bandwidth |
|---|---:|---:|
| f16/f16 | 14.4451 tok/s | 124.0825 GB/s |
| q8_0/q8_0 | 9.8929 tok/s | 45.1452 GB/s |
| q8_0 / f16 ratio | 0.68486x | 0.36383x |

The bandwidth ratio was below the 0.77 kill threshold, so the q8 layout
hypothesis remained live for P5.11. The q8 dot path assembles one 4-byte int
from two 2-byte-aligned `uint16_t` reads; it is not a single 2-byte load.

### P5.5 - existing TILE route for q8_0 GQA

**Change tested:** default-off `GGML_SYCL_FA_Q8_GQA_TILE=1`, routing only
q8_0/q8_0 D=128 single-token GQA decode to the existing TILE implementation.

**Result:** **diagnostic retained** in `84da8d97e`; **no default cutover**.

Paired tg128 medians versus VEC:

| Depth | Mistral | Llama-3.1 |
|---:|---:|---:|
| 0 | -5.47% | -5.46% |
| 2048 | -0.60% | -0.62% |
| 4096 | +3.50% | +3.44% |
| 8192 | +10.33% | +10.10% |
| 16384 | +20.40% | +20.03% |

Correctness passed and deep-cell paired lower bounds were positive. However,
every possible tested cutover retained the depth-0 regression below the -2%
shallow guard, so no token-count threshold was encoded and VEC remains the
default.

### P5.6 - SYCL graphs and Level Zero submission

**Changes tested:**

- graph disabled versus the existing per-call graph route;
- with graphs disabled, immediate command lists enabled versus disabled;
- `UR_L0_BATCH_SIZE={1,4,16,64}` with Level Zero V2 disabled.

**Result:** **graphs and batched submission survive as shape-specific
operational levers; no new source default**.

| Candidate | tg128 delta | Other result |
|---|---:|---|
| Graphs enabled | +44.55% | paired lower 95% +44.13%; pp512 -3.05% |
| Batch 1 | -12.31% | killed for this shape |
| Batch 4 | +20.74% | survives |
| Batch 16 | +38.45% | survives |
| Batch 64 | +47.26% | survives |

All five paired arms were valid with empty GPU-fault deltas. `ltrace` confirmed
that the Level Zero adapter read each requested value. The A770 exposed only
`ext_oneapi_limited_graph`, not full graph argument-update support.

### P5.7 - reorder/MMVQ versus DMMV routing

**Change tested:**
`GGML_SYCL_DISABLE_OPT={0,1} x GGML_SYCL_PRIORITIZE_DMMV={0,1}` on dense
Mistral Q4_K/Q6_K and Qwen3-Coder Q3_K MoE.

**Result:** **alternatives killed; default `0/0` retained**.

Each non-default arm regressed shallow tg128:

- Mistral: -24.14%, -24.34%, and -24.19%.
- Qwen3-Coder: -3.57%, -3.57%, and -3.34%.

All paired 95% upper bounds were below zero. Runtime tracing showed that the
default dense path used reorder+MMVQ, the prioritized-DMMV arm emitted
`dequantize_mul_mat_vec_*`, and Qwen's Q3_K MoE stayed on fused MMVQ-MoE in
both cases. Six campaigns were valid and had no GPU-fault delta.

### P5.8 - MMVQ launch geometry

**Change tested:** all twelve compile-time combinations
`GGML_SYCL_MMV_Y={1,2,4}` x
`GGML_SYCL_MMVQ_NUM_SUBGROUPS={4,8,16,32}` across all seven live subgroup
sites.

**Result:** **R6 killed; default 1x16 retained**.

Twelve isolated Release-JIT builds had distinct SYCL library hashes and all
passed correctness with `0 GATE-FAIL`. Twenty-four model campaigns produced
288 launches with valid pairing and no new GPU fault. No non-default geometry
reached +3% tg128 with a positive paired lower bound on both Mistral and
Qwen3-Coder. The best cross-model alternative, 1x4, bottomed out at -0.0148%
median and -0.4847% lower 95%. The guarded parameters remain useful for
measurement, but no architecture lookup or runtime geometry router was added.

Reusable isolated-binary and geometry runners landed in `071e396f5` and
`67411cce8`; their 16 focused tests passed and the reusable runner repeated all
12 correctness cells successfully.

A follow-up generic Q4_0 `MUL_MAT` audit found that the `4x32` cell requests
4,096 work-items per work-group on the A770, above its 1,024 limit. The
reusable runner now excludes every `Y * subgroups * 32 > 1024` cell. The
historical twelve-cell results remain valid for the recorded workload, but do
not establish all-op safety for those oversized geometries. This does not
change the retained `1x16` default or the decision to kill alternate geometry.

### P5.9 - JIT code split and compile-fast

**Changes tested:** isolated baseline, per-kernel device-code split,
compile-fast, and split+compile-fast builds.

**Result:** **per-kernel split promoted; compile-fast killed**.

Six forced-cold process launches per build/model used
`SYCL_CACHE_PERSISTENT=0` and measured first-valid-JSON timing:

- per-kernel split reduced conventional baseline-relative cold-start latency
  by 19.94% on Mistral and 17.01% on Llama-3.1 (equivalent to candidate-relative
  speedups of 24.91% and 20.50%);
- compile-fast changed cold start by only +1.52% and -0.93%.

Warm paired split results were:

| Model | pp512 | tg128 | lower 95% bounds |
|---|---:|---:|---|
| Mistral | +92.69% | +1.74% | +91.17% / +1.25% |
| Llama-3.1 | +85.22% | +1.05% | +84.74% / +0.82% |

All four correctness matrices ended at `0 GATE-FAIL` with zero new GPU faults.
Commit `76108af1a` makes per-kernel split the default and removes the
compile-fast experiment. `scripts/bench-sycl-cold-jit.py` and focused tests
remain as reusable tooling.

### P5.10 - non-PVC upload path

**Change tested:** replace the non-Windows, non-PVC tensor-upload bounce buffer
with direct blocking queue `memcpy`.

**Result:** **killed and reverted** (`c6c98bed1` -> `faa068cd4`).

After one warmup, six alternating warm-page-cache model-load-only processes per
arm measured 2.278841 s baseline versus 2.784785 s candidate median. The
candidate regressed by 0.505943 s, equivalent to a -22.20% "reduction," rather
than meeting the required improvement of at least 0.5 s. Both arms generated
the exact one-token response `Blue`; dmesg recorded no new GPU fault. The
bounce-buffer path remains intact.

### P5.11 - scale-separated q8_0 KV rows

**Change tested:** opt-in `GGML_SYCL_Q8_KV_QUANTS_FIRST=1` plus explicit
`GGML_TENSOR_FLAG_KV_Q8_QUANTS_FIRST` tensor metadata. The 136-byte D=128 row
stores 128 signed quant values followed by four fp16 scales. SET_ROWS,
same-type CPY/defrag, VEC, TILE, generic conversion, backend set/get, and state
I/O were made layout-aware.

**Result:** **opt-in promoted** in `5fa522c1d`; canonical q8_0 remains the
default.

Paired tg128 medians:

| Depth | Mistral | Llama-3.1 |
|---:|---:|---:|
| 4096 | +8.25% | +8.01% |
| 8192 | +13.81% | +13.37% |
| 16384 | +20.62% | +20.43% |

The minimum long-context lower 95% bounds were +7.90% and +7.71%; the worst
pp512 median was -0.39%. Both models retained five valid pairs per depth and
recorded zero new GPU faults.

Correctness covered quants-first GQA 4:1 and 8:1 through TILE and VEC with
cosine at least 0.999957 and `0 GATE-FAIL`. A server-state gate saved 12 tokens
and 836,576 canonical bytes, restored the same count and byte size in another
slot, and reproduced the deterministic continuation. Marker-absent q8_0
weights and host/session bytes remain canonical `block_q8_0`.

### P5.12 - isolated GPU oneDNN

**Changes tested:** isolated oneDNN v3.11.3 GPU-DPCPP build, effective
`GGML_SYCL_DNNL=1`, generated-kernel inspection, and paired Mistral/Llama
pp512.

**Result:** **G2/G1 killed; isolated prefix not adopted**.

The candidate linked successfully, reported `GGML_SYCL_DNNL: yes`, and passed
standard correctness with `0 GATE-FAIL`. `ONEDNN_JIT_DUMP=1` plus `iga64`
proved generated GPU GEMMs used 256/1024/256/256 `dpasw` instructions, so the
candidate was genuinely engaged rather than a configuration no-op.

Five retained pairs nevertheless measured only:

- Mistral pp512: -0.42%, lower 95% -1.21%.
- Llama-3.1 pp512: -0.20%, lower 95% -1.37%.

Both missed the required +5% median with positive lower bound and produced no
new GPU faults. No Q4_K/Q4_0 repacking, oneDNN decompression attributes, or
primitive caching was added. Default builds remain `GGML_SYCL_DNNL=0`.

### P5.13 - dense proxy for MoE reorder

**Change tested:** six alternating `test-backend-ops perf -o MUL_MAT` arms on
model-scale dense Q3_K and Q4_K shapes with reorder disabled/enabled.

**Result:** **G4 killed before MoE implementation**.

Five retained aggregate pairs measured:

| Type | Median | Lower 95% |
|---|---:|---:|
| Q3_K | +2.01% | +1.80% |
| Q4_K | -6.61% | -6.78% |
| Combined fleet | -2.43% | - |

The shape split explained why a global extension was unsafe: Q3_K/Q4_K `n=1`
improved +14.47%/+55.49%, Q4_K `n=8` collapsed -60.13%, and the other measured
shapes stayed within 3.1%. The proxy missed its +3% gate and dmesg stayed
clean. No MUL_MAT_ID reorder allocation, expert transform, SOA trait, or fused
MoE routing change was implemented.

### P5.14 - exact graph replay and FA capture safety

**Changes tested:**

1. exact-signature executable graph replay on DG2's limited graph extension;
2. pre-growth of quantized FA K/V conversion scratch before graph recording;
3. the R4 MoE graph-unlock prerequisite.

**Result:** **R2 killed and reverted; R4 skipped by gate; FA scratch fix
retained**.

Candidate `5df8a5cba` required identical node count, op sequence, tensor shapes,
data pointers, and op-parameter bytes. It preserved exact four-token output on
Mistral, Llama-3.1, and Qwen3-Coder, logged replay hits, passed the standard
correctness harness, and kept dmesg clean. Performance was nevertheless
catastrophic:

| Model/depth | tg128 median | Lower 95% | pp512 |
|---|---:|---:|---:|
| Mistral d8192 | -86.55% | -86.58% | within 0.02% |
| Mistral d16384 | -80.86% | below -80.88% | within 0.02% |
| Llama-3.1 d8192 | -86.21% | -86.25% | within 0.02% |

Replay was reverted by `e142abba1`; per-call graph record/finalize remains.
R4 was skipped because P5.7 produced no routing winner.

The long-depth run also exposed an independent existing abort: quantized FA
scratch growth called `queue::wait()` while the queue was recording. Commit
`f8f311aa4` scans graph FA nodes and grows K/V fp16 scratch before capture. A
fresh exact-source build passed the formerly aborting Mistral d8192 workload at
215.79 pp512 and 17.02 tg128, the standard correctness harness at
`0 GATE-FAIL`, syntax verification, and clean dmesg.

## Runtime and build controls retained

| Control | Default after P5 | Purpose | Campaign decision |
|---|---|---|---|
| `GGML_SYCL_FA_FORCE_VEC_STANDARD` | `0` | Force compatible standard D=128 decode through VEC | Diagnostic only |
| `GGML_SYCL_FA_Q8_GQA_TILE` | `0` | Route q8_0 GQA decode through existing TILE | Deep winner, unsafe shallow default |
| `GGML_SYCL_Q8_KV_QUANTS_FIRST` | `0` | Enable scale-separated q8_0 KV rows | Promoted opt-in |
| `GGML_SYCL_MMV_Y` | `1` | MMVQ compile-time Y geometry | Keep default |
| `GGML_SYCL_MMVQ_NUM_SUBGROUPS` | `16` | MMVQ compile-time subgroup count | Keep default |
| per-kernel device-code split | enabled | Split SYCL device code per kernel | Promoted default |
| `GGML_SYCL_DISABLE_OPT` | `0` | Disable optimized reorder/MMVQ paths | Keep default |
| `GGML_SYCL_PRIORITIZE_DMMV` | `0` | Prefer DMMV over MMVQ | Keep default |
| `GGML_SYCL_DNNL` | `0` with system package | Effective GPU-oneDNN integration | Isolated candidate killed |

The two diagnostic FA controls log their effective values. They should not be
silently enabled in production based only on a deep-context win.

## Commit ledger

| Commit | Disposition | Change |
|---|---|---|
| `d66649dae` | retained | tenancy-gated A770 campaign harness |
| `7dfee75cd` | retained | report effective oneDNN state |
| `7ce032b5a` | retained | harden paired A770 harness |
| `8cf0f14c5` | retained, default-off | standard VEC diagnostic |
| `84da8d97e` | retained, default-off | q8_0 GQA TILE diagnostic |
| `dd553ba69` | retained parameters | MMVQ geometry controls; defaults unchanged |
| `071e396f5` | retained tooling | compare isolated build binaries |
| `67411cce8` | retained tooling | reusable MMVQ sweep runner |
| `785351cf9` | partially retained | expose codegen discriminators |
| `0830a8419` | retained tooling | cold-JIT campaign runner |
| `76108af1a` | retained default | enable per-kernel split, remove compile-fast |
| `c6c98bed1` | reverted | direct non-PVC upload candidate |
| `faa068cd4` | retained revert | restore upload bounce buffer |
| `5fa522c1d` | retained, opt-in | scale-separated q8_0 KV rows |
| `5df8a5cba` | reverted | exact executable-graph replay candidate |
| `e142abba1` | retained revert | restore per-call graph recording |
| `f8f311aa4` | retained fix | grow FA scratch before graph capture |

The candidate/revert pairs remain in history intentionally: they preserve the
exact code that produced each negative result while leaving the branch's net
source state clean.

## Correctness and safety summary

- Every promoted behavioral change passed the repository's standard
  CPU-vs-SYCL correctness harness.
- Quants-first q8_0 specifically passed VEC and TILE GQA 4:1/8:1 coverage and
  canonical state save/restore.
- Exact graph replay matched generated output before its performance gate was
  considered.
- The retained FA scratch fix passed the formerly aborting long-depth workload.
- Every accepted performance campaign had an empty new i915/xe fault delta.
- Large-GRF was rejected on full-model compatibility despite a passing narrow
  oracle.
- oneDNN was rejected on performance despite confirmed DPAS engagement.
- Upload and graph-replay candidates were reverted after their gates failed.

## Review-adjudicated route invariants

- Optional K/V tensors are guarded by `quants_first_layer`, which requires
  both pointers before setting either layout flag.
- Non-contiguous q8_0 converter strides are canonical-block counts derived
  from `nb / type_size`, not byte offsets; grouped byte addressing is therefore
  intentional rather than a second stride scaling.
- Forced q8_0 GQA TILE supports quants-first rows through the source-aware
  `ggml_get_to_fp16_nc_sycl(type, tensor)` conversion used by TILE. The focused
  forced-TILE 4:1 and 8:1 correctness cells remain the binding route proof.

### Post-review hardening

The branch was hardened after the archived campaign without rewriting its
measurements:

- mixed canonical/quants-first q8_0 CPY now performs an explicit on-device
  layout conversion instead of asserting; canonical-to-quants-first,
  quants-first-to-canonical, and non-contiguous same-layout probes all pass;
- the product runner rejects incomplete inputs, non-finite timing values, fewer
  than three repetitions, and an explicit candidate binary without an explicit
  candidate environment;
- the cold-JIT runner invalidates stale products at startup, bounds shutdown
  after stdout EOF, records its effective library path, validates every JSON
  timing row, and uses Student-t critical values beyond 30 degrees of freedom;
- the geometry runner parses the exact `GATE-FAIL` count, requires same-identity
  correctness before benchmarking on `level_zero:0`, fingerprints dirty source
  contents, rejects duplicate model labels, and excludes workgroups above the
  A770's 1024-work-item limit before build.

The original 12-build geometry archive remains historical evidence for the
campaign as executed. The reusable runner now schedules only the nine valid
A770 cells. No model-scale performance measurements were rerun during this
review hardening.

Current verification after these fixes: 57 focused Python tests pass; the
rebuilt SYCL correctness harness reports `0 GATE-FAIL`, including all three
layout-aware q8_0 CPY probes; the host KV guard test passes; and the focused
MMVQ small-row matrix passes 10/10 cases.

## Reusable repository tooling

- `scripts/bench-a770-fork-unique.py`: fail-closed paired product campaigns,
  environment A/B, bandwidth accounting, raw evidence, and GPU-fault checks.
- `scripts/bench-sycl-cold-jit.py`: fail-closed cold-process JIT comparisons
  with persistent caching disabled and complete runtime provenance.
- `scripts/sweep-a770-mmvq-geometry.py`: identity-coupled
  build/correctness/benchmark sweeps across A770-valid MMVQ geometry.
- `scripts/test_bench_a770_fork_unique.py`: focused campaign-contract tests.
- `scripts/test_bench_sycl_cold_jit.py`: focused cold-JIT runner tests.
- `scripts/test_sweep_a770_mmvq_geometry.py`: focused geometry, identity, and
  correctness-ledger tests.
- `scripts/README.md`: command catalog and expected usage.

## Evidence map

The model-scale logs and compiler dumps are too large for the repository.
Their original `/tmp` names are retained below for correspondence with the
commands and verdicts, but the complete surviving P5 set was rescued to
durable ZFS storage after the campaign. Map any `/tmp/NAME` entry below to
`/home/svnbjrn/llama-p5-evidence/2026-07-19/tmp/NAME`.

| Task | Principal evidence |
|---|---|
| P5.1 | `/tmp/p5-p51-isolated-*` |
| P5.2 | `/tmp/p5-p52-smoke/out-1/` |
| P5.3 | `/tmp/p5-b2-7ce032b5a-*`, `/tmp/p5-p53-large-grf-7dfee75cd/` |
| P5.4 | `/tmp/p5-a1-bandwidth-8cf0f14c5/` |
| P5.5 | `/tmp/p5-p55-*` |
| P5.6 | `/tmp/p5-p56-*` |
| P5.7 | `/tmp/p5-p57-*` |
| P5.8 | `/tmp/p5-p58-dd553ba69-verdict.json` and related campaign dirs |
| P5.9 | `/tmp/p5-p59-785351cf9-verdict.json` and manifest |
| P5.10 | `/tmp/p5-p510-load-c6c98bed1/` |
| P5.11 | `/tmp/p5-p511-5fa522c1d/` |
| P5.12 | `/tmp/p5-p512-5fa522c1d/` |
| P5.13 | `/tmp/p5-p513-proxy-5fa522c1d/` |
| P5.14 | `/tmp/p5-p514-5df8a5cba-verdict.json`, `/tmp/p5-p514-5df8a5cba-manifest.sha256` |

P5.14's verdict SHA-256 is
`283ba030206d4110ff169c9a6e1623b485e6783f04b696e27c8b00553ef7ee73`; its
evidence-manifest SHA-256 is
`1c281d09ce91730f869d1ca06742364d03ea70a2a4bb558491e35a413a894590`.

### Durable evidence rescue

The 2026-07-19 rescue copied every surviving top-level `/tmp/p5-*` entry
without deleting or rewriting its source:

- durable root: `/home/svnbjrn/llama-p5-evidence/2026-07-19`;
- 366 top-level entries, 8,011 evidence files, and 411 directories;
- 2,011,368,030 evidence bytes (1.873 GiB);
- source-to-destination SHA-256 mismatches: zero;
- `SHA256SUMS`: 8,013 entries, covering all evidence plus
  `rescue-metadata.json` and `top-level-entries.txt`;
- `SHA256SUMS` SHA-256:
  `4579f9a79acc510b37a285d199ab55e87e588ddaa50b6f68dff614dead29291a`;
- `verification.json` SHA-256:
  `0ee61bb988dae4dc49b6c747010d82e18b1f446b9d500594aa06898219185178`.

The independent verification command completed successfully:

```bash
cd /home/svnbjrn/llama-p5-evidence/2026-07-19
sha256sum --quiet -c SHA256SUMS
```

## Operational recommendations

1. Keep per-kernel device-code split enabled.
2. Keep canonical q8_0 KV rows as the production default until the opt-in
   quants-first format receives broader architecture and lifecycle coverage.
3. Use q8_0 TILE only as an explicit deep-context experiment; do not infer a
   safe cutover from the deep wins because depth 0 regressed by about 5.5%.
4. Keep global reorder/MMVQ routing and 1x16 geometry unchanged.
5. Do not deploy the isolated GPU-oneDNN prefix for this workload.
6. Keep the upload bounce buffer and per-call graph record/finalize behavior.
7. Preserve FA scratch pre-growth before graph capture.
8. Reuse the paired harness for future claims; unpaired or contended A770
   numbers are not promotion evidence.

## Post-P5 performance campaign - 2026-07-22

The ordered post-P5 campaign ran from source base `d66649dae` and produced a
clean final candidate at binary source commit `d7a2bcf1d`. It retained the P5
behavioral state. No new backend default cleared every promotion gate.

### Final candidate matrix

The clean Release-JIT candidate used oneAPI 2026.0, IGC 2.36.3,
compute-runtime 26.22.38646.4-1.1, Level Zero loader 1.28.6-1.1, i915, graph
support, and per-kernel device-code split. Each cell alternated the held and
final binaries six times, discarded repetition zero, and retained five pairs.

| Model | Depth | pp512 paired median | tg128 paired median | tg128 held/final |
|---|---:|---:|---:|---:|
| Mistral | 0 | -0.317% | +0.019% | 24.601 / 24.597 |
| Mistral | 4096 | +0.066% | -0.064% | 17.932 / 17.920 |
| Mistral | 8192 | -0.017% | +0.029% | 14.119 / 14.138 |
| Mistral | 16384 | -0.129% | +0.010% | 9.931 / 9.933 |
| Llama-3.1 | 0 | +0.093% | +0.010% | 23.605 / 23.561 |
| Llama-3.1 | 4096 | +0.004% | -0.080% | 17.372 / 17.362 |
| Llama-3.1 | 8192 | -0.028% | +0.115% | 13.810 / 13.813 |
| Llama-3.1 | 16384 | -0.081% | -0.094% | 9.767 / 9.760 |

All cells were valid, every protected regression stayed well inside -2%, and
both harness-level pre/post kernel-log deltas were empty.

The final correctness binary reported
`0 GATE-FAIL, 0 XPASS, 6 xfail, 0 SKIP`. It covered standard f16/q8_0 VEC and
TILE at D64/D128, GQA 4:1 and 8:1, layout-aware q8_0 copies, and the known
turbo2 FA xfails. Its startup log confirmed the retained effective state:
Level Zero enabled, graph disabled only by the harness, standard VEC force off,
and q8 GQA TILE force off. The canonical state protocol saved 12 tokens and
836,576 bytes, restored the same counts across slots, and reproduced the
original continuation. The final kernel log query found no new i915 hang,
reset, fault, or wedge.

### Post-P5 decisions

| Arm | Verdict | Reason |
|---|---|---|
| Quants-first plus TILE | killed as a composition | Slower than either single mechanism at deep cells |
| Direct packed-q8 GQA stage 1 | killed and reverted | d16384 +7.59% missed its 20% keep gate; d4096 was -1.84% |
| Global graph/batching policy | no default | No one policy cleared +10% at every depth and parallel shape |
| Mutable Level Zero command lists | no-go before integration | Driver lacks kernel-argument mutation |
| Dense ROPE fusion | no-go before integration | 0% passed the production donor predicate; SET_ROWS destinations were q8_0 |
| Regular SYCL q8 prefetch | killed and reverted | All four engaged arms missed +3% and deep tg regressed about 1% |
| Global q8_0 speculation | not promoted | Copy-heavy prompts were exact, but free prose produced four hashes |

The direct-kernel candidate/revert pair is `2d53f3b25` / `0bb42498e`. The
prefetch candidate/revert pair is `4dba2654c` / `cadcd0393`. Gate-only arms did
not mutate backend behavior and therefore required no synthetic revert commit.
Successful route, submission, ROPE, and speculative evidence tooling remains
in history.

The final speculative oracle reproduced the product split. With canonical
q8_0 KV, `ngram-mod,ngram-map-k4v` was 4.15x on `code_edit` and 2.50x on
`multi_turn`, with exact target-only hashes, but 1.65x on `free_prose` with
four hashes across five deterministic repetitions. Global q8_0 speculation
therefore remains off. Controlled q8_0 copy-heavy classes may opt in. The
fixed f16 suite was target-exact and exceeded 1.5x in every class. The hostile
hard-off gate was also exact: target-only 21.82 t/s, unguarded 15.30 t/s, and
dead-off-3 27.83 t/s with six observed trips.

Durable post-P5 evidence is under
`scripts/perf/results/p5-post-campaign/`. The final build provenance, complete
CMake cache, matrix adjudication, correctness log, state result, speculative
summaries, and killed-arm ledger are in `phase7-final*`.
