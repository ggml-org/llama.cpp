# Round 2: SYCL decode-throughput candidates for Arc A770 (Xe-HPG, acm-g10)

Date: 2026-07-25. Fork HEAD at grounding: `028084d00`. Branch: `feature/quants-first-default`.

Round 1 asked "what avenues exist". Round 2 was asked for implementation-grade design
parameters and falsification criteria. What it found first is that a large part of the brief's
inherited state does not survive contact with the fork, and that the acceptance gate as stated
is unachievable for most of the candidates by arithmetic rather than by engineering.

This document leads with those two findings because they reorder everything after them.

---

## 1. The reframing: at depth 0, attention is 0.58% of the token

From the held-binary paired benchmarks in
`scripts/perf/results/p5-post-campaign/phase7-final/adjudication.json`, Mistral-7B Q4_K_M with
q8_0/q8_0 KV, tg128: 24.6008 tok/s at depth 0, 17.9318 at 4096, 14.1190 at 8192, 9.9310 at
16384.

Converted to ms/token, the marginal cost per KV row is almost perfectly linear:

| interval | delta ms | rows | us per KV row |
|---|---:|---:|---:|
| 0 to 4096 | 15.117 | 4096 | 3.691 |
| 4096 to 8192 | 15.060 | 4096 | 3.677 |
| 8192 to 16384 | 29.869 | 8192 | 3.646 |

Llama-3.1-8B reproduces it at 3.711 and 3.659 us. The fit is

```text
per-token time ~= 40.41 ms + 3.67 us * depth
```

A `tg128` run at `-d 0` generates 128 tokens from an empty cache, so mean depth is about 64.
That gives the attention share of a token, and therefore the flash-attention speedup required
to move the end-to-end number by 5%:

| depth | attention share of token time | FA speedup needed for +5% end-to-end |
|---:|---:|---:|
| 0 | 0.58% | impossible (exceeds infinite) |
| 4096 | 27.4% | +17.4% |
| 16384 | 60.0% | +7.9% |

**Consequence.** A conjunctive ">=5% at every depth" gate cannot be cleared by any
flash-attention change, however good, because making attention free gains 0.58% at depth 0.
Every FA candidate in this project's history shows the same deep-win/shallow-lose shape, and
that shape is Amdahl's law, not a tuning failure.

**Gate adopted for this round, disjunctive and depth-weighted.** This is the single authoritative
definition; every kill criterion in this document and in the campaign report refers back to it.

- **Promotion depths, improvement bar:** >=5% at **8192 and 16384**.
- **Protected depths, -2% no-regression guard:** **0, 2048 and 4096**. These carry a guard rather
  than an improvement bar, because section 1's arithmetic makes an improvement bar unreachable
  there for attention work.
- **Measurement set:** all five depths - 0, 2048, 4096, 8192, 16384 - are benched. The guard
  cannot be evaluated on a cell that was not measured.

This is also how P5.5 and P5.11 were actually adjudicated. The harness's own promotion rule
(>=+3% median, paired 95% lower bound above 0, -2% guard on protected cells) remains the merge
gate; candidates landing in the 3-5% band are surfaced, not discarded.

### 1.1 Where the other 40.4 ms goes - and the number that reframes the programme

Mistral-7B Q4_K_M streams 4.295 GB per token after excluding `tok_embd` (one row read at decode).
At an achievable 450 GB/s (about 80% of peak; ESTIMATE) the bandwidth floor decomposes as:

| Op class | est. ms/token | Basis |
|---|---:|---|
| MUL_MAT FFN (gate/up/down, 3.401 GB) | 7.56 | 5.637G elements x 0.6033 B/element |
| MUL_MAT attention projections (0.810 GB) | 1.80 | 1.342G elements |
| MUL_MAT output head (Q6_K, 0.108 GB) | 0.24 | 131.1M elements x 0.820 B/element |
| Everything else (GLU, ADD, ROPE, RMS_NORM, SET_ROWS, quantize, FA at d0, logits) | 0.04 | about 16 MB of traffic in total |
| **Sum** | **9.64** | |
| **Measured** | **40.41** | `phase7-final/adjudication.json` |
| **Unaccounted** | **30.77 (76% of the token)** | |

Effective bandwidth achieved is 4.295 GB / 40.41 ms = **106 GB/s, 19% of peak.** Note that every
non-`MUL_MAT` op in the decode graph moves about 16 MB in total - 36 microseconds of data. Their
entire cost is launch overhead. **The Amdahl argument generalises well past attention: everything
except weight streaming is about 0.1% of the token by physics.**

**Node inventory (ESTIMATE, enumerated from `ggml-sycl.cpp:5007-5029`, which skips
RESHAPE/VIEW/PERMUTE/TRANSPOSE/NONE):** 19 dispatched nodes per layer, plus **7 hidden
`quantize_row_q8_1_sycl` launches per layer** (`ggml-sycl.cpp:3020`, one per MMVQ `MUL_MAT`;
q, k and v each re-quantize the *same* 4096-float vector, as do gate and up). That is about
**837 kernel enqueues per token**.

Splitting the 30.77 ms:
- **Submission: 12.45 to 12.97 ms, measured-derived not estimated.** Graphs enabled at +44.55%
  implies 27.96 ms (delta 12.45); `UR_L0_BATCH_SIZE=64` at +47.26% implies 27.44 ms (delta 12.97).
  12.45 ms over 837 enqueues is **14.9 microseconds per enqueue**, and Intel's own guidance
  corroborates the mechanism: with a single queue and kernels under 10 microseconds, immediate
  command lists regress because host submission time dominates.
- **About 18.2 ms cannot be accounted for.** It is kernel execution above roofline, 45% of the
  token, explained by neither bandwidth nor measured submission cost. Stated plainly rather than
  given an invented name.

### 1.2 The comparison that matters, and it inverts the brief's hypothesis

No published tg128 exists for a 7-8B **Q4_K_M** on A770; every microarchitecture-tagged A770
number found is Q4_0. What exists, all Xe-HPG A770:

| Backend | Model | tg128 |
|---|---|---:|
| SYCL (upstream) | llama-2-7b Q4_0, FA off | 55.53 / 59.03 t/s |
| SYCL (upstream) | same, FA on | 64.49 / 67.09 t/s |
| Vulkan (upstream) | llama-2-7b Q4_0, FA off | 52.56 t/s |
| Vulkan (upstream) | same, FA on | 53.07 t/s |
| IPEX-LLM | - | no admissible number (untagged aggregator blogs only) |
| **This fork** | Mistral-7B **Q4_K_M**, q8_0 KV, FA on | **24.60 t/s** |

**Vulkan is not faster than SYCL on the A770** - they land within 12% of each other. There is no
backend tax to recover and no swap-the-backend escape hatch. That kills the hypothesis this
comparison was meant to test.

What it raises instead is worse. Upstream SYCL on this exact part reaches 59.03 t/s on 3.83 GB of
Q4_0 weights, which is **226 GB/s effective, 40% of peak**. If Q4_K_M's 4.295 GB streamed at that
same rate it would be 19.0 ms, i.e. **52.6 tok/s**. The fork measures 24.6. **The fork appears to
be about 2.1x slower than an upstream SYCL run on the same hardware.**

Confounds, stated rather than buried: Q4_0 versus Q4_K_M kernels (superblocks with 6-bit packed
scales are genuinely more ALU per byte); q8_0 KV here versus f16 there; different base commit; and
a much newer driver stack here. **None of them individually explains 2.1x**, and the whole claim
rests on a cross-source comparison rather than a paired run - which is exactly why reproducing it
locally is the first action in section 6, ahead of every optimisation.

**The two mechanisms above 40% in this fork's history** - graphs enabled at +44.55% tg128 (paired
lower-95% +44.13%, pp512 -3.05%) and `UR_L0_BATCH_SIZE=64` at +47.26% - were killed as *global*
defaults because "no one policy cleared the >=10% operational gate at every depth and parallel
shape" (`killed_arms[2]`). Rejected for lacking a universally safe setting, not for lacking
magnitude. Note also that the A770 exposes only `ext_oneapi_limited_graph`, not
`ext_oneapi_graph`, so `ggml-sycl.cpp:5313-5320` takes the `finalize()` branch on **every decode
step** - a full record and finalize per token - and still wins 44.55%.

---

## 2. Premise corrections

Load-bearing claims in the round-2 brief that the fork contradicts. Each was verified against
source or against the live host, not inferred.

| Brief claims | Actual | Effect |
|---|---|---|
| KMD is `xe`, not `i915`; driver-feature coverage is "an open variable" (2.5) | `lsmod` shows **`i915`** loaded, `xe` absent. Live toolchain matches the pins exactly: icpx 2026.0.0.20260331, IGC 2.36.3, NEO 26.22.38646.4, L0 loader 1.28.6, kernel 7.1.3-273-tkg-bore | **Angle E.3 void.** It asked which KMD is in play; the answer is measured, not researchable |
| L0 mutable command lists are an open track (E.2) | Already probed in-repo (`ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cpp`, 563 lines) and adjudicated in `phase3-mutable-command-list-adjudication.json`: A770/i915 advertises `ZE_experimental_mutable_command_list` v1.1 but reports `kernel_arguments: false`, `graph_arguments: false`, so `implementation_authorized: false` | **E.2's L0 half dead** |
| Retained wins include L0 v2 backend +8.72%, fused top-k MoE +8.47%, `RMS_NORM`+`MUL` fusion +3.28%, batched command lists +6.95% (2.1) | No L0 v2 backend exists. `RMS_NORM`+`MUL` fusion is **not implemented** (`ggml/src/ggml-sycl/norm.cpp:451` is standalone). Fused top-k MoE is **not implemented**; what exists is a different thing, fused MoE TG `mul_mat_id` mmvq (`ggml-sycl.cpp:4210`, `mmvq.cpp:2484/2534`). "Batched command lists" was a `UR_L0_BATCH_SIZE` env lever whose verdict was "no new source default", and batch-1 measured **-12.31%**. Only per-kernel device-code split is a real retained default (`ggml/CMakeLists.txt:209`, ON) | The assumed baseline is wrong. Any reasoning that chains these as compounding wins is unsound |
| Graph code at `ggml-sycl.cpp:5048-5059`, `update()` at `:5102` | Graph path is `ggml-sycl.cpp:5185-5379`; `check_graph_compatibility` at `:5186-5225`; `update()` plus its catch at `:5323-5344`. `GGML_SYCL_DISABLE_GRAPH` **defaults to 1** | Graphs are off by default despite the +44.55% measurement |
| `handler::set_arg` restructuring is the blocker for graph mutation | Worse than described: **276 `parallel_for` sites across 39 files with no launch helper at all**. Upstream's `sycl_parallel_for` wrapper is absent from this fork, so every launch is a hand-written lambda capture | A helper must be introduced before the refactor can even begin |
| Fleet is Mistral-7B, Llama-3.1-8B, Qwen3-Coder-30B-A3B, "all Q4_K_M" (2.5) | Only Mistral matched. Llama-3.1 existed solely as the abliterated "heretic" variant and Qwen solely as UD-Q3_K_XL. **Both stock Q4_K_M files have now been fetched** and verified `general.file_type = 15` | Fleet is now literal. No P5 number measured on Q3_K_XL Qwen or heretic Llama is a valid baseline for the new files |
| Acceptance gate is >=5% tg | The harness implements >=+3% median with paired 95% lower bound above 0, plus a -2% guard on protected cells | Both reported; see section 1 for the gate actually adopted |

### Confirmed model shapes (GGUF metadata, not assumed)

| Model | heads | KV heads | GQA r | head dim | layers | notes |
|---|---:|---:|---:|---:|---:|---|
| Mistral-7B-Instruct-v0.1 Q4_K_M | 32 | 8 | 4 | 128 | 32 | dense |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | 32 | 8 | 4 | 128 | 32 | dense, ctx 131072 |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | 32 | 4 | **8** | 128 | 48 | MoE, 128 experts, 8 used, ctx 262144 |

Trap worth restating: Qwen's head dim is 128 from `qwen3moe.attention.key_length`, **not**
`embedding_length / head_count` = 2048/32 = 64.

### Hardware constants used in the analysis below

| Constant | Value | Source |
|---|---|---|
| L2 | 16 MB | Chips and Cheese A770 microbenchmark; no source found claiming a lower effective size |
| L1/SLM per Xe-core | at least 192 KB, configurable (up to 192 KB L1 or up to 128 KB SLM) | as above; Intel oneAPI SLM guide |
| Xe-cores / XVEs | 32 Xe-cores, 512 XVEs, 16 XVEs per Xe-core | as above |
| GRF per thread | **128 registers x 32 B = 4 KB** default; `grf_size<256>` doubles it and halves threads per XVE from 8 to 4 | IGC `Platform.hpp:918`: `getGRFSize() { return isCoreChildOf(IGFX_XE_HPC_CORE) ? 64 : 32; }`. Cross-checks against the whitepaper: 32 KB register file, 8 threads/XVE, 128 x 32 B x 8 = 32 KB exactly |
| Cache line | 64 B | `intel/compute-runtime` `shared/source/xe_hpg_core/hw_cmds_xe_hpg_core_base.h:94` |
| DRAM | 560 GB/s (256-bit GDDR6 at 17.5 Gbps) | Intel product page |
| Peak L2 bandwidth | about 5 TB/s (32 banks x 64 B/cycle at 2.4 GHz), theoretical | Chips and Cheese |

Per-XVE throughput, from Intel's *Introduction to the Xe-HPG Architecture* whitepaper Table 1
(ops/clock): FP32 MAD 16, **FP16 MAD 32**, INT16 MAD 32, **INT8 DP4A 64**, XMX FP16 DPAS 128,
XMX INT8 DPAS 256. So **dp4a is exactly 2x the fp16 vector path, and DPAS is exactly 4x its
vector counterpart**. That 4x ceiling is the number that decides angle C.

**Correction issued during this round.** An earlier pass in this round recorded GRF as 128 x 64 B
= 8 KB per thread, sourced from a Chips and Cheese article describing Xe3. IGC's own platform
code says 64-byte GRFs are Xe-HPC-and-later; DG2 is 32-byte. The per-thread budget is 4 KB, so a
per-work-item dword at sub-group 16 occupies 64 B = two GRFs and the budget is about 64 dwords,
not 128. Any register-pressure estimate in this round that assumed 8 KB is off by 2x in the
optimistic direction.

---

## 3. Four of the six angles were already measured on this hardware

The brief treats angles A, B, C and F as open research questions. The campaign record contains
paired A/B results for all four.

### Angle A - separate scale planes. Built, won, and still not the default.

P5.11 (`docs/research/sycl-a770-p5-performance-campaign-2026-07-19.md:296-323`) is the brief's
own hypothesis, already implemented as the "quants-first" layout: a 136-byte D=128 row holding
128 signed quant values followed by four fp16 scales, replacing four interleaved 34-byte
`block_q8_0` records.

| Depth | Mistral | Llama-3.1 |
|---:|---:|---:|
| 4096 | +8.25% | +8.01% |
| 8192 | +13.81% | +13.37% |
| 16384 | +20.62% | +20.43% |

Minimum long-context lower 95% bounds +7.90% and +7.71%; worst pp512 median -0.39%; cosine at
least 0.999957; `0 GATE-FAIL`; server-state gate passed. Promoted **opt-in** (`5fa522c1d`)
behind `GGML_SYCL_Q8_KV_QUANTS_FIRST`; canonical q8_0 remained the default.

So angle A is not "should we try SoA". It is "SoA clears the gate at every depth from 4096 up
and is still not shipped".

Implementation anchors: `ggml/src/ggml-sycl/fattn-common.hpp:309-336` (K dot), `:673-703`
(V dequant); writer `ggml/src/ggml-sycl/set_rows.cpp:222-270`; host repack
`src/llama-kv-cache.cpp:23-40`; per-layer gate `src/llama-kv-cache.cpp:507-524`.

### Angle B - GQA grouping. Tried twice, same shape both times.

P5.5 (`campaign:164-184`) routed q8_0 GQA decode to the TILE kernel, which already carries the
full `ncols2` head-packing machinery (`ggml/src/ggml-sycl/fattn-tile.hpp:1155-1214`):

| Depth | 0 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|
| Mistral | **-5.47%** | -0.60% | +3.50% | +10.33% | **+20.40%** |
| Llama-3.1 | **-5.46%** | -0.62% | +3.44% | +10.10% | **+20.03%** |

Correctness passed and deep-cell lower bounds were positive. Every tested cutover kept the
depth-0 regression below the -2% shallow guard, so none was encoded.

A purpose-built packed-q8 GQA kernel was then written and reverted: commit `2d53f3b25`
"perf(sycl): add packed q8 GQA kernel" (`fattn-vec.hpp` +138, `fattn.cpp` +45), scoring
**d16384 +7.59%, d4096 -1.84%**, reverted in `0bb42498e` (`killed_arms[1]`).

The FA VEC decode kernel still hard-codes `ncols2 = 1` at `ggml/src/ggml-sycl/fattn-vec.hpp:604`.

**An L2-capacity explanation for the crossover was proposed and then REFUTED.** The model: with
`n_embd_k_gqa = 1024` a q8_0 KV row is 1088 B for K and the same for V, so a layer's working set
is 2176 B per token - 8.9 MB at 4k, 35.7 MB at 16k - and the 16 MB L2 holds the whole 4k layer,
making the `gqa_ratio` re-reads L2 hits below a capacity crossover at d = 7710. The arithmetic is
right and L2 = 16 MB is confirmed, but the inference is wrong on four counts:

1. **Wrong curve shape.** Marginal P5.5 gain per 2048 tokens is monotonically *decreasing*:
   +4.87, +4.10, +3.42, +2.52. A capacity threshold predicts flat-then-step-up at 7.7k. There is
   no kink anywhere.
2. **Wrong crossover.** The table crosses zero between 2048 (-0.60%) and 4096 (+3.50%), so about
   2400 - a 3.2x miss. The original fit was rescued only by measuring against where the table
   crosses *5%*, which is an unmotivated reference line.
3. **The mechanism cannot operate.** L2 is not a KV scratchpad. Between two visits to layer L,
   decode streams about 4.3 GB of Q4_K_M weights plus 31 other layers' KV through 16 MB. Only the
   `gqa_ratio` re-reads could hit L2 at all, so the model predicts packing saves nothing below
   7.7k - yet +3.50% is measured at 4096.
4. **A DRAM-capacity model answers the wrong question**, because the kernel is not DRAM-bound:
   P5.4 measured forced-VEC q8_0 at d16384 achieving 45.15 GB/s against a 560 GB/s bus.

OpenVINO's `min_gqa_sequence_len = 4096` further undercuts it: that single constant is applied
across Intel GPUs whose L2 spans roughly 4 MB (Xe-LPG) to 16 MB (A770) to about 200 MB (PVC). A
capacity crossover cannot be one constant across that range - so even the closest production
analogue is not capacity-derived, and its value sits nearer the measured 2400 than the model's
7710.

**Leading replacement hypothesis: occupancy, not capacity.** `fattn-common.hpp:1200` sets
`ntiles_z_gqa = ceil(gqa_ratio/ncols2)`, so at `ncols2 = 4` with `gqa_ratio = 4` the z-dimension
work-group count collapses from 32 to 8. The only compensator is `parallel_blocks`, capped at
`ntiles_KQ = ceil(K->ne[1]/nbatch_fa)` with `nbatch_fa = 64`. At depth 0 the KV is padded to
`FATTN_KQ_STRIDE = 256`, so `ntiles_KQ = 4` and the packed route fills roughly half a wave while
VEC fills all of it. The deficit relaxes continuously with depth, which matches the decelerating
curve rather than a step. It also matches the independently documented mechanism from the
Character.AI / Colfax FA-3 inference work (NVIDIA H100, mechanism-only): head packing "reduce[s]
the total number of waves by a factor of N" and therefore "requir[es] split-KV kernel to maintain
GPU occupancy at small batch sizes". **This hypothesis is source-derived and unconfirmed**, and it
is decidable by a host-side probe with no kernel work - see C2.

### Angle C - XMX. Already exists, and runs at 43% of baseline.

An opt-in `joint_matrix` FA path is present: `ggml/src/ggml-sycl/fattn-xmx.cpp` (self-labelled
SCAFFOLD), env `GGML_SYCL_FA_XMX`, router gate `fattn.cpp:494-501`, tiles TM=8/TN=8/TK=16,
sub-group 8, D in {128, 256}.

Measured on A770 with llama31-8b
(`docs/research/2026-07-09-a770-benchmark-results-incremental.md:118-155`):

| case | prompt tok/s | gen tok/s | gen vs baseline |
|---|---:|---:|---:|
| q8_0/q8_0 baseline | 324.31 | 25.29 | - |
| XMX q8_0/q8_0 | 222.95 | **10.90** | 43.1% |
| f16/f16 baseline | 326.65 | 24.85 | - |
| XMX f16/f16 | 157.97 | **6.72** | 27.0% |

Three structural defects, all in `fattn-xmx.cpp`: launch geometry is `local(1,1,8)` at
`:171-176`, one sub-group per work-group, so decode puts roughly one SIMD8 thread on each of 32
Xe-cores; `Br = 8` and `XMX_TM = 8` while decode has `ne1 = 1`, so seven of eight computed rows
are discarded; and the q8_0 dequant at `:27-29` is scalar with an integer divide and a modulo
per element, invoked per element in the SLM staging loop at `:89-98`. XMX is additionally
mutually exclusive with the quants-first layout by construction (`fattn.cpp:497-498`).

**But fixing all three would only reach parity, and two further findings close the angle for
decode outright.**

*The sub-group-16 ICE is a hardware gate, not a compiler bug.* IGC:
`bool hasExecSize16DPAS() const { return isCoreChildOf(IGFX_XE_HPC_CORE); }`
(`IGC/Compiler/CISACodeGen/Platform.hpp:587`). `IGFX_XE_HPG_CORE < IGFX_XE_HPC_CORE`, so DG2
returns false. Exec-size-16 DPAS is a Xe-HPC-and-later capability; the ICE is a missing frontend
diagnostic. **Sub-group 8 is not a workaround, it is the only legal mode, and no IGC release can
fix this.** A search of intel-graphics-compiler and intel/llvm issues returned nothing, because
there is nothing to file. This answers the brief's angle C.1 definitively and removes any reason
to bump the toolchain pin for it. Runtime-queried DG2 int8 DPAS shape is `M<=8, N=8, K=32`
(s8/u8 mixed to s32), so the fork's `XMX_TN = 8` is correct and *is* hitting real DPAS - the `N=16`
in most oneAPI prose is the PVC column.

*The arithmetic forecloses the whole family on this part.* Per the whitepaper table in section 2,
DPAS is 4x its vector counterpart on Xe-HPG. Decode attention with GQA ratio `r` has an intensity
of `r` MAC per KV element, so at 2.1 GHz and 560 GB/s the bandwidth ceiling for r=4 is about
1.12 T MAC/s against a 17.2 T MAC/s vector fp16 peak - **the vector ALU is already about 15x
idle**. A 4x ALU ceiling cannot cash a 15x surplus at any tile shape, occupancy or precision. The
crossover where the vector path even saturates is about **61-65 query rows**, not the 4 to 8 that
GQA grouping supplies; reaching it would need `n_draft` around 16 at r=4. This is why the NVIDIA
int8-attention literature transfers badly: tensor cores there are 8-16x the SIMD path, not 4x.

*Corollary worth stating, because it points the other way.* The same 15x ALU headroom is precisely
what makes quants-first and arbitrarily expensive dequant affordable - halving KV traffic converts
near 1:1 into decode speedup. XMX and quants-first are not in tension; the `fattn.cpp:497-498`
mutual exclusion costs nothing.

*Upstream reached the same gate independently and shipped the other side of it.* PR #25222 (merged
2026-07-15) adds `fattn-onednn.cpp` routing FA to the oneDNN Graph SDPA XMX primitive, gated to
f16 KV, single sequence, and **prefill of at least 32 tokens**: pp512 1.21x, pp80000 4.26x, with
**no decode numbers published or claimed**. The PR body states the motivation as prefill
degradation. Tag: Xe2 Arc Pro B70, so magnitudes are mechanism-only here.

*The brief's angle C.3, answered: the integer-attention papers' scale trick does not port to ggml
q8_0.* INT-FlashAttention (arXiv 2409.16997) applies **per-token, i.e. per-row** scales to Q and K
and rescales after accumulation as `S = diag(S_Q)(Q K^T) diag(S_K)`; V is per-tensor only, which
the authors flag, and P uses a fixed `S_P = 1/R` with R=256 folded into the softmax denominator.
It is evaluated on **prefill only, 1k to 16k, with no decode evaluation**. ggml's q8_0 scales are
per-block-of-32 **along D** - a K-axis scale, not a row scale - so it cannot be a post-accumulation
diagonal at all and must instead be folded per K-slice. K=32 happens to be exactly the DG2 int8
tile depth, so it lands at tile granularity and is workable, but doing so forfeits the papers'
central win. SageAttention2's P/V half has no Xe-HPG analogue whatsoever: `hasFP8Dpas()` requires
`IGFX_XE3P_CORE` (`Platform.hpp:585`). No SYCL or Intel-XPU port of either paper exists.

### Angle F - speculation does not move this metric at all.

The gate is a `tg128` row from `llama-bench`, which runs no drafter. Speculation was measured by
a separate oracle (`scripts/perf/bench_spec.py`; results in `phase6-speculation/` and
`phase7-final/adjudication.json` under `speculative_oracle`). With q8_0 KV and
`ngrammod+mapk4v`: `code_edit` 4.155x and target-exact; `multi_turn` 2.504x and target-exact;
`free_prose` 1.652x and **not** target-exact, producing four distinct output hashes across five
deterministic repetitions. Under f16 KV the same arms are target-exact (`free_prose` 11.456x,
one hash). Recorded as `killed_arms[6]`.

The open problem is therefore a correctness one - why q8_0 KV is non-deterministic under
speculative verification when f16 KV is not - and it has no `tg128` payoff.

**That correctness problem is now root-caused.** With `gqa_ratio = 4` and D=128 (Llama-3.1-8B),
`gqa_opt_applies` is true, so:
- **f16 KV** fails the `!gqa_opt_applies` guard at `fattn.cpp:507-512` and falls through to TILE
  (`:518`) at decode; verification at batch 49-65 also returns TILE. **Same kernel both times.**
- **q8_0 KV** returns VEC at decode via `fattn.cpp:513` (`Q->ne[1] <= 2`) but TILE at verification
  (`:518`). **The route flips.** TILE additionally hard-passes `need_f16_K = need_f16_V = true`
  (`fattn-tile.hpp:1091, 1126, 1139, 1149`), so `launch_fattn` dequantizes the whole q8_0 cache to
  f16 per launch (`fattn-common.hpp:1132-1192`) while VEC dequantizes inside the kernel. The
  verify pass therefore computes on `half(d*q)` values the decode pass never sees - a genuine
  arithmetic difference, not merely a reordering.

This is the only structural difference between the two arms and it uniquely predicts the observed
split. The replay-acceptance data confirms it: `ngram-mod` replay accept rates are identical
between q8_0 and f16 to three decimals on `code_edit` (0.641 x4) and `multi_turn` (0.969 x4), and
diverge only on `free_prose` (f16: 1.000 x4; q8_0: 0.391, 0.615, 0.636, 1.000).

**The oracle harness also has a defect worth fixing regardless.** Those four q8_0 hashes are not
four independent samples. Each run is deterministic given the drafter cache; one flipped argmax
changes the output, which re-seeds the cache, so the next repetition drafts a different
continuation. Hence the monotone staircase rather than random scatter. The five "deterministic
repetitions" are a feedback loop - the cache must be cleared between them.

Ruled out: q8_0 block-straddling at the draft boundary (K/V rows are quantized per token along
head-dim, so rollback is bit-exact). Downgraded to contributing rather than causal: the
`parallel_blocks` search, which is batch-shape dependent but applies identically to f16, which is
exact. `stream_k` is not in play - every caller passes `false`.

Free diagnostic, no code change: `GGML_SYCL_FA_Q8_GQA_TILE=1` forces TILE at `Q->ne[1] == 1` for
exactly this shape, making decode and verify take the same route. One benchmark run confirms or
kills the diagnosis. It is not a shipping fix - it would pay the full-cache dequant every decode
step, which is why it is off.

### Other closed arms worth carrying forward

- `killed_arms[0]`: "quants-first plus TILE composition" was **slower than either mechanism
  alone** at deep cells. The two must not be combined.
- `killed_arms[5]`: q8 prefetch perturbed the exact K/V load sites in the D=128 q8_0 VEC kernel
  with ISA-confirmed engagement, and all four arms came back between -0.95% and -1.01% at
  d16384. Evidence that this kernel is not issue-bound.
- P5.3 measured the same kernel at SIMD16/GRF128 with **zero spill and zero private bytes**;
  P5.4 measured its effective requested-KV bandwidth at **45.15 GB/s** at d16384, roughly 8% of
  the part's peak. A kernel with no spill running at 8% of peak bandwidth is limited by
  dependency latency and occupancy, not by load-message count.

---

## 4. Shipped this round

### 4.1 quants-first promoted to default

`src/llama-kv-cache.cpp` - the layout's enablement is inverted from opt-in to opt-out. The
per-layer predicate at `:507-524` is unchanged and still restricts the layout to a SYCL device
with both K and V present, both `GGML_TYPE_Q8_0`, `n_embd_head_k == n_embd_head_v == 128`, and
`!v_trans`; every other cache silently keeps canonical blocks exactly as before.

`GGML_SYCL_Q8_KV_QUANTS_FIRST=0` is the opt-out. A separate `quants_first_explicit` flag keeps
the "ignored" diagnostic attached to an explicit opt-in, so the default-on path stays silent
for the many caches that do not qualify (f16 KV, turbo KV, non-SYCL devices, 64-element heads).

`docs/backend/SYCL.md` gains a row for the variable, which was previously undocumented.

**Session-file compatibility was the main risk of flipping a layout default, and it is already
handled.** State write repacks to canonical before serialising
(`llama_kv_cache_q8_repack_groups(..., false)` at `src/llama-kv-cache.cpp:55`) and state read
repacks back on load (`..., true` at `:72`), so bytes on disk are always canonical
`block_q8_0` regardless of the in-memory layout. Session files written before this change
restore correctly after it, and vice versa. Round-trip coverage exists at
`tests/test-kv-cache-adaptive-mode.cpp:39-60`.

**This is a cutover of an already-measured mechanism, not a new kernel.** What was never
measured is the shallow end: P5.11 reported only depths 4096, 8192 and 16384. The disjunctive
gate requires the -2% no-regression check at depth 0 and depth 2048, and those cells do not
exist. Producing them is the gate.

### 4.2 Load-width probe

`scripts/perf/probe-q8-load-width.sh` - compile-only, no GPU execution. **Before this round**,
both q8_0 KV paths fetched their 4-byte quant words through `ggml_sycl_memcpy_1<N, 2>`, whose
second template argument is the literal per-copy width, not a hint:
`ggml/src/ggml-sycl/common.hpp` dispatches on `nb_per_cpy`, and 2 emits two 16-bit loads per
dword. The canonical path still does and must; the quants-first path is now `<N, 4>`.

- Canonical AoS (`fattn-common.hpp:298`, `:649`) **must keep the 2**. `qs` sits at offset 2 of a
  34-byte block, so the address is `base + ib*34 + 2 + 4*iqs`, which is congruent to 2 mod 4 for
  even `ib` and 0 for odd. Raising it is a misaligned dword load.
- Quants-first (`fattn-common.hpp:329`, `:683`) is genuinely 4-byte aligned: the group stride is
  136 B (`src/llama-kv-cache.cpp:20-21`, writer `set_rows.cpp:261`) and the payload offsets are
  `ib*32 + 4*iqs` and `ib*32 + iqs` with `iqs` a multiple of 4. These inherited the `, 2` from
  the canonical code they were copied from.

The fork already encodes the correct rule elsewhere, which is a useful cross-check: `q4_1` and
`q5_1` have a 4-byte `half2 dm` header and pass **no** alignment argument, while `q4_0`, `q5_0`
and `q8_0` pass `, 2`.

The probe compiles the D=128 q8_0 VEC instance and a minimal quants-first instance to
`spir64_gen -device acm-g10` with `IGC_ShaderDumpEnable`, then counts LSC message widths in the
SIMD16 entry dumps. It is run once before and once after the alignment change and the paired
counts are diffed; absolute counts are meaningless because unroll factors dominate.

**Go/no-go, stated before running:** if both builds emit the same load-message counts, IGC
already coalesced the adjacent 16-bit loads, the hypothesis is dead at zero cost, and no A770
time is spent on it.

**The probe ran. The kill criterion fired.**

Route (icpx 2026.0 removed `-fsycl-link=image`, so AOT goes the long way): device-only compile
to LLVM IR, `llvm-spirv --spirv-ext=+SPV_INTEL_subgroups` (the default set omits it and the
sub-group shuffles in the reductions then yield an empty module), then `ocloc compile -device
acm-g10 -spirv_input` under `IGC_ShaderDumpEnable`.

Captured evidence is committed under `scripts/perf/results/round2-load-width-probe/`; each
summary records its own `repo HEAD`, dirty-file count and toolchain versions. The full IGC ISA
dumps are about 11 MB per arm and are **not** committed - re-generate them with the script if
they are needed.

**Arm 1, the load-width question, measured at `028084d00`** (`baseline-summary.txt`,
`widened-summary.txt`):

| quants-first D=128 kernels | `load.ugm.d16*` | `load.ugm.d32*` | ISA bytes |
|---|---:|---:|---:|
| baseline (`, 2`) | 512 | 622 | 11,249,087 |
| widened (`, 4`) | **512** | **622** | 11,028,126 |

The LLVM IR genuinely differs (distinct module hashes, `OCL_asm385b9afb...` versus
`OCL_asm542956a3...`) and the generated code shrank about 2%, so the change reached the compiled
path. **The load-message counts are identical.** IGC was already merging the adjacent 16-bit
copies, and the 512 remaining `d16u32` loads are genuine `sycl::half` fetches - the per-group
scales, the mask, and Q - not split dwords. The canonical q8_0 instance (all D values) showed
2129 `d16*` of 4798 total ugm loads at the same commit; that is the AoS path and must stay at
alignment 2 regardless.

**Arm 2, the destination-alignment fix, measured at `9f57709cf`** (`noalignas-summary.txt`,
`alignas-summary.txt`): identical counts with and without `alignas(4)` - canonical 4258 `d16*` of
9594, quants-first 1024 `d16*` of 2270. IGC was already over-aligning the staging arrays, so the
fix removes a reliance on that without changing generated code.

**The two arms are not comparable to each other, only within themselves.** `9f57709cf` sits after
this branch was rebased onto master, which pulled in `61eed0aac` (upstream sync, renumbers
fork-private ggml type ids); that changes how many kernels the instance TU emits, which is why the
absolute counts roughly double between the arms. Each arm is an internally same-tree A/B, which is
the only comparison this probe supports. Any future re-run must re-establish its own baseline
rather than compare against the numbers above.

**Verdict: no performance change is expected or claimed from this edit.** It is retained purely
as hygiene - the alignment argument now states the invariant the layout actually guarantees, and
the added `static_assert(ne % 4 == 0)` makes a future change to `V_rows_per_thread` fail to
compile rather than silently emit a misaligned dword load. This corroborates two prior results
rather than contradicting them: P5.4 measured this kernel at 45.15 GB/s (about 8% of peak) with
zero spill, and the four q8 prefetch arms perturbed these exact sites with ISA-confirmed
engagement for -0.95% to -1.01% at d16384. The kernel is bound by dependency latency and
occupancy, not by load-message count.

---

## 5. Verification

Nothing in this round executed on the A770. The gates below are the operator's.

**Build.** Pinned recipe from `docs/research/sycl-build-runtime-pins.md`: `icx`/`icpx` 2026.0,
`-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON
-DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON`, `GGML_SYCL_DEVICE_ARCH` empty (JIT), Release, Ninja.

**Correctness.**
- `ctest -R test-sycl-turbo-correctness`, gate is `0 GATE-FAIL`
  (`tests/test-sycl-turbo-correctness.cpp:1558`). This harness is explicitly not bitwise
  (`:140-142`); it asserts NMSE and cosine.
- Forced-TILE 4:1 and 8:1 quants-first correctness cells, the binding route proof, since forced
  q8_0 GQA TILE reaches quants-first rows through `ggml_get_to_fp16_nc_sycl`.
- `python3 scripts/perf/server_state_roundtrip.py --server-bin <BUILD>/bin/llama-server
  --model <GGUF> --out-dir <DIR>` - all 8 assertions, including the 12-token / 836,576-byte
  save-restore identity and the exact cross-slot continuation.
- With `GGML_SYCL_Q8_KV_QUANTS_FIRST=0`, output must match the pre-change build. This is the
  inertness check and it is what makes the cutover reversible in the field.

**Performance.** `scripts/bench-a770-fork-unique.py` paired A/B, opt-out versus default:
`llama-bench -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 128 -b 512 -ub 512 --no-warmup -r 1
-o json -d <DEPTH>` at depths **0, 2048, 4096, 8192, 16384**; the first two are the new cells
that decide the cutover. Six launches per arm, repetition 0 discarded. Preconditions:
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, sole tenancy of `/dev/dri/renderD128` via `fuser -v`,
clean dmesg delta.

**Pre-registered kill criterion for the cutover, stated before any run.** Applying the gate of
section 1 unchanged: rejected if any protected depth - **0, 2048 or 4096** - regresses by more
than -2% paired median on any fleet model, or if the promotion depths 8192 and 16384 fail to
reproduce at >=5% on the newly fetched stock Q4_K_M files. Depth 4096 already has a measured
+8.25/+8.01% from P5.11 and is expected to clear its guard comfortably; depths 0 and 2048 are the
cells that do not yet exist. A regression confined to depth 0 with the deep gains intact is not an
automatic kill; it is the depth-conditioned-route question, reported and escalated rather than
quietly promoted.

**Determinism.** The layout change does not alter reduction order or split-K partitioning, so
output is expected to be bit-identical between the two arms for a given model. Any divergence
found by the server-state gate is a defect, not an expected numerical difference - which makes
that gate a sharper instrument here than it was for the GQA-packing candidates.

---

## 6. Ranked candidates

Ranked on expected value against the disjunctive gate of section 1, not on novelty.

### A0. Reproduce upstream's A770 number on this box. RUN THIS BEFORE ANYTHING ELSE.

- **Open problem.** Section 1.2 suggests the fork is about 2.1x slower than upstream SYCL on the
  same hardware, but that rests on a cross-source comparison, not a paired run. If it is real it
  dwarfs every candidate below. If it is an artefact of the driver stack, several candidates below
  are chasing the wrong thing.
- **Probe.** Build upstream master with the same pinned toolchain and run `llama-bench` on
  llama-2-7b Q4_0 with FA off, comparing against the published 55.53 / 59.03 t/s; then run the
  fork's binary on the same model file. Same box, same driver, same weights - source tree is the
  only variable.
- **Branch criterion, pre-registered.** If upstream also lands near 25 t/s on this driver stack,
  the regression belongs to compute-runtime 26.22 and the response is a driver bisect, not kernel
  work. If upstream reproduces 55-59 t/s, the fork carries a roughly 2x self-inflicted regression
  and locating it outranks every optimisation in this document.
- **Effort.** Medium, one extra build. **Cost of skipping it: potentially the whole round.**

### A0b. Quant-kernel isolation (cheap, run alongside A0)

Same binary, same shapes, quant the only variable: `llama-bench` on
`/mnt/mrgr/models/llama31-8b-q4_0/` versus the freshly fetched
`/mnt/mrgr/models/llama31-8b-q4km/`. Attributes part of the unaccounted 18.2 ms between Q4_K
superblock dequant cost and batch-1 occupancy. **Kill:** Q4_0 within 15% of Q4_K_M means the
residual is not quant-kernel-bound and the line dies. Effort: low.

### A4. Un-gate SYCL graphs for MoE (a bug, not a constraint)

`check_graph_compatibility` returns false on **any** `MUL_MAT_ID` node
(`ggml-sycl.cpp:5203-5207`), justified by the blocking `stream->wait()` at `:4323`. But the fused
MoE token-generation path (`ggml_sycl_mul_mat_id_mmvq_fused`, `:4210`) **returns before ever
reaching that wait**, and it is the path Qwen MoE actually takes - `should_reorder_tensor`
(`:4016`) requires `dst->op == GGML_OP_MUL_MAT`, so `MUL_MAT_ID` tensors never reorder, and P5.7
runtime tracing confirmed Qwen stays on fused MMVQ-MoE. **Consequence: Qwen3-Coder-30B-A3B can
never use SYCL graphs, on the strength of a host wait its own fast path skips.** With 48 layers it
has more enqueues per token than the dense models, so it stands to gain more than they do.
**Probe:** `GGML_SYCL_DISABLE_GRAPH=0 GGML_SYCL_GRAPH_PROFILE=1` on the Qwen MoE and confirm
`graph_calls = 0` today. **Kill:** if `direct_calls` is already 0, the gate is not firing. Effort:
low - condition the rejection on fused-path eligibility.

### C1. Promote quants-first q8_0 KV to default (SHIPPED THIS ROUND, needs the shallow cells)

- **Open problem.** The fork's largest measured decode win sits behind an env var nobody sets.
- **Mechanism.** Replacing four interleaved 34-byte `block_q8_0` records with a 136-byte row of
  128 contiguous quant bytes followed by four fp16 scales makes the payload dword-aligned and
  contiguous, so a KV row is fetched as coalesced dword runs instead of a strided gather that
  straddles a cache line on every other block, and the scale is no longer interleaved into the
  reduction stream.
- **Implementation parameters.** 136-byte group stride, D=128 only, four `QK8_0` groups per head,
  scales at byte offset `4 * QK8_0`. Gate: SYCL device, K and V both `GGML_TYPE_Q8_0`,
  `n_embd_head_k == n_embd_head_v == 128`, `!v_trans`. Opt-out `GGML_SYCL_Q8_KV_QUANTS_FIRST=0`.
- **Evidence.** Xe-HPG A770, paired tg128 medians: +8.25/+8.01% at 4096, +13.81/+13.37% at 8192,
  +20.62/+20.43% at 16384; lower-95% bounds +7.90/+7.71%; worst pp512 -0.39%; cosine >= 0.999957.
  `campaign:296-323`. Corroborated by peer designs: vLLM XPU, IPEX xetla, OpenVINO GPU plugin and
  oneDNN `micro_sdpa` all store quantized KV with grouped scales rather than per-block interleave.
- **Local gate probe.** The depth 0 and depth 2048 paired cells, which P5.11 never measured. See
  section 5 for the exact `llama-bench` invocation.
- **Pre-registered kill criterion.** The section 1 gate, unchanged: rejected if any protected
  depth - 0, 2048 or 4096 - regresses more than -2% on any fleet model, or if the promotion depths
  8192 and 16384 fail to reproduce at >=5% on the newly fetched stock Q4_K_M files.
- **Determinism.** Layout only. No change to reduction order or split-K partitioning, so output
  should be bit-identical between arms; the server-state gate is a sharp instrument here, and any
  divergence is a defect rather than expected numerical drift.
- **Non-duplication.** Not the killed DMMV/reorder work (that is weights). Not the killed SLM LUT
  (that is an in-loop lookup). This is a global-memory KV layout already merged as opt-in.
- **Effort.** Done: one predicate inverted in `src/llama-kv-cache.cpp`, plus a separate
  `quants_first_explicit` flag so the default-on path does not warn on every non-qualifying cache.
- **Compounding.** Must **not** be combined with `GGML_SYCL_FA_Q8_GQA_TILE`: `killed_arms[0]`
  records the composition as slower than either mechanism alone. Mutually exclusive with XMX by
  construction (`fattn.cpp:497-498`), which is moot given C3.

### C2. Depth-conditioned GQA route (the strongest genuinely new candidate)

- **Open problem.** GQA head packing is worth +10.33% at 8k and +20.40% at 16k on this hardware
  and has been rejected twice, both times solely because of a depth-0 regression.
- **Mechanism.** With GQA ratio `r`, a per-query-head loop dequantises every K element `r` times
  and re-reads the same KV rows `r` times. Packing amortises both by `r`. The saving is real only
  when those re-reads miss cache: at `n_embd_k_gqa = 1024` a q8_0 KV row is 1088 B for K and the
  same for V, so a layer's working set is 2176 B per token - 8.9 MB at 4k, 35.7 MB at 16k. The
  A770's 16 MB L2 holds the whole 4k layer, so below roughly 7.7k tokens the re-reads are L2 hits
  and packing buys nothing while still paying the packed kernel's fixed cost. **A route gated on
  KV depth captures the deep win and never pays the shallow cost.**
- **Implementation parameters.** Reuse the existing TILE `ncols2` machinery
  (`fattn-tile.hpp:1155-1214`) rather than porting packing into VEC. Gate the existing
  `GGML_SYCL_FA_Q8_GQA_TILE` block (`fattn.cpp:467-477`) on `K->ne[1] >= 8192` and flip its
  default on. Threshold justified by the fork's own crossover, which sits between 4096 (+3.50%)
  and 8192 (+10.33%).
- **Evidence.** Xe-HPG A770, this fork, P5.5 `campaign:164-184` (full depth table in section 3).
  Precedent for the mechanism, all code-read: **OpenVINO's Intel-GPU plugin gates its GQA-packed
  path on runtime sequence length** - `can_use_gqa_kernel` in
  `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp:112-121`, with
  `min_gqa_sequence_len = 16*256 = 4096` and the comment "Apply GQA optimization starting from a
  certain sequence length (4K tokens) value". Intel shipped exactly this gate on this GPU family.
  llama.cpp CUDA does the same shape of thing at `ggml/src/ggml-cuda/fattn.cu:464`
  (`!(gqa_ratio > 4 && K->ne[1] >= 8192)`), as does FlashAttention's `should_pack_gqa()`. Note the
  CUDA logic **routes quantized KV at batch 1 to the vector kernel on every generation from
  Turing to Blackwell**, which independently corroborates that packed-for-quantized-decode loses
  at shallow depth.
- **Run the occupancy probe first - it decides whether this candidate is a fix or a workaround.**
  Instrument `launch_fattn` to print `ntiles_total`, `parallel_blocks` and the efficiency search's
  best score (`fattn-common.hpp:1259-1287`) for both routes at depths 0, 2048, 4096, 8192, 16384.
  Host-side, no kernel work, one instrumented build. If TILE's depth-0 efficiency is far below
  VEC's, the occupancy hypothesis in section 3 is confirmed and **C2b below is the real fix** -
  a depth gate would merely be hiding a wave-fill deficit that could be removed at all depths.
  If the efficiencies match, the occupancy model dies and this gate stands on its own.
- **C2b, the competing higher-ceiling variant.** If occupancy is the cause, seed
  `parallel_blocks = min(ntiles_KQ, blocks_per_wave/ntiles_total)` when `ncols2 > 1` in the
  `!stream_k` branch and let the existing efficiency loop refine it. This is what Colfax's split-KV
  does for packed Q tiles. **Kill:** depth-0 paired regression still worse than -2%. **Determinism:
  much larger blast radius than C2** - it changes the split-K reduction tree for *all* FA shapes,
  so it needs the full `test-backend-ops` FA sweep, not just q8_0. Effort: medium.
- **Local gate probe.** Three-arm paired bench at depths 0, 2048, 4096, 8192, 16384: canonical
  baseline, quants-first default (C1), and quants-first plus depth-gated GQA. The third arm is
  mandatory because `killed_arms[0]` recorded the ungated composition as slower than either alone.
- **The shallow guard is satisfied by construction.** Below the threshold the route is
  byte-identical to today's VEC path, so depths 0, 2048 and 4096 cannot move except through a gate
  leak. That converts the criterion that killed P5.5 from a risk into a test.
- **Pre-registered kill criterion.** Rejected if the depth-gated arm fails to beat the C1 arm by
  >=5% at 8192 and 16384, **or if any sub-threshold depth moves by more than plus or minus 0.5%**
  (that movement would be a gate leak, i.e. an implementation bug, and the CUDA precedent is
  cautionary here - PR #21271 "CUDA: fix FA kernel selection logic" records a prior edit to this
  same selection function silently disabling FA for all unpadded KV caches).
- **Threshold.** Between 4096 (+3.50%) and 8192 (+10.33%). Default to 8192, the first *measured*
  depth clearing 5%; anything between is interpolation and should be env-overridable rather than
  asserted. OpenVINO's analogous constant is 4096.
- **Determinism.** Route changes with depth, so a generation can cross a kernel boundary
  mid-sequence, and TILE and VEC differ in reduction order and `parallel_blocks`. **But this
  introduces no new *class* of nondeterminism**: `parallel_blocks` is already chosen by a runtime
  occupancy search bounded by `ntiles_KQ = ceil(K->ne[1]/nbatch_fa)`
  (`fattn-common.hpp:1259-1285`), so the split-K reduction tree - and hence floating-point rounding
  - already changes with depth at every depth, unconditionally, today. **No production engine
  surveyed implements hysteresis** (llama.cpp, vLLM, FlashInfer, TensorRT-LLM all use hard
  stateless thresholds), so the precedent is to accept the discontinuity. Add `test-backend-ops` FA
  cases at KV = threshold-256 and KV = threshold to pin both sides.
- **Non-duplication.** Not P5.5, which proposed an unconditional cutover and was killed for the
  depth-0 cell it never guarded. Not the reverted `2d53f3b25` packed-q8 VEC kernel, which wrote a
  new kernel instead of gating the existing one.
- **Effort.** Small: a depth condition and a default flip in one router block. Explicitly **not**
  the `ncols2`-into-VEC port, which is a 5-file/~175-LOC change whose closest prior attempt scored
  d16384 +7.59% / d4096 -1.84% and was reverted.
- **Compounding.** Stacks with C1 only if the composition penalty in `killed_arms[0]` turns out to
  be a shallow-depth effect, which the depth gate would remove. That is the hypothesis the
  three-arm bench tests.

### C3. Load-width hygiene on the quants-first path (CLOSED by probe, retained as cleanup)

Kill criterion fired before any A770 time was spent. See section 4.2 for the measured result: IR
changed, code shrank 2%, load-message counts identical. **No performance claim.** Retained
because the alignment argument now states the invariant the layout guarantees and the added
`static_assert` prevents a future silent misalignment.

### C0. Dual-queue shape-routed submission mode (highest expected value; promote above C1)

- **Open problem.** The 40.4 ms non-attention floor, and the fact that the two largest measured
  wins in the fork's history were rejected for lacking a globally safe setting rather than for
  lacking magnitude.
- **Mechanism.** Submission mode is a **per-queue** property, not a global one. The Level Zero
  adapter picks it per queue: `if (isBatchedSubmission()) UsingImmCmdLists = false; else if
  (isImmediateSubmission()) UsingImmCmdLists = true; else Device->useImmediateCommandLists();`
  (`unified-runtime/source/adapters/level_zero/queue.cpp:1160-1165`), driven by
  `UR_QUEUE_FLAG_SUBMISSION_BATCHED` / `_IMMEDIATE` (`:1717-1722`). Batched submission coalesces
  N per-node submits into one command-buffer submit, which is exactly the cost that dominates a
  launch-bound decode step. Token generation is deep and serial and wants batching; prefill and
  batch-1 latency want immediate dispatch. One process can have both queues.
- **Implementation parameters.** Two in-order queues created at backend init: `Q_imm` with
  `sycl::ext::intel::property::queue::immediate_command_list` and `Q_batch` with
  `no_immediate_command_list` - **both properties are present in the installed toolchain**
  (`/opt/intel/oneapi/compiler/latest/include/sycl/properties/queue_properties.def:21,23`).
  `UR_L0_BATCH_SIZE=64` set once at process start; pinning a non-zero value also disables the
  adapter's dynamic batch adaptation (`queue.cpp:1269-1318`, which only adapts when `Size == 0`).
  Route on `n_tokens` at graph-compute entry. The fork currently creates plain in-order queues
  with no submission property at all (`ggml/src/ggml-sycl/dpct/helper.hpp:745-755`).
- **Evidence.** Xe-HPG A770, this fork, paired tg128: `UR_L0_BATCH_SIZE=64` **+47.26%**, batch 16
  +38.45%, batch 1 **-12.31%**, graphs enabled **+44.55%** (lower-95% +44.13%, pp512 -3.05%). The
  spread between +47.26% and -12.31% across settings is precisely the signal that a routed policy
  exploits and a global one cannot.
- **Two caveats that must be resolved first, and they are load-bearing.**
  1. **The +47.26% arm may be confounded.** All batching sits behind `if (!UsingImmCmdLists)`
     (`queue.cpp:1360-1390`), and on DG2/Linux with driver >= 1.5.30820 the adapter default is
     per-queue immediate command lists (`device.cpp:1988-2024`). Under the fork's current
     queue construction `UR_L0_BATCH_SIZE` **should have been inert**. Either immediate command
     lists were not active in the measured run, or that arm has a confound. Unresolved from source
     alone.
  2. **Graphs and batching are the same mechanism, so do not budget them as additive.** The fork
     builds a *fresh* `command_graph` and re-records every pass (`ggml-sycl.cpp:5294-5299`), so
     the +44.55% is not record-avoidance - it is one command-buffer submit replacing N submits,
     which is what batched submission also does.
- **Local gate probe.** Cross `UR_L0_USE_IMMEDIATE_COMMANDLISTS` in {0,1} against
  `UR_L0_BATCH_SIZE` in {unset, 64} on tg128 at depths 0 and 16384. This resolves caveat 1 before
  any code is written. If batch size only moves the needle when immediate lists are disabled, the
  dual-queue design is confirmed as the right shape.
- **Pre-registered kill criterion.** Rejected if the tg128 gain on the batched queue has a paired
  lower-95% bound below +10%, or if batch-1 latency on the immediate queue regresses more than 2%.
- **Determinism.** None. In-order queues, submission order preserved, no kernel or reduction
  change. This is the candidate's biggest advantage over every FA-side option.
- **Non-duplication.** Not the killed global batching policy (`killed_arms[2]`), which failed
  precisely because one setting had to serve every shape. Not SYCL-Graph: no recording, no
  `update()`, no UB exposure.
- **Effort.** Low. Two queue constructions and a routing predicate. **No launch-path refactor** -
  this is the one high-magnitude candidate that does not need the 276-site `set_arg` work.
- **Compounding.** Substitutive with graphs, not additive. Independent of C1 and C2.

### C5. Topology fingerprint guard for the graph `update()` path (cheap, strictly risk-reducing)

The SYCL-Graph specification requires that an updated graph have "the same number of nodes and
edges", nodes "added in the same order", identical `node_type` and "kernels with identical types",
and edges "created in the same order by using the same API invocation"; violating any of it "results
in undefined behavior". The fork calls `exec_graph->update()` per pass guarded only by a `catch`
(`ggml-sycl.cpp:5325-5344`), and **a `catch` cannot catch UB**. No topology-fingerprinting guard
exists in any public codebase, so this would be the first.

Worth doing regardless of C0, and there is a free diagnostic first: on this machine `update()`
almost certainly *never* succeeds. UR sets `MutableCommandListSpecExtensionSupported` from the
extension name and version alone (`platform.cpp:259-262`), never consulting `mutableCommandFlags`,
while the real gate asserts on `ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS`
(`helpers/mutable_helpers.cpp:417-419`) - which the fork's own probe reports as `false`. So every
`update()` throws into the re-finalize fallback. **The fork already instruments this**: compare
`graph_profile->update_fallbacks` against the update call count. If they are equal, the update path
is pure waste per pass and should be deleted rather than guarded. Effort: about 40 lines either way.

### C6. Per-kernel `grf_size<256>` on selected decode kernels

Distinct from the killed *global* large-GRF mode, and **supported**: the extension spec's own
table reads `DG2 | 128 (small register file), 256 (large register file)`
(`sycl_ext_intel_grf_size.asciidoc:170-173`), IGC agrees
(`supportLargeGRF() = isCoreChildOf(IGFX_XE_HPG_CORE) && !isCoreChildOf(IGFX_XE3_CORE)`,
`Platform.hpp:1309`), and the header `sycl/ext/intel/experimental/grf_size_properties.hpp` is
present. Alchemist offers a **binary 128/256 choice only** - `supportsVRT()` is Xe3-only (`:836`),
`getMinNumGRF()` is 128 (`:932`) - and `supportsAutoGRFSelection()` is **off for DG2**
(`:491-493`), so the size must be set explicitly per kernel; the compiler will not choose.

**Price it honestly before using it.** GRFs are 32 B on Xe-HPG (`Platform.hpp:918`), and the
whitepaper's 32 KB register file with 8 threads per XVE reconciles exactly: 128 x 32 B x 8 = 32 KB.
So `grf_size<256>` **halves threads per XVE from 8 to 4** - it spends precisely the resource a
bandwidth-bound decode kernel needs most, latency-hiding threads, to buy register blocking. PVC
has 4x the register file and eats that trade far more cheaply, which is the context vLLM's
`grf_size<256>` XPU kernel sits in. It pays only for a kernel that is register-blocked end to end;
the fork's XMX kernel stages everything through SLM (`fattn-xmx.cpp:88-99`), so as written it would
pay the occupancy cost and collect none of the benefit. One kernel at a time, benched, never
globally.

### C7. Fix the auto-asymmetric turbo-K predicate (correctness, not throughput)

- **Open problem.** `src/llama-kv-cache.cpp:240` upgrades K to `q8_0` when `gqa_ratio >= 6`, with
  the in-code rationale that "Turbo K quantization error gets amplified by the GQA broadcast
  factor" (`:213-218`). **The GQA ratio may not be the discriminator.**
- **Evidence.** KVLinC (arXiv:2510.05373) Table 1 measures QuaRot - rotated keys quantized
  token-wise, which is structurally what `turbo_wht<128>` over a head-dim-contiguous row does
  (`ggml/src/ggml-sycl/set_rows.cpp:53-87`, allocation `src/llama-kv-cache.cpp:511`) - and the
  blow-ups do not track GQA ratio: Qwen3-1.7B at **2:1** gives PPL 1963.3 while Llama-3.1-8B at
  **4:1** gives 7.3, and Llama-2-7B at 1:1 gives 5.8. Model family separates the rows. The paper
  attributes it (section 4.1) to Qwen2.5's bias in the Q/K projections and Qwen3's post-projection
  layernorm, both of which plant a fixed per-channel structure in K that a channel-wise zero-point
  absorbs for free, token-wise scales cannot represent, and a Hadamard rotation smears across all
  128 channels. If that holds, the current threshold separates Qwen2.5 (6-8:1) from Mistral (4:1)
  by coincidence and will **fail to fire for Qwen3-1.7B/4B/8B**, shipping a catastrophic K cache.
- **Scope limit, stated honestly.** Table 1 is at **2-bit** (average precision 2.46-2.96 bits with
  a 128-token FP16 residual window). The paper contains **no 3-, 4-, or 8-bit ablation**; it
  positions itself explicitly against the moderate regime where rotation is known to work. So this
  bears on `TURBO2_0` directly, on `TURBO3_0`/`TURBO4_0` by extrapolation, and on `q8_0` not at
  all. The fork's own 2887 blow-up was at turbo3, one step outside the evidence - the mechanism
  transfers, the magnitude does not.
- **Local gate probe.** Load Qwen3-1.7B (or any Qwen3 at GQA 2:1 or 4:1) with `-ctk turbo3 -ctv
  turbo3` and check the debug line at `:237-238` to confirm the gate does **not** fire, then
  measure PPL. Cheap, and it tests the hypothesis and the alleged bug in one run.
- **Pre-registered kill criterion.** If Qwen3-1.7B turbo3-K PPL is in fact acceptable, the family
  hypothesis collapses and this whole item is withdrawn. **Run this before acting on it.**
- **Throughput impact: none.** This is a correctness fix and does not compete for the gate.
- **Effort.** Low to diagnose; the replacement predicate (family or architecture test, or a
  load-time per-channel K kurtosis probe) is a design question deferred until the probe confirms.

### C8. Drop the WHT on K, keep it on V (the one throughput angle in angle D)

Rotation inflates the K quantization scale factor, and removing K rotation also makes the forward
WHT on Q dead work - Q is rotated *only* to match rotated K (`src/llama-graph.cpp:2411, 2540,
2737`), so this deletes an op from the per-decode-step critical path. The inverse WHT on the
attention output must stay for V (`:2114, 2213, 2234`). `LLAMA_ATTN_ROT_K_OVERRIDE` already exists
to A/B it. Expect accuracy to move the *wrong* way unless paired with per-channel K scales, since
Figure 4 of the paper ranks `Q_T(K)` worse than `Q_T(KH)`; the value here is removing Q-side work
and isolating the variable. **Kill: under 2% tok/s gain on A770 and it is dropped**, because the
accuracy story alone does not clear a throughput gate. Note this only matters for turbo KV, which
is not the default configuration.

### Demoted from angle D: per-channel K scales as a throughput candidate

The KVLinC result is **accuracy-only**. Its 2.55x figure is A40 (Ampere GA102, mechanism-only
under the tagging rule) and comes from batch-size headroom against FP16 FlashAttention-2 at a
per-sequence KV depth of at most 1280 - confounded with memory savings, and not a granularity
effect. Streaming append is solved and the solution is known - OpenVINO does per-block-per-channel
with running absmax and restatement scoped to a paged block
(`pa_kv_cache_update_ref.cl:119`, `quantize_and_save_by_channel_block_with_requantize`), so
statistics never span blocks and cost is bounded at O(head_dim x block_size) per token - but
adopting it would **degrade determinism**, since restatement makes the stored K bits depend on
chunk boundaries. Not a candidate under this project's gate.

### C9. Hoist quants-first scales from per-group to a per-row or per-page plane

The layout change already landed moved scale *ordering*; it did not fix *stride*. The row is
136 B against a 64 B cache line, so group `g` starts at `136g mod 64` with period 8 - the scale
quad at `+128` shares a line with the payload tail for some groups and not others. Moving scales
out to a trailing plane per row or per KV page makes the payload stride **128 B, line-aligned for
every group, at zero footprint cost**, and turns a tile's scales into one contiguous fetch instead
of an 8-byte quad per 128 elements. Bit-identical: same scale values, moved.

Corroboration: this is exactly OpenVINO's shape - `comp_ptr[token_pos_in_block] = 1.0/scale;` then
`comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + token_pos_in_block] = zp;`, both past the whole page's
quantized data (`pa_kv_cache_update_ref.cl:82-107`), i.e. two full planes indexed by token. Where
Intel *does* interleave, the stride is a 4-byte multiple (`// [packed_tokens (8 bytes)] [scale
(f16)] [zp (f16)] = 12 bytes`, `:204`) - the negative image of ggml's 34-byte block.

Files: host repack `src/llama-kv-cache.cpp:23-40`, writer `set_rows.cpp:222-270`, readers
`fattn-common.hpp:309-336` and `:673-703`. **Kill:** lower-95% tg128 delta at 8192 below +0.5%.
**Determinism:** none - identical values, identical accumulation order. Effort: about a day.

*Benchmark methodology warning inherited from upstream issue #25203:* reorder-style paths can be
gated on `src1->ne[1] <= 8`, so a plain `-p 512` arm never exercises them. Include a `-p 4` arm.

*Explicitly not corroborated:* padding the row to 144 or 192 B for line alignment. vLLM's XPU
TurboQuant backend pads only to an even byte count, and states the reason as shape plumbing
(`effective_head_size = slot_size_aligned // 2` must be integral), not alignment. The brief's
"peers pad to LSC transaction size" hypothesis is corroborated nowhere, and 192 B costs +41% KV
footprint. C9 obtains the alignment benefit for free, so try it first and treat padding as a
fallback only if C9 shows nothing.

### C10. Kill the redundant `quantize_row_q8_1` launches

Seven per layer (`ggml-sycl.cpp:3020`, one per MMVQ `MUL_MAT`), of which q/k/v re-quantize the
same 4096-float vector and gate/up do likewise - about 225 launches and 96 redundant enqueues per
token. Cache by `(src1 pointer, ne)`. Worth roughly 1.4 ms ungraphed but only about 0.15 ms once
graphs or batched submission land (ESTIMATE). **Do this only if C0 is rejected** - C0 subsumes
most of its value. Kill: under +2% paired. Effort: low.

### C11. oneDNN Graph SDPA for prefill (not a decode candidate, listed for completeness)

Prefill has M far above the 61-65 row crossover, so it is the one regime where XMX pays on this
part. Largely a cherry-pick of upstream PR #25222, mirroring its gate: f16 KV, single sequence,
prefill at least 32 tokens, no sinks or softcap. oneDNN is already linked
(`ggml/src/ggml-sycl/common.hpp:40-41`). **Kill:** oneDNN reports `unimplemented` for the SDPA
pattern on `acm-g10`, or pp4096 is under 1.15x the `fa=1` baseline. Note the compressed-KV variant
of that primitive is Xe2-validated only. **This does not move `tg128` and must not be counted
against the decode gate.**

### KILLED: persistent / megakernel decode on DG2

`sycl_ext_oneapi_root_group` requires the `use_root_sync` property and maps to
`zeCommandListAppendLaunchCooperativeKernel`, but the specification explicitly permits a trivial
fallback that restricts the kernel to **one work-group** and provides **no forward-progress
guarantee**. All 2025-2026 megakernel work (Hazy Research, Mirage MPK arXiv:2512.22219,
FlashInfer, vLLM-XPU) is NVIDIA, AMD or Xe2, so it is mechanism-only here. Cheap disproof before
anyone invests: query `max_num_work_groups_sync` for one decode kernel; if it returns 1, the
family is closed on this hardware.

## 7. Refuted and demoted

| Item | Status | Basis |
|---|---|---|
| Angle E.3, `xe` versus `i915` driver divergence | **Void** | Host runs i915; `xe` is not loaded. Not a research question |
| Angle E.2, L0 mutable command lists | **Dead** | A770/i915 reports `kernel_arguments: false`, `graph_arguments: false`; `implementation_authorized: false` |
| Angle C, XMX as a decode win | **SHELVED, not merely demoted** | Two independent closures. (a) DPAS is 4x the vector path on Xe-HPG while decode attention leaves the vector ALU about 15x idle, so a 4x ceiling cannot cash a 15x surplus at any tile shape or precision; the saturation crossover is 61-65 query rows against the 4-8 that GQA supplies. (b) Sub-group-16 DPAS is a Xe-HPC-and-later hardware capability (`Platform.hpp:587`), so the reported ICE can never be fixed by a toolchain bump. The measured 43.1%-of-baseline gap is a tuning artefact sitting on a strategy with no upside |
| "Wait for an IGC release that fixes the `joint_matrix` SG=16 ICE" | **Void** | Hardware gate, not a compiler bug. Nothing to wait for, and no reason to move off the 2.36.3 pin for it |
| Vulkan or IPEX-LLM as a faster backend on A770 | **Refuted** | Upstream Vulkan measures 52.56 t/s against SYCL's 55.53-59.03 on llama-2-7b Q4_0, same part. No backend tax to recover. IPEX-LLM has no admissible tagged number |
| The L2-capacity model of the GQA depth crossover | **Refuted** (it was mine) | Curve is monotonically decreasing with no kink; measured crossover is about 2400 not 7710; L2 cannot act as a KV scratchpad against 4.3 GB of weight streaming per token; and the kernel is at 8% of peak bandwidth so a DRAM-capacity model answers the wrong question. Replaced by the occupancy hypothesis in section 3 |
| Padding the quants-first row to 144 or 192 B | **Demoted below C9** | Corroborated by no peer stack; vLLM XPU pads only to an even byte for shape reasons. 192 B costs +41% KV footprint, which partly cancels the win at depth. C9 gets the alignment for free |
| Per-channel K scales as a throughput candidate | **Demoted to accuracy-only** | The supporting result is 2-bit only, its 2.55x is A40 with per-sequence KV depth at most 1280 and confounded with memory savings, and adopting the known streaming-append solution would degrade determinism |
| Angle F, speculation as a tg candidate | **Demoted to correctness work** | `llama-bench` runs no drafter; speculation reads as 0.000% on this gate. The live issue is that q8_0 KV is not target-exact under speculative verification while f16 KV is |
| "Port `ncols2` into the VEC kernel" | **Demoted below C2** | Closest prior attempt (`2d53f3b25`) scored d16384 +7.59% / d4096 -1.84% and was reverted. C2 obtains the same mechanism by gating a kernel that already exists |
| Round-1 retained wins (L0 v2 backend, top-k MoE fusion, `RMS_NORM`+`MUL` fusion, batched command lists) | **Not present in the fork** | See section 2. Three of the four are unimplemented; the fourth was an env lever with no source default |
| Load-width fix as a performance candidate | **Refuted by pre-registered probe** | Section 4.2 |

## 8. Open questions (external only)

Anything answerable by running a command on the A770 belongs in a candidate's local gate probe,
not here. **All five questions this section opened with were answered during the round** - see
section 7 and the candidate entries. What remains genuinely external:

1. **The per-command cost of `zeCommandListUpdateMutableCommandsExp` versus a re-record.** No
   vendor or third-party microbenchmark exists publicly. Do not plan around an assumed number.
   Moot on this machine anyway, since `update()` cannot succeed here.
2. **Any precedent for a lambda-capture to `handler::set_arg` refactor in a comparable codebase.**
   None found. The 276-site / 39-file estimate should be treated as unbounded risk, not as a
   sized task. Note C2 in the angle-E findings offers a way around it entirely:
   `dynamic_command_group` replaces a kernel node's whole command-group function, and its
   alternatives are ordinary lambdas - so the refactor is not required. That path is gated on
   `ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION`, which the existing mutable-command-list probe
   can be re-run to check.
3. **IGC 2.36.3 to 2.38.2 codegen deltas and any Alchemist regressions.** Intel publishes no
   per-release changelog at that granularity and no third-party report exists. Since nothing in
   this round's ranking needs a toolchain bump, hold the pin.
4. **Whether oneDNN's compressed-KV SDPA primitive is optimized on Xe-HPG.** Its grouped-scale
   shape `(N, H, D/G, S)` structurally matches the fork's quants-first layout, but the
   documentation scopes "optimized implementation" to "Intel Graphics Products with XMX support"
   without naming Xe-HPG, and the release notes cite only Lunar Lake and Battlemage.
5. **Whether a code-versus-prose acceptance split is a universal property of n-gram drafting.** No
   published split exists for any n-gram drafter. The local oracle shows one, but it is confounded
   by drafter-cache warming (see section 3) and is not yet evidence of a general property.
6. **Whether upstream PR #25089 (graph-compatibility loosening for MoE decode) holds on Xe-HPG.**
   Still open as of 2026-07-14 with no review activity, and its test plan is Arc Pro B70 (Xe2)
   only. Its correctness argument is microarchitecture-independent and reusable, but it carries
   zero Xe-HPG evidence. Relevant to A4.

## 9. Revised ordering

Round 1's suggested order was: local probe, then speculation bench, then graph hardening, then
split-KV q8_0. **Round 2 overturns it.**

| # | Action | Why here |
|---|---|---|
| 1 | **A0 / A0b** - reproduce upstream's A770 number, and isolate Q4_0 versus Q4_K_M | A cross-source comparison suggests a ~2.1x gap against upstream SYCL on the same part. If real, it dwarfs everything below; if it is the driver, several items below are chasing the wrong thing. Neither costs a kernel change |
| 2 | **C0** - dual-queue shape-routed submission | Largest measured mechanism (12.5-13.0 ms of a 40.4 ms token), no determinism impact, no launch-path refactor, and per-queue routing dissolves the exact objection that killed the global policy. Resolve the immediate-command-list confound first |
| 3 | **A4** - un-gate graphs for MoE | A bug: `MUL_MAT_ID` disables graphs model-wide on the strength of a host wait the fused path never reaches. Qwen has the most enqueues to save |
| 4 | **C1** - the quants-first default (already landed) | Mechanism already measured at +8 to +21%; only the depth 0 and 2048 cells are missing |
| 5 | **C2 occupancy probe**, then C2 or C2b | One instrumented host-side build decides whether the depth gate is a fix or a workaround. Do not write the gate first |
| 6 | **C5** - graph `update_fallbacks` check, then delete or guard | Free diagnostic; likely removes pure per-pass waste and real UB exposure |
| 7 | **C7** - the auto-asymmetric predicate probe | Cheap, and it is a correctness bug report rather than a gate candidate |
| 8 | **C9** - hoist scales to a plane | Free line alignment, bit-identical, about a day |
| 9 | **F/C1** - the speculation determinism diagnostic | One env var and one benchmark run; fix the oracle harness's cache-warming defect regardless |

**Angle F does not deserve the first bench slot, and not because of the RTX 3090 negative result.**
It does not deserve it because `llama-bench` runs no drafter, so speculation reads as 0.000% on
the gate this project is measured by. Its remaining value is the determinism root-cause, which is
now diagnosed and cheap to confirm. The 3090 result is real but its magnitude is unusable here
(NVIDIA Ampere) and it is contradicted by PR #19493's own gains on Qwen3-Coder-Next, so the honest
statement is that upstream spec-decode is **not** globally net-negative for MoE - it is untested on
any Intel GPU by anyone, and the fork's own oracle appears to be the only Xe-HPG measurement that
exists.
