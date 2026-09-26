# Round-2 decode follow-up probes (P6.6)

Falsifier-first probes of the four open items named in
`docs/research/sycl-a770-round2-decode-candidates-2026-07-25.md`. Each probe was run to
resolve a stated question before any source change, per that doc's own local-gate-probe
plans where one existed. No source change is made by this doc; findings that warrant
follow-up work are queued as new RALPH tasks, not implemented here.

Environment: Intel Arc A770/DG2, i915, Level Zero. Source `80d52e708` (merged master,
includes PR #33/#35/#36/#37/#38; the actual build/`build_commit` of record for every
sample below - the build directory predates the later, docs-only PR #39/#40/#41, which
touch nothing outside `docs/research/` and are a no-op diff against this commit,
confirmed by `git diff 80d52e708 ef1c35835 -- ':!docs/research'` being empty). Build:
Release JIT, `/home/svnbjrn/build-p63-80d52e708`, `GGML_SYCL_DEVICE_ARCH` empty. Sole
tenancy of `/dev/dri/renderD128` verified before every timed run; zero new i915/xe fault
lines in any run below.

## (a) Submission-mode local gate probe (section C0)

**Question.** Does `UR_L0_BATCH_SIZE` only move the needle when
`UR_L0_USE_IMMEDIATE_COMMANDLISTS` is disabled, as caveat 1 of section C0 requires before a
dual-queue shape-routed submission design can be trusted?

**Method.** Two paired product campaigns (Mistral-7B Q4_K_M, q8_0/q8_0, depths 0 and
16384, 6 launches/arm, sample-zero discard, 5 retained pairs), each holding
`UR_L0_USE_IMMEDIATE_COMMANDLISTS` fixed and varying `UR_L0_BATCH_SIZE` (unset vs 64):

| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | depth | pp512 delta | tg128 delta |
|---:|---:|---:|---:|
| 0 | 0 | +2.71% | **+32.18%** |
| 0 | 16384 | -0.12% | **+21.56%** |
| 1 | 0 | -1.95% | +0.04% |
| 1 | 16384 | -0.01% | -0.13% |

**Result: CONFIRMED.** `UR_L0_BATCH_SIZE=64` produces a large, real effect when immediate
command lists are disabled and is statistically inert (both deltas within noise) when they
are enabled. Caveat 1 is resolved in the direction section C0 needed: batching genuinely
helps, and only under the non-immediate regime, which is exactly the discriminator a
per-queue routed policy would exploit.

**Follow-up check: which regime is the fork's actual default?** A third campaign left both
variables unset (`UR_L0_BATCH_SIZE=64` vs completely unset, d16384 only) gave **+0.30%**
tg128 - statistically zero, matching the `imm=1` row above, not the `imm=0` row. **The
fork's default on this driver is effectively immediate command lists.** This confirms the
second half of caveat 1's suspicion: the historical `UR_L0_BATCH_SIZE=64` **+47.26%**
figure recorded in the source doc was measured under a regime where batching should have
been inert by this evidence, so it was likely confounded (a different queue-construction
state at measurement time, not the default this fork ships today). That older number should
not be relied on as-is; this probe's four-cell table is the trustworthy replacement.

**Disposition.** Section C0's dual-queue design is empirically well-motivated by this probe
(a routed policy has a real, large, orthogonal lever to exploit) but is **not implemented
here** - it needs the launch-path work section C0 already scoped (two queue constructions,
a routing predicate) plus the pre-registered kill criterion from that section (paired
lower-95% batched-queue gain >= +10%, immediate-queue batch-1 regression <= 2%). Queued as
a follow-up implementation task, not done as part of this probe.

## (b) Attribute the unaccounted decode time (section 1.1)

**Question.** The source doc's bandwidth decomposition left 18.2 ms/token (45% of a 40.41
ms token) as "kernel execution above roofline, ... neither bandwidth nor measured
submission cost." Can the existing `GGML_SYCL_FA_PROFILE`/`GGML_SYCL_GRAPH_PROFILE`
instrumentation attribute more of it?

**Important scope note.** The source doc's 40.41 ms/token reference point predates PR #35
(occupancy governor fix, `parallel_blocks` 2 -> 16) and PR #33 (quants-first default). This
probe re-measures on the **current, already-promoted** default state, which is a materially
different operating point, not a reproduction of the original number.

**Method.** Mistral-7B Q4_K_M, q8_0/q8_0 quants-first, `-p 0 -n 128 -d 16384`, with
`GGML_SYCL_FA_PROFILE=1 GGML_SYCL_GRAPH_PROFILE=1 GGML_SYCL_FFN_FUSION_PROFILE=1
GGML_SYCL_ROPE_FUSION_PROFILE=1`. Measured 20.24 t/s -> 49.41 ms/token.

**FA_PROFILE (measured, not estimated), per token over 128 generated tokens:**

```
GGML_SYCL_FA_PROFILE: route=VEC layout=quants-first launches=4128
  stage1_us=1751660 combine_us=204152 conversion_us=67
  parallel_blocks=16 repeated_packed_kv_bytes=444985245696
```

| Component | ms/token | % of token |
|---|---:|---:|
| FA stage1 | 13.68 | 27.7% |
| FA combine | 1.59 | 3.2% |
| FA conversion | ~0.001 | ~0% |
| **FA subtotal** | **15.28** | **30.9%** |
| Bandwidth floor (weight streaming, unchanged from source doc's estimate) | 9.64 | 19.5% |
| **Remainder (submission + non-FA/non-weight ops + above-roofline execution)** | **24.49** | **49.6%** |

**A concrete, measured explanation for FA's own cost:** `repeated_packed_kv_bytes` is
444,985,245,696 bytes over 128 tokens = **3.48 GB/token** of KV traffic - at `parallel_blocks
= 16` (the promoted occupancy default), the same KV data is read up to 16x redundantly
across split-K blocks before the combine step reduces it. At even a generous 560 GB/s peak,
3.48 GB alone costs 6.2 ms - about 45% of FA's measured 13.68 ms stage1 cost by itself. **The
source doc's "everything except weight streaming is about 0.1% of the token by physics"
claim was accurate for the pre-fix `parallel_blocks=2` state and is no longer accurate
post-promotion:** raising occupancy traded bandwidth (more redundant KV reads) for
parallelism, and that trade is now a real, measurable, non-negligible cost.

**Attempted further decomposition, BLOCKED by a crash.** The doc's own methodology for
isolating submission cost used SYCL graphs (`submit_us`/`wait_us` fields exist in
`GGML_SYCL_GRAPH_PROFILE` for exactly this purpose). Enabling
`GGML_SYCL_ENABLE_GRAPH=1` on this exact configuration (Mistral, q8_0/q8_0 quants-first,
d16384) **crashes**:

```
wait cannot be called for a queue which is recording to a command graph.
Exception caught at file:ggml/src/ggml-sycl/ggml-sycl.cpp, line:5335
Error OP FLASH_ATTN_EXT
```

This is graphs default-OFF (`GGML_SYCL_ENABLE_GRAPH` defaults to 0), so it does not affect
default operation, but it means the `submit_us`/`wait_us` direct-measurement path is
unavailable for this exact shape today. Queued as a new correctness finding, not fixed
here (out of scope for a probe).

**Result: PARTIALLY ATTRIBUTED.** FA's cost is now measured and mechanistically explained
(redundant KV reads at `parallel_blocks=16`), moving it from "unaccounted" to "accounted,
with a named cause." The remaining 24.49 ms/token (49.6%) is not further decomposed here -
the tool that would have split it into submission vs. above-roofline execution crashes on
this shape. This is progress (one large slice attributed with a mechanism) but not a full
resolution of the original question.

## (c) The `gqa_ratio >= 6` auto-asymmetric turbo-K predicate (section C7)

**Question.** Per KVLinC (arXiv:2510.05373), does the current threshold correctly separate
safe from unsafe configurations, or does model family (not GQA ratio) discriminate, meaning
the threshold would silently ship a broken K cache for lower-GQA Qwen3 models?

**Predicate confirmed unchanged** at
[`src/llama-kv-cache.cpp:240`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/80d52e708cacbb0e89c269f6237ca4bb378c4cf6/src/llama-kv-cache.cpp#L240)
(`!disabled && gqa_ratio >= 6 && type_k == type_v`).

**Method (exactly the doc's prescribed local gate probe).** Fetched
`Qwen/Qwen3-1.7B-GGUF` (`Qwen3-1.7B-Q8_0.gguf`, GQA 2:1 confirmed from GGUF metadata: 16
query heads / 8 KV heads, head_dim=128 - qualifies for turbo's 128-element block
requirement). Ran with `-ctk turbo3 -ctv turbo3` at `-lv 5` to read the debug gate line,
then an 8-chunk wikitext-2 PPL discriminator against a q8_0/q8_0 baseline.

**Gate confirmed not firing**, as predicted (`gqa_ratio=2 < 6`):

```
D llama_kv_cache: a2a3c-pre-auto: n_head=16 n_head_kv=8 gqa_ratio=2 disabled=0
D llama_kv_cache: a2a3c-post-auto: downgrade SKIPPED (... gqa_ratio=2 ...), type_k still=turbo3
```

**8-chunk PPL result:**

| KV config | PPL (8-chunk) |
|---|---:|
| turbo3/turbo3 | **659.6625 +/- 58.08** |
| q8_0/q8_0 (baseline) | 16.6391 +/- 1.33 |

**Result: PREDICATE CONFIRMED MISSPECIFIED (kill criterion for the doc's own probe fired
in the failure direction).** The doc's pre-registered kill criterion was: *"If Qwen3-1.7B
turbo3-K PPL is in fact acceptable, the family hypothesis collapses and this whole item is
withdrawn."* PPL is not acceptable - it is a ~40x blow-up versus the q8_0 baseline, on a
model the current `gqa_ratio >= 6` threshold does not gate. This directly confirms the
KVLinC family-not-ratio hypothesis for at least one Qwen3 dense model at the tested
precision. Note the scope limit already recorded in the source doc still applies: this
bears most directly on `TURBO2_0`/`TURBO3_0`; the fork's historical 2887 PPL blow-up was
also at turbo3, one step outside KVLinC's own 2-bit-focused evidence, so the *mechanism*
transfers cleanly but a similar magnitude at `TURBO4_0` or `q8_0` is not established by
this probe.

**Throughput impact: none** (correctness-only, as the source doc stated). **This is now a
confirmed correctness bug**, not just an open problem - queued as a follow-up task to
design and implement a family-or-architecture-aware replacement predicate, not done here
per the probe's own "design question deferred until the probe confirms" framing.

## (d) q8_0 speculative nondeterminism = VEC-to-TILE route flip at batch > 2 (section: root-caused claim)

**Question.** The source doc claims this is "now root-caused": q8_0 KV routes to VEC at
decode (`Q->ne[1] <= 2`) but TILE at speculative-verification batch sizes, while f16 KV
takes TILE both times, and TILE forces K dequant to f16 - a genuine arithmetic difference,
not mere reordering. Does this still hold on current master, three weeks and several PRs
later?

**Method.** Direct source read of `ggml/src/ggml-sycl/fattn.cpp` and
`ggml/src/ggml-sycl/fattn-tile.cpp`/`.hpp` at `ef1c35835`.

**Result: CONFIRMED, mechanism partially refactored.**

- Routing logic unchanged in substance; only the line numbers shifted from the doc's
  2026-07-25 citation:
  - `gqa_opt_applies` gate at
    [`fattn.cpp:511-521`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/80d52e708cacbb0e89c269f6237ca4bb378c4cf6/ggml/src/ggml-sycl/fattn.cpp#L511-L521).
  - Quantized-KV branch at
    [`fattn.cpp:661`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/80d52e708cacbb0e89c269f6237ca4bb378c4cf6/ggml/src/ggml-sycl/fattn.cpp#L661):
    `if (Q->ne[1] <= 2) return BEST_FATTN_KERNEL_VEC;`, else falls through to TILE.
  - That branch still hard-flips at the batch=2 boundary for quantized KV specifically,
    while f16 KV's route depends on `gqa_opt_applies` instead of batch size - so the two
    KV types can and do take different routes across the same batch-size transition.
- **The `need_f16_K`/`need_f16_V` bool parameters the doc cited by name and line number no
  longer exist** at those call sites. TILE has been refactored to select its dequant
  behavior via a **compile-time template parameter** instead: every TILE instantiation in
  [`fattn-tile.cpp:23-60`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/80d52e708cacbb0e89c269f6237ca4bb378c4cf6/ggml/src/ggml-sycl/fattn-tile.cpp#L23-L60)
  (`ggml_sycl_flash_attn_ext_tile_case<D, D, GGML_TYPE_F16>`) is
  baked to `type_K = GGML_TYPE_F16`. Functionally this is the same claim under a different
  mechanism: **TILE always computes on f16-typed K**, so a quantized cache must be
  dequantized to f16 before/during a TILE launch regardless of how that requirement is
  spelled in source, while VEC dequantizes per-element inside the kernel. The "genuine
  arithmetic difference, not merely a reordering" conclusion is unaffected by the
  refactor.

**Disposition.** No source change - this probe's job was verification, and the claim holds.
The doc's own next step (a depth-gated TILE cutover for q8_0/q8_0, section C2/C2b) remains
the scoped follow-up if anyone wants to remove the discontinuity; not undertaken here.

## Summary and follow-up queue

| Item | Verdict | Follow-up |
|---|---|---|
| (a) Submission mode | CONFIRMED, default regime identified (immediate lists) | Implement dual-queue routed submission (section C0 scope + kill criterion) |
| (b) Unaccounted time | PARTIALLY ATTRIBUTED (FA cost measured + explained); blocked by a crash for the rest | Fix the graphs+quants-first-q8_0+deep-context crash; re-attempt submission-cost isolation after |
| (c) `gqa_ratio>=6` predicate | CONFIRMED MISSPECIFIED (40x PPL blow-up on Qwen3-1.7B, GQA 2:1) | Design and implement a family/architecture-aware replacement predicate |
| (d) VEC/TILE route flip | CONFIRMED, mechanism refactored (bool params -> template param) | Section C2/C2b depth-gated TILE cutover, if pursued |

New correctness finding not previously tracked: `GGML_SYCL_ENABLE_GRAPH=1` crashes on
q8_0/q8_0 quants-first KV at deep context during `FLASH_ATTN_EXT`
([`ggml-sycl.cpp:5335`](https://github.com/Raudbjorn/ggml-llama.cpp/blob/80d52e708cacbb0e89c269f6237ca4bb378c4cf6/ggml/src/ggml-sycl/ggml-sycl.cpp#L5335),
"wait cannot be called for a queue which is recording to a command
graph"). Graphs are default-OFF, so this does not affect default operation, but it blocks
any future re-enablement work and blocked half of probe (b) here.
