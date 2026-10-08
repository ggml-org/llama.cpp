# SYCL MoE expert paging on Arc A770: research review and local measurements

Date: 2026-09-27. Host: Arc A770 16 GB (DG2), PCIe 4.0 x16, Ryzen 9 7900X3D, 64 GB DDR5,
Arch Linux, **i915** driver loaded (not xe), oneAPI 2026.0, Level Zero driver 1.17.39758.
Target model: Ornith-1.5-35B-A3B (Q4_K_M in production, IQ2_M also present).

Inputs: a 101-agent deep-research pass over 19 web sources (95 extracted claims, 25 verified by
3-vote adversarial check, 10 confirmed, 15 refuted) plus local probes and a SYCL microbenchmark
run the same night. Every ms and MB figure below is labelled measured or estimate.

## 1. Question and verdict

Three designs for decoding MoE models whose routed experts do not fit in VRAM:

1. Repair the CPU-hooked expert cache (TheTom/leloch design, SYCL port in PR #61).
2. GPU-resident experts with host-backed paging: cold experts in pinned host USM, hot experts in
   a heat-selected VRAM slab, MUL_MAT_ID stays on the GPU and gathers each routed expert through
   a device-side pointer table, promotion on a second queue.
3. Static heat-based placement, remaining experts on the CPU path.

**Verdict: approach 2, gated by a short spike.** It is the only design that removes the per-layer
host dependency, and both of its enabling primitives are now verified on this A770: kernels
reading pinned host USM reach link bandwidth at expert granularity (measured, section 2), and
SYCL-Graph record/replay does run on DG2 through `ext_oneapi_limited_graph` (PR #28725 evidence,
section 5). Approach 1 is ruled out by a measured sync cost (section 2), not by structure alone.
Approach 3 stays the fallback; its ceiling is the ~12 ms of host-layer cost.

The decisive unmeasured number is Ornith's hit-rate-vs-resident-bytes curve. On Q4_K_M the
break-even hit rate against the CPU path is roughly 60% of routed rows (estimate, section 3).

## 2. Local measurements (settle two of the research's open questions)

SYCL microbenchmark (`icpx -fsycl -O2`, in-order queue, `level_zero:0`, 2 GiB pools, slack and
codex also holding the render node, so treat as lower bounds). Source kept in the session
scratchpad as `pcie-bench.cpp`.

| Measurement | Result |
|---|---|
| memcpy H2D / D2H, 256 MiB pinned | 22.8 / 22.4 GB/s |
| kernel reading host USM, 24 x 660 KiB (one layer's routed experts) | 765 us, 21.2 GB/s |
| kernel reading host USM, 8 x 660 KiB | 289 us, 18.7 GB/s |
| kernel reading host USM, 1152 x 660 KiB (742 MiB) | 33.3 ms, 23.4 GB/s |
| kernel reading host USM, 256 x 64 KiB | 798 us, 21.0 GB/s |
| kernel reading device memory, 24 x 660 KiB | 106 us, 153 GB/s (launch-floor bound) |
| trivial kernel + `wait()` | 36.5 us |
| 4 KiB D2H + `wait()` | 72.5 us |
| 4 KiB pinned H2D + `wait()` | 0.9 us |
| provider per-node pattern (2 H2D, kernel, D2H, wait) | 101.5 us |
| kernel submit only, no wait | 20.3 us amortised |

What this settles:

- **Host-USM gather bandwidth on DG2** (research open question 1): a GPU kernel streaming
  expert-sized contiguous chunks from `sycl::malloc_host` gets 19-23 GB/s, i.e. the memcpy
  ceiling. Not measured: degradation while a second queue copies promotions concurrently.
- **Per-sync cost of approach 1** (research open question 4): one MUL_MAT_ID node costs ~100 us
  in the provider's H2D/kernel/D2H/wait pattern. Ornith has 41 blocks x 3 expert tensors = 123
  nodes per token (GGUF metadata), so **~12.5 ms/token of pure synchronisation**, against the
  ~12 ms the CPU path costs today. The 10-25 ms claim in the plan holds; hit rate cannot fix it.
- **Submission-bound decode**: a kernel submit alone costs ~20 us on i915. This is the mechanism
  behind the "launch-bound 59.5 ms step" and the reason graph replay matters (section 5).
- **PCIe topology footgun**: `lspci`/sysfs show the DG2 endpoint 0000:03:00.0 at 2.5 GT/s x1
  (current and max). That is the link to the card's internal bridge 02:01.0 (Intel 4fa4). The
  upstream port 0000:01:00.0 ran at 16 GT/s x16 during the benchmark. ReBAR on (BAR2 = 16 GB).

## 3. Ornith facts and byte arithmetic

GGUF (`gguf_dump`): `qwen35moe`, 41 blocks, 256 experts, 8 used, expert FF 512, embedding 2048.

| Quant | gate/up per expert | down per expert | per routed expert | per token (41 x 8) | all experts |
|---|---|---|---|---|---|
| Q4_K_M (production) | Q4_K 576 KiB | Q4_K 576 KiB or Q6_K 840 KiB by layer | ~1.7 MiB | ~554 MiB | ~17.7 GiB |
| IQ2_M | IQ2_S 320 KiB | IQ3_S 440 KiB | ~1.05 MiB | ~346 MiB | ~11 GiB |

Approach 2 PCIe cost per token at the measured 21 GB/s (estimates):

| row hit rate | Q4_K_M miss bytes | Q4_K_M time | IQ2_M time |
|---|---|---|---|
| 85% | 83 MiB | 4.1 ms | 2.6 ms |
| 75% | 139 MiB | 6.9 ms | 4.3 ms |
| 60% | 222 MiB | 11 ms (break-even with CPU path) | 6.9 ms |
| 0% (stream everything) | 554 MiB | 27.6 ms | 17.3 ms |

The plan's "28 MB, 1.4 ms" figure matches the IQ2_M file at ~92% row hits, not Q4_K_M. On the
production quant expect 3-7 ms at plausible hit rates, still under the ~12 ms CPU path, but the
margin is decided by the hit-rate curve, which no source measures for Ornith (section 4).

Reconciling the plan's own figures: 0.7 ms/layer x 41 layers would be ~29 ms, not 12 ms; the 12 ms
figure corresponds to the fit-placed configuration where roughly 17 of 41 layers' experts sit on
the host (inference from `-fitt 1024` vs `-ncmoe 24` t/s in the systemd unit comments).

Prefill (estimate): a 512-token ubatch touches every expert per layer. Streaming all 17.7 GiB of
Q4_K_M experts over PCIe at 21 GB/s is ~0.84 s per ubatch regardless of ubatch size, a ceiling of
~610 tokens/s; with 60% of bytes resident, ~0.34 s and ~1500 tokens/s. Measured CPU prefill today
is pp512@8k 155.7 tokens/s. So GPU prefill reading cold experts over PCIe likely beats the CPU
path once MUL_MAT_ID reads through the pointer table; the plan's "CPU may still win at prefill"
is probably too pessimistic. Unmeasured.

## 4. Prior art (verified claims, cited)

- **Per-layer host dependency is structural.** The router consumes the current layer's
  post-attention hidden state (fork source `src/models/qwen3moe.cpp:119-137`,
  `src/llama-graph.cpp:2065`), so expert identity is known only immediately before the expert
  matmul; Mixtral-offloading (arXiv 2312.17238, sec 3.2) states next-layer prefetch is impossible
  for MoE without speculation, and its speculative recall was only ~50-70%. Any host-load or
  CPU-compute scheme therefore has a load-or-compute-then-wait per MoE layer. Approaches 1 and 3
  keep it; approach 2 removes it. (Vote 2-1, corroborated by live source.)
- **Demand-copy caches over PCIe are bandwidth-bound and slow everywhere they were measured.**
  Mixtral-offloading Table 2: 3.06 / 2.66 / 2.28 / 2.09 tok/s (A100 / 3080M / 3060 / T4) with LRU
  plus 1-2 expert prefetch; removing them drops the 3080M to 1.76. HOBBIT (arXiv 2411.01433, sec
  2.1-2.2): expert loading is 85.5% (RTX 4090) to 94.5% (Jetson Orin) of inference time; its
  "~80 ms per Mixtral layer over PCIe 4.0" is derived from the theoretical 32 GB/s, not measured.
  (Vote 3-0.)
- **Hit rates are model-specific and the 85%-at-60% assumption is unverified.** Fiddler (ICLR'25,
  arXiv 2402.07033, App. C, Mixtral-8x7B): static popularity placement of 21.9% of experts gives
  25.2% expected hit vs 21.9% random; 48.8% gives 53.0% vs 48.8%, i.e. +3-5 points over random on
  a balanced router. "Not All Models Suit Expert Offloading" (ICLR 2026, arXiv 2505.16056, Table
  1, SRP at m=16): LLaMA-MoE-v2 78.2, Qwen3-30B-A3B 54.1, Phi-3.5-MoE 52.0, OLMoE 50.9,
  Mixtral-8x7B 49.4, DeepSeek-V2-Lite 37.9; shared experts and load balancing suppress locality.
  Cache-Aware Joint Router Adaptation (Sep 2026, arXiv 2609.04895): Qwen3-30B-A3B-Instruct-2507
  moves 1407 MB/token in bf16 at 61.2% LRU hit. Corroborating traces (Suram, arXiv 2608.18261,
  verifier notes only): static pin 59.2% vs LRU 65.9% vs Belady 79.1% at 13.4% residency; LRU
  ~84-88% at 35-45% residency on Qwen3-class routers. Ornith is `qwen35moe`, so the Qwen3-class
  numbers are the closest stand-in, and 85% at 60% resident bytes is plausible but unmeasured.
  (Vote 3-0 on the underlying claims.)
- **Kernels reading pinned host memory are proven on Intel Arc.** Upstream PR #21638 (merged
  2026-04-16) added `sycl_reorder_temp_buffer`: on device-alloc failure the reorder kernel reads
  the whole weight tensor from `sycl::malloc_host` over PCIe; commit message: "still works
  correctly reading from host memory over PCIe". Present in this tree (`ggml-sycl.cpp:4334`,
  default on via `GGML_SYCL_HOST_MEM_FALLBACK`). Tested upstream only on Arc Pro B70; no bandwidth
  figure there. Section 2 supplies the DG2 figure. Atomics on host USM are unsupported. (Vote 3-0.)
- **SYCL-Graph replay does run on A770.** Upstream PR #28725 (open, updated 2026-09-25) replaces
  per-call record/finalize with warmup + `replay_graph`, gated on `ext_oneapi_limited_graph`
  (present on this A770). A770 results: i915 tg128 29.12 -> 44.28 t/s (Qwen2.5-coder-7B Q6_K);
  xe 41.89 -> 44.68; Qwen3-8B Q4_K_M xe+AOT 49.42 -> 54.43; independent xe tester 50.05 -> 54.15;
  B70 outputs byte-identical; one A770/xe null run unexplained. Native recording errored on A770
  with the latest oneAPI; the working path is non-native recording. The fork's standing decision
  ("DG2 lacks `aspect::ext_oneapi_graph`, so replay cannot amortize") conflated graph *update*
  with graph *replay*. (Vote 3-0.)
- **No prior-art system implements approach 2's fast path** (zero-copy per-token gather through a
  device pointer table). Mixtral-offloading, HOBBIT, Fiddler and the trace studies all copy whole
  experts H2D on miss or compute them on the host; the claim that ExactMoE is direct prior art was
  refuted 1-2 (it stages misses into VRAM slots asynchronously). Approach 2 is novel relative to
  the corpus: upside and the main execution risk. (Vote 3-0 on the components.)
- **Approach 3 has measured prior art and bounded gains**: +3-5 points over random on balanced
  routers, ~7 points below LRU on Qwen3-class routers at equal budget, and as described it keeps
  the per-layer CPU/GPU split, so its ceiling is the host-layer cost (~12 of 59.5 ms, ~20%,
  estimate), below the +19% sarashin65 saw on a DDR4 B70 host. (Vote 3-0 on the components.)

## 5. Prerequisites and risks for approach 2 (this tree)

1. **MUL_MAT_ID disables SYCL graphs today.** `check_graph_compatibility()`
   (`ggml-sycl.cpp:6722-6727`) rejects any graph containing MUL_MAT_ID because the generic
   `ggml_sycl_mul_mat_id()` copies ids D2H and calls `stream->wait()` (`5389-5396`). The fused
   decode path `ggml_sycl_mul_mat_id_mmvq_fused()` (`5267-5325`, `ne12 == 1`) already passes
   `ids->data` to the kernel with no host wait, so the exclusion is stricter than the decode path
   needs. Consequence today: `GGML_SYCL_ENABLE_GRAPH=1` in `llama-gpu@.service` is inert for
   Ornith and every other MoE model. Approach 2's "nothing blocks graph capture" is a rewrite of
   the generic path plus a relaxed compatibility check, not a free consequence.
2. **This tree re-records and re-finalizes the graph every step** (`ggml-sycl.cpp:6848-6860`:
   without `ext_oneapi_graph` update support it finalizes a fresh graph each pass). PR #28725's
   warmup + replay is not ported. With ~20 us per submit measured, a 59.5 ms step that is
   launch-bound stands to gain materially from replay independent of any expert work; PR #28725's
   dense-7B i915 delta was ~11.8 ms/token (estimate from 29.12 -> 44.28 t/s). Port it early; it
   is the cheapest lever in this whole space and approach 2 depends on it.
3. **Hit-rate curve unmeasured.** Break-even on Q4_K_M is ~60% row hits (section 3). Needs an
   Ornith routing trace with static / LRU / heat policies against resident bytes.
4. **Concurrent promotion traffic.** Bandwidth measured with the link otherwise idle. Promotion
   copies on a second queue share PCIe with the gathers; the byte budget per token must be set
   from a measurement with both active.
5. **Device pointer table mutation between graph replays.** Replay executes recorded kernels with
   recorded arguments; the table itself is indirect data, so mutating it between replays is legal
   in principle, but ordering between the promotion queue's writes and the replayed gathers needs
   an explicit dependency. Unverified.
6. **Driver.** All timing here is i915. PR #28725 shows graph deltas 5-8x smaller on xe. Re-probe
   if the box moves to xe.
7. **Host USM limits.** Atomics on host USM unsupported; large pinned allocations (17.7 GiB for
   all Q4_K_M experts) untested; claims about a >2 GiB relaxed-allocation hazard were refuted, so
   no verified number exists either way.

## 6. Recommendation

Gated spike, 1-2 weeks, before committing 4-6 weeks:

- **Gate A (done, pass):** host-USM gather bandwidth at expert granularity >= 15 GB/s. Measured
  19-23 GB/s.
- **Gate B:** Ornith routing trace (llama.cpp `-lv 4` expert ids or a small hook) giving row hit
  rate vs resident bytes for static, LRU and heat-decayed policies. Pass if >= 75% at <= 60%
  resident bytes on Q4_K_M.
- **Gate C:** generic `ggml_sycl_mul_mat_id()` with device-side id resolution, MUL_MAT_ID removed
  from the graph exclusion, PR #28725 replay ported; Ornith decode with all experts on GPU (or the
  IQ2_M file) must be byte-stable under replay and faster than eager.
- **Gate D:** gather bandwidth re-measured with a concurrent promotion queue at the intended
  per-token byte budget.

If B fails, approach 3 with the same heat data is the fallback; its gain is capped near 20%.
If C fails on replay correctness, approach 2 still works eagerly and still removes the per-layer
host sync, but loses the launch-overhead recovery.

## 7. Not claimed

- Bandwidth and sync numbers were taken with a browser and two other processes holding the render
  node; they are lower bounds, not clean-room figures.
- No Ornith hit rate has been measured; every t/s projection for approach 2 is arithmetic.
- The graph-replay gains are from an unmerged upstream PR with COMMENTED reviews and one
  unexplained null run, on a dense model.
- The prefill estimate ignores compute time and KV traffic on the same link.
- The research pass refuted every third-party claim about Intel host-USM bandwidth, per-launch
  cost and pointer-table/graph compatibility; those areas rest only on section 2's local numbers.

## 8. Sources

- arXiv 2312.17238 Mixtral-offloading (Eliseev, Mazur)
- arXiv 2411.01433 HOBBIT
- arXiv 2402.07033 Fiddler (ICLR 2025)
- arXiv 2505.16056 Not All Models Suit Expert Offloading (ICLR 2026)
- arXiv 2609.04895 Cache-Aware Joint Router Adaptation
- arXiv 2608.18261 Suram expert-cache traces (verifier-cited; two derived claims refuted over
  model attribution, treat as medium quality)
- arXiv 2609.12978 SeqMoE (verifier-cited)
- github.com/ggml-org/llama.cpp/pull/21638 (host-memory reorder fallback)
- github.com/ggml-org/llama.cpp/pull/28725 (SYCL graph warmup + replay)
- github.com/ggml-org/llama.cpp/pull/27861, issues/26752, discussions/24528 (leloch expert-cache RFC)
- github.com/TheTom/llama-cpp-turboquant/pull/364
- github.com/intel/compute-runtime/issues/600, github.com/intel/llvm/issues/22958
- intel/llvm sycl_ext_oneapi_graph extension spec
- Local: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines cited above), GGUF metadata of
  `/mnt/mrgr/models/ornith-1.5-35b-a3b/*.gguf`, `pcie-bench.cpp` (session scratchpad), sysfs PCIe
  link state, `sycl-ls --verbose`.
