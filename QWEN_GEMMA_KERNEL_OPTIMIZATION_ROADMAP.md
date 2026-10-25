# Qwen and Gemma kernel optimization research roadmap

Status: the generic SM120 D512 feature-chunk pipeline is now qualified on its compiled F16 resume/general-mask 8-by-8 domain. A fresh full-runtime screen versus the shipping conversion-plus-ring implementation measures Gemma prefill gains of 7.34% target-only / 7.38% MTP3 at 64 Ki, with native-exact outputs, unchanged decode layouts and decode changes within 0.35%. All-pair, MTP/lifecycle, resource-admission and strict safety gates pass. Direct Q-fragment reloading was rejected and removed. The next authorized step is the post-commit Gemma MTP3 maximum-fit sweep from 8 through 256 Ki and an overlay with the historical 120 Ki series. Further decode optimization remains deferred.

The final [published profile](benchmarks/data/d512-feature-pipeline-v1.json) supersedes promotion decisions in the initial experiment notes below. Its consolidated audit contains 65 entries (two intended baseline reds), 17 strict zero-error CUDA gates, 120 native-matched server records and 94 portable HOST/ASan binary hashes. The 769 MiB wide sanitizer-only fixture is explicitly approved and passes; normal model execution remains 1 GiB. Checkpoint addressing is narrowed to the winning path, and the other seven 8-by-8 function bodies retain byte-identical SASS. Pilot timings below use different controls and are preserved as historical observations, not combined gains.

## Research objective

Find the fastest validated kernel implementation for each admitted Qwen and Gemma geometry, for both prefill and decode, within the existing memory budget and numerical contract. Optimize useful end-to-end performance, not register usage, occupancy or a single profiler percentage in isolation.

The expected result is a small set of kernel variants behind common capability-driven dispatch, not necessarily one universal kernel. Common code owns encoded layouts, source identity, leases, phase workspace, transfers and completion. Geometry and device capabilities choose arithmetic implementations; model names do not become kernel-selection rules.

Prefill is the first priority. Decode is an explicit research branch in this new plan, with transport and kernel execution investigated separately. Planning that branch does not start the previously deferred decode implementation work.

Success means a demonstrated improvement in prefill latency/throughput or useful generated tokens per second, with unchanged correctness, bounded storage and no material regression in the other phase. It does not mean proving a global optimum over every possible algorithm or GPU.

## Current model independent D512 focus

The immediate target is the D512 attention operator, not a Gemma-specific implementation. Gemma is an integration and end-to-end workload; Qwen D256 remains a regression control. The initial kernel contract has Q/K head dimension 512 and V head dimension 512, on SM120. A 512-value MLA operator with a different Q/K dimension is a separate geometry, not automatically covered by the D512 label.

Cover fully resident and bounded span/resume execution. Select kernels by head counts, query width, storage layout, masks, numerical policy and device capabilities. No layer number, model name, Gemma SWA allocation or assistant source mapping belongs in the arithmetic selector. A generic operator does not automatically qualify another model's context/cache integration.

The current streamed MMA validator accepts GQA ratios 2, 6 and 8, one sequence, up to three spans per launch, F32 Q/output and the admitted K/V reader formats, but admission also requires a compiled tile. Wide-query GQA 2 requests can select uncompiled 16-by-2 or 32-by-2 tiles and are rejected. It rejects nonzero bias/softcap and attention sinks. Native/tile paths may have different capabilities; do not advertise the MMA fast path as arbitrary D512 support.

Start with a pairwise coverage panel rather than a huge Cartesian benchmark:

| Q heads and KV heads | Purpose |
| --- | --- |
| 8 and 4 | GQA ratio 2 narrow-width execution and wide-width rejection probes |
| 24 and 4 | Admitted GQA ratio 6, including padded-head behavior |
| 32 and 4 | Existing profiled GQA ratio 8 |
| 8 and 1 | Same ratio with a single KV head |
| 64 and 8 | Same ratio with more independent heads |
| 8 and 8, 16 and 4, 32 and 2 | Admission probes for ratios 1, 4 and 16; new fast-path support requires its own implementation and qualification |

Combine representative head cases with decode/verification widths 1-4, selected wider supported widths, prefill widths 64/128/129/241/255/256, context tails, dense/compact masks and resident/wrapped/streamed sources. Preserve the existing 64 K/V-pair coverage, using broader geometry sampling without claiming every combination has been tested. Optional semantic features need independent gates rather than silent fallback or relaxed checks.

Workspace must also be model-independent. For the existing eligible quantized wide-prefill preparation, the decoded 2048-row F16 K/V stage costs `2048 * KV_heads * (Dk + Dv) * 2` bytes. D512 therefore costs 4 MiB per KV head: the measured 16 MiB stage assumes four KV heads. Derive requirements from the actual geometry; F16/F16 and small-query paths do not acquire this stage merely because they are D512.

The [initial D512 audit](benchmarks/results/d512-attention-research-20261021/initial-audit.json) records these first findings:

- SM120 currently selects the existing Ampere-family configuration heuristics for this path, not a dedicated SM120 D512 policy.
- The native 64-column D512 kernel uses 248 registers/thread. The profiled span/resume kernel uses 255; its matching 8-by-8 build object has a 368-byte stack frame. This suggests a bookkeeping/live-range investigation, not proof that resumability alone causes the cost.
- The profiled SASS and the 8-by-8 build object each contain 290 static local load/store instruction rows. This corroborates local-memory operations; it is not a dynamic cycle count.
- For the profiled 64-column shape, shared-memory accounting is 66560 bytes for Q, 16384 for the reused K/V tile, 640 for the mask, 168 for span descriptors and 1024 driver bytes, totaling 84776. Removing span descriptors alone cannot solve the shared-memory occupancy limit.

The first two operator experiments below test metadata/checkpoint-address lifetimes and Q-operand delivery without changing MMA fragments or reduction order. Their results are specific to the selected tile and delivery path, not universal gains for every model using D512.

### Historical initial operator experiments

The immutable native library remains the oracle. Each experiment replaces only the D512 8-by-8 runtime module in an isolated process; other tile geometries and the native kernel remain unchanged. No production library was relinked or model profile changed.

| Experiment and control | 8 Ki throughput | 64 Ki throughput | 120 Ki throughput | Decision |
| --- | --- | --- | --- | --- |
| Checkpoint address rematerialization versus the qualified starting implementation | +2.84% | +3.52% | +3.62% | Continue with narrower dispatch and integration tests |
| Direct Q-fragment reloading versus the checkpoint candidate | -30.47% | -30.74% | -30.60% | Reject this standalone reload design |
| Internal K/V feature-chunk double buffering versus the checkpoint candidate | +19.16% | +21.81% | +21.99% | Retain prototype for broader dispatch and model qualification |

These are five-sample, one-layer F16 operator screens with 256 queries, 32 Q heads, four KV heads and 2048-row waves. Each sample includes 16 operator iterations. Sequential process timing is provisional, not a full-model token-speed or statistically controlled A/B claim. [Checkpoint results](benchmarks/results/d512-attention-research-20261021/checkpoint-address-v1/analysis.json), [direct-Q results](benchmarks/results/d512-attention-research-20261021/query-direct-v1/analysis.json).

Checkpoint rematerialization reduced the profiled specialization's stack frame from 368 to 64 bytes and its static local load/store instruction count from 290 to 100; register count stayed at 255. However, the distinct F16/resume/dense-strided specialization's stack grew from 80 to 328 bytes. Scope the change and test those other paths before promoting the common-header prototype.

The direct-Q variant preserved operand conversion and attention order, but repeated global loads/conversions and increased the profiled stack to 120 bytes. Its host shared-memory grant was deliberately unchanged, so it does not test a reduced shared-memory allocation or rule out other Q-staging designs. Its source, patch, binaries and results are archived; the working tree retains the checkpoint and feature-pipeline prototypes.

Each candidate passed 1759 assertions over 100 head-layout profiles: 88 execute and 12 require clean rejection. The panel includes all five head layouts above, 2049/8193-token histories, query widths 1/4/64/128/256, F16/F16 and Q8_0/Q4_0, dense/compact masks and encoded/prepared delivery. Outputs and every native checkpoint payload byte match the frozen control; pointer-bearing tile descriptors are not numerical state. Each panel also passed strict CUDA memcheck with zero errors. The separate normal all-64-pair D256/D512 matrix passed 19969 assertions over 768 profiles for each candidate.

### Historical D512 feature pipeline pilot

The new prototype overlaps loading one K/V feature chunk with computation on the other. Q remains cached. K features still accumulate upper then lower, softmax/rescaling stays at the original boundary, and V features update lower then upper. Existing reader-retirement barriers protect both shared buffers; the last tile never prefetches beyond its current wave. This is GPU-internal scheduling, not the previously qualified host-to-GPU half-ring pipeline.

The isolated build replaces only the SM120 D512 8-by-8 F16/resume/general-mask module. It adds one 16384-byte on-chip shared tile without a global allocation or arena increase. Total user shared memory is 100136 bytes against the measured 101376-byte opt-in limit, leaving 1240 bytes. Thread/register counts remain 256/255; stack size changes from 64 to 96 bytes. The generated specialization retains 768 static MMA instructions, 198 asynchronous-copy instructions and 76 block-barrier instructions. These static counts do not establish dynamic stall costs.

In addition to the frozen 1759-assertion output/checkpoint panel and strict memcheck, the candidate passed 43778 progressive-attention assertions across all K/V pairs, the 19969-assertion prepared matrix and 250 focused mask/tail/wrapped-span assertions. Targeted racecheck and synccheck used 1057 active rows, 129 queries, 32 Q heads and three reordered spans, with zero reported hazards/errors. [Audited results and source/binary pins](benchmarks/results/d512-attention-research-20261021/feature-pipeline-v1/analysis.json).

Single matched full-model screens used Gemma Q3_K_M, Q8_0/Q4_0 KV, a fixed 1 GiB arena, batch/ubatch 256, UVM off, MTP off, fresh `capacity-271` prompts and 256 outputs. Both variants used the same frozen optimized server/libraries, differing only in the isolated module: checkpoint-only control versus checkpoint plus feature pipeline.

| Configured capacity | Control prefill tokens/s | Pipeline prefill tokens/s | Prefill change | Decode change |
| --- | --- | --- | --- | --- |
| 8 Ki | 1325.99 | 1340.60 | +1.10% | -0.15% |
| 64 Ki | 844.27 | 896.79 | +6.22% | -0.11% |

Each paired prompt and all 256 generated IDs matched exactly; decode pool, residency and ring layouts were identical. These single-pass observations are not repeated statistical qualification or a 120 Ki model result. The same-model comparison does not establish another D512 model's cache/context support. Production was restored healthy; no production library, commit or push was changed by this screen.

Before promotion, narrow eligibility to qualified tile/resource cases, address the checkpoint prototype's dense-strided resource regression, and qualify actual MTP/lifecycle paths and remaining dispatch geometries. Use targeted profiling to attribute the improvement rather than inferring achieved overlap from the source schedule alone.

Related [ROCm fork](https://github.com/stew675/llama-cpp-rdna-boosts/issues/188) and [upstream](https://github.com/ggml-org/llama.cpp/issues/30838) reports motivate the same geometry-focused investigation, but do not establish a shared root cause. That reported tile/WMMA dispatch differs from our already-MMA CUDA path. Do not transfer AMD tuning constants or infer a common spilling bottleneck from those reports.

## Evidence and starting point

The starting implementation already includes bounded prefill conversion reuse and the half-ring transfer pipeline. Preserve those qualified improvements rather than comparing against an obsolete pre-conversion implementation.

The latest evidence is the [audited 120 Ki analysis](benchmarks/results/nsight-120ki-20261020/20261021T045523Z-8uqoeija/complete-analysis.json) and its [counter summary](benchmarks/results/nsight-120ki-20261020/20261021T045523Z-8uqoeija/counter-summary.csv). These local artifacts pin source, binaries, inputs and captures. The workload used an RTX 5070 Ti, SM120, a fixed 1 GiB arena, Q8_0/Q4_0 KV, batch/ubatch 256, UVM off, a 122609-token uncached prompt and 256 outputs, with MTP off. All eight isolated attention/matrix samples matched their unprofiled output controls.

| Observation | Gemma | Qwen | Research implication |
| --- | --- | --- | --- |
| Global attention dimensions | D512, 32 query heads, 4 KV heads | D256, 24 query heads, 4 KV heads | Evaluate both geometries; kernel durations are not equal-work comparisons. |
| Prefill attention share of kernel-active time | About 72% | About 60% | Attention is the first prefill target. |
| Sampled prefill attention registers and occupancy | 255 registers/thread; 16.65% | 253 registers/thread; 14.58% | More live register state is not the default optimization. |
| Sampled prefill attention total shared memory per block | About 84.8 KB; one-block limit | About 35.0 KB; two-block limit | Register and shared-memory constraints must be considered jointly. |
| Reported prefill attention spilling requests | 1304000 | 0 | Attribute Gemma's spills to instructions; do not infer their runtime cost from the count. |
| Prefill attention scheduler cycles with no eligible warp | About 84% | About 75% | Investigate dependency chains and pipeline contention. |
| Sampled prefill weight GEMM | 255 registers; 16.65% occupancy | 255 registers; 16.66% occupancy | GEMMs also have resource pressure, but are a secondary long-context target. |
| Decode window containing H2D without a kernel | About 56% | About 35% | Decode needs a transport investigation, not just a faster arithmetic kernel. |
| Sampled decode weight matvec occupancy | About 70% | About 68% | Do not assume the decode matvec shares the attention kernel's occupancy problem. |

These counters cover one invocation per selected kernel family/model/phase. Kernel replay changes execution and cache conditions. The original reports do not establish that spilling is the main cause, that shared-memory bank conflicts dominate, or that every exposed-copy interval is removable.

## Constraints and decision gates

### Primary numerical contract

The default research track preserves the qualified native numerical order. Keep per-output MMA accumulation order, KV tile visitation, split ownership, softmax update/rescaling points, final reduction tree, intermediate precision, quantization conversion, masking and softcap behavior. Preserve checkpoint contents and exact bits in `VKQ_C`, `KQ_max` and `KQ_rowsum`.

Scheduling independent work differently can be eligible. Changing a tile size, warp mapping, pipeline configuration or compiler hint is not automatically eligible: first determine whether it changes the per-output operation sequence. The comment that a configuration affects only performance is not sufficient proof.

Fast math, a different exponential approximation, FP8/FP4 substitution, reassociation, a different split-K merge or a changed precision/reduction order require a separate user decision. Such alternatives may be researched as numerical-contract alternatives, but must not be promoted under exact-order qualification. A library advertising exact attention is not promising bit identity with this implementation.

### Memory and ownership contract

- Start with the same fixed parent arena, model weights, KV formats, batch sizes and request profile. Include conversion, state, graph scratch and all persistent local storage in accounting.
- Keep the accepted preparation policy inside the arena and reclaimed for decode. The eligible quantized wide-prefill stage is 8 MiB for D256 and 16 MiB for D512 at four KV heads; other head counts require geometry-derived quotes, not those constants. Do not add an automatic low-budget fallback policy or silently change an existing measurement profile's minimum.
- A new workspace requirement needs an exact quote and a decision before changing a profile. A second decoded staging buffer is not free merely because it fits inside the same parent: it reduces KV residency.
- Reuse existing allocators, source bindings and fences. No hidden full-history device gather, permanent converted-history cache, extra assistant histories or unbudgeted device buffers.
- Preserve native Gemma local SWA and history-availability guards, Qwen recurrent state, read-only assistant bindings, target/read feedback isolation and serial parent ownership. Assistant weights remain outside the arena.
- Selecting an existing kernel for an unprofitable geometry is normal dispatch; it does not authorize changing admission minima or silently accepting a failed quota.

### Hardware and research scope

Optimize the measured SM120 platform first. Query actual device attributes and occupancy breakpoints. CUTLASS explicitly distinguishes datacenter SM100 from GeForce SM120; an `sm100a` kernel is not a drop-in RTX 50-series implementation. [CUTLASS architecture guidance](https://github.com/NVIDIA/cutlass#target-architecture).

Keep other architectures on their qualified paths until separately validated. Preserve the existing distinction between older-architecture functional compilation and physical-device runtime qualification. Do not introduce a kernel-generator/packaging refactor or a clean compile-time benchmark as a side effect.

The original screening contexts were 8, 64 and 120 Ki. The user subsequently authorized a new post-commit Gemma MTP3 sweep, 8-256 Ki in 8 Ki steps, with 256 outputs and independent maximum-fit detection using the existing finder. Do not resume the stopped older sweep or mix its runtime cohort into the new one. Parallel slots, multi-GPU operation and new multimodal/model variants remain outside this scope.

## Milestones

K0/K1 produced frozen controls and source/resource hypotheses. The first K2/K3 winner has completed broader dispatch, MTP and safety qualification; further candidates and instruction attribution remain research work. Each experiment ends in retain, revise or reject; a rejected optimization is a valid research result.

| Milestone | Investigation | Required output before moving on |
| --- | --- | --- |
| K0 | Freeze objective, numerical contract and reproducible baselines | Named controls, actual geometry/capability inventory, memory accounting and an agreed measurement protocol. |
| K1 | Attribute costs to representative instructions and workloads | A source/SASS cost map and ranked hypotheses supported by controlled diagnostic evidence. |
| K2 | Reduce prefill register pressure and shared-memory constraints | Order-preserving variants with measured resource changes, exact outputs and lower complete operator time. |
| K3 | Reduce prefill state traffic and handoff costs | A bounded partial-accumulator/pipeline candidate, or evidence that the added complexity is unprofitable. |
| K4 | Investigate secondary prefill costs | Selective GEMM, conversion or small-operation improvements where full-model time justifies them. |
| K5 | Bound and reduce decode transfer exposure | Measured traffic/bandwidth/critical-path model and a safe transfer/residency candidate, if headroom exists. |
| K6 | Optimize decode attention and verification | Kernel variants for TG1 and speculative widths, with useful-token throughput gains rather than only faster isolated kernels. |
| K7 | Compare alternative kernel families | A hardware/numerical compatibility matrix and bounded prototypes only for viable candidates. |
| K8 | Select dispatch and qualify winners | The best validated variants by geometry, full-model performance evidence, safety regressions and documented rejected alternatives. |

K1 includes the literature and implementation survey for K7. Prefill K2/K3 is the recommended first implementation sequence; decode K5/K6 is a separate branch rather than a prerequisite for prefill progress.

## K0 Freeze contracts and baselines

Archive the current optimized server, native operator controls and all resolved runtime libraries, source/diff hashes, compiler flags, model/assistant/corpus hashes, commands and device configuration. Keep binary build origin separate from later documentation commits. Preserve existing accepted captures and original failed attempts. Ki denotes 1024 tokens in context labels.

Measure prefill latency/throughput, time to first token, steady-state decode latency and accepted output tokens per second separately. For MTP, do not count proposals as useful generated tokens. Record end-to-end request time, peak device and pinned-host memory, H2D bytes, phase workspace, resident/ring layout and startup resource cost. Measure complete prefill as well as early/middle/late windows; a faster late window alone does not establish a faster full prompt.

Use two explicitly separate comparisons:

1. Same parent arena and unchanged model/KV/request settings, to identify a kernel or scheduling improvement.
2. Independently maximum-fit arenas, only for finalists, to reveal whether extra scratch harms overall capacity or performance. Do not label this as an equal-memory comparison.

Honor the earlier single-pass sweep preference: no repeated full-model trial matrix is scheduled by this plan. Use a single matched A/B pair to screen candidates. A solid claim for a small gain needs approval for limited confirmation on selected winning points, not an automatic repeat of the whole sweep. Without confirmation beyond observed noise, label the result provisional rather than reliable.

## K1 Build a causal cost map

The current samples are a starting point, not a complete distribution. Rank kernels by cumulative timeline cost and record their launch shapes, native plans, query widths and cache roles. Include dominant weight variants, streamed and resident global attention, Gemma SWA, Qwen recurrent work, assistant reads and verification. Short-context profiles can have different priorities from 120 Ki.

Use the existing binary's SASS and resource metadata before building anything. Map spills, long live ranges, loads/stores, fixed-latency dependencies, math-pipe throttling, barriers and checkpoint accesses to actual instructions. Collect targeted PC/source counters only where needed. If line information requires a diagnostic build, retain the original control and compare generated code/resources before interpreting it as representative.

Construct focused tests using existing fixtures:

- Compare identical arithmetic with data already resident versus progressive delivery. This separates reader execution from transport without claiming a resident-only test qualifies the full server.
- Compare span counts, wrap positions and checkpoint boundaries at fixed logical contents and native operation order.
- Count checkpoint bytes and invocations, conversion work, DMA submissions, queue waits and graph transitions. Treat the source-count and batch-shape changes as diagnostic variables, not silently changed production profiles.
- Measure a pinned-host copy-only ceiling using the same sizes/queues. Compare it with achieved H2D bandwidth; copy activity alone is not proof of link saturation.

Use isolated kernel processes for Compute counters, validated Systems timelines for overlap, and interval unions instead of adding overlapping times. Keep profiler timing separate from performance evidence. NVIDIA documents both replay/cache effects and the distinction between active, eligible and issuing warps. [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

Exit: each leading hypothesis has an instruction/path, a diagnostic test, a predicted observable change and a plausible end-to-end benefit. Spilling counts, low occupancy or many launches alone do not satisfy this gate.

## K2 Improve prefill resource use and scheduling

Start with the common D512 attention paths and the head-geometry panel above, using Gemma as one integration workload and D256 Qwen as a regression control. The aim is fewer unnecessary live values and better overlap of independent instructions, not a register-count or occupancy target in isolation.

The [deeper D512 assessment](benchmarks/results/d512-attention-research-20261021/optimization-paths.md) identifies two larger structural differences: the starting D512 path has twice the output-state words per thread and uses one shared K/V tile, versus D256's two-stage pipeline. Both configurations allow about eight resident warps, so occupancy percentage alone does not explain the efficiency gap. GPU-internal feature-chunk double buffering now has the operator and single-pass model results above. SM120 producer/consumer register specialization and output-feature ownership changes remain proposals.

The supplied TurboQuant fork is also assessed there. Its lazy K/V transforms change KV representation and finite-precision evaluation order, and its dispatch uses vector attention. Keep that option separate from native-exact Q8/Q4 prefill optimization. SM120a register reallocation is supported by the PTX ISA; do not incorrectly exclude it together with Hopper-only WGMMA.

Prioritize one change at a time:

1. Shorten temporary lifetimes around loads, conversion, address calculations, mask handling and softmax updates. Check whether materialized arrays or repeated uniform calculations create avoidable pressure.
2. Audit Q placement and the lifetimes of `tile_Q`, `tile_K`, `tile_V` and `tile_mask`. Reuse shared regions only after their last reader, with the required barriers intact.
3. Reschedule independent address/load/math work while preserving each accumulator's arithmetic sequence. Attribute a reduced stall count to a lower elapsed time, not just a different instruction count.
4. Evaluate narrowly scoped stage/layout/compiler choices only after proving that native arithmetic and reduction order are unchanged. Do not use a lower register limit blindly: it can create more spills.

Calculate real occupancy breakpoints before compiling variants. For example, a 256-thread block at roughly 255 registers/thread is not close to allowing two blocks in a 64K-register SM. Shared memory independently prevents two current Gemma prefill blocks from residing together. A modest reduction in only one resource may leave occupancy unchanged yet still help instruction latency; measure both outcomes.

Relevant implementation: [native MMA configuration and accumulator code](ggml/src/ggml-cuda/fattn-mma-f16.cuh) and the dimension-aware runtime adapters. CUDA's guidance treats register/shared-memory usage as a tradeoff rather than prescribing maximum occupancy. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Exit: exact operator outputs/checkpoint bits, valid resource accounting and improved operator time including required preparation and checkpoint work. A resource improvement without a runtime improvement does not establish a winner.

## K3 Reduce partial accumulator traffic and handoffs

First quantify state traffic; do not assume it dominates. The existing resumable path copies native accumulator, maximum and rowsum state at boundaries. The experiment is to avoid unnecessary movement while preserving those exact state transitions and their precision.

Candidate order:

1. Optimize checkpoint addressing, packing and bit-preserving transfers without changing the numerical work.
2. Group work only where native tile order, split ownership, softmax rescaling and final reduction remain identical.
3. Investigate bounded accumulator persistence across an approved sequence of tiles. Preserve conversion, mask, source identity and checkpoint semantics rather than independently normalizing chunks and merging their outputs.
4. Consider a narrowly fused conversion/attention or persistent consumer only after establishing that removed handoffs exceed its extra register, shared-memory and synchronization cost.

Reuse the existing checked workspace and ring machinery in [the common CUDA adapter](ggml/src/ggml-cuda/kv-stream-partial.cu) and [resident scheduler](src/llama-kv-stream-resident.cpp). A persistent consumer must not occupy the GPU while waiting for a conversion producer that cannot be scheduled. Test producer progress, queue dependencies, cancellation, failure recovery and final-reader fences explicitly.

No full-history F16 cache or whole-model megakernel is assumed. If a second decoded slot or extra staging is necessary, quote its bytes and measure the residency loss at the same parent size before seeking a memory-policy decision.

Exit: exact output/state across streamed, wrapped and short-tail cases, no hidden allocation or deadlock, and a full-operator gain followed by a real-model gain. Preserve and defer a no-gain prototype rather than reviving the removed G5c patch by default.

## K4 Investigate secondary prefill costs

Use the updated time distribution after K2/K3 to decide whether weight GEMMs, row conversion, mask/index work or small operations justify attention. The current sampled prefill GEMMs also use 255 registers/thread and report some spill traffic; do not generalize the attention findings to every quantized weight kernel.

Prototype address/layout reuse, load scheduling and safe fusion only where the native conversion, accumulation and rounding sequence can be retained. Include dequantization, conversion, epilogues and workspace in operator timing. An F16-only benchmark is not evidence for a quantized GGUF path.

Retain existing reduced-build and all-pair capability checks. Specialize only measured profitable cases; do not introduce a large Cartesian product of quantization types and arithmetic bodies. Investigate Qwen recurrent or Gemma local-attention work when its measured share warrants it, especially at shorter contexts.

Exit: an end-to-end gain large enough to justify maintenance and compatibility cost, with the other phase and the other model preserved or explicitly dispatched to its existing path.

## K5 Establish the decode transport limit

Measure actual transferred bytes, cache hits/misses and reuse by source layer and access role. Distinguish compulsory history traffic from avoidable rereads, redundant conversions, tiny transfers and synchronization-induced gaps. Audit the existing source-aware lookahead and retained-ring reuse before adding another mechanism.

Use controlled resident, copy-only and progressive operator cases to bound compute and transport separately. For transferred bytes B and achieved sustained bandwidth W, B/W is a transport lower bound, not an additive prediction of decode latency. Dependencies may prevent perfect overlap.

Candidates include better transfer granularity, earlier safe lookahead and verified reuse of unchanged pages across target/assistant reads. Preserve source generations, role feedback and invalidation on mutation. Releasing an encoded slot after conversion requires proof that no attention path still reads it; a decoded slot remains live until its consumer finishes.

Keep the same physical budget. Extra residency that comes from a larger arena, different quantization or removing assistant weights is a different experiment. Cache-only tests with CPU-resident unused weights cannot qualify joint GPU target/assistant performance.

Exit: lower actual traffic or less exposed waiting with valid lifecycle behavior and higher useful decode throughput. Moving a wait to another function is not an improvement.

## K6 Optimize decode attention and verification

After separating transport, optimize the remaining reader cost for TG1 and the admitted speculative widths. Address dependency chains, address arithmetic, cache locality and redundant state movement while retaining native operations and reductions. Do not compare invocations processing different numbers of keys as equal work.

Profile MTP separately from target-only decode. Include assistant reads, target verification, acceptance 0 through the requested depth, corrections and rollback. Gemma's shared-cache assistant and Qwen's cache-producing assistant have different ownership; a common reader kernel must not erase that distinction. Keep Qwen DFlash as a regression control, not a new Gemma pairing.

The sampled decode matvecs have much higher occupancy than attention. Optimize them only when updated full-model evidence identifies a material remaining cost. New split/merge strategies that change reduction order belong behind the numerical-contract decision gate.

Exit: exact candidate/verification outputs and cache state, no additional histories or unbudgeted storage, and improved accepted output tokens per second at unchanged draft/accept counts for equivalent runs.

## K7 Compare alternative kernel families

Maintain a compact comparison for each external candidate: exact revision/license, actual SM120 instructions, head dimensions and GQA support, mask/softcap semantics, input/conversion precision, intermediate/reduction order, bounded spans/resume support, graph/lifetime requirements and scratch bytes. Prototype only candidates that pass the applicable numerical and hardware gates.

- [FlashAttention-4](https://arxiv.org/abs/2603.05451) is a source of asynchronous pipeline and attention-dataflow ideas. Its paper reports B200 results and techniques including tensor memory, altered exponential work and conditional rescaling; those results are neither SM120 performance evidence nor native-order equivalence.
- [FlashAttention implementations](https://github.com/Dao-AILab/flash-attention) are comparison references, not an automatic replacement. Check the specific implementation's supported dimensions; D256 support does not establish D512 support.
- [CUTLASS](https://github.com/NVIDIA/cutlass) offers hardware-specific implementation patterns. Use SM120-compatible primitives rather than assuming SM100 instructions are available.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is a comparison source for paged prefill/decode and backend selection. Verify the exact path against this fork's encoded formats, source leases and numerical contract.

An alternative that wins only after changing precision, softmax or reduction order is a useful documented option, not an accepted exact-order optimization. Obtain a decision before proceeding with that option. Do not introduce a new framework dependency merely to benchmark one isolated idea if existing fixtures can exercise it.

## K8 Select and qualify the best variants

### Correctness and lifecycle gate

Use existing tests before adding a new test system. Run focused red/green tests for the changed behavior, then the affected all-64 K/V operator matrix across D256/D512, resident/mixed/wrapped spans, dense/compact masks, affine formats and tails. Include admitted decode/verification widths and representative narrow/wide prefill widths, including 64, 128, 129, 241, 255 and 256.

Check native output bits, checkpoint bytes, canaries, exact/one-byte-short quota behavior, no-write failure paths, source identity, generation changes, cancellation, replay, state save/restore, window wrap and low-high-low phase transitions. Run ASAN/UBSAN and strict CUDA checks without weakening numerical/API reporting or accepting OOM as a pass.

Keep an immutable native operator binary or framed golden outputs/checkpoints for deterministic inputs. Recompiling both the candidate and its native reference from the same changed arithmetic header can hide a numerical-order change; agreement between those two newly built paths is not sufficient. Include signed zero and type-specific rounding behavior in the bitwise oracle.

Use fitting native model controls for full logits/hidden vectors. At long points where native allocation cannot fit, compare to the immutable qualified implementation on identical inputs and state; do not advertise long-context native equivalence. Preserve the recorded native 16 Ki Gemma MTP weight-loading OOM as capacity evidence, not a passed profile.

Qualify both actual GPU-resident targets and their admitted MTP paths, Qwen DFlash, shared-parent boundaries and existing supported consumers touched by common changes. Capability/API portability alone does not qualify another backend, model or multimodal path.

### Performance gate

For every candidate retain a record of the hypothesis, single changed variable, source/binary/input hashes, exactness result, kernel path, resource use, traffic, matched timings and decision. Include conversion, checkpoints and transport in the relevant comparison. A preparatory conversion cannot be excluded simply because a microbenchmark performs it once.

Use ordinary, uninstrumented model runs to establish gains. Use Nsight to verify the mechanism and actual dispatch, not to supply the speedup number. Separate fixed-arena comparisons from maximum-fit comparisons, and fresh prompts from cached prompts. Keep generated IDs, prompt/cache/draft counts and memory layouts aligned where they are invariants of the experiment.

Promote only gains above measured noise with no material correctness, capacity, prefill, decode or MTP regression. Declare the performance/regression decision thresholds before measurement rather than choosing them after seeing results. A small single-pass gain remains provisional until an explicitly approved confirmation demonstrates it. Do not force an arbitrary speedup promise or commit an unbeneficial optimization because its occupancy improved. If genuine prefill/decode tradeoffs remain, retain explicitly scoped alternatives on the performance/memory frontier instead of hiding a loss in a cross-model average.

### Dispatch and delivery gate

Prefer a few geometry/capability-based variants over one model-specific megakernel or an unconstrained autotuning grid. Keep the qualified implementation for nonwinning cases. Publish the selected domain, scratch requirements, observed full-model gains and rejected/no-gain alternatives.

Use incremental affected translation-unit builds and proportionate functional compatibility checks for retained changes, not repeated cold compilation. Keep compilation resource costs visible when choosing between otherwise similar implementations.

GPU jobs require exclusive ownership and must stay within the user's current authorization. The user has explicitly approved committing the qualified implementation and then running the new MTP3 sweep. Drain all owned profiler/server descendants before restoring the existing production service and verify port 1234. Preserve accepted attempts and runtime cohorts on resume. No production configuration/model changes, pushes or agents are authorized.

## Next investigation

Commit the now-qualified, narrowly dispatched feature pipeline, then run the new 32-point Gemma MTP3 sweep using `find-max-arena.py` independently at every capacity. Preserve the selected full-workload response, physical arena, resulting KV pool and actual rejected +1 MiB neighbor. Publish prefill/decode curves and arena differences against the historical 8-120 Ki MTP3 series without attributing every maximum-fit difference solely to the kernel. Exact native/checkpoint controls and broader regression gates have already passed; additional kernel candidates remain separate work.

For Q staging, avoid repeating the rejected global-load/conversion design without a new mechanism. Investigate whether a bounded layout or scheduling change removes shared-memory pressure without moving the cost into reloads, longer live ranges or extra phase storage. Output-feature blocking is a separate possible accumulator experiment: it must preserve full-D512 QK, per-output accumulation and softmax order, not compute two independent D256 attentions.
