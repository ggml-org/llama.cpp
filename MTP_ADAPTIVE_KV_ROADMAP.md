# Adaptive KV Streaming and MTP Integration Roadmap

## Background

This branch already provides a backend-neutral device-memory infrastructure, adaptive KV streaming for a single serial target context, phase-aware sharing between prefill compute workspace and decode KV storage, asynchronous target and attached-MTP K/V publication, cross-token prefetch, and sparse deadline feedback.

The next objective is to reintroduce Qwen3.8 Multi-Token Prediction (MTP) without giving up those properties. The target use case is a 16 GiB GPU running a Qwen3.8-27B target model with a 262144-token context. The target KV cache uses Q8_0 keys and Q4_0 values. The initial supported configuration remains one serial request, one accelerator, Flash Attention, and no parallel slots.

Qwen3.8-27B has 48 recurrent Gated DeltaNet blocks and 16 full-attention blocks. Its MTP head adds one logical full-attention layer. At 262144 tokens, the MTP layer has about 416 MiB of Q8_0/Q4_0 KV state. Reserving that entire layer as a separate permanent allocation would materially reduce target residency and long-context decode speed.

The desired design therefore treats the 16 target attention layers and the MTP attention layer as 17 logical layers sharing one physical KV budget. For each layer, a resident prefix remains in device memory and the nonresident suffix is copied into a shared ring. During an MTP phase, the complete MTP layer is present in device memory as two ordered spans: its resident prefix and its retained ring suffix. The suffix is loaded once, retained across catch-up and sequential draft steps, and released only after the MTP phase completes.

## Goals

- Preserve the current adaptive target-KV streaming behavior and its bounded device-memory budget.
- Support target verification widths from TG1 through TG4.
- Preserve stock attention arithmetic as closely as practical, especially the stock F16 MMA path used by quantized TG3/TG4 attention.
- Avoid a permanent 416 MiB MTP-only device allocation.
- Load the nonresident MTP KV suffix once per MTP phase, not once per draft token.
- Spill recurrent rollback snapshots to pinned host memory so speculative depth does not reserve several full device snapshot planes.
- Share phase-exclusive compute workspace among target decode, MTP catch-up, and MTP drafting.
- Keep policy, ownership, phase transitions, and span descriptions backend-neutral. Backend-specific code should implement execution only.
- Maintain explicit lifetime, generation, publication, and failure boundaries through the common memory infrastructure.

## Initial restrictions

The first production integration is intentionally restricted to the measured Qwen3.8-27B configuration:

- one serial sequence and one accelerator;
- 48 recurrent blocks, 16 target full-attention blocks, and one MTP full-attention block;
- Q8_0 key cache and Q4_0 value cache;
- target verification width of one through four tokens;
- Flash Attention and KV offload enabled;
- no mmproj, parallel requests, multi-GPU split, shared cache, or SWA cache;
- no claim of optimized streamed attention on non-CUDA backends until those adapters are implemented and measured.

The common contracts must not encode these restrictions. Unsupported configurations fail closed at the model integration gate.

## Core execution model

### Logical ownership

The target and MTP contexts keep separate logical cache identities, token frontiers, publication generations, and truncate/commit rules. They may share one physical arena, but they are not one logical cache.

### MTP layer residency

At the representative full-context layout, about 6593 Q8_0/Q4_0 pages are required across 17 attention layers. A feasible partition is approximately 348 resident pages per layer plus 677 shared ring pages. The MTP layer then needs 348 resident pages and 676 retained ring pages. Together those spans cover the complete layer.

The exact values are policy outputs, not constants. The invariant is:

```text
MTP resident prefix + retained MTP ring suffix = complete active MTP KV
```

The retained suffix is a lease on ring slots. Those slots cannot be recycled for target prefetch until the MTP catch-up and draft sequence finishes. Target-layer lookahead can use only genuinely spare ring capacity or capacity created by an explicit repartition.

### Speculative cycle

The intended steady-state cycle is:

```mermaid
flowchart LR
    A[MTP draft tokens] --> B[Target verifies TG1-TG4]
    B --> C[Spill recurrent rollback snapshots as needed]
    C --> D[Load and retain nonresident MTP KV suffix]
    D --> E[MTP catches up on target-evaluated tokens]
    E --> F[Accept or reject drafts]
    F --> G[Commit or truncate MTP logical KV]
    G --> H[Optionally restore selected recurrent state]
    H --> A
```

Acceptance is decided only after the MTP context has caught up to the tokens evaluated by the target. Rejected suffixes are truncated in the MTP logical cache. The retained MTP span lease can remain active for the next draft sequence when its logical identity and publication generation still match.

### Why segmented attention is required

The complete logical KV sequence is not necessarily one contiguous allocation. A layer can be split among resident and ring ranges. Gathering the whole sequence into another contiguous buffer would either exceed the budget or add an avoidable full-layer copy.

The attention backend therefore receives an ordered list of physical spans that exactly cover one logical token interval. The kernel must preserve the stock algorithm's arithmetic order. For TG3/TG4 on the target CUDA configuration, this means extending the stock F16 MMA path to consume spans and perform bounded tile-local quantized conversion, rather than replacing it with a different online-softmax reduction tree.

## Numerical policy

Bit identity is required where the ordinary all-resident stock path is used. Streamed execution should minimize absolute error and preserve stock dispatch and accumulation order wherever possible.

- TG1/TG2 may use the existing stock vector family extended with span addressing.
- TG3/TG4 must use a span-aware extension of the stock F16 MMA family.
- Q8_0/Q4_0 conversion for MMA must be bounded and tile-local. It must preserve the stock conversion and tile traversal order.
- A generic merge of independently normalized per-span attention results is a correctness reference and fallback experiment, not the preferred production arithmetic.
- Every new path is compared against stock logits, recurrent state, and continuation tokens before performance qualification.

## Milestone 7: span-aware attention foundation

Outcome: stock attention families can consume one logical KV sequence represented by multiple retained physical spans without changing cache policy or model semantics.

| Stage | Implementation boundary | Required TDD evidence |
| --- | --- | --- |
| 7.1 | Add a backend-neutral ordered segmented-KV contract. Validate exact logical coverage, nonempty ordered spans, overflow-safe byte ranges, query width, and buffer lifetime ownership. Do not change runtime dispatch. | Pure CPU tests for one/multiple spans, gaps, overlaps, reorder, zero lengths, arithmetic overflow, out-of-range K/V storage, transactional failure, and retained-buffer lifetime. Existing KV geometry tests remain green under Debug, ASan/leak checking, and UBSan. |
| 7.2 | Add an exact CPU/reference evaluator over the segmented contract. It is the semantic oracle, not the optimized production path. | Random and adversarial partitions, tails, masks, TG1-TG4, all-resident equivalence, empty/invalid rejection, and deterministic comparison with contiguous reference attention. |
| 7.3 | Add a retained complete-layer lease that atomically reserves resident and ring spans for one logical layer and prevents ring reuse until retirement. | Delayed consumers, cancellation, generation mismatch, repartition rejection while leased, failure rollback, and release after last execution. |
| 7.4 | Extend the stock vector attention family to address ordered spans for TG1/TG2 while retaining stock arithmetic. | Partition-boundary cases, quantized K/V tails, masks, CUDA memcheck, stock logit/state comparison, and no gather-sized scratch. |
| 7.5 | Extend the stock F16 MMA attention family to consume spans for TG3/TG4 without changing tile or reduction order. | Every boundary position around MMA tiles, TG3/TG4, causal masks, sinks if supported, stock numerical comparison, bounded scratch, and Nsight dispatch confirmation. |
| 7.6 | Add bounded tile-local Q8_0/Q4_0 conversion for span-aware MMA. Avoid full-layer F16 gathering. | Conversion boundary/tail tests, exact converted tile comparison, fixed scratch ceiling independent of context, OOM tests, memcheck, and stock TG3/TG4 error qualification. |
| 7.7 | Route target TG1-TG4 attention through the segmented contract when a layer is streamed; retain ordinary stock dispatch when it is contiguous. | Real Qwen3.8 target inference, all-resident zero-regression control, streamed numerical qualification, 8K-192K performance sweep, H2D telemetry, and explicit fallback/rejection matrix. |
| 7.8 | Admit TG2-TG4 decode through the ring with exact vector/MMA workspace planning and stock Qwen MMA geometry. TG3/TG4 require a complete layer in the ring. | Byte-identical Qwen 64/8 output after session publication, wrapped-ring tests, incomplete-ring rejection, memcheck, and two-layer short/long-context latency. |

Milestone 7 acceptance:

- One immutable plan exactly describes the logical K/V sequence and retains every backing buffer through execution.
- Invalid or stale span plans fail before backend work is submitted.
- Contiguous input retains the stock dispatch and output.
- Streamed TG1-TG4 uses bounded scratch and does not gather the full active cache.
- Quantized TG3/TG4 follows the stock MMA arithmetic structure closely enough to meet the recorded numerical threshold.

## Milestone 8: host-spilled recurrent rollback

Outcome: speculative recurrent rollback depth no longer consumes multiple permanent device-sized state planes.

| Stage | Implementation boundary | Required TDD evidence |
| --- | --- | --- |
| 8.1 | Inventory recurrent state tensors, exact snapshot bytes, mutation points, and rollback semantics for the target model. | Layout and overflow tests, token-depth accounting, and byte-exact stock snapshot comparison. |
| 8.2 | Allocate authoritative rollback snapshots in pinned host memory with explicit sequence/generation identity. | Bounded host allocation, cancellation, stale identity rejection, leak checks, and no device allocation increase. |
| 8.3 | Add a bounded device staging region and asynchronous D2H snapshot publication. | Delayed completion, overwrite prevention, publication ordering, failure injection, and exact host bytes. |
| 8.4 | Restore only the accepted rollback point through asynchronous H2D before target reuse. | Every acceptance length, full rejection, no-rollback fast path, dependency ordering, and recurrent-state equivalence. |
| 8.5 | Overlap snapshot spill/restore with independent target and MTP work where dependencies permit. | Timeline evidence, no extra device-wide synchronization, unchanged outputs, and measured latency impact. |

## Milestone 9: shared target/MTP phase arena

Outcome: target decode, MTP catch-up, and MTP drafting share one stable physical device budget while keeping separate logical executions.

| Stage | Implementation boundary | Required TDD evidence |
| --- | --- | --- |
| 9.1 | Extend the phase vocabulary and requirements to target verify, recurrent spill/restore, MTP catch-up, and MTP draft. | Repeated transitions, cancellation at every boundary, and no per-token replan in a stable phase. |
| 9.2 | Add a session-level owner for target and MTP requirements, leases, executions, and generations. | Independent logical identities, aliased physical grants, delayed work, destruction order, and stale capture rejection. |
| 9.3 | Plan the target TG4 maximum requirement using span-aware attention and rollback staging. | Exact accounting, maximum-width execution, smaller-width reuse, and phase-safe fit boundary. |
| 9.4 | Plan MTP prompt/catch-up and sequential draft requirements without simultaneous target-only workspace. | Catch-up widths one through four, TG1 draft reuse, fixed parent identity, and reclaimed-byte diagnostics. |
| 9.5 | Publish one shared compute workspace to both schedulers with explicit exclusive admission. | No simultaneous alias use, drain before handoff, graph invalidation/rebuild, and recovery after failed activation. |

## Milestone 10: adaptive MTP KV integration

Outcome: the MTP logical cache participates in the same bounded physical KV budget as the target cache.

| Stage | Implementation boundary | Required TDD evidence |
| --- | --- | --- |
| 10.1 | Generalize physical policy from 16 target layers to 17 logical attention layers without merging target and MTP cache identities. | Dynamic geometry, exact total bytes, minimum-ring feasibility, and no hardcoded 16-layer arithmetic. |
| 10.2 | Add separate authoritative host cache, revision, publication frontiers, and truncate/commit operations for MTP. | Accepted/rejected suffixes, cache reuse, cancellation, save/restore, and cross-context identity rejection. |
| 10.3 | Populate authoritative MTP prompt KV through the common writer/publication path without claiming device residency. | Prompt chunking, wide ubatch, exact host contents, delayed completion, an unadvanced device frontier, and restart from host state. |
| 10.4 | Acquire and populate a retained complete-MTP-layer lease from resident plus ring spans. | One H2D load per lease, exact resident/ring device bytes, full logical coverage, ring exclusion, failure rollback, and release/reuse rules. |
| 10.5 | Reconcile target/MTP frontier lag, coordinate target ring admission, and run catch-up over retained spans with TG1-TG4. | All catch-up widths, no overwrite of retained MTP spans, exact frontier advancement, span-aware attention qualification, and no repeated suffix transfer. |
| 10.6 | Run sequential MTP draft tokens while retaining the same complete-layer lease. | Multiple draft tokens, append publication, mutable tail handling, and one-transfer invariant. |
| 10.7 | Integrate acceptance, MTP truncate/commit, and retained-lease invalidation. | Every accepted length, full rejection, target/MTP frontier agreement, cache generation changes, and retry behavior. |
| 10.8 | Add optional target-L1 prefetch only from spare capacity, with deadline feedback. | No eviction of retained MTP spans, miss recovery, disabled-equivalence control, and measured benefit. |

Stage 10.4 checkpoint: lease population is backend-neutral and synchronized before return.

Stage 10.5 infrastructure checkpoint: phase-scoped ring guards mask MTP-owned slots from target prefetch and survive prime/adopt. A populated MTP lease can expose a committed prefix while reserving physical slots up to the target frontier; future bytes remain untouched.

Target resident and session now map cache-local target layers onto validated physical entries; only target layers receive roots, graphs, and publication pairs. CUDA tests cover resident attention, session append, and streamed target prefetch around MTP-held slots.


Stage 10.5 model/context checkpoint: an explicit, default-off target-context parameter
reserves one separate MTP logical host cache and includes it in the combined physical
KV policy and host-memory accounting. The target continues to use its cache-local
layers, while the physical policy has 17 slots. A real Qwen3.8 context test confirms
one target token executes without advancing MTP. Adaptive target repartition
remains enabled.

Stage 10.5 paired-context attachment checkpoint: when the target opts in, a serial
Qwen3.8 MTP context on the same loaded model instance, with compatible device,
context length, K/V types, and attention geometry, retains the target's separate
logical cache identity. Unsupported pairs fail before graph execution. The real
Unsloth UD-IQ4_XS GGUF, loaded once with its embedded MTP head, passes target/draft
attachment and mismatch/legacy controls. At that checkpoint, attachment was
metadata-only: MTP still allocated and executed against stock KV, while the
shared authoritative MTP cache remained at frontier zero. The host-backed
checkpoint below supersedes that allocation path.

Stage 10.5 host-publication checkpoint: the attached MTP graph exposes the final
post-transform K/V rows and publishes them through the common bounded writer to
the separate authoritative host cache. A real UD-IQ4_XS test covers a 256-token
prefill, one-token page-crossing append, byte-exact K/V parity with stock MTP storage, and
suffix removal/republication with generation change and prefix preservation.
The device frontier remains zero. At that checkpoint, the first live bridge
synchronized the producer and used separate bounded writer scratch; stock MTP
device KV and compute workspace had not yet been reclaimed.

Stage 10.5 live catch-up checkpoint: a populated complete-MTP-layer lease
retains the target pool and guards occupied ring slots. Target prefetch honors
that guard; a conflicting adaptive repartition declines admission and succeeds
after release/replan. New MTP K/V rows use tail-only H2D, while TG1-TG4 plans
reuse one reservation. The MTP flash-attention graph consumes those spans through
a buffer-local backend hook and borrows the target's phase-exclusive attention
workspace. CUDA's ratio-six GQA path now uses the same padded eight-column MMA
tile as stock. On the real UD-IQ4_XS model at 257-266 tokens, independent stock
MTP logits match the attached path exactly for TG1, TG2, TG3, and TG4.

Stage 10.6–10.7 live checkpoint: a bounded four-token future reservation
allows sequential MTP drafts beyond the target frontier. A rejected suffix
increments the logical generation, rebases the unchanged resident/ring prefix
without H2D, invalidates old span plans, and transfers only newly rewritten
tail rows. Real UD-IQ4_XS tests cover TG1–TG4 catch-up, two sequential draft
steps, all 0–4 accepted-prefix lengths, interior-hole rejection, and stock
MTP logit parity. Draft-context clear releases the guard; sequence checkpoint
restore reconstructs authoritative host KV from stock K/V in bounded
256-token gathers and checks every restored byte.

Stage 10.8 guarded-prefetch checkpoint: the existing backend-neutral FIFO
prefetch planner skips MTP-held ring slots and starts stable target-layer copies
ahead in genuinely spare slots. CPU planner and CUDA resident/session tests
check lookahead distance, ring bounds, protected sentinel bytes, complete-ring
failure before submission, and safe release/replan. Phase transitions retire
the guard; ordinary target verification retains it until a new KV page or
nonpoisoning adaptive-policy rejection requires release.

Stage 11 early integration checkpoint: `--kv-stream-auxiliary-layers 1` is a
strict, default-off serial/embedded-MTP Q8_0-K/Q4_0-V path. The draft inherits
the target quantization but not a second KV pool/arena. Until live host-spilled
recurrent rollback is attached, it uses the existing full-checkpoint path.
Real server tests passed two serial prompt-cache requests and 96 generated
tokens across a streamed page boundary and a target checkpoint rollback
(65/75 draft tokens accepted in that one synthetic 3K run). A matched
3005-token stock-vs-streamed TG4 control had zero maximum logit error.

The remaining Milestone 10/11 sign-off work is tracked in the four-step
completion table below. The observed 3K server rate is not a native-context
performance claim. Standalone sidecars loaded as a different model instance
still need explicit semantic-compatibility proof.

Native-context fit audit (CUDA, UVM disabled): at `--ctx-size 262144` and a
2,000 MiB physical target arena, draft construction failed allocating its
416 MiB stock MTP KV. At 1,800 MiB, that KV allocated but the separate MTP
prefill compute buffer then requested 1,196.27 MiB and OOMed. The target
prefill arena itself is approximately 1,192.27 MiB graph, 416 MiB attention,
and 191.70 MiB KV. Thus selectively parking MTP KV alone cannot solve startup.
The subsequent live hookup must alias the two serial compute grants within one
phase owner: target/MTP prefill need the maximum of their approximately
1.2 GiB graph workspaces, while target/MTP decode need only their much
smaller decode maxima. In this historical audit, the 416 MiB stock MTP KV
was still a separate persistent cost. This was a design requirement, not a
qualified native-context configuration.

Native-context serial-lifecycle checkpoint (CUDA, UVM disabled): the target
and draft now borrow one 1,800 MiB phase parent at 262,144 context tokens.
The native regression covers target prefill, draft prefill, target decode
growth, draft catch-up, and full sequence removal with a retained MTP ring
guard. An isolated server then completed two different serial requests with
`cache_prompt:true` and 64/96 generated tokens; the second request safely
returned through prefill and decode after the first request's reset. This is
not yet a long-context throughput or sustained-lifecycle qualification.

Host-backed MTP KV checkpoint (CUDA, UVM disabled): attached draft K/V now
bind to the target-owned authoritative host cache, rather than allocating a
separate 416 MiB device KV at 262,144 tokens. Buffer-local write and attention
execution uses bounded target-owned writer/attention workspaces. The 262K
target/draft context test passes, and the real UD-IQ4_XS TG1-TG4, draft,
truncation, and stock-logit comparisons pass. The later nonblocking-publication
checkpoint removes a host-side whole-backend wait but keeps D2H, device staging,
and attention ordered on one backend stream; sustained lifecycle and long-context
throughput qualification remain separate work.

## Milestone 10/11 completion track (2026-09-24)

The four-step completion track is ordered by dependency. Step 1 removes the *separate persistent* MTP device KV allocation; it does not claim that total VRAM falls by exactly 416 MiB, because retained MTP spans and bounded execution scratch still use the shared physical budget.

| Step | Status | Implementation boundary | Required evidence |
| --- | --- | --- | --- |
| 1. Remove stock MTP device KV | **Complete; ready for review** | Bind attached draft K/V to its authoritative host cache and use buffer-local execution hooks with target-owned writer/attention workspace. Preserve the ordinary unattached draft path. | At 262K, attached K/V are host-backed and no second 416 MiB device KV is allocated. Real UD-IQ4_XS prefill, TG1–TG4 catch-up/drafting, truncation, and stock-logit comparisons pass; focused writer, lease, session, model, span, publication, and execution tests pass. |
| 2. Nonblocking MTP host publication | **Implemented; ready for review** | Submit K/V publication without synchronizing the whole producer backend on every write. Retain source tensors, scratch, and cache generation until an explicit backend-neutral completion fence; advance the host frontier only after completion. | Delayed and back-to-back writes, exact host bytes, ordering, cancellation, stale-generation rejection, copy failure, and stock-logit parity. Trace confirms no per-token device-wide producer synchronization. |
| 3. Cancellation and sustained-lifecycle qualification | **Implemented; focused qualification passed** | Make partially submitted writes, attention, lease guards, phase handoffs, rollback, checkpoint restore, and teardown recover or fail closed without stale reuse. | Pending K and K/V cancellation, injected index-read and staged-copy failures, retained-lease teardown, guarded phase-transition tests, checkpoint parity, 24 varied serial requests with prompt-cache resets, CPU ASan/LSan, CUDA memcheck, and bounded post-warmup VRAM use pass. |
| 4. Numerical, memory, and throughput qualification | **Pending** | Sweep the supported configuration from 8K through native context with the maximum safe pool for each context, then compare MTP against target-only adaptive streaming and stock controls. | At least 256 decoded tokens per point; report prefill/decode throughput, MTP acceptance, pool/residency, H2D traffic, peak VRAM, numerical drift, and long-context failure/timeout behavior. |

Step 3's focused target/MTP lifecycle qualification passes. The 24-request UD-IQ4_XS test keeps free VRAM at 1579 MiB after warmup and reports zero maximum stock-logit error. The complete CUDA model, MTP lease, publication, session, and prefetch suites pass; six CPU ASan/LSan suites and focused CUDA memcheck also pass. A stale lease test was updated because full reset now releases a retained MTP lease. Fault injection for every attention-kernel and live server-disconnect boundary is still required by Stage 11.4; these focused tests do not establish production-wide cancellation safety.
Step 4 remains pending. Broader models, quantizations, backends, and separately loaded MTP sidecars remain outside the initial production gate.
MTP host publication now queues K/V conversion and D2H copies on the producer stream,
retains source/scratch ownership until backend-neutral completion events retire, and
advances the host frontier only after those events complete. For streamed decode,
encoded tiles are also staged into protected device spans before attention; the
retained lease adopts the new generation only after host publication. These
operations still share one ordered backend stream, so the trace does not show
copy/attention overlap. CPU/CUDA lease suites cover provisional bytes, delayed frontier, back-to-back append,
cancellation, stale generation, and injected first/second staged-copy failure.
The real UD-IQ4_XS test passes TG1–TG4 and successive draft appends with zero
reported stock-logit error; the 262K attached-MTP memory probe also passes.
A CUDA Nsight API trace of that real-model test recorded no
`cudaDeviceSynchronize` calls. It did record stream synchronizations elsewhere
in the full test, so this result does not establish zero stream-level stalls or
a throughput benefit; those remain part of step 4.

## Milestone 11: end-to-end speculative server integration

Outcome: supported Qwen3.8 MTP requests execute through the adaptive memory system with bounded device memory and measured long-context benefit.

| Stage | Implementation boundary | Required TDD evidence |
| --- | --- | --- |
| 11.1 | Add a strict capability gate and configuration diagnostics for the initial supported setup. | Every accepted/rejected configuration and clear resource estimates. |
| 11.2 | Integrate the complete speculative cycle into llama-server without parallel execution. | Real prompts, serial follow-up requests, prompt cache, cancellation, and context truncation. |
| 11.3 | Coordinate cross-token retained-MTP reuse, target prefetch, recurrent spill/restore, and sparse feedback. | Long decode across repartitions, no stale work, bounded leases, and transfer telemetry. |
| 11.4 | Complete allocation, publication, execution, and recovery failure matrices. | Injected failures at every transition, explicit retry or invalid-session outcome, sanitizers, and sustained leak tests. |
| 11.5 | Qualify numerical behavior, memory, acceptance, and throughput from 8K through native context. | Stock/no-MTP controls, old adaptive-KV comparison, 256+ decoded tokens per point, pool/residency/H2D plots, and acceptance statistics. |

## Milestone 12: generalization

After the initial configuration is correct and useful:

- support other K/V quant pairs through the existing generic geometry/capability model;
- support other Qwen3.8 sizes and differing target/MTP attention geometries;
- add native span-aware execution adapters for ROCm, SYCL, Vulkan, OpenCL, and other accelerators;
- evaluate parallel slots, multiple devices, and split-mode layer placement;
- generalize retained logical-layer leases for paged attention and other cache policies.

## Cross-cutting correctness rules

1. Host KV remains authoritative whenever a device layout can be rebuilt.
2. Logical cache identity, content generation, arena generation, KV layout revision, and executable generation remain distinct.
3. A span plan is immutable after publication and retains all backing storage until the final consumer retires.
4. Repartition cannot reuse leased ring slots.
5. Failure leaves the previously committed logical and physical state valid, or marks the session explicitly invalid.
6. No stage may depend on UVM eviction for correctness or capacity accounting.
7. Common policy and ownership code contains no CUDA stream, event, pointer-attribute, or kernel type.
8. Performance claims require matched model, context, batch/ubatch, K/V type, device budget, prompt, decode length, and UVM mode.

## TDD workflow

Each stage follows the same sequence:

1. Add the smallest behavioral test that fails for the missing contract.
2. Record the expected compile or runtime failure.
3. Implement only the stage boundary.
4. Run focused Debug tests, then relevant existing regressions.
5. Run ASan/leak checking and UBSan for host/common code.
6. Run real-backend tests and memory checking when device execution changes.
7. Run stock numerical comparison before performance measurement.
8. Update this document with actual evidence and limitations.
9. Stage only the reviewed stage files; the user creates the commit.

## Progress ledger

| Stage | Status | Commit | Evidence / limitations |
| --- | --- | --- | --- |
| 7.1 | Complete | bff7edad3 | Ordered exact-coverage plans retain arena leases and support reference-counted asynchronous consumers. The new suite passes 7 cases / 60 assertions in CPU Debug, CUDA, OpenCL, oneAPI SYCL, and Vulkan-ASan builds, plus CPU ASan/leak checking and UBSan. Six focused Debug ownership/KV suites pass. No production dispatch changes. |
| 7.2 | Complete | 82879f1a9 | Scalar host oracle consumes retained plans continuously across physical boundaries, supports GQA, TG1-TG4, explicit/no mask, softcap, F16 and Q8_0/Q4_0, and transactional failure. The suite passes 5 cases / 4,591 assertions in all configured builds and sanitizers. No production dispatch changes. |
| 7.3 | Complete | ffba29bea | One common owner reserves a complete layer's resident prefix and contiguous ring suffix, shares that reservation across query widths, rejects identity changes/repartition while live, and releases slots after final retirement. The suite passes 5 cases / 338 assertions across configured builds and sanitizers. Production prefetch is not connected yet. |
| 7.4 | Complete | c78a5f5ec | CUDA stock vector attention consumes retained ordered spans for TG1/TG2 with one continuous per-thread softmax state, tail-safe loads, global causal-mask coordinates, stock split topology, and bounded query-aware scratch. New and legacy CUDA suites pass with errors below 3.8e-9 and clean memcheck. The plan hook is not routed into production yet. |
| 7.5 | Complete | f32351849 | CUDA TG3/TG4 F16 MMA restores the stock two-stage `cp.async` pipeline and uses the stock affine loader for span-contained tiles; only boundary tiles resolve rows. Production-shaped TG1 maps stock vector partial blocks wholly onto one of at most three page-aligned regions. At 128K, synthetic one-layer overhead is 5.3%/5.0% for two/three-span TG1 and 1.5%/1.6% for TG4. Numerical suites and memcheck pass. |
| 7.6 | Complete | 3287593cc | Q8_0 keys and Q4_0 values use bounded shared-memory conversion with stock-exact FP16 bytes and no context-sized F16 plane. Compressed K/V tiles are prefetched asynchronously into one existing MMA buffer and decoded into the other while the previous phase computes. All 24 span/context cases remain exact; at 128K TG4 is 7.9-10.7% faster than stock with unchanged workspace and clean memcheck. |
| 7.7 | Ready for review | - | Production TG1-TG4 dispatch consumes the physical resident/pre-wrap/post-wrap layout as at most three retained spans; contiguous layers remain on stock direct attention and suffixes larger than the ring retain branch-free aligned TG1 resume. Qwen3.8 continuation tokens match the production control exactly. CUDA, sanitizer, cross-backend contract, no-all-quants, memcheck, and 8K-192K performance qualifications pass. |
| 7.8 | Ready for review | - | TG2 uses aligned stock-split resume and preserves all output bytes across 32-page transfers. TG3/TG4 use the stock 4x8 MMA shape over a complete resident-plus-ring layer, including wrap or zero resident pages. Exact scratch is queried from the backend and incomplete-ring layouts fail before mutation. Qwen 64/8 tests, session append, model arena tests, no-all-quants compile, and memcheck pass. |


### Stage 7.1 implementation and validation

The first test build failed because the segmented-KV source, immutable plan, view, and lifecycle API did not exist. The implementation adds those types to the existing common `ggml-kv-stream` layer rather than creating a backend-specific subsystem. Creation validates a nonempty ordered partition of `[0, active_tokens)`, query width, K/V geometry, alignment, overflow-safe physical extents, and exact buffer bounds before publishing anything. Failure leaves the caller's output handle unchanged.

A successful plan retains every input arena lease, converts the inputs to an immutable buffer-and-offset view for backend execution, and exposes no mutable policy state. The plan itself is reference counted so asynchronous consumers can share it. Releasing the caller's original leases does not permit arena repartition: the test observes both source arenas in `DRAINING` state until the last plan reference is freed, after which both become `QUIESCENT`.

TDD and validation evidence:

- The intentional red build reported all missing span-plan types and functions.
- The focused suite passes 7 cases / 60 assertions, covering one and multiple spans, gaps, overlaps, reordered and empty spans, invalid query widths, alignment, independent K/V bounds, logical and byte overflow, transactional failure, immutable views, retained arena lifetime, and final-consumer reference counting.
- Existing geometry passes 18 cases / 1,205 assertions. The Debug memory-executor, backend-memory, KV binding, and KV session regressions also pass; six selected suites pass in total.
- The span, geometry, and backend-memory suites pass under ASan with leak detection and UBSan.
- The final reference-counted span suite builds and passes unchanged in CPU, CUDA, OpenCL, oneAPI SYCL, and Vulkan-ASan configurations.
- This stage adds no attention kernel, runtime dispatch, cache policy, device transfer, or production-server behavior. Stage 7.2 remains the exact segmented CPU/reference evaluator.

### Stage 7.2 implementation and validation

The first test build failed because the host reference header and segmented-attention evaluator did not exist. The implementation adds a separate common `ggml-kv-stream-reference` component to `ggml-base`. It accepts only a retained 7.1 plan backed by host-readable buffers, packed FP32 queries, an optional packed mask, query-head count, scale, and optional logit softcap. It returns packed FP32 output plus an explicit empty marker for every all-masked query/head row.

The oracle decodes each K/V row through GGML's existing type traits and traverses logical tokens in one continuous order. Physical span boundaries do not start a new softmax, merge partial results, or change accumulation order. Q8_0/Q4_0 and F16 contiguous-versus-segmented controls are therefore bit-identical in the oracle. Double-precision scalar dot, softmax, and weighted-value intermediates make it a semantic qualification tool, not an emulation of a particular optimized kernel's rounding. Unsupported device-only buffers and malformed/nonfinite inputs fail before output publication.

TDD and validation evidence:

- The intentional red build stopped on the missing `ggml-kv-stream-reference.h`. The first implementation build then exposed and corrected one missing direct standard-library include before any runtime test ran.
- The focused suite passes 5 cases / 4,591 assertions. It covers TG1-TG4, random data, one-token and irregular tail partitions, every-token partitions, physically reversed span placement, causal and finite masks, no-mask operation, softcap, GQA, all-masked rows, F32/F16/Q8_0/Q4_0 storage, invalid plans/shapes/buffers/values, and transactional output.
- Arbitrarily segmented F32 results match an independent scalar oracle. F16 and Q8_0/Q4_0 results are exactly equal between one-span and multi-span plans. The F32 segmented oracle also matches ordinary contiguous CPU Flash Attention within 2e-5.
- Five focused Debug suites pass: reference, spans, partial attention, geometry, and backend memory. The reference, span, and partial suites pass under ASan with leak detection and UBSan.
- The final 4,591-assertion suite builds and passes unchanged in CPU, CUDA, OpenCL, oneAPI SYCL, and Vulkan-ASan configurations.
- This stage adds no backend execution hook, attention kernel, cache policy, transfer, or production-server behavior. Stage 7.3 remains the retained complete-layer lease and ring-reuse exclusion boundary.

### Stage 7.3 implementation and validation

The first test build failed because no complete-layer reservation or lease contract existed. The implementation adds a common `llama-kv-stream-layer-lease` owner above the stage 7.1 span plan and the existing policy-derived physical layout. It retains the pool arena lease, validates the complete layout and identity, allocates one nonwrapping contiguous ring run for a layer's nonresident suffix, and creates an immutable resident-plus-ring plan. It performs no KV copy or publication; content population remains stage 10.4.

The physical reservation is independent of query width. A TG4 catch-up plan and TG1 draft plan for the same layer, layout revision, content generation, and active frontier share one reservation and therefore one future suffix upload. Each query-specific handle is reference counted. Ring slots, layout identity, and pool storage remain protected until the final handle reference retires, even after the owner is closed or destroyed. Different-layer reservations cannot reuse occupied slots, and rebind/repartition is rejected while any reservation or handle is live.

TDD and validation evidence:

- The intentional red build stopped on the missing `llama-kv-stream-layer-lease.h`.
- The focused suite passes 5 cases / 338 assertions. It covers resident-plus-ring coverage, policy-derived K/V offsets, numerical attention over a known physical resident/ring split, shared TG4/TG1 reservation, final-reference retirement, insufficient/fragmented capacity, layer/query/revision/generation mismatch, transactional failure, rebind rejection and success, close, delayed consumers, and owner destruction before the consumer.
- Five focused Debug suites pass: layer lease, segmented reference, spans, policy, and backend memory. The layer-lease, reference, and span suites pass under ASan with leak detection and UBSan.
- The final suite builds and passes unchanged in CPU, CUDA, OpenCL, oneAPI SYCL, and Vulkan-ASan configurations. The TSan configuration builds, but its runtime cannot start on this host and exits with `unexpected memory mapping`; no TSan runtime claim is made.
- This owner is not connected to the existing production prefetch scheduler. When MTP integration begins, it must become the single ring-admission authority so ordinary prefetch cannot bypass retained slots. This stage adds no device copy, attention kernel, policy decision, or production-server behavior. Stage 7.4 remains span-aware stock vector attention for TG1/TG2.

### Stage 7.4 implementation and validation

The first CUDA test build failed on the missing query-aware resume layout, TG2 resume state, and plan-consuming backend hook. The implementation extends the existing stock CUDA vector kernel rather than adding a second attention algorithm. Per-thread maximum, normalizer, and weighted-value accumulators are saved for one or two queries and restored across every ordered physical span. Normalization and cross-split reduction occur only after the final span.

Backend hook version 6 accepts the immutable stage 7.1 span plan. The common plan remains backend-neutral; CUDA validates every buffer, byte range, mask/output alias, query width, type pair, and exact workspace before the first launch. Query-aware workspace accounting is tied to the logical token extent. CUDA instantiates TG1/TG2 stock and resumed vector variants per native K/V pair and derives split count from the same ordinary-vector occupancy topology as stock. Arbitrary final tails skip invalid K/V lanes rather than reading padded bytes outside a span.

TDD exposed two correctness details before acceptance. First, resumed-kernel occupancy produced a different split topology from stock, so planning now uses the matching ordinary vector specialization. Second, K/V pointers are span-local but the causal mask remains globally indexed. Advancing the mask by a local offset admitted one extra TG2 token and caused a context-inverse error; the kernel now keeps global mask coordinates, while the legacy per-span adapter reconstructs the original mask base.

TDD and validation evidence:

- The intentional red build reported the missing query dimension, `spans` hook, and query-aware planner ABI.
- The new suite passes 4 cases / 62 assertions. It covers TG1/TG2 workspace accounting, hook discovery, Q8_0/Q4_0 contexts of 257, 513, and 1017 tokens, contiguous and highly fragmented boundaries at 1/127/128/255/256/257 and final tails, causal masks, stock comparison, null/mismatched plans, undersized workspace, and output preservation.
- Across all six TG/context points, segmented-versus-contiguous and segmented-versus-stock maximum absolute error is at most 3.72529e-9, below the explicit 1e-7 partition gate. Byte identity is not claimed.
- The legacy TG1 resume suite passes 4 cases / 110 assertions across Q8_0/Q4_0, Q5_1/Q4_1, and Q4_0/F16 at 513, 769, and 4097 active tokens. Maximum stock error is at most 2.79397e-9.
- The full CUDA block suite passes 20 cases / 656,887 assertions, the session suite 9 / 481, and the model suite 7 / 159. Compute Sanitizer memcheck reports zero errors and zero leaked bytes for both new and legacy vector suites.
- CPU Debug, ASan with leak detection, UBSan, OpenCL, oneAPI SYCL, and Vulkan-ASan builds pass the common query-aware layout and host-only contract tests.
- Existing production workspace planning explicitly requests TG1, so its behavior and allocation remain unchanged. The new `spans` hook is not yet called by production inference. The production container was restored with its existing configuration and `/health` reports `ok`. Stage 7.5 remains the span-aware stock F16 MMA path for TG3/TG4.

### Stage 7.5 implementation and validation

The intentional red test changed the immutable-plan backend contract to request TG3/TG4 execution and exact workspace sizing. It failed because version 6 exposed only the resumed vector family. Version 7 adds a bounded `spans_workspace` query and one private generic CUDA span descriptor shared by the vector and MMA execution families. The common span plan remains unchanged and backend-neutral.

The CUDA implementation refactors the ordinary F16 MMA kernel body behind contiguous and span-aware wrappers. Span-contained K/V tiles resolve their physical interval once and delegate directly to the stock affine loader. Tiles crossing a physical boundary use tail-safe row resolution. Both paths retain the stock two-stage `cp.async` schedule, shared-memory layout, `ncols1=4`/`ncols2=2` MMA traversal, Stream-K block selection, partial metadata, and uniform/general fixup kernels. Physical span boundaries therefore do not create independently normalized MMA results or alter its reduction tree.

The production TG1 fast path takes advantage of the physical layout invariant: resident, pre-wrap ring, and post-wrap ring form at most three regions with 128-token-aligned starts. One launch assigns each stock vector partial block wholly to one region, runs the unchanged affine stock-vector body, and combines all partials through the stock combine kernel. An exact final tail is copied into one bounded zero-padded 128-token staging record and may become an internal fourth descriptor; TG2 and nonaligned interior boundaries retain the stage 7.4 resumable fallback. This keeps the general immutable-span contract correct without imposing its most conservative path on page-aligned production layouts.

Workspace contains only span descriptors plus the existing stock partial/fixup state; it never contains a full gathered layer. Descriptor publication is asynchronous on the execution stream. The ordinary all-resident wrappers retain their public launch ABI and arithmetic path; stage 7.7 remains responsible for bypassing segmented dispatch entirely when the logical cache is already contiguous.

TDD and validation evidence:

- The initial CUDA build failed on the missing version-7 workspace hook. Restoring two-stage MMA first exposed the dynamic shared-memory attribute requirement; matching stock attribute setup fixed occupancy planning before any kernel launch.
- The focused suite passes 7 cases / 253 assertions. It covers TG1/TG2 Q8_0/Q4_0, page-aligned one/two/wrapped-three-span TG1, TG3/TG4 F16, exact tails, and adversarial cuts at 1, 63/64/65, 127/128, 255/256/257, and final boundaries under causal masks.
- Page-aligned and exact-tail TG1 differs from stock by at most 2.79397e-9. Tail-safe/resumable TG1/TG2 differs by at most 4.65661e-9. All tested TG3/TG4 contiguous, two-span, wrapped-three-span, and adversarial fragmented outputs are exactly equal to ordinary stock F16 MMA.
- Both contiguous and heavily fragmented MMA workspace sizes remain below the corresponding full K/V gather size. Nsight Systems confirms the span-aware MMA plus stock Stream-K fixup dispatch and no gather kernel.
- The opt-in one-layer benchmark uses warmed repeated executions at 8K, 32K, 64K, and 128K. At 128K, resident-plus-ring overhead is 5.3% for TG1 and 1.5% for TG4; adding a ring-wrap region yields 5.0% and 1.6%. At 128K plus one token, bounded tail handling measures 9.2%/11.7% for two/three-region TG1 and approximately 12% for the current F16 TG4 boundary path. At 32K, production-shaped results remain within approximately 10% for both families. A 64K two-span TG1 point remains an observed scheduling discontinuity at 14.3%, while its wrapped-three-span case is 4.4%. Short-context percentages are dominated by fixed launch/planning cost; production should retain stock dispatch before streaming is necessary.
- Compute Sanitizer memcheck reports zero errors. The legacy TG1 resume suite passes 4 cases / 110 assertions, and the CUDA block suite passes 20 cases / 656,887 assertions across all native/fallback quant pairs.
- CPU Debug, ASan with leak detection, UBSan, OpenCL, oneAPI SYCL, and Vulkan-ASan builds pass the unchanged host contract. The CUDA backend also compiles with `GGML_CUDA_FA_ALL_QUANTS` disabled.
- The optimized vector path currently requires TG1 and 128-token-aligned physical starts; it handles one exact final tail through bounded staging, while other vector shapes fail over to the resumable implementation. The MMA executor still rejects sinks, softcap, ALiBi, other head geometries, multi-sequence batches, non-F16 spans, and widths outside TG3/TG4 before launch. Stage 7.6 supplies bounded Q8_0/Q4_0 tile conversion, and production routing remains stage 7.7.

### Stage 7.6 implementation and validation

The intentional red suite added Q8_0/Q4_0 TG3/TG4 execution over one, two, wrapped-three, and adversarial span layouts. All 24 executions were rejected by the version-7 F16-only gate while every Stage 7.5 control remained green.

Version 8 adds typed K/V storage parameters to the span-aware MMA specialization. F16 spans retain their asynchronous affine loader. Q8_0 keys and Q4_0 values preserve the same stock arithmetic while loading paired quant bytes and storing packed `half2` results. Q4_0 deliberately preserves stock FP32 multiply-then-add order before FP16 rounding; algebraically folding the offset initially caused 35 of 1,792 half values to differ and was rejected by the byte test.

The final multi-stage path reuses the two existing shared MMA buffers asymmetrically: one holds the asynchronously prefetched compressed K or V tile while the other holds the decoded F16 tile consumed by MMA. During KQ, V is prefetched; during VKQ, the next K tile is prefetched. No extra shared allocation, global conversion plane, arithmetic regrouping, or softmax-order change is introduced.

No full-cache conversion allocation is introduced. The only device workspace remains the descriptor table and stock Stream-K fixup metadata already planned in Stage 7.5. Conversion storage is the existing bounded shared MMA tile. A private version-8 row-conversion hook invokes the same device helper for direct byte qualification and reusable backend testing; it allocates nothing and remains outside the public llama API.

TDD and validation evidence:

- The focused CUDA suite passes 11 cases / 446 assertions. It covers TG3/TG4, contexts 257/513/1017, causal tails, one/two/wrapped-three/adversarial spans, exact final boundaries, and all Stage 7.4/7.5 controls.
- The tile-local Q8_0 and Q4_0 converters are byte-identical to stock for all 1,792 tested FP16 values per type. All 24 quantized span-attention comparisons have maximum absolute error exactly zero against ordinary stock attention.
- Quantized MMA workspace is 266,624 bytes in the focused matrix and 582,656 bytes at 8K, below a fixed 1 MiB qualification ceiling and far below the 16 MiB full F16 K/V gather. An exact-size-minus-one workspace is rejected before output mutation.
- Compute Sanitizer memcheck reports zero errors. The legacy resume suite passes 4 cases / 110 assertions and the full CUDA block/type-pair suite passes 20 cases / 656,887 assertions.
- CPU Debug, ASan with leak detection, UBSan, OpenCL, oneAPI SYCL, and Vulkan-ASan retain the backend-neutral host contract. The CUDA path remains gated to Q8_0/Q4_0 or F16/F16 with the measured Qwen3.8 geometry.
- Nsight isolated the rejected synchronous prototype: 93.47% of scheduler cycles had no eligible warp, 64.4% of sampled stalls were long-scoreboard waits, 84% of global sectors and 58% of shared wavefronts were excessive, while SM and DRAM throughput were only 5.61% and 4.69%. Paired loads/stores improved 128K by about 13%; restoring compressed `cp.async` overlap reduced TG4 from 2.35-2.41 ms to 0.73-0.75 ms.
- Relative to stock at 128K, one/two/three-span TG4 is 10.7%/8.4%/8.2% faster. At 64K it ranges from 2.2% faster to 0.7% slower; at 32K it costs 7.3-11.0%. The 8K fixed cost is 60-65%, so Stage 7.7 must route contiguous caches to ordinary stock execution and qualify the streamed real model. Production dispatch remains unchanged.

### Stage 7.7 implementation and validation

Production decode now constructs one immutable complete-layer plan directly from the resident prefix and the submitted ring ranges. The binding publishes its borrowed arena lease to the native factory, the resident owner retains that lease, and every asynchronous span plan remains retained until the complete layer sequence drains. Copy-slot retirement records a completion event after attention on the compute stream; a ring slot cannot be overwritten merely because its H2D transfer completed.

The optimized route is deliberately narrow. An all-resident layer continues through ordinary stock attention. A streamed production layer is represented by at most three page-aligned ranges: resident prefix, ring before wrap, and ring after wrap. Streamed Q8_0/Q4_0 TG1-TG4 and F16/F16 TG3/TG4 use that fixed-span executor whenever the complete suffix fits the ring. If the suffix exceeds ring capacity, TG1 retains the resumable copy/compute pipeline but selects a separate compile-time branch-free kernel for 256-token-aligned waves; arbitrary-tail TG1/TG2 remains a correctness fallback, not the production hot path. Zero-resident layouts, unsupported types or geometry, missing hooks, insufficient workspace, and non-covering plans fail over or fail closed rather than partially publishing output. The CUDA validator checks Q, mask, output, workspace, and every retained physical span independently; it no longer requires a fictitious contiguous K/V prototype allocation.

TDD and validation evidence:

- The production-shaped CUDA route covers 64 query heads, 8 KV heads, Q8_0/Q4_0 storage, TG1-TG4, 769/1025 active-token controls, and resident-plus-pre-wrap-plus-post-wrap layouts. The final resume suite passes 6 cases / 217 assertions; the span executor passes 11 / 476, resident 13 / 273, session 9 / 481, copy 23 / 3,211, and binding 16 / 3,109.
- All-resident graph execution remains on stock attention for query widths 1, 2, 3, 4, 8, and 33. Streamed production-geometry comparisons remain within 7.45058e-9 absolute error; TG3/TG4 quantized span tests remain exactly equal to stock.
- A real Qwen3.8-27B IQ4_XS run at 131072 prompt tokens produced the same 32/32 deterministic continuation token IDs as the production implementation. It measured 24.96 tok/s decode versus 23.20 tok/s for that production control while using the same 2688 MiB phase arena.
- The 8K-192K, 8K-step sweep showed no material pre-streaming regression. Relative to the preceding memory-infrastructure implementation, decode was within -1.91% to +0.15% through 112K and improved by 4.50%, 2.87%, and 0.45% at 120K, 128K, and 136K. Its long-context regression triggered the matched causal investigation below.
- A final 144K fallback A/B used an independently built clean `3287593cc` worktree with identical model, context, arena, batch, UVM, and request. Clean HEAD measured 10.6679 tok/s and Stage 7.7 measured 10.6700 tok/s with identical 256-token output, demonstrating that Stage 7.7 itself does not further regress fallback decode.
- A second matched causal A/B used the same current driver, model, 147456 context, 147200-token prompt, 256-token decode, 2688 MiB arena, 256/256 batch, resident-399/ring-209 layout, and 1,144.83 MiB H2D per token. The exact pre-Stage-7 boundary `ffba29bea` measured 23.57 tok/s while the Stage 7.4-derived path measured 10.67 tok/s; pinned H2D measured 49.22 GB/s. The regression came from applying arbitrary-tail guards and fragmented resume topology to the aligned production case.
- The final fixed-span build uses 106 H2D calls rather than 394 for the same 1,144.83 MiB and reaches 21.41 tok/s, recovering 90.8% of the 23.57 tok/s baseline and meeting the 5-10% overhead target within measurement rounding. The new 161-page-suffix test proves a suffix larger than the rejected cutoff still executes as one fixed-span attention launch. Aligned resumable Q8_0/Q4_0, Q5_1/Q4_1, and Q4_0/F16 controls are bit-exact with stock; arbitrary-tail TG1/TG2 tests retain their prior error bounds.
- Nsight Systems at 144K attributes 79.1% of the decode window to kernels and 52.3% to H2D, with 67.7% copy/compute overlap, 96.0% combined GPU activity, 25.13 GB/s averaged over the whole token window, and 48.02 GB/s while DMA is active. The apparent 50-60% H2D utilization is therefore exposed-copy duty cycle, not an underfilled PCIe burst; raw pinned H2D is 49.22 GB/s on this host.
- Cross-token feedback now advances with successful owner-session publication instead of being reset by every content generation. A primed sequence evaluates feedback before adoption and falls back transactionally when a new layout is required. Opt-in `LLAMA_KV_STREAM_TRACE_POLICY=1` logging exposes accepted sample deltas, misses, copy duty, peak slots, and proposed layout transitions without affecting default execution.
- Version 9 samples one complete immutable layer before any acquire waits. At 160K the controller converges from resident-395/ring-273 to resident-387/ring-401 and then reports 16/16 layer hits, but steady decode remains about 17.5 tok/s. This proves current-layer readiness is necessary but not sufficient to hide transfer time; ring growth beyond the zero-miss point is rejected as added traffic without throughput benefit.
- A matched stable ring-401 Nsight window measures 40.3 ms/token of kernels, 36.1 ms/token of H2D, but only 18.7 ms/token of overlap (52.0%). Historical phase arena measures 42.4/35.2/34.0 ms respectively (96.6% overlap). The current kernels are faster and active H2D remains 48.0 GB/s; approximately 17.3 ms/token is copy-only scheduling time.
- Version 10 assigns one retained ready fence and one retained final-consumer fence to each transfer batch while preserving per-slot logical ownership and partial-consumer fallback. Direct tests prove a two-page production batch uses one record/wait at each boundary, while partial consumers safely share readiness and retain individual consumed fences. The focused CUDA copy suite passes 25 cases / 3,260 assertions. The 160K fixed-span A/B is throughput-neutral (16.97 versus 16.95 tok/s), so this is retained as a complexity/dependency reduction rather than claimed as a speedup.
- Progressive TG1 reuses the stock 17 interleaved split states and final combine across streamed ranges. Transport and execution granularities are separate: each prefetch request and ready/consumed fence remains bounded to 32 pages, while up to three physically contiguous ready requests share one resumed-attention launch. A ring wrap ends the group rather than widening or copying across it. The production-shaped CUDA test proves 3 + 4 K/V DMA batches across two layers, 2 + 3 attention launches, and zero error against stock.
- The earlier opt-in A/B measured 19.48/19.69/20.08 tok/s for coupled 32/64/96-page copy/resume ceilings. With strict 32-page DMA and 96-page resume grouping enabled in production, the matched 160K IQ4_XS run reaches 20.66 tok/s over 256 decode tokens, versus approximately 17.42 tok/s for fixed-span execution, while retaining the 2,688 MiB arena and 1,602.33 MiB H2D per token. Measured H2D utilization is 69.4%.
- The zero-resident shrink test first exposed a wrapped-layer error of 0.00216693 on the complete-span route. Making zero residency an explicit resumable fallback restored the established 1.6e-5-class test error and all 481 session assertions.
- Compute Sanitizer reports zero errors. The CUDA backend compiles with `GGML_CUDA_FA_ALL_QUANTS` disabled. Binding and host-contract tests pass under ASan/leak checking, UBSan, OpenCL, oneAPI SYCL/Level Zero, and Vulkan-ASan.
- The remaining optimized production gate is intentionally limited to the documented single-sequence Qwen3.8 geometry. Other types, sinks, bias/softcap, multi-sequence batches, and unqualified head shapes retain existing paths; broader backend execution remains later work.

### Stage 7.8 implementation and validation

TG2 uses the page-aligned stock vector specialization, retaining each stock split's softmax state across at most 32-page DMA requests. A resume launch may consume up to three contiguous ready requests without widening their copy or fence ownership. TG3/TG4 use the stock 4x8 F16 MMA shape for Qwen's 64 query heads and 8 KV heads. The backend reports the exact maximum scratch requirement for three physical spans; the session checks that the complete streamed layer fits in the ring before changing the layout.

The production-shaped 25,601-token and 163,840-token tests compare the public FP32 attention output byte for byte with stock Q8_0/Q4_0 attention. TG2, TG3, and TG4 match for both layers at both lengths, including a wrapped ring layer. Session tests also match stock after appending and publishing new KV rows. A fixed ring too small to contain a full TG4 layer is rejected without changing the committed token frontier or layout revision.

| Active tokens | TG | Previous partial path (ms) | Exact native path (ms) | H2D calls |
| ---: | ---: | ---: | ---: | ---: |
| 25,601 | 2 | 3.515 | 4.264 | 14 |
| 25,601 | 3 | 3.875 | 3.600 | 14 |
| 25,601 | 4 | 4.780 | 3.583 | 14 |
| 163,840 | 2 | 13.693 | 8.790 | 32 |
| 163,840 | 3 | 20.351 | 10.729 | 32 |
| 163,840 | 4 | 27.321 | 11.065 | 32 |

These are uncontended medians of 12 measured two-layer evaluations after three warmups, not full-model token rates. The previous partial path has different arithmetic, so stock byte identity is qualified separately. TG2 is slower at the forced-streaming 25K point; the actual target normally stays all-resident there. The 160K two-layer speed gains are 35.8%, 47.3%, and 59.5% for TG2, TG3, and TG4 respectively.

The focused resume suite passes 9 cases / 311 assertions, including zero-resident complete-ring TG3/TG4 and a transactional MMA workspace query; session 13 / 603; model 7 / 159; span executor 11 / 476; copy 25 / 3,260. Compute Sanitizer reports zero errors. The CUDA target also compiles with `GGML_CUDA_FA_ALL_QUANTS=OFF`. TG3/TG4 multi-wave attention when one complete layer cannot fit the ring remains unsupported; the session fails closed. The MTP speculative cycle and its end-to-end token-rate benefit remain later milestones.

### Milestones 8-9 implementation checkpoint (2026-09-24)

The experimental `LLAMA_RS_HOST_SPILL=1` path is isolated from the default server configuration. `LLAMA_RS_STAGED_PUBLICATION=1` additionally selects the bounded device-stage publication path on CPU or CUDA; other backends fail closed until they add an adapter.

- Stage 8.1: an exact type/row/depth layout accounts for stock recurrent R/S device planes, one current device plane, host snapshots, and transient stage bytes. Overflow and invalid-quant geometry are tested.
- Stages 8.2-8.4: a versioned host bank supports CPU-host and CUDA-pinned allocations, asynchronous snapshot capture, mixed-generation and pristine-zero restore, cancellation, and identity checks. The staged path copies each R/S rollback snapshot to a backend-selected device ring, then uses a backend-local execution-buffer marker to queue D2H on an independent copy backend. No scheduler eval callback or whole-graph synchronization is used. Normal inference defers selected-state H2D until graph execution; host checkpoints read the selected snapshot directly, while device-native checkpoints materialize it before writing.
- Stage 8.3: a checked stage-ring layout allocates two device-local TG4 slots using 18.70 MiB for the measured 48-layer Qwen geometry. Per-part producer-ready events and per-slot D2H-consumed events prevent early reuse. CPU and CUDA tests verify exact bytes across three recurrent layers sharing two slots, partial-capture cancellation/retry, and CUDA memcheck with zero errors.
- Stage 8.4: tiny Qwen35 CPU/CUDA tests compare host snapshots and logits with stock, exercise rollback depths zero through eight, and verify host/device checkpoint replay. A generated three-recurrent-layer fixture covers actual graph slot reuse with TG4 depth. CPU ASan with leak detection passes. Full IQ4_XS on the 5070 Ti under UVM passes every TG4 rollback depth and checkpoint subcase against stock.
- Stage 8.5: a CUDA handoff test runs an MTP-side graph on the target's former arena before the target's staged D2H publication completes. Nsight Systems on the full IQ4_XS test records 4.15 GB of recurrent D2H traffic; 72.2% of copy time overlaps GPU kernels versus 38.9% for a matched direct-host-spill control. Active D2H throughput is 47.9 versus 43.9 GB/s. This is transfer overlap evidence, not a demonstrated end-to-end token-rate improvement.
- Stages 9.1–9.2: a backend-neutral work-phase tracker and serial session owner track independent target/MTP cache identities, admission, transition recovery, and stale-generation rejection.
- Stages 9.3–9.4, infrastructure only: an aligned byte planner takes measured TG1–TG4 target verification and MTP catch-up graphs plus *additional* staging bytes, chooses the peak shared-parent grant, and reports phase-specific reclaimed bytes. Width admission rejects a request wider than its planned grant. This planner is not yet fed by both live model contexts.
- Stage 9.5, infrastructure scope: two real CPU schedulers and two CUDA schedulers alternately compute on one physical arena, with one active lease and equal workspace addresses. The CUDA test retires native captures before release and proves that an independent staged D2H copy remains valid across handoff. The actual target/MTP contexts are not yet constructed under this owner; that bridge depends on MTP KV integration and remains part of Milestones 10-11.

The existing KV-streaming speculative gate remains in place. Do not claim MTP-on-adaptive-KV inference or a long-context token-rate speedup from these tests. Milestone 9's generic workspace contract is validated; real-context construction and MTP KV ownership remain for Milestones 10-11. Before enabling the path by default, benchmark verification and draft throughput at several context lengths, qualify cancellation under real server requests, and add adapters for other backends.

## MTP complete-layer prefill admission fix (2026-09-26)

Root cause: uniform strict-prefill placement does not enter the decode overlap-sizing path and has no decode deadline feedback. The startup ring can therefore stay much smaller than one MTP suffix. Complete-layer acquisition then rejects the retained lease, and the former stock-attention fallback has an incompatible bounded output allocation for narrow managed attention.

The backend-neutral policy now reserves one requested physical layer through its future-token frontier. If the current placement cannot provide the configured overlap headroom, it demotes resident pages within the existing pool and uses a uniform layout. The session drains old copies and captures before installing that layout; fixed or physically impossible budgets reject admission without changing the policy or cache frontiers. MTP acquisition invokes this admission before building or populating its retained TG1-TG4 plans. Admission failure returns an explicit error instead of silently using stock attention.

Wide strict prefill gathering remains unchanged. This fix concerns retained narrow catch-up and drafting; it does not alter attention arithmetic, add a new device allocation, or optimize lease renewal and recurrent rollback.

Validation:

- Pure policy tests cover the original 250-resident/12-ring, 17-layer failure, idempotence, future-tail page crossings, concentrated placement, fixed-ring rejection, impossible budgets, and exhaustive small-pool page conservation. Release and ASan/leak-check runs pass 32 cases and 144,897 assertions.
- The grown-ring CUDA regression checks the unchanged pool allocation and buffer identity, TG1-TG4 covering plans, release/reacquisition, and stock attention comparisons. TG1/TG2 maximum absolute errors are 3.72529e-9 and 1.86265e-9; TG3/TG4 are exact. CUDA memcheck reports zero errors.
- All CUDA model tests pass (11 cases / 388 assertions), including impossible-budget rejection with unchanged layout and frontiers. All session tests pass (15 cases / 669 assertions). The pool-growth session test now waits before reading asynchronous device output and compares against the host oracle only after the complete append publishes. It additionally checks stock CUDA outputs at 1e-5 tolerance; all four grown-pool stock comparisons are exact.
- Real UD-IQ4_XS requests at 96K with 1, 2, and 3 draft tokens complete 32-token continuations without the stock fallback. Prefill admission demotes resident pages from 250 to 241 and grows the ring from 12 to 165 slots inside the same 2,360 MiB parent.
- The exact original 98,044-token prompt also completes a 258-token continuation with one draft token. At the former failure boundary, target=98044 and reserved=98045, the prefill ring grows from 12 to 165 slots without stock fallback.
- A 192K check with the same arena reaches a separate CUDA graph-instantiation OOM before long prefill. With a 2,240 MiB arena, the 192K three-draft request completes 32 generated tokens; prefill admission changes resident pages 147->105 and ring slots 20->734. This is a correctness check, not a performance comparison.

Performance work remains deferred: capacity-based MTP lease reuse, rollback traffic, and host-fence reduction are separate from this correctness fix.

## MTP correctness investigation (2026-09-26)

The 8K IQ4_XS sweep sends the same 7,932-token article prefix and greedy sampling settings to every draft length. Each response contains exactly 256 token IDs, and their detokenized text matches the SSE content. Repeated no-MTP requests are identical. The original adaptive path first differs at generated token 17 for one draft, 108 for two drafts, and 30 for three drafts.

Two comparisons must remain separate:

1. Stock vector TG1/TG2 versus stock MMA TG3/TG4 is not a bit-equivalence contract. Quantized vector attention quantizes Q to Q8_1, while MMA uses FP16 inputs and different reductions. Recurrent snapshot versus full-checkpoint/replay modes also change execution and batching.
2. Adaptive versus stock at the same query width, token prefix, and rollback mode should not introduce the additional TG2 error found here.

The teacher-forced target test uses the real article, Q8_0 K/Q4_0 V, 8K capacity, 256/256 batches, separate stock/adaptive contexts, and typed recurrent snapshots. The GGUF has 24 query heads and four KV heads, not the 64/8 geometry used in earlier qualification fixtures. It tests identical TG2 inputs plus checkpoint restore and replay. Prefill is exact; the guarded span/resume path introduces a few-ULP attention-output difference that propagates into recurrent state and logits. In this fixed-input reproduction, the maximum later logit difference reaches 1.70069 and the recurrent-state difference reaches 2.26504.

CUDA graph disabling, explicit stream synchronization, and CUDA_LAUNCH_BLOCKING do not remove the error. Nsight records identical native/resumed vector grids (1,17,24) and blocks (32,4,1), ruling out different split counts.

The span executor currently selects the tail-safe TG2 resumable specialization even for completely page-aligned regions. Selecting only the existing aligned specialization inside that same executor restores exact stock logits and recurrent state in the reproduction. Buffers, masks, launch grids, checkpoint handling, and asynchronous behavior are unchanged. This is the actionable dispatch gap; the exact compiler-level rounding instruction has not been isolated.

Additional controls show that full checkpoint rollback alone does not reproduce the one-draft error in stock. With native target attention and matching full-checkpoint mode, adaptive and stock two/three-draft controls produce identical 64-token continuations, although they differ from no-MTP at tokens 40/30. Those higher-draft differences must not be attributed solely to KV streaming.

The strict opt-in regression reproduced 14 failures before the correction and now passes with exact stock logits and recurrent state:

```sh
build-device-memory-infra-cuda-release/bin/test-kv-stream-context --model MODEL.gguf --mtp-target-drift PREFILL.txt
```

Correction applied: the vector span executor selects the existing aligned resumable specialization only when the complete extent and every region start/length are multiples of 256 tokens. Partial regions retain guarded handling. The policy, allocation budget, transfer scheduling, and backend interface are unchanged.

Validation: CUDA span tests pass 13 cases / 549 assertions; resume 9 / 311; model 11 / 388; session 15 / 669; the real-model regression 2 / 1,334. New 24/4-head coverage checks exact contiguous, resident-plus-streamed, and physically wrapped TG2 regions, plus guarded partial tails and irregular cuts. Focused GQA6 CUDA memcheck passes 2 cases / 73 assertions with zero errors.

An 8K matched-mode comparison uses the same 7,932-token prompt, 256 generated tokens, 256/256 batches, no UVM, and Q8_0/Q4_0 for target and draft KV. Stock is forced to n_rs_seq=0 for this diagnostic comparison only; the temporary override is removed afterward. Complete token arrays and timings are retained in benchmarks/results/tg2-fixed-matched-stock-20260926/results.jsonl.

| Max draft length | Adaptive decode tok/s | Stock full-checkpoint decode tok/s | Output / acceptance |
| ---: | ---: | ---: | --- |
| 1 | 43.05 | 45.68 | 256/256 tokens identical; 101/127 accepted |
| 2 | 48.01 | 50.18 | 256/256 tokens identical; 129/180 accepted |
| 3 | 60.51 | 62.31 | 256/256 tokens identical; 160/210 accepted |

These are single-run correctness comparisons, not a full performance sweep. Adaptive prefill is about 1,596-1,598 tok/s versus 1,692-1,696 for stock, consistent with the retained strict-prefill gathering path. The no-MTP adaptive baseline measures 46.62 decode tok/s. Different outputs across draft lengths remain possible in stock's full-checkpoint mode; matching the execution mode is required before classifying them as streaming regressions.

A post-fix 96K real-article streaming smoke test with three drafts completes a 32-token continuation without OOM or admission failure. This is a lifecycle/capacity check, not a long-context stock-equivalence comparison. The test helper now initializes every query head, including the actual 24-head fixture. Temporary stock-mode instrumentation is removed; no production configuration or container state is changed.

## MTP optimization 1: suffix-aware target rollback (2026-09-26)

The 8K-72K sweep has not entered target KV streaming. Nsight nevertheless shows context-sized target uploads after rejected drafts: target suffix truncation called full content invalidation, marking all resident K/V rows dirty. Private truncation also rebuilt its session and prefill scratch, although the retained physical prefix had not moved.

The content owner now supports suffix invalidation. It advances the content generation to reject stale write tickets, marks the discarded suffix including padding dirty, and preserves existing prefix dirty marks and mirror identity. Target truncation drains pending work, reconstructs the logical publication frontier in the existing session, and retains the physical pool and scratch. No CUDA attention arithmetic, ring policy, recurrent checkpoint format, or MTP lease-renewal policy changes. Unknown external writes, full restores, and physical rebindings still invalidate the whole mirror. No additional MTP or target device allocation is introduced.

TDD and correctness qualification:

- The content regression first failed to compile because the suffix API did not exist. The model regression then failed against the original allocation-dependent truncation. Both pass after the implementation.
- Content tests cover zero/full suffixes, bitmap and page boundaries, dirty prefixes, padded rows, stale writes, invalid/reentrant requests, and 81 encoded K/V combinations. CPU ASan with leak checking passes 17 cases / 193,520 assertions.
- Resident tests cover rollback at tokens 255/256/257, bounded tail upload bytes, unchanged binding identity, TG1-TG4 attention, clean retry, and mandatory full refresh after unknown mutation. CPU ASan passes 15 cases / 275 assertions; CUDA passes 16 / 435.
- CUDA model tests pass 12 cases / 418 assertions. They verify truncation succeeds while device allocation is disabled, stable grants, suffix-only refresh, and streamed append/replay with changed replacement KV bytes. CUDA memcheck reports zero errors. Session tests pass 15 / 669 in isolation.
- The real IQ4_XS teacher-forced stock comparison passes TG1-TG4, including checkpoint restore/replay, with exact logits and recurrent state. The four runs pass 2,244 / 1,334 / 989 / 882 assertions respectively.

Paired Nsight measurements use a 32K context, the same 32,508-token article prefix, 256 generated tokens, a 2,240 MiB arena, 256/256 batches, Q8_0/Q4_0 KV, and no UVM. These are single profiled runs, not a new full sweep; timings include profiler overhead. Decode windows are estimated from the final GPU kernel and reported decode duration.

| Max draft length | Before tok/s | After tok/s | Decode H2D before/after | Acceptance before/after |
| ---: | ---: | ---: | --- | --- |
| 0 | 40.17 | 40.12 | 0.90 / 0.90 GiB | disabled |
| 1 | 35.91 | 37.74 | 29.66 / 11.88 GiB | 105 / 127 in both |
| 3 | 41.77 | 46.48 | 43.16 / 12.44 GiB | 144 / 223 in both |

The 34 MiB target-K prefix upload bursts fall from 336 to zero for one draft, and from 592 to zero for three drafts. The MTP decode pool remains exactly 2,227.868164 MiB. Recurrent checkpoint D2H traffic is unchanged, as expected. Trace artifacts are in benchmarks/results/mtp-optimization-profile-20260926 and benchmarks/results/mtp-suffix-optimization-profile-20260926, with adjacent .nsys-rep and .sqlite files.

Next optimization remains capacity-based MTP lease reuse and tail-only population, followed by recurrent checkpoint transfer optimization. Any proposal that increases the MTP device footprint requires user review before implementation.

## MTP optimization 2: resident reuse with phase-local ring ownership (2026-09-27)

The agreed policy keeps MTP as the 17th physical KV layer. Its resident prefix may survive across rounds, but its ring suffix must not shorten target streaming workspace. The target handoff now drains the draft and releases its MTP lease before main-model execution. Ring contents are never assumed to survive that handoff.

Reacquisition validates a non-owning stamp containing the pool buffer, arena generation, and layout revision. A new session or changed physical identity resets the auxiliary content mirror. With an unchanged placement, population flushes only dirty resident rows, but always uploads the ring suffix into its new reservation. The stamp retains no device lease or ring slot. Allocation replacement and full host restore require a refresh; logical suffix truncation preserves accepted resident rows and invalidates the rejected tail.

Within one MTP phase, catch-up and sequential predictions keep the existing complete-layer reservation. Host tail copies and completed D2D staging acknowledge only their resident rows; attention plans are refreshed under the new content generation. No additional page is reserved and the future-token limit remains unchanged. Population now reserves and installs the ring guard before writing KV, which drains target cross-token prefetch first. A failed acquisition clears its guard so the next attempt cannot remain blocked.

The implementation uses the existing backend-neutral content bitmap, buffer tensor-copy APIs, layer owner, and session guard. It introduces no accelerator kernel or backend-interface change. No additional device allocation is made.

TDD and qualification:

- The zero-copy reacquisition and accepted-prefix dirty-state tests failed against the preceding implementation. The ring-reuse test initially failed to compile without the opt-in resident-population contract.
- CPU and CUDA tests overwrite released ring bytes, reacquire the layer, verify suffix-only transfer counts, and compare the restored bytes. Tests cover changed resident rows, failed copy/retry, zero-copy clean reacquisition, full restore, allocation replacement, and guarded admission cleanup.
- CPU ASan/leak checking passes the MTP lease suite (13 cases / 179 assertions) and logical cache suite (8 / 171). CUDA model tests pass 12 / 427; the MTP lease suite passes 13 / 175. Both CUDA suites pass memcheck with zero errors. Session tests pass 15 / 669 and context-memory tests pass 7 / 425.
- Real IQ4_XS TG1-TG4 teacher-forced checks retain exact stock logits and recurrent state through verification and rollback/replay (2,244 / 1,334 / 989 / 882 assertions). The live shared-parent probe verifies that the next target decode has no MTP lease (2 cases / 28 assertions). Serial-request/prompt-cache checks pass 2 / 647.

Initial 32K Nsight measurements compare against optimization 1, with identical article prefix, 256 generated tokens, Q8_0/Q4_0 KV, 256/256 batches, a 2,240 MiB arena, and no UVM. These are single profiled measurements; the subsequent guard-order correction is qualified separately by final-code numerical and streaming tests.

| Max draft length | Optimization 1 tok/s | Resident-reuse tok/s | Decode H2D before/after | Acceptance |
| ---: | ---: | ---: | --- | --- |
| 0 | 40.12 | 40.20 | 0.90 / 0.90 GiB | disabled |
| 1 | 37.74 | 38.92 | 11.88 / 4.34 GiB | 105 / 127 in both |
| 3 | 46.49 | 47.65 | 12.44 / 6.78 GiB | 144 / 223 in both |

Full-prefix copy counting must exclude 16 target phase-initialization copies: the earlier 166/129 counts included those copies. The corresponding MTP-only populations are 150/113 before this optimization and one cold population in each new 32K trace. Recurrent checkpoint D2H traffic is unchanged. The effective MTP decode pool stays at 2,227.868164 MiB. Traces are under benchmarks/results/mtp-resident-reuse-profile-20260927.

Final-code 96K streaming checks use the same 98,044-token prompt and generate 256 tokens. Against the existing sweep reference, max draft 1 improves from 28.39 to 29.87 tok/s and max draft 3 from 43.65 to 46.37 tok/s. Acceptance is unchanged at 114/127 and 154/212; first/last output snippets match. These are reference comparisons, not repeated paired A/B trials or full-token-array equivalence checks. Pool size is unchanged at 2,227.368164 MiB, with 318 resident pages/layer and 76 ring slots in both versions. Artifacts are under benchmarks/results/mtp-resident-reuse-96k-20260927.

Next: recurrent checkpoint transfer optimization. Increasing device staging, snapshot storage, or reserved MTP ring space still requires user review before implementation.

## Default host-spilled recurrent rollback for attached MTP (2026-09-27)

Serial Qwen3.8 MTP launches with `--kv-stream-auxiliary-layers 1` and `--spec-draft-n-max` from 1 to 3 now use bounded recurrent rollback automatically. No enabling flag is needed. The server derives the depth from the draft length, allocates host-spilled snapshots, and enables the two-slot publication stage. The attached MTP draft context does not allocate a second snapshot bank. `--no-kv-stream-rs-rollback` restores the earlier full target checkpoint/replay path for A/B testing. Unsupported depths fail before decoding. Direct library callers set `llama_context_params.n_rs_seq` to the draft depth on the target; zero keeps full checkpoints because the library cannot infer a draft length without the server's speculative settings.

Example (replace `MODEL.gguf` with the actual model path):

```sh
build-device-memory-infra-cuda-release/bin/llama-server \
  --model MODEL.gguf --ctx-size 262144 --parallel 1 \
  --batch-size 256 --ubatch-size 256 --n-gpu-layers 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q4_0 \
  --kv-stream-arena-mib 2240 --kv-stream-auxiliary-layers 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --fit off --no-mmproj
```

The host-spilled mode uses approximately 149.6 MiB of pinned snapshots per draft depth and 6.2 MiB of GPU staging per depth, beyond the existing 149.62 MiB GPU recurrent state. At depth three this is approximately 448.9 MiB pinned system RAM and an 18.70 MiB GPU publication stage. Compared with full-checkpoint mode, the measured decode KV pool is about 3 MiB smaller per draft depth inside the unchanged 2,240 MiB phase arena. No new MTP KV ring reservation is made. At full context the GPU had only about 64 MiB unallocated during the test, so other GPU applications may force UVM weight eviction or cause OOM if managed-memory eviction cannot satisfy them. The measurements below used UVM disabled; this mode is not a promise of concurrent ffmpeg capacity.

Single-run IQ4_XS, Q8_0 K/Q4_0 V, 256-token continuation comparisons against the preceding full-checkpoint sweep:

| Context | Full checkpoint decode | Bounded rollback decode | Full checkpoint prefill | Bounded rollback prefill |
| ---: | ---: | ---: | ---: | ---: |
| 8 Ki | 62.68 tok/s | 77.43 tok/s | 1,598.3 tok/s | 1,589.0 tok/s |
| 144 Ki | 39.43 tok/s | 43.72 tok/s | 672.3 tok/s | 670.6 tok/s |
| 240 Ki | 16.34 tok/s | 20.68 tok/s | 449.0 tok/s | 448.1 tok/s |
| 256 Ki | no full-checkpoint result | 16.72 tok/s | no full-checkpoint result | 434.5 tok/s |

The rollback mode changes the recurrent execution schedule and can change generated text and draft acceptance compared with full checkpoints. A matched-mode 8 Ki stock/streamed comparison produced all 256 identical token IDs. TG1-TG4 teacher-forced comparisons are stock-exact in rollback mode, including tested restore/replay cases. A 24-cycle serial-request/prompt-cache test passes. An 8 Ki sweep tested all default draft depths 1-3: decode throughput was 64.33, 75.80, and 77.17 tok/s respectively. The full 262144-token-context request completed without OOM and kept the physical GPU allocation stable during prefill. More varied prompts, failure cases, and peak-memory tests are still warranted before enabling this mode in a multi-user production server.
