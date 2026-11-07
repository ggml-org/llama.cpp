# Backend grammar sampling: next optimization round

Date: 2026-11-07. Status: design proposal, based on source inspection; no new builds, tests, profiles or performance measurements.

**1. Recommendation and inspected baseline**

Implement in this order: comparable benchmark and tracing; genuinely lazy proposal-mode grammar preparation; small input-transfer batches and lifetime-safe retained-logit copies; reusable device-side rejection sampling; bounded GPU grammar execution. Keep the CPU grammar implementation as the semantic reference and exact fallback.

The inspected checkout is /devel/tools/llama.cpp at 8a5f7b069386a32899224b4c07193b7acc984d21 with substantial pre-existing tracked and untracked changes. HEAD alone does not identify the implementation. File locations below describe the inspected working tree. The locally available origin/master is c55b09488; it was inspected with git show and was not fetched. This is a pinned compatibility reference, not a claim about current upstream master.

Two supplied KB campaign records were read: 8849fd4d-50f4-41b9-9568-16e2465ef5c3 and 81922b4f-0b6f-4b9d-89b4-e009cc9c2d80. Their prior test and timing results are historical evidence, not validation of this proposal. Current source already contains packed masks, host-state caching, row-prefix preparation, resident mask slots, asynchronous grammar uploads and shared output readiness. Do not reimplement these or attribute their benefits to the next round.

Three corrections to the assignment's baseline matter:

- The standalone benchmark already gives CPU and GPU the same synthetic logits and grammar, but it measures grammar-first forced rejection exclusively. CPU candidate initialization is timed; GPU logit mutation is outside its timer.
- The current common/server integration already supports GPU proposal selection followed by CPU one-token grammar validation when grammar_first is false. However, its prepare hook still constructs full masks before decode. On rejection, it can mask retained logits on the GPU, download the full masked row, and execute the sampling chain on the CPU.
- The reported 26 synchronization calls, split into 14 scheduler copies and 12 buffer setters, were not independently reproduced or tied to a trace here. The source confirms these synchronization mechanisms, but exact counts and time attribution remain a profiling prerequisite.

**2. Shared semantic contract**

Keep two explicit policies: grammar-first and proposal-then-validate. Compare CPU and GPU within the same policy.

These policies are not generally distribution-equivalent. For a fixed sampler state with no proposal-dependent apply-time effects, let q be the unconstrained sampling-chain distribution, A the allowed tokens, and r the result of applying grammar before the complete chain. Proposal-then-validate produces, for an allowed token t:

P(t) = q(t) + q(not A) * r(t).

For positive proposal acceptance probability, this equals r only when r is q conditioned on A; if acceptance probability is zero, every proposal retries and the result is r. Top-k, top-p and other transformations can violate the conditioning relationship. Stateful apply-time effects require accounting for the retry distribution conditioned on the rejected proposal as well. Preserve the existing CPU policy; do not silently switch policy to improve timings.

Masking and validation do not commit grammar state. Rejected proposals do not enter accepted history. Commit the final selected token once, at the existing accept boundary. Preserve the chain's proposal/retry RNG consumption and any apply-time effects. A valid one-token check must use the current filtering semantics, not merely test whether accepting the token on a clone succeeds: mixed TOKEN and character handling has separate filtering and acceptance paths.

All implementations must preserve partial UTF-8, duplicate token pieces, EOG, empty pieces, token-ID terminals, lazy triggers, candidate-ID mapping, explicit row prefixes and reset/clone/copy behavior. Distinguish an empty allowed set from an unsupported optimization. Neither permits unconstrained output.

**3. Goal 1: a comparable standalone benchmark**

Architecture: one fixture reader and trace driver, with separately compiled CPU, upstream-backend and fork-backend adapters. Do not link binaries against a different revision's private structs. CMake compile probes select available sampler hooks; runtime capability checks identify the actually supported chain.

Use public CPU sampler operations for the portable reference. Build the backend proposal graph only from supported non-grammar samplers. The pinned master common implementation disables backend sampling with grammar, so its standalone adapter must orchestrate proposal selection and CPU validation outside common_sampler. This measures a harness-supported path, not an upstream server feature.

For grammar-first master GPU results, explicitly select a portability adapter: compute the CPU grammar mask, apply it to a host logit row, upload that row and use the supported GPU sampling chain. This is a hybrid baseline. Do not label it native GPU grammar or quietly substitute it for the fork's TOKEN_MASK path. If an adapter cannot implement a requested chain or policy, report unsupported with its reason.

Maintain these comparisons:

| Comparison | CPU arm | GPU arm | Purpose |
|---|---|---|---|
| Grammar-first | Full CPU grammar filter, then chain | Same full filter semantics, backend masking/selection where supported | Isolate grammar-first offload |
| Proposal policy | Chain, one-token check, full-mask retry on rejection | Same policy with backend proposal and supported retry | Representative common/server policy |
| Mask component | Exact CPU mask | Proposed GPU mask | Differential correctness and component cost |
| Native integration | Actual common/server path | Actual common/server path | Detect preparation, copy and lifecycle costs missed by the standalone driver |

Do not present cross-policy ratios as hardware speedups.

Fixture format:

- Store exact GBNF text, root, grammar mode, trigger configuration, vocabulary/token-piece identity hash, EOG metadata, sampling-chain order and parameters.
- Store F32 logits, token IDs, accepted-prefix tokens, reset boundaries and optional proposal/retry random variates. Include format version and checksums.
- Store model/tokenizer identity and capture provenance for real traces. Loading a different vocabulary must fail.
- Replay a fixed accepted prefix for every compared row. Selection mismatches are recorded; they must not silently send later rows down different grammar states.
- Preserve a separate autonomous replay mode that commits selected tokens and checks complete continuations. Fixed-prefix replay is a workload control, not evidence of free-running equivalence.

Use two complementary workloads.

First, a deterministic greedy control with unique maxima. For each state, choose an allowed continuation token and, where one exists, a forbidden token. A saved bitmap selects which rows put the forbidden token above the allowed token. Sweep realized rejection rates 0%, 1%, 10%, 50% and 100% over the same grammar-state sequence. Both policies receive identical rows. States with no forbidden token cannot be forced to reject and must be reported separately. The 100% case preserves the existing stress test. Synthetic logits are explicitly labeled.

Second, capture model logits before grammar and sampling transformations, then replay real tool-call/JSON traces without modifying them to force rejection. Use the same saved proposal and retry variates where injection is supported. A matching seed alone does not guarantee matching CPU/GPU draws or reductions. Report actual rejection rate, token mismatches and probability differences. Use greedy replay for strict token equality and stochastic cases for sampling-policy validation.

Cover small literals/enumerations, the existing tools fixture, recursive JSON, long strings with partial UTF-8, token-ID terminals and lazy triggers. Record rule elements, active-stack count/depth, allowed-token count, trie work, fallback count and cache hit rate. Grammar byte count alone is inadequate. Report cold program/trie creation separately from warmed recurring states; do not warm only one arm or let oracle checks populate the measured cache.

Timing contract:

| Field | Boundary |
|---|---|
| input_materialization_cpu | Build/copy CPU candidates from the saved row |
| input_upload_cpu / input_upload_gpu | Enqueue and device transfer of replay logits |
| row_prepare_cpu | Prefix validation/replay, state identity and row setup |
| mask_prep_cpu | Cache lookup/key construction, trie, exact fallback and packing, with nested counters |
| mask_upload_cpu / mask_upload_gpu | Mask descriptors/payload submission and device transfer |
| mask_eval_gpu | GPU grammar traversal/cache lookup, when implemented |
| token_mask_gpu | Applying validity to logits |
| selection_cpu / selection_gpu | Chain transforms and final selection, split into proposal/retry |
| validate_cpu | One-token grammar validation |
| readback_gpu / wait_cpu | Device output copy and host completion wait |
| accept_cpu / accept_gpu | Final state advancement |
| total_wall | Input ready in the declared memory domain through committed output |

Use host monotonic clocks for CPU work and blocked waits, stream timing events or native tracing for device intervals. Collect event results after the existing completion boundary; do not insert a device synchronization around every component. CPU waits overlap device execution, so component durations are not additive. Report wall time, CPU busy/blocked time and device intervals separately.

Provide both host-input and device-input timing scopes. The host-input scope includes both arms' row materialization and GPU upload. The device-input scope starts with model-style device-resident logits for GPU and reports that residency assumption; neither scope can hide preparation costs. Report setup, warmup and graph construction separately, and include any preparation performed before the old sample timer in the whole-step result.

Output JSONL with revision, dirty-source manifest hash, harness hash, fixture hashes, backend/device/build configuration, policy, supported path, chain, warm/cold mode, rejection counts, transfers/bytes, allocation counts, sync call counts, component times, total latency and correctness results. Report per-run distributions and paired comparisons, not only the ratio of aggregate means. A diagnostic instrumented run and an ordinary timing run should be separate if instrumentation changes scheduling.

Code locations: tools/sampler-bench/sampler-bench.cpp:72 (trace), :108 (grammar-first chain), :120 (backend graph), :179 (timing); tools/sampler-bench/CMakeLists.txt; common/sampling.cpp:781; tests/test-backend-sampler.cpp.

Complexity: medium. Risk: low for the fixture/greedy driver, medium for portable backend adapters and stochastic instrumentation. Expected impact: trustworthy attribution and rejection-dependent break-even curves; no direct production speedup.

**4. Goal 3: finish sample-and-validate offload**

This is the first production optimization. Full GPU mask generation is a later Goal 2 dependency; a CPU-generated mask with GPU application/selection is useful independently.

Current path:

1. common_sampler_grammar_adapter_prepare forwards every row to native grammar preparation.
2. Native preparation computes masks even when the adapter's mask flag is false.
3. The adapter builds grammar bindings but restores unmasked logits for the proposal graph.
4. common_sampler_sample reads the backend-selected token and validates a one-element candidate.
5. Rejection enters set_logits_full; apply_grammar creates a temporary mask graph, reads back a vocabulary-sized row, and apply_chain_full samples on the host.

Proposed path:

1. Separate row-state preparation from mask materialization. Proposal mode records owned row/prefix identity and checks invalid prefixes without enumerating the vocabulary. A mask has an explicit not-materialized state; this must not be confused with ALL, NONE or a cache miss.
2. Preserve raw logits on device, before any in-place chain transforms, in storage whose lifetime extends through validation/retry.
3. Execute the supported non-grammar chain on the GPU. Read back token plus status, and any explicitly requested reporting data.
4. Validate on CPU against the exact row state. If valid, return for the normal accept call. No full mask, descriptor upload or TOKEN_MASK execution is needed on this arm.
5. If rejected, materialize that row's full mask. Initially use the current exact CPU generator and upload cache; later use Goal 2.
6. Run a reusable grammar-first retry graph over retained raw logits, with the complete supported chain. Read back the final token/status. Do not rerun model evaluation.
7. If the chain has an unsupported CPU suffix, retain the explicit hybrid/CPU fallback. Do not advertise full GPU retry for partial offload.

The first implementation should handle one committed output row per sequence and retain existing behavior for speculative/multi-output rows. After that, extend lazy preparation using existing prefix types. A row with a valid prefix can still have no valid next vocabulary token; defer that determination until selected-token validation fails and a full mask is needed. Preserve explicit terminal/error status rather than assuming prefix-valid means token-available.

Own row-state snapshots until all dependent validations and retries complete. Existing common code advances the authoritative grammar as selected tokens are accepted; teacher-forced/tree rows need their supplied state, not whichever committed state exists later. Invalid/EOG prefixes remain unusable. Speculative rejection invalidates descendants; neither preparing nor sampling a draft commits grammar history.

Retain existing RNG transaction infrastructure in src/llama-sampler.cpp. Proposal and retry are distinct draws under the CPU policy. Late grammar-first policy replacement has different replay behavior. Preserve generated/committed draw accounting, copy-state semantics and stateful sampler behavior; initially gate chains whose effects cannot be faithfully replayed. Do not consume a discarded proposal twice or advance penalties for rejected candidates.

Build retry graphs once per backend, vocabulary, chain configuration and row capacity. Bind changing mask/state inputs each time. Keep a separate retry sampler execution context where necessary so graph bindings are not destroyed by cloning/copying state. Rebuild on configuration or allocator changes. Reuse existing retained-logit storage and sampler interfaces before adding public APIs.

Expected impact is rejection-dependent. Let M be eager mask preparation, U upload, K masking, S proposal selection, V one-token validation, R readback/wait, and D extra retention/transaction cost. With full-mask retry cost F:

T_proposal ~= S + R + V + D + p_reject * F.

Grammar-first is approximately M + U + K + S + R. This comparison describes latency only; the policies remain distinct. Against today's proposal mode, the direct opportunity is removing eager M and reducing the rejection arm's host readback/CPU-chain cost. Count CPU validation, raw-logit retention and clone overhead before claiming a win. At high rejection rates the extra proposal and round trip can lose. Do not automatically switch semantics based on rejection rate.

Code locations: common/sampling.cpp:147, :158, :190 (adapter), :287 (set_logits_full), :364 (RAW output request), :781 (selection and retry); src/llama-sampler.cpp:2991 (prepared rows), :1488/:1507 (RNG transactions/replay); src/llama-context.cpp:1156 (retain_logits), :1199 (apply_grammar), :2194 (preparation), :2591 (output copies). Use symbol names as the stable anchors.

Complexity: medium for lazy committed-row masks; medium/high for cached GPU retry; high for speculative extension. Main risks: policy drift, stale raw logits, incorrect row identity, double acceptance and RNG drift.

**5. Goal 2: GPU grammar state machine**

A general GBNF grammar is not a small finite-state DFA. The existing implementation maintains a set of alternative pushdown stacks. Port that meaning with bounded resources; do not replace it with one stack or enumerate all possible recursive states.

Immutable GPU program:

- Lower parsed elements into pointer-free opcode/value arrays, rule offsets, alternative starts and character-range tables. Use checked 32-bit indices only when all sizes fit.
- Precompute local structural metadata and validated epsilon-transition information on CPU. Do not attempt to pre-expand unbounded recursion.
- Upload token-piece byte offsets/lengths, special/EOG classification and a shared vocabulary trie once per vocabulary/backend.
- Give each immutable program a generation identity; do not use host addresses in device cache keys.
- Leave parsing, validation, memory sizing and allocation on CPU.

Dynamic state consists of program identity, sequence/row generation, partial UTF-8 accumulator and remaining-byte count, plus a canonical set of stack frame sequences. Each frame is a program offset. Use bounded global-memory arenas for stack frames/frontiers; shared memory may cache small hot portions. Kernel-local arrays must not silently truncate deep or ambiguous states.

Implement a bounded worklist for epsilon expansion and character transitions. Fork alternatives, push return locations, pop completed frames, and deduplicate complete resulting stacks. One warp/block per independent state or subtree is a starting point, not a claimed optimum. Preallocate all scratch. Return OK, NEED_CPU or NO_VALID_TOKEN distinctly; capacity exhaustion is not a valid empty mask.

Trie traversal:

1. Upload the existing byte trie initially, or lower it into contiguous child ranges while preserving terminal lists.
2. Traverse a frontier of (trie node, grammar-state set, partial UTF-8 state).
3. Share transition work for common token prefixes. A transition may cross several grammar rules within one token piece.
4. Emit bits for every terminal token ID at a valid node, including duplicate pieces. Process EOG under the CPU empty-stack semantics.
5. Split into subtrees or tiled frontier passes when one state provides insufficient parallelism.
6. Publish the mask only if the entire traversal completes within bounds. Any overflow or unsupported transition discards the partial result and requests exact CPU evaluation.

The current CPU trie explicitly falls back for TOKEN/TOKEN_NOT. Keep these grammars on CPU initially. Porting them later requires token-ID-aware boundary semantics and differential checks against both filtering and acceptance; byte strings alone cannot represent them.

Device-resident cache:

Use a small fixed-capacity set-associative cache with byte and entry limits. Key by immutable grammar/vocabulary identity, active state, normalized partial UTF-8 and the complete canonical stack set. Hash for lookup, then compare the full key; hash collisions cannot authorize tokens. A sequence generation checks handle freshness but must not prevent reuse of an equivalent semantic state.

Store packed bitsets first. At vocabulary size V, each mask costs 4 * ceil(V/32) bytes; V = 131072 costs 16 KiB, so 128 masks cost 2 MiB before keys/scratch. Avoid full dense transition tables over all states and tokens. Cache only completed masks or independently verified transitions.

Publish entries atomically after key/payload completion. Pin slots while consumers are in flight; use stream/event ordering for eviction and cross-stream access. Full cache means recompute or CPU fallback, never overwrite a live mask. Start with one owning stream per grammar execution context to reduce concurrency risk.

State ownership and acceptance:

First ship GPU mask computation from an uploaded CPU-authoritative state snapshot. This moves traversal/cache lookup but retains CPU acceptance and snapshot transfer; name that limitation.

Then add device advancement for committed tokens. Keep a CPU shadow advanced only by committed tokens for fallback and validation. Branches use independent snapshots or a persistent frame arena with explicit parent ownership. Acceptance is committed once after the host chooses the final token. Graph reset preserves accepted state; sampler reset clears it. Program/vocabulary replacement invalidates dependent caches.

When GPU computation fails, read its status, compute the exact CPU mask from the same row snapshot, upload it and rerun masking/selection. A downstream graph must be gated by status or safely discarded before any selected token can escape. Partial device results cannot be treated as constraints. CPU shadowing avoids expensive reconstruction on each overflow, but it means the state machine has not become CPU-free.

Feasibility:

| Feature | Initial treatment |
|---|---|
| Active character grammars, ranges, bounded ambiguity | GPU worklist/trie candidate |
| Partial UTF-8 and duplicate pieces | Required for supported GPU character path |
| Recursive grammars within arena limits | Supported with checked work/depth limits |
| Deep/large ambiguous states, frontier/cache-key overflow | Exact CPU fallback |
| TOKEN/TOKEN_NOT and mixed terminals | CPU until dedicated semantic port |
| Lazy regex triggers and buffered suffix replay | CPU; upload activated state afterward |
| llguidance | Existing provider-specific CPU/backend hooks; separate implementation |
| Tree speculation and rollback | Later phase using explicit row-state ownership |
| Unsupported backend or memory allocation failure | Existing CPU path |

Code locations: src/llama-grammar.h:130 (cache), :158 (state), :243 (bounded trie contract); src/llama-grammar.cpp:893 (stack expansion), :1168 (exact rejection), :1481 (canonical cache key), :1524 (trie), :1687 (exact mask), :1732 (lookup/fallback), :1819 (accept); src/llama-vocab.cpp:4375 (trie construction); src/llama-sampler.cpp:2908 and :3105 (mask graph/bindings); ggml/src/ggml-cuda/token-mask.cu.

Proposed new GPU interpreter kernels should be backend-private initially, with explicit tensor inputs, bounded scratch and status outputs. Keep TOKEN_MASK as the separate logits operation. Use a small isolated prototype before committing to a new cross-backend ggml operation or public API.

Complexity: high; a production general interpreter is a multi-stage research effort. Risks: state explosion, divergent execution, global-memory traffic, incorrect UTF-8/terminal behavior, cache collisions/lifetimes and fallback synchronization. Expected impact is strongest for frequent full masks, large grammars and multiple independent rows. With rare rejection after Goal 3, GPU interpreter work is paid only on the rejection arm, so its total benefit may be small. A single serial state may be faster on CPU.

**6. Goal 4: reduce synchronization with explicit ownership**

First collect a per-decode trace containing reason, tensor name, byte count, backend/stream, split, row, generation and blocked duration for each wait. Separate scheduler entry points from nested backend setters so the same physical synchronization is not counted twice. Include host staging delays and device idle gaps. The target metric is critical-path latency, not a lower API-call count alone.

| Current site | Proposed action | Boundary that must remain |
|---|---|---|
| ggml_backend_sched_copy_input, ggml/src/ggml-backend.cpp:2082 | Group compatible user-input transfers for a split/backend/copy slot; wait for the slot once, then batch copies | User data must be copied or owned before the caller may overwrite it |
| Async-copy fallback, same file :2114 | Use owned staging plus backend async H2D where supported | Source production and destination reuse dependencies |
| CUDA/HIP buffer setter, ggml/src/ggml-cuda/ggml-cuda.cu:831 | Use existing ggml_backend_tensor_set_n for a compatible group | Completion before synchronous setter returns |
| Batched setter, same file :867 | Reuse its one final wait per group; no new generic batch API needed | Backend/device grouping and ordered writes |
| Graph input setters, src/llama-graph.cpp:136, :262, :1464 | Collect uploads into an owned staging batch; persist local vector data until flush | Do not defer pointers to destroyed local vectors |
| Distribution uniform setter, src/llama-sampler.cpp:1409 | Pack active-row uniforms into owned staging and batch uploads | Preserve per-row RNG draw order |
| Grammar uploads, src/llama-sampler.cpp:3105 | Keep current cache skipping; add staging rings if reuse waits matter | Existing upload event protects host staging; device slot lifetime also needs protection |
| Retained raw logits, src/llama-context.cpp:1156 | Enqueue D2D copy on producing backend stream and retain destination through retry | Copy must precede allocator/source reuse; cross-stream consumers wait on an event |
| Output getters, src/llama-context.cpp:994 | Preserve outputs_ready sharing; batch token/status readbacks | CPU validation requires completion of the selected-token copy |
| Graph reuse, src/llama-context.cpp:1783 | Retain global guard initially; later replace with per-slot dependencies | Prior kernels may still read inputs being rewritten |

The scheduler's user-input branch explicitly performs an immediate copy to protect caller ownership. Simply removing its wait changes the API contract. The CUDA/HIP async tensor-copy implementation at :2891 accepts device-to-device copies and rejects CPU-to-device pairs, which then use the fallback. Solve this with scheduler-owned host staging and backend-aware asynchronous setters, not by assuming the existing D2D hook covers host inputs.

Start with synchronous batching. If all 12 reported setter waits form one compatible independent batch, they could become one wait, saving up to 11 calls. The 14 scheduler-copy waits may contain destination-reuse waits and nested setter waits; their saving cannot be added until the trace proves they are disjoint. Multiple backends, slots and interleaved compute require multiple groups.

Then use a bounded pinned host staging ring and matching device input slots. For each slot: wait only if its prior use is unfinished, fill owned host storage, enqueue H2D on the consuming stream, execute consumers, and record completion. Separate host-buffer upload completion from device-buffer final-consumer completion. Keep a synchronous fallback for backends without usable async/event support.

An Async API name does not guarantee nonblocking pageable-memory transfers. Use backend-appropriate pinned allocations and measure actual blocking behavior. See the [CUDA synchronization contract](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) and [HIP memory API](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html).

Do not replace per-thread-stream synchronous copies with unordered backend-stream work. Encode producer, copy, consumer and reuse dependencies explicitly. Avoid introducing separate copy streams until a same-stream implementation is correct and measured.

For retained logits, same-stream D2D ordering can remove the current host wait only if the next graph cannot reuse the source before the queued copy completes. Temporary view metadata, storage allocation, graph replay and backend changes need their own lifetime checks. A pointer alias to the model output alone is insufficient.

Goal 3 still needs one host completion boundary to validate each proposal batch and another for rejected rows. Batched independent sequences can share each boundary; dependent speculative rows cannot all be accepted at once. Goal 2 could later validate on device and reduce round trips, but that is a separate extension.

Complexity: low/medium for batched synchronous setters; high for scheduler-wide staging/reuse changes. Risk: use-after-free, stale inputs and cross-stream races. Expected benefit depends on measured blocked time and removed launch gaps. Empty waits may be cheap; waits covering useful model work are not all removable latency.

**7. Implementation order and acceptance gates**

| Order | Deliverable | Complexity / risk | Proceed when |
|---|---|---|---|
| 1 | Portable replay benchmark, policy labels and sync attribution | Medium / low-medium | CPU/GPU see identical fixtures and matched-policy results |
| 2 | Lazy mask preparation for committed proposal rows | Medium / medium | Accepted proposals perform no full-mask work; fallback preserves behavior |
| 3 | Small transfer batches and ordered retained-logit copies | Medium / medium | Trace proves eliminated waits and lifetime checks pass |
| 4 | Reusable GPU mask-plus-chain retry | Medium-high / high | Retry uses raw logits, correct draws and no unnecessary full-row readback |
| 5 | Extend row/transaction handling to speculative and multi-sequence paths | High / high | Prefix identity, rollback and repeated sampling remain correct |
| 6 | Bounded GPU character-grammar prototype and device cache | High / high | Exact masks match CPU and fallback is reliable |
| 7 | Device state advancement and broader grammar coverage | Very high / high | Profiling after earlier changes still justifies the complexity |

The numbering is recommended implementation order, not a claim that all later stages should ship. Reassess the GPU interpreter after lazy masking changes the workload. Broad scheduler or new interpreter work needs a separate reviewed design before implementation.

Validation should extend the existing grammar, backend-sampler and backend-op suites. Cover mask equality across complete vocabularies, allowed-logit preservation, EOG and empty sets, invalid/partial UTF-8, duplicate pieces, token terminals, trigger activation, forced capacity exhaustion, cache hash collisions, reset/clone/copy, stale slots and teardown with transfers in flight. Add acceptance/retry RNG checks, late policy overrides, interleaved sequences, prefix divergence and speculative rejection. Report unsupported/skipped backend cases explicitly.

For performance, compare matched policies over the rejection sweep and real traces, with warm/cold results separated. Capture full generation latency as well as model eval throughput: grammar preparation can occur outside existing model timing.

Every future end-to-end benchmark must retain the standard matrix:

llama-bench -p 1024 -n 128 -d 0,40000 -b 4096 -ub 1024,2048,4096

Use default repeats, with no -r override. For MTP speed also run tools/server/bench/speed-bench --bench throughput_1k --bench throughput_32k --limit 2 against a server using ubatch 4096 and --spec-type draft-mtp. Apply the required lazy-tensor flag where relevant. Standalone sampler microbenchmarks supplement this matrix; they do not replace it.

Acquire hardware_queue resource gpu before every GPU benchmark, model run or GPU test suite, and release immediately afterward. GPU commands require the appropriate sandbox escalation. Store evidence under .synagent/artifacts/backend-grammar-next/ and scratch builds under .synagent/scratch/backend-grammar-next/, never /tmp. Reference GGUF models in their existing approved stores.

No speedup or new correctness result is asserted by this document. The proposed gains are conditional on measured rejection rate, mask cost, chain support, transfer lifetime and backend behavior.

