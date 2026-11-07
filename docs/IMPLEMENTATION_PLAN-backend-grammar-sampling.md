The implementation plan has been saved to `docs/IMPLEMENTATION_PLAN-backend-grammar-sampling.md`, and implementation has been completed through Stages B, D, E, F, G, and H.

The recommended architecture is **CPU grammar state plus token-aware compilation and backend mask application**. GPU execution should initially cover masking and subsequent sampling; moving the pushdown grammar interpreter onto the GPU should remain a separate research direction.

**1. Scope and verified baseline**

This plan targets local commit `8a5f7b069386a32899224b4c07193b7acc984d21`. All local file references below refer to that revision. Proposed identifiers and new files are design suggestions, not existing APIs.

[PR #29557](https://github.com/ggml-org/llama.cpp/pull/29557), retrieved at head `daa2414ce934e6d3189f0e4d300cb05b8ca6cd46`, adds left-factoring before grammar stack construction. It reduces redundant alternatives but does not provide token-aware tables or backend grammar hooks. Its implementation is not present in the inspected local grammar file.

There is also overlapping work in [PR #31672](https://github.com/ggml-org/llama.cpp/pull/31672), retrieved as an open draft at head `3d6a41237fede981c91097eefc5b8e537ba6b968`. It proposes backend sampling followed by CPU validation of the selected token, with full-logit recovery on rejection. Reconcile the compatibility work below with that proposal before implementation. Its reported measurements were not independently verified in this investigation.

| Area | Verified source | Design consequence |
|---|---|---|
| Backend sampler interface | [include/llama.h:1378](/devel/tools/llama.cpp/include/llama.h:1378) | Hooks receive tensors but no sequence-prefix description. |
| Grammar state | [src/llama-grammar.h:128](/devel/tools/llama.cpp/src/llama-grammar.h:128) | State includes a set of stacks, partial UTF-8, and lazy-trigger history. |
| CPU grammar masking | [src/llama-grammar.cpp:1431](/devel/tools/llama.cpp/src/llama-grammar.cpp:1431) | Preserve token-piece, EOG, empty-piece, and partial UTF-8 behavior. |
| Grammar token acceptance | [src/llama-grammar.cpp:1474](/devel/tools/llama.cpp/src/llama-grammar.cpp:1474) | Acceptance remains authoritative on the CPU initially. |
| Grammar sampler hooks | [src/llama-sampler.cpp:2758](/devel/tools/llama.cpp/src/llama-sampler.cpp:2758) | All six backend/copy hooks are null. |
| Backend chain support | [src/llama-sampler.cpp:734](/devel/tools/llama.cpp/src/llama-sampler.cpp:734) | Offload covers a supported prefix, not necessarily the entire chain. |
| Common grammar guard | [common/sampling.cpp:422](/devel/tools/llama.cpp/common/sampling.cpp:422) | Removing this requires ownership, policy, and fallback changes. |
| Common sampling policy | [common/sampling.cpp:606](/devel/tools/llama.cpp/common/sampling.cpp:606) | Default behavior is sample, validate, then retry with grammar filtering. |
| Multiple output branches | [src/llama-graph.cpp:4281](/devel/tools/llama.cpp/src/llama-graph.cpp:4281) | Different rows cannot safely share one grammar state. |
| Backend acceptance | [src/llama-graph.cpp:4377](/devel/tools/llama.cpp/src/llama-graph.cpp:4377) | Invocation is commented out; it cannot drive grammar advancement. |
| Output transfers | [src/llama-context.cpp:2356](/devel/tools/llama.cpp/src/llama-context.cpp:2356) | Mask offload alone does not eliminate logits/probability readback. |

The initial supported configuration should be explicit grammar-first sampling, one output per sequence per decode, independent grammar state for each sequence, and CPU fallback for unsupported configurations. Later stages extend compatibility-mode acceleration and speculative multi-output execution.

**2. Semantic contract**

Correctness requires preserving both the grammar language and the sampling policy.

For a grammar state `S`, define:

```text
allowed(S, token_id) = whether the existing CPU grammar masker permits that ID
advance(S, token_id) = the existing CPU grammar acceptance operation
```

Treat these as separate operations. Do not implement `allowed()` by cloning the grammar and calling `advance()` until their equivalence has been demonstrated for the relevant grammar subset.

The current masking and acceptance paths handle mixed character and `TOKEN`/`TOKEN_NOT` terminals differently. In particular, inspect [src/llama-grammar.cpp:1078](/devel/tools/llama.cpp/src/llama-grammar.cpp:1078) against [src/llama-grammar.cpp:1548](/devel/tools/llama.cpp/src/llama-grammar.cpp:1548). Any inconsistency discovered there needs a separate semantic decision and regression test; this optimization must not silently redefine it.

Required invariants:

- A rejected token receives exactly `-INFINITY`.
- An allowed token retains its input logit.
- Mask computation does not advance grammar, trigger, or RNG state.
- Only committed output tokens advance the authoritative grammar.
- Candidate IDs remain vocabulary IDs after compaction or reordering.
- Graph rebuilds do not reset accepted history.
- Unsupported optimizations preserve constraints through an exact fallback.
- Empty allowed sets produce an explicit failure; they must never become unconstrained sampling.
- EOG eligibility follows the existing empty-stack rule. EOG is not treated as an ordinary token piece.
- Lazy grammars awaiting a trigger preserve the current unrestricted behavior.

There are two sampling policies:

| Policy | Behavior | Initial treatment |
|---|---|---|
| Existing rejection policy | Run the chain, validate the selected token, then retry from full logits with grammar filtering if necessary | Preserve; initially retain CPU fallback or reuse the compatibility approach from PR #31672 |
| Grammar-first | Apply grammar before the sampling chain | First target for backend grammar masking |

These policies can produce different distributions with top-k/top-p. Enabling backend sampling must not implicitly switch between them.

The existing `grammar_first` argument is supplied at sample time, after decode may already have executed backend sampling. For accelerated grammar-first execution, establish the policy before decode. Preserve existing callers through a compatibility path; a late request for a different policy must use retained raw logits and a correct replay path.

**3. Token-aware grammar transformation**

Do not replace a literal with its tokenizer’s preferred tokenization. For example, a grammar accepting `"hello"` must permit every vocabulary-token sequence whose pieces satisfy the grammar, including pieces that cross rule boundaries.

Likewise:

```text
root ::= left right
left ::= "ab"
right ::= "cd"
```

must allow a token with piece `"abcd"` when the vocabulary contains it. Per-rule tokenization would miss that transition.

The proposed transformation is a lazily compiled token transition system over the existing pushdown state:

```text
GBNF
  -> parsed and validated rules
  -> left-factored rules
  -> immutable grammar representation
  -> vocabulary token-piece trie
  -> token validity tables for encountered grammar states
```

It is token-aware without assuming that an arbitrary context-free grammar can be reduced to a finite DFA of manageable size.

Implement this in three increments.

1. **Exact mask extraction.** Add an internal operation that produces a validity bitset independently of a logits array. Initially derive it from the current rejection machinery. This establishes a reusable correctness reference and separates grammar evaluation from logit mutation.

2. **Vocabulary trie traversal.** Build a byte trie from the exact pieces used by `grammar.vocab->token_to_piece(id)`. Share traversal of common token prefixes and prune subtrees when no grammar branch can accept them.

3. **Bounded state and transition caching.** Cache masks for repeated complete grammar states. Add local rule-position optimizations only where their independence from the caller stack is proven.

The trie must preserve:

- Multiple token IDs with identical pieces.
- Tokens ending at internal trie nodes.
- Byte-fallback tokens and incomplete UTF-8.
- The current treatment of embedded NUL and malformed byte sequences.
- Special-token and EOG classification outside ordinary character matching.
- Exact token IDs for `TOKEN` and `TOKEN_NOT`.

For character-only paths, traversal carries the grammar frontier and UTF-8 decoder state. At a token terminal, evaluate validity using the same end-of-piece and partial-character rules as the current masker. Do not treat an incomplete but still viable UTF-8 prefix as a completed character.

For token terminals and mixed token/character paths, retain exact candidate-ID checks. A practical first implementation routes grammars containing these constructs through the existing matcher to build the same mask representation. Extend trie acceleration to them only after differential tests establish equivalence.

The trie improves shared-prefix work; it does not guarantee sublinear work for every grammar. Worst-case mask construction can still visit much of the vocabulary and many grammar states.

**4. State identity, caches, and resource limits**

A full-state mask cache key must contain:

```text
grammar-program identity
vocabulary identity
canonical set of complete stacks
partial_utf8.value
partial_utf8.n_remain
effective grammar-enabled state
```

Use `(rule_index, element_offset)` identifiers in cache keys, not raw pointers. The current stacks contain pointers into rule vectors; cloning already requires pointer relocation at [src/llama-grammar.cpp:1401](/devel/tools/llama.cpp/src/llama-grammar.cpp:1401).

Canonicalize the stack set, retain full continuations, and verify equality after hash matches. Two states with the same top terminal but different return stacks can admit different multi-character tokens.

Keep these objects separate:

| Object | Lifetime |
|---|---|
| Token-piece trie | Vocabulary/model lifetime |
| Factored grammar and position index | Grammar configuration lifetime |
| Full-state mask cache | Bounded grammar-program cache |
| Accepted stacks, UTF-8, trigger history | Per sampler/sequence |
| Graph tensors and upload storage | Destination graph/execution lifetime |

Initially cache masks only. Compute the accepted token’s successor with the existing acceptance implementation. Caching all successor stack sets for all vocabulary tokens can consume much more memory than storing validity masks.

Later, cache selected transitions `(state, token) -> successor` under a separate budget.

Bound preprocessing bytes, cache bytes, trie traversal work, stack expansion, and allocation arithmetic. An optimization-budget miss should use the exact reference matcher. Exhaustion of an overall request resource limit should return an explicit error, never an approximate mask.

Left-factoring must finish before constructing stable position identifiers or stack pointers. Validate against both factored and unfactored reference grammars; stack layout equality is not a language-equivalence test.

**5. Lookup-table representation**

Represent a completed allowed set with one of five encodings:

| Encoding | Meaning | Payload |
|---|---|---|
| `ALL` | Every vocabulary ID is allowed | None |
| `NONE` | No vocabulary ID is allowed | None |
| `ALLOW_IDS` | Only listed IDs are allowed | Sorted unique IDs |
| `DENY_IDS` | Every ID except listed IDs is allowed | Sorted unique IDs |
| `BITSET` | One validity bit per vocabulary ID | `ceil(V / 32)` words |

Choose the encoding from the actual completed mask. Do not assume that most states have small sets without measuring representative grammars.

With 32-bit IDs, sparse storage uses approximately `4*k` bytes and a bitset uses approximately `V/8` bytes. The storage crossover is roughly `k < V/32`, excluding headers. Kernel lookup cost will usually justify a smaller sparse threshold.

At `V = 131072`:

- Dense F32 mask: 512 KiB.
- Bitset: 16 KiB.
- A 32-ID sparse payload: 128 bytes.

These are calculated storage sizes, not measured transfer costs.

For alternatives, combine **allowed sets by union**. If branch masks use complements:

```text
(U \ A) union (U \ B) = U \ (A intersect B)
```

Unioning deny lists would incorrectly reject tokens accepted by another branch. Initially compose masks as CPU bitsets and compress only the final result.

Store EOG and empty-piece decisions in the completed mask according to the existing precedence rules. While awaiting a lazy trigger, use `ALL`.

Use immutable host cache entries. For device inputs, use fixed-capacity storage plus a mode/count descriptor, so changes between sparse and dense states do not force graph reconstruction. A capacity sufficient for one bitset can also hold sparse ID payloads below the storage crossover.

Device-resident multi-entry caching can be added later if upload measurements justify it. It introduces eviction and in-flight lifetime management that the first version does not need.

**6. GGML operation and CUDA/ROCm kernel**

A generic token-mask operation is the cleanest production interface. A provisional name is `ggml_token_mask`; finalize naming with GGML maintainers.

Its inputs should describe:

```text
F32 logits: candidate_count x row_count
I32 candidate IDs, or implicit identity mapping
I32 mask payload
I32 per-row descriptor: mode, count, payload offset, vocabulary size
```

Its output is an F32 tensor with the same logical shape as the logits:

```text
id = candidate_ids ? candidate_ids[column, row] : column
output[column, row] = allowed(mask[row], id)
                    ? logits[column, row]
                    : -INFINITY
```

Start with contiguous F32 rows and explicitly supported strides. Validate tensor types, descriptor dimensions, payload lengths, offsets, and vocabulary bounds before execution. Kernel bounds checks must prevent invalid candidate IDs from becoming out-of-bounds reads.

Use an out-of-place result initially. In-place masking would destroy raw logits needed for retries, probability reporting, or other consumers.

For the first CUDA/HIP kernel:

- Assign one thread to each candidate.
- Read one mask bit for `BITSET`.
- Use a small linear search for very short sparse lists; use binary search or bitset encoding beyond a measured threshold.
- Copy unchanged values for `ALL`; write `-INFINITY` for `NONE`.
- Avoid grammar traversal, recursion, dynamic allocation, and atomics.
- Do not hardcode warp size.

The full-width operation remains `O(V)` when all vocabulary logits must be written. The gain is removing grammar interpretation from that loop and reducing mask transfer size.

Possible later kernels are fill-plus-scatter for allow lists and copy-plus-scatter for deny lists. They require explicit dependency ordering, unique indices, and careful treatment of nonidentity candidate layouts. They should be added only if measurements beat the single-kernel implementation.

HIP already compiles the CUDA source directory: [ggml/src/ggml-hip/CMakeLists.txt:60](/devel/tools/llama.cpp/ggml/src/ggml-hip/CMakeLists.txt:60). Use shared portable source rather than maintaining a separate ROCm grammar implementation.

Expected integration points:

| Change | Existing anchor |
|---|---|
| Operation enum and API | [ggml/include/ggml.h:542](/devel/tools/llama.cpp/ggml/include/ggml.h:542) |
| Tensor construction and shape checks | [ggml/src/ggml.c:4061](/devel/tools/llama.cpp/ggml/src/ggml.c:4061) |
| CPU dispatch and reference implementation | [ggml/src/ggml-cpu/ggml-cpu.c:1915](/devel/tools/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1915) |
| CUDA operation dispatch | [ggml/src/ggml-cuda/ggml-cuda.cu:2482](/devel/tools/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2482) |
| CUDA support checks | [ggml/src/ggml-cuda/ggml-cuda.cu:6016](/devel/tools/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:6016) |
| Backend operation tests | [tests/test-backend-ops.cpp:2695](/devel/tools/llama.cpp/tests/test-backend-ops.cpp:2695) |

Add proposed `token-mask.cu/.cuh`, CPU implementation declarations, operation metadata, and backend support documentation as required by the existing GGML conventions.

Existing operations are useful for a prototype:

- `ggml_get_rows` can gather mask values by candidate ID.
- `ggml_fill` and `ggml_set_rows` can construct sparse masks.
- `ggml_add` can apply a dense additive mask.

However, `ggml_set_rows` aliases its destination, so graph dependencies and storage ownership matter. Additive masking also differs from assignment for values such as positive infinity and NaN. A direct select/write operation expresses the CPU contract precisely.

Do not use `MAP_CUSTOM` callbacks as a supposed CUDA implementation; those APIs describe host callbacks.

**7. Backend sampler hooks and lifecycle**

Extend `llama_sampler_grammar` at [src/llama-sampler.cpp:2667](/devel/tools/llama.cpp/src/llama-sampler.cpp:2667) with token-table/cache references, a state generation, prepared mask data, and graph-local input bindings.

| Hook | Required behavior |
|---|---|
| `backend_init` | Check supported configuration and operation support; initially reject more than one output per sequence. |
| `backend_apply` | Construct mask inputs and the mask operation; replace `data.logits`; preserve candidate identity. |
| `backend_set_input` | Prepare or reuse the current-state mask and upload descriptor/payload before execution. |
| `backend_reset` | Clear graph bindings and graph-local bookkeeping only. |
| `copy_state` | Copy accepted grammar and trigger state while retaining destination graph references. |
| `backend_accept` | Leave null initially; CPU acceptance commits history. |

Additional details:

- `backend_apply` must not advance state or perform host transfers.
- Grammar should precede probability construction in the common grammar-first path.
- If an explicitly constructed chain places grammar after a probability-producing stage, invalidate stale `data.probs`; subsequent sampling must derive probabilities from masked logits.
- A grammar stage after an already selected token is not a valid masking arrangement.
- `accept` advances the existing grammar exactly once and invalidates the prepared mask.
- `clone` copies mutable state but starts with no graph bindings.
- Semantic `reset` restores the initial grammar and trigger state.
- `copy_state` must handle destination rule-pointer relocation or immutable-program sharing without replacing graph-owned storage.

The distinction between copy and clone is essential: the generic fallback at [src/llama-sampler.cpp:4329](/devel/tools/llama.cpp/src/llama-sampler.cpp:4329) destroys destination context and transplants a clone. That is unsuitable once the grammar context contains graph bindings.

Also address support probing. The current helper constructs synthetic 128k-candidate tensors at [src/llama-sampler.cpp:648](/devel/tools/llama.cpp/src/llama-sampler.cpp:648). Vocabulary-bound grammar masks need shape-valid probes for the actual vocabulary and supported candidate layouts. Probing must not populate semantic caches or leave dangling input tensors.

Finally, a false return from chain initialization does not mean all offload was disabled. [src/llama-context.cpp:1587](/devel/tools/llama.cpp/src/llama-context.cpp:1587) retains the sampler after initialization. Capability and fallback decisions must inspect the actual supported prefix.

**8. Common-layer integration and ownership**

Common currently owns `grmr` separately from `chain`, with separate acceptance, cloning, and freeing. Simply adding `grmr` to the chain risks double acceptance and double free.

Preserve that ownership initially:

- `common_sampler` continues to own the authoritative grammar.
- Add a small internal chain adapter that delegates backend mask construction and input preparation to that grammar.
- The adapter does not own the grammar.
- Its CPU `apply` and `accept` are no-ops; common retains explicit CPU grammar handling.
- In grammar-first mode, put this adapter first in the backend chain.
- Rebind the adapter to the cloned destination grammar in `common_sampler_clone`.
- During `common_sampler_copy`, copy the grammar once and preserve the destination adapter binding.
- Make destruction order and adapter lifetime explicit.

This adapter is common-layer plumbing. Direct library users can place the real grammar sampler in a chain and use its normal CPU acceptance semantics.

Audit [common/sampling.cpp:449](/devel/tools/llama.cpp/common/sampling.cpp:449), [common/sampling.cpp:476](/devel/tools/llama.cpp/common/sampling.cpp:476), and [common/sampling.cpp:518](/devel/tools/llama.cpp/common/sampling.cpp:518) together. Prompt tokens must not accidentally advance a user grammar merely because other samplers accept prompt history. Preserve the generation-prompt prefill rules at [common/sampling.cpp:298](/devel/tools/llama.cpp/common/sampling.cpp:298).

Replace the unconditional guard with explicit eligibility:

```text
grammar backend masking eligible =
    grammar-first policy established before decode
    supported grammar mask implementation
    supported backend operation/layout
    supported output-state contract
    compatible common sampler controls
```

Preserve reasoning-budget and other existing restrictions until their behavior is covered.

In the default rejection policy, backend acceptance requires CPU validation of the selected token. On rejection:

1. Recover the original full-vocabulary logits.
2. Apply grammar filtering.
3. Replay the complete sampling chain with the correct state and RNG semantics.

Do not retry from top-k candidates; all surviving candidates may violate the grammar. Do not use normal chain application blindly, because it skips stages already marked as backend-executed.

For a later fully accelerated rejection retry, retain raw device logits and run a separate grammar-first sampling graph after CPU validation rejects the proposal. This avoids rerunning model evaluation but adds another sampling execution and synchronization. It requires explicit accounting of both random draws and mutable sampler effects.

Update the backend-token assertion at [common/sampling.cpp:629](/devel/tools/llama.cpp/common/sampling.cpp:629) only after this policy routing is implemented.

**9. CPU/GPU synchronization and output transfers**

For one output per sequence, execution should be:

```text
Committed CPU grammar state
  -> prepare mask
  -> upload mask
  -> execute model and backend sampling
  -> retrieve selected token
  -> common decides whether to commit it
  -> CPU grammar accepts committed token
  -> invalidate prepared mask
```

Masks must describe the state before the predicted token. Do not accept the decode input token again if common already committed it during the previous iteration.

Initially perform preparation through `backend_set_input`, called from [src/llama-graph.cpp:1371](/devel/tools/llama.cpp/src/llama-graph.cpp:1371). Upload buffers must remain alive until the transfer completes. A later asynchronous implementation needs owned staging buffers or execution slots, not stack-local vectors.

Reducing transfers is a separate milestone. The current context copies sampled logits, probabilities, and candidates even when a token has already been selected. Common also reconstructs candidates before checking the selected token.

Add an explicit output requirement distinguishing:

- Selected token only.
- Post-sampling candidates/probabilities.
- Raw logits for fallback or pre-sampling reporting.

Preserve the server’s reporting behavior and gates at [tools/server/server-context.cpp:2065](/devel/tools/llama.cpp/tools/server/server-context.cpp:2065). Device rows retained for deferred access must survive graph reuse, reserve/rebuild, output reordering, and later ubatches. Retaining a pointer into reusable graph memory is insufficient.

Measure transfer suppression independently from token-table and kernel improvements.

**10. Multiple sequences and multiple outputs**

Multiple independent sequences are compatible with the hybrid design when each has:

- Its own accepted grammar and trigger state.
- Its own prepared mask and generation.
- A correctly mapped output row.
- Immutable grammar/vocabulary data shared only where safe.

The current graph chooses `ubatch.seq_id[i][0]` when mapping sampling rows. Initially reject or fall back for output tokens associated with multiple sequence IDs unless all relevant state semantics are explicitly supported. Do not silently use the first sequence’s grammar.

Multiple outputs from one sequence require a new contract. For draft tokens `d0 ... d(n-1)`:

```text
S0 = committed state
S1 = advance(S0, d0)
S2 = advance(S1, d1)
...
mask for target row j = allowed(Sj, *)
```

Those states are provisional. They must not alter the committed state until speculative verification determines the accepted prefix.

The existing hooks cannot express this safely: `backend_set_input` receives no row prefixes, and graph construction builds several branches from the same sampler.

Initial release:

- Return unsupported when `n_outputs_max_per_seq > 1`.
- Fall back to sequential CPU grammar sampling for those configurations.
- Do not reuse one mask across speculative rows.
- Keep the limit honest even when a particular decode happens to request only one row.

Later extension:

1. Add an explicit pre-decode preparation operation describing output-row identity and the token prefix that precedes each row.
2. Supply prefixes from the speculative driver, not by guessing from output-row order.
3. Build provisional CPU grammar snapshots and mask rows.
4. Mark rows after an invalid draft prefix unusable; do not pass an all-masked row into ordinary sampling and interpret its result.
5. Commit only the accepted prefix and the final replacement/bonus token.
6. Discard unused grammar snapshots and sampler draws.
7. Reuse the existing transactional RNG model where applicable, extending it to grammar retries.

The transactional distribution sampler at [src/llama-sampler.cpp:1125](/devel/tools/llama.cpp/src/llama-sampler.cpp:1125) is a relevant pattern, but it does not supply grammar-prefix state.

Teacher-forced scoring, linear draft verification, and tree speculation need distinct row-prefix descriptions. For rows whose prefix depends on an unknown sampled token, use sequential sampling executions or fall back. Parallel branches cannot predict that state.

**11. Lazy grammars, triggers, and alternative grammar engines**

Keep trigger matching on the CPU.

While awaiting a trigger, upload `ALL`. On committed acceptance, execute the existing trigger logic at [src/llama-grammar.cpp:1479](/devel/tools/llama.cpp/src/llama-grammar.cpp:1479), including token-trigger consumption and regex replay of overlapping token pieces. Then invalidate the mask.

Required cases include:

- Trigger split across tokens.
- Trigger starting in the middle of a token piece.
- Trigger plus constrained suffix in the same piece.
- Partial UTF-8 at a trigger boundary.
- Reset, clone, and rollback before and after triggering.
- A trigger encountered in a speculative prefix that is later rejected.

Reasoning-budget suppression depends on common-layer state and acceptance ordering. Preserve [common/sampling.cpp:461](/devel/tools/llama.cpp/common/sampling.cpp:461) and its end-sequence replay behavior. Keep backend grammar masking disabled for that combination initially.

`llguidance` is a separate matcher. It already exposes a token bitmask in [common/llguidance.cpp:46](/devel/tools/llama.cpp/common/llguidance.cpp:46), so it can eventually reuse the generic backend mask operation. It should not be routed through the GBNF compiler. Support it in a separate change with its own lifecycle and error tests.

**12. Implementation sequence**

| Stage | Deliverable | Completion gate |
|---|---|---|
| A | Reconcile PR #29557 and overlapping PR #31672; freeze semantic fixtures | Documented policy, ownership, and supported configurations |
| B | CPU validity-bitset API and bounded full-state cache | Exact mask agreement with current CPU matcher |
| C | Vocabulary trie acceleration for character grammars; exact fallback elsewhere | Differential masks, UTF-8 and boundary tests |
| D | Generic mask operation with CPU implementation | Operator contract and backend-op tests |
| E | Shared CUDA/HIP implementation | Actual CPU/CUDA and CPU/HIP correctness evidence |
| F | Grammar sampler backend hooks and lifecycle | Copy/reset/clone/rebuild and fallback tests |
| G | Explicit grammar-first common integration | Correct ownership, prompt handling, and one-output generation |
| H | Compatibility-mode integration and output-transfer control | Full-logit retry and RNG/state tests |
| I | Explicit speculative row-prefix preparation | Multi-output acceptance, rollback, and row-mapping tests |
| J | Optional resident mask cache, specialized sparse kernels, llguidance adapter | Measured benefit with unchanged semantics |

Stage J uses graph-local resident payload slots in the grammar sampler. Each row has up to eight LRU entries under a 1 MiB payload budget, with at least one bitset-sized slot for larger vocabularies. Cache identity compares the complete encoding descriptor and payload, so equivalent completed masks can share a slot without equating their grammar states. Hits upload only a changed 16-byte descriptor; an unchanged binding needs no upload. Uploads use the scheduler-selected backend stream when events are available. Owned descriptor and payload staging survives until an upload event completes; unsupported configurations retain synchronous uploads. Exact unchanged grammar states retain their packed mask, and completed UTF-8 code points are normalized out of mask-cache identity. Each row owns its storage; graph reset releases the bindings and their cache, while semantic reset and copy retain valid resident masks in the destination graph.

The CUDA/HIP implementation uses one kernel for validation and masking. Each block owns a 4096-candidate tile. Identity sparse masks fill or copy that tile, then scatter its sorted IDs after a block barrier. Explicit candidate IDs use lookup, including reordered, repeated, compacted, and invalid IDs. Sparse payload validation is distributed across blocks and rejects unsorted or invalid IDs. LLGuidance supplies its own CPU bitsets to the generic operation and keeps matcher state and provisional prefix clones separate from GBNF.

Keep CPU grammar work, the GGML operation, backend kernels, and common integration separately reviewable. Public API expansion for speculative preparation is a design checkpoint, not a prerequisite for the first useful implementation.

**13. Validation and performance plan**

Use existing test files rather than adding a separate testing framework.

For grammar correctness, extend `tests/test-llama-grammar.cpp` and `tests/test-grammar-integration.cpp`. Compare complete vocabulary masks at successive states, not only final generated strings.

Include:

- Literals, alternatives, nullable rules, repetitions, and recursive grammars.
- Tokens crossing rule and alternative boundaries.
- Multiple IDs sharing one piece.
- Character ranges, negated ranges, and broad string content.
- Byte fallback, split Unicode, malformed UTF-8, and NUL behavior.
- Every EOG token, empty pieces, and special tokens.
- `TOKEN`, `TOKEN_NOT`, and mixed character/token rules.
- Factored versus unfactored grammars.
- Cache hits, misses, eviction, and fallback.
- Lazy triggers and generation-prompt prefill.

For the GGML operation, extend `tests/test-backend-ops.cpp`:

- All five encodings and representation transitions.
- Vocabulary sizes not divisible by 32.
- Empty/singleton masks.
- Identity, shuffled, duplicate, and compacted candidate IDs.
- Multiple rows with different masks.
- Supported nontrivial strides.
- Rejected logits equal to negative infinity; allowed values unchanged.
- NaN/infinity behavior and malformed descriptor rejection.
- Source logits unchanged after masking.

For integration, extend `tests/test-backend-sampler.cpp`:

- Grammar-first versus its CPU reference.
- Existing rejection policy, including a rejected backend proposal.
- A backend prefix followed by a CPU-only suffix.
- A valid token outside the backend’s truncated candidates.
- Clone, copy into an initialized destination, semantic reset, graph rebuild, and sampler replacement.
- Different grammars across sequences, inactive slots, row reordering, and ubatch splitting.
- Explicit fallback for unsupported multi-output and shared-sequence outputs.
- Later: speculative mismatch, EOG, trigger activation, rollback, and RNG draw accounting.

The HIP harness currently skips several sampler tests by default at [tests/test-backend-sampler.cpp:2179](/devel/tools/llama.cpp/tests/test-backend-sampler.cpp:2179). Explicitly select applicable cases and report skips; a default pass is insufficient qualification.

Performance evaluation should separate:

```text
grammar parse/factor time
token-index construction time and memory
cold mask construction
warm mask-cache lookup
accepted-token state advancement
mask upload
kernel execution
sampling/output synchronization
end-to-end generation
```

Compare the original CPU path, left-factored CPU path, token-aware CPU path, and backend path under the same sampling policy. Include short responses, large tool schemas, broad JSON strings, restrictive enums, Unicode grammars, and cache-unfriendly workloads.

Every end-to-end llama.cpp benchmark must retain the standing matrix:

```text
llama-bench:
  -p 1024 -n 128
  -d 0,40000
  -ub 1024,2048,4096
  -b 4096
  default repeats; do not pass -r
```

Add grammar-specific server workloads alongside that matrix; ordinary model throughput alone does not measure grammar sampling.

For MTP qualification, also run `tools/server/bench/speed-bench` with `--bench throughput_1k --bench throughput_32k --limit 2`, using a server at ubatch 4096 with `--spec-type draft-mtp`. Add the appropriate lazy-tensor flag for applicable models.

Acquire the shared `gpu` queue before GPU work and release it immediately afterward. Use project-local persistent artifact/scratch directories, never `/tmp`; reference GGUF files from the authorized model stores.

No speedup target is justified yet. The initial acceptance criteria are exact masks, correct sampler state, bounded memory, explicit fallbacks, and measured improvement on representative workloads without unacceptable startup regressions.

**14. Feasibility assessment**

The backend masking kernel is straightforward. The difficult work lies in preserving token-boundary semantics, common-layer sampling policy, and speculative state.

A useful first release is feasible without a GPU grammar interpreter: exact CPU-generated token masks, cached/trie-accelerated where beneficial, uploaded compactly and applied before backend sampling. Arbitrary recursive grammars remain supported through the existing CPU matcher when compilation or caching is ineffective.

Full GPU-resident grammar transitions would require bounded stack representations, branch management, UTF-8 handling, trigger semantics, and speculative commit/rollback. That is substantially larger than the requested masking integration and is not required to obtain the main architectural benefit.

This investigation used source inspection and upstream PR reads only. **No code was implemented, built, or tested, and no GPU/model run was performed.** The shared knowledge record and activity log were updated. Saving the document to the requested repository path remains the only delivery step blocked by this run’s read-only permissions.
