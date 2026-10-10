# P09 - Fixed-width padded verification

**Kind:** common-layer feature
**Depends on:** none
**Independent of:** P10

## Context

Target verification currently evaluates one sampled seed row plus every draft
row. The segment therefore has `n_draft + 1` rows, and varying draft lengths
can force `llama_context` to build a different target graph. P09 makes that
segment width stable by appending inert rows. It does not depend on P10 and does
not change draft-width policy.

`LLAMA_SPEC_PAD_VERIFY` is the single runtime contract. Unset or `0`
disables padding. Empty, `1`, or `auto` enables an automatic width: the
smallest power of two that covers
`common_speculative_n_max(spec) + 1` verification rows, including the sampled
seed. An integer of at least 2 is an explicit total verification-row width.
Invalid values fail initialization. The feature is disabled by default.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`, its
padded target verification path. The source's coupling to cumulative-probability
drafting is intentionally excluded.

## EARS requirements

- **R09.1 (State-driven):** WHILE fixed-width verification is enabled and the target batch is eligible, the server target-batch builder shall give each speculative slot exactly the configured total verification-row width.
- **R09.2 (State-driven):** WHILE `LLAMA_SPEC_PAD_VERIFY` is unset or `0`, the server target-batch builder shall preserve the exact current `n_draft + 1` verification segment shape.
- **R09.3 (Event-driven):** WHEN automatic fixed-width verification is selected, the server shall choose the smallest power of two greater than or equal to `common_speculative_n_max(spec) + 1`.
- **R09.4 (Ubiquitous):** The target output-validity map shall exclude fixed-width padding logits from sampler and public API access.
- **R09.5 (Ubiquitous):** The speculative acceptance index list shall exclude fixed-width padding rows from token emission.
- **R09.6 (Ubiquitous):** The target KV-cache write path shall skip fixed-width padding rows before allocating cells or applying stream offsets.
- **R09.7 (Unwanted behaviour):** IF a fixed width is smaller than a slot's real verification segment or exceeds target batch or microbatch capacity, THEN the server target-batch builder shall use the current unpadded shape for that target batch.
- **R09.8 (Event-driven):** WHEN fixed-width verification falls back, the server shall emit one structured reason with requested width, required rows, target `n_batch`, and target `n_ubatch`.
- **R09.9 (State-driven):** WHILE multiple speculative slots share a target batch, fixed-width verification shall apply to every eligible slot or to none of them.
- **R09.10 (Unwanted behaviour):** IF `LLAMA_SPEC_PAD_VERIFY` is neither a disabled value, an automatic value, nor an integer of at least 2, THEN the server initializer shall fail with the rejected value.
- **R09.11 (Event-driven):** WHEN `llama_context::process_ubatch` chooses graph reuse or rebuild for a tagged verification cycle, the P09 counter shall record that decision locally.
- **R09.12 (State-driven):** WHILE the target uses recurrent state or a backend without masked KV-row support, fixed-width verification shall use the current unpadded path.
- **R09.13 (State-driven):** WHILE fixed-width verification is active, the target graph shall retain fixed verification output capacity with a dynamic real-row validity mapping.
- **R09.14 (Ubiquitous):** The KV slot mapping shall preserve one row-aligned entry per verification input and mark every padding entry with the reserved invalid sentinel.

## Approach

1. Parse `LLAMA_SPEC_PAD_VERIFY` once and resolve automatic width after
   `common_speculative_n_max(spec)`, including the mandatory seed row.
2. In `server_slot::handle_last_sampled_token`, keep seed and real drafts in
   current order and keep `spec_i_batch` real-only. Decide padding for all
   active slots before adding any row; width must fit each real segment,
   `n_ubatch`, and combined `n_batch`, or the entire target batch is unpadded.
3. Append per-slot padding rows with an internal
   `LLAMA_BATCH_ROW_PAD_VERIFY` flag. Carry a row-validity sidecar through
   `common_batch`, optional `llama_batch`, `llama_batch_allocr`, and
   `llama_ubatch`; null means all rows valid and preserves disabled behavior.
4. Keep graph output topology fixed. For every padded verification segment,
   allocate output capacity for all fixed rows and provide a same-shape dynamic
   `verify_output_valid` input. Real rows map to normal output positions;
   padded logits map to scratch positions, are marked invalid, and are never
   returned by `llama_get_logits_ith` or placed in `spec_i_batch`. Extend
   `llm_graph_input_out_ids::can_reuse` so changing real draft length changes
   values, not `n_outputs` or graph shape.
5. Define `LLAMA_KV_PAD_IDX=UINT32_MAX` in `slot_info::idx_vec_t`. Keep the
   row-aligned slot vector at padded width plus an explicit valid-cell count.
   `find_slot` allocates only valid cells but inserts the sentinel for every
   padded row. Check the sentinel before any arithmetic or stream offset.
6. Update every row consumer: `apply_ubatch`, cache head advancement, cell
   metadata/sequence/position/token updates, rollback/recovery, KQ-mask input,
   `set_input_k_idxs`, and `set_input_v_idxs` skip invalid rows and assert
   that valid-row count equals allocated-cell count. Padded query-mask rows are
   fully masked and never alter cache metadata.
7. Convert the sentinel to signed `-1` only after the pre-offset validity
   check. Add an explicit `skip_negative_indices` op parameter used only by KV
   cache `GGML_OP_SET_ROWS` calls. CPU/SYCL kernels test it before address
   calculation; all other SET_ROWS callers retain existing negative-index error
   semantics. Unsupported backends force the unpadded path.
8. Tag padded verification ubatches with a P09 verification-cycle ID. Increment
   P09-owned reuse/rebuild/step counters at the actual decision branch in
   `llama_context::process_ubatch` (current lines 1853-1896), and expose them
   through structured server metrics. P04 may join by cycle ID when present but
   is not required for implementation or verification.
9. Add tests for row-flag propagation; slot search/allocation; apply/head/metadata
   invariants; recovery; pre-offset sentinel handling; scoped SET_ROWS behavior;
   fixed output capacity under alternating short/long drafts; invalid-logit
   invisibility; real acceptance mapping; multi-slot all-or-none fallback; and
   disabled byte-for-byte shapes.

## Critical files & anchors

- `common/speculative.cpp:3067-3115` - `common_speculative_n_max`.
- `tools/server/server-context.cpp:663-701` - real target verification segment assembly.
- `tools/server/server-context.cpp:4280-4355` - acceptance consumes only `spec_i_batch`.
- `common/common.h:1098-1104` - `common_batch` row construction and validity flag.
- `include/llama.h` and `src/llama-batch.cpp:170-252,784-850` - batch sidecars and ubatch propagation.
- `src/llama-graph.cpp:200-232` - output-ID capacity, validity mapping, and reuse.
- `src/llama-kv-cache.h:98-141` - row-aligned `slot_info`, valid count, and sentinel.
- `src/llama-kv-cache.cpp:1438-1700,1886-2035,2090-2317,2979-3067` - allocation, metadata, K/V indices, masks, and recovery.
- `ggml/src/ggml-cpu/ops.cpp:5370` and `ggml/src/ggml-sycl/set_rows.cpp` - scoped negative-index skip.
- `src/llama-context.cpp:1840-1896` - actual graph reuse/rebuild decision and P09 counters.
- `tests/test-backend-ops.cpp` and `tests/test-qwen4exp-mtp.cpp` - scoped sentinel, fixed-output, and speculative tests.
- `scripts/perf/verify-padded-verify.py` - planned token/KV/graph-count comparison.

## Verification

Run set-rows and speculative tests first:

```bash
timeout 180 ./build-sycl/bin/test-backend-ops -b CPU -o SET_ROWS
timeout 180 ./build-sycl/bin/test-backend-ops -b SYCL0 -o SET_ROWS
timeout 240 ctest --test-dir build-sycl -R 'test-qwen4exp-mtp' --output-on-failure
```

On A770, stop the service, verify sole tenancy, name the driver/build/model, and
run identical MTP requests with padding off, automatic, width 8, and oversized:

```bash
timeout 1800 python3 scripts/perf/verify-padded-verify.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --seed 123 --repeats 2 --modes off,auto,width:8,oversize --spec-args='--spec-type draft-mtp --spec-draft-n-max 7' --report /tmp/p09-padded-verify.json
```

Expected evidence:

- off, auto, and width-8 modes emit identical target tokens and accepted/drafted
  totals;
- padding logits occupy fixed scratch capacity but are invalid to sampler/API,
  absent from `spec_i_batch`, and never emit tokens;
- row-validity tests show padding allocates no KV cell, advances no head, updates
  no metadata, survives recovery, and is checked before stream offsets;
- committed K/V hashes and positions are identical to off mode;
- alternating short/long drafts keep constant `n_outputs` and output tensor
  shapes, while only the dynamic validity mapping changes;
- automatic width resolves correctly; oversized mode falls back with the
  structured capacity reason;
- at least 100 tagged steps show fewer P09-local rebuilds per 100 steps than off
  mode without requiring P04;
- CPU/SYCL tests prove only KV calls with `skip_negative_indices` skip `-1`;
  other SET_ROWS semantics remain unchanged;
- multi-slot padding is all-or-none and real indices remain attached to the
  correct slot;
- the post-run two-driver fault gate passes and the service is restarted.

## Assumptions & contingencies

- Width counts the complete target verification segment, including its sampled
  seed row.
- The supported path is ordinary causal transformer KV on CPU/SYCL; recurrent,
  cross-attention, or unsupported backends fall back unpadded.
- The sentinel is an internal row-validity marker checked before offset math;
  scoped SET_ROWS skipping is enabled only for validated KV-cache calls.
- P09 owns its graph-decision counters. P04 is an optional correlation source,
  not a dependency.
- P09 and P10 remain independently enabled, measured, and reversible.
