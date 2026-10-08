# Qwen4Exp MTP correctness, 2026-10-04

The MTP graph now uses the draft head's own `blk.N.nextn.hc_head_*` mixer.
All three mixer tensors are required when loading MTP weights. Ordinary loading
with MTP disabled still permits their absence. Adaptive MTP fitting now selects
an MTP context for both separate-head and shared-model configurations.

Qwen4Exp also supports opt-in chained drafting through `--spec-chain`. The chain
preserves the full hyper-connection state, uses the full vocabulary, and computes
confidence over the same top-10 candidates as sequential drafting. It reuses the
ordinary attention builder for cache rotations, padding, masks, and writes.

Commit ids quoted in this note name the tree a result was measured on. They are commits of
the PR 90 branch, which was merged as one squashed commit, so they are not in the history
of `master`; `git fetch origin pull/90/head` retrieves them.

Two additional correctness fixes came from the regression tests:

- Masked sequential MTP output now selects the requested hidden-state rows.
  Both the raw and gathered owning tensors remain protected outputs, because
  allocator reservation reuse checks sizes, not changed output flags. This keeps
  hidden states intact across masked/unmasked graph construction.
- Drafts exceeding the draft microbatch capacity use sequential execution.
  Splitting an in-graph chain would restart later steps from placeholder inputs.

## Evidence

This section records validation of the initial PR head, `54a2557`; review
follow-up validation is recorded separately below.

The new `test-qwen4exp-mtp` uses deterministic synthetic GGUFs with a recurrent
trunk block, a dense trunk block, and one dense MTP block. No external model or
network access is required. It exercises the real loader, graph execution,
public speculative driver, and fit initialization.

Observed baseline failures, using isolated original libraries:

| Check | Original behavior |
| --- | --- |
| Canonical MTP-only head | Abort at the trunk-mixer assertion, exit 134 |
| Change only trunk mixer | MTP logits change; independence assertion fails |
| Adaptive fit, separate head | Segmentation fault, exit 139 |
| Adaptive fit, shared model | Draft measurement context omitted |
| Chain depth 4, microbatch 2 | Draft tokens differ from sequential execution |

All corresponding checks pass with the fixes. Coverage includes missing and
partial mixers, rejection of output-only substitutes, non-MTP loading, depths
1/3/4, flash and non-flash attention, masked/unmasked output, catch-up rows,
rejected-tail rollback, the final context slots, multiple sequences, adaptive
depth changes, and a per-round context cap.

CPU validation:

```sh
ctest --test-dir /tmp/ggml-mtp-build \
  -R '^(test-qwen4exp-mtp|test-speculative-adaptive)$' \
  --output-on-failure --timeout 60
/tmp/ggml-mtp-build/bin/test-qwen4exp-mtp --q8-kv
```

Both CTest targets passed. The q8_0 KV run also passed.

SYCL validation on the Arc A770 passed for f16 KV with flash attention on/off,
and q8_0 KV with flash attention on. The final build used oneAPI 2026.1,
`GGML_SYCL_F16=ON`, `GGML_SYCL_DNN=OFF`, and LLVM's linker.

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=1 \
  GGML_SYCL_ENABLE_GRAPH=0 GGML_SYCL_GRAPH_PROFILE=1 \
  /tmp/ggml-mtp-sycl-build/bin/test-qwen4exp-mtp --backend SYCL0
# Also passed with --q8-kv, with GGML_SYCL_ENABLE_GRAPH=0 and =1.
```

The graph profiles recorded 355 direct calls for the f16 run and 245 for each
q8_0 run, with zero graph replays. Enabling graphs therefore verified the direct
fallback on this stack, not replay execution.

## Real-model attempt and limits

The original Coder IQ1_M shards and unmodified Q8_0 MTP head were located locally.
A 16K-context, `--fit on`, q8_0 KV server attempt crashed during target warmup in
`sycl::detail::PersistentDeviceCodeCache::getItemFromDisc`, reached through BF16
conversion in the auto-enabled oneDNN path. Disabling oneDNN at runtime did not
resolve startup. Rebuilding with `GGML_SYCL_DNN=OFF` hit a GNU BFD internal linker
error; using `-fuse-ld=lld` produced a successful build.

The subsequent real-model run was stopped when another service occupied the
A770. That service was left running. No completed real-model acceptance or speed
comparison is claimed, and these new prompts would not reproduce the original
benchmark prompts exactly.

## Review follow-up

Two public-API failures were reproduced against the initial PR library on CPU:

- A depth-32 chain exhausted graph tensor metadata and aborted. Qwen4Exp MTP
  contexts initially reserved metadata for every possible unrolled step. The
  allocation follow-up below replaces that eager reservation with growth on use.
- An output mask `[1, 0]` reached the graph's output-suffix assertion and aborted.
  Decode now rejects unsupported output masks, multiple sequences, missing token
  IDs, and total rows exceeding batch/microbatch capacity before cache updates.
  It returns `-1` without switching to a different output format.

The CPU regression suite now compares depth-32 tokens, probabilities, and all
hidden rows with sequential decoding. Invalid-chain cases verify byte-identical
sequence caches after rejection and a successful retry with chaining still on.
Driver cases exercise two catch-up rows at microbatch capacities 4/5/6 and a
supplied depth cap of two at the end of an occupied context. Both targeted CTest
tests and the CPU q8_0 KV variant passed after these changes.

The four disputed bot findings were checked against the surrounding source:

- Catch-up capacity is reserved by `defer_capacity()` before deferring any rows;
  incoming batches that exceed that allowance are processed immediately.
- The server computes remaining context/generation budget and supplies it via
  `dp.n_max`; chained and sequential drafting both apply that cap. The new driver
  test checks consumption of that cap, not the server's calculation itself.
- `build_inp_out_ids()` always allocates the output-index tensor.
- `build_attn_inp_kv()` always allocates the attention mask; flash attention
  changes its type, not its existence.

The duplicate adaptive-controller initialization was removed. SYCL f16 and q8_0
CTest entries now select `SYCL0` explicitly and share a resource lock. The SYCL
test target rebuilt successfully with the configuration above, and `ctest -N`
lists both backend-specific entries. No GPU execution of the expanded follow-up
suite is claimed here.

### Hidden-state validation follow-up

A token-only public chain was accepted at `21cab0124`, leaving the hidden-state
graph input unset. Decode now requires a hidden-state embedding on every chain
row, including zero placeholders for later generated rows. The regression failed
before the fix and now verifies rejection, unchanged serialized sequence caches,
and successful retry on the same context. Both targeted CPU CTest tests and the
CPU q8_0 variant passed. This additional guard was not rebuilt or run on SYCL;
the successful SYCL build above predates it.

## Allocation and regression follow-up

Ordinary context creation now rejects MTP-only Qwen4Exp heads before graph
construction reaches absent trunk tensors. The new regression reproduced exit
139 with the saved `4a9092768` libraries, then passed with the guard. It tests
heads loaded with MTP both enabled and disabled, and verifies that a combined
model still accepts an ordinary context. This rejection is superseded by the
trunkless ordinary context below.

Invalid-chain coverage now includes `LLAMA_TOKEN_NULL`, with unchanged serialized
sequence caches and a successful retry. The driver capacity tests observe actual
backend evaluation callbacks: ubatch 6 must execute exactly one packed four-token
chain graph with six input rows, including the two catch-up rows. Disabling chain
selection only for ubatch 6 preserved token parity but failed this new assertion.

Sequential-only MTP contexts retain the original graph-node budget. A validated
chain grows it using actual batch rows, retaining the high watermark across mode
switches. Dummy reservation graphs remain sequential. The regression executes
32- and 64-row chains and checks that shorter chains and mode toggles reuse the
reservation.

Measured synthetic-fixture metadata sizes (bytes per arena):

| Microbatch | Initial allocation | Prior eager policy, calculated | After chain execution |
| --- | ---: | ---: | ---: |
| 32 | 541,360 | 6,688,944 | 6,688,944 (32 rows) |
| 512 | 8,659,104 | 107,020,688 | 13,377,696 (64 rows) |

Initial and grown sizes are observed vector sizes. Prior sizes use the same ggml
allocator formula with the previous node budget. These are not total process RSS:
reservation and cached graphs have separate arenas, scheduler buffers are extra,
and a real model's tensor-count floor can change the result. `--fit` does not
establish peak compute memory for runtime chains.

The complete updated CPU suite passed both targeted CTest entries and the q8_0
variant. The rebuilt SYCL suite passed both registered A770 entries,
`test-qwen4exp-mtp-sycl-f16` and `test-qwen4exp-mtp-sycl-q8`, with
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, `SYCL_CACHE_PERSISTENT=1`, and
`GGML_SYCL_ENABLE_GRAPH=0`. These correctness runs supersede the earlier
follow-up's build-only SYCL evidence; their durations are not benchmarks.

## Trunkless ordinary context follow-up

The rejection above made `--fit` drop a head reached without an MTP speculative
type ("fitting without it") and turned `-m head.gguf` into a hard fit error. An
ordinary context on an MTP-only head now initializes instead:

- The trunk graph has a zero-block branch. The wide residual is `hc` copies of
  the token embedding, the absent trunk output mixer is skipped, and the shared
  LM head produces the logits. Hidden-state export keeps its masked and
  unmasked row contract.
- `llama_model::create_memory` gives that context a plain attention KV cache
  whose filter rejects every layer, following the existing MTP-on-hybrid
  pattern. Position tracking, rollback and state save/restore work; the cache
  holds no layer tensors. `llama_get_kv_cache_type_k/v` return
  `GGML_TYPE_COUNT` for it instead of indexing a missing first layer.
- A head with no token embedding still fails context creation with an error,
  since no graph can be built from it.
- Context creation logs one warning naming the file as an MTP draft head and
  pointing at `-md` with `--spec-type draft-mtp`.

Three external lines of work were read for comparison; none builds a context in
this situation. Upstream draft ggml-org/llama.cpp#27836 requires the trunk
tensors, so a separate head does not load. The `freqmod/llama.cpp` branch
`qwen4exp-mtp` relaxes them the same way this loader does and leaves the
ordinary graph unguarded. Upstream PR #28104 (closed, unmerged) aborts in the
trunk graph with guidance to load the file through `-md`; that guidance is kept
here as the warning. Its note that a hybrid memory with an empty recurrent layer
set is unsuitable matches the plain-cache choice: on this fork the hybrid
wrapper allocated but refused partial rollback.

Control runs on CPU with the synthetic fixtures and `--fit on`, using the saved
pre-PR libraries and the PR head before this change (`e68c49ca7`):

| Scenario | Pre-PR libraries | `e68c49ca7` | This change |
| --- | --- | --- | --- |
| Ordinary context on head | SIGSEGV, exit 139 | null context, error | context, logits |
| Head as `-md`, `none`/`draft-simple`/`ngram-mod` | SIGSEGV, exit 139 | fit drops the head | fit measures both contexts |
| Head as `-m` | SIGSEGV, exit 139 | hard fit error | fit and context succeed |

The pre-PR libraries crash in every row, so the earlier rejection was not a
regression against `master`; it replaced a crash with an error.

The regression suite now checks that trunkless logits equal the host product of
the LM head and the token embedding, across a multi-token batch, a continuation,
a rollback and both hidden-export modes. It checks zero context memory against a
nonzero combined-model control, sequence and whole-context state round trips,
the cache type query, the public draft-simple driver against the head's own
greedy chain, and fit measurement of the head as a draft and as the main model.
Removing the embedding guard reproduced exit 139 in the embedding-less case.

With the original `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` on CPU, `llama-completion
-m head -n 8` and `llama-cli -m head -st -n 8` both exited 0 and generated text.
`llama-completion` printed the warning once and evaluated eight tokens;
`llama-cli` hides library warnings at its default verbosity. The same
`llama-completion` binary on the pre-PR libraries exited 139. The two local
workaround heads that carry a mixer under the trunk's `output_hc_*` names also
exited 0 with one warning. The generated text is meaningless by construction:
without a trunk each token depends on its predecessor alone.

The full CPU suite, its q8_0 KV variant, both targeted CTest entries,
`test-llama-archs` (131 cases) and `test-kv-cache-adaptive-mode` passed.
Both registered A770 entries, `test-qwen4exp-mtp-sycl-f16` and
`test-qwen4exp-mtp-sycl-q8`, passed with `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
`SYCL_CACHE_PERSISTENT=1` and `GGML_SYCL_ENABLE_GRAPH=0` while another
`llama-server` held the device; these are correctness runs, not timings.

Limits: other architectures with MTP-only heads keep their existing behavior
and were not examined. No real-weight run paired the head with the 54 GiB trunk
under a non-MTP speculative type, and no server process was started. Draft-simple
drafts from a trunkless head are valid to verify but carry no trunk context, so
their acceptance is expected to be poor; it was not measured.

## Borrowed tables and chain sampler follow-up

A merge had kept the context-level rule that a Qwen4Exp head without its own
`token_embd.weight` or `output.weight` needs `ctx_other`, and dropped the graph
code that then uses the target's tables (commit `4deec5587`). With a target
context supplied, the MTP graph aborted on its embedding assertion. The graph now
takes both tables from the model of `ctx_other`. A target that has none to lend,
or a missing target context, fails context creation with an error instead of an
abort.

`--fit` could not measure such a head: the head only builds next to a target
context, so the measurement failed and the fit continued "without it". The fit
now retries the extra model beside a measuring context of the main model, reusing
the existing parent-measurement path.

The regression suite builds the table-less fixture as an MTP context beside the
combined model. Sequential logits and hidden states, and a chain with two
catch-up rows and depth three, equal the canonical head (the fixtures seed
tensors by name, so the tables are identical). A target whose `output.weight` is
doubled yields exactly doubled logits, which shows the LM head is the target's.
The public driver produces the same drafts with either head, sequential and
chained. A target without tables is rejected, and fit initialization constructs
four contexts: target, head alone (refused), target as parent, head beside it.
Before the port the first of these tests aborted at `qwen4exp.cpp:618`, and the
fit test failed with "fitting without it".

Real weights on CPU: a copy of `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` with the two
tables removed (2.60 GiB instead of 3.85 GiB; each table is 0.629 GiB at Q8_0),
borrowing from a context of the unmodified head, reproduces the unmodified head's
MTP output bit for bit: 744,960 logits and the hidden states, three tokens,
synthetic hidden input.

Real trunk on the Arc A770, one launch per head, `--fit on --fit-target 1024`,
sequential `draft-mtp`, `--spec-draft-n-max 6`, three prompts twice at 192
tokens. Observed once; the two launches are not a paired benchmark.

| | head with own tables | table-less head |
| --- | --- | --- |
| trunk weights on the GPU | 9455.72 MiB | 10134.54 MiB |
| head weights on the GPU | 3291.18 MiB | 2647.04 MiB |
| head weights on the host | 644.14 MiB | none |
| drafted tokens accepted | 838 of 1810 (46.3%) | 857 of 1702 (50.4%) |
| decode, six requests | 9.63 to 14.38 t/s | 11.26 to 15.64 t/s |
| time to healthy | 108 s | 84 s |

The borrowed tables are the trunk's `token_embd.weight` (IQ4_XS) and
`output.weight` (Q6_K) in place of the head's Q8_0 copies. Acceptance did not drop
in this sample. Before the fit change the same launch aborted: the fit dropped
the head, the trunk took 12886.98 MiB of the GPU, and loading the head failed
with a SYCL allocation error.

The second fix closes a decode-time hole. The Qwen4Exp chain preflight computed
whether backend samplers were attached and did not use it, so a chained batch
with one output row reached sampling with packed `[token, probability]` rows in
place of vocabulary logits. On the synthetic fixture that decode returned 0; a
sampler that indexes the vocabulary can abort instead. The preflight now returns
-1 before the KV cache changes, and the invalid-chain test covers it with an
unchanged-cache check and a retry on the same context.

Both registered A770 entries and the CPU suites passed on the final source (see
the PR description for the run list).

Limits: the fit retry runs for any extra model whose first measurement fails,
which costs one more metadata-only load of the main model. A head paired with a
target of different width or vocabulary is not checked and will assert in ggml.
Pinning the draft to a device that does not hold the target's tables
(`--spec-draft-device`) was not tested. The dense Qwen3.5 chain packs its output
the same way and has no such preflight; that path predates this branch and was
not changed. The A770 acceptance figures come from a decode that is not
reproducible run to run, so differences of a few points are within noise.

## Second review follow-up

Two review passes on the PR head `9d6b9b790` led to the changes below. Each has
a regression test that failed before its fix, except where noted.

| change | failure before | test |
|---|---|---|
| On-device sequence state restores block-quantized K/V: the reader sized its views in blocks, the writer in elements (`llama_io_read_device`). | abort at `GGML_ASSERT(n_copy > 0 && n_copy % el == 0)` on the first restore with q8_0 KV | `test_on_device_state` under `--q8-kv` |
| A failed chain decode keeps its catch-up rows: they were cleared before `llama_process`. | the next draft decoded with a hole below its position and differed from the reference | `test_driver_failed_draft` |
| A target with a separate draft head skips its own MTP block (`common_model_params_to_llama`). | a combined target without the draft mixer failed to load although the block is never used | `test_separate_head_target` |
| Borrowed tables are checked against the head's embedding width and vocabulary size. | context creation succeeded with tables of another vocabulary | `test_borrowed_tables`, "mismatched target tables rejected" |
| Embeddings are exported `n_embd_out` wide, for the trunkless head and the full trunk. | abort, "tensor read out of bounds", on any embeddings-enabled context; the full-trunk case predates this PR | `test_ordinary_context`, "embeddings width" |
| Switching the hidden export on plans the compute buffers again (`set_embeddings_nextn`). | the export read back the per-stream RMS-normalized residual: the allocator let the mixer's norm overwrite it in place | `test_ordinary_context`, "hidden export survives the graph" |
| The chain row high-water mark doubles instead of growing by one. | 31 scheduler reservations for chains of 1 to 32 rows, 5 after | `test_chain_reserve_growth` |
| A chain runs inside the compute buffers of the ordinary graph. | none: added as a guard for the claim that `--fit` covers chaining | assertion in `test_chain` |
| The fit logs why an extra model is measured next to its parent and does not repeat the failed attempt. | not tested: logging and control flow only | none |
| A file with some trunk block tensors is a trunk: one that lacks others fails to load. The loader used to take a missing `blk.0.hc_attn_norm.weight` alone as the mark of a head-only file. Third review pass. | a combined file without that one tensor loaded and ran as a head, embedding straight into the LM head | "a trunk that lacks a tensor is rejected, not run as a head" |
| A chained decode on a context with embeddings enabled returns -1. Third review pass. | abort at `llama-graph.cpp:3887`, `missing result_norm/result_embd tensor` | `test_invalid_chain`, case 7 |
| A table-less head whose devices cannot use the buffer of a borrowed table is rejected at context creation. Fifth review pass. | abort at `ggml-backend.cpp:1169`, `pre-allocated tensor (output.weight) in a buffer (SYCL0) that cannot run the operation`, for a host-only head next to a target on the A770 | `test_borrowed_tables`, "borrowed tables in a buffer the head's devices cannot use are rejected" (`--backend SYCL0` only) |
| A Qwen4Exp MTP context also reserves its full ubatch without an output row, for the scheduler and for `--fit`. Sixth review pass. | the first catch-up ubatch planned after a draft needed 11.0 MiB more than was reserved on the real head (152.0078 against 141.0157 MiB) | `test_catchup_reservation` (the reservation is made); `--catchup-real-head <gguf>` (opt-in, real weights, failed before the fix on the A770) |
| Checkpoint margins are looked up by device in the target model's device list, not by position in the context's own list. Fourth review pass. | a device the margins were not given for took the first margin | `test_checkpoint_placement`, "a device the margins were not given for keeps the largest one" (`--backend SYCL0` only) |
| The server's checkpoint guard decides before every checkpoint update instead of once (`common_speculative_checkpoint_flags`). Added after a third review pass, on `f5b0150d4`. | the first decision was kept while a checkpoint outgrew its device copy | `test_checkpoint_placement`; its host branch needs a device that reports memory, so it only runs under `--backend SYCL0` |

The hidden-export finding was not in either review. It surfaced while testing
the embeddings width. The Qwen4Exp head applies the same per-stream RMSNorm to
its hidden input, which is why drafting was not visibly affected: normalizing an
already normalized residual changes it only through the epsilon term. Drafts
therefore are not expected to be bit-identical to earlier builds.

Results on the final source: CPU `test-qwen4exp-mtp` 75 PASS (f16) and 50 PASS
(`--q8-kv`); Arc A770 `--backend SYCL0` 76 PASS and 51 PASS (one case needs a
device), 0 new i915/xe fault lines. Both real head files on disk hold only block 48, the head block, besides
their shared tensors, so the new head-only rule classifies them as before; the
Q8_0 head still loads and generates on CPU with the head-only warning.

The two SYCL test cases have a 180 s timeout. A cold run, with an empty NEO
compiler cache directory, took 5.4 s for either case on the A770 (55 kernels
compiled), against 1.1 s to 1.7 s warm; with the cache switched off it took
5.2 s. The 200 s cold start quoted in `AGENTS.md` does not apply to this
fixture. Real head on CPU, chain depth 6, 16 and 64 at microbatch sizes from the
chain length up to 512: the compute buffers stayed at their reserved size
(6.6 MiB to 566.0 MiB depending on the configuration).

Real trunk with its Q8_0 head on the A770, build of `f5b0150d4` (before the
guard moved to every update), `--fit on --fit-target
1024`, ctx 16384, q8_0 K/V, one launch per case with six requests of 192
tokens: sequential drafting at depth 6 accepted 850 of 1750 drafted tokens,
chained drafting at depth 4 accepted 818 of 1292. Neither log holds an
allocation failure, an assertion or a failed chain decode, and the kernel log
gained no i915/xe fault line. Both were observed once and are not a benchmark.

Compute buffer of the draft context (sixth pass). Every `draft-mtp` launch of the
real trunk ended with the draft context at 152.0078 MiB against an expectation of
141.0157 MiB. It is one reallocation, at the first full catch-up ubatch that has
to be planned from scratch, which is the case after any draft: a one-row or
chain graph has replaced the reserved plan by then. A catch-up ubatch has no
output row. The scheduler's device copy of the output indices, 128 bytes with one
output row, is empty then; the two free blocks it kept apart merge, best fit
places the next tensors differently, and the 50 MiB expert output that sets the
peak ends up 11.2 MiB higher. A planner trace of both plans (taken in a second
session with a tracing copy of `ggml-alloc.c` preloaded) shows the same 148
allocator events in the same order with different offsets. The hidden export is
not involved, the reservations before and after the switch are identical, and a
CPU-only context does not grow, because it has no input copies.

The context now reserves that shape as well. Measured on the A770 with the Q8_0
head, SYCL0 compute buffer:

| Configuration | Reserved before | Reserved now | After a catch-up ubatch |
| --- | --- | --- | --- |
| head alone, ctx 4096, one row then 512 rows without output (test, run twice) | 148.5236 MiB | 152.0000 MiB | 152.0078 MiB |
| real trunk, `--fit on --fit-target 1024`, ctx 16384, short request then long prompt (one launch) | 141.0157 MiB | 152.0000 MiB | 152.0078 MiB |

The fit-time measurement of the head reports 152 MiB of compute memory where it
reported 141 MiB. No i915/xe fault line in either run.

Limits of this follow-up:

- The on-device restore fix was exercised through the full-state path on a plain
  K/V cache. No model whose `PARTIAL_ONLY` state carries quantized K/V (the
  server's speculative checkpoint path) was run.
- The failed-chain test injects the failure with a backend sampler, which makes
  the decode return -1 before it touches the cache. Allocation and compute
  failures rely on `llama_context::decode` removing the failed batch's cells,
  which was read from source, not triggered.
- Borrowed tables of the right shape from an unrelated model are not detected.
- Embeddings on Qwen4Exp now return the wide residual in front of the output
  mixer, `hc * n_embd` values per token. Pooled embeddings were not tested.
- The catch-up reservation is 8192 bytes short of what a decode plans, so a
  `draft-mtp` launch still logs one `does not match expectation` line (152.0078
  against 152.0000 MiB) and reallocates the draft buffer once. The reserved
  graph carries the attention mask of a full cache, a decode a smaller one; with
  the smaller mask the free block above the attention output is smaller than the
  one below it, the 8 KiB `hc_inject` tensor goes there, and the 20 MiB residual
  lands 8 KiB higher. Best fit is not monotonic in tensor sizes, so no reserved
  shape bounds every decode. Head shapes other than this one and the other MTP
  architectures were not measured; the extra reservation is limited to Qwen4Exp.

## User-reported real-head evidence

The user independently tested `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` on CPU at
`4a9092768`, using synthetic hidden state and three tokens. These results were
reported by the user and were not reproduced in this follow-up:

- The unmodified head aborted with pre-PR libraries (exit 134), but loaded and
  decoded with PR libraries using flash attention and q8_0 KV.
- All 744,960 PR logits matched the pre-PR head-mixer workaround bit for bit.
  A trunk-mixer substitute differed by up to 4.11 under pre-PR libraries and
  changed argmax in two of three rows; PR libraries gave the canonical result.
- Reverting only the adaptive-fit predicate reproduced SIGSEGV in `build_hc_mix`;
  the full PR measured both contexts successfully.

This supports the original three bug fixes on a real head. It does not validate
chaining on real weights or an end-to-end run with the 54 GiB trunk. The original
79-84% code acceptance remains an earlier workaround measurement, not a PR result.

## Not claimed

- No improvement to target-model multi-token verification cost or a speedup target.
- No successful hardware graph replay; the observed profile used direct execution.
- No full test-suite pass: `test-arg-parser` fails its MoE-cache default assertion
  at line 440 in both the original and modified CPU builds.
- Synthetic coverage does not establish long-context model quality, PLE behavior,
  or performance on the 55 GiB checkpoint.
- No new production dependency or installed-binary replacement. Tests create
  temporary synthetic GGUF files; build and diagnostic artifacts are under `/tmp`.
- Chain use retains additional graph metadata until context destruction. Total
  host RSS, peak chained compute memory, real-checkpoint memory impact, and LoRA
  execution were not measured by the follow-up tests.
