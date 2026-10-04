# Qwen4Exp MTP correctness, 2026-10-04

The MTP graph now uses the draft head's own `blk.N.nextn.hc_head_*` mixer.
All three mixer tensors are required when loading MTP weights. Ordinary loading
with MTP disabled still permits their absence. Adaptive MTP fitting now selects
an MTP context for both separate-head and shared-model configurations.

Qwen4Exp also supports opt-in chained drafting through `--spec-chain`. The chain
preserves the full hyper-connection state, uses the full vocabulary, and computes
confidence over the same top-10 candidates as sequential drafting. It reuses the
ordinary attention builder for cache rotations, padding, masks, and writes.

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
model still accepts an ordinary context.

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
