# Tessera runtime trace capture - full spec

Status: FULL SPEC, architect decisions landed 2026-08-04. Ready for
implementation. Architect direction: session replay ships as a first-class
ANALYSIS AND CURATION stage ahead of the anonymization stage, not as a
stopgap.

Depends on: the phase 1 training bridge (`collect_training_traces`,
`TesseraTraceStore`, `tessera-train-lk`), `docs/tessera-lk-training-design.md`,
`docs/tessera-usage-dataset-spec.md` (sections 7 and 8 bind this spec).

## 1. Purpose

Phase 1 (shipped) harvests CALIBRATION traces: `llama-imatrix --model-draft
--telemetry-out` runs trunk + drafter over a neutral corpus and emits
`llama.tessera.spec.v1` records. That optimizes the drafter for the corpus
distribution.

Phase 2 captures traces DURING EXECUTION: every speculative step the app runs
for the user becomes a training signal, so the drafter converges on the
user's actual distribution - prompt style, tool-call JSON, code, domain
vocabulary. Real usage traces are the only data that optimizes for what the
user actually generates.

Captured sessions also become the raw material for the dataset pipeline:
replayed, analyzed, curated, anonymized, and only then eligible for egress
(section 12). The pipeline order is fixed:

```
capture -> replay + analysis -> curation verdicts -> anonymization -> train / egress
```

Two framing facts that shape everything below:

1. Runtime capture is only meaningful if the runtime does speculative
   decoding. Today it does not: the Studio's on-device path is single-model
   greedy decoding with no drafter in the loop. The prerequisite for capture
   is runtime spec decoding...
2. ...which is the same work the flywheel's OUTPUT side needs anyway: a
   trained drafter is worth nothing to the user until the runtime executes
   with it. One piece of work, two payoffs (user-visible speedup + capture).

## 2. What already exists (do not rebuild)

- `llama.tessera.spec.v1` record schema + emitter
  (`common/speculative-calibration.cpp`): per-step records with
  `drafted/accepted`, `drafted_tokens`, `accepted_tokens`, `confidence[]`,
  and (topk > 0) `verifier_argmax`, `drafter_argmax`, per-position top-k
  token/prob arrays for both models.
- The full speculative machinery (`common/speculative.h/.cpp`):
  `common_speculative_init_from_params`, `begin`, `draft`, `process`,
  `accept`, plus every head type the fork supports (MTP, DFlash, DSPark,
  Eagle3, adaptive muxer).
- `tessera-train-lk`: consumes spec.v1 with topk > 0, densifies verifier
  top-N into LK labels. Provenance-agnostic: it does not care where a
  record came from.
- `TesseraTraceStore` + the training gate + idle scheduler (phase 1 bridge).
  Store files are `traces-YYYYMMDD-HHMMSS.jsonl`, filtered by the
  `traces-` prefix; `appendRun(jsonlPath:)` is the sole writer.
- The CLlama dlopen pattern: the shim resolves libllama at runtime, so the
  SwiftPM package builds and runs with no native library present and
  degrades gracefully when one is missing.
- `common_speculative_calibration_run` (common/speculative-calibration.h:177):
  a prompt-replay loop over a tokenized corpus that emits one record per
  step. This IS the replay engine the curation stage reuses (section 12).

## 3. Runtime facts (investigation, 2026-08-04)

Verified against the tree before writing this design:

1. `LlamaLLMProvider` (Swift actor) -> `cllama_shim.c` (367 lines, plain C):
   dlopens `libllama.dylib`, resolves ~20 raw `llama.h` symbols
   (`llama_decode`, `llama_sampler_*`), builds a GREEDY sampler chain,
   generates one token at a time. No draft model, no spec decoding, no
   common layer.
2. The shim never touches the `common` layer. That matters because the spec
   machinery and the telemetry emitter live there, not in libllama.
3. KEY FINDING: the build already ships `libllama-common.dylib` next to
   `libllama.dylib` (build/bin, versioned). `nm` confirms it exports all 30
   `common_speculative_*` symbols, including
   `common_speculative_calibration_run` and
   `common_speculative_init_from_params`.
4. Those exports are C++-mangled and take C++ types (`common_params &`,
   `std::vector`). The shim must NOT dlsym mangled names: a small extern-C
   surface compiled into libllama-common is the honest fix (section 5).
5. The calibration emitter currently lives fused inside
   `common_speculative_calibration_run` (which drives its own loop over a
   tokenized prompt). The runtime needs a TOKEN-EMITTING loop instead, so
   the per-step record serialization must be factored into a shared
   function. That is the only refactor of existing C++ this design proposes.
6. `llama_detokenize` is public API (include/llama.h:1245): token ids back
   to UTF-8 text. One more symbol for the shim to resolve; it powers the
   curation stage's decode step (section 12).
7. Fork defaults that this spec inherits: `params.speculative.draft.n_max`
   defaults to 3 (common/common.h:366); `params.n_telemetry_topk` defaults
   to 0 (common/common.h:554).

## 4. Architecture

Add a small extern-C runtime entry point to the common layer (compiled into
the existing `libllama-common.dylib`, no new artifact). Extend the shim to
dlopen that library with the same candidate pattern it uses for libllama.
`LlamaLLMProvider` uses the spec engine when a runtime draft model is
configured; otherwise today's single-model path, unchanged.

```
LlamaLLMProvider (Swift actor)
    |
cllama_shim.c --dlopen--> libllama.dylib        (today: single-model greedy)
    |
    +--------dlopen--> libllama-common.dylib    (new: spec mode)
                            |
                      tessera_rt_generate()      new extern-C surface
                            |
                      common_speculative_*       draft -> verify -> accept,
                            |                    all head types inherited
                      spec.v1 record emitter     SAME records as imatrix
                            |                    (factored, +provenance)
    on_token(piece) <-------+------> on_trace(jsonl line)
    |                                       |
    v                                       v
  streamed text               TesseraTraceStore.appendRuntime
                              -> traces-runtime-<date>.jsonl
                                      |
                                      v   (idle agent duties, in order)
                              replay + analysis stage (section 12)
                                      |
                              curation verdicts ledger
                                      |
                              anonymization stage (dataset spec section 8)
                                      |
                              +-------+--------+
                              v                v
                        local training    dataset egress
```

Rejected alternatives:

- REIMPLEMENT the AR spec loop in the shim with raw llama.h: duplicates
  400-700 lines of delicate KV-rollback/accept logic, drifts from the
  reference implementation, and would be re-reimplemented per head type
  (MTP/DFlash/DSpark) while `common_speculative` already supports them all.
  The acceptance criterion the LK loss optimizes must match the acceptance
  semantics the runtime actually uses - one implementation, not two.
- SHELL OUT to a CLI per generation (imatrix-style): process launch per
  turn, no persistent KV across a conversation, breaks interactive
  streaming, and the agent loop is latency-sensitive.
- NEW standalone dylib: unnecessary - libllama-common.dylib already exists
  and already exports the machinery; an additive extern-C surface inside it
  needs zero new build artifacts and zero new install story.

## 5. The C ABI contract

New file `common/tessera-runtime.h` (and `.cpp`), extern-C, compiled into
libllama-common:

```c
typedef struct tessera_rt tessera_rt;

typedef void (*tessera_rt_token_cb)(const char * piece, int32_t token_id,
                                    void * ud);
typedef void (*tessera_rt_trace_cb)(const char * jsonl_line, void * ud);

// Load trunk + drafter, build contexts and the spec handle.
// draft_max: max drafted tokens per step.
tessera_rt * tessera_rt_load(const char * trunk_path,
                             const char * draft_path,
                             uint32_t n_ctx,
                             int32_t  n_threads,
                             int32_t  n_gpu_layers,
                             int32_t  draft_max);

// Tokenize + decode the prompt, then generate with spec decoding.
// telemetry_topk: 0 = no trace emission (cheap path); > 0 = emit one
// spec.v1 record per spec step through on_trace. on_trace may be NULL.
// Returns tokens generated, or -1 on error.
int32_t tessera_rt_generate(tessera_rt * rt,
                            const char * prompt,
                            int32_t max_tokens,
                            int32_t telemetry_topk,
                            tessera_rt_token_cb on_token,
                            tessera_rt_trace_cb on_trace,
                            void * ud);

void tessera_rt_free(tessera_rt * rt);
const char * tessera_rt_last_error(void);
```

Contract details:

- Records emitted through `on_trace` are schema-identical to
  `llama-imatrix --telemetry-out`, plus additive fields:
  `"provenance":"runtime"` and `"sid":"<session uuid>"` (section 8).
  Existing consumers ignore unknown keys, so nothing breaks; the training
  driver needs no change.
- `tessera_rt_generate` writes its own generation loop on top of
  `common_speculative_begin/draft/process/accept` (the live-loop API that
  llama-cli and llama-server already use), and calls the factored record
  emitter per spec step. It does NOT reuse
  `common_speculative_calibration_run` directly (that one drives a
  prompt-replay loop), only its emitter and its accept bookkeeping shape.
- Accepted tokens are decoded to text pieces inside the C side so the
  Swift callback stream stays per-token, preserving the current UX
  (streaming does not burst per accepted block).
- `telemetry_topk == 0` must be genuinely cheap: no softmax scans, no
  top-k extraction, no on_trace calls. Capture off costs only the spec
  loop itself.

Emitter factoring (the one touch of existing code): move the per-step
record serialization out of `common_speculative_calibration_run` into

```cpp
std::string common_spec_telemetry_record(
    int32_t step, llama_token id_last,
    const llama_tokens & draft, size_t n_acc,
    /* per-position logits accessors or precomputed rows */
    int32_t topk, const char * provenance, const char * sid);
```

`calibration_run` keeps byte-identical output (golden test in section 14)
and `provenance`/`sid` default to NULL -> fields omitted, so existing
imatrix files are unchanged.

## 6. Shim extension

Same conventions as the existing libllama load, in `cllama_shim.c`:

- `cllama_load_spec_library(override)`: candidates in order - explicit
  override, `TESSERA_LLAMA_COMMON_DYLIB` env var, SIBLING of the already
  resolved libllama path (the pairing pattern: both ship from the same
  build), then bare `libllama-common.dylib` default loader search.
- `cllama_is_spec_available()`: non-zero once the spec symbols resolved.
- `cllama_engine_load_spec(trunk, draft, n_ctx, n_threads, n_gpu_layers,
  draft_max)` and `cllama_engine_generate_spec(eng, prompt, max_tokens,
  topk, on_token, on_trace, ud)`: thin wrappers over `tessera_rt_*`.
- `cllama_detokenize(model, tokens, n, out_buf, out_len)`: thin wrapper
  over `llama_detokenize` for the curation stage's decode step. Stateless,
  cheap, no context needed.
- Missing library or missing symbols -> clean failure string, and the Swift
  provider falls back to the single-model path. Degrade open, honest error.
- The SwiftPM stub is untouched: spec mode only lights up when the real
  dylibs are present, exactly like on-device inference today.

## 7. Provider and settings

New settings (TesseraSettings):

- `learningRuntimeDraftModel` (string, default ""): runtime drafter GGUF.
  AUTO-DERIVE (decision 2): when empty, the provider looks for
  `<base>-tessera-trained.gguf` next to the configured base model; if it
  exists, spec decoding turns on with it. An explicit path always wins
  (same doctrine as `TesseraTrainBinaryResolver`: an explicit override is
  never silently second-guessed). A sentinel value "-" disables the
  auto-derive for users who want a trunk-only runtime.
- `learningRuntimeCapture` (bool, default true): when spec decoding runs,
  emit traces (topk > 0). Off = spec decoding with telemetry_topk 0.
- `learningRuntimeCaptureTopk` (int, default 16): top-k depth for runtime
  records. 16 keeps records light at interactive volume; the replay stage
  deepens promoted records to training parity offline (sections 11, 12).
- `learningRuntimeDraftMax` (int, default 3): max drafted tokens per step.
  Default matches the fork's `params.speculative.draft.n_max` so runtime
  acceptance semantics match what calibration measured. Configurable from
  day one via a bounded stepper (Settings numeric-input pattern).

`LlamaLLMProvider.complete()`: if a runtime drafter resolves (explicit or
auto-derived) AND `cllama_is_spec_available()`, load/generate through the
spec engine; otherwise today's path. Nothing else changes: prompt
construction, tool fence parsing, actor isolation, and the unload/deinit
story all stay. The trace callback appends into a session buffer the
provider flushes to `TesseraTraceStore` per completed generation
(section 8).

Model lifecycle: trunk + drafter are loaded once per provider lifetime
(same as the single model today), not per call. Hot-swapping the drafter
after a training cycle completes is a follow-up (section 16); v1 reads the
drafter path at provider init.

## 8. Trace store contract

- Provenance values: calibration records carry NO provenance field (absence
  = calibration; imatrix emitter untouched). Runtime records carry
  `"provenance":"runtime"`. Replayed records (section 12) carry
  `"provenance":"replay"` plus `"replayed_from":"runtime"`. All three are
  additive fields; readers ignore unknown keys.
- Session grouping: runtime records carry `"sid":"<uuid>"`, one uuid per
  provider generation call. Device-local random identifier, never leaves
  the machine (runtime records are local-only, section 9), stripped when a
  session is promoted to the replay corpus. It exists purely so the curation
  stage can group a turn's steps into one unit. NO-KEY invariant holds: a
  sid maps to nothing but its own records - no device, account, or
  contributor identity anywhere in the chain.
- File naming: `appendRuntime(records:)` (new TesseraTraceStore method)
  writes `traces-runtime-<date>.jsonl`; calibration runs keep
  `traces-<date>.jsonl`; replay output writes `traces-replay-<date>.jsonl`.
  All match the existing `traces-` prefix, so `totalRecords()` counts all
  three - all are legitimate training fuel, and the training gate sees the
  combined total. Filename prefix is the first egress-filter line; the
  provenance field is the second.
- Training: `stageCombinedTraces` concatenates all provenances unchanged.
  A calibration + on-policy + replayed mix is a feature, not a bug.
- Retention: a rolling cap trims the OLDEST runtime files first when the
  runtime share exceeds a budget (default 200 MB). Calibration and replay
  files are never touched by the runtime trimmer. Quarantined sessions
  (section 12) are exempt from automatic retention entirely: they hold
  content the user may want to inspect, and only user-initiated purge
  removes them. `learningDataRetentionDays` applies to non-quarantined
  files of all provenances, as today.

## 9. Privacy and tiering (binding under the usage dataset spec)

- Runtime traces record verifier/drafter top-k TOKEN IDS over the user's
  actual text. Token sequences are reconstructable text. Under the dataset
  spec's own criteria that makes them Tier B: personal, and no amount of
  rates/scores reshaping changes that while token ids are present.
- Local training: fine - that is the entire point of phase 2, and it never
  leaves the machine.
- Egress: the v1 published dataset EXCLUDES runtime traces entirely. The
  anonymization stage transforms text; re-anonymizing token-id
  distributions in place is not a meaningful operation. If runtime-derived
  data is ever published, it leaves only as re-derived Tier A aggregates
  (acceptance rates, mean draft length, block-size histograms - no token
  ids, no distributions), and only after the curation stage has promoted
  the source session and the anonymization stage has transformed any text.
- Ordering invariant (binding): curation runs BEFORE the anonymization
  stage (architect direction, 2026-08-04), and the anonymization stage
  runs BEFORE any training loop consumes (dataset spec section 8). Session
  replay is how the app decides what is worth transforming at all.
- Invariant (test-enforced): no record with `"provenance":"runtime"` may
  reach dataset staging. A unit test asserts the staging filter drops them.

## 10. UI surface

- Learning dashboard, Drafter Training section: a "runtime capture" row -
  records this session, total runtime records, live acceptance rate
  (accepted/drafted across captured steps), and curation state: promoted /
  quarantined / pending session counts. Read-only, refreshes with the
  section's existing refresh token.
- Quarantine list: session date, token count, and the probe class that
  quarantined it (secrets / contact info / paths / model-mismatch), never
  the matched content itself. Purge action with the app's existing
  destructive-confirmation pattern.
- Settings, Model tab, Learning section: runtime drafter path field (with
  the same found/not-found status row pattern as the training driver, and
  showing the auto-derived value when the field is empty), capture toggle,
  topk stepper, draft-depth stepper. Caption: the runtime drafter is read
  when the Playground provider initializes.
- No new notification surfaces: capture and curation are continuous
  background plumbing, not terminal events. Training-cycle pings already
  exist.

## 11. Volume and performance budget

- Top-k extraction cost: per spec step, softmax + top-k over verifier
  logits at n_dft + 1 positions, O(positions x n_vocab) on logits already
  resident in memory. At draft_max 3, vocab ~260k, topk 16 this is ~1 ms
  per step on CPU - small against a verifier forward, not zero.
  Capture off (topk 0) skips all of it.
- Runtime topk 16 vs calibration 64: smaller records (~1-2 KB/step at
  draft 3), thinner distribution tail. The tail is recovered OFFLINE: the
  replay stage re-runs the verifier over the decoded session and re-emits
  promoted records at topk 64 (topk deepening, section 12). Interactive
  capture pays for 16; training gets 64.
- Session volume: a heavy 4096-token session at draft 3, topk 16 is on the
  order of 2-4 MB of JSONL. The 200 MB rolling cap bounds the share.
- Latency: no extra forwards; telemetry reads logits the accept step
  already produced. Accept decisions are unaffected by capture on/off.
- Replay cost: one verifier forward pass over the decoded session (same
  cost as an imatrix calibration run of equal length). Idle-gated,
  on-power, batched - same scheduling envelope as training itself.

## 12. Session replay: analysis and curation stage

Architect direction (2026-08-04): session replay is not a stopgap. It is
the analysis stage that curates captured sessions BEFORE the anonymization
stage touches them. The anonymization stage decides HOW text leaves; the
curation stage decides WHETHER it deserves processing at all.

### 12.1 Why replay first

Raw runtime records are token ids + distributions. They cannot be
quality-scored, sensitivity-probed, or deduplicated in that form. Replay
materializes them into analyzable text and re-derived statistics, and it
does so DETERMINISTICALLY: the verifier is causal, so feeding the exact
accepted token sequence back through it reproduces the verifier
distributions at those positions faithfully.

Honest caveat: the DRAFTER's conditioning during replay matches the
fully-accepted-prefix case only (the drafter in replay never sees its own
rejected drafts mid-step). Replayed records are close to live capture, not
identical. That is acceptable for both uses: the LK training label comes
from the verifier distribution (exact under replay), and the curation
statistics (acceptance rates, lengths) are conservative estimates.

### 12.2 Replay mechanics (zero new native binaries)

Decode step: the curation stage reads a runtime trace file, groups records
by `sid`, and decodes each session's accepted token sequence to UTF-8 via
`cllama_detokenize` (the new shim wrapper over `llama_detokenize`). Output:
per-session text segments plus their recorded step statistics.

Recompute step: the decoded session text becomes a corpus for the EXISTING
machinery - the curation stage drives `collect_training_traces` (phase 1
tool: `llama-imatrix --model-draft --telemetry-out`) over it, requesting
topk 64. Because `common_speculative_calibration_run` is itself a
prompt-replay loop, this re-derives full per-step records at training
parity. Output: `traces-replay-<date>.jsonl` with
`"provenance":"replay", "replayed_from":"runtime"`.

No new CLI tool, no new library: decode is one shim symbol, recompute is
the phase 1 tool pointed at a different corpus.

### 12.3 Analysis pass

For each decoded session the stage computes a scorecard:

- QUALITY: acceptance rate, mean accepted run length, repetition ratio
  (n-gram self-overlap), token count floor (sessions below a minimum are
  noise), truncation/garbage heuristics (EOS ratio, out-of-distribution
  piece rate).
- SENSITIVITY: the dataset pipeline's deterministic scrubber pattern set
  (dataset spec section 8, phase 1) run READ-ONLY as a probe. Any hit -
  secrets, keys, credentials, addresses, phone numbers, account
  identifiers - is a quarantine signal. The probe shares its versioned
  rule set with the scrubber, so what the wall catches and what curation
  flags can never drift apart.
- DUPLICATION: normalized n-gram fingerprints checked against the store;
  near-duplicate sessions (retries, repeated prompts) collapse to one.
- COMPATIBILITY: the replayed session's tokenizer/vocab size must match
  the current trunk; a model update invalidates old sessions (verdict:
  drop, reason model-mismatch), since their token ids no longer decode.

### 12.4 Verdicts and ledger

Every analyzed session gets exactly one verdict, appended to an
append-only ledger `<learningStoreDir>/curation-ledger.jsonl`
(schema `llama.tessera.curation.v1`):

```json
{"schema":"llama.tessera.curation.v1",
 "sid":"<uuid>", "verdict":"promoted|quarantined|dropped",
 "reasons":["low-repetition","probe:none","dedup:kept"],
 "score":{"acceptance":0.71,"tokens":1204,"repetition":0.06},
 "anonymizer_required_version":">=1",
 "ts":"2026-08-04T22:30:00Z"}
```

- PROMOTED: passes quality floor, zero sensitivity hits, not a duplicate.
  Eligible for topk-deepened replay records (local training) and for the
  anonymization stage (dataset path). The sid is stripped at promotion:
  downstream artifacts carry no session identifier at all.
- QUARANTINED: any sensitivity probe hit. Stays local, exempt from
  automatic retention, visible in the dashboard quarantine list, excluded
  from replay, training, and egress. Only user-initiated purge removes it.
  A future re-analysis with a newer scrubber version can promote it
  (ledger gets a new entry; verdicts are append-only, latest wins).
- DROPPED: below quality floor, duplicate, or model-mismatch. Retention
  trims these first.

The ledger is device-local analysis metadata. It never leaves the machine:
no manifest, batch, or egress artifact references it.

### 12.5 Stage contract

- Runs under the idle agent's duties (same envelope as the anonymization
  stage and training sweeps): idle-gated, on-power, resumable, and each
  sweep records honest progress.
- Order: capture -> analysis -> verdict -> anonymization -> consumption.
  Nothing skips a stage. A training sweep that finds uncurated runtime
  records simply ignores them; it consumes calibration + promoted replay
  records only.
- The curation stage is also the natural ANALYSIS surface for calibration
  traces later (follow-up, section 16): calibration records come from a
  corpus the app chose, but the same scorecard can audit corpus quality.

## 13. Sequencing

Work units, in dependency order:

- S1: emitter factoring + golden test (C++). `common_spec_telemetry_record`
  out of `common_speculative_calibration_run`; calibration output
  byte-identical; provenance/sid NULL-omitted.
- S2: the big one. `tessera_rt_*` entry point + shim spec-library loading
  + `LlamaLLMProvider` spec mode + settings (incl. auto-derive).
  User-visible payoff even with capture off: faster generation with the
  trained drafter. Independently shippable.
- S3: capture plumbing. on_trace -> session buffer -> `appendRuntime`,
  `traces-runtime-` naming, sid stamping, dashboard capture row, rolling
  cap with quarantine exemption.
- S4: replay + curation stage. `cllama_detokenize` shim wrapper, analysis
  scorecard, verdict ledger, replay driver over `collect_training_traces`,
  topk deepening, dashboard curation row + quarantine list.
- S5: egress guard. Staging filter drops `provenance:runtime` (and
  unpromoted `replay`) records + the exclusion test. One filter, one test,
  but MUST land before the dataset pipeline ingests anything.

S4 depends on S3 for runtime inputs but NOT on S2 being in production use:
replay works over any decoded session text, so the stage can be developed
against manually exported conversations. S5 lands whenever, as long as it
precedes first dataset ingestion.

## 14. Testing

C++:
- Golden test for the factored emitter: records serialized by the new
  shared function are byte-identical to today's calibration output for the
  same inputs (provenance/sid omitted when NULL).
- Runtime generate smoke test with the tiny trunk + tiny drafter fixtures
  already used by test-telemetry-jsonl / test-spec-calibration: records
  parse, `accepted <= drafted`, `accepted_tokens` consistent,
  `provenance:"runtime"` present.
- Parity test: same corpus through `llama-imatrix --telemetry-out` and
  through `tessera_rt_generate` (topk matched) -> same schema, same
  acceptance counts.
- Topk-deepening test: replay of captured topk-16 records at topk 64
  reproduces the topk-64 verifier rows of a direct topk-64 calibration
  run on the same corpus (verifier side exact; drafter side tolerance).

Swift:
- Shim degradation: spec library absent -> provider uses today's path, no
  behavior change (the existing test surface keeps passing untouched).
- Store: appendRuntime naming, combined counting, sid stamping,
  runtime-first trimming, quarantine exemption from retention.
- Curation: scorecard thresholds on synthetic sessions; probe-hit session
  quarantines; duplicate collapses; model-mismatch drops; ledger
  append-only latest-wins decode; sid stripped from promoted output.
- Egress: staging filter drops `provenance:runtime` records (the section 9
  invariant).

## 15. Decisions landed (2026-08-04)

1. Runtime topk default: 16. Light records at interactive volume; the
   replay stage deepens promoted sessions to 64 offline (section 11), so
   training keeps calibration parity without paying for it online.
2. Runtime drafter default: AUTO-DERIVE. Empty setting + a
   `<base>-tessera-trained.gguf` next to the base model -> spec decoding
   on with the trained drafter. Explicit path always wins; sentinel "-"
   disables auto-derive.
3. Session replay: SHIPS, first-class. Not a stopgap: the permanent
   analysis and curation stage ahead of anonymization (section 12).
4. Draft depth default: 3, matching the fork's
   `params.speculative.draft.n_max`, configurable from day one.

## 16. Follow-ups (not v1)

- Hot-swap the runtime drafter when a training cycle completes, without
  reinitializing the provider.
- Multi-head runtime routing: the adaptive muxer already lives in
  `common_speculative`; the entry point inherits it once a unified
  multi-head GGUF lands (drafter extension roadmap).
- Tier A aggregate derivation from runtime traces for a future dataset
  version, if the no-egress-for-runtime rule is ever revisited.
- Per-scope capture granularity (e.g. capture Playground turns but not
  agent tool-call chains), if the consent model grows scopes.
- Curation pass over CALIBRATION traces: same scorecard auditing the
  bundled corpus's quality, using the replay stage.
