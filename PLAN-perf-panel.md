# PLAN v3: live performance panel in the llama-server web UI (Kimi-K3, port 18090)

Requirement (verbatim order kept): with a "Display performance" checkbox enabled, the chat UI shows in real time:
1 CPU busy %, 2 GPU busy %, 3 occupied system memory, 4 occupied GPU memory, 5 average generation speed ("0.53 t/s"),
6 average prompt processing speed, 7 NVMe loading bandwidth, 8 prediction accuracy, 9 preload wasted %.
Constraints: < 1% decode overhead, nothing per-token heavy, one lazy background sampler at 1 Hz, compact panel (dashboard,
not a trace). User decisions D1-D4 (section 9) are final and folded in below.

## 1. UI source, build, serving
- The UI source is in-tree: `tools/ui` (SvelteKit 5, Tailwind, shadcn-svelte, vitest + playwright; node 24 / npm 11 and
  playwright chromium installed). `scripts/ui-assets.cmake` priority: (1) `tools/ui/dist` if present (overrides everything,
  the launcher warns if it exists), (2) `npm run build` when `LLAMA_BUILD_UI=ON`, (3) HF `dist.tar.gz` (current build).
  The bundle is embedded in `libllama-ui.a`; `--path DIR` serves a directory instead.
- UI changes live as source in `tools/ui/src` on the branch (no vendored dist). Build with `-DLLAMA_BUILD_UI=ON`; `npm ci`
  runs once into `build/tools/ui/ui-src/node_modules` (network once), later builds are offline and skipped when sources are
  unchanged. `ui-assets.cmake`: with `BUILD_UI=ON` an npm failure is `FATAL_ERROR` (never silently embed the HF bundle).
- Launcher guards (`kimi-k3/run-server.sh`): die if `build/tools/ui/.ui-stamp` exists (HF bundle embedded) or if
  `build/tools/ui/ui.cpp` is newer than the binary; both skipped when `UI_PATH` is set (new env -> `--path`, the no-relink
  dev loop: `LLAMA_UI_OUT_DIR=/p npm run build`). Rebase cost: an upstream `package-lock.json` change makes the next build
  run `npm ci` (network); offline workaround: `UI_PATH` with the last built dist, or `-DLLAMA_BUILD_UI=OFF` temporarily.
- D2: the launcher passes `--ui-config '{"displayPerformance":true}'`. This is applied only on a browser's FIRST visit
  (settingsStore applies server defaults once); browsers that already have a saved config must tick the box once.

## 2. Metrics (sampler at 1 Hz; "window" = last 60 s; "total" = since the sampler baseline, taken right after model load)
| # | Headline | Definition / source | Expand (tap/hover) |
|---|---|---|---|
| 1 | `CPU 87%` | whole machine, /proc/stat deltas: (total - idle - iowait)/total | process share: /proc/self/stat (parse after the last ')'), utime+stime / (dt x ncpu), `sysconf(_SC_CLK_TCK)` |
| 2 | `GPU 3%` | NVML via `dlopen("libnvidia-ml.so.1")`, `nvmlDeviceGetUtilizationRates(dev 0).gpu`, device-wide | device name |
| 3 | `RAM 241 / 251 GB` | D1: occupied = MemTotal - MemFree (page cache counted; it holds the experts) | split: used (MemTotal - MemAvailable), cached, free |
| 4 | `VRAM 11.2 / 12 GB` | `nvmlDeviceGetMemoryInfo` used/total, device-wide | - |
| 5 | `gen 0.53 t/s` | A3: while generating = tokens so far / time since first token of the current request; idle = last request's average (never 0.00 when idle; `n/a` before the first request) | window = tokens / active seconds (seconds with >= 1 token), lifetime |
| 6 | `pp 14.8 t/s` | B5/C1 live: tokens_in_flight x layers_done / n_layer / elapsed for the running batch, layers_done from `llama_decode_progress()` (predictor-independent, both offload modes); idle = last request's prompt average | `layer 42/93`, window, lifetime |
| 7 | `NVMe 4.2 GB/s` | /proc/diskstats sectors read x 512 for the device backing the model file (stat st_dev -> /sys/dev/block -> `md0`), 1 s rate | 10 s smoothed, device |
| 8 | `guesses right 41%` | PRECISION (user decision): hits / guesses scored, window | recall (hits / actual experts), lifetime of both |
| 9 | `preload wasted 24%` | D4 + R1: LIFETIME since the post-load baseline (no window: eviction-based crediting resolves 55-200+ s later, a window would read ~0% in decode and > 100% idle): confirmed wasted bytes / total bytes the process read from disk, clamped 0..100, `n/a` while the denominator is 0 | "still pending X GB / credited Y GB", predictor share = bytes_read_predictor / bytes_read_total, the definition text |

## 3. Server side
a) Lock-free counters (relaxed atomics, no new locks on the graph path):
- `server_perf_counters`: `n_gen_live` (per token), `n_prompt_live` (per batch), per-request totals and last-request
  (n, t_us) for gen and prompt, `batch_in_flight {n_tokens, t_start_us}` set around `llama_decode` of a prompt batch.
- Predictor: `hits`, `actual`, `guesses` (3 adds at scoring in `on_layer`, already under `mtx`) plus the byte counters in 3b.
- C1 layer progress (primary, predictor-independent): `on_moe_selected` returns early for `n_tokens > 4` / `moe_cache`
  (kimi-k3.cpp:364), so the predictor's `layer_seen` never moves in prompt batches. For `n_tokens > 4` it instead inserts a
  trivial CPU custom op (src[0] = a lazy expert tensor pins it to the CPU backend) that stores `il` into a relaxed atomic
  owned by `llama_context` (`progress_il`, plus `progress_n_layer`, `progress_n_tokens`, `progress_t_start_us` set in
  `decode`); `sched_copy_experts` is a second writer with OFFLOAD=1. Cost: one extra CPU graph split per layer, negligible
  against a 76-108 s batch (measured in the overhead e2e). Exposed by `llama_decode_progress(ctx, &out)`.
b) Preload bytes (`src/llama-moe-predict.{h,cpp}`):
- Loaded bytes per guess: in `prefetch_expert` read `read_bytes` from `/proc/thread-self/io` (fd per worker, `pread` at
  0) before/after the expert's madvise loop (2 reads per expert, not per piece). B1: `drain_high()` between pieces runs
  nested HIGH loads; a `thread_local nested_bytes` accumulator is subtracted from the outer delta. The delta is whatever
  the kernel submitted (not assumed equal to the slice size); the kernel-contract test (5b) pins this once.
- Per layer arrays sized n_layer x n_expert (93 x 896 here; atomics, 5 x 8 B each = 3.3 MB): `pred_tag[e]` (step of
  the current guess, set when guesses are issued), `pred_bytes[e]`, `pend_bytes[e]`, `pend_clock[e]`, `pend_cached[e]`.
- Worker, LOW item (il,e): `b` measured; `if pred_tag[il][e] == w.step: pred_bytes[il][e] += b else bytes_late += b`;
  `bytes_read_predictor += b`. If `pend_bytes[il][e] > 0` (an earlier miss still pending) the re-guess rule: `b == 0`
  (or sampled mincore resident) -> keep pending, refresh clock; else the old pending bytes -> `bytes_wasted`, pend = b.
  Worker, HIGH item (actual expert of the current layer, B2): only `bytes_actual += b` (needed, never late/wasted); it
  does not resolve pending (C2: a worker probe is biased toward hit and dropped items would never resolve).
- Scoring in `on_layer` (graph thread, under `mtx`, before the layer's expert matmuls): for each guess
  `b = pred_bytes[e].exchange(0)`, then B3 `pred_tag[e] = 0` so in-flight late loads land in `bytes_late`; selected ->
  `bytes_hit += b`; not selected -> `pend_bytes[e] += b`, `pend_clock[e] = io_clock` (process read_bytes published by
  the sampler, 1 s resolution), `pend_cached[e] = Cached now`. C2: then for each `e` in `act` with `pend_bytes[e] > 0`,
  a sampled residency probe: `mincore` on 8 x 64 KB sub-ranges per expert tensor (thread-local scratch), resident ->
  `bytes_hit += pend`, `bytes_credited += pend` (D4 credit), else `bytes_wasted += pend`; pend = 0. Cap: at most n_used
  (16) probes per layer (~16 x 3 mincore calls, tens of us), included in the overhead e2e. mincore guard as in
  llama-mmap.cpp:1218 (file neither owned nor writable -> mincore lies "resident"): then fall back to "worker delta == 0
  -> resident", i.e. the HIGH load of that expert resolves pending by its measured bytes.
- Expiry (sampler thread, 1 Hz, ~100 us scan): pending with `io_clock_now - pend_clock > min(pend_cached, Cached now)`
  (the whole page cache turned over since the read -> evicted for sure) -> `bytes_wasted`. With ~200 GB cached and a
  few GB/s this takes minutes; it rarely fires within a session and mainly cleans up after long idle/prompt phases.
  Dropped/stale items issue no madvise -> 0 bytes. Observe mode issues no madvise -> `preload` null; precision/recall stay.
- Headline 9 (R1) = (bytes_wasted + bytes_late) / process read_bytes, both lifetime since the baseline, clamped 0..100
  (D4 denominator = all data loaded from disk: faults, CPU MoE prefetch, predictor, O_DIRECT reader).
c) Public API: `llama_moe_predict_stats(const llama_model *, llama_moe_predict_stats *)` -> `{enabled, mode, n_layer,
   layer_seen, step_seen, hits, actual, guesses, bytes_hit, bytes_wasted, bytes_late, bytes_actual, bytes_read_predictor,
   bytes_pending, bytes_credited}`; atomics only; `enabled=false` for dense models / predictor off. Also
   `llama_moe_predict_set_io_clock(model, read_bytes, cached)` and `llama_decode_progress(ctx, &{il, n_layer, n_tokens,
   t_start_us})` (C1; `n_tokens == 0` when no batch is running).
d) Sampler `tools/server/server-perf.{h,cpp}` (in the `server-context` library, link `${CMAKE_DL_LIBS}`): constructed
   right after model load (B4: baseline snapshot of /proc/self/io and the predictor counters at construction, independent
   of polling), joined before `llama_init.reset()`. Lazy: samples only while `/perf` was polled in the last 10 s. A5: a
   poll that wakes it ticks immediately, ticks again after 500 ms and answers with non-null rates (first response <= 1.5 s
   after enabling, `ts_ms` age < 2.5 s). Each tick ~200 us: /proc/stat, /proc/meminfo, /proc/diskstats (one line),
   /proc/self/io, /proc/self/stat, 2 NVML calls (dlopen once, failure -> gpu null forever, logged once), predictor stats,
   server atomics; 60-entry ring; pending expiry (3b). Computation is a pure function `server_perf_compute(ring, baseline,
   now) -> json` so tests inject synthetic ticks. Linux-only (`#ifdef __linux__`), elsewhere all groups null.
e) `GET /perf` (new route; API-key protected like `/slots`; no task-queue round trip; added to the `log_server_request`
   skip list; router mode: proxied to the active child like `/metrics`). JSON (bytes, seconds, pct 0..100):
   `{ts_ms, window_s:60, cpu:{busy_pct, proc_pct, n_cpu}, gpu:{busy_pct, mem_used_bytes, mem_total_bytes, name}|null,
   mem:{total_bytes, occupied_bytes, used_bytes, cached_bytes, free_bytes}, gen:{tps_headline, tps_window, tps_total,
   tps_last, generating, n_tokens}, prompt:{tps_headline, tps_window, tps_total, tps_last, processing, layer, n_layer,
   n_tokens}, nvme:{device, read_bps, read_bps_10s}|null, predict:{mode:"observe"|"prefetch", precision_window,
   precision_total, recall_window, recall_total, hits, actual, guesses}|null, preload:{wasted_pct (lifetime, null when
   bytes_read_total == 0), bytes_wasted, bytes_late, bytes_hit, bytes_credited, bytes_actual, bytes_pending,
   bytes_read_predictor, bytes_read_total}|null}`. Note: `recall_total` and the predictor's log line both start from zero
   after load (the persisted state file keeps slot sizes, not counters), so they are directly comparable.

## 4. UI side (tools/ui)
- Setting `SETTINGS_KEYS.DISPLAY_PERFORMANCE = 'displayPerformance'`, checkbox in the Display section, default false,
  help text; persisted/exported like every key. `API_PERF` constant; vite dev proxy entry for `/perf`; eslint sort-objects.
- `stores/perf.svelte.ts` owns polling: `setEnabled(bool)`, its own `visibilitychange` listener, immediate fetch on
  enable, 1 s interval while enabled and visible, 0 fetches while hidden or disabled, 5 s cadence after 3 consecutive
  failures, keeps polling on 503 (model loading), stops with `unavailable=true` on 404 (upstream server). `+layout.svelte`
  only has `$effect(() => perfStore.setEnabled(config.displayPerformance))`. `services/perf.service.ts` =
  `apiFetch('./perf', {authOnly:true})`. Types in `types/api.d.ts`.
- `components/app/perf/PerfPanel.svelte`: desktop (>= md): card in the existing fixed bottom-right stack above the build
  badge, 2-column `dl`, `text-[11px] tabular-nums`, 9 rows in the user's order, `<section aria-label="Performance">`,
  no `aria-live`, `pointer-events-none` except the expand toggle; the expand (tap or hover) shows every tooltip value of
  section 2 and the wasted definition. Mobile (< md, `deviceStore.isMobile`): one-line `text-xs` strip above the chat form
  positioned with `--chat-form-bottom-position` + `env(safe-area-inset-bottom)`; that variable is an inline style on the
  `.chat-screen` div (ChatScreen.svelte ~297), so the strip renders inside ChatScreen (or the variable is lifted); tap
  expands to the full card; never covers the composer.
- `utils/perf-format.ts` (pinned): memory base-1024 shown as "GB", 0 decimals if integral or >= 100 else 1
  (`241 / 251 GB`, `11.2 / 12 GB`); bandwidth base-1000 (`4.2 GB/s` >= 1e9, else `640 MB/s`); t/s 2 decimals < 10 else
  1 (`0.53 t/s`); percent integer (`87%`); null -> `n/a`.

## 5. Tests (intent-based numeric assertions; Opus writes unit tests with the code, Sonnet runs everything + e2e)
a) C++ `tests/test-moe-predict.cpp` (+ `params.test_bytes_per_expert` and a test residency hook): n_expert 8, guesses
   {0,1,2,3}, actual {0,5}: hits 1, actual 2, guesses 4 -> precision 0.250, recall 0.500 exactly; bytes: 1xB hit, 3xB
   pending; a later step selects 2 with the residency hook saying resident -> `bytes_credited += B` (and hit), saying
   evicted -> `bytes_wasted += B`; 1 re-guessed with delta B -> old B wasted; 3 re-guessed with delta 0 -> stays pending;
   HIGH bytes -> `bytes_actual` only; stale item -> 0 bytes; load after scoring -> `bytes_late`; nested HIGH inside a LOW
   load excluded from the LOW delta; overlapping router/popularity guess sets keep recall <= 1; `enabled=false` without a
   predictor; `llama_decode_progress` advances per layer for an n_tokens > 4 batch with the predictor off.
b) Kernel contract (Sonnet, worker-thread style, on the real model file path and a 256 MB scratch file on /mnt/k3):
   evict (`posix_fadvise DONTNEED`), `madvise WILLNEED` -> `/proc/thread-self/io read_bytes` delta in [0.9, 1.1] x size
   and within 10% of the md0 diskstats delta; warm repeat < 1 MB. Failure -> switch loaded-bytes to `mincore` too.
c) `tests/test-server-component.cpp` (A4): `server_perf_compute` with synthetic ticks: partial window, counter reset,
   idle gen (headline = last request), running batch (layer 42/93 -> pp live to the digit), precision/recall windows and
   the lifetime wasted% exactly as shipped; R1 cases: bytes read at tick 1 and expired at tick 100 count in the lifetime
   ratio; an idle stretch with expiries never exceeds 100%; a pending entry of B bytes credited at tick 70 ->
   bytes_pending -= B, bytes_hit and bytes_credited += B, wasted_pct unchanged; the same entry expired instead ->
   bytes_wasted += B and wasted_pct rises by exactly B / bytes_read_total; denominator 0 -> null; null groups.
d) pytest `tools/server/tests/unit/test_perf.py` (tinyllama): schema + ranges; 401 without key; after a completion of N
   tokens `gen.n_tokens == N` and `tps_last` within 5% of `timings.predicted_per_second`; first `/perf` after idle
   answers <= 1.5 s with non-null cpu/nvme rates and `ts_ms` age < 2.5 s; sampler thread CPU ticks unchanged over 5 s
   without polling; 503 handling by the UI is covered in 5e.
e) UI vitest: unit formatter tests assert the literal strings `0.53 t/s`, `241 / 251 GB`, `11.2 / 12 GB`, `4.2 GB/s`,
   `640 MB/s`, `87%`, `n/a`. Client tests (own `/perf` mock; the setup file passes `/perf` to the mock, not the real
   fetch): checkbox default false and persists `displayPerformance:true` in `CONFIG_LOCALSTORAGE_KEY`; immediate fetch on
   enable, N fetches after N s, 0 while `document.hidden`, 0 after disable, 5 s cadence after 3 failures, continues on
   503, stops on 404; window empty but last = 0.53 renders `0.53 t/s`; `deviceStore.isMobile` mocked (pattern
   `tests/client/context-panel-mobile.svelte.test.ts`) renders the strip and expands on tap.
f) E2E (Sonnet; D3: it starts its own Kimi-K3 server via run-server.sh with `PORT=18091` when no llama process runs):
   during a 1.6k-token prompt poll at t = 30..90 s: `prompt.tps_headline` non-zero every tick after 10 s and within 20%
   of the final `timings.prompt_per_second`, `layer` monotonic, run with OFFLOAD=0 and OFFLOAD=1; during a ~60-token decode: per-tick
   `nvme.read_bps` within 15% of an independent /proc/diskstats md0 reader (bytes, base-1000 only for display), gpu
   within 10 points / 5% of `nvidia-smi -lms 1000`, cpu within 5 points of an independent /proc/stat reader, mem within
   1% of `free -b`, `gen.tps_last` within 5% of `timings.predicted_per_second`, `recall_total` within 0.01 of the
   predictor's log line, `bytes_hit + bytes_wasted + bytes_late + bytes_actual + bytes_pending <= bytes_read_total`,
   `bytes_read_total` within 10% of the md0 delta, 0 <= wasted% <= 100, `bytes_credited <= bytes_hit`; observe mode run
   -> `preload` null, `predict.mode == "observe"`. Live seam
   (A7): playwright 1280x800 and 390x844 against that server: tick the checkbox, panel strings == `format(/perf JSON of
   the same second)`, gen row changes between ticks, all 9 values visible and the composer not obscured on the phone
   size. Overhead: 3x64-token decodes polling vs 3x not (includes the scoring-time mincore probes), mean tok/s difference
   < 1%; one 1.6k prompt with vs without the progress op within 1%; sampler thread < 1% of one core.

## 6. Degraded modes
gpu null (no NVML / non-NVIDIA), predict + preload null (dense model or predictor off), preload null in observe mode,
nvme null (device unresolvable); pp live needs no predictor (the progress op is model-side; models without it fall back
to batch granularity, `layer` null); the panel prints `n/a` per field. Upstream server without `/perf`: "unavailable", polling stops. Router mode: proxied to the active model.

## 7. Files
- modify: `src/llama-moe-predict.{h,cpp}`, `include/llama.h`, `src/llama-model.cpp`, `src/llama-context.{h,cpp}` (progress
  atomics, `sched_copy_experts` writer), `src/models/kimi-k3.cpp` (progress op), `tools/server/server-context.{h,cpp}`, `server.cpp`, `server-http.cpp` (log skip), `server-models.cpp` (proxy),
  `tools/server/CMakeLists.txt`, `scripts/ui-assets.cmake`, `kimi-k3/run-server.sh`, `tools/ui/vite.config.ts`,
  `tools/ui/src/lib/constants/{settings-keys,settings,api-endpoints}.constants.ts`, `routes/+layout.svelte`,
  `lib/types/api.d.ts`, `tools/ui/vitest-setup-client.ts`, `tests/test-moe-predict.cpp`, `tests/test-server-component.cpp`
- new: `tools/server/server-perf.{h,cpp}`, `tools/server/tests/unit/test_perf.py`, `tools/ui/src/lib/{services/perf.service.ts,
  stores/perf.svelte.ts, components/app/perf/PerfPanel.svelte, utils/perf-format.ts}`, `tools/ui/tests/unit/perf-format.test.ts`,
  `tools/ui/tests/client/perf-panel.svelte.test.ts`, `kimi-k3/tools/perf_e2e.py` (5b + 5f, kept as regression)
- build: reconfigure with `-DLLAMA_BUILD_UI=ON`

## 8. Roles and review loop
- Plan reviewers (Fable, parallel, APPROVE / CHANGES REQUESTED): server/kernel (io accounting, madvise, mincore, D4 rules,
  overhead), requirement traceability (9 metrics -> source -> JSON key -> panel text -> test), UI/build. Implementation
  starts only on unanimous APPROVE. Implementation + static review: Opus. Dynamic tests + e2e: Sonnet (own port/dirs).
- Tracer check: commits 06dd6f889, dfcdf9273, 34ed3b736 reverted the Perfetto tracer, c8d1cf38f removed the script; the
  remaining `LLAMA_TRACE` hits (`llama-model-loader.cpp:548`, `server-context.cpp:1019/1459`) are upstream's verbose flags.

## 9. User decisions (final)
D1 RAM headline = MemTotal - MemFree, split in the expand. D2 panel on by default via launcher `--ui-config` (first visit
only; existing browsers tick once). D3 e2e on a tester-started server on its own port when the machine is idle. D4 wasted
credits later use while resident; item 8 = precision labelled "guesses right", recall in expand/JSON; item 9 denominator =
all bytes loaded from disk, predictor share in the expand (R1, coordinator: lifetime ratio instead of a window, see 3b).

## Review history
- v3 APPROVED by all three reviewers (server/runtime, UI/build, traceability); 5c credit/expiry sentence corrected
  per the traceability reviewer (pending bytes are not in the numerator). Implementation notes: reset progress_il = -1 and
  set n_tokens at each llama_decode start, clear on return incl. error paths; keep mincore outside qmtx sections.
- v1 (2026-10-17): 3 Fable reviewers, all CHANGES REQUESTED. Server B1-B5 (nested HIGH loads, HIGH = needed bytes, clear
  pred_tag at scoring, baseline at construction, live prompt speed via layer progress) + minors; traceability A1-A7 (live
  pp e2e, precision label + recall kept, gen headline = current/last request, synthetic-tick sampler test, latency
  contract, phone layout, live seam test); UI/build 1-8 (D2 wording, mobile strip, tap-expand, polling in the store,
  pinned formats, launcher guards, vite proxy/a11y, rebase cost). All folded into v2 with user decisions D1-D4.
- v2 (2026-10-17): UI reviewer APPROVED (note: `--chat-form-bottom-position` lives in ChatScreen.svelte). Server and
  traceability CHANGES REQUESTED: R1 item 9 window incompatible with D4 crediting -> lifetime ratio, clamped, null at
  zero denominator, pending/credited in the expand, 5c cases; C1 predictor-independent per-layer progress op for
  n_tokens > 4 (on_moe_selected returns early there) + sched_copy_experts writer, `llama_decode_progress()`, e2e with
  OFFLOAD=0/1; C2 pending resolved at scoring with a capped sampled mincore probe, HIGH items only bytes_actual, mincore
  ownership guard with worker-delta fallback; minors (expiry min(Cached), mode string, recall zero-start note, thread-local
  scratch, array sizing). All folded into v3.
