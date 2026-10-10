# A770 matrix campaign field guide

This is the finding aid for [matrix-1006/](matrix-1006/), the October 6-7,
2026 Arc A770 campaign. It maps the seven measurement phases, their runners,
raw evidence and later corrections. It is a reading guide, not a runnable
benchmark recipe or a statement of current production settings.

Curated on 2026-10-08 from the archived scripts and artifacts. Numerical checks
below re-aggregate existing measurements; no GPU experiment was repeated.

## Start here

1. Read [FINAL-REPORT.md](matrix-1006/FINAL-REPORT.md), especially its harness
   problems and unproven claims. It supersedes early speculative-decoding
   recommendations in the phase-1 README.
2. Use [ALL-TABLES.md](matrix-1006/ALL-TABLES.md) for the common depth-0/8k
   throughput comparisons. Despite the final report's "phases 1-6" description,
   it does not tabulate every suite: deep context, clean workgroup sweeps,
   perplexity and server prompt results need their own raw directories.
3. Use [COMPLETE-MATRIX.md](matrix-1006/COMPLETE-MATRIX.md) and its
   [CSV](matrix-1006/COMPLETE-MATRIX.csv) for phase-1 settings and per-run values.
   "Complete" means the original phase-1 selection, not all seven phases.
4. Follow the suite table below to the raw cell, then join its tag to
   [index.tsv](matrix-1006/index.tsv), [master.log](matrix-1006/master.log)
   and the runner. A throughput table alone does not establish successful
   completion, a clean host, or correct output.

For the original snapshot, use [phase1-final/](matrix-1006/phase1-final/).
[README.md](matrix-1006/README.md) and
[summary-tables.md](matrix-1006/summary-tables.md) are earlier summaries.
[index-part1.tsv](matrix-1006/index-part1.tsv),
[index-phase1.tsv](matrix-1006/index-phase1.tsv) and
[master-part1.log](matrix-1006/master-part1.log) preserve partial history.

## Scope and provenance

The final report identifies build `b12327.c4b9d6430`, xe and oneAPI 2026.1.
The build footer is independently visible in
[ccs/ccs1.r0.md](matrix-1006/ccs/ccs1.r0.md). The
[baseline stderr](matrix-1006/envs/base.r0.err) records F16 enabled,
DNNL disabled, and graph support compiled in. These are dated evidence,
not a probe of today's installed binary. The report header does not supply
a complete per-cell compiler, compute-runtime, kernel and model checksum
manifest; do not fill those gaps from current host state.

| Model family | Role and placement in the archived runners |
| --- | --- |
| Llama-3.1-8B-Instruct Q4_K_M | Dense control, `-ngl 99`, fully on the GPU; most initial cells use q8_0 K/V, FA, 12 CPU threads, 512 prompt and 64 generated tokens at depths 0 and 8192 |
| Ornith-1.5-35B Q4_K_M | MoE production-style placement, fit target 1024; initial fit context 32768, later deep/server experiments reserve 131072; cache-off and load-mode-none are the production arms |
| Ornith-1.5-35B-A3B uncensored Q8_0 | Separate weight file in `q8` and `q8clean`; do not confuse Q8_0 weights with the q8_0 KV setting used across many other suites |
| Qwen2.5-Coder-7B Q6_K and Qwen3.5-9B Q4_K_M | MKL route and perplexity shape controls in `mkl` and `correct2` |
| Qwen4Exp / Qwen3.8-Flash-Next IQ1_M trunk with Q8_0 MTP head | Large-model drafting experiments in `spec3`, with fit-headroom follow-ups in `spec4`; a different model and memory regime from Ornith |

The shared [matrix-lib.sh](matrix-lib.sh) scrubs ambient tuning variables and
sets five baseline variables: `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
`GGML_SYCL_ENABLE_GRAPH=1`, `GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300`,
`GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0`, and `GGML_SYCL_FA_LARGE_GRF=1`.
The last-but-one setting disables the fork's xe-default hook, as the archived
stderr explicitly reports. Do not interpret it as a claim that copy engines
were disabled in every cell. Arms add their own environment delta.

The scripts use installed `/usr/bin/llama-*` programs and external models,
not a build from this checkout. Baseline choices are campaign controls, not
new recommended defaults.

## Phase and suite registry

Each directory link is an evidence bundle. Per-cell Markdown is normally
program stdout, not a human-authored report. "Rounds" below means outer
launches; a cell can also contain multiple internal `llama-bench` repetitions.

### Phase 1: broad screening

Runner: [matrix-run.sh](matrix-run.sh), shared helpers:
[matrix-lib.sh](matrix-lib.sh). The standalone
[matrix-ngram.sh](matrix-ngram.sh) and [matrix-ngram.py](matrix-ngram.py)
perform the server part; `ngram` appears in the main runner's header but it
does not define a `sec_ngram` function.

| Suite | Question and artifact interpretation |
| --- | --- |
| [ccs](matrix-1006/ccs/) | Single dense model, CCS modes 1/2/4, three rotated rounds; compare full-width and partitioned GPU configurations |
| [envs](matrix-1006/envs/) | Dense model, 21 arms with three rounds: graph, fusion, ESIMD, workgroup count, MKL FA, GRF, memory and command-list controls |
| [envs-aborted](matrix-1006/envs-aborted/) | Three early cell bundles retained as aborted history; do not pool with `envs` |
| [il](matrix-1006/il/) | CCS mode crossed with immediate command lists 0/1, three rounds; separates command submission from mode changes |
| [freq](matrix-1006/freq/) | Frequency floor/pinning and power profile, three rounds; use `freq2` for explicit write readbacks |
| [ornith](matrix-1006/ornith/) | MoE cache, load mode and selected environment variants, three rounds; establishes that dense-model conclusions do not transfer automatically |
| [two](matrix-1006/two/) | Two concurrent dense-model processes, modes 1/2/4 and timeslices 1000/20000 us; `.a` and `.b` identify the two processes, not an A/B candidate pair |
| [ngram](matrix-1006/ngram/) | Speculative types and placement variants; two launches, two prompts and three requests per prompt; startup failures also have server logs |

### Phase 2: mechanisms, controls and correctness probes

Runner: [matrix2-run.sh](matrix2-run.sh); client:
[matrix2-spec.py](matrix2-spec.py). Recovery runner:
[matrix2-oracle.sh](matrix2-oracle.sh).

| Suite | Question and artifact interpretation |
| --- | --- |
| [mode](matrix-1006/mode/) | Four rotated rounds of CCS 1/2/4 on Ornith production placement |
| [ornenv](matrix-1006/ornenv/) | Eighteen Ornith environment arms, three rounds; compare with dense `envs`, not across models by raw t/s |
| [8bnoise](matrix-1006/8bnoise/) | Six rounds for small dense-model effects; tests whether initial sub-2% differences persist |
| [mkl](matrix-1006/mkl/) | Four models, f16/q8_0 KV, MKL on/off, depths 0/2k/4k/8k/16k; one outer round, two internal repetitions |
| [deep](matrix-1006/deep/) | Ornith environment arms at 0/16k/32k/64k, one outer round; screening evidence preceding clean workgroup follow-ups |
| [fitc](matrix-1006/fitc/) | Fit-context 32k versus 128k; Q4 measurements exist, but the Q8 model path was wrong and those arms fail |
| [host](matrix-1006/host/) | Thread count, CCD affinity and microbatch size, two rounds; topology assignments are host-specific |
| [two2](matrix-1006/two2/) | Concurrent dense processes with 8k measurements and `.span` start/end timestamps; compare with `single6` thread-count control |
| [cost](matrix-1006/cost/) | Small prompt batches 1/2/4/8/16/32/64/128 on dense and MoE models; converts throughput into verification-batch time proxies |
| [correct](matrix-1006/correct/) | Dense-model perplexity under kernel-affecting knobs and CCS changes; includes the actual `text.txt` input assembled from local notes/docs |
| [oracle](matrix-1006/oracle/) | CPU/SYCL correctness logs with MKL on/off; original in-phase invocation hit `setvars.sh` under `set -u`, so inspect recovery logs and the final report |
| [spec2](matrix-1006/spec2/) | Ten heterogeneous prompts for dense/Ornith speculative arms and prefetch tests; early throughput interpretation is superseded by `spec6` |

### Phases 3-5: fill gaps, investigate failures, explore depth

| Runner | Suites and why they exist |
| --- | --- |
| [matrix3-run.sh](matrix3-run.sh) | [freq2](matrix-1006/freq2/) records verified sysfs writes; [single6](matrix-1006/single6/) compares one process at 6 versus 12 threads; [correct2](matrix-1006/correct2/) extends MKL perplexity to other shapes; [spec3](matrix-1006/spec3/) contains both `prefetch2` and `mtp` sections |
| [matrix4-run.sh](matrix4-run.sh) | [realtext](matrix-1006/realtext/) reconciles synthetic decode with a saved man-page prompt; [spec4](matrix-1006/spec4/) combines `specreal`, `prefetchng` and `mtpfit`; [q8](matrix-1006/q8/) fixes the earlier model path; [mkl32k](matrix-1006/mkl32k/) adds depth; [costpath](matrix-1006/costpath/) varies fusion/graphs and records profiler evidence; [correct3](matrix-1006/correct3/) retries Ornith PPL with more headroom and a smaller batch |
| [matrix5-run.sh](matrix5-run.sh) | [wg](matrix-1006/wg/) sweeps Ornith workgroup limits 16/24/32/48/64/128; 24/48/128 have only one outer round. [wg8b](matrix-1006/wg8b/) is the dense control; later clean reruns supersede contaminated timing |

Phase-3 and phase-4 server result directories combine different experiments.
Use `cfg` prefixes, not just the parent directory, to select a comparable set.
No `prefetch2/` or `mtp/` output directory is expected: both use `spec3/`.
The later Ornith PPL retry changes fit target and batch size together, so it
does not isolate which change resolved the abort.

### Phases 6-7: clean reruns and output checks

| Runner / suite | What to use it for |
| --- | --- |
| [matrix6-run.sh](matrix6-run.sh): [wgclean](matrix-1006/wgclean/) | Ornith workgroup 16/24/32/64, three rotated rounds at 0/16k/32k/64k, two internal repetitions; source comment says four rounds but the actual loop runs three |
| Phase 6: [wg8bclean](matrix-1006/wg8bclean/) | Dense workgroup 16/32/64, two rotated rounds at 0/8k/16k/32k |
| Phase 6: [mkl32kclean](matrix-1006/mkl32kclean/) | Dense and Ornith MKL on/off at 32k, two rounds |
| Phase 6: [q8clean](matrix-1006/q8clean/) | Q8_0-weight Ornith baseline/MKL-off/GRF-off/workgroup-32, three rounds |
| Phase 6: [spec6](matrix-1006/spec6/) | None/ngram-mod/ngram-simple, three rotated launches, ten prompts, temperatures 0 and 0.6; preferred speculative comparison |
| Phase 6: [wgclean-aborted-relaunch](matrix-1006/wgclean-aborted-relaunch/) | One retained interrupted bundle; not a fourth clean round |
| [matrix7-run.sh](matrix7-run.sh): [wgcheck](matrix-1006/wgcheck/) | Four dense PPL checks at workgroup 16/32 plus four Ornith server launches in 16/32/32/16 order; [longcheck.py](longcheck.py) supplies two source-text depths per launch |

The two unfiled issue drafts summarize proposed follow-up, not accepted fixes:
[MKL default/head dimension](matrix-1006/issues/01-mkl-fa-default-head-dim.md)
and [VMM allocation abort](matrix-1006/issues/02-pool-vmm-alloc-abort.md).

## Read the artifact schema correctly

| Artifact | Fields and interpretation |
| --- | --- |
| `<arm>.r<N>.md` | `llama-bench` stdout: test label, t/s with internal variation, model/backend/KV fields and build footer. For PPL or completion it may be mostly empty because results go to stderr |
| Matching `.err` | Build/runtime settings, model placement, PPL, completion timing, warnings, allocation errors and aborts; essential companion to stdout |
| `.freq` | Timestamped actual/current frequency and throttle samples; mean actual frequency is descriptive, not proof of a stable frequency cap |
| `.fdinfo` | DRM engine cycles/capacity snapshot around 25 seconds into the process; missing/empty files can mean a short run, and capacity is not proof of concurrent engine utilization |
| `.vmstat` | Periodic host load samples in later phases; keep distinct from per-run derived foreign CPU estimates |
| `index.tsv` | No header. First fields are suite, cell tag, environment delta, `idle=`, `load=`, `faults+`, `rc=`, elapsed seconds. Later lines add tenancy, times, memory, temperature, `idlerun=` and `foreignpct=`. Parse optional fields by key; old lines are shorter |
| `results.jsonl` | One server response summary: `cfg`, prompt name, `tg`, `pp`, generated `n`, drafted `draft_n`, accepted `draft_acc`; newer clients also record prompt tokens `pn` |
| `spec6/results.jsonl` | Selected attempt summaries additionally carry `idle_run`, `foreign_pct`, `attempt`; per-attempt JSONL/VMSTAT/server logs remain alongside them |
| `raw/*.json` | Generated text, token/logprob data and response timings; `wgcheck` also stores usage. These are output evidence, not copies of every source prompt |
| `two2/*.span` | A/B process start/end timestamps, useful to establish overlap before adding their rates |
| `master.log`, launcher stdout logs | Experiment order, retries, failures, skips, setting changes and cleanup messages; logs can outlive the exact revision of the shared library that produced them |
| `klog*.txt`, [monitor/](matrix-1006/monitor/), [sysfs-writes.log](matrix-1006/sysfs-writes.log) | Kernel snapshots/stream, fault matches and setting readbacks; inspect timestamps and distinguish intentional CCS resets from unexpected events |

There is no `summary.jsonl` in this matrix tree. The primary numerical records
are per-cell Markdown and the server `results.jsonl` files. Phase-1 CSV settings
are partly reconstructed by [matrix-complete.py](matrix-complete.py) from
hardcoded arm definitions and index deltas, rather than captured command lines.

`pp512 @ d8192` measures processing 512 prompt tokens after an 8192-token depth;
it is not processing an 8192-token prompt. `tg64` measures 64 generated tokens.
The server's `tg` is its response timing field, not HTTP wall latency, startup
time or aggregate multi-user throughput. The early `ngram` client requests up
to 384 tokens; `longcheck` requests up to 160 and may stop earlier. Compare
actual `n` and prompt lengths before comparing rates.

The summary generators average already summarized cell rates and do not
compute confidence intervals across launches. Internal repetition spread is
not the uncertainty of a campaign-wide effect. Rotated order reduces one
ordering bias, but small differences still need their own repeatability
evidence; selecting the fastest arm from a wide sweep is exploratory.

For `cost`, a batch of `b` tokens at `p` t/s has a time proxy `b / p` seconds;
compare that time to the one-token case. A throughput ratio alone is not the
verification overhead ratio. For draft acceptance, sum accepted and drafted
tokens and divide; null drafting fields in no-speculation rows are not failures.

## Independently checked archived values

The following are arithmetic means recomputed from the indicated existing
files on 2026-10-08, rounded only for display. They corroborate selected final
report claims; they do not revalidate hardware behavior or all report tables.

| Evidence | Checked measured values | Bound on interpretation |
| --- | --- | --- |
| [ccs/](matrix-1006/ccs/), three outer rounds per mode | Dense pp512: mode 1 = 1636.25, mode 2 = 1163.74, mode 4 = 623.90 t/s | Same campaign/build; early fault evidence is incomplete |
| [mode/](matrix-1006/mode/), four rounds | Ornith tg64: 49.31 / 42.81 / 31.76 t/s for modes 1/2/4 | Does not establish the hardware scheduling mechanism |
| [mkl32kclean/](matrix-1006/mkl32kclean/), two rounds | Dense MKL on/off: 63.51 / 103.73 pp512 t/s at 32k; Ornith: 207.85 / 117.98 | Effect reverses across these model/placement cases; this is not universal proof that head dimension alone causes it |
| [wgclean/](matrix-1006/wgclean/), three rounds | Ornith tg64 at 64k: WG16 = 21.62, WG24 = 24.11, WG32 = 25.52, WG64 = 23.11 t/s | WG32 gains about 18% versus WG16 in this synthetic depth case; no general service SLA follows |
| [q8clean/](matrix-1006/q8clean/), three rounds | Q8_0-weight Ornith tg64 at 8k: baseline = 25.51, WG32 = 27.02 t/s | Different weight model from the Q4 baseline |
| [wgcheck/](matrix-1006/wgcheck/), four PPL stderr files | All print `Final estimate: PPL = 12.7626 +/- 0.28235` | Identical printed scalar values; this check does not prove bit-identical logits, even though the report calls PPL bit-identical |

The clean speculative dataset contains 180 summaries: 30 per configuration and
temperature, comprising three launches of ten fixed prompts. Mean measured `tg`:

| Configuration | Temperature 0 | Temperature 0.6 |
| --- | ---: | ---: |
| None | 45.35 | 45.30 |
| ngram-mod | 45.33 | 42.51 |
| ngram-simple | 43.85 | 44.05 |

Source: [spec6/results.jsonl](matrix-1006/spec6/results.jsonl). Recorded foreign
CPU spans 0.7-1.0%; 160 rows select attempt 1 and 20 select attempt 2.
These means give each prompt equal weight and are not total generated tokens
divided by total elapsed time. The prompt mix was hand-selected, and prompt
order inside each launch is fixed. They support the report's withdrawal of an
earlier general ngram-simple recommendation, not a claim that speculation
never helps a particular workload.

## Corrections, exclusions and remaining uncertainty

- The phase-1 README says a whole-window dmesg scan found no fault lines; the
  final report explicitly says per-cell fault evidence for 03:14-06:00 does
  not exist. Preserve the later limitation. `faults+0` is not a clean-run
  certificate: the original counter had a grep fallback bug.
- `monitor/FAULTS.txt` contains reset lines. The final report attributes later
  GT resets to CCS writes and reports no devcoredump. This guide has not
  reconstructed causality for every reset or proved the absence of silent
  xe stalls.
- `cell()` logs nonzero child status but normally returns zero; it pauses
  after an increased fault count instead of failing the entire campaign.
  A "DONE" log does not mean all cases passed.
- The shared `cell_clean()` implementation gates on `CLEAN_FOREIGN`, not the
  stale comment's idle threshold. Its default is 4%; `srv6` defaults to 18%.
  The final report describes a campaign threshold of 18%. The last attempt
  can still be kept when dirty. Use recorded values and launch history, not
  the word "clean", to select data.
- Cleaner reruns are a disclosed selection criterion. Aborted directories,
  failed startup logs and discarded attempt files are evidence of exclusions,
  not additional independent repetitions. Successful-response JSONL alone
  omits launches that failed before a response.
- Earlier `spec2`/`spec4` aggregate performance conclusions are explicitly
  superseded by phase 6. Initial repeat-code prompts strongly favor some
  draft strategies and are not representative of arbitrary user traffic.
- `fitc` Q8 arms use a wrong path; `q8` and `q8clean` are the replacements.
  Ornith `correct2` allocation aborts lead to `correct3`; failing PPL runs
  are not infinite-PPL measurements and should not be averaged as numbers.
- The report records a leaked server retaining VRAM and leaves competing
  memory holders unexcluded for earlier allocation failures. More fit
  headroom helping later runs is evidence of a workaround, not a complete
  root-cause proof. No per-process VRAM history was recorded.
- The final report limits the MKL oracle to a particular d=128, GQA 4:1
  route and says MKL-off fails that route expectation. Read the log and
  [test source](../../tests/test-sycl-turbo-correctness.cpp) before interpreting
  it as a numerical regression. Perplexity checks do not replace missing
  d=256/GQA 7:1/fallback-route oracle coverage.
- The report's opening "production flags unchanged" language coexists with
  its later note that a WG32 service drop-in was applied on October 7.
  Treat those as historical narrative updates, not a verified current
  service inventory. This guide did not inspect or modify services.

## Runner safety and reproduction gaps

These archived scripts require adaptation before reuse. They are not safe
read-only entry points simply because they live under `docs/`.

| Script family | Concrete coupling or side effect |
| --- | --- |
| `matrix-lib.sh` and all sourcing runners | Creates `/mnt/nvme1/oneapi-ab/matrix-1006`, uses SSH alias `vinbonesjr` for service/sysfs actions, stops the Ornith service in `init()`, and restores hardcoded frequency/profile/timeslice/CCS values on exit rather than capturing all original state |
| `matrix-run.sh` | Changes CCS, frequency, power profile and timeslice; launches competing GPU processes in the concurrency suite |
| `matrix-ngram.sh`, `matrix2-run.sh` | Use broad `pkill -x llama-server`; these can kill servers from another session. Later phases use parent-PID teardown, but the final report still records a leaked process |
| `matrix2-oracle.sh` | Waits on an archived PID file and assumes an external oracle build plus oneAPI installation; a copied PID is not a reliable job identity |
| `matrix4-run.sh` | Rewrites the saved prompt from local `man ls`; always appends `correct3 prefetchng mtpfit` after requested sections, even when the caller requested something else |
| `matrix5-run.sh` | Runs both workgroup suites unconditionally |
| `matrix6-run.sh` | Removes per-attempt JSONL before retry, appends selected rows to results, and reuses raw response names across attempts; reruns can duplicate summaries or overwrite raw evidence |
| `matrix-complete.py` | Writes `COMPLETE-MATRIX.md/.csv` at its hardcoded external root; do not run it expecting to update this archived copy |
| `matrix-analyze.py` | Reads only four fixed test labels at depth 0/8k; reports `n` from pp512 count, ignores return codes and does not implement comprehensive aborted/dirty filtering |
| `matrix2-spec.py`, `longcheck.py` | Require external source/build/package/session files to construct prompts; raw outputs do not freeze those inputs. `longcheck.py` skips missing source files, changing the prompt silently |

The shared library has timeouts, but no `/tmp/a770.lock` lock. It waits for a
quiet host for a bounded period and then proceeds. The scripts' fault matcher
is not the current repository's full two-stage driver/failure gate. Consult
the [current GPU contract](../development/agents.md) before designing a new
run rather than treating these historical wrappers as the current protocol.

Dependencies include Bash, Python standard library clients, SSH privilege on
the named host, Linux `/proc`/xe sysfs, `vmstat`, `fuser`, `curl`, `taskset`,
installed llama programs, model files, and for phase 4 `man`/`col`.
The external helper, benchmark binary, two main model paths and oracle binary
existed during this guide's filesystem check, but existence does not establish
their version or compatibility. No SSH, service, sysfs, model load or API call
was performed for this guide.

## Copies and maintenance

There are 3,394 files in [matrix-1006/](matrix-1006/) in this snapshot. The
[oneapi-ab copy](oneapi-ab/matrix-1006/) has matching paths: 3,393 are
byte-identical; only `wgclean-aborted-relaunch/wg16.r0.vmstat` differs.
The root `matrix*` scripts and launcher logs checked against their
`oneapi-ab/` counterparts are identical. Do not count the second tree as an
independent campaign, and do not delete either copy based on this guide.

For a refresh, inventory the tree, compare duplicate bytes, read the newest
correction report, and recompute selected groups from raw cells/JSONL. Keep
the original reports and failed evidence intact. Add dates and source links
to changed conclusions; avoid silently modernizing paths or replacing the
archived runner's behavior with today's source behavior.

This curation checked artifact paths, ASCII in this new guide, duplicate
identity and the selected aggregates above. It did not rerun benchmarks,
audit every raw response/logprob or prove every historical result correct.
