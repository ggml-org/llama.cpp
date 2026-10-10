# Historical benchmark campaigns: finding aid

Source audit: 2026-10-08. This guide covers the compiler/runtime, driver, GRF,
plain-decode and server archives outside the October matrix. For the latter, use
[MATRIX-GUIDE.md](MATRIX-GUIDE.md); start at [README.md](README.md) for the whole
collection and its file inventory. No archived runner was executed for this guide.

These are historical experiments, including failed controls and interrupted runs.
A directory name is an experiment label, not a success criterion. Read the runner,
its stdout/stderr, and its provenance before quoting a result. The copied
[oneapi-ab/](oneapi-ab/) subtree contains byte-identical copies of all 26 runners
listed here as of the audit; these are duplicate sources, not independent trials.
Links below use the shorter root-level copy. Preserve both artifact trees.

## Choose an evidence trail

| Question | Read first | Then inspect | What it can establish |
| --- | --- | --- | --- |
| What changed between the old compiler/DNN builds? | [run-abcd.sh](run-abcd.sh), [run-dca.sh](run-dca.sh) | `A/C/D-final*` tables and stderr, [run-abcd.log](run-abcd.log), [run-dca.log](run-dca.log) | Historical same-workload build comparison; separate compiler and DNN arms |
| Why were xe copy engines disabled? | [dated driver investigation](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md) | `N-xe-x*`, before/after kernel logs, trace anchors and dumps | Failure mechanism and bounded control experiments on that stack |
| Does larger GRF help attention? | [dated GRF analysis](../research/sycl/sycl-fa-large-grf-2026-09-30.md) | `grf*/product.json`, provenance and sample files | Per-model, per-depth paired throughput with validity limits |
| Is a plain-decode difference host-load noise? | [decode/matrix.sh](decode/matrix.sh), [decode/quiet.sh](decode/quiet.sh) | [decode/matrix.log](decode/matrix.log), [decode/quiet.log](decode/quiet.log) | Interleaved builds and deliberate CPU-load controls |
| Did a real server survive the workaround? | [server-soak/soak.log](server-soak/soak.log) | Six saved request/response pairs in [server-soak/](server-soak/) | Completion and timing for those requests, not indefinite stability |
| Does work-group size change long-context greedy output? | [longcheck.py](longcheck.py), [longcheck-analyze.py](longcheck-analyze.py) | [matrix-1006/wgcheck/](matrix-1006/wgcheck/) and matrix guide | Token-string prefix/logprob differences, including same-size repeated launches |
| What did the b12327 update change? | [b12327-p1/meta](b12327-p1/meta), [b12327-d8k-p1/meta](b12327-d8k-p1/meta) | `run1-A`, `run2-B`, `run3-B`, `run4-A` tables/stderr | ABBA comparison of extracted old package and then-installed binary, at recorded CCS mode |

## Shared workload and artifact vocabulary

The early shell wrappers run the Ornith 1.5 35B Q4_K_M GGUF on `level_zero:0` with
`-fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -p 512 -n 64 -d 0,8192 -r 5 -t 12`.
They explicitly enable graphs. These are the archived arguments in
[bench.sh](bench.sh), not recommended defaults for today's build. In particular,
leaving MoE cache placement implicit differs from later `--moe-cache off` runs.
The installed-package workload also differs from the smaller Ornith IQ2_M used by
the paired GRF products. Do not compare these models as if only GRF changed.

| Artifact | Meaning and reading order |
| --- | --- |
| `bench.md`, `runN-A.md`, `runN-B.md` | `llama-bench` rows: prompt-processing and token-generation throughput, sometimes at depth; inspect build and model columns |
| `bench.err`, `*.err` | Effective backend settings, failures and fit messages; a table alone can hide truncation or changed placement |
| Runner `*.log`, `meta`, `done` | Execution order, exit statuses, CPU load/idle, time and service-restoration messages; `done` is not a correctness certificate |
| `product.md` | Human-readable paired throughput summary; read JSON validity before quoting it |
| `product.json` | Arm labels/environment, depth/KV cells, metric statistics, invalid diagnostics and provenance |
| `provenance.json` | Recorded CMake caches, binary hashes, repository commit, kernel/runtime probes and effective environments; failed metadata probes are preserved |
| `samples/*.json` | Per-arm, per-repetition subprocess records; sample 0 is discarded by the product harness |
| `dmesg-before.txt` / `dmesg-after.txt` or `dmesg.before.txt` / `dmesg.after.txt` | Kernel snapshots; distinguish newly introduced faults from old boot messages and unreadable/empty logs |
| `clock-anchor.txt`, `kernel-journal.txt`, `tlb-*.txt`, `perf-record.log` | Driver timeline support; the anchor relates monotonic tracing to wall-clock logs |
| `xe-devcoredump*.txt[.gz]` | Fault snapshots, sometimes from an earlier event than the enclosing directory; match PID, timestamp and sequence |
| `req*.json`, `resp*.json`, `out*.txt` | Real prompt/generated output evidence; successful HTTP or coherent text is not equality proof |

The current [paired product harness](../../scripts/bench-a770-fork-unique.py)
computes paired percentage changes by retained sample index, with 95% t intervals.
That differs from the mean/spread printed by a single `llama-bench -r 5` run.
Historical GRF products used four launches per arm and discarded sample 0; the
current harness default may differ. Its `effective KV` bandwidth is a modeled
byte count, not a hardware bandwidth counter.

## Compiler, installed-package and plain-decode families

A, C and D have explicit definitions in the runners: A is the old 2026.0 build
without DNN; C is the 2026.1 build without DNN; D is the 2026.1 build with DNN.
The saved stderr confirms `GGML_SYCL_DNNL: no/no/yes` respectively. The letters
are local labels, not backend names. N is identified in [run-n2.sh](run-n2.sh)
as the installed `4e7400c3a` build. The `K-aocc-*` labels are retained verbatim:
there is no matching K runner here that establishes a controlled compiler-only
comparison, so the label alone does not establish causation.

| Family | Purpose, contents and limits |
| --- | --- |
| [A-2026.0-nodnn/](A-2026.0-nodnn/), [A-2026.0-nodnn.contaminated/](A-2026.0-nodnn.contaminated/) | Early old-build tables/stderr; retain the explicitly contaminated variant as a failed-control record |
| [A-final/](A-final/), [C-final/](C-final/), [D-final/](D-final/) | Forward A/C/D order through the shared bench wrapper |
| [D-final2/](D-final2/), [C-final2/](C-final2/), [A-final2/](A-final2/) | Reverse D/C/A order, separate outputs; do not average away order or load differences |
| [A-pkg/](A-pkg/) and [A.ldpath](A.ldpath) | Extracted old package executables/libraries/headers and original runtime-library path input; package files are archived dependencies, not trusted current executables |
| [N-clean-r1/](N-clean-r1/), [N-clean-r2/](N-clean-r2/) | Two installed-build baseline rounds, with wait/load evidence in [run-n2.log](run-n2.log) |
| [N-xe-r1/](N-xe-r1/), [N-xe-r2/](N-xe-r2/) | Same early workload after xe switch; second directory also preserves hang diagnostics. Use [run-xe.log](run-xe.log) for driver/timing context |
| [K-aocc-ceoff-mmap/](K-aocc-ceoff-mmap/), [K-aocc-ceon-mmap/](K-aocc-ceon-mmap/) | mmap/copy-engine labeled bench controls with kernel snapshots; inspect stderr and result rows, not K as an assumed independent variable |
| [K-aocc-x4b-pinned-ceoff/](K-aocc-x4b-pinned-ceoff/), [K-aocc-x4c-pinned-ceon/](K-aocc-x4c-pinned-ceon/), [K-aocc-x4d-mmap-ceoff/](K-aocc-x4d-mmap-ceoff/) | Follow-up pinned/mmap and copy-engine controls; no standalone invocation manifest in these directories |
| [K-aocc-rt-ceon-pinned/](K-aocc-rt-ceon-pinned/), [K-aocc-oracle/](K-aocc-oracle/) | Real-text output plus before/after logs, and separate default/FA256 oracle output; not throughput equivalents |
| [decode/](decode/) | Plain-decode new/old trials, injected CPU load, quiet-host retries, [tables.json](decode/tables.json), and extracted `old-4e7400c3a` package |
| [old-b12321/](old-b12321/), [b12327-p1/](b12327-p1/), [b12327-d8k-p1/](b12327-d8k-p1/) | Old package and ABBA update trials; both `meta` files record `ccs_mode=2`, not the single-CCS baseline |
| [C-quiet.log](C-quiet.log), [N.log](N.log), [swap.log](swap.log), [swap-2.log](swap-2.log), [gputop.json](gputop.json) | Loose run/host diagnostics; contextual evidence, not independent throughput suites |

## xe failure investigation and controls

Start with the later [2026-09-30 synthesis](../research/software-stack/xe-kmd-bcs-copy-engine-2026-09-30.md),
then drill down into the raw families below. It distinguishes copy-source mapping,
placement, runtime patches and instrumented diagnostics. Its explanation does not
license transplanting old runtime debug switches into production.

| Family | Investigation question and evidence |
| --- | --- |
| [N-xe-copyeng0/](N-xe-copyeng0/), `N-xe-copyeng0-r2/r3/r4` | Copy engine disabled and repeated bounded soak, with [run-xe-soak.log](run-xe-soak.log) |
| [N-xe-pinned0/](N-xe-pinned0/), [N-xe-graph0/](N-xe-graph0/) | Disable host pinned memory or graphs separately; pinned-off includes coredump and kernel log |
| `N-xe-inorder0`, `N-xe-cbevents0`, `N-xe-immcl0`, `N-xe-nodirectsub`, `N-xe-norelaxed` | Five submission/event-ordering controls from [run-xe-levers2.sh](run-xe-levers2.sh); one passing setting is not a validated workaround |
| [N-xe-trace/](N-xe-trace/) | UR Level Zero debug log alongside bench stderr/table; instrumented throughput must not rank builds |
| [N-xe-copytrace/](N-xe-copytrace/), [N-xe-copytrace-nocache/](N-xe-copytrace-nocache/) | Async-copy tracer evidence, process maps, stacks and stall samples; no-cache copy-path variant adds live-copy and setting records |
| [N-xe-copyeng0-moeoff/](N-xe-copyeng0-moeoff/), [N-xe-copyeng0-moesoft/](N-xe-copyeng0-moesoft/) | Explicit MoE placement controls with copies on compute queue; early default-auto results are a different placement |
| [N-xe-realtext-moecache/](N-xe-realtext-moecache/), [N-xe-cachestats/](N-xe-cachestats/) | Text generation under cache/copy variants and additional cache-stat diagnostics |
| [N-xe-server/](N-xe-server/), [server-soak/](server-soak/) | Initial server request, response and failure evidence; later six-request soak under copy-engine-off configuration |
| [N-xe-x1-trace/](N-xe-x1-trace/) | Instrumented mmap failure timeline with anchors, TLB snapshots, kernel log and coredump |
| [N-xe-x4-pinned-ceon/](N-xe-x4-pinned-ceon/) | Instrumented pinned-source control with copy engine on; retains dump/log material that must be timestamp-matched |
| [N-xe-x4b-pinned-ceoff/](N-xe-x4b-pinned-ceoff/), [N-xe-x4c-pinned-ceon/](N-xe-x4c-pinned-ceon/), [N-xe-x4d-mmap-ceoff/](N-xe-x4d-mmap-ceoff/) | Clean pinned/copy-engine and mmap control series used in the dated synthesis |
| [N-xe-x5-bcsds1/](N-xe-x5-bcsds1/) | Blitter direct-submission experiment; diagnostic one-pass survival is not a stability guarantee |
| [N-xe-x6-noscratch/](N-xe-x6-noscratch/) | Attempt to remove scratch mappings; synthesis says requested switch did not remove the VM scratch-page flag |
| [N-xe-x7A-neomaster/](N-xe-x7A-neomaster/), [N-xe-x7B-neomaster/](N-xe-x7B-neomaster/) | NEO master read-only userptr retry disabled/enabled controls; trace logs explain stall versus completion |
| [N-xe-x7c-neopatch-rt-ceon/](N-xe-x7c-neopatch-rt-ceon/), [N-xe-x7d-neopatch-ceon-off/](N-xe-x7d-neopatch-ceon-off/), [N-xe-x7e-neopatch-ceon-auto/](N-xe-x7e-neopatch-ceon-auto/) | Patched runtime with real text, then off/auto MoE placement; foreign load/swapping invalidate speed comparisons in the latter runs |
| [N-xe-x7F-neopatch-hardened/](N-xe-x7F-neopatch-hardened/) | Hardened retry patch instrumented control |
| [N-xe-x7G-pkg-be85a8d685-retry/](N-xe-x7G-pkg-be85a8d685-retry/), [N-xe-x7H-pkg-be85a8d685-noretry/](N-xe-x7H-pkg-be85a8d685-noretry/) | Installed runtime package patch-on versus patch-off control; watchdog-stall evidence matters even without a kernel reset |
| `N-xe-rt-a-hook-mmap`, `N-xe-rt-b-hook-pinned`, `N-xe-rt-b2-hook-pinned-verbose`, `N-xe-rt-c-override-pinned`, `N-xe-rt-d-review-fix`, `N-xe-rt-lift-ceon` | Real-text follow-ups to the fork hook/override and review changes; directories retain `out.txt`, `err.txt` and kernel snapshots. Verbose b2 includes placement evidence |
| [N-xe-verify/](N-xe-verify/) | Saved correctness-oracle stdout/stderr, separate from timing |

The [investigation source bundle](xe-investigation-20260929/) has a different role
from measurement output:

- [COORDINATION.md](xe-investigation-20260929/COORDINATION.md) records ownership,
  hypotheses and tracer checks. Its correction identifies a server dump from an
  earlier PID/time than the nearby journal event; preserve that distinction.
- [driver-mechanisms.md](xe-investigation-20260929/driver-mechanisms.md) surveys
  xe/i915 residency and queue mechanisms, with unperformed discriminating experiments.
- [firmware-research.md](xe-investigation-20260929/firmware-research.md) separates
  GuC/HuC runtime firmware from board firmware and records limits of the snapshot.
- [interactive-tuning.md](xe-investigation-20260929/interactive-tuning.md) separates
  production placement from auto-placement results and explains scheduling knobs;
  its proposed profiles were not benchmarked by that note.
- `ur-reference-*`, `ur-current-*`, `neo-26.35-*` and
  `compute-walker-source-excerpt.txt` are frozen source excerpts for attribution.
- [trace-copies.c](xe-investigation-20260929/trace-copies.c),
  [check-trace.c](xe-investigation-20260929/check-trace.c), their compiled artifacts
  and [bin/llama-bench](xe-investigation-20260929/bin/llama-bench) are diagnostic
  interposition/forwarding tools, not clean timing runners. The wrapper uses an
  absolute `LD_PRELOAD` path; the coordinator's CPU check is historical evidence.

## GRF series: retain the rejected products

Every paired directory below contains `product.md`, `product.json`,
`provenance.json`, kernel snapshots and raw `samples/`. Each family has both
`-llama8b` and `-ornith-iq2m` variants. All use q8_0/q8_0 KV. Most cover depth 0
and 8192; initial `grf256-ornith-iq2m` also covers 2048.

| Product family (links to each model) | Actual comparison | Recorded status |
| --- | --- | --- |
| [grf256-llama8b](grf256-llama8b/product.json), [grf256-ornith-iq2m](grf256-ornith-iq2m/product.json) | Separate default and globally 256-GRF FA builds, before runtime knob; `GGML_SYCL_FA_GRF_SIZE_BUILD=256` is harness metadata for that build | Both `all_cells_valid=true`; broader kernel coverage differs from later tile-only variant |
| [grfmode1-llama8b](grfmode1-llama8b/product.json), [grfmode1-ornith-iq2m](grfmode1-ornith-iq2m/product.json) | Same-binary empty-env baseline versus `GGML_SYCL_FA_LARGE_GRF=1`, occupancy heuristic unchanged | Both valid under archived harness |
| [grfmode1occ-llama8b](grfmode1occ-llama8b/product.json), [grfmode1occ-ornith-iq2m](grfmode1occ-ornith-iq2m/product.json) | Mode 1 after large-GRF occupancy adjustment | Both valid under archived harness |
| [grfmode1pr89-llama8b](grfmode1pr89-llama8b/product.json), [grfmode1pr89-ornith-iq2m](grfmode1pr89-ornith-iq2m/product.json) | Review-head mode 1 | Both invalid: sample build `9ae0f9ad9` does not match recorded repository commit; note also reports high host load |
| [grfmode2pr89-llama8b](grfmode2pr89-llama8b/product.json), [grfmode2pr89-ornith-iq2m](grfmode2pr89-ornith-iq2m/product.json) | Historical mode 2 including VEC; later removed | Both invalid for same build/repository mismatch; exploratory evidence only |
| [grfmode1final-llama8b](grfmode1final-llama8b/product.json), [grfmode1final-ornith-iq2m](grfmode1final-ornith-iq2m/product.json) | Final tile-only mode 1 build | Both valid; dated note records substantial host load and wide short-context prefill intervals |
| [grfmode1aocc-llama8b](grfmode1aocc-llama8b/product.json), [grfmode1aocc-ornith-iq2m](grfmode1aocc-ornith-iq2m/product.json) | Same runtime knob on later recorded `linux73-tkg-bore-rc5-xe-aocc` kernel | Both invalid: sample build `b23424e30` mismatches recorded repository commit |
| [grfmode1aoccv-llama8b](grfmode1aoccv-llama8b/product.json), [grfmode1aoccv-ornith-iq2m](grfmode1aoccv-ornith-iq2m/product.json) | Later mode-1 product with matching provenance | Both valid under archived harness; suffix does not establish a controlled kernel/compiler comparison |

`invalid_cell_count=0` can coexist with `all_cells_valid=false`: global provenance
checks can reject the product even when individual cell failure lists are empty.
Read `invalid_diagnostics`. Likewise, `all_cells_valid=true` means the archived
harness accepted its gates, not that timing was uncontaminated or every provenance
probe succeeded. For example, saved provenance includes missing `llama-cli`
version probes. Do not silently repair those fields.

[grf/](grf/) holds same-binary PPL controls, while [grfbench/](grfbench/) and
[grfbench-moeoff/](grfbench-moeoff/) hold installed-package timing repeats with
explicit placement differences. [N-grf256-oracle/](N-grf256-oracle/),
[N-grf256-oracle-r3/](N-grf256-oracle-r3/), [N-grf256-oracle-r4/](N-grf256-oracle-r4/),
[N-split-oracles/](N-split-oracles/), [N-pr89-oracles/](N-pr89-oracles/) and
[N-pr89-final/](N-pr89-final/) retain correctness stdout/stderr for different
revisions and modes. Their test coverage and outcomes must be read from those
logs; a product result does not imply the corresponding oracle ran.

## Runner registry

This table covers every non-October-matrix `.sh`/`.py` runner in this archive,
including the separate decode matrix. Each also has an identical
`oneapi-ab/<same relative path>` copy. Argument sketches describe historical
interfaces; they are not copy-and-run instructions. All shell paths and outputs
below are hardcoded in the source unless an argument/environment override is named.

| Runner | Purpose and concrete controls | Inputs, outputs and operational limits |
| --- | --- | --- |
| [bench.sh](bench.sh) | `TAG [LD_LIBRARY_PATH] [BINDIR]`; shared Ornith workload above, graph on, timeout 3600 s | Uses `/usr/bin` by default; SSH service stop/start; writes original `/mnt/nvme1/oneapi-ab/TAG/bench.*` |
| [bench-xe.sh](bench-xe.sh) | Same workload; `GRAPH` and `BENCH_TIMEOUT` overrides; 15 s forced-kill grace | Same service/model/output dependencies |
| [bench-xe2.sh](bench-xe2.sh) | Adds shell-split `BENCH_EXTRA` to xe wrapper | Used for explicit MoE cache controls |
| [run-abcd.sh](run-abcd.sh) | Wait for load below 4, then A/C/D | Uses `A.ldpath`, extracted A package, external C/D build directories; produces `*-final` |
| [run-dca.sh](run-dca.sh) | Reverse D/C/A order | Same dependencies; produces `*-final2` |
| [run-n2.sh](run-n2.sh) | Two installed-build rounds after quiet waits | `N-clean-r1/r2`; failed wait does not stop execution |
| [run-xe.sh](run-xe.sh) | Two installed-build xe rounds, logs driver and engine/frequency settings | Calls `bench.sh`; `N-xe-r1/r2`; hardcoded PCI/sysfs topology |
| [run-xe-levers.sh](run-xe-levers.sh) | Copy engine off, host-pinned memory off, graph off; 420 s per arm | Calls `bench-xe.sh`; `N-xe-copyeng0/pinned0/graph0` |
| [run-xe-levers2.sh](run-xe-levers2.sh) | Driver in-order lists, counter events, immediate lists, NEO direct submission and relaxed ordering off separately | Five `N-xe-*` arms; `UR_*`/NEO variables are historical runtime controls, not validated current settings |
| [run-xe-moecache.sh](run-xe-moecache.sh) | Copy engine off with `--moe-cache soft` and `off`; 600 s each | `bench-xe2.sh`; `N-xe-copyeng0-moesoft/moeoff` |
| [run-xe-soak.sh](run-xe-soak.sh) | Three repeat rounds with `UR_L0_USE_COPY_ENGINE=0` | 420 s per round; `N-xe-copyeng0-r2/r3/r4` |
| [run-xe-trace.sh](run-xe-trace.sh) | `UR_LOG_LEVEL_ZERO` debug file and `UR_L0_DEBUG=1` | 600 s, `N-xe-trace/ur-l0-debug.log`; instrumented |
| [grf/ppl.sh](grf/ppl.sh) | GRF 0/1, `llama-perplexity -c 4096 -b 512 -ub 512 --chunks 10`, q8_0 KV, fit target 2048 | External build/model/WikiText; 2400 s per arm, `ppl-grf0/1.log`; SSH service mutation |
| [grf/ppl2.sh](grf/ppl2.sh) | Repeats same PPL arms to expose run-to-run noise | `ppl-grf0/1-repeat.log`; same dependencies |
| [grfbench/run.sh](grfbench/run.sh) | Installed package, warmup then off/on/off/on, pp128/512 and tg64 at 0/8192 | Five repetitions each; 3600 s each; quiet wait can expire and still launch; service mutation |
| [grfbench/run2.sh](grfbench/run2.sh) | Two phases of same GRF controls, idle-CPU gate and 1500 s per trial | Phase 1 deliberately leaves service stopped for phase 2; output shares `grfbench/` |
| [grfbench-moeoff/run.sh](grfbench-moeoff/run.sh) | Explicit `--moe-cache off`, same GRF repeats, then new/old plain decode | Deadline 1350 s checked before runs; 900 s each; archived comment records noisy host; extracted old build needed |
| [decode/matrix.sh](decode/matrix.sh) | New b12321 vs extracted b12305, plain tg64; injected 4/10 busy CPU loops, depths 0/8192 | Service stopped; cleans up its load loops; 1350 s scheduling deadline and 900 s per run; `decode/*/bench.*` |
| [decode/quiet.sh](decode/quiet.sh) | Require idle >=92% and load <3 for nine samples; new/old interleaved twice | Aborts without quiet window; 600 s per trial, extracted old package; `decode/Q-*` |
| [run-1006.sh](run-1006.sh) | `PHASE ORDER...`, A=b12321 extracted, B=then-installed b12327; pp512/tg64 depth 0, r3 | `--moe-cache off --load-mode none`, GRF1, graph on; 25 min scheduling deadline, 900 s/run; `b12327-PHASE` |
| [run-1006-d8k.sh](run-1006-d8k.sh) | Same package comparison, depths 0/8192, r5 | `b12327-d8k-PHASE`; both saved p1 trials use ABBA and record CCS2 |
| [server-soak/run.sh](server-soak/run.sh) | Six existing JSON requests to port 8089 `/completion`; 900 s each | Reads local API key without embedding it in source; needs running authenticated server and saved `req-r*.json`; writes responses and failure journal excerpts |
| [longcheck.py](longcheck.py) | `PORT TAG OUTDIR`; source-text prompts truncated to 95000/205000 characters, temperature 0, 160 output tokens, top-2 logprobs | Existing localhost server; hardcoded external source tree; missing source files silently omitted; actual prompt token counts are in response `usage`, not guaranteed by d28k/d60k labels |
| [longcheck-analyze.py](longcheck-analyze.py) | Same-WG and cross-WG pairs at both depths; identical prefix, logprob deltas and first divergence margin | Python standard library; reads hardcoded `matrix-1006/wgcheck/raw`; prints `missing` for absent pair inputs; compares token strings, not token IDs |
| [campaign-monitor.sh](campaign-monitor.sh) | Continuous kernel stream plus GPU/CPU/memory sample every 5 s | Hardcoded matrix monitor output, PCI/hwmon paths and SSH; no automatic stop; its fault regex is historical |
| [devcore-monitor.sh](devcore-monitor.sh) | Poll devcoredump every 3 s, copy up to 20 MB, then clear snapshot | Writes `1` to sysfs dump data over SSH; mutating, potentially truncating collector, not read-only analysis |

## Dependencies, gaps and safe interpretation

The wrappers assume Bash, coreutils/timeout, ripgrep, SSH access to `vinbonesjr`,
local systemd state, the original model/build trees and an A770 mapped to the
specified selector. Some old occupancy probes name `renderD129`, whereas newer
harnesses name `renderD128`; resolve the physical device before a new run.
Neither name is portable proof of sole tenancy.

As checked during this audit, the original C and D binary directories under
`/mnt/nvme1/llama-sycl-build/{C,D}/build/llama.cpp-sycl-f16-git/src/build/bin`
are absent. The original `/mnt/nvme1/oneapi-ab` path, Ornith Q4_K_M model and
WikiText path used by the PPL scripts exist on this host; their availability on
another host is not implied. Copied A/old package trees do not make the complete
compiler/runtime environment relocatable. Do not execute archived binaries merely
because they are present. Some investigation notes link to maintainer-local
`/home/svnbjrn/research-llama.cpp` sources outside this collection.

Many scripts use `set -uo pipefail` without `-e`, and print a subprocess exit code
before continuing. A zero shell exit is therefore not enough. `run-abcd.sh`,
`run-dca.sh`, `run-n2.sh` and `run-xe.sh` do not enforce quiet-wait success before
launch. Service stop failures are not consistently fatal either. Per-run timeouts
bound a subprocess but do not make a loop's scheduling deadline a strict total
wall-clock deadline. Monitors run forever until supervised termination.

For new work, use the current [SYCL backend guide](../backend/SYCL.md),
[agent GPU discipline](../development/agents.md), and appropriate maintained
runner after reviewing its current arguments:

- [bench-a770-fork-unique.py](../../scripts/bench-a770-fork-unique.py): paired
  product/depth comparison; explicit model, binary arms, environments and provenance.
- [perf/bench_spec.py](../../scripts/perf/bench_spec.py): real server requests,
  speculative acceptance and throughput, including paired builds; `llama-bench`
  alone does not exercise the server's speculative loop.
- [turbo-quality-gate.sh](../../scripts/turbo-quality-gate.sh): kernel, PPL and
  context-scaling stages. Non-strict mode permits skips; inspect stage results.
- [test-sycl-turbo-correctness.cpp](../../tests/test-sycl-turbo-correctness.cpp):
  CPU/SYCL correctness oracle, separate from performance conclusions.

## Evidence limits

This is source-grounded indexing, not a new experiment or an independent
recalculation of every raw result. It does not establish current driver safety,
performance, installed settings, or output equivalence. Historical `valid` fields,
HTTP successes and oracle logs retain their original scope. Measurements across
i915/xe boots, changed kernels/runtimes, different MoE placement, traced/clean
runs or different models must not be treated as a single controlled comparison.
