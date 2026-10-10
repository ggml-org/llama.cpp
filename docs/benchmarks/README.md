# Benchmark archive field guide

This is the finding aid for the local Arc A770 experiment archive. Start with
the question you need to answer, follow a campaign guide, then inspect its
report and raw evidence. Archived results describe their recorded builds and
host conditions; they do not establish the behavior of today's checkout.

| Read next | What it provides |
| --- | --- |
| [Matrix guide](MATRIX-GUIDE.md) | October 6-7 campaign: phases, every suite, report precedence, raw schemas, speculative decoding and validity limits |
| [Other campaigns](CAMPAIGNS.md) | Compiler/runtime and driver comparisons, copy-engine investigation, GRF experiments, correctness, soak tests and runner registry |
| [Complete file inventory](FILES.tsv) | Every archived regular file and symlink, with sizes, SHA-256 identities and duplicate references |
| [Repository field guide](../GUIDE.md) | Source architecture, build/test entry points and the rest of the documentation |
| [Current harness registry](../../scripts/README.md) | Maintained benchmark and analysis scripts outside this archive |
| [Research index](../research/README.md) | Dated explanations and decisions, including older campaigns outside this directory |

## Choose a reading or execution path

| Question | Start here | Runner or next evidence | Prerequisites and status |
| --- | --- | --- | --- |
| What did the October matrix conclude? | [Final report](matrix-1006/FINAL-REPORT.md), then [matrix guide](MATRIX-GUIDE.md) | Selected [tables](matrix-1006/ALL-TABLES.md), per-cell records and logs | Historical xe measurements. Early recommendations were revised; phase-1 tables are not the entire campaign. |
| Did a compiler/runtime/driver change help? | [Campaign guide](CAMPAIGNS.md) | `bench.md` + `bench.err`, package snapshots and driver investigation notes | Match binary, loaded libraries, driver, placement and model. A directory's letter is not a complete environment identity. |
| Did large GRFs help a particular model/depth? | [GRF catalog](CAMPAIGNS.md#grf-series-retain-the-rejected-products) | `product.md` -> `product.json` -> `provenance.json` -> `samples/` | Six of sixteen primary product campaigns fail their archived provenance gate. Check validity before using a percentage. |
| Compare steady-state throughput now | [Product harness](../../scripts/bench-a770-fork-unique.py) and [its registry](../../scripts/README.md) | `--campaign product`; separate candidate binaries for code/build changes | Python standard library, matching built tools, GGUF model, oneAPI environment and exclusive GPU use. Produces paired statistics and provenance. |
| Compare speculative decoding on actual text | [HTTP harness](../../scripts/perf/bench_spec.py), [method/example](../research/speculative/sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md) | Fixed [prompt fixture](../../scripts/perf/prompts.jsonl), `MODE=ab` for two builds | Compatible `llama-server`, model, available port, device and readable kernel logs. Ordinary `llama-bench` does not exercise server drafting. |
| Measure cold startup or MMVQ geometry | [Harness registry](../../scripts/README.md) | [Cold JIT](../../scripts/bench-sycl-cold-jit.py), [geometry sweep](../../scripts/sweep-a770-mmvq-geometry.py) | Cold JIT disables the persistent cache; the geometry sweep also builds variants and requires matching correctness evidence. |
| Establish correctness or quality | [Testing map](../GUIDE.md), [quality guide](../turboquant/quality-benchmarks.md) | [SYCL oracle](../../tests/test-sycl-turbo-correctness.cpp), [quality gate](../../scripts/turbo-quality-gate.sh), [perplexity tool](../../tools/perplexity/README.md) | Synthetic op correctness, corpus perplexity and application output are different checks. Throughput alone establishes none of them. |
| Find a log, trace or missing dependency | [FILES.tsv](FILES.tsv), then the relevant campaign guide | Filter `path`, compare `sha256`, inspect `link_target` | Read-only. All paths in the TSV are relative to this directory, including ignored and hidden files. |

There is no universal campaign duration: model load, cold compilation, context
depth, repetition count and timeouts dominate it. Read the selected runner's
parameters before reserving the device. No timing runs were performed to make
this guide.

## What was inventoried

Snapshot: 2026-10-08, checkout `strata` at
`d2f1f98348269a35e203c0dd01f3e588afa33d04`. The archive was **untracked** at
curation time, so that Git revision identifies the surrounding source, not
the origin or contents of these artifacts. [FILES.tsv](FILES.tsv) identifies
the individual bytes. The four new root guide/index files are excluded from
the inventory to avoid indexing the index itself.

| Inventory property | Observed value |
| --- | ---: |
| Archived entries | 9,121 |
| Regular files | 9,037 |
| Symlinks, recorded without following targets | 84 |
| Sum of regular-file lengths | 5,852,066,133 bytes |
| Distinct regular-file SHA-256 values | 3,543 |
| Regular files repeating an earlier digest | 5,494 |
| Zero-byte regular files | 502 |
| Entries in the primary `matrix-1006/` | 3,394 |

Lengths are logical file sizes, not filesystem space usage. Symlink targets
are not counted as additional library contents. Equal hashes include repeated
headers and empty logs as well as copied runs; they do not prove independent
measurements or a canonical choice of evidence.

### The `oneapi-ab` copy is nearly, but not exactly, redundant

Of its 4,563 entries, 4,553 match the primary tree at the same relative path
(regular-file SHA-256 or symlink target). One file differs:
[primary vmstat](matrix-1006/wgclean-aborted-relaunch/wg16.r0.vmstat) versus
[copied vmstat](oneapi-ab/matrix-1006/wgclean-aborted-relaunch/wg16.r0.vmstat).
Nine `perf.data` files exist only under `oneapi-ab/`, in:

- [N-xe-x1-trace](oneapi-ab/N-xe-x1-trace/)
- [N-xe-x4-pinned-ceon](oneapi-ab/N-xe-x4-pinned-ceon/)
- [N-xe-x5-bcsds1](oneapi-ab/N-xe-x5-bcsds1/)
- [N-xe-x6-noscratch](oneapi-ab/N-xe-x6-noscratch/)
- [N-xe-x7A-neomaster](oneapi-ab/N-xe-x7A-neomaster/)
- [N-xe-x7B-neomaster](oneapi-ab/N-xe-x7B-neomaster/)
- [N-xe-x7F-neopatch-hardened](oneapi-ab/N-xe-x7F-neopatch-hardened/)
- [N-xe-x7G-pkg-be85a8d685-retry](oneapi-ab/N-xe-x7G-pkg-be85a8d685-retry/)
- [N-xe-x7H-pkg-be85a8d685-noretry](oneapi-ab/N-xe-x7H-pkg-be85a8d685-noretry/)

Guides normally link the primary copy; retain both when provenance differs.
No files were deduplicated or removed. The TSV's `identical_to` is the first
lexicographic path with the same digest, not a judgment about which run is
authoritative.

### Existing packed reference

The existing [llamacpp-a770-benchmarks skill](.claude/skills/llama.cpp-a770-benchmarks/SKILL.md)
provides a Repomix search workflow. Its
[summary](.claude/skills/llama.cpp-a770-benchmarks/references/summary.md) and
[directory index](.claude/skills/llama.cpp-a770-benchmarks/references/project-structure.md)
are useful entry points. Its `files.md` is 391,857,595 bytes;
its advertised file count belongs to that export, not this inventory. Use
the original artifact and the new finding aids for current navigation.
The packed summary names `tech-stacks.md`, which is absent in that reference
directory. The summary also explicitly excludes `product.json`: use the
original products for the validity/provenance audit. Search the pack by file
header when needed, then verify the original source:

```bash
rg -n '^## File: matrix-1006/FINAL-REPORT.md$' \
  docs/benchmarks/.claude/skills/llama.cpp-a770-benchmarks/references/files.md
```

The four packed reference files are cataloged, but their embedded copies are
not counted as additional experiments. The broader
[repository skill](../../.claude/skills/llama.cpp-a770/SKILL.md) is described
in the [repository guide](../GUIDE.md#existing-clauderepomix-reference-skills).

## How to follow the evidence

```text
Question -> campaign guide -> report/table -> recorded configuration
                                             |
                                             v
                                    raw output + stderr
                                             |
                                             v
                                 driver/load/fault evidence
```

| Artifact pattern | Meaning and interpretation |
| --- | --- |
| `README.md`, `FINAL-REPORT.md`, issue drafts | Human interpretation at a stated time. Later corrections can withdraw earlier recommendations; an issue draft does not prove that it was filed or fixed. |
| `product.md` / `product.json` | Product-harness summaries and detailed cells. Check `all_cells_valid` and `invalid_diagnostics`, then retained samples, paired differences and intervals. |
| `provenance.json`, `*.meta`, `bench.err` | Build identity, flags, loaded environment and model placement. Inspect actual contents: different families record different subsets. |
| `*.md` paired with `*.err` | Often raw `llama-bench` tables plus backend diagnostics. An empty/header-only table is not a zero-throughput result. |
| `results.jsonl`, per-request `*.jsonl` | HTTP timing/acceptance or long-context output records. Schema and aggregation vary by runner; see the matrix guide. |
| `*.fdinfo`, `*.freq`, `*.vmstat`, `index.tsv` | Device, clocks and host-load context. Missing/empty telemetry leaves an evidence gap. |
| `dmesg*`, `klog*`, `kernel-journal*`, `FAULTS.txt` | Kernel/fault evidence. A zero-length log is not proof that collection succeeded; distinguish collected logs from filtered matches. |
| `*.stdout`, `*.log`, `*.out` | Process output, monitor output or command transcript; inspect the producer before assuming a schema. |
| `perf.data`, `*.bin`, `*.so`, captured package trees | Binary captures, instrumentation and historical executables. They are evidence/dependencies, not documentation or recommended installed builds. |
| `*.pid`, `__pycache__/` | Runtime residue. A saved PID does not identify a current process; bytecode is not a separate benchmark. |

`ppN` measures prompt processing for N tokens; `tgN` measures generation for
N tokens. Context depth is an additional axis, not part of the model identity.
Weight quantization (for example Q4_K_M) and K/V-cache types (for example
q8_0/q8_0) are separate controls. Fit budget, expert placement, load mode,
driver and build can change the work being measured.

For paired A/B, compare matching samples and report the paired uncertainty.
For a median over selected successful runs, retain the failed attempts and
explain the selection. Acceptance percentage, server-reported tokens/s,
end-to-end request latency, perplexity and effective KV bandwidth answer
different questions. Effective KV GB/s is a derived traffic estimate, not
a hardware memory-controller measurement.

## Audit findings that affect reuse

- **Report precedence matters.** The final matrix report withdraws its earlier
  ngram-simple recommendation after a varied-prompt rerun. Read the
  [matrix guide](MATRIX-GUIDE.md) before quoting phase-1 numbers.
- **Provenance failures are preserved.** Both models in `grfmode1aocc-*`,
  `grfmode1pr89-*` and `grfmode2pr89-*` report `all_cells_valid=false` for
  binary/repository commit mismatches despite `invalid_cell_count=0`.
  The [campaign catalog](CAMPAIGNS.md) distinguishes these six products.
- **Structured capture is not always complete.** Parsing each distinct
  JSON/JSONL byte sequence checked 1,297 JSON and 17 JSONL contents. The single
  failing content is [gputop.json](gputop.json), also copied under `oneapi-ab`:
  JSON parsing reaches EOF at line 1390. Other successful parses establish
  syntax only, not schema validity or measurement correctness.
- **Archived runners change the host.** Some SSH to another host, stop/start
  services, write sysfs controls, clear devcoredumps or overwrite results.
  Some quiet gates do not prevent launching after failure. The campaign
  guides identify these behaviors; do not use the archive as an unattended
  execution suite.
- **Paths are historical.** `/mnt/nvme1/oneapi-ab`, model mounts, build
  directories and SSH hosts in recorded commands are original provenance.
  Moving the logs here did not make those dependencies portable. Preserve
  recorded paths and configure new runs separately.
- **A Git checkout is not this whole archive.** Existing ignore rules omit
  many `.log`, `.so`, `.bin` and cache files. No ignore rules were changed and
  no archive was staged. Do not assume that committing just these guides
  publishes their linked raw evidence.

## Reproduce a new measurement

Use the [current harness registry](../../scripts/README.md), the
[SYCL build guide](../backend/SYCL.md) and the
[GPU protocol](../development/agents.md#gpu-protocol). Reserve the A770,
verify holders and readable before/after fault logs, and keep a timeout
around every GPU run. Do not change shared services merely to read this
archive. Pin the source/build, loaded libraries, driver/runtime, model,
corpus, placement, context and environment for both arms.

For example, this is the **baseline-only** form of the current product
harness, after those prerequisites, with paths replaced for the new run:

```bash
flock -w 5 /tmp/a770.lock timeout 1800s \
  python3 scripts/bench-a770-fork-unique.py \
    --campaign product --bin-dir /path/to/build/bin \
    --model /path/to/model.gguf --out-dir /path/to/new-results \
    --depths 0,8192 --kv-types q8_0/q8_0 --repetitions 4
```

The command is a source-checked template, not a reproduced campaign. Adding
`--env NAME=VALUE` enables a candidate arm; `--candidate-bin-dir` selects a
separate build and requires an explicit candidate environment. Use a fresh
output directory and read the selected harness's validity diagnostics.

## Refresh the index and guides

The [inventory script](../../scripts/index_benchmarks.py) uses only the Python
standard library. It walks hidden and Git-ignored files, streams SHA-256 over
regular files, and records symlinks without opening their targets. It does
not import archived Python, execute shell scripts or load captured binaries.
Run from the repository root:

```bash
python3 -m unittest discover -s scripts -p test_index_benchmarks.py
inventory_tmp=$(mktemp)
if python3 scripts/index_benchmarks.py docs/benchmarks > "$inventory_tmp"; then
  mv "$inventory_tmp" docs/benchmarks/FILES.tsv
else
  rm -f "$inventory_tmp"
  exit 1
fi
```

The temporary output prevents a failed scan from replacing the old index.
The script checks each regular file for size, modification-time and inode
changes during its read; it does not create a filesystem-wide snapshot.
Run it while artifacts are not being written. `bytes` for a symlink is its
link length, and its `sha256`/`identical_to` fields are blank.

To find a family without reading thousands of logs:

```bash
rg '^grfmode1final-' docs/benchmarks/FILES.tsv
rg '^oneapi-ab/.*/perf\.data\t' docs/benchmarks/FILES.tsv
```

After scanning, read new runners and reports, assign them to a campaign in
[CAMPAIGNS.md](CAMPAIGNS.md) or [MATRIX-GUIDE.md](MATRIX-GUIDE.md), connect
the raw paths, and revise this page's dated counts and audit findings.
Preserve superseded measurements and label corrections instead of silently
rewriting their conclusions. Validate new relative links and ASCII; check
numerical claims against raw records, not directory names.

## Limits of this curation

Inventory, byte comparisons, structured-data parsing, source inspection and
navigation checks were performed. The guides are additive; archived evidence
was not edited. No model inference, GPU correctness run, benchmark, binary
execution, hardware reconfiguration or external publication was performed.
Historical performance and runtime state still require a matched rerun to
establish present behavior.
