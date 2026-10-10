# P01 - Bench token count at depth

**Kind:** measurement methodology
**Depends on:** none
**Supplies:** re-baselined evidence for P10

## Context

Kmic-68 found that `llama-bench -n 128` at long context amortised a 2-3 s
first-token cost over too few tokens, reporting 12.2 t/s instead of 21.5 t/s.
The current product and depth-sweep examples in `AGENTS.md:203-209` both use
`-n 128`, so their long-depth results are not interchangeable with a corrected
run.

This plan changes measurement instructions and records only. It does not change
`llama-bench` or `scripts/perf/bench_spec.py`; the server harness has its own
`n_predict` setting. The P01 A770 research record is
`docs/research/kmic68-a770-rebaseline.md`, with a `Long-depth comparison`
table that retains both old and corrected observations, and a separate
`Token-count calibration` table for the controlled unequal-token experiment.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`p100-docs/FINDINGS.md`, "How the measurements lied", item 3.

## EARS requirements

- **R01.1 (Ubiquitous):** The `AGENTS.md` product-bench and depth-sweep commands shall specify `-n 512` or greater.
- **R01.2 (Event-driven):** WHEN an A770 turbo KV throughput result is recorded at or above depth 4096, the P01 A770 research record shall store its `n` value beside its throughput.
- **R01.3 (Event-driven):** WHEN the depth sweep is rerun with `-n 512` or greater, the P01 A770 research record shall label every retained `-n 128` result at or above depth 4096 as superseded and not directly comparable.
- **R01.4 (Unwanted behaviour):** IF two results use different token counts, THEN the P01 long-depth comparison table shall exclude them from a paired delta.
- **R01.5 (Event-driven):** WHEN an A770 campaign is re-baselined under P01, the P01 A770 research record shall preserve the superseded measurements with their original commands.
- **R01.6 (Event-driven):** WHEN a paired comparison is excluded for unequal token counts, the P01 long-depth comparison table shall display a token-count-mismatch label.
- **R01.7 (Event-driven):** WHEN the controlled token-count experiment completes with all other settings fixed, the P01 token-count calibration table shall record its unequal-token delta with both token counts and a calibration-only label.

## Approach

1. Change only the two `llama-bench` examples under `AGENTS.md` from
   `-n 128` to `-n 512`; retain all other example options.
2. Before replacing any baseline, create the P01 A770 research record and copy
   the old `-n 128` command, result, build, model, and driver label into its
   comparison table.
The research record is a file to create when P01 is executed; it is not a
currently resolvable code anchor for this documentation pass.
3. Run the paired correction experiment below from one checkout and one binary.
   Record raw rows and the correction in the separate calibration table before
   changing a downstream baseline; that delta is not a performance A/B.
4. Update each affected A770 table by preserving the old row, adding the new
   row, and calculating a delta only between rows with equal `n`.
5. Leave `scripts/perf/bench_spec.py` unchanged because request length is a
   different measurement contract.

## Critical files & anchors

- `AGENTS.md:203-209` - authoritative product and depth-sweep examples.
- `scripts/perf/bench_spec.py:326` - server request-length default, inspected
  only to keep it out of scope.
- `Kmic-68/llama.cpp:p100-docs/FINDINGS.md` on `p100-optimizations` - source
  observation and numeric example.

## Verification

Prerequisites: use the AOT production build, name the model and kernel driver,
stop `llama-sycl.cpp.service`, confirm sole tenancy with
`fuser -v /dev/dri/renderD128`, and wrap each command in the campaign timeout.
Run in this order with the same binary, model, environment, and driver:

```bash
timeout 900 ./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -p 0 -n 128 -d 16384 -r 3
timeout 900 ./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -p 0 -n 512 -d 16384 -r 3
```

Expected evidence:

- each command reports three repetitions at depth 16384;
- the record identifies `n=128` and `n=512` on every raw result;
- the calibration table records `100*(tps_512/tps_128-1)` with both token
  counts and a calibration-only label before any baseline row is replaced;
- a deliberate attempt to pair those rows in the ordinary long-depth comparison
  table produces a token-count-mismatch label and no percentage delta;
- the post-run two-driver fault gate from `AGENTS.md` finds no matching fault,
  and the stopped service is restarted.

## Assumptions & contingencies

- `-n 512` is the minimum corrected token count, not a promised throughput
  threshold.
- If `-n 512` cannot complete under the declared timeout, keep the old baseline
  marked uncorrected and record the timeout; do not substitute an `-n 128`
  comparison.
- Existing historical results remain evidence. This plan changes their
  comparability label, not their measured values.
- `scripts/bench-a770-fork-unique.py` has a separate product-command contract
  and remains out of scope; a campaign that uses it must update its token count
  before claiming P01-comparable results.
