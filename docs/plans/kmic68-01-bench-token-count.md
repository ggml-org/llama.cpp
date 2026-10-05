# P01 - Bench token count at depth

**Kind:** methodology
**Depends on:** none
**Blocks:** the baseline that P10 is judged against

## Purpose

Kmic-68 measured that `llama-bench -n 128` at long context amortises a 2-3 s
first-token cost over too few tokens: it reported 12.2 t/s where the truth was
21.5, and their fix was to use 512 or more. Our `AGENTS.md` specifies `-n 128`
in both the product bench and the depth sweep, so every turbo3 tg-at-depth figure
recorded to date may be biased low by an unknown amount.

The server harness is not affected: `scripts/perf/bench_spec.py` defaults
`n_predict` to 256, which a prompt may override (line 306).

## Source

- Kmic-68 `p100-docs/FINDINGS.md`, section *How the measurements lied*, item 3.

## In this fork

- `AGENTS.md:196` and `AGENTS.md:200`, the two `llama-bench` invocations.
- `scripts/perf/bench_spec.py:306`, `n_predict` default, for contrast.

## Requirements

- **R01.1** (ubiquitous) The `llama-bench` product bench and depth-sweep commands in `AGENTS.md` shall specify `-n 512` or greater.
- **R01.2** (event-driven) WHEN a turbo KV throughput figure is recorded at a depth above 4096, the <research log> shall record the token count used alongside the throughput.
- **R01.3** (event-driven) WHEN the depth sweep is re-run under R01.1, the <research log> shall mark every pre-existing `-n 128` turbo3 tg-at-depth figure as not comparable to the new baseline.
- **R01.4** (unwanted) IF a throughput figure and the baseline it is compared against were recorded with different token counts, THEN the <reporting format> shall refuse to place them in the same paired comparison. The calibration run under R01.5 is the sole exemption: measuring the delta between token counts is the one comparison that requires them to differ.
- **R01.5** (optional feature) WHERE a campaign is re-baselined under R01.1, the <research log> shall preserve the superseded figures rather than delete them.

## Acceptance

A paired `-n 128` against `-n 512` run at `-d 16384` on the real trunk, A770,
driver named, sole tenancy, three reps. This pair is exempt from R01.4 by its own
terms. The measured delta is recorded *before* the campaign is re-baselined, so the
size of the correction is on record.
