# P02 - Minimum free memory over a run

**Kind:** methodology
**Depends on:** none

## Purpose

Kmic-68 recorded that a headroom claim of roughly 250 MiB was measured at 8%
context fill; at a full prompt the margin was negative. Their rule: take the
minimum of free memory over a whole long run, because early samples look safe
and mean nothing.

We fell into exactly this trap in PR #90. The draft context reported 152.0 MiB
against 141.0 MiB at fit time, and every sample taken during startup said the
margin was comfortable.

## Source

- Kmic-68 `p100-docs/FINDINGS.md`, section *How the measurements lied*, item 5.

## In this fork

- `common/speculative.cpp:3199-3244`, `common_speculative_checkpoint_flags`, which
  queries `ggml_backend_dev_memory` once per checkpoint placement decision.
- `scripts/perf/bench_spec.py`, the A770 campaign harness.

## Requirements

- **R02.1** (ubiquitous) The <memory sampling routine> shall report the minimum device free memory observed across the sampling window, not the value at any single point.
- **R02.2** (event-driven) WHEN a benchmark run completes, the <result summary> shall record the minimum free memory alongside the final value.
- **R02.3** (unwanted) IF a headroom claim is derived from a sample taken before the first long-context decode, THEN the <claim> shall be rejected as evidence.
- **R02.4** (optional feature) WHERE the <A770 bench harness> runs, the harness shall sample free memory per launch and emit the minimum.

## Acceptance

The same workload run once at 8% context fill and once at full fill. The minimum
catches the dip that the final sample misses, and the two runs are reported side
by side rather than the later one alone.
