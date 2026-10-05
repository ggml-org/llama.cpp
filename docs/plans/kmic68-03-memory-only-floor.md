# P03 - Memory-only floor ceiling analysis

**Kind:** methodology
**Depends on:** none

## Purpose

The technique: stub out the arithmetic while keeping every load and staging
operation, measure the resulting time, and treat it as the floor below which no
amount of dot-product work can go. Kmic-68 used it to prove 40 t/s unreachable on
their card rather than argue it: the full kernel ran 164.5 us, loads-only ran
114.5 us, so 40 t/s would have required the whole kernel inside 84 us, below the
memory-only floor.

Our central open question, that TurboQuant is dequant-compute-bound on the A770,
has been argued from mechanism and never measured this way.

## Source

- Kmic-68 `OPTLOG.md`, section *Ceiling analysis: why 40 t/s is not reachable*.

## In this fork

- `ggml/src/ggml-sycl/fattn-vec.hpp`, the turbo KQ and V loops that P11 and P12
  would change.

## Requirements

- **R03.1** (ubiquitous) The <turbo ceiling analysis> shall separate measured memory time from measured arithmetic time in the FA VEC kernel.
- **R03.2** (event-driven) WHEN the arithmetic is stubbed and every load and staging operation is retained, the <harness> shall report the memory-only floor in microseconds per run.
- **R03.3** (event-driven) WHEN the floor is known, the <analysis> shall compute the maximum throughput available at that floor, independent of dot-product quality.
- **R03.4** (unwanted) IF a proposed optimisation cannot reach its target on the floor arithmetic alone, THEN the <analysis> shall record the avenue as arithmetically unavailable rather than as an open problem.
- **R03.5** (state-driven) WHILE the stub build is compiled, the <harness> shall skip all output-correctness assertions.
- **R03.6** (unwanted) IF the loaded K/V values and their staging become dead once the arithmetic is removed, THEN the <stub> shall be rejected as a floor, because the compiler is free to delete the loads and the timing understates memory cost.
- **R03.7** (event-driven) WHEN the stub build is produced, the <harness> shall verify from the generated device code, or through an observable dependency that forces the loads, that the memory operations survived.

## Acceptance

A recorded floor for turbo3 and turbo4 on the A770 with the kernel driver named,
and the arithmetic ceiling recomputed for each. P11 is judged against this number.
