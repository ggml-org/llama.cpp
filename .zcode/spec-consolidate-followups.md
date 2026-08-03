# spec-consolidate agent-g: follow-up dispatch

## Status: APPROVED

Scratch branch: `scratch/spec-consolidate/agent-g` @ `697fd3ae4`
Review branch: `evolve-review/spec-consolidate/agent-g` @ `697fd3ae4`

User has explicitly approved the refactor. The review branch tip is ready to be
promoted to main when the user is ready. Do not auto-promote.

## Pre-existing bugs found in baseline (NOT introduced by agent-g)

These were discovered while testing the agent-g refactor on the build dir.
They are not regressions. Each is logged in `.zcode/alphaevolve/findings.jsonl`
and ready for `findings-curator fix <N>` to pick up.

### Bug 1 (high): test-spec-calibration model-based path aborts

- **File**:tests/test-spec-calibration.cpp:185-217
- **File**:common/speculative.cpp:2340-2344 (the `has_draft || spec_mtp` assert)
- **File**:common/common.h:390 (the `has_dft()` check, which only looks at
  `draft.mparams`)

The test sets `params_dft.speculative.draft.target_model_path` then calls
`common_speculative_init_from_params()`. The init function asserts
`has_draft || spec_mtp`, but `has_dft()` only checks `!draft.mparams.empty()`.
The test does not set `mparams`, so the assert aborts. Same failure on
`main @ 7034c6ace` (confirmed). NOT introduced by agent-g.

**Suggested fix**: either (a) update the test to set `draft.mparams` via
`common_base_params_to_speculative` + a real drafter model load, or (b) relax
`has_dft()` to also accept `!draft.target_model.empty()`.

### Finding 2 (medium): Duplicate spec-calib paths

- **File**:tools/imatrix/imatrix.cpp:2062 (`compute_imatrix_spec`, inlined)
- **File**:common/speculative-calibration.cpp (`common_speculative_calibration_run`)

Both paths emit the spec-decoding telemetry. The new `common/` module is the
operational path; the inlined `imatrix` path is dead code but still compiles.
The 20f75c8cf commit message acknowledges this duplication. agent-g's
emission fix is in the new module only; the old path is unaffected. If either
path diverges (e.g. one is updated for a new field) the other will silently
emit the old shape.

**Suggested fix**: delete `compute_imatrix_spec` from imatrix.cpp and have the
imatrix main path call `common_speculative_calibration_run` directly. This is
a small refactor (the imatrix wiring is ~30 lines of observer hooks).

## Suggested dispatch

Use the `findings-curator` agent (`Agent(subagent_type="findings-curator")`)
in `fix 2` mode:

```
fix 2
```

The two items above are well-scoped, both have < 1-day effort estimates, and
they unblock the test-spec-calibration coverage (Bug 1) and reduce future
drift risk (Finding 2). The findings ledger already has both entries; the
backlog file should pick them up at the top.

## What's NOT in scope for the follow-up

- The spec-consolidate refactor itself is approved; do NOT revisit field
  ordering, schema name, or emission behavior.
- The brief's "test fixture refactor" in test_dataset.cpp /
  test_dflash_train_data.cpp is intentional (use `some.other.schema` as the
  wrong-schema fixture). Do not "fix" that.
- The `import sys` cosmetic in test-telemetry-schema-stability.py is a no-op;
  not worth a separate commit.
