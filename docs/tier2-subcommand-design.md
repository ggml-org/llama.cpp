# Tier 2: Tessera CLI subcommand restructure (agent-o)

## Goal

Collapse the flat 60-flag `--tessera-*` surface into 19 subcommands of a
single `llama-tessera` binary. HARD BREAK: all existing `--tessera-*`
flags are removed. No deprecation aliases, no back-compat layer.

## Top-level (no subcommand, main quantize path)

`llama-tessera --model X --output Y` runs the default quantize pipeline.
The top-level has only the common flags:

- `--model`, `--output-dir` (existing llama.cpp)
- `--tessera-config` (Tier 3 surface, reserved)
- `--tessera-imatrix`
- `--threads` (existing llama.cpp; aliased to `--tessera-nthreads` internally)
- Plus the existing common llama.cpp flags: `--ctx-size`, `--n-gpu-layers`,
  `-t`, etc.
- Plus the dispatch-level flags that apply to the main quantize path AND
  to any subcommand that touches a model: `--progress-file`,
  `--quantize-db`, `--force-requantize`. (These are needed by the dispatch
  at subcommand entry too, so they live at top-level.)
- Plus the debug sidecar flags: `--tessera-dequant-dir`,
  `--tessera-dequant-stride`, `--tessera-matmul-output-dir`,
  `--tessera-matmul-output-stride`. (These are pure sidecar capture; they
  apply regardless of subcommand and were previously registered against
  the common example. Keep at top-level so any subcommand can capture
  sidecars.)
- The dispatch mode (`--tessera-mode` off|default|calibrate-only|evolve-only)
  is REMOVED entirely. The equivalent functionality is reachable via the
  `calibrate --only` and `evolve --only` subcommands.

## Subcommand flag map (per-flag table)

Each row is one existing flag -> one new subcommand + new flag name. The
new flag name drops the `--tessera-` prefix and (where redundant) the
subcommand name itself. Where two subcommands would want the same short
name (e.g. `--out` in many), the one with the same field already exposed
by another subcommand is renamed (e.g. `--acceptance-out`, `--l5-out`).

### `accept` (G6 acceptance gate tuning)

| current flag | new flag | field |
|---|---|---|
| `--tessera-acceptance` | `--enabled` (bool) | `tessera_params.acceptance = true` |
| `--no-tessera-acceptance` | `--no-enabled` (negated bool) | `tessera_params.acceptance = false` |
| `--tessera-acceptance-out` | `--report` | `tessera_params.acceptance_out` |

Note: short `--enabled` is too generic. Renamed:
- `--tessera-acceptance` -> `--acceptance` (the subcommand is `accept`,
  the noun is implicit)
- `--no-tessera-acceptance` -> `--no-acceptance`
- `--tessera-acceptance-out` -> `--out`

### `adapt` (one-shot guarded adaptation step)

| current flag | new flag | field |
|---|---|---|
| `--tessera-adapt` | `--eval` | `tessera_params.adapt_eval` |
| `--tessera-adapt-out` | `--out` | `tessera_params.adapt_out` |
| `--tessera-adapt-dry-run` | `--dry-run` (bool) | `tessera_params.adapt_dry_run` |
| `--tessera-adapt-epsilon` | `--epsilon` | `tessera_params.adapt_epsilon` |

### `anonymize` (tier-2 escalation scrub)

| current flag | new flag | field |
|---|---|---|
| `--tessera-anonymize` | `--in` | `tessera_params.anonymize_in` |
| `--tessera-anonymize-out` | `--out` | `tessera_params.anonymize_out` |
| `--tessera-anonymize-level` | `--level` | `tessera_params.anonymize_level` |
| `--tessera-anonymize-map` | `--map` | `tessera_params.anonymize_map` |

### `awq` (AWQ tuning)

| current flag | new flag | field |
|---|---|---|
| `--tessera-awq-alpha` | `--alpha` | `tessera_params.awq_alpha` |
| `--tessera-awq-clip` | `--clip` | `tessera_params.awq_clip` |

### `calibrate` (calibration; --only runs calibration then exits)

| current flag | new flag | field |
|---|---|---|
| `--tessera-calibrate-only` | `--only` (bool) | `tessera_params.calibrate_only` |
| `--calib-corpus` | `--corpus` | `tessera_params.calib_corpus` |
| `--calib-corpus-out` | `--corpus-out` | `tessera_params.calib_corpus_out` |

Note: `--tessera-imatrix` is top-level (per the design).

### `capability` (per-axis capability score reduction)

| current flag | new flag | field |
|---|---|---|
| `--tessera-capability-eval` | `--eval` | `tessera_params.capability_eval` |
| `--tessera-capability-out` | `--out` | `tessera_params.capability_out` |

### `champq` (CHAMP-Q permutation toggle)

| current flag | new flag | field |
|---|---|---|
| `--tessera-champq` | (the subcommand itself is the toggle) | `tessera_params.champq = true` |

`llama-tessera champq` enables CHAMP-Q for the current quantize run; no
flag needed.

### `dataset` (drafter training data prep)

| current flag | new flag | field |
|---|---|---|
| `--tessera-dataset` | `--in` | `tessera_params.dataset_in` |
| `--tessera-dataset-out` | `--out` | `tessera_params.dataset_out` |
| `--tessera-dataset-mode` | `--mode` | `tessera_params.dataset_mode` |

### `dpace` (D-PACE adaptive position weights)

| current flag | new flag | field |
|---|---|---|
| `--tessera-dpace` | (positional) | `tessera_params.dpace_in` |
| `--tessera-dpace-out` | `--out` | `tessera_params.dpace_out` |
| `--tessera-dpace-alpha` | `--alpha` | `tessera_params.dpace_alpha` |
| `--tessera-dpace-gamma` | `--gamma` | `tessera_params.dpace_gamma` |

Note: `llama-tessera dpace <path>` is the input (like `cp`); the
subcommand-style positional is consistent with `anonymize --in` if we
prefer a flag. Kept as positional for dpace (matches the existing
"compute from this file" usage pattern of `runtime-probe`).

### `evolve` (GA tuning; --only runs GA then exits)

| current flag | new flag | field |
|---|---|---|
| `--tessera-evolve-only` | `--only` (bool) | `tessera_params.evolve_only` |
| `--tessera-evolve-iters` | `--iters` | `tessera_params.evolve_iters` |
| `--tessera-evolve-islands` | `--islands` | `tessera_params.evolve_islands` |
| `--tessera-evolve-population` | `--population` | `tessera_params.evolve_population` |
| `--tessera-evolve-seed` | `--seed` | `tessera_params.evolve_seed` |

### `ga` (GA checkpoint)

| current flag | new flag | field |
|---|---|---|
| `--tessera-ga-checkpoint` | `--checkpoint` | `tessera_params.ga_checkpoint` |

### `kernel-fitness` (L1 sidecar kernel-direct fitness)

| current flag | new flag | field |
|---|---|---|
| `--tessera-kernel-fitness` | `--enabled` (bool) | `tessera_params.kernel_fitness` |
| `--tessera-kernel-fitness-dir` | `--dir` | `tessera_params.kernel_fitness_dir` |
| `--tessera-kernel-fitness-blend` | `--blend` | `tessera_params.kernel_fitness_blend` |

### `l15` (L1.5 reference sidecar dtype)

| current flag | new flag | field |
|---|---|---|
| `--tessera-l15-dtype` | `--dtype` | `tessera_debug::set_l15_dtype` |

### `l2` (L2 forward-pass differential output)

| current flag | new flag | field |
|---|---|---|
| `--tessera-l2-out` | `--out` | `tessera_params.runtime_probe_l2_out` |

### `l5` (L5 adaptive requantize loop)

| current flag | new flag | field |
|---|---|---|
| `--tessera-l5-out` | `--out` | `tessera_params.l5_out` |
| `--tessera-l5-generations` | `--generations` | `tessera_params.l5_max_generations` |
| `--tessera-l5-flag-multiplier` | `--flag-multiplier` | `tessera_params.l5_flag_multiplier` |
| `--tessera-l5-alpha-min` | `--alpha-min` | `tessera_params.l5_alpha_min` |
| `--tessera-l5-clip-min` | `--clip-min` | `tessera_params.l5_clip_min` |
| `--tessera-l5-outlier-overshoot-scale` | `--outlier-overshoot-scale` | `tessera_params.l5_outlier_overshoot_scale` |
| `--tessera-l5-outlier-frac-cap` | `--outlier-frac-cap` | `tessera_params.l5_outlier_frac_cap` |
| `--tessera-adaptive-requantize` | `--enabled` (bool) | `tessera_params.adaptive_requantize` |
| `--no-tessera-adaptive-requantize` | `--no-enabled` (negated bool) | `tessera_params.adaptive_requantize = false` |

### `policy` (calibration policy + range selection)

| current flag | new flag | field |
|---|---|---|
| `--tessera-policy` | `--in` | `tessera_params.policy` |
| `--tessera-policy-out` | `--out` | `tessera_params.policy_out` |
| `--tessera-outlier-frac` | `--outlier-frac` | `tessera_params.outlier_frac` |
| `--tessera-range-selection` | `--range-selection` | `tessera_params.range_selection` |
| `--tessera-ternary-threshold` | `--ternary-threshold` | `tessera_params.ternary_threshold` |

### `runtime-probe` (L2 forward-pass orchestrator marker)

| current flag | new flag | field |
|---|---|---|
| `--tessera-runtime-probe` | (positional) | `tessera_params.runtime_probe` |
| `--tessera-runtime-probe-bf16` | `--bf16` | `tessera_params.runtime_probe_bf16` |
| `--tessera-runtime-probe-bf16` (out) | `--bf16` | `tessera_params.runtime_probe_bf16` |

`runtime-probe` and `l2` overlap on `--tessera-runtime-probe-l2-out`:
this is the l2 subcommand's `--out`. Both kept (different fields).

### `throughput` (north-star batched-throughput workload harness)

| current flag | new flag | field |
|---|---|---|
| `--tessera-throughput` | `--workload` | `tessera_params.throughput_workload` |
| `--tessera-throughput-out` | `--out` | `tessera_params.throughput_out` |

### `w4a4` (W4A4 activation quantization toggle + outlier threshold)

| current flag | new flag | field |
|---|---|---|
| `--tessera-w4a4` | (the subcommand is the toggle) | `tessera_params.w4a4 = true` |
| `--tessera-w4a4-outlier-thresh` | `--outlier-thresh` | `tessera_params.w4a4_outlier_thresh` |

`llama-tessera w4a4` enables W4A4 for the current quantize run; use
`--outlier-thresh` to override the LLM.int8 threshold.

## Top-level flags (NOT migrated, stay at top-level)

- `--tessera-imatrix` (per the architect's design)
- `--tessera-nthreads` (per the design; exposed as `--threads` which is
  the existing llama.cpp flag, but `common_tessera_params.nthreads` is
  the dispatcher override)
- `--progress-file`, `--quantize-db`, `--force-requantize` (dispatch-level)
- `--tessera-dequant-dir`, `--tessera-dequant-stride`,
  `--tessera-matmul-output-dir`, `--tessera-matmul-output-stride`
  (debug sidecar capture, applies to any subcommand)

## Removed flags

- `--tessera-mode` (replaced by `calibrate --only` / `evolve --only`)

## Subcommand dispatch (STEP 2 design choice)

The existing `llama_example` enum already has per-example scoping for
`add_opt`. The cleanest extension is to add a single
`LLAMA_EXAMPLE_TESSERA` and a parallel `enum tessera_subcommand`
(`TESSERA_SC_ACCEPT`, `_ADAPT`, ..., `_W4A4`). At parse time,
`common_params_parse` inspects `argv[1]` and:

1. If it is a known subcommand name, sets `ctx_arg.tessera_sc` to that
   subcommand and recurses with the rest of argv shifted by one.
2. Otherwise, treats it as the legacy top-level quantize path (no
   subcommand scope; only top-level flags are visible).

This avoids inflating the `llama_example` enum (one new variant covers
all 19 subcommands) and keeps the per-subcommand filtering in the
existing `add_opt` lambda. Per-subcommand flags are tagged with
`set_examples({LLAMA_EXAMPLE_TESSERA, LLAMA_EXAMPLE_TESSERA_<X>})`
where `<X>` is the specific subcommand; the lambda in
`common_params_parser_init` filters by both `llama_example` and
`tessera_sc`.

The top-level Tessera flags (imatrix, nthreads, progress-file,
quantize-db, force-requantize, dequant-*, matmul-output-*) are tagged
with `set_examples({LLAMA_EXAMPLE_TESSERA})` only - they are always
visible regardless of which subcommand is active.

## Hard break verification (STEP 4)

After migration, `grep -n '"--tessera-\|"--no-tessera-\|"--calib-corpus\|--progress-file\|--quantize-db\|--force-requantize' common/arg.cpp` should return zero hits in:
- `add_opt` blocks (no registrations)
- `common_tessera_parse_one` body (no parser handlers)

`llama-tessera --tessera-evolve-iters 4` errors with "unrecognized
argument". `llama-tessera evolve --iters 4` succeeds (or fails on a
downstream validation, not the parse).

## Help system (STEP 5)

`llama-tessera --help` lists:
- The 5 (or so) top-level Tessera flags
- A "Subcommands:" section with the 19 subcommand names + one-line
  description each
- Points users to `llama-tessera <subcommand> --help`

`llama-tessera <subcommand> --help` lists:
- The top-level flags (model, imatrix, output-dir, threads, config, etc.)
- The subcommand's specific flags
- A one-line description of the subcommand

Implementation: a new `print_tessera_subcommand_list()` helper for the
top-level help; per-subcommand help uses the existing
`common_params_print_usage` with the subcommand's flag set.
