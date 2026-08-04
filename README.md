# Tessera

Tessera is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for
running large language models locally with **calibrated, audited
quantization**, first-class speculative decoding, and an Apple Neural
Engine prefill path on M-series Macs.

Quantization in Tessera is not a single setting. It is a per-tensor policy
that you measure, evolve, and version, with a runtime that agrees with the
offline reference down to F16 precision. The result is a schema-versioned
receipt you can replay, audit, and re-run.

## Who this is for

- **LLM inference engineers** who want tighter quantization on top of
  llama.cpp without giving up its runtime, server, or model coverage.
- **Quantization researchers** who need reproducibility, schema-versioned
  policies, and an evidence trail from imatrix to final GGUF.
- **Apple Silicon power users** running models on M-series Macs who care
  about ANE prefill, IOSurface hand-off, and on-device throughput.

If you just want stock llama.cpp with no quantization research, this fork is
not the right starting point.

## What you get

- **Per-tensor calibration, not bulk quant knobs.** Tessera runs an
  importance matrix over real activations, then evolves a per-tensor policy
  `(ternary threshold, outlier fraction, AWQ alpha, AWQ clip)` per layer
  with a small genetic algorithm. Fitness is round-trip error against the
  BF16 source, with held-out activation rows. The result is a
  schema-versioned policy you can replay and re-run.
- **Runtime agrees with offline calibration.** The Tile640 dequant kernel
  is the ground truth. Any change to the kernel math ships with a
  calibration delta and a new threshold search. Calibration decisions land
  in versioned JSONL receipts, not in unlabelled build flags.
- **First-class speculative decoding on Apple Silicon.** DFlash and the
  DSpark markov head load as native architectures. The verifier and
  drafter share a tokenizer. Per-step telemetry
  (`llama.tessera.spec.v1`) is emitted for drafter fine-tuning via
  distillation or rejection sampling.
- **Apple Neural Engine prefill.** The verifier's prefill runs on the ANE
  through a Core ML package, with IOSurface async hand-off into the GPU.
  M-series only, opt-in via the auto-trigger; pass `--no-embedded-mtp` to
  bypass.

The internal name for this stack is "constitutional quantization":
per-tensor policy, schema-versioned evidence, runtime agreement. The word
"constitutional" is for distinguishing the project, not a category label.
If you are writing about Tessera, the safe public phrasing is "calibrated
and audited quantization" with the per-tensor GA spelled out.

## Subsystems

Tessera adds the following on top of llama.cpp:

- **Tessera-T640** - a Tile640-aware ternary + outlier quantization
  algorithm with AWQ pre-scaling, importance-weighted range selection, and
  evolutionary policy search.
- **ANE prefill** - Apple Neural Engine Core ML prefill for gemma 4 /
  qwen 3 multi-token prediction, with IOSurface async hand-off.
- **DFlash / DSpark drafters** - speculative decoding with the gemma 4
  drafter model and the deepseek markov head, loaded as first-class
  architectures.
- **Per-tensor evolutionary calibration** - small GA per tensor over
  `(ternary_threshold, outlier_fraction, awq_alpha, awq_clip)`, fitness is
  the round-trip relative Frobenius between the BF16 source and the
  dequantized reconstruction.
- **Spec-decoding telemetry** - `llama.tessera.spec.v1` JSONL with per-step
  verifier and drafter top-k distributions (top-k fields included only
  when `--telemetry-topk > 0`), used for drafter fine-tuning via
  distillation / rejection sampling. Single canonical schema; the previous
  v1/v2/v3 split is gone.

Tessera is a fork of llama.cpp, not a wrapper. The C++ changes live in the
same files as llama.cpp. The Python quantizer lives in `tools/tessera/`,
`tools/ane-mtp/`, and `tools/tile640/`.

## Quick start

### Build

Tessera is a fork of llama.cpp. It builds the same way:

```sh
cd tessera
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build --target llama-cli llama-imatrix llama-server llama-tessera -j 8
```

`llama-tessera` is the renamed quantizer binary (formerly `llama-quantize`),
reorganized into 19 named subcommands. `llama-imatrix`, `llama-cli`, and
`llama-server` are upstream-compatible.

### Quantize a model

The Tessera quantizer and per-tensor calibration are pure Python:

```sh
cd tessera
python3 tools/tile640/calibrate_quantize.py \
    --model-dir /path/to/safetensors \
    --f16-model /path/to/f16.gguf \
    --calibration-data /path/to/calib.txt \
    --output /path/to/tessera-q4km.gguf \
    --imatrix /path/to/calib.imatrix.gguf
```

### Subcommand interface

The Tessera quantizer is organized as 19 named subcommands. Run
`llama-tessera --help` to see the list, or `llama-tessera <subcommand> --help`
for a subcommand's specific flags:

```
Subcommands (run `llama-tessera <subcommand> --help` for details):
  accept           G6 acceptance gate tuning
  adapt            guarded one-shot adaptation step
  anonymize        tier-2 escalation text scrub
  awq              AWQ per-tensor tuning
  calibrate        calibration pass; --only runs calibrate then exits
  capability       per-axis capability score reduction
  champq           enable CHAMP-Q permutation for the current quantize run
  dataset          prepare drafter training data from spec JSONL
  dpace            compute D-PACE adaptive position weights from DFlash telemetry
  evolve           GA tuning; --only runs GA then exits
  ga               GA checkpoint resume
  kernel-fitness   L1 sidecar kernel-direct fitness blend
  l15              L1.5 reference sidecar dtype (f16 | f32)
  l2               L2 forward-pass differential output
  l5               L5 adaptive requantize loop tuning
  policy           calibration policy I/O and range selection
  runtime-probe    L2 forward-pass orchestrator marker
  throughput       north-star batched-throughput workload harness
  w4a4             enable W4A4 activation quantization for the current quantize run
```

Each subcommand only sees its own flag set. A config file (`--tessera-config
FILE` or the `TESSERA_CONFIG` env var) can supply default values for any
subcommand, with `CLI > env > config` precedence. See `examples/tessera-config.ini`
for the supported sections and key names.

## Calibration pipeline

The full pipeline - telemetry, per-tensor GA, runtime-aware requantization -
is documented in [`docs/pipeline-design.md`](docs/pipeline-design.md). The
short version:

1. **imatrix** - run `llama-imatrix` to get per-tensor importance. Clean
   `gemma4-12b-rich.imatrix.gguf` is the canonical calibration data.
2. **per-tensor GA** - run `tools/tessera/per_tensor_calibrate.py` on the
   layer bundles to find `(ternary_threshold, outlier_fraction, awq_alpha,
   awq_clip)` per tensor. Output: a per-tensor JSON policy.
3. **quantize** - `tools/tile640/calibrate_quantize.py` consumes the
   policy and writes a Tessera-T640 GGUF.
4. **end-to-end probe** - generate text on standard prompts, compare to
   F16 generation. The verifier's coherence is the calibration target.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) - Tessera architecture
  and merge strategy.
- [`docs/pipeline-design.md`](docs/pipeline-design.md) - full calibration
  pipeline (L1-L6 layers).
- [`docs/runtime-aware-pipeline.md`](docs/runtime-aware-pipeline.md) -
  layer-by-layer pipeline status.
- [`docs/speculative.md`](docs/speculative.md) - speculative decoding
  architecture.
- [`docs/audit-2026-07-29.md`](docs/audit-2026-07-29.md) - the quality
  bar every change must meet.
- [`docs/tier2-subcommand-design.md`](docs/tier2-subcommand-design.md) -
  the subcommand interface design.
- [`docs/PROJECT-STATUS.md`](docs/PROJECT-STATUS.md) - per-subsystem
  status table.

## Repository layout

```
tessera/
├── LICENSE                  # llama.cpp MIT license (applies to upstream code)
├── LICENSE-TESSERA          # PolyForm Noncommercial License 1.0.0 (Tessera code)
├── NOTICE                   # patent disclosure + upstream attributions
├── TESSERA-LICENSING.md     # licensing notes (multi-license layering)
├── docs/                    # architecture, audit, pipeline design
├── common/                  # llama.cpp common/ + tessera additions
├── src/                     # llama.cpp src/ + tessera additions
├── tools/
│   ├── tessera/             # Python quantizer (AWQ evolution, etc.)
│   ├── ane-mtp/             # ANE prefill Python toolkit
│   ├── dspark-gguf-patch/   # legacy dspark .gguf preprocessor
│   ├── tile640/             # main quantizer + orchestrator
│   ├── imatrix/             # llama-imatrix + spec-decoding hook
│   ├── server/              # llama-server + auto-MTP trigger
│   ├── quantize/            # llama-tessera (renamed from llama-quantize)
│   └── cli/                 # llama-cli
└── tests/                   # C++ tests
```

The C++ additions in `common/`, `src/`, `tools/imatrix/`, and `tools/server/`
are committed directly to the same files as llama.cpp. This is intentional -
Tessera is a fork, not a downstream patch set.

## License

This repository is a multi-licensed work. The split is:

- **PolyForm Noncommercial License 1.0.0** (see `LICENSE-TESSERA` and
  `NOTICE`). Applies to software authored specifically for Tessera: the
  Python quantizer under `tools/tile640/` and `tools/tessera/`, the ANE
  prefill toolkit under `tools/ane-mtp/`, the Tessera quantizer binary
  under `tools/quantize/`, the calibration telemetry schemas, and the
  first-party calibration / training datasets. Noncommercial purposes
  only, including personal research and use by charitable, educational,
  public research, public safety, health, environmental, and government
  organizations. Commercial use requires a separate agreement with
  `julian@tribunus.dev`.
- **MIT** (see `LICENSE`). Applies to the upstream llama.cpp and ggml code,
  and to Tessera additions that occur in files containing upstream MIT
  code, to the extent of the upstream contribution. Nothing in the
  PolyForm Noncommercial terms withdraws, replaces, or restricts rights
  already granted under MIT or another third-party license.
- **Apache-2.0** for the `tools/dspark-gguf-patch/` preprocessor (DeepSeek
  dspark upstream).

The Tessera software includes technology that may be covered by one or more
pending patent applications owned or controlled by Julian Alejandro Torres
Nieto, Tribunus.dev. The PolyForm Noncommercial License includes a Patent
License covering claims the licensor can license, subject to the
noncommercial purpose restriction and the Patent Defense provision. See
`NOTICE` for the full disclosure.

The full layering rules and the artifact notice required for
Tessera-published GGUF and safetensors are in `TESSERA-LICENSING.md` and
`docs/TESSERA_ARTIFACT_LICENSE_NOTICE.md`.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor
expectations, including the AI usage policy. The architecture and merge
strategy live in [`docs/architecture.md`](docs/architecture.md); the
audit and quality bar every change must meet is in
[`docs/audit-2026-07-29.md`](docs/audit-2026-07-29.md). Each contribution
should land on its own feature branch in a worktree, with a clean merge
to `main` once tests pass.

## Acknowledgments

Tessera is built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[ggml](https://github.com/ggml-org/ggml). The upstream MIT license terms
govern those contributions; see `LICENSE`.

The DFlash drafter architecture, the DSpark markov head, the Tile640
dequant kernel, and the AWQ pre-scaling algorithm are third-party
contributions with their own provenance. Each is attributed at its point
of inclusion in the source tree.
