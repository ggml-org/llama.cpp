# Tessera

Tessera is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for
running large language models locally with **calibrated, audited
quantization**, first-class speculative decoding, and an Apple Neural
Engine prefill path on M-series Macs.

Quantization in Tessera is not a single setting. It is a per-tensor
policy that you measure, evolve, and version, with a runtime that agrees
with the offline reference down to F16 precision. The result is a
schema-versioned receipt you can replay, audit, and re-run.

## Who this is for

- **LLM inference engineers** who want tighter quantization on top of
  llama.cpp without giving up its runtime, server, or model coverage.
- **Quantization researchers** who need reproducibility, schema-versioned
  policies, and an evidence trail from imatrix to final GGUF.
- **Apple Silicon power users** running models on M-series Macs who
  care about ANE prefill, IOSurface hand-off, and on-device throughput.

If you just want stock llama.cpp with no quantization research, this
fork is not the right starting point.

## What you get

- **Per-tensor calibration, not bulk quant knobs.** Tessera runs an
  importance matrix over real activations, then evolves a per-tensor
  policy `(ternary threshold, outlier fraction, AWQ alpha, AWQ clip)`
  per layer with a small genetic algorithm. Fitness is round-trip error
  against the BF16 source, with held-out activation rows. The result is
  a schema-versioned policy you can replay and re-run.
- **Runtime agrees with offline calibration.** The Tile640 dequant
  kernel is the ground truth. Any change to the kernel math ships with
  a calibration delta and a new threshold search. Calibration decisions
  land in versioned JSONL receipts, not in unlabelled build flags.
- **First-class speculative decoding on Apple Silicon.** DFlash and the
  DSpark markov head load as native architectures. The verifier and
  drafter share a tokenizer. Per-step telemetry
  (`llama.spec_calib.v2`) is emitted for drafter fine-tuning via
  distillation or rejection sampling.
- **Apple Neural Engine prefill.** The verifier's prefill runs on the
  ANE through a Core ML package, with IOSurface async hand-off into
  the GPU. M-series only, opt-in via the auto-trigger; pass
  `--no-embedded-mtp` to bypass.

`Constitutional quantization` is the internal name for this stack:
per-tensor policy, schema-versioned evidence, runtime agreement. The
`constitutional` word is for distinguishing the project, not a category
label. If you are writing about Tessera, the safe public phrasing is
"calibrated and audited quantization" with the per-tensor GA spelled
out.

## Subsystems

Tessera adds the following on top of llama.cpp:

- **Tessera-T640** - a Tile640-aware ternary + outlier quantization
  algorithm with AWQ pre-scaling, importance-weighted range selection,
  and evolutionary policy search.
- **ANE prefill** - Apple Neural Engine Core ML prefill for gemma 4 /
  qwen 3 multi-token prediction, with IOSurface async hand-off.
- **DFlash / DSpark drafters** - speculative decoding with the gemma 4
  drafter model and the deepseek markov head, loaded as first-class
  architectures.
- **Per-tensor evolutionary calibration** - small GA per tensor over
  `(ternary_threshold, outlier_fraction, awq_alpha, awq_clip)`, fitness
  is the round-trip relative Frobenius between the BF16 source and the
  dequantized reconstruction.
- **Spec-decoding telemetry** - `llama.spec_calib.v2` JSONL with
  per-step verifier and drafter top-k distributions, used for drafter
  fine-tuning via distillation / rejection sampling.

Tessera is a fork of llama.cpp, not a wrapper. The C++ changes live in
the same files as llama.cpp. The Python quantizer lives in
`tools/tessera/`, `tools/ane-mtp/`, and `tools/tile640/`.

## Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Tessera-T640 quantizer (Python) | **Production** | AWQ + outliers + per-tensor GA. The `ternary_threshold` knob was added 2026-07-29 in response to the layer-level error audit. |
| AWQ evolution (`awq-evolve.py`) | **Production** | Multi-generation GA with islands and MAP-Elites. Per-tensor `ternary_threshold` joined the mutation space. |
| Per-tensor calibration (`per_tensor_calibrate.py`) | **Production** | New tool 2026-07-29. Direct round-trip fitness + lossless target early stop. |
| DFlash drafter (`models/dflash.cpp`) | **Production** | DSpark folded into DFlash per upstream PR #25173. |
| DSpark markov head | **Production** | Loaded from `markov_w1`/`markov_w2`/`conf_proj` tensors. |
| Spec hook in `llama-imatrix` | **Production** | `--model-draft`, `--telemetry-out`, `--telemetry-topk`, `--spec-steps`. |
| V2 telemetry schema (`llama.spec_calib.v2`) | **Production** | Per-position verifier + drafter top-k. |
| ANE prefill (`common/ane-mtp.mm`) | **WIP** | Implementation present, full integration with the verifier's MTP context is not yet wired. `--no-embedded-mtp` flag bypasses the auto-trigger. |
| dft. observer protocol (`llama-graph.cpp`) | **WIP** | String-prefix hack. See `docs/audit-2026-07-29.md`. To be replaced with proper per-context observer state. |
| `dspark-gguf-patch/` | **Legacy** | Preprocessor for pre-PR-#25173 dspark drafters. Will be removed once the legacy converter is no longer in production. |
| Kernel dequant debug mode (L1 sidecar) | **Production** | v3 TDQT sidecar in `common/tessera-debug/`, wired into all three backends; consumed as the GA's kernel-direct fitness in `tessera-dispatch.cpp` (L6). See `docs/runtime-aware-pipeline.md` for the full layer-by-layer status: L1 + L6 shipped; L1.5 partial (suffix bug); L2/L3 weight-level and per-row; L4 partial PPL substitute; L5 library + tests, not yet on the dispatch path. |

## Build

Tessera is a fork of llama.cpp. It builds the same way:

```sh
cd tessera
cmake -B build
cmake --build build --target llama-cli llama-imatrix llama-server -j 8
```

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

## Calibration pipeline

The full pipeline - telemetry, per-tensor GA, runtime-aware
requantization - is documented in `docs/pipeline-design.md`. The short
version:

1. **imatrix** - run `llama-imatrix` to get per-tensor importance.
   Clean `gemma4-12b-rich.imatrix.gguf` is the canonical calibration
   data.
2. **per-tensor GA** - run `tools/tessera/per_tensor_calibrate.py` on
   the layer bundles to find `(ternary_threshold, outlier_fraction,
   awq_alpha, awq_clip)` per tensor. Output: a per-tensor JSON policy.
3. **quantize** - `tools/tile640/calibrate_quantize.py` consumes the
   policy and writes a Tessera-T640 GGUF.
4. **end-to-end probe** - generate text on standard prompts, compare
   to F16 generation. The verifier's coherence is the calibration
   target.

## Repository layout

```
tessera/
├── LICENSE                  # llama.cpp MIT license (applies to upstream code)
├── LICENSE-TESSERA          # the Tessera research + education license
├── TESSERA-LICENSING.md     # licensing notes
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
│   └── cli/                 # llama-cli
└── tests/                   # C++ tests
```

The C++ additions in `common/`, `src/`, `tools/imatrix/`, and
`tools/server/` are committed directly to the same files as llama.cpp.
This is intentional - Tessera is a fork, not a downstream patch set.

## License

This repository is a multi-licensed work. The split is:

- **Tessera Research and Education License 1.0** (see `LICENSE-TESSERA`).
  Applies to software authored specifically for Tessera: the Python
  quantizer under `tools/tile640/` and `tools/tessera/`, the ANE
  prefill toolkit under `tools/ane-mtp/`, the calibration telemetry
  schemas, and the first-party calibration / training datasets.
  Noncommercial education and noncommercial research only. Commercial
  use requires a separate agreement with `julian@tribunus.dev`.
- **MIT** (see `LICENSE`). Applies to the upstream llama.cpp and ggml
  code, and to Tessera additions that occur in files containing upstream
  MIT code, to the extent of the upstream contribution. Nothing in the
  Tessera terms withdraws, replaces, or restricts rights already
  granted under MIT or another third-party license.
- **Apache-2.0** for the `tools/dspark-gguf-patch/` preprocessor
  (DeepSeek dspark upstream).

The full layering rules and the artifact notice required for
Tessera-published GGUF and safetensors are in `TESSERA-LICENSING.md`
and `docs/TESSERA_ARTIFACT_LICENSE_NOTICE.md`.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor
expectations, including the AI usage policy. The architecture and
merge strategy live in `docs/architecture.md`; the audit and quality
bar every change must meet is in `docs/audit-2026-07-29.md`. Each
contribution should land on its own feature branch in a worktree, with
a clean merge to `main` once tests pass.
