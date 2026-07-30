# Tessera

**Constitutional quantization for llama.cpp.** Tessera is a research and
production fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that adds:

- **Tessera-T640** — a Tile640-aware ternary + outlier quantization
  algorithm with AWQ pre-scaling, importance-weighted range selection, and
  evolutionary policy search.
- **ANE prefill** — Apple Neural Engine Core ML prefill for gemma 4 / qwen 3
  multi-token prediction, with IOSurface async hand-off.
- **DFlash / DSpark drafters** — speculative decoding with the gemma 4
  drafter model and the deepseek markov head, loaded as first-class
  architectures.
- **Per-tensor evolutionary calibration** — small GA per tensor over
  `(ternary_threshold, outlier_fraction, awq_alpha, awq_clip)`, fitness is
  the round-trip relative Frobenius between the BF16 source and the
  dequantized reconstruction.
- **Spec-decoding telemetry** — `llama.spec_calib.v3` JSONL with per-step
  verifier and drafter top-k distributions, used for drafter fine-tuning
  via distillation / rejection sampling. v3 is a strict superset of v1
  (`llama.dflash.acceptance.v1`) and the legacy v2 schema; the legacy
  v1 schema is still emitted as a documented adapter via
  `--telemetry-v1-compat`.

Tessera is a fork of llama.cpp, not a wrapper. The C++ changes live in the
same files as llama.cpp. The Python quantizer lives in `tools/tessera/`,
`tools/ane-mtp/`, and `tools/tile640/`.

## Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Tessera-T640 quantizer (Python) | **Production** | AWQ + outliers + per-tensor GA. The `ternary_threshold` knob was added 2026-07-29 in response to the layer-level error audit. |
| AWQ evolution (`awq-evolve.py`) | **Production** | Multi-generation GA with islands and MAP-Elites. Per-tensor `ternary_threshold` joined the mutation space. |
| Per-tensor calibration (`per_tensor_calibrate.py`) | **Production** | New tool 2026-07-29. Direct round-trip fitness + lossless target early stop. |
| DFlash drafter (`models/dflash.cpp`) | **Production** | DSpark folded into DFlash per upstream PR #25173. |
| DSpark markov head | **Production** | Loaded from `markov_w1`/`markov_w2`/`conf_proj` tensors. |
| Spec hook in `llama-imatrix` | **Production** | `--model-draft`, `--telemetry-out`, `--telemetry-topk`, `--spec-steps`. |
| V3 telemetry schema (`llama.spec_calib.v3`) | **Production** | Per-position verifier + drafter top-k. Strict superset of v1 + v2. v1 available as documented adapter via `--telemetry-v1-compat`. |
| ANE prefill (`common/ane-mtp.mm`) | **WIP** | Implementation present, full integration with the verifier's MTP context is not yet wired. `--no-embedded-mtp` flag bypasses the auto-trigger. |
| dft. observer protocol (`llama-graph.cpp`) | **WIP** | String-prefix hack. See `docs/audit-2026-07-29.md`. To be replaced with proper per-context observer state. |
| `dspark-gguf-patch/` | **Legacy** | Preprocessor for pre-PR-#25173 dspark drafters. Will be removed once the legacy converter is no longer in production. |
| Kernel dequant debug mode | **Not started** | See `docs/pipeline-design.md` Layer 1. The foundation for runtime-aware calibration. |

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

The full pipeline — telemetry, per-tensor GA, runtime-aware requantization —
is documented in `docs/pipeline-design.md`. The short version:

1. **imatrix** — run `llama-imatrix` to get per-tensor importance. Clean
   `gemma4-12b-rich.imatrix.gguf` is the canonical calibration data.
2. **per-tensor GA** — run `tools/tessera/per_tensor_calibrate.py` on the
   layer bundles to find `(ternary_threshold, outlier_fraction, awq_alpha,
   awq_clip)` per tensor. Output: a per-tensor JSON policy.
3. **quantize** — `tools/tile640/calibrate_quantize.py` consumes the policy
   and writes a Tessera-T640 GGUF.
4. **end-to-end probe** — generate text on standard prompts, compare to
   F16 generation. The verifier's coherence is the calibration target.

## Repository layout

```
tessera/
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
This is intentional — Tessera is a fork, not a downstream patch set.

## License

- The C++ code in `common/`, `src/`, and the bulk of `tools/` follows the
  llama.cpp MIT license.
- The Tessera-specific Python quantizer in `tools/tile640/` and
  `tools/tessera/`, the ANE prefill toolkit in `tools/ane-mtp/`, and the
  calibration telemetry schemas are under the **Tessera Research and
  Education License 1.0** — see `LICENSE-TESSERA`. This is a non-commercial
  license; commercial use requires a separate agreement with
  `julian@tribunus.dev`.
- The `tools/dspark-gguf-patch/` preprocessor is Apache-2.0 (DeepSeek
  dspark upstream).

## Contributing

Each contribution should land on its own feature branch in a worktree,
with a clean merge to `main` once tests pass. See `docs/architecture.md`
for the system's invariants and `docs/audit-2026-07-29.md` for the
quality bar every new piece must meet.
