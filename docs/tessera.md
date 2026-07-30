# Tessera quantization

Tessera is the model-level quantization architecture implemented by the
TSQ encoding. GGUF remains the container, while Tessera describes how learned
tensors are represented and reconstructed. T640 is the first physical Tessera
layout and retains the established 640-value kernel tile.

The canonical calibrated unified profile is `TSQ-T640-AWQ-SR-U`. `TSQ`
identifies Tessera Quantization, `T640` identifies the physical layout, `AWQ`
identifies activation-aware scaling and outlier selection, `SR` identifies the
sparse residual refinement, and `U` identifies unified component packaging.
Profiles without calibration omit `AWQ`; profiles without unified components
omit `U`.

| GGUF metadata | Meaning |
| --- | --- |
| `tessera.name` | Reader-facing format name |
| `tessera.version` | Tessera metadata and representation contract version |
| `tessera.profile` | Complete abbreviated encoding profile |
| `tessera.features` | Machine-readable enabled feature names |
| `tessera.core.*` | Core numerical representation |
| `tessera.layout.*` | Physical page, lane, and packed-word geometry |
| `tessera.scale.*` | Page-scale and lane-scale storage types |
| `tessera.residual.*` | Sparse refinement representation |
| `tessera.sensitive.exact` | Whether sensitive tensors use exact residual representation |
| `tessera.calibration.*` | Calibration inputs used during quantization |
| `tessera.coverage` | Learned-tensor coverage contract |
| `tessera.passthrough` | Whether conventional source tensors remain |
| `tessera.unified` | Whether additional model components share the GGUF |
| `tessera.source.*` | BF16 source epoch, hashes, size, and tensor count |
| `tessera.dataset.*` | Calibration dataset epoch and aggregate evidence digest |
| `tessera.shape.<tensor>` | Logical shape of a generic componentized tensor |
| `tessera.matrix_shape.<tensor>` | Matrix view used by the T640 encoding |

Tessera v1 uses a balanced ternary core, BF16 page scales, INT8 lane-scale
codes, and row-sparse F16 residual values. Sensitive vectors, norms, biases,
routers, and other protected tensors remain exact through the same component
representation rather than conventional passthrough tensors.

The loader detects T640 weights by their component tensors. New files use the
`tessera.shape.*` namespace for generic text and multimodal tensors. The loader
continues accepting `tile640.shape.*` and `tile640.matrix_shape.*` so existing
experimental GGUFs remain usable. Tile640 remains the internal kernel and
physical-layout name; it is no longer the reader-facing format name.

## Rich calibration and evolutionary AWQ

The graph-resident importance observer collects four compact per-channel
statistics while activations remain on their execution backend:
`sum(x²)`, `sum(abs(x))`, `sum(x⁴)`, and `max(abs(x))`. The GGUF imatrix keeps
the original `.in_sum2` and `.counts` tensors and adds `.in_sumabs`,
`.in_sum4`, and `.in_maxabs`, so older quantizers retain their existing input
while Tessera can model activation tails and clipping pressure.

On Metal, the ordinary observer and the F32-to-F16 activation cast remain
independent graph nodes with read-only access to the same activation. Metal's
concurrent graph encoder may therefore interleave the tiled reduction with the
cast while Tile640 consumes the resulting F16 activation cache. An experimental
direct-F32 Tile640 input specialization is available with
`TESSERA_TILE640_F32_INPUT=1`, and the combined heterogeneous cast-observer
allocation is available with `TESSERA_FUSED_CAST_OBSERVER=1`. Both are opt-in:
full-model Gemma 4 12B measurements showed that repeatedly reading F32
activations, or placing all observer work on the cast's critical path, is slower
than materializing the reusable F16 activation cache and concurrently reducing
compact statistics.

Adaptive convergence is coverage-aware for routed models. `--min-expert-coverage`
defaults to `0.25`, so a routed tensor cannot freeze only because aggregate
moments look stable while its least-observed expert has received fewer than a
quarter of the samples of its most-observed expert. Dense tensors are
unaffected, allowing one calibration profile to safely serve dense, MoE, and
multimodal architectures.

`tools/tessera/awq-evolve.py` searches AWQ alpha, clipping, sparse-residual
fraction, moment mixing, and tail protection. It uses deterministic island
populations and a MAP-Elites archive, scores held-out activations when
available, and checkpoints the complete population and random state. Its
output uses `llama.speculative.calibration-policy.v1` and can refine an
existing DFlash/MTP acceptance policy without discarding its telemetry.

When AWQ evolution is enabled, `tessera-calibrate` automatically runs the
shadow-calibration stage before writing the GGUF. It compares each provisional
reconstruction with sampled source outputs, holds out a portion of activation
rows when available, and adds bounded full-name overrides for difficult
depth/family strata. `--auxiliary-evolution-layers` may be repeated to add
vision, audio, or drafter bundles to the same shadow receipt. Routed-expert
bundles retain their expert identity and receive coverage-aware per-expert
residual overrides rather than being collapsed into one decoder-family score.

`tools/tessera/unsloth-policy.py` can add Unsloth's dynamic-quantization
sensitivity guidance and evidence-selected activation tails before the
evolutionary search. The resulting GGUF records this with
`tessera.calibration.unsloth_prior`; this is provenance for an offline policy
input and does not create an inference-time Unsloth dependency.

The calibration wrapper on the external workspace can run the complete path
with `--evolve-awq`. It exports sampled source rows and observer moments,
evolves the policy, then passes the resulting alpha, clipping, and residual
budget to the Tessera quantizer. Source weights are streamed from safetensors;
the complete BF16 model is never required in memory.

Repair replays use deterministic semantic-family stratification rather than a
flat hash sample. Approved paragraphs are grouped by task family and length,
ordered deterministically inside each family, and emitted round-robin so the
earliest calibration chunks cover every represented family. The replay receipt
records source and selected family counts and a recommended convergence floor.

`llama-imatrix` supports adaptive stopping with
`--convergence-min-chunks`, `--convergence-interval`,
`--convergence-patience`, and `--convergence-tolerance`. At each window it
compares a deterministic sampled signature of normalized second, absolute,
fourth, and maximum activation moments across tensors. Calibration stops only
after the minimum coverage floor and several consecutive stable windows.
Periodic GGUF checkpoints remain resumable, and every convergence decision is
written to the run log.
