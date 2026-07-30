# Tessera calibration search

Tessera family coverage is enforced by `family_coverage.py`. Gemma 4 dense,
unified multimodal, MoE, and assistant/MTP checkpoints share the `gemma4`
contract. Qwen 3.6 dense, MoE, multimodal, and NextN/MTP checkpoints share the
`qwen3.6` contract while retaining the internal `qwen35` and `qwen35moe` GGUF
architecture names. The quantizer classifies every source tensor before
reading weights, rejects unknown tensors and incomplete multimodal payloads,
and embeds a compact `llama.tessera.family-coverage.v1` receipt containing the
family, detected features, component counts, runtime treatments, and canonical
tensor-manifest digest.

`awq-evolve.py` implements a hierarchical evolutionary search for Tessera AWQ
scales, clipping, and sparse-residual allocation. Numeric candidates are
evaluated in island populations. A MAP-Elites archive retains good policies
from different alpha, residual-budget, and tail-handling regions instead of
collapsing immediately to one local optimum. The global objective includes a
90th-percentile layer penalty. Reduced searches then generate exact-name
overrides for unusually sensitive tensors while preserving family defaults.
Each family also measures residual sensitivity curves and uses a greedy
marginal-error allocator to keep the mean sparse-residual fraction within the
requested model budget. The policy records the selected fraction and quality
cost for every sampled tensor so the allocation is auditable and resumable.

Each input layer is an `.npz` bundle with a two-dimensional `weight` array.
For exact reconstruction evaluation it may contain `train_activations` and
`heldout_activations`, both shaped `[tokens, input_channels]`. For compact
observer evaluation it may instead contain `in_sum2`, `in_sum4`,
`in_maxabs`, and `counts`. Optional scalar string arrays `name` and `family`
identify the tensor. Supported families are `attention`, `ffn`, `fusion`, and
`output_embedding`.

Bundles can be generated without loading the full source model. The exporter
selects up to 24 tensors per family across model depth instead of taking the
first tensors encountered:

```sh
python tools/tessera/make-awq-layer-bundles.py \
  --model-dir /Volumes/Julian\ T7/models/gemma-4-12B-it-qat-q4_0-unquantized \
  --imatrix /Volumes/Julian\ T7/calibration/gemma4-rich.imatrix.gguf \
  --output /Volumes/Julian\ T7/calibration/gemma4-layers
```

```sh
python tools/tessera/awq-evolve.py \
  --layers /Volumes/Julian\ T7/calibration/gemma4-layers \
  --output /Volumes/Julian\ T7/calibration/gemma4-evolved-policy.json \
  --checkpoint /Volumes/Julian\ T7/calibration/gemma4-evolution.json
```

The output uses `llama.speculative.calibration-policy.v1`, so it can be passed
directly to `tile640_quantize_v3.py --calibration-policy`. Checkpoints include
the populations, random-number-generator state, archive, and score history,
allowing a long search to resume deterministically.

The default progressive evaluator applies deterministic successive halving to
each island population: all candidates receive a depth-stratified observer
screen, promising candidates receive a larger tail-aware refinement, and only
the survivors receive exact full-family held-out scoring. Candidates close to
the promotion boundary retain a configurable margin, while a bounded number
of otherwise-unrepresented MAP-Elites cells survive to preserve exploration.
Per-candidate stage scores are checkpointed with the layer-bundle digest, so a
resumed search reuses compatible evidence rather than recomputing it. Use
`--no-progressive-eval` only for an intentionally exhaustive baseline.

Within one active search, the evaluator also caches exact candidate-by-layer
scores. A candidate promoted from screen to refinement to final validation
therefore never reconstructs a layer twice. The transient layer cache is kept
in memory only; checkpoints retain compact stage scores and the telemetry
digest, preventing a long run from exchanging compute savings for an
unbounded checkpoint file.

## Unsloth policy bridge

`unsloth-policy.py` integrates Unsloth's dynamic-quantization guidance without
adding PyTorch, Transformers, bitsandbytes, or Unsloth to the llama.cpp
runtime. It reads the public Unsloth sensitive-module list with Python's AST,
translates Hugging Face module names to GGUF tensor aliases, and can combine
that prior with the top observer tails in a Tessera evidence store:

```sh
/Volumes/Julian\ T7/calibration-venv/bin/python \
  tools/tessera/unsloth-policy.py \
  --unsloth-root /Volumes/Julian\ T7/unsloth-zoo \
  --config /path/to/model/config.json \
  --evidence-store /path/to/evidence \
  --run-id gemma4-pilot \
  --base-policy /path/to/dflash-policy.json \
  --output /path/to/unsloth-tessera-policy.json
```

The default `protected` mode keeps the whole model in Tessera while assigning
Unsloth-sensitive modules and telemetry-selected tensors a larger sparse
residual budget. `--unsloth-skip-mode exact` translates Unsloth skips
literally, using exact Tessera residual encoding rather than passthrough
tensors. The generated policy remains a valid base policy for `awq-evolve.py`.

The workspace launcher uses the dedicated MLX/Polars environment and performs
all stages with one option:

```sh
/Volumes/Julian\ T7/tessera-calibrate \
  --model-dir /Volumes/Julian\ T7/models/gemma-4-12B-it-qat-q4_0-unquantized \
  --f16-model /path/to/loadable-gemma4-calibration.gguf \
  --metadata-model /path/to/canonical-gemma4.gguf \
  --calibration-data /Volumes/Julian\ T7/calib_data.txt \
  --output /Volumes/Julian\ T7/models/gemma4-tessera.gguf \
  --chunks 4 \
  --evolve-awq \
  --evidence-store /Volumes/Julian\ T7/calibration/evidence \
  --evidence-run-id gemma4-pilot \
  --evolution-checkpoint /Volumes/Julian\ T7/calibration/gemma4-evolution.json
```

## Parquet evidence store

`evidence-store.py` stores observer channels, evolutionary populations, and
MTP/DFlash acceptance events as append-only Zstandard-compressed Parquet
partitions. Polars scans these partitions lazily, applies projection and
predicate pushdown, and uses its streaming engine for summaries.

## Hugging Face evidence commons

`hf-evidence.py` publishes only architecture-fingerprinted sufficient
statistics. It never uploads prompts, completions, request logs, raw
activations, or model weights. Create the public dataset once while logged in:

```sh
python tools/tessera/hf-evidence.py init
```

Publish one completed evidence run with `publish`, and bootstrap a compatible
local store with `pull`. Pulling rejects unrelated architectures implicitly by
downloading only the model configuration fingerprint:

Publication requires explicit acceptance of
`Tessera-Calibration-Contribution-1.0`. The terms grant Julian Alejandro Torres
Nieto a perpetual, irrevocable, transferable, sublicensable, royalty-free
license to redistribute and commercially relicense the submitted calibration
aggregate. The exact license identifier, text hash, grantee, assent mechanism,
and authenticated Hugging Face contributor identity are recorded in every new
aggregate manifest. Existing manifests are not retroactively relicensed.

```sh
python tools/tessera/hf-evidence.py pull \
  --model-dir /path/to/model \
  --store /path/to/evidence
```

The canonical public commons is
`juliantorr/tessera-calibration-commons`, recorded in `hf-commons.json`.
The calibration wrapper pulls compatible aggregates by default; use
`--no-hf-evidence` for an explicitly offline run. Publishing is opt-in with
`--hf-evidence-publish` and requires a Hugging Face account with write access.

Each accepted aggregate advances a model-specific evidence counter. One
dataset epoch represents one million conservatively measured calibration
tokens by default. When the dataset epoch exceeds the epoch recorded by the
published GGUF, `publish` and `status` emit a macOS notification and leave
`data/<fingerprint>/epoch.json` with `requantization_due: true`. After
quantizing and publishing the replacement GGUF, bind it to the exact evidence
snapshot with:

```sh
python tools/tessera/hf-evidence.py status \
  --model-dir /path/to/model \
  --output tessera-epoch.json

python /path/to/tile640_quantize_v3.py \
  --tessera-epoch-receipt tessera-epoch.json \
  ...

python tools/tessera/hf-evidence.py mark-model \
  --model-dir /path/to/model \
  --gguf /path/to/model.gguf \
  --model-repo owner/model
```

The dataset and model repositories then carry matching receipts with the
epoch number and aggregate-set digest. `--observer-tokens-per-epoch` adjusts
the refresh cadence.

Every newly published aggregate manifest also records the CC BY-NC-SA 4.0
outbound license, attribution to Julian Alejandro Torres Nieto, Tribunus.dev,
and the hash of the public artifact notice. Repositories containing legacy
Apache-2.0 aggregates use mixed-license metadata because those prior grants
remain effective.

## Engram third-pass preparation

`engram-build.py` prepares the deterministic hash ABI and the bounded n-gram
inventory for the compression-aware third pass. The implementation follows
the official DeepSeek demo's normalized-token compression, per-layer odd
multipliers, prime-sized heads, and XOR mixing. Generated multipliers and
moduli are sealed into a manifest so native inference does not depend on a
Python random-number generator.

The index command refuses corpora without a matching
`llama.tessera.training-corpus.v1` receipt, explicit redistribution
clearance, a resolved license, and a declaration that user inference data is
absent. It also rejects lines and decoded n-grams matching the initial PII and
secret filters. These filters are a publication gate, not a substitute for
source review.

The first-party Tessera corpus is released under CC BY-NC-SA 4.0. That license
permits study, sharing, and adaptation for noncommercial purposes, requires
attribution, and requires shared adaptations to retain the same license. It
does not relicense llama.cpp, upstream models, or separately published model
weights.

Tessera-authored software is separately licensed under the Tessera Research
and Education License 1.0. Upstream llama.cpp and ggml code remains MIT.
Tessera-published GGUF and safetensors artifacts carry CC BY-NC-SA 4.0 terms
only for Tessera-controlled material and must retain all applicable upstream
model terms. See `TESSERA-LICENSING.md` at the repository root for the exact
boundary and the treatment of legacy releases.

```sh
python tools/tessera/engram-build.py spec \
  --tokenizer /path/to/gemma4-safetensors \
  --layers 1 15 \
  --vocab-size 65521 \
  --output /path/to/engram-hash.json

python tools/tessera/engram-build.py index \
  --tokenizer /path/to/gemma4-safetensors \
  --corpus /path/to/approved-training-corpus.txt \
  --receipt /path/to/training-corpus-receipt.json \
  --output /path/to/engram-index.json
```

`engram_hash.h` is the runtime-side integer contract. Its focused native test
uses fixed overflow-sensitive vectors so changes to signed wrapping or modulo
behavior cannot silently alter lookup IDs.

## Compression-aware MLX training

`kv-compress-reference` is a native GGML oracle for KV quantize/dequantize
behavior. It supports the cache formats used by the third-pass curriculum and
can apply the same 64-wide orthonormal Hadamard rotation used by llama.cpp.
`kv_compression_mlx.py` mirrors those operations on MLX's Metal backend and
uses a straight-through estimator so the exact compressed forward values keep
useful gradients.

The parity test covers Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, and IQ4_NL with and
without rotation. The current fixtures compare every dequantized float
exactly, not only aggregate error.

`third_pass_losses.py` combines next-token cross entropy, teacher-logit KL,
hidden-state reconstruction, attention-output reconstruction, and
distribution-overlap draft acceptance. `third_pass_trainer.py` wraps an MLX
module and optimizer into one evaluated training step and can save only the
trainable Engram and adapter tensors as safetensors.

`kv_compaction_mlx.py` supplies the compaction state machine used by that
curriculum. Its `seq_rm`, `seq_add`, and `seq_div` transitions use the same
half-open position ranges as llama.cpp. Lossless defragmentation preserves
slot order. Lossy compaction can either evict unprotected middle entries or
merge them with importance weights while preserving configured prefix and
recent windows. Before key rows are merged, every row is RoPE-shifted to the
destination position; values are mixed without that rotation. The resulting
fixed-size state can then pass through the exact KV compression simulator.

`KVCompactionMachine` owns the current state, a monotonic compaction epoch,
and a bounded metadata-only transition history. The history records counts
and policy parameters but never key, value, token, or prompt contents.
Training code can therefore schedule progressively smaller cache targets and
different merge policies without making user inference telemetry part of the
training corpus.

## MoE calibration and repair

The Qwen 3.5/3.6 MoE path retains one observer row per routed expert. Use
`make-moe-awq-bundles.py` for fused three-dimensional expert banks; it reads
one expert slice at a time and writes separate gate/up/down search bundles
without materializing the complete bank. The ordinary bundle builder also
recognizes checkpoints that store experts as separate two-dimensional
tensors, along with router and shared-expert projections.

`awq-evolve.py` searches routed-expert bundles independently and allocates
their sparse residuals from one family-wide budget. Its policy records
per-layer, per-tensor, per-expert residual fractions, AWQ alpha, and clipping.
`tile640_quantize_v3.py` consumes those values when quantizing each expert in
a fused bank. Router matrices remain exact Tessera encodings so quantization
cannot perturb top-k selection merely to save a negligible amount of space.

`moe_calibration.py` accumulates only sufficient router statistics:
population counts, selection counts, probability and confidence sums, top-k
boundary margins, expert-output error, and downstream divergence. It never
retains token IDs, prompts, logits, activations, or expert assignments for an
individual token. Rare-expert estimates are shrunk toward the median error of
their layer before the global residual allocator runs. Public router evidence
also has independent minimum total-observation and per-expert-selection
gates.

The third-pass loss automatically enables router KL, teacher top-k margin,
and expert-output reconstruction terms when the student and teacher return
`router_logits` and `expert_output`. Dense models remain compatible because
those terms are optional. Qwen MoE and its MTP graph already expose
`ffn_moe_logits`, `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_down`, and
`ffn_moe_out` callback boundaries, so a validation run can compare BF16 and
Tessera at each amplification point instead of relying only on final text.

`build-calibration-corpus.py` generates the first-party balanced calibration
corpus used for this path. Its text is clean-room procedural material authored
for Tessera rather than copied or paraphrased benchmark questions. The ten
categories cover code, English, Korean, Chinese, Japanese, tool calling,
reasoning, chat, mixed-format documents, and long structured contexts. The
generator emits the plain calibration text, a sample index, a deterministic
manifest, and a `llama.tessera.training-corpus.v1` receipt under CC BY-NC-SA
4.0 with attribution to tribunus.dev.

`moe-calibrate.py` connects that corpus to llama-imatrix. It begins with a
deterministic stratified set of 128 samples, accumulates graph-resident
observer output across rounds, and adds 128 samples per round up to 1024.
Calibration stops only when the configured percentile of routed-expert
populations meets its minimum and the normalized per-expert observer profiles
remain within the stability threshold for two consecutive rounds. Reaching
the maximum sample count produces a valid bounded run but records
`maximum-samples` rather than falsely claiming convergence.

```sh
python tools/tessera/make-moe-awq-bundles.py \
  --model-dir /path/to/qwen-moe \
  --imatrix /path/to/qwen-moe.imatrix.gguf \
  --output /path/to/moe-bundles

python tools/tessera/awq-evolve.py \
  --layers /path/to/moe-bundles \
  --base-policy /path/to/current-policy.json \
  --output /path/to/qwen-moe-policy.json

python tools/tessera/evidence-store.py ingest-router \
  --store /path/to/evidence \
  --run-id qwen-moe \
  --telemetry /path/to/router-aggregates.jsonl

python tools/tessera/build-calibration-corpus.py \
  --output-dir /path/to/tessera-balanced-v1 \
  --epoch 1

python tools/tessera/moe-calibrate.py \
  --model /path/to/qwen-moe-canonical.gguf \
  --corpus-index /path/to/tessera-balanced-v1/samples.jsonl \
  --work-dir /path/to/qwen-moe-adaptive-calibration
```

The student module passed to `ThirdPassTrainer` receives a batch dictionary
and returns `logits`, `draft_logits`, `hidden`, and `attention`. The teacher
dictionary contains the corresponding uncompressed `logits`, `hidden`, and
`attention`; the batch also supplies `targets`. Model-specific Gemma, MTP,
DFlash, and Engram wiring can therefore evolve independently of the shared
loss and cache simulation contract.

## BF16 source epochs

`source-epoch.py` turns approved upstream safetensor checkpoints into one
reproducible, namespaced BF16 source bundle. Its public receipt retains the
upstream repositories, revisions, licenses, file hashes, and component
boundaries without exposing local paths. Assembly streams one tensor at a
time and writes bounded safetensor shards, so the entire combined model is
never resident in memory.

Start from `source-manifest.example.json`, then seal, assemble, validate, and
publish an immutable epoch:

```sh
python tools/tessera/source-epoch.py fetch \
  --manifest source-manifest.json \
  --cache /path/to/upstream-cache \
  --output resolved-source-manifest.json
python tools/tessera/source-epoch.py seal \
  --manifest resolved-source-manifest.json \
  --output sealed-source-manifest.json
python tools/tessera/source-epoch.py assemble \
  --manifest resolved-source-manifest.json \
  --output /path/to/tessera-source-epoch-0
python tools/tessera/source-epoch.py validate \
  --bundle /path/to/tessera-source-epoch-0
python tools/tessera/source-epoch.py publish \
  --bundle /path/to/tessera-source-epoch-0 \
  --repo owner/tessera-unified-bf16
```

Pass the resulting `tessera-source-epoch.json` to the quantizer with
`--tessera-source-receipt`. The GGUF then embeds the BF16 source epoch and
both its logical-source and assembled-artifact digests.

```sh
/Volumes/Julian\ T7/calibration-venv/bin/python \
  tools/tessera/evidence-store.py summarize \
  --store /Volumes/Julian\ T7/calibration/evidence \
  --run-id gemma4-pilot
```
