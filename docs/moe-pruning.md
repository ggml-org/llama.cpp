# Static MoE expert pruning

`llama-prune` implements static expert pruning for Gemma 4 26B A4B. The initial hard-pruning implementation supports the Q4_0 QAT GGUF layout loaded by `src/models/gemma4.cpp`. Other architectures and Gemma 4 variants fail with an unsupported-architecture error.

> Soft pruning does not modify model weights. It statically disables selected experts when the model is loaded. The selected profile cannot be changed while the server is running.

## Soft and hard pruning

Soft pruning writes a JSON profile containing original expert IDs. At model initialization, the profile identity hashes and shape constraints are validated. Disabled router logits are set to negative infinity before routing selection. Surviving router probabilities retain the model's original softmax and selected-weight normalization behavior.

Hard pruning consumes exactly one soft profile. It creates a new GGUF with compacted routed expert tensors and router rows, updates `gemma4.expert_count`, and records each original-to-new expert mapping. The dense parallel MLP in every Gemma 4 MoE layer is the shared expert path and is always copied without changes.

Hard pruning never edits the source model. It writes a temporary sibling file, finishes and validates it, then renames it to the requested output.

## ChatML JSONL

Each line must contain one object with a non-empty `messages` array:

```json
{"messages":[{"role":"user","content":"What is MoE?"},{"role":"assistant","content":"A sparse expert architecture."}]}
```

Assistant messages may contain separate reasoning and final content:

```json
{"messages":[{"role":"user","content":"Solve it."},{"role":"assistant","reasoning":"Work...","content":"Answer."}]}
```

The model's Jinja chat template and tokenizer render every record. For the extended form, reasoning and final content are rendered in that order in one assistant message with a newline separator. Errors include the JSONL line number. Supported loss masks are `all`, `assistant`, `reasoning`, and `content`.

Field masks use a deterministic token-start rule. A rendered token belongs to the field containing the byte offset at which that token starts. Template and special tokens belong to no message field and are evaluated only by the `all` mask.

## Analyze

```sh
build/bin/llama-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --ratios 0.05,0.10,0.15,0.20,0.25 \
  --metric router-output \
  --ppl-mask assistant \
  --max-layer-ratio 0.25 \
  --seed 42 \
  --output-dir pruning-results
```

Calibration collects selection count and frequency, router probability sum and mean, routed expert output L2 norm, and `mean(router_probability * output_norm)`. That last quantity is the `router-output` metric and is the REAP saliency score — see [Importance metric](#importance-metric) below. Experts are ranked once per layer by the final metric with original expert ID as the deterministic tie-breaker. Every larger ratio takes a longer prefix of the same ranking, so pruning sets are nested.

The first calibration writes `pruning-results/importance-cache.json`. The cache contains model and dataset identities, context size, baseline token/NLL aggregates, and the raw per-layer expert statistics. It does not contain a ratio-specific ranking. A later `analyze` command with compatible inputs loads this cache instead of running baseline calibration again. Use `--importance-cache FILE` to share one cache across output directories.

Ratio evaluation is enabled by default. Use `--no-evaluate` to generate profiles without loading and evaluating each soft-pruned model:

```sh
build/bin/llama-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --importance-cache calibration-importance.json \
  --ratios 0.05,0.10,0.15,0.20,0.25 \
  --no-evaluate \
  --output-dir pruning-profiles
```

`--evaluate` explicitly selects the default evaluated mode. Skipping ratio evaluation does not skip a cache miss's initial calibration. It skips only the full dataset passes for the generated soft-pruned profiles. `analysis.json` records `evaluation_enabled`, and unevaluated ratio entries record `evaluated: false`.

Once a cache exists, the `profiles` subcommand can create arbitrary compatible ratio profiles without opening or hashing the model or reading the dataset:

```sh
build/bin/llama-prune profiles \
  --importance-cache calibration-importance.json \
  --ratios 0.06,0.12,0.18,0.24 \
  --output-dir another-profile-set
```

The selected `--ppl-mask` must have at least one evaluated token in the cached calibration. `--max-layer-ratio` and router Top-K safety checks still apply when profiles are generated.

Perplexity uses stable accumulated negative log-likelihood:

```text
ppl = exp(total_nll / evaluated_token_count)
```

Baseline and pruned evaluations use identical rendered tokens, context windows, batches, and field masks. `analysis.json` contains all four field perplexities, timing, throughput, routing entropy, load imbalance, invalid-route count, and per-expert calibration statistics. `analysis.csv` contains one row per ratio. `README.txt` is a short human-readable summary.

JSONL reading, message parsing, template rendering, and tokenization use ordered parallel work queues. `--dataset-threads N` sets their worker limit; zero selects the physical core count. Results are collected in input order, and the pruning field masks use the same rendered text and token-start semantics in serial and parallel modes.

## Run the server

```sh
build/bin/llama-server \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --moe-prune-profile pruning-results/profile-020.json
```

The profile is loaded once before context creation. There is no hot reload, HTTP update, per-request mask, or per-sequence mask.

Validation covers format, version, architecture, full model SHA-256, expert identity SHA-256, routed MoE layer set, expert count, Top-K, expert ranges, duplicate IDs, equal surviving counts, and remaining-expert safety.

## Inspect and hard prune

```sh
build/bin/llama-prune inspect \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json

build/bin/llama-prune hard \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json \
  --output gemma-4-26b-a4b-pruned.gguf \
  --dataset calibration.jsonl \
  --validate
```

The hard converter accepts routed expert weights only when they are Q4_0 tensors with the original expert axis at GGML dimension 2, dimension 0 divisible by the Q4_0 block size of 32, and a tensor size equal to `row_size(ne0) * ne1 * ne2`. Each surviving expert slice is copied byte-for-byte. Router rows and optional per-expert scales are compacted without dequantization.

After conversion, the command reopens the output with the normal model loader and runs a short inference smoke test. When `--dataset` is present, it evaluates the hard model and the source model with the static soft profile, then records both perplexities and their absolute difference. The mapping, byte counts, and optional validation results are written to `<output>.report.json`.

## Importance metric

`--metric router-output` implements REAP (Router-weighted Expert Activation Pruning). The saliency score for expert `j` is the mean, over tokens routed to `j`, of the router probability times the L2 norm of that expert's output:

```
REAP(j) = mean_{t in X_j} [ g_j(t) * ||f_j(t)||_2 ]
```

`f_j` is the routed expert output *before* gate weighting (`ffn_moe_down`) and `g_j` is the post-normalization router probability (`ffn_moe_weights_norm`). Calibration also accumulates the ungated mean `||f_j||_2` (the EAN criterion) and the mean router probability, both recorded in the importance cache for comparison but not currently selectable as ranking metrics.

Reference: "REAP: Router-weighted Expert Activation Pruning", [arXiv:2510.13999](https://arxiv.org/abs/2510.13999) — the score above is Equation 9.

## Provenance

The expert profiling engine — the REAP/EAN accumulators and the ggml eval-callback interception of `ffn_moe_topk`, `ffn_moe_weights_norm`, and `ffn_moe_down` — derives from the `feat/moe-expert-profiling` branch by Salvatore Rossitto (`tools/expert-profile/expert-profile.cpp` and the `tools/moe-pruning` scripts, March 2026), reimplemented in C++ for this tool. GGUF expert compaction follows the approach of that branch's `gguf_prune.py`.

Built on top of that here: the soft-pruning graph mask (`llama_model_set_moe_prune` and `llm_graph_input_moe_mask`, which disable experts at inference time without touching weights), the hashed and validated JSON profile format, the ChatML dataset and per-field perplexity masks, and the reusable importance cache.

## Limitations

- Hard pruning supports only Gemma 4 26B A4B Q4_0 QAT GGUF.
- All routed layers must retain the same expert count.
- Grouped routing, heterogeneous expert counts, interleaved expert storage, packed expert axes, and non-Q4_0 routed expert weights are rejected.
- Shared-expert pruning is not supported.
- Automatic benchmark-suite execution is not included.
