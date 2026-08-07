# Static MoE expert pruning

`aikar-prune` implements static expert pruning for Gemma 4 26B A4B. The initial hard-pruning implementation supports the Q4_0 QAT GGUF layout loaded by `src/models/gemma4.cpp`. Other architectures and Gemma 4 variants fail with an unsupported-architecture error.

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
build/bin/aikar-prune analyze \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --dataset calibration.jsonl \
  --ratios 0.05,0.10,0.15,0.20,0.25 \
  --metric router-output \
  --ppl-mask assistant \
  --max-layer-ratio 0.25 \
  --seed 42 \
  --output-dir pruning-results
```

Calibration collects selection count and frequency, router probability sum and mean, routed expert output L2 norm, and `mean(router_probability * output_norm)`. Experts are ranked once per layer by the final metric with original expert ID as the deterministic tie-breaker. Every larger ratio takes a longer prefix of the same ranking, so pruning sets are nested.

Perplexity uses stable accumulated negative log-likelihood:

```text
ppl = exp(total_nll / evaluated_token_count)
```

Baseline and pruned evaluations use identical rendered tokens, context windows, batches, and field masks. `analysis.json` contains all four field perplexities, timing, throughput, routing entropy, load imbalance, invalid-route count, and per-expert calibration statistics. `analysis.csv` contains one row per ratio. `README.txt` is a short human-readable summary.

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
build/bin/aikar-prune inspect \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json

build/bin/aikar-prune hard \
  --model gemma-4-26b-a4b-q4_0.gguf \
  --profile pruning-results/profile-020.json \
  --output gemma-4-26b-a4b-pruned.gguf \
  --dataset calibration.jsonl \
  --validate
```

The hard converter accepts routed expert weights only when they are Q4_0 tensors with the original expert axis at GGML dimension 2, dimension 0 divisible by the Q4_0 block size of 32, and a tensor size equal to `row_size(ne0) * ne1 * ne2`. Each surviving expert slice is copied byte-for-byte. Router rows and optional per-expert scales are compacted without dequantization.

After conversion, the command reopens the output with the normal model loader and runs a short inference smoke test. When `--dataset` is present, it evaluates the hard model and the source model with the static soft profile, then records both perplexities and their absolute difference. The mapping, byte counts, and optional validation results are written to `<output>.report.json`.

## Limitations

- Hard pruning supports only Gemma 4 26B A4B Q4_0 QAT GGUF.
- All routed layers must retain the same expert count.
- Grouped routing, heterogeneous expert counts, interleaved expert storage, packed expert axes, and non-Q4_0 routed expert weights are rejected.
- Shared-expert pruning is not supported.
- Automatic benchmark-suite execution is not included.
