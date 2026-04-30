# llama-semantic-bench

Semantic fidelity benchmark for llama.cpp using Sentence Textual Similarity (STS) evaluation.

While `llama-perplexity` measures language-model quality (how well a model predicts text), it does not measure **semantic preservation** — whether the model's embedding space correctly captures the *meaning* of sentences. `llama-semantic-bench` fills this gap by evaluating how well the model's embeddings rank sentence pairs by semantic similarity.

## Metrics

| Metric | Description |
|--------|-------------|
| **Pearson r** | Linear correlation between model similarity scores and ground-truth labels |
| **Spearman rho** | Rank correlation — more robust to outliers |
| **Bhattacharyya Coefficient (BC)** | Distributional overlap between two embedding vectors (1.0 = identical distributions, 0.0 = orthogonal) |

The **Bhattacharyya Coefficient** is the core fidelity metric: it measures how much of the original embedding distribution is preserved after quantization. This makes it directly applicable to comparing F16 vs. Q4_K_M vs. Q2_K models on the same prompt set.

Theoretical reference: Sathyavageeswaran (2026), *US Patent 19/287,703*; IJITCE Vol. 13 (2025).

## Build

```bash
cmake -B build .
cmake --build build --config Release -t llama-semantic-bench
```

## Dataset Format

TSV file with three tab-separated columns:

```
score	sentence1	sentence2
4.2	A man is playing a guitar.	A man is playing music.
0.1	A cat is sleeping.	A dog is running.
```

- `score`: Ground-truth similarity label (default range: 0–5 for STS-B)
- Compatible with [STS-B](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark), STS12–16, SICK-R datasets

## Usage

```bash
./build/bin/llama-semantic-bench \
    -m models/my-model.gguf \
    -f data/sts-b-dev.tsv \
    --output-file results.csv \
    --score-min 0 \
    --score-max 5
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to GGUF model file |
| `-f, --tsv-file` | required | Input TSV file |
| `--output-file` | (none) | Optional CSV file for per-pair results |
| `--score-min` | `0` | Minimum score value in the TSV |
| `--score-max` | `5` | Maximum score value in the TSV |

Standard `llama.cpp` flags (`-c`, `-t`, `-ngl`, etc.) are also supported.

## Example Output

```
Loaded 1379 sentence pairs from sts-b-dev.tsv

=== Semantic Fidelity Results ===
Pairs evaluated   : 1379
Pearson  r        : +0.8124  (cosine sim vs label)
Spearman rho      : +0.8034  (cosine sim vs label)
Mean BC score     : 0.7821   (1.0=identical, 0.0=orthogonal)
Per-pair CSV      : results.csv
```

## Measuring Quantization Fidelity

Run the benchmark on the same dataset with different quantization levels to find the "semantic cliff":

```bash
for QUANT in f16 q8_0 q5_k_m q4_k_m q3_k_m q2_k; do
    echo "=== $QUANT ===" && \
    ./build/bin/llama-semantic-bench -m models/llama-${QUANT}.gguf \
        -f data/sts-b-dev.tsv 2>&1 | grep -E "Pearson|Spearman|Mean BC"
done
```

This produces a table showing exactly where semantic meaning degrades:

| Quantization | Pearson r | Spearman rho | Mean BC |
|---|---|---|---|
| F16 | 0.812 | 0.803 | 0.782 |
| Q8_0 | 0.811 | 0.802 | 0.780 |
| Q4_K_M | 0.797 | 0.788 | 0.764 |
| Q2_K | 0.721 | 0.712 | 0.689 |
