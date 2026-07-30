# llama.cpp/tools/imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.
More information is available in <https://github.com/ggml-org/llama.cpp/pull/4861>.

## Usage

```
./llama-imatrix \
    -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \
    [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \
    [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \
    [--show-statistics] [...]
```

Here `-m | --model` with a model name and `-f | --file` with a file containing calibration data (such as e.g. `wiki.train.raw`) are mandatory.
The parameters in square brackets are optional and have the following meaning:

* `-h | --help` shows usage information and exits.
* `-lv | --verbosity` specifies the verbosity level. If set to `0`, no output other than the perplexity of the processed chunks will be generated. If set to `1`, each time the results are saved a message is written to `stderr`. If `>=2`, a message is output each time data is collected for any tensor. Default verbosity level is `1`.
* `-o | --output-file` specifies the name of the file where the computed data will be stored. If missing `imatrix.gguf` is used.
* `-ofreq | --output-frequency` specifies how often the so far computed result is saved to disk. Default is 10 (i.e., every 10 chunks)
* `--output-format` specifies the output format of the generated imatrix file. Either "gguf", or "dat" (the legacy format). Defaults to "gguf".
* `--save-frequency` specifies how often to save a copy of the imatrix in a separate file. Default is 0 (i.e., never)
* `--process-output` specifies if data will be collected for the `output.weight` tensor. Typically, it is better not to utilize the importance matrix when quantizing `output.weight`, so this is set to `false` by default.
* `--in-file` one or more existing imatrix files to load and combine. Useful for merging files from multiple runs/datasets.
* `--parse-special` enables parsing of special tokens (e.g., `<|im_start|>` in some models). Useful for models with custom tokenizers.
* `--chunk | --from-chunk` to skip the first `n` chunks of tokens from the input data. Useful for resuming or skipping initial low-quality data.
* `--chunks` maximum number of chunks to process. Default is -1 for all available chunks.
* `--no-ppl` disables the calculation of perplexity for the processed chunks. Useful if you want to speed up the processing and do not care about perplexity.
* `--show-statistics` displays imatrix file's statistics.

## Speculative-decoding calibration

When `--model-draft FNAME` is set, `llama-imatrix` rolls the target model
forward under real speculative-decoding forward passes instead of plain
text forward passes. This captures verifier activations that match the
distribution the drafter will see at inference time, and is the basis
for the drafter fine-tuning (rejection sampling / distillation) flow.

* `--model-draft PATH` (also `-md` / `--spec-draft-model`) drafter model used to drive spec-decoding forward passes. When set, the imatrix is collected on the verifier activations produced while the drafter is run side-by-side; when unset, plain autoregressive forward passes are used.
* `--spec-steps N` number of spec-decoding steps to roll forward (default: 0, i.e. until the context limit is reached). Each step is one draft-token proposal plus the verifier check; set this to bound the calibration run when you do not need to consume the full context.
* `--telemetry-out PATH` path to write a per-step accept/reject JSONL stream when `--model-draft` is set (default: disabled, no output). The v1 schema is `llama.dflash.acceptance.v1` with one record per spec step carrying `seq_id`, `drafted`, `accepted`, and a per-position `confidence[]` array. The stream is append-only and flushed after each step, so it survives an interrupted run.
* `--telemetry-topk N` when `> 0` together with `--telemetry-out`, additionally emit the `llama.spec_calib.v2` schema with the full top-N verifier and drafter distributions at every draft position (default: 0, v1 only). Suggested value: 64. The v2 stream is what the drafter fine-tuning / distillation tooling consumes; without it, only the accept/reject signal is available.

For faster computation, make sure to use GPU offloading via the `-ngl | --n-gpu-layers` argument.

Recent versions of `llama-imatrix` store data in GGUF format by default. For the legacy format, use an extension other than `.gguf` when saving the output file. More information is available in <https://github.com/ggml-org/llama.cpp/pull/9400>.

## Examples

```bash
# generate importance matrix using default filename (imatrix.gguf), offloading 99 layers to GPU
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt -ngl 99

# use the imatrix to perform a Q4_K_M quantization
./llama-quantize --imatrix imatrix.gguf ggml-model-f16.gguf ./ggml-model-q4_k_m.gguf q4_k_m
```

```bash
# generate and save the imatrix using legacy format
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --output-format dat -o imatrix-legcy-format.dat -ngl 99
```

```bash
# convert legacy (binary) imatrix format to new (GGUF) format
./llama-imatrix --in-file imatrix-legacy-format.dat -o imatrix-new-format.gguf
```

```bash
# convert new (GGUF) imatrix format to legacy (binary) format
./llama-imatrix --in-file imatrix-new-format.gguf --output-format dat -o imatrix-legacy-format.dat
```

```bash
# combine existing imatrices
./llama-imatrix --in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf -o imatrix-combined.gguf
```

```bash
# skip first 5 chunks, save intermediates every 20 chunks and snapshots every 50, parsing special tokens
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --chunk 5 --output-frequency 20 --save-frequency 50 --parse-special
```

```bash
# analyse imatrix file and display summary statistics instead of running inference
./llama-imatrix --in-file imatrix.gguf --show-statistics
```

```bash
# spec-decoding calibration: collect verifier imatrix under real draft-and-verify
# forward passes, write the v2 top-64 telemetry stream for drafter fine-tuning
./llama-imatrix \
    -m ggml-model-f16.gguf -f calibration-data.txt \
    -md ggml-drafter.gguf \
    --spec-steps 256 \
    --telemetry-out spec-acceptance.jsonl \
    --telemetry-topk 64 \
    -ngl 99 -o imatrix.gguf
```

`--show-statistics` will display the following statistics:

#### Per tensor

* Σ(Act²): sum of all squared activations (the importance scores)
* Min & Max: minimum and maximum squared activations values
* μ & σ: Squared activations' mean and standard deviation
* % Active: proportion of elements whose average squared activation exceeds a small threshold (1e-5). Helpful to determine how alive/dormant the tensor is during inference
* N: number of squared activations
* Entropy: entropy of the squared activation distribution, in bits (standard Shannon entropy measurement) $S = -\sum_{i=1}^N p_i \log_2 p_i$
* E (norm): Normalized entropy. $E(norm)=\frac{-\sum_{i=1}^N p_i \log_2 p_i}{log_2 N}$. These two metrics can be used to determine how well a prompt "exercises" the model's capabilities
* ZD Score: z-score distribution as described in _3.1 Layer Importance Scores_ of [Layer-Wise Quantization](https://arxiv.org/abs/2406.17415)
* CosSim: cosine similarity with respect to the previous layer's tensor. Useful to determine how similar the squared activations of the current layer are to the previous layer's squared activations.

#### Per layer

Weighted averages of Σ(Act²), ZD Score and CosSim are also calculated.

#### Important note on the computed Statistics

When using these statistics, please note that they are computed on the squared activations, **not on the actual (raw) activations**.
Whilst the results are still useful, they're less reliable than using the raw values, and in the case of the cosine similarity, could be misleading if the tensor contains opposite vectors.
