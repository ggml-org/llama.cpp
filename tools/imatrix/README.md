# llama.cpp/tools/imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.
More information is available in <https://github.com/ggml-org/llama.cpp/pull/4861>.

## Usage

```
./llama-imatrix \
    -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \
    [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \
    [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \
    [--output-format gguf|dat] [--show-statistics] [...]
```

Here `-m | --model` with a model name and `-f | --file` with a file containing calibration data (such as e.g. `wiki.train.raw`) are mandatory.
The parameters in square brackets are optional and have the following meaning:

* `-h | --help` shows usage information and exits.
* `-lv | --verbosity` specifies the verbosity level. If set to `0`, no output other than the perplexity of the processed chunks will be generated. If set to `1`, each time the results are saved a message is written to `stderr`. If `>=2`, a message is output each time data is collected for any tensor. Default verbosity level is `1`.
* `-o | --output-file` specifies the name of the file where the computed data will be stored. If missing `imatrix.gguf` is used.
* `-ofreq | --output-frequency` specifies how often the so far computed result is saved to disk. Default is 10 (i.e., every 10 chunks)
* `--output-format` specifies the output format of the generated imatrix file. Either `gguf`, or `dat` (the legacy format). Defaults to `gguf`.
* `--save-frequency` specifies how often to save a copy of the imatrix in a separate file. Default is 0 (i.e., never)
* `--process-output` specifies if data will be collected for the `output.weight` tensor. Typically, it is better not to utilize the importance matrix when quantizing `output.weight`, so this is set to `false` by default.
* `--in-file` one or more existing imatrix files to load and combine. Useful for merging files from multiple runs/datasets.
* `--parse-special` enables parsing of special tokens (e.g., `<|im_start|>` in some models). Useful for models with custom tokenizers.
* `--chunk | --from-chunk` to skip the first `n` chunks of tokens from the input data. Useful for resuming or skipping initial low-quality data.
* `--chunks` maximum number of chunks to process. Default is `-1` for all available chunks.
* `--no-ppl` disables the calculation of perplexity for the processed chunks. Useful if you want to speed up the processing and do not care about perplexity.
* `--show-statistics` displays imatrix file's statistics.

For faster computation, make sure to use GPU offloading via the `-ngl | --n-gpu-layers` argument.

Versions **b5942** and newer of `llama-imatrix` store data in GGUF format by default. For the legacy format, use `--output-format dat` when saving the output file. More information is available in <https://github.com/ggml-org/llama.cpp/pull/9400>.

## Examples

```bash
# generate importance matrix using default filename (imatrix.gguf), offloading 99 layers to GPU
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt -ngl 99

# use the imatrix to perform a Q4_K_M quantization
./llama-quantize --imatrix imatrix.gguf ggml-model-f16.gguf ./ggml-model-q4_k_m.gguf q4_k_m 99
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
# analyze imatrix file and display summary statistics instead of running inference
./llama-imatrix --in-file imatrix.gguf --show-statistics
```

## Statistics

Please note that the L₂ Distance can only be calculated if the imatrix is in GGUF format. If a value lacks proper statistical interpretability, **nan** will be shown instead. The following statistics are computed:

#### Per tensor

* **Min / Max / μ / σ**: Tensor elements Min, Max, Mean, and Standard Deviation.
* **H Norm**: Shannon Entropy normalized over log₂(N). Defined as $H Norm=\frac{-\sum_{i=1}^N p_i \log_2 p_i}{log_2 N}$. Used to determine how well a prompt "exercises" the model's capabilities. Higher values indicate more uniform distribution of activations. Every neuron is firing equally; hard to prune.
* **Z-score Distribution (ZD)**: % of elements whose ZD-score is > 1.0 (an indicator of outliers), as described in _3.1 Layer Importance Scores_ of [Layer-Wise Quantization](https://arxiv.org/abs/2406.17415).
* **∑ E[A²]**: The sum of squares of activations (Energy) for the tensor. Tensors with high "energy" contribute most to the final output. Quantization errors here propagate strongly. These tensors usually need higher precision (e.g., Q6_K vs Q4_K).
* **L₂ Distance**: Euclidean Distance from the tensor in the previous layer. Measure of transformation magnitude; higher values indicate more significant transformation on the data.
* **CosSim**: Cosine Similarity with the tensor in the previous layer. _~1.0_, the tensor output points in the exact same direction as the previous layer's tensor (the layer is refining magnitude, not direction). _< 1.0_, the layer is rotating the vector space (changing semantic meaning).
* **PCC**: Pearson Correlation Coefficient with the tensor in the previous layer. Checks for linear correlation excluding the mean shift. Similar to CosSim but centers geometric data first. Indicates if the pattern of activation changes or just the offset.

#### Per layer

Aggregated metrics per block/layer:

* **Z-score Distribution (ZD)**: % of this layer's concatenated tensors' elements with |Z| > 1. Indicates general "spikiness" of the layer's activations.
* **∑ E[A²]:** Total energy of the layer's concatenated tensors. Indicates the layer's overall contribution amplitude.
* **L₂ Distance:** Euclidean Distance of the layer's concatenated tensors from the previous layer’s. Global measure of transformation magnitude.
* **CosSim**: Cosine Similarity of this layer's concatenated tensors with the previous layer.
* **PCC**: Average Pearson Correlation of the tensors in the layer.

More information is available in https://github.com/ggml-org/llama.cpp/pull/14891
