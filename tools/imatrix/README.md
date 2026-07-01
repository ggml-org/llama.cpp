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
# analyze imatrix file and display summary statistics instead of running inference
./llama-imatrix -m ggml-model-f16.gguf --in-file imatrix.gguf --show-statistics
```

## Statistics
Please note that if a value lacks statistical interpretability, **nan** will be shown instead.

#### Per tensor
Statistical properties of a single tensor's average activation or activation energy (squared magnitude).

* **Mean / StdDev**: $\mu = \frac{1}{N} \sum v_i$ and $\sigma = \sqrt{\frac{1}{N} \sum (v_i - \mu)^2}$
  - Establishes the baseline distribution of the tensor's outputs. Low variance means the tensor outputs a mostly constant projection; high variance implies high information density across dimensions.
* **Skewness & Kurtosis**: $skew = \frac{\frac{1}{N} \sum (v_i - \mu)^3}{\sigma^3}$ and $kurt = \frac{\frac{1}{N} \sum (v_i - \mu)^4}{\sigma^4} - 3.0$
  - Skewness measures the asymmetry of a distribution around its mean. Kurtosis measures the "tailedness" of the feature activations. A high kurtosis indicates a highly sparse/heavy-tailed activation distribution (e.g., outlier features). High-kurtosis tensors typically require higher precision quantization to prevent outlier degradation.
* **H Norm**: $H = -\sum_{i} P_i \log_2(P_i)$, where $P_i = \frac{v_{val_i}}{\sum v_{val_i}}$
  - Shannon Entropy normalized over log₂(N). Used to determine how well a prompt "exercises" the model's capabilities. Higher values indicate more uniform distribution of activations. Every neuron is firing equally; hard to prune.
* **$\sum E[A^2]$**: $\sum E[x_i^2]$
  - The sum of squares of activations (Energy) for the tensor. Tensors with high "energy" contribute most to the final output. Quantization errors here propagate strongly. These tensors usually need higher precision.

#### Intra-layer
These statistics compare identical tensor between the current layer $L$ and the previsou layer $L-1$ (e.g., `blk.1.attn_v` vs `blk.0.attn_v`).

* **Gain**: $G = \frac{\sqrt{\sum C_i^2 / N_{curr}}}{\sqrt{\sum P_i^2 / N_{prev}}}$
  - Indicates if a layer acts as an "amplifier" ($G > 1$) or a "dampener" ($G < 1$).
* **L2 Distance**: $L2 = \sqrt{ \sum (C_i - P_i)^2 }$ where $C$ is the current layer and $P$ is the previous layer's tensor.
  - Measures absolute representational shift. Huge leaps in L2 distance indicate that a layer fundamentally transforms the hidden states.
* **Pearson Correlation Coefficient (PCC)**: $r = \frac{\sum (C_i - \bar{C})(P_i - \bar{P})}{\sqrt{\sum (C_i - \bar{C})^2} \sqrt{\sum (P_i - \bar{P})^2}}$
  - Similar to Cosine Similarity, but invariant to scalar or offset biases (centers the data first). Highly correlated adjacent layers signify structural repetition.
* **Covariance (Cov)**: $\frac{1}{N}\sum (c_i-\bar c)(p_i-\bar p)$
  - The **unnormalized covariance** between current and previous layer activations. Captures both the correlation structure and the magnitude of the joint variation. Large absolute covariance indicates the layers are jointly processing strong, correlated signals.

#### Per layer
Aggregated metrics per block/layer:

* **∑ E[A²]:** Total energy of the layer's concatenated tensors. Indicates the layer's overall contribution amplitude.
* **Gain**: Indicates if a layer acts as an "amplifier" ($G > 1$) or a "dampener" ($G < 1$).
* **L₂ Distance:** Euclidean Distance of the layer's concatenated tensors from the previous layer’s. Global measure of transformation magnitude.
* **CosSim**: $\text{CosSim}_{Layer} = \frac{\sum_{\text{tensors}} (\text{Dot Prod})}{\sqrt{\sum_{\text{tensors}} (\text{Norm1}^2)} \sqrt{\sum_{\text{tensors}} (\text{Norm2}^2)}}$
  - Cosine Similarity of the current layer's concatenated tensors with the previous layer.
* **PCC**: $\text{PCC}_{Layer} = \frac{\sum \text{Covariance}^2}{\sqrt{\sum \text{Var}_{curr}} \sqrt{\sum \text{Var}_{prev}}}$
  - Average Pearson Correlation of the tensors in the layer.
* **Cov**: The **unnormalized covariance** between current layer's concatenated tensors and the previous layer.

More information is available in https://github.com/ggml-org/llama.cpp/pull/14891
