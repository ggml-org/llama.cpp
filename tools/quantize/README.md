# quantize

This tool takes a GGUF input model file, typically in a high-precision format like F32 or BF16, and converts it to a quantized format.
Quantization reduces the precision of model weights (e.g., from 32-bit floats to 4-bit integers), which shrinks the model's size and can speed up inference.
This process however, may introduce some accuracy loss which is usually measured in [Perplexity](https://huggingface.co/docs/transformers/en/perplexity) (ppl) and/or [Kullback–Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (kld).
This can be minimized by using a suitable imatrix file.

You can also use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to build your own quants without any setup. It syncs from llama.cpp `main` every 6 hours.

## Overview

Quantization is done in two phases:
- Convert the original model to GGUF format.
- Quantize the converted GGUF file.

If the model supports multimodal inputs (images or audio), you also need to convert and quantize the multimodal encoders and projectors.

To perform these tasks, you need to install the Python requirements:

```bash
python3 -m pip install -r requirements.txt
```

Or if you use `uv`:

```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

## Prepare the input GGUF file

To convert a model from a Hugging Face repo, you can use a command like the following:

```
python convert_hf_to_gguf.py --outfile gemma-4-E2B-it-bf16.gguf --outtype bf16 --remote google/gemma-4-E2B-it
```

Notes:
- In the usual case where the model is distributed in 16-bit format, `--outtype auto` (or omitting `--outtype` entirely) also works well.
- If you have previously downloaded the model locally, specify the directory and remove the `--remote` flag.
- For compatibility reasons, the Python requirements install transformers 4, but more and more models (like Gemma 4) require transformers 5. You can safely `pip install -U transformers` to get the latest version.

## Quantize the GGUF

After you have created a high-quality GGUF version of the model, you use `llama-quantize` to apply quantization. For example, quantize to `Q4_K_M` using a command like the following:

```bash
./build/bin/llama-quantize gemma-4-E2B-it-bf16.gguf gemma-4-E2B-it-Q4_K_M.gguf Q4_K_M
```

Various quantization methods are described [later in this document](#quantize).

Options:
* `--allow-requantize` allow requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
* `--leave-output-tensor` leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
* `--pure` disable k-quant mixtures and quantizes all tensors to the same type
* `--imatrix file_name` use data in file_name as importance matrix for quant optimizations
* `--include-weights tensor_name` use importance matrix for this tensor (can be specified multiple times)
* `--exclude-weights tensor_name` use importance matrix for the tensors **not** specified (include/exclude cannot be mixed)
* `--output-tensor-type` use a specific quant type for the output.weight tensor
* `--token-embedding-type` use a specific quant type for the token embeddings tensor
* `--keep-split` generate the quantized model in the same shards as the input file instead of a single quantized file
* `--dry-run` simulate the quantization process

Advanced options:
* `--tensor-type` quantize specific tensor(s) to specific quant types. Supports regex syntax. May be specified multiple times
* `--prune-layers` prune (remove) the layers in the list
* `--override-kv` option to override model metadata by key in the quantized model. May be specified multiple times
* `--target-bpw` automatically choose quant types to meet an overall bits per weight (bpw) target
* `--target-size` automatically choose quant types to meet a file size target
* `--state-file` file name to use or save to the bpw/size error computations; if no name is provided it defaults to `<model name>-<model hash>.bpw_state`

## (Optional) Convert the multimodal components

llama.cpp will convert the LLM portion of the source model, which is enough for conversational applications. If the model accepts multimodal inputs and you wish to take advantage of them, you need to create a separate GGUF file. This file is generically known as `mmproj`, for "multimedia projector"; however, it may contain various components such as vision or audio encoders in addition to projections.

Multimodal components are usually much smaller than the LLMs they come with. In addition, their quality has a direct impact on the quality of LLM generations, because these components are in charge of preparing the inputs for the LLM: the closer inputs are to data seen during training, the better LLM results will be.

For these reasons, multimodal components are usually kept in a high-quality format such as bf16 or q8. The impact on speed and memory from using a smaller quant is negligible, but overall quality could be impacted.

```bash
python convert_hf_to_gguf.py --mmproj --outfile mmproj-gemma-4-E2B-it-Q8_0.gguf --outtype q8_0 --remote google/gemma-4-E2B-it
```

## Run the quantized model


```bash
./build/bin/llama cli -m ./gemma-4-E2B-it-Q4_K_M.gguf --mmproj ./mmproj-gemma-4-E2B-it-Q8_0.gguf --image <input_image> --prompt "Describe this image"
```

## Quantization Examples

```bash
# naive Q4_K_M quantization using default settings and 8 CPU threads. Output will be "ggml-model-Q4_K_M.gguf"
./llama-quantize input-model-f32.gguf q4_k_m 8
```

```bash
# simulate naive Q4_K_M quantization process. No model will be generated
./llama-quantize --dry-run input-model-f32.gguf q4_k_m 8
```

```bash
# naive Q4_K_M quantization overriding general.name and model.rope.freq_base metadata values
./llama-quantize --override-kv general.name=str:"New model name" --override-kv model.rope.freq_base=float:1.234e+08 input-model-f32.gguf q4_k_m 8
```

```bash
#  quantize model enabling re-quantization, leaving the output tensor unquantized and all others quantized at the same level (Q4_K)
./llama-quantize --allow-requantize --leave-output-tensor --pure input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model using an importance matrix for specified tensors only (attn_v and ffn_down)
./llama-quantize --imatrix imatrix.gguf --include-weights attn_v --include-weights ffn_down input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model setting output tensor to Q5_K_M, token embeddings to Q3_K_M, and keeping the input file's shards
./llama-quantize --imatrix imatrix.gguf --output-tensor-type q5_k --token-embedding-type q3_k --keep-split input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model using a regex to quantize attn_k tensors in odd layers to Q5_K_M and attn_q tensors in even layers to Q3_K_M
./llama-quantize --imatrix imatrix.gguf --tensor-type "\.(\d*[13579])\.attn_k=q5_k" --tensor-type "\.(\d*[02468])\.attn_q=q3_k" input-model-f32.gguf q4_k_m 8
```

```bash
# quantize model setting tensors attn_v and ffn_down to Q5_K_M and pruning layers 20, 21, and 22
./llama-quantize --imatrix imatrix.gguf --tensor-type attn_v=q5_k --tensor-type ffn_down=q5_k --prune-layers 20,21,22 input-model-f32.gguf q4_k_m 8
```

```bash
# override expert used count metadata to 16, prune layers 20, 21, and 22 without quantizing the model (copy tensors) and use specified name for the output file
./llama-quantize --imatrix imatrix.gguf --override-kv qwen3moe.expert_used_count=int:16 --prune-layers 20,21,22 input-model-f32.gguf pruned-model-f32.gguf copy 8
```

```bash
# quantize model targeting a specific bpw average and save the target computations to the default file. Model type is optional and can be omitted
./llama-quantize --target-bpw 4.5678 --state-file --imatrix imatrix.gguf input-model-f32.gguf 8
```

```bash
# quantize model targeting a specific bpw average reusing previous target computations in deault file (e.g. input-model-8fd7a8bef0803042.bpw_state)
./llama-quantize --target-bpw 2.345 ---state-file --imatrix imatrix.gguf input-model-f32.gguf 8
```

```bash
# quantize model targeting a specific file size and save the target computations to a custom file.
./llama-quantize --target-size 1.5g --state-file my-state-file.dat --imatrix imatrix.gguf input-model-f32.gguf 8
```

## Memory/Disk Requirements

When running the larger models, make sure you have enough disk space to store all the intermediate files.
As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same. For example (Llama 3.1):

| Model | Original size | Quantized size (Q4_K_M) |
|------:|--------------:|------------------------:|
|    8B |       32.1 GB |                  4.9 GB |
|   70B |      280.9 GB |                 43.1 GB |
|  405B |    1,625.1 GB |                249.1 GB |

## Quantization

Several quantization methods are supported. They differ in the resulting model disk size and inference speed. For example,

### [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

| Quant Type | bits/weight | size (GiB) | prompt processing t/s @ 512 | text generation t/s @ 128 |
|:----------:|------------:|-----------:|----------------------------:|--------------------------:|
|   IQ1_S    |      2.0042 |       1.87 |                858.88 ±1.22 |               79.73 ±0.79 |
|   IQ1_M    |      2.1460 |       2.01 |                847.99 ±0.47 |               72.92 ±0.14 |
|  IQ2_XXS   |      2.3824 |       2.23 |                852.39 ±0.85 |               79.86 ±0.22 |
|   IQ2_XS   |      2.5882 |       2.42 |               826.99 ±12.51 |               78.04 ±0.46 |
|   IQ2_S    |      2.7403 |       2.56 |               783.55 ±13.73 |               77.30 ±2.47 |
|   IQ2_M    |      2.9294 |       2.74 |                787.68 ±7.00 |               74.44 ±0.15 |
|  IQ3_XXS   |      3.2548 |       3.04 |                813.88 ±6.53 |               73.95 ±0.20 |
|   IQ3_XS   |      3.4977 |       3.27 |                708.71 ±1.26 |               71.67 ±0.54 |
|   IQ3_S    |      3.6606 |       3.42 |                798.78 ±8.81 |               69.31 ±0.63 |
|   IQ3_M    |      3.7628 |       3.52 |               768.70 ±13.73 |               70.15 ±0.33 |
|   IQ4_XS   |      4.4597 |       4.17 |               771.80 ±11.38 |               77.51 ±0.20 |
|   IQ4_NL   |      4.6818 |       4.38 |                818.55 ±9.58 |               76.71 ±0.20 |
|   Q2_K_S   |      2.9697 |       2.78 |                798.91 ±6.40 |               90.01 ±0.12 |
|    Q2_K    |      3.1593 |       2.95 |                784.45 ±7.85 |               79.85 ±0.20 |
|   Q3_K_S   |      3.6429 |       3.41 |                752.17 ±7.94 |               71.68 ±0.22 |
|   Q3_K_L   |      4.2979 |       4.02 |                761.17 ±7.55 |               69.38 ±0.49 |
|   Q4_K_S   |      4.6672 |       4.36 |                818.55 ±9.58 |               76.71 ±0.20 |
|   Q4_K_S   |      4.6672 |       4.36 |                818.55 ±9.58 |               76.71 ±0.20 |
|   Q4_K_M   |      4.8944 |       4.58 |               821.81 ±21.44 |               71.93 ±1.52 |
|   Q5_K_S   |      5.5704 |       5.21 |                752.52 ±0.99 |               69.53 ±0.18 |
|   Q5_K_M   |      5.7036 |       5.33 |                758.69 ±7.43 |               67.23 ±1.08 |
|    Q6_K    |      6.5633 |       6.14 |               812.01 ±10.82 |               58.67 ±3.13 |
|    Q8_0    |      8.5008 |       7.95 |                865.09 ±8.30 |               50.93 ±0.08 |
|    F16     |     16.0005 |      14.96 |                923.49 ±0.53 |               29.17 ±0.04 |

## Background information on llama-quantize

- [k-quants](https://github.com/ggml-org/llama.cpp/pull/1684)
- k-quants improvements and i-quants
  - [#2707](https://github.com/ggml-org/llama.cpp/pull/2707)
  - [#2807](https://github.com/ggml-org/llama.cpp/pull/2807)
  - [#4773 - 2-bit i-quants (inference)](https://github.com/ggml-org/llama.cpp/pull/4773)
  - [#4856 - 2-bit i-quants (inference)](https://github.com/ggml-org/llama.cpp/pull/4856)
  - [#4861 - importance matrix](https://github.com/ggml-org/llama.cpp/pull/4861)
  - [#4872 - MoE models](https://github.com/ggml-org/llama.cpp/pull/4872)
  - [#4897 - 2-bit quantization](https://github.com/ggml-org/llama.cpp/pull/4897)
  - [#4930 - imatrix for all k-quants](https://github.com/ggml-org/llama.cpp/pull/4930)
  - [#4951 - imatrix on the GPU](https://github.com/ggml-org/llama.cpp/pull/4957)
  - [#4969 - imatrix for legacy quants](https://github.com/ggml-org/llama.cpp/pull/4969)
  - [#4996 - k-quants tuning](https://github.com/ggml-org/llama.cpp/pull/4996)
  - [#5060 - Q3_K_XS](https://github.com/ggml-org/llama.cpp/pull/5060)
  - [#5196 - 3-bit i-quants](https://github.com/ggml-org/llama.cpp/pull/5196)
  - [quantization tuning](https://github.com/ggml-org/llama.cpp/pull/5320), [another one](https://github.com/ggml-org/llama.cpp/pull/5334), and [another one](https://github.com/ggml-org/llama.cpp/pull/5361)
- [target bpw/size](https://github.com/ggml-org/llama.cpp/pull/15550)
