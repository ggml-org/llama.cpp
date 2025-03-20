# quantize

You can also use the [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) space on Hugging Face to build your own quants without any setup.

Note: It is synced from llama.cpp `main` every 6 hours.

Example usage:

```bash
# obtain the official LLaMA model weights and place them in ./models
ls ./models
llama-2-7b tokenizer_checklist.chk tokenizer.model
# [Optional] for models using BPE tokenizers
ls ./models
<folder containing weights and tokenizer json> vocab.json
# [Optional] for PyTorch .bin models like Mistral-7B
ls ./models
<folder containing weights and tokenizer json>

# install Python dependencies
python3 -m pip install -r requirements.txt

# convert the model to ggml FP16 format
python3 convert_hf_to_gguf.py models/mymodel/

# quantize the model to 4-bits (using Q4_K_M method)
./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M

# update the gguf filetype to current version if older version is now unsupported
./llama-quantize ./models/mymodel/ggml-model-Q4_K_M.gguf ./models/mymodel/ggml-model-Q4_K_M-v2.gguf COPY
```

Run the quantized model:

```bash
# start inference on a gguf model
./llama-cli -m ./models/mymodel/ggml-model-Q4_K_M.gguf -cnv -p "You are a helpful assistant"
```

When running the larger models, make sure you have enough disk space to store all the intermediate files.

## Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| Model | Original size | Quantized size (Q4_0) |
|------:|--------------:|----------------------:|
|    7B |         13 GB |                3.9 GB |
|   13B |         24 GB |                7.8 GB |
|   30B |         60 GB |               19.5 GB |
|   65B |        120 GB |               38.5 GB |

## Quantization

Several quantization methods are supported. They differ in the resulting model disk size and inference speed.

*(outdated)*

| Model | Measure      |    F16 |   Q4_0 |   Q4_1 |   Q5_0 |   Q5_1 |   Q8_0 |
|------:|--------------|-------:|-------:|-------:|-------:|-------:|-------:|
|    7B | perplexity   | 5.9066 | 6.1565 | 6.0912 | 5.9862 | 5.9481 | 5.9070 |
|    7B | file size    |  13.0G |   3.5G |   3.9G |   4.3G |   4.7G |   6.7G |
|    7B | ms/tok @ 4th |    127 |     55 |     54 |     76 |     83 |     72 |
|    7B | ms/tok @ 8th |    122 |     43 |     45 |     52 |     56 |     67 |
|    7B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |
|   13B | perplexity   | 5.2543 | 5.3860 | 5.3608 | 5.2856 | 5.2706 | 5.2548 |
|   13B | file size    |  25.0G |   6.8G |   7.6G |   8.3G |   9.1G |    13G |
|   13B | ms/tok @ 4th |      - |    103 |    105 |    148 |    160 |    131 |
|   13B | ms/tok @ 8th |      - |     73 |     82 |     98 |    105 |    128 |
|   13B | bits/weight  |   16.0 |    4.5 |    5.0 |    5.5 |    6.0 |    8.5 |

- [k-quants](https://github.com/ggml-org/llama.cpp/pull/1684)
- recent k-quants improvements and new i-quants
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

**Llama 2 7B**

| Quantization | Bits per Weight (BPW) |
|--------------|-----------------------|
| Q2_K         | 3.35                  |
| Q3_K_S       | 3.50                  |
| Q3_K_M       | 3.91                  |
| Q3_K_L       | 4.27                  |
| Q4_K_S       | 4.58                  |
| Q4_K_M       | 4.84                  |
| Q5_K_S       | 5.52                  |
| Q5_K_M       | 5.68                  |
| Q6_K         | 6.56                  |

**Llama 2 13B**

Quantization | Bits per Weight (BPW)
-- | --
Q2_K | 3.34
Q3_K_S | 3.48
Q3_K_M | 3.89
Q3_K_L | 4.26
Q4_K_S | 4.56
Q4_K_M | 4.83
Q5_K_S | 5.51
Q5_K_M | 5.67
Q6_K | 6.56

**Llama 2 70B**

Quantization | Bits per Weight (BPW)
-- | --
Q2_K | 3.40
Q3_K_S | 3.47
Q3_K_M | 3.85
Q3_K_L | 4.19
Q4_K_S | 4.53
Q4_K_M | 4.80
Q5_K_S | 5.50
Q5_K_M | 5.65
Q6_K | 6.56
