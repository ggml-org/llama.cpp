# Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts [thousands of models](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)

You can use any `llama.cpp`-compatible model from [Hugging Face](https://huggingface.co/) using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama cli -hf ggml-org/gemma-3-1b-it-GGUF
```

You can use the same CLI invocation to download from other sites, by pointing the `MODEL_ENDPOINT` environment variable to an endpoint compatible with the Hugging Face API.
`llama.cpp` can also run models you have downloaded locally to your filesystem.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.
To learn more about model quantization, [read this documentation](../tools/quantize/README.md)

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

## Exporting per-layer embeddings separately

For models with a per-layer embedding (PLE) table, export the backbone and table from the same checkpoint:

```sh
python3 convert_hf_to_gguf.py /path/to/model --no-ple --outfile model-no-ple.gguf
python3 convert_hf_to_gguf.py /path/to/model --ple-only --outfile model-ple.gguf
llama-cli -m model-no-ple.gguf --shard model-ple.gguf
```

`--no-ple` excludes only `per_layer_token_embd.weight` and keeps the PLE metadata and projection weights. `--ple-only` exports only that table, without a tokenizer. Each export supports `--outtype`, so the table and backbone can use different precisions. Default output names start with `no-ple-` and `ple-` respectively.

These modes are mutually exclusive and require a model with a PLE table, such as Qwen3.8 Flash Next or Gemma 4. They cannot be combined with vocabulary, multimodal projector, MTP-only, or DSpark-only export. `--no-ple` can be combined with `--no-mtp`.

`--ple` is an alias for `--shard`. Extra shards can be comma separated or supplied with repeated options, for example `-m backbone.gguf --shard /fast/ple.gguf,/other/token-embd.gguf`. Paths may be on different drives. Tensor names must be unique across all files, except for the PLE case below.

A numbered backbone is auto-discovered and checked as usual; explicitly supplied paths outside that numbered set are appended as extra shards. Their split numbering is ignored, and the loader uses the actual combined tensor count. Paths already in the discovered set are loaded once. If the main filename does not follow the numbered naming scheme, an explicit backbone list must still match its declared split count and order.

If an extra shard holds only `per_layer_token_embd.weight`, it replaces the PLE table of the model. If the model already has the table, the extra shard overrides it. A backbone split that holds only that tensor is skipped, and it can be absent from disk. This lets one PLE file be shared across quants and releases, for example `-m model-Q4_K_M-00001-of-00003.gguf --shard /fast/ple-Q8_0.gguf`.

## SenseNova Loop

[SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview) supports text generation and image understanding. The reference implementation is in [OpenSenseNova/SenseNova-Loop](https://github.com/OpenSenseNova/SenseNova-Loop). Related discussion: [#31571](https://github.com/ggml-org/llama.cpp/issues/31571).

Convert the text model and its Qwen3.5 vision projector separately:

```sh
python3 convert_hf_to_gguf.py /path/to/SenseNova-Loop-9B-Preview --outtype bf16 --outfile loop-bf16.gguf
python3 convert_hf_to_gguf.py /path/to/SenseNova-Loop-9B-Preview --mmproj --outtype f16 --outfile loop-vision-f16.gguf
```

The GGUF stores the 32 physical layers once. The runtime evaluates 56 logical layers, with separate attention KV, convolution and recurrent states for every visit. The two-stream mHC routers remain F32 during conversion and quantization. If the checkpoint contains an MTP head, the converter appends it after the logical layers. Checkpoints without MTP weights remain supported.

```sh
llama-quantize loop-bf16.gguf loop-Q4_K_M.gguf Q4_K_M
llama-cli -m loop-Q4_K_M.gguf -c 4096
llama-mtmd-cli -m loop-Q4_K_M.gguf --mmproj loop-vision-f16.gguf --image image.jpg -p "Describe this image."
llama-server -m loop-Q4_K_M.gguf --mmproj loop-vision-f16.gguf -c 4096
```

Use `--spec-type draft-mtp` with an MTP-enabled GGUF to enable speculative decoding. The draft head uses its own attention KV cache and shares the target model's weights:

```sh
llama-server -m loop-bf16.gguf --spec-type draft-mtp --spec-draft-n-max 3 -c 4096
```

The converter also accepts `--no-mtp` to omit the head, or `--mtp` to export only the head and shared embedding/output weights. A separate head can be passed with `-md loop-mtp.gguf --spec-type draft-mtp`. The head must come from the same checkpoint as the target model.

Use `-ngl 0` to run on the CPU. GPU offload uses existing backend operations; this implementation adds no model-specific GPU kernels. Tensor split mode and preserved FP8/NVFP4 checkpoints are not supported. GGUF files from experimental forks may use different tensor names and metadata; convert the original checkpoint with the matching converter.
