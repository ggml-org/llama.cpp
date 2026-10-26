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
