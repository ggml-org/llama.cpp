# Gemma 3 vision

> [!IMPORTANT]
>
> This is very experimental, only used for demo purpose.

## Quick started

You can use pre-quantized model from [ggml-org](https://huggingface.co/ggml-org)'s Hugging Face account

```bash
# build
cmake -B build
cmake --build build --target llama-mtmd-cli

# alternatively, install from brew (MacOS)
brew install llama.cpp

# run it
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF
llama-mtmd-cli -hf ggml-org/gemma-3-12b-it-GGUF
llama-mtmd-cli -hf ggml-org/gemma-3-27b-it-GGUF

# note: 1B model does not support vision
```

## How to get mmproj.gguf?

Simply to add `--mmproj` in when converting model via `convert_hf_to_gguf.py`:

```bash
cd gemma-3-4b-it
python ../llama.cpp/convert_hf_to_gguf.py --outfile model.gguf --outtype f16 --mmproj .
# output file: mmproj-model.gguf
```

## How to run it?

What you need:
- The text model GGUF, can be converted using `convert_hf_to_gguf.py`
- The mmproj file from step above
- An image file

```bash
# build
cmake -B build
cmake --build build --target llama-mtmd-cli

# run it
./build/bin/llama-mtmd-cli -m {text_model}.gguf --mmproj mmproj.gguf --image your_image.jpg
```
