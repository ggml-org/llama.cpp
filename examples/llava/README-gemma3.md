# Gemma 3 vision

> [!IMPORTANT]
>
> This is very experimental, only used for demo purpose.

## How to get mmproj.gguf?

```bash
cd gemma-3-4b-it
python ../llama.cpp/examples/llava/gemma3_convert_encoder_to_gguf.py .

# output file is mmproj.gguf
```

## How to run it?

What you need:
- The text model GGUF, can be converted using `convert_hf_to_gguf.py`
- The mmproj file from step above
- An image file

```bash
# build
cmake -B build
cmake --build build --target llama-gemma3-cli

# run it
./build/bin/llama-gemma3-cli -m {text_model}.gguf --mmproj mmproj.gguf --image your_image.jpg
```
