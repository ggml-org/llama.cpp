# Sesame CSM

This demo shows running inference of [Sesame CSM](https://github.com/SesameAILabs/csm) using llama.cpp / GGML

It contains 3 components (each has its own GGUF file):
1. Backbone LLM
2. Decoder LLM
3. Mimi decoder

## Quick start

By default, all GGUF files are downloaded from [ggml-org Hugging Face's account](https://huggingface.co/ggml-org/sesame-csm-1b-GGUF)

```sh
# build (make sure to have LLAMA_CURL enabled)
cmake -B build -DLLAMA_CURL=ON
cmake --build build -j --target llama-tts-csm

# run it
./build/bin/llama-tts-csm -p "[0]Hi, my name is Xuan Son. I am software engineer at Hugging Face."
```

## Convert the model yourself

To get the GGUF:

```sh
python examples/tts/convert_csm_to_gguf.py

# default output files:
# sesame-csm-backbone.gguf
# sesame-csm-decoder.gguf

# optionally, quantize it
# (lowest scheme is q8_0, it does not make sense to quantize further, quality degrades too much)
python examples/tts/convert_csm_to_gguf.py --outtype q8_0
```

Run the example using local file:

```sh
./build/bin/llama-tts-csm -m sesame-csm-backbone.gguf -mv kyutai-mimi.gguf -p "[0]Hello world."
# sesame-csm-backbone.gguf will automatically be loaded
# make sure the place these 2 GGUF files in the same directory

# output file: output.wav
```
