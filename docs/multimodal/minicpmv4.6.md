## MiniCPM-V 4.6

### Prepare models and code

Download [MiniCPM-V-4_6](https://huggingface.co/openbmb/MiniCPM-V-4_6) PyTorch model from huggingface to "MiniCPM-V-4_6" folder.

The model must be the standard `transformers` v5.7.0+ checkpoint (no `trust_remote_code`); the architecture in `config.json` is `MiniCPMV4_6ForConditionalGeneration` with a `qwen3_5_text` text model and a SigLIP-based vision tower plus a window-attention `vit_merger`.

### Build llama.cpp

If there are differences in usage, please refer to the official build [documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Clone llama.cpp:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Build llama.cpp using `CMake`:
```bash
cmake -B build
cmake --build build --config Release
```


### Usage of MiniCPM-V 4.6

Unlike older MiniCPM-V variants, MiniCPM-V 4.6 is converted directly through `convert_hf_to_gguf.py`. The same script is invoked twice on the original Hugging Face directory: once to produce the language-model GGUF and once with `--mmproj` to produce the multimodal projector GGUF.

```bash
# language model
python ./convert_hf_to_gguf.py ../MiniCPM-V-4_6 --outfile ../MiniCPM-V-4_6/ggml-model-f16.gguf

# multimodal projector (vision tower + window-attention vit_merger + DownsampleMLP merger)
python ./convert_hf_to_gguf.py ../MiniCPM-V-4_6 --mmproj --outfile ../MiniCPM-V-4_6/mmproj-model-f16.gguf

# optional: quantize to Q4_K_M
./build/bin/llama-quantize ../MiniCPM-V-4_6/ggml-model-f16.gguf ../MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf Q4_K_M
```


Inference on Linux or Mac
```bash
# run in single-turn mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-V-4_6/ggml-model-f16.gguf --mmproj ../MiniCPM-V-4_6/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run in conversation mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-V-4_6/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4_6/mmproj-model-f16.gguf
```
