## MiniCPM-V 4.7

### Prepare models and code

Download [MiniCPM-V-4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7) PyTorch model from huggingface to "MiniCPM-V-4.7" folder.

The model must be the standard `transformers` checkpoint (no `trust_remote_code` for the text and vision graph used here); the architecture in `config.json` is `MiniCPMV4_7ForConditionalGeneration` with a `qwen3_5_text` (or `qwen3_5_moe_text`) text model and a SigLIP-based vision tower plus a window-attention `vit_merger`, same as MiniCPM-V 4.6. The text tower uses canvas M-RoPE.

If the checkpoint ships no MTP weights, pass `--no-mtp` to skip the nextn layers.

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


### Usage of MiniCPM-V 4.7

MiniCPM-V 4.7 is converted directly through `convert_hf_to_gguf.py`. The same script is invoked twice on the original Hugging Face directory: once to produce the language-model GGUF and once with `--mmproj` to produce the multimodal projector GGUF.

```bash
# language model
python ./convert_hf_to_gguf.py ../MiniCPM-V-4.7 --outfile ../MiniCPM-V-4.7/ggml-model-f16.gguf --no-mtp

# multimodal projector (vision tower + window-attention vit_merger + DownsampleMLP merger)
python ./convert_hf_to_gguf.py ../MiniCPM-V-4.7 --mmproj --outfile ../MiniCPM-V-4.7/mmproj-model-f16.gguf

# optional: quantize to Q4_K_M
./build/bin/llama-quantize ../MiniCPM-V-4.7/ggml-model-f16.gguf ../MiniCPM-V-4.7/ggml-model-Q4_K_M.gguf Q4_K_M
```

The default projector merges 16x (4x4 patches into one token). To keep 4x more visual tokens, copy the model dir and set `"downsample_mode": "4x"` in the copy's `preprocessor_config.json` before running the `--mmproj` conversion; the loader reads `clip.vision.projector.scale_factor` to pick the graph.


Inference on Linux or Mac
```bash
# run in single-turn mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-V-4.7/ggml-model-f16.gguf --mmproj ../MiniCPM-V-4.7/mmproj-model-f16.gguf -c 4096 --jinja --image xx.jpg -p "What is in the image?"

# run in conversation mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-V-4.7/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4.7/mmproj-model-f16.gguf --jinja
```

The chat template enables thinking by default. Pass `--chat-template-kwargs '{"enable_thinking": false}'` to `llama-server` to turn it off.
