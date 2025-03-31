# GLMV-EDGE

Currently this implementation supports [glm-edge-v-2b](https://huggingface.co/THUDM/glm-edge-v-2b) and [glm-edge-v-5b](https://huggingface.co/THUDM/glm-edge-v-5b).

## Usage
Build with cmake or run `make llama-llava-cli` to build it.

After building, run: `./llama-llava-cli` to see the usage. For example:

```sh
./llama-llava-cli -m model_path/ggml-model-f16.gguf --mmproj model_path/mmproj-model-f16.gguf --image img_path/image.jpg -p "<|system|>\n system prompt <image><|user|>\n prompt <|assistant|>\n"
```

**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.
**note**: For GPU offloading ensure to use the `-ngl` flag just like usual

## GGUF conversion

1. Clone a GLMV-EDGE model ([2B](https://huggingface.co/THUDM/glm-edge-v-2b) or [5B](https://huggingface.co/THUDM/glm-edge-v-5b)). For example:

```sh
git clone https://huggingface.co/THUDM/glm-edge-v-5b or https://huggingface.co/THUDM/glm-edge-v-2b
```

2. Use `glmedge-surgery.py` to split the GLMV-EDGE model to LLM and multimodel projector constituents:

```sh
python ./examples/llava/glmedge-surgery.py -m ../model_path
```

4. Use `glmedge-convert-image-encoder-to-gguf.py` to convert the GLMV-EDGE image encoder to GGUF:

```sh
python ./examples/llava/glmedge-convert-image-encoder-to-gguf.py -m ../model_path --llava-projector ../model_path/glm.projector --output-dir ../model_path
```

5. Use `examples/convert_hf_to_gguf.py` to convert the LLM part of GLMV-EDGE to GGUF:

```sh
python convert_hf_to_gguf.py ../model_path
```

Now both the LLM part and the image encoder are in the `model_path` directory.
