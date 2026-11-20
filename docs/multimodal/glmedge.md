# GLMV-EDGE

Currently this implementation supports [glm-edge-v-2b](https://huggingface.co/THUDM/glm-edge-v-2b) and [glm-edge-v-5b](https://huggingface.co/THUDM/glm-edge-v-5b).

## Usage
Build the `llama-mtmd-cli` binary.

After building, run: `./build/bin/llama-mtmd-cli` to see the usage. For example:

```sh
./build/bin/llama-mtmd-cli -m glm-edge-v-5b/ggml-model-f16.gguf --mmproj glm-edge-v-5b/mmproj-model-f16.gguf
```

**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.
**note**: For GPU offloading ensure to use the `-ngl` flag just like usual

## GGUF conversion

1. Clone a GLMV-EDGE model ([2B](https://huggingface.co/THUDM/glm-edge-v-2b) or [5B](https://huggingface.co/THUDM/glm-edge-v-5b)). For example:

```sh
git clone https://huggingface.co/THUDM/glm-edge-v-5b
```

2. Use `glmedge-surgery.py` to split the GLMV-EDGE model to LLM and multimodel projector constituents:

```sh
python ./tools/mtmd/legacy-models/glmedge-surgery.py -m glm-edge-v-5b
```

3. Use `glmedge-convert-image-encoder-to-gguf.py` to convert the GLMV-EDGE image encoder to GGUF:

```sh
python ./tools/mtmd/legacy-models/glmedge-convert-image-encoder-to-gguf.py -m glm-edge-v-5b --llava-projector glm-edge-v-5b/glm.projector --output-dir glm-edge-v-5b
```

4. Use `convert_hf_to_gguf.py` to convert the LLM part of GLMV-EDGE to GGUF:

```sh
python convert_hf_to_gguf.py glm-edge-v-5b --outtype f16 --outfile glm-edge-v-5b/ggml-model-f16.gguf
```

Now both the LLM part and the image encoder are in the `glm-edge-v-5b` directory.
