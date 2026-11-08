# LLaVA

Currently this implementation supports [llava-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b) variants,
as well as llava-1.6 [llava-v1.6](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2) variants.

The pre-converted [7b](https://huggingface.co/mys/ggml_llava-v1.5-7b)
and [13b](https://huggingface.co/mys/ggml_llava-v1.5-13b)
models are available.
For llava-1.6 a variety of prepared gguf models are available as well [7b-34b](https://huggingface.co/cmp-nct/llava-1.6-gguf)

After API is confirmed, more models will be supported / uploaded.

## Usage
Build the `llama-mtmd-cli` binary.

After building, run: `./llama-mtmd-cli` to see the usage. For example:

```sh
./llama-mtmd-cli -m ../llava-v1.5-7b/ggml-model-f16.gguf \
    --mmproj ../llava-v1.5-7b/mmproj-model-f16.gguf \
    --chat-template vicuna
```

**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.
**note**: For GPU offloading ensure to use the `-ngl` flag just like usual

## LLaVA 1.5

1. Clone a LLaVA and a CLIP model ([available options](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)). For example:

```sh
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

2. Install the required Python packages:

```sh
pip install -r tools/mtmd/requirements.txt
```

3. Use `llava_surgery.py` to split the LLaVA model to LLaMA and multimodel projector constituents:

```sh
python ./tools/mtmd/llava_surgery.py -m ../llava-v1.5-7b
```

4. Use `convert_image_encoder_to_gguf.py` to convert the LLaVA image encoder to GGUF:

```sh
python ./tools/mtmd/convert_image_encoder_to_gguf.py -m ../clip-vit-large-patch14-336 --llava-projector ../llava-v1.5-7b/llava.projector --output-dir ../llava-v1.5-7b
```

5. Use `examples/convert_legacy_llama.py` to convert the LLaMA part of LLaVA to GGUF:

```sh
python ./examples/convert_legacy_llama.py ../llava-v1.5-7b --skip-unknown
```

Now both the LLaMA part and the image encoder are in the `llava-v1.5-7b` directory.

## LLaVA 1.6 gguf conversion
1) First clone a LLaVA 1.6 model:
```console
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
```

2) Install the required Python packages:

```sh
pip install -r tools/mtmd/requirements.txt
```

3) Use `llava_surgery_v2.py` which also supports llava-1.5 variants pytorch as well as safetensor models:
```console
python tools/mtmd/llava_surgery_v2.py -C -m ../llava-v1.6-vicuna-7b/
```
- you will find a llava.projector and a llava.clip file in your model directory

4) Copy the llava.clip file into a subdirectory (like vit), rename it to pytorch_model.bin and add a fitting vit configuration to the directory:
```console
mkdir vit
cp ../llava-v1.6-vicuna-7b/llava.clip vit/pytorch_model.bin
cp ../llava-v1.6-vicuna-7b/llava.projector vit/
curl -s -q https://huggingface.co/cmp-nct/llava-1.6-gguf/raw/main/config_vit.json -o vit/config.json
```

5) Create the visual gguf model:
```console
python ./tools/mtmd/convert_image_encoder_to_gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-vision
```
- This is similar to llava-1.5, the difference is that we tell the encoder that we are working with the pure vision model part of CLIP

6) Then convert the model to gguf format:
```console
python ./examples/convert_legacy_llama.py ../llava-v1.6-vicuna-7b/ --skip-unknown
```

7) And finally we can run the llava cli using the 1.6 model version:
```console
./llama-mtmd-cli -m ../llava-v1.6-vicuna-7b/ggml-model-f16.gguf --mmproj vit/mmproj-model-f16.gguf
```

**note** llava-1.6 needs more context than llava-1.5, at least 3000 is needed (just run it at -c 4096)

**note** llava-1.6 greatly benefits from batched prompt processing (defaults work)

**note** if the language model in step `6)` is incompatible with the legacy conversion script, the easiest way handle the LLM model conversion is to load the model in transformers, and export only the LLM from the llava next model.

```python
import os
import transformers

model_path = ...
llm_export_path = ...

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForImageTextToText.from_pretrained(model_path)

tokenizer.save_pretrained(llm_export_path)
model.language_model.save_pretrained(llm_export_path)
```

Then, you can convert the LLM using the `convert_hf_to_gguf.py` script, which handles more LLM architectures.

## Chat template

For llava-1.5 and llava-1.6, you need to use `vicuna` chat template. Simply add `--chat-template vicuna` to activate this template.


## How to know if you are running in llava-1.5 or llava-1.6 mode

When running llava-cli you will see a visual information right before the prompt is being processed:

**Llava-1.5:**
`encode_image_with_clip: image embedding created: 576 tokens`

**Llava-1.6 (anything above 576):**
`encode_image_with_clip: image embedding created: 2880 tokens`


Alternatively just pay notice to how many "tokens" have been used for your prompt, it will also show 1000+ tokens for llava-1.6
