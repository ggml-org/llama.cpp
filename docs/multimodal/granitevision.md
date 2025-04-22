# Granite Vision

Download the model and point your `GRANITE_MODEL` environment variable to the path.

```bash
$ git clone https://huggingface.co/ibm-granite/granite-vision-3.2-2b
$ export GRANITE_MODEL=./granite-vision-3.2-2b
```


### 1. Running llava surgery v2.
First, we need to run the llava surgery script as shown below:

`python llava_surgery_v2.py -C -m $GRANITE_MODEL`

You should see two new files (`llava.clip` and `llava.projector`) written into your model's directory, as shown below.

```bash
$ ls $GRANITE_MODEL | grep -i llava
llava.clip
llava.projector
```

We should see that the projector and visual encoder get split out into the llava files. Quick check to make sure they aren't empty:
```python
import os
import torch

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

encoder_tensors = torch.load(os.path.join(MODEL_PATH, "llava.clip"))
projector_tensors = torch.load(os.path.join(MODEL_PATH, "llava.projector"))

assert len(encoder_tensors) > 0
assert len(projector_tensors) > 0
```

If you actually inspect the `.keys()` of the loaded tensors, you should see a lot of `vision_model` tensors in the `encoder_tensors`, and 5 tensors (`'multi_modal_projector.linear_1.bias'`, `'multi_modal_projector.linear_1.weight'`, `'multi_modal_projector.linear_2.bias'`, `'multi_modal_projector.linear_2.weight'`, `'image_newline'`) in the multimodal `projector_tensors`.


### 2. Creating the Visual Component GGUF
Next, create a new directory to hold the visual components, and copy the llava.clip/projector files, as shown below.

```bash
$ ENCODER_PATH=$PWD/visual_encoder
$ mkdir $ENCODER_PATH

$ cp $GRANITE_MODEL/llava.clip $ENCODER_PATH/pytorch_model.bin
$ cp $GRANITE_MODEL/llava.projector $ENCODER_PATH/
```

Now, we need to write a config for the visual encoder. In order to convert the model, be sure to use the correct `image_grid_pinpoints`, as these may vary based on the model. You can find the `image_grid_pinpoints` in `$GRANITE_MODEL/config.json`.

```json
{
    "_name_or_path": "siglip-model",
    "architectures": [
      "SiglipVisionModel"
    ],
    "image_grid_pinpoints": [
        [384,384],
        [384,768],
        [384,1152],
        [384,1536],
        [384,1920],
        [384,2304],
        [384,2688],
        [384,3072],
        [384,3456],
        [384,3840],
        [768,384],
        [768,768],
        [768,1152],
        [768,1536],
        [768,1920],
        [1152,384],
        [1152,768],
        [1152,1152],
        [1536,384],
        [1536,768],
        [1920,384],
        [1920,768],
        [2304,384],
        [2688,384],
        [3072,384],
        [3456,384],
        [3840,384]
    ],
    "mm_patch_merge_type": "spatial_unpad",
    "hidden_size": 1152,
    "image_size": 384,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
    "projection_dim": 0,
    "vision_feature_layer": [-24, -20, -12, -1]
}
```

At this point you should have something like this:
```bash
$ ls $ENCODER_PATH
config.json             llava.projector         pytorch_model.bin
```

Now convert the components to GGUF; Note that we also override the image mean/std dev to `[.5,.5,.5]` since we use the SigLIP visual encoder - in the transformers model, you can find these numbers in the `preprocessor_config.json`.
```bash
$ python convert_image_encoder_to_gguf.py \
    -m $ENCODER_PATH \
    --llava-projector $ENCODER_PATH/llava.projector \
    --output-dir $ENCODER_PATH \
    --clip-model-is-vision \
    --clip-model-is-siglip \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5
```

This will create the first GGUF file at `$ENCODER_PATH/mmproj-model-f16.gguf`; we will refer to the absolute path of this file as the `$VISUAL_GGUF_PATH.`


### 3. Creating the LLM GGUF.
The granite vision model contains a granite LLM as its language model. For now, the easiest way to get the GGUF for LLM is by loading the composite model in `transformers` and exporting the LLM so that it can be directly converted with the normal conversion path.

First, set the `LLM_EXPORT_PATH` to the path to export the `transformers` LLM to.
```bash
$ export LLM_EXPORT_PATH=$PWD/granite_vision_llm
```

```python
import os
import transformers

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

LLM_EXPORT_PATH = os.getenv("LLM_EXPORT_PATH")
if not LLM_EXPORT_PATH:
    raise ValueError("env var LLM_EXPORT_PATH is unset!")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

# NOTE: granite vision support was added to transformers very recently (4.49);
# if you get size mismatches, your version is too old.
# If you are running with an older version, set `ignore_mismatched_sizes=True`
# as shown below; it won't be loaded correctly, but the LLM part of the model that
# we are exporting will be loaded correctly.
model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)

tokenizer.save_pretrained(LLM_EXPORT_PATH)
model.language_model.save_pretrained(LLM_EXPORT_PATH)
```

Now you can convert the exported LLM to GGUF with the normal converter in the root of the llama cpp project.
```bash
$ LLM_GGUF_PATH=$LLM_EXPORT_PATH/granite_llm.gguf
...
$ python convert_hf_to_gguf.py --outfile $LLM_GGUF_PATH $LLM_EXPORT_PATH
```


### 4. Quantization
If you want to quantize the LLM, you can do so with `llama-quantize` as you would any other LLM. For example:
```bash
$ ./build/bin/llama-quantize $LLM_EXPORT_PATH/granite_llm.gguf $LLM_EXPORT_PATH/granite_llm_q4_k_m.gguf Q4_K_M
$ LLM_GGUF_PATH=$LLM_EXPORT_PATH/granite_llm_q4_k_m.gguf
```

Note that currently you cannot quantize the visual encoder because granite vision models use SigLIP as the visual encoder, which has tensor dimensions that are not divisible by 32.


### 5. Running the Model in Llama cpp
Build llama cpp normally; you should have a target binary named `llama-mtmd-cli`, which you can pass two binaries to. As an example, we pass the the llama.cpp banner.

```bash
$ ./build/bin/llama-mtmd-cli -m $LLM_GGUF_PATH \
    --mmproj $VISUAL_GGUF_PATH \
    -c 16384 \
    --temp 0
```
