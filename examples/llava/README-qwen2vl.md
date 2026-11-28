# QWEN2-VL

This implementation supports all versions of Qwen2VL, e.g. [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

## Usage

After building, run `./llama-qwen2vl-cli` to use it. Or you can also get the ready one on Huggingface, e.g. [Qwen2-VL-2B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF) :

### The basic one for running with an image and a prompt

```sh
./bin/llama-qwen2vl-cli -m /models/Qwen2-VL-2B-Instruct-Q4_0.gguf --mmproj /models/mmproj-Qwen2-VL-2B-Instruct-f32.gguf -p 'Describe this image.' --image '/models/test_image.jpg'
```

The image argument is optional in case you just want to use the model for text. However, the mmproj still has to be there as it will be loaded.

Without defining the system prompt in the prompt, it will default to `You are a helpful assistant.`.

### Or if you want the image to be directly in the prompt as a base64

```sh
./llama-qwen2vl-cli -m /models/Qwen2-VL-2B-Instruct-Q4_0.gguf --mmproj /models/mmproj-Qwen2-VL-2B-Instruct-f32.gguf -p '<img src="{base64}">Describe this image.'
```

### Or a complete prompt with the system message

```sh
./llama-qwen2vl-cli -m /models/Qwen2-VL-2B-Instruct-Q4_0.gguf --mmproj /models/mmproj-Qwen2-VL-2B-Instruct-f32.gguf -p '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|vision_pad|><|vision_end|>Describe this image.' --image '/models/test_image.jpg'
```

**Note**: A lower temperature like 0.1 is recommended for better quality. Add `--temp 0.1` to the command to do so.
**Note**: For GPU offloading, ensure to use the `-ngl` flag as usual.

## GGUF Conversion

1. Clone the Qwen2-VL model:

```sh
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
```

2. Use `qwen2_vl_surgery.py` to prepare the model for conversion:

```sh
python ./examples/llava/qwen2_vl_surgery.py ./model_path --data_type fp32
```

It will generate the vision model, and output the filename in the log.

3. Use `examples/convert_hf_to_gguf.py` to convert the Qwen2-VL model to GGUF:

```sh
python convert_hf_to_gguf.py ./model_path -outtype f32
```

Now the model is ready to use in the `model_path` directory. You can quantize them as you normally would with other GGUF files.

*Have fun with the models ! :)*

## Current limitations

* This only supports the image to be in the very beginning of the input prompt to the LLM.
* The vision model (clip.cpp)'s GPU backend support, which Qwen2VL uses, is disabled.
