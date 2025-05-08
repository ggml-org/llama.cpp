# Quantizing CLIP Visual Projector

This is the tool for quantizing the CLIP visual projector model. Quantization reduces the precision of the model's weights, which can significantly decrease the model size and improve inference speed, often with minimal impact on performance.

## Usage

To quantize a CLIP visual projector model, use the following command:

```sh
./bin/llama-llava-clip-quantize-cli /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf <type>
```

After the quantization, the visual projector can be used freely with the existing LLAVA cli (LLAVA, Qwen2VL, etc).

### Arguments

- `/path/to/ggml-model-f32.gguf`: The path to the input model file in FP32 or FP16 format.
- `/path/to/ggml-model-quantized.gguf`: The path where the quantized model will be saved.
- `<type>`: The quantization type to apply. This should be an integer corresponding to one of the quantization types defined in the `enum ggml_type`.

### Quantization Types

The following quantization types are supported, based on the `enum ggml_type` definition:

- `2` - `q4_0`: 4-bit quantization with a single scale value.
- `3` - `q4_1`: 4-bit quantization with a separate scale value for each block.
- `6` - `q5_0`: 5-bit quantization with a single scale value.
- `7` - `q5_1`: 5-bit quantization with a separate scale value for each block.
- `8` - `q8_0`: 8-bit quantization with a single scale value.

### Example

To quantize a model using the `q4_0` quantization type, you would run:

```sh
./bin/llama-llava-clip-quantize-cli /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf 2
```

This command will generate a quantized model at `/path/to/ggml-model-quantized.gguf` using the `q4_0` quantization method.

## Notes

- Quantization can lead to a loss in model accuracy, depending on the chosen quantization type. It is recommended to evaluate the quantized model's performance on your specific task to ensure it meets your requirements.
- The quantized model will typically be smaller in size and faster to run, making it more suitable for deployment in resource-constrained environments.
