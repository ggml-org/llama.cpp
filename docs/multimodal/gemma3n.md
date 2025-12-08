# Gemma 3n Vision

> [!IMPORTANT]
>
> This is very experimental, only used for demo purpose.

## Overview

Gemma 3n is an advanced multimodal model that uses the **MobileNetV5** vision encoder architecture (instead of SigLIP used in Gemma 3). The MobileNetV5 encoder provides efficient CNN-based feature extraction with Multi-Scale Fusion Adapter (MSFA) for combining features at different resolutions.

## Architecture Differences

- **Gemma 3**: Uses SigLIP vision encoder (Vision Transformer)
- **Gemma 3n**: Uses MobileNetV5 vision encoder (CNN with MSFA)

Both models share the same projection mechanism to the language model embedding space.

## Quick Start

You can use pre-quantized models from [ggml-org](https://huggingface.co/ggml-org)'s Hugging Face account (when available)

```bash
# build
cmake -B build
cmake --build build --target llama-mtmd-cli

# alternatively, install from brew (MacOS)
brew install llama.cpp

# run it (example - update with actual model names when available)
llama-mtmd-cli -hf ggml-org/gemma-3n-VARIANT-GGUF
```

## How to get mmproj.gguf?

Simply add `--mmproj` when converting the model via `convert_hf_to_gguf.py`:

```bash
cd gemma-3n-model-directory
python ../llama.cpp/convert_hf_to_gguf.py --outfile model.gguf --outtype f16 --mmproj .
# output file: mmproj-model.gguf
```

## How to run it?

What you need:
- The text model GGUF, can be converted using `convert_hf_to_gguf.py`
- The mmproj file from step above (contains MobileNetV5 vision encoder)
- An image file

```bash
# build
cmake -B build
cmake --build build --target llama-mtmd-cli

# run it
./build/bin/llama-mtmd-cli -m {text_model}.gguf --mmproj mmproj-model.gguf --image your_image.jpg
```

## Model Conversion Details

The conversion process handles:
1. **Text Model**: Standard Gemma 3n language model weights
2. **Vision Encoder**: MobileNetV5 architecture with:
   - Stem convolution layer
   - Multiple inverted residual blocks
   - Multi-Query Attention (MQA) blocks
   - Multi-Scale Fusion Adapter (MSFA)
3. **Projection Layers**: RMSNorm and linear projection to language model space

### Image Processing

- **Input Resolution**: Depends on model configuration (typically 384x384 or similar)
- **Output Tokens**: 256 soft tokens (16×16 grid)
- **Preprocessing**: Normalization based on model metadata

## Technical Implementation

### MobileNetV5 Components

1. **Inverted Residual Blocks**: Expansion → Depthwise Conv → Squeeze-Excitation → Projection
2. **RMSNorm2d**: 2D RMS normalization for feature maps
3. **Approximate GELU**: Activation function throughout the network
4. **MSFA**: Combines features from multiple scales for robust representation

### Integration with Language Model

The vision encoder outputs are processed through:
1. RMSNorm normalization
2. Soft embedding normalization
3. Linear projection to language model embedding dimension
4. Non-causal attention during vision processing

## Notes

- Gemma 3n uses `Gemma3p5RMSNorm` which has different normalization behavior than Gemma 3
- The MobileNetV5 architecture is more efficient than Vision Transformers for certain use cases
- Image tokens are processed with non-causal attention masks

## Troubleshooting

If you encounter issues:

1. **Model Loading Errors**: Ensure you're using the correct `--mmproj` file that matches the text model version
2. **Vision Encoder Not Found**: Make sure the mmproj file was generated with the `--mmproj` flag
3. **Image Size Mismatches**: Check the model's expected input resolution in the preprocessor config

## References

- [MobileNetV5 Paper](https://github.com/huggingface/pytorch-image-models) (timm implementation)
- [Gemma 3n Model Card](https://huggingface.co/transformers/models/gemma3n)
- [llama.cpp Multimodal Documentation](../)
