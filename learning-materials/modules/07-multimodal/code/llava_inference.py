#!/usr/bin/env python3
"""
LLaVA Inference Script
======================

This script demonstrates how to use LLaVA (Large Language and Vision Assistant)
models with llama-cpp-python for vision-language tasks.

Requirements:
    pip install llama-cpp-python pillow numpy

Usage:
    python llava_inference.py --image cat.jpg --prompt "What is in this image?"
"""

import argparse
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import base64
from io import BytesIO


class LLaVAInference:
    """LLaVA inference wrapper for easy image understanding"""

    def __init__(self, model_path, mmproj_path, n_ctx=2048, n_gpu_layers=0):
        """
        Initialize LLaVA model

        Args:
            model_path: Path to the GGUF language model
            mmproj_path: Path to the vision projector (mmproj-*.gguf)
            n_ctx: Context length (default 2048, recommend 4096 for multimodal)
            n_gpu_layers: Number of layers to offload to GPU
        """
        print(f"Loading LLaVA model...")
        print(f"  Language model: {model_path}")
        print(f"  Vision projector: {mmproj_path}")

        # Initialize chat handler with vision support
        self.chat_handler = Llava15ChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )

        # Load language model
        self.llm = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,  # Required for multimodal
            verbose=False
        )

        print("✓ Model loaded successfully")

    def encode_image_base64(self, image_path):
        """
        Encode image to base64 data URI

        Args:
            image_path: Path to image file

        Returns:
            Base64 data URI string
        """
        image = Image.open(image_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large (optional, for efficiency)
        max_size = 672  # LLaVA-1.6 can handle up to 672x672
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Encode to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"

    def query(self, image_path, prompt, max_tokens=512, temperature=0.7):
        """
        Ask a question about an image

        Args:
            image_path: Path to the image file
            prompt: Question or instruction about the image
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            dict: Response with text, usage stats, and metadata
        """
        # Encode image
        image_data_uri = self.encode_image_base64(image_path)

        # Create message with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Generate response
        print(f"\nProcessing image: {image_path}")
        print(f"Prompt: {prompt}")
        print("Generating response...\n")

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "USER:", "\n\n\n"]
        )

        return {
            'text': response['choices'][0]['message']['content'],
            'usage': response['usage'],
            'model': response['model']
        }

    def batch_query(self, image_paths, prompt, max_tokens=256):
        """
        Process multiple images with the same prompt

        Args:
            image_paths: List of image paths
            prompt: Common prompt for all images
            max_tokens: Maximum tokens per response

        Returns:
            list: Responses for each image
        """
        results = []

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing {image_path}...")

            try:
                result = self.query(image_path, prompt, max_tokens=max_tokens)
                results.append({
                    'image': image_path,
                    'response': result['text'],
                    'success': True
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image': image_path,
                    'error': str(e),
                    'success': False
                })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaVA inference on images"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to GGUF language model"
    )
    parser.add_argument(
        "--mmproj",
        required=True,
        help="Path to vision projector (mmproj-*.gguf)"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (or directory for batch processing)"
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Prompt/question about the image"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU"
    )

    args = parser.parse_args()

    # Initialize model
    llava = LLaVAInference(
        model_path=args.model,
        mmproj_path=args.mmproj,
        n_ctx=4096,
        n_gpu_layers=args.n_gpu_layers
    )

    # Check if image is directory or file
    image_path = Path(args.image)

    if image_path.is_dir():
        # Batch processing
        image_files = list(image_path.glob("*.jpg")) + \
                     list(image_path.glob("*.jpeg")) + \
                     list(image_path.glob("*.png"))

        print(f"\nBatch processing {len(image_files)} images...")

        results = llava.batch_query(
            image_files,
            args.prompt,
            max_tokens=args.max_tokens
        )

        # Print results
        print("\n" + "="*80)
        print("BATCH RESULTS")
        print("="*80)

        for result in results:
            print(f"\nImage: {result['image']}")
            if result['success']:
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result['error']}")
            print("-" * 80)

    else:
        # Single image
        result = llava.query(
            args.image,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        # Print result
        print("\n" + "="*80)
        print("RESPONSE")
        print("="*80)
        print(result['text'])
        print("\n" + "="*80)
        print(f"Tokens used: {result['usage']['total_tokens']}")


if __name__ == "__main__":
    main()


# Example usage:
"""
# Single image
python llava_inference.py \\
    --model llava-v1.5-7b-Q4_K_M.gguf \\
    --mmproj mmproj-model-f16.gguf \\
    --image cat.jpg \\
    --prompt "What animal is in this image and what is it doing?"

# Batch processing
python llava_inference.py \\
    --model llava-v1.5-7b-Q4_K_M.gguf \\
    --mmproj mmproj-model-f16.gguf \\
    --image ./images/ \\
    --prompt "Describe this image in one sentence." \\
    --max-tokens 100

# With GPU acceleration
python llava_inference.py \\
    --model llava-v1.5-7b-Q4_K_M.gguf \\
    --mmproj mmproj-model-f16.gguf \\
    --image photo.jpg \\
    --prompt "What objects are in this image?" \\
    --n-gpu-layers 35
"""
