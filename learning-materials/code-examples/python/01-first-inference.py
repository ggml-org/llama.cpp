"""
Module: 01-first-inference.py
Purpose: Demonstrate basic inference with llama.cpp Python bindings
Learning Objectives:
    - Understand how to load a GGUF model
    - Perform basic text generation
    - Handle common errors and edge cases
    - Understand model initialization parameters

Prerequisites: Module 1 Lesson 1.1-1.3 complete
Estimated Time: 15 minutes
Module: 1.4 - Basic Inference
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


def load_model(
    model_path: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
    verbose: bool = True
) -> Optional[Llama]:
    """
    Load a GGUF model for inference.

    This function demonstrates the basic model loading process, including
    error handling and parameter configuration.

    Args:
        model_path: Path to .gguf model file (e.g., "models/llama-2-7b.Q4_K_M.gguf")
        n_ctx: Context window size in tokens (default: 2048)
            - Larger values allow longer prompts/conversations
            - More context = more memory usage
        n_gpu_layers: Number of layers to offload to GPU (default: 0 for CPU-only)
            - Set to -1 to offload all layers
            - Requires CUDA/Metal support
        verbose: Whether to print loading information

    Returns:
        Initialized Llama model instance, or None if loading fails

    Raises:
        FileNotFoundError: If model_path doesn't exist
        RuntimeError: If model loading fails

    Example:
        >>> model = load_model("models/llama-2-7b.Q4_K_M.gguf", n_ctx=4096)
        >>> if model:
        >>>     print(f"Model loaded successfully with {model.n_ctx()} context size")
    """
    # Validate model path exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download a GGUF model first. Try:\n"
            f"  - https://huggingface.co/models?search=gguf\n"
            f"  - https://huggingface.co/TheBloke (popular quantizations)"
        )

    if not model_file.suffix == ".gguf":
        print(f"Warning: File does not have .gguf extension: {model_path}")
        print("This may not be a valid GGUF model file.")

    try:
        if verbose:
            print(f"Loading model from: {model_path}")
            print(f"Context size: {n_ctx} tokens")
            print(f"GPU layers: {n_gpu_layers}")
            print("This may take a few moments...")

        # Initialize the model
        # Key parameters:
        # - n_ctx: Maximum context length
        # - n_gpu_layers: GPU offloading for better performance
        # - verbose: Control logging output
        llm = Llama(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose
        )

        if verbose:
            print(f"✓ Model loaded successfully!")
            print(f"  - Context window: {llm.n_ctx()} tokens")
            print(f"  - Vocabulary size: {llm.n_vocab()} tokens")

        return llm

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print(f"Common issues:", file=sys.stderr)
        print(f"  1. Insufficient RAM (model may be too large)", file=sys.stderr)
        print(f"  2. Corrupted model file (try re-downloading)", file=sys.stderr)
        print(f"  3. Incompatible model format (ensure it's GGUF)", file=sys.stderr)
        return None


def generate_text(
    model: Llama,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    stop: Optional[list] = None
) -> str:
    """
    Generate text from a prompt using the loaded model.

    Args:
        model: Loaded Llama model instance
        prompt: Input text to continue/complete
        max_tokens: Maximum number of tokens to generate (default: 128)
        temperature: Sampling temperature (default: 0.7)
            - 0.0: Deterministic (always picks most likely token)
            - 0.7: Balanced creativity and coherence (recommended)
            - 1.0+: More random and creative
        stop: List of strings that stop generation when encountered

    Returns:
        Generated text completion

    Example:
        >>> text = generate_text(model, "The capital of France is", max_tokens=50)
        >>> print(text)
    """
    if stop is None:
        stop = []

    try:
        # Generate text using the model
        # The model returns a dictionary with generation information
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False  # Don't include the prompt in the output
        )

        # Extract the generated text from the response
        generated_text = output["choices"][0]["text"]
        return generated_text

    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        return ""


def main() -> None:
    """
    Main demonstration function showing basic inference workflow.

    This example shows:
    1. How to load a GGUF model
    2. How to generate text from a prompt
    3. Proper error handling
    4. Basic parameter tuning
    """
    print("=" * 60)
    print("LLaMA.cpp Python - First Inference Example")
    print("=" * 60)
    print()

    # Example model path - adjust this to your actual model location
    # You can download models from HuggingFace, for example:
    # https://huggingface.co/TheBloke/Llama-2-7B-GGUF
    model_path = "models/llama-2-7b.Q4_K_M.gguf"

    # For testing, you can also try smaller models like:
    # - TinyLlama: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
    # model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    # Check if user provided a model path as command-line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print(f"Model path: {model_path}")
    print()

    # Step 1: Load the model
    print("Step 1: Loading model...")
    print("-" * 60)
    model = load_model(
        model_path=model_path,
        n_ctx=2048,  # Context window size
        n_gpu_layers=0,  # Use CPU only (set to -1 for GPU acceleration)
        verbose=True
    )

    if model is None:
        print("\nFailed to load model. Please check the error messages above.")
        print("\nTip: Make sure you have downloaded a GGUF model file.")
        sys.exit(1)

    print()

    # Step 2: Prepare a prompt
    print("Step 2: Preparing prompt...")
    print("-" * 60)
    prompt = "The capital of France is"
    print(f"Prompt: {prompt}")
    print()

    # Step 3: Generate text
    print("Step 3: Generating text...")
    print("-" * 60)
    print("Generating... (this may take a few seconds)")
    print()

    generated = generate_text(
        model=model,
        prompt=prompt,
        max_tokens=128,
        temperature=0.7
    )

    # Step 4: Display results
    print("Step 4: Results")
    print("-" * 60)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print()

    # Additional example: Try a longer prompt
    print("=" * 60)
    print("Additional Example: Longer prompt")
    print("=" * 60)
    print()

    long_prompt = """Write a short poem about artificial intelligence:

"""
    print(f"Prompt:\n{long_prompt}")
    print("Generating...")
    print()

    poem = generate_text(
        model=model,
        prompt=long_prompt,
        max_tokens=150,
        temperature=0.8  # Slightly higher temperature for creativity
    )

    print(f"Generated:\n{poem}")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("1. Model loading requires a valid GGUF file path")
    print("2. Context size (n_ctx) determines maximum input/output length")
    print("3. Temperature controls randomness (lower = more deterministic)")
    print("4. Always handle errors when loading models or generating text")
    print()
    print("Next steps:")
    print("- Try different prompts and parameters")
    print("- Experiment with temperature values (0.0 to 1.5)")
    print("- Try different model sizes and quantizations")
    print("- See 02-basic-chat.py for interactive chat examples")


if __name__ == "__main__":
    main()
