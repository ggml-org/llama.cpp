"""
Module: 03-sampling-parameters.py
Purpose: Demonstrate different sampling methods and parameters
Learning Objectives:
    - Understand temperature and its effect on generation
    - Learn about top-k and top-p (nucleus) sampling
    - Compare greedy vs stochastic sampling
    - Master generation parameter tuning
    - Observe quality vs creativity trade-offs

Prerequisites: Module 1 Lesson 1.4 - Basic Inference complete
Estimated Time: 25 minutes
Module: 1.4 - Basic Inference
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


def load_model(model_path: str) -> Optional[Llama]:
    """Load a GGUF model for experiments."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return None

    try:
        print(f"Loading model: {model_path}\n")
        return Llama(
            model_path=str(model_file),
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def generate_with_params(
    model: Llama,
    prompt: str,
    params: Dict[str, Any],
    label: str
) -> str:
    """
    Generate text with specific sampling parameters.

    Args:
        model: Loaded Llama model
        prompt: Input prompt
        params: Dictionary of generation parameters
        label: Descriptive label for this configuration

    Returns:
        Generated text
    """
    print(f"\n{label}")
    print("-" * 70)
    print(f"Parameters: {params}")
    print()

    try:
        output = model(prompt, echo=False, **params)
        generated = output["choices"][0]["text"]
        print(f"Generated:\n{generated}\n")
        return generated
    except Exception as e:
        print(f"Error: {e}\n")
        return ""


def experiment_temperature(model: Llama, prompt: str) -> None:
    """
    Experiment with different temperature values.

    Temperature controls randomness in token selection:
    - 0.0: Deterministic (greedy) - always picks most likely token
    - 0.1-0.3: Very focused, factual, conservative
    - 0.5-0.7: Balanced (default for most tasks)
    - 0.8-1.0: Creative, diverse
    - 1.0+: Very random, potentially incoherent

    Key insight: Temperature scales the logits before softmax.
    Lower temperature → more peaked distribution → less randomness
    Higher temperature → flatter distribution → more randomness
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temperature Effects")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]

    for temp in temperatures:
        params = {
            "max_tokens": 100,
            "temperature": temp,
            "top_k": 40,  # Keep these constant to isolate temperature effect
            "top_p": 0.9
        }

        label = f"Temperature: {temp}"
        if temp == 0.0:
            label += " (Greedy/Deterministic)"
        elif temp < 0.5:
            label += " (Very Focused)"
        elif temp < 0.8:
            label += " (Balanced)"
        elif temp < 1.2:
            label += " (Creative)"
        else:
            label += " (Very Random)"

        generate_with_params(model, prompt, params, label)


def experiment_top_k(model: Llama, prompt: str) -> None:
    """
    Experiment with top-k sampling.

    Top-k sampling: Only consider the k most likely next tokens.
    - Filters out unlikely tokens
    - Prevents sampling from the "long tail" of the distribution
    - Smaller k = more focused, less diverse
    - Larger k = more diverse, potentially less coherent

    Common values:
    - k=1: Greedy decoding (deterministic)
    - k=10: Very focused
    - k=40: Balanced (common default)
    - k=100: Diverse
    - k=0: Disabled (consider all tokens)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Top-K Sampling")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    top_k_values = [1, 10, 40, 100, 0]  # 0 = disabled

    for k in top_k_values:
        params = {
            "max_tokens": 100,
            "temperature": 0.7,  # Keep constant
            "top_k": k,
            "top_p": 1.0  # Disable top_p to isolate top_k effect
        }

        label = f"Top-K: {k}"
        if k == 0:
            label += " (Disabled - all tokens considered)"
        elif k == 1:
            label += " (Greedy)"
        elif k <= 20:
            label += " (Very Focused)"
        elif k <= 50:
            label += " (Balanced)"
        else:
            label += " (Diverse)"

        generate_with_params(model, prompt, params, label)


def experiment_top_p(model: Llama, prompt: str) -> None:
    """
    Experiment with top-p (nucleus) sampling.

    Top-p sampling: Consider smallest set of tokens whose cumulative
    probability exceeds p.
    - More adaptive than top-k
    - Automatically adjusts number of tokens based on confidence
    - High confidence → fewer tokens
    - Low confidence → more tokens

    Common values:
    - p=0.1: Very focused (only most confident predictions)
    - p=0.5: Moderately focused
    - p=0.9: Balanced (common default)
    - p=0.95: Diverse
    - p=1.0: Disabled (all tokens)

    Example:
    If token probabilities are [0.4, 0.3, 0.2, 0.05, 0.03, 0.02, ...]
    - p=0.5: Consider only [0.4, 0.3] (sum = 0.7 > 0.5)
    - p=0.9: Consider [0.4, 0.3, 0.2, 0.05, 0.03] (sum = 0.98 > 0.9)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Top-P (Nucleus) Sampling")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    top_p_values = [0.1, 0.5, 0.9, 0.95, 1.0]

    for p in top_p_values:
        params = {
            "max_tokens": 100,
            "temperature": 0.7,  # Keep constant
            "top_k": 0,  # Disable top_k to isolate top_p effect
            "top_p": p
        }

        label = f"Top-P: {p}"
        if p <= 0.3:
            label += " (Very Focused)"
        elif p <= 0.7:
            label += " (Moderately Focused)"
        elif p < 1.0:
            label += " (Balanced/Diverse)"
        else:
            label += " (Disabled)"

        generate_with_params(model, prompt, params, label)


def experiment_repeat_penalty(model: Llama, prompt: str) -> None:
    """
    Experiment with repeat penalty.

    Repeat penalty: Penalize tokens that have already appeared.
    - Helps prevent repetitive text
    - Value > 1.0 = penalize repetitions
    - Value = 1.0 = no penalty (default)
    - Value < 1.0 = encourage repetitions (rarely used)

    Common values:
    - 1.0: No penalty
    - 1.1: Light penalty (good for most tasks)
    - 1.3: Strong penalty (prevent loops)
    - 1.5+: Very strong penalty (may hurt coherence)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Repeat Penalty")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    penalties = [1.0, 1.1, 1.3, 1.5]

    for penalty in penalties:
        params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": penalty
        }

        label = f"Repeat Penalty: {penalty}"
        if penalty == 1.0:
            label += " (No Penalty)"
        elif penalty < 1.2:
            label += " (Light Penalty)"
        elif penalty < 1.4:
            label += " (Moderate Penalty)"
        else:
            label += " (Strong Penalty)"

        generate_with_params(model, prompt, params, label)


def experiment_combined(model: Llama, prompt: str) -> None:
    """
    Demonstrate recommended parameter combinations for different use cases.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Recommended Combinations")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    configurations = [
        {
            "name": "Factual/Deterministic (Q&A, Facts)",
            "params": {
                "max_tokens": 100,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "repeat_penalty": 1.0
            }
        },
        {
            "name": "Balanced (General Purpose)",
            "params": {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        },
        {
            "name": "Creative (Stories, Poetry)",
            "params": {
                "max_tokens": 100,
                "temperature": 0.9,
                "top_k": 100,
                "top_p": 0.95,
                "repeat_penalty": 1.1
            }
        },
        {
            "name": "Code Generation (Precise)",
            "params": {
                "max_tokens": 100,
                "temperature": 0.2,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.05
            }
        }
    ]

    for config in configurations:
        generate_with_params(
            model,
            prompt,
            config["params"],
            f"Use Case: {config['name']}"
        )


def print_summary() -> None:
    """Print a summary of sampling parameters."""
    print("\n" + "=" * 70)
    print("SAMPLING PARAMETERS SUMMARY")
    print("=" * 70)
    print("""
┌──────────────────┬─────────────────┬─────────────────────────────────────┐
│ Parameter        │ Range           │ Effect                               │
├──────────────────┼─────────────────┼─────────────────────────────────────┤
│ temperature      │ 0.0 - 2.0       │ Controls randomness                  │
│                  │ • 0.0: greedy   │ • Low: focused, deterministic        │
│                  │ • 0.7: balanced │ • High: creative, random             │
│                  │ • 1.0+: random  │                                      │
├──────────────────┼─────────────────┼─────────────────────────────────────┤
│ top_k            │ 1 - 100+        │ Consider top K tokens                │
│                  │ • 1: greedy     │ • Low: focused                       │
│                  │ • 40: default   │ • High: diverse                      │
│                  │ • 0: disabled   │ • 0: all tokens                      │
├──────────────────┼─────────────────┼─────────────────────────────────────┤
│ top_p            │ 0.0 - 1.0       │ Nucleus sampling                     │
│                  │ • 0.5: focused  │ • Low: only confident tokens         │
│                  │ • 0.9: default  │ • High: more tokens                  │
│                  │ • 1.0: disabled │ • More adaptive than top_k           │
├──────────────────┼─────────────────┼─────────────────────────────────────┤
│ repeat_penalty   │ 1.0 - 2.0       │ Penalize repeated tokens             │
│                  │ • 1.0: none     │ • 1.0: no penalty                    │
│                  │ • 1.1: light    │ • >1.0: reduce repetition            │
│                  │ • 1.3+: strong  │ • Very high may hurt coherence       │
└──────────────────┴─────────────────┴─────────────────────────────────────┘

Recommended Combinations:
• Factual/Precise:   temp=0.0-0.3, top_k=10, top_p=0.9, repeat=1.0
• General Purpose:   temp=0.7, top_k=40, top_p=0.9, repeat=1.1
• Creative Writing:  temp=0.9-1.0, top_k=100, top_p=0.95, repeat=1.1
• Code Generation:   temp=0.2, top_k=40, top_p=0.9, repeat=1.05

Pro Tips:
1. Start with defaults (temp=0.7, top_k=40, top_p=0.9)
2. Adjust one parameter at a time
3. Lower temperature for factual tasks
4. Higher temperature for creative tasks
5. Use repeat_penalty to prevent loops
6. top_p is generally better than top_k (more adaptive)
""")


def main() -> None:
    """
    Main function demonstrating all sampling experiments.
    """
    print("=" * 70)
    print("LLaMA.cpp Python - Sampling Parameters Demonstration")
    print("=" * 70)
    print()
    print("This example compares different sampling strategies.")
    print("Each experiment isolates one parameter to show its effect.")
    print()

    # Get model path
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Load model
    model = load_model(model_path)
    if model is None:
        sys.exit(1)

    # Test prompts for different experiments
    factual_prompt = "The capital of France is"
    creative_prompt = "Once upon a time, in a magical forest"

    # Run experiments
    print("\n[Running 5 experiments. This will take several minutes...]\n")

    # Experiment 1: Temperature
    experiment_temperature(model, factual_prompt)

    # Experiment 2: Top-K
    experiment_top_k(model, creative_prompt)

    # Experiment 3: Top-P
    experiment_top_p(model, creative_prompt)

    # Experiment 4: Repeat Penalty
    experiment_repeat_penalty(model, "The word 'the' is")

    # Experiment 5: Combined
    experiment_combined(model, creative_prompt)

    # Print summary
    print_summary()

    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Temperature is the most important parameter")
    print("2. top_p is generally preferred over top_k")
    print("3. Repeat penalty helps prevent loops")
    print("4. Different tasks need different settings")
    print("5. Always experiment with your specific use case")
    print("\nNext: Try 04-context-management.py")


if __name__ == "__main__":
    main()
