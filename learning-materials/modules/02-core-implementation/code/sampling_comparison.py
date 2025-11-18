#!/usr/bin/env python3
"""
Sampling Strategy Comparison

Compare different sampling methods and their effects on generation.
Demonstrates all major sampling strategies with side-by-side comparison.

Usage:
    python sampling_comparison.py model.gguf "Once upon a time"
"""

import sys
from llama_cpp import Llama
from typing import Dict, Any
import time


class SamplingComparator:
    """Compare different sampling strategies"""

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=32,  # Use GPU if available
            verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 50, **sampling_params) -> Dict[str, Any]:
        """Generate with specific sampling parameters"""
        start_time = time.time()

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            **sampling_params
        )

        elapsed = time.time() - start_time

        return {
            'text': output['choices'][0]['text'],
            'tokens': output['usage']['completion_tokens'],
            'time': elapsed,
            'tokens_per_sec': output['usage']['completion_tokens'] / elapsed if elapsed > 0 else 0,
        }

    def compare_strategies(self, prompt: str, max_tokens: int = 50):
        """Compare multiple sampling strategies"""
        strategies = {
            'Greedy': {
                'temperature': 0.0,
                'top_k': 1,
                'description': 'Deterministic, always picks highest probability'
            },
            'Low Temperature': {
                'temperature': 0.2,
                'top_p': 0.95,
                'description': 'Very focused, low diversity'
            },
            'Balanced': {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
                'description': 'Good balance (recommended)'
            },
            'Creative': {
                'temperature': 0.9,
                'top_p': 0.95,
                'repeat_penalty': 1.15,
                'description': 'More diverse, creative'
            },
            'Top-K Only': {
                'temperature': 0.8,
                'top_k': 20,
                'top_p': 1.0,
                'description': 'Top-K sampling (k=20)'
            },
            'Top-P Only': {
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': -1,
                'description': 'Nucleus sampling (p=0.9)'
            },
            'High Temperature': {
                'temperature': 1.5,
                'top_p': 0.95,
                'description': 'Very random, often incoherent'
            },
        }

        print("\n" + "=" * 100)
        print(f"🎲 Sampling Strategy Comparison")
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print("=" * 100)

        results = {}
        for name, params in strategies.items():
            print(f"\n{name}: {params['description']}")
            print(f"  Parameters: {', '.join(f'{k}={v}' for k, v in params.items() if k != 'description')}")
            print("  " + "-" * 96)

            # Remove description from params
            gen_params = {k: v for k, v in params.items() if k != 'description'}

            try:
                result = self.generate(prompt, max_tokens, **gen_params)
                results[name] = result

                print(f"  Output: {result['text']}")
                print(f"  Stats: {result['tokens']} tokens in {result['time']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        return results

    def test_temperature_sweep(self, prompt: str, max_tokens: int = 30):
        """Test different temperature values"""
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]

        print("\n" + "=" * 100)
        print("🌡️  Temperature Sweep")
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print("=" * 100)

        for temp in temperatures:
            print(f"\nTemperature = {temp}")
            print("  " + "-" * 96)

            result = self.generate(
                prompt,
                max_tokens,
                temperature=temp,
                top_p=0.95,
                repeat_penalty=1.0
            )

            print(f"  {result['text']}")

    def test_repetition_penalty(self, prompt: str, max_tokens: int = 50):
        """Test repetition penalty effects"""
        penalties = [1.0, 1.05, 1.1, 1.2, 1.3]

        print("\n" + "=" * 100)
        print("🔁 Repetition Penalty Test")
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print("=" * 100)

        for penalty in penalties:
            print(f"\nRepetition Penalty = {penalty}")
            print("  " + "-" * 96)

            result = self.generate(
                prompt,
                max_tokens,
                temperature=0.8,
                top_p=0.9,
                repeat_penalty=penalty
            )

            print(f"  {result['text']}")

    def test_top_k_vs_top_p(self, prompt: str, max_tokens: int = 40):
        """Compare Top-K vs Top-P"""
        configs = [
            {'name': 'Top-K = 10', 'top_k': 10, 'top_p': 1.0},
            {'name': 'Top-K = 40', 'top_k': 40, 'top_p': 1.0},
            {'name': 'Top-K = 100', 'top_k': 100, 'top_p': 1.0},
            {'name': 'Top-P = 0.9', 'top_k': -1, 'top_p': 0.9},
            {'name': 'Top-P = 0.95', 'top_k': -1, 'top_p': 0.95},
            {'name': 'Top-P = 0.99', 'top_k': -1, 'top_p': 0.99},
        ]

        print("\n" + "=" * 100)
        print("🎯 Top-K vs Top-P Comparison")
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print("=" * 100)

        for config in configs:
            print(f"\n{config['name']}")
            print("  " + "-" * 96)

            result = self.generate(
                prompt,
                max_tokens,
                temperature=0.8,
                top_k=config['top_k'],
                top_p=config['top_p']
            )

            print(f"  {result['text']}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python sampling_comparison.py <model.gguf> <prompt>")
        print("\nExamples:")
        print("  python sampling_comparison.py model.gguf \"Once upon a time\"")
        print("  python sampling_comparison.py model.gguf \"The meaning of life is\"")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]

    print(f"📚 Loading model: {model_path}")

    try:
        comparator = SamplingComparator(model_path)
        print("✅ Model loaded successfully\n")

        # Run all comparisons
        comparator.compare_strategies(prompt, max_tokens=50)
        comparator.test_temperature_sweep(prompt, max_tokens=30)
        comparator.test_repetition_penalty(prompt, max_tokens=50)
        comparator.test_top_k_vs_top_p(prompt, max_tokens=40)

        print("\n" + "=" * 100)
        print("✅ Comparison complete!")
        print("\n💡 Key Takeaways:")
        print("  • Greedy: Deterministic but can be repetitive")
        print("  • Low temp (0.2-0.5): Good for factual tasks")
        print("  • Balanced (0.7-0.9): Best for most use cases")
        print("  • High temp (1.0+): Creative but may be incoherent")
        print("  • Repetition penalty: Use 1.1-1.2 to reduce repetition")
        print("  • Top-P usually better than fixed Top-K (adapts to distribution)")
        print("=" * 100)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
