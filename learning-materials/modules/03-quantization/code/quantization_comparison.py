#!/usr/bin/env python3
"""
Quantization Comparison Script

Compares different quantization formats for quality and performance.
Helps choose the optimal quantization for your use case.

Usage:
    python quantization_comparison.py --model model.gguf --formats Q4_K_M Q5_K_M Q8_0
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd


class QuantizationComparer:
    """Compare different quantization formats"""

    def __init__(self, model_path: str, llama_cpp_dir: str = "../../../"):
        self.model_path = Path(model_path)
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.results = []

    def quantize_model(self, output_path: Path, quant_format: str) -> bool:
        """Quantize model to specified format"""
        print(f"\n🔄 Quantizing to {quant_format}...")

        cmd = [
            str(self.llama_cpp_dir / "llama-quantize"),
            str(self.model_path),
            str(output_path),
            quant_format
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"✅ Quantized to {quant_format}")
                return True
            else:
                print(f"❌ Failed to quantize: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"❌ Quantization timeout")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def measure_perplexity(self, model_path: Path, test_file: str = "wikitext-2-test.txt") -> float:
        """Measure perplexity on test dataset"""
        print(f"📊 Measuring perplexity...")

        cmd = [
            str(self.llama_cpp_dir / "llama-perplexity"),
            "-m", str(model_path),
            "-f", test_file,
            "--perplexity"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # Parse perplexity from output
            for line in result.stdout.split('\n'):
                if 'perplexity' in line.lower():
                    # Extract number after 'perplexity:'
                    parts = line.split(':')
                    if len(parts) > 1:
                        ppl = float(parts[1].split()[0])
                        print(f"  Perplexity: {ppl:.2f}")
                        return ppl

            print("⚠️  Could not parse perplexity")
            return -1.0
        except Exception as e:
            print(f"❌ Error measuring perplexity: {e}")
            return -1.0

    def measure_performance(self, model_path: Path, prompt: str = "Hello, world!",
                           n_tokens: int = 100, n_trials: int = 5) -> Dict[str, float]:
        """Measure inference performance"""
        print(f"⚡ Measuring performance ({n_trials} trials)...")

        times = []

        for trial in range(n_trials):
            cmd = [
                str(self.llama_cpp_dir / "llama-cli"),
                "-m", str(model_path),
                "-p", prompt,
                "-n", str(n_tokens),
                "--no-display-prompt"
            ]

            try:
                start = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                elapsed = time.time() - start

                if result.returncode == 0:
                    times.append(elapsed)
                    print(f"  Trial {trial + 1}: {elapsed:.2f}s ({n_tokens/elapsed:.2f} t/s)")
            except Exception as e:
                print(f"⚠️  Trial {trial + 1} failed: {e}")

        if times:
            avg_time = sum(times) / len(times)
            tps = n_tokens / avg_time
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            return {
                'avg_time_sec': avg_time,
                'tokens_per_sec': tps,
                'std_dev_sec': std_dev,
                'trials': len(times)
            }
        else:
            return {
                'avg_time_sec': -1,
                'tokens_per_sec': -1,
                'std_dev_sec': -1,
                'trials': 0
            }

    def get_model_size(self, model_path: Path) -> float:
        """Get model file size in GB"""
        size_bytes = model_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        return size_gb

    def compare_formats(self, formats: List[str], test_file: str = None,
                       measure_quality: bool = True,
                       measure_speed: bool = True) -> pd.DataFrame:
        """Compare multiple quantization formats"""

        print(f"\n{'='*60}")
        print(f"Comparing {len(formats)} quantization formats")
        print(f"{'='*60}")

        for quant_format in formats:
            result = {
                'format': quant_format,
                'model_size_gb': 0,
                'perplexity': -1,
                'tokens_per_sec': -1,
                'avg_time_sec': -1
            }

            # Generate output path
            output_path = self.model_path.parent / f"{self.model_path.stem}-{quant_format.lower()}.gguf"

            # Quantize if needed
            if not output_path.exists():
                if not self.quantize_model(output_path, quant_format):
                    continue
            else:
                print(f"✓ Using existing {quant_format} model")

            # Get model size
            result['model_size_gb'] = self.get_model_size(output_path)
            print(f"  Size: {result['model_size_gb']:.2f} GB")

            # Measure quality
            if measure_quality and test_file:
                result['perplexity'] = self.measure_perplexity(output_path, test_file)

            # Measure performance
            if measure_speed:
                perf = self.measure_performance(output_path)
                result.update(perf)

            self.results.append(result)

        # Create DataFrame
        df = pd.DataFrame(self.results)
        return df

    def visualize_results(self, df: pd.DataFrame, output_path: str = "comparison_results.png"):
        """Create visualization of comparison results"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Model Size comparison
        ax = axes[0, 0]
        df_sorted = df.sort_values('model_size_gb')
        ax.barh(df_sorted['format'], df_sorted['model_size_gb'], color='skyblue')
        ax.set_xlabel('Model Size (GB)')
        ax.set_title('Model Size by Quantization Format')
        ax.grid(axis='x', alpha=0.3)

        # 2. Performance comparison
        ax = axes[0, 1]
        df_perf = df[df['tokens_per_sec'] > 0].sort_values('tokens_per_sec', ascending=False)
        ax.barh(df_perf['format'], df_perf['tokens_per_sec'], color='lightgreen')
        ax.set_xlabel('Tokens per Second')
        ax.set_title('Inference Speed Comparison')
        ax.grid(axis='x', alpha=0.3)

        # 3. Perplexity comparison
        ax = axes[1, 0]
        df_ppl = df[df['perplexity'] > 0].sort_values('perplexity')
        ax.barh(df_ppl['format'], df_ppl['perplexity'], color='lightcoral')
        ax.set_xlabel('Perplexity (lower is better)')
        ax.set_title('Quality Comparison (Perplexity)')
        ax.grid(axis='x', alpha=0.3)

        # 4. Size vs Speed tradeoff
        ax = axes[1, 1]
        df_valid = df[(df['tokens_per_sec'] > 0) & (df['model_size_gb'] > 0)]
        ax.scatter(df_valid['model_size_gb'], df_valid['tokens_per_sec'], s=100, alpha=0.6)
        for _, row in df_valid.iterrows():
            ax.annotate(row['format'], (row['model_size_gb'], row['tokens_per_sec']),
                       fontsize=8, ha='right')
        ax.set_xlabel('Model Size (GB)')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Size vs Speed Trade-off')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved visualization to {output_path}")

    def generate_report(self, df: pd.DataFrame, output_path: str = "comparison_report.md"):
        """Generate markdown report"""

        with open(output_path, 'w') as f:
            f.write("# Quantization Comparison Report\n\n")
            f.write(f"**Model**: {self.model_path.name}\n\n")
            f.write(f"**Formats Tested**: {', '.join(df['format'].tolist())}\n\n")

            f.write("## Summary Table\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            # Best in each category
            if len(df) > 0:
                f.write("## Recommendations\n\n")

                # Best quality
                if (df['perplexity'] > 0).any():
                    best_quality = df[df['perplexity'] > 0].nsmallest(1, 'perplexity').iloc[0]
                    f.write(f"**Best Quality**: {best_quality['format']} ")
                    f.write(f"(perplexity: {best_quality['perplexity']:.2f})\n\n")

                # Best speed
                if (df['tokens_per_sec'] > 0).any():
                    best_speed = df[df['tokens_per_sec'] > 0].nlargest(1, 'tokens_per_sec').iloc[0]
                    f.write(f"**Best Speed**: {best_speed['format']} ")
                    f.write(f"({best_speed['tokens_per_sec']:.2f} tokens/sec)\n\n")

                # Smallest size
                best_size = df.nsmallest(1, 'model_size_gb').iloc[0]
                f.write(f"**Smallest Size**: {best_size['format']} ")
                f.write(f"({best_size['model_size_gb']:.2f} GB)\n\n")

                # Best balance
                if (df['tokens_per_sec'] > 0).any() and (df['perplexity'] > 0).any():
                    df_valid = df[(df['tokens_per_sec'] > 0) & (df['perplexity'] > 0)].copy()
                    # Normalize metrics
                    df_valid['norm_speed'] = df_valid['tokens_per_sec'] / df_valid['tokens_per_sec'].max()
                    df_valid['norm_quality'] = df_valid['perplexity'].min() / df_valid['perplexity']
                    df_valid['score'] = df_valid['norm_speed'] + df_valid['norm_quality']
                    best_balance = df_valid.nlargest(1, 'score').iloc[0]
                    f.write(f"**Best Balance**: {best_balance['format']} ")
                    f.write(f"(quality: {best_balance['perplexity']:.2f}, ")
                    f.write(f"speed: {best_balance['tokens_per_sec']:.2f} t/s)\n\n")

        print(f"\n📝 Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantization formats")
    parser.add_argument("--model", required=True, help="Path to base GGUF model")
    parser.add_argument("--formats", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"],
                       help="Quantization formats to compare")
    parser.add_argument("--test-file", help="Test file for perplexity measurement")
    parser.add_argument("--no-quality", action="store_true", help="Skip quality measurement")
    parser.add_argument("--no-speed", action="store_true", help="Skip speed measurement")
    parser.add_argument("--llama-cpp-dir", default="../../../",
                       help="Path to llama.cpp directory")
    parser.add_argument("--output-viz", default="comparison_results.png",
                       help="Output path for visualization")
    parser.add_argument("--output-report", default="comparison_report.md",
                       help="Output path for report")
    parser.add_argument("--output-json", default="comparison_results.json",
                       help="Output path for JSON results")

    args = parser.parse_args()

    # Create comparer
    comparer = QuantizationComparer(args.model, args.llama_cpp_dir)

    # Run comparison
    df = comparer.compare_formats(
        formats=args.formats,
        test_file=args.test_file,
        measure_quality=not args.no_quality and args.test_file is not None,
        measure_speed=not args.no_speed
    )

    # Display results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    # Save results
    df.to_json(args.output_json, orient='records', indent=2)
    print(f"\n💾 Saved results to {args.output_json}")

    # Generate visualization
    if len(df) > 0:
        comparer.visualize_results(df, args.output_viz)

    # Generate report
    comparer.generate_report(df, args.output_report)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
