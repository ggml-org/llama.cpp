#!/usr/bin/env python3
"""
Performance Profiler

Comprehensive performance profiling for llama.cpp models.
Measures tokens/sec, memory usage, and provides detailed timing breakdowns.

Usage:
    python performance_profiler.py --model model.gguf --profile all
    python performance_profiler.py --model model.gguf --threads 1,2,4,8
"""

import argparse
import json
import subprocess
import time
import psutil
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ProfileResult:
    """Performance profile result"""
    model: str
    threads: int
    prompt_tokens: int
    generated_tokens: int
    prompt_eval_time_ms: float
    token_eval_time_ms: float
    total_time_ms: float
    tokens_per_sec: float
    prompt_tokens_per_sec: float
    memory_rss_mb: float
    memory_peak_mb: float
    cpu_percent: float


class PerformanceProfiler:
    """Profile llama.cpp performance"""

    def __init__(self, llama_cpp_dir: str = "../../../"):
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.llama_cli = self.llama_cpp_dir / "llama-cli"

        if not self.llama_cli.exists():
            raise FileNotFoundError(f"llama-cli not found at {self.llama_cli}")

    def parse_llama_output(self, output: str) -> Dict:
        """Parse timing information from llama-cli output"""
        stats = {}

        patterns = {
            'prompt_tokens': r'prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens',
            'generated_tokens': r'eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs',
            'prompt_eval_time': r'prompt eval time\s*=\s*([\d.]+)\s*ms',
            'token_eval_time': r'eval time\s*=\s*([\d.]+)\s*ms',
            'total_time': r'total time\s*=\s*([\d.]+)\s*ms',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                stats[key] = float(match.group(1))

        return stats

    def measure_memory_usage(self, pid: int) -> Dict[str, float]:
        """Measure memory usage of process"""
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()

            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
            }
        except:
            return {'rss_mb': 0, 'vms_mb': 0}

    def profile_inference(self,
                         model_path: str,
                         prompt: str = "Hello, world!",
                         n_predict: int = 100,
                         threads: int = 4,
                         gpu_layers: int = 0,
                         ctx_size: int = 2048) -> Optional[ProfileResult]:
        """Profile a single inference run"""

        cmd = [
            str(self.llama_cli),
            "-m", model_path,
            "-p", prompt,
            "-n", str(n_predict),
            "-t", str(threads),
            "-c", str(ctx_size),
            "--no-display-prompt",
            "--log-disable"
        ]

        if gpu_layers > 0:
            cmd.extend(["-ngl", str(gpu_layers)])

        print(f"Running: {' '.join(cmd[:6])}...")

        try:
            # Run with resource monitoring
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)

            # Monitor memory usage
            peak_memory = 0
            cpu_samples = []

            while process.poll() is None:
                try:
                    proc_info = psutil.Process(process.pid)
                    mem = proc_info.memory_info().rss / (1024 * 1024)
                    cpu = proc_info.cpu_percent(interval=0.1)

                    peak_memory = max(peak_memory, mem)
                    cpu_samples.append(cpu)
                except:
                    pass

                time.sleep(0.1)

            stdout, stderr = process.communicate()
            elapsed = time.time() - start_time

            if process.returncode != 0:
                print(f"Error running inference: {stderr}")
                return None

            # Parse output
            output = stdout + stderr
            stats = self.parse_llama_output(output)

            # Calculate metrics
            prompt_tokens = int(stats.get('prompt_tokens', 0))
            generated_tokens = int(stats.get('generated_tokens', 0))
            prompt_eval_time = stats.get('prompt_eval_time', 0)
            token_eval_time = stats.get('token_eval_time', 0)
            total_time = stats.get('total_time', elapsed * 1000)

            tokens_per_sec = (generated_tokens / (token_eval_time / 1000.0)
                             if token_eval_time > 0 else 0)
            prompt_tps = (prompt_tokens / (prompt_eval_time / 1000.0)
                         if prompt_eval_time > 0 else 0)

            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

            result = ProfileResult(
                model=Path(model_path).name,
                threads=threads,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                prompt_eval_time_ms=prompt_eval_time,
                token_eval_time_ms=token_eval_time,
                total_time_ms=total_time,
                tokens_per_sec=tokens_per_sec,
                prompt_tokens_per_sec=prompt_tps,
                memory_rss_mb=peak_memory,
                memory_peak_mb=peak_memory,
                cpu_percent=avg_cpu
            )

            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            print(f"  Memory: {peak_memory:.1f} MB")
            print(f"  CPU: {avg_cpu:.1f}%")

            return result

        except Exception as e:
            print(f"Error: {e}")
            return None

    def profile_thread_scaling(self,
                               model_path: str,
                               thread_counts: List[int],
                               prompt: str = "Test prompt",
                               n_predict: int = 100) -> pd.DataFrame:
        """Profile performance across different thread counts"""

        print(f"\n{'='*60}")
        print("THREAD SCALING ANALYSIS")
        print(f"{'='*60}")

        results = []

        for threads in thread_counts:
            print(f"\nTesting with {threads} threads...")

            result = self.profile_inference(
                model_path=model_path,
                prompt=prompt,
                n_predict=n_predict,
                threads=threads
            )

            if result:
                results.append(asdict(result))

        df = pd.DataFrame(results)
        return df

    def profile_prompt_lengths(self,
                               model_path: str,
                               prompt_lengths: List[int],
                               threads: int = 4) -> pd.DataFrame:
        """Profile performance across different prompt lengths"""

        print(f"\n{'='*60}")
        print("PROMPT LENGTH ANALYSIS")
        print(f"{'='*60}")

        results = []
        base_prompt = "This is a test prompt. " * 20  # ~100 tokens

        for length in prompt_lengths:
            # Generate prompt of specified token length (approximate)
            num_repeats = length // 20
            prompt = (base_prompt * num_repeats)[:length * 5]  # Rough estimate

            print(f"\nTesting with ~{length} token prompt...")

            result = self.profile_inference(
                model_path=model_path,
                prompt=prompt,
                n_predict=100,
                threads=threads
            )

            if result:
                results.append(asdict(result))

        df = pd.DataFrame(results)
        return df

    def visualize_thread_scaling(self, df: pd.DataFrame, output_path: str = "thread_scaling.png"):
        """Visualize thread scaling performance"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Tokens/sec vs threads
        ax = axes[0, 0]
        ax.plot(df['threads'], df['tokens_per_sec'], marker='o', linewidth=2)
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Tokens per Second')
        ax.set_title('Inference Speed vs Thread Count')
        ax.grid(alpha=0.3)

        # 2. Speedup vs threads
        ax = axes[0, 1]
        baseline_tps = df.iloc[0]['tokens_per_sec']
        speedup = df['tokens_per_sec'] / baseline_tps
        ax.plot(df['threads'], speedup, marker='o', linewidth=2, label='Actual')
        ax.plot(df['threads'], df['threads'] / df['threads'].min(),
                '--', alpha=0.5, label='Ideal')
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Speedup')
        ax.set_title('Parallel Speedup')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Memory usage vs threads
        ax = axes[1, 0]
        ax.plot(df['threads'], df['memory_peak_mb'], marker='o', linewidth=2)
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage vs Thread Count')
        ax.grid(alpha=0.3)

        # 4. Efficiency (tokens/sec per thread)
        ax = axes[1, 1]
        efficiency = df['tokens_per_sec'] / df['threads']
        ax.plot(df['threads'], efficiency, marker='o', linewidth=2)
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Tokens/sec per Thread')
        ax.set_title('Threading Efficiency')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved visualization to {output_path}")

    def generate_report(self, df: pd.DataFrame, output_path: str = "profile_report.md"):
        """Generate markdown report"""

        with open(output_path, 'w') as f:
            f.write("# Performance Profile Report\n\n")

            if 'model' in df.columns:
                f.write(f"**Model**: {df.iloc[0]['model']}\n\n")

            f.write("## Summary Statistics\n\n")
            f.write(df.describe().to_markdown())
            f.write("\n\n")

            f.write("## Detailed Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if 'threads' in df.columns and len(df) > 1:
                best_tps = df.nlargest(1, 'tokens_per_sec').iloc[0]
                f.write(f"**Optimal Thread Count**: {int(best_tps['threads'])} ")
                f.write(f"({best_tps['tokens_per_sec']:.2f} tokens/sec)\n\n")

                # Calculate efficiency
                df_copy = df.copy()
                df_copy['efficiency'] = df_copy['tokens_per_sec'] / df_copy['threads']
                best_eff = df_copy.nlargest(1, 'efficiency').iloc[0]
                f.write(f"**Most Efficient**: {int(best_eff['threads'])} threads ")
                f.write(f"({best_eff['efficiency']:.2f} tokens/sec per thread)\n\n")

        print(f"\n📝 Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile llama.cpp performance")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--profile", choices=["threads", "prompts", "all"], default="all",
                       help="Profiling mode")
    parser.add_argument("--threads", default="1,2,4,8",
                       help="Thread counts to test (comma-separated)")
    parser.add_argument("--prompt-lengths", default="50,100,200,500",
                       help="Prompt lengths to test (comma-separated)")
    parser.add_argument("--prompt", default="Hello, world! This is a test prompt.",
                       help="Test prompt")
    parser.add_argument("--n-predict", type=int, default=100,
                       help="Number of tokens to generate")
    parser.add_argument("--gpu-layers", type=int, default=0,
                       help="Number of GPU layers")
    parser.add_argument("--output-json", default="profile_results.json",
                       help="Output JSON path")
    parser.add_argument("--output-viz", default="profile_visualization.png",
                       help="Output visualization path")
    parser.add_argument("--output-report", default="profile_report.md",
                       help="Output report path")
    parser.add_argument("--llama-cpp-dir", default="../../../",
                       help="Path to llama.cpp directory")

    args = parser.parse_args()

    # Create profiler
    try:
        profiler = PerformanceProfiler(args.llama_cpp_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Parse thread counts
    thread_counts = [int(x) for x in args.threads.split(',')]
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(',')]

    # Run profiling
    if args.profile in ["threads", "all"]:
        df_threads = profiler.profile_thread_scaling(
            model_path=args.model,
            thread_counts=thread_counts,
            prompt=args.prompt,
            n_predict=args.n_predict
        )

        if not df_threads.empty:
            print("\n" + "="*60)
            print("THREAD SCALING RESULTS")
            print("="*60)
            print(df_threads.to_string(index=False))

            # Save results
            df_threads.to_json(args.output_json, orient='records', indent=2)
            print(f"\n💾 Saved results to {args.output_json}")

            # Generate visualization
            profiler.visualize_thread_scaling(df_threads, args.output_viz)

            # Generate report
            profiler.generate_report(df_threads, args.output_report)

    if args.profile in ["prompts", "all"]:
        df_prompts = profiler.profile_prompt_lengths(
            model_path=args.model,
            prompt_lengths=prompt_lengths
        )

        if not df_prompts.empty:
            print("\n" + "="*60)
            print("PROMPT LENGTH RESULTS")
            print("="*60)
            print(df_prompts.to_string(index=False))

    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    main()
