#!/usr/bin/env python3
"""
Benchmark Automation System

Automated benchmarking suite for continuous performance monitoring.
Can be integrated into CI/CD pipelines.

Usage:
    python benchmark_automation.py --config benchmark_config.yaml
    python benchmark_automation.py --models "*.gguf" --suite standard
"""

import argparse
import json
import subprocess
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkSuite:
    """Automated benchmark suite"""

    def __init__(self, config_path: Optional[str] = None):
        self.results = []
        self.start_time = datetime.now()

        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.default_config()

    @staticmethod
    def default_config() -> Dict:
        """Default benchmark configuration"""
        return {
            'llama_cpp_dir': '../../../',
            'benchmarks': {
                'perplexity': {
                    'enabled': True,
                    'test_file': 'wikitext-2-test.txt',
                    'context_size': 512
                },
                'performance': {
                    'enabled': True,
                    'prompt': 'Hello, world! This is a benchmark test.',
                    'n_predict': 128,
                    'trials': 5,
                    'thread_counts': [1, 2, 4, 8]
                },
                'memory': {
                    'enabled': True
                }
            },
            'thresholds': {
                'perplexity_increase_percent': 5.0,
                'performance_decrease_percent': 10.0,
                'memory_increase_percent': 15.0
            },
            'output': {
                'json': 'benchmark_results_{timestamp}.json',
                'csv': 'benchmark_results_{timestamp}.csv',
                'plots': True,
                'plots_dir': 'benchmark_plots'
            }
        }

    def benchmark_perplexity(self, model_path: str) -> Dict[str, Any]:
        """Benchmark model perplexity"""
        print(f"\n📊 Benchmarking perplexity for {Path(model_path).name}...")

        llama_perplexity = Path(self.config['llama_cpp_dir']) / 'llama-perplexity'
        test_file = self.config['benchmarks']['perplexity']['test_file']

        if not Path(test_file).exists():
            print(f"⚠️  Test file not found: {test_file}, skipping")
            return {'perplexity': -1, 'error': 'Test file not found'}

        cmd = [
            str(llama_perplexity),
            '-m', model_path,
            '-f', test_file,
            '--perplexity'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # Parse perplexity from output
            for line in result.stdout.split('\n'):
                if 'perplexity' in line.lower() and ':' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        ppl = float(parts[1].split()[0])
                        print(f"  Perplexity: {ppl:.2f}")
                        return {'perplexity': ppl}

            print("⚠️  Could not parse perplexity")
            return {'perplexity': -1, 'error': 'Parse error'}

        except subprocess.TimeoutExpired:
            print("❌ Perplexity benchmark timeout")
            return {'perplexity': -1, 'error': 'Timeout'}
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'perplexity': -1, 'error': str(e)}

    def benchmark_performance(self, model_path: str) -> Dict[str, Any]:
        """Benchmark inference performance"""
        print(f"\n⚡ Benchmarking performance for {Path(model_path).name}...")

        llama_cli = Path(self.config['llama_cpp_dir']) / 'llama-cli'
        perf_config = self.config['benchmarks']['performance']

        results = {
            'tokens_per_sec': [],
            'thread_results': {}
        }

        for threads in perf_config['thread_counts']:
            print(f"  Testing with {threads} threads...")

            thread_times = []
            for trial in range(perf_config['trials']):
                cmd = [
                    str(llama_cli),
                    '-m', model_path,
                    '-p', perf_config['prompt'],
                    '-n', str(perf_config['n_predict']),
                    '-t', str(threads),
                    '--no-display-prompt',
                    '--log-disable'
                ]

                try:
                    start = time.time()
                    subprocess.run(cmd, capture_output=True, timeout=120)
                    elapsed = time.time() - start

                    tps = perf_config['n_predict'] / elapsed
                    thread_times.append(tps)

                except Exception as e:
                    print(f"    Trial {trial + 1} failed: {e}")

            if thread_times:
                avg_tps = sum(thread_times) / len(thread_times)
                results['thread_results'][threads] = {
                    'avg_tps': avg_tps,
                    'min_tps': min(thread_times),
                    'max_tps': max(thread_times)
                }
                results['tokens_per_sec'].append(avg_tps)
                print(f"    Average: {avg_tps:.2f} tokens/sec")

        # Overall best performance
        if results['tokens_per_sec']:
            results['best_tps'] = max(results['tokens_per_sec'])
            results['avg_tps'] = sum(results['tokens_per_sec']) / len(results['tokens_per_sec'])
        else:
            results['best_tps'] = -1
            results['avg_tps'] = -1

        return results

    def benchmark_memory(self, model_path: str) -> Dict[str, Any]:
        """Benchmark memory usage"""
        print(f"\n💾 Benchmarking memory for {Path(model_path).name}...")

        model_size = Path(model_path).stat().st_size
        model_size_gb = model_size / (1024 ** 3)

        return {
            'model_size_bytes': model_size,
            'model_size_gb': model_size_gb
        }

    def benchmark_model(self, model_path: str) -> Dict[str, Any]:
        """Run complete benchmark suite on model"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {Path(model_path).name}")
        print(f"{'='*60}")

        result = {
            'model': Path(model_path).name,
            'model_path': str(model_path),
            'timestamp': datetime.now().isoformat()
        }

        # Run benchmarks
        if self.config['benchmarks']['perplexity']['enabled']:
            result.update(self.benchmark_perplexity(model_path))

        if self.config['benchmarks']['performance']['enabled']:
            perf = self.benchmark_performance(model_path)
            result.update(perf)

        if self.config['benchmarks']['memory']['enabled']:
            mem = self.benchmark_memory(model_path)
            result.update(mem)

        self.results.append(result)
        return result

    def benchmark_models(self, model_paths: List[str]) -> pd.DataFrame:
        """Benchmark multiple models"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUITE")
        print(f"{'='*60}")
        print(f"Models: {len(model_paths)}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        for model_path in model_paths:
            self.benchmark_model(model_path)

        end_time = datetime.now()
        elapsed = end_time - self.start_time

        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {elapsed}")

        return pd.DataFrame(self.results)

    def compare_with_baseline(self, current: pd.DataFrame,
                              baseline_path: str) -> Dict[str, Any]:
        """Compare results with baseline"""
        print(f"\n{'='*60}")
        print(f"BASELINE COMPARISON")
        print(f"{'='*60}")

        try:
            baseline = pd.read_json(baseline_path)
        except Exception as e:
            print(f"⚠️  Could not load baseline: {e}")
            return {'status': 'no_baseline'}

        comparison = {'regressions': [], 'improvements': [], 'status': 'ok'}
        thresholds = self.config['thresholds']

        # Compare each model
        for _, curr_row in current.iterrows():
            model_name = curr_row['model']
            baseline_row = baseline[baseline['model'] == model_name]

            if baseline_row.empty:
                print(f"⚠️  No baseline for {model_name}")
                continue

            baseline_row = baseline_row.iloc[0]

            # Check perplexity
            if 'perplexity' in curr_row and 'perplexity' in baseline_row:
                curr_ppl = curr_row['perplexity']
                base_ppl = baseline_row['perplexity']

                if curr_ppl > 0 and base_ppl > 0:
                    ppl_change = ((curr_ppl - base_ppl) / base_ppl) * 100

                    if ppl_change > thresholds['perplexity_increase_percent']:
                        comparison['regressions'].append({
                            'model': model_name,
                            'metric': 'perplexity',
                            'baseline': base_ppl,
                            'current': curr_ppl,
                            'change_percent': ppl_change
                        })
                        print(f"❌ REGRESSION: {model_name} perplexity +{ppl_change:.1f}%")

            # Check performance
            if 'best_tps' in curr_row and 'best_tps' in baseline_row:
                curr_tps = curr_row['best_tps']
                base_tps = baseline_row['best_tps']

                if curr_tps > 0 and base_tps > 0:
                    tps_change = ((curr_tps - base_tps) / base_tps) * 100

                    if tps_change < -thresholds['performance_decrease_percent']:
                        comparison['regressions'].append({
                            'model': model_name,
                            'metric': 'performance',
                            'baseline': base_tps,
                            'current': curr_tps,
                            'change_percent': tps_change
                        })
                        print(f"❌ REGRESSION: {model_name} performance {tps_change:.1f}%")
                    elif tps_change > 5:
                        comparison['improvements'].append({
                            'model': model_name,
                            'metric': 'performance',
                            'change_percent': tps_change
                        })
                        print(f"✅ IMPROVEMENT: {model_name} performance +{tps_change:.1f}%")

        if comparison['regressions']:
            comparison['status'] = 'failed'
            print(f"\n❌ {len(comparison['regressions'])} regression(s) detected")
        else:
            print(f"\n✅ No regressions detected")

        if comparison['improvements']:
            print(f"✨ {len(comparison['improvements'])} improvement(s)")

        return comparison

    def generate_plots(self, df: pd.DataFrame, output_dir: str = "benchmark_plots"):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Perplexity comparison
        if 'perplexity' in df.columns:
            df_ppl = df[df['perplexity'] > 0].sort_values('perplexity')
            if not df_ppl.empty:
                plt.figure(figsize=(10, 6))
                plt.barh(df_ppl['model'], df_ppl['perplexity'])
                plt.xlabel('Perplexity (lower is better)')
                plt.title('Model Perplexity Comparison')
                plt.tight_layout()
                plt.savefig(output_dir / 'perplexity_comparison.png', dpi=150)
                plt.close()

        # 2. Performance comparison
        if 'best_tps' in df.columns:
            df_perf = df[df['best_tps'] > 0].sort_values('best_tps', ascending=False)
            if not df_perf.empty:
                plt.figure(figsize=(10, 6))
                plt.barh(df_perf['model'], df_perf['best_tps'])
                plt.xlabel('Tokens per Second')
                plt.title('Model Performance Comparison')
                plt.tight_layout()
                plt.savefig(output_dir / 'performance_comparison.png', dpi=150)
                plt.close()

        # 3. Size vs Performance
        if 'model_size_gb' in df.columns and 'best_tps' in df.columns:
            df_valid = df[(df['model_size_gb'] > 0) & (df['best_tps'] > 0)]
            if not df_valid.empty:
                plt.figure(figsize=(10, 6))
                plt.scatter(df_valid['model_size_gb'], df_valid['best_tps'], s=100)
                for _, row in df_valid.iterrows():
                    plt.annotate(row['model'], (row['model_size_gb'], row['best_tps']),
                               fontsize=8, ha='right')
                plt.xlabel('Model Size (GB)')
                plt.ylabel('Performance (tokens/sec)')
                plt.title('Size vs Performance Trade-off')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'size_vs_performance.png', dpi=150)
                plt.close()

        print(f"\n📊 Saved plots to {output_dir}/")

    def save_results(self, df: pd.DataFrame):
        """Save benchmark results"""
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')

        # JSON
        json_path = self.config['output']['json'].format(timestamp=timestamp)
        df.to_json(json_path, orient='records', indent=2)
        print(f"\n💾 Saved JSON results to {json_path}")

        # CSV
        csv_path = self.config['output']['csv'].format(timestamp=timestamp)
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV results to {csv_path}")

        # Plots
        if self.config['output']['plots']:
            self.generate_plots(df, self.config['output']['plots_dir'])


def main():
    parser = argparse.ArgumentParser(description="Automated benchmark suite")
    parser.add_argument("--config", help="Configuration YAML file")
    parser.add_argument("--models", nargs="+", help="Model paths to benchmark")
    parser.add_argument("--model-pattern", help="Glob pattern for models (e.g., '*.gguf')")
    parser.add_argument("--baseline", help="Baseline results to compare against")
    parser.add_argument("--suite", choices=["quick", "standard", "comprehensive"],
                       default="standard", help="Benchmark suite preset")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory")

    args = parser.parse_args()

    # Create benchmark suite
    suite = BenchmarkSuite(args.config)

    # Determine models to benchmark
    if args.models:
        model_paths = args.models
    elif args.model_pattern:
        model_paths = [str(p) for p in Path('.').glob(args.model_pattern)]
    else:
        print("❌ Must specify --models or --model-pattern")
        return 1

    if not model_paths:
        print("❌ No models found")
        return 1

    print(f"Found {len(model_paths)} model(s) to benchmark")

    # Run benchmarks
    df = suite.benchmark_models(model_paths)

    # Save results
    suite.save_results(df)

    # Compare with baseline if provided
    if args.baseline:
        comparison = suite.compare_with_baseline(df, args.baseline)

        # Exit with error if regressions detected
        if comparison['status'] == 'failed':
            print("\n❌ Benchmark failed due to regressions")
            return 1

    print("\n✅ Benchmark suite complete!")
    return 0


if __name__ == "__main__":
    exit(main())
