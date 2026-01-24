#!/usr/bin/env python3
"""
Aggregate benchmark results from all tiers into a unified analysis dataset.

Usage:
    python3 aggregate_results.py --results-dir ../../benchmark_results --output aggregated.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import csv

# Quant type priority for analysis
QUANT_PRIORITY = {
    'Q4_0': 0, 'q4_0': 0,
    'Q8_0': 0, 'q8_0': 0,
    'Q6_K': 1, 'q6_k': 1,
    'Q4_K': 1, 'q4_k': 1,
    'MXFP4': 2, 'mxfp4': 2,
    'Q2_K': 2, 'q2_k': 2,
    'Q3_K': 2, 'q3_k': 2,
    'Q5_K': 2, 'q5_k': 2,
}

# Kernel tier classification
KERNEL_TIERS = {
    # Tier 0: Reference
    'memory_bandwidth': 0,
    'onednn_fp16_gemm': 0,
    'onednn_int8_gemm': 0,
    'roofline_compute': 0,

    # Tier 1: Memory-bound (batch=1-4)
    'mmvq_aos_baseline': 1,
    'mmvq_soa_baseline': 1,
    'mmvq_coalesced': 1,
    'mmvq_slm_cached': 1,
    'mmvq_prefetch': 1,
    'mmvq_wide_load': 1,
    'mmvq_esimd_block_load': 1,
    'mmvq_esimd_slm': 1,

    # Tier 2: Transitional (batch=8-64)
    'mmvq_xmx_tile_8x8': 2,
    'mmvq_xmx_tile_16x16': 2,
    'mmvq_xmx_cf2': 2,
    'mmvq_xmx_cf4': 2,
    'mmvq_esimd_xmx': 2,

    # Tier 3: Compute-bound (batch=64+)
    'mmvq_xmx_tile_64x64': 3,
    'mmvq_xmx_persistent': 3,
    'mmvq_esimd_xmx_large': 3,

    # Tier 4: Experimental
    'hybrid_adaptive': 4,
    'xmx_fused': 4,
    'coalesced_xmx_aligned': 4,
    'esimd_hybrid': 4,
    'esimd_cooperative': 4,
    'q4_0_specialized': 4,
    'q6_k_specialized': 4,
    'mxfp4_native': 4,
}


def classify_regime(batch: int) -> str:
    """Classify batch size into performance regime."""
    if batch <= 4:
        return 'memory_bound'
    elif batch <= 64:
        return 'transitional'
    else:
        return 'compute_bound'


def normalize_quant(quant: str) -> str:
    """Normalize quant type to uppercase."""
    return quant.upper()


def normalize_kernel(kernel: str) -> str:
    """Normalize kernel name."""
    # Remove tier prefixes if present
    for prefix in ['tier1_', 'tier2_', 'tier3_', 'tier4_']:
        if kernel.startswith(prefix):
            kernel = kernel[len(prefix):]
    return kernel.lower()


def parse_jsonl_file(filepath: Path) -> List[Dict]:
    """Parse a JSONL file, skipping non-JSON lines."""
    results = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('[') or line.startswith('#'):
                    continue
                # Skip lines that don't look like JSON objects
                if not line.startswith('{'):
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'kernel' in data:
                        results.append(data)
                except json.JSONDecodeError:
                    pass  # Skip malformed lines
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    return results


def find_jsonl_files(results_dir: Path) -> List[Path]:
    """Find all JSONL files in the results directory."""
    jsonl_files = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith('.jsonl'):
                jsonl_files.append(Path(root) / f)
    return sorted(jsonl_files)


def aggregate_results(results_dir: Path) -> List[Dict]:
    """Aggregate all benchmark results."""
    all_results = []
    jsonl_files = find_jsonl_files(results_dir)

    print(f"Found {len(jsonl_files)} JSONL files", file=sys.stderr)

    for filepath in jsonl_files:
        results = parse_jsonl_file(filepath)
        for r in results:
            # Normalize and enrich data
            kernel = normalize_kernel(r.get('kernel', 'unknown'))
            quant = normalize_quant(r.get('quant', 'UNKNOWN'))
            batch = r.get('batch', 1)

            enriched = {
                'kernel': kernel,
                'quant': quant,
                'batch': batch,
                'dim_m': r.get('dim_m', 0),
                'dim_n': r.get('dim_n', 0),
                'dim_k': r.get('dim_k', 0),
                'layout': r.get('layout', 'AOS'),
                'throughput_tps': r.get('throughput_tps', 0.0),
                'latency_us': r.get('latency_us', 0.0),
                'bandwidth_gbps': r.get('bandwidth_gbps', 0.0),
                'xmx_util_pct': r.get('xmx_util_pct', 0.0),
                'variance_pct': r.get('variance_pct', 0.0),
                'regime': classify_regime(batch),
                'tier': KERNEL_TIERS.get(kernel, -1),
                'source_file': str(filepath.name),
            }
            all_results.append(enriched)

    return all_results


def find_winners(results: List[Dict]) -> Dict:
    """Find the winning kernel for each (quant, batch, dim) combination."""
    # Group by (quant, batch, dim_m)
    groups = {}
    for r in results:
        key = (r['quant'], r['batch'], r['dim_m'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Find winner in each group by throughput
    winners = {}
    for key, group in groups.items():
        best = max(group, key=lambda x: x['throughput_tps'])
        winners[key] = best['kernel']

    return winners


def write_csv(results: List[Dict], output_path: Path, winners: Dict):
    """Write results to CSV with winner annotation."""
    if not results:
        print("No results to write", file=sys.stderr)
        return

    fieldnames = [
        'kernel', 'quant', 'batch', 'dim_m', 'dim_n', 'dim_k', 'layout',
        'throughput_tps', 'latency_us', 'bandwidth_gbps', 'xmx_util_pct',
        'variance_pct', 'regime', 'tier', 'is_winner', 'source_file'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            key = (r['quant'], r['batch'], r['dim_m'])
            r['is_winner'] = winners.get(key) == r['kernel']
            writer.writerow(r)

    print(f"Wrote {len(results)} rows to {output_path}", file=sys.stderr)


def print_summary(results: List[Dict], winners: Dict):
    """Print summary of aggregated results."""
    print("\n=== Aggregation Summary ===\n")

    # Count by tier
    tier_counts = {}
    for r in results:
        tier = r['tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print("Results by tier:")
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier}: {tier_counts[tier]} results")

    # Count by quant
    quant_counts = {}
    for r in results:
        q = r['quant']
        quant_counts[q] = quant_counts.get(q, 0) + 1

    print("\nResults by quant type:")
    for q in sorted(quant_counts.keys()):
        print(f"  {q}: {quant_counts[q]} results")

    # Count by regime
    regime_counts = {}
    for r in results:
        regime = r['regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print("\nResults by regime:")
    for regime in sorted(regime_counts.keys()):
        print(f"  {regime}: {regime_counts[regime]} results")

    # Winners by quant and batch
    print("\n=== Winners by (quant, batch) ===\n")
    winner_summary = {}
    for (quant, batch, dim), kernel in winners.items():
        key = (quant, batch)
        if key not in winner_summary:
            winner_summary[key] = {}
        winner_summary[key][kernel] = winner_summary[key].get(kernel, 0) + 1

    for (quant, batch) in sorted(winner_summary.keys()):
        kernels = winner_summary[(quant, batch)]
        top_kernel = max(kernels.items(), key=lambda x: x[1])
        print(f"  {quant} batch={batch}: {top_kernel[0]} ({top_kernel[1]} dims)")


def main():
    parser = argparse.ArgumentParser(description='Aggregate benchmark results')
    parser.add_argument('--results-dir', type=str,
                        default='../../benchmark_results',
                        help='Directory containing benchmark results')
    parser.add_argument('--output', type=str,
                        default='aggregated.csv',
                        help='Output CSV file')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary, do not write CSV')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    results = aggregate_results(results_dir)
    winners = find_winners(results)

    print_summary(results, winners)

    if not args.summary_only:
        output_path = Path(args.output)
        write_csv(results, output_path, winners)


if __name__ == '__main__':
    main()
