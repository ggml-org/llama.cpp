#!/usr/bin/env python3
"""
Find kernel crossover points from aggregated benchmark data.

Identifies batch sizes where one kernel type starts outperforming another.

Usage:
    python3 find_crossovers.py --input aggregated.csv --output crossovers.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Kernel categories for crossover analysis
KERNEL_CATEGORIES = {
    # Memory-bound kernels (Tier 1)
    'mmvq_coalesced': 'mmvq',
    'mmvq_soa_baseline': 'mmvq',
    'mmvq_aos_baseline': 'mmvq',
    'mmvq_slm_cached': 'mmvq',
    'mmvq_prefetch': 'mmvq',
    'mmvq_wide_load': 'mmvq',
    'mmvq_esimd_block_load': 'mmvq',
    'mmvq_esimd_slm': 'mmvq',

    # XMX small tile (Tier 2)
    'mmvq_xmx_tile_8x8': 'xmx_small',
    'mmvq_xmx_tile_16x16': 'xmx_small',
    'mmvq_xmx_tile_soa': 'xmx_small',
    'mmvq_xmx_cf2': 'xmx_small',
    'mmvq_xmx_cf2_soa': 'xmx_small',
    'mmvq_xmx_cf4': 'xmx_small',
    'mmvq_xmx_cf4_soa': 'xmx_small',
    'mmvq_xmx_double_buffer': 'xmx_small',
    'mmvq_xmx_aos_direct': 'xmx_small',
    'mmvq_esimd_v2': 'xmx_small',
    'mmvq_esimd_xmx': 'xmx_small',

    # XMX large tile (Tier 3)
    'mmvq_xmx_tile_64x64': 'xmx_large',
    'mmvq_xmx_persistent': 'xmx_large',
    'mmvq_xmx_register_accum': 'xmx_large',
    'mmvq_xmx_multi_wg': 'xmx_large',
    'mmvq_esimd_xmx_large': 'xmx_large',

    # MMQ kernels (fallback for large batch)
    'mmq_aos': 'mmq',
    'mmq_soa': 'mmq',
    'mmq_coalesced': 'mmq',
}


def load_aggregated_data(filepath: Path) -> List[Dict]:
    """Load aggregated CSV data."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['batch'] = int(row['batch'])
            row['throughput_tps'] = float(row['throughput_tps'])
            row['bandwidth_gbps'] = float(row['bandwidth_gbps'])
            row['dim_m'] = int(row['dim_m'])
            results.append(row)
    return results


def categorize_kernel(kernel: str) -> str:
    """Get the category for a kernel."""
    return KERNEL_CATEGORIES.get(kernel, 'other')


def find_best_per_category(data: List[Dict], quant: str, batch: int) -> Dict[str, Dict]:
    """Find best kernel in each category for a given quant/batch."""
    filtered = [r for r in data if r['quant'] == quant and r['batch'] == batch]

    category_best = {}
    for r in filtered:
        kernel = r['kernel']
        category = categorize_kernel(kernel)

        if category not in category_best or r['throughput_tps'] > category_best[category]['throughput_tps']:
            category_best[category] = {
                'kernel': kernel,
                'throughput_tps': r['throughput_tps'],
                'bandwidth_gbps': r['bandwidth_gbps'],
            }

    return category_best


def find_crossover_point(
    data: List[Dict],
    quant: str,
    from_category: str,
    to_category: str,
    batch_sizes: List[int]
) -> Optional[Tuple[int, float]]:
    """
    Find the batch size where to_category starts outperforming from_category.
    Returns (batch_size, confidence) or None if no crossover found.
    """
    crossover_batch = None
    crossover_margin = 0.0

    for batch in batch_sizes:
        category_best = find_best_per_category(data, quant, batch)

        if from_category not in category_best or to_category not in category_best:
            continue

        from_tps = category_best[from_category]['throughput_tps']
        to_tps = category_best[to_category]['throughput_tps']

        if to_tps > from_tps:
            margin = (to_tps - from_tps) / from_tps * 100 if from_tps > 0 else 0
            if crossover_batch is None:
                crossover_batch = batch
                crossover_margin = margin

    return (crossover_batch, crossover_margin) if crossover_batch else None


def analyze_crossovers(data: List[Dict]) -> Dict:
    """Analyze crossover points for all quant types."""
    # Get unique quant types and batch sizes
    quants = sorted(set(r['quant'] for r in data))
    batch_sizes = sorted(set(r['batch'] for r in data))

    crossovers = {}

    for quant in quants:
        crossovers[quant] = {}

        # Find MMVQ -> XMX small crossover
        result = find_crossover_point(data, quant, 'mmvq', 'xmx_small', batch_sizes)
        if result:
            crossovers[quant]['mmvq_to_xmx_small'] = {
                'batch': result[0],
                'margin_pct': round(result[1], 1),
            }

        # Find XMX small -> XMX large crossover
        result = find_crossover_point(data, quant, 'xmx_small', 'xmx_large', batch_sizes)
        if result:
            crossovers[quant]['xmx_small_to_large'] = {
                'batch': result[0],
                'margin_pct': round(result[1], 1),
            }

        # Find XMX -> MMQ crossover (for very large batch)
        result = find_crossover_point(data, quant, 'xmx_small', 'mmq', batch_sizes)
        if result:
            crossovers[quant]['xmx_to_mmq'] = {
                'batch': result[0],
                'margin_pct': round(result[1], 1),
            }

        # Determine best kernel per regime
        for batch in batch_sizes:
            category_best = find_best_per_category(data, quant, batch)
            if category_best:
                best_category = max(category_best.items(), key=lambda x: x[1]['throughput_tps'])
                crossovers[quant][f'best_at_batch_{batch}'] = {
                    'category': best_category[0],
                    'kernel': best_category[1]['kernel'],
                    'throughput_tps': round(best_category[1]['throughput_tps'], 2),
                }

    return crossovers


def generate_dispatch_thresholds(crossovers: Dict) -> Dict:
    """Generate dispatch threshold values from crossover analysis."""
    thresholds = {}

    for quant, data in crossovers.items():
        thresholds[quant] = {
            'MMVQ_MAX_BATCH': 4,  # Default
            'XMX_SMALL_MAX_BATCH': 32,  # Default
        }

        # Update from crossover data
        if 'mmvq_to_xmx_small' in data:
            # Use batch - 1 as the max for MMVQ (crossover is where XMX wins)
            thresholds[quant]['MMVQ_MAX_BATCH'] = max(1, data['mmvq_to_xmx_small']['batch'] - 1)

        if 'xmx_small_to_large' in data:
            thresholds[quant]['XMX_SMALL_MAX_BATCH'] = max(1, data['xmx_small_to_large']['batch'] - 1)

    return thresholds


def print_summary(crossovers: Dict, thresholds: Dict):
    """Print summary of crossover analysis."""
    print("\n=== Crossover Analysis Summary ===\n")

    for quant in sorted(crossovers.keys()):
        print(f"\n{quant}:")
        data = crossovers[quant]
        thresh = thresholds.get(quant, {})

        # Crossover points
        if 'mmvq_to_xmx_small' in data:
            c = data['mmvq_to_xmx_small']
            print(f"  MMVQ -> XMX: batch >= {c['batch']} (+{c['margin_pct']}%)")
        else:
            print(f"  MMVQ -> XMX: no crossover found")

        if 'xmx_small_to_large' in data:
            c = data['xmx_small_to_large']
            print(f"  XMX small -> large: batch >= {c['batch']} (+{c['margin_pct']}%)")

        if 'xmx_to_mmq' in data:
            c = data['xmx_to_mmq']
            print(f"  XMX -> MMQ: batch >= {c['batch']} (+{c['margin_pct']}%)")

        # Thresholds
        print(f"  Thresholds: MMVQ_MAX={thresh.get('MMVQ_MAX_BATCH', 'N/A')}, "
              f"XMX_SMALL_MAX={thresh.get('XMX_SMALL_MAX_BATCH', 'N/A')}")

    # Consolidated dispatch table
    print("\n\n=== Recommended Dispatch Thresholds ===\n")
    print("| Quant | MMVQ Max | XMX Small Max | Notes |")
    print("|-------|----------|---------------|-------|")

    for quant in sorted(thresholds.keys()):
        t = thresholds[quant]
        notes = []
        if 'mmvq_to_xmx_small' in crossovers.get(quant, {}):
            notes.append(f"XMX +{crossovers[quant]['mmvq_to_xmx_small']['margin_pct']}%")
        print(f"| {quant:5} | {t['MMVQ_MAX_BATCH']:8} | {t['XMX_SMALL_MAX_BATCH']:13} | {', '.join(notes)} |")


def main():
    parser = argparse.ArgumentParser(description='Find kernel crossover points')
    parser.add_argument('--input', type=str, default='aggregated.csv',
                        help='Input aggregated CSV file')
    parser.add_argument('--output', type=str, default='crossovers.json',
                        help='Output JSON file')
    parser.add_argument('--thresholds', type=str, default='thresholds.json',
                        help='Output thresholds JSON file')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        print("Run aggregate_results.py first to create the aggregated CSV.", file=sys.stderr)
        sys.exit(1)

    data = load_aggregated_data(input_path)
    print(f"Loaded {len(data)} results", file=sys.stderr)

    crossovers = analyze_crossovers(data)
    thresholds = generate_dispatch_thresholds(crossovers)

    print_summary(crossovers, thresholds)

    # Write outputs
    with open(args.output, 'w') as f:
        json.dump(crossovers, f, indent=2)
    print(f"\nWrote crossovers to {args.output}", file=sys.stderr)

    with open(args.thresholds, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Wrote thresholds to {args.thresholds}", file=sys.stderr)


if __name__ == '__main__':
    main()
