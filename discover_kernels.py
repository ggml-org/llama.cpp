#!/usr/bin/env python3
import argparse, subprocess, re, csv, os
from pathlib import Path
import matplotlib.pyplot as plt

LLAMA_BENCH_SCRIPT = "./SCRIPT_llama_bench.sh"
GFX906_ENV = {
    'HSA_OVERRIDE_GFX_VERSION': '9.0.6', 'HIP_VISIBLE_DEVICES': '0',
    'CUDA_VISIBLE_DEVICES': '0', 'ROCR_VISIBLE_DEVICES': '0',
    'GGML_BACKEND_HIP': '1', 'HCC_AMDGPU_TARGET': 'gfx906',
    'GGML_CUDA_DISABLE_GRAPHS': '1',
}
COLORS = {
    'bg': '#0d1117', 'text': '#e6edf3', 'grid': '#30363d', 'accent': '#58a6ff',
    'gradient': ['#00d4ff', '#00b4d8', '#0096c7', '#0077b6', '#023e8a',
                 '#7b2cbf', '#9d4edd', '#c77dff', '#e0aaff'],
}

def run_rocprofv3(command: str, output_dir: Path) -> Path:
    env = os.environ.copy()
    env.update(GFX906_ENV)
    cmd = ["rocprofv3", "--kernel-trace", "--stats", "-d", str(output_dir),
           "-o", "kernels", "-f", "csv", "--"] + command.split()
    print(f"Running: {' '.join(cmd)}\nThis may take several minutes...")
    subprocess.run(cmd, env=env, timeout=1200)
    return next(output_dir.glob("*kernel_stats.csv"))

def parse_stats_csv(stats_file: Path) -> list[dict]:
    with open(stats_file) as f:
        return [{'name': r['Name'], 'calls': int(r.get('Calls', 1)),
                 'total_ns': int(r.get('TotalDurationNs', 0))}
                for r in csv.DictReader(f) if r.get('Name')]

def analyze_kernels(kernels: list[dict], threshold: float = 1.0) -> dict:
    real = [k for k in kernels if not k['name'].startswith('<')]
    total = sum(k['total_ns'] for k in real)
    for k in real:
        k['pct'] = (k['total_ns'] / total * 100) if total else 0
    by_time = sorted(real, key=lambda x: x['total_ns'], reverse=True)
    return {'kernels': by_time, 'total_time_ns': total,
            'hot_kernels': [k for k in by_time if k['pct'] >= threshold]}

def short_name(name: str) -> str:
    if name.startswith('Cijk_'):
        mt, isa = re.search(r'MT(\d+x\d+x\d+)', name), re.search(r'ISA(\d+)', name)
        return f"rocBLAS_gemm<MT{mt.group(1) if mt else '?'}_ISA{isa.group(1) if isa else '?'}>"
    return name

def plot_chart(analysis: dict, output_dir: Path, top_n: int = 15):
    kernels = analysis['hot_kernels'][:top_n]
    if not kernels:
        return
    names = [short_name(k['name'])[:40] for k in kernels]
    pcts = [k['pct'] for k in kernels]
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, max(7, len(kernels) * 0.5)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    n = len(kernels)
    colors = [COLORS['gradient'][min(i * len(COLORS['gradient']) // max(n, 1),
              len(COLORS['gradient']) - 1)] for i in range(n)]
    bars = ax.barh(range(n), pcts, color=colors, height=0.7, alpha=0.95)
    for bar, c in zip(bars, colors):
        ax.barh(bar.get_y() + bar.get_height()/2, bar.get_width(),
                height=0.9, color=c, alpha=0.15, zorder=0)
    for bar, pct in zip(bars, pcts):
        inside = bar.get_width() > max(pcts) * 0.15
        ax.text(bar.get_width() - 0.5 if inside else bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center',
                ha='right' if inside else 'left',
                color='#000' if inside else COLORS['text'],
                fontsize=11, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=10, color=COLORS['text'], fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=10)
    ax.set_xlabel('GPU Time %', fontsize=12, color=COLORS['text'], labelpad=10)
    ax.set_xlim(0, max(pcts) * 1.12)
    ax.invert_yaxis()
    ax.tick_params(axis='x', colors=COLORS['text'], labelsize=10)
    ax.grid(axis='x', color=COLORS['grid'], linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('GPU Kernel Time Distribution\n', fontsize=16,
                 fontweight='bold', color=COLORS['text'], pad=20)
    ax.text(0.5, 1.02, f'Total: {analysis["total_time_ns"]/1e6:,.0f} ms  |  '
            f'{len(analysis["hot_kernels"])} hot kernels',
            transform=ax.transAxes, ha='center', fontsize=11,
            color=COLORS['accent'], style='italic')
    plt.tight_layout()
    plt.savefig(output_dir / 'kernel_pareto.png', dpi=150,
                facecolor=COLORS['bg'], bbox_inches='tight', pad_inches=0.3)
    plt.close()

def print_report(analysis: dict, top_n: int = 20):
    total_ms = analysis['total_time_ns'] / 1e6
    print(f"\n{'='*80}\nKERNEL DISCOVERY REPORT\n{'='*80}")
    print(f"Total GPU time: {total_ms:,.2f} ms | Kernels: {len(analysis['kernels'])} "
          f"| Hot (>1%): {len(analysis['hot_kernels'])}")
    print(f"\n{'#':<4} {'Kernel':<50} {'Calls':>8} {'Time(ms)':>10} {'%':>6}\n" + "-"*80)
    for i, k in enumerate(analysis['kernels'][:top_n], 1):
        marker = "* " if k['pct'] >= 5.0 else "  "
        print(f"{i:<4}{marker}{short_name(k['name']):<48} {k['calls']:>8} "
              f"{k['total_ns']/1e6:>10.2f} {k['pct']:>5.1f}%")
    if len(analysis['kernels']) > top_n:
        print(f"    ... and {len(analysis['kernels']) - top_n} more kernels")
    print("-"*80)

def save_results(analysis: dict, output_dir: Path):
    with open(output_dir / 'kernels.csv', 'w') as f:
        f.write("rank,name,short_name,calls,total_ns,pct\n")
        for i, k in enumerate(analysis['kernels'], 1):
            f.write(f"{i},{k['name']},{short_name(k['name'])},"
                    f"{k['calls']},{k['total_ns']},{k['pct']:.2f}\n")
    with open(output_dir / 'hot_kernels.txt', 'w') as f:
        for k in analysis['hot_kernels']:
            f.write(f"{short_name(k['name'])}  # {k['pct']:.1f}%\n")
    plot_chart(analysis, output_dir)
    print(f"Saved: {output_dir}/kernels.csv, hot_kernels.txt, kernel_pareto.png")

def main():
    p = argparse.ArgumentParser(description='Discover GPU kernels using rocprofv3')
    p.add_argument('-o', '--output', default='./discovery-results')
    p.add_argument('-t', '--threshold', type=float, default=1.0)
    p.add_argument('-n', '--top', type=int, default=20)
    p.add_argument('-e', '--existing', metavar='CSV')
    args = p.parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = Path(args.existing) if args.existing else run_rocprofv3(LLAMA_BENCH_SCRIPT, output_dir)
    analysis = analyze_kernels(parse_stats_csv(stats_file), args.threshold)
    print_report(analysis, args.top)
    save_results(analysis, output_dir)

if __name__ == '__main__':
    main()
