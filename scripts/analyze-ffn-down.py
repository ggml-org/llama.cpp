#!/usr/bin/env python3
"""Deep analysis of WHY ffn_down is hard to quantize.
Compares structural properties of all weight and activation tensors.
"""

import numpy as np
import struct
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_f32_tensor(name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        nrow, ncol = struct.unpack("qq", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.float32)
        assert len(data) == nrow * ncol, f"Expected {nrow * ncol}, got {len(data)}"
        return data.reshape(nrow, ncol)


def stats(label, arr):
    """Print comprehensive statistics for a flat array."""
    a = arr.ravel()
    print(f"  {label}:")  # noqa: NP100
    print(f"    shape={arr.shape}, n={len(a)}")  # noqa: NP100
    print(f"    mean={a.mean():.6f}, std={a.std():.6f}")  # noqa: NP100
    print(f"    min={a.min():.6f}, max={a.max():.6f}")  # noqa: NP100
    print(f"    median={np.median(a):.6f}")  # noqa: NP100
    print(  # noqa: NP100
        f"    |mean|/std = {abs(a.mean()) / (a.std() + 1e-10):.4f}  (offset-to-spread ratio)"
    )
    # Kurtosis (excess) - how heavy-tailed vs Gaussian
    kurt = np.mean(((a - a.mean()) / (a.std() + 1e-10)) ** 4) - 3.0
    # Skewness
    skew = np.mean(((a - a.mean()) / (a.std() + 1e-10)) ** 3)
    print(f"    skewness={skew:.4f}, excess_kurtosis={kurt:.4f}")  # noqa: NP100
    # Percentile ranges
    pcts = np.percentile(a, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    print(  # noqa: NP100
        f"    percentiles: 0.1%={pcts[0]:.4f}, 1%={pcts[1]:.4f}, 5%={pcts[2]:.4f}, "
        f"25%={pcts[3]:.4f}, 50%={pcts[4]:.4f}, 75%={pcts[5]:.4f}, "
        f"95%={pcts[6]:.4f}, 99%={pcts[7]:.4f}, 99.9%={pcts[8]:.4f}"
    )
    # Sparsity
    near_zero = np.sum(np.abs(a) < 0.001 * a.std()) / len(a)
    print(f"    fraction |x| < 0.001*std: {near_zero:.4f}")  # noqa: NP100
    return {
        "mean": a.mean(),
        "std": a.std(),
        "skew": skew,
        "kurt": kurt,
        "min": a.min(),
        "max": a.max(),
    }


# ============================================================================
# 1. BASIC WEIGHT TENSOR COMPARISON
# ============================================================================
print("=" * 80)  # noqa: NP100
print("SECTION 1: WEIGHT TENSOR GLOBAL STATISTICS")  # noqa: NP100
print("=" * 80)  # noqa: NP100

tensors = {
    "ffn_gate": ("blk_0_ffn_gate_weight.f32bin", "9728x2560 (wide→narrow proj)"),
    "ffn_up": ("blk_0_ffn_up_weight.f32bin", "9728x2560 (wide→narrow proj)"),
    "ffn_down": ("blk_0_ffn_down_weight.f32bin", "2560x9728 (narrow→wide proj)"),
    "attn_q": ("blk_0_attn_q_weight.f32bin", "4096x2560"),
    "attn_k": ("blk_0_attn_k_weight.f32bin", "1024x2560"),
    "attn_v": ("blk_0_attn_v_weight.f32bin", "1024x2560"),
    "attn_out": ("blk_0_attn_output_weight.f32bin", "2560x4096"),
}

weight_data = {}
for name, (fname, desc) in tensors.items():
    try:
        W = load_f32_tensor(fname)
        print(f"\n{'─' * 70}")  # noqa: NP100
        print(f"  {name} [{desc}] — file: {fname}")  # noqa: NP100
        weight_data[name] = W
        stats(name, W)
    except Exception as e:
        print(f"  {name}: SKIP ({e})")  # noqa: NP100

# ============================================================================
# 2. ROW-LEVEL STATISTICS (each row is a neuron output)
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 2: ROW-LEVEL VARIABILITY (per-neuron weight statistics)")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  Each row of the weight matrix produces one output dimension.")  # noqa: NP100
print("  High row-to-row variability in mean/std means the quantizer")  # noqa: NP100
print("  must handle very different distributions across rows.\n")  # noqa: NP100

for name, W in weight_data.items():
    row_means = W.mean(axis=1)
    row_stds = W.std(axis=1)
    row_ranges = W.max(axis=1) - W.min(axis=1)

    print(f"\n  {name} ({W.shape[0]} rows × {W.shape[1]} cols):")  # noqa: NP100
    print(  # noqa: NP100
        f"    Row means:  mean={row_means.mean():.6f}, std={row_means.std():.6f}, "
        f"range=[{row_means.min():.6f}, {row_means.max():.6f}]"
    )
    print(  # noqa: NP100
        f"    Row stds:   mean={row_stds.mean():.6f}, std={row_stds.std():.6f}, "
        f"range=[{row_stds.min():.6f}, {row_stds.max():.6f}]"
    )
    print(f"    Row ranges: mean={row_ranges.mean():.6f}, std={row_ranges.std():.6f}")  # noqa: NP100
    print(  # noqa: NP100
        f"    RowMeans CV (std/mean): {row_means.std() / (abs(row_means.mean()) + 1e-10):.4f}"
    )
    print(f"    RowStds CV:  {row_stds.std() / (row_stds.mean() + 1e-10):.4f}")  # noqa: NP100

# ============================================================================
# 3. GROUP-LEVEL ANALYSIS (16-element groups, like Q2_K)
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 3: GROUP-LEVEL ANALYSIS (16-element groups)")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  Quantization works on 16-element groups. Key question:")  # noqa: NP100
print("  How much does each group need its own OFFSET (dmin)?\n")  # noqa: NP100

GS = 16

for name, W in weight_data.items():
    # Look at first 256 rows for speed
    nr = min(W.shape[0], 256)
    nc = W.shape[1]

    group_means = []
    group_stds = []
    group_ranges = []
    group_offsets = []  # |mean| / range — how important is the offset

    for r in range(nr):
        for g_start in range(0, nc, GS):
            g = W[r, g_start : g_start + GS]
            gm = g.mean()
            gs = g.std()
            gr = g.max() - g.min()
            gmin = g.min()

            group_means.append(gm)
            group_stds.append(gs)
            group_ranges.append(gr)
            # Offset importance: how large is the group mean relative to its range?
            # If this is high, offset (dmin) matters a lot
            if gr > 1e-10:
                group_offsets.append(abs(gm) / gr)
            else:
                group_offsets.append(0)

    gm = np.array(group_means)
    gs = np.array(group_stds)
    gr = np.array(group_ranges)
    go = np.array(group_offsets)

    print(f"\n  {name} ({len(group_means)} groups):")  # noqa: NP100
    print(  # noqa: NP100
        f"    Group mean:  mean={gm.mean():.6f}, std={gm.std():.6f}, "
        f"range=[{gm.min():.6f}, {gm.max():.6f}]"
    )
    print(f"    Group std:   mean={gs.mean():.6f}, std={gs.std():.6f}")  # noqa: NP100
    print(f"    Group range: mean={gr.mean():.6f}, std={gr.std():.6f}")  # noqa: NP100
    print("    *** OFFSET IMPORTANCE (|group_mean| / range) ***")  # noqa: NP100
    print(  # noqa: NP100
        f"        mean={go.mean():.4f}, median={np.median(go):.4f}, "
        f"p90={np.percentile(go, 90):.4f}, max={go.max():.4f}"
    )
    print(f"        fraction with offset > 0.1: {np.mean(go > 0.1):.3f}")  # noqa: NP100
    print(f"        fraction with offset > 0.2: {np.mean(go > 0.2):.3f}")  # noqa: NP100
    print(f"        fraction with offset > 0.3: {np.mean(go > 0.3):.3f}")  # noqa: NP100

    # How well does zeroing the min (Q2_K style, clamping min to 0) work?
    # vs keeping the actual min
    mse_no_offset = 0  # Assume uniform 4 levels [0,1,2,3] * scale
    mse_with_offset = 0  # Assume uniform 4 levels [0,1,2,3] * scale + offset

    for r in range(nr):
        for g_start in range(0, nc, GS):
            g = W[r, g_start : g_start + GS]
            gmin = g.min()
            gmax = g.max()
            gr = gmax - gmin
            if gr < 1e-10:
                continue

            # No offset: clamp min to 0, scale = max/3
            if gmin > 0:
                scale_no = gmax / 3.0
                min_no = 0
            else:
                scale_no = gmax / 3.0
                min_no = 0  # lose the negative offset
                # Actually use (gmax - 0)/3 but we're clamping gmin to 0

            # Better: use actual min/max
            scale_w = gr / 3.0
            min_w = gmin

            for val in g:
                # No offset quantization
                norm_no = val / (scale_no + 1e-10)
                idx_no = max(0, min(3, int(round(norm_no))))
                recon_no = scale_no * idx_no
                mse_no_offset += (val - recon_no) ** 2

                # With offset quantization
                norm_w = (val - min_w) / (scale_w + 1e-10)
                idx_w = max(0, min(3, int(round(norm_w))))
                recon_w = min_w + scale_w * idx_w
                mse_with_offset += (val - recon_w) ** 2

    total_elements = nr * nc
    rmse_no = np.sqrt(mse_no_offset / total_elements)
    rmse_w = np.sqrt(mse_with_offset / total_elements)
    improvement = (rmse_no - rmse_w) / rmse_no * 100
    print(f"    Quant RMSE (no offset): {rmse_no:.6f}")  # noqa: NP100
    print(f"    Quant RMSE (with offset): {rmse_w:.6f}")  # noqa: NP100
    print(f"    Offset benefit: {improvement:.1f}% RMSE reduction")  # noqa: NP100

# ============================================================================
# 4. ACTIVATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 4: ACTIVATION DISTRIBUTION COMPARISON")  # noqa: NP100
print("=" * 80)  # noqa: NP100

activations = {
    "ffn_input (gate/up)": "act_blk0_ffn_input.f32bin",
    "ffn_down_input (swiglu)": "act_blk0_ffn_down_input.f32bin",
    "attn_input (q/k/v)": "act_blk0_attn_input.f32bin",
    "attn_output_input": "act_blk0_attn_output_input.f32bin",
}

act_data = {}
for name, fname in activations.items():
    try:
        A = load_f32_tensor(fname)
        act_data[name] = A
        print(f"\n{'─' * 70}")  # noqa: NP100
        print(f"  {name} — {fname}")  # noqa: NP100
        stats(name, A)
    except Exception as e:
        print(f"  {name}: SKIP ({e})")  # noqa: NP100

# ============================================================================
# 5. THE CRITICAL QUESTION: PER-DIMENSION ACTIVATION MAGNITUDE
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 5: PER-DIMENSION ACTIVATION POWER (per-column RMS)")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  If activation dimensions have very different magnitudes,")  # noqa: NP100
print("  the quantization error in each weight dimension is weighted differently.")  # noqa: NP100
print("  Dimensions with high activation power amplify weight errors.\n")  # noqa: NP100

for name, A in act_data.items():
    col_rms = np.sqrt(np.mean(A**2, axis=0))  # RMS per column (dimension)
    print(f"\n  {name} ({A.shape[1]} dimensions):")  # noqa: NP100
    print(f"    Col RMS: mean={col_rms.mean():.6f}, std={col_rms.std():.6f}")  # noqa: NP100
    print(f"    Col RMS range: [{col_rms.min():.6f}, {col_rms.max():.6f}]")  # noqa: NP100
    print(f"    Col RMS CV (std/mean): {col_rms.std() / (col_rms.mean() + 1e-10):.4f}")  # noqa: NP100
    print(f"    Max/Min ratio: {col_rms.max() / (col_rms.min() + 1e-10):.1f}x")  # noqa: NP100

    # Top 10 and bottom 10 dimensions by power
    top10 = np.argsort(col_rms)[-10:][::-1]
    bot10 = np.argsort(col_rms)[:10]
    print(  # noqa: NP100
        f"    Top-10 dims by RMS: {[(int(d), f'{col_rms[d]:.4f}') for d in top10[:5]]}..."
    )
    print(  # noqa: NP100
        f"    Bot-10 dims by RMS: {[(int(d), f'{col_rms[d]:.4f}') for d in bot10[:5]]}..."
    )

    # How much do the top 10% of dimensions contribute to total power?
    total_power = np.sum(col_rms**2)
    sorted_power = np.sort(col_rms**2)[::-1]
    top10pct = int(len(col_rms) * 0.1)
    top10pct_power = np.sum(sorted_power[:top10pct])
    top1pct = max(1, int(len(col_rms) * 0.01))
    top1pct_power = np.sum(sorted_power[:top1pct])
    print(  # noqa: NP100
        f"    Top 10% of dims contribute {top10pct_power / total_power * 100:.1f}% of total power"
    )
    print(  # noqa: NP100
        f"    Top 1% of dims contribute {top1pct_power / total_power * 100:.1f}% of total power"
    )

# ============================================================================
# 6. CROSS-CORRELATION: WEIGHT ERROR × ACTIVATION POWER
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 6: WHERE DO WEIGHT ERRORS MEET HIGH ACTIVATION POWER?")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  For each weight dimension, compute: activation_rms[dim] × weight_error[dim]")  # noqa: NP100
print("  This tells us which dimensions contribute most to matmul error.\n")  # noqa: NP100

# Focus on ffn_down vs ffn_gate for comparison
focus = [
    ("ffn_down", "blk_0_ffn_down_weight.f32bin", "act_blk0_ffn_down_input.f32bin"),
    ("ffn_gate", "blk_0_ffn_gate_weight.f32bin", "act_blk0_ffn_input.f32bin"),
    ("ffn_up", "blk_0_ffn_up_weight.f32bin", "act_blk0_ffn_input.f32bin"),
    ("attn_q", "blk_0_attn_q_weight.f32bin", "act_blk0_attn_input.f32bin"),
]

for name, wfile, afile in focus:
    W = load_f32_tensor(wfile)
    A = load_f32_tensor(afile)

    if W.shape[1] != A.shape[1]:
        print(f"  {name}: dim mismatch W={W.shape[1]} vs A={A.shape[1]}, SKIP")  # noqa: NP100
        continue

    nc = W.shape[1]

    # Per-column activation RMS
    act_rms = np.sqrt(np.mean(A**2, axis=0))

    # Per-column weight std and range (how "hard" to quantize)
    w_std = W.std(axis=0)
    w_range = W.max(axis=0) - W.min(axis=0)

    # Per-column weight kurtosis (heavy tails = harder to quantize)
    w_kurt = (
        np.mean(((W - W.mean(axis=0)) / (W.std(axis=0) + 1e-10)) ** 4, axis=0) - 3.0
    )

    # Weight error proxy: with 2-bit uniform quant on 16-element groups
    # Higher variance columns → more error
    nr = min(W.shape[0], 256)

    # Simple Q2_K-style error estimate per dimension:
    # For each group of 16 in the column direction, quantize and measure error
    dim_mse = np.zeros(nc)
    for g_start in range(0, nc, GS):
        g_end = min(g_start + GS, nc)
        for r in range(nr):
            g = W[r, g_start:g_end]
            gmin = min(g.min(), 0)  # Q2_K clamps min to ≤0
            gmax = g.max()
            gr = gmax - gmin
            if gr < 1e-10:
                continue
            scale = gr / 3.0
            for i, val in enumerate(g):
                norm = (val - gmin) / scale
                idx = max(0, min(3, int(round(norm))))
                recon = gmin + scale * idx
                dim_mse[g_start + i] += (val - recon) ** 2

    dim_rmse = np.sqrt(dim_mse / nr)

    # The key metric: dimension-level contribution to matmul error
    # matmul_error_contribution[d] ≈ act_rms[d] * weight_rmse[d]
    matmul_contrib = act_rms * dim_rmse

    print(f"\n  {name} ({nc} dimensions):")  # noqa: NP100
    print(  # noqa: NP100
        f"    act_rms: mean={act_rms.mean():.4f}, CV={act_rms.std() / act_rms.mean():.4f}"
    )
    print(  # noqa: NP100
        f"    w_rmse:  mean={dim_rmse.mean():.6f}, CV={dim_rmse.std() / (dim_rmse.mean() + 1e-10):.4f}"
    )
    print(  # noqa: NP100
        f"    matmul_contrib: mean={matmul_contrib.mean():.6f}, "
        f"std={matmul_contrib.std():.6f}"
    )

    # Correlation between activation power and weight error
    corr = np.corrcoef(act_rms, dim_rmse)[0, 1]
    print(f"    CORRELATION act_rms ↔ weight_rmse: {corr:.4f}")  # noqa: NP100
    print("      (>0 means high-power dims are also hard to quantize — BAD)")  # noqa: NP100

    # Top contributors to matmul error
    top_dims = np.argsort(matmul_contrib)[-20:][::-1]
    print("    Top-5 error-contributing dimensions:")  # noqa: NP100
    for d in top_dims[:5]:
        print(  # noqa: NP100
            f"      dim {d}: act_rms={act_rms[d]:.4f}, w_rmse={dim_rmse[d]:.6f}, "
            f"contrib={matmul_contrib[d]:.6f}, w_std={w_std[d]:.6f}, w_kurt={w_kurt[d]:.2f}"
        )

    # Distribution of matmul contributions
    total_contrib = matmul_contrib.sum()
    sorted_contrib = np.sort(matmul_contrib)[::-1]
    for pct in [0.01, 0.05, 0.10, 0.25]:
        n = max(1, int(nc * pct))
        print(  # noqa: NP100
            f"    Top {pct * 100:.0f}% dims: {sorted_contrib[:n].sum() / total_contrib * 100:.1f}% "
            f"of total matmul error"
        )

# ============================================================================
# 7. THE STRUCTURAL ASYMMETRY: COLUMN DIRECTION GROUP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 7: STRUCTURAL ASYMMETRY — COLUMN vs ROW GROUPING")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  Quantization groups along the ROW (inner dim). For ffn_down,")  # noqa: NP100
print("  each row has 9728 elements (38 groups of 256).")  # noqa: NP100
print("  For ffn_gate, each row has 2560 elements (10 groups of 256).")  # noqa: NP100
print("  More groups = more metadata (scales/offsets) relative to data bits.\n")  # noqa: NP100

for name, wfile, afile in focus:
    W = load_f32_tensor(wfile)
    nc = W.shape[1]
    n_groups_per_row = nc // 256  # super-blocks per row

    print(f"\n  {name}: {nc} cols → {n_groups_per_row} super-blocks per row")  # noqa: NP100
    print(f"    Groups per row: {nc // GS} (16-element groups)")  # noqa: NP100
    print(  # noqa: NP100
        f"    With Q2_K (2.625 bpw): {n_groups_per_row * 2} scale+offset bytes per row"
    )

    # How much do group means vary WITHIN a row?
    nr = min(W.shape[0], 64)
    intra_row_mean_var = []
    for r in range(nr):
        group_means = []
        for g_start in range(0, nc, GS):
            group_means.append(W[r, g_start : g_start + GS].mean())
        group_means = np.array(group_means)
        intra_row_mean_var.append(group_means.std())

    print(  # noqa: NP100
        f"    Intra-row group mean variability (avg across rows): "
        f"mean={np.mean(intra_row_mean_var):.6f}"
    )

    # How much does the sign of group means vary?
    pos_frac = 0
    neg_frac = 0
    total_groups = 0
    for r in range(nr):
        for g_start in range(0, nc, GS):
            gm = W[r, g_start : g_start + GS].mean()
            if gm > 0.001:
                pos_frac += 1
            elif gm < -0.001:
                neg_frac += 1
            total_groups += 1
    print(  # noqa: NP100
        f"    Group mean sign: {pos_frac / total_groups * 100:.1f}% positive, "
        f"{neg_frac / total_groups * 100:.1f}% negative, "
        f"{(1 - pos_frac / total_groups - neg_frac / total_groups) * 100:.1f}% near-zero"
    )

# ============================================================================
# 8. THE SWIGLU EFFECT: WHY ffn_down INPUT IS SPECIAL
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 8: THE SWIGLU EFFECT — ffn_down ACTIVATION STRUCTURE")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print("  ffn_down's activation is the SwiGLU output: silu(gate) * up")  # noqa: NP100
print("  This creates a specific activation pattern that differs from")  # noqa: NP100
print("  raw FFN input (RMSNorm output).\n")  # noqa: NP100

if "ffn_input (gate/up)" in act_data and "ffn_down_input (swiglu)" in act_data:
    A_in = act_data["ffn_input (gate/up)"]
    A_swiglu = act_data["ffn_down_input (swiglu)"]

    print(f"  FFN input (RMSNorm output): {A_in.shape}")  # noqa: NP100
    print(f"  SwiGLU output: {A_swiglu.shape}")  # noqa: NP100

    # Per-token analysis
    for t in range(min(A_swiglu.shape[0], 3)):
        tok_in = A_in[t]
        tok_sw = A_swiglu[t]
        print(f"\n  Token {t}:")  # noqa: NP100
        print(  # noqa: NP100
            f"    FFN input:   mean={tok_in.mean():.6f}, std={tok_in.std():.6f}, "
            f"|max|={np.abs(tok_in).max():.6f}"
        )
        print(  # noqa: NP100
            f"    SwiGLU out:  mean={tok_sw.mean():.6f}, std={tok_sw.std():.6f}, "
            f"|max|={np.abs(tok_sw).max():.6f}"
        )

        # SwiGLU creates lots of near-zero values (silu suppresses negatives)
        frac_nearzero_sw = np.mean(np.abs(tok_sw) < 0.01 * tok_sw.std())
        frac_nearzero_in = np.mean(np.abs(tok_in) < 0.01 * tok_in.std())
        print(  # noqa: NP100
            f"    Near-zero fraction: FFN input={frac_nearzero_in:.3f}, "
            f"SwiGLU={frac_nearzero_sw:.3f}"
        )

        # Sparsity pattern
        frac_neg = np.mean(tok_sw < 0)
        print(f"    SwiGLU negative fraction: {frac_neg:.3f}")  # noqa: NP100

    # Dimension-level analysis of SwiGLU
    print("\n  Dimension-level SwiGLU properties:")  # noqa: NP100
    dim_mean_sw = A_swiglu.mean(axis=0)
    dim_std_sw = A_swiglu.std(axis=0)
    dim_sparsity = np.mean(A_swiglu < 0, axis=0)  # fraction of tokens negative per dim

    print(f"    Dim mean range: [{dim_mean_sw.min():.6f}, {dim_mean_sw.max():.6f}]")  # noqa: NP100
    print(f"    Dim std range: [{dim_std_sw.min():.6f}, {dim_std_sw.max():.6f}]")  # noqa: NP100
    print(  # noqa: NP100
        f"    Dim negative fraction: mean={dim_sparsity.mean():.3f}, "
        f"range=[{dim_sparsity.min():.3f}, {dim_sparsity.max():.3f}]"
    )

    # Highly sparse dimensions (mostly near-zero after SwiGLU)
    high_sparsity = np.sum(dim_sparsity > 0.7)
    low_sparsity = np.sum(dim_sparsity < 0.3)
    print(f"    Dims with >70% negative tokens: {high_sparsity}/{len(dim_sparsity)}")  # noqa: NP100
    print(f"    Dims with <30% negative tokens: {low_sparsity}/{len(dim_sparsity)}")  # noqa: NP100

# ============================================================================
# 9. QUANTIZATION NOISE × ACTIVATION POWER: THE MATMUL ERROR DECOMPOSITION
# ============================================================================
print("\n" + "=" * 80)  # noqa: NP100
print("SECTION 9: MATMUL ERROR DECOMPOSITION")  # noqa: NP100
print("=" * 80)  # noqa: NP100
print(  # noqa: NP100
    "  matmul_error ≈ sum over groups of (activation_power_in_group × "
    "weight_mse_in_group)"
)
print(  # noqa: NP100
    "  If activation power is concentrated in groups with high weight error, "
    "matmul error explodes.\n"
)

# For ffn_down specifically, compare where activation power sits vs weight error
W_down = load_f32_tensor("blk_0_ffn_down_weight.f32bin")
A_swiglu = load_f32_tensor("act_blk0_ffn_down_input.f32bin")

W_gate = load_f32_tensor("blk_0_ffn_gate_weight.f32bin")
A_ffn_in = load_f32_tensor("act_blk0_ffn_input.f32bin")

for label, W, A in [("ffn_down", W_down, A_swiglu), ("ffn_gate", W_gate, A_ffn_in)]:
    nc = W.shape[1]
    nr = min(W.shape[0], 128)

    # Compute per-superblock (256) activation power and weight error
    n_sb = nc // 256
    sb_act_power = np.zeros(n_sb)
    sb_weight_mse = np.zeros(n_sb)

    for sb in range(n_sb):
        s = sb * 256
        e = s + 256
        # Activation power: mean squared activation in this region
        sb_act_power[sb] = np.mean(A[:, s:e] ** 2)

        # Weight MSE: Q2_K-style uniform quant error
        mse = 0
        count = 0
        for r in range(nr):
            for g in range(0, 256, GS):
                gvals = W[r, s + g : s + g + GS]
                gmin = min(gvals.min(), 0)
                gmax = gvals.max()
                gr = gmax - gmin
                if gr < 1e-10:
                    continue
                scale = gr / 3.0
                for v in gvals:
                    norm = (v - gmin) / scale
                    idx = max(0, min(3, int(round(norm))))
                    recon = gmin + scale * idx
                    mse += (v - recon) ** 2
                    count += 1
        sb_weight_mse[sb] = mse / max(count, 1)

    # Correlation between activation power and weight error across super-blocks
    valid = sb_act_power > 1e-10
    if valid.sum() > 10:
        corr = np.corrcoef(np.sqrt(sb_act_power[valid]), np.sqrt(sb_weight_mse[valid]))[
            0, 1
        ]
    else:
        corr = 0

    print(f"\n  {label}:")  # noqa: NP100
    print(f"    Super-blocks: {n_sb}")  # noqa: NP100
    print(  # noqa: NP100
        f"    act_power: mean={sb_act_power.mean():.6f}, "
        f"std={np.sqrt(sb_act_power.var()):.6f}, "
        f"range=[{sb_act_power.min():.6f}, {sb_act_power.max():.6f}]"
    )
    print(  # noqa: NP100
        f"    weight_mse: mean={sb_weight_mse.mean():.6f}, "
        f"range=[{sb_weight_mse.min():.6f}, {sb_weight_mse.max():.6f}]"
    )
    print(f"    CORRELATION (act_power ↔ weight_mse): {corr:.4f}")  # noqa: NP100

    # Show top-5 super-blocks by contribution to matmul error
    contrib = sb_act_power * sb_weight_mse
    top5 = np.argsort(contrib)[-5:][::-1]
    print(f"    Top-5 error-contributing super-blocks (of {n_sb}):")  # noqa: NP100
    for idx in top5:
        print(  # noqa: NP100
            f"      SB {idx * 256}-{(idx + 1) * 256 - 1}: act_power={sb_act_power[idx]:.6f}, "
            f"weight_mse={sb_weight_mse[idx]:.6f}, contrib={contrib[idx]:.6f}"
        )

print("\n" + "=" * 80)  # noqa: NP100
print("ANALYSIS COMPLETE")  # noqa: NP100
print("=" * 80)  # noqa: NP100
