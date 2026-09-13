#!/usr/bin/env python3
"""validate-conversion.py — compare llama.cpp converted model tensors with HuggingFace originals.

Runs both models on the same prompt, dumps intermediate tensors for the first,
middle, and last layer, then prints a side-by-side accuracy summary.

Usage:
  validate-conversion.py \\
      --hf-model  /path/to/hf-model \\
      --gguf-model /path/to/model.gguf \\
      [--config gemma4-tensor-pairs.yaml] \\
      [--output-dir data/<model-name>] \\
      [--prompt "Hello, my name is"] \\
      [--llama-debug /path/to/llama-debug] \\
      [--n-layers 35] \\
      [--layers 0,17,34] \\
      [--device cuda]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    print("PyYAML not found. Install with: pip install pyyaml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a llama.cpp model conversion by comparing intermediate tensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--hf-model", "-H", required=True, metavar="DIR",
                   help="Path to the HuggingFace model directory")
    p.add_argument("--gguf-model", "-G", required=True, metavar="FILE",
                   help="Path to the GGUF model file")
    p.add_argument("--config", "-c", default=None, metavar="FILE",
                   help="Tensor pairs config YAML (default: gemma4-tensor-pairs.yaml in script dir)")
    p.add_argument("--output-dir", "-o", default=None, metavar="DIR",
                   help="Directory to save tensor dumps (default: data/<model-name>)")
    p.add_argument("--prompt", "-p", default="Hello, my name is",
                   help="Prompt to use for both models (default: 'Hello, my name is')")
    p.add_argument("--llama-debug", default=None, metavar="BIN",
                   help="Path to llama-debug binary (auto-detected if omitted)")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Override number of layers from config")
    p.add_argument("--layers", default=None, metavar="N,N,...",
                   help="Comma-separated layer indices to dump (default: first,middle,last)")
    p.add_argument("--device", default="auto",
                   help="PyTorch device for HF model: cpu, cuda, auto (default: auto)")
    p.add_argument("--skip-llama", action="store_true",
                   help="Skip running llama-debug (reuse existing dumps)")
    p.add_argument("--skip-torch", action="store_true",
                   help="Skip running HF model (reuse existing dumps)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tensor comparison utilities (mirrors compare-tensors.py)
# ---------------------------------------------------------------------------

def nmse_db(ref, test):
    mse  = float(np.mean((ref - test) ** 2))
    var  = float(np.var(ref))
    nmse = mse / var if var > 1e-30 else 0.0
    return nmse, 10 * np.log10(nmse + 1e-300)


def load_tensor(directory, name):
    path = Path(directory) / f"{name}.bin"
    if not path.exists():
        return None, None
    arr = np.fromfile(path, dtype=np.float32)
    shape_path = Path(directory) / f"{name}.shape"
    shape = None
    if shape_path.exists():
        shape = tuple(int(x) for x in shape_path.read_text().split())
    return arr, shape


def compare_pair(llama_dir, torch_dir, llama_name, torch_name):
    """Return (status, metrics_dict) for one tensor pair."""
    ref, ref_shape = load_tensor(torch_dir, torch_name)
    llm, llm_shape = load_tensor(llama_dir, llama_name)

    if ref is None and llm is None:
        return "both_missing", {}
    if ref is None:
        return "torch_missing", {}
    if llm is None:
        return "llama_missing", {}
    if ref.size != llm.size:
        # llama.cpp strips to the last token via inp_out_ids in the final layer.
        # PyTorch keeps all tokens. If ref is an exact multiple of llm, compare
        # only the last llm.size elements of ref (= last token in row-major layout).
        if ref.size > llm.size and ref.size % llm.size == 0:
            ref = ref[-llm.size:]
        else:
            return "size_mismatch", {"ref_size": ref.size, "llm_size": llm.size,
                                      "ref_shape": ref_shape, "llm_shape": llm_shape}

    nmse, db = nmse_db(ref, llm)
    max_err  = float(np.max(np.abs(ref - llm)))
    mean_err = float(np.mean(np.abs(ref - llm)))
    return "ok", {"n": ref.size, "nmse": nmse, "db": db,
                  "max_err": max_err, "mean_err": mean_err}


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run(cmd, env=None, check=True):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env or os.environ.copy(), check=check)
    return result.returncode


def find_llama_debug(hint, script_dir):
    if hint:
        p = Path(hint)
        if not p.exists():
            print(f"Error: llama-debug not found at {hint}")
            sys.exit(1)
        return str(p.resolve())

    candidates = [
        script_dir / "../../../../build/bin/llama-debug",
        script_dir / "../../../../build/bin/Debug/llama-debug",
        Path("/usr/local/bin/llama-debug"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    print("Error: llama-debug not found. Specify with --llama-debug.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    script_dir = Path(__file__).parent

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config_path = args.config or (script_dir / "gemma4-tensor-pairs.yaml")
    if not Path(config_path).exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    n_layers = args.n_layers or config.get("n_layers", 35)

    if args.layers:
        target_layers = sorted(set(int(x) for x in args.layers.split(",")))
    else:
        target_layers = sorted(set([0, n_layers // 2, n_layers - 1]))

    print(f"Layers to validate : {target_layers} (model has {n_layers} layers)")

    global_pairs = config.get("global_tensors", [])
    layer_pairs  = config.get("layer_tensors",  [])

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    model_name = Path(args.hf_model).name
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir / "data" / model_name)
    llama_dir  = output_dir / "llama"
    torch_base = output_dir / "torch"
    llama_dir.mkdir(parents=True, exist_ok=True)
    torch_base.mkdir(parents=True, exist_ok=True)
    print(f"Output directory   : {output_dir}")

    # ------------------------------------------------------------------
    # Step 1: llama-debug
    # ------------------------------------------------------------------
    if not args.skip_llama:
        llama_debug = find_llama_debug(args.llama_debug, script_dir)
        print(f"llama-debug binary : {llama_debug}")

        # Build --tensor-filter: include all tensor names we care about
        filter_names = [p["llama"] for p in global_pairs]
        for layer in target_layers:
            for p in layer_pairs:
                filter_names.append(p["llama"].format(layer=layer))
        tensor_filter = "|".join(filter_names)

        print("\n" + "=" * 60)
        print("Step 1: Running llama-debug")
        print("=" * 60)
        run([
            llama_debug,
            "-m", args.gguf_model,
            "-p", args.prompt,
            "--no-warmup",
            "--tensor-filter", tensor_filter,
            "--save-tensors", str(llama_dir),
        ])
    else:
        print("\nStep 1: Skipped (--skip-llama)")

    # ------------------------------------------------------------------
    # Step 2: HuggingFace model — one run per target layer
    # ------------------------------------------------------------------
    run_org = script_dir / "run-org-model.py"
    env = os.environ.copy()
    env["MODEL_TESTING_PROMPT"] = args.prompt

    if not args.skip_torch:
        print("\n" + "=" * 60)
        print("Step 2: Running HuggingFace model")
        print("=" * 60)
        for layer in target_layers:
            layer_out = torch_base / f"layer-{layer}"
            layer_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  Layer {layer} → {layer_out}/")
            run([
                sys.executable, str(run_org),
                "--model-path", args.hf_model,
                "--dump-tensors", str(layer_out),
                "--dump-layer", str(layer),
                "--device", args.device,
            ], env=env)
    else:
        print("\nStep 2: Skipped (--skip-torch)")

    # ------------------------------------------------------------------
    # Step 3: Comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Comparison summary")
    print("=" * 60)

    # For global tensors we use the first layer's torch dir
    global_torch_dir = torch_base / f"layer-{target_layers[0]}"

    col_name = 28
    header = (f"\n{'Layer':>6}  {'Tensor':<{col_name}}  {'N-elem':>8}  "
              f"{'NMSE':>12}  {'dB':>8}  {'MaxErr':>10}  {'MeanErr':>10}  Status")
    sep = "-" * (len(header) - 1)
    print(header)
    print(sep)

    all_results = []

    def print_row(layer_label, llama_name, torch_name, torch_dir):
        status, m = compare_pair(llama_dir, torch_dir, llama_name, torch_name)
        if status == "ok":
            nmse, db, n = m["nmse"], m["db"], m["n"]
            max_e, mean_e = m["max_err"], m["mean_err"]
            if nmse > 0.1:
                flag = "DIVERGED"
            elif nmse > 1e-2:
                flag = "high"
            elif nmse > 1e-4:
                flag = "~ ok"
            else:
                flag = "OK"
            display = llama_name if llama_name == torch_name else f"{llama_name} / {torch_name}"
            print(f"{layer_label:>6}  {display:<{col_name}}  {n:>8d}  "
                  f"{nmse:>12.3e}  {db:>8.2f}  {max_e:>10.4f}  {mean_e:>10.4f}  {flag}")
        elif status == "size_mismatch":
            print(f"{layer_label:>6}  {llama_name:<{col_name}}  SIZE MISMATCH  "
                  f"torch={m['ref_size']} {m['ref_shape']}  llama={m['llm_size']} {m['llm_shape']}")
        elif status == "torch_missing":
            print(f"{layer_label:>6}  {llama_name:<{col_name}}  MISSING (torch)")
        elif status == "llama_missing":
            print(f"{layer_label:>6}  {llama_name:<{col_name}}  MISSING (llama)")
        elif status == "both_missing":
            print(f"{layer_label:>6}  {llama_name:<{col_name}}  MISSING (both)")
        all_results.append((status, m))

    # Global tensors
    for p in global_pairs:
        print_row("global", p["llama"], p["torch"], global_torch_dir)

    # Layer tensors
    for layer in target_layers:
        layer_torch_dir = torch_base / f"layer-{layer}"
        for p in layer_pairs:
            llama_name = p["llama"].format(layer=layer)
            torch_name = p["torch"].format(layer=layer)
            print_row(str(layer), llama_name, torch_name, layer_torch_dir)

    print(sep)

    # Final verdict
    ok_count      = sum(1 for s, _ in all_results if s == "ok")
    diverged      = [(s, m) for s, m in all_results if s == "ok" and m.get("nmse", 0) > 0.1]
    missing_count = sum(1 for s, _ in all_results if s not in ("ok", "size_mismatch"))
    total         = len(all_results)

    print(f"\n{ok_count}/{total} tensor pairs compared")
    if missing_count:
        print(f"{missing_count} tensor(s) missing — check that both runs completed successfully")
    if diverged:
        print(f"{len(diverged)} tensor(s) diverged (NMSE > 0.1) — inspect the first divergence point")
    else:
        print("No diverged tensors.")


if __name__ == "__main__":
    main()
