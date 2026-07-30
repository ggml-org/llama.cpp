#!/usr/bin/env python3
"""Translate Unsloth dynamic-quant guidance into a Tessera policy.

This is deliberately an offline bridge.  llama.cpp does not import Unsloth,
PyTorch, Transformers, or bitsandbytes at runtime.  The bridge reads
Unsloth's public sensitive-module list (or a Hugging Face quantization
configuration), combines it with Tessera observer evidence, and emits the
policy schema consumed by the Tessera quantizer.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

# Polars is only required by the bridge / observer-evidence path.  Lazy-import
# so the PE-QAT sub-mode stays usable on systems where polars is not pinned
# (e.g. inside the calibration harness container that ships without it).
try:
    import polars  # noqa: F401
    _HAS_POLARS = True
except ImportError:  # pragma: no cover - exercised on polars-free systems
    _HAS_POLARS = False


POLICY_SCHEMA = "llama.speculative.calibration-policy.v1"
BRIDGE_SCHEMA = "llama.tessera.unsloth-bridge.v1"

# Hugging Face/Unsloth module names do not normally survive GGUF conversion.
# These aliases cover llama.cpp names while retaining the original name in
# provenance.  Matching remains substring-based, as it is in the quantizer.
GGUF_ALIASES = {
    "lm_head": ["output.weight"],
    "multi_modal_projector": ["mm.", "multi_modal_projector", "encoder_proj"],
    "merger": ["mm.", "merger"],
    "modality_projection": ["mm.", "modality_projection", "encoder_proj"],
    "router": ["ffn_gate_inp", "router"],
    "mlp.gate": ["ffn_gate_inp"],
    "block_sparse_moe.gate": ["ffn_gate_inp"],
    "audio_tower": ["a.", "audio_tower"],
    "vision_tower": ["v.", "vision_tower"],
    "vision_embedder": ["v.", "vision_embedder"],
    "embed_vision": ["v.", "embed_vision"],
    "embed_audio": ["a.", "embed_audio"],
}


def read_unsloth_skip_modules(unsloth_root: Path) -> tuple[list[str], Path]:
    candidates = [
        unsloth_root / "unsloth_zoo" / "peft_utils.py",
        unsloth_root / "peft_utils.py",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise ValueError(f"{unsloth_root}: could not locate unsloth_zoo/peft_utils.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "SKIP_QUANTIZATION_MODULES" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{source}: SKIP_QUANTIZATION_MODULES is not a string list")
        return value, source
    raise ValueError(f"{source}: SKIP_QUANTIZATION_MODULES was not found")


def config_skip_modules(config_path: Path) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    found: list[str] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            modules = value.get("llm_int8_skip_modules")
            if isinstance(modules, list):
                found.extend(item for item in modules if isinstance(item, str))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    return found


def unique_fragments(module: str) -> list[str]:
    values = [module, *GGUF_ALIASES.get(module, [])]
    return list(dict.fromkeys(value for value in values if value))


def observer_candidates(store: Path, run_id: str | None, fraction: float) -> list[dict]:
    if not _HAS_POLARS:
        raise RuntimeError(
            "polars is required for --evidence-store; install it via the tessera requirements"
        )
    import polars as pl
    files = list((store / "observer").glob("*.parquet"))
    if not files or fraction <= 0:
        return []
    query = pl.scan_parquet(str(store / "observer" / "*.parquet"))
    if run_id:
        query = query.filter(pl.col("run_id") == run_id)
    frame = (
        query.group_by("tensor")
        .agg(
            pl.col("tail_ratio").quantile(0.99).alias("tail"),
            pl.col("kurtosis").quantile(0.99).alias("kurtosis"),
            pl.col("rms").mean().alias("rms"),
            pl.col("count").max().alias("count"),
        )
        .filter(pl.col("count") > 0)
        .collect(engine="streaming")
    )
    if frame.is_empty():
        return []

    def finite_log(value: float) -> float:
        return math.log1p(max(float(value), 0.0)) if math.isfinite(float(value)) else 0.0

    records = frame.to_dicts()
    for record in records:
        record["score"] = (
            finite_log(record["tail"])
            + finite_log(record["kurtosis"])
            + 0.25 * finite_log(record["rms"])
        )
    records.sort(key=lambda record: (-record["score"], record["tensor"]))
    count = max(1, math.ceil(len(records) * fraction))
    return records[:count]


def load_base_policy(path: Path | None) -> dict:
    if path is None:
        return {}
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: unsupported policy schema {policy.get('schema')!r}")
    return policy


def command_pe_qat(args: argparse.Namespace) -> None:
    """PE-QAT sub-mode.  Trains a LoRA + SmoothQuant policy on a layer bundle."""
    # Importing here keeps the existing bridge path dependency-free of
    # ``tools.tessera.pe_qat`` (which is only relevant to the DSpark LoRA
    # finetuning lane).  We try the package-relative import first (so the
    # tool behaves identically whether invoked as a module or as a script)
    # and fall back to a sys.path-anchored import for the script path.
    try:
        from .pe_qat import apply_pe_qat, pe_qat_train, save_pe_qat_policy
        from .pe_qat_demo import _per_position_mse  # type: ignore[attr-defined]
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tools.tessera.pe_qat import apply_pe_qat, pe_qat_train, save_pe_qat_policy
        from tools.tessera.pe_qat_demo import _per_position_mse  # type: ignore[attr-defined]

    data = np.load(Path(args.pe_qat_bundle), allow_pickle=False)
    if "weight" not in data or "train_activations" not in data:
        raise ValueError(
            f"{args.pe_qat_bundle}: expected `weight` and `train_activations` arrays"
        )
    W = np.asarray(data["weight"], dtype=np.float32)
    x = np.asarray(data["train_activations"], dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"{args.pe_qat_bundle}: weight must be 2-D, got {W.shape}")
    if x.ndim != 2 or x.shape[1] != W.shape[1]:
        raise ValueError(
            f"{args.pe_qat_bundle}: train_activations must be (batch, in={W.shape[1]}), got {x.shape}"
        )
    print(f"loaded {args.pe_qat_bundle}: W={W.shape}, x={x.shape}")
    result = pe_qat_train(
        W,
        x,
        rank=args.pe_qat_rank,
        alpha=args.pe_qat_alpha,
        iters=args.pe_qat_iters,
        lr=args.pe_qat_lr,
        smooth_prior_weight=args.pe_qat_smooth_prior_weight,
        log_every=args.pe_qat_log_every,
        seed=args.pe_qat_seed,
    )
    output = Path(args.output)
    save_pe_qat_policy(output, result, family=args.pe_qat_family)
    print(f"wrote {output}")

    y_ref, y_q = apply_pe_qat(W, x, result)
    per_pos = _per_position_mse(y_ref, y_q)
    print(
        f"output MSE: quant-vs-ref={float(np.mean(per_pos)):.4e}  "
        f"per-position min={per_pos.min():.4e} max={per_pos.max():.4e} "
        f"median={np.median(per_pos):.4e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Tessera policy from Unsloth and observer evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unsloth-root", default="/Volumes/Julian T7/unsloth-zoo")
    parser.add_argument("--config", default=None, help="Optional Hugging Face config.json")
    parser.add_argument("--base-policy", default=None)
    parser.add_argument("--evidence-store", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--evidence-top-fraction", type=float, default=0.05)
    parser.add_argument("--protected-outlier-frac", type=float, default=0.02)
    parser.add_argument(
        "--unsloth-skip-mode",
        choices=("protected", "exact"),
        default="protected",
        help="Encode Unsloth skips with a larger residual budget or exact Tessera residuals",
    )
    # PE-QAT mode flags.  All optional; when --pe-qat-bundle is omitted the
    # script behaves exactly as it did before this mode was added.  The
    # ``--pe-qat-*`` namespace keeps the existing flat CLI intact so the
    # calibrate_quantize.py and test-tessera-unsloth-policy.py consumers
    # don't need to be updated.
    parser.add_argument(
        "--pe-qat-bundle",
        default=None,
        help="Layer bundle .npz to run PE-QAT on; switches the script into PE-QAT mode",
    )
    parser.add_argument("--pe-qat-rank", type=int, default=16, help="LoRA rank for PE-QAT")
    parser.add_argument("--pe-qat-alpha", type=float, default=32.0, help="LoRA alpha for PE-QAT")
    parser.add_argument("--pe-qat-iters", type=int, default=100, help="PE-QAT training iterations")
    parser.add_argument("--pe-qat-lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument(
        "--pe-qat-smooth-prior-weight", type=float, default=1e-3, help="SmoothQuant log-space prior weight"
    )
    parser.add_argument("--pe-qat-seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--pe-qat-family", default="attention", help="Family tag stored in the policy JSON")
    parser.add_argument("--pe-qat-log-every", type=int, default=10, help="Loss-log interval")
    args = parser.parse_args()

    if args.pe_qat_bundle:
        command_pe_qat(args)
        return

    if not 0 <= args.evidence_top_fraction <= 1:
        raise ValueError("--evidence-top-fraction must be in [0, 1]")
    if not 0 < args.protected_outlier_frac <= 1:
        raise ValueError("--protected-outlier-frac must be in (0, 1]")

    modules, source = read_unsloth_skip_modules(Path(args.unsloth_root))
    if args.config:
        modules.extend(config_skip_modules(Path(args.config)))
    modules = list(dict.fromkeys(modules))
    evidence = (
        observer_candidates(Path(args.evidence_store), args.run_id, args.evidence_top_fraction)
        if args.evidence_store
        else []
    )

    base = load_base_policy(Path(args.base_policy) if args.base_policy else None)
    protected: dict[str, dict] = {}
    for index, module in enumerate(modules):
        protected[f"unsloth_{index:02d}_{module.replace('.', '_')}"] = {
            "match": unique_fragments(module),
            "exact": args.unsloth_skip_mode == "exact",
            "awq_alpha": 0.0,
            "awq_clip": 1.0,
            "outlier_fraction": 1.0 if args.unsloth_skip_mode == "exact" else args.protected_outlier_frac,
            "source_module": module,
        }
    for index, record in enumerate(evidence):
        protected[f"observer_sensitive_{index:04d}"] = {
            "match": [record["tensor"]],
            "exact": False,
            "awq_alpha": "auto",
            "awq_clip": 1.0,
            "outlier_fraction": args.protected_outlier_frac,
            "observer_score": record["score"],
        }

    # Protected, tensor-specific matches must precede broad evolved families
    # because the quantizer intentionally uses first-match policy semantics.
    families = {**protected, **base.get("tensor_families", {})}
    policy = {
        **base,
        "schema": POLICY_SCHEMA,
        "tensor_families": families,
        "unsloth_bridge": {
            "schema": BRIDGE_SCHEMA,
            "source": str(source),
            "config": args.config,
            "skip_mode": args.unsloth_skip_mode,
            "protected_outlier_fraction": args.protected_outlier_frac,
            "modules": modules,
            "evidence_store": args.evidence_store,
            "evidence_run_id": args.run_id,
            "evidence_selected": len(evidence),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {destination}: {len(modules)} Unsloth modules and "
        f"{len(evidence)} evidence-selected tensors"
    )


if __name__ == "__main__":
    main()
