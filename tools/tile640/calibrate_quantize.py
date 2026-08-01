#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(str(arg) for arg in command), file=sys.stderr)
    subprocess.run(command, check=True)


def apply_runtime_benchmark(args) -> None:
    """Select the fastest verified microbatch geometry from a Tessera TSV."""
    if not args.runtime_benchmark:
        return
    path = Path(args.runtime_benchmark)
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    candidates = []
    for row in rows:
        try:
            ubatch = int(row["ubatch"])
            elapsed = float(row["real_seconds"])
            batches = int(row.get("observer_batches", "0"))
        except (KeyError, TypeError, ValueError):
            continue
        if ubatch > 0 and elapsed > 0.0 and batches > 0:
            candidates.append((elapsed, ubatch, row))
    if not candidates:
        raise ValueError(
            f"{path}: no completed observer benchmark rows; run "
            "benchmark-observer-ubatch.zsh first"
        )
    elapsed, ubatch, row = min(candidates)
    args.ubatch_size = ubatch
    print(
        f"Selected --ubatch-size {ubatch} from {path.name} "
        f"({elapsed:.3f}s, observer copy={row.get('copy_seconds', 'n/a')}s, "
        f"reduce={row.get('reduce_seconds', 'n/a')}s)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and Tessera-quantize a model")
    parser.add_argument("--model-dir", required=True, help="Hugging Face safetensors directory")
    parser.add_argument(
        "--f16-model",
        required=True,
        help="GGUF used for calibration (may itself be a compact Tessera baseline)",
    )
    parser.add_argument(
        "--metadata-model",
        default=None,
        help="Canonical GGUF supplying metadata and transformed tensors; defaults to --f16-model",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Canonical mmproj GGUF to embed into the quantized model",
    )
    parser.add_argument("--calibration-data", required=True, help="Calibration text file")
    parser.add_argument(
        "--challenge-data",
        default=None,
        help="Optional clean-room challenge holdout merged into the initial imatrix before policy evolution",
    )
    parser.add_argument(
        "--challenge-chunks", type=int, default=0,
        help="Maximum challenge chunks; 0 uses the full challenge corpus",
    )
    parser.add_argument("--output", required=True, help="Output Tessera GGUF")
    parser.add_argument("--imatrix", default=None, help="Imatrix output path")
    parser.add_argument(
        "--quantize-imatrix-merge",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Extra imatrix file(s) to merge with the primary imatrix via "
            "geometric mean during tessera quantization. Repeat for multiple. "
            "Only used if --imatrix is also set. The merge policy is fixed at "
            "geometric-mean for now (sqrt(a*b*c*...) per tensor)."
        ),
    )
    parser.add_argument("--llama-imatrix", default="/Volumes/Julian T7/llama-cpp-build/bin/llama-imatrix")
    parser.add_argument("--quantizer", default=str(Path(__file__).with_name("tile640_quantize_v3.py")))
    parser.add_argument(
        "--draft-telemetry",
        "--dflash-telemetry",
        dest="draft_telemetry",
        default=None,
        help="DFlash or MTP acceptance JSONL captured during speculative decoding",
    )
    parser.add_argument(
        "--calibration-policy",
        default=None,
        help="DFlash calibration policy output; defaults beside --output",
    )
    parser.add_argument(
        "--per-tensor-calibration",
        default=None,
        help=(
            "Per-tensor evolutionary calibration JSON from "
            "tools/tessera/per_tensor_calibrate.py. Carries per-tensor "
            "ternary_threshold, outlier_fraction, awq_alpha, awq_clip. "
            "Applied to both the verifier (main model) and the drafter."
        ),
    )
    parser.add_argument(
        "--per-tensor-bundles",
        default=None,
        help=(
            "Directory of layer .npz bundles (from "
            "tools/tessera/make-awq-layer-bundles.py). When --per-tensor-"
            "calibration is also set the orchestrator runs the per-tensor "
            "evolutionary calibration as a pre-step and writes the JSON to "
            "the path supplied here. The JSON is then forwarded as "
            "--per-tensor-calibration to the quantizer."
        ),
    )
    parser.add_argument("--per-tensor-generations", type=int, default=6)
    parser.add_argument("--per-tensor-population", type=int, default=8)
    parser.add_argument(
        "--per-tensor-include-family", nargs="*", default=None,
        help="Restrict the per-tensor search to these families (e.g. attention ffn)",
    )
    parser.add_argument(
        "--per-tensor-fitness",
        choices=("lrq", "awq", "flrq", "dartquant", "compare"),
        default="awq",
        help=(
            "Per-tensor fitness mode, forwarded to per_tensor_calibrate.py. "
            "The choices must match that tool's --fitness. 'awq' = island GA "
            "via awq-evolve.py (the production path). 'lrq' = low-rank "
            "weight-scaling closed-form fit. 'flrq' = outlier-aware low-rank "
            "plus clipped residual. 'dartquant' = QR-Orth pre-rotation with "
            "a Whip loss. 'compare' = run awq and lrq and write a "
            "side-by-side report."
        ),
    )
    parser.add_argument(
        "--per-tensor-lossless-target", type=float, default=None,
        help=(
            "Early-stop per-tensor GA when relative MSE drops below this "
            "value. The 'effectively lossless in reference to the BF16 "
            "source' criterion. Default: no early stop."
        ),
    )
    parser.add_argument(
        "--per-tensor-tool",
        default=os.path.join(
            os.path.dirname(__file__), "..", "tessera", "per_tensor_calibrate.py"
        ),
    )
    parser.add_argument(
        "--policy-generator",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/ane-mtp/dflash-calibration-policy.py",
    )
    parser.add_argument(
        "--evolution-layers",
        default=None,
        help="Directory of AWQ layer bundles; enables island/MAP-Elites policy search",
    )
    parser.add_argument(
        "--auxiliary-evolution-layers",
        action="append",
        default=[],
        help="Additional vision, audio, or drafter bundle directories included in shadow calibration",
    )
    parser.add_argument(
        "--evolve-awq",
        action="store_true",
        help="Export sampled layers from --model-dir/--imatrix and evolve an AWQ policy",
    )
    parser.add_argument(
        "--bundle-exporter",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/make-awq-layer-bundles.py",
    )
    parser.add_argument(
        "--evolution-output",
        default=None,
        help="Evolved policy output; defaults beside --output",
    )
    parser.add_argument(
        "--evolution-checkpoint",
        default=None,
        help="Checkpoint prefix for resumable AWQ evolution",
    )
    parser.add_argument(
        "--evolver",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/awq-evolve.py",
    )
    parser.add_argument(
        "--shadow-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="score provisional Tessera reconstructions before writing the GGUF and refine sensitive overrides; enabled automatically with AWQ evolution",
    )
    parser.add_argument(
        "--shadow-tool",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/shadow-calibrate.py",
    )
    parser.add_argument("--shadow-top-fraction", type=float, default=0.12)
    parser.add_argument("--shadow-max-overrides", type=int, default=24)
    parser.add_argument("--shadow-residual-boost", type=float, default=1.0)
    parser.add_argument("--evolution-generations", type=int, default=24)
    parser.add_argument("--evolution-population", type=int, default=16)
    parser.add_argument("--evolution-islands", type=int, default=4)
    parser.add_argument(
        "--calibration-prior",
        default=None,
        help="Compatible prior policy; only portable family rules are merged after local evolution",
    )
    parser.add_argument("--prior-family", default=None, help="Architecture-family label recorded with --calibration-prior")
    parser.add_argument(
        "--prior-tool",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/policy-prior.py",
    )
    parser.add_argument(
        "--evidence-store",
        default=None,
        help="Append observer, evolution, and acceptance evidence as Parquet partitions",
    )
    parser.add_argument("--evidence-run-id", default=None)
    parser.add_argument(
        "--evidence-tool",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/evidence-store.py",
    )
    parser.add_argument(
        "--hf-evidence-repo",
        default=os.environ.get(
            "TESSERA_HF_EVIDENCE_REPO",
            "juliantorr/tessera-calibration-commons",
        ),
        help="Public HF dataset to pull compatible aggregate evidence from before calibration",
    )
    parser.add_argument(
        "--no-hf-evidence",
        action="store_true",
        help="Disable the default compatible-evidence bootstrap for an offline run",
    )
    parser.add_argument("--hf-evidence-revision", default="main")
    parser.add_argument(
        "--hf-evidence-publish",
        action="store_true",
        help="Publish privacy-filtered aggregate evidence after a successful run",
    )
    parser.add_argument(
        "--hf-evidence-tool",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/hf-evidence.py",
    )
    parser.add_argument(
        "--unsloth-policy",
        action="store_true",
        help="Merge Unsloth sensitive-module guidance and observer tails into the Tessera policy",
    )
    parser.add_argument("--unsloth-root", default="/Volumes/Julian T7/unsloth-zoo")
    parser.add_argument(
        "--unsloth-config",
        default=None,
        help="Hugging Face config.json containing optional dynamic-quant skip modules",
    )
    parser.add_argument(
        "--unsloth-tool",
        default="/Users/user/Developer/GitHub/llama.cpp/tools/tessera/unsloth-policy.py",
    )
    parser.add_argument(
        "--unsloth-skip-mode",
        choices=("protected", "exact"),
        default="protected",
    )
    parser.add_argument("--unsloth-protected-outlier-frac", type=float, default=0.02)
    parser.add_argument("--unsloth-evidence-top-fraction", type=float, default=0.05)
    parser.add_argument(
        "--gguf-py",
        default="/Users/user/Developer/GitHub/llama.cpp/gguf-py",
        help="Path to llama.cpp/gguf-py",
    )
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=0, help="Maximum calibration chunks; 0 uses all input")
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="Skip this many calibration chunks (for resuming from an --input-imatrix checkpoint)",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument(
        "--runtime-benchmark",
        default=None,
        help="observer-ubatch benchmark TSV; selects its fastest completed u-batch",
    )
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument(
        "--input-imatrix",
        action="append",
        default=[],
        help="Existing GGUF imatrix to merge into the new calibration (repeatable)",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=0,
        help="Write a resumable imatrix checkpoint every N calibration iterations",
    )
    parser.add_argument(
        "--awq-search-target",
        choices=("per-row", "layer-output"),
        default="per-row",
        help=(
            "Optimization target for the AWQ alpha search. 'layer-output' uses "
            "per-tensor X (synthetic with banded correlation by default; real "
            "calibration activations if --calibration-activations is provided)."
        ),
    )
    parser.add_argument(
        "--calibration-activations",
        default=None,
        help="Path to .npz with per-layer X arrays for the layer-output AWQ search.",
    )
    parser.add_argument("--awq-synthetic-batch", type=int, default=32)
    parser.add_argument("--awq-synthetic-correlation", type=float, default=0.25)
    parser.add_argument("--awq-alpha", default="auto", help="'auto' or a fixed value in [0, 1]")
    parser.add_argument("--outlier-frac", type=float, default=0.005)
    parser.add_argument(
        "--range-selection",
        choices=("legacy", "imatrix-mse"),
        default="legacy",
        help=(
            "Tessera range-selection strategy. 'imatrix-mse' runs a vllm-style "
            "importance-weighted MSE grid search per row, picking the ternary "
            "code's range to minimize weighted reconstruction error. "
            "Bypasses AWQ per-channel scaling because the importance signal "
            "is consumed by the range selection."
        ),
    )
    parser.add_argument("--imatrix-mse-norm", type=float, default=3.0,
                        help="Exponent p for imatrix_mse importance-weighted error")
    parser.add_argument("--imatrix-mse-grid", type=int, default=20,
                        help="Number of grid steps for imatrix_mse shrink search")
    parser.add_argument("--imatrix-mse-maxshrink", type=float, default=0.95,
                        help="Maximum shrink factor (fraction of max(|W|)) for imatrix_mse grid")
    parser.add_argument(
        "--force-gemma4-n-swa",
        type=int,
        default=512,
        help=(
            "Override gemma4.attention.sliding_window (and the MTP variant) "
            "to this value at export time. 0 disables the override. Default "
            "512 fixes the 1024-instead-of-512 bug in Google's QAT gguf."
        ),
    )
    parser.add_argument(
        "--gemma4-sensitive-patterns",
        nargs="*",
        default=None,
        help=(
            "Substring patterns of canonical tensor names that must be forced "
            "to exact (outlier_frac=1.0) Tile640 encoding. Default for "
            "gemma4/gemma4-assistant covers QK-norm, post-norm, attention "
            "output, and FFN down projection."
        ),
    )
    parser.add_argument("--force-calibration", action="store_true")
    args = parser.parse_args()
    apply_runtime_benchmark(args)
    metadata_model = args.metadata_model or args.f16_model

    output = Path(args.output)
    evidence_run_id = args.evidence_run_id or output.stem
    imatrix = Path(args.imatrix) if args.imatrix else output.with_suffix(".imatrix.gguf")
    imatrix.parent.mkdir(parents=True, exist_ok=True)

    if args.hf_evidence_repo and not args.no_hf_evidence:
        if not args.evidence_store:
            args.evidence_store = str(output.with_suffix(".evidence"))
        run([
            sys.executable,
            args.hf_evidence_tool,
            "pull",
            "--repo", args.hf_evidence_repo,
            "--model-dir", args.model_dir,
            "--store", args.evidence_store,
            "--revision", args.hf_evidence_revision,
        ])

    if args.force_calibration or not imatrix.exists():
        command = [
            args.llama_imatrix,
            "-m", args.f16_model,
            "-f", args.calibration_data,
            "-o", str(imatrix),
            "--output-format", "gguf",
            "--ctx-size", str(args.ctx_size),
            "--batch-size", str(args.batch_size),
            "--ubatch-size", str(args.ubatch_size),
            "--no-ppl",
            "-ngl", str(args.gpu_layers),
        ]
        # With zero GPU layers, llama.cpp may still offload weightless graph
        # operations to Metal unless this is disabled explicitly. That creates
        # repeated CPU/Metal activation transfers and is dramatically slower
        # for source-model calibration on a memory-constrained Mac.
        if args.gpu_layers == 0:
            command.append("--no-op-offload")
        for input_imatrix in args.input_imatrix:
            command.extend(["--in-file", input_imatrix])
        if args.chunk_start > 0:
            command.extend(["--chunk", str(args.chunk_start)])
        if args.chunks > 0:
            command.extend(["--chunks", str(args.chunks)])
        if args.save_frequency > 0:
            command.extend(["--save-frequency", str(args.save_frequency)])
        run(command)
    else:
        print(f"Using existing imatrix: {imatrix}", file=sys.stderr)

    if args.challenge_data:
        challenge_imatrix = imatrix.with_name(imatrix.stem + ".challenge.imatrix.gguf")
        challenge_command = [
            args.llama_imatrix,
            "-m", args.f16_model,
            "-f", args.challenge_data,
            "-o", str(challenge_imatrix),
            "--output-format", "gguf",
            "--ctx-size", str(args.ctx_size),
            "--batch-size", str(args.batch_size),
            "--ubatch-size", str(args.ubatch_size),
            "--no-ppl",
            "-ngl", str(args.gpu_layers),
            "--in-file", str(imatrix),
        ]
        if args.gpu_layers == 0:
            challenge_command.append("--no-op-offload")
        if args.challenge_chunks > 0:
            challenge_command.extend(["--chunks", str(args.challenge_chunks)])
        run(challenge_command)
        challenge_imatrix.replace(imatrix)

    if args.evidence_store:
        run([
            sys.executable,
            args.evidence_tool,
            "ingest-imatrix",
            "--store", args.evidence_store,
            "--run-id", evidence_run_id,
            "--imatrix", str(imatrix),
            "--gguf-py", args.gguf_py,
        ])

    calibration_policy = None
    if args.draft_telemetry:
        calibration_policy = Path(args.calibration_policy) if args.calibration_policy else output.with_suffix(".draft-policy.json")
        run([
            sys.executable,
            args.policy_generator,
            "--telemetry", args.draft_telemetry,
            "--output", str(calibration_policy),
            "--base-outlier-frac", str(args.outlier_frac),
        ])
        if args.evidence_store:
            run([
                sys.executable,
                args.evidence_tool,
                "ingest-acceptance",
                "--store", args.evidence_store,
                "--run-id", evidence_run_id,
                "--telemetry", args.draft_telemetry,
            ])

    if args.unsloth_policy:
        unsloth_policy = output.with_suffix(".unsloth-policy.json")
        command = [
            sys.executable,
            args.unsloth_tool,
            "--unsloth-root", args.unsloth_root,
            "--output", str(unsloth_policy),
            "--unsloth-skip-mode", args.unsloth_skip_mode,
            "--protected-outlier-frac", str(args.unsloth_protected_outlier_frac),
            "--evidence-top-fraction", str(args.unsloth_evidence_top_fraction),
        ]
        if args.unsloth_config:
            command.extend(["--config", args.unsloth_config])
        if calibration_policy:
            command.extend(["--base-policy", str(calibration_policy)])
        if args.evidence_store:
            command.extend([
                "--evidence-store", args.evidence_store,
                "--run-id", evidence_run_id,
            ])
        run(command)
        calibration_policy = unsloth_policy

    evolution_layers = args.evolution_layers
    if args.evolve_awq and not evolution_layers:
        evolution_layers = str(output.with_suffix(".awq-layers"))
        run([
            sys.executable,
            args.bundle_exporter,
            "--model-dir", args.model_dir,
            "--imatrix", str(imatrix),
            "--output", evolution_layers,
            "--gguf-py", args.gguf_py,
        ])

    if evolution_layers:
        evolution_checkpoint = args.evolution_checkpoint
        if args.evidence_store and not evolution_checkpoint:
            evolution_checkpoint = str(output.with_suffix(".evolution.json"))
        evolved_policy = (
            Path(args.evolution_output)
            if args.evolution_output
            else output.with_suffix(".evolved-policy.json")
        )
        evolve_command = [
            sys.executable,
            args.evolver,
            "--layers", evolution_layers,
            "--output", str(evolved_policy),
            "--generations", str(args.evolution_generations),
            "--population", str(args.evolution_population),
            "--islands", str(args.evolution_islands),
            "--base-outlier-frac", str(args.outlier_frac),
        ]
        if evolution_checkpoint:
            evolve_command.extend(["--checkpoint", evolution_checkpoint])
        if calibration_policy:
            evolve_command.extend(["--base-policy", str(calibration_policy)])
        run(evolve_command)
        calibration_policy = evolved_policy
        if args.evidence_store and evolution_checkpoint:
            checkpoint_path = Path(evolution_checkpoint)
            run([
                sys.executable,
                args.evidence_tool,
                "ingest-evolution",
                "--store", args.evidence_store,
                "--run-id", evidence_run_id,
                "--checkpoint-glob", str(
                    checkpoint_path.with_name(f"{checkpoint_path.stem}.*.json")
                ),
            ])

    if args.calibration_prior:
        if not calibration_policy:
            raise ValueError("--calibration-prior requires a locally evolved or draft calibration policy")
        if not args.prior_family:
            raise ValueError("--calibration-prior requires --prior-family")
        prior_policy = output.with_suffix(".prior-policy.json")
        run([
            sys.executable,
            args.prior_tool,
            "--base-policy", str(calibration_policy),
            "--prior-policy", args.calibration_prior,
            "--family", args.prior_family,
            "--output", str(prior_policy),
        ])
        calibration_policy = prior_policy

    shadow_calibration = args.shadow_calibration
    if shadow_calibration is None:
        shadow_calibration = evolution_layers is not None
    if shadow_calibration:
        if not evolution_layers:
            raise ValueError("--shadow-calibration requires --evolve-awq or --evolution-layers")
        if not calibration_policy:
            raise RuntimeError("shadow calibration requires a provisional calibration policy")
        shadow_policy = output.with_suffix(".shadow-policy.json")
        run([
            sys.executable,
            args.shadow_tool,
            "--layers", evolution_layers,
            "--base-policy", str(calibration_policy),
            "--output", str(shadow_policy),
            "--top-fraction", str(args.shadow_top_fraction),
            "--max-overrides", str(args.shadow_max_overrides),
            "--residual-boost", str(args.shadow_residual_boost),
            "--evolver", args.evolver,
        ] + sum((["--layers", directory] for directory in args.auxiliary_evolution_layers), []))
        calibration_policy = shadow_policy
        if args.evidence_store:
            run([
                sys.executable,
                args.evidence_tool,
                "ingest-shadow",
                "--store", args.evidence_store,
                "--run-id", evidence_run_id,
                "--policy", str(shadow_policy),
            ])

    per_tensor_calibration = (
        Path(args.per_tensor_calibration) if args.per_tensor_calibration else None
    )
    if args.per_tensor_bundles:
        # Run the per-tensor evolutionary calibration as a pre-step. This is
        # the missing calibration knob: per-tensor (ternary_threshold,
        # outlier_fraction, awq_alpha, awq_clip) tuned by a small GA. Used
        # for both the verifier and the drafter.
        per_tensor_calibration = (
            per_tensor_calibration
            if per_tensor_calibration
            else output.with_suffix(".per-tensor.json")
        )
        calibrate_cmd = [
            sys.executable,
            args.per_tensor_tool,
            "--layers", args.per_tensor_bundles,
            "--output", str(per_tensor_calibration),
            "--base-outlier-frac", str(args.outlier_frac),
            "--generations", str(args.per_tensor_generations),
            "--population", str(args.per_tensor_population),
            "--fitness", args.per_tensor_fitness,
        ]
        if args.per_tensor_lossless_target is not None:
            calibrate_cmd.extend(["--lossless-target", str(args.per_tensor_lossless_target)])
        if calibration_policy is not None:
            calibrate_cmd.extend(["--base-policy", str(calibration_policy)])
        if args.per_tensor_include_family:
            calibrate_cmd.extend(["--include-family", *args.per_tensor_include_family])
        run(calibrate_cmd)
        print(
            f"per-tensor calibration written to {per_tensor_calibration}",
            file=sys.stderr,
        )

    quantize_command = [
        sys.executable,
        args.quantizer,
        "--model-dir", args.model_dir,
        "--output", args.output,
        "--metadata-from", metadata_model,
        "--gguf-py", args.gguf_py,
        "--imatrix", str(imatrix),
        "--awq-alpha", args.awq_alpha,
        "--outlier-frac", str(args.outlier_frac),
        "--range-selection", args.range_selection,
        "--imatrix-mse-norm", str(args.imatrix_mse_norm),
        "--imatrix-mse-grid", str(args.imatrix_mse_grid),
        "--imatrix-mse-maxshrink", str(args.imatrix_mse_maxshrink),
        "--force-gemma4-n-swa", str(args.force_gemma4_n_swa),
        "--awq-search-target", args.awq_search_target,
        "--awq-synthetic-batch", str(args.awq_synthetic_batch),
        "--awq-synthetic-correlation", str(args.awq_synthetic_correlation),
    ]
    if args.calibration_activations:
        quantize_command.extend(["--calibration-activations", args.calibration_activations])
    for merge_path in args.quantize_imatrix_merge:
        quantize_command.extend(["--imatrix-merge", merge_path])
    if args.gemma4_sensitive_patterns:
        quantize_command.extend(["--gemma4-sensitive-patterns"] + args.gemma4_sensitive_patterns)
    if calibration_policy:
        quantize_command.extend(["--calibration-policy", str(calibration_policy)])
    if per_tensor_calibration is not None:
        quantize_command.extend(["--calibration-policy", str(per_tensor_calibration)])
    if args.vision_model:
        quantize_command.extend(["--vision-from", args.vision_model])
    run(quantize_command)

    if args.hf_evidence_publish:
        if not args.hf_evidence_repo:
            raise ValueError("--hf-evidence-publish requires --hf-evidence-repo")
        run([
            sys.executable,
            args.hf_evidence_tool,
            "publish",
            "--repo", args.hf_evidence_repo,
            "--model-dir", args.model_dir,
            "--store", args.evidence_store,
            "--run-id", evidence_run_id,
        ])


if __name__ == "__main__":
    main()
