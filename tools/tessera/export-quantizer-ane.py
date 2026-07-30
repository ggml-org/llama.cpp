#!/usr/bin/env python3
"""Export Tessera lane statistics and residual scoring as one Core ML program."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import torch


LANE_WIDTH = 20
PAGE_WIDTH = 640


class ResidualScore(torch.nn.Module):
    def forward(
        self,
        weights: torch.Tensor,
        ternary: torch.Tensor,
        lane_scale: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        expanded = torch.repeat_interleave(
            lane_scale, LANE_WIDTH, dim=1
        )
        error = weights - ternary * expanded
        return error * error * (importance * importance)


class LaneTargets(torch.nn.Module):
    def forward(
        self,
        weights: torch.Tensor,
        ternary: torch.Tensor,
    ) -> torch.Tensor:
        shaped_weights = torch.abs(weights).reshape(
            weights.shape[0], PAGE_WIDTH // LANE_WIDTH, LANE_WIDTH
        )
        retained = (ternary != 0).to(weights.dtype).reshape(
            weights.shape[0], PAGE_WIDTH // LANE_WIDTH, LANE_WIDTH
        )
        # Give Core ML an explicit finite upper bound.  An open-ended clamp
        # otherwise lowers its implicit FP32 maximum to an overflowing FP16
        # constant when the program is compiled for ANE execution.
        count = torch.clamp(torch.sum(retained, dim=-1), min=1.0, max=float(LANE_WIDTH))
        return torch.sum(shaped_weights * retained, dim=-1) / count


def convert_module(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    output: Path,
    precision: ct.precision,
) -> None:
    exported = torch.export.export(module.eval(), inputs).run_decompositions({})
    model = ct.convert(
        exported,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=name, shape=value.shape)
            for name, value in zip(names, inputs, strict=True)
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision,
    )
    model.save(output)


def export(output: Path, row_buckets: tuple[int, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tessera-ane-") as directory:
        temporary = Path(directory)
        descriptor = ct.utils.MultiFunctionDescriptor()
        functions: list[str] = []
        for rows in row_buckets:
            weights = torch.zeros((rows, PAGE_WIDTH), dtype=torch.float32)
            ternary = torch.zeros_like(weights)
            lane_scale = torch.ones(
                (rows, PAGE_WIDTH // LANE_WIDTH), dtype=torch.float32
            )
            importance = torch.ones((PAGE_WIDTH,), dtype=torch.float32)

            for variant, precision in (
                ("ane", ct.precision.FLOAT16),
                ("exact", ct.precision.FLOAT32),
            ):
                residual_path = (
                    temporary / f"residual-{variant}-r{rows}.mlpackage"
                )
                convert_module(
                    ResidualScore(),
                    (weights, ternary, lane_scale, importance),
                    ("weights", "ternary", "lane_scale", "importance"),
                    residual_path,
                    precision,
                )
                residual_name = f"residual_score_{variant}_r{rows}"
                descriptor.add_function(
                    str(residual_path), "main", residual_name
                )
                functions.append(residual_name)

                lane_path = temporary / f"lanes-{variant}-r{rows}.mlpackage"
                convert_module(
                    LaneTargets(),
                    (weights, ternary),
                    ("weights", "ternary"),
                    lane_path,
                    precision,
                )
                lane_name = f"lane_targets_{variant}_r{rows}"
                descriptor.add_function(
                    str(lane_path), "main", lane_name
                )
                functions.append(lane_name)

        descriptor.default_function_name = functions[0]
        package = output / "tessera-quantizer.mlpackage"
        ct.utils.save_multifunction(descriptor, str(package))

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(output)],
        check=True,
    )
    manifest = {
        "format": "tessera-quantizer-multifunction-v2",
        "page_width": PAGE_WIDTH,
        "lane_width": LANE_WIDTH,
        "row_buckets": list(row_buckets),
        "precision_contract": {
            "ane": "float16 proposal path on CPU and Neural Engine",
            "exact": "float32 canonical path on CPU and GPU",
        },
        "functions": functions,
    }
    (output / "tessera-quantizer.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--row-buckets",
        type=int,
        nargs="+",
        default=(64, 256, 1024),
    )
    args = parser.parse_args()
    export(args.output, tuple(sorted(set(args.row_buckets))))


if __name__ == "__main__":
    main()
