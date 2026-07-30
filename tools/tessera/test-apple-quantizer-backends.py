#!/usr/bin/env python3
"""Verify Tessera Core ML, Accelerate, and NumPy quantizer parity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apple_accelerate import AccelerateBackend
from apple_ane_quantizer import ANEQuantizerBackend


def encoded_scales(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    page = values.max(axis=-1)
    page = np.where(page < 1e-30, 1.0, page)
    lanes = np.clip(
        np.round(values / page[:, None] * 127.0), 1, 127
    ).astype(np.int8)
    return page.astype(np.float16), lanes


def verify(model: Path) -> None:
    coreml = ANEQuantizerBackend(model)
    accelerate = AccelerateBackend()
    random = np.random.default_rng(20260728)
    for rows in (64, 256, 1024):
        weights = random.normal(0.0, 0.15, (rows, 640)).astype(np.float32)
        weights[:, ::37] = np.nextafter(np.float32(0.0625), np.float32(1.0))
        weights[:, ::53] = np.float32(7.75)
        ternary = np.where(
            weights > 0.06, 1, np.where(weights < -0.06, -1, 0)
        ).astype(np.float32)

        lane_accelerate = accelerate.lane_targets(weights, ternary)
        lane_coreml = coreml.lane_targets(weights, ternary)
        if not np.array_equal(lane_coreml, lane_accelerate):
            raise AssertionError(f"lane targets differ for row bucket {rows}")
        if any(
            not np.array_equal(left, right)
            for left, right in zip(
                encoded_scales(lane_coreml),
                encoded_scales(lane_accelerate),
                strict=True,
            )
        ):
            raise AssertionError(f"encoded scales differ for row bucket {rows}")

        importance = np.linspace(0.125, 2.0, 640, dtype=np.float32)
        reconstructed = ternary * np.repeat(
            np.maximum(lane_accelerate, np.float32(1e-5)), 20, axis=1
        )
        residual_coreml = coreml.residual_score(
            weights,
            ternary,
            np.maximum(lane_accelerate, np.float32(1e-5)),
            importance,
        )
        residual_accelerate = accelerate.weighted_square_error(
            weights, reconstructed, importance
        )
        residual_numpy = (
            np.square(weights - reconstructed, dtype=np.float32)
            * np.square(importance, dtype=np.float32)
        )
        if not np.array_equal(residual_coreml, residual_accelerate):
            raise AssertionError(f"residual scores differ from Accelerate at {rows}")
        if not np.array_equal(residual_coreml, residual_numpy):
            raise AssertionError(f"residual scores differ from NumPy at {rows}")

        proposal_lane = coreml.lane_targets_ane(weights, ternary)
        proposal_residual = coreml.residual_score_ane(
            weights,
            ternary,
            np.maximum(lane_accelerate, np.float32(1e-5)),
            importance,
        )
        if not (
            np.all(np.isfinite(proposal_lane))
            and np.all(np.isfinite(proposal_residual))
        ):
            raise AssertionError(f"ANE proposal produced nonfinite values at {rows}")
        print(f"row bucket {rows}: exact parity and finite ANE proposal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    verify(args.model)


if __name__ == "__main__":
    main()
