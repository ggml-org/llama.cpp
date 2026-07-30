#!/usr/bin/env python3
"""Export a fixed-bucket hybrid DFlash/MTP arbitration mlpackage."""

import argparse
from pathlib import Path

import coremltools as ct
import torch


class HybridArbiter(torch.nn.Module):
    def __init__(self, block: int) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        dflash_tokens: torch.Tensor,
        dflash_confidence: torch.Tensor,
        dflash_counts: torch.Tensor,
        mtp_tokens: torch.Tensor,
        mtp_confidence: torch.Tensor,
        mtp_counts: torch.Tensor,
        dflash_cutoff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(
            self.block, device=dflash_tokens.device, dtype=torch.int32
        )[None, :]
        limit = torch.minimum(dflash_counts, mtp_counts)
        comparable = offsets < limit[:, None]
        mismatch = torch.logical_and(comparable, dflash_tokens != mtp_tokens)
        first = torch.argmax(mismatch.to(torch.int32), dim=1).to(torch.int32)
        has_mismatch = torch.any(mismatch, dim=1)
        agreement = torch.where(has_mismatch, first, limit)
        index = torch.clamp(agreement, 0, self.block - 1).to(torch.int64)
        pd = torch.gather(dflash_confidence, 1, index[:, None])[:, 0]
        pm = torch.gather(mtp_confidence, 1, index[:, None])[:, 0]
        both = torch.logical_and(dflash_counts > 0, mtp_counts > 0)
        uncertain = torch.logical_and(
            torch.logical_and(
                agreement < dflash_counts,
                pd < dflash_cutoff,
            ),
            pm > pd,
        )
        extension = torch.logical_and(
            agreement == dflash_counts,
            mtp_counts > dflash_counts,
        )
        choose_mtp = torch.logical_or(
            torch.logical_and(dflash_counts == 0, mtp_counts > 0),
            torch.logical_and(both, torch.logical_or(uncertain, extension)),
        )
        selected_source = torch.where(
            choose_mtp,
            torch.full_like(dflash_counts, 2),
            torch.where(
                dflash_counts > 0,
                torch.ones_like(dflash_counts),
                torch.zeros_like(dflash_counts),
            ),
        )
        return selected_source, agreement


def export(output: Path, batch: int, block: int) -> None:
    shape = (batch, block)
    d_tokens = torch.zeros(shape, dtype=torch.int32)
    d_confidence = torch.zeros(shape, dtype=torch.float32)
    d_counts = torch.zeros((batch,), dtype=torch.int32)
    m_tokens = torch.zeros(shape, dtype=torch.int32)
    m_confidence = torch.zeros(shape, dtype=torch.float32)
    m_counts = torch.zeros((batch,), dtype=torch.int32)
    cutoff = torch.full((batch,), 0.65, dtype=torch.float32)
    program = torch.export.export(
        HybridArbiter(block).eval(),
        (
            d_tokens,
            d_confidence,
            d_counts,
            m_tokens,
            m_confidence,
            m_counts,
            cutoff,
        ),
    ).run_decompositions({})
    model = ct.convert(
        program,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="dflash_tokens", shape=shape, dtype=int),
            ct.TensorType(name="dflash_confidence", shape=shape),
            ct.TensorType(name="dflash_counts", shape=(batch,), dtype=int),
            ct.TensorType(name="mtp_tokens", shape=shape, dtype=int),
            ct.TensorType(name="mtp_confidence", shape=shape),
            ct.TensorType(name="mtp_counts", shape=(batch,), dtype=int),
            ct.TensorType(name="dflash_cutoff", shape=(batch,)),
        ],
        outputs=[
            ct.TensorType(name="selected_source", dtype=int),
            ct.TensorType(name="agreement", dtype=int),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--block", type=int, required=True)
    args = parser.parse_args()
    export(args.output, args.batch, args.block)


if __name__ == "__main__":
    main()
