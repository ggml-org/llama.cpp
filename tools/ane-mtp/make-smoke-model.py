#!/usr/bin/env python3
"""Build tiny fixed-batch Core ML programs for ANE MTP residency tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import torch


class SmokeState(torch.nn.Module):
    def __init__(self, batch: int) -> None:
        super().__init__()
        self.register_buffer("kv_state", torch.zeros((batch, 96)))


class SmokeSync(SmokeState):
    def forward(
        self,
        active: torch.Tensor,
        positions: torch.Tensor,
        base_keys: torch.Tensor,
        base_values: torch.Tensor,
        swa_keys: torch.Tensor,
        swa_values: torch.Tensor,
    ) -> torch.Tensor:
        pos = positions.to(torch.int64)
        base_columns = torch.arange(2, device=positions.device)
        swa_columns = torch.arange(4, device=positions.device)
        base_idx = pos[..., None] * 2 + base_columns
        base_v_idx = 16 + base_idx
        swa_idx = 32 + pos[..., None] * 4 + swa_columns
        swa_v_idx = 64 + pos[..., None] * 4 + swa_columns
        indices = torch.cat((base_idx, base_v_idx, swa_idx, swa_v_idx), dim=-1)
        source = torch.cat((base_keys, base_values, swa_keys, swa_values), dim=-1)
        old = torch.gather(self.kv_state, 1, indices.flatten(1))
        mask = active.to(torch.bool)[:, None].expand_as(old)
        source_flat = torch.where(mask, source.flatten(1), old)
        updated = self.kv_state.scatter(1, indices.flatten(1), source_flat)
        self.kv_state.copy_(updated)
        return self.kv_state[:, :2]


class SmokeMTP(SmokeState):
    def forward(
        self,
        token_ids: torch.Tensor,
        h_nextn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_signal = token_ids.to(torch.float32).unsqueeze(-1)
        state_signal = (
            self.kv_state[:, 0:1] + self.kv_state[:, 16:17]
            + self.kv_state[:, 32:33] + self.kv_state[:, 64:65]
        )
        next_hidden = h_nextn + token_signal + state_signal
        scores = torch.cat((next_hidden[:, :1], -next_hidden[:, :1]), dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        confidence, top_token = torch.max(probabilities, dim=-1)
        # Keep the buffer as Core ML state in the predict function.
        self.kv_state.copy_(self.kv_state + next_hidden.sum() * 0)
        return top_token.to(torch.int32), confidence, next_hidden


class SmokeReset(SmokeState):
    def forward(self, active: torch.Tensor) -> torch.Tensor:
        mask = active.to(torch.bool)[:, None].expand_as(self.kv_state)
        self.kv_state.copy_(torch.where(mask, torch.zeros_like(self.kv_state), self.kv_state))
        return self.kv_state[:, :2]


class SmokePrefill(torch.nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        value = token_ids.to(torch.float32) + positions.to(torch.float32)
        offsets = torch.arange(8, device=value.device, dtype=torch.float32)
        hidden = value.unsqueeze(-1) + offsets
        key = torch.stack((value, value + 0.25), dim=-1).unsqueeze(-2)
        output_value = torch.stack((value + 0.5, value + 0.75), dim=-1).unsqueeze(-2)
        return hidden, key, output_value


class SmokeDFlash(torch.nn.Module):
    def __init__(self, block: int) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        target_features: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.arange(
            1, self.block + 1, device=token_ids.device, dtype=torch.int32
        )
        draft_tokens = token_ids[:, None] + offsets[None, :]
        signal = target_features[:, :1] + positions.to(torch.float32)[:, None]
        confidence = torch.sigmoid(signal).expand(-1, self.block)
        return draft_tokens, confidence


class SmokeHybrid(torch.nn.Module):
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


def build_bucket(
    output_root: Path,
    batch: int,
    prefill_sequence: int,
    dflash_block: int,
) -> None:
    module = SmokeMTP(batch).eval()
    token_ids = torch.zeros((batch,), dtype=torch.int32)
    h_nextn = torch.zeros((batch, 8), dtype=torch.float32)
    positions = torch.zeros((batch, 4), dtype=torch.int32)
    active = torch.ones((batch,), dtype=torch.int32)
    base = torch.zeros((batch, 4, 2), dtype=torch.float32)
    base_v = torch.ones((batch, 4, 2), dtype=torch.float32)
    swa = torch.zeros((batch, 4, 4), dtype=torch.float32)
    swa_v = torch.ones((batch, 4, 4), dtype=torch.float32)
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=value.shape), name=name)
        for name, value in module.named_buffers()
    ]

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        predict_program = torch.export.export(
            module, (token_ids, h_nextn)
        ).run_decompositions({})
        predict = ct.convert(
            predict_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="token_ids", shape=token_ids.shape, dtype=int),
                ct.TensorType(name="h_nextn", shape=h_nextn.shape),
            ],
            outputs=[
                ct.TensorType(name="top_token", dtype=int),
                ct.TensorType(name="confidence"),
                ct.TensorType(name="next_hidden"),
            ],
            states=states,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        predict_package = temporary_root / "predict.mlpackage"
        predict.save(predict_package)

        sync_module = SmokeSync(batch).eval()
        sync_program = torch.export.export(
            sync_module, (active, positions, base, base_v, swa, swa_v)
        ).run_decompositions({})
        sync_states = [
            ct.StateType(wrapped_type=ct.TensorType(shape=value.shape), name=name)
            for name, value in sync_module.named_buffers()
        ]
        sync = ct.convert(
            sync_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="active", shape=active.shape, dtype=int),
                ct.TensorType(name="positions", shape=positions.shape, dtype=int),
                ct.TensorType(name="base_keys", shape=base.shape),
                ct.TensorType(name="base_values", shape=base.shape),
                ct.TensorType(name="swa_keys", shape=swa.shape),
                ct.TensorType(name="swa_values", shape=swa.shape),
            ],
            states=sync_states,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        sync_package = temporary_root / "sync.mlpackage"
        sync.save(sync_package)

        reset_module = SmokeReset(batch).eval()
        reset_program = torch.export.export(
            reset_module, (active,)
        ).run_decompositions({})
        reset_states = [
            ct.StateType(wrapped_type=ct.TensorType(shape=value.shape), name=name)
            for name, value in reset_module.named_buffers()
        ]
        reset = ct.convert(
            reset_program,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="active", shape=active.shape, dtype=int)],
            states=reset_states,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        reset_package = temporary_root / "reset.mlpackage"
        reset.save(reset_package)

        prefill_tokens = torch.zeros(
            (batch, prefill_sequence), dtype=torch.int32
        )
        prefill_positions = torch.zeros_like(prefill_tokens)
        prefill_program = torch.export.export(
            SmokePrefill().eval(), (prefill_tokens, prefill_positions)
        ).run_decompositions({})
        prefill = ct.convert(
            prefill_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="token_ids", shape=prefill_tokens.shape, dtype=int
                ),
                ct.TensorType(
                    name="positions", shape=prefill_positions.shape, dtype=int
                ),
            ],
            outputs=[
                ct.TensorType(name="hidden_states"),
                ct.TensorType(name="key_states"),
                ct.TensorType(name="value_states"),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        prefill_package = temporary_root / "prefill.mlpackage"
        prefill.save(prefill_package)

        target_features = torch.zeros((batch, 8), dtype=torch.float32)
        draft_token = torch.zeros((batch,), dtype=torch.int32)
        draft_position = torch.zeros((batch,), dtype=torch.int32)
        dflash_program = torch.export.export(
            SmokeDFlash(dflash_block).eval(),
            (target_features, draft_token, draft_position),
        ).run_decompositions({})
        dflash = ct.convert(
            dflash_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="target_features", shape=target_features.shape
                ),
                ct.TensorType(
                    name="token_ids", shape=draft_token.shape, dtype=int
                ),
                ct.TensorType(
                    name="positions", shape=draft_position.shape, dtype=int
                ),
            ],
            outputs=[
                ct.TensorType(name="draft_tokens", dtype=int),
                ct.TensorType(name="confidence"),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        dflash_package = temporary_root / "dflash.mlpackage"
        dflash.save(dflash_package)

        candidate_shape = (batch, dflash_block)
        hybrid_d_tokens = torch.zeros(candidate_shape, dtype=torch.int32)
        hybrid_d_conf = torch.zeros(candidate_shape, dtype=torch.float32)
        hybrid_d_counts = torch.zeros((batch,), dtype=torch.int32)
        hybrid_m_tokens = torch.zeros(candidate_shape, dtype=torch.int32)
        hybrid_m_conf = torch.zeros(candidate_shape, dtype=torch.float32)
        hybrid_m_counts = torch.zeros((batch,), dtype=torch.int32)
        hybrid_cutoff = torch.full((batch,), 0.65, dtype=torch.float32)
        hybrid_program = torch.export.export(
            SmokeHybrid(dflash_block).eval(),
            (
                hybrid_d_tokens,
                hybrid_d_conf,
                hybrid_d_counts,
                hybrid_m_tokens,
                hybrid_m_conf,
                hybrid_m_counts,
                hybrid_cutoff,
            ),
        ).run_decompositions({})
        hybrid = ct.convert(
            hybrid_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(
                    name="dflash_tokens", shape=candidate_shape, dtype=int
                ),
                ct.TensorType(
                    name="dflash_confidence", shape=candidate_shape
                ),
                ct.TensorType(
                    name="dflash_counts", shape=(batch,), dtype=int
                ),
                ct.TensorType(
                    name="mtp_tokens", shape=candidate_shape, dtype=int
                ),
                ct.TensorType(
                    name="mtp_confidence", shape=candidate_shape
                ),
                ct.TensorType(
                    name="mtp_counts", shape=(batch,), dtype=int
                ),
                ct.TensorType(
                    name="dflash_cutoff", shape=(batch,)
                ),
            ],
            outputs=[
                ct.TensorType(name="selected_source", dtype=int),
                ct.TensorType(name="agreement", dtype=int),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        hybrid_package = temporary_root / "hybrid.mlpackage"
        hybrid.save(hybrid_package)

        descriptor = ct.utils.MultiFunctionDescriptor()
        descriptor.add_function(str(predict_package), "main", "predict")
        descriptor.add_function(str(sync_package), "main", "sync")
        descriptor.add_function(str(reset_package), "main", "reset")
        descriptor.add_function(
            str(prefill_package), "main", f"prefill_s{prefill_sequence}"
        )
        descriptor.add_function(
            str(dflash_package), "main", f"dflash_b{dflash_block}"
        )
        descriptor.add_function(
            str(hybrid_package), "main", f"hybrid_b{dflash_block}"
        )
        descriptor.default_function_name = "predict"
        package = output_root / f"batch-{batch}.mlpackage"
        ct.utils.save_multifunction(descriptor, str(package))

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(output_root)],
        check=True,
    )
    manifest = {
        "format": "ane-compute-image-v1",
        "batch": batch,
        "context": 16,
        "sync_chunk": 4,
        "functions": [
            "predict",
            "sync",
            "reset",
            f"prefill_s{prefill_sequence}",
            f"dflash_b{dflash_block}",
            f"hybrid_b{dflash_block}",
        ],
    }
    (output_root / f"batch-{batch}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prefill-sequence", type=int, default=4)
    parser.add_argument("--dflash-block", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        build_bucket(
            args.output, batch, args.prefill_sequence, args.dflash_block
        )


if __name__ == "__main__":
    main()
