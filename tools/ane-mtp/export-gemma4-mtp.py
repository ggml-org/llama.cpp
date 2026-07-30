#!/usr/bin/env python3
"""Export the Gemma 4 assistant as a stateful Core ML multifunction program."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import torch
import torch.nn.functional as F
from safetensors import safe_open


EPS = 1e-6
SWA_CONTEXT = 1024
BASE_WIDTH = 512
SWA_WIDTH = 2048


def load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as source:
        return source.get_tensor(name).to(torch.float16)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + EPS)
    return normalized.to(value.dtype) * (weight + 1.0)


def apply_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    rotating_pairs: int,
) -> torch.Tensor:
    head_dim = value.shape[-1]
    pairs = head_dim // 2
    frequency = theta ** (
        -torch.arange(pairs, device=value.device, dtype=torch.float32) / pairs
    )
    if rotating_pairs < pairs:
        frequency = torch.cat(
            (
                frequency[:rotating_pairs],
                torch.zeros(pairs - rotating_pairs, device=value.device),
            )
        )
    angle = positions.float()[:, None] * frequency[None, :]
    cosine = torch.cos(angle).to(value.dtype)[:, None, :]
    sine = torch.sin(angle).to(value.dtype)[:, None, :]
    first, second = value[..., :pairs], value[..., pairs:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)


class Gemma4MTPState(torch.nn.Module):
    def __init__(self, batch: int, context: int) -> None:
        super().__init__()
        self.batch = batch
        self.context = context
        self.base_k_offset = 0
        self.base_v_offset = context * BASE_WIDTH
        self.swa_k_offset = self.base_v_offset + context * BASE_WIDTH
        self.swa_v_offset = self.swa_k_offset + SWA_CONTEXT * SWA_WIDTH
        state_size = self.swa_v_offset + SWA_CONTEXT * SWA_WIDTH
        self.register_buffer("kv_state", torch.zeros((batch, state_size), dtype=torch.float16))


class Gemma4MTPSync(Gemma4MTPState):
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
        base_columns = torch.arange(BASE_WIDTH, device=positions.device)
        swa_columns = torch.arange(SWA_WIDTH, device=positions.device)
        base_idx = self.base_k_offset + pos[..., None] * BASE_WIDTH + base_columns
        base_v_idx = self.base_v_offset + pos[..., None] * BASE_WIDTH + base_columns
        swa_pos = torch.remainder(pos, SWA_CONTEXT)
        swa_idx = self.swa_k_offset + swa_pos[..., None] * SWA_WIDTH + swa_columns
        swa_v_idx = self.swa_v_offset + swa_pos[..., None] * SWA_WIDTH + swa_columns
        indices = torch.cat((base_idx, base_v_idx, swa_idx, swa_v_idx), dim=-1)
        source = torch.cat(
            (base_keys, base_values, swa_keys, swa_values), dim=-1
        ).to(self.kv_state.dtype)
        flat_indices = indices.flatten(1)
        old = torch.gather(self.kv_state, 1, flat_indices)
        mask = active.to(torch.bool)[:, None].expand_as(old)
        updated = self.kv_state.scatter(
            1, flat_indices, torch.where(mask, source.flatten(1), old)
        )
        self.kv_state.copy_(updated)
        return self.kv_state[:, :1]


class Gemma4MTPReset(Gemma4MTPState):
    def forward(self, active: torch.Tensor) -> torch.Tensor:
        mask = active.to(torch.bool)[:, None].expand_as(self.kv_state)
        self.kv_state.copy_(torch.where(mask, torch.zeros_like(self.kv_state), self.kv_state))
        return self.kv_state[:, :1]


class AssistantLayer(torch.nn.Module):
    def __init__(self, tensors: dict[str, torch.Tensor], layer: int) -> None:
        super().__init__()
        prefix = f"model.layers.{layer}"
        for name, suffix in (
            ("attn_norm", "input_layernorm.weight"),
            ("q", "self_attn.q_proj.weight"),
            ("q_norm", "self_attn.q_norm.weight"),
            ("o", "self_attn.o_proj.weight"),
            ("post_attn_norm", "post_attention_layernorm.weight"),
            ("ffn_norm", "pre_feedforward_layernorm.weight"),
            ("gate", "mlp.gate_proj.weight"),
            ("up", "mlp.up_proj.weight"),
            ("down", "mlp.down_proj.weight"),
            ("post_ffn_norm", "post_feedforward_layernorm.weight"),
            ("scale", "layer_scalar"),
        ):
            self.register_buffer(name, tensors[f"{prefix}.{suffix}"])


class Gemma4MTPPredict(Gemma4MTPState):
    def __init__(
        self,
        batch: int,
        context: int,
        target_embedding: torch.Tensor,
        assistant: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(batch, context)
        self.register_buffer("target_embedding", target_embedding)
        self.register_buffer("assistant_embedding", assistant["model.embed_tokens.weight"])
        self.register_buffer("pre_projection", assistant["pre_projection.weight"])
        self.register_buffer("post_projection", assistant["post_projection.weight"])
        self.register_buffer("final_norm", assistant["model.norm.weight"])
        self.layers = torch.nn.ModuleList(
            AssistantLayer(assistant, layer) for layer in range(4)
        )

    def attention(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        batch = query.shape[0]
        if layer < 3:
            state_k = self.kv_state[
                :, self.swa_k_offset:self.swa_v_offset
            ].reshape(batch, SWA_CONTEXT, 8, 256)
            state_v = self.kv_state[
                :, self.swa_v_offset:
            ].reshape(batch, SWA_CONTEXT, 8, 256)
            logical = positions[:, None] - (SWA_CONTEXT - 1) + torch.arange(
                SWA_CONTEXT, device=query.device
            )[None, :]
            indices = torch.remainder(logical, SWA_CONTEXT)
            gather_idx = indices[..., None, None].expand(-1, -1, 8, 256)
            keys = torch.gather(state_k, 1, gather_idx)
            values = torch.gather(state_v, 1, gather_idx)
            valid = logical >= 0
        else:
            keys = self.kv_state[
                :, self.base_k_offset:self.base_v_offset
            ].reshape(batch, self.context, 1, 512)
            values = self.kv_state[
                :, self.base_v_offset:self.swa_k_offset
            ].reshape(batch, self.context, 1, 512)
            valid = torch.arange(self.context, device=query.device)[None, :] < positions[:, None]

        repeats = 16 // keys.shape[2]
        keys = keys.repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
        values = values.repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
        scores = torch.einsum("bhd,bhld->bhl", query, keys)
        scores = torch.where(valid[:, None, :], scores, torch.full_like(scores, -1e4))
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.einsum("bhl,bhld->bhd", weights, values)

    def forward(
        self,
        token_ids: torch.Tensor,
        h_nextn: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = F.embedding(token_ids.to(torch.int64), self.target_embedding)
        token = token * math.sqrt(3840.0)
        current = F.linear(torch.cat((token, h_nextn), -1), self.pre_projection)

        for index, layer in enumerate(self.layers):
            head_dim = 256 if index < 3 else 512
            query = F.linear(rms_norm(current, layer.attn_norm), layer.q)
            query = query.reshape(self.batch, 16, head_dim)
            query = rms_norm(query, layer.q_norm)
            query = apply_rope(
                query,
                positions,
                10000.0 if index < 3 else 1000000.0,
                head_dim // 2 if index < 3 else 64,
            )
            attended = self.attention(query, positions, index).reshape(
                self.batch, 16 * head_dim
            )
            attended = F.linear(attended, layer.o)
            attended = rms_norm(attended, layer.post_attn_norm) + current
            ffn_input = rms_norm(attended, layer.ffn_norm)
            ffn = F.gelu(F.linear(ffn_input, layer.gate), approximate="tanh")
            ffn = ffn * F.linear(ffn_input, layer.up)
            current = (
                rms_norm(F.linear(ffn, layer.down), layer.post_ffn_norm) + attended
            ) * layer.scale

        normalized = rms_norm(current, self.final_norm)
        logits = F.linear(normalized, self.assistant_embedding)
        probabilities = torch.softmax(logits.float(), -1)
        confidence, top_token = torch.max(probabilities, -1)
        next_hidden = F.linear(normalized, self.post_projection)
        self.kv_state.copy_(self.kv_state + next_hidden.sum() * 0)
        return top_token.to(torch.int32), confidence, next_hidden


def assistant_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as source:
        return {name: source.get_tensor(name).to(torch.float16) for name in source.keys()}


def state_types(module: torch.nn.Module) -> list[ct.StateType]:
    return [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=value.shape),
            name=name,
        )
        for name, value in module.named_buffers()
        if name == "kv_state"
    ]


def export(args: argparse.Namespace) -> None:
    assistant_file = args.assistant / "model.safetensors"
    target_file = args.target / "model.safetensors"
    target_embedding = load_tensor(
        target_file, "model.language_model.embed_tokens.weight"
    )
    tensors = assistant_tensors(assistant_file)

    token_ids = torch.zeros((args.batch,), dtype=torch.int32)
    hidden = torch.zeros((args.batch, 3840), dtype=torch.float32)
    draft_positions = torch.ones((args.batch,), dtype=torch.int32)
    active = torch.ones((args.batch,), dtype=torch.int32)
    sync_positions = torch.zeros((args.batch, args.sync_chunk), dtype=torch.int32)
    base = torch.zeros((args.batch, args.sync_chunk, BASE_WIDTH), dtype=torch.float32)
    base_v = torch.ones_like(base)
    swa = torch.zeros((args.batch, args.sync_chunk, SWA_WIDTH), dtype=torch.float32)
    swa_v = torch.ones_like(swa)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predict_module = Gemma4MTPPredict(
            args.batch, args.context, target_embedding, tensors
        ).eval()
        predict_program = torch.export.export(
            predict_module, (token_ids, hidden, draft_positions)
        ).run_decompositions({})
        predict = ct.convert(
            predict_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="token_ids", shape=token_ids.shape, dtype=int),
                ct.TensorType(name="h_nextn", shape=hidden.shape),
                ct.TensorType(name="positions", shape=draft_positions.shape, dtype=int),
            ],
            outputs=[
                ct.TensorType(name="top_token", dtype=int),
                ct.TensorType(name="confidence"),
                ct.TensorType(name="next_hidden"),
            ],
            states=state_types(predict_module),
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        predict_path = root / "predict.mlpackage"
        predict.save(predict_path)

        sync_module = Gemma4MTPSync(args.batch, args.context).eval()
        sync_program = torch.export.export(
            sync_module,
            (active, sync_positions, base, base_v, swa, swa_v),
        ).run_decompositions({})
        sync = ct.convert(
            sync_program,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="active", shape=active.shape, dtype=int),
                ct.TensorType(name="positions", shape=sync_positions.shape, dtype=int),
                ct.TensorType(name="base_keys", shape=base.shape),
                ct.TensorType(name="base_values", shape=base.shape),
                ct.TensorType(name="swa_keys", shape=swa.shape),
                ct.TensorType(name="swa_values", shape=swa.shape),
            ],
            states=state_types(sync_module),
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        sync_path = root / "sync.mlpackage"
        sync.save(sync_path)

        reset_module = Gemma4MTPReset(args.batch, args.context).eval()
        reset_program = torch.export.export(
            reset_module, (active,)
        ).run_decompositions({})
        reset = ct.convert(
            reset_program,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="active", shape=active.shape, dtype=int)],
            states=state_types(reset_module),
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        reset_path = root / "reset.mlpackage"
        reset.save(reset_path)

        descriptor = ct.utils.MultiFunctionDescriptor()
        descriptor.add_function(str(predict_path), "main", "predict")
        descriptor.add_function(str(sync_path), "main", "sync")
        descriptor.add_function(str(reset_path), "main", "reset")
        extra_functions: list[str] = []
        for specification in args.compute_function:
            name, separator, source = specification.partition("=")
            if not separator or not re.fullmatch(
                r"(prefill_s|dflash_b|hybrid_b)[1-9][0-9]*", name
            ):
                raise ValueError(
                    "--compute-function must be prefill_sN=PATH, dflash_bN=PATH, or hybrid_bN=PATH"
                )
            source_path = Path(source)
            if not source_path.is_dir():
                raise ValueError(f"Core ML package does not exist: {source_path}")
            descriptor.add_function(str(source_path), "main", name)
            extra_functions.append(name)
        descriptor.default_function_name = "predict"
        package = args.output / f"batch-{args.batch}.mlpackage"
        args.output.mkdir(parents=True, exist_ok=True)
        ct.utils.save_multifunction(descriptor, str(package))

    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(args.output)],
        check=True,
    )
    manifest = {
        "format": "gemma4-mtp-state-v1",
        "batch": args.batch,
        "context": args.context,
        "sync_chunk": args.sync_chunk,
        "base_width": BASE_WIDTH,
        "swa_width": SWA_WIDTH,
        "functions": ["predict", "sync", "reset", *extra_functions],
    }
    (args.output / f"batch-{args.batch}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--assistant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--sync-chunk", type=int, default=512)
    parser.add_argument(
        "--compute-function",
        action="append",
        default=[],
        metavar="NAME=MLPACKAGE",
        help="Merge an exported prefill_sN, dflash_bN, or hybrid_bN mlpackage into the compute image.",
    )
    export(parser.parse_args())


if __name__ == "__main__":
    main()
