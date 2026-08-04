from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GptOssForCausalLM")
class GptOssModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT_OSS

    # TODO: remove once MXFP4 is supported more generally
    def dequant_model(self):
        if self._is_mxfp4:
            return
        return super().dequant_model()

    def transform_nibble_layout(self, tensor):
        assert tensor.dtype == torch.uint8
        assert tensor.shape[-1] == 16
        # swap nibbles
        t_lo = tensor & 0x0F
        t_hi = tensor & 0xF0
        t_swapped = (t_lo << 4) | (t_hi >> 4)
        tensor = t_swapped
        # transform aaaa...bbbb... to abababab...
        blk_a, blk_b = tensor.chunk(2, dim=-1)
        # get a_
        blk_a0 = (blk_a & 0xF0).view(-1, 1)
        blk_a1 = (blk_a << 4).view(-1, 1)
        blk_a = torch.stack((blk_a0, blk_a1), dim=2).view(tensor.shape)
        # get _b
        blk_b0 = (blk_b >> 4).view(-1, 1)
        blk_b1 = (blk_b & 0x0F).view(-1, 1)
        blk_b = torch.stack((blk_b0, blk_b1), dim=2).view(tensor.shape)
        # swap once more
        out = blk_a | blk_b
        out_h = out & 0xF0
        out_l = out & 0x0F
        out = (out_h >> 4) | (out_l << 4)
        return out

    def repack_mxfp4_parts(self, new_name: str, parts: tuple[tuple[Tensor, Tensor], ...]):
        if not parts:
            raise ValueError(f"No MXFP4 parts provided for {new_name}")

        blocks0, scales0 = parts[0]
        if blocks0.dtype != torch.uint8 or scales0.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 MXFP4 blocks and scales for {new_name}")
        if len(blocks0.shape) != 4 or len(scales0.shape) != 3:
            raise ValueError(f"Unexpected MXFP4 tensor rank for {new_name}")
        if blocks0.shape[:3] != scales0.shape or blocks0.shape[-1] != 16:
            raise ValueError(f"Mismatched MXFP4 blocks and scales for {new_name}")

        packed_parts = []
        n_rows = 0
        for blocks, scales in parts:
            if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 MXFP4 blocks and scales for {new_name}")
            if len(blocks.shape) != 4 or len(scales.shape) != 3:
                raise ValueError(f"Unexpected MXFP4 tensor rank for {new_name}")
            if blocks.device != blocks0.device or scales.device != blocks.device:
                raise ValueError(f"MXFP4 parts are on different devices for {new_name}")
            if blocks.shape[:3] != scales.shape or blocks.shape[-1] != 16:
                raise ValueError(f"Mismatched MXFP4 blocks and scales for {new_name}")
            if blocks.shape != blocks0.shape:
                raise ValueError(f"Mismatched MXFP4 part shapes for {new_name}")
            packed_parts.append(torch.concat((scales.unsqueeze(-1), self.transform_nibble_layout(blocks)), dim=-1))
            n_rows += blocks.shape[1]

        new_data = packed_parts[0] if len(packed_parts) == 1 else torch.concat(packed_parts, dim=1)
        expected_shape = (blocks0.shape[0], n_rows, blocks0.shape[2], blocks0.shape[3] + 1)
        if new_data.shape != expected_shape:
            raise ValueError(f"Unexpected repacked MXFP4 shape for {new_name}: {tuple(new_data.shape)}")

        new_shape = [new_data.shape[0], new_data.shape[1], new_data.shape[2] * 32]
        logger.info(f"Repacked {new_name} with shape {new_shape} and quantization MXFP4")
        new_data = new_data.view(new_data.shape[0], new_data.shape[1], new_data.shape[2] * new_data.shape[3])
        new_data = new_data.numpy()
        self.gguf_writer.add_tensor(new_name, new_data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

    def repack_mxfp4(self, new_name: str, blocks: Tensor, scales: Tensor):
        self.repack_mxfp4_parts(new_name, ((blocks, scales),))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        pending: dict[str, dict[str, Tensor]] = {}

        for name, data_torch in self.get_tensors():
            suffix = None
            if name.endswith("_blocks"):
                suffix = "blocks"
            elif name.endswith("_scales"):
                suffix = "scales"

            if suffix is None or "mlp.experts." not in name or not name.endswith(
                    ("down_proj_" + suffix, "gate_up_proj_" + suffix)):
                continue

            key = name.removesuffix("_" + suffix)
            pair = pending.setdefault(key, {})
            if suffix in pair:
                raise ValueError(f"Duplicate MXFP4 {suffix} tensor for {key}")
            pair[suffix] = data_torch
            if set(pair) != {"blocks", "scales"}:
                continue

            blocks = pair["blocks"]
            scales = pair["scales"]
            del pending[key]

            if key.endswith("down_proj"):
                new_name = self.map_tensor_name(key + ".weight")
                self.repack_mxfp4(new_name, blocks, scales)
                continue

            if blocks.shape[1] % 2 != 0 or scales.shape[1] % 2 != 0:
                raise ValueError(f"Expected interleaved gate/up rows for {key}")
            gate_blocks, up_blocks = blocks[:, ::2, :, :], blocks[:, 1::2, :, :]
            gate_scales, up_scales = scales[:, ::2, :], scales[:, 1::2, :]
            if gate_blocks.shape != up_blocks.shape or gate_scales.shape != up_scales.shape:
                raise ValueError(f"Mismatched gate/up MXFP4 shapes for {key}")

            if self.fuse_gate_up_exps:
                new_name = self.map_tensor_name(key + ".weight")
                self.repack_mxfp4_parts(new_name, ((gate_blocks, gate_scales), (up_blocks, up_scales)))
            else:
                new_name_gate = self.map_tensor_name(key.replace("gate_up_proj", "gate_proj") + ".weight")
                new_name_up = self.map_tensor_name(key.replace("gate_up_proj", "up_proj") + ".weight")
                self.repack_mxfp4(new_name_gate, gate_blocks, gate_scales)
                self.repack_mxfp4(new_name_up, up_blocks, up_scales)

        if pending:
            missing = ", ".join(
                f"{key}: {sorted({'blocks', 'scales'}.difference(pair))}" for key, pair in sorted(pending.items()))
            raise ValueError(f"Incomplete GPT-OSS MXFP4 tensor pairs: {missing}")
        return []

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "sinks" in name:
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # correct naming for down_proj
        if "down_proj" in name:
            if name.endswith("_bias"):
                name = name.replace("down_proj_bias", "down_proj.bias")
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name = name.replace("down_proj", "down_proj.weight")
                data_torch = data_torch.transpose(-1, -2)
            else:
                # otherwise, it should already be repacked to ggml MXFP4 format
                return

        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            if name.endswith("_bias"):
                if data_torch.shape[-1] % 2 != 0:
                    raise ValueError(f"Expected interleaved gate/up bias rows for {name}")
                gate_proj_bias, up_proj_bias = data_torch[..., ::2], data_torch[..., 1::2]
                if gate_proj_bias.shape != up_proj_bias.shape:
                    raise ValueError(f"Mismatched gate/up bias shapes for {name}")
                if self.fuse_gate_up_exps:
                    name = name.replace("gate_up_proj_bias", "gate_up_proj.bias")
                    data_torch = torch.cat((gate_proj_bias, up_proj_bias), dim=-1)
                    yield from super().modify_tensors(data_torch, name, bid)
                else:
                    name_up = name.replace("gate_up_proj_bias", "up_proj.bias")
                    name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
                    yield from super().modify_tensors(gate_proj_bias, name_gate, bid)
                    yield from super().modify_tensors(up_proj_bias, name_up, bid)
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                data_torch = data_torch.transpose(-1, -2)
                if data_torch.shape[1] % 2 != 0:
                    raise ValueError(f"Expected interleaved gate/up weight rows for {name}")
                gate_proj_weight, up_proj_weight = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                if gate_proj_weight.shape != up_proj_weight.shape:
                    raise ValueError(f"Mismatched gate/up weight shapes for {name}")
                if self.fuse_gate_up_exps:
                    name = name.replace("gate_up_proj", "gate_up_proj.weight")
                    data_torch = torch.cat((gate_proj_weight, up_proj_weight), dim=1)
                    yield from super().modify_tensors(data_torch, name, bid)
                else:
                    name_up = name.replace("gate_up_proj", "up_proj.weight")
                    name_gate = name.replace("gate_up_proj", "gate_proj.weight")
                    yield from super().modify_tensors(gate_proj_weight, name_gate, bid)
                    yield from super().modify_tensors(up_proj_weight, name_up, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size"])
