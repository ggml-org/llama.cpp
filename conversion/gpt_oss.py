from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GptOssForCausalLM")
class GptOssModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPT_OSS

    # TODO: remove once MXFP4 is supported more generally
    def dequant_model(self):
        quant_config = self.hparams.get("quantization_config")
        if quant_config is not None and quant_config.get("quant_method") == "mxfp4":
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

    def repack_mxfp4(self, new_name: str, blocks: Tensor, scales: Tensor):
        assert blocks.dtype == torch.uint8
        assert scales.dtype == torch.uint8
        scales = scales.unsqueeze(-1)
        assert len(blocks.shape) == 4
        assert len(scales.shape) == 4
        blocks = self.transform_nibble_layout(blocks)
        new_data = torch.concat((scales, blocks), dim=-1)
        new_shape = [new_data.shape[0], new_data.shape[1], new_data.shape[2] * 32]
        logger.info(f"Repacked {new_name} with shape {new_shape} and quantization MXFP4")
        # flatten last dim
        new_data = new_data.view(new_data.shape[0], new_data.shape[1], new_data.shape[2] * new_data.shape[3])
        new_data = new_data.numpy()
        self.gguf_writer.add_tensor(new_name, new_data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        blocks0: Tensor = torch.zeros(1)
        blocks1: Tensor = torch.zeros(1)
        # we assume that tensors are loaded in the correct order
        for name, data_torch in self.get_tensors():
            if "mlp.experts.down_proj_blocks" in name:
                blocks0 = data_torch
            elif "mlp.experts.down_proj_scales" in name:
                new_name = self.map_tensor_name(name.replace("_scales", ".weight"))
                self.repack_mxfp4(new_name, blocks0, data_torch)
            elif "mlp.experts.gate_up_proj_blocks" in name:
                blocks0, blocks1 = data_torch[:, ::2, :, :], data_torch[:, 1::2, :, :]
            elif "mlp.experts.gate_up_proj_scales" in name:
                scales0, scales1 = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                new_name_gate = self.map_tensor_name(name.replace("gate_up_proj_scales", "gate_proj.weight"))
                new_name_up = self.map_tensor_name(name.replace("gate_up_proj_scales", "up_proj.weight"))
                self.repack_mxfp4(new_name_gate, blocks0, scales0)
                self.repack_mxfp4(new_name_up, blocks1, scales1)
        return []

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if "sinks" in name:
            name += ".weight"
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
                return []
        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            if name.endswith("_bias"):
                name_up = name.replace("gate_up_proj_bias", "up_proj.bias")
                name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
                gate_proj_bias, up_proj_bias = data_torch[..., ::2], data_torch[..., 1::2]
                return [
                    (self.map_tensor_name(name_gate), gate_proj_bias),
                    (self.map_tensor_name(name_up), up_proj_bias)
                ]
            elif "_blocks" not in name and "_scales" not in name:
                logger.warning(f"{name} is not in MXFP4, performance may be degraded")
                name_up = name.replace("gate_up_proj", "up_proj.weight")
                name_gate = name.replace("gate_up_proj", "gate_proj.weight")
                data_torch = data_torch.transpose(-1, -2)
                gate_proj_weight, up_proj_weight = data_torch[:, ::2, :], data_torch[:, 1::2, :]
                return [
                    (self.map_tensor_name(name_gate), gate_proj_weight),
                    (self.map_tensor_name(name_up), up_proj_weight)
                ]
            else:
                # otherwise, it should already be repacked to ggml MXFP4 format
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size"])
        rope_scaling = self.hparams.get("rope_scaling") or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        assert rope_type == "yarn", f"GPT-OSS only supports yarn rope scaling, got {rope_type}"
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
        self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling.get("original_max_position_embeddings", 4096))
