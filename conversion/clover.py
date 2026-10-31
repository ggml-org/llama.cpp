from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from .base import ModelBase, gguf
from .qwen import _LinearAttentionVReorderBase

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("CloverMoeModelLM")
@ModelBase.example("Diffuspace/Clover-35B-A3B-Preview-1024")
class CloverMoeModel(_LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.CLOVER_MOE
    supports_mtp_export = False

    def __init__(self, *args, **kwargs):
        self.no_mtp = True
        super().__init__(*args, **kwargs)

    def set_vocab(self):
        from transformers import AutoConfig, PretrainedConfig

        class CloverTokenizerConfig(PretrainedConfig):
            model_type = "clover_moe"

        AutoConfig.register("clover_moe", CloverTokenizerConfig, exist_ok=True)
        self._set_vocab_gpt2()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.endswith(".conv_right_weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_bool("clover_moe.moe_block_coreset_enabled", self.hparams["moe_block_coreset_enabled"])
        self.gguf_writer.add_float32("clover_moe.moe_block_coreset_top_p", self.hparams["moe_block_coreset_top_p"])
        self.gguf_writer.add_uint32("clover_moe.block_size", self.hparams["block_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log"):
            # Clover stores raw A_log; GPU exponentiation must match the reference.
            nk = self.hparams["linear_num_key_heads"]
            nv = self.hparams["linear_num_value_heads"]
            original = data_torch.float()
            if nk != nv:
                original = self._reorder_v_heads(original.unsqueeze(-1), 0, nk, nv // nk, 1).squeeze(-1)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.SSM_A, bid, ""), original
            return
        extras = {
            "conv_right_weight": (gguf.MODEL_TENSOR.SSM_CONV1D, ".weight_right"),
            "bd_softmax_logit_scale": (gguf.MODEL_TENSOR.SSM_BD_LOGIT_SCALE, ".weight"),
            "bd_softmax_history_bias": (gguf.MODEL_TENSOR.SSM_BD_HISTORY, ".bias"),
            "bd_softmax_history_c": (gguf.MODEL_TENSOR.SSM_BD_HISTORY, ".weight"),
        }
        suffix = name.rsplit(".", 1)[-1]
        if suffix in extras:
            assert bid is not None
            nk = self.hparams["linear_num_key_heads"]
            nv = self.hparams["linear_num_value_heads"]
            if suffix == "conv_right_weight":
                data_torch = data_torch.squeeze(1)
                qk = 2 * nk * self.hparams["linear_key_head_dim"]
                if nk != nv:
                    import torch
                    v = self._reorder_v_heads(data_torch[qk:], 0, nk, nv // nk, self.hparams["linear_value_head_dim"])
                    data_torch = torch.cat((data_torch[:qk], v), dim=0)
            elif nk != nv:
                data_torch = self._reorder_v_heads(data_torch.unsqueeze(-1), 0, nk, nv // nk, 1).squeeze(-1)
            tensor, tensor_suffix = extras[suffix]
            if tensor == gguf.MODEL_TENSOR.SSM_BD_HISTORY:
                # Export b + c * log(block_length / 32) as a conventional weight/bias pair.
                name = name.rsplit(".", 1)[0] + ".bd_softmax_history" + tensor_suffix
                yield self.map_tensor_name(name), data_torch
            else:
                yield self.format_tensor_name(tensor, bid, tensor_suffix), data_torch
            return

        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch.float()
        yield from super().modify_tensors(data_torch, name, bid)
