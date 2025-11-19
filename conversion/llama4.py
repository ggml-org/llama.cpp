from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
class Llama4Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA4
    undo_permute = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # IMPORTANT: the normal "intermediate_size" is renamed to "intermediate_size_mlp", we need to undo this
        self.hparams["intermediate_size_moe"] = self.hparams["intermediate_size"]
        self.hparams["intermediate_size"] = self.hparams["intermediate_size_mlp"]

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_interleave_moe_layer_step(self.hparams["interleave_moe_layer_step"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["intermediate_size_moe"])
        if "layer_types" in self.hparams:
            if all(lt == "full_attention" for lt in self.hparams["layer_types"]):
                # all layers are full attention (for MobileLLM), disable swa
                self.gguf_writer.add_sliding_window(0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("language_model."):
            name = name.replace("language_model.", "")

        # split the gate_up into gate and up
        if "gate_up_proj" in name:
            name_up = name.replace("gate_up_proj", "up_proj.weight")
            name_gate = name.replace("gate_up_proj", "gate_proj.weight")
            dim_half = data_torch.shape[-1] // 2
            gate_proj_weight, up_proj_weight = data_torch.transpose(-1, -2).split(dim_half, dim=-2)
            return [
                (self.map_tensor_name(name_gate), gate_proj_weight),
                (self.map_tensor_name(name_up), up_proj_weight)
            ]

        if name.endswith("down_proj"):
            name += ".weight"
            data_torch = data_torch.transpose(-1, -2)

        if "multi_modal_projector" in name or "vision_model" in name:
            return []
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Llama4ForConditionalGeneration")
class Llama4VisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.LLAMA4)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams["norm_eps"])
        self.gguf_writer.add_vision_projector_scale_factor(int(1.0 / self.hparams["pixel_shuffle_ratio"]))
        assert self.hparams["hidden_act"] == "gelu"
        self.gguf_writer.add_vision_use_gelu(True)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid # unused
        if "multi_modal_projector" in name or "vision_model" in name:
            # process vision tensors
            if "positional_embedding_vlm" in name and ".weight" not in name:
                name += ".weight"
            if "multi_modal_projector.linear_1" in name:
                # despite the name with number postfix, this is a single fully connected layer
                return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_MMPROJ_FC] + '.weight', data_torch)]
            return [(self.map_tensor_name(name), data_torch)]
        return []
