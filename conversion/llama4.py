from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("Llama4ForCausalLM", "Llama4ForConditionalGeneration")
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
                self.gguf_writer.add_expert_swa(False)
            else:
                self.gguf_writer.add_expert_swa(True)
        else:
            self.gguf_writer.add_expert_swa(True)
        self.gguf_writer.add_leading_dense_block_count(self.hparams["leading_dense_block_count"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
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
        del bid  # unused
        valid_prefixes = (
            "multi_modal_projector.",
            "vision_tower.",
            "vision_encoder.",
        )
        if any(name.startswith(prefix) for prefix in valid_prefixes):
            return [(self.map_tensor_name(name), data_torch)]
        return [] # skip other tensors
