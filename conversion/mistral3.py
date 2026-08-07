from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

from .deepseek import DeepseekV2Model
from .llama import LlamaModel


@ModelBase.register(
    "Mistral3ForConditionalGeneration",
    "Ministral3ForCausalLM",
)
class Mistral3Model(TextModel):
    class Ministral3Model(LlamaModel):
        model_arch = gguf.MODEL_ARCH.MISTRAL3

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            rope_params = self.rope_parameters
            if self.hparams.get("model_type") == "ministral3":
                assert rope_params, "ministral3 must have 'rope_parameters' config"
                assert rope_params["rope_type"] == "yarn", "ministral3 rope_type must be 'yarn'"
                self.gguf_writer.add_rope_scaling_yarn_log_mul(rope_params["mscale_all_dim"])
                self.gguf_writer.add_attn_temperature_scale(rope_params["llama_4_scaling_beta"])

    class Mistral4Model(DeepseekV2Model):
        model_arch = gguf.MODEL_ARCH.MISTRAL4
        skip_mtp = False # model contains no MTP layers, so no need to skip
        merge_expert = False # experts are already stacked as 3D

        def modify_tensors(self, data_torch, name, bid):
            if name.endswith(".down_proj") or name.endswith(".gate_up_proj"):
                name = name + ".weight"
            yield from super().modify_tensors(data_torch, name, bid)

    model_arch = gguf.MODEL_ARCH.MISTRAL3 # unused
    impl: TextModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams.get("model_type") == "mistral4":
            self.impl = Mistral3Model.Mistral4Model(*args, **kwargs)
        else:
            self.impl = Mistral3Model.Ministral3Model(*args, **kwargs)

    def set_vocab(self):
        self.impl.set_vocab()

    def set_gguf_parameters(self):
        self.impl.set_gguf_parameters()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        yield from self.impl.modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        self.impl.prepare_tensors()

    def write_vocab(self):
        self.impl.write_vocab()

    def write(self):
        self.impl.write()
