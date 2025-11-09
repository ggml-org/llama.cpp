from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("JanusForConditionalGeneration")
class JanusProModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA  # reuse Llama arch

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Skip vision, aligner, and generation tensors
        skip_prefixes = (
            'model.vision_model.',
            'model.aligner.',
            'model.vqmodel.',
            'model.generation_embeddings.',
            'model.generation_aligner.',
            'model.generation_head.',
        )
        if name.startswith(skip_prefixes):
            return []
        if name.startswith('model.language_model.'):
            name = name.replace('model.language_model.', 'model.')
        elif name.startswith('language_model.'):
            name = name.replace('language_model.', '')
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("JanusForConditionalGeneration")
class JanusProVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        if "intermediate_size" not in self.hparams_vision:
            mlp_ratio = self.hparams_vision.get("mlp_ratio")
            hidden_size = self.hparams_vision.get("hidden_size")
            if mlp_ratio is not None and hidden_size is not None:
                self.hparams_vision["intermediate_size"] = int(round(hidden_size * mlp_ratio))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.JANUS_PRO)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision.get("layer_norm_eps", 1e-6))
        hidden_act = str(self.hparams_vision.get("hidden_act", "")).lower()
        if hidden_act == "gelu":
            self.gguf_writer.add_vision_use_gelu(True)
        elif hidden_act == "silu":
            self.gguf_writer.add_vision_use_silu(True)

    def _map_aligner_tensor(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:
        """Map aligner tensors to projector format"""
        suffix = ".bias" if name.endswith(".bias") else ".weight"
        if name.startswith("model.aligner."):
            local_name = name[len("model.aligner."):]
        elif name.startswith("aligner."):
            local_name = name[len("aligner."):]
        else:
            raise ValueError(f"Unsupported Janus aligner prefix: {name}")
        if local_name.startswith("fc1."):
            mm_index = 0
        elif local_name.startswith("hidden_layers."):
            parts = local_name.split(".", 2)
            if len(parts) < 3:
                raise ValueError(f"Unexpected Janus aligner tensor name: {name}")
            mm_index = int(parts[1]) + 1
        else:
            raise ValueError(f"Unsupported Janus aligner tensor: {name}")
        tensor_name = self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, mm_index, suffix=suffix)
        return [(tensor_name, data_torch)]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # Skip language model tensors as they will be handled by `JanusProModel`
        if name.startswith(('model.language_model.', 'language_model.')):
            return []
        # Skip generation-related components
        skip_generation_prefixes = (
            'model.vqmodel.',
            'vqmodel.',
            'model.generation_embeddings.',
            'generation_embeddings.',
            'model.generation_aligner.',
            'generation_aligner.',
            'model.generation_head.',
            'generation_head.',
        )
        if name.startswith(skip_generation_prefixes):
            return []
        # Handle aligner tensors
        if name.startswith(('model.aligner.', 'aligner.')):
            return list(self._map_aligner_tensor(data_torch, name))
        # Handle vision tensors
        if name.startswith(('model.vision_model.', 'vision_model.')):
            return [(self.map_tensor_name(name), data_torch)]
        return []
