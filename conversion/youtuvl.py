from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger


@ModelBase.register("YoutuVLForConditionalGeneration")
class YoutuVLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.hparams_vision["image_size"] = self.hparams_vision.get("image_size", 560)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.YOUTUVL)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))

        # Handle activation function
        hidden_act = str(self.hparams.get("hidden_act", "gelu_pytorch_tanh")).lower()
        if hidden_act in ("gelu", "gelu_pytorch_tanh", "gelu_fast", "gelu_new", "gelu_accurate"):
            self.gguf_writer.add_vision_use_gelu(True)
        elif hidden_act == "silu":
            self.gguf_writer.add_vision_use_silu(True)
        else:
            raise ValueError(f"Unsupported activation function for YOUTUVL: {hidden_act}")

        self.gguf_writer.add_vision_spatial_merge_size(self.hparams.get("spatial_merge_size", 2))

        window_size = self.hparams.get("window_size")
        if window_size is not None:
            self.gguf_writer.add_vision_window_size(window_size)
        # fullatt_block_indexes contains explicit layer indices that use full attention
        # e.g., [2, 5, 8, 11] means layers 2, 5, 8, 11 use full attention
        # All other layers use window attention
        fullatt_block_indexes = self.hparams.get("fullatt_block_indexes")
        assert fullatt_block_indexes is not None, "fullatt_block_indexes is required for youtuvl"
        # Store the explicit layer indices for YoutuVL (irregular pattern approach)
        self.gguf_writer.add_vision_wa_layer_indexes(layers=fullatt_block_indexes)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip language model tensors
        skip_prefixes = ('lm_head.', 'model.layers.', 'model.embed_tokens.', 'model.norm.')
        if name.startswith(skip_prefixes):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Try to map the tensor using TensorNameMap (handles vision encoder and projector)
        try:
            yield from super().modify_tensors(data_torch, name, bid)
        except ValueError:
            # If mapping fails, log warning and skip
            logger.warning(f"Cannot map tensor: {name}")
            return
