from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("InternVisionModel")
class InternVisionModel(MmprojModel):

    min_dynamic_tiles: int = 0
    max_dynamic_tiles: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.min_dynamic_tiles = self.global_config.get("min_dynamic_patch", 0)
        self.max_dynamic_tiles = self.global_config.get("max_dynamic_patch", 0)

    def set_gguf_parameters(self):
        assert self.hparams_vision is not None
        if isinstance(self.hparams_vision['image_size'], list):
            self.hparams_vision['image_size'] = self.hparams_vision['image_size'][0]
        if isinstance(self.hparams_vision['patch_size'], list):
            self.hparams_vision['patch_size'] = self.hparams_vision['patch_size'][0]
        super().set_gguf_parameters()

        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.INTERNVL)
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams["layer_norm_eps"])
        # hidden_act
        if hparams["hidden_act"] == "silu":
            self.gguf_writer.add_vision_use_silu(True)
        elif hparams["hidden_act"] == "gelu":
            self.gguf_writer.add_vision_use_gelu(True)
        else:
            raise ValueError(f"Unsupported hidden_act: {hparams['hidden_act']}")
        # downsample_ratio
        downsample_ratio = self.global_config.get("downsample_ratio")
        assert downsample_ratio is not None
        self.gguf_writer.add_vision_projector_scale_factor(int(1.0 / downsample_ratio))
        # older models may not have min/max_dynamic_patch in config
        if self.min_dynamic_tiles > 0:
            self.gguf_writer.add_vision_preproc_min_tiles(self.min_dynamic_tiles)
        if self.max_dynamic_tiles > 0:
            self.gguf_writer.add_vision_preproc_max_tiles(self.max_dynamic_tiles)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".position_embd." in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        vision_prefix = ['vision_model', 'mlp', 'model.vision_tower', 'model.multi_modal_projector']
        if not any([name.startswith(prefix) for prefix in vision_prefix]):
            return None
        # deal with intern-s1 special case
        names_map = {
            "model.multi_modal_projector.layer_norm.bias": "mlp1.0.bias",
            "model.multi_modal_projector.layer_norm.weight": "mlp1.0.weight",
            "model.multi_modal_projector.linear_1.bias": "mlp1.1.bias",
            "model.multi_modal_projector.linear_1.weight": "mlp1.1.weight",
            "model.multi_modal_projector.linear_2.bias": "mlp1.3.bias",
            "model.multi_modal_projector.linear_2.weight": "mlp1.3.weight",
        }
        if name in names_map:
            name = names_map[name]
        # correct name
        if name.startswith("vision_model"):
            name = "vision_tower." + name
        if (".ls" in name or ".lambda_" in name or "position_embedding" in name) and not name.endswith(".weight"):
            name += ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # split QKV tensors if needed
        if ".qkv." in name:
            if data_torch.ndim == 2: # weight
                c3, _ = data_torch.shape
            else: # bias
                c3 = data_torch.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = data_torch[:c]
            wk = data_torch[c: c * 2]
            wv = data_torch[c * 2:]
            yield from super().modify_tensors(wq, name.replace("attn.qkv", "self_attn.q_proj"), bid)
            yield from super().modify_tensors(wk, name.replace("attn.qkv", "self_attn.k_proj"), bid)
            yield from super().modify_tensors(wv, name.replace("attn.qkv", "self_attn.v_proj"), bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
