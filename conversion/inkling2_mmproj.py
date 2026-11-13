from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf


@ModelBase.register("Inkling2ForConditionalGeneration")
class Inkling2MmprojModel(MmprojModel):
    """Perception-Encoder style vision tower and its 2x2 adapter."""

    has_vision_encoder = True
    has_audio_encoder = False

    _IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _IMAGE_STD = [0.26862954, 0.2613026, 0.2757771]

    def dequant_model(self):
        return

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams_vision
        assert hp is not None
        expected = {
            "temporal_patch_size": 2,
            "native_patch_size": 20,
            "adapter_merge_size": 2,
            "projector_bias": False,
        }
        for key, want in expected.items():
            got = hp.get(key)
            if got != want:
                raise NotImplementedError(f"Inkling2 mmproj requires vision_config.{key}={want!r}, got {got!r}")
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio",
                    "posemb_grid_size", "projection_hidden_dim", "layer_norm_eps"):
            if key not in hp:
                raise NotImplementedError(f"Inkling2 mmproj requires vision_config.{key}")
        n_ff = hp["hidden_size"] * hp["mlp_ratio"]
        if n_ff != int(n_ff):
            raise NotImplementedError("Inkling2 mmproj requires an integer MLP width")
        self.n_ff = int(n_ff)

    def set_gguf_parameters(self):
        hp = self.hparams_vision
        assert hp is not None
        patch = hp["native_patch_size"]

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.INKLING2)
        self.gguf_writer.add_vision_projection_dim(self.n_embd_text)
        self.gguf_writer.add_vision_image_size(hp["posemb_grid_size"] * patch)
        self.gguf_writer.add_vision_patch_size(patch)
        self.gguf_writer.add_vision_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_vision_feed_forward_length(self.n_ff)
        self.gguf_writer.add_vision_block_count(hp["num_hidden_layers"])
        self.gguf_writer.add_vision_head_count(hp["num_attention_heads"])
        self.gguf_writer.add_vision_attention_layernorm_eps(hp["layer_norm_eps"])
        self.gguf_writer.add_vision_projector_scale_factor(hp["adapter_merge_size"])
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_image_mean(self._IMAGE_MEAN)
        self.gguf_writer.add_vision_image_std(self._IMAGE_STD)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith("model.visual."):
            return None
        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        del bid
        hp = self.hparams_vision
        assert hp is not None
        local = name.removeprefix("model.visual.")

        if local == "proj_in.weight":
            yield "mm.up.weight", data_torch
            return
        if local == "proj_out.weight":
            yield "mm.down.weight", data_torch
            return

        prefix = "backbone."
        if not local.startswith(prefix):
            raise ValueError(f"unexpected Inkling2 vision tensor {name!r}")
        local = local.removeprefix(prefix)

        if local == "embeddings.patch_embedding.weight":
            p = hp["native_patch_size"]
            n_out = data_torch.shape[0]
            assert data_torch.shape[1] == 3 * p * p
            yield "v.patch_embd.weight", data_torch.reshape(n_out, p, p, 3).permute(0, 3, 1, 2).contiguous()
            return
        if local == "embeddings.position_embedding":
            g = hp["posemb_grid_size"]
            assert data_torch.shape[0] == g * g
            yield "v.position_embd.weight", data_torch.reshape(g, g, -1)
            return
        if local == "embeddings.class_embedding":
            yield "v.class_embd", data_torch
            return
        if local == "embeddings.cls_position_embedding":
            yield "v.class_pos_embd", data_torch
            return
        if local in ("ln_pre.weight", "ln_pre.bias"):
            yield "v.pre_ln." + local.split(".")[-1], data_torch
            return

        parts = local.split(".")
        if parts[0] == "layers" and len(parts) >= 4 and parts[1].isdigit():
            il = int(parts[1])
            sub = ".".join(parts[2:])
            names = {
                "ln_1.weight": "ln1.weight", "ln_1.bias": "ln1.bias",
                "ln_2.weight": "ln2.weight", "ln_2.bias": "ln2.bias",
                "ls_1.gamma": "ls1.weight", "ls_2.gamma": "ls2.weight",
                "self_attn.q_proj.weight": "attn_q.weight", "self_attn.q_proj.bias": "attn_q.bias",
                "self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_proj.bias": "attn_k.bias",
                "self_attn.v_proj.weight": "attn_v.weight", "self_attn.v_proj.bias": "attn_v.bias",
                "self_attn.out_proj.weight": "attn_out.weight", "self_attn.out_proj.bias": "attn_out.bias",
                "mlp.c_fc.weight": "ffn_up.weight", "mlp.c_fc.bias": "ffn_up.bias",
                "mlp.c_proj.weight": "ffn_down.weight", "mlp.c_proj.bias": "ffn_down.bias",
            }
            if sub in names:
                yield f"v.blk.{il}.{names[sub]}", data_torch
                return

        raise ValueError(f"unexpected Inkling2 vision tensor {name!r}")
