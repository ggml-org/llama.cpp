from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf, logger


def _is_inkling(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type") in ("inkling_mm_model", "inkling_text")


@ModelBase.register_hparams_loader(_is_inkling)
def _load_inkling_hparams(dir_model: Path):
    # AutoConfig replaces the checkpoint's expert width with the dense width.
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f)


@ModelBase.register("InklingForCausalLM", "InklingForConditionalGeneration")
@ModelBase.example("thinkingmachines/Inkling", "thinkingmachines/Inkling-Small")
class InklingModel(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING

    def __init__(self, dir_model: Path, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)
        hp = self.hparams
        for key, expected in (
            ("hidden_act", "silu"),
            ("gate_activation", "sigmoid"),
            ("q_bias", False),
            ("o_bias", False),
            ("shared_expert_sink", True),
            ("norm_after_topk", True),
            ("use_global_scale", True),
            ("use_embed_norm", True),
            ("use_sconv", True),
            ("use_gate_bias", True),
        ):
            if hp.get(key, expected) != expected:
                raise ValueError(f"Inkling: unsupported {key}={hp[key]!r}")
        if hp.get("final_logit_softcapping") not in (None, 0):
            raise ValueError("Inkling: logit softcapping is unsupported")
        types = hp.get("layer_types")
        if types is None:
            local = hp["local_layer_ids"]
            types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.block_count)]
        if len(types) != self.block_count or set(types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError("Inkling: invalid layer_types")
        self.is_swa = [t == "hybrid_sliding" for t in types]
        logger.warning("Inkling: converting text only; vision, audio and MTP tensors are skipped")

    def dequant_model(self):
        # Inkling 1 checkpoints use only NVFP4 packing, which is unsupported here.
        pass

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, tensor in super().get_tensors():
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise SystemExit("Inkling: NVFP4 checkpoints are unsupported; use BF16, F16 or F32 weights")
            yield name, tensor

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(("model.visual.", "model.audio.", "model.mtp.", "model.vision_tower.", "model.audio_tower.")):
            return None
        name = name.replace("model.llm.", "model.").replace("model.language_model.", "model.")
        name = name.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
        return name, gen

    def get_vocab_base(self):
        tokens, types, _ = super().get_vocab_base()
        n = self.hparams["unpadded_vocab_size"]
        return tokens[:n], types[:n], "inkling"

    def set_vocab(self):
        tokens, types, pre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(pre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(types)
        special = gguf.SpecialVocab(self.dir_model, load_merges=True)
        eos = self.hparams["eos_token_id"]
        special.special_token_ids["eos"] = eos
        special.special_token_ids["eot"] = eos
        special.add_special_token["bos"] = False
        special.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        hp["num_attention_heads"] = [hp["swa_num_attention_heads"] if s else hp["num_attention_heads"] for s in self.is_swa]
        hp["num_key_value_heads"] = [hp["swa_num_key_value_heads"] if s else hp["num_key_value_heads"] for s in self.is_swa]
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_vocab_size(hp["vocab_size"])
        w.add_key_length_swa(hp["swa_head_dim"])
        w.add_value_length_swa(hp["swa_head_dim"])
        w.add_rope_dimension_count(0)
        w.add_sliding_window(hp["sliding_window_size"])
        w.add_sliding_window_pattern(self.is_swa)
        w.add_shortconv_l_cache(self.find_hparam(["conv_kernel_size", "sconv_kernel_size"]))
        w.add_expert_feed_forward_length(self.find_hparam(["moe_intermediate_size", "intermediate_size"]))
        w.add_expert_shared_count(hp["n_shared_experts"])
        w.add_leading_dense_block_count(hp["dense_mlp_idx"])
        w.add_expert_weights_scale(hp["route_scale"])
        w.add_logit_scale(1.0 / hp["logits_mup_width_multiplier"])
        w.add_relative_length(hp["d_rel"])
        w.add_relative_extent(hp["rel_extent"])
        w.add_log_scale_floor(hp["log_scaling_n_floor"])
        w.add_vocab_size_unpadded(hp["unpadded_vocab_size"])
        w.add_log_scale_alpha(hp["log_scaling_alpha"])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(("_conv.weight", "attn_rel_bias.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith((".w13_dn.weight", ".shared_w13_weight", ".w13_weight")):
            gate, up = data_torch[..., 0::2, :].contiguous(), data_torch[..., 1::2, :].contiguous()
            if name.endswith(".w13_weight"):
                data_torch = torch.cat((gate, up), dim=-2)
            else:
                shared = name.endswith(".shared_w13_weight")
                for tensor, value in (
                    (gguf.MODEL_TENSOR.FFN_GATE_SHEXP if shared else gguf.MODEL_TENSOR.FFN_GATE, gate),
                    (gguf.MODEL_TENSOR.FFN_UP_SHEXP if shared else gguf.MODEL_TENSOR.FFN_UP, up),
                ):
                    yield self.format_tensor_name(tensor, bid, ".weight"), value
                return
        if name.endswith((".conv1d.weight", "sconv.weight")) and data_torch.ndim == 3:
            data_torch = data_torch.squeeze(1)
        elif name.endswith(".rel_logits_proj.proj"):
            data_torch = data_torch.T.contiguous()
        new_name = self.tensor_map.get_name(name)
        if new_name is not None:
            yield new_name + (".bias" if name.endswith(".e_score_correction_bias") else ".weight"), data_torch
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("InklingForConditionalGeneration")
class InklingVisionModel(MmprojModel):
    has_audio_encoder = True

    def get_vision_config(self):
        config = dict(self.global_config["vision_config"])
        n_stages = config.get("n_layers", config.get("num_hidden_layers", 4))
        config.update(image_size=40, hidden_size=128, intermediate_size=4800, num_attention_heads=1, n_layers=n_stages - 1)
        return config

    def get_audio_config(self):
        config = dict(self.global_config["audio_config"])
        config.update(hidden_size=self.n_embd_text, intermediate_size=0, num_attention_heads=1, n_layers=0)
        return config

    def set_gguf_parameters(self):
        p = self.preprocessor_config
        p.setdefault("image_mean", [0.48145466, 0.4578275, 0.40821073])
        p.setdefault("image_std", [0.26862954, 0.26130258, 0.27577711])
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.INKLINGA)
        self.gguf_writer.add_audio_num_mel_bins(self.global_config["audio_config"]["n_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.global_config["audio_config"].get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.INKLINGV)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.global_config["vision_config"].get("rms_norm_eps", 1e-6))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        return item if name.startswith(("model.visual.", "model.vision_tower.", "model.audio.", "model.audio_tower.")) else None

    def dequant_model(self):
        # The tower weights are BF16 even when the text model uses NVFP4.
        pass

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, tensor in super().get_tensors():
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise SystemExit("Inkling: NVFP4 checkpoints are unsupported; use BF16, F16 or F32 weights")
            yield name, tensor

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in ("model.visual.layers.linear_3.weight", "model.vision_tower.encoder_layers.3.projection.weight"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, 0, ".weight"), data_torch
            return
        yield from super().modify_tensors(data_torch, name, bid)
