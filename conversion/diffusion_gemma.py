from __future__ import annotations

import json
from typing import Iterable

from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes
from .gemma import Gemma4Model
import gguf


@ModelBase.register("DiffusionGemma4ModelForBlockDiffusion", "DiffusionGemmaForBlockDiffusion")
class DiffusionGemmaModel(Gemma4Model):
    model_arch = gguf.MODEL_ARCH.DIFFUSION_GEMMA

    def _create_vocab_sentencepiece(self):
        tokens, scores, toktypes = super()._create_vocab_sentencepiece()

        def looks_control(s: str) -> bool:
            return (s in ("<s>", "</s>")
                    or (s.startswith("<|") and s.endswith(">"))
                    or (s.startswith("<") and s.endswith("|>")))

        for i, tok in enumerate(tokens):
            s = tok.decode("utf-8", "ignore") if isinstance(tok, (bytes, bytearray)) else str(tok)
            if toktypes[i] in (SentencePieceTokenTypes.NORMAL, SentencePieceTokenTypes.USER_DEFINED) and looks_control(s):
                toktypes[i] = SentencePieceTokenTypes.CONTROL
        return tokens, scores, toktypes

    def set_gguf_parameters(self):
        self.hparams.setdefault("num_kv_shared_layers", 0)
        self.hparams.setdefault("hidden_size_per_layer_input", 0)

        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)

        canvas_length = self.find_hparam(["canvas_length"], optional=False)
        if canvas_length is None or int(canvas_length) <= 0:
            raise ValueError("DiffusionGemma conversion requires a positive root canvas_length")
        self.gguf_writer.add_diffusion_canvas_length(int(canvas_length))

        gen_cfg_path = self.dir_model / "generation_config.json"
        if gen_cfg_path.is_file():
            with open(gen_cfg_path, encoding="utf-8") as f:
                gen_cfg = json.load(f)
            sampler_cfg = gen_cfg.get("sampler_config", {})
            if "max_denoising_steps" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_max_steps(int(gen_cfg["max_denoising_steps"]))
            if "t_min" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_t_min(float(gen_cfg["t_min"]))
            if "t_max" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_t_max(float(gen_cfg["t_max"]))
            if "entropy_bound" in sampler_cfg:
                self.gguf_writer.add_diffusion_eb_entropy_bound(float(sampler_cfg["entropy_bound"]))
            if "stability_threshold" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_stability_threshold(int(gen_cfg["stability_threshold"]))
            if "confidence_threshold" in gen_cfg:
                self.gguf_writer.add_diffusion_eb_confidence_threshold(float(gen_cfg["confidence_threshold"]))

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item

        if name.endswith("layer_scalar"):
            name = name + ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "vision" in name or "embed_vision" in name:
            return

        if name.startswith("model.encoder.layers.") and "layer_scalar" in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ENC_LAYER_OUT_SCALE, bid), data_torch)
            return

        if name.startswith("model.encoder."):
            return

        if name.startswith("model.decoder.self_conditioning."):
            sub = name[len("model.decoder.self_conditioning."):]
            sc_map = {
                "pre_norm.weight":  gguf.MODEL_TENSOR.SC_PRE_NORM,
                "gate_proj.weight": gguf.MODEL_TENSOR.SC_GATE,
                "up_proj.weight":   gguf.MODEL_TENSOR.SC_UP,
                "down_proj.weight": gguf.MODEL_TENSOR.SC_DOWN,
            }
            if sub in sc_map:
                yield (self.format_tensor_name(sc_map[sub]), data_torch)
            return

        if name.startswith("model.decoder."):
            name = "model." + name[len("model.decoder."):]

        yield from super().modify_tensors(data_torch, name, bid)
