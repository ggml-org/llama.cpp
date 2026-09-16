from __future__ import annotations

import json
from typing import Iterable

from torch import Tensor

from .base import ModelBase, SentencePieceTokenTypes
from .gemma import Gemma4Model
import gguf


@ModelBase.register("DiffusionGemma4ModelForBlockDiffusion", "DiffusionGemmaForBlockDiffusion")
class DiffusionGemmaModel(Gemma4Model):
    """Block text-diffusion MoE on a Gemma-4 backbone.

    Encoder (causal prefill) and decoder (bidirectional canvas denoising) share all weights except a
    per-layer layer_scalar; the backbone lives under model.decoder.*. Strategy: rewrite model.decoder.<x>
    -> model.<x> so the inherited Gemma4 tensor map handles it, then export the encoder layer_scalars
    (ENC_LAYER_OUT_SCALE) and the self_conditioning gated MLP (SC_*) explicitly. Vision tower ignored;
    lm_head tied to model.decoder.embed_tokens.
    """

    model_arch = gguf.MODEL_ARCH.DIFFUSION_GEMMA

    # TextModel.__init__ merges text_config into root hparams; root-only keys (canvas_length) are preserved.

    def _create_vocab_sentencepiece(self):
        tokens, scores, toktypes = super()._create_vocab_sentencepiece()
        # Some Gemma special tokens ship non-control ('</s>', and tool/channel tokens with asymmetric
        # '<|...>' / '<...|>' brackets the generic heuristic misses); tag them control so the vocab is correct.

        def looks_control(s: str) -> bool:
            return (s in ("<s>", "</s>")
                    or (s.startswith("<|") and s.endswith(">"))    # <|tool_response>, <|...|>
                    or (s.startswith("<") and s.endswith("|>")))   # <tool_response|>, <turn|>
        for i, tok in enumerate(tokens):
            s = tok.decode("utf-8", "ignore") if isinstance(tok, (bytes, bytearray)) else str(tok)
            if toktypes[i] in (SentencePieceTokenTypes.NORMAL, SentencePieceTokenTypes.USER_DEFINED) and looks_control(s):
                toktypes[i] = SentencePieceTokenTypes.CONTROL
        return tokens, scores, toktypes

    def set_gguf_parameters(self):
        # plain Gemma-4 MoE: disable gemma3n-only features (per-layer-input embeddings, KV-sharing)
        self.hparams.setdefault("num_kv_shared_layers", 0)
        self.hparams.setdefault("hidden_size_per_layer_input", 0)

        super().set_gguf_parameters()

        # bidirectional decoder; the forward fills its own region-aware mask
        self.gguf_writer.add_causal_attention(False)

        # canvas_length is required (the runtime splits [prompt | canvas] on it)
        canvas_length = self.find_hparam(["canvas_length"], optional=False)
        if canvas_length is None or int(canvas_length) <= 0:
            raise ValueError("DiffusionGemma conversion requires a positive root canvas_length")
        self.gguf_writer.add_diffusion_canvas_length(int(canvas_length))

        # entropy-bound sampler defaults (the real decoder) from generation_config; missing keys fall back to
        # the runtime's reference defaults, so older configs still convert.
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

        # encoder contributes only layer_scalar buffers; suffix them like decoder scalars (raw 1-D)
        if name.endswith("layer_scalar"):
            name = name + ".weight"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # base filter_tensors strips "language_model.", so encoder tensors arrive as "model.encoder.layers.N.*"

        # drop vision tower entirely (diffusion path is text-only)
        if "vision" in name or "embed_vision" in name:
            return

        # encoder-mode per-layer scalar -> dedicated ENC_LAYER_OUT_SCALE tensor
        if name.startswith("model.encoder.layers.") and "layer_scalar" in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ENC_LAYER_OUT_SCALE, bid), data_torch)
            return

        # ignore any other encoder-only tensors (its backbone weights are tied to the decoder)
        if name.startswith("model.encoder."):
            return

        # decoder-only self-conditioning gated MLP
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

        # remap the backbone (everything else under model.decoder.*) to model.<x> for Gemma4Model
        if name.startswith("model.decoder."):
            name = "model." + name[len("model.decoder."):]

        yield from super().modify_tensors(data_torch, name, bid)
