from __future__ import annotations

import math

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("RicursiveForCausalLM")
@ModelBase.example("ricursiveAI/Ripple-65B-Preview")
class RicursiveModel(TextModel):
    model_arch = gguf.MODEL_ARCH.RICURSIVE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams

        if hp.get("quantization_config") is not None or (self.dir_model / "hf_quant_config.json").is_file():
            raise ValueError("Ricursive: quantized checkpoints are not supported")
        if (act := hp.get("hidden_act", "silu")) != "silu":
            raise ValueError(f"Ricursive: unsupported hidden_act {act!r}")

        if (layer_types := hp.get("layer_types")) is None:
            # RicursiveConfig default: every 5th layer is full attention
            layer_types = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(self.block_count)]
        if len(layer_types) != self.block_count:
            raise ValueError(f"Ricursive: layer_types has {len(layer_types)} entries, expected {self.block_count}")
        if unknown := set(layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError(f"Ricursive: unsupported layer_types {sorted(unknown)}")
        self.is_swa = [t == "sliding_attention" for t in layer_types]
        # RicursiveConfig discards the stored hybrid_layer_pattern and derives it from layer_types
        if (pattern := hp.get("hybrid_layer_pattern")) is not None and pattern != [int(swa) for swa in self.is_swa]:
            logger.warning("Ricursive: hybrid_layer_pattern does not match layer_types, ignoring it")
        if not any(self.is_swa):
            raise ValueError("Ricursive: layer_types has no sliding_attention layer, all-full-attention models are not supported")

        if not hp.get("gate_full_precision", True):
            logger.warning("Ricursive: gate_full_precision is false, llama.cpp computes the attention gate in f32")

        self.use_valres = hp.get("use_valres", False)
        self.use_gate = hp.get("use_gated_attention", True)
        names = list(self.model_tensors)
        if bias := [name for name in names if name.endswith("bias")]:
            raise ValueError(f"Ricursive: bias tensors are not supported, found {bias[0]!r}")
        for enabled, suffix in ((self.use_valres, ".self_attn.value_res_lambda"), (self.use_gate, ".self_attn.g_proj.weight")):
            if enabled and sum(name.endswith(suffix) for name in names) != self.block_count:
                raise ValueError(f"Ricursive: expected one {suffix!r} tensor per layer")

    def set_vocab(self):
        self._set_vocab_gpt2()
        # tokenizer_config.json has a short template without tools, transformers uses this file instead
        if (tmpl_file := self.dir_model / "chat_template.jinja").is_file():
            self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
            self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_sliding_window(hp["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(self.is_swa)
        # sliding layers use RoPE, full attention layers use no positional encoding
        self.gguf_writer.add_rope_pattern(self.is_swa)

        if (embed_scale := hp.get("embed_scale")) is not None:
            if embed_scale == 0.0:
                raise ValueError(f"Ricursive: unsupported embed_scale {embed_scale!r}")
            self.gguf_writer.add_embedding_scale(embed_scale)
        if clamp := hp.get("ffn_clamp_threshold"):
            if clamp < 0 or not math.isfinite(clamp):
                raise ValueError(f"Ricursive: unsupported ffn_clamp_threshold {clamp!r}")
            self.gguf_writer.add_swiglu_clamp(clamp)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".value_res_lambda"):
            if not self.use_valres:
                return
            name += ".weight"
        elif name.endswith(".g_proj.weight") and not self.use_gate:
            return
        yield from super().modify_tensors(data_torch, name, bid)
