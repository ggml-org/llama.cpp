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
        hparams = self.hparams

        if (act := hparams.get("hidden_act", "silu")) != "silu":
            raise ValueError(f"Ricursive: unsupported hidden_act {act!r}")

        # RicursiveConfig derives the layer pattern from layer_types alone, by default full attention iff i % 5 == 4
        if (layer_types := hparams.get("layer_types")) is None:
            layer_types = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(self.block_count)]
        if len(layer_types) != self.block_count:
            raise ValueError(f"Ricursive: layer_types has {len(layer_types)} entries, expected {self.block_count}")
        if unknown := set(layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError(f"Ricursive: unsupported layer_types {sorted(unknown)}")
        self.is_swa = [t == "sliding_attention" for t in layer_types]
        # only sliding layers use RoPE, an all-full-attention model would have no positional encoding
        if not any(self.is_swa):
            raise ValueError("Ricursive: layer_types has no sliding_attention layer, all-full-attention models are not supported")
        # RicursiveConfig pops a stored hybrid_layer_pattern and recomputes it from layer_types
        if (pattern := hparams.get("hybrid_layer_pattern")) is not None and list(pattern) != [int(swa) for swa in self.is_swa]:
            logger.warning("Ricursive: hybrid_layer_pattern does not match layer_types, ignoring it")

        if not hparams.get("gate_full_precision", True):
            logger.warning("Ricursive: gate_full_precision is false, llama.cpp computes the attention gate in f32")

        # RicursiveConfig defaults; the reference ignores the tensors of a disabled feature
        self.use_valres = hparams.get("use_valres", False)
        self.use_gate = hparams.get("use_gated_attention", True)
        names = list(self.model_tensors)
        if bias := [name for name in names if name.endswith("bias")]:
            raise ValueError(f"Ricursive: bias tensors are not supported, found {bias[0]!r}")
        for enabled, suffix in ((self.use_valres, ".self_attn.value_res_lambda"), (self.use_gate, ".self_attn.g_proj.weight")):
            if enabled and sum(name.endswith(suffix) for name in names) != self.block_count:
                raise ValueError(f"Ricursive: expected one {suffix!r} tensor per layer")

    def set_vocab(self):
        self._set_vocab_gpt2()
        # eos_token_id is [<|im_end|>, <|endoftext|>]: register the turn end as EOT so generation stops on it
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            extra = [e for e in eos if e != self.hparams.get("bos_token_id")]
            if extra:
                self.gguf_writer.add_eot_token_id(extra[0])
                logger.info(f"gguf: eot_token_id = {extra[0]}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])

        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(self.is_swa)
        # sliding layers use RoPE, full attention layers use no positional encoding
        self.gguf_writer.add_rope_pattern(self.is_swa)

        if (embed_scale := hparams.get("embed_scale")) is not None:
            if embed_scale == 0:
                raise ValueError(f"Ricursive: unsupported embed_scale {embed_scale!r}")
            self.gguf_writer.add_embedding_scale(float(embed_scale))
        # absent and 0 both mean no clamp (the reference tests `if clamp_threshold:`)
        if (clamp := hparams.get("ffn_clamp_threshold")) is not None:
            if clamp < 0 or not math.isfinite(clamp):
                raise ValueError(f"Ricursive: unsupported ffn_clamp_threshold {clamp!r}")
            self.gguf_writer.add_swiglu_clamp([float(clamp)] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # value_res_lambda has no .weight suffix in the checkpoint
        if name.endswith(".value_res_lambda"):
            if not self.use_valres:
                return
            name += ".weight"
        elif name.endswith(".g_proj.weight") and not self.use_gate:
            return
        yield from super().modify_tensors(data_torch, name, bid)
