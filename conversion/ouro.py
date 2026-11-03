from __future__ import annotations

from .base import ModelBase, gguf, logger
from .llama import LlamaModel


@ModelBase.register("OuroForCausalLM")
@ModelBase.example("ByteDance/Ouro-1.4B")
class OuroModel(LlamaModel):
    """Ouro (ByteDance Seed) — a looped LM: the whole stack runs `total_ut_steps` times with shared
    weights, and each (loop, layer) pair keeps its own KV slot. llama.cpp represents that by
    unrolling to n_layer * num_loops logical layers, so only the loop count has to be recorded.

    Two things differ from a plain Llama conversion:

    * the block uses sandwich norms, and the HF names collide with the standard ones —
      `post_attention_layernorm` here is the PRE-FFN norm (Llama convention), while the norms
      applied to the sublayer outputs are the `_2` suffixed ones. The generic tensor map would
      route `post_attention_layernorm` to ATTN_POST_NORM (its gemma2/olmo2 meaning), so the four
      norms are mapped explicitly below instead.
    * `model.early_exit_gate` is dropped. It selects which already-computed loop feeds the LM head
      rather than changing what is computed, and with the shipped early_exit_threshold of 1.0 it
      never fires at all, so a fixed-depth graph reproduces the reference exactly. It must be
      dropped explicitly: the converter raises on any tensor it cannot map.
    """

    model_arch = gguf.MODEL_ARCH.OURO
    undo_permute = True

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        n_loops = int(self.hparams.get("total_ut_steps", 1) or 1)
        if n_loops < 1:
            n_loops = 1
        self.gguf_writer.add_num_loops(n_loops)
        logger.info(f"gguf: num_loops (total_ut_steps) = {n_loops}")

        # The reference applies model.norm at the end of every loop, including the last, so the
        # between-loop norm must run: skip_loop_final_norm stays False.
        self.gguf_writer.add_skip_loop_final_norm(False)

        # The graph always runs all num_loops steps. That matches the reference only while the exit
        # gate never fires, which is what early_exit_threshold = 1.0 guarantees (a sigmoid cannot
        # reach 1.0 in bf16 for any logit these weights produce). A checkpoint shipping a lower
        # threshold would exit early in transformers and would NOT match this conversion.
        threshold = self.hparams.get("early_exit_threshold", 1.0)
        if threshold is None or float(threshold) < 1.0:
            logger.warning(
                f"early_exit_threshold={threshold} < 1.0: the reference may exit before "
                f"{n_loops} loops, but llama.cpp always runs all of them; outputs will differ")

    def modify_tensors(self, data_torch, name, bid):
        # The early-exit gate is a readout-only head; it never feeds the next loop.
        if name.startswith("model.early_exit_gate"):
            logger.info(f"gguf: skipping readout-only tensor {name}")
            return []

        # Map the four block norms explicitly (see the class docstring for why).
        if bid is not None:
            suffix = name.split(f"layers.{bid}.", 1)[-1]
            explicit = {
                "input_layernorm.weight":            gguf.MODEL_TENSOR.ATTN_NORM,
                "input_layernorm_2.weight":          gguf.MODEL_TENSOR.ATTN_POST_NORM,
                "post_attention_layernorm.weight":   gguf.MODEL_TENSOR.FFN_NORM,
                "post_attention_layernorm_2.weight": gguf.MODEL_TENSOR.FFN_POST_NORM,
            }.get(suffix)
            if explicit is not None:
                return [(self.format_tensor_name(explicit, bid), data_torch)]

        return super().modify_tensors(data_torch, name, bid)
