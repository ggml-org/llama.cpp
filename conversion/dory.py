from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("DoryForCausalLM")
@ModelBase.example("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
class DoryModel(TextModel):
    """Dory: a looped nGPT stack whose 72 physical modules alternate attention and MLP.

    The checkpoint is flat (`backbone.layers.{i}.mixer.*`), one module per layer, so the HF layer
    index is the GGUF blk index. `mixer.g_proj` is the attention output gate and `mixer.gate_proj`
    is the SwiGLU gate.
    """

    model_arch = gguf.MODEL_ARCH.DORY

    # nGPT parameters are 1-D and carry no `.weight` suffix in the checkpoint
    _param_suffixes = (".sqk", ".s_gate", ".suv_gate", ".suv_up", ".attn_alpha", ".mlp_alpha", ".logit_scale")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = self.hparams

        n_groups = hparams["n_input_layers"] + hparams["n_recurrent_layers"] + hparams["n_output_layers"]
        if n_groups != self.block_count:
            raise ValueError(
                f"Dory: input({hparams['n_input_layers']}) + recurrent({hparams['n_recurrent_layers']})"
                f" + output({hparams['n_output_layers']}) modules != num_hidden_layers({self.block_count})"
            )

        layer_types = hparams["layer_types"]
        if len(layer_types) != self.block_count:
            raise ValueError(f"Dory: layer_types has {len(layer_types)} entries, expected {self.block_count}")
        if unknown := set(layer_types) - {"sliding_attention", "full_attention", "mlp"}:
            raise ValueError(f"Dory: unsupported layer_types {sorted(unknown)}")
        # attention and MLP alternate: even modules attend, odd modules are MLP
        for i, layer_type in enumerate(layer_types):
            if (layer_type == "mlp") != (i % 2 == 1):
                raise ValueError(f"Dory: layer_types[{i}] is {layer_type!r}, expected the attention/MLP alternation")
        if not any(layer_type == "sliding_attention" for layer_type in layer_types):
            raise ValueError("Dory: layer_types has no sliding_attention module")
        self.layer_types = layer_types

        if (act := hparams.get("hidden_act", "silu")) != "silu":
            raise ValueError(f"Dory: unsupported hidden_act {act!r}")

    def set_gguf_parameters(self):
        hparams = self.hparams
        w = self.gguf_writer
        rope = self.rope_parameters

        hidden_size = hparams["hidden_size"]
        head_dim = hparams.get("head_dim") or hidden_size // hparams["num_attention_heads"]

        w.add_block_count(self.block_count)
        w.add_context_length(hparams["max_position_embeddings"])
        w.add_embedding_length(hidden_size)
        w.add_feed_forward_length(hparams["intermediate_size"])
        w.add_head_count(hparams["num_attention_heads"])
        w.add_head_count_kv(hparams["num_key_value_heads"])
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        w.add_rope_dimension_count(head_dim)
        w.add_rope_freq_base(rope["full_attention"]["rope_theta"])
        w.add_sliding_window(hparams["sliding_window"])

        def key(template: str) -> str:
            return template.format(arch=w.arch)

        # dual RoPE: full attention uses freq_base, sliding attention freq_base_2
        w.add_float32(key(gguf.Keys.Rope.FREQ_BASE_2), rope["sliding_attention"]["rope_theta"])
        # 1 = primary base, 2 = secondary base, one entry per physical module (MLP modules carry 1)
        w.add_array(key(gguf.Keys.Rope.PROFILE), [2 if t == "sliding_attention" else 1 for t in self.layer_types])

        w.add_uint32(key(gguf.Keys.LLM.INPUT_LAYER_COUNT), hparams["n_input_layers"])
        w.add_uint32(key(gguf.Keys.LLM.RECURRENT_LAYER_COUNT), hparams["n_recurrent_layers"])
        w.add_uint32(key(gguf.Keys.LLM.OUTPUT_LAYER_COUNT), hparams["n_output_layers"])
        w.add_uint32(key(gguf.Keys.LLM.NUM_LOOPS), hparams["n_recurrent_loops"])
        # one character per physical module: 'S' sliding, '*' full, '-' MLP
        pattern = "".join(
            "S" if t == "sliding_attention" else "*" if t == "full_attention" else "-" for t in self.layer_types
        )
        w.add_string(key(gguf.Keys.LLM.LAYER_PATTERN), pattern)
        w.add_string(key(gguf.Keys.LLM.RECURRENT_KV_CACHE_MODE), hparams["recurrent_kv_cache_mode"])

        # the *_scale tensors hold the raw learned parameters, so the reference fold constants are kept here
        w.add_float32(key(gguf.Keys.LLM.INIT_STD), hparams["initializer_range"])
        w.add_float32(key(gguf.Keys.LLM.ALPHA_INIT), hparams["ngpt_alpha_init"])
        w.add_float32(key(gguf.Keys.LLM.GATE_INIT), hidden_size**0.5)

        w.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(self._param_suffixes):
            name += ".weight"
        yield from super().modify_tensors(data_torch, name, bid)
