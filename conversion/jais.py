from __future__ import annotations
import math
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("JAISLMHeadModel")
class JaisModel(TextModel):
    model_arch = gguf.MODEL_ARCH.JAIS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SwigLU activation
        assert self.hparams["activation_function"] == "swiglu"
        # ALiBi position embedding
        assert self.hparams["position_embedding_type"] == "alibi"
        # Embeddings scale
        self.embeddings_scale = 1.0
        if 'mup_embeddings_scale' in self.hparams:
            self.embeddings_scale = self.hparams['mup_embeddings_scale']
        elif 'embeddings_scale' in self.hparams:
            self.embeddings_scale = self.hparams['embeddings_scale']
        else:
            assert False
        self.width_scale = 1.0
        if 'mup_output_alpha' in self.hparams:
            assert 'mup_width_scale' in self.hparams
            self.width_scale = self.hparams['mup_output_alpha'] * self.hparams['mup_width_scale']
        elif 'width_scale' in self.hparams:
            self.width_scale = self.hparams['width_scale']
        else:
            assert False
        self.max_alibi_bias = 8.0

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(self.hparams["n_inner"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        tensors: list[tuple[str, Tensor]] = []
        # we don't need these
        if name.endswith((".attn.bias")):
            return tensors
        if name.endswith(("relative_pe.slopes")):
            # Calculate max ALiBi bias (this is the inverse of the ALiBi calculation)
            # Some other models has max_alibi_bias spelled out explicitly in the hyperparams,
            # but Jais's PyTorch model simply precalculates the slope values and places them
            # in relative_pes.slopes
            n_head_closest_log2 = 2 ** math.floor(math.log2(self.hparams["n_head"]))
            first_val = float(data_torch[0].item())
            self.max_alibi_bias = -round(math.log2(first_val) * n_head_closest_log2)
            return tensors
        if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_fc2.weight")):
            data_torch = data_torch.transpose(1, 0)
        new_name = self.map_tensor_name(name)
        if new_name == self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD):
            tensors.append((new_name, data_torch * self.embeddings_scale))
        elif new_name == self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT):
            tensors.append((new_name, data_torch * self.width_scale))
        else:
            tensors.append((new_name, data_torch))
        return tensors

    def prepare_tensors(self):
        super().prepare_tensors()
        self.gguf_writer.add_max_alibi_bias(self.max_alibi_bias)
