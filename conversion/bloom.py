from __future__ import annotations
import re
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("BloomForCausalLM", "BloomModel")
class BloomModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BLOOM

    def set_gguf_parameters(self):
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        name = re.sub(r'transformer\.', '', name)
        tensors: list[tuple[str, Tensor]] = []
        if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
            # Map bloom-style qkv_linear to gpt-style qkv_linear
            # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
            # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
            qkv_weights = data_torch.reshape((n_head, 3, n_embed // n_head, n_embed))
            data_torch = torch.cat(
                (
                    qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                    qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.weight")
        elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
            qkv_bias = data_torch.reshape((n_head, 3, n_embed // n_head))
            data_torch = torch.cat(
                (
                    qkv_bias[:, 0, :].reshape((n_embed,)),
                    qkv_bias[:, 1, :].reshape((n_embed,)),
                    qkv_bias[:, 2, :].reshape((n_embed,)),
                ),
                dim=0,
            )
            logger.info("re-format attention.linear_qkv.bias")
        tensors.append((self.map_tensor_name(name), data_torch))
        return tensors
