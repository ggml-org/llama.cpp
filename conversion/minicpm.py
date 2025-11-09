from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("MiniCPMForCausalLM")
class MiniCPMModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MINICPM

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        embedding_scale = float(self.hparams["scale_emb"])
        self.gguf_writer.add_embedding_scale(embedding_scale)
        logger.info(f"gguf: (minicpm) embedding_scale = {embedding_scale}")
        residual_scale = self.hparams["scale_depth"] / self.hparams["num_hidden_layers"] ** 0.5
        self.gguf_writer.add_residual_scale(residual_scale)
        logger.info(f"gguf: (minicpm) residual_scale = {residual_scale}")
        logit_scale = self.hparams["hidden_size"] / self.hparams["dim_model_base"]
        self.gguf_writer.add_logit_scale(logit_scale)
        logger.info(f"gguf: (minicpm) logit_scale = {logit_scale}")
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "longrope":
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LONGROPE)
            logger.info(f"gguf: (minicpm) rope_scaling_type = {gguf.RopeScalingType.LONGROPE}")

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        rope_dims = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        rope_scaling = self.find_hparam(['rope_scaling'], True)
        if rope_scaling is not None:
            long_factors = rope_scaling.get('long_factor', None)
            short_factors = rope_scaling.get('short_factor', None)
            if long_factors is None or short_factors is None:
                raise KeyError('Missing the required key rope_scaling.long_factor or rope_scaling_short_factor')
            if len(long_factors) != len(short_factors) or len(long_factors) != rope_dims / 2:
                raise ValueError(f'The length of rope long and short factors must be {rope_dims / 2}')
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_LONG), torch.tensor(long_factors, dtype=torch.float32))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_SHORT), torch.tensor(short_factors, dtype=torch.float32))

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        # HF models permute some of the tensors, so we need to undo that
        if name.endswith(("q_proj.weight")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("MiniCPM3ForCausalLM")
class MiniCPM3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINICPM3

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        rope_scaling = self.find_hparam(['rope_scaling'], True)
        if rope_scaling is not None:
            rope_dims = self.hparams["qk_rope_head_dim"]
            long_factors = rope_scaling.get('long_factor', None)
            short_factors = rope_scaling.get('short_factor', None)
            if long_factors is None or short_factors is None:
                raise KeyError('Missing the required key rope_scaling.long_factor or rope_scaling_short_factor')
            if len(long_factors) != len(short_factors) or len(long_factors) != rope_dims / 2:
                raise ValueError(f'The length of rope long and short factors must be {rope_dims / 2}')
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_LONG), torch.tensor(long_factors, dtype=torch.float32))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FACTORS_SHORT), torch.tensor(short_factors, dtype=torch.float32))

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head
        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )
