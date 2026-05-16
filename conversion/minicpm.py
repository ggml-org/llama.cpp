from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger

from .llama import LlamaModel
from .qwen import Qwen3_5TextModel


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
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith(("q_proj.weight")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        yield from super().modify_tensors(data_torch, name, bid)


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


# MiniCPM-V 4.6: text tower is Qwen3.5 (linear+full hybrid attention) wrapped under
# `model.language_model.*`; vision tower is SigLIP + a window-attention ViT merger
# + a final DownsampleMLP merger. The same HF arch is registered twice below: once as
# the LM (text mode) and once as the mmproj (vision mode), mirroring the Qwen3-VL setup.

@ModelBase.register("MiniCPMV4_6ForConditionalGeneration")
class MiniCPMV4_6TextModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("model.merger."):
            return None
        # MTP tensors are not used at inference yet; align with Qwen3Next behaviour
        if name.startswith("mtp"):
            return None

        return super().filter_tensors(item)


@ModelBase.register("MiniCPMV4_6ForConditionalGeneration")
class MiniCPMV4_6VisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams_vision is not None:
            # In MiniCPM-V 4.6 `vision_config.image_size` (980) describes the SigLIP
            # positional embedding bucket grid (70 x 70), while the per-slice processing
            # resolution is the preprocessor's `scale_resolution` (typically 448).
            # The CLIP loader in tools/mtmd/clip.cpp consumes `clip.vision.image_size`
            # as the slice size and warmup resolution, so report `scale_resolution` there
            # to match the upstream MiniCPMV4_6ImageProcessorPil slicing rules.
            scale_resolution = self.preprocessor_config.get("scale_resolution")
            if scale_resolution is not None:
                self.hparams_vision["image_size"] = int(scale_resolution)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        # projector type string is consumed by clip_projector_type_from_string() in clip.cpp
        # (mapped to PROJECTOR_TYPE_MINICPMV4_6).
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MINICPMV4_6)

        # ViT merger 2x2 + final merger 2x2 = 4x spatial merge per dimension; used for slice alignment
        self.gguf_writer.add_vision_projector_scale_factor(4)

        # borrow wa_layer_indexes for vit_merger insertion point
        insert_layer_id = int(self.global_config.get(
            "insert_layer_id", self.hparams_vision.get("insert_layer_id", 6)))
        self.gguf_writer.add_vision_wa_layer_indexes([insert_layer_id])

        # SigLIP vision body uses gelu_pytorch_tanh, which matches ggml_gelu (tanh approx).
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(
            self.hparams_vision.get("layer_norm_eps", 1e-6))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # lm_head / MTP -> belong to the LM file
        if name.startswith(("lm_head.", "mtp")):
            return None

        return super().filter_tensors(item)
