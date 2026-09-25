from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger

from .llama import LlamaModel
from .qwen import Qwen3_5TextModel


@ModelBase.register("MiniCPMForCausalLM")
@ModelBase.example("openbmb/MiniCPM-2B-sft-bf16")
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

        long_factors = self.rope_parameters.get('long_factor')
        short_factors = self.rope_parameters.get('short_factor')
        if long_factors or short_factors:
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
@ModelBase.example("openbmb/MiniCPM3-4B")
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
        long_factors = self.rope_parameters.get('long_factor')
        short_factors = self.rope_parameters.get('short_factor')
        if long_factors or short_factors:
            rope_dims = self.hparams["qk_rope_head_dim"]

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
@ModelBase.example("openbmb/MiniCPM-V-4_6")
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
@ModelBase.example("openbmb/MiniCPM-V-4_6")
class MiniCPMV4_6VisionModel(MmprojModel):
    projector_type = gguf.VisionProjectorType.MINICPMV4_6
    # fallback for checkpoints whose preprocessor config omits `scale_resolution`
    default_scale_resolution: int | None = None

    def get_downsample_mode(self) -> str:
        return self.preprocessor_config.get("downsample_mode", "16x")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downsample_mode = self.get_downsample_mode()
        if self.downsample_mode not in {"4x", "16x"}:
            raise ValueError(f"Unsupported downsample mode: {self.downsample_mode}")
        if self.downsample_mode == "4x":
            self.model_tensors = {
                name: tensor for name, tensor in self.model_tensors.items()
                if ".vit_merger." not in name
            }

        if self.hparams_vision is not None:
            # In MiniCPM-V 4.6 `vision_config.image_size` (980) describes the SigLIP
            # positional embedding bucket grid (70 x 70), while the per-slice processing
            # resolution is the preprocessor's `scale_resolution` (typically 448).
            # The CLIP loader in tools/mtmd/clip.cpp consumes `clip.vision.image_size`
            # as the slice size and warmup resolution, so report `scale_resolution` there
            # to match the upstream MiniCPMV4_6ImageProcessorPil slicing rules.
            scale_resolution = self.preprocessor_config.get(
                "scale_resolution", self.default_scale_resolution)
            if scale_resolution is not None:
                self.hparams_vision["image_size"] = int(scale_resolution)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        # projector type string is consumed by clip_projector_type_from_string() in clip.cpp
        self.gguf_writer.add_clip_projector_type(self.projector_type)

        self.gguf_writer.add_vision_projector_scale_factor(
            2 if self.downsample_mode == "4x" else 4)

        # llava-uhd slice cap: the reference image processor decides how many slices to
        # cut from this value, so it has to travel with the model
        max_slice_nums = self.preprocessor_config.get("max_slice_nums")
        if max_slice_nums is not None:
            self.gguf_writer.add_vision_max_slice_nums(int(max_slice_nums))

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


# MiniCPM-V 4.7 shares the MiniCPM-V 4.6 stack: a Qwen3.5 text tower wrapped under
# `model.language_model.*` and the same SigLIP + vit_merger + merger vision tower.
# The text tower swaps to the MoE variant when the checkpoint says so, and the
# vision tower keeps the v4.6 graph but reports its own projector type.

@ModelBase.register("MiniCPMV4_7ForConditionalGeneration")
@ModelBase.example("openbmb/MiniCPM-V-4.7")
class MiniCPMV4_7TextModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # Canvas M-RoPE keeps the time component constant across an image group while the
        # cache/attention key has to stay strictly increasing, so the time component is
        # carried by the 4th position slot, which RoPE does not use (sections [11, 11, 10, 0]).
        self.gguf_writer.add_rope_mrope_time_slot(3)

    def __init__(self, dir_model, ftype, fname_out, *, hparams: dict | None = None, **kwargs):
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        text_config = hparams.get("text_config", {})
        if text_config.get("model_type") == "qwen3_5_moe_text":
            self.model_arch = gguf.MODEL_ARCH.QWEN35MOE
        else:
            self.model_arch = gguf.MODEL_ARCH.QWEN35
        super().__init__(dir_model, ftype, fname_out, hparams=hparams, **kwargs)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # MTP tensors are not used at inference yet; align with Qwen3Next behaviour
        if name.startswith("mtp"):
            return None

        return super().filter_tensors(item)


@ModelBase.register("MiniCPMV4_7ForConditionalGeneration")
@ModelBase.example("openbmb/MiniCPM-V-4.7")
class MiniCPMV4_7VisionModel(MiniCPMV4_6VisionModel):
    projector_type = gguf.VisionProjectorType.MINICPMV4_7
    # MiniCPMV4_7ImageProcessorPil default; some 4.7 checkpoints ship no
    # `scale_resolution` (and no `patch_size`) in their preprocessor config
    default_scale_resolution = 448

    def get_downsample_mode(self) -> str:
        # 4.7 moved downsample_mode from the preprocessor config to the model config;
        # an explicit preprocessor value still wins so that a copy of the model dir
        # can export the 4x variant
        return self.preprocessor_config.get(
            "downsample_mode", self.global_config.get("downsample_mode", "16x"))
