from __future__ import annotations

import math
import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, _MISTRAL_COMMON_DATASET_MEAN, _MISTRAL_COMMON_DATASET_STD, gguf

from .qwen import Qwen3Model


@ModelBase.register("StepVLForConditionalGeneration", "Step3p7ForConditionalGeneration")
class Step3VLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None

        if not self.hparams_vision.get("intermediate_size"):
            hidden_size = self.hparams_vision.get("hidden_size") or self.hparams_vision.get("width") or 0
            assert hidden_size > 0
            mlp_ratio = float(self.hparams_vision.get("mlp_ratio", 8960 / 1536))
            self.hparams_vision["intermediate_size"] = int(round(hidden_size * mlp_ratio))

        self.preprocessor_config.setdefault("image_mean", list(_MISTRAL_COMMON_DATASET_MEAN))
        self.preprocessor_config.setdefault("image_std", list(_MISTRAL_COMMON_DATASET_STD))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None

        projector_stride = int(self.global_config.get("understand_projector_stride", -1))
        hidden_size = int(self.hparams_vision.get("hidden_size", self.hparams_vision.get("width", -1)))
        num_layers = int(self.hparams_vision.get("num_hidden_layers", self.hparams_vision.get("layers", -1)))
        assert (projector_stride, int(self.hparams_vision.get("image_size", -1)), hidden_size, num_layers) == (2, 728, 1536, 47), (
            "current Step3-VL conversion path is only validated for Step3-VL-10B"
        )

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.STEP3VL)
        self.gguf_writer.add_vision_attention_layernorm_eps(float(self.hparams_vision.get("layer_norm_eps", 1e-5)))
        self.gguf_writer.add_vision_projector_scale_factor(projector_stride ** 2)
        # 3024 max resize comes from step3-vl-10b processing_step3.py.
        self.gguf_writer.add_vision_preproc_image_size(3024)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".position_embd." in new_name:
            return gguf.GGMLQuantizationType.F32
        if ("mm.0." in new_name or "mm.1." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(("model.", "lm_head.")):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("vision_model.vit_downsampler"):
            match = re.match(r"vision_model\.vit_downsampler(\d+)\.(weight|bias)", name)
            if match is None:
                raise ValueError(f"Unexpected Step3-VL projector tensor {name!r}")

            proj_id = int(match.group(1)) - 1
            suffix = f".{match.group(2)}"
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, proj_id, suffix=suffix), data_torch)
            return

        if name == "vit_large_projector.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ_FC), data_torch)
            return

        if name.startswith("vision_model."):
            if name == "vision_model.positional_embedding":
                name += ".weight"
            elif name.endswith(".gamma") and ".ls_" in name:
                name = name.removesuffix(".gamma") + ".weight"

            name = name.replace("attn.in_proj_weight", "attn.in_proj.weight")
            name = name.replace("attn.in_proj_bias", "attn.in_proj.bias")

            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("StepVLForConditionalGeneration")
class Step3VLTextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3


@ModelBase.register("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration")
class Step35Model(TextModel):
    model_arch = gguf.MODEL_ARCH.STEP35
    supports_mtp_export = True

    # The --mtp / --no-mtp toggles are ModelBase.mtp_only / no_mtp (set in
    # convert_hf_to_gguf.py main()). Unlike Qwen3.5, which stores MTP under a
    # `mtp.*` namespace, Step3.5 appends MTP layers at
    # `model.layers.{num_hidden_layers + i}`, so we filter them by layer index.
    # The trunk layer count is captured before indexing so the classmethod
    # filter_tensors can tell the appended MTP block(s) apart from the trunk.
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NextN/MTP layers are appended past num_hidden_layers; extend the
        # tensor map to cover them so the MTP block's tensors get correctly
        # indexed names. When --no-mtp drops the MTP blocks, fall back to the
        # base num_hidden_layers so we don't reserve unused slots.
        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0))
        if n_nextn > 0 and not self.no_mtp:
            self.block_count += n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # filter_tensors is a classmethod and can't reach self.hparams; stash
        # the trunk layer count here (before indexing runs) so it can detect
        # the appended MTP layers by index.
        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        key = next((k for k in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"] if k in hparams), None)
        type(self)._n_main_layers = hparams.get(key)
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_gguf_parameters(self):
        rope_theta = self.hparams.get("rope_theta")
        if isinstance(rope_theta, list):
            self.hparams["rope_theta"] = float(rope_theta[0])
            self.hparams["local_rope_theta"] = float(rope_theta[1])
            self.rope_parameters["rope_theta"] = self.hparams["rope_theta"]
            self.rope_parameters["sliding_attention"] = {"rope_theta": self.hparams["local_rope_theta"]}

        super().set_gguf_parameters()

        layer_types = self.hparams.get("layer_types") or []
        partial_rotary_factors = self.hparams.get("partial_rotary_factors") or []
        attn_other = self.hparams.get("attention_other_setting") or {}

        n_head_base = self.hparams["num_attention_heads"]
        n_kv_base = self.hparams["num_attention_groups"]

        n_head_swa = attn_other.get("num_attention_heads", n_head_base)
        n_kv_swa = attn_other.get("num_attention_groups", n_kv_base)

        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0))

        # The Step3p5 HF checkpoint stores layer_types/partial_rotary_factors
        # entries for the MTP blocks past num_hidden_layers; preserve them so
        # the MTP layer's attention shape, SWA flag, and partial RoPE dim are
        # set correctly. Pad with full-attention defaults if the checkpoint
        # truncated them.
        def _pad(arr, n, default):
            arr = list(arr)
            if len(arr) < n:
                arr = arr + [default] * (n - len(arr))
            return arr[:n]

        layer_types = _pad(layer_types, self.block_count, "full_attention")
        partial_rotary_factors = _pad(
            partial_rotary_factors,
            self.block_count,
            0.5,  # full_attention default for Step3p5
        )
        assert [1.0 if lt == "sliding_attention" else 0.5 for lt in layer_types] == partial_rotary_factors
        head_arr = [n_head_swa if lt == "sliding_attention" else n_head_base for lt in layer_types]
        kv_arr = [n_kv_swa if lt == "sliding_attention" else n_kv_base for lt in layer_types]
        swa_pat = [lt == "sliding_attention" for lt in layer_types]

        self.gguf_writer.add_head_count(head_arr)
        self.gguf_writer.add_head_count_kv(kv_arr)

        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_pat)

        self.gguf_writer.add_value_length(self.hparams["head_dim"])

        # MoE params
        self.gguf_writer.add_expert_count(self.hparams["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["moe_top_k"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["share_expert_dim"])

        if (moe_router_scaling_factor := self.hparams.get("moe_router_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(moe_router_scaling_factor)
        if (norm_expert_weight := self.hparams.get("norm_expert_weight")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_expert_weight)

        # leading dense blocks
        leading_dense = 0
        moe_layers_enum = self.hparams.get("moe_layers_enum")
        if isinstance(moe_layers_enum, str) and moe_layers_enum.strip():
            moe_layers = sorted(int(i) for i in moe_layers_enum.strip().split(","))
            if moe_layers:
                leading_dense = max(0, moe_layers[0])
        self.gguf_writer.add_leading_dense_block_count(leading_dense)
        self.gguf_writer.add_moe_every_n_layers(int(self.hparams.get("moe_every_n_layer", 1)))

        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-5))

        # Optional per-layer SwiGLU clamps. MTP layers default to no clamping (0.0).
        if (limits := self.hparams.get("swiglu_limits")) is not None:
            limits_f = _pad(
                [0.0 if v is None else float(v) for v in limits],
                self.block_count,
                0.0,
            )
            self.gguf_writer.add_swiglu_clamp_exp(limits_f)
        if (limits_shared := self.hparams.get("swiglu_limits_shared")) is not None:
            limits_shared_f = _pad(
                [0.0 if v is None else float(v) for v in limits_shared],
                self.block_count,
                0.0,
            )
            self.gguf_writer.add_swiglu_clamp_shexp(limits_shared_f)

        if n_nextn > 0 and not self.no_mtp:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        # Map router bias (expert selection bias) to a GGUF bias tensor
        if name.endswith(".moe.router_bias"):
            name += ".bias"

        # Step3.5 appends the MTP block(s) past num_hidden_layers.
        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        # --no-mtp: drop the appended MTP block(s) entirely.
        if is_mtp and cls.no_mtp:
            return None
        # --mtp: keep ONLY MTP-block tensors plus the shared embeddings/norm/
        # lm_head (so the resulting GGUF carries just the draft head).
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        # The checkpoint nests the per-MTP-layer shared head under
        # `model.layers.{N+i}.transformer.shared_head.{norm,output}.weight`;
        # strip the `transformer.` infix and rename `output` → `head` so the
        # existing NEXTN_SHARED_HEAD_{NORM,HEAD} tensor mapping picks them up.
        # Mirrors vllm's `_rewrite_spec_layer_name` (step3p5_mtp.py).
        if is_mtp:
            name = name.replace(".transformer.", ".")
            name = name.replace("shared_head.output", "shared_head.head")

        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if name.endswith("norm.weight"):
            data_torch += 1.0

        if name.endswith((".self_attn.g_proj.weight", ".moe.gate.weight", ".moe.up_proj.weight", ".moe.gate_proj.weight", ".moe.down_proj.weight")):
            data_torch = data_torch.squeeze().contiguous()

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        # Mirror Qwen3.5's behavior: when emitting a draft-only file into a
        # directory, prefix with "mtp-" so it doesn't collide with the trunk.
        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # Step35 can optionally use Llama-3 style RoPE scaling (HF: rope_scaling.rope_type == "llama3").
        # llama.cpp represents this via a single extra tensor: "rope_freqs.weight" (aka MODEL_TENSOR.ROPE_FREQS).
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        rope_type = rope_params.get("rope_type") or ""
        if rope_type.lower() != "llama3":
            return

        # Step35 configs can carry per-layer rope_theta as a list; for llama3 rope factors we use the base value.
        rope_theta = self.hparams.get("rope_theta", 10000.0)
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[0]
        base = float(rope_theta)

        if (storage_dim := self.hparams.get("head_dim")) is None:
            storage_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        storage_dim = int(storage_dim)

        # Llama 3 factors apply only to the rotary dims used by full_attention layers
        # (partial_rotary_factor * head_dim). Remaining slots are padded with 1.0 so
        # sliding_attention layers remain unaffected. set_gguf_parameters already
        # guarantees at least one full_attention layer.
        layer_types = (self.hparams.get("layer_types") or [])[: self.block_count]
        partial_rotary_factors = (self.hparams.get("partial_rotary_factors") or [])[: self.block_count]
        full_attention_factor = next(
            float(f) for lt, f in zip(layer_types, partial_rotary_factors) if lt == "full_attention"
        )
        rotary_dim = int(storage_dim * full_attention_factor)

        freqs = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

        factor = float(rope_params.get("factor", 8.0))
        low_freq_factor = float(rope_params.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_params.get("high_freq_factor", 4.0))
        old_context_len = int(rope_params.get("original_max_position_embeddings", 8192))

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        rope_factors: list[float] = []
        for freq in freqs:
            wavelen = 2 * math.pi / float(freq)
            if wavelen < high_freq_wavelen:
                rope_factors.append(1.0)
            elif wavelen > low_freq_wavelen:
                rope_factors.append(factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                rope_factors.append(1.0 / ((1.0 - smooth) / factor + smooth))

        # Pad to head_dim/2 with 1.0 so non-scaled layers remain neutral.
        if len(rope_factors) < storage_dim // 2:
            rope_factors.extend([1.0] * (storage_dim // 2 - len(rope_factors)))

        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))
