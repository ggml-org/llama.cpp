from __future__ import annotations

import math

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf
from .qwenvl import Qwen2VLVisionModel


@ModelBase.register("ExaoneForCausalLM")
class ExaoneModel(TextModel):
    model_arch = gguf.MODEL_ARCH.EXAONE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        assert (hparams["activation_function"] == "silu")

        rotary_factor = self.rope_parameters.get("partial_rotary_factor")
        rotary_factor = rotary_factor if rotary_factor is not None else 1.0
        self.gguf_writer.add_rope_dimension_count(int(rotary_factor * (hparams["hidden_size"] // hparams["num_attention_heads"])))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = self.rope_parameters.get("rope_theta", 10000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_params.get("factor", 8.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
                old_context_len = rope_params.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))


@ModelBase.register("Exaone4ForCausalLM")
class Exaone4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.EXAONE4

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(hparams["sliding_window"])
            if "layer_types" in hparams:
                self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])
            elif "sliding_window_pattern" in hparams:
                sliding_window_pattern = []
                if isinstance(hparams["sliding_window_pattern"], str):  # e.g. LLLG
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append(hparams["sliding_window_pattern"][i % len(hparams["sliding_window_pattern"])] == "L")
                if isinstance(hparams["sliding_window_pattern"], int):  # e.g. 4
                    for i in range(hparams["num_hidden_layers"]):
                        sliding_window_pattern.append((i + 1) % hparams["sliding_window_pattern"] != 0)
                if len(sliding_window_pattern) == hparams["num_hidden_layers"]:
                    self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_params := self.rope_parameters.get("full_attention", self.rope_parameters):
            if rope_params.get("rope_type", '').lower() == "llama3":
                base = rope_params.get("rope_theta", 10_000.0)
                if (dim := self.hparams.get("head_dim")) is None:
                    dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
                freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

                factor = rope_params.get("factor", 16.0)
                low_freq_factor = rope_params.get("low_freq_factor", 1.0)
                high_freq_factor = rope_params.get("high_freq_factor", 4.0)
                old_context_len = rope_params.get("original_max_position_embeddings", 8192)

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(rope_factors, dtype=torch.float32))


@ModelBase.register("ExaoneMoEForCausalLM")
class ExaoneMoEModel(Exaone4Model):
    model_arch = gguf.MODEL_ARCH.EXAONE_MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        moe_intermediate_size = self.hparams["moe_intermediate_size"]
        num_shared_experts = self.hparams["num_shared_experts"]
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        self.gguf_writer.add_expert_shared_count(num_shared_experts)
        self.gguf_writer.add_expert_shared_feed_forward_length(moe_intermediate_size * num_shared_experts)
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])
        n_dense_layer = self.hparams.get("first_k_dense_replace", self.hparams.get("first_last_k_dense_replace", 0))
        self.gguf_writer.add_leading_dense_block_count(n_dense_layer)
        self.gguf_writer.add_nextn_predict_layers(self.hparams.get("num_nextn_predict_layers", 0))

        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("mtp."):
            if name.find("layers.") != -1:
                # `mtp.layers.0.[module_name]` format
                name = name.replace(f"mtp.layers.{bid}", f"model.layers.{bid + self.hparams['num_hidden_layers']}")
            else:
                # mtp fc/norm weights
                remapper = {
                    "mtp.fc": "model.layers.{bid}.eh_proj",
                    "mtp.pre_fc_norm_embedding": "model.layers.{bid}.enorm",
                    "mtp.pre_fc_norm_hidden": "model.layers.{bid}.hnorm",
                    "mtp.norm": "model.layers.{bid}.shared_head.norm",
                }
                _n = Path(name)
                new_name = remapper[_n.stem] + _n.suffix

                # set shared weights for all NextN/MTP layers
                for bid in range(self.hparams['num_hidden_layers'], self.block_count):
                    yield from super().modify_tensors(data_torch, new_name.format(bid=bid), bid)
                return

        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    yield from super().modify_tensors(data_torch, new_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Exaone4_5_ForConditionalGeneration")
class Exaone4_5_TextModel(Exaone4Model):
    """Text tower of EXAONE 4.5; Tensors match EXAONE4"""

    model_arch = gguf.MODEL_ARCH.EXAONE4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        if n_nextn > 0:
            self.block_count = self.hparams["num_hidden_layers"] + n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        if n_nextn > 0:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("mtp."):
            n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
            if n_nextn <= 0:
                return
            nh = self.hparams["num_hidden_layers"]
            if ".layers." in name:
                share = self.hparams.get("mtp_share_layers", False)
                mtp_bid = bid if bid is not None else 0
                if share:
                    for k in range(n_nextn):
                        nn = name.replace(f"mtp.layers.{mtp_bid}", f"model.layers.{nh + k}")
                        yield from super().modify_tensors(data_torch, nn, nh + k)
                    return
                name = name.replace(f"mtp.layers.{mtp_bid}", f"model.layers.{mtp_bid + nh}")
            else:
                remapper = {
                    "mtp.fc": gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                    "mtp.pre_fc_norm_embedding": gguf.MODEL_TENSOR.NEXTN_ENORM,
                    "mtp.pre_fc_norm_hidden": gguf.MODEL_TENSOR.NEXTN_HNORM,
                    "mtp.norm": gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
                }
                _n = Path(name)
                key = _n.stem
                if key not in remapper:
                    return
                for bid_mtp in range(nh, self.block_count):
                    mapped_name = self.format_tensor_name(remapper[key], bid_mtp, suffix=_n.suffix)
                    yield from ModelBase.modify_tensors(self, data_torch, mapped_name, bid_mtp)
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Exaone4_5_ForConditionalGeneration")
class Exaone4_5VisionModel(Qwen2VLVisionModel):
    """Vision tower for EXAONE 4.5; Qwen2-VL-style ViT (GQA) + patch merger"""

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        name = name.replace("model.visual.", "visual.", 1)
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        MmprojModel.set_gguf_parameters(self)
        assert self.hparams_vision is not None
        hparams = self.hparams_vision
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.EXAONE4_5)
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_min_pixels(self.preprocessor_config["min_pixels"])
        self.gguf_writer.add_vision_max_pixels(self.preprocessor_config["max_pixels"])
        num_kv_head = self.find_vparam(["num_key_value_heads"], optional=True)
        if num_kv_head is not None:
            self.gguf_writer.add_vision_head_count_kv(num_kv_head)
        eps = hparams.get("rms_norm_eps", self.global_config.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_vision_attention_layernorm_eps(eps)
        if (window_size := hparams.get("window_size")) is not None:
            self.gguf_writer.add_vision_window_size(window_size)
        fullatt_block_indexes = hparams.get("fullatt_block_indexes")
        if fullatt_block_indexes:
            n_wa_pattern = fullatt_block_indexes[0] + 1
            for i in range(1, len(fullatt_block_indexes)):
                if fullatt_block_indexes[i] - fullatt_block_indexes[i - 1] != n_wa_pattern:
                    raise ValueError(f"Invalid EXAONE4.5 fullatt_block_indexes: {fullatt_block_indexes}")
            self.gguf_writer.add_vision_n_wa_pattern(n_wa_pattern)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".qkv." in name:
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
            return

        yield from Qwen2VLVisionModel.modify_tensors(self, data_torch, name, bid)
