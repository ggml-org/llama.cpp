from __future__ import annotations

import json
import math
import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf


@ModelBase.register("Ernie4_5_ForCausalLM", "Ernie4_5ForCausalLM")
class Ernie4_5Model(TextModel):
    model_arch = gguf.MODEL_ARCH.ERNIE4_5

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(tokenizer_config_json["add_prefix_space"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "ernie." in name:
            name = name.replace("ernie.", "model.")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_heads = self.hparams["num_attention_heads"]
        num_kv_heads = self.hparams["num_key_value_heads"]
        if (head_dim := self.hparams.get("head_dim")) is None:
            head_dim = self.hparams["hidden_size"] // num_heads

        # split the qkv weights
        # qkv_proj shape: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
        if "qkv_proj" in name:
            name_q = name.replace("qkv_proj.weight", "q_proj.weight")
            name_k = name.replace("qkv_proj.weight", "k_proj.weight")
            name_v = name.replace("qkv_proj.weight", "v_proj.weight")
            total_q_dim = num_heads * head_dim
            total_k_dim = num_kv_heads * head_dim
            total_v_dim = num_kv_heads * head_dim
            q_proj_weight, k_proj_weight, v_proj_weight = data_torch.split([total_q_dim, total_k_dim, total_v_dim], dim=0)
            yield from super().modify_tensors(q_proj_weight, name_q, bid)
            yield from super().modify_tensors(k_proj_weight, name_k, bid)
            yield from super().modify_tensors(v_proj_weight, name_v, bid)
        # split the up_gate_proj into gate and up
        # up_gate_proj shape: [2 * intermediate_size, hidden_size]
        elif "up_gate_proj" in name:
            name_up = name.replace("up_gate_proj.weight", "up_proj.weight")
            name_gate = name.replace("up_gate_proj.weight", "gate_proj.weight")
            dim_half = data_torch.shape[0] // 2
            gate_proj_weight, up_proj_weight = data_torch.split(dim_half, dim=0)
            yield from super().modify_tensors(gate_proj_weight, name_gate, bid)
            yield from super().modify_tensors(up_proj_weight, name_up, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Ernie4_5_MoeForCausalLM")
class Ernie4_5MoeModel(Ernie4_5Model):
    model_arch = gguf.MODEL_ARCH.ERNIE4_5_MOE
    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experts = [{} for _ in range(self.block_count)]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_count(self.hparams["moe_num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["moe_k"])
        self.gguf_writer.add_interleave_moe_layer_step(self.hparams["moe_layer_interval"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["moe_layer_start_index"])
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (shared_expert_count := self.hparams.get('moe_num_shared_experts')) is not None:
            self.gguf_writer.add_expert_shared_count(shared_expert_count)
            if shared_expert_count > 0 and (shared_expert_intermediate_size := self.hparams.get('intermediate_size')) is not None and (num_key_value_heads := self.hparams.get('num_key_value_heads')) is not None:
                self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size // num_key_value_heads)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # skip Multi-Token Prediction (MTP) layers (again, same as DeepseekV2)
        match = re.match(r"model.mtp_block.(\d+)", name)
        if match:
            return None

        # skip all other MTP tensors for now
        match = re.match(r"model.mtp_emb_norm.(\d+)", name)
        if match:
            return None

        match = re.match(r"model.mtp_hidden_norm.(\d+)", name)
        if match:
            return None

        match = re.match(r"model.mtp_linear_proj.(\d+)", name)
        if match:
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["moe_num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(data_torch, merged_name, bid)
        else:
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("PaddleOCRVLForConditionalGeneration")
class PaddleOCRModel(Ernie4_5Model):
    model_arch = gguf.MODEL_ARCH.PADDLEOCR


@ModelBase.register("PaddleOCRVisionModel")
class PaddleOCRVisionModel(MmprojModel):
    # PaddleOCR-VL uses a modified version of Siglip
    min_pixels: int = 0
    max_pixels: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        self.min_pixels = self.preprocessor_config["min_pixels"]
        self.max_pixels = self.preprocessor_config["max_pixels"]
        self.hparams_vision["image_size"] = int(math.sqrt(self.max_pixels))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        hparams = self.hparams_vision
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.PADDLEOCR)
        self.gguf_writer.add_vision_max_pixels(self.max_pixels)
        self.gguf_writer.add_vision_min_pixels(self.min_pixels)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("rms_norm_eps", 1e-6))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if "vision_model" not in name and "mlp_AR" not in name:
            return None
        name = name.replace("visual.", "model.")
        if "packing_position_embedding" in name:
            # unused
            return None
        if "vision_model.head" in name:
            # we don't yet support image embeddings for this model
            return None

        return super().filter_tensors((name, gen))
