from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Cohere2 runtime in llama.cpp expects no bias tensors;
        # the actual weight only contains 0-value tensors as bias, we can skip them
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r} for Cohere2 conversion.")
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Cohere2MoeForCausalLM")
class Cohere2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2MOE
    _n_main_layers: int | None = None
    _expert_tensor_re = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down_proj|gate_proj|up_proj)\.weight"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (n_nextn := int(self.hparams.get("num_nextn_predict_layers", 0) or 0)) > 0 and not self.no_mtp:
            self.block_count += n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self._experts: list[dict[str, Tensor]] = [{} for _ in range(self.block_count)]

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hparams = self.hparams
        expert_intermediate_size = hparams["intermediate_size"]
        mlp_layer_types = hparams.get("mlp_layer_types")
        n_dense_lead = hparams.get("first_k_dense_replace", 0)
        if mlp_layer_types is not None:
            n_dense_lead = next((i for i, t in enumerate(mlp_layer_types) if t != "dense"), len(mlp_layer_types))

        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hparams["layer_types"]])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(expert_intermediate_size)
        self.gguf_writer.add_leading_dense_block_count(n_dense_lead)
        self.gguf_writer.add_expert_weights_norm(hparams.get("norm_topk_prob", False))
        if (num_shared_experts := hparams.get("num_shared_experts", 0)) > 0:
            if hparams.get("shared_expert_combination_strategy", "average") != "average":
                raise ValueError("Cohere2 MoE only supports average shared expert combination")
            self.gguf_writer.add_expert_shared_count(num_shared_experts)
            self.gguf_writer.add_expert_shared_feed_forward_length(expert_intermediate_size * num_shared_experts)
        if (n_nextn := hparams.get("num_nextn_predict_layers", 0)) > 0 and not self.no_mtp:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)
        self.gguf_writer.add_rope_dimension_count(hparams["head_dim"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        self._n_main_layers = hparams.get("num_hidden_layers")
        type(self)._n_main_layers = self._n_main_layers
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item):
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        if cls._n_main_layers is not None:
            is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers
            if is_mtp and cls.no_mtp:
                return None
            if cls.mtp_only and not is_mtp and name not in (
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            ):
                return None

        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r}.")
            return

        if (m := self._expert_tensor_re.fullmatch(name)) is not None:
            n_experts = self.hparams["num_experts"]
            layer_idx = int(m.group(1))
            assert bid is None or bid == layer_idx

            self._experts[layer_idx][name] = data_torch

            expected = {
                f"model.layers.{layer_idx}.mlp.experts.{xid}.{w_name}.weight"
                for xid in range(n_experts)
                for w_name in ("down_proj", "gate_proj", "up_proj")
            }
            if expected.issubset(self._experts[layer_idx]):
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{layer_idx}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[layer_idx][ename])
                        del self._experts[layer_idx][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{layer_idx}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, layer_idx)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        experts = [k for d in self._experts for k in d.keys()]
        if len(experts) > 0:
            raise ValueError(f"Unprocessed experts: {experts}")
