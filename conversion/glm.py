from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Glm4ForCausalLM", "Glm4vForConditionalGeneration")
class Glm4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GLM4

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.visual."): # ignore visual part of Glm4v
            return []
        elif name.startswith("model.language_model."):
            name = name.replace("language_model.", "") # for Glm4v
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Glm4MoeForCausalLM")
class Glm4MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GLM4_MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GLM4_MOE has num_hidden_layers + 1 actual layers (including NextN layer)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        # Special tokens
        # Note: Using <|endoftext|> (151329) for eot causes endless generation
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["[gMASK]"])  # 151331
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # 151336
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"]) # 151329
        special_vocab._set_special_token("eom", tokenizer.get_added_vocab()["<|observation|>"])  # 151338
        # Patch broken chat template
        if isinstance(special_vocab.chat_template, str) and "visible_text(m.content).endswith" in special_vocab.chat_template:
            special_vocab.chat_template = special_vocab.chat_template.replace(
                """{{ visible_text(m.content) }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not visible_text(m.content).endswith("/nothink")) else '' -}}""",
                """{% set content = visible_text(m.content) %}{{ content }}\n{{- '/nothink' if (enable_thinking is defined and not enable_thinking and not content.endswith("/nothink")) else '' -}}""")
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = (
                self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            )
        self.gguf_writer.add_rope_dimension_count(
            int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5))
        )
        # MoE parameters - Use only routed expert count (shared experts handled separately)
        if (n_routed_experts := self.hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_shared_experts := self.hparams.get("n_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (first_k_dense_replace := self.hparams.get("first_k_dense_replace")) is not None:
            self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)
        # Expert gating function (sigmoid for GLM4_MOE)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        # Routed scaling factor
        if (routed_scaling_factor := self.hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)
        # Normalise topk probabilities
        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)
        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.visual."):  # ignore visual part
            return []
        elif name.startswith("model.language_model."):
            name = name.replace("language_model.", "")  # for multimodal variants
        # Handle main token embedding (but not layer-specific NextN embeddings)
        if name == "model.embed_tokens.weight" and ".layers." not in name:
            return [(self.map_tensor_name("token_embd.weight"), data_torch)]
        # Handle routed experts
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
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
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")
        new_name = self.map_tensor_name(name)
        return [(new_name, data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
