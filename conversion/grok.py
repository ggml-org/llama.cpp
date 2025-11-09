from __future__ import annotations
import sys
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GrokForCausalLM", "Grok1ForCausalLM")
class GrokModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GROK

    def set_vocab(self):
        if (self.dir_model / 'tokenizer.model').is_file():
            self._set_vocab_sentencepiece()
            return
        if not (self.dir_model / 'tokenizer.json').is_file() or not (self.dir_model / 'chat_template.jinja').is_file():
            logger.error('Error: Missing vocab and chat template, download files from https://huggingface.co/alvarobartt/grok-2-tokenizer')
            sys.exit(1)
        self._set_vocab_gpt2()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_attn_logit_softcapping(self.hparams.get("attn_logit_softcapping", 30.0))
        self.gguf_writer.add_router_logit_softcapping(self.hparams.get("router_logit_softcapping", 30.0))
        if (final_logit_softcap := self.hparams.get("final_logit_softcapping")):
            self.gguf_writer.add_final_logit_softcapping(final_logit_softcap)
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        # Treat "original" as "yarn", seems to have been a mistake
        if self.hparams.get("rope_type") in ("yarn", "original"):
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(self.hparams["scaling_factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["original_max_position_embeddings"])
            self.gguf_writer.add_rope_scaling_yarn_ext_factor(self.hparams["extrapolation_factor"])
            self.gguf_writer.add_rope_scaling_yarn_attn_factor(self.hparams["attn_factor"])
            self.gguf_writer.add_rope_scaling_yarn_beta_fast(self.hparams["beta_fast"])
            self.gguf_writer.add_rope_scaling_yarn_beta_slow(self.hparams["beta_slow"])
        if temp_len := self.hparams.get("attn_temperature_len"):
            self.gguf_writer.add_attn_temperature_length(temp_len)
        self.gguf_writer.add_attn_output_scale(self.hparams.get("attn_output_multiplier", rope_dim**-0.5))
        self.gguf_writer.add_embedding_scale(self.hparams["embedding_multiplier_scale"])
        self.gguf_writer.add_logit_scale(self.hparams["output_multiplier_scale"])
    _experts: list[dict[str, list[Tensor]]] | None = None
    _cur_expert = ""

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        tensors: list[tuple[str, Tensor]] = []
        is_expert = ".moe." in name or ".block_sparse_moe.experts." in name
        if not is_expert:
            tensors.append((self.map_tensor_name(name), data_torch))
        # process the experts separately
        if is_expert or self._cur_expert:
            n_experts = self.hparams["num_local_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            # concatenate split tensors
            if name in self._experts[bid]:
                self._cur_expert = name
                self._experts[bid][name].append(data_torch)
                return []
            elif is_expert:
                self._cur_expert = name
                self._experts[bid][name] = [data_torch]
                return []
            else:
                self._cur_expert = ""
            for bid in range(self.block_count):
                if len(self._experts[bid]) >= n_experts * 3:
                    # merge the experts into a single 3d tensor
                    for wid in [("linear", "w1", 0), ("linear_1", "w2", 1), ("linear_v", "w3", 0)]:
                        datas: list[Tensor] = []
                        for xid in range(n_experts):
                            ename = f"transformer.decoder_layer.{bid}.moe.{xid}.{wid[0]}.weight"
                            if ename not in self._experts[bid]:
                                ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid[1]}.weight"
                            tensor_list = self._experts[bid][ename]
                            datas.append(torch.cat(tensor_list, dim=wid[2]) if len(tensor_list) > 1 else tensor_list[0])
                            del self._experts[bid][ename]
                        data_torch = torch.stack(datas, dim=0)
                        merged_name = f"transformer.decoder_layer.{bid}.moe.{wid[0]}.weight"
                        new_name = self.map_tensor_name(merged_name)
                        yield (new_name, data_torch)
        yield from tensors
