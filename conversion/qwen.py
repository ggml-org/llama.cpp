from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("QWenLMHeadModel")
class QwenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_vocab(self):
        self._set_vocab_qwen()

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register("Qwen2Model", "Qwen2ForCausalLM", "Qwen2AudioForConditionalGeneration")
class Qwen2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.hf_arch == "Qwen2Model":
            name = f"model.{name}"  # map to Qwen2ForCausalLM tensors
        if "language_model." in name:
            name = name.replace("language_model.", "") # for InternVL
        if name.startswith("mlp") or name.startswith("multi_modal_projector") \
                or name.startswith("vision_model") or name.startswith("audio_tower") \
                or name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            # skip vision and audio tensors
            return []
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen2MoeForCausalLM")
class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")
        # YaRN is not enabled by default
        # To enable it, please refer to this guide: https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        name = name.replace("language_model.", "") # InternVL
        # handle aggregated expert tensors
        # GGUF stores dimensions reversed from PyTorch, so:
        # PyTorch (A,B,C) -> GGUF writes [C,B,A] -> GGML reads ne={C,B,A}
        # Input shapes from HF: (n_expert, n_ff_exp, n_embd) or (n_expert, n_embd, n_ff_exp)
        # Expected GGML ne: {n_embd, n_ff_exp, n_expert} for gate/up, {n_ff_exp, n_embd, n_expert} for down
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            # Input: (n_expert=128, n_ff_exp=768, n_embd=2048)
            # Want GGML ne: {n_ff_exp, n_embd, n_expert} = {768, 2048, 128}
            # Need PyTorch: (128, 2048, 768) [reversed of GGML]
            # So: permute(0, 2, 1): (128, 768, 2048) -> (128, 2048, 768)
            permuted = data_torch.permute(0, 2, 1).contiguous()
            return [(self.map_tensor_name(mapped), permuted)]
        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-1] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            split_dim = data_torch.shape[-1] // 2
            gate = data_torch[..., :split_dim].contiguous()
            up = data_torch[..., split_dim:].contiguous()
            # Input gate/up: (n_expert=128, n_embd=2048, n_ff_exp=768)
            # Want GGML ne: {n_embd, n_ff_exp, n_expert} = {2048, 768, 128}
            # Need PyTorch: (128, 768, 2048) [reversed of GGML]
            # So: permute(0, 2, 1): (128, 2048, 768) -> (128, 768, 2048)
            base_name = name.removesuffix(".weight")
            base = base_name.rsplit('.', 1)[0]
            mapped_gate = f"{base}.gate_proj.weight"
            mapped_up = f"{base}.up_proj.weight"
            perm_gate = gate.permute(0, 2, 1).contiguous()
            perm_up = up.permute(0, 2, 1).contiguous()
            return [
                (self.map_tensor_name(mapped_gate), perm_gate),
                (self.map_tensor_name(mapped_up), perm_up),
            ]
        if name.startswith("mlp") or name.startswith("vision_model") or name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") or name.startswith("model.visual"):
            # skip visual tensors
            return []
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
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
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Qwen3ForCausalLM")
class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3
    # extra logic for rerank models
    is_rerank: bool = False
    is_tied_embeddings: bool = False
    token_false_id: int | None = None
    token_true_id: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # track for intern-s1-mini
        hparams = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]
        # a bit hacky, but currently the only way to detect if this is a rerank model
        # ref: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
        readme_path = self.dir_model / "README.md"
        readme_text = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf-8") as f:
                readme_text = f.read()
        if "# Qwen3-Reranker" in readme_text:
            self._find_rerank_config()

    def set_vocab(self):
        # deal with intern-s1-mini
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return
        super().set_vocab()

    def _find_rerank_config(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        self.is_rerank = True
        self.is_tied_embeddings = self.hparams.get("tie_word_embeddings", False)
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self.sep_token_id = tokenizer.convert_tokens_to_ids("|")
        assert self.token_false_id is not None and self.token_true_id is not None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.is_rerank:
            self.gguf_writer.add_pooling_type(gguf.PoolingType.RANK)
            self.gguf_writer.add_classifier_output_labels(["yes", "no"])
            self.gguf_writer.add_chat_template([{
                "name": "rerank",
                "template": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                            "<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            }])

    def _get_cls_out_tensor(self, data_torch: Tensor) -> Tensor:
        # extract "yes" and "no" tokens from the output lm_head tensor
        false_row = data_torch[self.token_false_id]
        true_row = data_torch[self.token_true_id]
        return torch.stack([true_row, false_row], dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "model.vision_" in name:
            # skip multimodal tensors
            return []
        if self.is_rerank:
            is_tied_head = self.is_tied_embeddings and "embed_tokens" in name
            is_real_head = not self.is_tied_embeddings and "lm_head" in name
            if is_tied_head or is_real_head:
                cls_out_head = (
                    gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.CLS_OUT] + ".weight",
                    self._get_cls_out_tensor(data_torch),
                )
                if is_tied_head:
                    embed = (self.map_tensor_name(name), data_torch)
                    return [cls_out_head, embed]
                if is_real_head:
                    return [cls_out_head]
        return super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3MoeForCausalLM")
class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = ModelBase.load_hparams(self.dir_model, False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

    def set_vocab(self):
        # deal with intern-s1
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return
        super().set_vocab()
