from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from .qwen import QwenModel
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("HunYuanMoEV1ForCausalLM")
class HunYuanMoEModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HUNYUAN_MOE

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        # 1. Get the pre-tokenizer identifier hash
        tokpre = self.get_vocab_base_pre(tokenizer)
        # 2. Reverse-engineer the merges list from mergeable_ranks
        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            if len(merged) == 2: # todo this is an assert in Qwen, why?
                merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))
        # 3. Generate the tokens and toktypes lists
        vocab_size = self.hparams["vocab_size"]
        assert tokenizer.vocab_size == vocab_size
        special_tokens = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token = reverse_vocab[i]
                tokens.append(token)
                if i in special_tokens.values():
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
        # 4. Write all vocab-related fields to the GGUF writer
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)
        # 5. Add special tokens and chat templates
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)
        # FIX for BOS token: Overwrite incorrect id read from config.json
        self.gguf_writer.add_bos_token_id(127959) # <|bos|>

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["intermediate_size"])
        moe_intermediate_size = hparams["moe_intermediate_size"]
        assert all(n == moe_intermediate_size[0] for n in moe_intermediate_size)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size[0])
        moe_topk = hparams["moe_topk"]
        assert all(topk == moe_topk[0] for topk in moe_topk)
        self.gguf_writer.add_expert_used_count(moe_topk[0])
        moe_shared_expert = hparams["num_shared_expert"]
        assert all(n == moe_shared_expert[0] for n in moe_shared_expert)
        self.gguf_writer.add_expert_shared_count(moe_shared_expert[0])
        # Rope
        rope_scaling = hparams.get("rope_scaling", {})
        if rope_scaling.get("type") == "dynamic":
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = rope_scaling.get("alpha", 1000)
            base = hparams.get("rope_theta", 10000.0)
            dim = (hparams["hidden_size"] // hparams["num_attention_heads"]) # 128
            scaled_base = base * (alpha ** (dim / (dim - 2))) # 10000 * (1000 ** (128 / 126)) = 11158839.9251
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
            self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
            self.gguf_writer.add_context_length(256 * 1024) # 256k context length
            # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
            assert alpha == 1000 and base == 10000.0 and dim == 128 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return []
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                tensors: list[tuple[str, Tensor]] = []
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
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("HunYuanDenseV1ForCausalLM")
class HunYuanModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HUNYUAN_DENSE

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
            # 1. Get the pre-tokenizer identifier hash
            tokpre = self.get_vocab_base_pre(tokenizer)
            # 2. Reverse-engineer the merges list from mergeable_ranks
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.mergeable_ranks
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))
            # 3. Generate the tokens and toktypes lists
            vocab_size = self.hparams["vocab_size"]
            assert tokenizer.vocab_size == vocab_size
            special_tokens = tokenizer.special_tokens
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []
            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)
            # 4. Write all vocab-related fields to the GGUF writer
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)
            # 5. Add special tokens and chat templates
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
            # FIX for BOS token: Overwrite incorrect id read from config.json
            if self.hparams['hidden_size'] == 4096:
                self.gguf_writer.add_bos_token_id(127958) # only for 7b dense, fix <|bos|> token

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        # Rope
        rope_scaling = hparams.get("rope_scaling", {})
        if rope_scaling.get("type") == "dynamic":
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = rope_scaling.get("alpha", 50)
            base = hparams.get("rope_theta", 10000.0)
            dim = hparams["head_dim"]
            scaled_base = base * (alpha ** (dim / (dim - 2)))
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
            self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
            self.gguf_writer.add_context_length(256 * 1024) # 256k context length
            # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
            assert base == 10000.0 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return []
        return [(self.map_tensor_name(name), data_torch)]
