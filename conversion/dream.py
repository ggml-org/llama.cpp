from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("DreamModel")
class DreamModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DREAM

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab_dict = tokenizer.get_vocab()
        vocab_size = self.hparams.get("vocab_size", len(vocab_dict))
        assert max(vocab_dict.values()) < vocab_size
        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab_dict.items()}
        added_vocab = tokenizer.get_added_vocab()
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                # Check if it's a special token - treat special tokens as CONTROL tokens
                if hasattr(tokenizer, 'added_tokens_decoder') and i in tokenizer.added_tokens_decoder:
                    if tokenizer.added_tokens_decoder[i].special:
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    # Fallback: treat all added vocab as control tokens for special tokens like <|im_start|>
                    toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)
        return tokens, toktypes, tokpre

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()
        # Dream models use non-causal attention for diffusion
        self.gguf_writer.add_causal_attention(False)
        # Handle RoPE scaling similar to Qwen2
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        # Add Dream-specific parameters
        mask_token_id = self.hparams.get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(mask_token_id)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Dream model tensors should be mapped directly since it's the base model
        yield from super().modify_tensors(data_torch, name, bid)
