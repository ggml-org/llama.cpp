from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("DreamModel")
class DreamModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DREAM

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        vocab_dict = tokenizer.get_vocab()  # ty: ignore[unresolved-attribute]
        vocab_size = self.hparams.get("vocab_size", len(vocab_dict))
        assert max(vocab_dict.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab_dict.items()}
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

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

        # Add Dream-specific parameters
        mask_token_id = self.hparams.get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(mask_token_id)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Dream model tensors should be mapped directly since it's the base model
        yield from super().modify_tensors(data_torch, name, bid)
