from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("XverseForCausalLM")
class XverseModel(TextModel):
    model_arch = gguf.MODEL_ARCH.XVERSE

    def set_vocab(self):
        assert (self.dir_model / "tokenizer.json").is_file()
        dir_model = self.dir_model
        hparams = self.hparams

        tokens: list[bytes] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))  # ty: ignore[unresolved-attribute]
        # Since we are checking the maximum index, we need to ensure it's strictly less than vocab_size,
        # because vocab_size is the count of items, and indexes start at 0.
        max_vocab_index = max(tokenizer.get_vocab().values())  # ty: ignore[unresolved-attribute]
        if max_vocab_index >= vocab_size:
            raise ValueError("Vocabulary size exceeds expected maximum size.")

        reverse_vocab: dict[int, str] = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}  # ty: ignore[unresolved-attribute]
        added_vocab = tokenizer.get_added_vocab()  # ty: ignore[unresolved-attribute]

        for token_id in range(vocab_size):
            token_text = reverse_vocab[token_id].encode('utf-8')
            # replace "\x00" to string with length > 0
            if token_text == b"\x00":
                toktype = gguf.TokenType.BYTE  # special
                token_text = f"<{token_text}>".encode('utf-8')
            elif re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
                toktype = gguf.TokenType.BYTE  # special
            elif reverse_vocab[token_id] in added_vocab:
                if tokenizer.added_tokens_decoder[token_id].special:  # ty: ignore[unresolved-attribute]
                    toktype = gguf.TokenType.CONTROL
                else:
                    toktype = gguf.TokenType.USER_DEFINED
            else:
                toktype = gguf.TokenType.NORMAL

            tokens.append(token_text)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        # HF models permute some of the tensors, so we need to undo that
        if name.endswith("q_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count)
        if name.endswith("k_proj.weight"):
            data_torch = self._reverse_hf_permute(data_torch, head_count, head_count_kv)

        yield from super().modify_tensors(data_torch, name, bid)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )
