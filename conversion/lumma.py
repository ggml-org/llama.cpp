from __future__ import annotations

import json
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .llama import LlamaModel


class _LummaPreTokenizer:
    def __init__(self, tok) -> None:
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids


@ModelBase.register("LummaForCausalLM")
class LummaModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LUMMA
    # Lumma HF weights already match ggml NeoX layout; permuting Q/K breaks inference.
    undo_permute = False

    def set_vocab(self) -> None:
        from tokenizers import Tokenizer

        tokenizer_path = self.dir_model / "tokenizer.json"
        hf_tokenizer = Tokenizer.from_file(str(tokenizer_path))

        tokpre = self.get_vocab_base_pre(_LummaPreTokenizer(hf_tokenizer))

        vocab = hf_tokenizer.get_vocab()
        reverse_vocab = {idx: tok for tok, idx in vocab.items()}
        vocab_size = self.hparams.get("vocab_size", len(vocab))

        with open(tokenizer_path, encoding="utf-8") as f:
            added_tokens_decoder = {
                item["id"]: item for item in json.load(f).get("added_tokens", [])
            }

        tokens: list[str] = []
        toktypes: list[int] = []

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif i in added_tokens_decoder:
                token = reverse_vocab[i]
                if added_tokens_decoder[i].get("special") or self.does_token_look_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
                tokens.append(token)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_add_space_prefix(True)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams

        if hparams.get("factorized_embedding", False):
            rank = hparams["embedding_rank"]
            self.gguf_writer.add_embedding_length_out(rank)
            logger.info(f"Lumma factorized embeddings: hidden={hparams['hidden_size']} rank={rank}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("self_attn.v_proj.weight") or name.endswith("self_attn.v_proj.bias"):
            return

        if name.endswith("embedding_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD_PROJ), data_torch.contiguous())
            return

        if name.endswith("lm_head_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_PROJ), data_torch.contiguous())
            return

        yield from super().modify_tensors(data_torch, name, bid)
