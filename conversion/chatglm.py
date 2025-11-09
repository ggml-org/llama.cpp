from __future__ import annotations
from .base import (
    ModelBase, TextModel, SentencePieceTokenTypes,
    gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GlmForCausalLM", "ChatGLMModel", "ChatGLMForConditionalGeneration")
class ChatGLMModel(TextModel):
    model_arch = gguf.MODEL_ARCH.CHATGLM

    def set_vocab_chatglm3(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytes] = []
        toktypes: list[int] = []
        scores: list[float] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams.get("padded_vocab_size", len(tokenizer.get_vocab()))
        assert max(tokenizer.get_vocab().values()) < vocab_size
        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        for token_id in range(vocab_size):
            piece = tokenizer._convert_id_to_token(token_id)
            if token_id == 0:
                piece = "<unk>"
            elif token_id == 1:
                piece = "<bos>"
            elif token_id == 2:
                piece = "<eos>"
            text = piece.encode("utf-8")
            score = 0.0
            # Referencing the tokenizer Python implementation(https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py),
            # it is only valid if it is less than tokenizer.tokenizer.sp_model.vocab_size()
            if len(piece) != 0 and token_id < tokenizer.tokenizer.sp_model.vocab_size():
                score = tokenizer.tokenizer.sp_model.get_score(token_id)
            if token_id >= tokenizer.tokenizer.sp_model.vocab_size():
                if piece in special_tokens:
                    toktype = SentencePieceTokenTypes.CONTROL
                elif len(piece) == 0:
                    text = f"[PAD{token_id}]".encode("utf-8")
                    toktype = SentencePieceTokenTypes.UNUSED
                else:
                    toktype = SentencePieceTokenTypes.USER_DEFINED
                tokens.append(text)
                scores.append(score)
                toktypes.append(toktype)
                continue
            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.tokenizer.sp_model.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.tokenizer.sp_model.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.tokenizer.sp_model.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.tokenizer.sp_model.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
        self.gguf_writer.add_tokenizer_model("llama")
        # glm3 needs prefix and suffix formatted as:
        # prompt = "[gMASK]sop<|user|>\n" + prompt + "<|assistant|>"
        self.gguf_writer.add_tokenizer_pre("chatglm-spm")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

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
        if "THUDM/chatglm3-6b" in self.hparams.get("_name_or_path", ""):
            self.set_vocab_chatglm3()
            return
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams.get("padded_vocab_size",hparams["vocab_size"])
        assert max(tokenizer.get_vocab().values()) < vocab_size
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # only add special tokens when they were not already loaded from config.json
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])
        # this one is usually not in config.json anyway
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_head_kv = self.hparams.get("multi_query_group_num", self.hparams.get("num_key_value_heads", n_head))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(self.hparams.get("ffn_hidden_size", self.hparams.get("intermediate_size", 4 * n_embed)))
        self.gguf_writer.add_block_count(self.hparams.get("num_layers", self.hparams["num_hidden_layers"]))
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("layernorm_epsilon",1e-5))
        self.gguf_writer.add_file_type(self.ftype)
        if "attention_dim" in self.hparams:
            rope_dim = self.hparams["attention_dim"]
        else:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        self.gguf_writer.add_add_bos_token(False)
        rope_freq = 10000
        if "rope_ratio" in self.hparams:
            rope_freq = rope_freq * self.hparams["rope_ratio"]
        self.gguf_writer.add_rope_freq_base(rope_freq)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if name.endswith(".rotary_pos_emb.inv_freq") or name.startswith("model.vision."):
            return []
        name = name.removeprefix("transformer.")
        return [(self.map_tensor_name(name), data_torch)]
