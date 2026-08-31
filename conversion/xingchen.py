from __future__ import annotations

import json
import re

from typing import TYPE_CHECKING, Callable, Iterable

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .deepseek import DeepseekV2Model


@ModelBase.register("XingChen4ForCausalLM")
class XingChen4Model(DeepseekV2Model):
    # DeepSeek-V3 style MLA/MoE + Manifold-Constrained Hyper-Connections (MHC)
    # multi-residual-stream blocks + a DeepSeek-V3 MTP head.
    model_arch = gguf.MODEL_ARCH.XINGCHEN4
    skip_mtp = False
    supports_mtp_export = True
    _n_main_layers: int | None = None

    _HC_ALPHA = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        # (bid, "attn_hc"|"ffn_hc") -> {0: alpha_pre, 1: alpha_post, 2: alpha_res}
        self._hc_alpha: dict[tuple[int, str], dict[int, Tensor]] = {}

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        # legacy bias_* tensors are unused (vLLM load_weights skips them)
        if re.search(r"(attn_hc|ffn_hc)\.bias_(pre|post|res)$", name):
            return None

        # the NextN/MTP block lives past num_hidden_layers (model.layers.40 -> blk.40)
        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        # --no-mtp: drop the NextN block entirely.
        if is_mtp and cls.no_mtp:
            return None
        # --mtp: keep only NextN-block tensors plus the shared embeddings/norm/lm_head.
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_vocab(self):
        # the checkpoint ships only the slow TeleChat3 tokenizer.model; build the
        # BPE vocab and merges straight from the SentencePiece model
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / 'tokenizer.model'
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = tokenizer.vocab_size()

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []
        pieces: list[str] = []
        for token_id in range(vocab_size):
            piece = tokenizer.IdToPiece(token_id)
            pieces.append(piece)
            tokens.append(piece.encode("utf-8"))
            scores.append(tokenizer.GetScore(token_id))

            if tokenizer.IsUnknown(token_id):
                toktypes.append(gguf.TokenType.UNKNOWN)
            elif tokenizer.IsControl(token_id):
                toktypes.append(gguf.TokenType.CONTROL)
            elif tokenizer.IsUnused(token_id):
                toktypes.append(gguf.TokenType.UNUSED)
            elif tokenizer.IsByte(token_id):
                toktypes.append(gguf.TokenType.BYTE)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        # reclassify special tokens as CONTROL, but only when the added-token id
        # lines up with the spm piece; tokenizer_config.json ids can drift from
        # the spm model and the slow tokenizer (sp.encode) uses the spm ids
        tokenizer_config_file = self.dir_model / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
            for token_id, token_data in tokenizer_config_json.get("added_tokens_decoder", {}).items():
                token_id = int(token_id)
                if token_id >= vocab_size:
                    continue
                token: str = token_data["content"]
                if pieces[token_id] != token:
                    continue
                if token_data.get("special") or self.does_token_look_special(token):
                    toktypes[token_id] = gguf.TokenType.CONTROL
                else:
                    toktypes[token_id] = gguf.TokenType.USER_DEFINED

        # merges, replicating transformers' SentencePieceExtractor.generate_merges():
        # sorted by piece score desc, which is sentencepiece's merge priority and
        # maps to llama.cpp's bpe rank 0 = highest priority
        vocab = {piece: i for i, piece in enumerate(pieces)}
        vocab_scores = {piece: score for piece, score in zip(pieces, scores)}
        merges_raw: list[tuple[str, str, float]] = []
        for merge, piece_score in vocab_scores.items():
            local: list[tuple[str, str, float]] = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges_raw.extend(local)
        merges_raw = sorted(merges_raw, key=lambda val: (val[2], len(val[0]), len(val[1])), reverse=True)
        merges = [f"{val[0]} {val[1]}" for val in merges_raw]

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("xingchen4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.merges = merges
        # unk/pad ids are absent from config.json; take them from the spm trainer spec
        special_vocab._set_special_token("unk", tokenizer.unk_id())
        special_vocab._set_special_token("pad", tokenizer.pad_id())
        special_vocab.add_to_gguf(self.gguf_writer)

        # sentencepiece add_dummy_prefix -> BPE runtime prepends U+2581 to the text
        self.gguf_writer.add_add_space_prefix(True)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # NextN/MTP prediction layers
        if not self.no_mtp and (num_nextn_predict_layers := hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # Manifold-Constrained Hyper-Connections (MHC)
        self.gguf_writer.add_hyper_connection_count(hparams["num_residual_streams"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["mhc_sinkhorn_iterations"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-6)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hc_match = re.match(r"model\.layers\.(\d+)\.(attn_hc|ffn_hc)\.(\w+)$", name)
        if hc_match is not None:
            layer, module, leaf = hc_match.groups()
            key = (int(layer), module)
            hc = "attn" if module == "attn_hc" else "ffn"

            if leaf == "mapping_weight":
                # HF shape [hc_mix_dim, hc_dim]; the saver writes dims reversed,
                # giving GGUF {hc_dim, hc_mix_dim} as the C++ loader expects
                new_name = self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{hc.upper()}_FN"), key[0])
                yield new_name, data_torch.to(torch.float32)
            elif leaf == "bias":
                new_name = self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{hc.upper()}_BASE"), key[0])
                yield new_name, data_torch.to(torch.float32)
            else:
                alpha_index = self._HC_ALPHA.get(leaf)
                if alpha_index is None:
                    return
                buf = self._hc_alpha.setdefault(key, {})
                buf[alpha_index] = data_torch.to(torch.float32)
                if len(buf) == 3:
                    # all three alpha_* present -> emit hc_scale [3]
                    scale = torch.stack([buf[i] for i in range(3)]).reshape(-1)
                    new_name = self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{hc.upper()}_SCALE"), key[0])
                    yield new_name, scale
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # keep the small MHC weights in F32 for exact vLLM logit parity
        if re.search(r"blk\.\d+\.hc_(attn|ffn)_(fn|base|scale)\.weight$", new_name):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_metadata(self, vocab_only: bool):
        # give --mtp draft exports an 'mtp-' prefixed file name (mirrors DeepSeek-V4)
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"
