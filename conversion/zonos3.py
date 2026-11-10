from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, LazyTorchTensor, gguf, logger

LM_DIR = Path("lm") / "zonos3_mini"

# GDN geometry, hard-coded in the reference (zonos3.reference.lm)
GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4

# the text table keeps the tokenizer ids shifted by this offset, ids 2 and 3 are BOS and EOS
TEXT_OFFSET = 4
TEXT_BOS, TEXT_EOS = 2, 3


def _round_up(value: float, multiple: int) -> int:
    return multiple * ((int(value) + multiple - 1) // multiple)


def _load_params(dir_model: Path) -> dict[str, Any] | None:
    path = dir_model / LM_DIR / "params.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        params = json.load(f).get("model", {})
    return params if params.get("model_type") == "zonos2" else None


def _is_zonos3_checkpoint(dir_model: Path) -> bool:
    return _load_params(dir_model) is not None


@ModelBase.register_hparams_loader(_is_zonos3_checkpoint)
def _load_zonos3_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected ZONOS3 checkpoint")
    params = _load_params(dir_model)
    assert params is not None
    for key, want in (("attention_pattern", "gdn3_global1"), ("residual_type", "full_attnres"),
                      ("moe_router_type", "linear"), ("moe_router_use_eda", False), ("use_xsa", False),
                      ("moe_end_from_layer", 0), ("speaker_enabled", True)):
        if params.get(key) != want:
            raise NotImplementedError(f"ZONOS3: unsupported {key} = {params.get(key)!r}, expected {want!r}")
    return {"architectures": ["Zonos3Model"], "text_config": dict(params)}


@ModelBase.register("Zonos3Model")
class Zonos3TextModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZONOS3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self.n_embd       = hp["dim"]
        self.n_codebooks  = hp["n_codebooks"]
        self.audio_vocab  = hp["codebook_size"] + 2  # codes, EOA, pad
        self.audio_pad_id = hp["audio_pad_id"]
        self.text_rows    = hp["text_vocab"] + 1     # + the padding row, used by the generated rows
        self.cond_counts  = [n + 1 for n in hp["condition_bucket_counts"]]  # + the unknown bucket
        self.gdn_period   = int(hp["attention_pattern"][len("gdn"):].split("_")[0]) + 1
        self.n_ff         = _round_up(hp["ffn_dim_multiplier"] * hp["dim"], hp["multiple_of"])
        self.n_ff_exp     = _round_up(hp["moe_ffn_dim_multiplier"] * hp["dim"], hp["moe_multiple_of"])
        self.n_latent     = int(hp["dim"] * hp["moe_alpha"])
        self._embd: dict[str, Tensor] = {}
        self._router: dict[int, dict[str, Tensor]] = {}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        path = self.dir_model / LM_DIR / "model.pth"
        logger.info(f"gguf: indexing model part '{LM_DIR / 'model.pth'}'")
        state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        tensors: dict[str, Callable[[], Tensor]] = {}
        for name, data in state.items():
            gen = (lambda data=data: LazyTorchTensor.from_eager(data)) if self.lazy else (lambda data=data: data)
            titem = self.filter_tensors((name, gen))
            self._source_tensor_names[name] = titem[0] if titem else None
            if titem:
                tensors[titem[0]] = titem[1]
        return tensors

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # audio heads and speaker projection go to the mmproj
        if name.startswith(("multi_output.", "speaker_projection.")):
            return None
        return item

    def set_vocab(self):
        with open(self.dir_model / LM_DIR / "tokenizer" / "superbpe8192.json", encoding="utf-8") as f:
            tok = json.load(f)
        model = tok["model"]
        if model["type"] != "BPE" or model.get("byte_fallback") or tok.get("normalizer") is not None:
            raise NotImplementedError("ZONOS3: unexpected tokenizer type")

        vocab: dict[str, int] = model["vocab"]
        n_bpe = len(vocab)
        if n_bpe + TEXT_OFFSET != self.hparams["text_vocab"]:
            raise ValueError(f"ZONOS3: tokenizer has {n_bpe} tokens, text_vocab is {self.hparams['text_vocab']}")
        added = {t["id"] for t in tok.get("added_tokens", [])}

        tokens: list[str] = ["<|unused_0|>", "<|unused_1|>", "<|bos|>", "<|eos|>"]
        toktypes: list[int] = [gguf.TokenType.CONTROL] * TEXT_OFFSET
        inv = {i: t for t, i in vocab.items()}
        for i in range(n_bpe):
            tokens.append(inv[i])
            toktypes.append(gguf.TokenType.CONTROL if i in added else gguf.TokenType.NORMAL)

        # padding row of the text table (the text column of every generated row)
        tokens.append("<|text_pad|>")
        toktypes.append(gguf.TokenType.CONTROL)

        # condition rows, the last bucket of each table is "unknown"
        for cond, n in zip(self.hparams["condition_names"], self.cond_counts):
            tokens += [f"<|{cond}_{i}|>" for i in range(n - 1)] + [f"<|{cond}_unk|>"]
            toktypes += [gguf.TokenType.CONTROL] * n

        merges = [" ".join(m) if isinstance(m, list) else m for m in model["merges"]]

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("zonos3")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_bos_token_id(TEXT_BOS)
        self.gguf_writer.add_eos_token_id(TEXT_EOS)
        self.gguf_writer.add_pad_token_id(TEXT_OFFSET + 1)  # <|pad|> of the tokenizer
        # the reference wraps the text as [BOS] + ids + [EOS]
        self.gguf_writer.add_add_bos_token(True)
        self.gguf_writer.add_add_eos_token(True)

    def set_gguf_parameters(self):
        hp = self.hparams
        n_layer = self.block_count
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_block_count(n_layer)
        self.gguf_writer.add_context_length(hp["max_seqlen"])
        self.gguf_writer.add_embedding_length(self.n_embd)
        self.gguf_writer.add_feed_forward_length(self.n_ff)
        self.gguf_writer.add_head_count(hp["n_heads"])
        self.gguf_writer.add_head_count_kv(hp["n_kv_heads"])
        self.gguf_writer.add_key_length(hp["head_dim"])
        self.gguf_writer.add_value_length(hp["head_dim"])
        self.gguf_writer.add_rope_dimension_count(hp["head_dim"])
        self.gguf_writer.add_rope_freq_base(hp["rope_theta"])
        self.gguf_writer.add_layer_norm_rms_eps(hp["norm_eps"])
        self.gguf_writer.add_full_attention_interval(self.gdn_period)

        # gated delta net
        self.gguf_writer.add_ssm_conv_kernel(GDN_KERNEL)
        self.gguf_writer.add_ssm_inner_size(GDN_HEADS * GDN_HEAD_DIM)
        self.gguf_writer.add_ssm_state_size(GDN_HEAD_DIM)
        self.gguf_writer.add_ssm_time_step_rank(GDN_HEADS)
        self.gguf_writer.add_ssm_group_count(GDN_HEADS)

        # latent MoE, the gating (softmax over the selected logits) is hard-coded in libllama
        self.gguf_writer.add_leading_dense_block_count(hp["moe_start_from_layer"])
        self.gguf_writer.add_expert_count(hp["moe_n_experts"])
        self.gguf_writer.add_expert_used_count(hp["moe_router_topk"])
        self.gguf_writer.add_expert_feed_forward_length(self.n_ff_exp)
        self.gguf_writer.add_moe_latent_size(self.n_latent)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # router logits pick the experts, keep them exact
        if new_name.endswith(("ffn_gate_inp.weight", "exp_probs_b.bias")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _token_embd(self) -> Tensor:
        n_cb = self.n_codebooks
        pad = sum(self._embd[f"multi_embedder.embedders.{cb}.weight"][self.audio_pad_id].float() for cb in range(n_cb))
        text = self._embd[f"multi_embedder.embedders.{n_cb}.weight"].float()
        if text.shape[0] != self.text_rows:
            raise ValueError(f"ZONOS3: text table has {text.shape[0]} rows, expected {self.text_rows}")
        conds = [self._embd[f"condition_embeddings.{i}.weight"].float() for i in range(len(self.cond_counts))]
        for t, n in zip(conds, self.cond_counts):
            assert t.shape[0] == n
        # a text row is [pad x n_codebooks, text_id], condition rows replace the summed row
        return torch.cat([text + pad] + conds, dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        hp = self.hparams
        n_layer = self.block_count

        if name.startswith(("multi_embedder.", "condition_embeddings.")):
            self._embd[name] = data_torch
            if len(self._embd) == self.n_codebooks + 1 + len(self.cond_counts):
                yield (self.format_tensor_name(T.TOKEN_EMBD), self._token_embd())
                self._embd.clear()
            return

        if name == "block_attnres_queries":
            # one query per routing point: 2*il before the mixer, 2*il+1 before the FFN, the last one before out_norm
            assert data_torch.shape[0] == 2 * n_layer + 1
            for il in range(n_layer):
                yield (self.format_tensor_name(T.ATTN_RES_SCORE, il), data_torch[2 * il])
                yield (self.format_tensor_name(T.FFN_RES_SCORE, il), data_torch[2 * il + 1])
            yield (self.format_tensor_name(T.OUTPUT_RES_SCORE), data_torch[2 * n_layer])
            return

        if name == "out_norm.weight":
            yield (self.format_tensor_name(T.OUTPUT_NORM), data_torch)
            return

        assert bid is not None, name
        key = name.split(".", 2)[2]  # strip "layers.{bid}."

        if key == "attention_norm.weight":
            yield (self.format_tensor_name(T.ATTN_NORM, bid), data_torch)
            return
        if key == "ffn_norm.weight":
            yield (self.format_tensor_name(T.FFN_NORM, bid), data_torch)
            return

        # gated delta net
        if key == "attention.in_proj.weight":
            # split the fused [qkv | z | a | b] weight into the qwen35 GDN tensors
            # attn_gate holds z, per-head output gate in the attention layers
            inner = GDN_HEADS * GDN_HEAD_DIM
            qkv, z, a, b = data_torch.split([3 * inner, inner, GDN_HEADS, GDN_HEADS], dim=0)
            yield (self.format_tensor_name(T.ATTN_QKV, bid), qkv)
            yield (self.format_tensor_name(T.ATTN_GATE, bid), z)
            yield (self.format_tensor_name(T.SSM_ALPHA, bid), a)
            yield (self.format_tensor_name(T.SSM_BETA, bid), b)
            return
        if key == "attention.conv.weight":
            yield (self.format_tensor_name(T.SSM_CONV1D, bid), data_torch.squeeze(1))
            return
        if key == "attention.A_log":
            yield (self.format_tensor_name(T.SSM_A, bid, suffix=""), -torch.exp(data_torch.float()))
            return
        if key == "attention.dt_bias":
            yield (self.format_tensor_name(T.SSM_DT, bid, suffix=".bias"), data_torch)
            return
        if key == "attention.out_weight":
            yield (self.format_tensor_name(T.SSM_NORM, bid), data_torch)
            return
        if key == "attention.wo.weight" and (bid + 1) % self.gdn_period != 0:
            yield (self.format_tensor_name(T.SSM_OUT, bid), data_torch)
            return

        # global attention
        if key == "attention.wq.weight":
            yield (self.format_tensor_name(T.ATTN_Q, bid), data_torch)
            return
        if key == "attention.wkv.weight":
            yield (self.format_tensor_name(T.ATTN_K, bid), data_torch[0])
            yield (self.format_tensor_name(T.ATTN_V, bid), data_torch[1])
            return
        if key == "attention.wo.weight":
            yield (self.format_tensor_name(T.ATTN_OUT, bid), data_torch)
            return
        if key == "attention.gater.weight":
            # one sigmoid gate per head, same tensor name as the GDN z gate
            yield (self.format_tensor_name(T.ATTN_GATE, bid), data_torch)
            return
        if key == "attention.temp":
            # Q = rms_norm(q) * |temp|, the per-head scale is stored as the Q norm weight
            temp = data_torch.float().abs().reshape(hp["n_heads"], 1).expand(hp["n_heads"], hp["head_dim"])
            yield (self.format_tensor_name(T.ATTN_Q_NORM, bid), temp.contiguous())
            return

        # dense FFN (layer 0): out = w_out(w_in[0](x) * silu(w_in[1](x)))
        if key == "feed_forward.w_in.weight":
            yield (self.format_tensor_name(T.FFN_UP, bid), data_torch[0])
            yield (self.format_tensor_name(T.FFN_GATE, bid), data_torch[1])
            return
        if key == "feed_forward.w_out.weight":
            yield (self.format_tensor_name(T.FFN_DOWN, bid), data_torch)
            return

        # latent MoE
        if key in ("feed_forward.router.router_linear.weight", "feed_forward.router.rmsnorm_eda.weight"):
            # logits = router_linear(rms_norm(x) * w), the norm weight is folded into the router columns
            r = self._router.setdefault(bid, {})
            r[key] = data_torch
            if len(r) == 2:
                w = r["feed_forward.router.router_linear.weight"].float() * r["feed_forward.router.rmsnorm_eda.weight"].float()[None, :]
                yield (self.format_tensor_name(T.FFN_GATE_INP, bid), w)
                del self._router[bid]
            return
        if key == "feed_forward.router.balancing_biases":
            # the reference selects on logits - bias
            yield (self.format_tensor_name(T.FFN_EXP_PROBS_B, bid, suffix=".bias"), -data_torch.float())
            return
        if key == "feed_forward.latent_down.weight":
            yield (self.format_tensor_name(T.MOE_LATENT_DOWN, bid), data_torch)
            return
        if key == "feed_forward.latent_up.weight":
            yield (self.format_tensor_name(T.MOE_LATENT_UP, bid), data_torch)
            return
        if key == "feed_forward.experts.gate_up_proj":
            gate, up = data_torch.chunk(2, dim=1)
            yield (self.format_tensor_name(T.FFN_GATE_EXP, bid), gate)
            yield (self.format_tensor_name(T.FFN_UP_EXP, bid), up)
            return
        if key == "feed_forward.experts.down_proj":
            yield (self.format_tensor_name(T.FFN_DOWN_EXP, bid), data_torch)
            return

        raise ValueError(f"ZONOS3: unexpected tensor {name}")
