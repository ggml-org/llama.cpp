from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger

_INKLING_PRE_REGEX = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


class _StreamedBytes:
    """Raw tensor bytes produced one expert at a time at write time."""
    def __init__(self, shape: tuple[int, ...], chunks: Callable[[], Iterable[np.ndarray]]):
        self.shape = shape
        self.dtype = np.dtype(np.uint8)
        self.nbytes = int(np.prod(shape))
        self._chunks = chunks

    def tofile(self, fout) -> None:
        written = 0
        for chunk in self._chunks():
            chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
            chunk.tofile(fout)
            written += chunk.nbytes
        if written != self.nbytes:
            raise ValueError(f"streamed {written} bytes, expected {self.nbytes}")

    def byteswap(self, inplace: bool = False):
        raise NotImplementedError("streamed tensors are written little-endian only")


def nvfp4_to_ggml(packed: Tensor, scale: Tensor) -> np.ndarray:
    """Checkpoint NVFP4 rows (uint8 [N, K/2], e4m3 scales [N, K/16]) -> block_nvfp4 rows, bit-exact."""
    n, half = packed.shape
    nb = scale.shape[1]
    assert half * 2 == nb * 16 and nb % 4 == 0, (packed.shape, scale.shape)
    sbits = scale.view(torch.uint8)
    if bool((sbits & 0x80).any()) or bool(((sbits & 0x7F) == 0x7F).any()):
        raise ValueError("NVFP4 block scales must be positive and finite (UE4M3)")
    w = packed.reshape(n, nb, 8)
    vals = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(n, nb, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).to(torch.uint8).numpy()
    d = sbits.numpy().reshape(n, nb // 4, 4)
    q = qs.reshape(n, nb // 4, 32)
    return np.concatenate([d, q], axis=-1).reshape(n, nb // 4 * 36)


def mxfp8_to_f32(values: Tensor, scale: Tensor) -> Tensor:
    """e4m3 [N, K] * 2^(ue8m0 - 127) per 32 -> fp32, exact in BF16 for ue8m0 in [3, 246]."""
    if scale.numel() and (int(scale.min()) < 3 or int(scale.max()) > 246):
        raise ValueError("MXFP8 ue8m0 exponents outside [3, 246]: BF16 would not be exact")
    return values.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)


@ModelBase.register("Inkling2ForConditionalGeneration")
class Inkling2Model(TextModel):
    """NVFP4 experts are repacked bit-exact with a per-expert F32 scale (amax / outer_scale_denominator), MXFP8
    experts are stored as BF16 (exact). The short-conv residual and output scale are folded into the taps."""
    model_arch = gguf.MODEL_ARCH.INKLING2
    undo_permute = False

    _SKIP_PREFIXES = ("model.visual.", "model.vision.", "model.audio.", "model.mtp.", "mtp.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        if hp.get("q_bias", False) or hp.get("o_bias", False):
            raise NotImplementedError("attention q_bias / o_bias are not supported")
        for a, b in (("swa_head_dim", "head_dim"), ("swa_num_attention_heads", "num_attention_heads"),
                     ("swa_num_key_value_heads", "num_key_value_heads")):
            if hp.get(a, hp[b]) != hp[b]:
                raise ValueError(f"{a} {hp[a]} must equal {b} {hp[b]}")
        for k in ("use_global_scale", "pre_norm_pre_affine", "final_norm_pre_affine", "n_shared_experts"):
            if hp.get(k):
                raise NotImplementedError(f"{k} is not supported")
        if (mml := hp.get("model_max_length")) is not None:
            self.hparams["max_position_embeddings"] = mml
        n_layer = hp["num_hidden_layers"]
        local = hp.get("local_layer_ids")
        if local is None:
            local = [i for i in range(n_layer) if i % 6 != 5]
        self._local = [i in set(local) for i in range(n_layer)]
        self._qfmt: dict[str, str] = {}
        qf = Path(self.dir_model) / "hf_quant_config.json"
        if qf.is_file():
            for g in json.loads(qf.read_text()).get("model", {}).get("parameter_groups", []):
                for t in g["tensors"]:
                    self._qfmt[t] = g["compute_dtype"]

    def dequant_model(self):
        return

    def get_vocab_base_pre(self, tokenizer) -> str:
        tj = Path(self.dir_model) / "tokenizer.json"
        pre = json.loads(tj.read_text()).get("pre_tokenizer") or {}
        pats = [p.get("pattern", {}).get("Regex") for p in pre.get("pretokenizers", []) if p.get("type") == "Split"]
        if pats != [_INKLING_PRE_REGEX]:
            raise NotImplementedError("Inkling2 tokenizer pre-tokenizer differs from the Inkling regex")
        return "inkling"

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens, toktypes, tokpre = super().get_vocab_base()
        n_vocab = self.hparams["vocab_size"]
        n_unpadded = self.hparams.get("unpadded_vocab_size") or n_vocab
        if len(tokens) != n_vocab:
            raise ValueError(f"Inkling2 tokenizer produced {len(tokens)} entries, expected {n_vocab}")
        tc = json.loads((Path(self.dir_model) / "tokenizer_config.json").read_text())
        for tid, d in tc.get("added_tokens_decoder", {}).items():
            if d.get("special") and 0 <= int(tid) < n_vocab:
                toktypes[int(tid)] = gguf.TokenType.CONTROL
        if any(t != gguf.TokenType.UNUSED for t in toktypes[n_unpadded:]):
            raise ValueError("real tokens found at/above unpadded_vocab_size; padded-vocab mask would hide them")
        return tokens, toktypes, tokpre

    def set_vocab(self):
        self._set_vocab_gpt2()
        eos_id = int(self.hparams.get("eos_token_id", 200006))
        self.gguf_writer.add_eos_token_id(eos_id)
        self.gguf_writer.add_bos_token_id(eos_id)
        self.gguf_writer.add_add_bos_token(False)
        tmpl = Path(__file__).resolve().parent.parent / "models" / "templates" / "Inkling2.jinja"
        if tmpl.is_file():
            self.gguf_writer.add_chat_template(tmpl.read_text())

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(hp["route_scale"])
        if hp.get("final_logit_softcapping"):
            self.gguf_writer.add_final_logit_softcapping(hp["final_logit_softcapping"])
        window = hp["sliding_window_size"]
        self.gguf_writer.add_sliding_window(window)
        self.gguf_writer.add_sliding_window_pattern(self._local)
        self.gguf_writer.add_uint32(f"{arch}.d_rel", hp["d_rel"])
        self.gguf_writer.add_uint32(f"{arch}.rel_extent", hp["rel_extent"])
        self.gguf_writer.add_uint32(f"{arch}.rel_extent_swa", window)
        self.gguf_writer.add_uint32(f"{arch}.shortconv_kernel", hp["sconv_kernel_size"])
        self.gguf_writer.add_uint32(f"{arch}.log_scaling_n_floor", int(hp.get("log_scaling_n_floor") or 0))
        self.gguf_writer.add_float32(f"{arch}.log_scaling_alpha", hp.get("log_scaling_alpha", 0.0))
        self.gguf_writer.add_uint32(f"{arch}.unpadded_vocab_size", hp.get("unpadded_vocab_size") or hp["vocab_size"])
        self.gguf_writer.add_float32(f"{arch}.qk_clip_threshold", hp["qk_l2_clip_threshold"])
        self.gguf_writer.add_float32(f"{arch}.rel_clip_threshold", hp["rel_l2_clip_threshold"])
        self.gguf_writer.add_attn_res_block_size(hp["n_layers_per_attn_res_block"])
        self.gguf_writer.add_bool(f"{arch}.attn_res_normalize_values", bool(hp.get("attn_res_normalize_values", False)))
        self.gguf_writer.add_float32(f"{arch}.mlp_gate_softcap", float(hp.get("mlp_gate_softcap", 0.0)))
        self.gguf_writer.add_float32(f"{arch}.mlp_up_softcap", float(hp.get("mlp_up_softcap", 0.0)))
        self.gguf_writer.add_bool(f"{arch}.value_norm", bool(hp.get("use_value_norm_no_gamma", False)))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(cls._SKIP_PREFIXES):
            return None
        return super().filter_tensors((name, gen))

    _ATTN = {
        "wq_du.weight": gguf.MODEL_TENSOR.ATTN_Q, "wk_dv.weight": gguf.MODEL_TENSOR.ATTN_K,
        "wv_dv.weight": gguf.MODEL_TENSOR.ATTN_V, "wr_du.weight": gguf.MODEL_TENSOR.ATTN_R,
        "wo_ud.weight": gguf.MODEL_TENSOR.ATTN_OUT, "gate_attn.weight": gguf.MODEL_TENSOR.ATTN_GATE,
        "attn_sink_proj.weight": gguf.MODEL_TENSOR.ATTN_SINK_PROJ, "q_l2_clip.scale": gguf.MODEL_TENSOR.ATTN_Q_CLIP,
        "rel_l2_clip.scale": gguf.MODEL_TENSOR.ATTN_R_CLIP, "rel_logits_proj.proj": gguf.MODEL_TENSOR.ATTN_REL_PROJ,
    }
    _NORMS = {
        "attn_norm": gguf.MODEL_TENSOR.ATTN_NORM, "q_input_norm": gguf.MODEL_TENSOR.ATTN_Q_IN_NORM,
        "k_input_norm": gguf.MODEL_TENSOR.ATTN_K_IN_NORM, "v_input_norm": gguf.MODEL_TENSOR.ATTN_V_IN_NORM,
        "mlp_norm": gguf.MODEL_TENSOR.FFN_NORM,
    }

    def _eager(self, name: str) -> Tensor:
        return LazyTorchTensor.to_eager(self.model_tensors[name]())

    def _fold_static(self, w: Tensor, dynamic: bool) -> Tensor:
        """[C, W] taps (tap W-1 = current token) -> (w + e_last) * (W (x2 if dynamic) + 1)^-1/2."""
        w = LazyTorchTensor.to_eager(w).float().clone()
        k = w.shape[-1]
        w[:, k - 1] += 1.0
        return w * (k * (2 if dynamic else 1) + 1) ** -0.5

    def _write_experts(self, bid: int, src: str, which: str) -> list[tuple[str, Tensor]]:
        base = src + which
        fmt = self._qfmt.get(base)
        if which == "w13_weight":
            outs = [(gguf.MODEL_TENSOR.FFN_GATE_EXP, 0), (gguf.MODEL_TENSOR.FFN_UP_EXP, 1)]
        else:
            outs = [(gguf.MODEL_TENSOR.FFN_DOWN_EXP, None)]
        if fmt == "nvfp4":
            amax = self._eager(base + "_amax").float()
            den = self._eager(base + "_outer_scale_denominator").float()
            scale = (amax / den).numpy().astype(np.float32)
            w_gen, s_gen = self.model_tensors[base], self.model_tensors[base + "_scale"]
            e, n, half = w_gen().shape
            for key, sel in outs:
                rows = n if sel is None else n // 2

                def chunks(sel=sel):
                    w_all = LazyTorchTensor.to_eager(w_gen())
                    s_all = LazyTorchTensor.to_eager(s_gen())
                    for x in range(e):
                        w, s = w_all[x], s_all[x]
                        if sel is not None:
                            w, s = w[sel::2], s[sel::2]
                        yield nvfp4_to_ggml(w.contiguous(), s.contiguous())

                name = self.format_tensor_name(key, bid)
                self.gguf_writer.add_tensor(name, _StreamedBytes((e, rows, half * 2 // 64 * 36), chunks),
                                            raw_dtype=gguf.GGMLQuantizationType.NVFP4)
                self.gguf_writer.add_tensor(self.format_tensor_name(key, bid, ".scale"), scale)
        elif fmt == "mxfp8":
            v_gen, s_gen = self.model_tensors[base], self.model_tensors[base + "_scale"]
            e, n, k = v_gen().shape
            for key, sel in outs:
                rows = n if sel is None else n // 2

                def chunks(sel=sel):
                    v_all = LazyTorchTensor.to_eager(v_gen())
                    s_all = LazyTorchTensor.to_eager(s_gen())
                    for x in range(e):
                        v, s = v_all[x], s_all[x]
                        if sel is not None:
                            v, s = v[sel::2], s[sel::2]
                        f = mxfp8_to_f32(v, s)
                        bf = f.to(torch.bfloat16)
                        assert torch.equal(bf.float(), f), "MXFP8 value not exact in BF16"
                        yield bf.view(torch.uint8).numpy()

                self.gguf_writer.add_tensor(self.format_tensor_name(key, bid), _StreamedBytes((e, rows, k * 2), chunks),
                                            raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            data = self.model_tensors[base]()
            return [(self.format_tensor_name(key, bid), data if sel is None else data[:, sel::2, :]) for key, sel in outs]
        return []

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n = name.replace("model.llm.", "")
        if n == "embed.weight":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)]
        if n == "embed_norm.weight":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD_NORM), data_torch)]
        if n == "unembed.weight":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)]
        if n == "norm.weight":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)]
        if n == "attn_res.output_query":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_RES_SCORE), data_torch.float())]
        if n.startswith("attn_res.attn_queries."):  # layers 1..L-1
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_RES_SCORE, int(n.rsplit(".", 1)[1]) + 1), data_torch.float())]
        if n.startswith("attn_res.mlp_queries."):
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_RES_SCORE, int(n.rsplit(".", 1)[1])), data_torch.float())]
        assert bid is not None, name
        rest = n.split(f"layers.{bid}.", 1)[1]
        root, _, leaf = rest.partition(".")
        if root in self._NORMS:
            return [(self.format_tensor_name(self._NORMS[root], bid, "." + leaf), data_torch)]
        if root in ("attn_sconv", "mlp_sconv"):
            key = gguf.MODEL_TENSOR.SHORTCONV_ATTN if root == "attn_sconv" else gguf.MODEL_TENSOR.SHORTCONV_MLP
            return [(self.format_tensor_name(key, bid), self._fold_static(data_torch.squeeze(1), False))]
        if root == "attn":
            if leaf in self._ATTN:
                return [(self.format_tensor_name(self._ATTN[leaf], bid), data_torch)]
            kv, _, part = leaf.partition("_sconv.")
            key = gguf.MODEL_TENSOR.SHORTCONV_K if kv == "k" else gguf.MODEL_TENSOR.SHORTCONV_V
            dyn = gguf.MODEL_TENSOR.SHORTCONV_K_DYN if kv == "k" else gguf.MODEL_TENSOR.SHORTCONV_V_DYN
            if part == "weight":  # local: [C, 1, W], tap W-1 = current token
                return [(self.format_tensor_name(key, bid), self._fold_static(data_torch.squeeze(1), False))]
            if part == "static_w":  # global: [W, C] indexed by lag (0 = current) -> [C, W] taps
                taps = LazyTorchTensor.to_eager(data_torch).flip(0).T.contiguous()
                return [(self.format_tensor_name(key, bid), self._fold_static(taps, True))]
            if part == "weight_proj":  # global: [n_kv*W, H], row = head*W + lag; fixed scale folded in
                k = self.hparams["sconv_kernel_size"]
                return [(self.format_tensor_name(dyn, bid), LazyTorchTensor.to_eager(data_torch).float() * (2 * k + 1) ** -0.5)]
            raise ValueError(f"unexpected tensor {name}")
        if root == "mlp":
            if leaf == "gate.weight":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid), data_torch)]
            if leaf == "gate.bias":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid, ".bias"), data_torch.float())]
            src = name[: name.index(f"layers.{bid}.")] + f"layers.{bid}.mlp.experts."
            if leaf in ("experts.w13_weight", "experts.w2_weight"):
                return self._write_experts(bid, src, leaf.split(".")[1])
            if leaf.startswith("experts."):  # scales, consumed by _write_experts
                return []
        raise ValueError(f"unexpected tensor {name}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if new_name.endswith(("attn_rel_proj.weight", "_clip.weight", "_res_score.weight")) or "shortconv_" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
