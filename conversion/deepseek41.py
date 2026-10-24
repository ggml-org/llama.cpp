from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf
from .deepseek import DeepseekV4FlashVisionModel, DeepseekV4Model


# Engram hash constants from inference/engram.py, stored in the GGUF: the token map needs the HF tokenizers
# normalizers and the multipliers numpy's PCG64 stream, neither of which the runtime has.

def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def engram_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> list[list[list[int]]]:
    """EngramLayout.from_args: [layer][ngram order 2..max][head] bucket moduli, never reused across layers."""
    seen: set[int] = set()
    out: list[list[list[int]]] = []
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not _is_prime(cand) or cand in seen:
                    cand += 1
                cur = cand
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        out.append(per_ngram)
    return out


def engram_multipliers(layer_ids: list[int], max_ngram_size: int, compressed_vocab_size: int) -> list[list[int]]:
    """compute_hash_multipliers: odd, bounded so that token_id * multiplier cannot overflow int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


def engram_compressed_token_map(tokenizer_dir: Path) -> tuple[list[int], int]:
    """build_compressed_token_map: token id -> compressed id (tokens that normalize alike collapse)."""
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class _StreamedBytes:
    """Raw tensor bytes produced chunk by chunk at write time (an Engram table is ~200 GB in BF16)."""
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


def _e8m0_check(scale_u8: np.ndarray, name: str) -> None:
    # e4m3 (|x| in [2^-9, 448]) * 2^(e - 127) is exact in BF16 for e in [3, 246]
    if scale_u8.size and (int(scale_u8.min()) < 3 or int(scale_u8.max()) > 246):
        raise ValueError(f"{name}: E8M0 scale exponents outside [3, 246], the dequantized values would not be exact")


@ModelBase.register("DeepseekV41ForCausalLM")
class DeepseekV41Model(DeepseekV4Model):
    """DeepSeek-V4.1 text model, lossless: FP8 (e4m3, 32x32 ue8m0 blocks) linears dequantized to BF16 (F32 with
    --outtype f32), routed experts copied bit-exact as MXFP4, Engram tables (e4m3 rows, one e8m0 scale per 32) copied
    bit-exact as MXFP8 (DSV41_ENGRAM_TYPE=bf16: the exact BF16 lookup, F32 with --outtype f32), router and Engram gate
    F32. DSpark (mtp.*) is not converted here (--dspark, DeepseekV41DSparkModel); the vision tower goes to an mmproj
    (--mmproj, DeepseekV41VisionModel).
    The Engram tables of all layers are written as one per_layer_token_embd (rows concatenated in layer order) with
    the PLE keys of qwen4exp: absolute head offsets and bucket counts, so the runtime can read it lazily."""
    model_arch = gguf.MODEL_ARCH.DEEPSEEK41
    supports_mtp_export = False

    _ENGRAM_EMBED_RE = re.compile(r"layers\.(\d+)\.engram\.embed\.(weight|scale)$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.setdefault("num_hash_layers", 0)  # no hash-routed layers in V4.1
        self._f32 = self.ftype == gguf.LlamaFileType.ALL_F32
        self._engram_mxfp8 = os.environ.get("DSV41_ENGRAM_TYPE", "mxfp8").lower() != "bf16"
        qc = self.hparams.get("quantization_config") or {}
        self._fp8_block = tuple(qc.get("weight_block_size") or (32, 32))
        if qc.get("scale_fmt", "ue8m0") != "ue8m0":
            raise ValueError(f"unsupported FP8 scale format {qc.get('scale_fmt')!r}")
        if self.hparams.get("gate_temp", 1.0) != 1.0:
            raise ValueError("gate_temp != 1 is not supported (the router logits would need dividing by it)")

        # V4.1 does not use the DeepSeek-V4 template; a template shipped with the model is still added by set_vocab
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CHAT_TEMPLATE)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # num_hidden_layers lives in text_config, which is merged only after indexing; MTP stages are skipped anyway
        return TextModel.index_tensors(self, remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith("vision."):
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        w = self.gguf_writer
        w.add_attention_kv_source_layers(hp["kv_source_layer_ids"])
        w.add_indexer_source_layers(hp["index_source_layer_ids"])
        if hp.get("candidate_source_layer_id", -1) >= 0:
            w.add_indexer_candidate_source_layer(hp["candidate_source_layer_id"])
            w.add_indexer_candidate_top_k_blocks(hp["candidate_topk_blocks"])
            w.add_indexer_candidate_block_size(hp["candidate_block_size"])

        # GGUF cannot hold empty arrays: the Engram keys are omitted when the model has no Engram layers
        layer_ids = list(hp.get("engram_layer_ids") or [])
        if layer_ids:
            max_ngram, n_heads = hp["engram_max_ngram_size"], hp["engram_n_heads"]
            primes = engram_primes(len(layer_ids), max_ngram, n_heads, hp["engram_vocab_size"])
            for li, layer_primes in enumerate(primes):
                if sum(p for per in layer_primes for p in per) != hp["engram_num_embeddings"][li]:
                    raise ValueError(f"engram layer {layer_ids[li]}: prime sum != num_embeddings")
            token_map, cvocab = engram_compressed_token_map(self.dir_model)
            if cvocab != hp["engram_compressed_vocab_size"] or len(token_map) != hp["vocab_size"]:
                raise ValueError(f"compressed token map ({len(token_map)} -> {cvocab}) does not match the config")
            # absolute row offsets into the concatenated table
            sizes = [p for layer in primes for per in layer for p in per]
            offsets = [sum(sizes[:i]) for i in range(len(sizes))]
            w.add_key_value(gguf.Keys.PerLayerEmbedding.LAYERS.format(arch=w.arch), layer_ids,
                            gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
            w.add_ple_ngram_size(max_ngram)
            w.add_ple_heads_per_ngram(n_heads)
            w.add_embedding_length_per_layer_input(hp["engram_head_dim"])
            w.add_ple_layer_multipliers([m for row in engram_multipliers(layer_ids, max_ngram, cvocab) for m in row])
            w.add_ple_head_offsets(offsets)
            w.add_ple_head_vocab_sizes(sizes)
            w.add_engram_pad_token_id(hp["engram_pad_token_id"])
            w.add_engram_token_map(token_map)

    def dequant_model(self):
        bo, bi = self._fp8_block

        def dequant(weight: Tensor, scale: Tensor) -> Tensor:
            out_f, in_f = weight.shape
            s = self._e8m0_to_float(scale).repeat_interleave(bo, 0)[:out_f].repeat_interleave(bi, 1)[:, :in_f]
            return weight.float() * s

        for name in [n for n in self.model_tensors if n.endswith(".scale") and not self._ENGRAM_EMBED_RE.match(n)]:
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name not in self.model_tensors or self.model_tensors[weight_name]().dtype not in self._float8_dtypes():
                continue
            w, s = self.model_tensors[weight_name], self.model_tensors.pop(name)
            out_f, in_f = w().shape
            if tuple(s().shape) != (-(-out_f // bo), -(-in_f // bi)):
                raise ValueError(f"{weight_name}: scale shape {tuple(s().shape)} does not match {(bo, bi)} blocks")
            _e8m0_check(LazyTorchTensor.to_eager(s()).view(torch.uint8).numpy(), name)
            self.model_tensors[weight_name] = lambda w=w, s=s: dequant(w(), s())
            self._dsv4_fp8_dequantized.add(weight_name)

    def _tensor_part(self, name: str) -> Path:
        index = self.dir_model / "model.safetensors.index.json"
        if not index.is_file():
            return self.dir_model / "model.safetensors"
        with open(index, "r", encoding="utf-8") as f:
            return self.dir_model / json.load(f)["weight_map"][name]

    def _write_mxfp4_experts(self, bid: int, proj: str, key: gguf.MODEL_TENSOR, n_expert: int | None = None) -> None:
        n_expert = self.hparams["n_routed_experts"] if n_expert is None else n_expert
        names = [f"layers.{bid}.ffn.experts.{e}.{proj}" for e in range(n_expert)]
        pairs = [(self.model_tensors.pop(n + ".weight"), self.model_tensors.pop(n + ".scale")) for n in names]
        rows, packed_cols = pairs[0][0]().shape

        def chunks() -> Iterable[np.ndarray]:
            for w, s in pairs:
                yield self.repack_mxfp4_blocks(LazyTorchTensor.to_eager(w()), LazyTorchTensor.to_eager(s()))
        data = cast(np.ndarray, _StreamedBytes((len(pairs), rows, packed_cols * 2 // 32 * 17), chunks))
        self.gguf_writer.add_tensor(self.format_tensor_name(key, bid), data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

    def _write_engram_tables(self, layer_ids: list[int]) -> None:
        # official lookup e4m3 * 2^(e8m0 - 127): stored as the source MXFP8 blocks (bit-exact) or as exact BF16
        parts, dim = [], None
        for bid in layer_ids:
            wn, sn = f"layers.{bid}.engram.embed.weight", f"layers.{bid}.engram.embed.scale"
            rows, d = self.model_tensors.pop(wn)().shape
            if d % 32 != 0 or tuple(self.model_tensors.pop(sn)().shape) != (rows, d // 32) or d != (dim or d):
                raise ValueError(f"{wn}: expected one e8m0 scale per 32 elements and the same width in every layer")
            if rows != self.hparams["engram_num_embeddings"][len(parts)]:
                raise ValueError(f"{wn}: {rows} rows, config says {self.hparams['engram_num_embeddings'][len(parts)]}")
            dim = d
            parts.append((wn, sn, rows, self._tensor_part(wn), self._tensor_part(sn)))
        assert dim is not None
        name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
        n_rows = sum(p[2] for p in parts)

        if self._engram_mxfp8:
            def blocks() -> Iterable[np.ndarray]:
                from safetensors import safe_open
                for wn, sn, rows, wpart, spart in parts:
                    with safe_open(wpart, framework="pt") as fw, safe_open(spart, framework="pt") as fs:
                        ws, ss = fw.get_slice(wn), fs.get_slice(sn)
                        for r0 in range(0, rows, 1 << 18):
                            q = ws[r0:r0 + (1 << 18)].view(torch.uint8).reshape(-1, 32).numpy()
                            e = ss[r0:r0 + (1 << 18)].view(torch.uint8).reshape(-1, 1).numpy()
                            if (e == 0xFF).any() or ((q & 0x7F) == 0x7F).any():
                                raise ValueError(f"{wn}: NaN E8M0 scale or E4M3 code")
                            yield np.concatenate([e, q], axis=1)
            data = cast(np.ndarray, _StreamedBytes((n_rows, dim // 32 * 33), blocks))
            self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP8)
            return

        def chunks() -> Iterable[np.ndarray]:
            from safetensors import safe_open
            for wn, sn, rows, wpart, spart in parts:
                with safe_open(wpart, framework="pt") as fw, safe_open(spart, framework="pt") as fs:
                    ws, ss = fw.get_slice(wn), fs.get_slice(sn)
                    for r0 in range(0, rows, 1 << 18):
                        q = ws[r0:r0 + (1 << 18)].view(torch.uint8).reshape(-1, 32)
                        e = ss[r0:r0 + (1 << 18)].view(torch.uint8).reshape(-1, 1)
                        _e8m0_check(e.numpy(), sn)
                        v = (q.view(torch.float8_e4m3fn).float() * torch.exp2(e.float() - 127)).reshape(-1, dim)
                        yield (v if self._f32 else v.bfloat16()).view(torch.uint8).numpy()
        qt = gguf.GGMLQuantizationType.F32 if self._f32 else gguf.GGMLQuantizationType.BF16
        data = cast(np.ndarray, _StreamedBytes((n_rows, dim * (4 if self._f32 else 2)), chunks))
        self.gguf_writer.add_tensor(name, data, raw_dtype=qt)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if not self._dsv4_mxfp4_generated:
            for bid in range(self.block_count):
                self._write_mxfp4_experts(bid, "w1", gguf.MODEL_TENSOR.FFN_GATE_EXP)
                self._write_mxfp4_experts(bid, "w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP)
                self._write_mxfp4_experts(bid, "w3", gguf.MODEL_TENSOR.FFN_UP_EXP)
            if self.hparams.get("engram_layer_ids"):
                self._write_engram_tables(list(self.hparams["engram_layer_ids"]))
            self._dsv4_mxfp4_generated = True
        return ()

    _LAYER_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
        "attn.indexer.wk.weight": (gguf.MODEL_TENSOR.INDEXER_ATTN_K, ".weight"),
        "attn.indexer.k_norm.weight": (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".weight"),
        "engram.wkv.weight": (gguf.MODEL_TENSOR.ENGRAM_KV, ".weight"),
    }

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        m = re.match(r"layers\.\d+\.(.+)$", name)
        if m is not None and m.group(1) in self._LAYER_MAP:
            return self._LAYER_MAP[m.group(1)]
        return super()._map_dsv4_tensor_name(name, bid)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".engram.k_weight"):
            return []  # folded into engram_gate
        if name.endswith(".engram.q_weight"):
            # Engram gate weight = q_weight * k_weight (bf16 * bf16 is exact in f32)
            k = self.model_tensors[name.removesuffix("q_weight") + "k_weight"]
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENGRAM_GATE, bid), data_torch.float() * k().float())]
        return super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith((".engram_gate.weight", ".ffn_gate_inp.weight")) or name in self._dsv4_f32_tensors:
            return gguf.GGMLQuantizationType.F32
        if n_dims >= 2 and (name in self._dsv4_fp8_dequantized or name in self._dsv4_bf16_tensors):
            return gguf.GGMLQuantizationType.F32 if self._f32 else gguf.GGMLQuantizationType.BF16
        return False


class DeepseekV41DSparkModel(DeepseekV41Model):
    """DeepSeek-V4.1 DSpark speculative draft (mtp.0..2 of the checkpoint) as a DFLASH GGUF (`--dspark`).

    Layout = the V4 DSpark draft (conversion/deepseek.py DeepseekV4DSparkModel) with V4.1's differences flagged by
    `dflash.hyper_connection.single_pass`: stages blk.0..2 (MXFP4 experts bit-exact, dense BF16), mtp.0.main_proj /
    main_norm -> fc / enc.output_norm, mtp.2.norm -> output_norm, markov_head.embed / head -> markov_w1 / markov_w2,
    confidence_head.proj -> conf_proj (F32), no hc_head. Token embeddings and the lm head are the target's (shared at
    runtime through ctx_other). target_layers are the V4.1 ids as-is: the draft reads the attention input of those
    layers, which is what llama_set_embeddings_layer_inp extracts.
    """
    model_arch = gguf.MODEL_ARCH.DFLASH
    mtp_only = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self._n_main = hp["num_hidden_layers"]
        self._n_stage = hp["num_nextn_predict_layers"]
        if self._n_stage <= 0 or "dspark_block_size" not in hp:
            raise ValueError("DeepSeek-V4.1 DSpark export needs num_nextn_predict_layers > 0 and the dspark_* config")
        # stages keep main-model block ids (mtp.M -> layers.n_main+M); format_tensor_name() maps them back
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self._n_stage)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        hp = {**self.hparams, **self.hparams.get("text_config", {})}  # text_config is merged only after indexing
        type(self)._dsv4_main_layers = hp["num_hidden_layers"]
        type(self)._dsv4_nextn_layers = hp.get("num_nextn_predict_layers", 0)
        return TextModel.index_tensors(self, remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        m = re.match(r"mtp\.(\d+)\.(.+)$", name)
        if m is None:
            return None  # shared with the target at runtime
        stage, rest = int(m.group(1)), m.group(2)
        assert cls._dsv4_main_layers is not None
        if stage >= cls._dsv4_nextn_layers:
            raise ValueError(f"Unexpected DSpark stage {stage}")
        if rest.startswith(("markov_head.", "confidence_head.")):
            return f"dspark.{rest}", gen
        return f"layers.{cls._dsv4_main_layers + stage}.{rest}", gen

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if bid is not None and bid >= self._n_main:
            bid -= self._n_main
        return super().format_tensor_name(key, bid, suffix)

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_mask_token_id(self.hparams["dspark_noise_token_id"])

    def set_gguf_parameters(self):
        hp = self.hparams
        n = self._n_stage
        w = self.gguf_writer
        w.add_block_count(n)
        w.add_context_length(hp["max_position_embeddings"])
        w.add_embedding_length(hp["hidden_size"])
        w.add_feed_forward_length(hp["moe_intermediate_size"])
        w.add_head_count(hp["num_attention_heads"])
        w.add_head_count_kv(1)
        w.add_key_length(hp["head_dim"])
        w.add_value_length(hp["head_dim"])
        w.add_rope_dimension_count(hp["qk_rope_head_dim"])
        # compress-ratio-0 layers: plain rope at rope_theta, no YaRN
        w.add_rope_freq_base(float(hp["rope_theta"]))
        w.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        w.add_q_lora_rank(hp["q_lora_rank"])
        w.add_sliding_window(hp["sliding_window"])
        w.add_expert_count(hp["dspark_n_routed_experts"])
        w.add_expert_used_count(hp["dspark_num_experts_per_tok"])
        w.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        w.add_expert_shared_count(hp["n_shared_experts"])
        w.add_expert_weights_scale(hp["routed_scaling_factor"])
        w.add_expert_weights_norm(hp["norm_topk_prob"])
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)
        w.add_swiglu_clamp_exp([hp["swiglu_limit"]] * n)
        w.add_swiglu_clamp_shexp([hp["swiglu_limit"]] * n)
        w.add_attention_output_group_count(hp["o_groups"])
        w.add_attention_output_lora_rank(hp["o_lora_rank"])
        w.add_attention_compress_ratios([0] * n)
        w.add_hyper_connection_count(hp["hc_mult"])
        w.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        w.add_hyper_connection_epsilon(hp["hc_eps"])
        w.add_hyper_connection_single_pass(True)
        w.add_block_size(hp["dspark_block_size"])
        w.add_target_layers(hp["dspark_target_layer_ids"])
        w.add_sample_from_anchor(True)
        w.add_has_confidence_head(True)
        w.add_file_type(self.ftype)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # no Engram tables in the draft
        if not self._dsv4_mxfp4_generated:
            n_expert = self.hparams["dspark_n_routed_experts"]
            for bid in range(self._n_main, self._n_main + self._n_stage):
                self._write_mxfp4_experts(bid, "w1", gguf.MODEL_TENSOR.FFN_GATE_EXP, n_expert)
                self._write_mxfp4_experts(bid, "w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP, n_expert)
                self._write_mxfp4_experts(bid, "w3", gguf.MODEL_TENSOR.FFN_UP_EXP, n_expert)
            self._dsv4_mxfp4_generated = True
        return ()

    _DSPARK_ROOT: dict[str, gguf.MODEL_TENSOR] = {
        "main_proj.weight": gguf.MODEL_TENSOR.FC,
        "main_norm.weight": gguf.MODEL_TENSOR.ENC_OUTPUT_NORM,
        "norm.weight": gguf.MODEL_TENSOR.OUTPUT_NORM,
        "dspark.markov_head.embed.weight": gguf.MODEL_TENSOR.DSPARK_MARKOV_W1,
        "dspark.markov_head.head.weight": gguf.MODEL_TENSOR.DSPARK_MARKOV_W2,
        "dspark.confidence_head.proj.weight": gguf.MODEL_TENSOR.DSPARK_CONF_PROJ,
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".ffn.gate.bias_vl"):
            return []
        if name in self._DSPARK_ROOT:
            return [(self.format_tensor_name(self._DSPARK_ROOT[name], None), data_torch)]
        m = re.match(r"layers\.\d+\.(main_proj\.weight|main_norm\.weight|norm\.weight)$", name)
        if m is not None:
            return [(self.format_tensor_name(self._DSPARK_ROOT[m.group(1)], None), data_torch)]
        return super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name == "conf_proj.weight":
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("DeepseekV41ForCausalLM")
@ModelBase.example("deepseek-ai/DeepSeek-V4.1-Flash-1017")
class DeepseekV41VisionModel(DeepseekV4FlashVisionModel):
    """DeepSeek-V4.1 vision tower (projector deepseek41v): the V4-Flash-Vision ViT + 3x3 aligner, with the V4.1 token
    block ([START] ([IMAGE]*w [NEWLINE])*h [END], no pads) and resize plan. ref: inference/vision.py,
    inference/image_processor.py of deepseek-ai/DeepSeek-V4.1-Flash-1017. The checkpoint config is HF-style
    (vision_config); image_start / image_end / image_newline are written like V4's sentinels, there is no image_pad."""

    def get_vision_config(self) -> dict[str, Any] | None:
        vc = self.global_config.get("vision_config")
        if not vc or vc.get("num_hidden_layers", 0) == 0:
            raise ValueError("DeepseekV41VisionModel requires vision_config.num_hidden_layers > 0")
        return {
            "num_hidden_layers":   vc["num_hidden_layers"],
            "hidden_size":         vc["hidden_size"],
            "num_attention_heads": vc["num_attention_heads"],
            "intermediate_size":   vc["intermediate_size"],
            "patch_size":          vc["patch_size"],
            # dynamic resolution; only used for compat / warmup
            "image_size":          vc["patch_size"] * vc["downsample_ratio"] * 16,
            "rope_theta":          vc.get("rope_theta", 10000.0),
            "downsample_ratio":    vc["downsample_ratio"],
            "min_pixels":          vc["min_pixels"],
            "max_image_tokens":    vc["max_image_tokens"],
            "max_wh_ratio":        vc.get("max_wh_ratio"),
        }

    def set_gguf_parameters(self):
        assert self.hparams_vision is not None
        super(DeepseekV4FlashVisionModel, self).set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DEEPSEEK41V)
        # pytorch default eps (vision.py), not the LLM rms_norm_eps (1e-20)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_projector_scale_factor(self.hparams_vision["downsample_ratio"])
        self.gguf_writer.add_vision_min_pixels(self.hparams_vision["min_pixels"])
        # hardcoded in clip.cpp (PROJECTOR_TYPE_DEEPSEEK41V); add GGUF keys if a later model differs
        assert self.hparams_vision["max_image_tokens"] == 1024, self.hparams_vision["max_image_tokens"]
        assert self.hparams_vision["max_wh_ratio"] is None, self.hparams_vision["max_wh_ratio"]
        assert self.hparams_vision["rope_theta"] == 10000, self.hparams_vision["rope_theta"]
