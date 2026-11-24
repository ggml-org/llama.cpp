from __future__ import annotations

from math import prod
from typing import Any, Callable, Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import LazyTorchTensor, ModelBase, logger
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


_NVFP4_PLE_CHUNK_ROWS = 8192


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # the MTP head: one full-attention QSA block after the trunk, fed by the trunk's hc-wide residual
    supports_mtp_export = True

    # MTP tensors the shared Qwen remapper does not know
    _MTP_EXTRA = {
        "fc_embedding":           "nextn_fc_embedding",
        "fc_hidden":              "nextn_fc_hidden",
        "hyper_connection_mixer": "nextn_hc_head",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None
        self._nvfp4_ple_shards: dict[str, tuple[Callable[[], Tensor], Callable[[], Tensor], Callable[[], Tensor]]] = {}
        self._mtp_fc: dict[str, Tensor] = {}
        self._ple_scale: float | None = None

    def dequant_model(self):
        # FP8 checkpoints: all n-gram shards share one scale, stored under the parent module name
        # there is no "ngram_embedding.weight", so the generic FP8 pass drops it or cannot map it
        for name in list(self.model_tensors):
            if name.endswith(".ngram_embedding.weight_scale"):
                scale = LazyTorchTensor.to_eager(self.model_tensors.pop(name)()).float().flatten()
                if scale.numel() != 1:
                    raise ValueError(f"PLE scale {name!r} must be a scalar, got {scale.numel()} values")
                self._ple_scale = float(scale[0])
        super().dequant_model()

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        part = name.split(".")[1] if name.startswith("mtp.") else None
        if part in cls._MTP_EXTRA:
            if cls.no_mtp:
                return None
            assert cls._original_block_count is not None
            rest = name.split(".", 2)[2]
            return f"model.layers.{cls._original_block_count}.{cls._MTP_EXTRA[part]}.{rest}", gen
        return super().filter_tensors(item)

    def _validate_quantized_layers(self, quant_layers: dict[str, Any], matched: set[str]):
        redundant = set()
        n_layer = self.hparams["num_hidden_layers"]
        n_mtp = self.hparams.get("mtp_num_hidden_layers", 0)
        for name in quant_layers.keys() - matched:
            parts = name.split(".", 3)
            if len(parts) != 4 or parts[:2] != ["mtp", "layers"] or not parts[2].isdecimal():
                continue
            mtp_idx = int(parts[2]) - n_layer
            if not 0 <= mtp_idx < n_mtp:
                continue
            if any(source.startswith(name + ".") for source in self._source_tensor_names):
                continue
            canonical = f"mtp.layers.{mtp_idx}.{parts[3]}"
            # Some ModelOpt exports repeat MTP metadata with trunk-offset indices.
            if canonical in matched and quant_layers[name] == quant_layers[canonical]:
                redundant.add(name)
                logger.warning("Ignoring redundant MTP quantization metadata %r (same settings as %r)", name, canonical)
        super()._validate_quantized_layers({name: config for name, config in quant_layers.items() if name not in redundant}, matched)

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        # the MTP block is a full-attention QSA layer too
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
            + [ratio] * (self.block_count - n_layer)
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        # the MTP head never reads PLE, so an MTP-only file carries none of it
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers or self.mtp_only:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return

        if ".ngram_embedding.shard_" in name:
            yield from self._place_ple_shard(data_torch, name)
            return

        # eh_proj([e ; h_s]) = fc_embedding(e) + fc_hidden(h_s) for every hc stream s
        if name.endswith((".nextn_fc_embedding.weight", ".nextn_fc_hidden.weight")):
            self._mtp_fc[name.rsplit(".", 2)[1]] = data_torch
            if len(self._mtp_fc) < 2:
                return
            eh = torch.cat([self._mtp_fc.pop("nextn_fc_embedding"), self._mtp_fc.pop("nextn_fc_hidden")], dim=1)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, ".weight"), eh)
            return

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), data_torch[:n_q])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), data_torch[n_q:])
            return

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            yield (self.map_tensor_name(name), data_torch + 1)
            return

        if name.endswith(".ple.conv1d.weight"):
            yield (self.map_tensor_name(name), data_torch.squeeze())
            return

        yield from super().modify_tensors(data_torch, name, bid)

    # matmuls that read the same input, written as one weight [a | b] so the runtime does one matmul:
    # (merged tensor, (a, b), linear-attention layers only)
    _SHARED_INPUT_MERGES = (
        (gguf.MODEL_TENSOR.HC_ATTN_DOWN_INJECT, (gguf.MODEL_TENSOR.HC_ATTN_DOWN, gguf.MODEL_TENSOR.HC_ATTN_INJECT), False),
        (gguf.MODEL_TENSOR.HC_FFN_DOWN_INJECT,  (gguf.MODEL_TENSOR.HC_FFN_DOWN,  gguf.MODEL_TENSOR.HC_FFN_INJECT),  False),
        (gguf.MODEL_TENSOR.ATTN_QKV_GATE,       (gguf.MODEL_TENSOR.ATTN_QKV,     gguf.MODEL_TENSOR.ATTN_GATE),      True),
        (gguf.MODEL_TENSOR.SSM_BETA_ALPHA_CAT,  (gguf.MODEL_TENSOR.SSM_BETA,     gguf.MODEL_TENSOR.SSM_ALPHA),      True),
    )

    def _generate_nvfp4_tensors(self):
        consumed = set()
        for name, weight_gen in self.model_tensors.items():
            if ".ngram_embedding.shard_" not in name or not name.endswith(".weight"):
                continue
            weight = weight_gen()
            if weight.dtype != torch.uint8:
                continue
            scale_name = name + "_scale"
            scale2_name = name.rpartition(".shard_")[0] + ".weight_scale_2"
            if scale_name not in self.model_tensors or scale2_name not in self.model_tensors:
                raise ValueError(f"NVFP4 PLE shard {name!r} requires block scales and a shared weight_scale_2")
            scale_gen = self.model_tensors[scale_name]
            scale2_gen = self.model_tensors[scale2_name]
            scale, scale2 = scale_gen(), scale2_gen()
            if weight.ndim != 2 or min(weight.shape) <= 0 or weight.shape[1] % 8 or scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != (weight.shape[0], weight.shape[1] // 8):
                raise ValueError(f"Invalid NVFP4 PLE weight/block scale layout for {name!r}")
            if prod(scale2.shape) != 1 or scale2.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"NVFP4 PLE shared scale {scale2_name!r} must be a floating-point scalar")
            self._nvfp4_ple_shards[name] = (weight_gen, scale_gen, scale2_gen)
            consumed.update((scale_name, scale2_name))
        for name in consumed:
            del self.model_tensors[name]
        super()._generate_nvfp4_tensors()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if self._nvfp4_ple_shards and new_name == gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight":
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _load_nvfp4_ple_shard(self, name: str, rows: slice):
        weight_gen, scale_gen, scale2_gen = self._nvfp4_ple_shards[name]

        def load() -> np.ndarray:
            packed = LazyTorchTensor.to_eager(weight_gen()[rows]).numpy()
            scales = LazyTorchTensor.to_eager(scale_gen()[rows]).float().numpy()
            scale2 = float(LazyTorchTensor.to_eager(scale2_gen()).float().item())
            if not np.isfinite(scale2) or scale2 < 0 or not np.isfinite(scales).all() or np.any(scales < 0):
                raise ValueError(f"NVFP4 PLE shard {name!r} has invalid scales")
            levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
            values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
            values[:, 0::2] = levels[packed & 15]
            values[:, 1::2] = levels[packed >> 4]
            values.reshape(values.shape[0], -1, 16)[:] *= scales[:, :, None]
            values *= scale2
            if not np.isfinite(values).all():
                raise ValueError(f"NVFP4 PLE shard {name!r} has non-finite decoded values")
            return values

        return load

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]
        if not 0 <= idx < n_parts:
            raise ValueError(f"PLE shard index {idx} is outside the expected range 0..{n_parts - 1}")

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1]) * (2 if name in self._nvfp4_ple_shards else 1)

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            row_dim = int(shape[-1]) * (2 if shard in self._nvfp4_ple_shards else 1)
            if row_dim != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {row_dim}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        # an FP8 table keeps its bytes and gets a scale tensor, any other type has the scale folded in
        keep_fp8 = name in self._fp8_e4m3_preserved
        chunks = []
        for shard in shards:
            if shard in self._nvfp4_ple_shards:
                # Match the bounded row decoding used by packed expert conversion.
                chunks.extend(self._load_nvfp4_ple_shard(shard, slice(start, start + _NVFP4_PLE_CHUNK_ROWS)) for start in range(0, int(self.model_tensors[shard]().shape[0]), _NVFP4_PLE_CHUNK_ROWS))
            else:
                chunks.append(self._load_ple_shard(shard, keep_fp8))
        table = gguf.LazyChunkedTensor(
            chunks,
            shape=(rows, self._ple_row_dim),
            dtype=np.uint8 if keep_fp8 else np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        out = [(gguf_name + ".weight", cast(Tensor, table))]
        if keep_fp8 and self._ple_scale is not None:
            out.append((gguf_name + ".scale", torch.tensor([self._ple_scale], dtype=torch.float32)))
        return out

    def _load_ple_shard(self, name: str, keep_fp8: bool):
        def load() -> np.ndarray:
            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            if keep_fp8:
                if eager.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"PLE shard {name} is {eager.dtype}, expected FP8 like the other shards")
                return eager.contiguous().view(torch.uint8).numpy()
            data = eager.to(torch.float32)
            if self._ple_scale is not None:
                data = data * self._ple_scale
            return data.contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._mtp_fc:
            raise ValueError(f"MTP projection missing its other half: {sorted(self._mtp_fc)}")
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
