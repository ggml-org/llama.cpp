from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import gguf
import numpy as np
import torch
from torch import Tensor

from .base import LazyTorchTensor, ModelBase, logger
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("unsloth/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mtp_only and self.ftype not in (
                gguf.LlamaFileType.ALL_F32,
                gguf.LlamaFileType.MOSTLY_F16,
                gguf.LlamaFileType.MOSTLY_BF16,
        ):
            raise ValueError(
                "Qwen4Exp MTP must remain F16/BF16: whole-head Q8_0 quantization "
                "changes expert routing and collapses real draft acceptance"
            )
        self._ple_row_dim: int | None = None
        self._ple_rows: int | None = None
        self._ple_map: np.memmap | None = None
        self._ple_path: Path | None = None
        self._ple_multipliers: list[int] | None = None
        self._ple_offsets: list[int] | None = None
        self._ple_vocab_sizes: list[int] | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None):
        if not self.mtp_only or remote_hf_model_id is not None:
            return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

        # The released checkpoint keeps its MTP head in model_mtp.safetensors.
        # MTP export needs only that file plus the shared embedding and LM head;
        # do not open the enormous target/PLE shards merely to filter them out.
        index_path = self.dir_model / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as file:
            weight_map = json.load(file).get("weight_map", {})
        selected = {
            name: part for name, part in weight_map.items()
            if name.startswith("mtp.") or name in (
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
            )
        }
        required = {"model.language_model.embed_tokens.weight", "lm_head.weight"}
        if not required.issubset(selected) or not any(name.startswith("mtp.") for name in selected):
            raise ValueError("Qwen4Exp MTP export is missing its shared or auxiliary tensors")

        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        type(self)._original_block_count = int(hparams["num_hidden_layers"])
        type(self).opt_num_mtp_layers = 0
        type(self).saw_mtp_tensor = False
        tensors = {}
        for part_name in sorted(set(selected.values())):
            with gguf.utility.SafetensorsLocal(self.dir_model / part_name) as model_part:
                for name, source_part in selected.items():
                    if source_part != part_name:
                        continue
                    data = model_part[name]
                    data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)
                    if titem := self.filter_tensors((name, data_gen)):
                        tensor_name, tensor_gen = titem
                        tensors[tensor_name] = tensor_gen
        return tensors

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if len(t.shape) != 1:
                    raise ValueError(f"PLE constant {suffix!r} must be one-dimensional")
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                values = [int(x) for x in t.tolist()]
                if any(x < 0 or x > (1 << 64) - 1 for x in values):
                    raise ValueError(f"PLE constant {suffix!r} is outside UINT64 range")
                return values
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def _ple_constants(self) -> tuple[list[int], list[int], list[int]]:
        if self._ple_multipliers is None:
            self._ple_multipliers = self._read_hash_constants("ple_embedding.layer_multipliers")
            self._ple_offsets = self._read_hash_constants("ple_embedding.ngram_heads_offsets")
            self._ple_vocab_sizes = self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes")
        assert self._ple_offsets is not None and self._ple_vocab_sizes is not None
        return self._ple_multipliers, self._ple_offsets, self._ple_vocab_sizes

    def _validate_ple(self, row_dim: int | None = None, total_rows: int | None = None) -> None:
        ngram = int(self.hparams["ngram_size"])
        heads_per_ngram = int(self.hparams["heads_per_ngram"])
        conv_kernel = int(self.hparams["ple_conv_kernel_size"])
        if ngram < 2 or ngram > 8:
            raise ValueError("ngram_size must be in [2, 8]")
        if heads_per_ngram <= 0:
            raise ValueError("heads_per_ngram must be positive")
        if conv_kernel <= 0:
            raise ValueError("ple_conv_kernel_size must be positive")

        n_heads = (ngram - 1) * heads_per_ngram
        if n_heads > 64:
            raise ValueError("PLE head count exceeds 64")
        multipliers, offsets, vocab_sizes = self._ple_constants()
        if len(multipliers) != ngram:
            raise ValueError("PLE multiplier count does not match ngram_size")
        if len(offsets) != n_heads or len(vocab_sizes) != n_heads:
            raise ValueError("PLE head constants do not match the configured head count")
        if any(x <= 0 for x in multipliers):
            raise ValueError("PLE multipliers must be positive")
        if any(x <= 0 for x in vocab_sizes):
            raise ValueError("PLE head vocabulary sizes must be positive")

        ranges: list[tuple[int, int]] = []
        for offset, size in zip(offsets, vocab_sizes, strict=True):
            end = offset + size
            if end > (1 << 64) - 1:
                raise ValueError("PLE head range exceeds UINT64")
            ranges.append((offset, end))
        ranges.sort()
        if any(left[1] > right[0] for left, right in pairwise(ranges)):
            raise ValueError("PLE head ranges overlap")

        if row_dim is not None:
            hidden_size = int(self.hparams["hidden_size"])
            if row_dim <= 0 or row_dim * n_heads != hidden_size:
                raise ValueError("PLE row width and head count do not flatten to hidden_size")
        if total_rows is not None:
            required_rows = max((end for _, end in ranges), default=0)
            if total_rows <= 0 or total_rows > (1 << 31):
                raise ValueError("PLE table row count exceeds the signed index range")
            if required_rows > total_rows:
                raise ValueError("PLE head range exceeds the streamed table")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        hc_count = int(hp["hc_count"])
        hc_lowrank = int(hp["hc_lowrank"])
        hidden_size = int(hp["hidden_size"])
        if hc_count <= 0 or hc_lowrank <= 0 or hc_count * hidden_size > (1 << 32) - 1:
            raise ValueError("invalid hyper-connection dimensions")
        self.gguf_writer.add_hyper_connection_count(hc_count)
        self.gguf_writer.add_hyper_connection_low_rank(hc_lowrank)

        n_layer = int(hp["num_hidden_layers"])
        indexer_heads = int(hp["indexer_n_heads"])
        indexer_dim = int(hp["indexer_head_dim"])
        indexer_budget = int(hp["indexer_budget"])
        if n_layer <= 0 or indexer_heads <= 0 or indexer_dim <= 0 or indexer_budget <= 0:
            raise ValueError("invalid QSA dimensions")
        self.gguf_writer.add_indexer_head_count(indexer_heads)
        self.gguf_writer.add_indexer_key_length(indexer_dim)
        self.gguf_writer.add_indexer_top_k(indexer_budget)
        ratio = int(hp["indexer_compress_ratio"])
        layer_types = hp["layer_types"]
        if len(layer_types) != n_layer:
            raise ValueError("layer_types length does not match num_hidden_layers")
        if ratio <= 0 or not any(kind == "full_attention" for kind in layer_types):
            raise ValueError("QSA needs a positive ratio and at least one full-attention layer")
        if any(kind not in ("full_attention", "linear_attention") for kind in layer_types):
            raise ValueError("layer_types contains an unsupported attention kind")
        compress_ratios = [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        # The released MTP block has its own QSA indexer/cache and uses the
        # same compression ratio as the target full-attention layers.
        compress_ratios.extend([ratio] * (self.block_count - n_layer))
        self.gguf_writer.add_attention_compress_ratios(compress_ratios)

        # An MTP-only GGUF carries no PLE table or hash metadata.
        if self.mtp_only:
            return

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [int(i) - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        if len(set(ple_layers)) != len(ple_layers):
            raise ValueError("ple_layer_ids contains a duplicate")
        if any(i < 0 or i >= n_layer for i in ple_layers):
            raise ValueError("ple_layer_ids contains an out-of-range layer")
        if any(layer_types[i] == "full_attention" for i in ple_layers):
            raise ValueError("PLE must be attached to a recurrent layer")
        self._validate_ple(self._ple_row_dim, self._ple_rows)
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(int(hp["ngram_size"]))
        self.gguf_writer.add_ple_heads_per_ngram(int(hp["heads_per_ngram"]))
        self.gguf_writer.add_ple_conv_kernel(int(hp["ple_conv_kernel_size"]))
        vocab_size = int(hp["vocab_size"])
        eos_token = self._eos_token_id()
        if eos_token >= vocab_size:
            raise ValueError("eos_token_id is outside vocab_size")
        self.gguf_writer.add_ple_eos_token_id(eos_token)
        # The PLE hash runs over token ids, but a multimodal batch arrives as embeddings
        # with the placeholder consumed. Carry it so those positions hash what the
        # reference sees in input_ids instead of being undefined.
        _img = self._image_token_id()
        if _img is not None:
            if _img >= vocab_size:
                raise ValueError("image_token_id is outside vocab_size")
            self.gguf_writer.add_ple_image_token_id(_img)
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        multipliers, offsets, vocab_sizes = self._ple_constants()
        self.gguf_writer.add_ple_layer_multipliers(multipliers)
        self.gguf_writer.add_ple_head_offsets(offsets)
        self.gguf_writer.add_ple_head_vocab_sizes(vocab_sizes)

    def _image_token_id(self) -> int | None:
        # image_token_id is top-level in config.json, not in self.hparams once that is
        # narrowed to text_config, and the text model has no global_config; read the file
        img = self.hparams.get("image_token_id")
        if img is not None:
            return self._uint32_token(img, "image_token_id")
        try:
            with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
                img = json.load(f).get("image_token_id")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return None if img is None else self._uint32_token(img, "image_token_id")

    @staticmethod
    def _uint32_token(value: object, name: str) -> int:
        if not isinstance(value, (int, str)):
            raise TypeError(f"{name} must be an integer")
        token = int(value)
        if token < 0 or token > (1 << 32) - 1:
            raise ValueError(f"{name} is outside UINT32 range")
        return token

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            if not eos:
                raise ValueError("eos_token_id must not be an empty list")
            # the PLE hash resets n-grams on the primary EOS
            return self._uint32_token(eos[-1], "eos_token_id")
        if eos is None:
            raise ValueError("eos_token_id is required for the PLE hash")
        return self._uint32_token(eos, "eos_token_id")

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # The MTP block has its own final hyper-connection mixer. In an
        # MTP-only file it occupies the model-level head tensor names.
        if name.startswith("mtp.hyper_connection_mixer."):
            if cls.no_mtp:
                return None
            return name.replace("mtp.", "model.", 1), gen
        return super().filter_tensors(item)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        # The reference computes fc_embedding(e) + fc_hidden(h). Joining the
        # two weights makes this one projection of concat(e, h), matching the
        # existing generic nextn tensor layout.
        e_name = "mtp.fc_embedding.weight"
        h_name = "mtp.fc_hidden.weight"
        have_e = e_name in self.model_tensors
        have_h = h_name in self.model_tensors
        if not have_e and not have_h:
            return
        if not have_e or not have_h:
            raise KeyError(f"unpaired MTP input projection: need both {e_name} and {h_name}")

        e = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
        h = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
        if e.ndim != 2 or h.ndim != 2 or e.shape[0] != h.shape[0]:
            raise ValueError(f"incompatible MTP input projections: {tuple(e.shape)} and {tuple(h.shape)}")
        yield (
            self.format_tensor_name(
                gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
                int(self.hparams["num_hidden_layers"]),
            ),
            torch.cat([e, h], dim=1).contiguous(),
        )
        del self.model_tensors[e_name]
        del self.model_tensors[h_name]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith((
                "ple_embedding.layer_multipliers",
                "ple_embedding.ngram_heads_offsets",
                "ple_embedding.ngram_heads_vocab_sizes",
        )):
            return []

        if ".ngram_embedding.shard_" in name:
            raise RuntimeError("PLE shards must be streamed before the regular tensor pass")

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = int(self.hparams["indexer_n_heads"]) * int(self.hparams["indexer_head_dim"])
            n_k = int(self.hparams["indexer_head_dim"])
            if data_torch.ndim != 2 or data_torch.shape[0] != n_q + n_k:
                raise ValueError(
                    f"index_qk_proj has shape {tuple(data_torch.shape)}, expected first dimension {n_q + n_k}"
                )
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            if data_torch.ndim != 3 or data_torch.shape[1] != 1:
                raise ValueError(f"PLE conv1d weight has unexpected shape {tuple(data_torch.shape)}")
            return [(self.map_tensor_name(name), data_torch.squeeze(1))]

        return super().modify_tensors(data_torch, name, bid)

    # -- the PLE table ----------------------------------------------------
    #
    # The PLE table is roughly 98 GiB before quantization and is split into many
    # checkpoint tensors. Concatenating it, or letting the normal converter
    # quantize the concatenation, allocates another table-sized array. Stream
    # one shard and a small row group at a time into an output-typed memmap,
    # then hand that memmap directly to the GGUF writer. Peak resident memory is
    # bounded by one source shard plus a small quantization buffer.

    def _ple_output_qtype(self, source_name: str, gguf_name: str, row_dim: int) -> gguf.GGMLQuantizationType:
        qtype = self.tensor_force_quant(source_name, gguf_name, None, 2)
        if isinstance(qtype, bool):
            if self.ftype == gguf.LlamaFileType.ALL_F32:
                qtype = gguf.GGMLQuantizationType.F32
            elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                qtype = gguf.GGMLQuantizationType.BF16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                qtype = gguf.GGMLQuantizationType.Q8_0
            elif self.ftype in (gguf.LlamaFileType.MOSTLY_TQ1_0, gguf.LlamaFileType.MOSTLY_TQ2_0):
                # Match base.py: token-embedding tables stay F16 for ternary output.
                qtype = gguf.GGMLQuantizationType.F16
            else:
                raise ValueError(f"unsupported PLE output file type: {self.ftype.name}")

        try:
            gguf.quants.quantize(np.zeros((1, row_dim), dtype=np.float32), qtype)
        except gguf.QuantError as exc:
            logger.warning("%s; storing the PLE table as F16", exc)
            qtype = gguf.GGMLQuantizationType.F16
        return qtype

    def _stream_ple_table(self) -> None:
        if self.mtp_only or not self.hparams.get("ple_layer_ids"):
            return

        marker = ".ngram_embedding.shard_"
        shards: dict[int, tuple[str, Tensor]] = {}
        for name, gen in self.model_tensors.items():
            if marker not in name:
                continue
            suffix = name.rpartition(".shard_")[2]
            try:
                idx = int(suffix.partition(".")[0])
            except ValueError as exc:
                raise ValueError(f"invalid PLE shard name: {name}") from exc
            if idx in shards:
                raise ValueError(f"duplicate PLE shard index {idx}")
            shards[idx] = (name, gen())

        n_parts = int(self.hparams["split_ngram_parts"])
        if n_parts <= 0 or set(shards) != set(range(n_parts)):
            missing = sorted(set(range(max(n_parts, 0))) - set(shards))
            extra = sorted(set(shards) - set(range(max(n_parts, 0))))
            raise ValueError(f"invalid PLE shard set; missing={missing}, extra={extra}")

        first = shards[0][1]
        if len(first.shape) != 2:
            raise ValueError(f"PLE shard 0 must be two-dimensional, got {tuple(first.shape)}")
        first_rows = int(first.shape[0])
        row_dim = int(first.shape[1])
        if first_rows <= 0 or row_dim <= 0:
            raise ValueError("PLE shards must have positive dimensions")

        total_rows = 0
        for idx, (_, shard) in shards.items():
            rows = int(shard.shape[0])
            if len(shard.shape) != 2 or int(shard.shape[1]) != row_dim:
                raise ValueError(f"PLE shard {idx} has inconsistent shape {tuple(shard.shape)}")
            if idx != n_parts - 1 and rows != first_rows:
                raise ValueError(
                    f"PLE shard {idx} has {rows} rows, expected {first_rows}; "
                    "only the last shard may be short"
                )
            if rows <= 0 or (idx == n_parts - 1 and rows > first_rows):
                raise ValueError(f"PLE shard {idx} has invalid row count {rows}")
            total_rows += rows

        self._validate_ple(row_dim, total_rows)
        self._ple_row_dim = row_dim
        self._ple_rows = total_rows
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight"
        qtype = self._ple_output_qtype(shards[0][0], gguf_name, row_dim)
        probe = gguf.quants.quantize(np.zeros((1, row_dim), dtype=np.float32), qtype)

        self._ple_path = self.fname_out.parent / f".{self.fname_out.stem}.ple.{qtype.name.lower()}.tmp"
        ple_map = np.memmap(
            self._ple_path, dtype=probe.dtype, mode="w+",
            shape=(total_rows, *probe.shape[1:]))
        self._ple_map = ple_map

        row_offset = 0
        rows_per_group = 16
        for idx in range(n_parts):
            name, shard = shards[idx]
            eager = LazyTorchTensor.to_eager(shard)
            rows = int(eager.shape[0])
            for start in range(0, rows, rows_per_group):
                stop = min(start + rows_per_group, rows)
                source = eager[start:stop].to(torch.float32).contiguous().numpy()
                ple_map[row_offset + start:row_offset + stop] = gguf.quants.quantize(source, qtype)
            row_offset += rows
            ple_map.flush()
            del eager
            logger.info("Streamed PLE shard %d/%d (%s)", idx + 1, n_parts, name)

        self.gguf_writer.add_tensor(gguf_name, ple_map, raw_dtype=qtype)
        for name, _ in shards.values():
            del self.model_tensors[name]

    def prepare_tensors(self):
        self._stream_ple_table()
        super().prepare_tensors()

    def write(self):
        try:
            super().write()
        finally:
            ple_map = self._ple_map
            self._ple_map = None
            if ple_map is not None:
                ple_map.flush()
                mmap_handle = getattr(ple_map, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
            path = self._ple_path
            if path is not None and path.exists():
                path.unlink()


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("unsloth/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
