from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, cast

import numpy as np
import torch

from .base import LazyTorchTensor, MmprojModel, ModelBase, gguf, logger
from .deepseek41 import _StreamedBytes
from .hy_v4 import HYV4Model

# the reference derives the hash multipliers from the Engram seed, the 1-based layer id and the compressed vocab size
ENGRAM_SEED_STRIDE = 10007


@ModelBase.register("HYV4FlashForCausalLM", "HYV4FlashForConditionalGeneration")
@ModelBase.example("tencent/Hy4-Flash")
class HYV4FlashModel(HYV4Model):
    """hy_v4 with sparse and sliding-window MLA layers, sandwich norms, latent routed experts, Engram and a k-pool indexer.

    The MTP draft block is exported as one extra sparse layer. The vision tower and the dfly draft layers are not converted.
    """

    model_arch = gguf.MODEL_ARCH.HY_V4_FLASH

    # the draft block is exported under block ids past the trunk, which skip_mtp would drop
    skip_mtp = False

    # dropped subsystems, see the class docstring
    DROP_PREFIXES = ("vit.", "model.dfly_layers.")
    DROP_SUBSTRINGS = ()

    # the released config renamed sparse_attention, both name the same layer type
    SPARSE_TYPES = ("sparse_attention", "deepseek_sparse_attention")

    # the released checkpoints ship the token map as a tensor instead of engram_vocab_map.npz
    ENGRAM_LOOKUP_TABLE = "lookup_table"

    # compressed_indexer tail -> the tensor the graph reads it as
    INDEXER_MAP = {
        "token_key_projection.weight":   gguf.MODEL_TENSOR.INDEXER_ATTN_K,
        "token_key_norm.weight":         gguf.MODEL_TENSOR.INDEXER_K_NORM,
        "key_value_projection.weight":   gguf.MODEL_TENSOR.INDEXER_KPOOL_VALUE,
        "key_gate_projection.weight":    gguf.MODEL_TENSOR.INDEXER_KPOOL_GATE,
        "intra_group_ape":               gguf.MODEL_TENSOR.INDEXER_KPOOL_APE,
        "key_norm.weight":               gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_NORM,
        "query_projection.weight":       gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B,
        "head_weight_projection.weight": gguf.MODEL_TENSOR.INDEXER_PROJ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the draft block is an extra block, the loader subtracts the nextn count from block_count
        if not getattr(self, "no_mtp", False):
            self.block_count += self._n_nextn()
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        # Engram weights arrive one at a time but are written fused, so they wait here
        self._engram_buf: dict[int, dict[str, torch.Tensor]] = {}
        self._engram_tables: dict[int, torch.Tensor] = {}
        self._engram_done: set[str] = set()

    @classmethod
    def filter_tensors(cls, item):
        name = item[0]
        if name.startswith(cls.DROP_PREFIXES) or any(s in name for s in cls.DROP_SUBSTRINGS):
            return None
        if name.startswith("model.mtp_layers."):
            # hy_v4 drops the draft block; this model exports it, so skip that filter
            return super(HYV4Model, cls).filter_tensors(item)
        return super().filter_tensors(item)

    def _layer_types(self) -> list[str]:
        cached = getattr(self, "_layer_types_cache", None)
        if cached is not None:
            return cached
        layer_types = self.hparams.get("layer_types")
        n_layer = self.hparams["num_hidden_layers"]
        if not layer_types or len(layer_types) < n_layer:
            raise ValueError("HY_V4_FLASH needs a layer_types entry per layer")
        unknown = set(layer_types[:n_layer]) - set(self.SPARSE_TYPES) - {"sliding_attention"}
        if unknown:
            raise ValueError(f"HY_V4_FLASH unknown layer_types values: {sorted(unknown)}")
        # the released config renamed the sparse layers, normalize so the rest reads one name
        self._layer_types_cache = ["sparse_attention" if t in self.SPARSE_TYPES else t for t in layer_types[:n_layer]]
        return self._layer_types_cache

    # Engram: one n-gram hash module per Engram layer, all buckets in one table as in deepseek41

    def _engram_enabled(self) -> bool:
        # engram_enable is optional, a False turns Engram off
        return bool(self.hparams.get("engram_enable", True)) and bool(self.hparams.get("engram_layer_ids"))

    def _engram_ids(self) -> list[int]:
        """engram_layer_ids, 1-based and sorted."""
        ids = sorted(int(i) for i in self.hparams["engram_layer_ids"])
        n_layer = self.hparams["num_hidden_layers"]
        for i in ids:
            # 1-based: the tensors live at model.layers.{i-1}.engram_layer.*
            if not 1 <= i <= n_layer:
                raise ValueError(f"HY_V4_FLASH engram_layer_ids has {i}, outside 1..{n_layer}")
        return ids

    def _engram_primes(self) -> dict[int, list[int]]:
        """Per 1-based layer id, the flattened [ngram][head] bucket counts."""
        raw = self.hparams.get("engram_layer_head_vocab_sizes") or {}
        n_gram = self.hparams["engram_max_ngram_size"]
        n_head = self.hparams["engram_n_head_per_ngram"]
        out = {}
        for i in self._engram_ids():
            # the keys are strings in the config
            groups = raw.get(str(i)) or raw.get(i)
            if groups is None:
                raise ValueError(f"HY_V4_FLASH engram_layer_head_vocab_sizes has no entry for layer {i}")
            if len(groups) != n_gram - 1 or any(len(g) != n_head for g in groups):
                raise ValueError(
                    f"HY_V4_FLASH engram layer {i}: expected {n_gram - 1} groups of {n_head} bucket "
                    f"counts, got {[len(g) for g in groups]}")
            out[i] = [int(p) for g in groups for p in g]
        return out

    def _engram_lookup_names(self) -> list[str]:
        """The lookup_table tensor of every Engram layer, empty when the checkpoint carries the npz instead."""
        names = [f"model.layers.{i - 1}.engram_layer.{self.ENGRAM_LOOKUP_TABLE}" for i in self._engram_ids()]
        present = [n for n in names if n in self.model_tensors]
        if present and len(present) != len(names):
            raise ValueError(f"HY_V4_FLASH engram is missing the lookup table of {sorted(set(names) - set(present))}")
        return present

    def _engram_vocab_map(self) -> tuple[list[int], int]:
        """The token -> compressed id map, extended by identity as the reference does, and the compressed vocab size."""
        names = self._engram_lookup_names()
        if names:
            table = LazyTorchTensor.to_eager(self.model_tensors[names[0]]()).numpy().astype(np.int64)
            for other in names[1:]:
                if not np.array_equal(LazyTorchTensor.to_eager(self.model_tensors[other]()).numpy().astype(np.int64), table):
                    raise ValueError(f"HY_V4_FLASH engram lookup table of {other} differs from {names[0]}")
            # the map is onto a dense id range from 0, so its largest id gives the compressed vocab size
            compressed = int(table.max()) + 1
            if int(table.min()) != 0 or len(np.unique(table)) != compressed:
                raise ValueError("HY_V4_FLASH engram lookup table is not a dense id range from 0, supply engram_vocab_map.npz")
        else:
            path = Path(self.dir_model) / "engram_vocab_map.npz"
            if not path.is_file():
                raise FileNotFoundError(f"HY_V4_FLASH engram needs a lookup_table tensor or {path}")
            with np.load(path) as z:
                table = z["lookup_table"].astype(np.int64)
                meta = json.loads(bytes(z["metadata"]).decode("utf-8").rstrip("\x00"))

            compressed = int(meta["compressed_vocab_size"])
            if int(table.max()) + 1 > compressed:
                raise ValueError("HY_V4_FLASH engram lookup table exceeds its compressed vocab size")
            if int(meta["engram_seed"]) != int(self.hparams.get("engram_seed", 0)):
                raise ValueError("HY_V4_FLASH engram seed disagrees between the config and the vocab map")

        n_vocab = self.hparams["vocab_size"]
        token_map = list(map(int, table))
        token_map += list(range(len(token_map), n_vocab))
        return token_map, compressed

    def _engram_multipliers(self, layer_id: int, compressed: int) -> list[int]:
        seed = int(self.hparams.get("engram_seed", 0))
        n_gram = self.hparams["engram_max_ngram_size"]
        bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
        rng = np.random.default_rng(seed + ENGRAM_SEED_STRIDE * layer_id)
        return [int(m) for m in rng.integers(0, bound, size=n_gram) * 2 + 1]

    def _engram_set_gguf_parameters(self):
        hparams = self.hparams
        ids = self._engram_ids()
        primes = self._engram_primes()
        token_map, compressed = self._engram_vocab_map()

        n_gram = hparams["engram_max_ngram_size"]
        n_head = hparams["engram_n_head_per_ngram"]
        head_dim, rem = divmod(hparams["engram_n_embed_per_ngram"], n_head)
        if rem:
            raise ValueError("HY_V4_FLASH engram_n_embed_per_ngram is not a multiple of the head count")

        # rebase each module's buckets into the shared table, row ids are int32
        offsets: list[int] = []
        multipliers: list[int] = []
        base = 0
        for i in ids:
            for p in primes[i]:
                offsets.append(base)
                base += p
            multipliers += self._engram_multipliers(i, compressed)
        if base > np.iinfo(np.int32).max:
            raise ValueError(f"HY_V4_FLASH engram table needs {base} rows, past the int32 row id limit")

        w = self.gguf_writer
        w.add_ple_layers([i - 1 for i in ids])
        w.add_ple_ngram_size(n_gram)
        w.add_ple_heads_per_ngram(n_head)
        w.add_embedding_length_per_layer_input(head_dim)
        w.add_ple_layer_multipliers(multipliers)
        w.add_ple_head_offsets(offsets)
        w.add_ple_head_vocab_sizes([p for i in ids for p in primes[i]])
        w.add_engram_pad_token_id(hparams["engram_pad_id"])
        w.add_engram_token_map(token_map)

        logger.info(
            "HY_V4_FLASH engram: %d module(s) at 0-based layers %s, %d-gram x %d heads x %d dims, "
            "%d table rows, compressed vocab %d",
            len(ids), [i - 1 for i in ids], n_gram, n_head, head_dim, base, compressed)

    def _n_nextn(self) -> int:
        return int(self.hparams.get("num_nextn_predict_layers", 0) or 0)

    def _layer_type_of(self, bid: int) -> str:
        """Layer type of any block, draft blocks included. Only sparse draft blocks are supported."""
        n_layer = self.hparams["num_hidden_layers"]
        if bid < n_layer:
            return self._layer_types()[bid]

        types = self.hparams.get("mtp_layer_types") or []
        if not types:
            return "sparse_attention"

        t = types[min(bid - n_layer, len(types) - 1)]
        if t != "sparse_attention":
            raise ValueError(
                f"HY_V4_FLASH: mtp_layer_types gives {t!r} for draft block {bid}, "
                "but only 'sparse_attention' draft blocks are supported")
        return t

    def _write_rope_sections(self, sparse_rope: dict, window_rope: dict):
        """Write the M-RoPE sections of the VL checkpoints. Text-only files keep the loader default."""
        sections = sparse_rope.get("mrope_section")
        if sections is None:
            return
        if window_rope.get("mrope_section") != sections:
            raise ValueError(
                f"HY_V4_FLASH needs one mrope_section for both layer types, but "
                f"{window_rope.get('mrope_section')} != {sections}")
        if not sparse_rope.get("mrope_interleaved", False):
            raise NotImplementedError("HY_V4_FLASH expects mrope_interleaved = true")

        rope_dim = self.hparams["qk_rope_head_dim"]
        if len(sections) != 3 or 2 * sum(sections) != rope_dim:
            raise ValueError(f"HY_V4_FLASH mrope_section {sections} must sum to qk_rope_head_dim / 2")

        self.gguf_writer.add_rope_dimension_sections(list(sections) + [0])

    def _rope_for(self, layer_type: str) -> dict:
        params = self.hparams.get("rope_parameters")
        names = self.SPARSE_TYPES if layer_type == "sparse_attention" else (layer_type,)
        found = [params[n] for n in names if isinstance(params, dict) and n in params]
        if not found:
            raise ValueError(f"HY_V4_FLASH needs rope_parameters[{layer_type!r}]")
        if any(f != found[0] for f in found[1:]):
            raise ValueError(f"HY_V4_FLASH has conflicting rope_parameters for {names}")
        return found[0]

    def set_gguf_parameters(self):
        hparams = self.hparams
        layer_types = self._layer_types()
        is_swa = [t == "sliding_attention" for t in layer_types]

        if not any(is_swa) or all(is_swa):
            raise ValueError("HY_V4_FLASH expects a mix of sparse and sliding attention layers")

        # the graph always applies the sandwich post-norms
        if not hparams.get("sandwich_norm", False):
            raise ValueError("HY_V4_FLASH expects sandwich_norm, the graph always applies the post-norms")
        if hparams.get("dsa_skip_attn_post_norm", False):
            raise ValueError(
                "HY_V4_FLASH: dsa_skip_attn_post_norm is not supported - the graph applies "
                "attn_post_norm on the sparse layers too")

        # the non-sliding rope entry is keyed "sparse_attention", not "full_attention" as the base expects
        sparse_rope = self._rope_for("sparse_attention")
        window_rope = self._rope_for("sliding_attention")

        # hide the key from hy_v4, which warns that it drops the draft block
        n_nextn = self.hparams.pop("num_nextn_predict_layers", None)
        try:
            super().set_gguf_parameters()
        finally:
            if n_nextn is not None:
                self.hparams["num_nextn_predict_layers"] = n_nextn

        self.gguf_writer.add_rope_freq_base(sparse_rope["rope_theta"])

        # the graph reorders the rope rows itself, so the sections only say how to split them
        self._write_rope_sections(sparse_rope, window_rope)

        # per-layer head count: the window layers carry swa_num_attention_heads
        n_head = hparams["num_attention_heads"]
        n_head_swa = hparams["swa_num_attention_heads"]
        self.gguf_writer.remove_key(
            gguf.Keys.Attention.HEAD_COUNT.format(arch=self.gguf_writer.arch))
        # the arrays cover n_layer_all, the draft block is a sparse layer
        is_swa_all = is_swa + [False] * (0 if getattr(self, "no_mtp", False) else self._n_nextn())
        self.gguf_writer.add_head_count(
            [n_head_swa if swa else n_head for swa in is_swa_all])

        # the base already derives rope.freq_base_swa from rope_parameters["sliding_attention"]
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(is_swa_all)

        # routed experts run here, the router and the shared expert stay at hidden_size
        latent = hparams.get("routed_expert_hidden_size")
        if latent:
            self.gguf_writer.add_expert_latent_length(latent)

        if self._n_nextn() and not getattr(self, "no_mtp", False):
            self.gguf_writer.add_nextn_predict_layers(self._n_nextn())

        if self._indexer_enabled():
            # the pool size is compress_ratio, and the current group is always kept
            ratio = hparams["compressed_indexer_compress_ratio"]
            self.gguf_writer.add_indexer_kpool(ratio)
            self.gguf_writer.add_indexer_kpool_select_tail(True)

            # same check as the reference
            group_topk = hparams.get("compressed_indexer_group_topk")
            candidate = hparams.get("compressed_indexer_candidate_topk")
            if group_topk and candidate and group_topk * ratio != candidate:
                raise ValueError(
                    f"HY_V4_FLASH indexer: candidate_topk {candidate} != "
                    f"group_topk {group_topk} x compress_ratio {ratio}")
            if hparams["index_topk"] % ratio:
                raise ValueError(
                    f"HY_V4_FLASH indexer: index_topk {hparams['index_topk']} is not a multiple "
                    f"of compress_ratio {ratio}")
            is_full = self.indexer_is_full() or []
            logger.info(
                "HY_V4_FLASH indexer: %d pooled groups of %d, keeping the best %d plus the current group (index_topk %d)",
                sum(is_full), ratio, hparams["index_topk"] // ratio, hparams["index_topk"])

        if self._engram_enabled():
            self._engram_set_gguf_parameters()

        self._check_mla_geometry()

        logger.info(
            "HY_V4_FLASH: %d sparse (%d heads, rope_theta %g) + %d sliding "
            "(%d heads, rope_theta %g, window %d) layers",
            is_swa.count(False), n_head, sparse_rope["rope_theta"],
            is_swa.count(True), n_head_swa, window_rope["rope_theta"],
            hparams["sliding_window"],
        )
        if hparams.get("engram_layer_ids") and not self._engram_enabled():
            logger.warning(
                "HY_V4_FLASH: engram_enable is off, dropping %d Engram layer(s)",
                len(hparams["engram_layer_ids"]))
        if hparams.get("use_compressed_indexer") and not self._indexer_enabled():
            logger.warning(
                "HY_V4_FLASH: dropping the compressed indexer - the sparse layers will "
                "attend their whole context instead of the top %s tokens",
                hparams.get("index_topk"),
            )

    def _check_mla_geometry(self):
        """Both layer types must share one MLA geometry, only the head count may differ."""
        hparams = self.hparams
        for sparse_key, window_key in (
            ("qk_nope_head_dim", "swa_qk_nope_head_dim"),
            ("qk_rope_head_dim", "swa_qk_rope_head_dim"),
            ("v_head_dim",       "swa_v_head_dim"),
            ("qk_head_dim",      "swa_qk_head_dim"),
            ("kv_lora_rank",     "swa_kv_lora_rank"),
        ):
            if window_key not in hparams:
                continue
            if hparams[window_key] != hparams[sparse_key]:
                raise NotImplementedError(
                    f"HY_V4_FLASH needs one MLA geometry for both layer types, but "
                    f"{window_key}={hparams[window_key]} != {sparse_key}={hparams[sparse_key]}"
                )

    def _indexer_enabled(self) -> bool:
        return bool(self.hparams.get("use_dsa")) and bool(self.hparams.get("use_compressed_indexer"))

    def indexer_is_full(self) -> list[bool] | None:
        """Which layers own a k-pool indexer: the sparse ones. The config indexer_types marks every layer."""
        if not self._indexer_enabled():
            return None
        n_all = self.hparams["num_hidden_layers"] + self._n_nextn()
        return [self._layer_type_of(i) == "sparse_attention" for i in range(n_all)]

    def prepare_tensors(self):
        """Check the compressed_indexer tensors instead of hy_v4's self_attn.indexer ones."""
        self.hparams["num_key_value_heads"] = self.hparams["num_attention_heads"]

        is_full = self.indexer_is_full()
        if is_full is not None:
            want = set(self.INDEXER_MAP)
            n_layer = self.hparams["num_hidden_layers"]
            present: dict[int, set[str]] = {}
            for name in self.model_tensors:
                head, sep, tail = name.partition(".self_attn.compressed_indexer.")
                if not sep:
                    continue
                # the draft block carries its own indexer, under a different prefix
                if head.startswith("model.layers."):
                    present.setdefault(int(head.split(".")[2]), set()).add(tail)
                elif head.startswith("model.mtp_layers."):
                    present.setdefault(n_layer + int(head.split(".")[2]), set()).add(tail)
            for il, expect in enumerate(is_full):
                seen = present.get(il, set())
                if expect and seen != want:
                    raise ValueError(
                        f"HY_V4_FLASH layer {il} owns an indexer but is missing "
                        f"{sorted(want - seen)}" + (f" and carries unexpected {sorted(seen - want)}"
                                                    if seen - want else ""))
                if not expect and seen:
                    raise ValueError(
                        f"HY_V4_FLASH layer {il} owns no indexer but carries {sorted(seen)}")

        # straight to the grandparent: hy_v4's own validation does not apply here
        super(HYV4Model, self).prepare_tensors()

        if self._engram_enabled():
            T = gguf.MODEL_TENSOR
            want = {self.format_tensor_name(T.PER_LAYER_TOKEN_EMBD)}
            for i in self._engram_ids():
                for t in (T.ENGRAM_KV, T.ENGRAM_KV_B, T.ENGRAM_GATE, T.ENGRAM_LOOKUP_NORM, T.ENGRAM_HC_SCALE, T.ENGRAM_PRE_GAMMA):
                    want.add(self.format_tensor_name(t, i - 1))
            if want - self._engram_done:
                raise ValueError(f"HY_V4_FLASH: incomplete Engram tensors {sorted(want - self._engram_done)}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        # ggml_add / ggml_mul operands, kept f32 as deepseek41 does for engram_gate
        if new_name.endswith((".engram_gate.weight", ".engram_kv_b.weight", ".engram_hc_scale.weight")):
            return gguf.GGMLQuantizationType.F32
        # the pooled-key position bias is a ggml_add operand too
        if new_name.endswith(".indexer_compressor_ape.weight"):
            return gguf.GGMLQuantizationType.F32
        # the base rule misses the router of the renumbered draft block
        if new_name.endswith((".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _engram_modify(self, data_torch: torch.Tensor, name: str,
                       bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        """Fuse a layer's Engram weights as deepseek41 does, once all parts of a group arrived."""
        if bid is None:
            raise ValueError(f"HY_V4_FLASH: engram tensor without a layer id: {name}")

        hc = self.hparams["hc_mult"]
        n_head = self.hparams["engram_n_head_per_ngram"] * (self.hparams["engram_max_ngram_size"] - 1)

        tail = name.split(".engram_layer.", 1)[1]
        if tail == self.ENGRAM_LOOKUP_TABLE:
            # the token map, read through _engram_vocab_map() and written as GGUF metadata
            return

        buf = self._engram_buf.setdefault(bid, {})
        buf[tail] = data_torch

        tn = self.format_tensor_name
        T = gguf.MODEL_TENSOR

        def ready(*names) -> bool:
            return all(n in buf for n in names)

        def take(*names):
            return [buf.pop(n) for n in names]

        # hc key projections then the value projection, the order the graph views assume
        kv_parts = [f"key_projs.{c}.weight" for c in range(hc)] + ["value_proj.weight"]
        if ready(*kv_parts):
            yield tn(T.ENGRAM_KV, bid), torch.cat(take(*kv_parts), dim=0)

        kvb_parts = [f"key_projs.{c}.bias" for c in range(hc)] + ["value_proj.bias"]
        if ready(*kvb_parts):
            yield tn(T.ENGRAM_KV_B, bid), torch.stack(take(*kvb_parts), dim=0)

        # engram_gate folds the key and query norms, as deepseek41 does
        gate_parts = [f"norm_key.{c}.weight" for c in range(hc)] + [f"norm_query.{c}.weight" for c in range(hc)]
        if ready(*gate_parts):
            parts = take(*gate_parts)
            yield tn(T.ENGRAM_GATE, bid), torch.stack(
                [parts[c] * parts[hc + c] for c in range(hc)], dim=0)

        norm_parts = [f"lookup_per_head_norms.{c}.weight" for c in range(n_head)]
        if ready(*norm_parts):
            yield tn(T.ENGRAM_LOOKUP_NORM, bid), torch.stack(take(*norm_parts), dim=0)

        scale_parts = [f"unified_per_hc_scales.{c}" for c in range(hc)]
        if ready(*scale_parts):
            yield tn(T.ENGRAM_HC_SCALE, bid), torch.stack(take(*scale_parts), dim=0)

        if ready("unified_pre_gamma"):
            yield tn(T.ENGRAM_PRE_GAMMA, bid), take("unified_pre_gamma")[0]

        # the tables of all modules are concatenated in layer order
        if ready("multi_head_embedding.embedding.weight"):
            self._engram_tables[bid] = take("multi_head_embedding.embedding.weight")[0]
            want = [i - 1 for i in self._engram_ids()]
            if all(i in self._engram_tables for i in want):
                self._write_engram_tables([self._engram_tables.pop(i) for i in want])

    def _write_engram_tables(self, tables: list[torch.Tensor]) -> None:
        # the concatenated tables are ~127 GB in BF16, so they are streamed one table at a time as deepseek41 does
        f32 = self.ftype == gguf.LlamaFileType.ALL_F32
        dim = tables[0].shape[1]
        if any(t.shape[1] != dim for t in tables):
            raise ValueError("HY_V4_FLASH engram tables differ in width")
        n_rows = sum(t.shape[0] for t in tables)

        def chunks() -> Iterable[np.ndarray]:
            while tables:
                table = LazyTorchTensor.to_eager(tables.pop(0))
                for r0 in range(0, table.shape[0], 1 << 18):
                    # a copy, so that no chunk still held by the writer keeps the whole table alive
                    rows = (table[r0:r0 + (1 << 18)].float() if f32 else table[r0:r0 + (1 << 18)].bfloat16()).clone()
                    yield rows.view(torch.uint8).numpy()
                del table, rows

        qt = gguf.GGMLQuantizationType.F32 if f32 else gguf.GGMLQuantizationType.BF16
        data = cast(np.ndarray, _StreamedBytes((n_rows, dim * (4 if f32 else 2)), chunks))
        name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
        self.gguf_writer.add_tensor(name, data, raw_dtype=qt)
        self._engram_done.add(name)

    NEXTN_MAP = {
        "eh_proj.weight":          gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
        "enorm.weight":            gguf.MODEL_TENSOR.NEXTN_ENORM,
        "hnorm.weight":            gguf.MODEL_TENSOR.NEXTN_HNORM,
        "final_layernorm.weight":  gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
    }

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        if name.startswith("model.mtp_layers."):
            if not self._n_nextn() or getattr(self, "no_mtp", False):
                return
            parts = name.split(".", 3)
            nbid = self.hparams["num_hidden_layers"] + int(parts[2])
            tail = parts[3]
            key = self.NEXTN_MAP.get(tail)
            if key is not None:
                yield self.format_tensor_name(key, nbid), data_torch
                return
            # the rest of the block is an ordinary sparse layer, so re-enter under its block index
            yield from self.modify_tensors(data_torch, f"model.layers.{nbid}.{tail}", nbid)
            return

        if ".compressed_indexer." in name:
            # dropped when the indexer is off, the sparse layers then attend their whole context
            if not self._indexer_enabled():
                return
            tail = name.split(".compressed_indexer.", 1)[1]
            key = self.INDEXER_MAP.get(tail)
            if key is None:
                raise ValueError(f"HY_V4_FLASH: unmapped compressed_indexer tensor {name}")
            # intra_group_ape [kpool, head_dim] is already ne (head_dim, kpool)
            yield self.format_tensor_name(key, bid), data_torch
            return

        if ".engram_layer." in name:
            if not self._engram_enabled():
                return
            for new_name, t in self._engram_modify(data_torch, name, bid):
                self._engram_done.add(new_name)
                yield new_name, t
            return

        # split kv_b_proj with the head count of this layer, the base uses a global one
        if name.endswith("kv_b_proj.weight") and bid is not None:
            hparams = self.hparams
            n_head = (hparams["swa_num_attention_heads"]
                      if self._layer_type_of(bid) == "sliding_attention"
                      else hparams["num_attention_heads"])
            v_head_dim = hparams["v_head_dim"]
            qk_nope_head_dim = hparams["qk_nope_head_dim"]

            expected = n_head * (v_head_dim + qk_nope_head_dim)
            if data_torch.shape[0] != expected:
                raise ValueError(
                    f"HY_V4_FLASH layer {bid}: {name} has {data_torch.shape[0]} rows, "
                    f"expected {expected} = {n_head} heads x ({qk_nope_head_dim} + {v_head_dim})"
                )

            kv_b = data_torch.view(n_head, qk_nope_head_dim + v_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            tn = self.format_tensor_name
            yield from ModelBase.modify_tensors(self, k_b, tn(gguf.MODEL_TENSOR.ATTN_K_B, bid), bid)
            yield from ModelBase.modify_tensors(self, v_b, tn(gguf.MODEL_TENSOR.ATTN_V_B, bid), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("HYV4FlashForCausalLM", "HYV4FlashForConditionalGeneration")
class HYV4FlashVisionModel(MmprojModel):
    """Vision tower of HY_V4_FLASH: a SigLIP-shaped encoder with a rmsnorm_mlp_adapter projector.

    Positions come from a learned grid, resampled to the patch grid. The projector merges 2x2 patches with a strided conv2d, runs a 1x1 conv2d and appends a newline embedding to every row.
    """

    projector_type = gguf.VisionProjectorType.HY_V4_FLASH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # there is no image_size, the position-embedding grid is max_image_size / patch_size
        self.hparams_vision["image_size"] = self.hparams_vision.get("max_image_size", 2048)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        vcfg = self.hparams_vision
        projector_type = vcfg.get("projector_type")
        if projector_type != "rmsnorm_mlp_adapter":
            raise ValueError(f"unsupported projector_type {projector_type!r}")
        # the graph implements the plain variant only
        unsupported = {
            "use_2d_rope": bool(vcfg.get("use_2d_rope", False)),
            "cat_extra_token": bool(vcfg.get("cat_extra_token", False)),
            "use_image_newline_token": not vcfg.get("use_image_newline_token", True),
            "perceive_post_norm": bool(vcfg.get("perceive_post_norm", False)),
            "hidden_act": vcfg.get("hidden_act", "gelu") != "gelu",
            "interpolate_mode": vcfg.get("interpolate_mode", "bilinear") != "bilinear",
        }
        for key, bad in unsupported.items():
            if bad:
                raise NotImplementedError(f"HY_V4_FLASH vision config {key}={vcfg.get(key)!r} is not supported")
        self.gguf_writer.add_clip_projector_type(self.projector_type)
        self.gguf_writer.add_vision_use_gelu(True)
        # rms_norm_eps is also the LayerNorm eps of the encoder blocks
        self.gguf_writer.add_vision_attention_layernorm_eps(vcfg.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_vision_spatial_merge_size(vcfg.get("spatial_merge_size", 2))
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], torch.Tensor]]) -> tuple[str, Callable[[], torch.Tensor]] | None:
        if not item[0].startswith("vit."):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        # row 0 of the position embedding is an unused CLS slot, the reference resamples rows 1:
        if "position_embedding" in name:
            data_torch = data_torch[1:]
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # conv weights in F32 or F16, BF16 IM2COL is not supported on Metal
        if ("mm.0." in new_name or "mm.2." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
