from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .base import ModelBase, gguf, logger
from .hy_v4 import HYV4Model

# the hash multipliers are not stored in the checkpoint: the reference derives them from the
# Engram seed, the 1-based layer id and the compressed vocabulary size
ENGRAM_SEED_STRIDE = 10007


@ModelBase.register("HYV4NextForCausalLM", "HYV4NextVLForConditionalGeneration")
@ModelBase.example("tencent/Hy4-Flash")
class HYV4FlashModel(HYV4Model):
    """HY_V4_FLASH: hy_v4's iHC + gated MLA + learnable sink with a hybrid attention stack.

    The stack mixes "sparse" full-attention layers with sliding-window layers, in the order given
    by layer_types rather than a fixed alternation. Both are MLA over the same latent geometry and
    differ only in head count and rope base, so the head count is exported per layer and the window
    layers get their own rope frequency.

    Added over hy_v4: sandwich norms (after the attention output and after the FFN) and routed
    experts that run in a latent space reached by routed_expert_down_proj / _up_proj, while the
    router and the shared expert stay at hidden_size.

    The MTP (nextn) draft block is exported as one extra sparse layer with no hyper-connections of
    its own; it is fed the next token's embedding and the trunk's final hidden state.

    Dropped: the vision tower, which is converted separately as an mmproj.
    """

    model_arch = gguf.MODEL_ARCH.HY_V4_FLASH

    # the draft block is exported, and _engram_modify re-enters the normal path under a block index
    # past the trunk. DeepseekV2Model.modify_tensors drops exactly that range when skip_mtp is set,
    # which would silently lose most of the block.
    skip_mtp = False

    # dropped subsystems, see the class docstring
    DROP_PREFIXES = ("vit.",)
    DROP_SUBSTRINGS = ()

    # compressed_indexer tail -> the tensor the graph reads it as
    INDEXER_MAP = {
        "token_key_projection.weight":   gguf.MODEL_TENSOR.INDEXER_ATTN_K,
        "token_key_norm.weight":         gguf.MODEL_TENSOR.INDEXER_K_NORM,
        "key_value_projection.weight":   gguf.MODEL_TENSOR.INDEXER_KPOOL_VALUE,
        "key_gate_projection.weight":    gguf.MODEL_TENSOR.INDEXER_KPOOL_GATE,
        "intra_group_ape":               gguf.MODEL_TENSOR.INDEXER_KPOOL_APE,
        "key_norm.weight":               gguf.MODEL_TENSOR.INDEXER_KPOOL_NORM,
        "query_projection.weight":       gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B,
        "head_weight_projection.weight": gguf.MODEL_TENSOR.INDEXER_PROJ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the draft block is an extra block as far as tensor naming goes; block_count is what the
        # loader reads as n_layer_all and it subtracts the nextn count back off
        if not getattr(self, "no_mtp", False):
            self.block_count += self._n_nextn()
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        # Engram weights arrive one at a time but are written fused, so they wait here
        self._engram_buf: dict[int, dict[str, torch.Tensor]] = {}
        self._engram_tables: dict[int, torch.Tensor] = {}

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
        unknown = set(layer_types[:n_layer]) - {"sparse_attention", "sliding_attention"}
        if unknown:
            raise ValueError(f"HY_V4_FLASH unknown layer_types values: {sorted(unknown)}")
        self._layer_types_cache = list(layer_types[:n_layer])
        return self._layer_types_cache

    # ---- Engram ------------------------------------------------------------------------------
    #
    # Four n-gram hash modules, one per Engram layer. All of their buckets are packed into a single
    # table, so each module's head offsets are rebased into it, matching deepseek41's layout.

    def _engram_enabled(self) -> bool:
        return bool(self.hparams.get("engram_enable")) and bool(self.hparams.get("engram_layer_ids"))

    def _engram_ids(self) -> list[int]:
        """The config's engram_layer_ids, 1-based, in ascending order."""
        ids = sorted(int(i) for i in self.hparams["engram_layer_ids"])
        n_layer = self.hparams["num_hidden_layers"]
        for i in ids:
            # 1-based: the tensors live at model.layers.{i-1}.engram_layer.*
            if not 1 <= i <= n_layer:
                raise ValueError(f"HY_V4_FLASH engram_layer_ids has {i}, outside 1..{n_layer}")
        return ids

    def _engram_primes(self) -> dict[int, list[int]]:
        """Per 1-based layer id, the 16 bucket counts flattened from their [ngram][head] nesting."""
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

    def _engram_vocab_map(self) -> tuple[list[int], int]:
        """The token -> compressed id map, extended by identity, and the compressed vocab size.

        The map covers only the source tokenizer, which is shorter than the model's vocabulary. The
        reference extends it by identity, which puts the trailing reserved ids outside the
        compressed range; they hash to a row anyway because the hash is taken modulo the bucket
        count, so they stay in range but are not reachable by any trained token.
        """
        path = Path(self.dir_model) / "engram_vocab_map.npz"
        if not path.is_file():
            raise FileNotFoundError(f"HY_V4_FLASH engram needs {path}")
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

        # offsets rebase each module's buckets into the single shared table; row ids are int32, so
        # the loader rejects a table whose last bucket ends past INT32_MAX
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
        """Layer type of any block, including the draft one.

        A draft block defaults to a sparse layer that owns an indexer, which is the shape the
        checkpoints are exported with, and `mtp_layer_types` overrides it per draft layer. The
        graph only drives a sparse draft block, so refuse anything else rather than write a file
        whose head count and rope would silently disagree with it.
        """
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

    def _rope_for(self, layer_type: str) -> dict:
        params = self.hparams.get("rope_parameters")
        if not isinstance(params, dict) or layer_type not in params:
            raise ValueError(f"HY_V4_FLASH needs rope_parameters[{layer_type!r}]")
        return params[layer_type]

    def set_gguf_parameters(self):
        hparams = self.hparams
        layer_types = self._layer_types()
        is_swa = [t == "sliding_attention" for t in layer_types]

        if not any(is_swa) or all(is_swa):
            raise ValueError("HY_V4_FLASH expects a mix of sparse and sliding attention layers")

        # The graph applies the sandwich post-norms on every layer unconditionally, so refuse a
        # config that asks for either of them to be skipped: the tensors ship in both cases, so
        # the file would convert cleanly and then quietly compute something else.
        if not hparams.get("sandwich_norm", False):
            raise ValueError("HY_V4_FLASH expects sandwich_norm, the graph always applies the post-norms")
        if hparams.get("dsa_skip_attn_post_norm", False):
            raise ValueError(
                "HY_V4_FLASH: dsa_skip_attn_post_norm is not supported - the graph applies "
                "attn_post_norm on the sparse layers too")

        # the base reads rope_parameters["full_attention"], which this config does not have:
        # its non-sliding entry is keyed "sparse_attention". Without this the sparse layers
        # would silently fall back to the default rope base.
        sparse_rope = self._rope_for("sparse_attention")
        window_rope = self._rope_for("sliding_attention")

        # hy_v4 warns that it drops the draft block. This model exports it, so keep the key out of
        # sight for that one call rather than let the warning mislead.
        n_nextn = self.hparams.pop("num_nextn_predict_layers", None)
        try:
            super().set_gguf_parameters()
        finally:
            if n_nextn is not None:
                self.hparams["num_nextn_predict_layers"] = n_nextn

        self.gguf_writer.add_rope_freq_base(sparse_rope["rope_theta"])

        # per-layer head count: the window layers carry swa_num_attention_heads
        n_head = hparams["num_attention_heads"]
        n_head_swa = hparams["swa_num_attention_heads"]
        self.gguf_writer.remove_key(
            gguf.Keys.Attention.HEAD_COUNT.format(arch=self.gguf_writer.arch))
        # the arrays are read over n_layer_all, so the draft block needs an entry too: it is a
        # sparse layer, so the global head count and no window
        is_swa_all = is_swa + [False] * self._n_nextn()
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
            # the base writes head_count / key_length / top_k / types off indexer_is_full(); the
            # pool size is this model's compress_ratio, and the current partial group is always kept
            ratio = hparams["compressed_indexer_compress_ratio"]
            self.gguf_writer.add_indexer_kpool(ratio)
            self.gguf_writer.add_indexer_kpool_select_tail(True)

            # the reference's own config check: the candidate budget is the group budget expanded
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
            logger.info(
                "HY_V4_FLASH indexer: %d pooled groups of %d, keeping the best %d plus the "
                "current group (index_topk %d)",
                sum(self.indexer_is_full()), ratio, hparams["index_topk"] // ratio,
                hparams["index_topk"])

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
        """The window layers must share the sparse layers' MLA geometry.

        Only the head count may differ: the graph keeps one latent cache for both layer types
        and reads a single set of MLA dimensions.
        """
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
        """Which layers own a k-pool indexer.

        Derived from layer_types, not from indexer_types: the config marks all 42 layers "full",
        which there means the layer carries an indexer *config*, while only the sparse layers
        actually hold compressed_indexer weights. llama.cpp's flag means "owns one".
        """
        if not self._indexer_enabled():
            return None
        n_all = self.hparams["num_hidden_layers"] + self._n_nextn()
        return [self._layer_type_of(i) == "sparse_attention" for i in range(n_all)]

    def prepare_tensors(self):
        """Validate against this model's own indexer, then skip hy_v4's check.

        hy_v4 looks for the lightning indexer's `self_attn.indexer.*`; this model instead carries
        `self_attn.compressed_indexer.*` on its sparse layers, so the inherited check would reject
        every file.
        """
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

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        # the graph feeds these straight to ggml_add / ggml_mul as the second operand, which has no
        # f16 kernel. deepseek41 keeps engram_gate f32 for the same reason; all three are tiny.
        if new_name.endswith((".engram_gate.weight", ".engram_kv_b.weight", ".engram_hc_scale.weight")):
            return gguf.GGMLQuantizationType.F32
        # same for the pooled-key position bias, which is a ggml_add operand. glm keeps every
        # indexer_compressor_ tensor f32; here the norms already fall under the base class's
        # *_norm.weight rule and the rest are matmul operands, so only the ape needs it.
        if new_name.endswith(".indexer_compressor_ape.weight"):
            return gguf.GGMLQuantizationType.F32
        # The router picks which experts fire, so it stays f32 like every other arch's. The base
        # class's rule for it matches the tensor name against the *source* layer id, which no
        # longer agrees with the name once the draft block has been renumbered into the trunk's
        # range - so without this the draft block's router alone would ship at the source dtype.
        if new_name.endswith((".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def _engram_modify(self, data_torch: torch.Tensor, name: str,
                       bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        """Fuse a layer's Engram weights into the tensors the graph reads.

        The per-hc key projections and the single value projection become one matmul plus one bias,
        and the two gate norms multiply into one, both matching deepseek41's fused layout. Nothing
        is emitted until a group is complete, because the fused tensor needs every part.
        """
        if bid is None:
            raise ValueError(f"HY_V4_FLASH: engram tensor without a layer id: {name}")

        hc = self.hparams["hc_mult"]
        n_head = self.hparams["engram_n_head_per_ngram"] * (self.hparams["engram_max_ngram_size"] - 1)

        buf = self._engram_buf.setdefault(bid, {})
        tail = name.split(".engram_layer.", 1)[1]
        buf[tail] = data_torch

        tn = self.format_tensor_name
        T = gguf.MODEL_TENSOR

        def ready(*names) -> bool:
            return all(n in buf for n in names)

        def take(*names):
            return [buf.pop(n) for n in names]

        # the fused matmul: hc key projections then the value projection, which is the order the
        # graph's views assume
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

        # every module's buckets share one table, concatenated in layer order, so it can only be
        # written once the last module has arrived
        if ready("multi_head_embedding.embedding.weight"):
            self._engram_tables[bid] = take("multi_head_embedding.embedding.weight")[0]
            want = [i - 1 for i in self._engram_ids()]
            if all(i in self._engram_tables for i in want):
                tables = [self._engram_tables.pop(i) for i in want]
                yield (tn(T.PER_LAYER_TOKEN_EMBD),
                       tables[0] if len(tables) == 1 else torch.cat(tables, dim=0))

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
            # dropped wholesale when the indexer is off, so the GGUF stays loadable as a
            # full-attention model
            if not self._indexer_enabled():
                return
            tail = name.split(".compressed_indexer.", 1)[1]
            key = self.INDEXER_MAP.get(tail)
            if key is None:
                raise ValueError(f"HY_V4_FLASH: unmapped compressed_indexer tensor {name}")
            # intra_group_ape is numpy [kpool, head_dim], which gguf writes as ne (head_dim, kpool)
            # already, so no transpose is needed anywhere here
            yield self.format_tensor_name(key, bid), data_torch
            return

        if ".engram_layer." in name:
            if not self._engram_enabled():
                return
            yield from self._engram_modify(data_torch, name, bid)
            return

        # the base splits kv_b_proj with one global head count; here the sliding-window
        # layers carry swa_num_attention_heads over the same MLA geometry, so the split
        # has to read the head count of this layer
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
