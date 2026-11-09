from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .afmoe import AfmoeModel
from .base import ModelBase, gguf, logger


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1")
class Kolibri1Model(AfmoeModel):
    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AfmoeModel reads these keys, this config stores the same values under other names
        self.hparams["num_shared_experts"] = 1
        self.hparams["route_norm"] = self.hparams["norm_topk_prob"]
        self.hparams["intermediate_size"] = self.hparams["moe_intermediate_size"]

        # A dense prefix replaces the MoE of the leading layers by dense FFNs
        if dense_layers := self.hparams.get("mlp_only_layers"):
            if dense_layers != list(range(len(dense_layers))):
                raise ValueError(f"Dense FFN layers must be the leading layers, not {dense_layers}")
            self.hparams["num_dense_layers"] = len(dense_layers)
            self.hparams["intermediate_size"] = self.hparams["mlp_only_intermediate_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)


@ModelBase.register("Kolibri1_1ForCausalLM")
class Kolibri11Model(Kolibri1Model):
    """Kolibri-1.1: Kolibri-1 with sparse attention in all full-attention layers."""

    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if bid is not None and self.hparams["kv_source_layers"][bid] != bid:
            if name.endswith((".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.k_norm.weight")):
                # Checkpoints retain K/V projections that shared-KV layers do not use.
                logger.info("Skipping unused shared-KV tensor %s", name)
                return
        yield from super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        # The sparse policy llama.cpp implements: NVFP4 K/V and routing keys in the sparse layers,
        # 8 sink keys, 128 recent keys and routing over the query heads of the first KV head.
        supported = {
            "kv_cache_qat": True,
            "index_num_sink_tokens": 8,
            "index_num_local_tokens": 128,
            "index_num_heads": hp["num_attention_heads"] // hp["num_key_value_heads"],
        }
        for key, value in supported.items():
            if hp.get(key) != value:
                raise ValueError(f"Unsupported sparse attention: {key} is {hp.get(key)!r}, expected {value!r}")

        n_layers = hp["num_hidden_layers"]
        swa_layers = {i for i, t in enumerate(hp["layer_types"]) if t == "sliding_attention"}
        kv_sources = hp["kv_source_layers"]
        indexers = set(hp["index_layers"])
        if len(kv_sources) != n_layers or any(kv_sources[i] != i for i in swa_layers) or indexers & swa_layers:
            raise ValueError("Sliding-window layers must neither share K/V nor be indexers")
        if hp["index_pool_size"] % hp["index_compression"] != 0:
            raise ValueError("The pool must hold whole blocks")

        self.gguf_writer.add_indexer_types([i in indexers for i in range(n_layers)])
        self.gguf_writer.add_kv_source_map(kv_sources)
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_candidate_block_size(hp["index_compression"])
        self.gguf_writer.add_indexer_candidate_top_k_blocks(hp["index_pool_size"] // hp["index_compression"])
