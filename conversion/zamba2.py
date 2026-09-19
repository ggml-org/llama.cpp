from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .mamba import Mamba2Model


@ModelBase.register("Zamba2ForCausalLM")
class Zamba2Model(Mamba2Model):
    model_arch = gguf.MODEL_ARCH.ZAMBA2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Zamba2's SSM d_inner is mamba_expand * hidden_size (intermediate_size is the FFN dim)
        mamba_expand = self.find_hparam(["mamba_expand"], optional=True) or 2
        self.d_inner = int(mamba_expand * self.d_model)

        # Layer classification from config
        block_types = self.hparams.get("layers_block_type", [])
        self._hybrid_layers: list[int] = [
            i for i, t in enumerate(block_types) if t == "hybrid"
        ]
        self._num_mem_blocks: int = self.hparams.get("num_mem_blocks", 1)

        # Map shared block index -> list of hybrid layer indices that use it.
        # Hybrid visit N uses shared block (N % num_mem_blocks).
        self._shared_block_layers: dict[int, list[int]] = {}
        for visit_idx, layer_idx in enumerate(self._hybrid_layers):
            block_idx = visit_idx % self._num_mem_blocks
            self._shared_block_layers.setdefault(block_idx, []).append(layer_idx)

        # Source layers: first num_mem_blocks hybrid layers store the shared weights
        self._shared_block_src: list[int] = self._hybrid_layers[:self._num_mem_blocks]

        # Pre-load per-layer adapter weights for merging into shared transformer
        # weights. Zamba2 has rank-128 LoRA-style adapters that are integral to
        # the architecture (not optional fine-tuning). Each adapter is a
        # Sequential(Linear(in, 128), Linear(128, out)), stored as:
        #   adapter_list.{visit_idx}.0.weight  (down projection)
        #   adapter_list.{visit_idx}.1.weight  (up projection)
        # The merged weight = shared_weight + up @ down
        self._adapter_cache: dict[str, Tensor] = {}
        self._use_attn_adapters = self.hparams.get("use_shared_attention_adapter", False)
        self._use_mlp_adapters = self.hparams.get("use_shared_mlp_adapter", True)

        adapter_pattern = re.compile(
            r"model\.layers\.(\d+)\.shared_transformer\."
            r"(?:self_attn\.(linear_[qkv]_adapter_list)|feed_forward\.(gate_up_proj_adapter_list))"
            r"\.(\d+)\.([01])\.weight"
        )
        for tname in list(self.model_tensors.keys()):
            m = adapter_pattern.match(tname)
            if m:
                layer_idx = int(m.group(1))
                adapter_type = m.group(2) or m.group(3)
                visit_idx = int(m.group(4))
                seq_idx = int(m.group(5))  # 0=down, 1=up
                # Only cache from source layers (avoid duplicates from weight ties)
                if layer_idx in self._shared_block_src:
                    key = f"{layer_idx}.{adapter_type}.{visit_idx}.{seq_idx}"
                    self._adapter_cache[key] = self.model_tensors[tname]()

    def _get_adapter_contribution(self, src_layer: int, adapter_type: str, visit_idx: int) -> Tensor | None:
        """Compute up @ down for a given adapter, returning the additive correction."""
        down_key = f"{src_layer}.{adapter_type}.{visit_idx}.0"
        up_key = f"{src_layer}.{adapter_type}.{visit_idx}.1"
        down = self._adapter_cache.get(down_key)
        up = self._adapter_cache.get(up_key)
        if down is None or up is None:
            return None
        # Merge in float32 for precision, result will be cast by caller
        return (up.float() @ down.float())

    def set_vocab(self):
        # Zamba2 uses LlamaTokenizer (sentencepiece); tokenizer.json may be the
        # only file present (no tokenizer.model).  Follow JambaModel's pattern.
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        elif (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_llama_hf()
        else:
            self._set_vocab_builtin("gpt-neox", self.hparams["vocab_size"])

    def set_gguf_parameters(self):
        hparams = self.hparams
        n_head = hparams["num_attention_heads"]
        n_kv   = hparams.get("num_key_value_heads", n_head)
        block_types = hparams.get("layers_block_type", [])

        # Per-layer KV head count: 0 for mamba-only, n_kv for hybrid
        n_kv_vec = [n_kv if t == "hybrid" else 0 for t in block_types]

        head_dim = hparams.get("attention_head_dim",
                               hparams.get("attention_hidden_size", 2 * self.d_model) // n_head)

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_embedding_length(self.d_model)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_kv_vec)
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_feed_forward_length(
            hparams.get("ffn_hidden_size", hparams.get("intermediate_size")))
        self.gguf_writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 4096))
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # RoPE (only when use_mem_rope is enabled)
        if hparams.get("use_mem_rope", False):
            self.gguf_writer.add_rope_dimension_count(head_dim)
            self.gguf_writer.add_rope_freq_base(hparams.get("rope_theta", 10000.0))
        else:
            self.gguf_writer.add_rope_dimension_count(0)

        # Mamba-2 SSM parameters
        d_conv  = hparams.get("mamba_d_conv", 4)
        d_state = hparams.get("mamba_d_state", 64)
        headdim = hparams.get("mamba_headdim", 64)
        n_heads = self.d_inner // headdim

        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(n_heads)
        self.gguf_writer.add_ssm_group_count(self.n_group)

        self.gguf_writer.add_file_type(self.ftype)

    # Map from shared weight tail names to their adapter type names
    _ADAPTER_MAP: dict[str, str] = {
        "self_attn.q_proj.weight": "linear_q_adapter_list",
        "self_attn.k_proj.weight": "linear_k_adapter_list",
        "self_attn.v_proj.weight": "linear_v_adapter_list",
        "feed_forward.gate_up_proj.weight": "gate_up_proj_adapter_list",
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Skip adapter weights — they are pre-merged into shared weights below
        if "adapter" in name.lower():
            return

        # Shared transformer tensors: duplicate to all hybrid layers using this block,
        # merging per-layer adapter corrections where applicable.
        if "shared_transformer" in name:
            if bid is None or bid not in self._shared_block_src:
                return
            block_idx = self._shared_block_src.index(bid)
            target_layers = self._shared_block_layers[block_idx]

            # Strip the shared_transformer prefix so names match the standard HF layout
            tail = name.partition("shared_transformer.")[2]

            # input_layernorm sits before the attention-on-concat block, so it maps
            # to ATTN_POST_NORM rather than the default ATTN_NORM.  Handle explicitly.
            if tail == "input_layernorm.weight":
                for target_bid in target_layers:
                    yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_POST_NORM, target_bid), data_torch)
                return

            # Check if this weight has a corresponding adapter type
            adapter_type = self._ADAPTER_MAP.get(tail)
            use_adapter = False
            if adapter_type:
                if adapter_type.startswith("linear_") and self._use_attn_adapters:
                    use_adapter = True
                elif adapter_type == "gate_up_proj_adapter_list" and self._use_mlp_adapters:
                    use_adapter = True

            # Delegate to the parent's tensor-name mapping, once per target layer.
            for visit_local, target_bid in enumerate(target_layers):
                # Compute the global visit index for this target layer
                global_visit = self._hybrid_layers.index(target_bid)

                if use_adapter and adapter_type is not None:
                    correction = self._get_adapter_contribution(bid, adapter_type, global_visit)
                    if correction is not None:
                        # Merge: per-layer weight = shared + adapter_up @ adapter_down
                        merged = data_torch.float() + correction
                        layer_data = merged.to(data_torch.dtype)
                    else:
                        layer_data = data_torch
                else:
                    layer_data = data_torch

                rewritten = f"model.layers.{target_bid}.{tail}"
                yield from super().modify_tensors(layer_data, rewritten, target_bid)
            return

        # Linear mixing weight (hybrid layers)
        if name.endswith(".linear.weight"):
            if bid is not None:
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_MIX, bid), data_torch)
            return

        # Mamba/SSM tensors: strip mamba_decoder prefix then delegate to parent
        # Parent handles: dt_bias rename, map_tensor_name, conv1d squeeze,
        #   A_log -> -exp, A/D reshape, NORM reshape
        if ".mamba_decoder." in name:
            name = name.replace(".mamba_decoder.", ".")

        yield from super().modify_tensors(data_torch, name, bid)
