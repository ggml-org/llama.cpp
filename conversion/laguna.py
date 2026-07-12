from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("LagunaForCausalLM")
class LagunaModel(TextModel):
    """poolside Laguna-XS-2.1 — 33B MoE, agentic coding.

    Architecturally close to Step 3.5 (head-wise attention gate, QK-norm, sigmoid
    MoE + correction bias + shared expert, mixed sliding/full attention with a
    per-type RoPE). Laguna deltas handled here:
      - per-layer variable query heads (48 full / 64 sliding, KV=8 constant),
      - partial YaRN RoPE on full-attention layers (n_rot=64) vs full rotary on
        sliding layers (n_rot=128),
      - fused expert weights (gate_up_proj) split into gate/up,
      - moe_routed_scaling_factor (2.5), e_score_correction_bias key.
    See LAGUNA-PORT.md.
    """

    model_arch = gguf.MODEL_ARCH.LAGUNA

    def set_gguf_parameters(self):
        # Base emits: block_count, context_length, embedding_length,
        # feed_forward_length (dense n_ff), scalar head_count/head_count_kv, and
        # RoPE (reads rope_parameters["full_attention"] → YaRN, and
        # rope_parameters["sliding_attention"].rope_theta → local theta).
        super().set_gguf_parameters()

        hp = self.hparams
        head_dim = hp["head_dim"]
        layer_types = hp.get("layer_types") or ["full_attention"] * self.block_count

        # --- per-layer variable query heads (KV heads are constant here) ---
        per_layer_heads = hp.get("num_attention_heads_per_layer")
        if per_layer_heads is not None:
            self.gguf_writer.add_head_count(list(per_layer_heads))
        self.gguf_writer.add_head_count_kv(hp["num_key_value_heads"])
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_key_length(head_dim)

        # --- mixed sliding/full attention ---
        swa_pattern = [lt == "sliding_attention" for lt in layer_types]
        self.gguf_writer.add_sliding_window(hp["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_pattern)

        # --- partial dual RoPE: full layers rotate n_rot dims (partial 0.5),
        #     sliding layers rotate the full head_dim (partial 1.0) ---
        rp = hp.get("rope_parameters", {})
        full_partial = float(rp.get("full_attention", {}).get("partial_rotary_factor", 1.0))
        swa_partial = float(rp.get("sliding_attention", {}).get("partial_rotary_factor", 1.0))
        self.gguf_writer.add_rope_dimension_count(int(head_dim * full_partial))
        self.gguf_writer.add_rope_dimension_count_swa(int(head_dim * swa_partial))

        # --- MoE ---
        self.gguf_writer.add_expert_count(hp["num_experts"])
        self.gguf_writer.add_expert_used_count(hp["num_experts_per_tok"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_weights_scale(float(hp.get("moe_routed_scaling_factor", 1.0)))
        self.gguf_writer.add_expert_weights_norm(bool(hp.get("norm_topk_prob", True)))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # leading dense blocks: mlp_only_layers lists the dense layer indices
        # (contiguous from 0 here), and decoder_sparse_step controls MoE cadence.
        mlp_only = sorted(hp.get("mlp_only_layers") or [])
        leading_dense = 0
        while leading_dense in mlp_only:
            leading_dense += 1
        self.gguf_writer.add_leading_dense_block_count(leading_dense)
        self.gguf_writer.add_moe_every_n_layers(int(hp.get("decoder_sparse_step", 1)))

        self.gguf_writer.add_layer_norm_rms_eps(hp.get("rms_norm_eps", 1e-6))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # aux-loss-free routing bias: on disk `mlp.experts.e_score_correction_bias`;
        # the loader presents it as `...e_score_correction.bias`. Move it under
        # `mlp.gate.` so the FFN_EXP_PROBS_B mapping (deepseek-v3 stem) matches; the
        # base strips the trailing .bias suffix during lookup.
        if "e_score_correction" in name:
            name = name.replace(".mlp.experts.", ".mlp.gate.")
            yield from super().modify_tensors(data_torch.squeeze(), name, bid)
            return

        # Routed experts are stored per-expert (gate_proj/up_proj/down_proj) — merge
        # each projection across all experts into one 3D tensor (qwen2moe/afmoe style).
        if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) < n_experts * 3:
                return

            for w_name in ["gate_proj", "up_proj", "down_proj"]:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                    datas.append(self._experts[bid][ename])
                    del self._experts[bid][ename]
                merged = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                yield from super().modify_tensors(merged, merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
