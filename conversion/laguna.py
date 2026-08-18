from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("LagunaForCausalLM")
class LagunaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LAGUNA
    _experts: list[dict] | None = None
    _gate_types: list[str] | None = None

    # --- vocab ---------------------------------------------------------------

    def set_vocab(self) -> None:
        self._set_vocab_gpt2()

        # Some Laguna releases wrap the chat template in tokenizer_config.json as
        # "{% include 'chat_template.jinja' %}", which SpecialVocab embeds verbatim
        # and llama.cpp's jinja engine cannot process. Prefer the resolved template
        # from the chat_template.jinja file so the GGUF is self-contained.
        tmpl_file = self.dir_model / "chat_template.jinja"
        if tmpl_file.is_file():
            self.gguf_writer.add_chat_template(tmpl_file.read_text(encoding="utf-8"))
            logger.info("gguf: embedded resolved chat_template.jinja (overriding include directive)")

        # eos_token_id is a list [2, 24]: token 2 (EOS, also BOS) and token 24
        # (</assistant>, the turn-end). _set_vocab_gpt2 only records the scalar
        # eos, so register the extra id as eot; llama.cpp folds eot into its EOG
        # set, so the model halts on </assistant> natively.
        eos_ids = self.hparams.get("eos_token_id")
        if isinstance(eos_ids, list):
            bos_id = self.hparams.get("bos_token_id")
            extra = [e for e in eos_ids if e != bos_id]
            if extra:
                self.gguf_writer.add_eot_token_id(extra[0])
                logger.info(f"gguf: registered eot_token_id={extra[0]} from eos list {eos_ids}")

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        # </assistant> is the assistant turn-end (registered as eot below). The
        # HF tokenizer flags it special=false, so the base classifies it as
        # USER_DEFINED and llama.cpp renders its text into generated content,
        # leaking "</assistant>" and breaking response parsing. It is a control
        # marker, so promote it to CONTROL: llama.cpp then treats it as
        # end-of-generation and suppresses its text.
        tokens, toktypes, tokpre = super().get_vocab_base()
        for i, tok in enumerate(tokens):
            if tok == "</assistant>":
                toktypes[i] = gguf.TokenType.CONTROL
                logger.info(f"gguf: marked </assistant> (id {i}) as CONTROL token")
        return tokens, toktypes, tokpre

    # --- hparams -------------------------------------------------------------

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams

        # super() does not emit vocab_size for the gpt2 vocab path; head_count is
        # overridden with a per-layer array (XS.2 varies heads per layer via
        # num_attention_heads_per_layer; M.1 is uniform and omits it).
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        per_layer_heads = hparams.get("num_attention_heads_per_layer")
        if not per_layer_heads:
            per_layer_heads = [hparams["num_attention_heads"]] * hparams["num_hidden_layers"]
        assert len(per_layer_heads) == hparams["num_hidden_layers"], (
            f"num_attention_heads_per_layer length {len(per_layer_heads)} != "
            f"num_hidden_layers {hparams['num_hidden_layers']}"
        )
        self.gguf_writer.add_head_count(per_layer_heads)

        # Resolve + validate the attention gate type now so an inconsistent
        # `gating` field fails at conversion time. See _attn_gate_types.
        self._attn_gate_types()

        # SWA window size (M.1 has none -> key omitted, swa_type stays NONE).
        sliding_window = hparams.get("sliding_window") or 0
        if sliding_window > 0:
            self.gguf_writer.add_sliding_window(sliding_window)

        # MoE (expert_count / expert_used_count come from super().set_gguf_parameters())
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_weights_norm(True)  # HF reference always sum-normalises after top-k
        self.gguf_writer.add_expert_weights_scale(float(hparams["moe_routed_scaling_factor"]))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Leading dense layers (XS.2 has 1, M.1 has 3) before the MoE layers.
        mlp_layer_types: list[str] = hparams["mlp_layer_types"]
        leading_dense = 0
        for t in mlp_layer_types:
            if t == "dense":
                leading_dense += 1
            else:
                break
        self.gguf_writer.add_leading_dense_block_count(leading_dense)

        # Per-layer-type RoPE dimension count (partial rotary). base emits
        # rope_freq_base(_swa) and the YaRN params from self.rope_parameters.
        head_dim = hparams["head_dim"]
        full_rope = self.rope_parameters["full_attention"]
        self.gguf_writer.add_rope_dimension_count(
            int(head_dim * float(full_rope.get("partial_rotary_factor", 1.0))))
        swa_rope = self.rope_parameters.get("sliding_attention")
        if swa_rope is not None:
            self.gguf_writer.add_rope_dimension_count_swa(
                int(head_dim * float(swa_rope.get("partial_rotary_factor", 1.0))))

    def _attn_gate_types(self) -> list[str]:
        """Per-layer attention output gate type: "per_head" or "per_element".

        `gating_types` (per layer) is authoritative when present; otherwise the
        scalar `gating` field is used (the "per-element"/"per-head" string, or
        the legacy boolean True == per-head, as in Laguna-XS.2).

        Fails loudly when the model is per-element but the `gating` field does
        not declare that as a string: runtimes that key off `gating` (vLLM,
        transformers) ignore gating_types and read a bare boolean True as
        per-head, silently corrupting the model. Surfacing it here keeps a
        broken checkpoint from being packaged as if it were fine.
        """
        if self._gate_types is not None:
            return self._gate_types
        hparams = self.hparams
        n_layer = hparams["num_hidden_layers"]
        gating = hparams.get("gating")
        gating_types = hparams.get("gating_types")

        def _norm(t: object) -> str:
            sval = str(t).replace("-", "_")
            if sval in ("per_element", "per_head"):
                return sval
            raise ValueError(f"Laguna: unrecognised attention gate type {t!r}")

        if gating_types:
            assert len(gating_types) == n_layer, (
                f"gating_types length {len(gating_types)} != num_hidden_layers {n_layer}")
            types = [_norm(t) for t in gating_types]
        elif isinstance(gating, str):
            types = [_norm(gating)] * n_layer
        elif gating is True:
            types = ["per_head"] * n_layer
        else:
            raise ValueError(
                f"Laguna: cannot determine attention gate type "
                f"(gating={gating!r}, gating_types={gating_types!r})")

        if any(t == "per_element" for t in types) and not (
                isinstance(gating, str) and _norm(gating) == "per_element"):
            raise ValueError(
                f"Laguna config declares a per-element attention gate but "
                f"`gating`={gating!r} is not the string \"per-element\". Runtimes that "
                f"read `gating` (vLLM, transformers) will mis-handle this checkpoint as "
                f"per-head. Set gating=\"per-element\" in the source config.")

        self._gate_types = types
        return types

    # --- tensor handling -----------------------------------------------------

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Per-expert MoE weights: model.layers.{bid}.mlp.experts.{xid}.{w}.weight.
        # Only the NUMBERED per-expert weights are stacked; the router bias
        # (mlp.experts.e_score_correction_bias) takes the normal mapping path.
        if re.search(r"mlp\.experts\.\d+\.", name):
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            needed = [f"model.layers.{bid}.mlp.experts.{x}.{w}.weight"
                      for x in range(n_experts) for w in ("gate_proj", "up_proj", "down_proj")]
            if all(e in self._experts[bid] for e in needed):
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas = [self._experts[bid][f"model.layers.{bid}.mlp.experts.{x}.{w_name}.weight"]
                             for x in range(n_experts)]
                    stacked = torch.stack(datas, dim=0)
                    merged = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from TextModel.modify_tensors(self, stacked, merged, bid)
                self._experts[bid].clear()
                return
            return
        # Cross-check the gate projection width against the declared gate type;
        # a mismatch means the weights and config disagree -> fail, do not guess.
        if bid is not None and name.endswith("self_attn.g_proj.weight"):
            heads = (self.hparams.get("num_attention_heads_per_layer")
                     or [self.hparams["num_attention_heads"]] * self.hparams["num_hidden_layers"])
            n_head = heads[bid]
            head_dim = self.hparams["head_dim"]
            gate_type = self._attn_gate_types()[bid]
            expected = n_head * head_dim if gate_type == "per_element" else n_head
            out_features = int(data_torch.shape[0])
            if out_features != expected:
                raise ValueError(
                    f"Laguna layer {bid}: g_proj output width {out_features} contradicts the "
                    f"declared {gate_type} gate (expected {expected}); weights and config disagree.")

        yield from TextModel.modify_tensors(self, data_torch, name, bid)
