from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger

from .deepseek import DeepseekV2Model


@ModelBase.register("Glm4ForCausalLM", "Glm4vForConditionalGeneration")
@ModelBase.example("zai-org/GLM-4-9B-0414")
class Glm4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GLM4
    use_mrope = False
    partial_rotary_factor = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 0.5)
        if "mrope_section" in self.rope_parameters:
            self.use_mrope = True
            logger.info("Q/K weight will need to be permuted for M-RoPE")

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|user|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.partial_rotary_factor))

    @staticmethod
    def normal_to_neox(weights: Tensor, n_head: int, n_head_kv: int, head_dim: int, partial_rotary_factor: float) -> Tensor:
        orig_shape = weights.shape
        if len(orig_shape) == 1:
            weights = weights.unsqueeze(1)  # [out_dim, 1]
        if len(weights.shape) != 2:
            raise ValueError("Only 1D and 2D tensors are supported.")
        n_effective_heads = weights.shape[0] // head_dim
        if n_head_kv is not None and n_effective_heads != n_head:
            if n_effective_heads != n_head_kv:
                raise AssertionError(f"Mismatch in effective heads: computed {n_effective_heads}, expected {n_head} or {n_head_kv}")
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even.")
        reshaped = weights.reshape(n_effective_heads, head_dim, -1)
        rot_part = reshaped[:, :rotary_dim, :]
        non_rot_part = reshaped[:, rotary_dim:, :]
        permuted_rot = torch.cat((rot_part[:, ::2, :], rot_part[:, 1::2, :]), dim=1)
        combined = torch.cat((permuted_rot, non_rot_part), dim=1)
        result = combined.reshape(weights.shape)
        return result if len(orig_shape) != 1 else result.squeeze(1)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.use_mrope:
            n_head = self.hparams["num_attention_heads"]
            n_kv_head = self.hparams["num_key_value_heads"]
            n_embd = self.hparams["hidden_size"]
            head_dim = self.hparams.get("head_dim", n_embd // n_head)
            # because llama.cpp M-RoPE kernel only supports Neox ordering, we have to permute the weights here
            if name.endswith(("q_proj.weight", "q_proj.bias")):
                data_torch = Glm4Model.normal_to_neox(data_torch, n_head, n_head, head_dim, self.partial_rotary_factor)
            if name.endswith(("k_proj.weight", "k_proj.bias")):
                data_torch = Glm4Model.normal_to_neox(data_torch, n_head, n_kv_head, head_dim, self.partial_rotary_factor)
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("GlmOcrForConditionalGeneration")
@ModelBase.example("zai-org/GLM-OCR")
class GlmOCRModel(Glm4Model):
    model_arch = gguf.MODEL_ARCH.GLM4
    use_mrope = False
    partial_rotary_factor = 0.5

    # Note: GLM-OCR is the same as GLM4, but with an extra NextN/MTP prediction layer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GLM-OCR has num_hidden_layers + 1 actual layers (including NextN layer)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)


@ModelBase.register("Glm4MoeForCausalLM", "Glm4vMoeForConditionalGeneration")
@ModelBase.example("zai-org/GLM-4.5-Air")
class Glm4MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GLM4_MOE
    supports_mtp_export = True
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.no_mtp:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        key = next((k for k in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"] if k in hparams), None)
        type(self)._n_main_layers = hparams.get(key)
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_vocab(self):
        return self._set_vocab_glm()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = (
                self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            )
        self.gguf_writer.add_rope_dimension_count(
            int(rope_dim * self.rope_parameters.get("partial_rotary_factor", 0.5))
        )

        # MoE parameters - Use only routed expert count (shared experts handled separately)
        if (n_routed_experts := self.hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_shared_experts := self.hparams.get("n_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (first_k_dense_replace := self.hparams.get("first_k_dense_replace")) is not None:
            self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)

        # Expert gating function (sigmoid for GLM4_MOE)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # Routed scaling factor
        if (routed_scaling_factor := self.hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        # Normalise topk probabilities
        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        if not self.no_mtp and (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    _experts: list[dict[str, Tensor]] | None = None

    # note: unlike GLM4V non-MoE, we don't need to permute Q/K here since GLM4V_MOE uses Neox ordering already
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Handle main token embedding (but not layer-specific NextN embeddings)
        if name == "model.embed_tokens.weight" and ".layers." not in name:
            yield from super().modify_tensors(data_torch, "token_embd.weight", bid)
            return

        # Handle routed experts
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Glm4MoeLiteForCausalLM")
@ModelBase.example("zai-org/GLM-4.7-Flash")
class Glm4MoeLiteModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2
    skip_mtp = False
    supports_mtp_export = True
    _n_main_layers: int | None = None

    def set_vocab(self):
        return self._set_vocab_glm()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_hidden_layers = self.hparams["num_hidden_layers"]
        self.num_nextn_predict_layers = self.hparams.get("num_nextn_predict_layers", 0)
        self.skip_mtp = self.no_mtp or self.num_nextn_predict_layers == 0

        if self.skip_mtp:
            self.block_count = num_hidden_layers
        else:
            self.block_count = num_hidden_layers + self.num_nextn_predict_layers

        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        if self.skip_mtp:
            return

        self.gguf_writer.add_nextn_predict_layers(self.num_nextn_predict_layers)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item):
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        if cls._n_main_layers is not None:
            match = re.match(r"model\.layers\.(\d+)\.", name)
            is_mtp = match is not None and int(match.group(1)) >= cls._n_main_layers
            if is_mtp and cls.no_mtp:
                return None
            if cls.mtp_only and not is_mtp and name not in (
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            ):
                return None

        return name, gen

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"


@ModelBase.register("GlmMoeDsaForCausalLM")
@ModelBase.example("zai-org/GLM-5.2")
class GlmMoeDsaModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.GLM_DSA
    skip_mtp = False
    supports_mtp_export = True

    # Trunk layer count, stashed before indexing so the classmethod
    # filter_tensors can identify the appended NextN/MTP block (mirrors
    # HYV3Model / Step35Model).
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        # GLM-5.2 appends the NextN/MTP block past num_hidden_layers
        # (model.layers.78 -> blk.78 in the 79-block file).
        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        # --no-mtp: drop the appended NextN block entirely.
        if is_mtp and cls.no_mtp:
            return None
        # --mtp: keep ONLY NextN-block tensors plus the shared embeddings/
        # norm/lm_head (so the resulting GGUF carries just the draft head).
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_vocab(self):
        return self._set_vocab_glm()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        rope_dim = self.hparams["qk_rope_head_dim"]
        partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * partial_rotary_factor))

        # NextN/MTP prediction layers
        if not self.no_mtp and (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # DSA indexer parameters
        self.gguf_writer.add_indexer_head_count(self.hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(self.hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(self.hparams["index_topk"])
        if (indexer_types := self.hparams.get("indexer_types")) is not None:
            indexer_types = [t == "full" for t in indexer_types]
            self.gguf_writer.add_indexer_types(indexer_types)


@ModelBase.register("SolarOpenForCausalLM")
@ModelBase.example("upstage/Solar-Open-100B")
class SolarOpenModel(Glm4MoeModel):
    model_arch = gguf.MODEL_ARCH.GLM4_MOE
    supports_mtp_export = False

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab._set_special_token("eos", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|endoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("unk", tokenizer.get_added_vocab()["<unk>"])  # ty: ignore[unresolved-attribute]
        special_vocab._set_special_token("bos", tokenizer.get_added_vocab()["<|startoftext|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("Glm5NextForConditionalGeneration")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(TextModel):

    model_arch = gguf.MODEL_ARCH.GLM5_NEXT
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_nextn_layers = self.hparams.get("num_nextn_predict_layers", 0)
        self.skip_mtp = self.no_mtp or self.n_nextn_layers == 0

        if not self.skip_mtp:
            self.block_count += self.n_nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.hparams.pop("head_dim", None)

    def set_vocab(self):
        # requires transformers >= 5, tokpre hash-resolves to glm4
        return self._set_vocab_glm()

    def index_tensors(self, remote_hf_model_id: str | None = None):
        hp = self.hparams.get("text_config", self.hparams)
        type(self)._n_main_layers = hp["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        assert cls._n_main_layers is not None
        m = re.match(r"model\.layers\.(\d+)\.", name)
        is_mtp = m is not None and int(m.group(1)) >= cls._n_main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        layer_types = hp["layer_types"]
        n_kv_heads = [0 if t == "linear_attention" else 1 for t in layer_types]
        assert len(n_kv_heads) == hp["num_hidden_layers"]
        # Pad to block_count
        n_kv_heads += [1] * (self.block_count - len(n_kv_heads))
        self.gguf_writer.add_head_count_kv(n_kv_heads)
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_layer_norm_eps(1e-6)

        if not self.skip_mtp:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn_layers)

        # KDA
        lin = hp["linear_attn_config"]
        assert lin["num_heads"] == hp["num_attention_heads"]
        self.gguf_writer.add_ssm_conv_kernel(lin["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(lin["head_dim"])
        if (lb := lin.get("gate_lower_bound")) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(lb)

        # MLA (nope only)
        assert hp.get("mla_use_nope") and hp["qk_rope_head_dim"] == 0, "expected nope-only MLA"
        kv_lora_rank = hp["kv_lora_rank"]
        qk_rope = hp["qk_rope_head_dim"]
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + qk_rope)
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])

        # DSA indexer with k-pool compression
        self.gguf_writer.add_indexer_head_count(hp["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_kpool(hp["index_kpool"])
        self.gguf_writer.add_indexer_kpool_select_tail(hp.get("index_kpool_always_select_tail", True))
        self.gguf_writer.add_indexer_index_share_mtp(hp.get("index_share_for_mtp_iteration", False))
        if (indexer_types := hp.get("indexer_types")) is not None:
            self.gguf_writer.add_indexer_types([t == "full" for t in indexer_types])

        # mHC
        assert hp.get("mhc", True)
        self.gguf_writer.add_hyper_connection_count(hp["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hp["hc_eps"])

        # MoE
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])
        if (limit := hp.get("swiglu_limit")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp([limit] * self.block_count)
            self.gguf_writer.add_swiglu_clamp_shexp([limit] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight" and self.hparams.get("tie_word_embeddings", False):
            return

        # routed experts
        if ".mlp.experts." in name:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) < n_experts * 3:
                return
            for w_name in ("down_proj", "gate_proj", "up_proj"):
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                    datas.append(self._experts[bid].pop(ename))
                merged = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                yield from super().modify_tensors(torch.stack(datas, dim=0), merged, bid)
            return

        # MLA absorption
        if name.endswith("kv_b_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            yield from super().modify_tensors(k_b.transpose(1, 2), name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from super().modify_tensors(v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        # KDA conv1d
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
            elif data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
            else:
                raise ValueError(f"unexpected conv1d rank {data_torch.ndim} for {name}")
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        if name.endswith(".A_log"):
            n_head = self.hparams["num_attention_heads"]
            data_torch = -torch.exp(data_torch.float().flatten()[:n_head])

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        if re.search(r"\.(hc_(?:attn|ffn)_(?:fn|base|scale)|index_kpool_compress_(?:ape|gate))$", name):
            yield self.map_tensor_name(name) + ".weight", data_torch
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # keep the small mHC / gating parameters exact
        exact_keys = ("hc_attn_", "hc_ffn_", "indexer_compressor_", "ssm_a", "ssm_dt", "exp_probs_b")
        if new_name.startswith(("blk.", "output_hc")) and any(k in new_name for k in exact_keys):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)
        if not self.mtp_only or not from_dir:
            return
        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftover = [k for d in self._experts for k in d.keys()]
            if leftover:
                raise ValueError(f"Unprocessed experts: {leftover}")
