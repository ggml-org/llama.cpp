from __future__ import annotations

import json
import re
from pathlib import Path

from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, MmprojModel, ModelBase, TextModel, gguf, logger

from .qwen import QwenModel


@ModelBase.register("DeepseekOCRForCausalLM", "UnlimitedOCRForCausalLM")
class DeepseekOCRVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_projector_type = gguf.VisionProjectorType.DEEPSEEKOCR

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(self.clip_projector_type)
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)
        # calculate proj_scale_factor (used by tinygemma3 test model)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        n_per_side = int(image_seq_length ** 0.5)
        image_size = self.hparams["image_size"]
        patch_size = self.hparams["patch_size"]
        proj_scale_factor = (image_size // patch_size) // n_per_side
        if proj_scale_factor > 0 and proj_scale_factor != 4:
            # we only need to write this if it's not the default value
            # in this case, we are converting a test model
            self.gguf_writer.add_vision_projector_scale_factor(proj_scale_factor)
        # @bluebread: there's no window_size in config but just add it here anyway
        self.gguf_writer.add_vision_window_size(self.hparams.get("window_size", 14))

        # SAM configuration
        sam_hparams = hparams['sam']
        self.gguf_writer.add_vision_sam_layers_count(sam_hparams['layers'])
        self.gguf_writer.add_vision_sam_embedding_length(sam_hparams['width'])
        self.gguf_writer.add_vision_sam_head_count(sam_hparams['heads'])

    def get_vision_config(self) -> dict[str, Any]:
        vision_config: dict[str, Any] | None = self.global_config.get("vision_config")

        if not vision_config:
            raise ValueError("DeepseekOCR model requires 'vision_config' in the model configuration, but it was not found")

        vision_config['sam'] = vision_config['width']['sam_vit_b']
        if vision_config['width'].get('clip-l-14-224') is not None:
            vision_config.update(vision_config['width']['clip-l-14-224'])
        if isinstance(vision_config['width'], int):
            vision_config['hidden_size'] = vision_config['width']
        if vision_config.get('heads') is not None:
            vision_config['num_heads'] = vision_config['heads']
            vision_config['intermediate_size'] = vision_config['heads'] * 4

        return vision_config

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        for nq_name in ('.embeddings.', 'pos_embed', '.rel_pos_h', '.rel_pos_w', '.neck.', '.net_'):
            if nq_name in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("view_seperator"):
            data_torch = data_torch.unsqueeze(0)
        yield from super().modify_tensors(data_torch, name, bid)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Only process vision-related tensors, skip language model tensors
        # Vision components: sam_model, vision_model, projector, image_newline, view_seperator
        # Language model components to skip: lm_head, embed_tokens, layers, norm
        if name.startswith(("lm_head.", "model.embed_tokens.", "model.layers.", "model.norm.")):
            return None

        if name.endswith("pos_embed") or name.endswith("rel_pos_h") or name.endswith("rel_pos_w"):
            name += ".weight"

        return super().filter_tensors((name, gen))


@ModelBase.register("DeepseekOCR2ForCausalLM")
class DeepseekOCR2VisionModel(DeepseekOCRVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_projector_type = gguf.VisionProjectorType.DEEPSEEKOCR2

    def set_gguf_parameters(self):
        # the vision tower's qwen2 encoder is built from fixed defaults,
        # see build_qwen2_decoder_as_encoder() in deepencoderv2.py
        if self.hparams.get("patch_size") is None:
            self.hparams["patch_size"] = 16
        if self.hparams.get("intermediate_size") is None:
            self.hparams["intermediate_size"] = 4864
        if self.hparams.get("num_attention_heads") is None:
            self.hparams["num_attention_heads"] = 14
        super().set_gguf_parameters()
        # qwen2 encoder is GQA: 14 Q heads, 2 KV heads
        self.gguf_writer.add_vision_head_count_kv(2)

    def get_vision_config(self) -> dict[str, Any]:
        vision_config = super().get_vision_config()
        vision_config['hidden_size'] = vision_config['width']['qwen2-0-5b']['dim']
        if vision_config.get('layers') is None:
            vision_config['layers'] = 24
        return vision_config


@ModelBase.register("DeepseekForCausalLM")
class DeepseekModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
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


@ModelBase.register(
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekOCRForCausalLM",
    "UnlimitedOCRForCausalLM",
    "KimiVLForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
    "YoutuForCausalLM",
    "YoutuVLForConditionalGeneration",
)
class DeepseekV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    # TODO @ngxson : remove this when we support MTP for deepseek models
    skip_mtp = True

    merge_expert = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams: dict = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

        # special handling for Deepseek OCR
        if self.origin_hf_arch in ("DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM", "UnlimitedOCRForCausalLM"):
            self.model_arch = gguf.MODEL_ARCH.DEEPSEEK2OCR
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            # default jinja template
            self.gguf_writer.add_chat_template("{% for m in messages %}{{m['content']}}{% endfor %}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # DeepSeek-OCR vision encoder (SAM + DeepSeek-OCR-2 qwen2 tower)
        if "sam_model" in name or "qwen2_model" in name:
            return None
        return super().filter_tensors(item)

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
            return
        except Exception:
            pass

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokpre = self.get_vocab_base_pre(tokenizer)

        if tokpre == "kimi-k2":
            # Build merges list using the approach similar to HunYuanMoE
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

            # Build token list
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            raise NotImplementedError(f"Deepseek pre-tokenizer {tokpre!r} is not supported yet!")

    def set_gguf_parameters(self):
        is_ocr = (self.model_arch == gguf.MODEL_ARCH.DEEPSEEK2OCR)

        if is_ocr:
            self.hparams['rope_theta'] = self.hparams.get('rope_theta', 10000.0)
        else:
            # note: deepseek2 using MLA converts into MQA (ie: GQA with 1 group)
            self.hparams["num_key_value_heads"] = 1

        self.hparams['rms_norm_eps'] = self.hparams.get('rms_norm_eps', 1e-6)

        super().set_gguf_parameters()
        hparams = self.hparams

        # first_k_dense_replace: number of leading layers using dense FFN instead of MoE
        # For non-MoE models (like Youtu), set to n_layer to use dense FFN for all layers
        # For MoE models (like DeepSeek-V2), this is the number of leading non-MoE layers
        has_moe = hparams.get("n_routed_experts") is not None
        first_k_dense_replace = hparams.get("first_k_dense_replace")
        if first_k_dense_replace is None:
            # Default: if no MoE, all layers are dense; if MoE, none are dense
            first_k_dense_replace = hparams["num_hidden_layers"] if not has_moe else 0
        self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)
        kv_lora_rank = hparams.get("kv_lora_rank", 512)
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        # note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
        if not is_ocr:
            self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
            self.gguf_writer.add_key_length(kv_lora_rank + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length(kv_lora_rank)
            self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        # MoE parameters (required by C++ code for DEEPSEEK2 arch)
        # For non-MoE models like Youtu, use intermediate_size as expert_feed_forward_length
        moe_intermediate_size = self.find_hparam(["moe_intermediate_size", "intermediate_size"], optional=False)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)

        if (n_routed_experts := hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)

        # expert_shared_count is required by C++ code, default to 0 for non-MoE models
        n_shared_experts = hparams.get("n_shared_experts", 0)
        self.gguf_writer.add_expert_shared_count(n_shared_experts)

        # When not set, C++ code will use scale_w = false to skip the no-op scaling
        if (routed_scaling_factor := hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        if (norm_topk_prob := hparams.get("norm_topk_prob")) is not None and norm_topk_prob:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        # Unlimited-OCR sliding window; written for metadata, the decoder ignores it (full MHA)
        if is_ocr:
            sliding_window = hparams.get("sliding_window_size") or hparams.get("sliding_window")
            if sliding_window:
                self.gguf_writer.add_sliding_window(sliding_window)

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            # note: for legacy reasons, this is not consistent with the other usages of self.gguf_writer.add_rope_scaling_yarn_log_mul
            # ref https://github.com/ggml-org/llama.cpp/pull/17945
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # skip lm_head.weight if tie_word_embeddings is True
        if self.hparams.get("tie_word_embeddings", False):
            if name == "lm_head.weight" or name == "model.lm_head.weight":
                logger.info("Skipping tied output layer 'lm_head.weight' (will use token_embd.weight)")
                return

        # skip Multi-Token Prediction (MTP) layers
        if self.skip_mtp:
            block_count = self.hparams["num_hidden_layers"]
            match = re.match(r"model.layers.(\d+)", name)
            if match and int(match.group(1)) >= block_count:
                return

        # process the experts separately
        if self.merge_expert and name.find("mlp.experts") != -1:
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

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("DeepseekV32ForCausalLM")
class DeepseekV32Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK32
    skip_mtp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        assert getattr(tokenizer, "add_bos_token", False), "Change value of add_bos_token to true in tokenizer_config.json file."
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # DSA indexer parameters
        self.gguf_writer.add_indexer_head_count(self.hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(self.hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(self.hparams["index_topk"])


@ModelBase.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK4
    _skipped_mtp_tensors = 0

    def __init__(self, *args, **kwargs):
        type(self)._skipped_mtp_tensors = 0
        super().__init__(*args, **kwargs)

        with open(self.dir_model / "config.json", "r", encoding="utf-8") as f:
            raw_hparams = json.load(f)
        for key, value in raw_hparams.items():
            self.hparams.setdefault(key, value)

        self.block_count = self.hparams["num_hidden_layers"]
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self._dsv4_fp8_dequantized: set[str] = set()
        self._dsv4_bf16_tensors: set[str] = set()
        self._dsv4_f32_tensors: set[str] = set()
        self._dsv4_mxfp4_generated = False
        self._collect_source_dtypes()

        if type(self)._skipped_mtp_tensors:
            logger.info("Skipping %d DeepSeek-V4 MTP tensor(s) for conversion v0", type(self)._skipped_mtp_tensors)

        # add a default chat template; if the model has a built-in template, it will be overridden later
        template_path = Path(__file__).parent.parent / "models" / "templates" / "deepseek-ai-DeepSeek-V4.jinja"
        if template_path.is_file():
            with open(template_path, "r", encoding="utf-8") as f:
                self.gguf_writer.add_chat_template(f.read())

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if name.startswith("mtp."):
            cls._skipped_mtp_tensors += 1
            return None
        return super().filter_tensors(item)

    @staticmethod
    def _float8_dtypes() -> tuple[torch.dtype, ...]:
        return tuple(
            dtype for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            ) if dtype is not None
        )

    @staticmethod
    def _e8m0_to_float(scale: Tensor) -> Tensor:
        torch_float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if torch_float8_e8m0 is not None and scale.dtype == torch_float8_e8m0:
            return scale.float()

        bits = scale.view(torch.uint8).float()
        return torch.exp2(bits - 127.0)

    def _collect_source_dtypes(self) -> None:
        for name, gen in self.model_tensors.items():
            dtype = gen().dtype
            if dtype == torch.bfloat16:
                self._dsv4_bf16_tensors.add(name)
            elif dtype == torch.float32:
                self._dsv4_f32_tensors.add(name)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_swiglu_clamp_exp([hparams["swiglu_limit"]] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([hparams["swiglu_limit"]] * self.block_count)

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        self.gguf_writer.add_attention_output_group_count(hparams["o_groups"])
        self.gguf_writer.add_attention_output_lora_rank(hparams["o_lora_rank"])
        self.gguf_writer.add_attention_compress_ratios(hparams["compress_ratios"])
        self.gguf_writer.add_attention_compress_rope_freq_base(hparams["compress_rope_theta"])
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])
        self.gguf_writer.add_hash_layer_count(hparams["num_hash_layers"])

    def dequant_model(self):
        fp8_dtypes = self._float8_dtypes()
        tensors_to_remove: list[str] = []

        def dequant_fp8_weight(weight: Tensor, scale: Tensor) -> Tensor:
            out_features, in_features = weight.shape
            scale_f = self._e8m0_to_float(scale)
            scale_f = scale_f.repeat_interleave(128, 0)[:out_features]
            scale_f = scale_f.repeat_interleave(128, 1)[:, :in_features]
            return weight.float() * scale_f

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".scale"):
                continue
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name not in self.model_tensors:
                continue

            weight = self.model_tensors[weight_name]
            scale = self.model_tensors[name]
            if weight().dtype not in fp8_dtypes:
                continue

            self.model_tensors[weight_name] = lambda w=weight, s=scale: dequant_fp8_weight(w(), s())
            self._dsv4_fp8_dequantized.add(weight_name)
            tensors_to_remove.append(name)

        for name in tensors_to_remove:
            del self.model_tensors[name]

    @staticmethod
    def _pack_mxfp4_blocks(weight: Tensor, scale: Tensor) -> np.ndarray:
        packed = weight.contiguous().view(torch.uint8)
        scale_u8 = scale.contiguous().view(torch.uint8)

        out_features, packed_cols = packed.shape
        logical_cols = packed_cols * 2
        if logical_cols % 32 != 0:
            raise ValueError(f"MXFP4 source row has {logical_cols} values, expected a multiple of 32")

        n_blocks = logical_cols // 32
        if tuple(scale_u8.shape) != (out_features, n_blocks):
            raise ValueError(f"MXFP4 scale shape {tuple(scale_u8.shape)} does not match {(out_features, n_blocks)}")

        src = packed.reshape(out_features, n_blocks, 16)
        low = src & 0x0F
        high = (src >> 4) & 0x0F

        # The safetensors bytes store adjacent values as low/high nibbles.
        # ggml MXFP4 blocks store values 0..15 in low nibbles and 16..31 in high nibbles.
        vals = torch.stack((low, high), dim=-1).reshape(out_features, n_blocks, 32)
        qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
        raw = torch.cat((scale_u8.unsqueeze(-1), qs.to(torch.uint8)), dim=-1)
        return raw.reshape(out_features, n_blocks * 17).cpu().numpy()

    def _write_mxfp4_expert_tensor(self, bid: int, proj: str, tensor_key: gguf.MODEL_TENSOR) -> list[str]:
        n_experts = self.hparams["n_routed_experts"]
        data: np.ndarray | None = None
        consumed: list[str] = []

        for eid in range(n_experts):
            weight_name = f"layers.{bid}.ffn.experts.{eid}.{proj}.weight"
            scale_name = f"layers.{bid}.ffn.experts.{eid}.{proj}.scale"
            if weight_name not in self.model_tensors or scale_name not in self.model_tensors:
                raise KeyError(f"Missing routed expert tensors for {weight_name}")

            weight = LazyTorchTensor.to_eager(self.model_tensors[weight_name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
            packed = self._pack_mxfp4_blocks(weight, scale)
            if data is None:
                data = np.empty((n_experts, *packed.shape), dtype=packed.dtype)
            data[eid] = packed
            consumed.extend((weight_name, scale_name))

        assert data is not None
        new_name = self.format_tensor_name(tensor_key, bid)
        shape = gguf.quant_shape_from_byte_shape(data.shape, gguf.GGMLQuantizationType.MXFP4)
        logger.info(f"{new_name}: repacked routed experts to MXFP4, shape = {{{', '.join(str(n) for n in reversed(shape))}}}")
        self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

        return consumed

    def _write_hash_routing_tensors(self) -> list[str]:
        consumed: list[str] = []

        for bid in range(self.hparams["num_hash_layers"]):
            name = f"layers.{bid}.ffn.gate.tid2eid"
            if name not in self.model_tensors:
                raise KeyError(f"Missing hash routing tensor {name}")

            data_torch = LazyTorchTensor.to_eager(self.model_tensors[name]())
            data = data_torch.to(torch.int32).cpu().numpy()
            new_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_TID2EID, bid, ".weight")
            logger.info(f"{new_name}: converted hash routing table to I32, shape = {{{', '.join(str(n) for n in reversed(data.shape))}}}")
            self.gguf_writer.add_tensor(new_name, data)
            consumed.append(name)

        return consumed

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._dsv4_mxfp4_generated:
            return ()

        consumed: list[str] = self._write_hash_routing_tensors()
        for bid in range(self.block_count):
            consumed.extend(self._write_mxfp4_expert_tensor(bid, "w1", gguf.MODEL_TENSOR.FFN_GATE_EXP))
            consumed.extend(self._write_mxfp4_expert_tensor(bid, "w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP))
            consumed.extend(self._write_mxfp4_expert_tensor(bid, "w3", gguf.MODEL_TENSOR.FFN_UP_EXP))

        for name in consumed:
            del self.model_tensors[name]

        self._dsv4_mxfp4_generated = True
        return ()

    def _format_dsv4_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> str:
        return self.format_tensor_name(key, bid, suffix)

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        root_map: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "embed.weight": (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "norm.weight": (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "head.weight": (gguf.MODEL_TENSOR.OUTPUT, ".weight"),
            "hc_head_fn": (gguf.MODEL_TENSOR.HC_HEAD_FN, ".weight"),
            "hc_head_base": (gguf.MODEL_TENSOR.HC_HEAD_BASE, ".weight"),
            "hc_head_scale": (gguf.MODEL_TENSOR.HC_HEAD_SCALE, ".weight"),
        }
        if name in root_map:
            return root_map[name]

        match = re.match(r"layers\.(\d+)\.(.+)$", name)
        if match is None:
            raise ValueError(f"Unsupported DeepSeek-V4 tensor {name!r}")

        layer = int(match.group(1))
        if bid != layer:
            raise ValueError(f"Tensor {name!r} parsed bid {bid} but layer name has {layer}")

        layer_map: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "hc_attn_fn": (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "hc_attn_base": (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "hc_attn_scale": (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "hc_ffn_fn": (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "hc_ffn_base": (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "hc_ffn_scale": (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            "attn.attn_sink": (gguf.MODEL_TENSOR.ATTN_SINKS, ".weight"),
            "attn.wq_a.weight": (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "attn.wq_b.weight": (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "attn.q_norm.weight": (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "attn.wkv.weight": (gguf.MODEL_TENSOR.ATTN_KV, ".weight"),
            "attn.kv_norm.weight": (gguf.MODEL_TENSOR.ATTN_KV_NORM, ".weight"),
            "attn.wo_a.weight": (gguf.MODEL_TENSOR.ATTN_OUT_A, ".weight"),
            "attn.wo_b.weight": (gguf.MODEL_TENSOR.ATTN_OUT_B, ".weight"),
            "attn.compressor.ape": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE, ".weight"),
            "attn.compressor.wkv.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WKV, ".weight"),
            "attn.compressor.wgate.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WGATE, ".weight"),
            "attn.compressor.norm.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_NORM, ".weight"),
            "attn.indexer.wq_b.weight": (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "attn.indexer.weights_proj.weight": (gguf.MODEL_TENSOR.INDEXER_PROJ, ".weight"),
            "attn.indexer.compressor.ape": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE, ".weight"),
            "attn.indexer.compressor.wkv.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WKV, ".weight"),
            "attn.indexer.compressor.wgate.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_WGATE, ".weight"),
            "attn.indexer.compressor.norm.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_NORM, ".weight"),
            "attn_norm.weight": (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "ffn_norm.weight": (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
            "ffn.gate.weight": (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            "ffn.gate.bias": (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "ffn.gate.tid2eid": (gguf.MODEL_TENSOR.FFN_GATE_TID2EID, ".weight"),
            "ffn.shared_experts.w1.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "ffn.shared_experts.w2.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            "ffn.shared_experts.w3.weight": (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
        }

        tensor_name = match.group(2)
        if tensor_name in layer_map:
            return layer_map[tensor_name]

        if re.match(r"ffn\.experts\.\d+\.w[123]\.(weight|scale)$", tensor_name):
            return gguf.MODEL_TENSOR.FFN_GATE_EXP, ".weight"

        raise ValueError(f"Unsupported DeepSeek-V4 tensor {name!r}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if re.match(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$", name):
            return []

        tensor_key, suffix = self._map_dsv4_tensor_name(name, bid)
        if tensor_key == gguf.MODEL_TENSOR.FFN_GATE_TID2EID:
            return []

        return [(self._format_dsv4_tensor_name(tensor_key, bid, suffix), data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        del new_name, bid  # unused

        if name in self._dsv4_fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.Q8_0
        if name in self._dsv4_f32_tensors:
            return gguf.GGMLQuantizationType.F32
        if name in self._dsv4_bf16_tensors and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16

        return False

    def prepare_tensors(self):
        super().prepare_tensors()
        self._is_mxfp4 = True
        self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE
