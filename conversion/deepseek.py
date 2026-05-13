from __future__ import annotations

import re

from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger

from .qwen import QwenModel


@ModelBase.register("DeepseekOCRForCausalLM")
class DeepseekOCRVisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DEEPSEEKOCR)
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
        vision_config.update(vision_config['width']['clip-l-14-224'])
        vision_config['hidden_size'] = vision_config['width']
        vision_config['num_heads'] = vision_config['heads']
        vision_config['intermediate_size'] = vision_config['heads'] * 4

        return vision_config

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".embeddings." in name or 'pos_embed' in name:
            return gguf.GGMLQuantizationType.F32
        if ".rel_pos_h" in name or '.rel_pos_w' in name:
            return gguf.GGMLQuantizationType.F32
        if ".neck." in name or ".net_" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

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
        if self.origin_hf_arch == "DeepseekOCRForCausalLM":
            self.model_arch = gguf.MODEL_ARCH.DEEPSEEK2OCR
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            # default jinja template
            self.gguf_writer.add_chat_template("{% for m in messages %}{{m['content']}}{% endfor %}")

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
