from __future__ import annotations

import math
import re

import torch

from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf

from .deepseek import DeepseekV2Model


@ModelBase.register("Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration", "Dots3NoteTextForCausalLM")
class Dots3NoteModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DOTS3NOTE
    skip_mtp = False
    supports_mtp_export = True

    # trunk layer count, stashed before indexing for filter_tensors (mirrors DeepseekV32Model)
    _n_main_layers: int | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hparams = self.hparams

        # config file doesn't specify MTP block, detect it from model weight
        self.n_nextn = 1 if "model.mtp.embed_tokens.weight" in self.model_tensors else 0
        if self.n_nextn:
            self.block_count += self.n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.layer_types = hparams["layer_types"]
        if len(self.layer_types) < hparams["num_hidden_layers"]:
            raise ValueError("layer_types is shorter than num_hidden_layers")

        if hparams.get("use_dsa", True) is not True:
            raise ValueError("dots3-note conversion requires use_dsa=true")
        if hparams.get("normalization", "RMSNorm") != "RMSNorm" or hparams.get("final_norm", "RMSNorm") != "RMSNorm":
            raise ValueError("dots3-note conversion only supports RMSNorm")
        if hparams.get("k_rope_only_layernorm", True) is not True:
            raise ValueError("dots3-note conversion requires k_rope_only_layernorm=true")
        if hparams.get("topk_method", "noaux_tc") != "noaux_tc" or hparams.get("scoring_func") != "sigmoid":
            raise ValueError("dots3-note conversion only supports noaux_tc/sigmoid expert gating")
        if hparams.get("n_group", 1) != 1 or hparams.get("topk_group", 1) != 1:
            raise ValueError("dots3-note conversion does not support grouped expert routing")
        if hparams.get("use_dynamic_rsf", False) or hparams.get("moe_gating_fp32", False):
            raise ValueError("dots3-note conversion does not support use_dynamic_rsf/moe_gating_fp32")
        for key in ("attention_gate_type", "swa_attention_gate_type"):
            if hparams.get(key, "headwise") != "headwise":
                raise ValueError(f"dots3-note conversion only supports headwise attention gate, got {key}={hparams.get(key)!r}")
        if hparams["swa_qk_nope_head_dim"] + hparams["swa_qk_rope_head_dim"] != hparams.get("swa_head_dim", 256):
            raise ValueError("swa_head_dim must equal swa_qk_nope_head_dim + swa_qk_rope_head_dim")
        if hparams["swa_qk_rope_head_dim"] != hparams["qk_rope_head_dim"]:
            # both layer kinds share a single rope_dimension_count
            raise ValueError("swa_qk_rope_head_dim must match qk_rope_head_dim")

        self.apply_lora_rescale = hparams.get("apply_mla_qkv_lora_rescale", False)

    def _is_swa_layer(self, bid: int) -> bool:
        if bid >= self.hparams["num_hidden_layers"]:
            # note: the NextN/MTP block uses the sliding-attention MLA
            return True
        return self.layer_types[bid] == "sliding_attention"

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab._set_special_token("eot", tokenizer.get_added_vocab()["<|endofassistant|>"])  # ty: ignore[unresolved-attribute]
        special_vocab.add_to_gguf(self.gguf_writer)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem
        if name.startswith(("vision_encoder.", "audio_encoder.")):
            return None

        assert cls._n_main_layers is not None
        is_mtp = name.startswith("model.mtp.") or \
            ((m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers)

        # --no-mtp: drop the NextN/MTP block; --mtp: keep only that block plus the shared embeddings/norm/lm_head
        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def set_gguf_parameters(self):
        hparams = self.hparams

        # head_count is a per-layer array because the two layer kinds have different head counts
        n_layer = hparams["num_hidden_layers"]
        hparams["num_attention_heads"] = [
            hparams["swa_num_attention_heads"] if self._is_swa_layer(il) else hparams["num_attention_heads"]
            for il in range(self.block_count)
        ]

        # prevent the base class from emitting key/value_length from the unused head_dim
        hparams.pop("head_dim", None)

        super().set_gguf_parameters()

        # MLA geometry of the sliding-window layers (rope.freq_base_swa is emitted by the base class)
        swa_kv_lora_rank = hparams["swa_kv_lora_rank"]
        self.gguf_writer.add_sliding_window(hparams["sliding_window_size"])
        self.gguf_writer.add_sliding_window_pattern([self._is_swa_layer(il) for il in range(n_layer)])
        self.gguf_writer.add_kv_lora_rank_swa(swa_kv_lora_rank)
        self.gguf_writer.add_key_length_swa(swa_kv_lora_rank + hparams["swa_qk_rope_head_dim"])
        self.gguf_writer.add_value_length_swa(swa_kv_lora_rank)
        self.gguf_writer.add_key_length_mla_swa(hparams["swa_qk_nope_head_dim"] + hparams["swa_qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla_swa(hparams["swa_v_head_dim"])
        if hparams["swa_q_lora_rank"] != hparams["q_lora_rank"]:
            raise ValueError("dots3-note conversion assumes a shared q_lora_rank for both layer kinds")

        if self.n_nextn:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn)

        # DSA indexer (full-attention layers only)
        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])
        self.gguf_writer.add_indexer_types([not self._is_swa_layer(il) for il in range(n_layer)])

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

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # move the MTP token embedding into the NextN block so the standard nextn mapping picks it up
        if name == "model.mtp.embed_tokens.weight":
            name = f"model.layers.{self.hparams['num_hidden_layers']}.embed_tokens.weight"
            bid = self.hparams["num_hidden_layers"]

        # fold the activation rescale sqrt(n_embd/lora_rank) into the preceding RMSNorm weight
        # this also covers the indexer wq_b, which reads the same rescaled q_lora activation
        if self.apply_lora_rescale and bid is not None:
            if name.endswith("q_a_layernorm.weight"):
                data_torch = data_torch * math.sqrt(self.hparams["hidden_size"] / self.hparams["q_lora_rank"])
            elif name.endswith("kv_a_layernorm.weight"):
                rank = self.hparams["swa_kv_lora_rank"] if self._is_swa_layer(bid) else self.hparams["kv_lora_rank"]
                data_torch = data_torch * math.sqrt(self.hparams["hidden_size"] / rank)

        # MLA absorption: split kv_b_proj into k_b (transposed) and v_b, per-layer-kind geometry
        if name.endswith("kv_b_proj.weight"):
            assert bid is not None
            if self._is_swa_layer(bid):
                n_head = self.hparams["swa_num_attention_heads"]
                qk_nope_head_dim = self.hparams["swa_qk_nope_head_dim"]
                v_head_dim = self.hparams["swa_v_head_dim"]
            else:
                n_head = self.hparams["num_attention_heads"]
                qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
                v_head_dim = self.hparams["v_head_dim"]
            if isinstance(n_head, list):  # set_gguf_parameters turns this into a per-layer array
                n_head = n_head[bid]

            assert data_torch.shape[0] == n_head * (qk_nope_head_dim + v_head_dim)

            kv_b = data_torch.view(n_head, qk_nope_head_dim + v_head_dim, data_torch.shape[-1])
            k_b, v_b = kv_b.split([qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from ModelBase.modify_tensors(self, k_b, name.replace("kv_b_proj", "k_b_proj"), bid)
            yield from ModelBase.modify_tensors(self, v_b, name.replace("kv_b_proj", "v_b_proj"), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration")
class Dots3NoteMmprojModel(MmprojModel):
    has_vision_encoder = True
    has_audio_encoder = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        assert self.hparams_audio is not None

        # preprocessor_config.json nests the image params under vision_config
        self.preprocessor_config = {**self.preprocessor_config, **self.preprocessor_config.get("vision_config", {})}

        vis = self.hparams_vision
        # in this config, hidden_size is the adapter output width; embed_dim is the tower width
        vis["hidden_size"] = vis["embed_dim"]
        vis["image_size"] = 0  # dynamic resolution
        self.pyramid = [max(0, n) for n in vis["pyramid_num_routed"]]

        if vis.get("adapter_type") != "patch_merger" or not vis.get("pre_pixel_shuffle"):
            raise ValueError("dots3-note vision conversion requires adapter_type=patch_merger and pre_pixel_shuffle")
        if vis.get("router_scoring_func", "sigmoid") != "sigmoid" or vis.get("router_scale", 1.0) != 1.0:
            raise ValueError("dots3-note vision conversion only supports sigmoid routing with router_scale=1.0")
        if vis.get("temporal_patch_size", 1) != 1 or vis.get("use_bias") or not vis.get("use_qk_norm"):
            raise ValueError("unsupported dots3-note vision config variant")

        aud = self.hparams_audio
        if not aud.get("use_conv2d_stem") or not aud.get("use_rope") or not aud.get("use_rms_norm") or aud.get("use_causal"):
            raise ValueError("unsupported dots3-note audio config variant")
        if aud["whisper_config"].get("activation_function") != "swiglu":
            raise ValueError("dots3-note audio conversion requires the swiglu activation")
        if aud.get("merge_factor", 1) != 1 or aud.get("chunk_seconds") != 60:
            raise ValueError("unsupported dots3-note audio chunking config")
        # the graph hard-codes these rope parameters
        rope = aud.get("rope_parameters", {})
        if rope.get("partial_rotary_factor") != 0.5 or rope.get("rope_theta") != 10000.0:
            raise ValueError("unsupported dots3-note audio rope config")

    def get_audio_config(self) -> dict[str, Any] | None:
        cfg = self.global_config.get("audio_config")
        if cfg is not None:
            # aliases so MmprojModel.find_aparam() / n_block_keys can resolve them
            whisper = cfg["whisper_config"]
            cfg["hidden_size"] = whisper["d_model"]
            cfg["intermediate_size"] = whisper["encoder_ffn_dim"]
            cfg["num_attention_heads"] = whisper["encoder_attention_heads"]
            cfg["num_hidden_layers"] = whisper["encoder_layers"]
        return cfg

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        assert self.hparams_audio is not None

        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.DOTS3NOTE_V)
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams_vision["rms_norm_eps"])
        self.gguf_writer.add_vision_spatial_merge_size(self.hparams_vision["spatial_merge_size"])
        self.gguf_writer.add_vision_min_pixels(self.preprocessor_config["min_pixels"])
        self.gguf_writer.add_vision_max_pixels(self.preprocessor_config["max_pixels"])
        # pyramid MoE: per-block routed expert count, 0 = dense block
        self.gguf_writer.add_vision_expert_count_per_layer(self.pyramid)
        self.gguf_writer.add_vision_expert_used_count(int(self.hparams_vision["capacity_factor"]))

        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.DOTS3NOTE_A)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams_audio["whisper_config"]["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-6)  # Dots3NoteAudioRMSNorm default

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if not name.startswith(("vision_encoder.", "audio_encoder.")):
            return None
        return super().filter_tensors(item)

    _vis_experts: dict[int, dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # router params have no .weight suffix in the checkpoint, but gguf tools expect one
        if name.endswith((".gate_weight", ".router_bias")):
            name += ".weight"

        # audio fc1 fuses gate and up for swiglu; split it
        if ".speech_encoder.layers." in name and ".fc1." in name:
            gate, up = data_torch.chunk(2, dim=0)
            yield from super().modify_tensors(gate, name.replace(".fc1.", ".fc1_gate."), bid)
            yield from super().modify_tensors(up, name.replace(".fc1.", ".fc1_up."), bid)
            return

        # vision MoE: stack per-expert weights into a single 3D tensor per block
        if ".mlp.experts." in name:
            assert bid is not None
            n_expert = self.pyramid[bid]
            if self._vis_experts is None:
                self._vis_experts = {}
            buf = self._vis_experts.setdefault(bid, {})
            buf[name] = data_torch

            if len(buf) >= n_expert * 3:
                for w_name in ("fc1", "fc2", "fc3"):
                    datas: list[Tensor] = []
                    for xid in range(n_expert):
                        ename = f"vision_encoder.blocks.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(buf.pop(ename))
                    merged = torch.stack(datas, dim=0)
                    yield from super().modify_tensors(merged, f"vision_encoder.blocks.{bid}.mlp.experts.{w_name}.weight", bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._vis_experts is not None:
            leftover = [k for d in self._vis_experts.values() for k in d.keys()]
            if leftover:
                raise ValueError(f"unprocessed vision experts: {leftover}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # FP32 routing is load-bearing for the vision MoE (near-tied expert scores)
        if ".ffn_gate_inp." in new_name or ".exp_probs_b." in new_name:
            return gguf.GGMLQuantizationType.F32
        if ".conv2d" in new_name or "a.conv_out" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
