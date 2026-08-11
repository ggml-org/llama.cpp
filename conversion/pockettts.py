from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, SentencePieceTokenTypes, TextModel, gguf, logger

# Pocket TTS is a CALM: the backbone conditions a flow-matching decoder that generates one
# continuous 32-d latent per frame. There is no codebook in this model.
# The checkpoint ships no config.json, hparams come from _load_hparams() below.
#
# Tricks being used to support this model via existing llama.cpp code paths:
# - bos_before_voice and bos_emb are learned input vectors, not tokens
#   they are appended to the embedding table as extra tokens, to be looked up like any other row
# - bos_emb lives in latent space, so input_linear is folded into it here
# - the backbone has no lm_head, the embedding table is reused as output for the unused logits
#
# pipeline stage mapping:
#   mimi encoder + speaker_proj --> mapped to normal mtmd audio encoder
#   flow_lm.transformer         --> mapped to normal libllama text model (autoregressive)
#   flow_lm.flow_net + out_eos  --> MTMD_GEN_PROCESS_TYPE_GEN_CODE
#   mimi decoder                --> MTMD_GEN_PROCESS_TYPE_GEN_WAV

# indices into mimi.encoder.model / mimi.decoder.model for stage i, see SEANetEncoder/SEANetDecoder
_ENC_RES_IDX   = lambda i: 1 + 3 * i  # noqa: E731
_ENC_SCALE_IDX = lambda i: 3 + 3 * i  # noqa: E731
_DEC_SCALE_IDX = lambda i: 2 + 3 * i  # noqa: E731
_DEC_RES_IDX   = lambda i: 3 + 3 * i  # noqa: E731

_N_SEANET_STAGES = 3
_SAMPLE_RATE = 24000


def _tensor_shapes(dir_model: Path) -> dict[str, tuple[int, ...]]:
    part_names = ModelBase.get_model_part_names(dir_model, "model", ".safetensors")
    if len(part_names) != 1:
        return {}
    with gguf.utility.SafetensorsLocal(dir_model / part_names[0]) as part:
        return {name: tuple(part[name].shape) for name in part.keys()}


@ModelBase.register_hparams_loader(lambda dir_model: "flow_lm.bos_emb" in _tensor_shapes(dir_model))
def _load_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected pocket-tts checkpoint, deriving hparams from tensor shapes")
    shapes = _tensor_shapes(dir_model)
    n_vocab, n_embd = shapes["flow_lm.conditioner.embed.weight"]
    n_layer = sum(1 for name in shapes if re.fullmatch(r"flow_lm\.transformer\.layers\.\d+\.norm1\.weight", name))
    n_layer_a = sum(1 for name in shapes if re.fullmatch(r"mimi\.encoder_transformer\.transformer\.layers\.\d+\.norm1\.weight", name))
    n_embd_a = shapes["mimi.encoder_transformer.transformer.layers.0.norm1.weight"][0]
    return {
        "architectures": ["PocketTTSModel"],
        "model_type": "pockettts",
        "num_hidden_layers": n_layer,
        "hidden_size": n_embd,
        "intermediate_size": shapes["flow_lm.transformer.layers.0.linear1.weight"][0],
        # the transformer is fully causal with no context limit, this only bounds the KV cache
        "max_position_embeddings": 4096,
        # not in the checkpoint, but every released variant uses head_dim 64
        "num_attention_heads": n_embd // 64,
        # extra rows for the learned input vectors, see _embd_table()
        "vocab_size": n_vocab + (2 if "flow_lm.bos_before_voice" in shapes else 1),
        "rope_theta": 10000.0,
        "layer_norm_eps": 1e-5,
        "audio_config": {
            "num_hidden_layers": n_layer_a,
            "hidden_size": n_embd_a,
            "intermediate_size": shapes["mimi.encoder_transformer.transformer.layers.0.linear1.weight"][0],
            "num_attention_heads": n_embd_a // 64,
        },
    }


@ModelBase.register("PocketTTSModel")
class PocketTTSModel(TextModel):
    model_arch = gguf.MODEL_ARCH.POCKETTTS

    _LAYER_TENSOR_MAP = {
        "norm1":               gguf.MODEL_TENSOR.ATTN_NORM,
        "norm2":               gguf.MODEL_TENSOR.FFN_NORM,
        "self_attn.out_proj":  gguf.MODEL_TENSOR.ATTN_OUT,
        "linear1":             gguf.MODEL_TENSOR.FFN_UP,
        "linear2":             gguf.MODEL_TENSOR.FFN_DOWN,
    }

    def set_vocab(self):
        # this is a unigram sentencepiece model, llama.cpp's SPM tokenizer cannot do
        # unigram segmentation, so use the UGM tokenizer instead
        from sentencepiece import sentencepiece_model_pb2 as model

        proto = model.ModelProto()  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore[unresolved-attribute]
        proto.ParseFromString(open(self.dir_model / "tokenizer.model", "rb").read())
        assert proto.trainer_spec.model_type == 1, "expected a unigram tokenizer"

        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        # the last rows of the embedding table are not sentencepiece pieces
        extra = self._extra_tokens()
        for i, name in enumerate(extra):
            tokens[len(tokens) - len(extra) + i] = name.encode("utf-8")
            toktypes[len(tokens) - len(extra) + i] = SentencePieceTokenTypes.CONTROL
            scores[len(tokens) - len(extra) + i] = -1000.0

        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_add_space_prefix(proto.normalizer_spec.add_dummy_prefix)
        self.gguf_writer.add_remove_extra_whitespaces(proto.normalizer_spec.remove_extra_whitespaces)
        if proto.normalizer_spec.precompiled_charsmap:
            self.gguf_writer.add_precompiled_charsmap(proto.normalizer_spec.precompiled_charsmap)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if not name.startswith("flow_lm."):
            return  # mimi and the flow net go to the mmproj

        if name == "flow_lm.conditioner.embed.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), self._embd_table(data_torch))
            return

        if name.startswith("flow_lm.out_norm."):
            suffix = "." + name.rsplit(".", 1)[1]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM, suffix=suffix), data_torch)
            return

        if name.startswith("flow_lm.transformer.layers."):
            assert bid is not None
            key_with_suffix = name.split(f"layers.{bid}.", 1)[1]
            key, suffix = key_with_suffix.rsplit(".", 1)

            if key == "self_attn.in_proj":
                q, k, v = data_torch.chunk(3, dim=0)
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q)
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k)
                yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v)
                return

            tensor = self._LAYER_TENSOR_MAP.get(key)
            if tensor is not None:
                yield (self.format_tensor_name(tensor, bid, suffix="." + suffix), data_torch)
                return

        return

    def _extra_tokens(self) -> list[str]:
        # the conditioner's padding row, then the learned vectors appended by _embd_table().
        # bos_before_voice only exists when the pack sets insert_bos_before_voice
        names = ["<|pad|>"]
        if "flow_lm.bos_before_voice" in self.model_tensors:
            names.append("<|bos_before_voice|>")
        names.append("<|audio_bos|>")
        return names

    def _embd_table(self, embed: Tensor) -> Tensor:
        rows = [embed]
        if "flow_lm.bos_before_voice" in self.model_tensors:
            rows.append(self.model_tensors["flow_lm.bos_before_voice"]().reshape(1, -1).to(embed.dtype))

        # bos_emb is a latent, it only enters the backbone through input_linear
        bos_emb = self.model_tensors["flow_lm.bos_emb"]()
        input_linear = self.model_tensors["flow_lm.input_linear.weight"]()
        audio_bos = torch.nn.functional.linear(bos_emb.float(), input_linear.float()).reshape(1, -1)
        rows.append(audio_bos.to(embed.dtype))

        return torch.cat(rows, dim=0)


@ModelBase.register("PocketTTSModel")
class PocketTTSMmprojModel(MmprojModel):
    has_audio_encoder = True
    has_vision_encoder = False

    _MIMI_TFM_MAP = {
        "norm1":              (gguf.MODEL_TENSOR.A_ENC_INPUT_NORM,  gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_NORM),
        "norm2":              (gguf.MODEL_TENSOR.A_ENC_OUTPUT_NORM, gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_NORM),
        "self_attn.out_proj": (gguf.MODEL_TENSOR.A_ENC_OUTPUT,      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_OUT),
        "linear1":            (gguf.MODEL_TENSOR.A_ENC_FFN_UP,      gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_UP),
        "linear2":            (gguf.MODEL_TENSOR.A_ENC_FFN_DOWN,    gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_DOWN),
        "layer_scale_1.scale": (gguf.MODEL_TENSOR.A_ENC_ATTN_SCALE, gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_SCALE),
        "layer_scale_2.scale": (gguf.MODEL_TENSOR.A_ENC_FFN_SCALE_LS, gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_SCALE),
    }
    _MIMI_TFM_QKV = (
        (gguf.MODEL_TENSOR.A_ENC_ATTN_Q, gguf.MODEL_TENSOR.A_ENC_ATTN_K, gguf.MODEL_TENSOR.A_ENC_ATTN_V),
        (gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_Q, gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_K, gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_V),
    )

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        assert self.hparams_audio is not None

        # voice-prompt encoder: mimi encoder + speaker_proj
        self.gguf_writer.add_clip_has_audio_encoder(True)
        # note: the 24kHz sample rate is hardcoded on the clip.cpp side, like the other audio models
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.POCKETTTS_SPKENC)
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_audio_block_count(self.hparams_audio["num_hidden_layers"])
        self.gguf_writer.add_audio_embedding_length(self.hparams_audio["hidden_size"])
        self.gguf_writer.add_audio_feed_forward_length(self.hparams_audio["intermediate_size"])
        self.gguf_writer.add_audio_head_count(self.hparams_audio["num_attention_heads"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)
        # mimi convolves the waveform directly, it is passed around as a 1-row "mel"
        self.gguf_writer.add_audio_num_mel_bins(1)

        # generation: flow-matching decoder + mimi decoder
        # the SEANet and flow net hparams are constant across the family, clip.cpp holds them
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.POCKETTTS_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(self.hparams_audio["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(self.hparams_audio["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(self.hparams_audio["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(self.hparams_audio["num_attention_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-5)

        self.gguf_writer.add_gen_audio_model_variant(self.dir_model.name)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if ".seanet." in new_name or new_name in ("a.downsample.conv.weight", "a.gen.wav.upsample.weight"):
            return gguf.GGMLQuantizationType.F16
        return False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # the block index of the mimi transformers is parsed here, not by the base class
        T = gguf.MODEL_TENSOR

        if name in ("flow_lm.bos_emb", "flow_lm.bos_before_voice", "flow_lm.conditioner.embed.weight"):
            return  # folded into the backbone embedding table
        if name.startswith("flow_lm.transformer.") or name.startswith("flow_lm.out_norm."):
            return  # backbone

        if name == "flow_lm.speaker_proj_weight":
            yield (self.format_tensor_name(T.A_ENC_SPEAKER_PROJ), data_torch)
            return
        if name == "flow_lm.input_linear.weight":
            yield (self.format_tensor_name(T.A_GEN_INPUT_LINEAR), data_torch)
            return
        if name == "flow_lm.emb_mean":
            yield (self.format_tensor_name(T.A_GEN_EMB_MEAN, suffix=""), data_torch)
            return
        if name == "flow_lm.emb_std":
            yield (self.format_tensor_name(T.A_GEN_EMB_STD, suffix=""), data_torch)
            return
        if name.startswith("flow_lm.out_eos."):
            suffix = "." + name.rsplit(".", 1)[1]
            yield (self.format_tensor_name(T.A_GEN_OUT_EOS, suffix=suffix), data_torch)
            return

        if name.startswith("flow_lm.flow_net."):
            yield from self._flow_net_tensor(name, data_torch)
            return

        if name == "mimi.downsample.conv.conv.weight":
            yield (self.format_tensor_name(T.A_ENC_DOWNSAMPLE_CONV), data_torch)
            return
        if name == "mimi.upsample.convtr.convtr.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_UPSAMPLE), data_torch)
            return
        if name == "mimi.quantizer.output_proj.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_QUANT_OUT), data_torch.squeeze(-1))
            return

        if "_transformer.transformer.layers." in name:
            yield from self._mimi_tfm_tensor(name, data_torch)
            return

        if name.startswith("mimi.encoder.model.") or name.startswith("mimi.decoder.model."):
            yield from self._seanet_tensor(name, data_torch)
            return

        return

    def _flow_net_tensor(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        key = name.split("flow_lm.flow_net.", 1)[1]
        suffix = "." + key.rsplit(".", 1)[1]

        simple = {
            "input_proj":                 T.A_GEN_FLOW_INPUT_PROJ,
            "cond_embed":                 T.A_GEN_FLOW_COND_EMBD,
            "final_layer.linear":         T.A_GEN_FLOW_FINAL_PROJ,
            "final_layer.adaLN_modulation.1": T.A_GEN_FLOW_FINAL_ADA,
        }
        tensor = simple.get(key.rsplit(".", 1)[0])
        if tensor is not None:
            yield (self.format_tensor_name(tensor, suffix=suffix), data_torch)
            return

        if key.startswith("time_embed."):
            bid = int(key.split(".")[1])
            rest = key.split(f"time_embed.{bid}.", 1)[1]
            time_map = {
                "freqs":       (T.A_GEN_FLOW_TIME_FREQS, ""),
                "mlp.0":       (T.A_GEN_FLOW_TIME_UP,    suffix),
                "mlp.2":       (T.A_GEN_FLOW_TIME_DOWN,  suffix),
                "mlp.3.alpha": (T.A_GEN_FLOW_TIME_NORM,  ""),
            }
            entry = time_map.get(rest) or time_map.get(rest.rsplit(".", 1)[0])
            if entry is not None:
                yield (self.format_tensor_name(entry[0], bid, suffix=entry[1]), data_torch)
            return

        if key.startswith("res_blocks."):
            bid = int(key.split(".")[1])
            rest = key.split(f"res_blocks.{bid}.", 1)[1].rsplit(".", 1)[0]
            blk_map = {
                "in_ln":                T.A_GEN_FLOW_BLK_NORM,
                "mlp.0":                T.A_GEN_FLOW_BLK_UP,
                "mlp.2":                T.A_GEN_FLOW_BLK_DOWN,
                "adaLN_modulation.1":   T.A_GEN_FLOW_BLK_ADA,
            }
            tensor = blk_map.get(rest)
            if tensor is not None:
                yield (self.format_tensor_name(tensor, bid, suffix=suffix), data_torch)
            return

    def _mimi_tfm_tensor(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        is_decoder = name.startswith("mimi.decoder_transformer.")
        bid = int(name.split("_transformer.transformer.layers.", 1)[1].split(".")[0])
        key_with_suffix = name.split(f".layers.{bid}.", 1)[1]

        if key_with_suffix == "self_attn.in_proj.weight":
            q, k, v = data_torch.chunk(3, dim=0)
            names = self._MIMI_TFM_QKV[1 if is_decoder else 0]
            for tensor, part in zip(names, (q, k, v)):
                yield (self.format_tensor_name(tensor, bid), part)
            return

        key, suffix = key_with_suffix.rsplit(".", 1)
        entry = self._MIMI_TFM_MAP.get(key) or self._MIMI_TFM_MAP.get(key_with_suffix)
        if entry is None:
            return
        tensor = entry[1 if is_decoder else 0]
        suffix = ".weight" if key_with_suffix.endswith(".scale") else "." + suffix
        yield (self.format_tensor_name(tensor, bid, suffix=suffix), data_torch)

    def _seanet_tensor(self, name: str, data_torch: Tensor) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        is_decoder = name.startswith("mimi.decoder.")
        idx = int(name.split(".model.", 1)[1].split(".")[0])
        suffix = "." + name.rsplit(".", 1)[1]

        conv_in, conv_out, res1, res2, scale = (
            (T.A_GEN_WAV_SEANET_CONV_IN, T.A_GEN_WAV_SEANET_CONV_OUT, T.A_GEN_WAV_SEANET_RES_CONV1,
             T.A_GEN_WAV_SEANET_RES_CONV2, T.A_GEN_WAV_SEANET_SCALE_CONV)
            if is_decoder else
            (T.A_ENC_SEANET_CONV_IN, T.A_ENC_SEANET_CONV_OUT, T.A_ENC_SEANET_RES_CONV1,
             T.A_ENC_SEANET_RES_CONV2, T.A_ENC_SEANET_SCALE_CONV)
        )

        if idx == 0:
            yield (self.format_tensor_name(conv_in, suffix=suffix), data_torch)
            return
        if idx == 3 * _N_SEANET_STAGES + 2:
            yield (self.format_tensor_name(conv_out, suffix=suffix), data_torch)
            return

        for stage in range(_N_SEANET_STAGES):
            res_idx = _DEC_RES_IDX(stage) if is_decoder else _ENC_RES_IDX(stage)
            scale_idx = _DEC_SCALE_IDX(stage) if is_decoder else _ENC_SCALE_IDX(stage)
            if idx == scale_idx:
                yield (self.format_tensor_name(scale, stage, suffix=suffix), data_torch)
                return
            if idx == res_idx:
                # block.1 is the dilated conv, block.3 the pointwise one (0 and 2 are ELU)
                inner = int(name.split(".block.", 1)[1].split(".")[0])
                tensor = res1 if inner == 1 else res2
                yield (self.format_tensor_name(tensor, stage, suffix=suffix), data_torch)
                return
