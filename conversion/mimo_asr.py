from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf
from .qwen import Qwen2Model

@ModelBase.register("MiMoV2ASRForCausalLM")
class MiMoAudioMmprojModel(MmprojModel):
    has_audio_encoder = True
    has_vision_encoder = False

    n_block_keys = ("input_local_layers",)

    _AUDIO_TENSOR_PREFIXES = (
        "mimo_audio_tokenizer",
        "speech_embeddings",
        "input_local_transformer",
        "speech_group_downcast"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.audio_dir = Path(self.dir_model).parent / "MiMo-Audio-Tokenizer"
        self.audio_model_file = self.audio_dir / "model.safetensors"
        self.audio_config_file = self.audio_dir / "config.json"

        self.hparams_audio = self.hparams_audio or self.hparams["audio_config"]
        hp, hpa = self.hparams, self.hparams_audio
        
        hpa["hidden_size"] = hp["input_local_dim"]
        hpa["num_hidden_layers"] = hp["input_local_layers"]
        hpa["num_attention_heads"] = hpa["input_local_attn_heads"]
        hpa["intermediate_size"] = hpa["input_local_intermediate_size"]
        
        max_blocks = max(self.block_count or 0, hp["audio_channels"])
        if self.audio_model_file.exists():
            from safetensors import safe_open
            with safe_open(self.audio_model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if match := re.search(r'\.layers\.(\d+)', key):
                        max_blocks = max(max_blocks, int(match.group(1)) + 1)
                        
        self.block_count = max_blocks
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        gw = self.gguf_writer
        hp = self.hparams
        
        ch = hp["audio_channels"]
        
        gw.add_clip_audio_projector_type(gguf.VisionProjectorType.MIMO_V2_ASR)
        
        tok_cfg = {}
        if self.audio_config_file.exists():
            with open(self.audio_config_file, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
        else:
            print(f"\nWarning: '{self.audio_config_file}' not found, using default audio tokenizer params.")
            
        gw.add_audio_num_mel_bins(tok_cfg.get("n_mels", 128))
        gw.add_audio_attention_layernorm_eps(tok_cfg.get("layer_norm_eps", 1e-6))
        
        gw.add_audio_embedding_length(tok_cfg.get("d_model", 768)) 
        gw.add_audio_block_count(tok_cfg.get("encoder_layers", 8))
        gw.add_audio_head_count(tok_cfg.get("encoder_attention_heads", 12))
        gw.add_audio_feed_forward_length(tok_cfg.get("encoder_ffn_dim", 3072))
        gw.add_audio_projection_dim(hp.get("input_local_dim", 1024))
        gw.add_uint32("mimo.rvq_codebook_count", ch)
        
        def parse_config_array(value, length):
            val_str = str(value)
            return [int(x) for x in val_str.split("-")] if "-" in val_str else [int(val_str)] * length

        gw.add_array("mimo.rvq_vocab_sizes", parse_config_array(hp["speech_vocab_size"], ch))
        gw.add_array("mimo.speech_zeroemb_idx", parse_config_array(hp["speech_zeroemb_idx"], ch))

        gw.add_uint32("mimo.text_empty_idx", 151667)
        gw.add_uint32("mimo.group_size", hp["group_size"])
        gw.add_bool("mimo.input_full_attention", hp["input_full_attention"])

    def is_audio_tensor(self, name: str) -> bool:
        return name.startswith(self._AUDIO_TENSOR_PREFIXES)

    def get_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().get_tensors()

        if not self.audio_model_file.exists():
            raise FileNotFoundError(
                f"\nError: Audio encoder weights not found at '{self.audio_model_file}'.\n"
                "Please place the audio encoder weights in the sibling directory 'MiMo-Audio-Tokenizer'.\n"
            )

        from safetensors.torch import load_file

        for name, data in load_file(self.audio_model_file).items():
            if name.startswith("encoder."):
                if "quantizer.vq" in name and not name.endswith("embed"):
                    continue
                yield f"mimo_audio_tokenizer.{name}", data

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | None:
        # Keep RVQ codebooks at F32 precision
        if self.is_audio_tensor(name) and "quantizer.vq.layers" in name and "codebook" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid

        if not self.is_audio_tensor(name):
            return

        if data_torch.ndim == 0:
            data_torch = data_torch.unsqueeze(0)

        name = name.replace("encoder.conv1", "encoder.conv.0").replace("encoder.conv2", "encoder.conv.1")
        yield (self.map_tensor_name(name, (".weight", ".bias")), data_torch)


@ModelBase.register("MiMoV2ASRForCausalLM")
class MiMoV2AsrModel(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN2

    _ASR_SPECIFIC_PREFIXES = (
        "mimo_audio_tokenizer", 
        "speech_embeddings", 
        "input_local_transformer", 
        "speech_group_downcast", 
        "local_transformer", 
        "hidden_states_downcast",
        "local_transformer_lm_heads"
    )

    def set_vocab(self):
        super().set_vocab()
        template = (
            "{%- for message in messages -%}"
                "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}"
                "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n' -}}"
            "{%- endif -%}"
        )
        self.gguf_writer.add_chat_template(template)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Intercept MiMoASR-specific tensors
        if name.startswith(self._ASR_SPECIFIC_PREFIXES):
            return
            
        yield from super().modify_tensors(data_torch, name, bid)