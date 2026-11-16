from __future__ import annotations

import base64
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, gguf, logger
from .qwen import QwenModel, Qwen3Model

# token ids of the yue2_infer protocol (protocol.py)
EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START = 184621

# the VAE is a separate repo (m-a-p/YuE2-Vae), downloaded into this subdir of the model dir
VAE_DIR = "vae"

INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}

# the text is "[Tags]\n<style>\n[Lyrics]\n<lyrics>", optionally followed by "\n[ABC]\n<score>" (cover mode)
# voice picks the planning mode: full (default), melody or off
TTS_TEMPLATE = (
    "{%- set text = messages[0]['content'] -%}"
    "{%- set mode = (voice | lower) if voice is defined and voice and (voice | lower) in ['full', 'melody', 'off'] else 'full' -%}"
    "{%- set parts = text.split('\\n[ABC]\\n') -%}"
    "{%- set instructions = {"
    + ", ".join(f"'{k}': '{v}'" for k, v in INSTRUCTIONS.items())
    + "} -%}"
    "{{- '<|endoftext|>' + instructions[mode] + '\\n' + parts[0].rstrip() + '\\n' -}}"
    "{%- if mode == 'off' -%}"
    "{{- '<abc></abc><|music_start|>' -}}"
    "{%- elif parts | length > 1 -%}"
    "{{- '<abc>' + parts[1] + '</abc><|music_start|>' -}}"
    "{%- else -%}"
    "{{- '<abc>' -}}"
    "{%- endif -%}"
)


def _is_nar_tensor(name: str) -> bool:
    return ".nar_" in name or name.startswith(("llm2vae.", "vae2llm.", "time_embedder.", "latent_pos_embed."))


@ModelBase.register("YuE2ForCausalLM")
class YuE2TextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # the NAR path goes to the mmproj
        if _is_nar_tensor(item[0]):
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # the codes are sampled with classifier-free guidance, the uncond prompt runs as a second sequence
        self.gguf_writer.add_classifier_free_guidance(True)

    def set_vocab(self):
        # checkpoint-native tiktoken file, see tokenization_yue2.py
        ranks: dict[bytes, int] = {}
        with open(self.dir_model / "qwen.tiktoken", "rb") as f:
            for line in f.read().splitlines():
                if line:
                    tok, rank = line.split()
                    ranks[base64.b64decode(tok)] = int(rank)
        if len(ranks) != EOD:
            raise ValueError(f"YuE2: expected {EOD} BPE tokens, got {len(ranks)}")

        vocab_size = self.hparams["vocab_size"]
        tokens: list[str] = [""] * vocab_size
        toktypes: list[int] = [gguf.TokenType.UNUSED] * vocab_size
        merges: list[str] = []
        for tok, rank in ranks.items():
            tokens[rank] = QwenModel.token_bytes_to_string(tok)
            toktypes[rank] = gguf.TokenType.NORMAL
            if len(tok) > 1:
                merged = QwenModel.bpe(ranks, tok, max_rank=rank)
                assert len(merged) == 2
                merges.append(" ".join(map(QwenModel.token_bytes_to_string, merged)))

        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        named = {EOD + i: s for i, s in enumerate(specials)}
        named[MUSIC_START] = "<|music_start|>"
        named[MUSIC_END]   = "<|music_end|>"
        for i in range(CODEC_SIZE):
            named[CODEC_OFFSET + i] = f"<|codec_{i}|>"
        for i, s in enumerate(["<|latent_start|>", "<|latent_end|>", "<|latent_pad|>"]):
            named[LATENT_START + i] = s
        for i in range(EOD, vocab_size):
            tokens[i] = named.get(i, f"[PAD{i}]")
            toktypes[i] = gguf.TokenType.CONTROL if i in named else gguf.TokenType.UNUSED

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("qwen2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)
        self.gguf_writer.add_bos_token_id(EOD)
        self.gguf_writer.add_eos_token_id(MUSIC_END)
        self.gguf_writer.add_pad_token_id(EOD)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_chat_template([{"name": "tts", "template": TTS_TEMPLATE}])


@ModelBase.register("YuE2ForCausalLM")
class YuE2MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def get_audio_config(self) -> dict[str, Any] | None:
        # the NAR path: same shapes as the backbone layers
        return {k: self.global_config[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                                   "num_attention_heads", "num_key_value_heads", "rms_norm_eps")}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name = item[0]
        if name == "latent_pos_embed.pe":
            return None  # sinusoidal, computed at runtime
        if _is_nar_tensor(name) or name == "model.norm.weight":
            return item
        return None

    def set_gguf_parameters(self):
        hp = self.hparams
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.YUE2_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(hp["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(hp["num_attention_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(hp["num_key_value_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(hp["rms_norm_eps"])
        # rope theta, the ODE solver and the VAE hparams are hard-coded in tools/mtmd

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # read on the host, or small convs and projections of the flow / VAE graphs
        if not new_name.startswith("a.gen.nar.blk.") or n_dims < 2:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @staticmethod
    def _wn(sd: dict[str, Tensor], prefix: str) -> Tensor:
        # torch weight norm, over every dim but 0
        g, v = sd[prefix + ".weight_g"].float(), sd[prefix + ".weight_v"].float()
        return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        from safetensors.torch import load_file
        path = self.dir_model / VAE_DIR / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"YuE2: VAE not found at {path}, download m-a-p/YuE2-Vae into it")
        logger.info(f"gguf: loading VAE decoder from '{path}'")
        sd = {k[len("decoder.layers."):]: v for k, v in load_file(str(path)).items() if k.startswith("decoder.layers.")}
        p = "a.gen.vae"

        # SnakeBeta with log-scale params: x + 1/(exp(beta) + 1e-9) * sin(exp(alpha) * x)^2
        def snake(dst: str, src: str) -> Iterable[tuple[str, Tensor]]:
            yield (f"{dst}.alpha", torch.exp(sd[src + ".alpha"].float()))
            yield (f"{dst}.beta",  1.0 / (torch.exp(sd[src + ".beta"].float()) + 1e-9))

        yield (f"{p}.conv_in.weight", self._wn(sd, "0"))
        yield (f"{p}.conv_in.bias",   sd["0.bias"])
        n_blk = sum(1 for k in sd if k.endswith(".layers.1.weight_g") and k.count(".") == 3)
        for b in range(n_blk):
            s = f"{b + 1}.layers"
            yield from snake(f"{p}.blk.{b}.snake", f"{s}.0")
            yield (f"{p}.blk.{b}.up.weight", self._wn(sd, f"{s}.1"))
            yield (f"{p}.blk.{b}.up.bias",   sd[f"{s}.1.bias"])
            for r in range(3):
                rs = f"{s}.{r + 2}.layers"
                yield from snake(f"{p}.blk.{b}.res.{r}.snake1", f"{rs}.0")
                yield (f"{p}.blk.{b}.res.{r}.conv1.weight", self._wn(sd, f"{rs}.1"))
                yield (f"{p}.blk.{b}.res.{r}.conv1.bias",   sd[f"{rs}.1.bias"])
                yield from snake(f"{p}.blk.{b}.res.{r}.snake2", f"{rs}.2")
                yield (f"{p}.blk.{b}.res.{r}.conv2.weight", self._wn(sd, f"{rs}.3"))
                yield (f"{p}.blk.{b}.res.{r}.conv2.bias",   sd[f"{rs}.3.bias"])
        yield from snake(f"{p}.post_snake", f"{n_blk + 1}")
        yield (f"{p}.conv_out.weight", self._wn(sd, f"{n_blk + 2}"))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("a.gen.vae."):
            yield (name, data_torch)
            return

        p = "a.gen.nar"
        aux = {
            "model.norm.weight":         f"{p}.output_norm.weight",
            "vae2llm.weight":            f"{p}.inp.weight",
            "vae2llm.bias":              f"{p}.inp.bias",
            "llm2vae.weight":            f"{p}.out.weight",
            "llm2vae.bias":              f"{p}.out.bias",
            "time_embedder.mlp.0.weight": f"{p}.time.0.weight",
            "time_embedder.mlp.0.bias":   f"{p}.time.0.bias",
            "time_embedder.mlp.2.weight": f"{p}.time.2.weight",
            "time_embedder.mlp.2.bias":   f"{p}.time.2.bias",
        }
        if name in aux:
            yield (aux[name], data_torch)
            return

        # the NAR layers keep the libllama names, mtmd loads them as a text model
        assert bid is not None, name
        layer = {
            "nar_input_layernorm.weight":   "attn_norm.weight",
            "nar_self_attn.q_proj.weight":  "attn_q.weight",
            "nar_self_attn.k_proj.weight":  "attn_k.weight",
            "nar_self_attn.v_proj.weight":  "attn_v.weight",
            "nar_self_attn.o_proj.weight":  "attn_output.weight",
            "nar_self_attn.q_norm.weight":  "attn_q_norm.weight",
            "nar_self_attn.k_norm.weight":  "attn_k_norm.weight",
            "nar_pre_mlp_layernorm.weight": "ffn_norm.weight",
            "nar_mlp.gate_proj.weight":     "ffn_gate.weight",
            "nar_mlp.up_proj.weight":       "ffn_up.weight",
            "nar_mlp.down_proj.weight":     "ffn_down.weight",
        }
        key = name.split(".", 3)[3]  # strip "model.layers.{bid}."
        yield (f"{p}.blk.{bid}.{layer[key]}", data_torch)
