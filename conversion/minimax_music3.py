from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, gguf
from .drama3 import fold_snake
from .qwen import Qwen3Model

# MiniMax Music 3, diffusers layout: one checkpoint per component
# text model: the global LLM (language_model/), samples the semantic code of each frame
# mmproj: RVQ depth decoder, condition encoder, flow-matching DiT and DAC decoder (vocoder)

LM_DIR = "language_model"
TOK_DIR = "tokenizer"
MMPROJ_DIRS = ("rvq_depth_decoder", "condition_encoder", "transformer", "vocoder")

# token ids of the checkpoint, see encoders.py of the diffusers pipeline
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_CODE_OFFSET = 151675
N_SEMANTIC = 16384


def _load_hparams(dir_model: Path) -> dict[str, Any]:
    return ModelBase.load_hparams(dir_model / LM_DIR, is_mistral_format=False)


def _index_dirs(model: ModelBase, dirs: Iterable[str], keep: Callable[[str], bool]) -> dict[str, Callable[[], Tensor]]:
    # the names get the component dir as prefix
    root = model.dir_model
    tensors: dict[str, Callable[[], Tensor]] = {}
    try:
        for d in dirs:
            model.dir_model = root / d
            prefix = "model" if d == LM_DIR else "diffusion_pytorch_model"
            for name, gen in model.index_local_tensors(prefix).items():
                if keep(f"{d}.{name}"):
                    tensors[f"{d}.{name}"] = gen
    finally:
        model.dir_model = root
    return tensors


def _fold_wn(g: Tensor, v: Tensor) -> Tensor:
    # torch weight norm, over every dim but 0
    g, v = g.float(), v.float()
    return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))


@ModelBase.register("MiniMaxMusic3ForConditionalGeneration")
class MiniMaxMusic3TextModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        super().__init__(dir_model, *args, hparams=_load_hparams(dir_model), **kwargs)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = _index_dirs(self, [LM_DIR], lambda _: True)
        return {name.removeprefix(LM_DIR + "."): gen for name, gen in tensors.items()}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # the codes are sampled with classifier-free guidance, the uncond prompt runs as a second sequence
        self.gguf_writer.add_classifier_free_guidance(True)

    def set_vocab(self):
        root = self.dir_model
        self.dir_model = root / TOK_DIR
        try:
            tokens, toktypes, tokpre = self.get_vocab_base()
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        finally:
            self.dir_model = root

        # the semantic codes are not in the tokenizer, name them for debugging
        for i in range(N_SEMANTIC):
            tokens[AUDIO_CODE_OFFSET + i] = f"<|audio_code_{i}|>"
            toktypes[AUDIO_CODE_OFFSET + i] = gguf.TokenType.CONTROL

        # the generation ends with <|audio_end|>, the chat template of the tokenizer is not used
        template_path = Path(__file__).parent.parent / "models" / "templates" / "MiniMaxAI-MiniMax-Music3-tts.jinja"
        with open(template_path, encoding="utf-8") as f:
            special_vocab.chat_template = [{"name": "tts", "template": f.read()}]
        special_vocab.special_token_ids["eos"] = AUDIO_END

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelBase.register("MiniMaxMusic3ForConditionalGeneration")
class MiniMaxMusic3MmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    def __init__(self, dir_model: Path, *args, **kwargs):
        kwargs.pop("hparams", None)
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        hparams["text_config"] = _load_hparams(dir_model)
        for d in MMPROJ_DIRS:
            hparams[d] = ModelBase.load_hparams(dir_model / d, is_mistral_format=False)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    def get_audio_config(self) -> dict[str, Any] | None:
        # the generic gen layers are the depth decoder
        return self.global_config["rvq_depth_decoder"]

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # from the backbone, only the embeddings of the semantic codes
        return _index_dirs(self, (LM_DIR, *MMPROJ_DIRS),
                           lambda name: not name.startswith(LM_DIR + ".") or name == f"{LM_DIR}.model.embed_tokens.weight")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        return item

    def set_gguf_parameters(self):
        hp = self.hparams
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.MINIMAX_MUSIC3_GEN)
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(hp["num_layers"])
        self.gguf_writer.add_gen_audio_head_count(hp["num_attention_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(1e-6)
        # the DiT, the condition encoder and the vocoder hparams come from the tensor shapes or are hard-coded in tools/mtmd

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # only the transformer layers follow --outtype, the convs and the small projections stay F32
        if n_dims >= 2 and new_name.startswith(("a.gen.code.blk.", "a.gen.flow.blk.")):
            return super().tensor_force_quant(name, new_name, bid, n_dims)
        if new_name.startswith(("a.gen.code.head", "a.gen.code.embd", "a.gen.code.out_embd", "a.gen.code.proj_in")):
            return super().tensor_force_quant(name, new_name, bid, n_dims)
        return gguf.GGMLQuantizationType.F32

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("a."):
            yield (name, data_torch)
            return

        comp, key = name.split(".", 1)
        if comp == LM_DIR:
            yield ("a.gen.code.out_embd.weight", data_torch[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + N_SEMANTIC])
        elif comp == "rvq_depth_decoder":
            yield from self._depth(data_torch, key)
        elif comp == "condition_encoder":
            yield from self._cond(data_torch, key)
        elif comp == "transformer":
            yield from self._dit(data_torch, key)
        # the vocoder is folded in generate_extra_tensors()

    def _depth(self, t: Tensor, key: str) -> Iterable[tuple[str, Tensor]]:
        p = "a.gen.code"
        n_cb = self.hparams["num_codebooks"] - 1
        if key == "audio_embeddings.weight":
            # one table per residual codebook
            yield (f"{p}.embd.weight", t.reshape(n_cb, -1, t.shape[-1]))
        elif key.startswith("audio_heads."):
            if key == "audio_heads.0.weight":
                heads = [self._tensor(f"rvq_depth_decoder.audio_heads.{i}.weight") for i in range(n_cb)]
                yield (f"{p}.head.weight", torch.stack(heads))
        elif key == "projection.weight":
            yield (f"{p}.proj_in.weight", t)
        elif key == "pos_embedding.weight":
            yield (f"{p}.position_embd.weight", t)
        elif key == "norm.weight":
            yield (f"{p}.post_ln.weight", t)
        else:
            _, il, rest = key.split(".", 2)
            layer = {
                "input_layernorm.weight":          "ln1.weight",
                "attn.to_q.weight":                "attn_q.weight",
                "attn.to_k.weight":                "attn_k.weight",
                "attn.to_v.weight":                "attn_v.weight",
                "attn.to_out.weight":              "attn_out.weight",
                "post_attention_layernorm.weight": "ln2.weight",
                "gate_proj.weight":                "ffn_gate.weight",
                "up_proj.weight":                  "ffn_up.weight",
                "down_proj.weight":                "ffn_down.weight",
            }
            yield (f"{p}.blk.{il}.{layer[rest]}", t)

    def _cond(self, t: Tensor, key: str) -> Iterable[tuple[str, Tensor]]:
        p = "a.gen.cond"
        if key == "layer_weight_logits":
            # softmax mix of the 8 hidden states of a frame, with the scale folded in
            scale = self._tensor("condition_encoder.layer_scale").float()
            yield (f"{p}.layer_w", torch.softmax(t.float(), dim=0) * scale)
        elif key == "proj.weight":
            yield (f"{p}.proj.weight", t)
        elif key == "proj.bias":
            yield (f"{p}.proj.bias", t)

    def _dit(self, t: Tensor, key: str) -> Iterable[tuple[str, Tensor]]:
        p = "a.gen.flow"
        if key.startswith("transformer_blocks."):
            _, il, rest = key.split(".", 2)
            q = f"{p}.blk.{il}"
            if rest.startswith("ff_in."):
                # value half first, then the silu gate half
                suffix = rest.removeprefix("ff_in.")
                up, gate = t.chunk(2, dim=0)
                yield (f"{q}.ffn_up.{suffix}", up)
                yield (f"{q}.ffn_gate.{suffix}", gate)
                return
            layer = {
                "norm1.weight":          "ln1.weight",
                "norm1.bias":            "ln1.bias",
                "attn.to_q.weight":      "attn_q.weight",
                "attn.to_k.weight":      "attn_k.weight",
                "attn.to_v.weight":      "attn_v.weight",
                "attn.to_out.0.weight":  "attn_out.weight",
                "norm2.weight":          "ln2.weight",
                "norm2.bias":            "ln2.bias",
                "ff_out.weight":         "ffn_down.weight",
                "ff_out.bias":           "ffn_down.bias",
            }
            yield (f"{q}.{layer[rest]}", t)
            return
        if key == "time_proj.weight":
            # random Fourier features, 2 * pi folded in
            yield (f"{p}.time_freq", t.float().flatten() * 2.0 * torch.pi)
            return
        out = {
            "time_embed.linear_1.weight": "time_in.weight",
            "time_embed.linear_1.bias":   "time_in.bias",
            "time_embed.linear_2.weight": "time_out.weight",
            "time_embed.linear_2.bias":   "time_out.bias",
            "preprocess_conv.weight":     "pre.weight",
            "proj_in.weight":             "proj_in.weight",
            "proj_out.weight":            "proj_out.weight",
            "postprocess_conv.weight":    "post.weight",
        }[key]
        yield (f"{p}.{out}", t.squeeze(-1) if key.endswith("conv.weight") else t)

    def _tensor(self, name: str) -> Tensor:
        # read directly, without the dtype handling of prepare_tensors()
        return self.model_tensors[name]().float()

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # DAC decoder, same layout as the yue2 VAE (a.gen.vae.*), with the snake and the weight norm folded
        def get(key: str) -> Tensor:
            return self._tensor(f"vocoder.{key}")

        def wn(key: str) -> Tensor:
            return _fold_wn(get(key + ".weight_g"), get(key + ".weight_v"))

        def snake(dst: str, key: str) -> Iterable[tuple[str, Tensor]]:
            a, b = fold_snake(get(key + ".alpha"))
            yield (f"{dst}.alpha", a)
            yield (f"{dst}.beta", b)

        p = "a.gen.vae"
        yield (f"{p}.in_proj.weight", get("dec_in_proj.weight"))
        yield (f"{p}.in_proj.bias",   get("dec_in_proj.bias"))
        yield (f"{p}.conv_in.weight", wn("conv_in"))
        yield (f"{p}.conv_in.bias",   get("conv_in.bias"))
        n_blk = len(self.global_config["vocoder"]["upsampling_ratios"])
        for b in range(n_blk):
            s = f"blocks.{b}"
            yield from snake(f"{p}.blk.{b}.snake", f"{s}.snake1")
            yield (f"{p}.blk.{b}.up.weight", wn(f"{s}.conv_t1"))
            yield (f"{p}.blk.{b}.up.bias",   get(f"{s}.conv_t1.bias"))
            for r in range(3):
                rs = f"{s}.res_unit{r + 1}"
                yield from snake(f"{p}.blk.{b}.res.{r}.snake1", f"{rs}.snake1")
                yield (f"{p}.blk.{b}.res.{r}.conv1.weight", wn(f"{rs}.conv1"))
                yield (f"{p}.blk.{b}.res.{r}.conv1.bias",   get(f"{rs}.conv1.bias"))
                yield from snake(f"{p}.blk.{b}.res.{r}.snake2", f"{rs}.snake2")
                yield (f"{p}.blk.{b}.res.{r}.conv2.weight", wn(f"{rs}.conv2"))
                yield (f"{p}.blk.{b}.res.{r}.conv2.bias",   get(f"{rs}.conv2.bias"))
        yield from snake(f"{p}.post_snake", "snake_out")
        yield (f"{p}.conv_out.weight", wn("conv_out"))
        yield (f"{p}.conv_out.bias",   get("conv_out.bias"))
