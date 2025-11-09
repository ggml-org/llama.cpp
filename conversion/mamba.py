from __future__ import annotations
import json
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("MambaForCausalLM", "MambaLMHeadModel", "FalconMambaForCausalLM")
class MambaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAMBA

    def __init__(self, dir_model: Path, *args, **kwargs):
        # Avoid using AutoConfig for hparams
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    def set_vocab(self):
        vocab_size = self.hparams["vocab_size"]
        # Round vocab size to next multiple of 8
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 8)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        elif (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            self._set_vocab_builtin("gpt-neox", vocab_size)

    def set_gguf_parameters(self):
        d_model = self.find_hparam(["hidden_size",       "d_model"])
        d_conv  = self.find_hparam(["conv_kernel",       "d_conv"],  optional=True) or 4
        d_inner = self.find_hparam(["intermediate_size", "d_inner"], optional=True) or 2 * d_model
        d_state = self.find_hparam(["state_size",        "d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["time_step_rank",     "dt_rank"],      optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5
        use_dt_b_c_norm = False
        # For falconmamba we do apply RMS norm on B / DT and C layers
        if self.find_hparam(["model_type"], optional=True) in ("falcon_mamba",):
            use_dt_b_c_norm = True
        # Fail early for models which don't have a block expansion factor of 2
        assert d_inner == 2 * d_model
        self.gguf_writer.add_context_length(2**20) # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(0) # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0) # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_ssm_dt_b_c_rms(use_dt_b_c_norm) # For classic Mamba we don't apply rms norm on B / DT layers
        self.gguf_writer.add_file_type(self.ftype)
    _tok_embd = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
        tok_embd_name = self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD)
        new_name = self.map_tensor_name(name)
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)
        # [4 1 8192 1] -> [4 8192 1 1]
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()
        # assuming token_embd.weight is seen before output.weight
        if self._tok_embd is not None and new_name == output_name:
            if torch.equal(self._tok_embd, data_torch):
                logger.debug(f"{output_name} is equivalent to {tok_embd_name}, omitting")
                return []
        elif new_name == tok_embd_name:
            self._tok_embd = data_torch
        return [(new_name, data_torch)]


@ModelBase.register("Mamba2ForCausalLM")
class Mamba2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MAMBA2

    def __init__(self, dir_model: Path, *args, **kwargs):
        # Avoid using AutoConfig for hparams
        # It wrongly assumes all Mamba2 models are Mamba-Codestral-7B-v0.1
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self.d_model = self.find_hparam(["hidden_size", "d_model", "dim"])
        self.d_inner = self.find_hparam(["mamba_d_ssm", "intermediate_size", "d_inner"], optional=True) or 2 * self.d_model
        self.n_group = self.find_hparam(["n_groups"], optional=True) or 1

    def set_vocab(self):
        vocab_size = self.hparams["vocab_size"]
        # Round vocab size to next multiple of 16
        pad_vocab = self.hparams.get("pad_vocab_size_multiple", 16)
        # pad using ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        vocab_size = -(vocab_size // -pad_vocab) * pad_vocab
        self.hparams["vocab_size"] = vocab_size
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        elif (self.dir_model / "tokenizer.model.v3").is_file():
            # mamba-codestral
            raise NotImplementedError(f"Please rename {self.dir_model / 'tokenizer.model.v3'} to {self.dir_model / 'tokenizer.model'}")
        elif (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            # Use the GPT-NeoX tokenizer when no tokenizer files are present
            self._set_vocab_builtin("gpt-neox", vocab_size)

    def set_gguf_parameters(self):
        d_conv  = self.find_hparam(["conv_kernel", "d_conv"],     optional=True) or 4
        d_state = self.find_hparam(["state_size",  "d_state"],    optional=True) or 128
        head_dim = self.find_hparam(["mamba_d_head", "head_dim"], optional=True) or 64
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-5
        # Fail early for models which don't have a block expansion factor of 2
        # TODO: does this really matter?
        # skip the assertion for FalconH1 Model
        if self.model_arch != gguf.MODEL_ARCH.FALCON_H1:
            assert self.d_inner == 2 * self.d_model
            assert self.d_inner % head_dim == 0
        self.gguf_writer.add_context_length(2**20)  # arbitrary value; for those who use the default
        self.gguf_writer.add_embedding_length(self.d_model)
        self.gguf_writer.add_feed_forward_length(0)  # unused, but seemingly required when loading
        self.gguf_writer.add_head_count(0)  # unused, but seemingly required when loading
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(self.d_inner // head_dim)
        self.gguf_writer.add_ssm_group_count(self.n_group)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.backbone") or name.startswith("model.lm_head"):
            # map Mamba-Codestral-7B-v0.1 tensor names to the names used by Mamba-2
            name = name.removeprefix("model.")
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        new_name = self.map_tensor_name(name)
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()
        elif any(self.match_model_tensor_name(new_name, t, bid, suffix="") for t in [
            gguf.MODEL_TENSOR.SSM_A,
            gguf.MODEL_TENSOR.SSM_D,
        ]):
            # unsqueeze A to use similar shape semantics as Mamba-1
            # (D is also unsqueezed, but for more straightforward broadcast internally)
            data_torch = data_torch.reshape((*data_torch.shape, 1))
        elif self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_NORM, bid):
            data_torch = data_torch.reshape((self.n_group, self.d_inner // self.n_group))
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)
        yield (new_name, data_torch)
