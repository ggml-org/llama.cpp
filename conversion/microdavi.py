from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .mamba import Mamba2Model


# Layer indices (0-indexed) that are TransformerBlock instead of Mamba2Block.
# Derived from the actual checkpoint: layers 5, 11, 17, 23 out of 24 total
# (every 6th layer, matching the original "(i + 1) % 6 == 0" rule in the
# training code). NOTE: this is hardcoded against the confirmed checkpoint
# shapes, not derived from any HF config field -- there is no HF config for
# this model. If you retrain with a different layer count/period, update
# this set (or better: move it into a small custom config.json you control
# and read it via find_hparam, see __init__ below for where that would slot in).
ATTN_LAYER_INDICES = {5, 11, 17, 23}


@ModelBase.register("MicroDaviPMTModel")
class MicroDaviModel(Mamba2Model):
    """
    Hybrid Mamba-2 / Transformer model (MicroDavi).

    - d_model=320, n_layers=24, n_heads=8 (confirmed against checkpoint tensor shapes)
    - Mamba2 layers: d_state=64, d_conv=4, expand=2, d_inner=640, n_heads_mamba=10,
      headdim=64, n_groups=1 (confirmed via direct mamba_ssm.Mamba2 instantiation)
    - Attention layers (5, 11, 17, 23): full MHA (no GQA, n_head_kv == n_head),
      RoPE base 10000.0, SwiGLU FFN at 4x expansion (fused gate+up Linear)
    - RMSNorm throughout, no biases, tied embedding/lm_head
    - Mamba2Block has NO feed-forward sublayer -- it is strictly
      x = x + mamba(norm(x)). Only attention layers have a separate FFN.
    """

    model_arch = gguf.MODEL_ARCH.MICRODAVI

    def __init__(self, dir_model, *args, **kwargs):
        # This model has no config.json. Build the hparams dict directly
        # instead of reading one from disk, then hand it to Mamba2Model.__init__
        # (which expects either a real config.json or a pre-built hparams dict
        # passed via kwargs -- check base.py's ModelBase.__init__ signature to
        # confirm "hparams" is accepted as a kwarg the same way Mamba2Model's
        # own __init__ pops it; this mirrors exactly what Mamba2Model.__init__
        # does on lines 106-115 of mamba.py).
        d_model    = 320
        n_layers   = 24
        n_heads    = 8
        d_state    = 64
        d_conv     = 4
        expand     = 2
        d_inner    = 640
        n_groups   = 1
        head_dim_mamba = 64  # d_inner / n_heads_mamba = 640/10, NOT the SSM head_dim used by Mamba2Model.set_gguf_parameters's own "head_dim" lookup -- see note below
        vocab_size = 100277

        print("DEBUG kwargs hparams:", kwargs.get("hparams"))
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = {
                "architectures": ["MicroDaviPMTModel"],
                "hidden_size": d_model,
                "num_hidden_layers": 24,
                "vocab_size": vocab_size,
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_heads,   # full MHA, no GQA
                # "head_dim": d_model // n_heads,   # attention head dim = 40
                "max_position_embeddings": 1024,  # SEQ_LEN from training notebook
                "rope_theta": 10000.0,
                "intermediate_size": 4 * d_model, # FFN hidden dim (1280), matches mlp_in/2
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
                "n_groups": n_groups,
                "mamba_d_ssm": d_inner,
                "mamba_d_state": d_state,
                "mamba_d_conv": d_conv,
                "mamba_expand": expand,
                # Mamba2Model.set_gguf_parameters looks up "mamba_d_head"/"head_dim"
                # for the SSM head dim (640/10=64). This collides in name with the
                # *attention* head_dim above -- both are called "head_dim" by
                # different pieces of code. Resolve this collision explicitly,
                # don't let find_hparam's fallback ordering silently pick the
                # wrong one. See note at bottom of file.
                "mamba_d_head": head_dim_mamba,
                "state_size": 64,
                "conv_kernel": 4,
                "head_dim": 64,
                "model_type": "microdavi",
            }
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

        self.d_model  = d_model
        self.expand   = expand
        self.d_inner  = d_inner
        self.n_group  = n_groups
        self.n_heads_attn = n_heads
        self.attn_layer_indices = ATTN_LAYER_INDICES

    def set_vocab(self):
        # Tokenizer is Xenova/gpt-4 (tiktoken-cl100k-style BPE via tokenizers
        # lib), with custom <|im_start|>/<|im_end|> ChatML special tokens added
        # on top per the training notebook. _set_vocab_gpt2() reads
        # tokenizer.json directly (BPE path) -- confirm your exported model
        # directory actually contains a tokenizer.json with those special
        # tokens already merged in (i.e. export via
        # tokenizer.save_pretrained(...) AFTER add_special_tokens, not before).
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        # Deliberately NOT calling Mamba2Model.set_gguf_parameters() via super()
        # for the whole thing, because that method unconditionally calls
        # add_head_count(0) and skips per-layer attention setup entirely, the
        # same conflict flagged for FalconH1Model. FalconH1Model dodges this by
        # calling super().set_gguf_parameters() and then overriding
        # add_head_count() with a second call afterward (see falcon_h1.py
        # lines 97-108) -- so the "last write wins" pattern is the existing
        # precedent. We replicate that here rather than skip super() entirely,
        # since Mamba2Model's SSM-param logic is what we actually want reused.
        super().set_gguf_parameters()

        # Override the head_count(0) that Mamba2Model.set_gguf_parameters set,
        # and add the real per-layer KV head vector (Jamba's pattern, see
        # jamba.py lines 36-39,46). Mamba layers -> 0, attention layers -> n_heads.
        self.gguf_writer.add_head_count(self.n_heads_attn)

        n_kv_vec = [
            self.n_heads_attn if i in self.attn_layer_indices else 0
            for i in range(self.block_count)
        ]
        self.gguf_writer.add_head_count_kv(n_kv_vec)

        self.gguf_writer.add_key_length(self.d_model // self.n_heads_attn)
        self.gguf_writer.add_value_length(self.d_model // self.n_heads_attn)
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])

        assert self.hparams.get("hidden_act") in [None, "silu"], "Only SILU activation supported"

    def modify_tensors(self, data_torch, name, bid):
        # --- Model-level tensors (no bid) ---
        if name == "embeddings.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)
            return
        if name == "final_norm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)
            return

        assert bid is not None  # everything below this point is per-layer

        # --- Attention-layer tensors ---
        if name.endswith("qkv_proj.weight"):
            d = self.d_model
            q, k, v = data_torch.split(d, dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v)
            return
        if name.endswith("out_proj.weight") and ".mamba." not in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), data_torch)
            return
        if name.endswith("ln1.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch)
            return
        if name.endswith("ln2.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_PRE_NORM, bid), data_torch)
            return
        if name.endswith("mlp_in.weight"):
            ffn_dim = self.hparams["intermediate_size"]
            gate, up = data_torch.split(ffn_dim, dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), gate)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), up)
            return
        if name.endswith("mlp_out.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid), data_torch)
            return

        # --- Mamba-layer tensors ---
        if name.endswith("layers.{}.norm.weight".format(bid)):
            # block-level pre-norm for mamba layers (ATTN_NORM slot reused, same
            # role as ln1 on attention layers -- both are "the norm right before
            # this layer's main sublayer")
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch)
            return
        if name.endswith("mamba.norm.weight"):
            reshaped = data_torch.reshape((self.n_group, self.d_inner // self.n_group))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_NORM, bid), reshaped)
            return
        if name.endswith("mamba.in_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_IN, bid), data_torch)
            return
        if name.endswith("mamba.out_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_OUT, bid), data_torch)
            return
        if name.endswith("mamba.conv1d.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), data_torch.squeeze())
            return
        if name.endswith("mamba.conv1d.bias"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".bias"), data_torch)
            return
        if name.endswith("mamba.dt_bias"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_DT, bid, suffix=".bias"), data_torch)
            return
        if name.endswith("mamba.A_log"):
            print("DEBUG hit A_log branch for", name)
            logger.debug("A_log --> A ==> " + name)
            reshaped = (-torch.exp(data_torch)).reshape((*data_torch.shape, 1))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_A, bid, suffix=""), reshaped)
            return
        if name.endswith("mamba.D"):
            reshaped = data_torch.reshape((*data_torch.shape, 1))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_D, bid, suffix=""), reshaped)
            return

        raise ValueError(f"microdavi: unhandled tensor name {name!r}")

    def modify_tensors(self, data_torch, name, bid):
        # --- Model-level tensors (no bid) ---
        if name == "embeddings.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)
            return
        if name == "final_norm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)
            return

        assert bid is not None  # everything below this point is per-layer

        # --- Attention-layer tensors ---
        if name.endswith("qkv_proj.weight"):
            d = self.d_model
            q, k, v = data_torch.split(d, dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v)
            return
        if name.endswith("out_proj.weight") and ".mamba." not in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), data_torch)
            return
        if name.endswith("ln1.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch)
            return
        if name.endswith("ln2.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_PRE_NORM, bid), data_torch)
            return
        if name.endswith("mlp_in.weight"):
            ffn_dim = self.hparams["intermediate_size"]
            gate, up = data_torch.split(ffn_dim, dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), gate)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), up)
            return
        if name.endswith("mlp_out.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid), data_torch)
            return

        # --- Mamba-layer tensors ---
        if name.endswith("layers.{}.norm.weight".format(bid)):
            # block-level pre-norm for mamba layers (ATTN_NORM slot reused, same
            # role as ln1 on attention layers -- both are "the norm right before
            # this layer's main sublayer")
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid), data_torch)
            return
        if name.endswith("mamba.norm.weight"):
            reshaped = data_torch.reshape((self.n_group, self.d_inner // self.n_group))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_NORM, bid), reshaped)
            return
        if name.endswith("mamba.in_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_IN, bid), data_torch)
            return
        if name.endswith("mamba.out_proj.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_OUT, bid), data_torch)
            return
        if name.endswith("mamba.conv1d.weight"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid), data_torch.squeeze())
            return
        if name.endswith("mamba.conv1d.bias"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".bias"), data_torch)
            return
        if name.endswith("mamba.dt_proj.bias"):
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_DT, bid, suffix=".bias"), data_torch)
            return
        if name.endswith("mamba.A_log"):
            logger.debug("A_log --> A ==> " + name)
            reshaped = (-torch.exp(data_torch)).reshape((*data_torch.shape, 1))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_A, bid, suffix=""), reshaped)
            return
        if name.endswith("mamba.D"):
            reshaped = data_torch.reshape((*data_torch.shape, 1))
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_D, bid, suffix=""), reshaped)
            return

        raise ValueError(f"microdavi: unhandled tensor name {name!r}")