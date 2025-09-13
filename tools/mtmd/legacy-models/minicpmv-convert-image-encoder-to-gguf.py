# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Siglip model. """
# Copied from  HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit and add tgt_sizes


import os
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    logging,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

class SiglipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    Example:
    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel
    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()
    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

_CHECKPOINT_FOR_DOC = "google/siglip-base-patch16-224"

SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/siglip-base-patch16-224",
    # See all SigLIP models at https://huggingface.co/models?filter=siglip
]

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        # The `erfinv_` op is not (yet?) defined in float16+cpu, bfloat16+gpu
        og_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        tensor.erfinv_()
        tensor = tensor.to(og_dtype)
    else:
        tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    if tensor.dtype == torch.float16:
        # The `clamp_` op is not (yet?) defined in float16+cpu
        tensor = tensor.to(torch.float32)
        tensor.clamp_(min=a, max=b)
        tensor = tensor.to(torch.float16)
    else:
        tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.
    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    denom = fan_in
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->Siglip
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.self_attn = (
            SiglipAttention(config)
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SiglipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, SiglipVisionEmbeddings):
            width = self.config.hidden_size
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.normal_(module.q_proj.weight)
            nn.init.normal_(module.k_proj.weight)
            nn.init.normal_(module.v_proj.weight)
            nn.init.normal_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.normal_(module.fc1.weight)
            nn.init.normal_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SIGLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`SiglipVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SIGLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->Siglip
class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].
    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class SiglipVisionTransformer(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

import argparse
import json
import re

import numpy as np
from gguf import *
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig

TEXT = "clip.text"
VISION = "clip.vision"


def add_key_str(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def should_skip_tensor(name: str, has_text: bool, has_vision: bool, has_minicpmv: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    if has_minicpmv and name in ["visual_projection.weight"]:
        return True

    if name.startswith("v") and not has_vision:
        return True

    if name.startswith("t") and not has_text:
        return True

    return False


def get_tensor_name(name: str) -> str:
    if "projection" in name:
        return name
    if "mm_projector" in name:
        name = name.replace("model.mm_projector", "mm")
        name = re.sub(r'mm\.mlp\.mlp', 'mm.model.mlp', name, count=1)
        name = re.sub(r'mm\.peg\.peg', 'mm.model.peg', name, count=1)
        return name

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", required=True)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
ap.add_argument("--text-only", action="store_true", required=False,
                help="Save a text-only model. It can't be used to encode images")
ap.add_argument("--vision-only", action="store_true", required=False,
                help="Save a vision-only model. It can't be used to encode texts")
ap.add_argument("--clip-model-is-vision", action="store_true", required=False,
                help="The clip model is a pure vision model (ShareGPT4V vision extract for example)")
ap.add_argument("--clip-model-is-openclip", action="store_true", required=False,
                help="The clip model is from openclip (for ViT-SO400M type))")
ap.add_argument("--minicpmv-projector", help="Path to minicpmv.projector file. If specified, save an image encoder for MiniCPM-V models.")
ap.add_argument("--projector-type", help="Type of projector. Possible values: mlp, ldp, ldpv2", choices=["mlp", "ldp", "ldpv2"], default="mlp")
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)
# Example --image_mean 0.48145466 0.4578275 0.40821073 --image_std 0.26862954 0.26130258 0.27577711
# Example --image_mean 0.5 0.5 0.5 --image_std 0.5 0.5 0.5
default_image_mean = [0.5, 0.5, 0.5]
default_image_std = [0.5, 0.5, 0.5]
ap.add_argument('--image-mean', type=float, nargs='+', help='Mean of the images for normalization (overrides processor) ', default=None)
ap.add_argument('--image-std', type=float, nargs='+', help='Standard deviation of the images for normalization (overrides processor)', default=None)
ap.add_argument('--minicpmv_version', type=int, help='minicpmv_version: MiniCPM-V-2 use 1; MiniCPM-V-2.5 use 2; MiniCPM-V-2.6 use 3; MiniCPM-o-2.6 use 4; MiniCPM-V 4.0 use 5; MiniCPM-o-4.0 use 6', default=2)

# with proper
args = ap.parse_args()


if args.text_only and args.vision_only:
    print("--text-only and --image-only arguments cannot be specified at the same time.")
    exit(1)

if args.use_f32:
    print("WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")

# output in the same directory as the model if output_dir is None
dir_model = args.model_dir

# Read config.json to get actual model configuration
config_path = os.path.join(dir_model, "config.json")
model_config = {}
if os.path.isfile(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    print(f"Loaded config from {config_path}")
else:
    print(f"Warning: config.json not found at {config_path}")

# If minicpmv_projector is not specified but the default path exists, use the default path
if args.minicpmv_projector is None:
    default_projector_path = os.path.join(dir_model, "minicpmv.projector")
    if os.path.isfile(default_projector_path):
        args.minicpmv_projector = default_projector_path
        print(f"Found default projector file: {default_projector_path}")

# If output_dir is not specified, use model_dir as the default value
if args.output_dir is None:
    args.output_dir = dir_model

if args.clip_model_is_vision or not os.path.exists(dir_model + "/vocab.json") or args.clip_model_is_openclip:
    vocab = None
    tokens = None
else:
    with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        tokens = [key for key in vocab]

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if args.use_f32:
    ftype = 0

# if args.clip_model_is_vision or args.clip_model_is_openclip:
#     model = CLIPVisionModel.from_pretrained(dir_model)
#     processor = None
# else:
#     model = CLIPModel.from_pretrained(dir_model)
#     processor = CLIPProcessor.from_pretrained(dir_model)

minicpmv_version = args.minicpmv_version

# Use actual config values instead of hardcoded ones
if model_config:
    # For the projector/resampler, use the main model's hidden_size
    emb_dim = model_config.get("hidden_size", 1536)

    # For the vision model, use vision_config values
    vision_config_dict = model_config.get("vision_config", {})
    default_vision_config = {
        "hidden_size": vision_config_dict.get("hidden_size", 1152),
        "image_size": vision_config_dict.get("image_size", 980),
        "intermediate_size": vision_config_dict.get("intermediate_size", 4304),
        "model_type": vision_config_dict.get("model_type", "siglip"),
        "num_attention_heads": vision_config_dict.get("num_attention_heads", 16),
        "num_hidden_layers": vision_config_dict.get("num_hidden_layers", 27),
        "patch_size": vision_config_dict.get("patch_size", 14),
    }

    # Use vision model's num_hidden_layers for block_count
    block_count = vision_config_dict.get("num_hidden_layers", 27)

    print(f"Using config values: emb_dim={emb_dim}, block_count={block_count}")
    print(f"Vision config: {default_vision_config}")
else:
    # Fallback to original hardcoded logic if config.json not found
    emb_dim = 4096
    block_count = 26
    if minicpmv_version == 1:
        emb_dim = 2304
        block_count = 26
    elif minicpmv_version == 2:
        emb_dim = 4096
        block_count = 27
    elif minicpmv_version == 3:
        emb_dim = 3584
        block_count = 27
    elif minicpmv_version == 4:
        emb_dim = 3584
        block_count = 27
    elif minicpmv_version == 5:
        emb_dim = 2560
        block_count = 27
    elif minicpmv_version == 6:
        emb_dim = 4096
        block_count = 27

    default_vision_config = {
            "hidden_size": 1152,
            "image_size": 980,
            "intermediate_size": 4304,
            "model_type": "idefics2",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14,
        }

vision_config = Idefics2VisionConfig(**default_vision_config)
model = Idefics2VisionTransformer(vision_config)
if minicpmv_version == 3 or (model_config and model_config.get("vision_config", {}).get("model_type") == "siglip"):
    vision_config = SiglipVisionConfig(**default_vision_config)
    model = SiglipVisionTransformer(vision_config)
elif minicpmv_version == 4:
    vision_config = SiglipVisionConfig(**default_vision_config)
    model = SiglipVisionTransformer(vision_config)
elif minicpmv_version == 5:
    default_vision_config["model_type"] = "siglip_vision_model"
    vision_config = SiglipVisionConfig(**default_vision_config)
    model = SiglipVisionTransformer(vision_config)
elif minicpmv_version == 6:
    default_vision_config["model_type"] = "siglip_vision_model"
    vision_config = SiglipVisionConfig(**default_vision_config)
    model = SiglipVisionTransformer(vision_config)

processor = None
# if model.attn_pool is not None:
#     model.attn_pool = torch.nn.Identity()

# model.blocks = model.blocks[:-1]
model.load_state_dict(torch.load(os.path.join(dir_model, "minicpmv.clip")))

fname_middle = None
has_text_encoder = True
has_vision_encoder = True
has_minicpmv_projector = False

if args.text_only:
    fname_middle = "text-"
    has_vision_encoder = False
elif args.minicpmv_projector is not None:
    fname_middle = "mmproj-"
    has_text_encoder = False
    has_minicpmv_projector = True
elif args.vision_only:
    fname_middle = "vision-"
    has_text_encoder = False
else:
    fname_middle = ""

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(output_dir).replace("ggml_", "")
fname_out = os.path.join(output_dir, f"{fname_middle}model-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch="clip")

fout.add_bool("clip.has_text_encoder", has_text_encoder)
fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
fout.add_bool("clip.has_minicpmv_projector", has_minicpmv_projector)
fout.add_file_type(ftype)
if args.text_only:
    fout.add_description("text-only CLIP model")
elif args.vision_only and not has_minicpmv_projector:
    fout.add_description("vision-only CLIP model")
elif has_minicpmv_projector:
    fout.add_description("image encoder for MiniCPM-V")
    # add projector type
    fout.add_string("clip.projector_type", "resampler")
    fout.add_int32("clip.minicpmv_version", minicpmv_version)
else:
    fout.add_description("two-tower CLIP model")

if has_vision_encoder:
    # vision_model hparams - use actual config values
    vision_image_size = model_config.get("image_size", 448) if model_config else 448
    vision_patch_size = default_vision_config.get("patch_size", 14)
    vision_hidden_size = default_vision_config.get("hidden_size", 1152)
    vision_intermediate_size = default_vision_config.get("intermediate_size", 4304)
    vision_attention_heads = default_vision_config.get("num_attention_heads", 16)

    fout.add_uint32("clip.vision.image_size", vision_image_size)
    fout.add_uint32("clip.vision.patch_size", vision_patch_size)
    fout.add_uint32(add_key_str(KEY_EMBEDDING_LENGTH, VISION), vision_hidden_size)
    fout.add_uint32(add_key_str(KEY_FEED_FORWARD_LENGTH, VISION), vision_intermediate_size)
    fout.add_uint32("clip.vision.projection_dim", 0)
    fout.add_uint32(add_key_str(KEY_ATTENTION_HEAD_COUNT, VISION), vision_attention_heads)
    fout.add_float32(add_key_str(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    fout.add_uint32(add_key_str(KEY_BLOCK_COUNT, VISION), block_count)

    # Add MiniCPM-V specific parameters
    query_num = model_config.get("query_num", 0) if model_config else 0
    resampler_emb_dim = model_config.get("hidden_size", 0) if model_config else 0
    fout.add_uint32("clip.minicpmv_query_num", query_num)

    if processor is not None:
        image_mean = processor.image_processor.image_mean if args.image_mean is None or args.image_mean == default_image_mean else args.image_mean
        image_std = processor.image_processor.image_std if args.image_std is None or args.image_std == default_image_std else args.image_std
    else:
        image_mean = args.image_mean if args.image_mean is not None else default_image_mean
        image_std = args.image_std if args.image_std is not None else default_image_std
    fout.add_array("clip.vision.image_mean", image_mean)
    fout.add_array("clip.vision.image_std", image_std)

use_gelu = True
fout.add_bool("clip.use_gelu", use_gelu)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def _replace_name_resampler(s, v):
    if re.match("resampler.pos_embed", s):
        return {
            s: v,
            re.sub("pos_embed", "pos_embed_k", s): torch.from_numpy(get_2d_sincos_pos_embed(emb_dim, (70, 70))),
        }
    if re.match("resampler.proj", s):
        return {
            re.sub("proj", "pos_embed_k", s): torch.from_numpy(get_2d_sincos_pos_embed(emb_dim, (70, 70))),
            re.sub("proj", "proj.weight", s): v.transpose(-1, -2).contiguous(),
        }
    if re.match("resampler.attn.in_proj_.*", s):
        return {
            re.sub("attn.in_proj_", "attn.q.", s): v.chunk(3, dim=0)[0],
            re.sub("attn.in_proj_", "attn.k.", s): v.chunk(3, dim=0)[1],
            re.sub("attn.in_proj_", "attn.v.", s): v.chunk(3, dim=0)[2],
        }
    return {s: v}

if has_minicpmv_projector:
    projector = torch.load(args.minicpmv_projector)
    new_state_dict = {}
    for k, v in projector.items():
        kvs = _replace_name_resampler(k, v)
        for nk, nv in kvs.items():
            new_state_dict[nk] = nv
    projector = new_state_dict
    ftype_cur = 0
    for name, data in projector.items():
        name = get_tensor_name(name)
        data = data.squeeze().numpy()

        n_dims = len(data.shape)
        if ftype == 1:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        fout.add_tensor(name, data)
        print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")

    print("Projector tensors added\n")

def _replace_name(s, v):
    s = "vision_model." + s
    if re.match("vision_model.embeddings.position_embedding", s):
        v = v.unsqueeze(0)
        return {s: v}

    return {s: v}

state_dict = model.state_dict()
new_state_dict = {}
for k, v in state_dict.items():
    kvs = _replace_name(k, v)
    for nk, nv in kvs.items():
        new_state_dict[nk] = nv
state_dict = new_state_dict
for name, data in state_dict.items():
    if should_skip_tensor(name, has_text_encoder, has_vision_encoder, has_minicpmv_projector):
        # we don't need this
        print(f"skipping parameter: {name}")
        continue

    name = get_tensor_name(name)
    data = data.squeeze().numpy()

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if n_dims == 4:
        print(f"tensor {name} is always saved in f16")
        data = data.astype(np.float16)
        ftype_cur = 1
    elif ftype == 1:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name, data)


fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)
