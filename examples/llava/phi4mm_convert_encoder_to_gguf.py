import gguf
import argparse
import logging
import sys
import torch
import json
import os
import numpy as np
from typing import cast, ContextManager, Any, Iterator
from pathlib import Path
from torch import Tensor

logger = logging.getLogger("phi4-mmproj")


# https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/modeling_phi4mm.py
# https://huggingface.co/google/siglip-base-patch16-224/blob/main/preprocessor_config.json
# https://github.com/EricLBuehler/mistral.rs/pull/1163/files
SIGLIP_MODEL = {
    "model_id": "google/siglip-base-patch16-224",
    "image_size": 448,
    "patch_size": 14, # I had very had time finding this number
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "SiglipProcessor",
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "height": 224,
        "width": 224
    }
}
N_LAYERS = 27
FEATURE_LAYER = -2
HEAD_COUNT = 16


# (copied from convert_hf_to_gguf.py)
# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


class Phi4MM:
    hparams: dict
    gguf_writer: gguf.GGUFWriter
    fname_out: Path
    ftype: gguf.LlamaFileType

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)
        
    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)
        part_names.sort()
        return part_names

    def __init__(self,
                 dir_model: Path,
                 fname_out: Path,
                 ftype: gguf.LlamaFileType,
                 is_big_endian: bool,):
        hparams = Phi4MM.load_hparams(dir_model)
        self.hparams = hparams
        self.fname_out = fname_out
        self.ftype = ftype
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="clip", endianess=endianess)
        self.gguf_writer.add_string ("clip.projector_type",              "mlp")
        self.gguf_writer.add_bool   ("clip.has_text_encoder",            False)
        self.gguf_writer.add_bool   ("clip.has_vision_encoder",          True)
        self.gguf_writer.add_bool   ("clip.has_llava_projector",         False)
        self.gguf_writer.add_uint32 ("clip.vision.image_size",           SIGLIP_MODEL["image_size"])
        self.gguf_writer.add_uint32 ("clip.vision.patch_size",           SIGLIP_MODEL["patch_size"])
        self.gguf_writer.add_uint32 ("clip.vision.embedding_length",     1152)
        self.gguf_writer.add_uint32 ("clip.vision.feed_forward_length",  4304)
        self.gguf_writer.add_uint32 ("clip.vision.projection_dim",       hparams["hidden_size"])
        self.gguf_writer.add_uint32 ("clip.vision.block_count",          N_LAYERS)
        self.gguf_writer.add_uint32 ("clip.vision.attention.head_count", HEAD_COUNT)
        self.gguf_writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        self.gguf_writer.add_array  ("clip.vision.image_mean",           SIGLIP_MODEL["image_mean"])
        self.gguf_writer.add_array  ("clip.vision.image_std",            SIGLIP_MODEL["image_std"])
        self.gguf_writer.add_bool   ("clip.use_gelu",                    False)
        self.gguf_writer.add_array  ("clip.vision.feature_layer",        [N_LAYERS + FEATURE_LAYER])

        # load tensors
        for name, data_torch in self.get_tensors(dir_model):
            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch)

    def get_tensors(self, dir_model: Path) -> Iterator[tuple[str, Tensor]]:
        part_names = Phi4MM.get_model_part_names(dir_model, "model", ".safetensors")
        tensor_names_from_parts: set[str] = set()
        for part_name in part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            from safetensors import safe_open
            ctx = cast(ContextManager[Any], safe_open(dir_model / part_name, framework="pt", device="cpu"))
            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    data = model_part.get_slice(name)
                    data = LazyTorchTensor.from_safetensors_slice(data)
                    yield name, data

    def add_tensor(self, name: str, data_torch: Tensor):
        if not name.startswith("model.embed_tokens_extend.image_embed.") \
                or "img_processor.head." in name \
                or "glb_GN" in name \
                or "sub_GN" in name:
            return  # skip

        is_1d = len(data_torch.shape) == 1
        is_embd = ".embeddings." in name
        old_dtype = data_torch.dtype
        can_quantize = not is_1d and not is_embd
        data_qtype = gguf.GGMLQuantizationType.F32

        # prefix
        name = name.replace("model.embed_tokens_extend.image_embed.img_processor.", "")
        name = name.replace("encoder.", "v.")
        name = name.replace("layers.", "blk.")
        # projector and input embd
        name = name.replace("embeddings.patch_embedding.", "v.patch_embd.")
        name = name.replace("embeddings.position_embedding.", "v.position_embd.")
        name = name.replace("post_layernorm.", "post_ln.")
        # each block
        name = name.replace(".self_attn.k_proj.", ".attn_k.")
        name = name.replace(".self_attn.v_proj.", ".attn_v.")
        name = name.replace(".self_attn.q_proj.", ".attn_q.")
        name = name.replace(".self_attn.out_proj.", ".attn_out.")
        name = name.replace(".layer_norm1.", ".ln1.")
        name = name.replace(".layer_norm2.", ".ln2.")
        name = name.replace(".mlp.fc1.", ".ffn_down.")
        name = name.replace(".mlp.fc2.", ".ffn_up.")
        # projector
        name = name.replace("model.embed_tokens_extend.image_embed.img_projection.", "mm.")

        if can_quantize:
            if self.ftype == gguf.LlamaFileType.ALL_F32:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                data_qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                data_qtype = gguf.GGMLQuantizationType.BF16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                data_qtype = gguf.GGMLQuantizationType.Q8_0
            else:
                raise ValueError(f"Unsupported file type: {self.ftype}")
        data = data_torch.numpy()

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except Exception as e:
            logger.error(f"Error quantizing tensor '{name}': {e}, fallback to F16")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        # reverse shape to make it similar to the internal ggml dimension order
        shape_str = f"{{{', '.join(str(n) for n in reversed(data_torch.shape))}}}"
        logger.info(f"{f'%-32s' % f'{name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        self.gguf_writer.add_tensor(name, data, raw_dtype=data_qtype)

    def write(self):
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Phi 4 vision encoder safetensors to GGUF format",)
    parser.add_argument(
        "--outfile", type=Path, default="mmproj.gguf",
        help="path to write mmproj file to",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0"], default="f16",
        help="output format",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
        nargs="?",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )

    args = parser.parse_args()
    if args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir_model = args.model

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    logger.info(f"Loading model: {dir_model.name}")

    with torch.inference_mode():
        phi4_mm = Phi4MM(
            dir_model=dir_model,
            fname_out=args.outfile,
            ftype=ftype_map[args.outtype],
            is_big_endian=args.bigendian,
        )
        phi4_mm.write()


if __name__ == '__main__':
    main()
