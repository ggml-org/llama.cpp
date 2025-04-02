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

logger = logging.getLogger("gemma3-mmproj")


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


class Gemma3VisionTower:
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
        hparams = Gemma3VisionTower.load_hparams(dir_model)
        self.hparams = hparams
        self.fname_out = fname_out
        self.ftype = ftype
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="clip", endianess=endianess)

        text_config = hparams["text_config"]
        vision_config = hparams["vision_config"]

        assert hparams["architectures"][0] == "Gemma3ForConditionalGeneration"
        assert text_config is not None
        assert vision_config is not None

        self.gguf_writer.add_string ("clip.projector_type",              "gemma3")
        self.gguf_writer.add_bool   ("clip.has_text_encoder",            False)
        self.gguf_writer.add_bool   ("clip.has_vision_encoder",          True)
        self.gguf_writer.add_bool   ("clip.has_llava_projector",         False) # legacy
        self.gguf_writer.add_uint32 ("clip.vision.image_size",           vision_config["image_size"])
        self.gguf_writer.add_uint32 ("clip.vision.patch_size",           vision_config["patch_size"])
        self.gguf_writer.add_uint32 ("clip.vision.embedding_length",     vision_config["hidden_size"])
        self.gguf_writer.add_uint32 ("clip.vision.feed_forward_length",  vision_config["intermediate_size"])
        self.gguf_writer.add_uint32 ("clip.vision.projection_dim",       text_config["hidden_size"])
        self.gguf_writer.add_uint32 ("clip.vision.block_count",          vision_config["num_hidden_layers"])
        self.gguf_writer.add_uint32 ("clip.vision.attention.head_count", vision_config["num_attention_heads"])
        self.gguf_writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_config.get("layer_norm_eps", 1e-6))
        # default values taken from HF tranformers code
        self.gguf_writer.add_array  ("clip.vision.image_mean", [0.5, 0.5, 0.5])
        self.gguf_writer.add_array  ("clip.vision.image_std",  [0.5, 0.5, 0.5])
        self.gguf_writer.add_bool   ("clip.use_gelu", True)

        # load tensors
        for name, data_torch in self.get_tensors(dir_model):
            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch)

    def get_tensors(self, dir_model: Path) -> Iterator[tuple[str, Tensor]]:
        part_names = Gemma3VisionTower.get_model_part_names(dir_model, "model", ".safetensors")
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
        is_1d = len(data_torch.shape) == 1
        is_embd = ".embeddings." in name
        old_dtype = data_torch.dtype
        can_quantize = not is_1d and not is_embd
        data_qtype = gguf.GGMLQuantizationType.F32

        # this is to support old checkpoint
        # TODO: remove this when we have the final model
        name = name.replace("vision_model.vision_model.", "vision_tower.vision_model.")
        name = name.replace("multimodal_projector.", "multi_modal_projector.")

        # filter only vision tensors
        if not name.startswith("vision_tower.vision_model.") and not name.startswith("multi_modal_projector."):
            return
        # prefix
        name = name.replace("vision_tower.vision_model.encoder.layers.", "v.blk.")
        name = name.replace("vision_tower.vision_model.", "v.")
        # projector and input embd
        name = name.replace(".embeddings.patch_embedding.", ".patch_embd.")
        name = name.replace(".embeddings.position_embedding.", ".position_embd.")
        name = name.replace(
            "multi_modal_projector.mm_input_projection_weight",
            "mm.input_projection.weight"
        )
        name = name.replace(
            "multi_modal_projector.mm_soft_emb_norm.weight",
            "mm.soft_emb_norm.weight"
        )
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

        # corrent norm value ; only this "soft_emb_norm" need to be corrected as it's part of Gemma projector
        # the other norm values are part of SigLIP model, and they are already correct
        # ref code: Gemma3RMSNorm
        if "soft_emb_norm.weight" in name:
            logger.info(f"Correcting norm value for '{name}'")
            data_torch = data_torch + 1

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
        description="Convert Gemma 3 vision tower safetensors to GGUF format",)
    parser.add_argument(
        "--outfile", type=Path, default="mmproj.gguf",
        help="path to write to",
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
        gemma3_vision_tower = Gemma3VisionTower(
            dir_model=dir_model,
            fname_out=args.outfile,
            ftype=ftype_map[args.outtype],
            is_big_endian=args.bigendian,
        )
        gemma3_vision_tower.write()


if __name__ == '__main__':
    main()

