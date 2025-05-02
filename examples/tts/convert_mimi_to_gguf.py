import gguf
import argparse
import logging
import torch
from typing import Union
from pathlib import Path
from torch import Tensor
from transformers import MimiModel, PreTrainedModel

logger = logging.getLogger("mimi")


class MimiModelConverter:
    mimi_model: PreTrainedModel
    gguf_writer: gguf.GGUFWriter
    fname_out: Path
    ftype: gguf.LlamaFileType

    def __init__(self,
                 pretrained_model_name_or_path: Union[Path, str],
                 fname_out: Path,
                 ftype: gguf.LlamaFileType,
                 is_big_endian: bool,):
        self.mimi_model = MimiModel.from_pretrained(pretrained_model_name_or_path)
        self.fname_out = fname_out
        self.ftype = ftype
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch="this model cannot be used as LLM, use it via --model-vocoder in TTS examples",
            endianess=endianess)

        assert self.mimi_model.config.architectures[0] == "MimiModel"

        # load tensors
        for name, data_torch in self.mimi_model.state_dict().items():
            # convert any unsupported data types to float32
            old_dtype = data_torch.dtype
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch, old_dtype)

    def add_tensor(self, name: str, data_torch: Tensor, old_dtype: torch.dtype):
        is_1d = len(data_torch.shape) == 1
        is_bias = ".bias" in name
        can_quantize = not is_1d and not is_bias
        data_qtype = gguf.GGMLQuantizationType.F32

        n_head = self.mimi_model.config.num_attention_heads
        n_kv_head = self.mimi_model.config.num_key_value_heads
        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = self.undo_permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = self.undo_permute(data_torch, n_head, n_kv_head)

        # process codebook
        if ".codebook.initialized" in name:
            # "initialized" tensor
            state_dict = self.mimi_model.state_dict()
            embed_sum = state_dict[name.replace(".initialized", ".embed_sum")]
            cluster_usage = state_dict[name.replace(".initialized", ".cluster_usage")]
            # see modeling_mimi.py --> MimiEuclideanCodebook
            data_torch = embed_sum / cluster_usage.clamp(min=self.mimi_model.config.norm_eps)[:, None]
            name = name.replace(".initialized", "")

        # ignore processed tensors
        if ".cluster_usage" in name or ".embed_sum" in name:
            return

        # transpose some tensors
        if ".conv.bias" in name:
            data_torch = data_torch.view((1, data_torch.shape[0]))
            data_torch = data_torch.transpose(0, 1)

        # change view 3d to 2d
        if "quantizer" in name and "_proj." in name:
            assert data_torch.shape[2] == 1
            data_torch = data_torch.view((data_torch.shape[0], data_torch.shape[1]))

        # shorten name, otherwise it will be too long for ggml to read
        name = name.replace("_residual_vector_quantizer", "_rvq")

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

        # Conv kernels are always F16
        if ".conv.weight" in name:
            data_qtype = gguf.GGMLQuantizationType.F16

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

    @staticmethod
    def undo_permute(weights: Tensor, n_head: int, n_head_kv: int):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mimi safetensors model to GGUF",)
    parser.add_argument(
        "--outfile", type=Path, default="kyutai-mimi.gguf",
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
        help="directory or model ID containing model file (if model ID is specified, download from Hugging Face hub)",
        nargs="?",
        default="kyutai/mimi",
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

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    logger.info(f"Loading model: {dir_model}")

    with torch.inference_mode():
        converter = MimiModelConverter(
            pretrained_model_name_or_path=dir_model,
            fname_out=args.outfile,
            ftype=ftype_map[args.outtype],
            is_big_endian=args.bigendian,
        )
        converter.write()


if __name__ == '__main__':
    main()

