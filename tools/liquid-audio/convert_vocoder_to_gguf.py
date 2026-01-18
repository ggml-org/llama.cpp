from pathlib import Path
from safetensors import safe_open
from torch import Tensor
from typing import Union
import argparse
import gguf
import json
import logging
import torch

logger = logging.getLogger()


class Lfm2AudioDecoderModelConverter:
    mimi_tensors: dict()
    gguf_writer: gguf.GGUFWriter
    fname_out: Path
    ftype: gguf.LlamaFileType
    decoder_tensors: dict()

    def __init__(
        self,
        pretrained_path: Union[Path, str],
        fname_out: Path,
        ftype: gguf.LlamaFileType,
    ):
        self.fname_out = fname_out
        self.ftype = ftype
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch="this model cannot be used as LLM, use it via --model-vocoder in TTS examples",
            endianess=gguf.GGUFEndian.LITTLE,
        )

        self.decoder_tensors = self.load_tensors(
            pretrained_path / "model.safetensors",
            Lfm2AudioDecoderModelConverter._is_decoder_tensor,
        )

        self.detokenizer_tensors = self.load_tensors(
            pretrained_path / "audio_detokenizer" / "model.safetensors", lambda _: True
        )

        for name, data_torch in (
            self.detokenizer_tensors | self.decoder_tensors
        ).items():
            # convert any unsupported data types to float32
            old_dtype = data_torch.dtype
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch, old_dtype)

        # populate config entries
        with open(pretrained_path / "config.json", "r", encoding="utf-8") as f:
            config_json = json.load(f)
            assert config_json["architectures"] == ["Lfm2AudioForConditionalGeneration"]
            self.gguf_writer.add_uint32(
                "depthformer_n_layer",
                config_json["depthformer"]["layers"],
            )
            self.gguf_writer.add_uint32(
                "depthformer_n_embd",
                config_json["depthformer"]["dim"],
            )

    def load_tensors(self, path, predicate):
        tensors = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if predicate(key):
                    tensors[key] = f.get_tensor(key)
        return tensors

    def add_tensor(self, name: str, data_torch: Tensor, old_dtype: torch.dtype):
        if name.startswith("lfm") or name.startswith("lin"):
            return

        is_1d = len(data_torch.shape) == 1
        is_bias = ".bias" in name
        can_quantize = not is_1d and not is_bias
        data_qtype = gguf.GGMLQuantizationType.F32

        # conv kernels are always F32
        if ".conv.weight" in name:
            data_torch = data_torch.squeeze(1)
            can_quantize = False

        # shorten name, otherwise it will be too long for ggml to read
        name = name.replace("bounded_attention", "attention")

        if can_quantize:
            if self.ftype == gguf.LlamaFileType.ALL_F32:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                data_qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                data_qtype = gguf.GGMLQuantizationType.BF16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                data_qtype = gguf.GGMLQuantizationType.Q8_0
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q4_0:
                data_qtype = gguf.GGMLQuantizationType.Q4_0
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
        logger.info(
            f"{'%-32s' % f'{name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
        )

        self.gguf_writer.add_tensor(name, data, raw_dtype=data_qtype)

    @staticmethod
    def _is_decoder_tensor(key):
        audio_out_tensor_prefixes = [
            "depthformer",
            "depth_embeddings",
            "depth_linear",
            "audio_embedding",
        ]
        return any(key.startswith(p) for p in audio_out_tensor_prefixes)

    def write(self):
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LFM2-Audio decoder model to GGUF",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        required=True,
        help="path to write to",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "q4_0"],
        default="f16",
        help="output format",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to LFM2-Audio model",
        nargs="?",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
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
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }

    logger.info(f"Loading model: {dir_model}")

    with torch.inference_mode():
        converter = Lfm2AudioDecoderModelConverter(
            pretrained_path=dir_model,
            fname_out=args.outfile,
            ftype=ftype_map[args.outtype],
        )
        converter.write()


if __name__ == "__main__":
    main()
