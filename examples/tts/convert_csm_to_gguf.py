import os
import sys
import argparse
import logging
import torch
from safetensors.torch import load_file
from typing import Union, Any, Dict
from pathlib import Path
from torch import Tensor
from huggingface_hub import hf_hub_download

cur_path = sys.path
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'gguf-py'))
import gguf

sys.path = cur_path

logger = logging.getLogger("csm")


# This converts directly one safetensors file to 2 GGUFs
# It is easier to do this way, rather than convert to 2 smaller HF models and then convert to GGUF
# This is because the Sesame model does not have built-in tokenizer

def get_field_data(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None

# copied from https://github.com/SesameAILabs/csm/blob/main/models.py
class Llama_3_2_1B:
    vocab_size=128_256
    num_layers=16
    num_heads=32
    num_kv_heads=8
    embed_dim=2048
    max_seq_len=2048
    intermediate_dim=8192
    attn_dropout=0.0
    norm_eps=1e-5
    rope_base=500_000
    scale_factor=32

    def write_gguf_metadata(self, fout: gguf.GGUFWriter, fvocab: gguf.GGUFReader):
        arch = get_field_data(fvocab, gguf.Keys.General.ARCHITECTURE)
        assert arch == "llama"
        fout.add_type("model")
        fout.add_block_count(self.num_layers)
        fout.add_context_length(self.max_seq_len)
        fout.add_feed_forward_length(self.intermediate_dim)
        fout.add_embedding_length(self.embed_dim)
        # attn
        fout.add_head_count(self.num_heads)
        fout.add_head_count_kv(self.num_kv_heads)
        fout.add_rope_freq_base(self.rope_base)
        # fout.add_rope_scaling_factor(self.scale_factor) # breaks if this is added
        fout.add_rope_dimension_count(self.embed_dim // self.num_heads)
        fout.add_layer_norm_rms_eps(self.norm_eps)
        fout.add_key_length(self.embed_dim // self.num_heads)
        fout.add_value_length(self.embed_dim // self.num_heads)
        # vocab
        fout.add_vocab_size(self.vocab_size)
        fout.add_tokenizer_model(get_field_data(fvocab, gguf.Keys.Tokenizer.MODEL))
        fout.add_tokenizer_pre(get_field_data(fvocab, gguf.Keys.Tokenizer.PRE))
        fout.add_token_list(get_field_data(fvocab, gguf.Keys.Tokenizer.LIST)[:self.vocab_size])
        fout.add_token_types(get_field_data(fvocab, gguf.Keys.Tokenizer.TOKEN_TYPE)[:self.vocab_size])
        fout.add_token_merges(get_field_data(fvocab, gguf.Keys.Tokenizer.MERGES))
        fout.add_bos_token_id(get_field_data(fvocab, gguf.Keys.Tokenizer.BOS_ID))
        fout.add_eos_token_id(get_field_data(fvocab, gguf.Keys.Tokenizer.EOS_ID))

class Llama_3_2_100M(Llama_3_2_1B):
    vocab_size=65_632 #128_256
    num_layers=4
    num_heads=8
    num_kv_heads=2
    embed_dim=1024
    max_seq_len=2048
    intermediate_dim=8192
    attn_dropout=0.0
    norm_eps=1e-5
    rope_base=500_000
    scale_factor=32

class CSMModelConverter:
    state_dict: Dict[str, Tensor]
    gguf_writer_backbone: gguf.GGUFWriter
    gguf_writer_decoder: gguf.GGUFWriter
    gguf_reader_vocab: gguf.GGUFReader
    fname_out: Path
    ftype: gguf.LlamaFileType

    def __init__(self,
                 safetensors_path: Union[Path, str],
                 path_to_vocab_gguf: Path,
                 fname_out: Path,
                 ftype: gguf.LlamaFileType,
                 is_big_endian: bool,):

        if "<component>" not in fname_out.name:
            raise ValueError("Output file name must contain '<component>' placeholder, for example: 'sesame-csm-<component>.gguf'")

        self.state_dict = load_file(safetensors_path, device="cpu")
        self.fname_out = fname_out
        self.ftype = ftype
        self.gguf_reader_vocab = gguf.GGUFReader(path_to_vocab_gguf)
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE

        # backbone
        self.gguf_writer_backbone = gguf.GGUFWriter(
            path=None,
            arch="llama-csm",
            endianess=endianess)

        # decoder
        self.gguf_writer_decoder = gguf.GGUFWriter(
            path=None,
            arch="llama-csm",
            endianess=endianess)

        Llama_3_2_1B().write_gguf_metadata(self.gguf_writer_backbone, self.gguf_reader_vocab)
        Llama_3_2_100M().write_gguf_metadata(self.gguf_writer_decoder, self.gguf_reader_vocab)

        # load tensors
        for component in ("backbone", "decoder"):
            print()
            print(f"Converting {component}...")
            print()
            for name, data_torch in self.state_dict.items():
                # convert any unsupported data types to float32
                old_dtype = data_torch.dtype
                if data_torch.dtype not in (torch.float16, torch.float32):
                    data_torch = data_torch.to(torch.float32)
                self.add_tensor(name, data_torch, old_dtype, component)

    def add_tensor(self, name: str, data_torch: Tensor, old_dtype: torch.dtype, component: str):
        is_1d = len(data_torch.shape) == 1
        #is_embd = "_embeddings" in name
        can_quantize = not is_1d #and not is_embd
        data_qtype = gguf.GGMLQuantizationType.F32

        is_backbone = False
        is_decoder = False

        def rename_transformer(name: str) -> str:
            # transformer
            name = name.replace(".scale", ".weight")
            name = name.replace("attn.k_proj", "attn_k")
            name = name.replace("attn.q_proj", "attn_q")
            name = name.replace("attn.v_proj", "attn_v")
            name = name.replace("attn.output_proj", "attn_output")
            name = name.replace("sa_norm", "attn_norm")
            name = name.replace("mlp.w1", "ffn_gate")
            name = name.replace("mlp.w2", "ffn_down")
            name = name.replace("mlp.w3", "ffn_up")
            name = name.replace("mlp_norm", "ffn_norm")
            return name

        if "audio_embeddings." in name:
            is_decoder = True
            name = name.replace("audio_embeddings.", "audio_embd.")

        elif "text_embeddings." in name:
            is_backbone = True
            name = name.replace("text_embeddings.", "token_embd.")

        elif "backbone." in name or "codebook0_head." in name:
            is_backbone = True
            name = name.replace("backbone.layers.", "blk.")
            name = name.replace("backbone.norm.scale", "output_norm.weight")
            name = rename_transformer(name)

        elif "decoder." in name:
            is_decoder = True
            name = name.replace("decoder.layers.", "blk.")
            name = name.replace("decoder.norm.scale", "output_norm.weight")
            name = rename_transformer(name)

        elif name == "audio_head":
            is_decoder = True
            name = "audio_head.weight"
            if component == "decoder":
                # add padding at the beginning and the end so that build_lora_mm_id can be used
                zero_tensor = torch.zeros(1, 1024, 2051)
                data_torch = torch.cat([zero_tensor, data_torch, zero_tensor], dim=0)
                assert data_torch.shape == (33, 1024, 2051)
                # then, transpose it
                data_torch = data_torch.transpose(1, 2)

        elif name == "projection.weight":
            is_decoder = True
            is_backbone = True
            name = "csm_proj.weight"

        if can_quantize:
            if self.ftype == gguf.LlamaFileType.ALL_F32:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                data_qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                data_qtype = gguf.GGMLQuantizationType.BF16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                # decoder is very sensitive to quantization, do not quantize it lower than F16
                data_qtype = gguf.GGMLQuantizationType.Q8_0 if component != "decoder" \
                                else gguf.GGMLQuantizationType.F16
            else:
                raise ValueError(f"Unsupported file type: {self.ftype}")

        data = data_torch.numpy()

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except Exception as e:
            logger.error(f"Error quantizing tensor '{name}': {e}, fallback to F16")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        if (is_backbone and component == "backbone") or (is_decoder and component == "decoder"):
            # reverse shape to make it similar to the internal ggml dimension order
            shape_str = f"{{{', '.join(str(n) for n in reversed(data_torch.shape))}}}"
            logger.info(f"{f'%-32s' % f'{name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

            if component == "backbone":
                self.gguf_writer_backbone.add_tensor(name, data, raw_dtype=data_qtype)
            elif component == "decoder":
                self.gguf_writer_decoder.add_tensor(name, data, raw_dtype=data_qtype)

    def write(self):
        self._write_single(self.gguf_writer_backbone, "backbone")
        self._write_single(self.gguf_writer_decoder, "decoder")

    def _write_single(self, gguf_writer: gguf.GGUFWriter, component: str):
        output_path = str(self.fname_out).replace("<component>", component)
        gguf_writer.write_header_to_file(path=Path(output_path))
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file(progress=True)
        gguf_writer.close()

    @staticmethod
    def undo_permute(weights: Tensor, n_head: int, n_head_kv: int):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sesame model to GGUFs (multiple files)",)
    parser.add_argument(
        "--outfile", type=Path, default="sesame-csm-<component>.gguf",
        help="path to write to, the '<component>' placeholder is required and will be replaced with 'backbone' and 'decoder'",
    )
    parser.add_argument(
        "--vocab", type=Path, default="models/ggml-vocab-llama-bpe.gguf",
        help="path to vocab GGUF",
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
        help="path to safetensors or model ID containing model file (if model ID is specified, download from Hugging Face hub)",
        nargs="?",
        default="sesame/csm-1b:model.safetensors",
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
    path_vocab = args.vocab

    dir_parts = str(dir_model).split(":")
    if len(dir_parts) == 2:
        try:
            dir_model = Path(hf_hub_download(dir_parts[0], dir_parts[1]))
        except Exception as e:
            print("Error downloading model from Hugging Face hub:", e)
            print()
            print("Please make sure you have access to the model")
            print("Hint: you may need to set HF_TOKEN by running: huggingface-cli login")

    if not path_vocab.exists():
        raise FileNotFoundError(f"Vocab file not found: {path_vocab} ; Hint: download it from https://github.com/ggml-org/llama.cpp/blob/master/models/ggml-vocab-llama-bpe.gguf")

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    logger.info(f"Loading model: {dir_model}")

    with torch.inference_mode():
        converter = CSMModelConverter(
            safetensors_path=dir_model,
            fname_out=args.outfile,
            path_to_vocab_gguf=path_vocab,
            ftype=ftype_map[args.outtype],
            is_big_endian=args.bigendian,
        )
        converter.write()


if __name__ == '__main__':
    main()

