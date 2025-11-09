#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

# Import all model classes from the conversion module
from conversion import (
    ModelBase, ModelType, get_model_architecture, logger
)

try:
    from mistral_common.tokens.tokenizers.base import TokenizerVersion
    from mistral_common.tokens.tokenizers.multimodal import DATASET_MEAN as _MISTRAL_COMMON_DATASET_MEAN, DATASET_STD as _MISTRAL_COMMON_DATASET_STD
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.sentencepiece import (
        SentencePieceTokenizer,
    )

    _mistral_common_installed = True
    _mistral_import_error_msg = ""
except ImportError:
    _MISTRAL_COMMON_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _MISTRAL_COMMON_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    _mistral_common_installed = False
    TokenizerVersion = None
    Tekkenizer = None
    SentencePieceTokenizer = None
    _mistral_import_error_msg = (
        "Mistral format requires `mistral-common` to be installed. Please run "
        "`pip install mistral-common[image,audio]` to install it."
    )


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="do not write file; just process and print stats",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="for split output: do not place small tensors in the first split",
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="print the list of supported model architectures",
    )
    parser.add_argument(
        "--mmproj", type=Path,
        help="path to mmproj file (for mulimodal models)",
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="path to metadata override file",
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="download model from huggingface",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="model is in mistral format",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help="do not use mistral community chat template",
    )
    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help="include sentence-transformers dense modules safetensors files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        # Load all model modules to ensure they're registered
        from conversion import _load_all_models
        _load_all_models()
        logger.error("Supported models:")
        ModelBase.print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download
        allowed_patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"]
        if args.sentence_transformers_dense_modules:
            # include sentence-transformers dense modules safetensors files
            allowed_patterns.append("*.safetensors")
        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)

    if not dir_model.is_dir():
        logger.error(f'Error: {dir_model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    is_mistral_format = args.mistral_format
    if is_mistral_format and not _mistral_common_installed:
        raise ImportError(_mistral_import_error_msg)
    disable_mistral_community_chat_template = args.disable_mistral_community_chat_template

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        if not is_mistral_format:
            model_architecture = get_model_architecture(hparams, model_type)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                model_class = ModelBase.from_model_architecture(model_architecture, model_type=model_type)
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported")
                sys.exit(1)
        elif args.mmproj:
            assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
            from conversion import get_model_class
            model_class = get_model_class("PixtralModel")
        else:
            from conversion import get_model_class
            model_class = get_model_class("MistralModel")

        model_instance = model_class(dir_model, output_type, fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split,
                                     remote_hf_model_id=hf_repo_id, disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                                     sentence_transformers_dense_modules=args.sentence_transformers_dense_modules
                                     )

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()
