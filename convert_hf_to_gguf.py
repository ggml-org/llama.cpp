#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    MmprojModel,
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
    _mistral_common_installed,
    _mistral_import_error_msg,
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
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="auto",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type",
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
        help="only print out a split plan and exit, without writing any new files",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split (disabled by default)"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="Print the supported models"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj", action="store_true",
        help="Export multimodal projector (mmproj) for vision models. This will only work on some vision models. An 'mmproj-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--embed-mmproj", action="store_true",
        help="Embed multimodal projector tensors and metadata in the text-model GGUF. "
             "This preserves the normal model output name and is only supported for models with an mmproj converter.",
    )
    parser.add_argument(
        "--embed-mtp", type=Path, metavar="DIR",
        help="Embed a standalone MTP/draft checkpoint as a namespaced component in the model GGUF. "
             "The component tensors and metadata are stored under the 'mtp.' prefix.",
    )
    parser.add_argument(
        "--embed-mtp-ane", type=Path, metavar="MLMODELC",
        help="Embed a compiled multifunction ANE compute image containing MTP and optional prefill/DFlash functions, "
             "or a directory containing batch-{1,2,4,8}.mlmodelc variants. Files are materialized into a "
             "content-addressed cache at runtime.",
    )
    parser.add_argument(
        "--mtp", action="store_true",
        help="Export only the multi-token prediction (MTP) head as a separate GGUF, suitable for use as a speculative draft. An 'mtp-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="Exclude the multi-token prediction (MTP) head from the converted GGUF. Pair with --mtp on a second run to publish trunk and MTP as two files. Note: the split form duplicates embeddings, but even though the bundled default is more space-efficient overall, this allows differing quantization which may be more performant.",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="Whether the model is stored following the Mistral format.",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help=(
            "Whether to disable usage of Mistral community chat templates. If set, use the Mistral official `mistral-common` library for tokenization and detokenization of Mistral models. "
            "Using `mistral-common` ensure correctness and zero-day support of tokenization for models converted from the Mistral format but requires to manually setup the tokenization server."
        )
    )

    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help=("Whether to include sentence-transformers dense modules. "
              "It can be used for sentence-transformers models, like google/embeddinggemma-300m. "
              "Default these modules are not included.")
    )

    parser.add_argument(
        "--fuse-gate-up-exps", action="store_true",
        help="Fuse gate_exps and up_exps tensors into a single gate_up_exps tensor for MoE models.",
    )
    parser.add_argument(
        "--fp8-as-q8", action="store_true",
        help="Store tensors dequantized from FP8 as Q8_0 instead of BF16/F16.",
    )

    parser.add_argument(
        "--target-model-dir", type=str, default=None,
        help=(
            "path to the target model directory; required when converting a standalone draft model "
            "(e.g. EAGLE3 / DFlash) that needs target-model metadata such as tokenizer, hidden size, and "
            "layer count to populate its GGUF."
        ),
    )

    args = parser.parse_args()
    if not args.print_supported_models and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        print_registered_models()
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
        if args.mmproj and args.embed_mmproj:
            logger.error("--mmproj and --embed-mmproj are mutually exclusive")
            sys.exit(1)
        if args.vocab_only and (args.embed_mmproj or args.embed_mtp):
            logger.error("--embed-mmproj / --embed-mtp cannot be used with --vocab-only")
            sys.exit(1)
        if args.mtp and args.embed_mtp:
            logger.error("--mtp and --embed-mtp are mutually exclusive")
            sys.exit(1)
        if args.embed_mtp_ane and not args.embed_mtp:
            logger.error("--embed-mtp-ane requires --embed-mtp")
            sys.exit(1)
        if args.embed_mtp_ane and not args.embed_mtp_ane.is_dir():
            logger.error("--embed-mtp-ane must point to a compiled .mlmodelc directory or batch-bundle directory")
            sys.exit(1)
        if args.embed_mtp_ane and (args.split_max_tensors > 0 or args.split_max_size != "0"):
            logger.error("--embed-mtp-ane currently requires a single, unsplit GGUF")
            sys.exit(1)

        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        if not is_mistral_format:
            model_architecture = get_model_architecture(hparams, model_type)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                model_class = get_model_class(model_architecture, mmproj=(model_type == ModelType.MMPROJ))
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported")
                sys.exit(1)
        elif args.mmproj:
            assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
            from conversion.pixtral import PixtralModel
            model_class = PixtralModel
        elif hparams.get("moe") is not None:
            from conversion.mistral import MistralMoeModel
            model_class = MistralMoeModel
        else:
            from conversion.mistral import MistralModel
            model_class = MistralModel

        if args.mtp and args.no_mtp:
            logger.error("--mtp and --no-mtp are mutually exclusive")
            sys.exit(1)

        if args.mtp or args.no_mtp:
            if not model_class.supports_mtp_export:
                logger.error("--mtp / --no-mtp are not supported for %s", model_architecture)
                sys.exit(1)
            if args.no_mtp:
                model_class.no_mtp = True
            if args.mtp:
                model_class.mtp_only = True

        model_instance = model_class(dir_model, output_type, fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split,
                                     remote_hf_model_id=hf_repo_id, disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                                     sentence_transformers_dense_modules=args.sentence_transformers_dense_modules,
                                     target_model_dir=Path(args.target_model_dir) if args.target_model_dir else None,
                                     fuse_gate_up_exps=args.fuse_gate_up_exps,
                                     fp8_as_q8=args.fp8_as_q8,
                                     )

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        elif args.embed_mmproj or args.embed_mtp:
            component_instances: list[tuple[str, ModelBase]] = []
            if args.embed_mmproj:
                try:
                    mmproj_class = get_model_class(model_architecture, mmproj=True)
                except NotImplementedError:
                    logger.error("--embed-mmproj is not supported for %s", model_architecture)
                    sys.exit(1)

                component_instances.append(("", mmproj_class(
                    dir_model, output_type, fname_out,
                    is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                    eager=args.no_lazy,
                    metadata_override=args.metadata, model_name=args.model_name,
                    split_max_tensors=0, split_max_size=0, dry_run=args.dry_run,
                    small_first_shard=False,
                    remote_hf_model_id=hf_repo_id,
                    disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                    sentence_transformers_dense_modules=args.sentence_transformers_dense_modules,
                    target_model_dir=Path(args.target_model_dir) if args.target_model_dir else None,
                    fuse_gate_up_exps=args.fuse_gate_up_exps,
                    fp8_as_q8=args.fp8_as_q8,
                )))

            if args.embed_mtp:
                mtp_hparams = ModelBase.load_hparams(args.embed_mtp, is_mistral_format=False)
                mtp_architecture = get_model_architecture(mtp_hparams, ModelType.TEXT)
                mtp_class = get_model_class(mtp_architecture, mmproj=False)
                component_instances.append(("mtp.", mtp_class(
                    args.embed_mtp, output_type, fname_out,
                    is_big_endian=args.bigendian, use_temp_file=False,
                    eager=args.no_lazy,
                    metadata_override=None, model_name=f"{args.embed_mtp.name} MTP",
                    split_max_tensors=0, split_max_size=0, dry_run=args.dry_run,
                    small_first_shard=False,
                    remote_hf_model_id=None,
                    disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                    sentence_transformers_dense_modules=False,
                    target_model_dir=dir_model,
                    fuse_gate_up_exps=args.fuse_gate_up_exps,
                    fp8_as_q8=args.fp8_as_q8,
                )))

            logger.info("Exporting model with %d embedded component(s)...", len(component_instances))
            model_instance.prepare_tensors()
            model_instance.prepare_metadata(vocab_only=False)
            ane_mmaps: list[np.memmap] = []

            for prefix, component in component_instances:
                if not prefix:
                    component.gguf_writer = model_instance.gguf_writer
                    component.prepare_tensors()
                    if isinstance(component, MmprojModel):
                        component.embedded_in_model = True
                    component.set_gguf_parameters()
                    continue

                component.prepare_tensors()
                component.prepare_metadata(vocab_only=False)
                for shard in component.gguf_writer.tensors:
                    for name, tensor_info in shard.items():
                        model_instance.gguf_writer.add_tensor_info(
                            prefix + name,
                            tensor_info.shape,
                            tensor_info.tensor.dtype,
                            tensor_info.nbytes,
                            raw_dtype=tensor_info.dtype,
                        )
                        model_instance.gguf_writer.tensors[-1][prefix + name].tensor = tensor_info.tensor
                for key, value in component.gguf_writer.kv_data[0].items():
                    model_instance.gguf_writer.add_key_value(
                        prefix + key,
                        value.value,
                        value.type,
                        value.sub_type,
                    )
                model_instance.gguf_writer.add_bool(prefix + "component.present", True)

            if args.embed_mtp_ane:
                bucket_dirs = [
                    (batch, args.embed_mtp_ane / f"batch-{batch}.mlmodelc")
                    for batch in (1, 2, 4, 8)
                    if (args.embed_mtp_ane / f"batch-{batch}.mlmodelc").is_dir()
                ]
                if not bucket_dirs:
                    bucket_dirs = [(1, args.embed_mtp_ane)]

                embedded_buckets: list[int] = []
                for batch, bundle_dir in bucket_dirs:
                    ane_files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
                    if not ane_files:
                        logger.error("ANE MTP bundle for batch %d contains no files: %s", batch, bundle_dir)
                        sys.exit(1)
                    digest = hashlib.sha256()
                    key_prefix = f"mtp.ane.bucket.{batch}"
                    for index, path in enumerate(ane_files):
                        relative = path.relative_to(bundle_dir).as_posix()
                        data = np.memmap(path, mode="r", dtype=np.uint8)
                        ane_mmaps.append(data)
                        digest.update(relative.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(memoryview(data))
                        tensor_name = f"{key_prefix}.file.{index:04d}"
                        model_instance.gguf_writer.add_tensor(tensor_name, data.view(np.int8))
                        model_instance.gguf_writer.add_string(f"{tensor_name}.path", relative)
                    model_instance.gguf_writer.add_uint32(f"{key_prefix}.file_count", len(ane_files))
                    model_instance.gguf_writer.add_string(f"{key_prefix}.bundle_sha256", digest.hexdigest())
                    manifest_path = args.embed_mtp_ane / f"batch-{batch}.json"
                    if manifest_path.is_file():
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        functions = manifest.get("functions", [])
                        if functions:
                            model_instance.gguf_writer.add_array(f"{key_prefix}.functions", functions)
                        if "context" in manifest:
                            model_instance.gguf_writer.add_uint32(
                                f"{key_prefix}.context_length", int(manifest["context"]))
                        if "sync_chunk" in manifest:
                            model_instance.gguf_writer.add_uint32(
                                f"{key_prefix}.sync_chunk", int(manifest["sync_chunk"]))
                    embedded_buckets.append(batch)

                model_instance.gguf_writer.add_string("mtp.ane.format", "mlmodelc-buckets-v2")
                model_instance.gguf_writer.add_array("mtp.ane.batch_buckets", embedded_buckets)
                model_instance.gguf_writer.add_bool("mtp.ane.keep_warm", True)

            model_instance.gguf_writer.write_header_to_file(path=model_instance.fname_out)
            model_instance.gguf_writer.write_kv_data_to_file()
            model_instance.gguf_writer.write_tensors_to_file(progress=True)
            model_instance.gguf_writer.close()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model with embedded components successfully exported to {out_path}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()
