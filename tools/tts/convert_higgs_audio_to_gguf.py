#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parents[2] / "gguf-py"))
import gguf


CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
CODEBOOK_EMBD = "tied.embedding.modality_embeddings.0.embedding.weight"
CODEBOOK_HEAD = "tied.head.modality_heads.0.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Higgs Audio v3 TTS audio/codebook tensors to a companion GGUF."
    )
    parser.add_argument("model", type=Path, help="Higgs Audio v3 HF checkpoint directory")
    parser.add_argument("--outfile", type=Path, required=True, help="output GGUF path")
    parser.add_argument(
        "--outtype",
        choices=("f16", "f32"),
        default="f16",
        help="floating point storage for exported tensors",
    )
    parser.add_argument("--bigendian", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the export plan without writing tensor data",
    )
    return parser.parse_args()


def load_config(model_dir: Path) -> dict:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_safetensors(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        names = sorted(set(weight_map.values()))
        for name in names:
            yield model_dir / name
        return

    yield from sorted(model_dir.glob("*.safetensors"))


def as_numpy(tensor: torch.Tensor, outtype: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        if outtype == "f16":
            return tensor.to(torch.float16).numpy()
        return tensor.to(torch.float32).numpy()

    if tensor.dtype == torch.bool:
        return tensor.to(torch.int8).numpy()
    return tensor.numpy()


def export_name(name: str, codec_index: int) -> str | None:
    if name == CODEBOOK_EMBD:
        return "higgs.codebook_embd.weight"
    if name == CODEBOOK_HEAD:
        return "higgs.codebook_head.weight"
    if name.startswith(CODEC_PREFIX):
        return f"higgs.codec.{codec_index:04d}"
    return None


def main() -> int:
    args = parse_args()
    model_dir = args.model
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")

    config = load_config(model_dir)
    audio_cfg = config.get("audio_encoder_config") or {}
    if config.get("model_type") != "higgs_multimodal_qwen3":
        raise SystemExit(f"not a Higgs multimodal Qwen3 config: {config.get('model_type')!r}")

    endian = gguf.GGUFEndian.BIG if args.bigendian else gguf.GGUFEndian.LITTLE
    writer = gguf.GGUFWriter(
        path=None if args.dry_run else args.outfile,
        arch="higgs_audio",
        endianess=endian,
        dry_run=args.dry_run,
    )

    writer.add_name("Higgs Audio v3 TTS audio companion")
    writer.add_string("higgs_audio.format", "higgs-audio-v3-tts")
    writer.add_string("higgs_audio.backbone_arch", "qwen3")
    writer.add_uint32("higgs_audio.num_codebooks", int(audio_cfg.get("num_codebooks", 8)))
    writer.add_uint32("higgs_audio.codebook_size", int(audio_cfg.get("vocab_size", 1026)))
    writer.add_uint32("higgs_audio.hidden_size", int(audio_cfg.get("out_dim", config.get("_hidden_size", 2560))))
    writer.add_uint32("higgs_audio.boc_id", 1024)
    writer.add_uint32("higgs_audio.eoc_id", 1025)
    writer.add_uint32("higgs_audio.sample_rate", 24000)
    writer.add_uint32("higgs_audio.frame_rate", 25)
    writer.add_bool("higgs_audio.use_delay_pattern", bool(audio_cfg.get("use_delay_pattern", True)))
    writer.add_bool("higgs_audio.tie_codebook_embeddings", bool(audio_cfg.get("tie_word_embeddings", True)))

    exported = 0
    codec_index = 0
    total_bytes = 0
    seen: set[str] = set()
    codec_output_names: list[str] = []
    codec_original_names: list[str] = []
    for shard in iter_safetensors(model_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in sorted(f.keys()):
                is_codec = name.startswith(CODEC_PREFIX)
                out_name = export_name(name, codec_index)
                if out_name is None:
                    continue
                if out_name in seen:
                    raise RuntimeError(f"duplicate output tensor: {out_name}")
                seen.add(out_name)
                if is_codec:
                    codec_output_names.append(out_name)
                    codec_original_names.append(name[len(CODEC_PREFIX):])
                    codec_index += 1

                data = as_numpy(f.get_tensor(name), args.outtype)
                writer.add_tensor(out_name, data)
                exported += 1
                total_bytes += data.nbytes
                print(f"{out_name:80s} {str(data.dtype):8s} shape={list(data.shape)}")

    required = {"higgs.codebook_embd.weight", "higgs.codebook_head.weight"}
    missing = sorted(required - seen)
    if missing:
        raise RuntimeError(f"missing required Higgs codebook tensors: {missing}")

    writer.add_uint32("higgs_audio.codec_tensor_count", len(codec_output_names))
    writer.add_array("higgs_audio.codec_tensor_names", codec_output_names)
    writer.add_array("higgs_audio.codec_original_tensor_names", codec_original_names)

    print(f"Exported {exported} tensors, {total_bytes / 1_000_000_000:.2f} GB payload")
    if args.dry_run:
        print(f"Dry run, not writing {args.outfile}")
        return 0

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
