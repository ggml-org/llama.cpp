#!/usr/bin/env python3
"""Verify a qwen3-tts-talker GGUF against its safetensors source.

Checks: arch + TTS metadata, tensor inventory (404 talker tensors, speaker
encoder dropped, code predictor at blk.{block_count+i}, per-codebook
.{cid} tensors), text vocab size, and raw-bf16 byte parity on a spread of
tensors (1D tensors are compared as F32 values: the converter stores all
1D tensors as F32).

usage:
  python3 verify_qwen3tts_gguf.py <gguf> <safetensors>
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "gguf-py")

import torch  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from safetensors import safe_open  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", help="converted qwen3-tts-talker GGUF")
    ap.add_argument("safetensors", help="source model.safetensors")
    args = ap.parse_args()

    reader = GGUFReader(args.gguf, "r")

    arch = reader.fields["general.architecture"].parts[reader.fields["general.architecture"].data[0]].tobytes().decode()
    check(arch == "qwen3-tts-talker", f"arch = {arch}")

    def get_field(key):
        f = reader.fields.get(key)
        if f is None:
            return None
        if len(f.data) == 1:
            return f.parts[f.data[0]][0]
        parts = [f.parts[i] for i in f.data]
        if parts and parts[0].dtype == np.uint8:
            try:
                return [p.tobytes().decode("utf-8") for p in parts]
            except UnicodeDecodeError:
                pass
        return [p[0] for p in parts]

    expect_meta = {
        "qwen3-tts-talker.block_count": 28,
        "qwen3-tts-talker.context_length": 32768,
        "qwen3-tts-talker.embedding_length": 2048,
        "qwen3-tts-talker.feed_forward_length": 6144,
        "qwen3-tts-talker.attention.head_count": 16,
        "qwen3-tts-talker.attention.head_count_kv": 8,
        "qwen3-tts-talker.codec_vocab_size": 3072,
        "qwen3-tts-talker.num_code_groups": 16,
        "qwen3-tts-talker.predictor_layers": 5,
        "qwen3-tts-talker.cp_hidden_size": 1024,
        "qwen3-tts-talker.cp_feed_forward_length": 3072,
        "qwen3-tts-talker.cp_head_count": 16,
        "qwen3-tts-talker.cp_head_count_kv": 8,
        "qwen3-tts-talker.codec_pad_id": 2148,
        "qwen3-tts-talker.codec_bos_id": 2149,
        "qwen3-tts-talker.codec_eos_id": 2150,
        "qwen3-tts-talker.codec_think_id": 2154,
        "qwen3-tts-talker.codec_nothink_id": 2155,
        "qwen3-tts-talker.codec_think_bos_id": 2156,
        "qwen3-tts-talker.codec_think_eos_id": 2157,
        "qwen3-tts-talker.position_id_per_seconds": 13,
    }
    for key, want in expect_meta.items():
        got = get_field(key)
        check(got == want, f"{key} = {got} (want {want})")

    lang_names = get_field("qwen3-tts-talker.codec_language_names")
    lang_ids = get_field("qwen3-tts-talker.codec_language_ids")
    check(lang_names is not None and len(lang_names) == len(lang_ids), f"codec_language ids/names aligned ({len(lang_names or [])} langs)")
    check(get_field("qwen3-tts-talker.rope.dimension_sections") is not None, "rope.dimension_sections present")

    n_vocab = len(get_field("tokenizer.ggml.tokens") or [])
    check(n_vocab == 151936, f"text vocab size = {n_vocab}")
    merges = get_field("tokenizer.ggml.merges")
    check(merges is not None and len(merges) > 0, f"bpe merges present ({len(merges) if merges else 0})")

    names = {t.name: t for t in reader.tensors}
    print(f"\ntensor count: {len(names)}")
    check(len(names) == 404, f"tensor count = {len(names)} (want 404: all talker tensors, speaker_encoder dropped)")

    def expect_shape(name, want):
        t = names.get(name)
        if t is None:
            check(False, f"tensor {name} present")
            return
        check(list(t.shape) == want, f"{name} ggml shape {list(t.shape)} (want {want})")

    # ggml ne is reversed vs torch: torch (A,B) -> ggml ne [B,A]
    expect_shape("token_embd.weight", [2048, 151936])
    expect_shape("codec_embd.weight", [2048, 3072])
    expect_shape("codec_head.weight", [2048, 3072])
    expect_shape("output_norm.weight", [2048])
    expect_shape("text_proj_1.weight", [2048, 2048])
    expect_shape("text_proj_2.weight", [2048, 2048])
    expect_shape("cp_proj.weight", [2048, 1024])
    expect_shape("cp_norm.weight", [1024])
    expect_shape("blk.0.attn_q.weight", [2048, 2048])
    expect_shape("blk.0.attn_k.weight", [2048, 1024])
    expect_shape("blk.0.attn_q_norm.weight", [128])
    expect_shape("blk.0.ffn_gate.weight", [2048, 6144])
    expect_shape("blk.27.ffn_down.weight", [6144, 2048])
    expect_shape("blk.28.attn_q.weight", [1024, 2048])  # cp layer 0: torch [2048, 1024]
    expect_shape("blk.28.ffn_gate.weight", [1024, 3072])
    expect_shape("blk.32.attn_output.weight", [2048, 1024])  # cp layer 4
    expect_shape("cp_codec_embd.0.weight", [2048, 2048])
    expect_shape("cp_codec_embd.14.weight", [2048, 2048])
    expect_shape("cp_head.0.weight", [1024, 2048])
    expect_shape("cp_head.14.weight", [1024, 2048])

    want_suffixes = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm",
                     "attn_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"}
    for i in range(28, 33):
        prefix = f"blk.{i}."
        got = {n[len(prefix):].replace(".weight", "") for n in names if n.startswith(prefix)}
        check(got == want_suffixes, f"blk.{i} has all {len(want_suffixes)} tensors (got {len(got)})")

    for cid in range(15):
        for base in ("cp_codec_embd", "cp_head"):
            check(f"{base}.{cid}.weight" in names, f"{base}.{cid}.weight present")

    print("\nparity spot checks:")
    parity = [
        ("talker.codec_head.weight", "codec_head.weight"),
        ("talker.model.text_embedding.weight", "token_embd.weight"),
        ("talker.model.codec_embedding.weight", "codec_embd.weight"),
        ("talker.model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
        ("talker.model.layers.27.mlp.down_proj.weight", "blk.27.ffn_down.weight"),
        ("talker.code_predictor.model.layers.2.self_attn.q_proj.weight", "blk.30.attn_q.weight"),
        ("talker.code_predictor.model.layers.4.mlp.gate_proj.weight", "blk.32.ffn_gate.weight"),
        ("talker.code_predictor.small_to_mtp_projection.bias", "cp_proj.bias"),
        ("talker.code_predictor.model.codec_embedding.7.weight", "cp_codec_embd.7.weight"),
        ("talker.code_predictor.lm_head.13.weight", "cp_head.13.weight"),
        ("talker.text_projection.linear_fc2.bias", "text_proj_2.bias"),
        ("talker.model.norm.weight", "output_norm.weight"),
    ]
    with safe_open(args.safetensors, framework="pt") as f:
        for st_name, gg_name in parity:
            src = f.get_tensor(st_name)
            t = names[gg_name]
            if t.tensor_type == GGMLQuantizationType.F32:
                # 1D tensors are stored F32 by the converter; bf16 -> f32 is exact
                a = np.ascontiguousarray(t.data).astype(np.float32)
                b = src.float().numpy()
                ok = a.shape == b.shape and bool((a == b).all())
                check(ok, f"{gg_name} == {st_name} (F32, {a.size} elts)")
                continue
            if src.dtype == torch.bfloat16:
                b16 = src.contiguous().view(torch.uint16).numpy().view(np.uint8)
            else:
                b16 = src.contiguous().numpy().view(np.uint8)
            a = np.ascontiguousarray(t.data).view(np.uint8)
            ok = a.shape == b16.shape and bool((a == b16).all())
            check(ok, f"{gg_name} == {st_name} ({a.size} bytes)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for m in failures:
            print("  -", m)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
