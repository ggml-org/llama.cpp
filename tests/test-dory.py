"""Small deterministic GGUF, independent F32 Dory reference, and cache tests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf

H, FF, HD, NH, NK, VOCAB = 256, 256, 64, 8, 2, 32
PATTERN = "*-S-*-"
STD = np.float32(0.0177)


def norm(x):
    return x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + np.float32(1e-6))


def fixture(path, dtype=gguf.GGMLQuantizationType.F32, pattern=PATTERN, counts=(2, 2, 2), cache_mode=None):
    rng = np.random.default_rng(1729)
    w = gguf.GGUFWriter(path, "dory")
    w.add_name("tiny-dory-test-only")
    w.add_block_count(len(pattern))
    w.add_context_length(64)
    w.add_embedding_length(H)
    w.add_feed_forward_length(FF)
    w.add_head_count(NH)
    w.add_head_count_kv(NK)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_rope_dimension_count(HD)
    w.add_rope_freq_base(1e7)
    w.add_sliding_window(4)
    for key, value in zip(("input_layer_count", "recurrent_layer_count", "output_layer_count", "recurrent_loop_count"), (*counts, 4)):
        w.add_uint32(f"dory.{key}", value)
    w.add_string("dory.layer_pattern", pattern)
    if cache_mode is not None:
        w.add_string("dory.recurrent_kv_cache_mode", cache_mode)
    w.add_array("dory.rope.profile", [2 if char == "S" else 1 for char in pattern])
    for key, value in {"rope.freq_base_2": 1e4, "init_std": STD, "alpha_init": 0.05, "gate_init": H**0.5}.items():
        w.add_float32(f"dory.{key}", float(value))
    w.add_tokenizer_model("llama")
    w.add_token_list(["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(3, VOCAB)])
    w.add_token_scores([0.0]*VOCAB)
    w.add_token_types([2, 3, 3] + [1]*(VOCAB-3))
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    weights = {}

    def add(name, shape, vector=False):
        x = rng.normal(0, 0.06, shape).astype(np.float32)
        if vector:
            x = (rng.uniform(0.7, 1.3, shape)*STD).astype(np.float32)
        packed = gguf.quantize(x, dtype)
        weights[name] = gguf.dequantize(packed, dtype)
        w.add_tensor(name, packed, raw_dtype=dtype)

    add("token_embd.weight", (VOCAB, H))
    add("output.weight", (VOCAB, H))
    add("dory_logit_scale.weight", (VOCAB,), True)
    for i, char in enumerate(pattern):
        prefix = f"blk.{i}."
        add(prefix + "dory_alpha.weight", (H,), True)
        if char == "-":
            for role, shape in (("ffn_gate", (FF, H)), ("ffn_up", (FF, H)), ("ffn_down", (H, FF))):
                add(prefix + role + ".weight", shape)
            for role in ("dory_suv_gate", "dory_suv_up"):
                add(prefix + role + ".weight", (FF,), True)
        else:
            for role, rows in (("attn_q", NH*HD), ("attn_k", NK*HD), ("attn_v", NK*HD), ("attn_gate", NH*HD)):
                add(prefix + role + ".weight", (rows, H))
            add(prefix + "attn_output.weight", (H, NH*HD))
            for role in ("dory_sqk", "dory_s_gate"):
                add(prefix + role + ".weight", (NH*HD,), True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weights


def forward(weights, loops, pattern=PATTERN, counts=(2, 2, 2)):
    x = weights["token_embd.weight"][np.arange(3, 11)]
    n = len(x)

    def rope(x, theta):
        angle = np.arange(n, dtype=np.float32)[:, None, None] / np.float32(theta)**(np.arange(0, HD, 2, dtype=np.float32)/HD)
        c, s = np.cos(angle), np.sin(angle)
        a, b = np.split(x, 2, axis=-1)
        return np.concatenate((a*c-b*s, b*c+a*s), axis=-1)

    def layer(x, i):
        def get(role):
            return weights[f"blk.{i}.{role}.weight"]
        if pattern[i] == "-":
            g = (x @ get("ffn_gate").T) * (get("dory_suv_gate")*np.sqrt(np.float32(H)))
            u = (x @ get("ffn_up").T) * (get("dory_suv_up")*np.sqrt(np.float32(H)))
            y = norm((g/(1+np.exp(-g)))*u) @ get("ffn_down").T
        else:
            q = (x @ get("attn_q").T).reshape(n, NH, HD)
            k = (x @ get("attn_k").T).reshape(n, NK, HD)
            v = (x @ get("attn_v").T).reshape(n, NK, HD)
            theta = 1e4 if pattern[i] == "S" else 1e7
            q = norm(rope(q, theta)) * (get("dory_sqk").reshape(NH, HD)*np.sqrt(np.float32(HD))/STD)**2
            k = norm(rope(k, theta))
            k, v = np.repeat(k, NH//NK, axis=1), np.repeat(v, NH//NK, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(np.float32(HD))
            for t in range(n):
                scores[:, t, t+1:] = -np.inf
                if pattern[i] == "S":
                    scores[:, t, :max(0, t-3)] = -np.inf
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            y = np.einsum("hts,shd->thd", probs, v).reshape(n, NH*HD)
            sg = (get("dory_s_gate") * np.float32(H**0.25) / STD)**2
            y *= 1/(1+np.exp(-(x @ get("attn_gate").T)*sg))
            y = norm(y) @ get("attn_output").T
        a, b = norm(x), norm(y)
        return norm(a + np.abs(get("dory_alpha")*(np.float32(0.05)/STD))*(b-a))

    ni, nr, no = counts
    for i in range(ni):
        x = layer(x, i)
    inp = x.copy()
    for _ in range(loops):
        x = norm(x+inp)
        for i in range(ni, ni+nr):
            x = layer(x, i)
        x = norm(x)
    for i in range(ni+nr, ni+nr+no):
        x = layer(x, i)
    return (x @ weights["output.weight"].T) * (weights["dory_logit_scale.weight"]/STD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/parity")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.output / "tiny-f32.gguf"
    weights = fixture(model)
    models = [(model, weights, 3e-5)]
    bf16 = args.output / "tiny-bf16.gguf"
    bf16_weights = fixture(bf16, gguf.GGMLQuantizationType.BF16)
    models.append((bf16, bf16_weights, 3e-3))
    if args.quantizer:
        for preset in ("Q4_K_M", "Q5_K_M"):
            quantized = args.output / f"tiny-{preset.lower()}.gguf"
            with (args.output / f"quantize-{preset}.log").open("w") as log:
                subprocess.run([str(args.quantizer), "--output-tensor-type", "bf16", "--token-embedding-type", "bf16", str(bf16), str(quantized), preset, "2"], check=True, stdout=log, stderr=log)
            reader = gguf.GGUFReader(quantized)
            for tensor in reader.tensors:
                if len(tensor.shape) == 1 or tensor.name in ("token_embd.weight", "output.weight"):
                    assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16, tensor.name
            qw = {t.name: gguf.dequantize(t.data, t.tensor_type).reshape(tuple(reversed(t.shape))).astype(np.float32) for t in reader.tensors}
            # Quantized CPU matrix products round inputs internally; allow that error here.
            models.append((quantized, qw, 3e-3))
    results = []
    for path, w, tolerance in models:
        for loops in (1, 2, 4):
            reference = forward(w, loops)
            previous = None
            for chunk in (8, 3, 1):
                tag = f"{path.stem}-k{loops}-chunk{chunk}"
                output = args.output / f"{tag}.bin"
                with (args.output / f"{tag}.log").open("w") as log:
                    subprocess.run([str(args.runner), str(path), str(loops), str(chunk), str(output)], check=True, stdout=log, stderr=log)
                actual = np.fromfile(output, dtype=np.float32).reshape(8, VOCAB)
                error = float(np.max(np.abs(actual-reference)))
                cache_error = None if previous is None else float(np.max(np.abs(actual-previous)))
                if not np.isfinite(actual).all() or error > tolerance or (cache_error is not None and cache_error > tolerance):
                    raise AssertionError(f"{tag}: reference error={error}, cache error={cache_error}, tolerance={tolerance}")
                results.append(dict(case=tag, max_abs_error=error, cache_error=cache_error, tolerance=tolerance))
                previous = actual
    # With no prior history and one full chunk, both cache modes have identical logits.
    for path, w, tolerance in models:
        for loops in (1, 2, 4):
            tag = f"{path.stem}-shared-k{loops}"
            output = args.output / f"{tag}.bin"
            with (args.output / f"{tag}.log").open("w") as log:
                subprocess.run([str(args.runner), str(path), str(loops), "8", str(output)],
                               env=dict(os.environ, DORY_TEST_KV_MODE="last_loop"), check=True, stdout=log, stderr=log)
            actual = np.fromfile(output, dtype=np.float32).reshape(8, VOCAB)
            error = float(np.max(np.abs(actual-forward(w, loops))))
            caches = sum(map(int, re.findall(r"llama_kv_cache: size =.*?cells,\s+(\d+) layers", (args.output/f"{tag}.log").read_text())))
            if not np.isfinite(actual).all() or error > tolerance or caches != 3:
                raise AssertionError(f"{tag}: error={error}, caches={caches}")
            results.append(dict(case=tag, max_abs_error=error, tolerance=tolerance, cache_count=caches))
    # Full physical/virtual depth, with smaller tensor dimensions.
    deep = args.output / "deep-layout-f32.gguf"
    deep_pattern, counts = "S-S-S-*-"*9, (16, 40, 16)
    deep_weights = fixture(deep, pattern=deep_pattern, counts=counts)
    reference = forward(deep_weights, 4, deep_pattern, counts)
    previous = None
    for chunk in (8, 1):
        tag = f"deep-layout-f32-k4-chunk{chunk}"
        output = args.output / f"{tag}.bin"
        with (args.output / f"{tag}.log").open("w") as log:
            subprocess.run([str(args.runner), str(deep), "4", str(chunk), str(output)], check=True, stdout=log, stderr=log)
        actual = np.fromfile(output, dtype=np.float32).reshape(8, VOCAB)
        error = float(np.max(np.abs(actual-reference)))
        cache_error = None if previous is None else float(np.max(np.abs(actual-previous)))
        if not np.isfinite(actual).all() or error > 3e-5 or (cache_error is not None and cache_error > 3e-5):
            raise AssertionError(f"{tag}: reference error={error}, cache error={cache_error}")
        results.append(dict(case=tag, max_abs_error=error, cache_error=cache_error, tolerance=3e-5))
        previous = actual
    for loops in (0, 5):
        with (args.output / f"invalid-k{loops}.log").open("w") as log:
            result = subprocess.run([str(args.runner), str(model), str(loops), "1", str(args.output / "invalid.bin")], stdout=log, stderr=log)
        if result.returncode != 3:
            raise AssertionError(f"Invalid K={loops} was not rejected cleanly: {result.returncode}")
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: {len(results)} numerical/cache cases and 2 invalid-K checks")


if __name__ == "__main__":
    main()
