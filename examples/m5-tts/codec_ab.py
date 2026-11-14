#!/usr/bin/env python3
"""Stateful chunked codec decode A/B: run models on the same codes and compare.

Mirrors the engine's Codec::run() loop (chunks of N frames with past->present
state feedback) so the comparison covers the real streaming path.
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort


def load_feed(meta, voices, chunk):
    codes = np.asarray(voices["voices"][0]["codes"], dtype=np.int64)  # [T, books]
    inputs = meta["inputs"]
    # "past" (kv cache) starts empty; meta["states"] covers conv_0..conv_20.
    zeros = {"past": np.zeros((8, 2, 1, 16, 0, 64), np.float32)}
    zeros.update({name: np.zeros(s["shape"], np.float32)
                  for name, s in zip(inputs[3:], meta["states"])})
    return codes, inputs, zeros


def run_model(path, meta, voices, chunk, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(path, so)
    load_s = time.time() - t0
    codes, inputs, states = load_feed(meta, voices, chunk)
    T = len(codes)
    outs = []
    t0 = time.time()
    offset = 0
    while offset < T:
        n = min(chunk, T - offset)
        block = codes[offset:offset + n]           # [n, books]
        feed = {
            "codes": block.T[None],                 # [1, books, n]
            "offset": np.array([offset], np.int64),
        }
        feed.update({k: v.copy() for k, v in states.items()})
        res = sess.run(None, feed)
        outs.append(res[0].reshape(-1))
        # present -> past, next_X -> X (same convention as the engine).
        out_names = [o.name for o in sess.get_outputs()]
        for name, val in zip(out_names[1:], res[1:]):
            target = "past" if name == "present" else (name[5:] if name.startswith("next_") else None)
            if target and val is not None and val.size:
                states[target] = val
        offset += n
    run_s = time.time() - t0
    return np.concatenate(outs), load_s, run_s


def metrics(ref, cand):
    n = min(ref.size, cand.size)
    ref, cand = ref[:n], cand[:n]
    power = float(np.dot(ref, ref))
    err = cand - ref
    noise = float(np.dot(err, err))
    corr = float(np.dot(ref, cand) / np.sqrt(power * float(np.dot(cand, cand)))) if power else 0.0
    snr = 10 * np.log10(power / noise) if noise > 0 else float("inf")
    return {"n": n, "corr": round(corr, 4), "snr_db": round(float(snr), 2),
            "rms_ref": round(float(np.sqrt(power / max(n, 1))), 1),
            "rms_cand": round(float(np.sqrt(np.dot(cand, cand) / max(n, 1))), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save", default=None, help="write WAVs with this prefix")
    ap.add_argument("models", nargs="+", help="tag=path pairs")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    voices = json.load(open(args.voices))
    results = {}
    for spec in args.models:
        tag, path = spec.split("=", 1)
        pcm, load_s, run_s = run_model(path, meta, voices, args.chunk, args.threads)
        results[tag] = pcm
        print(f"[{tag}] samples={pcm.size} load={load_s:.1f}s run={run_s:.2f}s "
              f"rms={np.sqrt((pcm.astype(np.float64)**2).mean()):.1f}", flush=True)
        if args.save:
            import wave
            with wave.open(f"{args.save}-{tag}.wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(np.clip(pcm * 32768, -32768, 32767).astype("<i2").tobytes())

    tags = list(results)
    base = tags[0]
    for tag in tags[1:]:
        print(f"[{base} vs {tag}]", json.dumps(metrics(results[base], results[tag])))


if __name__ == "__main__":
    main()
