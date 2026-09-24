#!/usr/bin/env python3
"""Verify laya GGUF quantization: structure, precision protection, dtypes and
tensor-level deviation of the quantized models against the F16 reference.

Usage:
    PYTHONPATH=gguf-py python3 tests/laya/verify_quantize.py [model-dir]
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'gguf-py'))

from gguf.gguf_reader import GGUFReader  # noqa: E402
from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
F16 = f'{MODEL_DIR}/laya-f16.gguf'
QUANTS = [
    ('Q4_K_M', f'{MODEL_DIR}/laya-q4_k_m.gguf'),
    ('Q5_K_M', f'{MODEL_DIR}/laya-q5_k_m.gguf'),
    ('Q8_0',   f'{MODEL_DIR}/laya-q8_0.gguf'),
]

# precision-protected tensor families: must never be quantized
PROTECTED_PREFIXES = (
    'token_embd.weight',
    'token_embd_norm.',
    'output_norm.',
    'type_emb.',
    'scorer.',
    'act_head.',
)


def is_protected(name: str) -> bool:
    # all norm tensors are protected regardless of prefix
    if '_norm.weight' in name or '_norm.bias' in name:
        return True
    for p in PROTECTED_PREFIXES:
        if name.startswith(p):
            return True
    return False


def norm_shape(shape):
    # GGUFReader may drop trailing singleton dims; compare by element count
    return int(np.prod(shape))


def load_tensors(path: str):
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        out[t.name] = (t.tensor_type, t.shape, t.data)
    return out


def dequant_tensor(tensor_type, shape, data):
    raw = np.ascontiguousarray(data).ravel()
    if tensor_type == GGMLQuantizationType.F16:
        return raw.view(np.float16).reshape(shape).astype(np.float32)
    if tensor_type == GGMLQuantizationType.F32:
        return raw.view(np.float32).reshape(shape)
    if tensor_type == GGMLQuantizationType.Q8_0:
        # block_q8_0 = { ggml_fp16_t d; int8_t qs[32]; } -- scale first
        blocks = raw.view(np.uint8).reshape(-1, 34)
        s = blocks[:, :2].view(np.float16).astype(np.float32)[:, 0]
        d = blocks[:, 2:].view(np.int8).astype(np.float32)
        return (d * s[:, None]).reshape(shape)
    # k-quants via gguf-py
    dq = quants.dequantize(raw.view(np.uint8), tensor_type)
    return np.asarray(dq).reshape(shape)


def main():
    ref = load_tensors(F16)
    print(f'{F16}: {len(ref)} tensors')
    errors = []
    for qname, qpath in QUANTS:
        q = load_tensors(qpath)
        print(f'\n=== {qname} ===')
        if set(q.keys()) != set(ref.keys()):
            missing = set(ref) - set(q)
            extra = set(q) - set(ref)
            print(f'  TENSOR LIST MISMATCH: missing={len(missing)} extra={len(extra)}')
            errors.append(f'{qname}: tensor list mismatch')
            continue
        n_protected = n_quantized = n_q8_0_fallback = 0
        dev_lines = []
        for name in sorted(ref.keys()):
            rt, rshape, rdata = ref[name]
            qt, qshape, qdata = q[name]
            if norm_shape(rshape) != norm_shape(qshape):
                errors.append(f'{qname}: {name} element count mismatch {rshape} vs {qshape}')
                continue
            protected = is_protected(name)
            if qt in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                if protected:
                    n_protected += 1
                dev = 0.0
                rel = 0.0
            else:
                n_quantized += 1
                # Q8_0 used as a 256-block fallback (ncols not divisible by 256)
                if qt == GGMLQuantizationType.Q8_0 and rshape[0] % 256 != 0:
                    n_q8_0_fallback += 1
                if not protected:
                    # tensor-level deviation for quantized (lossy) tensors
                    f32 = dequant_tensor(rt, rshape, rdata)
                    dq = dequant_tensor(qt, qshape, qdata)
                    delta = np.abs(f32 - dq)
                    denom = np.abs(f32).max()
                    if denom > 0:
                        dev = float(delta.max()) / denom
                        rel = float(delta.mean()) / denom
                    else:
                        dev = rel = 0.0
                    dev_lines.append((name, qt.name, float(delta.max()), float(delta.mean()), dev, rel))
                else:
                    dev = rel = -1.0
            if protected and qt not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                errors.append(f'{qname}: PROTECTED tensor {name} got quantized type {qt.name}!')
        print(f'  protected (kept F16/F32): {n_protected} tensors')
        print(f'  quantized (lossy):        {n_quantized} tensors, of which {n_q8_0_fallback} are Q8_0 (ncols%256 != 0 fallback)')
        if dev_lines:
            print(f'  {"tensor":38s} {"type":6s} {"max_abs":>12s} {"mean_abs":>12s} {"max_rel":>10s}')
            worst = sorted(dev_lines, key=lambda x: -x[4])[:5]
            for name, tname, mabs, mabsmean, rel, _ in worst:
                print(f'  {name:38s} {tname:6s} {mabs:12.6g} {mabsmean:12.6g} {rel:10.4%}')
            all_max_rel = max(d[4] for d in dev_lines)
            all_mean_rel = np.mean([d[4] for d in dev_lines])
            print(f'  -> worst tensor max_rel={all_max_rel:.4%}, mean tensor max_rel={all_mean_rel:.4%}')
    print()
    if errors:
        print('FAIL:')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    print('PASS: structure + precision protection verified')


if __name__ == '__main__':
    main()
