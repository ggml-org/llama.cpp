#!/usr/bin/env python3
"""Static QDQ (per-channel INT8) quantization of the M5-TTS codec decoder.

ORT CPU fuses QDQ Conv into an INT8 QLinearConv path; a standalone probe of
/conv_14/Conv measured 5.7 ms FP32 vs 0.93 ms QDQ on the target device.
Only Conv/ConvTranspose are quantized to limit accuracy risk.

Activations can only be quantized per tensor: QLinearConv rejects a non-scalar
output scale, so per-channel activations and int16 activations are both out.
Clipping activations to a percentile is also a loss, because the peaks of the
waveform layers are the signal. What does help is leaving the last convolutions
in FP32: `/conv_31..34` carry the largest relative quantization error, and
dropping them from the quantized set costs no speed while gaining ~2.7 dB.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization import calibrate

# ORT 1.18 raises when adjust_tensor_ranges writes a Softmax output that the
# calibration pass never collected. Those ranges are irrelevant here because
# only Conv/ConvTranspose are quantized.
def _relaxed_setitem(self, key, value):
    self.data[key] = value

calibrate.TensorsData.__setitem__ = _relaxed_setitem


class Reader(CalibrationDataReader):
    def __init__(self, meta, voices, samples=6):
        self.inputs = meta["inputs"]
        self.codes = np.asarray(voices["voices"][0]["codes"], dtype=np.int64)
        self.states = [np.zeros((8, 2, 1, 16, 0, 64), np.float32),
                       *[np.zeros(s["shape"], np.float32) for s in meta["states"]]]
        if len(self.states) != len(self.inputs) - 2:
            raise ValueError("unexpected number of codec state tensors")
        self.samples = samples
        self.i = 0

    def get_next(self):
        if self.i >= self.samples:
            return None
        i = self.i
        self.i += 1
        count = [4, 8, 16, 4, 8, 16][i % 6]
        offset = i * 2
        chunk = self.codes[np.arange(offset, offset + count) % len(self.codes)]
        return dict(zip(self.inputs,
                        [chunk.T[None], np.array([offset], np.int64),
                         *[s.copy() for s in self.states]]))


def quantize(source: Path, output: Path, meta_path: Path, voices_path: Path, samples: int, exclude=(), conv_only=False, exclude_last=0, ops=None) -> None:
    meta = json.loads(meta_path.read_text())
    voices = json.loads(voices_path.read_text())
    exclude = list(exclude)
    if exclude_last:
        import onnx
        graph = onnx.load(source).graph
        convs = [node.name for node in graph.node if node.op_type == "Conv"]
        exclude.extend(convs[-exclude_last:])
    if exclude:
        import onnx
        graph = onnx.load(source).graph
        conv_names = {node.name for node in graph.node if node.op_type in ("Conv", "ConvTranspose", "MatMul")}
        unknown = set(exclude) - conv_names
        if unknown:
            raise ValueError(f"unknown convolution exclusions: {sorted(unknown)}")
    if ops is not None:
        op_list = ops
    else:
        op_list = ["Conv"] if conv_only else ["Conv", "ConvTranspose"]
    t0 = time.time()
    quantize_static(
        source, output, Reader(meta, voices, samples),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_list,
        nodes_to_exclude=list(exclude),
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print("saved %s (%d bytes) in %.1fs" % (output, output.stat().st_size, time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--exclude", action="append", default=[], help="keep a named convolution in FP32")
    parser.add_argument("--conv-only", action="store_true", help="quantize Conv and leave ConvTranspose in FP32")
    parser.add_argument("--exclude-last", type=int, default=0,
                        help="keep the last N Conv nodes in FP32 (4 is the measured sweet spot)")
    parser.add_argument("--ops", default=None,
                        help="comma-separated op types to quantize (overrides --conv-only), "
                             "e.g. Conv,MatMul or MatMul to extend an already-QDQ model")
    args = parser.parse_args()
    ops = args.ops.split(",") if args.ops else None
    quantize(args.source, args.output, args.meta, args.voices, args.samples,
             args.exclude, args.conv_only, args.exclude_last, ops)
