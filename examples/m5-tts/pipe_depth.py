"""Run the official sampling loop around a resident llama.cpp depth core."""

import struct

import numpy as np

from pipe_backbone import PipeBackbone


def export_weights(model, output, android=False):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(model).graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    nodes = {node.name: node for node in graph.node}
    projections = ["/head0/MatMul", *[f"/MatMul_{i}" if i else "/MatMul" for i in range(15)]]
    arrays = {
        name: numpy_helper.to_array(weights[name])
        for name in ("projected_audio", "audio_embeddings.weight")
    }
    arrays["heads"] = np.stack([numpy_helper.to_array(weights[nodes[name].input[1]]) for name in projections])
    arrays["projection"] = numpy_helper.to_array(weights[nodes["/projection/MatMul"].input[1]])
    if android:
        with open(output, "wb") as file:
            for array in (arrays["heads"].transpose(0, 2, 1), arrays["projection"].T,
                          arrays["projected_audio"], arrays["audio_embeddings.weight"]):
                array.astype("<f4").tofile(file)
    else:
        np.savez(output, **arrays)


def sample(logits, noise, allow_eos):
    if not allow_eos:
        logits[0] = -1e5
    logits /= np.float32(0.9)
    cutoff = np.partition(logits, -10)[-10]
    scores = np.exp(np.where(logits >= cutoff, logits - logits.max(), -np.inf))
    scores /= scores.sum()
    return int(np.argmax(scores / noise))


class PipeDepth(PipeBackbone):
    def __init__(self, command, weights):
        with np.load(weights, allow_pickle=False) as data:
            self.heads = data["heads"]
            self.projection = data["projection"]
            self.audio = data["audio_embeddings.weight"]
            self.projected = data["projected_audio"]
        super().__init__(command)

    def run(self, outputs, feed):
        hidden = np.asarray(feed["hidden"], dtype=np.float32).reshape(1, 512)
        noise = np.asarray(feed["noise"], dtype=np.float32).reshape(16, 2051)
        allow_eos = bool(np.asarray(feed["allow_eos"]).item())
        if not np.isfinite(hidden).all() or not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("expected finite hidden and positive finite sampling noise")

        self.process.stdin.write(struct.pack("<I", 0))
        self.position = 0
        codes = [sample((hidden @ self.heads[0])[0], noise[0], allow_eos)]
        local = np.concatenate((hidden @ self.projection, self.projected[codes]), axis=0)[None]
        for index in range(1, 16):
            positions = np.arange(self.position, self.position + local.shape[1])[None]
            decoded, _ = super().run(None, {"hidden": local, "positions": positions})
            code = sample((decoded @ self.heads[index])[0], noise[index], True)
            codes.append(code)
            local = self.projected[None, [code + index * 2051]]
        indices = np.asarray(codes, dtype=np.int64)[None]
        next_hidden = self.audio[indices + np.arange(16) * 2051].sum(axis=1, keepdims=True)
        return indices, next_hidden


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--android", action="store_true", help="write the NativeDepth binary layout")
    args = parser.parse_args()
    export_weights(args.model, args.output, args.android)
