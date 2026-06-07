# NVFP4 quantization

NVFP4 is NVIDIA's 4-bit floating-point weight format. In llama.cpp it is
represented by `GGML_TYPE_NVFP4`, with packed `E2M1` values and per-block
`E4M3`/UE4M3 scale data.

llama.cpp support is split into two parts:

- GGUF / ggml type support was added in
  [#19769](https://github.com/ggml-org/llama.cpp/pull/19769).
- A native CUDA path for Blackwell FP4 matrix-matrix workloads was added in
  [#22196](https://github.com/ggml-org/llama.cpp/pull/22196).

This page summarizes when NVFP4 is useful and how to compare it with mature
4-bit formats such as `Q4_K_M`.

## When to try NVFP4

NVFP4 is most useful when at least one of the following is true:

- You are converting or running a Hugging Face checkpoint that is already
  distributed in ModelOpt or compressed-tensors NVFP4 format.
- You are running on a CUDA Blackwell build and care about prompt processing or
  batched matrix-matrix workloads where the native FP4 path can be selected.
- You want to benchmark NVIDIA's FP4 checkpoint ecosystem against llama.cpp's
  existing integer 4-bit formats.

NVFP4 is not automatically faster for every workload. One-token-at-a-time text
generation is often memory-bandwidth or vector-dot limited, so it may not see
the same gains as prompt processing or batched workloads. Always benchmark
prompt processing and text generation separately.

## Hardware and backend notes

- Native CUDA FP4 acceleration requires a Blackwell CUDA build. The CUDA backend
  selects the native FP4 path for MXFP4 / NVFP4 matrix-matrix workloads when
  `blackwell_mma_available()` is true for the active build and device.
- Other backends can still load NVFP4 GGUF tensors through their supported
  dequantization / vector-dot paths, but performance depends on the backend and
  workload.
- Support is still evolving. If a device is Blackwell-class but performance is
  unexpected, check the build architecture flags, backend log output, and
  `llama-bench` results before assuming the native FP4 path is active.

## NVFP4 versus Q4_K_M

`Q4_K_M` remains a strong default for general interactive use because it has
mature kernels and broad backend coverage.

NVFP4 is a good candidate when you want to preserve an upstream NVFP4 checkpoint
format or benchmark Blackwell FP4 hardware. On single-stream interactive decode,
compare against `Q4_K_M` rather than assuming NVFP4 will win. On Blackwell
prompt processing or batched workloads, NVFP4 may be more attractive because
the native FP4 path can reduce compute cost.

Quality also depends on the model, calibration recipe, tensor selection, and
whether the source checkpoint was trained or calibrated for NVFP4. There is no
universal quality ordering between NVFP4 and `Q4_K_M`; run task-specific
evaluation for your model.

## Conversion and usage

For ModelOpt or compressed-tensors NVFP4 checkpoints on Hugging Face, use
`convert_hf_to_gguf.py` to convert the checkpoint to GGUF. The converter detects
NVFP4 metadata and repacks supported NVFP4 tensors into the GGUF layout used by
llama.cpp.

Example:

```bash
python3 convert_hf_to_gguf.py \
  --outfile model-nvfp4.gguf \
  --remote <user>/<nvfp4-model>
```

Then run the resulting GGUF as usual:

```bash
./build/bin/llama-cli -m model-nvfp4.gguf -p "Hello"
```

If you are quantizing a high-precision GGUF locally, check
`llama-quantize --help` in your build for the currently supported quantization
targets. Not every build exposes NVFP4 as a direct `llama-quantize` target.

## Benchmarking checklist

Use `llama-bench` or your normal serving benchmark to record both prompt
processing and text generation:

```bash
./build/bin/llama-bench -m model-nvfp4.gguf -p 512 -n 128
./build/bin/llama-bench -m model-q4_k_m.gguf -p 512 -n 128
```

When comparing formats, keep the same model architecture, prompt length,
generation length, batch/concurrency setting, backend, and GPU build flags.
Report prompt processing and text generation numbers separately because NVFP4
can affect them differently.
