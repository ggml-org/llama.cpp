Vulkan builds of this fork for AMD Strix Halo (Radeon 8060S / gfx1151).

A fork of llama.cpp with **adaptive speculative decoding** — draft length follows measured
acceptance instead of a fixed `n` — and a Vulkan backend tuned for this hardware on stock
Mesa, no ROCm toolchain. See the [README](https://github.com/LaurentZuijdwijk/llama.cpp#readme)
for the measured numbers and ready-to-run `llama-server` commands.

## Download

| | |
|---|---|
| Linux x86-64 | `llama-__TAG__-bin-ubuntu-vulkan-x64.tar.gz` |
| Windows x64 | `llama-__TAG__-bin-win-vulkan-x64.zip` |

```bash
tar xzf llama-__TAG__-bin-ubuntu-vulkan-x64.tar.gz
cd llama-__TAG__/
./llama --version
```

Self-contained. On Linux the shared libraries sit next to the executables and resolve
through an `$ORIGIN` rpath, so there is nothing to install and no `LD_LIBRARY_PATH` to
set. On Windows the DLLs sit next to the `.exe` files. Run them from the folder you
extracted.

Prefer a container, presets and a preflight check?
[**agention-llama**](https://github.com/LaurentZuijdwijk/agention-llama) packages all of
this — `curl | sh`, then `agention-llama run dflash-fp4`.

## Requirements

- x86-64, Linux or Windows
- A Vulkan 1.3 driver. Measured on Mesa RADV 26.0.8.
- Check your device is visible: `vulkaninfo --summary`

Built portable: `GGML_NATIVE=OFF` with every CPU variant compiled in and selected at run
time, `GGML_BACKEND_DL=ON`, `GGML_VULKAN=ON`. Exact flags in
`.github/workflows/release-vulkan.yml`. The benchmark figures in the README came from a
`GGML_NATIVE=ON` build; the work is on the GPU, so the two measure the same, though the
builds are not bit-identical.

> **Windows note.** The LDS stride fix (~12–14 % prefill) is **RADV-only** and stays off
> here. `ggml_vk_coopmat_shmem_pad()` takes the padded stride only when the driver
> reports as Mesa RADV at 25.3 or newer, because older RADV lowers `coopMatLoad` to
> `ds_read_b128` and the misaligned rows then cost more than the bank spread wins.
> AMD's Windows driver reports a different driver id and takes the upstream default pad.
> Everything else applies on both platforms: adaptive speculation, the ROCmFPx quant
> types, the batch 3–8 mat-vec fixes, and the tiled concat-transpose and f16-B prefill
> gates.

## Highlights

- `--spec-draft-adaptive` — draft length tracks measured acceptance, so `--spec-draft-n-max`
  becomes a safety ceiling rather than a tuning parameter. Recommended with
  `--spec-draft-n-min 3` for DFlash2, `2` for MTP.
- ROCmFPx quant types (`Q4_0_ROCMFP4`, `_FAST`, `Q2/Q3/Q6/Q8_0_ROCMFPX`) with CPU codecs
  and Vulkan dequant / mat-vec / matmul kernels — models mainline cannot load at all.
- Vulkan batched mat-vec fixes: an IQ3_S register spill at `NUM_COLS > 4` that made the
  batch widths speculation runs at 5× slower than they should be.
- Vulkan prefill gates: tiled concat-transpose (+45 % on delta-net MoE prefill), f16 B
  operand for quantized `mul_mat` and `mul_mat_id`.
