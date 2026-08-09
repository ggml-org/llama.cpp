## Overview

This PR adds support for two new **mtmd projectors** — **LocateAnything-3B** (referring-expression grounding) and **Mage-VL** (custom vision tower).

Both are implemented and fully functional on **CPU** (the primary target per the contribution guidelines). They are registered in `tools/mtmd/` and ship as new files `tools/mtmd/models/locateanything.cpp` and `tools/mtmd/models/magevl.cpp`.

- `LocateAnything-3B`: given an image + a natural-language referring expression, outputs bounding boxes as `0–1000` normalized integer coordinates.
- `Mage-VL`: a second custom mmproj vision-tower backend.

The `ggml-cpu/binary-ops.cpp` change adds the reverse upcast combinations so the CPU path handles the `(F32,F16,F32)` binary-op combos these projectors emit.

The matching **CUDA** correctness fix (a `device_supports_op` type-combo guard + `(F32,F16,F32)` dispatch in `binbcast.cu` / `ggml-cuda.cu`) is submitted separately in branch `fix/cuda-binary-op-f16f32`, so this PR intentionally touches only the CPU / CPU-adjacent path.

## Additional information

- Build: `cmake -B build -DGGML_CUDA=ON` then `--target mtmd llama` (the CPU path also works without CUDA).
- Verified: `mmproj-LocateAnything-3B-BF16.gguf` loads and is recognized as a projector; referring-expression queries decode boxes on CPU via the fallback path.
- Coordinate convention: outputs are `0–1000` normalized integers; `pixel = coord / 1000 * image_size`.
- New API: `mtmd_bitmap_set/get_temporal_idx` for temporal (video) indexing.

## Requirements

- [x] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: **YES** — parts of this PR (the mtmd projector wiring and this description) were drafted with assistance from an AI coding agent (WorkBuddy). I have reviewed and tested all changes and am responsible for the submitted code.

<!-- Reminder (added by AI agent): you are responsible for all submitted changes. Note that llama.cpp restricts AI-generated content — see AGENTS.md and CONTRIBUTING.md. -->
