## Overview

Fixes a CUDA assertion crash in the binary-ops broadcast path that is triggered by models emitting mixed `F16/F32` operands (e.g. the new mtmd projectors in the companion PR `locateanything-magevl-grounding`).

Previously `device_supports_op` returned `true` for *any* `F32/F16` combination, so these ops were dispatched to the broadcast kernel, which then read an `F16` tensor as `float` (`nb10 == 2`, `2 % 4 != 0`) and tripped the `binbcast.cu:293` assertion — crashing CUDA.

- `binbcast.cu`: `device_supports_op` now only allows the exact supported type combinations (+ a contiguity check).
- `ggml-cuda.cu`: add the matching `(F32,F16,F32)` non-fused dispatch.

This is intentionally **CUDA-only**; the CPU-side reverse-upcast fallback (`binary-ops.cpp`) ships with the projector PR so each backend is introduced separately per the contribution guidelines.

## Additional information

- Repro: run a model that produces a `(F32,F16,F32)` binary-op on CUDA before this fix → `binbcast.cu:293` assertion. After the fix, the op is either correctly dispatched or falls back to CPU.
- Build: `cmake -B build -DGGML_CUDA=ON --target ggml-cuda llama`.

## Requirements

- [x] I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: **YES** — the `device_supports_op` guard logic and this description were drafted with assistance from an AI coding agent (WorkBuddy). I have reviewed and tested the change and am responsible for the submitted code.

<!-- Reminder (added by AI agent): you are responsible for all submitted changes. Note that llama.cpp restricts AI-generated content — see AGENTS.md and CONTRIBUTING.md. -->
