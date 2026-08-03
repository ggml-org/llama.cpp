# VLA

`libvla` loads control-generation components from standalone GGUF files and
conditions them on final per-token hidden states produced by libllama. The
public API and model factory are model-independent. MiniCPM-Robot is the first
model implementation.

A complete model consists of three independent files:

- a text model loaded by libllama
- an `mmproj.gguf` loaded by libmtmd
- a `vla.gguf` loaded by libvla

The VLA file uses `general.architecture = vla` and `vla.model_type` for factory
dispatch.

## Build

The experimental subsystem is disabled by default:

```bash
cmake -B build -DLLAMA_BUILD_VLA=ON
cmake --build build --target vla llama-vla-cli test-vla
```

The subsystem is a library and CLI tool. It does not add model-specific routes
or state to `llama-server`.

## Convert checkpoint -> GGUF

From the llama.cpp root (needs `torch` + `gguf-py`):

```bash
PYTHONPATH=gguf-py python3 tools/vla/convert_hf_to_vla_gguf.py \
  --model /path/to/MiniCPM-RobotManip \
  --output vla-f32.gguf \
  --control-horizon 30
```

Dims / layer counts are inferred from tensor shapes. `--action-horizon` must be
passed explicitly (`position_embedding` stores `max_seq_len`, not horizon).

The script writes metadata, maps all required tensors, and runs an L0
bit-exact re-read check.

## L1 numerical check

Reference dumps (from the matching PyTorch head):

```text
vl_embs.bin      [S, cross_dim]
state.bin        [state_dim]
noise.bin        [control_horizon, control_dim]
controls_ref.bin [control_horizon, control_dim]
```

```bash
./build/bin/test-vla /path/to/vla-f32.gguf /path/to/l1_ref
```

Pass bar (CUDA): MAE `< 1e-4`, max abs diff `< 5e-3`.

## End-to-end CLI

The prompt must contain the image markers expected by the selected mtmd model.
State and noise files contain raw little-endian f32 values.

```bash
./build/bin/llama-vla-cli \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --vla /path/to/vla-f32.gguf \
  --image /path/to/image.jpg \
  --prompt '<image>\nPredict the robot controls.' \
  --state /path/to/state.bin \
  --noise /path/to/noise.bin \
  --output /path/to/controls.bin
```

Common metadata (`vla.state_dim`, `vla.control_dim`,
`vla.control_horizon`, `vla.conditioning_dim`, and
`vla.n_embodiments`) is validated before model dispatch. Model-specific
metadata remains under the `mra.*` namespace.

## Server

Build with `LLAMA_BUILD_VLA=ON`, then load all three components:

```bash
./build/bin/llama-server \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --vla-model /path/to/vla-f32.gguf
```

`POST /v1/vla/predictions` accepts one text or multimodal input together with
`state`, optional `noise`, and optional `embodiment_id`. `GET /props` reports
the loaded VLA type and its state/control dimensions.

## Current limits

- Weights: **F32 only**
- Denoising: **fixed 4 steps** (`clean_action`)
- Proprio: **concat** only
- `control_horizon` is a convert-time CLI flag (not unique in tensor shapes)

Verbose load/predict logs: set `VLA_VERBOSE=1`.
