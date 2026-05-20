# Build & setup — desktop solver

## Prerequisites

- Python **3.10+** (3.11 recommended)
- A C/C++ toolchain (`build-essential` on Debian / Xcode CLT on macOS),
  required by `llama-cpp-python` if a prebuilt wheel is not available
- ~6 GB free disk for the model + dependencies
- 8 GB RAM minimum (16 GB recommended for larger context windows)

## Install

Recommended: a project-local virtual environment.

```bash
cd desktop-solver
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `llama-cpp-python` falls back to building from source and you want to
enable a specific backend, set its CMake args before `pip install`:

```bash
# CPU (default) — no extra flags needed.

# CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-binary=:llama-cpp-python: llama-cpp-python

# Apple Metal:
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-binary=:llama-cpp-python: llama-cpp-python

# Vulkan (cross-vendor GPUs):
CMAKE_ARGS="-DGGML_VULKAN=on" pip install --no-binary=:llama-cpp-python: llama-cpp-python
```

## Models

**Auto-discovery.** On launch the GUI scans for `.gguf` files under common
locations (`~/Downloads`, `~/Documents`, `~/Models`,
`~/.cache/huggingface`, `~/.cache/lm-studio`, `~/.lmstudio/models`,
`~/.ollama/models`, `~/.local/share/models`, `/opt/models`, …) and your
home tree, with a total deadline of ~90 s. The highest-scoring local
model is loaded automatically — math-tuned filenames
(`math` / `qwen-math` / `deepseek-math` / `mathstral`) and Q4 quantisations
score highest, and anything larger than ~70 % of system RAM is heavily
penalised. Pass `--model /path/to/model.gguf` to skip the scan.

Pick one math-tuned GGUF that fits your RAM budget:

| Model                                       | Approx RSS | Notes                                    |
| ------------------------------------------- | ---------- | ---------------------------------------- |
| `Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf`      | ~4.4 GB    | Default recommendation for 8 GB systems  |
| `Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf`   | ~1 GB      | If memory is tight or for quick testing  |
| `Phi-3.5-mini-Q4_K_M.gguf`                  | ~2.5 GB    | Strong general reasoning, less math-tuned|
| `DeepSeek-Math-7B-Instruct-Q4_K_M.gguf`     | ~4.3 GB    | Comparable to Qwen-Math 7B               |

Download from Hugging Face (any of `bartowski`, `lmstudio-community`, etc.),
place anywhere on disk, then pass `--model <path>` when launching.

## Run

```bash
# GUI (will prompt for a model file if --model is omitted)
python run.py --model /path/to/model.gguf
python run.py --model /path/to/model.gguf --n-ctx 8192 --n-threads 6

# Headless CLI
python -m desktop_solver.cli --model /path/to/model.gguf \
    "Solve x^2 + 5x + 6 = 0"

# Pipe the problem in from stdin (handy for shell loops)
echo "Find sin 30 degrees" | python -m desktop_solver.cli --model model.gguf
```

## Memory budget (8 GB system)

| Component                                | Approx RSS |
| ---------------------------------------- | ---------- |
| 7 B Q4_K_M model (mmap'd)                | ~4.4 GB    |
| llama.cpp context (`n_ctx=4096`)         | ~600 MB    |
| Python + SymPy + PyQt6 + matplotlib      | ~250 MB    |
| GUI runtime                              | ~80 MB     |
| **Total**                                | **~5.3 GB**|

Headroom is ~2.5 GB on an 8 GB machine — comfortable.

## GPU offload (optional)

```bash
python run.py --model model.gguf --n-gpu-layers 32
```

`--n-gpu-layers` only has an effect if `llama-cpp-python` was built with a
GPU backend (see "Install" above).

## Tests

The pure-Python tests do not need the model:

```bash
pip install -e .[test]
pytest -q
```

## Troubleshooting

**"failed to load model" on launch.**
The GGUF file is missing or unreadable. Verify with `ls -l` and that the
file is a valid llama.cpp GGUF (binary, starts with `GGUF` magic).

**LaTeX renders as plain text.**
matplotlib's mathtext doesn't need a system TeX install; if rendering is
blank, check the log panel — usually a malformed LaTeX from the LLM. The
answer is also shown as text in the SymPy panel below.

**Solve button stays disabled.**
The model is still loading. The status bar says "Loading model — please
wait" until the worker emits "model ready".
