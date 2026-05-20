# Desktop GCSE math solver

A desktop variant of the GCSE math solver, intended for workstations with
~8 GB of RAM. Unlike the embedded `gcse-solver/` (Pi Zero 2 W, C++ + ExprTk +
Lua), this variant uses:

- **A larger math-tuned LLM** (e.g. Qwen2.5-Math-7B-Instruct-Q4_K_M, ~4.4 GB)
- **SymPy** for exact symbolic computation
- **A verification step** that substitutes the answer back into the original
  problem; if the residual is non-zero, the LLM is re-prompted with the error
- **PyQt6 GUI** with LaTeX rendering for both the problem and the answer

## On the "100% correct" goal

No LLM-based solver can be proven 100% correct. What this project does
provide:

1. The LLM emits SymPy code, not freeform arithmetic; numbers stay exact.
2. Every answer is **verified** by re-substitution into the source equation.
   If verification fails, up to two self-correction passes are tried.
3. The GUI labels unverified answers explicitly so you can spot them.
4. The CLI returns a non-zero exit code on unverified or failed solves —
   handy for batch sweeping a question paper.

If a problem leaves the model genuinely stuck, the GUI surfaces the failing
code and the verification residual so you can intervene.

## Architecture

```
   user
    │
    ▼  natural language or LaTeX
┌────────────────┐
│  PyQt6 GUI     │
└─────┬──────────┘
      │ problem
      ▼
┌────────────────┐
│  LLM engine    │  llama-cpp-python, math-tuned 7B GGUF
└─────┬──────────┘
      │ raw text with [CALC] [VERIFY] [LATEX] [STEPS] blocks
      ▼
┌────────────────┐
│  tag_extractor │
└─────┬──────────┘
      │ four code/text fragments
      ▼
┌────────────────┐       ┌────────────────┐
│  executor      │──────►│  SymPy / math  │   exec [CALC], read `answer`
└─────┬──────────┘       └────────────────┘
      │
      ▼
┌────────────────┐
│  verifier      │  eval [VERIFY] in the same namespace → must be 0 / True
└─────┬──────────┘
      │ verified? → render in GUI ; not verified? → retry once or twice
      ▼
┌────────────────┐
│  GUI output    │  rendered LaTeX answer + working + SymPy code + log
└────────────────┘
```

## Quick start

```bash
cd desktop-solver
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pick a math-tuned GGUF and launch the GUI:
python run.py --model /path/to/Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf
```

The first launch loads the model (~10–30 s on modern CPUs). Once
"Ready" is shown in the status bar, type a problem and press **Solve**.

For full setup and model recommendations, see [BUILD.md](BUILD.md).

## CLI mode

```bash
python -m desktop_solver.cli --model /path/to/model.gguf "Solve x^2 + 5x + 6 = 0"
```

Add `--json` for machine-readable output. Exit code is `0` only when the
answer is both produced and verified.

## Examples

| Problem                                              | Verified answer            |
| ---------------------------------------------------- | -------------------------- |
| Solve `3x + 7 = 22`                                  | `5`                        |
| Solve `x^2 + 5x + 6 = 0`                             | `[-3, -2]`                 |
| Find `∫₀¹ (2x + 3) dx`                               | `4`                        |
| Find the gradient of `y = x^2 + 3x` at `x = 2`       | `7`                        |
| `sin(30°)`                                           | `1/2`                      |
| `P(no rain in 5 days)` if daily `P(rain) = 0.3`      | `16807/100000`             |
| Simplify `(x^2 - 4) / (x - 2)` for `x ≠ 2`           | `x + 2`                    |

## Layout

```
desktop-solver/
├── pyproject.toml
├── requirements.txt
├── run.py                          # convenience launcher
├── README.md
├── BUILD.md
├── src/desktop_solver/
│   ├── __init__.py
│   ├── __main__.py                 # python -m desktop_solver
│   ├── prompts.py                  # LLM system prompt + correction template
│   ├── llm_engine.py               # llama-cpp-python wrapper
│   ├── tag_extractor.py            # [CALC] [VERIFY] [LATEX] [STEPS] parsing
│   ├── executor.py                 # SymPy-backed exec in a curated namespace
│   ├── verifier.py                 # re-substitution check
│   ├── latex_io.py                 # parse + render LaTeX (matplotlib mathtext)
│   ├── solver.py                   # full pipeline with self-correction
│   ├── gui.py                      # PyQt6 main window + worker thread
│   └── cli.py                      # headless CLI / batch mode
└── tests/
    └── test_executor.py            # pytest suite (no LLM required)
```

## Tests

The pipeline below the LLM (executor, verifier, tag extractor) has full unit
test coverage and does not require a GGUF model:

```bash
cd desktop-solver
pip install -e .[test]
pytest
```

## Differences from `gcse-solver/`

| Dimension          | `gcse-solver/` (Pi Zero 2 W)     | `desktop-solver/` (8 GB desktop) |
| ------------------ | -------------------------------- | -------------------------------- |
| Language           | C++20                            | Python 3.10+                     |
| Model              | 1.5 B Q4 (~180 MB)               | 7 B Q4 (~4.4 GB)                 |
| Symbolic engine    | ExprTk (numeric) + Lua 5.4       | SymPy (exact)                    |
| LaTeX              | none                             | mathtext render, optional input  |
| Interface          | CLI / REPL                       | PyQt6 GUI + CLI                  |
| Verification       | retry once on error              | algebraic re-substitution        |
```
