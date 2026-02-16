# llama-eval Codebase Guidelines

## Overview

This directory contains Python evaluation tools for llama.cpp:
- `llama-eval.py` - Main evaluation tool with multiple datasets (AIME, AIME2025, GSM8K, GPQA)
- `llama-server-simulator.py` - Flask-based server simulator for testing
- `test-simulator.sh` - Test script for the simulator

## Build/Run Commands

### Virtual Environment
The project uses a virtual environment located at `venv/`:
```bash
source venv/bin/activate
```

### Running the Main Evaluator
```bash
python llama-eval.py \
  --server http://127.0.0.1:8013 \
  --model gpt-oss-20b-hf-low \
  --dataset aime \
  --n_cases 10 \
  --grader-type llm \
  --seed 42
```

### Running the Simulator (for testing)
```bash
python llama-server-simulator.py --port 8033 --success-rate 0.8
```

### Running Tests
```bash
./test-simulator.sh
```

## Code Style Guidelines

### Imports
- Standard library imports first (argparse, json, os, re, subprocess, sys, time)
- Third-party imports (requests, tqdm, datasets, flask) after standard library
- Relative imports not used
- Group imports by category with blank line between groups

### Formatting
- 4-space indentation
- Max line length: 125 characters (per parent project's .flake8)
- Use double quotes for strings
- Use triple double quotes for docstrings
- Binary operators at the beginning of continued lines

### Naming Conventions
- Classes: PascalCase (e.g., `AimeDataset`, `Grader`, `Processor`)
- Functions: snake_case (e.g., `normalize_number`, `get_prompt`)
- Variables: snake_case (e.g., `question_text`, `correct_count`)
- Constants: UPPER_SNAKE_CASE (e.g., `GRADER_PATTERNS`, `TEMPLATE_REGISTRY`)
- Private methods: prefix with underscore (e.g., `_load_dataset`, `_grade_regex`)

### Types
- Use type hints for all function signatures
- Import from `typing` module: `Dict`, `List`, `Optional`, `Any`, `Tuple`
- Use `@dataclass` for data structures
- Prefer `Optional[T]` over `Union[T, None]`

### Error Handling
- Use try/except for network requests and file operations
- Return `None` or `False` on errors when appropriate
- Use `ValueError` for invalid arguments
- Use `FileNotFoundError` for missing files
- CLI scripts should handle exceptions gracefully

### Dataclasses
- Use `@dataclass` for structured data
- Define fields with explicit types
- Use `Optional[T]` for nullable fields
- Provide default values where appropriate

### String Formatting
- Use f-strings for formatting (Python 3.6+)
- Use triple double quotes for multi-line strings
- Escape backslashes in regex patterns: `r'\\boxed{(\d+)}'`

### File Paths
- Use `pathlib.Path` instead of string paths
- Create directories with `mkdir(parents=True, exist_ok=True)`
- Use `Path.home()` for user home directory

### Logging
- Use `print()` for user-facing output
- Use `sys.stderr` for debug logging
- Simulator writes debug logs to `/tmp/simulator-debug.log`

### Testing

- Test script uses bash with `set -e` for strict error handling
- Simulator runs in background with PID tracking
- Tests verify correct answers, error cases, and edge cases
- Use `curl` for HTTP testing in shell scripts

### Whitespace Cleanup
- Remove trailing whitespace from all lines
- When making edits, do not leave trailing whitespace

## Dataset Support

### AIME Dataset
- 90 questions from 2025 AIME competition
- Answers in `\boxed{answer}` format
- Supports regex, CLI, and LLM grading

### AIME2025 Dataset
- 30 questions from 2025 AIME I & II
- Answers in `\boxed{answer}` format
- Requires loading two config parts

### GSM8K Dataset
- 7473 math word problems
- Answers numeric values with `####` separator
- Supports regex, CLI, and LLM grading

### GPQA Dataset
- 198 questions from GPQA Diamond
- Multiple choice with shuffled options (A, B, C, D)
- **Requires LLM grader** (returns letter A/B/C/D)

## Grading Types

### Regex Grader
- Built-in patterns per dataset
- Prioritizes `\boxed{}` for AIME datasets
- Extracts last number for GSM8K

### CLI Grader
- External script interface
- Call: `grader.sh --answer <pred> --expected <gold>`
- Exit code 0 = correct, non-zero = incorrect

### LLM Grader
- Uses judge model for answer extraction
- Includes few-shot examples
- Case-insensitive comparison
- Required for GPQA

## Configuration

### Sampling Parameters (Optional)
- `--temperature`: Sampling temperature
- `--top-k`: Top K sampling
- `--top-p`: Top P sampling
- `--min-p`: Min P sampling
- Only passed to API if explicitly specified

### Default Values
- `--n_predict`: -1 (infinite)
- `--grader-type`: llm
- `--seed`: 1234
- `--threads`: 32
- `--output`: llama-eval-state.json

## Output Format

### Progress Table
- Shows task ID, dataset, prompt (truncated to 43 chars), expected answer, status
- Uses `tqdm` for progress bars

### Results Summary
- Format: `Results: X/Y correct (Z%)`
- Displayed after all tasks complete

### JSON Output
- Complete eval state saved to output file
- Contains: task IDs, correctness, prompts, extracted answers, sampling config
- Uses `dataclasses.asdict()` for serialization

## HuggingFace Datasets

- Cache directory: `~/.cache/huggingface/datasets`
- Set via `HF_DATASETS_CACHE` environment variable
- Telemetry disabled via `HF_HUB_DISABLE_TELEMETRY=1`
- Datasets loaded with `datasets.load_dataset()`

## Flask Simulator

- Runs on configurable port (default: 5000)
- Endpoint: `/v1/chat/completions` (OpenAI-compatible)
- Uses Dice coefficient for question matching
- Configurable success rate for testing
- Debug logs to `/tmp/simulator-debug.log`
