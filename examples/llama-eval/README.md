# llama-eval Evaluation Tool

Simple evaluation tool for llama.cpp with support for multiple datasets.

## Features

- **Multiple Datasets**: AIME, GSM8K, GPQA
- **Flexible Grading**: Regex, CLI, or LLM-based grading
- **Parallel Processing**: Configurable thread count
- **Real-time Feedback**: Progress tracking with detailed output
- **Sampling Parameters**: Temperature, Top K, Top P, Min P
- **JSON Output**: Complete eval state saved for debugging

## Usage

```bash
python llama-eval.py \
  --server http://127.0.0.1:8013 \
  --model gpt-oss-20b-hf-low \
  --judge-model gpt-oss-20b-hf-medium \
  --dataset aime \
  --n_cases 10 \
  --grader-type llm \
  --seed 42
```

## CLI Arguments

- `--server`: llama-server URL (default: http://127.0.0.1:8013)
- `--model`: Model name for evaluation (default: llama)
- `--judge-model`: Model name for LLM judge (default: same as main model)
- `--judge-server`: Server URL for LLM judge (default: same as main server)
- `--dataset`: Dataset type (aime, gsm8k, gpqa)
- `--n_cases`: Number of cases to evaluate (default: all)
- `--n_predict`: Max tokens to predict per prompt (default: -1, infinite)
- `--temperature`: Sampling temperature (default: not passed)
- `--top-k`: Top K sampling (default: not passed)
- `--top-p`: Top P sampling (default: not passed)
- `--min-p`: Min P sampling (default: not passed)
- `--threads`: Number of threads for parallel requests (default: 32)
- `--verbose`: Show detailed output for each case
- `--output`: Output file for eval state (default: llama-eval-state.json)
- `--grader-type`: Grader type (regex, cli, llm, default: llm)
- `--grader-script`: Path to CLI grader script (required for --grader-type cli)
- `--seed`: Random seed for shuffling (default: 1234)

## Datasets

### AIME
- 90 questions from 2025 AIME competition
- Answers in boxed format: `\boxed{answer}`
- Requires regex grader or LLM grader

### GSM8K
- 7473 math word problems
- Answers are numeric values
- Requires regex grader or LLM grader

### GPQA
- 198 questions from GPQA Diamond dataset
- Multiple choice with shuffled options
- Requires LLM grader (returns letter A, B, C, or D)

## Grading Types

### Regex Grader
Built-in patterns for different datasets:
- AIME: `\boxed{(\d+)}|\b(\d+)\b`
- GSM8K: `\b(\d+)\b`
- GPQA: Letter extraction (A, B, C, D)

### CLI Grader
External script interface:
```bash
./grader.sh --answer <pred> --expected <gold>
```
Returns exit code 0 if correct, non-zero if incorrect.

### LLM Grader
Uses LLM to extract and compare answers:
- Configurable server and model
- Includes few-shot examples from sample answers
- Case-insensitive comparison

## Output

### Progress Table
```
  Task ID             Dataset  Prompt (first 43 chars)                        Expected    Status
  aime_000_001         AIME   Complete the following reactions and sel...    A          pending
```

### Results
```
============================================================
Results: 8/10 correct (80.0%)
============================================================
```

### JSON Output
Complete eval state saved to output file with:
- Task IDs and correctness status
- Prompts and extracted answers
- Sampling configuration
- Processing metadata
