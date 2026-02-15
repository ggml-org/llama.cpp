# llama-eval Implementation Summary

## Overview

Simple evaluation tool for llama.cpp with support for multiple datasets (AIME, GSM8K, GPQA) and flexible grading (regex, CLI, LLM).

## Key Features

- **Multiple Datasets**: AIME, GSM8K, GPQA with proper answer extraction
- **Flexible Grading**: Regex, CLI, or LLM-based grading
- **Parallel Processing**: Configurable thread count for concurrent requests
- **Sampling Parameters**: Temperature, Top K, Top P, Min P (optional)
- **Real-time Feedback**: Progress tracking with detailed output
- **JSON Output**: Complete eval state saved for debugging
- **GPQA Support**: Answer shuffling with reproducible results

## Architecture

### Eval State
```python
@dataclass
class EvalState:
    id: str
    tasks: List[str]
    task_states: Dict[str, Dict[str, Any]]
    sampling_config: Dict[str, Any]
```

### Processor
- Handles processing, grading, and state management
- Thread-safe concurrent execution
- Configurable sampling parameters

### Grader
- Abstract grading interface supporting multiple types
- Regex grader with dataset-specific patterns
- CLI grader with external script interface
- LLM grader with configurable server and model

### Datasets
- `AimeDataset`: 90 AIME 2025 questions
- `Gsm8kDataset`: 7473 math word problems
- `GpqaDataset`: 198 GPQA Diamond questions with shuffling

## Configuration

### Sampling Parameters (Optional)
- `--temperature`: Sampling temperature
- `--top-k`: Top K sampling
- `--top-p`: Top P sampling
- `--min-p`: Min P sampling
- Only passed if explicitly specified

### Grading Types
- **regex**: Built-in patterns for each dataset
- **cli**: External script with `--answer` and `--expected` args
- **llm**: LLM-based extraction with configurable server/model

## Output Format

### Progress Table
```
  Task ID             Dataset  Prompt (first 43 chars)                        Expected    Status
  aime_000_001         AIME   Complete the following reactions and sel...    A          pending
```

### Results Summary
```
============================================================
Results: 8/10 correct (80.0%)
============================================================
```

### JSON Output
Complete eval state with task IDs, correctness, prompts, extracted answers, and sampling configuration.

## Technical Details

- Default max tokens: -1 (infinite)
- Default grader type: llm
- Default seed: 1234
- Default threads: 32
- Prompt truncation: First 43 chars + padding + "..."
- GPQA requires LLM grader (returns letter A/B/C/D)
- Judge model defaults to evaluated model if not specified
