# llama-eval Implementation Discussion

## Overview
Discussion about implementing a lean evaluation tool for llama.cpp based on ggerganov's feedback in PR #18892.

## Key Requirements from ggerganov

### 1. Simplify and Focus on One Eval
- Start with AIME2025 (most familiar with it)
- Don't support multiple evals initially

### 2. Implement an "eval state" object
- ID
- List of tasks
- Task states
- Sampling config

### 3. Implement a "processor" object
- List of endpoints
- Threads per endpoint
- Grade/judge type (regex, endpoint, or CLI tool)

### 4. Processor responsibilities
- Accepts eval state
- Starts processing
- Dumps eval state periodically as it progresses

### 5. Real-time feedback
- Default: show "correct / not correct" for each task
- Verbose mode: show produced answer vs expected answer as soon as it completes

### 6. Grading approach
- Abstract grading to support external "grader" or "judge"
- Use LLM post-processing instead of regex (to avoid issues from GPT-OSS evals)

### 7. Output format
- Use structured output (JSON) instead of boxed text

## Current Implementation Analysis

### What exists in llama-eval.py:
- Multiple task implementations (AIME, GSM8K, MMLU, HellaSwag, ARC, WinoGrande)
- Regex-based answer extraction
- HTTP requests to OpenAI-compatible endpoint
- Checkpointing/resume capability
- Thread-based parallel execution
- Summary reporting

### What needs to be removed:
- All task implementations except AIME
- Regex-based grading
- Multiple endpoint support
- Complex task loading logic
- Summary reporting (replace with real-time feedback)

## Discussion Points

### 1. Eval State Object Structure
**Status: Under Discussion**

Questions:
- What fields should be in the eval state object?
- Should it include the actual prompts, or just metadata?
- How should task states be tracked?

### 2. Processor Architecture
**Status: Not Started**

Questions:
- Should the processor handle multiple endpoints (for distributed evaluation)?
- What's the threading model?
- How are endpoints configured?

### 3. Grader Interface
**Status: Not Started**

Questions:
- How should the grader be configured?
- Should it be a separate service, or a local LLM call?
- What's the interface for grading?

### 4. Checkpointing
**Status: Not Started**

Questions:
- Should the eval state be serialized to disk?
- How often should it be dumped?
- What format should it use?

### 5. Real-time Output
**Status: Not Started**

Questions:
- How should progress be displayed?
- Console output, file logging, or both?
- What verbosity levels are needed?

### 6. Output Format
**Status: Not Started**

Questions:
- Should responses be in JSON format?
- How should the grader interface work with JSON output?

## Next Steps

1. **Eval State Object** - Currently discussing
2. Processor Architecture
3. Grader Interface
4. Checkpointing
5. Real-time Output
6. Output Format

## References
- PR #18892: https://github.com/ggml-org/llama.cpp/pull/18892
- Discussion #18195: https://github.com/ggml-org/llama.cpp/discussions/18195
