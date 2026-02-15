# llama-server-simulator Implementation Plan

## Overview
Create a standalone Python script that simulates a llama-server HTTP endpoint for testing the eval script.

## Goals
1. Simulate llama-server's `/v1/chat/completions` endpoint
2. Accept requests and respond with expected answers from AIME dataset
3. Implement configurable success rate (sometimes right, sometimes wrong)
4. Use regex matching to find questions in incoming requests
5. Test with curl requests before integrating with eval script

## Implementation Plan

### Phase 1: Basic Simulator Structure
- Create `llama-server-simulator.py` script
- Set up Flask/FastAPI HTTP server
- Implement `/v1/chat/completions` endpoint
- Handle basic request/response format

### Phase 2: AIME Dataset Integration
- Load AIME dataset
- Store questions and expected answers
- Implement regex matching to find questions in incoming requests
- Extract expected answer from matched question

### Phase 3: Response Generation
- Implement success rate configuration
- Randomly determine if response should be correct or incorrect
- Generate appropriate response based on success determination
- Format response in OpenAI-compatible format

### Phase 4: Testing
- Write curl commands to test basic functionality
- Test correct responses
- Test incorrect responses
- Test edge cases (no question found, etc.)

## Technical Details

### Server Framework
- Use Flask for simplicity
- Listen on configurable port
- Support JSON request/response format

### Request Format
```json
{
  "model": "llama",
  "messages": [
    {"role": "user", "content": "Question text here"}
  ],
  "temperature": 0,
  "max_tokens": 2048
}
```

### Response Format
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Answer text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

### AIME Dataset Integration
- Load from HuggingFace: "AI-MO/aimo-validation-aime"
- Store in memory for fast lookup
- Regex pattern to find question text in request
- Extract answer from matched question

### Success Rate Configuration
- Command-line argument: `--success-rate 0.8` (80% success rate)
- Randomly determine correctness based on rate
- Log when responses are correct vs incorrect

### Testing Strategy
1. Start simulator with default settings
2. Send curl request with known question
3. Verify response contains expected answer
4. Test with different success rates
5. Test edge cases

## Implementation Steps

### Step 1: Basic Server Setup
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    # Handle request
    return jsonify(response)
```

### Step 2: Load AIME Dataset
```python
import datasets

ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
# Store in memory
```

### Step 3: Regex Matching
```python
import re

def find_question_in_request(request_text):
    # Regex pattern to find question
    pattern = r"question:\s*(.*?)\n"
    match = re.search(pattern, request_text, re.DOTALL)
    return match.group(1) if match else None
```

### Step 4: Response Generation
```python
import random

def generate_response(question, success_rate):
    if random.random() < success_rate:
        return get_expected_answer(question)
    else:
        return get_wrong_answer(question)
```

### Step 5: Testing with Curl
```bash
curl -X POST http://localhost:8033/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Question text"}]
  }'
```

## Configuration Options
- `--port`: Server port (default: 8033)
- `--success-rate`: Success rate 0-1 (default: 0.8)
- `--host`: Server host (default: localhost)
- `--dataset-split`: AIME split to use (default: train)

## Expected Output
```
=== llama-server-simulator ===
Server running on http://localhost:8033
Success rate: 0.8
AIME dataset loaded: 1000 questions
```

## Testing Checklist
- [ ] Server starts successfully
- [ ] Basic request/response works
- [ ] Correct answer returned when success rate allows
- [ ] Wrong answer returned when success rate doesn't allow
- [ ] No question found returns error
- [ ] Multiple requests work correctly
- [ ] Different success rates work as expected

## Next Steps

1. ✓ Implement basic server structure
2. ✓ Load AIME dataset
3. ✓ Implement regex matching
4. ✓ Add response generation with success rate
5. ✓ Test with curl commands
6. ✓ Integrate with eval script once simulator works
7. ✓ Implement eval state object
8. ✓ Implement processor object
9. ✓ Add real-time progress reporting
10. ✓ Add enhanced grading system with LLM judge
