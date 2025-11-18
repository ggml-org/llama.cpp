# Tutorial 3: Advanced Prompt Engineering with Constraints

**Module 5 - Advanced Inference**
**Duration**: 60-90 minutes

## Overview

Master advanced prompting techniques using grammar constraints, optimized sampling, and production-grade patterns for reliable LLM outputs.

## Techniques Covered

1. **Grammar-constrained prompting** for structured outputs
2. **Sampling strategy optimization** per task
3. **Few-shot learning** with examples
4. **Chain-of-thought** for complex reasoning
5. **Self-consistency** for reliability

## Part 1: Grammar-Constrained Prompting (20 min)

### Function Calling with Guaranteed Format

**Goal**: Generate valid function calls every time

```python
# Define function call grammar
FUNCTION_CALL_GRAMMAR = """
root ::= function-call

function-call ::= function-name "(" arguments ")"

function-name ::=
  "get_weather" |
  "search_web" |
  "send_email" |
  "calculate"

arguments ::= "{" argument-list "}"

argument-list ::= argument ("," argument)*

argument ::= string-key ":" value

string-key ::= "\\"" [a-zA-Z_]+ "\\""

value ::= string-value | number-value

string-value ::= "\\"" [^"]* "\\""

number-value ::= [0-9]+
"""

# Prompt with grammar
response = model.generate(
    prompt="""
    User: What's the weather in San Francisco?
    Assistant: I'll check the weather for you.
    Function call:
    """,
    grammar=FUNCTION_CALL_GRAMMAR,
    temperature=0.2  # Low temp for accuracy
)

# Output: get_weather({"location": "San Francisco"})
# Guaranteed valid format!
```

### JSON Schema Enforcement

```python
USER_PROFILE_SCHEMA = """
root ::= user-object

user-object ::=
  "{" ws
    "\\"name\\"" ws ":" ws string ws "," ws
    "\\"age\\"" ws ":" ws number ws "," ws
    "\\"email\\"" ws ":" ws email-string ws "," ws
    "\\"role\\"" ws ":" ws role ws
  "}"

string ::= "\\"" [a-zA-Z ]+ "\\""
email-string ::= "\\"" [a-zA-Z0-9._-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,} "\\""
number ::= [0-9]+
role ::= "\\"admin\\"" | "\\"user\\"" | "\\"guest\\""
ws ::= [ \\t\\n]*
"""

response = model.generate(
    prompt="Generate a user profile for Alice, a 30-year-old admin:",
    grammar=USER_PROFILE_SCHEMA,
    temperature=0.5
)

# Output: {"name": "Alice", "age": 30, "email": "alice@example.com", "role": "admin"}
# Always valid JSON!
```

## Part 2: Task-Optimized Sampling (20 min)

### Code Generation (Low Temperature)

```python
def generate_code(prompt: str) -> str:
    """Optimized for syntactically correct code"""
    return model.generate(
        prompt=f"```python\n{prompt}\n",
        temperature=0.2,    # Low for determinism
        top_p=0.95,         # Slightly permissive
        max_tokens=512,
        stop=["```"]        # Stop at code block end
    )

# Example
code = generate_code(
    "def fibonacci(n):\n    # Calculate nth fibonacci number"
)
```

### Creative Writing (Mirostat)

```python
def generate_story(prompt: str) -> str:
    """Optimized for creative, coherent stories"""
    return model.generate(
        prompt=prompt,
        mirostat=2,         # Mirostat V2
        mirostat_tau=6.0,   # Target perplexity
        mirostat_eta=0.1,   # Learning rate
        temperature=1.0,    # Let Mirostat control
        max_tokens=1000
    )

# Example
story = generate_story(
    "Once upon a time in a magical forest, a young wizard discovered"
)
```

### Factual Q&A (Balanced)

```python
def answer_question(question: str) -> str:
    """Optimized for accurate, concise answers"""
    return model.generate(
        prompt=f"Question: {question}\nAnswer:",
        temperature=0.3,    # Low but not greedy
        top_p=0.9,
        min_p=0.05,        # Filter unlikely tokens
        max_tokens=100,
        stop=["\n\n"]      # Stop at paragraph break
    )
```

## Part 3: Few-Shot Learning (15 min)

### Pattern: Format Examples

```python
def few_shot_prompt(task: str, examples: List[tuple], query: str) -> str:
    """Build few-shot prompt with examples"""
    prompt = f"Task: {task}\n\n"

    # Add examples
    for input_ex, output_ex in examples:
        prompt += f"Input: {input_ex}\n"
        prompt += f"Output: {output_ex}\n\n"

    # Add query
    prompt += f"Input: {query}\n"
    prompt += f"Output:"

    return prompt

# Example: Sentiment classification
examples = [
    ("I love this product!", "positive"),
    ("Terrible experience.", "negative"),
    ("It's okay, nothing special.", "neutral")
]

prompt = few_shot_prompt(
    task="Classify sentiment",
    examples=examples,
    query="Best purchase I ever made!"
)

response = model.generate(
    prompt=prompt,
    temperature=0.2,
    max_tokens=10
)
# Output: "positive"
```

### Pattern: Grammar + Few-Shot

```python
# Combine examples with grammar constraints
EMAIL_EXTRACTION_GRAMMAR = """
root ::= email-list
email-list ::= email ("," email)*
email ::= "\\"" [a-zA-Z0-9._-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,} "\\""
"""

prompt = """
Extract all email addresses from the text.

Example 1:
Text: Contact us at support@example.com or sales@company.org
Emails: "support@example.com", "sales@company.org"

Example 2:
Text: Reach out to john.doe@email.com for more info
Emails: "john.doe@email.com"

Text: Send your resume to jobs@startup.io or hr@bigcorp.com
Emails:"""

emails = model.generate(
    prompt=prompt,
    grammar=EMAIL_EXTRACTION_GRAMMAR,
    temperature=0.1
)
# Guaranteed valid email format!
```

## Part 4: Chain-of-Thought (15 min)

### Explicit Reasoning Steps

```python
def chain_of_thought(problem: str) -> str:
    """Prompt for step-by-step reasoning"""
    prompt = f"""
Solve this problem step by step:

Problem: {problem}

Let's think through this carefully:

Step 1:"""

    return model.generate(
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )

# Example
solution = chain_of_thought(
    "If a train travels 120 miles in 2 hours, "
    "how far will it travel in 5 hours at the same speed?"
)

"""
Output:
Step 1: Calculate the train's speed
Speed = Distance / Time = 120 miles / 2 hours = 60 mph

Step 2: Use the speed to find distance for 5 hours
Distance = Speed × Time = 60 mph × 5 hours = 300 miles

Answer: 300 miles
"""
```

### Structured CoT with Grammar

```python
COT_GRAMMAR = """
root ::= reasoning

reasoning ::= steps answer

steps ::= "Steps:" ws step+

step ::= "- " [^\\n]+ "\\n"

answer ::= "Answer:" ws [^\\n]+

ws ::= [ \\t\\n]*
"""

response = model.generate(
    prompt=f"""
Solve: {problem}

Think step-by-step:
""",
    grammar=COT_GRAMMAR,
    temperature=0.3
)
# Guaranteed structured reasoning!
```

## Part 5: Self-Consistency (15 min)

### Multiple Samples + Voting

```python
def self_consistent_answer(
    question: str,
    num_samples: int = 5
) -> str:
    """Generate multiple answers and pick most common"""
    answers = []

    for _ in range(num_samples):
        response = model.generate(
            prompt=f"Question: {question}\nAnswer:",
            temperature=0.7,  # Some diversity
            max_tokens=100
        )
        answers.append(response.strip())

    # Vote for most common answer
    from collections import Counter
    vote = Counter(answers)
    best_answer, _ = vote.most_common(1)[0]

    return best_answer

# Example
answer = self_consistent_answer(
    "What is the capital of France?"
)
# More reliable than single sample!
```

### Verification Pattern

```python
def verify_answer(question: str, answer: str) -> bool:
    """Ask model to verify its own answer"""
    verification_prompt = f"""
Question: {question}
Proposed Answer: {answer}

Is this answer correct? Reply with only "yes" or "no".
Verification:"""

    verification = model.generate(
        verification_prompt,
        temperature=0.1,
        max_tokens=5
    ).strip().lower()

    return "yes" in verification

# Use in loop
best_answer = None
for _ in range(3):  # Try up to 3 times
    answer = model.generate(question)
    if verify_answer(question, answer):
        best_answer = answer
        break

return best_answer
```

## Part 6: Production Patterns (15 min)

### Retry with Degradation

```python
class RobustGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str, **kwargs):
        """Generate with automatic retry and degradation"""
        strategies = [
            # Try 1: Optimal settings
            {'temperature': 0.7, 'top_p': 0.9, 'min_p': 0.05},

            # Try 2: More conservative
            {'temperature': 0.5, 'top_p': 0.95},

            # Try 3: Very conservative
            {'temperature': 0.2, 'top_p': 0.99},

            # Try 4: Greedy (fallback)
            {'temperature': 0.0}
        ]

        last_error = None

        for i, strategy in enumerate(strategies):
            try:
                return self.model.generate(
                    prompt,
                    **{**kwargs, **strategy}
                )
            except Exception as e:
                last_error = e
                logging.warning(f"Attempt {i+1} failed: {e}")

        raise last_error

# Usage
generator = RobustGenerator(model)
response = generator.generate(
    "Explain quantum computing",
    max_tokens=200
)
```

### Caching for Common Patterns

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(prompt: str, temperature: float) -> str:
    """Cache responses for deterministic prompts"""
    return model.generate(
        prompt,
        temperature=temperature,
        max_tokens=200
    )

# Fast repeated calls
response1 = cached_generate("What is AI?", 0.2)
response2 = cached_generate("What is AI?", 0.2)  # Instant!
```

## Best Practices Summary

### Grammar Constraints

✅ **When to use**:
- Structured outputs (JSON, XML, code)
- Function calling
- Data extraction
- APIs

✅ **Tips**:
- Keep grammars simple
- Test with edge cases
- Cache parsed grammars
- Measure overhead

### Sampling Strategies

| Task | Temperature | Top-P | Min-P | Mirostat |
|------|------------|-------|-------|----------|
| Code | 0.1-0.3 | 0.95 | 0.0 | No |
| Q&A | 0.3-0.5 | 0.9 | 0.05 | No |
| Creative | 0.7-0.9 | 0.85 | 0.1 | Yes (tau=6) |
| Analysis | 0.5-0.7 | 0.9 | 0.05 | Optional |

### Reliability Patterns

✅ **Self-consistency**: Multiple samples + voting
✅ **Verification**: Ask model to verify
✅ **Retry with degradation**: Fallback strategies
✅ **Few-shot examples**: Show desired format
✅ **Chain-of-thought**: Step-by-step reasoning

## Production Checklist

- [ ] Grammar constraints for structured outputs
- [ ] Task-optimized sampling settings
- [ ] Few-shot examples in prompts
- [ ] Retry logic with fallbacks
- [ ] Response validation
- [ ] Caching for common prompts
- [ ] Logging and monitoring
- [ ] A/B testing of prompts

## Common Pitfalls

❌ **Don't**: Use same settings for all tasks
✅ **Do**: Optimize per use case

❌ **Don't**: Trust single sample for critical tasks
✅ **Do**: Use self-consistency

❌ **Don't**: Ignore grammar overhead
✅ **Do**: Benchmark and optimize

❌ **Don't**: Hard-code prompts
✅ **Do**: Version and test prompts

## Further Reading

- Prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering
- Chain-of-thought paper: "Chain-of-Thought Prompting Elicits Reasoning"
- Self-consistency paper: "Self-Consistency Improves Chain of Thought Reasoning"

🎯 **You now have production-grade prompting skills!**
