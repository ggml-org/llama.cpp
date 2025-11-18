# Simple CLI Chat - Usage Examples

Quick examples to get you started with the chat application.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (edit paths to match your setup)
cp config.yaml ~/.llama_chat/config.yaml
nano ~/.llama_chat/config.yaml

# 3. Start chatting!
python3 chat.py
```

## Example Sessions

### 1. General Chat

```bash
$ python3 chat.py

You: What is the capital of France?
Assistant: The capital of France is Paris.

You: Tell me more about it
Assistant: Paris is the largest city in France and has been the country's capital
since the 12th century. It's known for landmarks like the Eiffel Tower, the Louvre
Museum, and Notre-Dame Cathedral...

You: /save
[Session saved to ~/.llama_chat/history/session_20240115_143022.json]

You: /quit
[Goodbye!]
```

### 2. Coding Assistant

```bash
$ python3 chat.py --system-prompt coding --session python-help

You: How do I read a JSON file in Python?
Assistant: Here's how to read a JSON file in Python:

```python
import json

# Read JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Now 'data' contains your JSON data
print(data)
```

You: What if the file doesn't exist?
Assistant: You should add error handling:

```python
import json

try:
    with open('data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found!")
except json.JSONDecodeError:
    print("Invalid JSON format!")
```
```

### 3. Learning Session

```bash
$ python3 chat.py --system-prompt tutor --session learning

You: Explain gradient descent
Assistant: Gradient descent is an optimization algorithm. Think of it like walking
down a hill - you take steps in the direction that goes down most steeply...

You: Can you give a simple example?
Assistant: Sure! Imagine you want to find the minimum of f(x) = x². Here's the process:
1. Start at some point, say x = 4
2. Calculate the slope (derivative): f'(4) = 2*4 = 8
3. Move opposite to the slope: x = 4 - (learning_rate * 8)
...
```

### 4. Named Session Management

```bash
# Create a project-specific session
$ python3 chat.py --session my-project

You: Let's plan the architecture for a web app
Assistant: Great! Let's break down the architecture...

# Later, resume the same session
$ python3 chat.py --session my-project
[Loaded session from ~/.llama_chat/history/my-project.json]
[Messages in history: 12]

You: What did we decide about the database?
Assistant: Looking at our previous discussion, we decided to use...
```

### 5. Switching Models

```bash
# Use a different model
$ python3 chat.py --model ./models/mistral-7b.Q4_K_M.gguf

# Use with custom config
$ python3 chat.py --config my-config.yaml --model ./models/codellama.gguf
```

## Configuration Examples

### High-Quality Output

```yaml
generation:
  temperature: 0.3      # More deterministic
  top_p: 0.9
  top_k: 40
  max_tokens: 1024      # Longer responses
```

### Creative Writing

```yaml
generation:
  temperature: 1.2      # More creative
  top_p: 0.95
  top_k: 80
  max_tokens: 2048
```

### Fast Responses

```yaml
generation:
  temperature: 0.7
  max_tokens: 256       # Shorter responses

model:
  context_size: 1024    # Smaller context
  threads: 8            # More threads
```

### GPU Acceleration

```yaml
model:
  gpu_layers: 33        # Offload to GPU
  threads: 4            # Fewer CPU threads when using GPU
  context_size: 4096    # Can use more context with GPU
```

## Advanced Examples

### Batch Processing Questions

```bash
# Create a file with questions
cat > questions.txt << 'EOF'
What is Python?
How do I use virtual environments?
EOF

# Process each question (requires script modification)
# Or use in interactive mode
```

### Custom System Prompts

Add to config.yaml:

```yaml
system_prompts:
  sql_expert: |
    You are an expert SQL database administrator.
    Provide efficient, secure SQL queries.
    Explain your reasoning.

  math_tutor: |
    You are a patient mathematics tutor.
    Break down complex problems step by step.
    Use examples and visual descriptions.
```

Use them:

```bash
python3 chat.py --system-prompt sql_expert
python3 chat.py --system-prompt math_tutor
```

## Workflow Examples

### Code Review Workflow

```bash
# Start code review session
$ python3 chat.py --session code-review --system-prompt coding

You: Review this code:
[paste code]

# Get feedback
Assistant: [provides review]

# Make changes
You: Here's the updated version:
[paste updated code]

# Save session
You: /save
```

### Research Workflow

```bash
# Start research session
$ python3 chat.py --session research-topic --system-prompt researcher

# Multiple focused questions
You: What are the main approaches to attention mechanisms?
You: Compare self-attention vs cross-attention
You: What are the computational costs?

# List sessions to find this later
You: /sessions
```

## Tips and Tricks

### 1. Quick Commands

```
/help      - Show commands
/save      - Save now (auto-saves on exit anyway)
/clear     - Start fresh conversation
/sessions  - List all saved sessions
/quit      - Exit
```

### 2. Session Organization

```bash
# Use descriptive session names
python3 chat.py --session project-planning
python3 chat.py --session python-learning
python3 chat.py --session blog-writing
```

### 3. Context Management

```bash
# Use /clear when conversation gets too long
You: /clear
[History cleared]

# Or start fresh without loading history
python3 chat.py --no-history
```

### 4. Finding Old Sessions

```bash
# List all sessions
You: /sessions

# Or check the directory
ls -lh ~/.llama_chat/history/

# View a session file
cat ~/.llama_chat/history/my-session.json | jq .
```

### 5. Exporting Conversations

```bash
# Sessions are JSON files, easy to export
cd ~/.llama_chat/history/

# View messages
jq '.messages' my-session.json

# Export to text
jq -r '.messages[] | "\(.role): \(.content)\n"' my-session.json > conversation.txt
```

## Common Patterns

### Pattern 1: Iterative Development

```
You: Write a function to sort a list
Assistant: [provides code]
You: Add error handling
Assistant: [adds error handling]
You: Add type hints
Assistant: [adds type hints]
You: /save
```

### Pattern 2: Learning with Examples

```
You: Explain concept X
Assistant: [explains]
You: Give me an example
Assistant: [example]
You: What about edge case Y?
Assistant: [addresses edge case]
```

### Pattern 3: Brainstorming

```
You: Ideas for a blog post about Python
Assistant: [lists ideas]
You: Expand on idea #3
Assistant: [expands]
You: Create an outline
Assistant: [creates outline]
```

## Troubleshooting Examples

### Model Too Slow

```yaml
# In config.yaml, reduce complexity
generation:
  max_tokens: 256     # Shorter responses

model:
  context_size: 1024  # Smaller context
  threads: 8          # More threads
```

### Out of Memory

```yaml
# Use smaller model or reduce context
model:
  context_size: 512   # Much smaller
  path: ./models/model.Q4_K_M.gguf  # Use Q4 instead of Q6
```

### Poor Quality Responses

```yaml
# Adjust temperature and use better model
generation:
  temperature: 0.7    # Balanced
  top_p: 0.9

model:
  path: ./models/model.Q6_K.gguf  # Higher quality quantization
```
