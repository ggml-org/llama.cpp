# Example 02: Basic Chat

**Module**: 1.4 - Basic Inference
**Difficulty**: Beginner
**Estimated Time**: 20 minutes

## Overview

This example demonstrates how to build an interactive chatbot with conversation history and context management. You'll learn how to create a multi-turn conversation system that remembers previous exchanges.

## Learning Objectives

By completing this example, you will:
- ✅ Build an interactive chat loop
- ✅ Manage conversation history across multiple turns
- ✅ Handle user commands (exit, clear, help)
- ✅ Build prompts that include conversation context
- ✅ Understand context window limitations
- ✅ Implement proper input handling and error recovery

## Prerequisites

- Python 3.8 or higher
- llama-cpp-python installed
- Completed example 01-first-inference.py
- A GGUF model file

## Installation

```bash
pip install llama-cpp-python
```

## Usage

### Basic Usage

```bash
# Run with default model
python 02-basic-chat.py

# Run with custom model
python 02-basic-chat.py path/to/your/model.gguf
```

### Example Session

```
======================================================================
LLaMA.cpp Python - Basic Chat Example
======================================================================

This example demonstrates an interactive chat loop with context.

Commands:
  Type your message and press Enter to chat
  'exit' or 'quit' - Exit the chat
  'clear' or 'reset' - Clear conversation history
  'history' - Show conversation summary
  'help' - Show this help message

======================================================================

Loading model: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
Please wait...
✓ Model loaded (context: 2048 tokens)

Chat started! Type your message or 'help' for commands.

You: Hello! Who are you?
Assistant: Hello! I am an AI assistant designed to help answer questions and
provide information. How can I assist you today?

You: What's the capital of France?
Assistant: The capital of France is Paris. It's a beautiful city known for
its art, culture, and iconic landmarks like the Eiffel Tower.

You: Tell me more about it
Assistant: Paris is home to many world-famous museums like the Louvre, which
houses the Mona Lisa. The city is also known for its cuisine, fashion, and
romantic atmosphere. [Note: The bot remembers we're discussing Paris!]

You: history
Conversation has 6 messages

You: clear
✓ Conversation history cleared.

You: What were we talking about?
Assistant: I don't have any previous conversation history. Could you please
let me know what you'd like to discuss?
[After clearing, the bot doesn't remember Paris]

You: exit
Goodbye! Thanks for chatting.

======================================================================
Chat session ended.
Final status: Conversation has 4 messages
======================================================================
```

## Key Concepts Explained

### 1. ChatBot Class

The `ChatBot` class encapsulates conversation management:

```python
chatbot = ChatBot(
    model=model,
    system_prompt="You are a helpful AI assistant.",
    max_history=10  # Keep last 10 exchanges
)
```

**Key Features**:
- Maintains conversation history
- Builds prompts with context
- Limits history to prevent context overflow
- Provides conversation management methods

### 2. Conversation History Management

```python
# Message structure
{
    "role": "user" or "assistant",
    "content": "message text"
}

# Adding messages
chatbot.add_message("user", "Hello!")
chatbot.add_message("assistant", "Hi there!")
```

**Why Limit History?**
- Context windows have token limits (e.g., 2048 tokens)
- Old messages are dropped to make room for new ones
- Typical strategy: Keep last N user-assistant pairs

### 3. Prompt Building

The chatbot builds prompts in this format:

```
System: You are a helpful AI assistant.

User: Hello!
Assistant: Hi there! How can I help?

User: What's the capital of France?
Assistant:
```

The model completes from "Assistant:", and we parse its response.

**Note**: Different models may prefer different formats:
- **ChatML**: `<|im_start|>user\n...<|im_end|>`
- **Alpaca**: `### Instruction:\n...\n### Response:`
- **Llama-2**: `[INST] ... [/INST]`

### 4. Stop Sequences

```python
stop=["User:", "System:", "\n\n"]
```

Stop sequences tell the model when to stop generating:
- Prevents generating the next turn
- Avoids the model talking to itself
- Helps maintain turn structure

### 5. Interactive Loop

```python
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = chatbot.chat(user_input)
    print(f"Assistant: {response}")
```

**Key Features**:
- `input()` for user input
- Command parsing before generation
- Error handling for interrupts (Ctrl+C)
- Graceful exit handling

## Code Structure

```
02-basic-chat.py
├── ChatBot class
│   ├── __init__()           # Initialize with model and settings
│   ├── add_message()        # Add to conversation history
│   ├── build_prompt()       # Format history as prompt
│   ├── chat()               # Process user message → response
│   ├── reset()              # Clear history
│   └── get_history_summary() # Get stats
├── load_model()             # Load GGUF model
├── print_welcome()          # Show intro message
├── print_help()             # Show help text
└── main()                   # Interactive chat loop
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `exit`, `quit` | Exit the chat session |
| `clear`, `reset` | Clear conversation history |
| `history` | Show number of messages in history |
| `help` | Display help message |
| `Ctrl+C` | Interrupt and exit |

## Common Issues and Solutions

### Issue 1: Context Overflow

**Symptom**: Model becomes slow or generates poor responses after many turns

**Solution**:
- Reduce `max_history` parameter
- Implement smarter history pruning
- Increase `n_ctx` when loading model (if you have enough RAM)

```python
# Option 1: Reduce history
chatbot = ChatBot(model, max_history=5)

# Option 2: Increase context window
model = Llama(model_path=path, n_ctx=4096)
```

### Issue 2: Model Talks to Itself

**Symptom**: Response includes both user and assistant parts

**Example**:
```
You: Hello
Assistant: Hello! User: How are you? Assistant: I'm fine.
```

**Solution**: Add better stop sequences

```python
stop=["User:", "Human:", "\n\n", "###"]
```

### Issue 3: Responses Cut Off

**Symptom**: Responses end mid-sentence

**Solution**: Increase `max_tokens`

```python
response = chatbot.chat(user_message, max_tokens=512)
```

### Issue 4: No Memory Between Turns

**Symptom**: Bot doesn't remember previous messages

**Check**:
- Is history being added? (Check `add_message()` calls)
- Is `build_prompt()` including history?
- Was history cleared accidentally?

## Experiments to Try

### 1. Different System Prompts

Try various system prompts to change behavior:

```python
# Technical expert
system_prompt = "You are a computer science expert. Provide detailed technical explanations."

# Creative writer
system_prompt = "You are a creative writer. Be imaginative and poetic."

# Concise assistant
system_prompt = "You are a helpful assistant. Keep all responses under 50 words."
```

### 2. Adjust History Length

```python
# Short-term memory (good for quick Q&A)
max_history = 3

# Long-term memory (good for complex discussions)
max_history = 20
```

### 3. Custom Prompt Formats

Try implementing ChatML format:

```python
def build_prompt_chatml(self) -> str:
    prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
    for msg in self.conversation_history:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt
```

### 4. Token Counting

Add token counting to monitor context usage:

```python
def count_tokens(self, text: str) -> int:
    """Estimate token count (approximate)."""
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4
```

## Advanced Features to Implement

1. **Streaming Responses**: Display tokens as they're generated
2. **Conversation Export**: Save chat history to JSON
3. **Multi-User Support**: Separate contexts for different users
4. **Conversation Summarization**: Compress old history
5. **Typing Indicators**: Show "Assistant is typing..."
6. **Formatted Output**: Markdown rendering, syntax highlighting

## Performance Tips

1. **Use GPU acceleration**: `n_gpu_layers=-1`
2. **Limit context window**: Don't make `n_ctx` larger than needed
3. **Prune history aggressively**: Keep only relevant recent messages
4. **Cache common prompts**: Reuse system prompt across sessions
5. **Batch when possible**: (See example 05)

## Interview Topics

This example covers these interview topics:

**Conversation State Management**:
- How do you maintain context across turns?
- What are the trade-offs of keeping long vs. short history?

**Memory Constraints**:
- How do you handle context window limits?
- What strategies exist for compressing conversation history?

**User Experience**:
- How do you design intuitive chat interfaces?
- How do you handle errors gracefully?

**Typical Interview Question**:
> "Design a chatbot that remembers user preferences across sessions. How would you implement this?"

**Answer**: Use persistent storage (database/files) to save conversation history and user metadata. Load on session start, save on exit. Consider privacy implications and implement data retention policies.

## Next Steps

After completing this example:
- ✅ Implement conversation history export/import
- ✅ Try `03-sampling-parameters.py` to control generation quality
- ✅ Add streaming support for real-time responses
- ✅ Build a web UI using Flask or Gradio

## Related Documentation

- [Module 1.4: Basic Inference](../../modules/01-foundations/docs/)
- [Context Management](../../modules/01-foundations/docs/)
- [Lab 1.4: First Inference](../../modules/01-foundations/labs/)

---

**Author**: Agent 3 (Code Developer)
**Last Updated**: 2025-11-18
**Module**: 1.4 - Basic Inference
