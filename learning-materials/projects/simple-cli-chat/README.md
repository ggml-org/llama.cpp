# Simple CLI Chat

A production-ready command-line chat application for interacting with llama.cpp models. This project demonstrates best practices for building CLI tools and provides a solid foundation for learning about LLM integration.

## Features

- **Multi-turn Conversations**: Maintain context across multiple exchanges
- **Conversation History**: Automatic saving and loading of chat sessions
- **System Prompts**: Pre-configured prompts for different use cases (coding, creative writing, technical, etc.)
- **Configurable**: YAML-based configuration for easy customization
- **Session Management**: Create, save, and resume named chat sessions
- **Production Quality**: Error handling, logging, and clean architecture

## Project Structure

```
simple-cli-chat/
├── chat.py              # Main application script
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file

Generated at runtime:
~/.llama_chat/
├── config.yaml         # User configuration (auto-created)
└── history/            # Saved chat sessions
    ├── session_20240101_120000.json
    └── my-chat.json
```

## Installation

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python3 --version
   ```

2. **llama.cpp compiled and ready**
   ```bash
   # Make sure llama-cli is available in your llama.cpp directory
   ./llama-cli --version
   ```

3. **A GGUF model file**
   - Download from Hugging Face or other sources
   - Place in a `models/` directory

### Setup

1. **Clone or navigate to this directory**
   ```bash
   cd simple-cli-chat
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**
   ```bash
   # Copy the example config
   cp config.yaml ~/.llama_chat/config.yaml

   # Edit the configuration
   nano ~/.llama_chat/config.yaml
   ```

   Update these key settings:
   - `model.path`: Path to your GGUF model file
   - `llama_cpp_path`: Path to your llama-cli executable
   - `model.threads`: Number of CPU threads to use
   - `model.gpu_layers`: GPU layers to offload (if using GPU)

4. **Make the script executable (optional)**
   ```bash
   chmod +x chat.py
   ```

## Usage

### Basic Usage

Start a new chat session:
```bash
python3 chat.py
```

### Named Sessions

Create or load a named session:
```bash
python3 chat.py --session my-project-chat
```

### System Prompts

Use different personalities/modes:
```bash
# Coding assistant
python3 chat.py --system-prompt coding

# Creative writing
python3 chat.py --system-prompt creative

# Technical expert
python3 chat.py --system-prompt technical

# Patient tutor
python3 chat.py --system-prompt tutor
```

### Custom Configuration

Use a different config file:
```bash
python3 chat.py --config my-custom-config.yaml
```

### Override Model

Use a different model without changing config:
```bash
python3 chat.py --model ./models/mistral-7b.Q4_K_M.gguf
```

### Fresh Start

Start without loading history:
```bash
python3 chat.py --no-history
```

## Interactive Commands

While chatting, you can use these commands:

- `/help` - Show help message
- `/save` - Save current session
- `/clear` - Clear conversation history
- `/sessions` - List all saved sessions
- `/quit` or `/exit` - Exit the chat (auto-saves)

## Configuration Reference

### Model Settings

```yaml
model:
  path: ./models/llama-2-7b-chat.Q4_K_M.gguf  # Model file
  context_size: 2048                            # Context window
  threads: 4                                    # CPU threads
  gpu_layers: 0                                 # GPU offload (0=CPU only)
```

**Tips:**
- Increase `context_size` for longer conversations (uses more RAM)
- Set `threads` to your CPU core count for best performance
- Increase `gpu_layers` if you have a CUDA-compatible GPU

### Generation Parameters

```yaml
generation:
  temperature: 0.7        # Randomness (0.0-2.0)
  top_p: 0.9             # Nucleus sampling threshold
  top_k: 40              # Token selection limit
  repeat_penalty: 1.1    # Repetition penalty
  max_tokens: 512        # Max response length
```

**Parameter Guide:**
- **Temperature**: Lower = more focused, Higher = more creative
  - 0.1-0.3: Factual, deterministic
  - 0.7-0.9: Balanced
  - 1.0+: Creative, diverse
- **Top-p**: Keep at 0.9 for most uses
- **Top-k**: 40-100 works well for most cases
- **Repeat Penalty**: 1.1-1.2 reduces repetition

### Custom System Prompts

Add your own in `config.yaml`:

```yaml
system_prompts:
  my_custom_prompt: |
    You are a specialized assistant for...
    Your behavior should be...
```

Use it:
```bash
python3 chat.py --system-prompt my_custom_prompt
```

## Examples

### Example 1: Code Review Session

```bash
$ python3 chat.py --session code-review --system-prompt coding

You: Can you review this Python function?
def calc(x, y):
    return x + y
Assistant: This function works but could be improved:
1. The name 'calc' is not descriptive
2. No type hints
3. No docstring

Here's an improved version:

def add_numbers(x: float, y: float) -> float:
    """Add two numbers and return the result."""
    return x + y
```

### Example 2: Creative Writing

```bash
$ python3 chat.py --session story --system-prompt creative

You: Help me write a story about a robot learning to paint
```

## Troubleshooting

### llama.cpp not found

If you get an error about llama.cpp not being found:

1. Check the `llama_cpp_path` in your config file
2. Make sure it points to the correct location
3. Verify llama-cli is executable:
   ```bash
   chmod +x ./llama-cli
   ```

### Model file not found

Update the model path in `~/.llama_chat/config.yaml`:

```yaml
model:
  path: /full/path/to/your/model.gguf
```

### Out of memory

If you run out of memory:

1. Reduce `context_size` in config
2. Reduce `max_tokens` for shorter responses
3. Use a smaller quantized model (Q4_K_M instead of Q6_K)
4. Reduce `threads` if using too many

### Slow responses

To speed up generation:

1. Increase `threads` to match your CPU cores
2. Enable GPU acceleration with `gpu_layers`
3. Use a smaller model
4. Reduce `context_size`

## Extension Ideas

This project can be extended in many ways:

### Easy Extensions

1. **Colorized output** - Use `rich` or `colorama` for better formatting
2. **Streaming responses** - Show tokens as they're generated
3. **Custom prompts** - Add more system prompts to the config
4. **Session search** - Search through saved sessions
5. **Export sessions** - Export to text or markdown

### Intermediate Extensions

6. **Chat templates** - Support different chat templates (ChatML, Alpaca, etc.)
7. **RAG integration** - Add document retrieval for context
8. **Web interface** - Build a simple web UI with Flask
9. **Multi-model support** - Switch between models mid-session
10. **Conversation branches** - Branch conversations at any point

### Advanced Extensions

11. **Function calling** - Integrate with external tools
12. **Voice I/O** - Add speech-to-text and text-to-speech
13. **Image support** - For multimodal models
14. **Collaborative chat** - Multiple users in one session
15. **Cloud sync** - Sync sessions across devices

## Project Architecture

### Class Diagram

```
ChatHistory
├── save_session()
├── load_session()
├── add_message()
├── get_context()
└── clear_history()

Config
├── load_config()
├── save_config()
├── get_system_prompt()
└── _merge_configs()

LlamaChatClient
├── generate_response()
└── (wraps llama.cpp)

main()
└── (orchestrates above components)
```

### Data Flow

```
User Input
    ↓
Command Parsing (/quit, /save, etc.)
    ↓
Add to History
    ↓
Build Context from History
    ↓
Generate Response (llama.cpp)
    ↓
Add Response to History
    ↓
Display to User
    ↓
Auto-save on Exit
```

## Best Practices Demonstrated

This project showcases several best practices:

### 1. Configuration Management
- Centralized configuration
- Default values with override capability
- YAML for human-readable config

### 2. Data Persistence
- Automatic session saving
- JSON for structured data storage
- Timestamped sessions

### 3. Error Handling
- Graceful error recovery
- Informative error messages
- Continue on non-fatal errors

### 4. User Experience
- Clear command structure
- Helpful prompts and feedback
- Colored output for readability

### 5. Code Organization
- Separation of concerns (history, config, client)
- Clear class responsibilities
- Reusable components

### 6. Documentation
- Comprehensive help text
- Clear examples
- Inline code documentation

## Session File Format

Sessions are saved as JSON files in `~/.llama_chat/history/`:

```json
{
  "session_file": "/home/user/.llama_chat/history/my-chat.json",
  "created_at": "2024-01-15T10:00:00",
  "updated_at": "2024-01-15T10:30:00",
  "message_count": 5,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant.",
      "timestamp": "2024-01-15T10:00:00"
    },
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2024-01-15T10:01:00"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "timestamp": "2024-01-15T10:01:05"
    }
  ]
}
```

## Advanced Usage

### Environment Variables

You can override config with environment variables:

```bash
export LLAMA_CPP_PATH=/usr/local/bin/llama-cli
export MODEL_PATH=/models/my-model.gguf
python3 chat.py
```

### Batch Mode

For scripting, you can pipe input:

```bash
echo "What is Python?" | python3 chat.py --no-history
```

### Custom Config for Different Tasks

Create task-specific configs:

```bash
# For coding tasks
python3 chat.py --config coding-config.yaml

# For creative writing
python3 chat.py --config creative-config.yaml
```

## Performance Tips

### CPU Optimization

```yaml
model:
  threads: 8  # Set to your CPU core count
  context_size: 2048  # Smaller = faster
```

### GPU Acceleration

If you have an NVIDIA GPU with CUDA:

```yaml
model:
  gpu_layers: 33  # Offload layers to GPU
  # For 7B models: 33 layers typical
  # For 13B models: 40 layers typical
```

### Memory Management

```yaml
model:
  context_size: 1024  # Reduce for less memory usage

generation:
  max_tokens: 256  # Shorter responses = less memory
```

## Learning Outcomes

By working with this project, you'll learn:

1. **CLI Application Design**
   - Argument parsing with argparse
   - User interaction patterns
   - Command structures

2. **Data Persistence**
   - JSON serialization
   - File I/O
   - Session management

3. **Configuration Management**
   - YAML configuration
   - Default values
   - Config merging

4. **Process Integration**
   - Subprocess management
   - Command building
   - Output capturing

5. **Error Handling**
   - Exception handling
   - User feedback
   - Graceful degradation

6. **LLM Integration Basics**
   - Context management
   - Prompt engineering
   - Parameter tuning

## FAQ

**Q: Can I use this with other llama.cpp compatible models?**
A: Yes! Any GGUF model supported by llama.cpp will work.

**Q: How do I clear a session?**
A: Use `/clear` during chat or delete the session file from `~/.llama_chat/history/`.

**Q: Can I export conversations?**
A: Yes, session files are JSON and can be processed with any JSON tool.

**Q: Does this work on Windows?**
A: Yes, but you may need to adjust paths and use `python` instead of `python3`.

**Q: Can I use multiple models?**
A: Use the `--model` flag to switch between models without changing config.

**Q: How do I make responses faster?**
A: Increase threads, enable GPU layers, use smaller models, or reduce max_tokens.

## Contributing to the Project

Ideas for improvements:

1. Add streaming output
2. Implement conversation search
3. Add export to different formats
4. Create a web interface
5. Add prompt templates library
6. Implement conversation branching
7. Add statistics tracking
8. Create automated tests

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Python argparse](https://docs.python.org/3/library/argparse.html)
- [YAML in Python](https://pyyaml.org/wiki/PyYAMLDocumentation)

## Credits

This project is part of the llama.cpp learning materials, designed to teach practical LLM integration through hands-on coding.
