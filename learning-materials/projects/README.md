# Module 1 - Starter Projects

Production-ready projects for learning llama.cpp fundamentals. These projects demonstrate best practices and provide hands-on experience with LLM integration.

## Projects Overview

### 1. Simple CLI Chat

A complete command-line chat application for interacting with llama.cpp models.

**Location:** `simple-cli-chat/`

**Key Features:**
- Multi-turn conversation support
- Persistent conversation history
- Multiple system prompt modes (coding, creative, technical, tutor)
- Session management (create, save, resume named sessions)
- YAML-based configuration
- Production-quality error handling
- Clean, modular architecture

**Use Cases:**
- Interactive model testing
- Learning prompt engineering
- Building custom chat applications
- Understanding context management

**Technologies:**
- Python 3.8+
- YAML configuration
- JSON session storage
- subprocess integration with llama.cpp

**What You'll Learn:**
- CLI application design
- Configuration management
- Data persistence
- Context window management
- LLM parameter tuning
- Process integration

---

### 2. Model Info Tool

A comprehensive CLI tool for inspecting and analyzing GGUF model files.

**Location:** `model-info-tool/`

**Key Features:**
- Complete GGUF file parsing
- Detailed model analysis (architecture, quantization, size)
- Model comparison tool
- Multiple export formats (JSON, Markdown, CSV)
- Memory requirement estimation
- Zero external dependencies (pure Python stdlib)
- Well-structured Python package

**Use Cases:**
- Model selection and comparison
- Hardware planning (RAM/VRAM requirements)
- Model cataloging and documentation
- Learning about model internals
- Pre-download verification

**Technologies:**
- Python 3.8+
- Binary file parsing
- Structured data export
- Package development (setuptools)

**What You'll Learn:**
- Binary file format parsing
- Model architecture understanding
- Quantization types and trade-offs
- Python package structure
- CLI design patterns
- Data serialization

---

## Getting Started

### Prerequisites

Both projects require:
- Python 3.8 or higher
- llama.cpp compiled (for simple-cli-chat)
- GGUF model files

### Quick Start

#### Simple CLI Chat

```bash
cd simple-cli-chat/

# Install dependencies
pip install -r requirements.txt

# Configure (update paths)
cp config.yaml ~/.llama_chat/config.yaml
nano ~/.llama_chat/config.yaml

# Run
python3 chat.py

# Or with options
python3 chat.py --session my-chat --system-prompt coding
```

#### Model Info Tool

```bash
cd model-info-tool/

# Option 1: Direct usage (no installation)
python3 -m model_info_tool.cli info /path/to/model.gguf

# Option 2: Install as package
pip install -e .
model-info info /path/to/model.gguf

# Compare models
model-info compare model1.gguf model2.gguf

# Export information
model-info export model.gguf -f json -o info.json
```

## Project Structure

### Simple CLI Chat

```
simple-cli-chat/
├── chat.py              # Main application (480+ lines)
├── config.yaml          # Configuration template
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables
├── README.md           # Comprehensive documentation
└── EXAMPLES.md         # Usage examples

Runtime files (auto-created):
~/.llama_chat/
├── config.yaml         # User configuration
└── history/            # Saved sessions
    └── *.json          # Session files
```

### Model Info Tool

```
model-info-tool/
├── model_info_tool/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # CLI interface (370+ lines)
│   ├── gguf_reader.py       # GGUF parser (190+ lines)
│   ├── model_analyzer.py    # Analysis logic (280+ lines)
│   └── exporter.py          # Export functionality (230+ lines)
├── model-info               # Entry point script
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies (none required!)
├── README.md               # Comprehensive documentation
└── EXAMPLES.md             # Usage examples
```

## Learning Path

### Beginner Path

1. **Start with Model Info Tool**
   - Inspect models you have
   - Understand quantization types
   - Compare different models
   - Learn about memory requirements

2. **Move to Simple CLI Chat**
   - Set up configuration
   - Run basic chat sessions
   - Experiment with system prompts
   - Try different models

### Intermediate Path

1. **Customize Simple CLI Chat**
   - Add custom system prompts
   - Modify temperature/parameters
   - Create task-specific configs
   - Build chat workflows

2. **Extend Model Info Tool**
   - Add new export formats
   - Create analysis scripts
   - Build model databases
   - Integrate with other tools

### Advanced Path

1. **Enhance Simple CLI Chat**
   - Add streaming responses
   - Implement RAG integration
   - Create web interface
   - Add function calling

2. **Advanced Model Info Tool**
   - Add performance benchmarking
   - Create visualization tools
   - Build web dashboard
   - Add cloud storage support

## Extension Ideas

### Simple CLI Chat Extensions

**Easy:**
- Colorized output with `rich`
- More system prompts
- Session search functionality
- Export to markdown

**Intermediate:**
- Streaming token generation
- Chat template support
- RAG with document retrieval
- Multi-model switching

**Advanced:**
- Web UI with Flask/FastAPI
- Voice input/output
- Multimodal support
- Collaborative sessions

### Model Info Tool Extensions

**Easy:**
- More export formats
- Batch processing scripts
- Model recommendations
- Better visualizations

**Intermediate:**
- GUI with tkinter
- Performance benchmarking
- Model testing integration
- Database backend

**Advanced:**
- Web service API
- Cloud integration
- Real-time monitoring
- Format conversion tools

## Best Practices Demonstrated

### Code Organization
- Modular design with clear separation of concerns
- Reusable components
- Well-defined class responsibilities

### Error Handling
- Comprehensive exception handling
- Informative error messages
- Graceful degradation

### Documentation
- Comprehensive README files
- Detailed code comments
- Type hints throughout
- Usage examples

### User Experience
- Clear command structures
- Helpful feedback
- Multiple output formats
- Sensible defaults

### Configuration
- YAML for human-readable config
- Default values with overrides
- Environment variable support

### Testing
- Edge case handling
- Input validation
- File existence checks

## Documentation

Each project includes:

- **README.md**: Comprehensive guide with installation, usage, configuration, and troubleshooting
- **EXAMPLES.md**: Practical examples and common use cases
- **Inline comments**: Well-documented code
- **Type hints**: For better IDE support

## Development Tips

### For Simple CLI Chat

1. Start with the default config
2. Test with small models first
3. Use named sessions for organization
4. Experiment with different system prompts
5. Monitor memory usage with large contexts

### For Model Info Tool

1. Test with various GGUF models
2. Use JSON export for scripting
3. Compare before downloading large models
4. Build a model catalog
5. Use memory estimates for hardware planning

## Common Use Cases

### Use Case 1: Model Selection

```bash
# Check what you have
model-info compare models/*.gguf

# Test the best candidate
python3 chat.py --model models/best-model.gguf

# If it works well, make it default in config
```

### Use Case 2: Learning Session

```bash
# Start learning session with tutor prompt
python3 chat.py --session learning --system-prompt tutor

# Ask questions, get explanations
# Session auto-saves on exit
```

### Use Case 3: Code Development

```bash
# Use coding assistant
python3 chat.py --session dev --system-prompt coding

# Get code reviews, suggestions, examples
# Resume later with same context
```

### Use Case 4: Model Documentation

```bash
# Document all your models
for model in models/*.gguf; do
  name=$(basename "$model" .gguf)
  model-info export "$model" -f markdown -o "docs/${name}.md"
done
```

## Troubleshooting

### Simple CLI Chat

**Issue:** llama.cpp not found
- Update `llama_cpp_path` in config
- Use absolute path

**Issue:** Out of memory
- Reduce `context_size`
- Use smaller model (Q4 instead of Q6)
- Reduce `max_tokens`

**Issue:** Slow responses
- Increase `threads`
- Enable GPU with `gpu_layers`
- Use smaller model

### Model Info Tool

**Issue:** File not found
- Use absolute paths
- Check file exists with `ls`

**Issue:** Invalid GGUF file
- Re-download model
- Check file integrity
- Verify GGUF format

**Issue:** Missing metadata
- Normal for some models
- Tool shows available data only

## Resources

### llama.cpp
- [Main Repository](https://github.com/ggerganov/llama.cpp)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Model Quantization](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

### Python
- [argparse Tutorial](https://docs.python.org/3/howto/argparse.html)
- [YAML in Python](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [JSON Module](https://docs.python.org/3/library/json.html)

### Learning
- Module 1: Fundamentals (theory)
- These projects (practice)
- Code examples (reference)

## Contributing

Ideas for contributions:

1. Add unit tests
2. Improve documentation
3. Add more examples
4. Create video tutorials
5. Build integration guides
6. Add internationalization
7. Create Docker containers
8. Add CI/CD pipelines

## Next Steps

After completing these projects:

1. **Module 2**: Inference optimization
   - Performance tuning
   - Hardware acceleration
   - Batch processing

2. **Module 3**: Advanced features
   - Custom sampling
   - Fine-tuning integration
   - Production deployment

3. **Build Your Own**
   - Combine both projects
   - Add custom features
   - Create specialized tools

## Support

For questions or issues:

1. Check README files
2. Review EXAMPLES.md
3. Read inline code comments
4. Consult llama.cpp documentation
5. Check Module 1 learning materials

## License

These projects are educational resources. Use freely for learning and experimentation.

---

**Happy Learning!**

Build, experiment, and extend these projects to deepen your understanding of llama.cpp and LLM integration.
