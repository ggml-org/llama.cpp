# Model Info Tool

A production-ready command-line tool for inspecting, analyzing, and comparing GGUF (GPT-Generated Unified Format) model files. This tool provides detailed information about model architecture, quantization, memory requirements, and more.

## Features

- **Comprehensive Model Inspection**: View detailed information about GGUF models
- **Model Comparison**: Compare multiple models side-by-side
- **Multiple Export Formats**: Export model information to JSON, Markdown, or CSV
- **Memory Estimation**: Estimate RAM/VRAM requirements
- **Architecture Analysis**: Detailed architecture and quantization information
- **Zero Dependencies**: Uses only Python standard library (optional dependencies for enhanced output)
- **Well-Structured Package**: Clean, modular architecture for easy extension

## Project Structure

```
model-info-tool/
├── model_info_tool/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── gguf_reader.py       # GGUF file parser
│   ├── model_analyzer.py    # Model analysis logic
│   └── exporter.py          # Export functionality
├── model-info               # Entry point script
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies (none required!)
└── README.md               # This file
```

## Installation

### Method 1: Direct Usage (No Installation)

```bash
cd model-info-tool
python3 -m model_info_tool.cli info /path/to/model.gguf
```

### Method 2: Install as Package

```bash
cd model-info-tool
pip install -e .

# Now you can use 'model-info' from anywhere
model-info info /path/to/model.gguf
```

### Method 3: Standalone Script

```bash
chmod +x model-info
./model-info info /path/to/model.gguf
```

### Prerequisites

- Python 3.8 or higher
- A GGUF model file to inspect

No external dependencies required!

## Usage

### Basic Model Information

Display basic model information:

```bash
model-info info model.gguf
```

Output:
```
============================================================
  MODEL INFORMATION
============================================================

Name: Llama 2 7B Chat
Architecture: llama
File: llama-2-7b-chat.Q4_K_M.gguf
Path: /models/llama-2-7b-chat.Q4_K_M.gguf
Size: 3.83 GB (3897.45 MB)
Parameters: 6.74B (6,738,415,616)
Tensors: 291
```

### Verbose Output

Show all available information:

```bash
model-info info model.gguf -v
```

or

```bash
model-info info model.gguf --verbose
```

### Summary View

Quick summary of the model:

```bash
model-info info model.gguf -s
```

or

```bash
model-info info model.gguf --summary
```

### Specific Information

Show only specific sections:

```bash
# Architecture details only
model-info info model.gguf --architecture

# Tokenizer information only
model-info info model.gguf --tokenizer

# Quantization information only
model-info info model.gguf --quantization

# Memory requirements only
model-info info model.gguf --memory

# Tensor details
model-info info model.gguf --tensors
```

### Compare Multiple Models

Compare 2 or more models:

```bash
model-info compare model1.gguf model2.gguf model3.gguf
```

Output:
```
============================================================
  MODEL COMPARISON
============================================================

Model                          Architecture    Parameters   Size (GB)  Quantization
----------------------------------------------------------------------------------
llama-2-7b-chat.Q4_K_M.gguf   llama           6.74B        3.83       Q4_K_M
llama-2-13b-chat.Q4_K_M.gguf  llama           13.02B       7.37       Q4_K_M
mistral-7b-v0.1.Q5_K_M.gguf   llama           7.24B        4.78       Q5_K_M

Memory Requirements Comparison
------------------------------------------------------------
Model                          CPU RAM (GB)    GPU VRAM (GB)
------------------------------------------------------------
llama-2-7b-chat.Q4_K_M.gguf   4.594          4.213
llama-2-13b-chat.Q4_K_M.gguf  8.844          8.107
mistral-7b-v0.1.Q5_K_M.gguf   5.736          5.258
```

### Export Model Information

#### Export to JSON

```bash
# Print to stdout
model-info export model.gguf -f json

# Save to file
model-info export model.gguf -f json -o model_info.json

# Include tensor details
model-info export model.gguf -f json --include-tensors -o model_full.json

# Custom indentation
model-info export model.gguf -f json -i 4 -o model_info.json
```

#### Export to Markdown

```bash
# Print to stdout
model-info export model.gguf -f markdown

# Save to file
model-info export model.gguf -f markdown -o model_info.md
```

#### Export to CSV

```bash
# Print to stdout
model-info export model.gguf -f csv

# Save to file
model-info export model.gguf -f csv -o model_info.csv
```

### List All Metadata

View all metadata keys and values:

```bash
model-info metadata model.gguf
```

## Python API

You can also use the tool as a Python library:

```python
from model_info_tool import GGUFReader, ModelAnalyzer, ModelExporter

# Read model
analyzer = ModelAnalyzer("model.gguf")

# Get basic information
basic_info = analyzer.get_basic_info()
print(f"Model: {basic_info['name']}")
print(f"Parameters: {basic_info['parameter_count_formatted']}")

# Get architecture details
arch = analyzer.get_architecture_details()
print(f"Layers: {arch.get('block_count')}")

# Get memory requirements
memory = analyzer.get_memory_requirements()
print(f"RAM needed: {memory['estimated_ram_cpu_only_gb']} GB")

# Generate summary
summary = analyzer.generate_summary()
print(summary)

# Export to JSON
exporter = ModelExporter(analyzer)
exporter.to_json("model_info.json", include_tensors=True)

# Compare models
comparison = analyzer.compare_with("another_model.gguf")
print(comparison)
```

## Output Examples

### JSON Export

```json
{
  "export_info": {
    "timestamp": "2024-01-15T10:30:00",
    "tool_version": "1.0.0"
  },
  "basic_info": {
    "name": "Llama 2 7B Chat",
    "architecture": "llama",
    "parameters": 6738415616,
    "parameter_count_formatted": "6.74B",
    "file_size_gb": 3.83,
    "tensor_count": 291
  },
  "architecture": {
    "embedding_length": 4096,
    "block_count": 32,
    "feed_forward_length": 11008,
    "attention.head_count": 32,
    "attention.head_count_kv": 32
  },
  "quantization": {
    "predominant_type": "Q4_K_M",
    "tensor_types": {
      "Q4_K": 225,
      "Q6_K": 64,
      "F32": 2
    }
  }
}
```

### Markdown Export

```markdown
# Llama 2 7B Chat

**Generated:** 2024-01-15 10:30:00

## Basic Information

- **File:** `llama-2-7b-chat.Q4_K_M.gguf`
- **Architecture:** llama
- **Parameters:** 6.74B (6,738,415,616)
- **File Size:** 3.83 GB (3897.45 MB)
- **Quantization:** Q4_K_M
- **Tensor Count:** 291

## Architecture Details

| Property | Value |
|----------|-------|
| embedding_length | 4096 |
| block_count | 32 |
| feed_forward_length | 11008 |
| attention.head_count | 32 |
| attention.head_count_kv | 32 |

## Memory Requirements (Estimated)

- **Model Size:** 3.83 GB
- **CPU-only RAM:** ~4.59 GB
- **GPU-full VRAM:** ~4.21 GB

> Estimates include model weights and minimal overhead. Actual usage depends on context size and batch size.
```

## Understanding the Output

### Model Size vs Memory Requirements

- **Model Size**: The size of the GGUF file on disk
- **CPU RAM**: Estimated memory when running entirely on CPU (includes ~20% overhead)
- **GPU VRAM**: Estimated memory when fully offloaded to GPU (includes ~10% overhead)

### Quantization Types

Common quantization formats you'll see:

- **F32/F16**: Full precision (32-bit) or half precision (16-bit) floating point
- **Q8_0**: 8-bit quantization
- **Q6_K**: 6-bit quantization (K-quants)
- **Q5_K_M**: 5-bit medium quality K-quants
- **Q4_K_M**: 4-bit medium quality K-quants (good balance)
- **Q4_K_S**: 4-bit small/fast K-quants
- **Q3_K_M**: 3-bit medium quality K-quants
- **Q2_K**: 2-bit quantization (smallest)

Lower bits = smaller file size but potentially lower quality. K-quants generally provide better quality than legacy quants at the same bit level.

## Use Cases

### 1. Model Selection

Compare different quantization levels of the same model to choose the right balance between size and quality:

```bash
model-info compare   llama-2-7b-chat.Q2_K.gguf   llama-2-7b-chat.Q4_K_M.gguf   llama-2-7b-chat.Q6_K.gguf
```

### 2. Hardware Planning

Check if a model will fit in your available RAM/VRAM:

```bash
model-info info model.gguf --memory
```

### 3. Model Inventory

Export information about all your models to JSON for cataloging:

```bash
for model in models/*.gguf; do
  model-info export "$model" -f json -o "info/$(basename "$model" .gguf).json"
done
```

### 4. Documentation

Generate Markdown documentation for your models:

```bash
model-info export model.gguf -f markdown -o MODEL_INFO.md
```

### 5. Quick Model Check

Verify model file integrity and get basic stats:

```bash
model-info info model.gguf --summary
```

## Extension Ideas

This project can be extended in several ways:

1. **Graphical Interface**: Add a GUI using tkinter or PyQt
2. **Batch Processing**: Add batch export for multiple models
3. **Model Testing**: Integrate performance benchmarking
4. **Web Interface**: Create a Flask/FastAPI web service
5. **Cloud Integration**: Add support for inspecting models from cloud storage
6. **Visualization**: Create charts showing quantization distribution
7. **Model Recommendations**: Suggest optimal quantization based on hardware
8. **Format Conversion**: Add ability to export metadata for other tools

## Best Practices Demonstrated

This project demonstrates several best practices:

### Code Organization
- Modular design with separation of concerns
- Clear class and function responsibilities
- Well-defined interfaces between components

### Error Handling
- Comprehensive exception handling
- Informative error messages
- Graceful degradation on missing data

### Documentation
- Comprehensive docstrings
- Type hints throughout
- Clear README with examples

### CLI Design
- Intuitive command structure
- Helpful error messages
- Comprehensive help text
- Multiple output formats

### Extensibility
- Easy to add new export formats
- Pluggable analyzer components
- Clean separation of reading and analysis logic

## Troubleshooting

### "File not found" Error

Make sure the path to your GGUF file is correct:

```bash
# Use absolute path
model-info info /full/path/to/model.gguf

# Or relative path from current directory
model-info info ./models/model.gguf
```

### "Invalid GGUF file" Error

The file might be:
- Corrupted during download
- Not a GGUF file (wrong format)
- Incompatible GGUF version

Try downloading the model again or check the file format.

### Missing Metadata

Some older or converted models may not have all metadata fields. The tool will display available information and skip missing fields.

### Permission Error

Make sure you have read permissions for the model file:

```bash
chmod +r model.gguf
```

## Technical Details

### GGUF Format

GGUF (GPT-Generated Unified Format) is a binary format for storing Large Language Model (LLM) weights and metadata. It includes:

- **Header**: Magic number, version, tensor count, metadata count
- **Metadata**: Key-value pairs describing the model
- **Tensor Info**: Names, dimensions, types, and offsets for all tensors
- **Tensor Data**: The actual weight data

### Reading Process

1. Read and validate header (magic number, version)
2. Parse metadata key-value pairs
3. Read tensor information (without loading actual data)
4. Analyze and compute derived information

The tool only reads headers and metadata, not the actual tensor data, making it very fast even for large models.

## Contributing

Ideas for contributions:

1. Add support for other model formats
2. Improve memory estimation accuracy
3. Add performance benchmarking integration
4. Create visualization tools
5. Add more export formats
6. Improve documentation
7. Add unit tests
8. Add support for streaming large files

## Learning Resources

This project teaches:

- Binary file format parsing
- Working with model metadata
- CLI application design
- Python package structure
- Data serialization (JSON, CSV, Markdown)
- Error handling and validation

## License

This is an educational project. Use freely for learning and experimentation.

## Acknowledgments

- Based on the GGUF format specification
- Part of the llama.cpp learning materials
- Designed for Module 1: Fundamentals
