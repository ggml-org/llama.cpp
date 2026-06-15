# Guide: Using the WebUI of llama.cpp

The new SvelteKit-based WebUI (`llama-ui`) provides a full-featured browser interface for llama-server. This guide covers getting started and exploring its capabilities.

## Getting started

### 1. Build or install llama.cpp

See the [build guide](../build.md) or [install guide](../install.md).

### 2. Start llama-server

```sh
# Basic text-only chat
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --jinja -c 0

# With HTML/JS preview enabled
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --jinja -c 0 --preview-js

# Vision-enabled model
llama-server -hf ggml-org/Qwen3-VL-30B-A3B-Instruct-Q8_0-GGUF --jinja -c 0 --mmproj <mmproj-file>

# Hybrid SSM model with 1M context
llama-server -hf ggml-org/granite-4.0-h-small-Q8_0-GGUF --jinja -c 0
```

### 3. Open the browser

Navigate to `http://127.0.0.1:8080`. The WebUI is served automatically by llama-server.

For a GUI-based setup on Mac, try the [LlamaBarn](https://github.com/ggml-org/llama.cpp/discussions/16938) application.

## Key features

### Text document processing

Add multiple text files from disk or from the clipboard as context for your conversation.

### PDF document processing

Attach PDFs to your conversation. By default, contents are extracted as raw text. For vision-enabled models, PDFs can also be processed as images.

### Image inputs

When the model supports vision (multimodal), insert images directly into your conversation alongside text.

### Conversation branching

Edit or regenerate any previous message to branch from earlier points in the conversation.

### Parallel conversations

Run multiple independent chat conversations simultaneously. Parallel image processing is also supported.

### Default sampling parameters

Set default parameters via command line:

```sh
llama-server --top-k 5 --temp 0.80 --alias "my-model" -m model.gguf
```

The WebUI settings panel will reflect these defaults.

### Math rendering

Mathematical expressions (LaTeX) are rendered inline in chat messages.

### Input via URL parameters

Pass input through URL query parameters:

```
http://127.0.0.1:8080/?prompt=Tell%20me%20a%20joke
```

### HTML/JS preview

Enable with `--preview-js`:

```sh
llama-server -m model.gguf --preview-js
```

Generated HTML/JavaScript code is rendered inline in an iframe.

### Constrained generation / JSON schema

Specify a JSON schema to constrain the model output to a specific format. Use the WebUI's schema editor to define the output structure.

### Import/Export conversations

Manage private conversations through the WebUI's Import/Export options.

### Efficient SSM context management

SSM models (e.g., Mamba, hybrid architectures) benefit from automatic context management and prefix caching with minimal reprocessing.

### Mobile compatibility

The WebUI is responsive and works on mobile browsers.

## Sample configurations

```sh
# Lightweight, text-only, greedy sampling
llama-server --jinja -c 0 --port 8033 \
  -hf ggml-org/gpt-oss-20b-GGUF \
  --alias "gpt-oss-20b" --top-k 1

# Accessible from local network
llama-server --jinja -c 0 --port 8033 \
  -hf ggml-org/Qwen3-VL-30B-A3B-Instruct-Q8_0-GGUF \
  --alias "Qwen3 VL 30B" --host 192.168.100.3

# Large context, hybrid model
llama-server --jinja -c 0 --port 8033 \
  -hf ggml-org/granite-4.0-h-small-Q8_0-GGUF \
  --alias "Granite 4.0 Hybrid"
```
