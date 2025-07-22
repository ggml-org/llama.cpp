#!/bin/bash

# llama.cpp WebUI Development Script
# Usage: ./scripts/webui-dev.sh -hf <model> [-b|--build] [-p|--port <port>]

set -e

# Default values
MODEL=""
PORT="8080"
BUILD=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to display usage
usage() {
    echo "Usage: $0 (-hf|-m) <model> [-b|--build] [-p|--port <port>]"
    echo ""
    echo "Options:"
    echo "  -b, --build        Build llama-server before running"
    echo "  -h, --help         Show this help message"
    echo "  -hf <model>        Hugging Face model to use (required)"
    echo "  -m <model>         Path to local model file (required)"
    echo "  -p, --port <port>  Port to run the server on (default: 8080)"
    echo ""
    echo "Note: -hf and -m are interchangeable and can accept either HF models or local paths"
    echo ""
    echo "Examples:"
    echo "  $0 -hf ggml-org/SmolLM3-3B-GGUF"
    echo "  $0 -m /path/to/model.gguf"
    echo "  $0 -hf ggml-org/SmolLM3-3B-GGUF -b -p 8081"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -hf)
            MODEL="$2"
            shift 2
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -m)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: Model (-hf or -m) is required"
    usage
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port must be a number"
    exit 1
fi

echo "🚀 Starting llama.cpp WebUI Development Environment"
echo "📁 Project root: $PROJECT_ROOT"
echo "🤖 Model: $MODEL"
echo "🌐 Port: $PORT"
echo "🔨 Build: $BUILD"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Build if requested
if [[ "$BUILD" == true ]]; then
    echo "🔨 Building llama-server..."
    cmake -B build
    cmake --build build --config Release -t llama-server
    echo "✅ Build completed"
    echo ""
fi

# Check if binary exists
if [[ ! -f "./build/bin/llama-server" ]]; then
    echo "❌ Error: llama-server binary not found at ./build/bin/llama-server"
    echo "💡 Try running with the -b flag to build first"
    exit 1
fi

echo "🎯 Starting llama-server..."
echo "📡 Server will be available at: http://localhost:$PORT"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the server
# Check if model looks like a HuggingFace model or local path
if [[ "$MODEL" == *"/"* && ! "$MODEL" == *"ggml-org/"* && ! "$MODEL" == *"microsoft/"* && ! "$MODEL" == *"meta-llama/"* ]]; then
    # Looks like a local path
    ./build/bin/llama-server -m "$MODEL" --port "$PORT"
else
    # Treat as HuggingFace model
    ./build/bin/llama-server -hf "$MODEL" --port "$PORT"
fi