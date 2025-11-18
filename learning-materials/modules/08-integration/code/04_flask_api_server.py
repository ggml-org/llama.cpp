#!/usr/bin/env python3
"""
Flask API Server for llama.cpp

OpenAI-compatible API endpoints:
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/embeddings
- GET /v1/models
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from llama_cpp import Llama
import json
import time
from typing import Dict, List
import os


app = Flask(__name__)

# Global model instance
llm = None
config = {}


def initialize_model(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 35):
    """Initialize the model."""
    global llm, config

    print(f"Loading model from {model_path}...")

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format="llama-2",
        verbose=False
    )

    config = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers
    }

    print("Model loaded successfully!")


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatible)."""
    model_name = os.path.basename(config.get('model_path', 'unknown'))

    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama.cpp",
                "permission": [],
                "root": model_name,
                "parent": None
            }
        ]
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.

    Request body:
    {
        "model": "model-name",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": false
    }
    """
    data = request.json

    # Extract parameters
    messages = data.get('messages', [])
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 512)
    top_p = data.get('top_p', 0.95)
    stream = data.get('stream', False)

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        if stream:
            # Streaming response
            def generate():
                response_stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True
                )

                for chunk in response_stream:
                    chunk_data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": config.get('model_path', 'unknown'),
                        "choices": [chunk['choices'][0]]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream'
            )
        else:
            # Non-streaming response
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Format response
            formatted_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.get('model_path', 'unknown'),
                "choices": response['choices'],
                "usage": response.get('usage', {})
            }

            return jsonify(formatted_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/completions', methods=['POST'])
def completions():
    """
    OpenAI-compatible completions endpoint.

    Request body:
    {
        "model": "model-name",
        "prompt": "Hello, how are you?",
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": false
    }
    """
    data = request.json

    # Extract parameters
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 512)
    top_p = data.get('top_p', 0.95)
    stream = data.get('stream', False)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        if stream:
            # Streaming response
            def generate():
                response_stream = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True
                )

                for output in response_stream:
                    chunk_data = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": config.get('model_path', 'unknown'),
                        "choices": output['choices']
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream'
            )
        else:
            # Non-streaming response
            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Format response
            formatted_response = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": config.get('model_path', 'unknown'),
                "choices": response['choices'],
                "usage": response.get('usage', {})
            }

            return jsonify(formatted_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """
    OpenAI-compatible embeddings endpoint.

    Request body:
    {
        "model": "model-name",
        "input": "Text to embed"
    }
    """
    data = request.json

    # Extract parameters
    input_text = data.get('input', '')

    if not input_text:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Generate embedding
        result = llm.create_embedding(input_text)

        # Format response
        formatted_response = {
            "object": "list",
            "data": result['data'],
            "model": config.get('model_path', 'unknown'),
            "usage": result.get('usage', {})
        }

        return jsonify(formatted_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": llm is not None
    })


@app.route('/', methods=['GET'])
def index():
    """API information."""
    return jsonify({
        "name": "llama.cpp API Server",
        "version": "1.0.0",
        "endpoints": {
            "/v1/models": "List available models (GET)",
            "/v1/chat/completions": "Chat completions (POST)",
            "/v1/completions": "Text completions (POST)",
            "/v1/embeddings": "Generate embeddings (POST)",
            "/health": "Health check (GET)"
        }
    })


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='llama.cpp Flask API Server')
    parser.add_argument('model_path', help='Path to GGUF model file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--n-ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--n-gpu-layers', type=int, default=35, help='GPU layers')

    args = parser.parse_args()

    # Initialize model
    initialize_model(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers
    )

    # Run server
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/")
    print("\nExample usage:")
    print(f"  curl http://{args.host}:{args.port}/v1/models")
    print(f"  curl http://{args.host}:{args.port}/v1/chat/completions \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
    print()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
