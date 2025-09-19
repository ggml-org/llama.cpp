#!/bin/bash

cd ../../../

# Check and install git hooks if missing
check_and_install_hooks() {
    local hooks_missing=false
    
    # Check for required hooks
    if [ ! -f ".git/hooks/pre-commit" ] || [ ! -f ".git/hooks/pre-push" ] || [ ! -f ".git/hooks/post-push" ]; then
        hooks_missing=true
    fi
    
    if [ "$hooks_missing" = true ]; then
        echo "🔧 Git hooks missing, installing them..."
        cd tools/server/webui
        if bash scripts/install-git-hooks.sh; then
            echo "✅ Git hooks installed successfully"
        else
            echo "⚠️  Failed to install git hooks, continuing anyway..."
        fi
        cd ../../../
    else
        echo "✅ Git hooks already installed"
    fi
}

# Install git hooks if needed
check_and_install_hooks

# Check if llama-server binary already exists
if [ ! -f "build/bin/llama-server" ]; then
    echo "Building llama-server..."
    cmake -B build && cmake --build build --config Release -t llama-server
else
    echo "llama-server binary already exists, skipping build."
fi

# Start llama-server and capture output
echo "Starting llama-server..."
mkfifo server_output.pipe
build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF --jinja -c 0 --no-webui > server_output.pipe 2>&1 &
SERVER_PID=$!

# Function to wait for server to be ready
wait_for_server() {
    echo "Waiting for llama-server to be ready..."
    local max_wait=60
    local start_time=$(date +%s)
    
    # Read server output in background and look for the ready message
    (
        while IFS= read -r line; do
            echo "🔍 Server: $line"
            if [[ "$line" == *"server is listening on http://127.0.0.1:8080 - starting the main loop"* ]]; then
                echo "✅ llama-server is ready!"
                echo "READY" > server_ready.flag
                break
            fi
        done < server_output.pipe
    ) &
    
    # Wait for ready flag or timeout
    while [ ! -f server_ready.flag ]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $max_wait ]; then
            echo "❌ Server failed to start within $max_wait seconds"
            rm -f server_ready.flag
            return 1
        fi
        
        sleep 1
    done
    
    rm -f server_ready.flag
    return 0
}

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    kill $SERVER_PID 2>/dev/null
    rm -f server_output.pipe server_ready.flag
    exit
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for server to be ready
if wait_for_server; then
    echo "🚀 Starting development servers..."
    cd tools/server/webui
    storybook dev -p 6006 --ci & vite dev --host 0.0.0.0 &
    
    # Wait for all background processes
    wait
else
    echo "❌ Failed to start development environment"
    cleanup
fi
