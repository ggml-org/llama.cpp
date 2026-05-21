#!/bin/bash

# Development script for llama-ui
#
# This script starts the llama-ui development servers (Storybook and Vite).
# Note: You need to start llama-server separately.
#
# Usage:
#   bash scripts/dev.sh
#   npm run dev

cd ../../

# Ensure node_modules are installed
if [ ! -d "tools/ui/node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    cd tools/ui && npm install && cd ../../
fi

# Check and install git hooks if missing
check_and_install_hooks() {
    local hooks_missing=false

    # Check for required hooks
    if [ ! -f ".git/hooks/pre-commit" ] || [ ! -f ".git/hooks/pre-push" ]; then
        hooks_missing=true
    fi

    if [ "$hooks_missing" = true ]; then
        echo "🔧 Git hooks missing, installing them..."
        if bash "$(dirname "$0")/git-hooks/install.sh"; then
            echo "✅ Git hooks installed successfully"
        else
            echo "⚠️  Failed to install git hooks, continuing anyway..."
        fi
    else
        echo "✅ Git hooks already installed"
    fi
}

# Install git hooks if needed
check_and_install_hooks

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    exit
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🚀 Starting development servers..."
echo "📝 Note: Make sure to start llama-server separately if needed"
cd tools/ui
# Use --insecure-http-parser to handle malformed HTTP responses from llama-server
# (some responses have both Content-Length and Transfer-Encoding headers)
storybook dev -p 6006 --ci & NODE_OPTIONS="--insecure-http-parser" vite dev --host 0.0.0.0 &

# Wait for all background processes
wait
