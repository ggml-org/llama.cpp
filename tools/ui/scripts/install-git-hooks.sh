#!/bin/bash

# Script to install pre-commit hook for llama-ui
# Pre-commit: formats, checks, and builds the UI app

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"

echo "Installing pre-commit hook for llama-ui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Check if there are any changes in the tools/ui directory
if git diff --cached --name-only | grep -q "^tools/ui/"; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    cd "$REPO_ROOT/tools/ui"

    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in tools/ui"
        exit 1
    fi

    echo "Formatting and checking llama-ui code..."

    # Run the format command
    npm run format
    if [ $? -ne 0 ]; then
        echo "Error: npm run format failed"
        exit 1
    fi

    # Run the lint command
    npm run lint
    if [ $? -ne 0 ]; then
        echo "Error: npm run lint failed"
        exit 1
    fi

    # Run the check command
    npm run check
    if [ $? -ne 0 ]; then
        echo "Error: npm run check failed"
        exit 1
    fi

    echo "✅ llama-ui code formatted and checked successfully"

    # Build the llama-ui
    echo "Building llama-ui..."
    npm run build
    if [ $? -ne 0 ]; then
        echo "❌ npm run build failed"
        exit 1
    fi

    echo "✅ llama-ui built successfully"
fi

exit 0
EOF

# Make hook executable
chmod +x "$PRE_COMMIT_HOOK"

if [ $? -eq 0 ]; then
    echo "✅ Git hook installed successfully!"
    echo "   Pre-commit: $PRE_COMMIT_HOOK"
    echo ""
    echo "The hook will automatically:"
    echo "  • Format, lint and check llama-ui code before commits"
    echo "  • Build llama-ui"
else
    echo "❌ Failed to make hook executable"
    exit 1
fi
