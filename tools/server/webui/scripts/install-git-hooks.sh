#!/bin/bash

# Script to install pre-commit hook for webui
# Pre-commit: formats, checks, builds, and stages build output

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"

echo "Installing pre-commit hook for webui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Check if there are any changes in the webui directory
if git diff --cached --name-only | grep -q "^tools/server/webui/"; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    cd "$REPO_ROOT/tools/server/webui"

    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in tools/server/webui"
        exit 1
    fi

    echo "Formatting and checking webui code..."

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

    echo "✅ Webui code formatted and checked successfully"

    # Build the webui
    echo "Building webui..."
    npm run build
    if [ $? -ne 0 ]; then
        echo "❌ npm run build failed"
        exit 1
    fi

    # Stage the build output alongside the source changes
    cd "$REPO_ROOT"
    git add tools/server/public/

    echo "✅ Webui built and build output staged"
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
    echo "  • Format, lint and check webui code before commits"
    echo "  • Build webui and stage tools/server/public/ into the same commit"
else
    echo "❌ Failed to make hook executable"
    exit 1
fi
