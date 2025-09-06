#!/bin/bash

# Script to install pre-commit hook for webui formatting
# Run this from the webui directory: ./install-git-hook.sh

# Get the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Path to the hooks directory
HOOKS_DIR="$GIT_ROOT/.git/hooks"
HOOK_FILE="$HOOKS_DIR/pre-commit"

# Create the pre-commit hook content
cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash

# Check if there are any changes in the webui directory
if git diff --cached --name-only | grep -q "^tools/server/webui/"; then
    echo "Formatting webui code..."
    
    # Change to webui directory and run format
    cd tools/server/webui
    
    # Check if npm is available and package.json exists
    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in tools/server/webui"
        exit 1
    fi
    
    # Run the format command
    npm run format

    # Check if format command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run format failed"
        exit 1
    fi

    # Run the check command
    npm run check
    
    # Check if check command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run check failed"
        exit 1
    fi

    # Run the build command
    npm run build
    
    # Check if build command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run build failed"
        exit 1
    fi
    
    # Go back to repo root
    cd ../../..
    
    # Add any files that were formatted back to the staging area
    git add tools/server/webui/
    
    echo "Webui code formatted successfully"
fi

exit 0
EOF

# Make the hook executable
chmod +x "$HOOK_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Pre-commit hook installed successfully at: $HOOK_FILE"
    echo ""
    echo "The hook will automatically format webui code when you commit changes to files in tools/server/webui/"
    echo ""
    echo "To test the hook, make a change to a file in the webui directory and commit it."
else
    echo "❌ Failed to make hook executable"
    exit 1
fi
