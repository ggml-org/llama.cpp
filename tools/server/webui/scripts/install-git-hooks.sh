#!/bin/bash

# Script to install pre-commit and post-commit hooks for webui
# Pre-commit: formats, lints, checks, and builds code, stashes unstaged changes
# Post-commit: automatically unstashes changes

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"
POST_COMMIT_HOOK="$REPO_ROOT/.git/hooks/post-commit"

echo "Installing pre-commit and post-commit hooks for webui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
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
    
    # Stash any unstaged changes to avoid conflicts during format/build
    echo "Stashing unstaged changes..."
    git stash push --keep-index --include-untracked -m "Pre-commit hook: stashed unstaged changes"
    STASH_CREATED=$?
    
    # Run the format command
    npm run format

    # Check if format command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run format failed"
        if [ $STASH_CREATED -eq 0 ]; then
            echo "You can restore your unstaged changes with: git stash pop"
        fi
        exit 1
    fi

    # Run the lint command
    npm run lint
    
    # Check if lint command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run lint failed"
        if [ $STASH_CREATED -eq 0 ]; then
            echo "You can restore your unstaged changes with: git stash pop"
        fi
        exit 1
    fi

    # Run the check command
    npm run check
    
    # Check if check command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run check failed"
        if [ $STASH_CREATED -eq 0 ]; then
            echo "You can restore your unstaged changes with: git stash pop"
        fi
        exit 1
    fi

    # Run the build command
    npm run build
    
    # Check if build command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run build failed"
        if [ $STASH_CREATED -eq 0 ]; then
            echo "You can restore your unstaged changes with: git stash pop"
        fi
        exit 1
    fi

    # Go back to repo root to add build output
    cd ../../..
    
    # Add the build output to staging area
    git add tools/server/public/index.html.gz
    
    if [ $STASH_CREATED -eq 0 ]; then
        echo "✅ Build completed. Your unstaged changes have been stashed."
        echo "They will be automatically restored after the commit."
        # Create a marker file to indicate stash was created by pre-commit hook
        touch .git/WEBUI_STASH_MARKER
    fi
    
    echo "Webui code formatted successfully"
fi

exit 0
EOF

# Create the post-commit hook
cat > "$POST_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Check if we have a stash marker from the pre-commit hook
if [ -f .git/WEBUI_STASH_MARKER ]; then
    echo "Restoring your unstaged changes..."
    git stash pop
    rm -f .git/WEBUI_STASH_MARKER
    echo "✅ Your unstaged changes have been restored."
fi

exit 0
EOF

# Make both hooks executable
chmod +x "$PRE_COMMIT_HOOK"
chmod +x "$POST_COMMIT_HOOK"

if [ $? -eq 0 ]; then
    echo "✅ Pre-commit and post-commit hooks installed successfully!"
    echo "   Pre-commit:  $PRE_COMMIT_HOOK"
    echo "   Post-commit: $POST_COMMIT_HOOK"
    echo ""
    echo "The hooks will automatically:"
    echo "  • Format, lint, check, and build webui code before commits"
    echo "  • Stash unstaged changes during the process"
    echo "  • Restore your unstaged changes after the commit"
    echo ""
    echo "To test the hooks, make a change to a file in the webui directory and commit it."
else
    echo "❌ Failed to make hooks executable"
    exit 1
fi
