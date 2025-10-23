#!/bin/bash

# Script to install pre-commit and pre-push hooks for webui
# Pre-commit: formats code and runs checks
# Pre-push: builds the project, stashes unstaged changes

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"
PRE_PUSH_HOOK="$REPO_ROOT/.git/hooks/pre-push"

echo "Installing pre-commit and pre-push hooks for webui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Check if there are any changes in the webui directory
if git diff --cached --name-only | grep -q "^tools/server/webui/"; then
    echo "Formatting and checking webui code..."
    
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

    # Run the lint command
    npm run lint
    
    # Check if lint command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run lint failed"
        exit 1
    fi

    # Run the check command
    npm run check
    
    # Check if check command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: npm run check failed"
        exit 1
    fi

    # Go back to repo root
    cd ../../..
    
    echo "✅ Webui code formatted and checked successfully"
fi

exit 0
EOF

# Create the pre-push hook
cat > "$PRE_PUSH_HOOK" << 'EOF'
#!/bin/bash

# Check if there are any webui changes that need building
WEBUI_CHANGES=$(git diff --name-only @{push}..HEAD | grep "^tools/server/webui/" || true)

if [ -n "$WEBUI_CHANGES" ]; then
    echo "Webui changes detected, checking if build is up-to-date..."
    
    # Change to webui directory
    cd tools/server/webui
    
    # Check if npm is available and package.json exists
    if [ ! -f "package.json" ]; then
        echo "Error: package.json not found in tools/server/webui"
        exit 1
    fi
    
    # Check if build output exists and is newer than source files
    BUILD_FILE="../public/index.html.gz"
    NEEDS_BUILD=false
    
    if [ ! -f "$BUILD_FILE" ]; then
        echo "Build output not found, building..."
        NEEDS_BUILD=true
    else
        # Check if any source files are newer than the build output
        if find src -newer "$BUILD_FILE" -type f | head -1 | grep -q .; then
            echo "Source files are newer than build output, rebuilding..."
            NEEDS_BUILD=true
        fi
    fi
    
    if [ "$NEEDS_BUILD" = true ]; then
        echo "Building webui..."
        
        # Stash any unstaged changes to avoid conflicts during build
        echo "Checking for unstaged changes..."
        if ! git diff --quiet || ! git diff --cached --quiet --diff-filter=A; then
            echo "Stashing unstaged changes..."
            git stash push --include-untracked -m "Pre-push hook: stashed unstaged changes"
            STASH_CREATED=$?
        else
            echo "No unstaged changes to stash"
            STASH_CREATED=1
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

        # Go back to repo root
        cd ../../..
        
        # Check if build output was created/updated
        if [ -f "tools/server/public/index.html.gz" ]; then
            # Add the build output and commit it
            git add tools/server/public/index.html.gz
            if ! git diff --cached --quiet; then
                echo "Committing updated build output..."
                git commit -m "chore: update webui build output"
                echo "✅ Build output committed successfully"
            else
                echo "Build output unchanged"
            fi
        else
            echo "Error: Build output not found after build"
            if [ $STASH_CREATED -eq 0 ]; then
                echo "You can restore your unstaged changes with: git stash pop"
            fi
            exit 1
        fi
        
        if [ $STASH_CREATED -eq 0 ]; then
            echo "✅ Build completed. Your unstaged changes have been stashed."
            echo "They will be automatically restored after the push."
            # Create a marker file to indicate stash was created by pre-push hook
            touch .git/WEBUI_PUSH_STASH_MARKER
        fi
    else
        echo "✅ Build output is up-to-date"
    fi
    
    echo "✅ Webui ready for push"
fi

exit 0
EOF

# Create the post-push hook (for restoring stashed changes after push)
cat > "$REPO_ROOT/.git/hooks/post-push" << 'EOF'
#!/bin/bash

# Check if we have a stash marker from the pre-push hook
if [ -f .git/WEBUI_PUSH_STASH_MARKER ]; then
    echo "Restoring your unstaged changes after push..."
    git stash pop
    rm -f .git/WEBUI_PUSH_STASH_MARKER
    echo "✅ Your unstaged changes have been restored."
fi

exit 0
EOF

# Make all hooks executable
chmod +x "$PRE_COMMIT_HOOK"
chmod +x "$PRE_PUSH_HOOK"
chmod +x "$REPO_ROOT/.git/hooks/post-push"

if [ $? -eq 0 ]; then
    echo "✅ Git hooks installed successfully!"
    echo "   Pre-commit: $PRE_COMMIT_HOOK"
    echo "   Pre-push:   $PRE_PUSH_HOOK"
    echo "   Post-push:  $REPO_ROOT/.git/hooks/post-push"
    echo ""
    echo "The hooks will automatically:"
    echo "  • Format and check webui code before commits (pre-commit)"
    echo "  • Build webui code before pushes (pre-push)"
    echo "  • Stash unstaged changes during build process"
    echo "  • Restore your unstaged changes after the push"
    echo ""
    echo "To test the hooks:"
    echo "  • Make a change to a file in the webui directory and commit it (triggers format/check)"
    echo "  • Push your commits to trigger the build process"
else
    echo "❌ Failed to make hooks executable"
    exit 1
fi
