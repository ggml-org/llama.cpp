#!/usr/bin/env bash
#
# Pre-commit hook for llama-ui
# Runs: format (staged files only) + type-check
# Stashes unstaged changes temporarily and restores them after.

# Only run when there are staged changes in tools/ui/
if ! git diff --cached --name-only | grep -q "^tools/ui/"; then
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT/tools/ui"

# Check that node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ node_modules not found. Run 'npm install' first."
    exit 1
fi

# Stash unstaged changes in tools/ui/ so they don't interfere
stash_name="pi-ui-precommit"
git stash push --keep-index -u -m "$stash_name" -- tools/ui/ 2>/dev/null || true

echo "Running pre-commit checks for llama-ui..."

# Format only staged files
staged_ui=$(git diff --cached --name-only -- tools/ui/)
if [ -n "$staged_ui" ]; then
    echo "$staged_ui" | xargs npx --no-install prettier --write
    format_ok=$?
    # Re-stage formatted files
    git add tools/ui/
else
    format_ok=0
fi

# Type-check the clean tree
npm run check
check_ok=$?

# Restore stashed changes
if git stash list | grep -q "$stash_name"; then
    git stash pop 2>/dev/null || true
fi

if [ $format_ok -ne 0 ]; then
    echo "❌ Format failed"
    exit 1
fi
if [ $check_ok -ne 0 ]; then
    echo "❌ Type check failed"
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
