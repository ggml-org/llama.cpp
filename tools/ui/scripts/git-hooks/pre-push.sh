#!/usr/bin/env bash
#
# Pre-push hook for llama-ui
# Runs: lint + test
# Ignores unstaged changes (stashes them temporarily and restores after).

needs_check=false

# Read refs from stdin: local_ref local_sha remote_ref remote_sha
while read local_ref local_sha remote_ref remote_sha; do
    # New branch or force-push — always check
    if [ "$local_sha" = "0000000000000000000000000000000000000000" ] || \
       [ "$remote_sha" = "0000000000000000000000000000000000000000" ]; then
        needs_check=true
        continue
    fi

    # Check for changes in tools/ui/ between remote and local
    if git diff --name-only "$remote_sha...$local_sha" -- tools/ui/ | grep -q .; then
        needs_check=true
    fi
done

if [ "$needs_check" = false ]; then
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT/tools/ui"

# Check that node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ node_modules not found. Run 'npm install' first."
    exit 1
fi

# Stash unstaged changes so they don't interfere with checks
stash_name="pi-ui-prepush"
git stash push -u -m "$stash_name" -- tools/ui/ 2>/dev/null || true

echo "Running pre-push checks for llama-ui..."

# Lint
npm run lint
lint_ok=$?

# Test
npm test
test_ok=$?

# Restore stashed changes
if git stash list | grep -q "$stash_name"; then
    git stash pop 2>/dev/null || true
fi

if [ $lint_ok -ne 0 ]; then
    echo "❌ Lint failed"
    exit 1
fi

if [ $test_ok -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ Pre-push checks passed"
exit 0
