#!/usr/bin/env bash
#
# Install git hooks for llama-ui
# Copies pre-commit and pre-push hooks into the repo's .git/hooks directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HOOKS_DIR="$REPO_ROOT/$(cd "$REPO_ROOT" && git rev-parse --git-path hooks)"

# Verify package.json exists
if [ ! -f "$REPO_ROOT/tools/ui/package.json" ]; then
    echo "❌ package.json not found in tools/ui"
    exit 1
fi

echo "Installing git hooks for llama-ui..."

for hook in pre-commit pre-push; do
    src="$SCRIPT_DIR/${hook}.sh"
    dst="$HOOKS_DIR/$hook"

    if cp "$src" "$dst" && chmod +x "$dst"; then
        echo "  ✅ $hook"
    else
        echo "  ❌ Failed to install $hook"
        exit 1
    fi
done

echo ""
echo "Pre-commit:  format (staged) + type-check"
echo "Pre-push:    lint + test"
echo ""
echo "Hooks stash unstaged changes temporarily and restore them after."
echo "Skip with:  git commit --no-verify / git push --no-verify"
