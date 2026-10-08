#!/usr/bin/env bash
# Exercise SDK pruning in an isolated repository without configuring a build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/repo" "$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp"
cd "$SCRATCH/repo"
git init -q -b fixture
git config user.name 'Prune test'
git config user.email 'prune-test@example.invalid'
git config commit.gpgsign false
git config core.hooksPath /dev/null

mkdir -p scripts .claude/agents .codex/agents docs/development docs/research
cp "$SCRIPT_DIR/prune-to-lib.sh" scripts/
printf 'runbook\n' > .claude/agents/probe.md
printf 'name = "probe"\n' > .codex/agents/probe.toml
printf 'contract\n' > docs/development/agents.md
printf 'Read [contract](docs/development/agents.md).\n' > AGENTS.md
printf 'Read [guidance](AGENTS.md) and [contract](docs/development/agents.md).\n' > CLAUDE.md
printf 'SDK\n' > docs/SDK.md
printf 'evidence\n' > docs/research/probe.md
printf 'keep\n' > .claude/settings.json
printf 'keep\n' > .codex/config.toml
git add -f .
git commit -qm 'fixture'
SOURCE_SHA="$(git rev-parse HEAD)"

bash scripts/prune-to-lib.sh --from HEAD --branch sdk-probe --no-check

# The source branch stays intact; only the generated artifact loses the roles.
test "$(git rev-parse HEAD)" = "$SOURCE_SHA"
test "$(git branch --show-current)" = fixture
test -f .claude/agents/probe.md
test -f .codex/agents/probe.toml
test -f AGENTS.md
test -f CLAUDE.md
test -z "$(git status --porcelain)"
test -z "$(git ls-tree -r --name-only sdk-probe -- .claude/agents .codex/agents docs/development AGENTS.md CLAUDE.md)"
for path in docs/SDK.md docs/research/probe.md .claude/settings.json .codex/config.toml; do
    git cat-file -e "sdk-probe:$path"
done
test "$(git worktree list --porcelain | grep -c '^worktree ')" = 1
printf 'PASS: SDK prunes unusable roles and root guidance, preserves retained files and leaves source unchanged\n'
