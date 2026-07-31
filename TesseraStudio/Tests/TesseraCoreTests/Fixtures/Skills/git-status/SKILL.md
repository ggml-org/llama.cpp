---
name: git-status
description: Summarize a git working tree - branch, staged and unstaged changes, and untracked files.
emoji: "🌿"
os: ["darwin", "linux"]
requires:
  bins: ["git"]
install:
  - xcode-select --install
---

# Git Status

Report the state of a git repository without mutating it.

## When to Use

- Before committing, to confirm what is staged and what is dirty.
- When the user asks "what changed?" or "is the tree clean?".

## When NOT to Use

- When a mutating action is requested (commit, push, reset): this skill only reads.
- Outside a git repository.

## Setup

Requires git. On macOS it ships with the Xcode command line tools:

```
xcode-select --install
```

## Common Commands

```
git status --short --branch
git diff --stat
git ls-files --others --exclude-standard
```
