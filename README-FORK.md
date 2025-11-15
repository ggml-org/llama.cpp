# Custom Fork Instructions

This is a personal fork of llama.cpp with custom modifications.

## Custom Changes

This fork includes the following modifications:

1. **Timestamp format** - Changed log timestamp from relative format (M.ss.mmm.uuu) to absolute format (YYYY-MM-DD HH:MM:SS.mmm)
   - Files: `common/log.cpp`, `common/log.h`
   - Makes logs more readable and suitable for production monitoring

2. **Thinking tags support** - Added optional thinking tags in JSON schema grammar
   - File: `common/json-schema-to-grammar.cpp`
   - Allows model to output `<think>...</think>` before JSON response

## Branch Structure

- `master` - Clean branch tracking upstream (ggml-org/llama.cpp)
- `custom-fixes` - Branch with custom modifications (work here)

## Daily Workflow

### Working with custom changes

```bash
cd ~/work/llama.cpp-fork

# Switch to custom branch
git checkout custom-fixes

# Build the project
cmake -B build && cmake --build build

# Your binaries are in: build/bin/
```

### Updating from upstream

When you want to get latest changes from the original llama.cpp project:

```bash
cd ~/work/llama.cpp-fork

# 1. Switch to master and update from upstream
git checkout master
git pull upstream master

# 2. Push updated master to your fork
git push origin master

# 3. Apply your changes on top of new updates
git checkout custom-fixes
git rebase master

# 4. If rebase successful, push updated branch
git push origin custom-fixes --force-with-lease
```

### If rebase has conflicts

If you get conflicts during rebase:

```bash
# 1. Fix conflicts in the files git shows
# 2. After fixing, mark as resolved:
git add <fixed-files>

# 3. Continue rebase:
git rebase --continue

# If too many conflicts, abort and ask for help:
git rebase --abort
```

## Remote Repositories

- `origin` - Your fork: https://github.com/itiu/llama.cpp
- `upstream` - Original project: https://github.com/ggml-org/llama.cpp

View remotes:
```bash
git remote -v
```

## Quick Commands

```bash
# Check current branch and status
git status

# View all branches
git branch -vv

# See your custom changes
git diff master custom-fixes

# View commit history
git log --oneline --graph --all -10
```

## Backup

Your changes are automatically backed up to GitHub when you push:
```bash
git push origin custom-fixes
```

You can view them at: https://github.com/itiu/llama.cpp/tree/custom-fixes

