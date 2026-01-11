# Branch Management Rules for llama.cpp Development

## Branch Hierarchy

```
origin/master (upstream)
    │
    └── production-consolidated (PROTECTED - production/benchmarks)
            │
            └── research/* (experimental work)
```

## Branch Purposes

| Branch | Purpose | Build From | Modify? |
|--------|---------|------------|---------|
| `production-consolidated` | Production benchmarks, stable features | This branch | NO - cherry-pick only |
| `research/*` | Experimental features, testing | `production-consolidated` | YES |
| `feature/*` | Isolated feature development | `origin/master` | YES |

## Rules

### Rule 1: Never Modify production-consolidated Directly
- All changes go through cherry-pick from tested feature branches
- Create `research/your-experiment` for testing
- Only cherry-pick to production-consolidated after validation

### Rule 2: Always Rebuild After Branch Switch
```bash
# ALWAYS do this after switching branches:
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF
cmake --build build -j$(nproc)
```

### Rule 3: Verify Build Before Benchmarking
```bash
# Check for undefined symbols (should return nothing):
nm build/bin/llama-cli | grep "U.*llama_"

# Verify required flags exist:
./build/bin/llama-cli --help | grep moe-n-expert
```

### Rule 4: Research Branch Workflow
```bash
# Start new research from production-consolidated:
git checkout production-consolidated
git checkout -b research/my-experiment

# Work on experiment...
# When done and validated, cherry-pick to production-consolidated:
git checkout production-consolidated
git cherry-pick <commit-hash>
git push fork production-consolidated
```

### Rule 5: Tag Working States
```bash
# After successful benchmark session:
git tag benchmark-$(date +%Y-%m-%d) -m "Working benchmark state"
git push fork --tags
```

### Rule 6: Document Feature Dependencies
Before cherry-picking, verify the commit doesn't depend on uncommitted infrastructure.
Check with:
```bash
# See what files the commit touches:
git show <commit> --stat

# Check if any new types/functions are referenced but not defined:
git show <commit> | grep -E "^[\+].*\(" | head -20
```

## Current Branch Status (2026-01-10)

| Branch | Status | Features |
|--------|--------|----------|
| `production-consolidated` | STABLE | MoE hard mask, layer skip, lookahead fixes, parallel repack |
| `feature/moe-hard-mask` | MERGED | → production-consolidated |
| `feature/eagle-penultimate-layer` | DO NOT USE | Has undefined symbol issues |
| `mtp-branch` | INCOMPATIBLE | Requires infrastructure not in origin/master |

## Recovering from Build Issues

If you encounter `undefined symbol` errors or SIGSEGV:

1. **Check current branch**: `git branch --show-current`
2. **Check build date**: `ls -la build/bin/llama-cli`
3. **Clean rebuild**: `rm -rf build && cmake -B build ... && cmake --build build -j$(nproc)`
4. **Verify symbols**: `nm build/bin/llama-cli | grep "U.*llama"`
5. **If still broken**: `git checkout production-consolidated` and rebuild
